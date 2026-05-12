#include "MixtureDP.h"
#include <numeric>

MixtureDP::MixtureDP(int mode, size_t n_total, size_t K,
                     double prior_atom_variance,
                     double mass_init,
                     const int* source_indicator,
                     double sigma_init,
                     double nu_sigma,
                     double lambda_sigma)
    : mode_(mode),
      n_total_(n_total),
      K_(K),
      num_groups_(0),
      prior_atom_variance_(prior_atom_variance),
      psi1_(2.0),
      psi2_(0.1),
      nu_sigma_(nu_sigma),
      lambda_sigma_(lambda_sigma) {

  if (mode_ == GAUSSIAN) {
    // Even in Gaussian mode, store a single sigma so callers can query it.
    sigma_g_.assign(1, sigma_init);
    return;
  }

  if (mode_ == SHARED_DP) {
    num_groups_ = 1;
    obs_group_.assign(n_total_, 0);
  } else { // SOURCE_DP or SOURCE_DP_SCALE
    num_groups_ = 2;
    obs_group_.resize(n_total_);
    for (size_t i = 0; i < n_total_; ++i)
      obs_group_[i] = (source_indicator[i] == 1) ? 1 : 0;
  }

  group_indices_.assign(num_groups_, std::vector<size_t>());
  for (size_t i = 0; i < n_total_; ++i)
    group_indices_[obs_group_[i]].push_back(i);

  obs_index_in_group_.assign(n_total_, 0);
  {
    std::vector<size_t> g_count(num_groups_, 0);
    for (size_t i = 0; i < n_total_; ++i) {
      int g = obs_group_[i];
      obs_index_in_group_[i] = g_count[g]++;
    }
  }

  labels_         .assign(num_groups_, std::vector<int>());
  mix_prop_       .assign(num_groups_, std::vector<double>(K_, 1.0 / static_cast<double>(K_)));
  locations_      .assign(num_groups_, std::vector<double>(K_, 0.0));
  cluster_counts_ .assign(num_groups_, std::vector<int>(K_, 0));
  mass_           .assign(num_groups_, mass_init);
  sigma_g_        .assign(num_groups_, sigma_init);

  // HDP-only state
  if (mode_ == SOURCE_HDP) {
    locations_shared_.assign(K_, 0.0);
    mu_g_           .assign(num_groups_, 0.0);
    beta_           .assign(K_, 1.0 / static_cast<double>(K_));
    table_counts_   .assign(num_groups_, std::vector<int>(K_, 0));
    gamma_top_       = 1.0;
  } else {
    gamma_top_       = 0.0;
  }

  for (int g = 0; g < num_groups_; ++g) {
    size_t n_g = group_indices_[g].size();
    labels_[g].assign(n_g, 1);                 // all in cluster 1 initially
    cluster_counts_[g][0] = static_cast<int>(n_g);
  }
}

void MixtureDP::Update(const double* residuals, double sigma, Random& random) {
  if (mode_ == GAUSSIAN) return;

  if (mode_ == SOURCE_HDP) {
    // Stage B: shared atoms theta_k* with per-source weights, sticks, and means.
    for (int g = 0; g < num_groups_; ++g) {
      if (group_indices_[g].empty()) continue;
      updateLabelsGroup(g, residuals, sigma, random);
      tabCountsGroup(g);
    }
    updateBeta(random);
    for (int g = 0; g < num_groups_; ++g) {
      if (group_indices_[g].empty()) continue;
      updateMixHDP(g, random);
      updateMassHDP(g, random);
    }
    updateAtomsShared(residuals, sigma, random);
    recomputeCenteredLocations();
    updateGamma(random);
    return;
  }

  const bool scale_mode = (mode_ == SOURCE_DP_SCALE);
  for (int g = 0; g < num_groups_; ++g) {
    if (group_indices_[g].empty()) continue;
    const double sigma_g_eff = scale_mode ? sigma_g_[g] : sigma;
    updateLabelsGroup   (g, residuals, sigma_g_eff, random);
    tabCountsGroup      (g);
    updateMixGroup      (g, random);
    updateLocationsGroup(g, residuals, sigma_g_eff, random);
    if (scale_mode) updateSigmaGroup(g, residuals, random);
  }
}

double MixtureDP::sigma_pooled() const {
  if (mode_ != SOURCE_DP_SCALE) {
    return sigma_g_.empty() ? 1.0 : sigma_g_.front();
  }
  double ssq = 0.0;
  size_t total = 0;
  for (int g = 0; g < num_groups_; ++g) {
    size_t n_g = group_indices_[g].size();
    ssq += static_cast<double>(n_g) * sigma_g_[g] * sigma_g_[g];
    total += n_g;
  }
  if (total == 0) return 1.0;
  return std::sqrt(ssq / static_cast<double>(total));
}

void MixtureDP::GetIndividualShifts(double* theta_shift) const {
  if (mode_ == GAUSSIAN) {
    for (size_t i = 0; i < n_total_; ++i) theta_shift[i] = 0.0;
    return;
  }
  for (size_t i = 0; i < n_total_; ++i) {
    int g    = obs_group_[i];
    size_t j = obs_index_in_group_[i];
    int lbl  = labels_[g][j];
    if (lbl < 1) lbl = 1;
    if (static_cast<size_t>(lbl) > K_) lbl = static_cast<int>(K_);
    theta_shift[i] = locations_[g][lbl - 1];
  }
}

// ----- private helpers ---------------------------------------------------

void MixtureDP::updateLabelsGroup(int g, const double* residuals,
                                   double sigma, Random& random) {
  const std::vector<size_t>& idx = group_indices_[g];
  const std::vector<double>& prop = mix_prop_[g];
  const std::vector<double>& loc  = locations_[g];
  std::vector<int>&          lbl  = labels_[g];
  const size_t n_g = idx.size();

  const double inv_two_sigsq = 0.5 / (sigma * sigma);

  std::vector<double> w(K_);

  for (size_t jj = 0; jj < n_g; ++jj) {
    const double r = residuals[idx[jj]];

    // log-stabilised softmax over clusters
    double max_log = -std::numeric_limits<double>::infinity();
    for (size_t k = 0; k < K_; ++k) {
      double diff = r - loc[k];
      // log(pi_k) - 0.5 (r - mu_k)^2 / sigma^2  (drop common 1/sqrt(2pi sig^2))
      double log_w = (prop[k] > 0.0)
        ? std::log(prop[k]) - diff * diff * inv_two_sigsq
        : -1.0e300;
      w[k] = log_w;
      if (log_w > max_log) max_log = log_w;
    }
    double tot = 0.0;
    for (size_t k = 0; k < K_; ++k) {
      w[k] = std::exp(w[k] - max_log);
      tot += w[k];
    }

    if (!(tot > 0.0)) {
      // Degenerate: fall back to sampling from the stick weights
      double u = random.uniform();
      double cum = 0.0;
      int chosen = static_cast<int>(K_);
      for (size_t k = 0; k < K_; ++k) {
        cum += prop[k];
        if (u < cum) { chosen = static_cast<int>(k) + 1; break; }
      }
      lbl[jj] = chosen;
      continue;
    }

    double u = random.uniform() * tot;
    double cum = 0.0;
    int chosen = static_cast<int>(K_);
    for (size_t k = 0; k < K_; ++k) {
      cum += w[k];
      if (u < cum) { chosen = static_cast<int>(k) + 1; break; }
    }
    lbl[jj] = chosen;
  }
}

void MixtureDP::tabCountsGroup(int g) {
  std::fill(cluster_counts_[g].begin(), cluster_counts_[g].end(), 0);
  const std::vector<int>& lbl = labels_[g];
  for (size_t jj = 0; jj < lbl.size(); ++jj) {
    int k = lbl[jj];
    if (k >= 1 && static_cast<size_t>(k) <= K_) cluster_counts_[g][k - 1]++;
  }
}

void MixtureDP::updateMixGroup(int g, Random& random) {
  std::vector<double>& prop = mix_prop_[g];
  const std::vector<int>& nn = cluster_counts_[g];
  double& mass = mass_[g];

  // Ishwaran-James truncated stick-breaking: V_h ~ Beta(n_h+1, mass + sum_{j>h} n_j)
  double log_prod_1m  = 0.0;   // running log prod_{j<h}(1 - V_j)
  double sum_log_1m   = 0.0;   // sum_{h=0..K-2} log(1 - V_h), used by concentration update

  for (size_t h = 0; h + 1 < K_; ++h) {
    int n_h = nn[h];
    int n_after = 0;
    for (size_t j = h + 1; j < K_; ++j) n_after += nn[j];

    double shape1 = static_cast<double>(n_h) + 1.0;
    double shape2 = mass + static_cast<double>(n_after);
    double V = random.beta(shape1, shape2);

    if (V <= 0.0)         V = std::nextafter(0.0, 1.0);
    if (V >= 1.0)         V = std::nextafter(1.0, 0.0);

    prop[h] = std::exp(std::log(V) + log_prod_1m);

    double log_1mV = std::log(1.0 - V);
    log_prod_1m += log_1mV;
    sum_log_1m  += log_1mV;
  }

  double partial = 0.0;
  for (size_t h = 0; h + 1 < K_; ++h) partial += prop[h];
  prop[K_ - 1] = std::max(0.0, 1.0 - partial);

  // Concentration update (Gamma conjugate, AFTrees defaults psi1=2, psi2=0.1)
  double gam_shape = psi1_ + static_cast<double>(K_) - 1.0;
  double gam_rate  = psi2_ - sum_log_1m;
  if (gam_rate <= 0.0) gam_rate = 1.0e-10;
  double new_mass = random.gamma(gam_shape, gam_rate);
  if (!(new_mass > 0.0)) new_mass = 1.0e-6;
  mass = new_mass;
}

void MixtureDP::updateLocationsGroup(int g, const double* residuals,
                                      double sigma, Random& random) {
  std::vector<double>&       loc  = locations_[g];
  const std::vector<int>&    nn   = cluster_counts_[g];
  const std::vector<size_t>& idx  = group_indices_[g];
  const std::vector<int>&    lbl  = labels_[g];
  const std::vector<double>& prop = mix_prop_[g];
  const size_t n_g = idx.size();

  const double sigsq       = sigma * sigma;
  const double prior_sigsq = prior_atom_variance_;

  for (size_t k = 0; k < K_; ++k) {
    double clust_sum = 0.0;
    for (size_t jj = 0; jj < n_g; ++jj) {
      if (lbl[jj] == static_cast<int>(k) + 1)
        clust_sum += residuals[idx[jj]];
    }
    double wts       = prior_sigsq / (prior_sigsq * static_cast<double>(nn[k]) + sigsq);
    double post_mean = wts * clust_sum;
    double post_var  = sigsq * wts;
    double post_sd   = std::sqrt(post_var);
    loc[k] = post_mean + post_sd * random.normal();
  }

  // Centring: atoms are recentred to weighted mean zero within this group.
  // (Per-source recentre when SOURCE_DP*, global recentre when SHARED_DP.)
  double muG = 0.0;
  for (size_t k = 0; k < K_; ++k) muG += prop[k] * loc[k];
  for (size_t k = 0; k < K_; ++k) loc[k] -= muG;
}

void MixtureDP::updateSigmaGroup(int g, const double* residuals,
                                  Random& random) {
  // IG conjugate update of sigma_g^2 using residuals minus the assigned atom,
  // restricted to observations in group g.  Mirrors the AFTrees Mixdev::updateSigma
  // pattern, stratified by source.  Prior: sigma^2 ~ IG(nu/2, nu*lambda/2).
  const std::vector<size_t>& idx = group_indices_[g];
  const std::vector<int>&    lbl = labels_[g];
  const std::vector<double>& loc = locations_[g];
  const size_t n_g = idx.size();
  if (n_g == 0) return;

  double ss = 0.0;
  for (size_t jj = 0; jj < n_g; ++jj) {
    int k = lbl[jj];
    if (k < 1) k = 1;
    if (static_cast<size_t>(k) > K_) k = static_cast<int>(K_);
    double r = residuals[idx[jj]] - loc[k - 1];
    ss += r * r;
  }
  double new_sigma2 = (nu_sigma_ * lambda_sigma_ + ss) /
                      random.chi_square(static_cast<double>(n_g) + nu_sigma_);
  double new_sigma  = std::sqrt(new_sigma2);
  if (!(new_sigma > 0.0))         new_sigma = 1.0e-6;
  if (new_sigma > 10.0)           new_sigma = 10.0;   // mirror UpdateSigma's cap
  sigma_g_[g] = new_sigma;
}

// =================================================================
// HDP-CDP (Stage B) Gibbs steps
// =================================================================

void MixtureDP::updateAtomsShared(const double* residuals, double sigma,
                                   Random& random) {
  // Conjugate Gaussian draw for each shared atom theta_k*, pooling observations
  // from all source groups in cluster k.  The model decomposes the residual as
  //
  //     R_i = theta_{Z_i}^* - mu_{S_i} + sigma * eps_i,
  //
  // so the augmented quantity (R_i + mu_{S_i}) ~ N(theta_k^*, sigma^2) when
  // Z_i = k.  We collect (R_i + mu_{S_i}) sums per cluster across both groups.
  const double sigsq       = sigma * sigma;
  const double prior_sigsq = prior_atom_variance_;

  for (size_t k = 0; k < K_; ++k) {
    double clust_sum = 0.0;
    int    n_k       = 0;
    for (int g = 0; g < num_groups_; ++g) {
      const std::vector<size_t>& idx = group_indices_[g];
      const std::vector<int>&    lbl = labels_[g];
      const double mu_s              = mu_g_[g];
      for (size_t jj = 0; jj < idx.size(); ++jj) {
        if (lbl[jj] == static_cast<int>(k) + 1) {
          clust_sum += residuals[idx[jj]] + mu_s;
          ++n_k;
        }
      }
    }
    double wts       = prior_sigsq / (prior_sigsq * static_cast<double>(n_k) + sigsq);
    double post_mean = wts * clust_sum;
    double post_var  = sigsq * wts;
    double post_sd   = std::sqrt(post_var);
    locations_shared_[k] = post_mean + post_sd * random.normal();
  }
}

void MixtureDP::recomputeCenteredLocations() {
  // mu_s = sum_k pi_{sk} * theta_k*  ; locations_[g][k] = theta_k* - mu_s
  for (int g = 0; g < num_groups_; ++g) {
    const std::vector<double>& prop = mix_prop_[g];
    double mu = 0.0;
    for (size_t k = 0; k < K_; ++k) mu += prop[k] * locations_shared_[k];
    mu_g_[g] = mu;
    std::vector<double>& loc = locations_[g];
    for (size_t k = 0; k < K_; ++k) loc[k] = locations_shared_[k] - mu;
  }
}

void MixtureDP::updateBeta(Random& random) {
  // Step 5a: sample table counts t_{sk} via Bernoulli sum
  //          (Antoniak 1974, written out in Teh et al. 2006, eq.(5)).
  for (int g = 0; g < num_groups_; ++g) {
    const std::vector<int>& nn = cluster_counts_[g];
    const double M_s = mass_[g];
    std::vector<int>& tg = table_counts_[g];
    for (size_t k = 0; k < K_; ++k) {
      const int n_gk = nn[k];
      if (n_gk <= 0) { tg[k] = 0; continue; }
      const double M_beta = M_s * beta_[k];
      int t = 0;
      for (int j = 1; j <= n_gk; ++j) {
        const double p = M_beta / (M_beta + static_cast<double>(j) - 1.0);
        if (random.uniform() < p) ++t;
      }
      if (t < 1) t = 1;          // at least one table when n_{gk} > 0
      tg[k] = t;
    }
  }

  // Step 5b: stick-breaking for beta given the table-count totals.
  std::vector<int> t_dot(K_, 0);
  for (int g = 0; g < num_groups_; ++g)
    for (size_t k = 0; k < K_; ++k) t_dot[k] += table_counts_[g][k];

  double log_prod_1m = 0.0;
  for (size_t h = 0; h + 1 < K_; ++h) {
    int t_after = 0;
    for (size_t j = h + 1; j < K_; ++j) t_after += t_dot[j];
    double shape1 = 1.0 + static_cast<double>(t_dot[h]);
    double shape2 = gamma_top_ + static_cast<double>(t_after);
    double V = random.beta(shape1, shape2);
    if (V <= 0.0) V = std::nextafter(0.0, 1.0);
    if (V >= 1.0) V = std::nextafter(1.0, 0.0);
    beta_[h] = std::exp(std::log(V) + log_prod_1m);
    log_prod_1m += std::log(1.0 - V);
  }
  double partial = 0.0;
  for (size_t h = 0; h + 1 < K_; ++h) partial += beta_[h];
  beta_[K_ - 1] = std::max(0.0, 1.0 - partial);
}

void MixtureDP::updateMixHDP(int g, Random& random) {
  // pi_s | beta, M_s, n_s ~ Dirichlet(M_s*beta_1 + n_{s1}, ..., M_s*beta_K + n_{sK})
  std::vector<double>& prop = mix_prop_[g];
  const std::vector<int>& nn = cluster_counts_[g];
  const double M_s = mass_[g];

  std::vector<double> alpha(K_);
  for (size_t k = 0; k < K_; ++k) {
    alpha[k] = M_s * beta_[k] + static_cast<double>(nn[k]);
    if (alpha[k] < 1.0e-12) alpha[k] = 1.0e-12;
  }

  // Sample Dirichlet via normalised gammas; use the log-Dirichlet helper for
  // numerical safety when some alpha_k are very small.
  std::vector<double> log_pi = random.log_dirichlet(alpha);
  // exponentiate and renormalise (log_dirichlet already normalises in log-space)
  double sum = 0.0;
  for (size_t k = 0; k < K_; ++k) {
    prop[k] = std::exp(log_pi[k]);
    sum += prop[k];
  }
  if (sum > 0.0) for (size_t k = 0; k < K_; ++k) prop[k] /= sum;
}

void MixtureDP::updateMassHDP(int g, Random& random) {
  // Escobar-West auxiliary-variable update for M_s.
  //   eta ~ Beta(M_s + 1, n_s)
  //   M_s ~ omega * Gamma(psi1 + t_{s.},     psi2 - log(eta))   +
  //         (1-omega) * Gamma(psi1 + t_{s.} - 1, psi2 - log(eta))
  // with omega / (1 - omega) = (psi1 + t_{s.} - 1) / (n_s * (psi2 - log(eta)))
  const double n_s = static_cast<double>(group_indices_[g].size());
  int t_sum = 0;
  for (size_t k = 0; k < K_; ++k) t_sum += table_counts_[g][k];
  const double t_sd = static_cast<double>(t_sum);
  if (n_s == 0.0 || t_sd == 0.0) return;

  double M = mass_[g];
  double eta = random.beta(M + 1.0, n_s);
  if (eta <= 0.0) eta = std::nextafter(0.0, 1.0);
  if (eta >= 1.0) eta = std::nextafter(1.0, 0.0);
  double rate = psi2_ - std::log(eta);
  if (rate <= 0.0) rate = 1.0e-10;

  double odds = (psi1_ + t_sd - 1.0) / (n_s * rate);
  double omega = odds / (1.0 + odds);
  if (!(omega >= 0.0 && omega <= 1.0)) omega = 0.5;

  double shape = (random.uniform() < omega) ? psi1_ + t_sd
                                            : psi1_ + t_sd - 1.0;
  if (shape < 1.0e-6) shape = 1.0e-6;
  double new_M = random.gamma(shape, rate);
  if (!(new_M > 0.0)) new_M = 1.0e-6;
  mass_[g] = new_M;
}

void MixtureDP::updateGamma(Random& random) {
  // Escobar-West for the top-level concentration.  K* = number of components
  // with t_{.k} > 0; t_{..} = total table count.
  int t_total = 0;
  int K_star  = 0;
  for (size_t k = 0; k < K_; ++k) {
    int t_dot_k = 0;
    for (int g = 0; g < num_groups_; ++g) t_dot_k += table_counts_[g][k];
    if (t_dot_k > 0) ++K_star;
    t_total += t_dot_k;
  }
  if (t_total == 0 || K_star == 0) return;

  const double t_total_d = static_cast<double>(t_total);
  const double K_star_d  = static_cast<double>(K_star);

  double eta = random.beta(gamma_top_ + 1.0, t_total_d);
  if (eta <= 0.0) eta = std::nextafter(0.0, 1.0);
  if (eta >= 1.0) eta = std::nextafter(1.0, 0.0);
  double rate = psi2_ - std::log(eta);
  if (rate <= 0.0) rate = 1.0e-10;

  double odds = (psi1_ + K_star_d - 1.0) / (t_total_d * rate);
  double omega = odds / (1.0 + odds);
  if (!(omega >= 0.0 && omega <= 1.0)) omega = 0.5;

  double shape = (random.uniform() < omega) ? psi1_ + K_star_d
                                            : psi1_ + K_star_d - 1.0;
  if (shape < 1.0e-6) shape = 1.0e-6;
  double new_gamma = random.gamma(shape, rate);
  if (!(new_gamma > 0.0)) new_gamma = 1.0e-6;
  gamma_top_ = new_gamma;
}
