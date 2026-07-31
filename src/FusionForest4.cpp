#include "FusionForest4.h"
#include "MixtureDP.h"

// [[Rcpp::export]]
Rcpp::List FusionForest4_cpp(
  SEXP nSEXP, SEXP p_treatSEXP, SEXP p_controlSEXP, SEXP X_train_treatSEXP,
  SEXP X_train_controlSEXP, SEXP ySEXP, SEXP status_indicatorSEXP, SEXP is_survivalSEXP,
  SEXP observed_left_timeSEXP, SEXP observed_right_timeSEXP,
  SEXP interval_censoring_indicatorSEXP,
  SEXP treatment_indicatorSEXP, SEXP source_indicatorSEXP,
  SEXP n_testSEXP, SEXP X_test_controlSEXP, SEXP X_test_treatSEXP, SEXP X_test_deconfSEXP,
  SEXP X_test_deviationSEXP,
  SEXP treatment_indicator_testSEXP, SEXP source_indicator_testSEXP,
  SEXP n_deconfSEXP, SEXP p_deconfSEXP, SEXP X_train_deconfSEXP,
  SEXP no_trees_deconfSEXP, SEXP power_deconfSEXP, SEXP base_deconfSEXP,
  SEXP p_grow_deconfSEXP, SEXP p_prune_deconfSEXP, SEXP omega_deconfSEXP,
  SEXP n_deviationSEXP, SEXP p_deviationSEXP, SEXP X_train_deviationSEXP,
  SEXP no_trees_deviationSEXP, SEXP power_deviationSEXP, SEXP base_deviationSEXP,
  SEXP p_grow_deviationSEXP, SEXP p_prune_deviationSEXP, SEXP omega_deviationSEXP,
  SEXP no_trees_treatSEXP, SEXP power_treatSEXP, SEXP base_treatSEXP,
  SEXP p_grow_treatSEXP, SEXP p_prune_treatSEXP, SEXP omega_treatSEXP,
  SEXP no_trees_controlSEXP, SEXP power_controlSEXP, SEXP base_controlSEXP,
  SEXP p_grow_controlSEXP, SEXP p_prune_controlSEXP, SEXP omega_controlSEXP,
  SEXP sigma_knownSEXP, SEXP sigmaSEXP, SEXP lambdaSEXP, SEXP nuSEXP,
  SEXP N_postSEXP, SEXP N_burnSEXP,
  SEXP store_posterior_sampleSEXP,
  SEXP verboseSEXP,
  SEXP treatment_codingSEXP,
  SEXP propensity_trainSEXP,
  SEXP propensity_testSEXP,
  SEXP mixture_modeSEXP,
  SEXP mixture_KSEXP,
  SEXP mixture_prior_atom_varianceSEXP,
  SEXP mixture_mass_initSEXP
) {

  // ---- Argument conversion ----

  size_t n           = Rcpp::as<size_t>(nSEXP);
  size_t n_deconf    = Rcpp::as<size_t>(n_deconfSEXP);
  size_t n_deviation = Rcpp::as<size_t>(n_deviationSEXP);
  size_t p_treat     = Rcpp::as<size_t>(p_treatSEXP);
  size_t p_control   = Rcpp::as<size_t>(p_controlSEXP);
  size_t p_deconf    = Rcpp::as<size_t>(p_deconfSEXP);
  size_t p_deviation = Rcpp::as<size_t>(p_deviationSEXP);

  Rcpp::NumericVector X_train_treat_vector(X_train_treatSEXP);
  double* X_train_treat = &X_train_treat_vector[0];
  Rcpp::NumericVector X_train_control_vector(X_train_controlSEXP);
  double* X_train_control = &X_train_control_vector[0];
  Rcpp::NumericVector X_train_deconf_vector(X_train_deconfSEXP);
  double* X_train_deconf = &X_train_deconf_vector[0];
  Rcpp::NumericVector X_train_deviation_vector(X_train_deviationSEXP);
  double* X_train_deviation = &X_train_deviation_vector[0];
  Rcpp::NumericVector y_vector(ySEXP);
  double* y = &y_vector[0];

  bool is_survival = Rcpp::as<bool>(is_survivalSEXP);
  Rcpp::IntegerVector treatment_indicator_vector(treatment_indicatorSEXP);
  int* treatment_indicator = &treatment_indicator_vector[0];
  Rcpp::NumericVector status_indicator_vector(status_indicatorSEXP);
  double* status_indicator = &status_indicator_vector[0];
  std::vector<double> y_observed_vector(y_vector.begin(), y_vector.end());
  double* y_observed = y_observed_vector.data();
  // Interval-censoring bounds; see FusionForest_cpp for the contract.
  Rcpp::NumericVector observed_left_time_vector(observed_left_timeSEXP);
  double* observed_left_time = &observed_left_time_vector[0];
  Rcpp::NumericVector observed_right_time_vector(observed_right_timeSEXP);
  double* observed_right_time = &observed_right_time_vector[0];
  Rcpp::NumericVector interval_censoring_indicator_vector(interval_censoring_indicatorSEXP);
  double* interval_censoring_indicator = &interval_censoring_indicator_vector[0];
  Rcpp::IntegerVector source_indicator_vector(source_indicatorSEXP);
  int* source_indicator = &source_indicator_vector[0];

  size_t n_test = Rcpp::as<size_t>(n_testSEXP);
  Rcpp::NumericVector X_test_treat_vector(X_test_treatSEXP);
  double* X_test_treat = &X_test_treat_vector[0];
  Rcpp::NumericVector X_test_control_vector(X_test_controlSEXP);
  double* X_test_control = &X_test_control_vector[0];
  Rcpp::NumericVector X_test_deconf_vector(X_test_deconfSEXP);
  double* X_test_deconf = &X_test_deconf_vector[0];
  Rcpp::NumericVector X_test_deviation_vector(X_test_deviationSEXP);
  double* X_test_deviation = &X_test_deviation_vector[0];
  Rcpp::IntegerVector treatment_indicator_test_vector(treatment_indicator_testSEXP);
  int* treatment_indicator_test = &treatment_indicator_test_vector[0];
  Rcpp::IntegerVector source_indicator_test_vector(source_indicator_testSEXP);
  int* source_indicator_test = &source_indicator_test_vector[0];

  // Hyperparameters — treatment effect forest
  size_t no_trees_treat = Rcpp::as<size_t>(no_trees_treatSEXP);
  double power_treat    = Rcpp::as<double>(power_treatSEXP);
  double base_treat     = Rcpp::as<double>(base_treatSEXP);
  double p_grow_treat   = Rcpp::as<double>(p_grow_treatSEXP);
  double p_prune_treat  = Rcpp::as<double>(p_prune_treatSEXP);
  double omega_treat    = Rcpp::as<double>(omega_treatSEXP);

  // Hyperparameters — mu (shared baseline) forest
  size_t no_trees_control = Rcpp::as<size_t>(no_trees_controlSEXP);
  double power_control    = Rcpp::as<double>(power_controlSEXP);
  double base_control     = Rcpp::as<double>(base_controlSEXP);
  double p_grow_control   = Rcpp::as<double>(p_grow_controlSEXP);
  double p_prune_control  = Rcpp::as<double>(p_prune_controlSEXP);
  double omega_control    = Rcpp::as<double>(omega_controlSEXP);

  // Hyperparameters — deconfounding forest
  size_t no_trees_deconf = Rcpp::as<size_t>(no_trees_deconfSEXP);
  double power_deconf    = Rcpp::as<double>(power_deconfSEXP);
  double base_deconf     = Rcpp::as<double>(base_deconfSEXP);
  double p_grow_deconf   = Rcpp::as<double>(p_grow_deconfSEXP);
  double p_prune_deconf  = Rcpp::as<double>(p_prune_deconfSEXP);
  double omega_deconf    = Rcpp::as<double>(omega_deconfSEXP);

  // Hyperparameters — deviation (g) forest
  size_t no_trees_deviation = Rcpp::as<size_t>(no_trees_deviationSEXP);
  double power_deviation    = Rcpp::as<double>(power_deviationSEXP);
  double base_deviation     = Rcpp::as<double>(base_deviationSEXP);
  double p_grow_deviation   = Rcpp::as<double>(p_grow_deviationSEXP);
  double p_prune_deviation  = Rcpp::as<double>(p_prune_deviationSEXP);
  double omega_deviation    = Rcpp::as<double>(omega_deviationSEXP);

  // Error variance hyperparameters
  bool   sigma_known = Rcpp::as<bool>(sigma_knownSEXP);
  double sigma       = Rcpp::as<double>(sigmaSEXP);
  double lambda      = Rcpp::as<double>(lambdaSEXP);
  double nu          = Rcpp::as<double>(nuSEXP);

  // MCMC dimensions
  size_t N_post = Rcpp::as<size_t>(N_postSEXP);
  size_t N_burn = Rcpp::as<size_t>(N_burnSEXP);

  // Storage flags
  bool store_posterior_sample = Rcpp::as<bool>(store_posterior_sampleSEXP);

  // Random number generation — uses R's RNG state
  RandomGenerator random;

  bool verbose = Rcpp::as<bool>(verboseSEXP);

  // Residual mixture configuration (MixtureDP) — see FusionForest_cpp.
  int    mixture_mode                = Rcpp::as<int>(mixture_modeSEXP);
  size_t mixture_K                   = Rcpp::as<size_t>(mixture_KSEXP);
  double mixture_prior_atom_variance = Rcpp::as<double>(mixture_prior_atom_varianceSEXP);
  double mixture_mass_init           = Rcpp::as<double>(mixture_mass_initSEXP);

  // Treatment-effect coding b_i (see FusionForest_cpp for full documentation):
  //   "binary"   : b = 1 if treated, 0 if control
  //   "centered" : b = 0.5 / -0.5
  //   "adaptive" : b = z_i - propensity_i
  // The tau and c forests use weights b_i^2 to recover the correct full-data
  // likelihood; tiny |b| are floored to weight 0 to avoid 0/0.
  std::string treatment_coding = Rcpp::as<std::string>(treatment_codingSEXP);
  Rcpp::NumericVector propensity_train_vector(propensity_trainSEXP);
  double* propensity_train = (propensity_train_vector.size() > 0)
    ? &propensity_train_vector[0] : nullptr;
  Rcpp::NumericVector propensity_test_vector(propensity_testSEXP);
  double* propensity_test = (propensity_test_vector.size() > 0)
    ? &propensity_test_vector[0] : nullptr;

  const double b_eps = 1e-10;

  double* b_train = new double[n];
  double* b_test  = n_test ? new double[n_test] : nullptr;

  if (treatment_coding == "binary") {
    for (size_t k = 0; k < n; ++k)
      b_train[k] = (treatment_indicator[k] == 1) ? 1.0 : 0.0;
    for (size_t k = 0; k < n_test; ++k)
      b_test[k]  = (treatment_indicator_test[k] == 1) ? 1.0 : 0.0;
  } else if (treatment_coding == "adaptive") {
    if (propensity_train == nullptr || propensity_test == nullptr) {
      delete[] b_train; if (b_test) delete[] b_test;
      Rcpp::stop("treatment_coding = 'adaptive' requires propensity_train "
                 "and propensity_test.");
    }
    for (size_t k = 0; k < n; ++k)
      b_train[k] = static_cast<double>(treatment_indicator[k])
                   - propensity_train[k];
    for (size_t k = 0; k < n_test; ++k)
      b_test[k]  = static_cast<double>(treatment_indicator_test[k])
                   - propensity_test[k];
  } else if (treatment_coding == "centered") {
    for (size_t k = 0; k < n; ++k)
      b_train[k] = (treatment_indicator[k] == 1) ? 0.5 : -0.5;
    for (size_t k = 0; k < n_test; ++k)
      b_test[k]  = (treatment_indicator_test[k] == 1) ? 0.5 : -0.5;
  } else {
    delete[] b_train; if (b_test) delete[] b_test;
    Rcpp::stop("Unknown treatment_coding: '%s'. "
               "Use 'binary', 'centered', or 'adaptive'.",
               treatment_coding.c_str());
  }

  // Per-observation weights b_i^2 for the tau forest (length n) and the
  // deconfounding c forest (length n_deconf, RWD subset only).  These are
  // fixed for the whole MCMC run.
  double* weights_treat  = new double[n];
  double* weights_deconf = n_deconf ? new double[n_deconf] : nullptr;
  {
    size_t j = 0;
    for (size_t k = 0; k < n; ++k) {
      double bk = b_train[k];
      weights_treat[k] = (std::abs(bk) < b_eps) ? 0.0 : bk * bk;
      if (source_indicator[k] == 0) {
        weights_deconf[j] = weights_treat[k];
        ++j;
      }
    }
  }


  // ---- Storage containers ----

  Rcpp::NumericVector train_predictions_mean(n,            0.0);
  Rcpp::NumericVector test_predictions_mean(n_test,        0.0);
  Rcpp::NumericVector train_predictions_mean_control(n,        0.0);
  Rcpp::NumericVector test_predictions_mean_control(n_test,    0.0);
  Rcpp::NumericVector train_predictions_mean_treat(n,          0.0);
  Rcpp::NumericVector test_predictions_mean_treat(n_test,      0.0);
  Rcpp::NumericVector train_predictions_mean_deconf(n_deconf,  0.0);
  Rcpp::NumericVector test_predictions_mean_deconf(n_test,     0.0);
  Rcpp::NumericVector train_predictions_mean_deviation(n_deviation, 0.0);
  Rcpp::NumericVector test_predictions_mean_deviation(n_test,      0.0);

  Rcpp::NumericMatrix train_predictions_sample_control;
  Rcpp::NumericMatrix test_predictions_sample_control;
  Rcpp::NumericMatrix train_predictions_sample_treat;
  Rcpp::NumericMatrix test_predictions_sample_treat;
  Rcpp::NumericMatrix train_predictions_sample_deconf;
  Rcpp::NumericMatrix test_predictions_sample_deconf;
  Rcpp::NumericMatrix train_predictions_sample_deviation;
  Rcpp::NumericMatrix test_predictions_sample_deviation;
  if (store_posterior_sample) {
    train_predictions_sample_control   = Rcpp::NumericMatrix(N_post, n);
    test_predictions_sample_control    = Rcpp::NumericMatrix(N_post, n_test);
    train_predictions_sample_treat     = Rcpp::NumericMatrix(N_post, n);
    test_predictions_sample_treat      = Rcpp::NumericMatrix(N_post, n_test);
    train_predictions_sample_deconf    = Rcpp::NumericMatrix(N_post, n_deconf);
    test_predictions_sample_deconf     = Rcpp::NumericMatrix(N_post, n_test);
    train_predictions_sample_deviation = Rcpp::NumericMatrix(N_post, n_deviation);
    test_predictions_sample_deviation  = Rcpp::NumericMatrix(N_post, n_test);
  }

  Rcpp::NumericVector store_sigma = sigma_known
    ? Rcpp::NumericVector::create(sigma)
    : Rcpp::NumericVector(N_post + N_burn);

  // Acceptance ratio trackers
  bool* accepted_control   = new bool[no_trees_control]();
  bool* accepted_treat     = new bool[no_trees_treat]();
  bool* accepted_deconf    = new bool[no_trees_deconf]();
  bool* accepted_deviation = new bool[no_trees_deviation]();
  double sum_accept_control = 0, sum_accept_treat = 0;
  double sum_accept_deconf = 0,  sum_accept_deviation = 0;
  double acceptance_ratio_control, acceptance_ratio_treat;
  double acceptance_ratio_deconf,  acceptance_ratio_deviation;

  // Working arrays
  double* testpred_treat     = n_test ? new double[n_test] : nullptr;
  double* testpred_control   = n_test ? new double[n_test] : nullptr;
  double* testpred_deconf    = n_test ? new double[n_test] : nullptr;
  double* testpred_deviation = n_test ? new double[n_test] : nullptr;
  double* total_predictions         = new double[n];
  double* total_plus_shift          = new double[n];   // total_predictions + theta_shift
  double* theta_shift               = new double[n];   // DP cluster shift (0 if Gaussian)
  double* residuals_for_dp          = new double[n];   // y - total_predictions
  double* augmented_outcome_treat   = new double[n];
  double* augmented_outcome_control = new double[n];
  double* augmented_outcome_deconf    = new double[n_deconf];
  double* augmented_outcome_deviation = new double[n_deviation];

  for (size_t i = 0; i < n; ++i) {
    augmented_outcome_treat[i]   = y[i] / 2.0;
    augmented_outcome_control[i] = y[i] / 2.0;
    theta_shift[i]               = 0.0;
  }
  for (size_t i = 0; i < n_deconf; ++i) augmented_outcome_deconf[i] = 0.0;
  for (size_t i = 0; i < n_deviation; ++i) augmented_outcome_deviation[i] = 0.0;

  // --- MixtureDP: residual-distribution prior ---
  MixtureDP mixture(mixture_mode, n, mixture_K,
                    mixture_prior_atom_variance, mixture_mass_init,
                    source_indicator,
                    /*sigma_init=*/sigma,
                    /*nu_sigma=*/nu,
                    /*lambda_sigma=*/lambda);
  const bool dp_active     = mixture.active();
  const int  dp_groups     = mixture.num_groups();
  const bool dp_hdp_scale_mode = (mixture_mode == MixtureDP::SOURCE_HDP_SCALE);
  const bool dp_scale_mode = (mixture_mode == MixtureDP::SOURCE_DP_SCALE) || dp_hdp_scale_mode;
  const bool dp_hdp_mode   = (mixture_mode == MixtureDP::SOURCE_HDP) || dp_hdp_scale_mode;

  Rcpp::List dp_mix_prop_list (dp_active ? dp_groups : 0);
  Rcpp::List dp_locations_list(dp_active ? dp_groups : 0);
  Rcpp::List dp_mass_list     (dp_active ? dp_groups : 0);
  Rcpp::List dp_sigma_g_list  (dp_scale_mode ? dp_groups : 0);
  Rcpp::List dp_mu_g_list     (dp_hdp_mode   ? dp_groups : 0);
  Rcpp::NumericMatrix dp_locations_shared_mat;
  Rcpp::NumericMatrix dp_beta_mat;
  Rcpp::NumericVector dp_gamma_vec;
  if (dp_active) {
    for (int g = 0; g < dp_groups; ++g) {
      dp_mix_prop_list[g]  = Rcpp::NumericMatrix(N_post, static_cast<int>(mixture_K));
      dp_locations_list[g] = Rcpp::NumericMatrix(N_post, static_cast<int>(mixture_K));
      dp_mass_list[g]      = Rcpp::NumericVector(N_post);
      if (dp_scale_mode) dp_sigma_g_list[g] = Rcpp::NumericVector(N_post);
      if (dp_hdp_mode)   dp_mu_g_list[g]    = Rcpp::NumericVector(N_post);
    }
  }
  if (dp_hdp_mode) {
    dp_locations_shared_mat = Rcpp::NumericMatrix(N_post, static_cast<int>(mixture_K));
    dp_beta_mat             = Rcpp::NumericMatrix(N_post, static_cast<int>(mixture_K));
    dp_gamma_vec            = Rcpp::NumericVector(N_post);
  }

  // ---- Per-observation precision weight buffers (SCALE mode only) ----
  double* weight_treat_dyn     = new double[n];
  double* weight_deconf_dyn    = n_deconf    ? new double[n_deconf]    : nullptr;
  double* weight_control_dyn   = dp_scale_mode ? new double[n]              : nullptr;
  double* weight_deviation_dyn = dp_scale_mode && n_deviation
                                  ? new double[n_deviation] : nullptr;
  for (size_t k = 0; k < n; ++k) weight_treat_dyn[k] = weights_treat[k];
  for (size_t j = 0; j < n_deconf; ++j) weight_deconf_dyn[j] = weights_deconf[j];
  if (dp_scale_mode) {
    for (size_t k = 0; k < n; ++k) weight_control_dyn[k] = 1.0;
    for (size_t j = 0; j < n_deviation; ++j) weight_deviation_dyn[j] = 1.0;
  }


  // ---- Set up forests ----

  // mu: shared baseline forest (all observations)
  ForestEngine forest_mu(no_trees_control);
  forest_mu.SetTreePrior(base_control, power_control, omega_control,
                         p_grow_control, p_prune_control,
                         0.5, 1.0, static_cast<double>(p_control),
                         true, false, 1.0);
  forest_mu.SetUpForest(p_control, n, X_train_control,
                        augmented_outcome_control, nullptr, omega_control);

  // g: RWD-specific deviation forest (RWD observations only)
  ForestEngine forest_g(no_trees_deviation);
  forest_g.SetTreePrior(base_deviation, power_deviation, omega_deviation,
                        p_grow_deviation, p_prune_deviation,
                        0.5, 1.0, static_cast<double>(p_deviation),
                        true, false, 1.0);
  forest_g.SetUpForest(p_deviation, n_deviation, X_train_deviation,
                       augmented_outcome_deviation, nullptr, omega_deviation);

  // tau: treatment effect forest (all observations)
  ForestEngine forest_tau(no_trees_treat);
  forest_tau.SetTreePrior(base_treat, power_treat, omega_treat,
                          p_grow_treat, p_prune_treat,
                          0.5, 1.0, static_cast<double>(p_treat),
                          true, false, 1.0);
  forest_tau.SetUpForest(p_treat, n, X_train_treat,
                         augmented_outcome_treat, nullptr, omega_treat);
  forest_tau.SetWeights(weight_treat_dyn);

  // c: deconfounding forest (RWD observations only)
  ForestEngine forest_c(no_trees_deconf);
  forest_c.SetTreePrior(base_deconf, power_deconf, omega_deconf,
                        p_grow_deconf, p_prune_deconf,
                        0.5, 1.0, static_cast<double>(p_deconf),
                        true, false, 1.0);
  forest_c.SetUpForest(p_deconf, n_deconf, X_train_deconf,
                       augmented_outcome_deconf, nullptr, omega_deconf);
  forest_c.SetWeights(weight_deconf_dyn);

  // SOURCE_DP_SCALE: enable per-obs precision weighting on the otherwise
  // unweighted mu and g forests.  Weight buffers are updated each sweep.
  if (dp_scale_mode) {
    forest_mu.SetWeights(weight_control_dyn);
    forest_g .SetWeights(weight_deviation_dyn);
  }


  // ---- Timing ----

  time_t time_stamp;
  int time_start = time(&time_stamp);
  int barWidth = 70;
  if (verbose) Rcpp::Rcout << "\nProgress of the MCMC sampler:\n\n";


  // ---- MCMC loop ----

  for (size_t i = 0; i < N_post + N_burn; ++i) {

    // Progress bar
    if (verbose) {
      float progress = static_cast<float>(i) /
                       static_cast<float>(N_post + N_burn);
      int pos = static_cast<int>(barWidth * progress);
      Rcpp::Rcout << "|";
      for (int j = 0; j < barWidth; ++j)
        Rcpp::Rcout << (j < pos ? "=" : (j == pos ? ">" : " "));
      Rcpp::Rcout << "| " << static_cast<int>(progress * 100.0)
                  << " %\r";
      Rcpp::Rcout.flush();
    }

    // -- Update mu (shared baseline, all observations) --
    forest_mu.UpdateForest(sigma, accepted_control, random);

    // Refresh augmented outcomes after mu update
    {
      size_t j = 0;  // index into RWD subset
      for (size_t k = 0; k < n; ++k) {
        const double b      = b_train[k];
        const bool   b_zero = (std::abs(b) < b_eps);
        const double s =
          (source_indicator[k] == 1) ? 1.0 : 0.0;

        double g_k = 0.0, c_k = 0.0;
        if (source_indicator[k] == 0) {
          g_k = forest_g.GetPrediction(j);
          c_k = forest_c.GetPrediction(j);
          augmented_outcome_deviation[j] =
            y[k] - forest_mu.GetPrediction(k)
                 - b * (forest_tau.GetPrediction(k) + c_k)
                 - theta_shift[k];
          augmented_outcome_deconf[j] = b_zero ? 0.0
            : (y[k] - forest_mu.GetPrediction(k) - g_k
                    - b * forest_tau.GetPrediction(k)
                    - theta_shift[k]) / b;
          ++j;
        }
        augmented_outcome_treat[k] = b_zero ? 0.0
          : (y[k] - forest_mu.GetPrediction(k)
                  - (1.0 - s) * g_k
                  - b * (1.0 - s) * c_k
                  - theta_shift[k]) / b;
      }
    }

    // -- Update g (RWD deviation, RWD observations only) --
    forest_g.UpdateForest(sigma, accepted_deviation, random);

    // Refresh augmented outcomes after g update
    {
      size_t j = 0;
      for (size_t k = 0; k < n; ++k) {
        const double b      = b_train[k];
        const bool   b_zero = (std::abs(b) < b_eps);
        const double s =
          (source_indicator[k] == 1) ? 1.0 : 0.0;

        double g_k = 0.0, c_k = 0.0;
        if (source_indicator[k] == 0) {
          g_k = forest_g.GetPrediction(j);
          c_k = forest_c.GetPrediction(j);
          augmented_outcome_deconf[j] = b_zero ? 0.0
            : (y[k] - forest_mu.GetPrediction(k) - g_k
                    - b * forest_tau.GetPrediction(k)
                    - theta_shift[k]) / b;
          ++j;
        }
        augmented_outcome_control[k] =
          y[k] - (1.0 - s) * g_k
               - b * (forest_tau.GetPrediction(k)
                     + (1.0 - s) * c_k)
               - theta_shift[k];
        augmented_outcome_treat[k] = b_zero ? 0.0
          : (y[k] - forest_mu.GetPrediction(k)
                  - (1.0 - s) * g_k
                  - b * (1.0 - s) * c_k
                  - theta_shift[k]) / b;
      }
    }

    // -- Update tau (treatment effect, all observations) --
    forest_tau.UpdateForest(sigma, accepted_treat, random);

    // Refresh augmented outcomes after tau update
    {
      size_t j = 0;
      for (size_t k = 0; k < n; ++k) {
        const double b      = b_train[k];
        const bool   b_zero = (std::abs(b) < b_eps);
        const double s =
          (source_indicator[k] == 1) ? 1.0 : 0.0;

        double g_k = 0.0, c_k = 0.0;
        if (source_indicator[k] == 0) {
          g_k = forest_g.GetPrediction(j);
          c_k = forest_c.GetPrediction(j);
          augmented_outcome_deviation[j] =
            y[k] - forest_mu.GetPrediction(k)
                 - b * (forest_tau.GetPrediction(k) + c_k)
                 - theta_shift[k];
          augmented_outcome_deconf[j] = b_zero ? 0.0
            : (y[k] - forest_mu.GetPrediction(k) - g_k
                    - b * forest_tau.GetPrediction(k)
                    - theta_shift[k]) / b;
          ++j;
        }
        augmented_outcome_control[k] =
          y[k] - (1.0 - s) * g_k
               - b * (forest_tau.GetPrediction(k)
                     + (1.0 - s) * c_k)
               - theta_shift[k];
      }
    }

    // -- Update c (deconfounding, RWD observations only) --
    forest_c.UpdateForest(sigma, accepted_deconf, random);

    // Refresh augmented outcomes after c update
    {
      size_t j = 0;
      for (size_t k = 0; k < n; ++k) {
        const double b      = b_train[k];
        const bool   b_zero = (std::abs(b) < b_eps);
        const double s =
          (source_indicator[k] == 1) ? 1.0 : 0.0;

        double g_k = 0.0, c_k = 0.0;
        if (source_indicator[k] == 0) {
          g_k = forest_g.GetPrediction(j);
          c_k = forest_c.GetPrediction(j);
          augmented_outcome_deviation[j] =
            y[k] - forest_mu.GetPrediction(k)
                 - b * (forest_tau.GetPrediction(k) + c_k)
                 - theta_shift[k];
          ++j;
        }
        augmented_outcome_control[k] =
          y[k] - (1.0 - s) * g_k
               - b * (forest_tau.GetPrediction(k)
                     + (1.0 - s) * c_k)
               - theta_shift[k];
        augmented_outcome_treat[k] = b_zero ? 0.0
          : (y[k] - forest_mu.GetPrediction(k)
                  - (1.0 - s) * g_k
                  - b * (1.0 - s) * c_k
                  - theta_shift[k]) / b;
      }
    }

    // -- Compute total predictions --
    {
      size_t j = 0;
      for (size_t k = 0; k < n; ++k) {
        const double b = b_train[k];
        double g_k = 0.0, c_k = 0.0, s = 1.0;
        if (source_indicator[k] == 0) {
          s   = 0.0;
          g_k = forest_g.GetPrediction(j);
          c_k = forest_c.GetPrediction(j);
          ++j;
        }
        total_predictions[k] =
          forest_mu.GetPrediction(k) + (1.0 - s) * g_k
          + b * (forest_tau.GetPrediction(k) + (1.0 - s) * c_k);
      }
    }

    // -- Update residual-mixture (DP) --
    if (dp_active) {
      for (size_t k = 0; k < n; ++k)
        residuals_for_dp[k] = y[k] - total_predictions[k];
      mixture.Update(residuals_for_dp, sigma, random);
      mixture.GetIndividualShifts(theta_shift);
    }

    // SCALE mode: refresh per-obs precision weights and pooled sigma.
    if (dp_scale_mode) {
      const double sg_rwd    = mixture.sigma_g(0);
      const double sg_rct   = mixture.sigma_g(1);
      const double sigma_ref = mixture.sigma_pooled();
      const double sigma_ref_sq = sigma_ref * sigma_ref;
      const double factor_rwd  = sigma_ref_sq / (sg_rwd  * sg_rwd );
      const double factor_rct = sigma_ref_sq / (sg_rct * sg_rct);
      for (size_t k = 0; k < n; ++k) {
        const double fac = (source_indicator[k] == 1) ? factor_rct : factor_rwd;
        weight_treat_dyn  [k] = weights_treat[k] * fac;
        weight_control_dyn[k] = fac;
      }
      for (size_t j = 0; j < n_deconf; ++j)
        weight_deconf_dyn[j] = weights_deconf[j] * factor_rwd;
      for (size_t j = 0; j < n_deviation; ++j)
        weight_deviation_dyn[j] = factor_rwd;          // g is RWD-only
      sigma = sigma_ref;
      if (!sigma_known) store_sigma[i] = sigma;
    }

    // total_predictions + cluster shift (collapses to total_predictions when
    // dp_active is false because theta_shift is zero-initialised).
    for (size_t k = 0; k < n; ++k)
      total_plus_shift[k] = total_predictions[k] + theta_shift[k];

    // -- Update sigma -- (skipped in SCALE mode; DP owns sigma)
    if (!dp_scale_mode)
      UpdateSigma(sigma_known, sigma, store_sigma, i,
                  y, n, total_plus_shift, nu, lambda, random);

    // -- Augment censored observations --
    AugmentCensoredObservations(is_survival, y,
                                observed_left_time, status_indicator,
                                observed_right_time,
                                interval_censoring_indicator,
                                total_plus_shift, sigma, n, random);


    // -- Post-burn-in storage --
    if (i >= N_burn) {

      size_t j_rwd = 0;
      for (size_t k = 0; k < n; ++k) {
        const double b = b_train[k];
        const double mu_k  = forest_mu.GetPrediction(k);
        const double tau_k = forest_tau.GetPrediction(k);
        double g_k = 0.0, c_k = 0.0, s = 1.0;
        if (source_indicator[k] == 0) {
          s   = 0.0;
          g_k = forest_g.GetPrediction(j_rwd);
          c_k = forest_c.GetPrediction(j_rwd);
          train_predictions_mean_deviation[j_rwd] += g_k;
          train_predictions_mean_deconf[j_rwd]    += c_k;
          ++j_rwd;
        }
        train_predictions_mean[k] +=
          mu_k + (1.0 - s) * g_k
          + b * (tau_k + (1.0 - s) * c_k);
        train_predictions_mean_control[k] += mu_k;
        train_predictions_mean_treat[k]   += tau_k;
      }

      if (store_posterior_sample) {
        for (size_t k = 0; k < n; ++k) {
          train_predictions_sample_control(i - N_burn, k) =
            forest_mu.GetPrediction(k);
          train_predictions_sample_treat(i - N_burn, k) =
            forest_tau.GetPrediction(k);
        }
        size_t j_rwd2 = 0;
        for (size_t k = 0; k < n; ++k) {
          if (source_indicator[k] == 0) {
            train_predictions_sample_deconf(i - N_burn, j_rwd2) =
              forest_c.GetPrediction(j_rwd2);
            train_predictions_sample_deviation(i - N_burn, j_rwd2) =
              forest_g.GetPrediction(j_rwd2);
            ++j_rwd2;
          }
        }
      }

      if (n_test > 0) {
        forest_mu.Predict(p_control, n_test,
                          X_test_control, testpred_control);
        forest_tau.Predict(p_treat, n_test,
                           X_test_treat, testpred_treat);
        forest_c.Predict(p_deconf, n_test,
                         X_test_deconf, testpred_deconf);
        forest_g.Predict(p_deviation, n_test,
                         X_test_deviation, testpred_deviation);

        if (store_posterior_sample) {
          for (size_t k = 0; k < n_test; ++k) {
            test_predictions_sample_control(i - N_burn, k) =
              testpred_control[k];
            test_predictions_sample_treat(i - N_burn, k) =
              testpred_treat[k];
            test_predictions_sample_deconf(i - N_burn, k) =
              testpred_deconf[k];
            test_predictions_sample_deviation(i - N_burn, k) =
              testpred_deviation[k];
          }
        }

        for (size_t k = 0; k < n_test; ++k) {
          const double b = b_test[k];
          const double s =
            (source_indicator_test[k] == 1) ? 1.0 : 0.0;
          const double g_k = testpred_deviation[k];
          const double c_k = testpred_deconf[k];
          test_predictions_mean[k] +=
            testpred_control[k] + (1.0 - s) * g_k
            + b * (testpred_treat[k] + (1.0 - s) * c_k);
          test_predictions_mean_control[k]   += testpred_control[k];
          test_predictions_mean_treat[k]     += testpred_treat[k];
          test_predictions_mean_deconf[k]    += c_k;
          test_predictions_mean_deviation[k] += g_k;
        }
      }

      // Acceptance ratios
      for (size_t j = 0; j < no_trees_control; ++j)
        sum_accept_control += accepted_control[j];
      for (size_t j = 0; j < no_trees_treat; ++j)
        sum_accept_treat += accepted_treat[j];
      for (size_t j = 0; j < no_trees_deconf; ++j)
        sum_accept_deconf += accepted_deconf[j];
      for (size_t j = 0; j < no_trees_deviation; ++j)
        sum_accept_deviation += accepted_deviation[j];

      // DP posterior draws
      if (dp_active) {
        for (int g = 0; g < dp_groups; ++g) {
          Rcpp::NumericMatrix mp = dp_mix_prop_list[g];
          Rcpp::NumericMatrix lc = dp_locations_list[g];
          Rcpp::NumericVector ms = dp_mass_list[g];
          const std::vector<double>& prop = mixture.mix_prop(g);
          const std::vector<double>& locs = mixture.locations(g);
          for (size_t k = 0; k < mixture_K; ++k) {
            mp(i - N_burn, static_cast<int>(k)) = prop[k];
            lc(i - N_burn, static_cast<int>(k)) = locs[k];
          }
          ms[i - N_burn] = mixture.mass(g);
          if (dp_scale_mode) {
            Rcpp::NumericVector sg = dp_sigma_g_list[g];
            sg[i - N_burn] = mixture.sigma_g(g);
          }
          if (dp_hdp_mode) {
            Rcpp::NumericVector mug = dp_mu_g_list[g];
            mug[i - N_burn] = mixture.mu_g_vec()[g];
          }
        }
        if (dp_hdp_mode) {
          const std::vector<double>& locs_sh = mixture.locations_shared();
          const std::vector<double>& betas   = mixture.beta();
          for (size_t k = 0; k < mixture_K; ++k) {
            dp_locations_shared_mat(i - N_burn, static_cast<int>(k)) = locs_sh[k];
            dp_beta_mat            (i - N_burn, static_cast<int>(k)) = betas[k];
          }
          dp_gamma_vec[i - N_burn] = mixture.gamma_top();
        }
      }
    }

  } // end MCMC loop


  // ---- Post-loop ----

  acceptance_ratio_control =
    sum_accept_control / (N_post * no_trees_control);
  acceptance_ratio_treat =
    sum_accept_treat / (N_post * no_trees_treat);
  acceptance_ratio_deconf =
    sum_accept_deconf / (N_post * no_trees_deconf);
  acceptance_ratio_deviation =
    sum_accept_deviation / (N_post * no_trees_deviation);

  int time_end = time(&time_stamp);

  if (verbose) {
    Rcpp::Rcout << "|";
    for (int j = 0; j < barWidth; ++j) Rcpp::Rcout << "=";
    Rcpp::Rcout << "| 100 %\r";
    Rcpp::Rcout.flush();
    Rcpp::Rcout << "\n" << std::endl;
    Rcpp::Rcout << "Mean acceptance ratio (shared baseline):   "
                << acceptance_ratio_control << std::endl;
    Rcpp::Rcout << "Mean acceptance ratio (RWD deviation):     "
                << acceptance_ratio_deviation << std::endl;
    Rcpp::Rcout << "Mean acceptance ratio (treatment effect):  "
                << acceptance_ratio_treat << std::endl;
    Rcpp::Rcout << "Mean acceptance ratio (deconfounding):     "
                << acceptance_ratio_deconf << std::endl;
    Rcpp::Rcout << "\nDone in " << (time_end - time_start)
                << " seconds.\n" << std::endl;
  }

  // Rescale posterior means
  for (size_t k = 0; k < n; ++k) {
    train_predictions_mean[k]         /= N_post;
    train_predictions_mean_control[k] /= N_post;
    train_predictions_mean_treat[k]   /= N_post;
  }
  for (size_t k = 0; k < n_deconf; ++k)
    train_predictions_mean_deconf[k] /= N_post;
  for (size_t k = 0; k < n_deviation; ++k)
    train_predictions_mean_deviation[k] /= N_post;
  for (size_t k = 0; k < n_test; ++k) {
    test_predictions_mean[k]           /= N_post;
    test_predictions_mean_control[k]   /= N_post;
    test_predictions_mean_treat[k]     /= N_post;
    test_predictions_mean_deconf[k]    /= N_post;
    test_predictions_mean_deviation[k] /= N_post;
  }

  // ---- Build result list ----

  Rcpp::List results;
  results["sigma"]                       = store_sigma;
  results["train_predictions"]           = train_predictions_mean;
  results["test_predictions"]            = test_predictions_mean;
  results["train_predictions_control"]   = train_predictions_mean_control;
  results["test_predictions_control"]    = test_predictions_mean_control;
  results["train_predictions_treat"]     = train_predictions_mean_treat;
  results["test_predictions_treat"]      = test_predictions_mean_treat;
  results["train_predictions_deconf"]    = train_predictions_mean_deconf;
  results["test_predictions_deconf"]     = test_predictions_mean_deconf;
  results["train_predictions_deviation"] = train_predictions_mean_deviation;
  results["test_predictions_deviation"]  = test_predictions_mean_deviation;
  results["acceptance_ratio_control"]    = acceptance_ratio_control;
  results["acceptance_ratio_treat"]      = acceptance_ratio_treat;
  results["acceptance_ratio_deconf"]     = acceptance_ratio_deconf;
  results["acceptance_ratio_deviation"]  = acceptance_ratio_deviation;
  if (store_posterior_sample) {
    results["train_predictions_sample_control"]   =
      train_predictions_sample_control;
    results["test_predictions_sample_control"]    =
      test_predictions_sample_control;
    results["train_predictions_sample_treat"]     =
      train_predictions_sample_treat;
    results["test_predictions_sample_treat"]      =
      test_predictions_sample_treat;
    results["train_predictions_sample_deconf"]    =
      train_predictions_sample_deconf;
    results["test_predictions_sample_deconf"]     =
      test_predictions_sample_deconf;
    results["train_predictions_sample_deviation"] =
      train_predictions_sample_deviation;
    results["test_predictions_sample_deviation"]  =
      test_predictions_sample_deviation;
  }
  if (dp_active) {
    results["dp_mode"]       = mixture_mode;
    results["dp_K"]          = static_cast<int>(mixture_K);
    results["dp_num_groups"] = dp_groups;
    results["dp_mix_prop"]   = dp_mix_prop_list;
    results["dp_locations"]  = dp_locations_list;
    results["dp_mass"]       = dp_mass_list;
    if (dp_scale_mode) results["dp_sigma_g"] = dp_sigma_g_list;
    if (dp_hdp_mode) {
      results["dp_locations_shared"] = dp_locations_shared_mat;
      results["dp_beta"]             = dp_beta_mat;
      results["dp_gamma"]            = dp_gamma_vec;
      results["dp_mu_g"]             = dp_mu_g_list;
    }
  }

  // ---- Clean up ----

  delete[] testpred_control;
  delete[] testpred_treat;
  delete[] testpred_deconf;
  delete[] testpred_deviation;
  delete[] accepted_control;
  delete[] accepted_treat;
  delete[] accepted_deconf;
  delete[] accepted_deviation;
  delete[] total_predictions;
  delete[] total_plus_shift;
  delete[] theta_shift;
  delete[] residuals_for_dp;
  delete[] augmented_outcome_control;
  delete[] augmented_outcome_treat;
  delete[] augmented_outcome_deconf;
  delete[] augmented_outcome_deviation;
  delete[] b_train;
  if (b_test) delete[] b_test;
  delete[] weights_treat;
  if (weights_deconf) delete[] weights_deconf;
  delete[] weight_treat_dyn;
  if (weight_deconf_dyn)    delete[] weight_deconf_dyn;
  if (weight_control_dyn)   delete[] weight_control_dyn;
  if (weight_deviation_dyn) delete[] weight_deviation_dyn;

  return results;
}
