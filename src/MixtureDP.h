#ifndef MIXTUREDP_H
#define MIXTUREDP_H

#include "Prerequisites.h"
#include <vector>

// Centered Dirichlet process mixture of Gaussians on the residual distribution,
// adapted from Henderson et al. (2020, Biostatistics) and Yang et al. (2010).
//
// Six modes:
//   GAUSSIAN        : no mixture, single normal residual (this class is a no-op)
//   SHARED_DP       : one DP pooled across all observations
//   SOURCE_DP       : independent DPs per data source, shared sigma (Stage A)
//   SOURCE_DP_SCALE : independent DPs per data source AND per-source sigma_g.
//                     Adds an inverse-gamma conjugate update for sigma_g each
//                     sweep.  Designed to compare against SOURCE_DP when only
//                     the residual *scale* (not shape) differs across sources.
//   SOURCE_HDP      : Stage B -- shared atoms theta_k* across sources via a
//                     top-level DP(gamma, H), per-source weights pi_{sk},
//                     per-source means mu_s, table-count augmentation for the
//                     top-level sticks (Antoniak-Teh), Escobar-West updates
//                     for the concentration parameters gamma and M_s.
//   SOURCE_HDP_SCALE: Stage B + per-source sigma_g.  Same hierarchical sharing
//                     of atoms as SOURCE_HDP, but each source carries its own
//                     residual scale.  Label sampling and shared-atom posterior
//                     use the per-source sigma_g; sigma_g is updated each sweep
//                     via the same IG conjugate step as SOURCE_DP_SCALE.
//
// Per-source state is held in std::vector-of-vector containers indexed by
// group g in [0, num_groups_).  SHARED_DP uses num_groups_ = 1; all SOURCE_*
// modes use num_groups_ = 2 with group 0 = real-world data (S=0) and
// group 1 = randomised trial (S=1).
class MixtureDP {
public:
  enum Mode { GAUSSIAN = 0, SHARED_DP = 1, SOURCE_DP = 2,
              SOURCE_DP_SCALE = 3, SOURCE_HDP = 4,
              SOURCE_HDP_SCALE = 5 };

  MixtureDP(int mode, size_t n_total, size_t K,
            double prior_atom_variance,
            double mass_init,
            const int* source_indicator,
            double sigma_init      = 1.0,
            double nu_sigma        = 3.0,
            double lambda_sigma    = 1.0);

  bool active() const          { return mode_ != GAUSSIAN; }
  int mode() const             { return mode_; }
  size_t truncation_level() const { return K_; }
  int num_groups() const       { return num_groups_; }

  // Run one Gibbs sweep of the DP block (labels -> tabulate -> stick weights
  // & concentration -> atoms with centring).  No-op when GAUSSIAN.
  // `residuals` has length n_total_ and equals y - total_predictions.
  void Update(const double* residuals, double sigma, Random& random);

  // Fill theta_shift[i] (length n_total_) with the current centred atom for
  // observation i.  Fills zeros when GAUSSIAN.
  void GetIndividualShifts(double* theta_shift) const;

  const std::vector<double>& mix_prop(int g)  const { return mix_prop_[g];  }
  const std::vector<double>& locations(int g) const { return locations_[g]; }
  double mass(int g) const                          { return mass_[g];      }

  // Per-source error scale (only meaningful in SOURCE_DP_SCALE; equals
  // sigma_init for the other modes).
  double sigma_g(int g) const { return sigma_g_[g]; }

  // sqrt of weight-by-group pooled variance.  Used as the reference sigma the
  // caller passes to forest backfitting; weights compensate per observation.
  double sigma_pooled() const;

  // HDP-only accessors (return empty / zero for other modes).
  const std::vector<double>& locations_shared() const { return locations_shared_; }
  const std::vector<double>& mu_g_vec()         const { return mu_g_;             }
  const std::vector<double>& beta()             const { return beta_;             }
  double gamma_top() const { return gamma_top_; }

private:
  int    mode_;
  size_t n_total_;
  size_t K_;
  int    num_groups_;
  double prior_atom_variance_;
  double psi1_;
  double psi2_;
  double nu_sigma_;
  double lambda_sigma_;

  std::vector<int>                obs_group_;            // length n_total_
  std::vector<size_t>             obs_index_in_group_;   // length n_total_
  std::vector<std::vector<size_t>> group_indices_;       // group_indices_[g][jj] -> obs index

  std::vector<std::vector<int>>    labels_;          // labels_[g][jj] in 1..K
  std::vector<std::vector<double>> mix_prop_;        // mix_prop_[g][k]
  std::vector<std::vector<double>> locations_;       // locations_[g][k] (centred)
  std::vector<std::vector<int>>    cluster_counts_;  // cluster_counts_[g][k]
  std::vector<double>              mass_;            // mass_[g]
  std::vector<double>              sigma_g_;         // sigma_g_[g], only updated in SCALE mode

  // HDP-only state (allocated only when mode_ == SOURCE_HDP).
  std::vector<double>              locations_shared_;   // length K -- unconstrained atoms theta_k*
  std::vector<double>              mu_g_;               // length num_groups_ -- per-source mu_s
  std::vector<double>              beta_;               // length K -- top-level sticks
  double                           gamma_top_;          // top-level concentration
  std::vector<std::vector<int>>    table_counts_;       // table_counts_[g][k] = t_{sk}

  void updateLabelsGroup   (int g, const double* residuals, double sigma, Random& random);
  void tabCountsGroup      (int g);
  void updateMixGroup      (int g, Random& random);
  void updateLocationsGroup(int g, const double* residuals, double sigma, Random& random);
  void updateSigmaGroup    (int g, const double* residuals, Random& random);

  // HDP Gibbs steps
  void updateLabelsGroupHDP(int g, const double* residuals, double sigma, Random& random);
  void updateAtomsShared   (const double* residuals, double sigma, Random& random);
  void updateAtomsSharedPerSourceSigma(const double* residuals, Random& random);
  void updateBeta          (Random& random);
  void updateMixHDP        (int g, Random& random);
  void updateMassHDP       (int g, Random& random);
  void updateGamma         (Random& random);
  void recomputeCenteredLocations();
};

#endif // MIXTUREDP_H
