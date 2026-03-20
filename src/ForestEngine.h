#ifndef GUARD_ForestEngine_h
#define GUARD_ForestEngine_h


#include "StanForest.h"


// Wrapper around StanForest for standard BART and Dirichlet BART (DART) models.

struct ForestEngine {

  std::unique_ptr<StanForest> stan_forest;

  ForestEngine(size_t m) {
    stan_forest = std::make_unique<StanForest>(m);
  }

  // Set-up the forest
  void SetTreePrior(double base, double power, double eta, double p_grow,
                   double p_prune, double a_dirichlet, double b_dirichlet,
                   double rho_dirichlet, bool augment, bool dirichlet_bool,
                   double alpha_dirichlet) {
    stan_forest->SetPriorParameters(base, power, eta);
    stan_forest->SetDartParameters(a_dirichlet, b_dirichlet, rho_dirichlet, augment, dirichlet_bool, alpha_dirichlet);
    // Usually, rho_dirichlet = p, and augment = true
  }

  void SetUpForest(size_t p, size_t n, double* X, double* augment_outcome, int* nc,
                   double omega) {
    stan_forest->SetData(p, n, X, augment_outcome, nc);
  }

  // Overload with uniform number of cutpoints per variable.
  void SetUpForest(size_t p, size_t n, double* X, double* augment_outcome,
                   size_t num_cuts, double omega) {
    stan_forest->SetData(p, n, X, augment_outcome, num_cuts);
  }

  // Set IRS mode: 0=off, 1=skip-then-draw, 2=draw-then-decide, 3=uniform.
  void SetIRS(int mode) {
    stan_forest->SetIRS(mode);
  }

  // Set per-observation weights for weighted regression (e.g., b_i^2 in BCF)
  // Note: StanForest does not currently support per-observation weights
  void SetWeights(double* w) {}

  void StartDirichlet() {
    stan_forest->ToggleDart();
  }

  // Update the forest
  void UpdateForest(double sigma,
              bool* accepted,
              Random& rng)
  {
    stan_forest->Draw(sigma, rng, accepted);
  }

  // Predict new outcomes
  void Predict(size_t p, size_t n_test, double* X, double* out) {
    stan_forest->Predict(p, n_test, X, out);
  }

  // IRS: test-time prediction with uniform random routing at NaN splits.
  void Predict(size_t p, size_t n_test, double* X, double* out, Random& random) {
    stan_forest->Predict(p, n_test, X, out, random);
  }

  // Accessors
  inline double GetPrediction(size_t i) {
    return stan_forest->GetFitAt(i);
  }

  inline double* GetPredictions() {
    return stan_forest->GetAllFit();
  }

  std::vector<size_t>& GetVariableInclusionCount() {
    return stan_forest->GetVariableSplitCounts();
  }

  std::vector<double>& GetVariableInclusionProb() {
    return stan_forest->GetSplitProbabilities();
  }

  void UpdateGlobalScaleParameters(string prior_type,
                                   double global_parameter,
                                   double& storage_eta,
                                   Random& random) {
    stan_forest->UpdateGlobalScaleParameters(prior_type, global_parameter, storage_eta, random);
  }
};

#endif

