
#ifndef GUARD_StanForest_h
#define GUARD_StanForest_h

#include <ctime>

#include "StanTree.h"
#include "StanTreeFunctions.h"
#include "Info.h"
#include "StanForestFunctions.h"
#include "StanBirthDeath.h"

// An ensemble of StanTree objects implementing standard BART and DART
// (Dirichlet Additive Regression Trees).  StanForest manages the trees,
// cutpoints, MCMC state, and DART hyperparameters.
class StanForest {
public:

  // Friend declaration for the birth-death step function.
  friend bool BirthDeathStep(StanTree& tree, CutpointMatrix& cutpoints,
                             DataInfo& data_info, PriorInfo& prior_info,
                             double sigma,
                             std::vector<size_t>& variable_split_counts,
                             std::vector<double>& split_probabilities,
                             bool use_augmentation, Random& random);

  // Constructor / Destructor
  explicit StanForest(size_t num_trees);
  ~StanForest();

  // Getters and Setters

  size_t GetNumTrees() { return num_trees; }
  void   SetNumTrees(size_t new_num_trees);

  // Set the feature matrix, working-response vector, and cutpoints.
  void SetData(size_t p, size_t n, double* x, double* y, size_t nc = 100);
  void SetData(size_t p, size_t n, double* x, double* y, int* nc);

  void SetPriorInfo(PriorInfo& info)  { this->prior_info = info; }
  void SetPriorParameters(double base, double power, double eta) {
    prior_info.base  = base;
    prior_info.power = power;
    prior_info.eta   = eta;
  }

  // Configure the DART (Dirichlet) prior parameters.
  // If dart_theta_init == 0, the sparsity parameter is treated as random;
  // otherwise it is held fixed at dart_theta_init.
  void SetDartParameters(double dart_a_init, double dart_b_init,
                         double dart_rho_init, bool use_augmentation_init,
                         bool use_dart_init, double dart_theta_init = 0.0) {
    this->dart_a            = dart_a_init;
    this->dart_b            = dart_b_init;
    this->dart_rho          = dart_rho_init;
    this->use_augmentation  = use_augmentation_init;
    this->use_dart          = use_dart_init;
    if (dart_theta_init == 0.0) {
      this->fixed_theta  = false;
      this->dart_theta   = 1.0;
    } else {
      this->fixed_theta  = true;
      this->dart_theta   = dart_theta_init;
    }
  }

  // Toggle the DART split-probability updates on/off.
  void ToggleDart() { dart_active = !dart_active; }

  StanTree&       GetTree(size_t i)         { return trees[i]; }
  CutpointMatrix& GetCutpointMatrix()       { return cutpoints; }
  void            SetCutpointMatrix(CutpointMatrix& new_cutpoints);

  std::vector<size_t>& GetVariableSplitCounts() { return variable_split_counts; }
  std::vector<double>& GetSplitProbabilities()  { return split_probabilities; }
  double               GetDartTheta()           { return dart_theta; }

  // Tree modification wrappers (by tree index and node id)
  void Birth(size_t tree_index, size_t nid, size_t split_var, size_t cut_val,
             double left_step_height, double right_step_height) {
    trees[tree_index].Birth(nid, split_var, cut_val,
                            left_step_height, right_step_height);
  }
  void Death(size_t tree_index, size_t nid, double new_step_height) {
    trees[tree_index].Death(nid, new_step_height);
  }

  void Print();
  void Clear() {
    for (size_t i = 0; i < trees.size(); i++) trees[i].Clear();
  }

  // Compute the forest prediction for all n observations and store in fp.
  void Predict(size_t p, size_t n, double* x, double* fp);

  // Perform one full MCMC update (birth-death + leaf parameter draws).
  bool Draw(double sigma, Random& random, bool* accept);

  // Return the current forest fitted value for observation i.
  double GetFitAt(size_t i) { return all_fit[i]; }
  double* GetAllFit()       { return all_fit; }

  // Update the global scale parameter of the prior (horseshoe / half-Cauchy).
  void UpdateGlobalScaleParameters(string prior_type,
                                   double global_parameter,
                                   double& storage_eta,
                                   Random& random);
  void UpdateHalfCauchyScale(double global_parameter, double& storage_eta,
                             Random& random);

protected:
  size_t num_trees;              // Number of trees in the ensemble
  std::vector<StanTree> trees;   // The trees
  PriorInfo prior_info;          // Prior and MCMC tuning parameters
  // Data dimensions and pointers
  size_t p, n;
  double* x;                     // Feature matrix (column-major, p x n)
  double* y;                     // Working response (length n)
  CutpointMatrix cutpoints;      // Cutpoints[v][c]: c-th cutpoint for variable v
  // Working arrays
  double* all_fit;               // Current forest fitted values (length n)
  double* residuals;             // Working residuals (length n)
  double* tree_fit_temp;         // Temporary per-tree fit buffer (length n)
  DataInfo data_info;            // Thin wrapper pointing into the arrays above
  // DART state
  bool use_dart;                 // Whether a Dirichlet prior is used
  bool dart_active;              // Whether DART split-probability updates are on
  bool use_augmentation;         // Whether data augmentation is used
  bool fixed_theta;              // Whether dart_theta is held fixed
  double dart_a;                 // Beta prior shape parameter a
  double dart_b;                 // Beta prior shape parameter b
  double dart_rho;               // Dirichlet concentration scaling parameter
  double dart_theta;             // Current Dirichlet concentration parameter
  std::vector<size_t> variable_split_counts; // Per-variable split counts
  std::vector<double> split_probabilities;   // Current split probabilities
  std::vector<double> log_split_probabilities; // Log of split probabilities
};

#endif

