#include "StanForest.h"


// Constructor: initialise all members; arrays are allocated lazily in SetData.
StanForest::StanForest(size_t num_trees_init)
  : num_trees(num_trees_init), trees(num_trees_init), prior_info(),
    p(0), n(0), x(nullptr), y(nullptr), cutpoints(),
    all_fit(nullptr), residuals(nullptr), tree_fit_temp(nullptr), data_info(),
    use_dart(false), dart_active(false), use_augmentation(false),
    fixed_theta(false), dart_a(0.0), dart_b(0.0), dart_rho(0.0),
    dart_theta(1.0) {}

// Destructor: release heap-allocated arrays.
StanForest::~StanForest()
{
  if (all_fit)       delete[] all_fit;
  if (residuals)     delete[] residuals;
  if (tree_fit_temp) delete[] tree_fit_temp;
}

// Change the number of trees and recompute the current forest fit.
void StanForest::SetNumTrees(size_t new_num_trees)
{
  trees.resize(new_num_trees);
  num_trees = trees.size();
  if (all_fit && (cutpoints.size() == p)) Predict(p, n, x, all_fit);
}

// Deep-copy new_cutpoints into the internal cutpoints.
void StanForest::SetCutpointMatrix(CutpointMatrix& new_cutpoints)
{
  size_t num_predictors = new_cutpoints.size();
  cutpoints.resize(num_predictors);
  for (size_t i = 0; i < num_predictors; i++) {
    size_t num_cuts = new_cutpoints[i].size();
    cutpoints[i].resize(num_cuts);
    for (size_t j = 0; j < num_cuts; j++)
      cutpoints[i][j] = new_cutpoints[i][j];
  }
}

// Set the feature matrix and working-response, build cutpoints, and allocate
// working arrays.  nc == uniform cut count per variable (scalar overload).
void StanForest::SetData(size_t p, size_t n, double* x, double* y,
                         size_t num_cuts_per_var)
{
  int* nc = new int[p];
  for (size_t i = 0; i < p; ++i) nc[i] = static_cast<int>(num_cuts_per_var);
  this->SetData(p, n, x, y, nc);
  delete[] nc;
}

void StanForest::SetData(size_t p, size_t n, double* x, double* y, int* nc)
{
  this->p = p; this->n = n; this->x = x; this->y = y;
  if (cutpoints.empty()) MakeCutpoints(p, n, &x[0], cutpoints, nc);

  if (all_fit) delete[] all_fit;
  all_fit = new double[n];
  Predict(p, n, x, all_fit);

  if (residuals) delete[] residuals;
  residuals = new double[n];

  if (tree_fit_temp) delete[] tree_fit_temp;
  tree_fit_temp = new double[n];

  data_info.n         = n;
  data_info.p         = p;
  data_info.X         = &x[0];
  data_info.residuals = residuals;

  variable_split_counts.clear();
  split_probabilities.clear();
  for (size_t j = 0; j < p; j++) {
    variable_split_counts.push_back(0);
    split_probabilities.push_back(1.0 / static_cast<double>(p));
  }
}

// Compute the forest prediction for all n observations.
void StanForest::Predict(size_t p, size_t n, double* x, double* fp)
{
  double* temp = new double[n];

  for (size_t j = 0; j < n; j++) fp[j] = 0.0;
  for (size_t j = 0; j < num_trees; j++) {
    FitTree(trees[j], cutpoints, p, n, x, temp);
    for (size_t k = 0; k < n; k++) fp[k] += temp[k];
  }

  delete[] temp;
}

// Perform one full MCMC sweep: for each tree, update its structure and leaf
// parameters, then optionally update the DART split probabilities.
bool StanForest::Draw(double sigma, Random& random, bool* accept)
{
  for (size_t j = 0; j < num_trees; j++) {

    // Remove the current contribution of tree j from all_fit.
    FitTree(trees[j], cutpoints, p, n, x, tree_fit_temp);
    for (size_t k = 0; k < n; k++) {
      all_fit[k]  -= tree_fit_temp[k];
      residuals[k] = y[k] - all_fit[k];
    }

    // Propose a birth or death for tree j.
    accept[j] = BirthDeathStep(trees[j], cutpoints, data_info, prior_info,
                                sigma, variable_split_counts,
                                split_probabilities, use_augmentation, random);

    // Draw new leaf parameters for tree j.
    DrawAllLeafMeans(trees[j], cutpoints, data_info, prior_info, sigma, random);

    // Add the updated contribution of tree j back to all_fit.
    FitTree(trees[j], cutpoints, p, n, x, tree_fit_temp);
    for (size_t k = 0; k < n; k++) all_fit[k] += tree_fit_temp[k];
  }

  if (dart_active) {
    DrawSplitProbabilities(variable_split_counts, log_split_probabilities,
                           dart_theta, random);
    DrawSparsityParameter(fixed_theta, dart_theta, log_split_probabilities,
                          dart_a, dart_b, dart_rho, random);
    for (size_t j = 0; j < p; j++)
      split_probabilities[j] = std::exp(log_split_probabilities[j]);
  }

  return true;
}

void StanForest::Print()
{
  cout << "*****StanForest object:\n";
  cout << "num_trees: " << num_trees << std::endl;
  cout << "prior and mcmc info:\n";
  if (use_dart) {
    cout << "*****dart prior (On):\n";
    cout << "dart_a: "   << dart_a   << std::endl;
    cout << "dart_b: "   << dart_b   << std::endl;
    cout << "dart_rho: " << dart_rho << std::endl;
    cout << "use_augmentation: " << use_augmentation << std::endl;
  } else {
    cout << "*****dart prior (Off):\n";
  }
  if (p) cout << "data set: n, p: " << n << ", " << p << std::endl;
  else   cout << "data not set\n";
}

void StanForest::UpdateGlobalScaleParameters(string prior_type,
                                             double global_parameter,
                                             double& storage_eta,
                                             Random& random)
{
  if (prior_type == "standard-halfcauchy") {
    this->UpdateHalfCauchyScale(global_parameter, storage_eta, random);
  } else if (prior_type == "standard-halfnormal") {
    return;
  } else {
    return;
  }
}

void StanForest::UpdateHalfCauchyScale(double global_parameter,
                                       double& storage_eta,
                                       Random& random)
{
  double sum_sq    = 0.0;
  double num_leaves = 0.0;

  for (size_t j = 0; j < num_trees; j++) {
    std::vector<StanTree*> leaves;
    trees[j].CollectLeaves(leaves);

    for (auto* leaf : leaves) {
      double step_height = leaf->GetStepHeight();
      sum_sq    += (step_height * step_height) / (prior_info.eta * prior_info.eta);
      num_leaves += 1.0;
    }
  }

  double shape = 0.5 * (1.0 + num_leaves);
  double rate  = 0.5 * (1.0 + sum_sq);

  double aux = random.gamma(shape, 1.0) / rate;
  prior_info.eta = global_parameter /
                   (std::sqrt(aux) * std::sqrt(static_cast<double>(num_trees)));
  storage_eta = prior_info.eta;
}

