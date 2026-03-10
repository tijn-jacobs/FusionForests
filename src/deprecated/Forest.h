// The Forest class represents a collection (ensemble) of trees in a Bayesian 
// Additive Regression Tree (BART) model. It is designed to manage the collection 
// of trees, their predictions, and perform updates during the MCMC process.

// Key components of the Forest class:
// - General properties: These include the number of trees (m), a vector of Tree 
//   objects, and prior information (tree_prior). The class also stores feature 
//   and observation data, including pointers to the input features (x) and 
//   response variable (y), as well as working variables for predictions and 
//   residuals.
// - Data properties: The dimensions of the input data (p and n), pointers to the 
//   input and target data, and an object of class Cutpoints to manage split points 
//   for tree nodes.
// - Working variables: Arrays for storing predictions (predictions), residuals 
//   (residual), and temporary storage for individual tree predictions during 
//   updates (temporary_fit).
// - Feature inclusion tracking: The variable_inclusion_count tracks how often each 
//   feature is used in splits across the trees, and variable_inclusion_prob manages 
//   probabilities for feature inclusion.

// Constructors and Destructor:
// - Default constructor initializes a forest with 200 trees and sets all pointers 
//   and data structures to null or empty values.
// - Parameterized constructor allows setting the number of trees (m) at the time 
//   of instantiation.
// - Copy constructor ensures deep copying of dynamically allocated memory, such as 
//   predictions, residual, and temporary_fit arrays.
// - Destructor ensures proper memory management by freeing any allocated memory.

// Core methods include:
// - SetUpForest: Initializes the forest with the given data and sets up the 
//   necessary data structures like predictions, residuals, and cutpoints. It also 
//   manages the initialization of feature inclusion tracking.
// - SetTreePrior: Sets the prior parameters for tree growth and pruning.
// - Predict: Predicts outcomes for the input data by evaluating each tree in the 
//   forest and aggregating the predictions.
// - UpdateForest: Performs a single iteration of MCMC, updating the structure and 
//   parameters of the trees in the forest, recalculating predictions, and updating 
//   the residuals.
// - PrintForest: Prints either the entire forest or a specific tree based on the 
//   provided index.

// This class is essential for managing the ensemble of decision trees and their 
// interaction with the data. It provides methods to perform model fitting via MCMC, 
// predict outputs, and track feature usage across the ensemble. The class also 
// ensures proper memory management and allows flexible control over the number of 
// trees and their parameters.




#ifndef GUARD_Forest_h
#define GUARD_Forest_h

#include <vector>
using std::vector; // DOUBLE?!?!?

#include "TreeModifications.h"



class Forest {

  // General properties
  size_t m;                          // Number of trees in the forest
  double omega;                      // Adjust the prior to the forest
  std::vector<Tree> trees;           // Vector of trees in the forest
  TreePrior tree_prior;              // Prior information and MCMC settings

  // Data properties
  size_t p;                          // Number of features (covariates)
  size_t n;                          // Number of observations
  double* x;                         // Pointer to input data (pxn column-major)
  double* y;                         // Pointer to response variable (nx1)
  Cutpoints cutpoints;               // Information on cutpoints for splitting
  Data data;                         // Data object containing dataset and residuals

  // Working variables
  double* predictions;               // Predictions of the forest (nx1)
  double* residual;                  // Residuals (y - f(x)) for each observation
  double* temporary_fit;             // Temporary storage for tree predictions

  // Variables related to feature selection
  std::vector<size_t> variable_inclusion_count; // Number of splits per feature
  std::vector<double> variable_inclusion_prob, log_vip;  // Feature inclusion probabilities


public:

  // Friend function declaration for ReversibleJump
  friend bool ReversibleJump(Tree& tree, Cutpoints& cutpoints, Data& data, 
                             TreePrior& tree_prior, const double& sigma, 
                             const double& omega,
                             std::vector<size_t>& nodes, 
                             std::vector<double>& variable_inclusion_prob,
                             const size_t& delayed_proposal,
                             bool reversible, ScaleMixture& scale_mixture,
                             Random& random);

  // Constructor
  Forest(size_t im)
    : m(im), trees(m), tree_prior(), p(0), n(0), x(nullptr), y(nullptr),
      cutpoints(), data(), predictions(nullptr), residual(nullptr),
      temporary_fit(nullptr) {}

  // Destructor
  ~Forest() {
    if (predictions != nullptr) delete[] predictions;
    if (residual != nullptr) delete[] residual;
    if (temporary_fit != nullptr) delete[] temporary_fit;
  }

  // Setters and Getters
  void SetUpForest(size_t p, size_t n, double *x, double *y, int *nc,
                  double omega);
  void SetTreePrior(TreePrior& _tree_prior);
  void SetTreePrior(double _base, double _power, double _eta, double _p_GROW, 
                    double _p_PRUNE);
  std::vector<size_t>& GetVariableInclusionCount();
  std::vector<double>& GetVariableInclusionProb();
  double GetPrediction(size_t i) const;
  double* GetPredictions() const;
  std::vector<Tree>* GetTreesPointer();
  size_t GetM() const { return m; }; 
  // const vector<double>& GetVariableInclusionProb() const { return variable_inclusion_prob; }
  // const vector<size_t>& GetVariableInclusionCount() const { return variable_inclusion_count; }

  

  // Set per-observation weights for weighted regression (e.g., b_i^2 in BCF)
  void SetWeights(double* w) { data.SetWeights(w); }

  // Methods
  void Predict(size_t p, size_t n, double *X, double *predictions);
  void UpdateForest(const double& sigma, ScaleMixture& scale_mixture, bool reversible,
                    const size_t& delayed_proposal,
                    Random& random, bool* accepted);
  void PrintForest(size_t tree_index = std::numeric_limits<size_t>::max());

};



#endif // GUARD_Forest_h
