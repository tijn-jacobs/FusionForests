#include "Forest.h"

// Return the number of splits for each covariate as a reference
std::vector<size_t>& Forest::GetVariableInclusionCount() {
  return variable_inclusion_count;
}

// Return the covariate inclusion probabilities as a reference
std::vector<double>& Forest::GetVariableInclusionProb() {
  return variable_inclusion_prob;
}

// Set the tree prior parameters for the forest
void Forest::SetTreePrior(double base, double power, double eta, 
                          double p_GROW, double p_PRUNE) {
  tree_prior.base = base;
  tree_prior.power = power;
  tree_prior.eta = eta; // eta is the fixed variance parameter for standard BART
  tree_prior.p_GROW = p_GROW;
  tree_prior.p_PRUNE = p_PRUNE;
}

// Return the prediction for the i-th observation
double Forest::GetPrediction(size_t i) const {
  return predictions[i];
}

// Return the pointer to the predictions vector
double* Forest::GetPredictions( ) const {
  return predictions;
}

// Getter for the pointer to the trees vector
std::vector<Tree>* Forest::GetTreesPointer() {
    return &trees;
}

// Initialize data and related structures within the Forest class.
// Sets dimensions, pointers to input data, allocates memory for working
// variables (predictions, residuals, temporary_fit), initializes cutpoints, 
// sets up the Data object, and initializes feature usage counters and probs.
// Initialize data and related structures within the Forest class.
// Sets dimensions, pointers to input data, allocates memory for working
// variables (predictions, residuals, temporary_fit), initializes cutpoints,
// sets up the Data object, and initializes feature usage counters and probs.
void Forest::SetUpForest(size_t p, size_t n, double *x, double *y,
                         int *nc, double omega) {

  // Set the dimensions and pointers to the input data
  this->p = p;
  this->n = n;
  this->x = x;
  this->y = y;

  // Set omega
  this->omega = omega;

  // Initialize cutpoints if not set
  if (cutpoints.p == 0)
    cutpoints.SetCutpoints(p, n, x, nc);

  // Allocate memory for predictions
  if (predictions != nullptr)
    delete[] predictions;
  predictions = new double[n];

  // Allocate memory for residuals
  if (residual != nullptr)
    delete[] residual;
  residual = new double[n];

  // Allocate memory for temporary_fit
  if (temporary_fit != nullptr)
    delete[] temporary_fit;
  temporary_fit = new double[n];

  // Set up data in the Data object
  data.SetN(n);
  data.SetP(p);
  data.SetX(x);
  data.SetY(y);
  data.residual = residual;

  // Initialize predictions and compute initial fits
  Predict(p, n, x, predictions);

  // Initialize feature usage counters and probabilities
  variable_inclusion_count.resize(p, 0);
  variable_inclusion_prob.resize(p, 1.0 / static_cast<double>(p));
}


// Predicts the output for a given set of covariates using the entire forest
void Forest::Predict(size_t p, size_t n, double *X, double *predictions) {

  // Temporary storage for predictions from a single tree
  double *single_tree_prediction = new double[n];

  // Initialize the predictions array to zero
  for (size_t j = 0; j < n; j++) {
    predictions[j] = 0.0;
  }

  // Aggregate predictions from all trees in the forest
  for (size_t j = 0; j < m; j++) {

    // Evaluate the current tree to get predictions for all observations
    trees[j].EvaluateTree(cutpoints, p, n, X, data, single_tree_prediction);

    // Add the current tree's predictions to the cumulative predictions
    for (size_t i = 0; i < n; i++) {
      predictions[i] += single_tree_prediction[i];
    }
  }

  // Free the memory allocated for single_tree_prediction
  delete[] single_tree_prediction;
}



// Performs a single iteration of the MCMC process, updating the trees in the forest.
void Forest::UpdateForest(const double& sigma, ScaleMixture& scale_mixture, bool reversible,
                          const size_t& delayed_proposal,
                          Random& random, bool* accepted) {
  
  // Loop over each tree in the forest
  for (size_t j = 0; j < m; j++) {

    // Calculate the current contribution of tree `j` to the predictions
    trees[j].EvaluateTree(cutpoints, p, n, x, data, temporary_fit);

    // Subtract the contribution of tree `j` from the overall predictions 
    for (size_t k = 0; k < n; k++) {
      predictions[k] -= temporary_fit[k];
      
      // Update the residuals based on the new predictions
      data.residual[k] = data.GetOutcome(k) - predictions[k];
    }

    // Propose a new structure for tree `j` by making a move adn propose new parameters
    accepted[j] = ReversibleJump(trees[j], cutpoints, data, tree_prior, sigma, omega,
                                 variable_inclusion_count, 
                                 variable_inclusion_prob, delayed_proposal, 
                                 reversible, scale_mixture, random);

    // Draw new values for all the terminal nodes' parameters (mu) of tree `j`
    DrawMuAllLeaves(trees[j], cutpoints, data, tree_prior, sigma, omega,
                    scale_mixture, random);

    // Update the variables of all the leaf node parameters
    FullUpdate(trees[j], cutpoints, data, sigma, omega, scale_mixture, random);

    // Re-evaluate the updated tree to get its new contribution to the predictions
    trees[j].EvaluateTree(cutpoints, p, n, x, data, temporary_fit);

    // Add the updated contribution of tree `j` back to the overall predictions
    for (size_t k = 0; k < n; k++) {
      predictions[k] += temporary_fit[k];
    }
  }
}

// Prints a specific tree
void Forest::PrintForest(size_t tree_index) {

  if (tree_index == std::numeric_limits<size_t>::max()) {
    // Print the entire forest if no specific tree index is provided
    for (size_t i = 0; i < trees.size(); ++i) {
      cout << "Tree " << i + 1 << ":\n";
      PrintTreeWithSizes(trees[i], cutpoints, data);
      cout << endl;
    }
  } else {
    // Ensure the provided index is valid
    if (tree_index < trees.size()) {
      cout << "Tree " << tree_index + 1 << ":\n";
      PrintTreeWithSizes(trees[tree_index], cutpoints, data);
      cout << endl;
    } else {
      cout << "Error: Tree index out of range.\n";
      return;
    }
  }

  cout << endl;
}

