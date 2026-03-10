#include "Functions.h"

// The function checks if the given leaf node has ANY variables on which it can 
// perform a valid split. It iterates through all the potential variables (stored 
// in the `cutpoints` object) and checks if there is a range of possible cut 
// points for each variable. A valid split is determined by checking if the upper 
// bound index of possible cuts is greater than or equal to the lower bound index.
bool Splittable(Tree& leaf_node, Cutpoints& cutpoints) {
  int lower_bound_index, upper_bound_index;
  bool split_variable_found = false; 
  size_t variable_index = 0;

  // Loop through variables until a split variable is found or all are checked
  while (!split_variable_found && variable_index < cutpoints.p) {

    // Set lower and upper bounds for the possible cut points
    lower_bound_index = 0;
    upper_bound_index = cutpoints.values[variable_index].size() - 1;

    // Determine possible cut points for the current variable
    leaf_node.PossibleCuts(variable_index, 
                           &lower_bound_index, 
                           &upper_bound_index);

    // If a valid range is found, mark the variable as splittable
    if (upper_bound_index >= lower_bound_index) {
      split_variable_found = true;
    }

    // Move to the next variable
    variable_index++;
  }

  return split_variable_found;
}


// Function retrieves the addresses (i.e., pointers to) of leaf nodes that can 
// be split on. It collects all leaf nodes, checks if they are splittable, and 
// removes those that can't be split.
void CollectSplittableLeaves(Tree& tree, std::vector<Tree*>& leaf_vector, 
                            Cutpoints& cutpoints) {
  
  // Collect all leaf nodes.
  tree.CollectLeaves(leaf_vector);

  // Iterate over the collected leaf nodes
  for (size_t i = 0; i < leaf_vector.size(); i++) {
    
    // If the leaf node can't be split, remove it from the vector
    if (!Splittable(*leaf_vector[i], cutpoints)) {
      
      // Erase the node if it can't be split
      leaf_vector.erase(leaf_vector.begin() + i);

      // Adjust index to account for the removed element
      --i;
    }
  }
}


// Function searches for the variables that the given node can split on and 
// stores their indices in split_var. The function checks each variable to see if 
// there is a valid range of cut points. If a variable has a valid range, its 
// index is added to the split_var vector.
void GetSplittableVariables(Tree& leaf, Cutpoints& cutpoints, 
                            std::vector<size_t>& split_var) {

  // Ensure that the vector is empty
  split_var.clear();

  // Create placeholders for the lower and upper bound indices.
  int lower_bound_index, upper_bound_index;

  // Loop through all variables and check whether they can be split on
  for (size_t variable_index = 0; variable_index != cutpoints.p; 
       ++variable_index) {

    // Set lower and upper bounds for possible cut points
    lower_bound_index = 0; 
    upper_bound_index = cutpoints.values[variable_index].size() - 1;

    // Determine the possible range of cut points for the current variable
    leaf.PossibleCuts(variable_index, &lower_bound_index, 
                      &upper_bound_index);

    // If a valid range is found, add the variable index to split_var
    if (upper_bound_index >= lower_bound_index) {
      split_var.push_back(variable_index);
    }
  }
}


// Function returns the probability of a GROW move given the tree's current form.
// The probability of a GROW move is given by the prior, except in the case that 
// the tree is only a stump (i.e., consists of one node) or has leaf nodes which 
// cannot be split on.
// - std::vector<Tree*>& splittable_nodes: A vector to store all the leaf nodes 
//   that can be split.
double GrowProbability(Tree& tree, Cutpoints& cutpoints, TreePrior& tree_prior, 
                       std::vector<Tree*>& splittable_nodes) {

  // Probability of a birth move; to be returned
  double prob_birth; 
  std::vector<Tree*> leaf_nodes; // All the bottom nodes
  tree.CollectLeaves(leaf_nodes);
  
  // Find all splittable bottom nodes
  for (size_t i = 0; i != leaf_nodes.size(); i++) {
    if (Splittable(*leaf_nodes[i], cutpoints)) {
      splittable_nodes.push_back(leaf_nodes[i]);
    }
  }
  
  // Determine the probability of birth based on the number of splittable nodes
  if (splittable_nodes.size() == 0) {  // Are there any bottom nodes to split on?
    prob_birth = 0.0;
  } else {
    if (tree.TreeSize() == 1) {
      prob_birth = 1.0;  // Is there just one node (stump)?
    } else {
      prob_birth = tree_prior.p_GROW;
    }
  }
  
  return prob_birth;
}


// This function computes the sufficient statistics for the left and right 
// children of a node after a potential split on a specified variable (split_var) 
// and cutpoint (cut_val). The sufficient statistics include:
// 1. The number of observations in the left and right partitions (left_count, 
//    right_count).
// 2. The sum of residuals in the left and right partitions (left_sum, right_sum).
void SufficientStatistics(Tree& tree, Tree* target_node, size_t split_var,
                          size_t cut_val, Cutpoints& cutpoints, Data& data,
                          double& left_count, double& left_sum,
                          double& right_count, double& right_sum) {

  // Reset the counts and sums for left and right partitions
  left_count = 0.0;
  left_sum = 0.0;
  right_count = 0.0;
  right_sum = 0.0;

  // Create a placeholder for the row of covariates
  double* current_row;

  // Iterate over all data points
  for (size_t i = 0; i < data.GetN(); ++i) {

    // Retrieve the current data row (covariates) from the dataset
    current_row = data.GetDataRow(i);

    // Find the leaf node in the tree that this observation falls into
    if (target_node == tree.FindLeaf(current_row, cutpoints)) {

      // Get the value of the feature at the specified variable index
      double value_at_var = current_row[split_var];

      // Get the value of the cutpoint for the specified variable and index
      double cutpoint_value = cutpoints.values[split_var][cut_val];

      // Per-observation weight (1.0 when no weights are set)
      double w = data.GetWeight(i);

      // Determine if the data point falls to the left or right of the cutpoint
      if (value_at_var < cutpoint_value) {
        left_count += w;
        left_sum += w * data.residual[i];
      } else {
        right_count += w;
        right_sum += w * data.residual[i];
      }
    }
  }
}


// Compute the count (n) and sum of residuals (∑ res_i) for the left and right 
// child nodes. This function differs from the previous similarly named function 
// in that it specifically handles two child nodes (l and r) resulting from a 
// split and computes the sufficient statistics for these two nodes based on the 
// observations that fall into them.
void SufficientStatistics(Tree& tree, Tree* left_child, Tree* right_child,
                          Cutpoints& cutpoints, Data& data, double& left_count,
                          double& left_sum, double& right_count,
                          double& right_sum) {

  // Initialize counts and sums for the left and right child nodes
  left_count = 0.0;
  left_sum = 0.0;
  right_count = 0.0;
  right_sum = 0.0;

  // Iterate over all observations in the dataset
  for (size_t i = 0; i < data.GetN(); ++i) {

    // Get the current row of covariates
    double* current_row = data.GetDataRow(i);

    // Find the leaf node in the tree that the current observation falls into
    const Tree* current_leaf = tree.FindLeaf(current_row, cutpoints);

    // Per-observation weight (1.0 when no weights are set)
    double w = data.GetWeight(i);

    // If the observation falls into the left child node, update its statistics
    if (current_leaf == left_child) {
      left_count += w;
      left_sum += w * data.GetResidual(i);
    }

    // If the observation falls into the right child node, update its statistics
    if (current_leaf == right_child) {
      right_count += w;
      right_sum += w * data.GetResidual(i);
    }
  }
}


// Computes the log-posterior likelihood of the data given the tree structure and 
// updated variance. This function calculates the posterior log-likelihood based 
// on the number of observations (n), the sum of the residuals (sum_residuals), 
// the standard deviation of the residuals (sigma), and the prior variance (eta).
double LogPostLikelihood(double observation_count, double sum_residuals,
                         double sigma, double prior_variance) {

  // Calculate the variance squared for both the residuals and the prior
  double sigma_squared = sigma * sigma;
  double eta_squared = prior_variance * prior_variance;

  // Calculate the combined variance factor
  double combined_variance = observation_count * eta_squared + sigma_squared;

  // Compute the log-posterior likelihood
  return -0.5 * log(combined_variance) + 
         (eta_squared * sum_residuals * sum_residuals) / 
         (2.0 * sigma_squared * combined_variance);
}


// Calculate the probability that a node grows.
// If there are no good variables to split on, the probability is 0.
// Otherwise, the probability is computed as base / (1 + d)^power, where d is 
// the depth of the node in the tree.
double ProbNodeGrows(Tree& tree, Cutpoints& cutpoints, TreePrior& tree_prior) {

  // Check if the current node can be split based on the available cutpoints
  if (Splittable(tree, cutpoints)) {

    // Calculate the probability of growing the node according to the prior
    return tree_prior.base / 
           pow(1.0 + tree.NodeDepth(), tree_prior.power);

  } else {

    // If no good variables are available for splitting, the probability is 0
    return 0.0;
  }
}

//Helper functions to manage leaf_index_ during stats pass. Replaced old map-based approach.
// Assigns 0..(leaves.size()-1) to each leaf’s leaf_index_.
static inline void AssignLeafIndices(std::vector<Tree*>& leaves) {
  for (size_t i = 0; i < leaves.size(); ++i) {
    leaves[i]->SetLeafIndex(i);
  }
}

// Clears the indices after we’re done (defensive).
static inline void ClearLeafIndices(std::vector<Tree*>& leaves) {
  for (Tree* leaf : leaves) leaf->ResetLeafIndex();
}

// Compute sufficient statistics for all bottom nodes in the tree. 
// This method loops through all the data once, collecting the number of 
// observations and the sum of residuals for each bottom node.
/*
void SufficientStatisticsAllLeaves(Tree& tree, Cutpoints& cutpoints, Data& data, 
                                   std::vector<Tree*>& bottom_nodes, 
                                   std::vector<size_t>& observation_count_vector, 
                                   std::vector<double>& residual_sum_vector) {

  // Pointer to the bottom node for the current observation
  const Tree* current_bottom_node;

  // Index into the vector of the current bottom node
  size_t bottom_node_index;

  // Placeholder for the current row of covariates
  double* current_covariates;

  // Clear the vector of bottom nodes and collect them from the tree
  bottom_nodes.clear();
  tree.CollectLeaves(bottom_nodes);

  // Resize the observation count and residual sum vectors to match the number of 
  // bottom nodes
  size_t num_bottom_nodes = bottom_nodes.size();
  observation_count_vector.resize(num_bottom_nodes);
  residual_sum_vector.resize(num_bottom_nodes);

  // Map to associate each bottom node with its index in the vector
  std::map<const Tree*, size_t> bottom_node_map;
  for (size_t i = 0; i < num_bottom_nodes; ++i) {
    bottom_node_map[bottom_nodes[i]] = i;
    observation_count_vector[i] = 0;
    residual_sum_vector[i] = 0.0;
  }

  // Loop through all observations to collect sufficient statistics
  for (size_t i = 0; i < data.GetN(); ++i) {
    current_covariates = data.GetDataRow(i);

    // Find the bottom node where the current observation falls
    current_bottom_node = tree.FindLeaf(current_covariates, cutpoints);

    // Get the index of the bottom node
    bottom_node_index = bottom_node_map[current_bottom_node];

    // Update the observation count and residual sum for this bottom node
    ++(observation_count_vector[bottom_node_index]);
    residual_sum_vector[bottom_node_index] += data.GetResidual(i);
  }
}
*/

void SufficientStatisticsAllLeaves(
    Tree& tree,
    Cutpoints& cutpoints,
    Data& data,
    std::vector<Tree*>& bottom_nodes,
    std::vector<double>& observation_count_vector,
    std::vector<double>& residual_sum_vector) {

  bottom_nodes.clear();
  tree.CollectLeaves(bottom_nodes);

  // Assign dense indices to leaves once per pass
  AssignLeafIndices(bottom_nodes);

  const size_t L = bottom_nodes.size();
  observation_count_vector.assign(L, 0.0);
  residual_sum_vector.assign(L, 0.0);

  // Fast raw pointers
  const size_t N = data.GetN();
  const size_t P = data.GetP();
  double* Xrow = nullptr;
  double* X = data.GetX();
  double* resid = data.GetResidual();
  double* wts = data.GetWeights();

  // Main scan
  for (size_t i = 0; i < N; ++i) {
    Xrow = X + i * P;  // contiguous row
    Tree* leaf = tree.FindLeaf(Xrow, cutpoints);  // iterative inline
    const size_t idx = leaf->GetLeafIndex();      // O(1)
    double w = wts ? wts[i] : 1.0;
    observation_count_vector[idx] += w;
    residual_sum_vector[idx] += w * resid[i];
  }

  // Defensive cleanup (cheap, keeps invariants simple)
  ClearLeafIndices(bottom_nodes);
}


// New update function for scale mixture set-up
void DrawMuAllLeaves(Tree& tree, Cutpoints& cutpoints, Data& data,
                     TreePrior& tree_prior, double sigma, const double& omega,
                     ScaleMixture& scale_mixture, Random& random) {

  // Vectors to hold bottom nodes, their respective observation counts, and
  // sum of residuals
  std::vector<Tree*> bottom_nodes;
  std::vector<double> observation_counts;
  std::vector<double> residual_sums;

  // Compute sufficient statistics for all bottom nodes in the tree
  SufficientStatisticsAllLeaves(tree, cutpoints, data, bottom_nodes, 
                                observation_counts, residual_sums);

  // Draw and set the mu (parameter) for each bottom node
  for (std::vector<Tree*>::size_type i = 0; i != bottom_nodes.size(); ++i) {
    scale_mixture.Propose(bottom_nodes[i]->GetParameters(), 
                          tree.GetParameters(), 
                          residual_sums[i], observation_counts[i], 
                          sigma, 
                          omega,
                          random);
  }
}


// Performs a full update of both global and local parameters for all bottom 
// nodes in the tree. This includes updating the sufficient statistics, global 
// parameters, and proposing new values for the local parameters.
void FullUpdate(Tree& tree, Cutpoints& cutpoints, Data& data,
                const double& sigma,
                const double& omega,
                ScaleMixture& scale_mixture,
                Random& random) {

  // Vectors to hold leaf nodes, their respective observation counts, and
  // sum of residuals
  std::vector<Tree*> leaf_nodes;
  std::vector<Parameters*> leaf_parameters;
  std::vector<double> observation_counts;
  std::vector<double> residual_sums;

  // Compute sufficient statistics for all bottom nodes in the tree
  SufficientStatisticsAllLeaves(tree, cutpoints, data, leaf_nodes, 
                                observation_counts, residual_sums);

  // First, perform an update of all the global parameters
  if (scale_mixture.GlobalRequired()) {

    // Collect the addresses of the parameters objects of the leaf nodes
    for (Tree* leaf : leaf_nodes) {
      leaf_parameters.push_back(&leaf->GetParameters());
    }

    // Perform the global update step
    scale_mixture.GlobalUpdate(tree.GetParameters(), leaf_parameters, 
                               sigma, omega, random);
  }

  // Update the local parameters and step heights for each bottom node
  for (std::vector<Tree*>::size_type i = 0; i != leaf_nodes.size(); ++i) {
    scale_mixture.Propose(leaf_nodes[i]->GetParameters(), 
                          tree.GetParameters(), 
                          residual_sums[i], observation_counts[i], 
                          sigma, omega, random);
  }
}

// Draws a single mu (parameter) value from the posterior distribution
double DrawMuOneLeave(double n, double sum_residuals, double prior_variance,
                      double sigma, Random& random) {

  // Compute the square of the residual standard deviation
  double sigma_squared = sigma * sigma;

  // Calculate b, the weight based on the number of observations and residual variance
  double b = n / sigma_squared;

  // Calculate a, the weight based on the prior variance
  double a = 1.0 / (prior_variance * prior_variance);

  // Return the drawn mu value from the posterior distribution
  // The first term is the mean of the posterior distribution
  // The second term adds noise sampled from a normal distribution 
  // scaled by the posterior variance
  return (sum_residuals / sigma_squared) / (a + b) + 
         random.normal() / sqrt(a + b);
}


// Print the tree and include the size of each leaf (bottom node) and the residual sum
void PrintTreeWithSizes(Tree& tree, Cutpoints& cutpoints, Data& data) {
  std::vector<Tree*> bottom_nodes;                   // Vector to store bottom nodes
  std::vector<double> observation_count_vector;      // Vector to store the size of each bottom node
  std::vector<double> residual_sum_vector;           // Vector to store residual sums

  // Compute the sufficient statistics for all bottom nodes
  SufficientStatisticsAllLeaves(tree, cutpoints, data, bottom_nodes, 
                                observation_count_vector, residual_sum_vector);

  // Create a map from bottom node to its index in the vector for quick lookup
  std::map<const Tree*, size_t> bottom_node_map;
  for (size_t i = 0; i < bottom_nodes.size(); ++i) {
    bottom_node_map[bottom_nodes[i]] = i;
  }

  // Start recursive tree printing, including sizes and residual sums of nodes
  PrintTreeWithSizesRecursive(&tree, cutpoints, data, bottom_node_map, 
                              observation_count_vector, residual_sum_vector, 0);
}


// Recursive function to print tree structure with node sizes and residual sums
void PrintTreeWithSizesRecursive(Tree* node, Cutpoints& cutpoints, Data& data,
                                 std::map<const Tree*, size_t>& bottom_node_map,
                                 std::vector<double>& observation_count_vector,
                                 std::vector<double>& residual_sum_vector,
                                 int depth) {
  // Indentation based on depth to visualize the tree structure
  for (int i = 0; i < depth; ++i) {
    cout << "  ";  // Two spaces per depth level
  }

  // Print node ID
  cout << "Node " << node->NodeID() << ": ";

  // Print split information if it's not a leaf node
  if (node->GetLeft() != nullptr || node->GetRight() != nullptr) {
    cout << " (Splitting variable: " << node->GetSplitVar() 
              << ", Cut value: " 
              << cutpoints.values[node->GetSplitVar()][node->GetCutVal()]
              << ")";
  } else {
    // Print parameter, node size, and residual sum for leaf nodes
    size_t node_size = bottom_node_map.count(node) 
                       ? observation_count_vector[bottom_node_map[node]] 
                       : 0;
    double node_residual_sum = bottom_node_map.count(node) 
                               ? residual_sum_vector[bottom_node_map[node]] 
                               : 0.0;
    cout << " (Parameter: " << node->GetParameters(0) 
              << ", Node size: " << node_size 
              << ", Mean residual: " 
              << (node_size > 0 ? node_residual_sum / node_size : 0.0)
              << ")";
  }

  cout << "\n";

  // Recursively print left and right children if they exist
  if (node->GetLeft() != nullptr) {
    PrintTreeWithSizesRecursive(node->GetLeft(), cutpoints, data, 
                                bottom_node_map, observation_count_vector, 
                                residual_sum_vector, depth + 1);
  }

  if (node->GetRight() != nullptr) {
    PrintTreeWithSizesRecursive(node->GetRight(), cutpoints, data, 
                                bottom_node_map, observation_count_vector, 
                                residual_sum_vector, depth + 1);
  }
}


// Tree ratio for the GROW move. Tree ratio for the PRUNE move is the inverse.
double LogTreeRatio_PRUNE(Tree* node, const TreePrior& tree_prior, 
                          Cutpoints& cutpoints) {

  int depth = node->NodeDepth();

  // Compute the common denominators
  double depth_power = std::pow(1.0 + depth, tree_prior.power);
  double depth_plus_1_power = std::pow(2.0 + depth, tree_prior.power);

  // Compute p_depth and p_depth+1
  double PG = tree_prior.base / depth_power;
  double PGl = Splittable(*node->GetLeft(), cutpoints) 
               ? tree_prior.base / depth_plus_1_power : 0;
  double PGr = Splittable(*node->GetRight(), cutpoints) 
               ? tree_prior.base / depth_plus_1_power : 0;

  // Compute the log of the tree ratio
  double log_tree_ratio = std::log((1 - PG) / (PG * (1 - PGr) * (1 - PGl)));

  return log_tree_ratio;
}

// Tree ratio for the GROW move. Tree ratio for the PRUNE move is the inverse.
double LogTreeRatio_GROW(size_t depth, const TreePrior& tree_prior) {
  // Compute the common denominators
  double depth_power = std::pow(1.0 + depth, tree_prior.power);
  double depth_plus_1_power = std::pow(2.0 + depth, tree_prior.power);

  // Compute p_depth and p_depth+1
  double p_depth = tree_prior.base / depth_power;
  double p_depth_plus_1 = tree_prior.base / depth_plus_1_power;

  // Log of p_depth
  double log_p_depth = std::log(p_depth);

  // Log of (1 - p_depth+1)
  double log_one_minus_p_depth_plus_1 = std::log(1.0 - p_depth_plus_1);

  // Log of (1 - p_depth)
  double log_one_minus_p_depth = std::log(1.0 - p_depth);

  // Compute the log of the tree ratio
  double log_tree_ratio = log_p_depth + 
                          2 * log_one_minus_p_depth_plus_1 - 
                          log_one_minus_p_depth;

  return log_tree_ratio;
}

// Move ratio for the GROW move. Move ratio for the PRUNE move is the inverse.
double LogMoveRatio(size_t number_of_nogs, size_t number_of_leaves, 
                    double prune_prob, double grow_prob) {

  return std::log(prune_prob) - std::log(number_of_nogs) - 
         (std::log(grow_prob) - std::log(number_of_leaves));
}

