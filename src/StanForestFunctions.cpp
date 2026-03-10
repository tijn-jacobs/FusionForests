#include "StanForestFunctions.h"


// Numerically stable log-sum-exp: log( sum_i exp(v[i]) ).
// Declared in Prerequisites.h; used by Random::log_dirichlet and
// DrawSparsityParameter.
double LogSumExp(std::vector<double>& v)
{
  double max_val = v[0];
  for (size_t i = 0; i < v.size(); i++)
    if (v[i] > max_val) max_val = v[i];

  double sum = 0.0;
  for (size_t i = 0; i < v.size(); i++)
    sum += std::exp(v[i] - max_val);

  return max_val + std::log(sum);
}


// Build a CutpointMatrix from the feature matrix x (column-major, n x p).
// If nc != nullptr, nc[v] uniform cutpoints are created for each predictor v.
// If nc == nullptr, the unique observed values are used as cutpoints.
void MakeCutpoints(size_t p, size_t n, double* x, CutpointMatrix& cutpoints,
                   int* nc)
{
  cutpoints.resize(p);

  if (nc != nullptr) {
    // Uniform cutpoints: find min/max per predictor and build a grid.
    std::vector<double> min_val(p,  std::numeric_limits<double>::infinity());
    std::vector<double> max_val(p, -std::numeric_limits<double>::infinity());

    for (size_t i = 0; i < p; i++) {
      for (size_t j = 0; j < n; j++) {
        double value = *(x + p * j + i);
        if (value < min_val[i]) min_val[i] = value;
        if (value > max_val[i]) max_val[i] = value;
      }
    }

    for (size_t i = 0; i < p; i++) {
      size_t num_cuts = nc[i];
      double delta    = (max_val[i] - min_val[i]) / (num_cuts + 1.0);
      cutpoints[i].resize(num_cuts);
      for (size_t j = 0; j < num_cuts; j++) {
        cutpoints[i][j] = min_val[i] + (j + 1) * delta;
      }
    }

  } else {
    // Empirical cutpoints: unique sorted observed values per predictor.
    for (size_t i = 0; i < p; i++) {
      std::vector<double> observed(n);
      for (size_t j = 0; j < n; j++)
        observed[j] = *(x + p * j + i);

      std::sort(observed.begin(), observed.end());
      auto last = std::unique(observed.begin(), observed.end());
      observed.erase(last, observed.end());
      cutpoints[i] = observed;
    }
  }
}

// Return the birth probability and fill splittable_leaves with all leaf nodes
// that can be split.
double GetBirthProbability(StanTree& tree, CutpointMatrix& cutpoints,
                           PriorInfo& prior_info,
                           std::vector<StanTree*>& splittable_leaves)
{
  std::vector<StanTree*> all_leaves;
  tree.CollectLeaves(all_leaves);

  for (size_t i = 0; i < all_leaves.size(); i++) {
    if (CanSplit(all_leaves[i], cutpoints))
      splittable_leaves.push_back(all_leaves[i]);
  }

  if (splittable_leaves.empty()) return 0.0;
  if (tree.TreeSize() == 1) return 1.0; // Single-node tree must grow
  return prior_info.prob_birth;
}

// Compute sufficient statistics for the two children of a proposed split.
void GetSufficientStatistics(StanTree& tree, StanTree* target_leaf,
                             size_t split_var, size_t cut_val,
                             CutpointMatrix& cutpoints, DataInfo& data_info,
                             size_t& left_count, double& left_sum,
                             size_t& right_count, double& right_sum)
{
  left_count = 0;  left_sum  = 0.0;
  right_count = 0; right_sum = 0.0;

  for (size_t i = 0; i < data_info.n; i++) {
    double* x_row = data_info.X + i * data_info.p;
    if (target_leaf == tree.FindLeaf(x_row, cutpoints)) {
      if (x_row[split_var] < cutpoints[split_var][cut_val]) {
        left_count++;
        left_sum += data_info.residuals[i];
      } else {
        right_count++;
        right_sum += data_info.residuals[i];
      }
    }
  }
}

// Compute sufficient statistics for an existing left/right leaf pair.
void GetSufficientStatistics(StanTree& tree, StanTree* left_leaf,
                             StanTree* right_leaf,
                             CutpointMatrix& cutpoints, DataInfo& data_info,
                             size_t& left_count, double& left_sum,
                             size_t& right_count, double& right_sum)
{
  left_count = 0;  left_sum  = 0.0;
  right_count = 0; right_sum = 0.0;

  for (size_t i = 0; i < data_info.n; i++) {
    double* x_row = data_info.X + i * data_info.p;
    const StanTree* leaf = tree.FindLeaf(x_row, cutpoints);
    if (leaf == left_leaf) {
      left_count++;
      left_sum += data_info.residuals[i];
    }
    if (leaf == right_leaf) {
      right_count++;
      right_sum += data_info.residuals[i];
    }
  }
}

// Log-likelihood contribution of n observations whose residuals sum to
// sum_residuals, given noise std dev sigma and leaf prior std dev eta.
double LogLikelihood(size_t n, double sum_residuals, double sigma, double eta)
{
  double sigma_squared  = sigma * sigma;
  double prior_variance = eta * eta;
  double precision_sum  = n * prior_variance + sigma_squared;
  return -0.5 * std::log(precision_sum)
         + (prior_variance * sum_residuals * sum_residuals)
           / (2.0 * sigma_squared * precision_sum);
}

// Probability that a node grows; 0 if no valid split variable is available.
double ProbabilityNodeGrows(StanTree* node, CutpointMatrix& cutpoints,
                            PriorInfo& prior_info)
{
  if (CanSplit(node, cutpoints)) {
    return prior_info.base /
      std::pow(1.0 + static_cast<double>(node->NodeDepth()), prior_info.power);
  } else {
    return 0.0;
  }
}

// Compute sufficient statistics for every leaf in a single data pass.
void GetAllLeafStatistics(StanTree& tree, CutpointMatrix& cutpoints,
                          DataInfo& data_info,
                          std::vector<StanTree*>& leaves,
                          std::vector<size_t>& observation_counts,
                          std::vector<double>& residual_sums)
{
  leaves.clear();
  tree.CollectLeaves(leaves);

  size_t num_leaves = leaves.size();
  observation_counts.resize(num_leaves);
  residual_sums.resize(num_leaves);

  std::map<const StanTree*, size_t> leaf_index_map;
  for (size_t i = 0; i < num_leaves; i++) {
    leaf_index_map[leaves[i]] = i;
    observation_counts[i]     = 0;
    residual_sums[i]          = 0.0;
  }

  for (size_t i = 0; i < data_info.n; i++) {
    double* x_row        = data_info.X + i * data_info.p;
    const StanTree* leaf = tree.FindLeaf(x_row, cutpoints);
    size_t leaf_index    = leaf_index_map[leaf];
    observation_counts[leaf_index]++;
    residual_sums[leaf_index] += data_info.residuals[i];
  }
}

// Draw new step heights for all leaves from their Gaussian posteriors.
void DrawAllLeafMeans(StanTree& tree, CutpointMatrix& cutpoints,
                      DataInfo& data_info, PriorInfo& prior_info,
                      double sigma, Random& random)
{
  std::vector<StanTree*> leaves;
  std::vector<size_t>    observation_counts;
  std::vector<double>    residual_sums;
  GetAllLeafStatistics(tree, cutpoints, data_info, leaves,
                       observation_counts, residual_sums);

  for (size_t i = 0; i < leaves.size(); i++) {
    leaves[i]->SetStepHeight(
      DrawLeafMean(observation_counts[i], residual_sums[i],
                   prior_info.eta, sigma, random));
  }
}

// Generate a birth proposal.
void BirthProposal(StanTree& tree, CutpointMatrix& cutpoints,
                   PriorInfo& prior_info,
                   std::vector<StanTree*>& splittable_leaves,
                   double& prob_birth, StanTree*& target_leaf,
                   size_t& split_var, size_t& cut_val,
                   double& log_proposal_ratio,
                   std::vector<size_t>& variable_split_counts,
                   std::vector<double>& split_probabilities,
                   bool use_augmentation, Random& random)
{
  // Draw the target leaf uniformly from splittable leaves.
  size_t leaf_index = static_cast<size_t>(
    std::floor(random.uniform() * splittable_leaves.size()));
  target_leaf = splittable_leaves[leaf_index];

  std::vector<size_t> good_vars;
  int lower_bound = 0, upper_bound = 0;

  if (!use_augmentation) {
    // Degenerate trees strategy (Linero Assumption 2.2):
    // draw split_var with probability proportional to split_probabilities.
    GetSplittableVariables(target_leaf, cutpoints, good_vars);
    random.SetInclusionWeights(split_probabilities);
    split_var   = random.discrete();
    lower_bound = 0;
    upper_bound = static_cast<int>(cutpoints[split_var].size()) - 1;
    if (!std::binary_search(good_vars.begin(), good_vars.end(), split_var)) {
      // Variable not splittable here: use the constrained cut.
      cut_val = target_leaf->GetConstrainedCut(split_var);
    } else {
      target_leaf->FindRegionBounds(split_var, &lower_bound, &upper_bound);
      cut_val = lower_bound + static_cast<size_t>(
        std::floor(random.uniform() * (upper_bound - lower_bound + 1)));
    }

  } else {
    // Modified data-augmentation strategy (Linero Mod. Assumption 2.1):
    // draw split_var only from the good variables, augment bad-variable counts.
    std::vector<size_t> bad_vars;
    std::vector<double> prob_good_vars;
    std::vector<double> prob_bad_vars;
    GetSplittableVariables(target_leaf, cutpoints, good_vars);

    size_t num_bad_vars  = 0;
    double prob_sum_good = 0.0;
    double prob_sum_bad  = 0.0;

    for (size_t j = 0; j < split_probabilities.size(); j++) {
      if (good_vars[j - num_bad_vars] != j) {
        bad_vars.push_back(j);
        prob_bad_vars.push_back(split_probabilities[j]);
        prob_sum_bad += split_probabilities[j];
        num_bad_vars++;
      } else {
        prob_good_vars.push_back(split_probabilities[j]);
        prob_sum_good += split_probabilities[j];
      }
    }

    random.SetInclusionWeights(prob_good_vars);
    split_var = good_vars[random.discrete()];

    if (num_bad_vars != 0) {
      for (size_t j = 0; j < num_bad_vars; j++) {
        variable_split_counts[bad_vars[j]] +=
          (1.0 / prob_sum_good) *
          (split_probabilities[bad_vars[j]] / prob_sum_bad);
      }
    }

    lower_bound = 0;
    upper_bound = static_cast<int>(cutpoints[split_var].size()) - 1;
    target_leaf->FindRegionBounds(split_var, &lower_bound, &upper_bound);
    cut_val = lower_bound + static_cast<size_t>(
      std::floor(random.uniform() * (upper_bound - lower_bound + 1)));
  }

  // Compute the Metropolis ratio components.
  double prob_choose_leaf = 1.0 / splittable_leaves.size();
  size_t leaf_depth       = target_leaf->NodeDepth();
  double prob_grow_leaf   = prior_info.base /
    std::pow(1.0 + static_cast<double>(leaf_depth), prior_info.power);

  double prob_grow_left, prob_grow_right;
  if (good_vars.size() > 1) {
    prob_grow_left  = prior_info.base /
      std::pow(1.0 + leaf_depth + 1.0, prior_info.power);
    prob_grow_right = prob_grow_left;
  } else {
    prob_grow_left = ((int)(cut_val - 1) < lower_bound)
      ? 0.0
      : prior_info.base / std::pow(1.0 + leaf_depth + 1.0, prior_info.power);
    prob_grow_right = (upper_bound < (int)(cut_val + 1))
      ? 0.0
      : prior_info.base / std::pow(1.0 + leaf_depth + 1.0, prior_info.power);
  }

  double prob_death_proposed;
  if (splittable_leaves.size() > 1) {
    prob_death_proposed = 1.0 - prior_info.prob_birth;
  } else {
    prob_death_proposed = ((prob_grow_right == 0.0) && (prob_grow_left == 0.0))
      ? 1.0
      : 1.0 - prior_info.prob_birth;
  }

  size_t num_nogs       = tree.NumberOfNogs();
  StanTree* leaf_parent = target_leaf->GetParent();
  double prob_choose_nog;
  if (leaf_parent == nullptr) {
    prob_choose_nog = 1.0;
  } else {
    prob_choose_nog = (leaf_parent->NodeType() == 'n')
      ? 1.0 / num_nogs
      : 1.0 / (num_nogs + 1.0);
  }

  log_proposal_ratio =
    (prob_grow_leaf * (1.0 - prob_grow_left) * (1.0 - prob_grow_right) *
     prob_death_proposed * prob_choose_nog) /
    ((1.0 - prob_grow_leaf) * prob_choose_leaf * prob_birth);
}

// Generate a death proposal.
void DeathProposal(StanTree& tree, CutpointMatrix& cutpoints,
                   PriorInfo& prior_info,
                   std::vector<StanTree*>& splittable_leaves,
                   double& prob_birth, StanTree*& nog_node,
                   double& log_proposal_ratio, Random& random)
{
  std::vector<StanTree*> nog_nodes;
  tree.CollectNogs(nog_nodes);
  size_t nog_index = static_cast<size_t>(
    std::floor(random.uniform() * nog_nodes.size()));
  nog_node = nog_nodes[nog_index];

  size_t nog_depth     = nog_node->NodeDepth();
  double prob_grow_nog = prior_info.base /
    std::pow(1.0 + static_cast<double>(nog_depth), prior_info.power);

  double prob_grow_left_child  = ProbabilityNodeGrows(nog_node->GetLeft(),
                                                      cutpoints, prior_info);
  double prob_grow_right_child = ProbabilityNodeGrows(nog_node->GetRight(),
                                                      cutpoints, prior_info);

  double prob_birth_proposed = (nog_node->NodeType() == 't')
    ? 1.0
    : prior_info.prob_birth;

  int num_splittable_proposed = static_cast<int>(splittable_leaves.size());
  if (CanSplit(nog_node->GetLeft(),  cutpoints)) --num_splittable_proposed;
  if (CanSplit(nog_node->GetRight(), cutpoints)) --num_splittable_proposed;
  ++num_splittable_proposed;
  double prob_choose_leaf_proposed = 1.0 / num_splittable_proposed;

  double prob_death_current      = 1.0 - prob_birth;
  double prob_choose_nog_current = 1.0 / nog_nodes.size();

  log_proposal_ratio =
    ((1.0 - prob_grow_nog) * prob_birth_proposed * prob_choose_leaf_proposed) /
    (prob_grow_nog * (1.0 - prob_grow_left_child) *
     (1.0 - prob_grow_right_child) * prob_death_current *
     prob_choose_nog_current);
}

// Draw a single leaf mean from its Gaussian posterior.
double DrawLeafMean(size_t n, double sum_residuals, double eta,
                    double sigma, Random& random)
{
  double sigma_squared   = sigma * sigma;
  double precision_data  = n / sigma_squared;
  double precision_prior = 1.0 / (eta * eta);
  return (sum_residuals / sigma_squared) / (precision_prior + precision_data)
         + random.normal() / std::sqrt(precision_prior + precision_data);
}

// Draw the splitting probability vector from a Dirichlet posterior.
void DrawSplitProbabilities(std::vector<size_t>& variable_split_counts,
                            std::vector<double>& log_split_probabilities,
                            double& dart_theta, Random& random)
{
  size_t p = variable_split_counts.size();
  std::vector<double> dirichlet_params(p);
  for (size_t j = 0; j < p; j++) {
    dirichlet_params[j] = dart_theta / static_cast<double>(p)
                          + static_cast<double>(variable_split_counts[j]);
  }
  log_split_probabilities = random.log_dirichlet(dirichlet_params);
}

// Draw the Dirichlet sparsity parameter theta from a grid approximation.
void DrawSparsityParameter(bool fixed_theta, double& dart_theta,
                           std::vector<double>& log_split_probabilities,
                           double dart_a, double dart_b, double dart_rho,
                           Random& random)
{
  // Model: theta / (theta + rho) ~ Beta(dart_a, dart_b).
  // Grid over lambda = theta / (theta + rho) in {1/1001, ..., 1000/1001}.
  if (fixed_theta) return;

  size_t p = log_split_probabilities.size();
  double sum_log_probs = 0.0;
  for (size_t j = 0; j < p; j++) sum_log_probs += log_split_probabilities[j];

  const size_t grid_size = 1000;
  std::vector<double> lambda_grid(grid_size);
  std::vector<double> theta_grid(grid_size);
  std::vector<double> log_weights(grid_size);

  for (size_t k = 0; k < grid_size; k++) {
    lambda_grid[k] = static_cast<double>(k + 1) / 1001.0;
    theta_grid[k]  = (lambda_grid[k] * dart_rho) / (1.0 - lambda_grid[k]);

    double theta_log_lik =
      std::lgamma(theta_grid[k])
      - static_cast<double>(p) *
        std::lgamma(theta_grid[k] / static_cast<double>(p))
      + (theta_grid[k] / static_cast<double>(p)) * sum_log_probs;

    double beta_log_prior =
      (dart_a - 1.0) * std::log(lambda_grid[k])
      + (dart_b - 1.0) * std::log(1.0 - lambda_grid[k]);

    log_weights[k] = theta_log_lik + beta_log_prior;
  }

  double log_normalizer = LogSumExp(log_weights);
  for (size_t k = 0; k < grid_size; k++) {
    log_weights[k] = std::exp(log_weights[k] - log_normalizer);
  }

  random.SetInclusionWeights(log_weights);
  dart_theta = theta_grid[random.discrete()];
}

