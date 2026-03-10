#include "StanBirthDeath.h"

// Propose and accept/reject one birth or death step for tree.
// Returns true if the proposal was accepted.
bool BirthDeathStep(StanTree& tree, CutpointMatrix& cutpoints,
                    DataInfo& data_info, PriorInfo& prior_info,
                    double sigma,
                    std::vector<size_t>& variable_split_counts,
                    std::vector<double>& split_probabilities,
                    bool use_augmentation, Random& random)
{
  // Collect splittable leaves and compute the birth probability.
  std::vector<StanTree*> splittable_leaves;
  double prob_birth = GetBirthProbability(tree, cutpoints, prior_info,
                                          splittable_leaves);

  if (random.uniform() < prob_birth) {
    // ----------------------------------------------------------------
    // Birth proposal
    // ----------------------------------------------------------------
    StanTree* target_leaf        = nullptr;
    size_t    split_var          = 0;
    size_t    cut_val            = 0;
    double    log_proposal_ratio = 0.0;

    BirthProposal(tree, cutpoints, prior_info, splittable_leaves, prob_birth,
                  target_leaf, split_var, cut_val, log_proposal_ratio,
                  variable_split_counts, split_probabilities,
                  use_augmentation, random);

    // Sufficient statistics for the proposed left and right children.
    size_t left_count = 0, right_count = 0;
    double left_sum   = 0.0, right_sum = 0.0;
    GetSufficientStatistics(tree, target_leaf, split_var, cut_val,
                            cutpoints, data_info,
                            left_count, left_sum, right_count, right_sum);

    // Compute the log Metropolis acceptance probability.
    double log_alpha    = 0.0;
    double accept_prob  = 0.0;
    if ((left_count >= 5) && (right_count >= 5)) {
      double log_lik_left   = LogLikelihood(left_count,  left_sum,
                                            sigma, prior_info.eta);
      double log_lik_right  = LogLikelihood(right_count, right_sum,
                                            sigma, prior_info.eta);
      double log_lik_parent = LogLikelihood(left_count + right_count,
                                            left_sum + right_sum,
                                            sigma, prior_info.eta);
      accept_prob = 1.0;
      log_alpha   = std::log(log_proposal_ratio)
                    + (log_lik_left + log_lik_right - log_lik_parent)
                    + std::log(sigma);
      log_alpha   = std::min(0.0, log_alpha);
    }

    // Accept or reject.
    bool accepted = (accept_prob > 0.0) &&
                    (std::log(random.uniform()) < log_alpha);
    if (accepted) {
      double left_mean  = DrawLeafMean(left_count,  left_sum,
                                       prior_info.eta, sigma, random);
      double right_mean = DrawLeafMean(right_count, right_sum,
                                       prior_info.eta, sigma, random);
      tree.BirthAtNode(target_leaf, split_var, cut_val, left_mean, right_mean);
      variable_split_counts[split_var]++;
      return true;
    } else {
      return false;
    }

  } else {
    // ----------------------------------------------------------------
    // Death proposal
    // ----------------------------------------------------------------
    StanTree* nog_node           = nullptr;
    double    log_proposal_ratio = 0.0;

    DeathProposal(tree, cutpoints, prior_info, splittable_leaves, prob_birth,
                  nog_node, log_proposal_ratio, random);

    // Sufficient statistics for the two leaf children of the nog node.
    size_t left_count = 0, right_count = 0;
    double left_sum   = 0.0, right_sum = 0.0;
    GetSufficientStatistics(tree, nog_node->GetLeft(), nog_node->GetRight(),
                            cutpoints, data_info,
                            left_count, left_sum, right_count, right_sum);

    // Compute the log Metropolis acceptance probability.
    double log_lik_left   = LogLikelihood(left_count,  left_sum,
                                          sigma, prior_info.eta);
    double log_lik_right  = LogLikelihood(right_count, right_sum,
                                          sigma, prior_info.eta);
    double log_lik_parent = LogLikelihood(left_count + right_count,
                                          left_sum + right_sum,
                                          sigma, prior_info.eta);

    double log_alpha = std::log(log_proposal_ratio)
                       + (log_lik_parent - log_lik_left - log_lik_right)
                       - std::log(sigma);
    log_alpha = std::min(0.0, log_alpha);

    // Accept or reject.
    if (std::log(random.uniform()) < log_alpha) {
      double parent_mean = DrawLeafMean(left_count + right_count,
                                        left_sum  + right_sum,
                                        prior_info.eta, sigma, random);
      variable_split_counts[nog_node->GetSplitVar()]--;
      tree.DeathAtNode(nog_node, parent_mean);
      return true;
    } else {
      return false;
    }
  }
}

