#ifndef GUARD_StanForestFunctions_h
#define GUARD_StanForestFunctions_h

#include "StanTree.h"
#include "StanTreeFunctions.h"
#include "Info.h"
#include <algorithm>

// Build a CutpointMatrix for p predictors from n observations stored in x
// (column-major).  nc[v] controls the number of cutpoints for predictor v:
// if nc != nullptr, uniform grids are used; if nc == nullptr, unique observed
// values are used as cutpoints.
void MakeCutpoints(size_t p, size_t n, double* x, CutpointMatrix& cutpoints,
                   int* nc);

// Compute the probability of proposing a birth step given the current tree.
// splittable_leaves is populated with all leaf nodes that can be split.
double GetBirthProbability(StanTree& tree, CutpointMatrix& cutpoints,
                           PriorInfo& prior_info,
                           std::vector<StanTree*>& splittable_leaves);

// Compute observation counts and residual sums for the left and right
// partitions induced by splitting target_leaf on (split_var, cut_val).
void GetSufficientStatistics(StanTree& tree, StanTree* target_leaf,
                             size_t split_var, size_t cut_val,
                             CutpointMatrix& cutpoints, DataInfo& data_info,
                             size_t& left_count, double& left_sum,
                             size_t& right_count, double& right_sum);

// Compute observation counts and residual sums for an existing left/right
// leaf pair (used during a death proposal).
void GetSufficientStatistics(StanTree& tree, StanTree* left_leaf,
                             StanTree* right_leaf,
                             CutpointMatrix& cutpoints, DataInfo& data_info,
                             size_t& left_count, double& left_sum,
                             size_t& right_count, double& right_sum);

// Compute sufficient statistics for every leaf in the tree in a single pass
// over the data.  Populates leaves, observation_counts, and residual_sums.
void GetAllLeafStatistics(StanTree& tree, CutpointMatrix& cutpoints,
                          DataInfo& data_info,
                          std::vector<StanTree*>& leaves,
                          std::vector<size_t>& observation_counts,
                          std::vector<double>& residual_sums);

// Log-likelihood of n residuals with sum sum_residuals under a Gaussian
// model with noise std dev sigma and leaf prior std dev eta.
double LogLikelihood(size_t n, double sum_residuals, double sigma, double eta);

// Probability that a node at the given depth grows; 0 if no valid split
// variable is available.
double ProbabilityNodeGrows(StanTree* node, CutpointMatrix& cutpoints,
                            PriorInfo& prior_info);

// Draw new step heights for all leaves of the tree from their posterior.
void DrawAllLeafMeans(StanTree& tree, CutpointMatrix& cutpoints,
                      DataInfo& data_info, PriorInfo& prior_info,
                      double sigma, Random& random);

// Generate a birth proposal: draw the target leaf, split variable, cut
// value, and compute the Metropolis ratio component (log_proposal_ratio).
void BirthProposal(StanTree& tree, CutpointMatrix& cutpoints,
                   PriorInfo& prior_info,
                   std::vector<StanTree*>& splittable_leaves,
                   double& prob_birth, StanTree*& target_leaf,
                   size_t& split_var, size_t& cut_val,
                   double& log_proposal_ratio,
                   std::vector<size_t>& variable_split_counts,
                   std::vector<double>& split_probabilities,
                   bool use_augmentation, Random& random);

// Generate a death proposal: draw the nog node to collapse and compute
// the Metropolis ratio component (log_proposal_ratio).
void DeathProposal(StanTree& tree, CutpointMatrix& cutpoints,
                   PriorInfo& prior_info,
                   std::vector<StanTree*>& splittable_leaves,
                   double& prob_birth, StanTree*& nog_node,
                   double& log_proposal_ratio, Random& random);

// Draw a single leaf mean from its Gaussian posterior.
double DrawLeafMean(size_t n, double sum_residuals, double eta,
                    double sigma, Random& random);

// Draw the vector of splitting probabilities from a Dirichlet posterior
// (Linero, 2018).  Updates log_split_probabilities in-place.
void DrawSplitProbabilities(std::vector<size_t>& variable_split_counts,
                            std::vector<double>& log_split_probabilities,
                            double& dart_theta, Random& random);

// Draw the Dirichlet sparsity parameter theta from a grid approximation
// to its posterior (Linero, 2018).
void DrawSparsityParameter(bool fixed_theta, double& dart_theta,
                           std::vector<double>& log_split_probabilities,
                           double dart_a, double dart_b, double dart_rho,
                           Random& random);

// ===== Informed Random Splitting (IRS) =====

// Compute the informed routing probability P(go left) for a missing
// observation at a split node, based on posterior predictive densities.
double ComputeIRSProbability(double residual_i,
                             size_t n_left, double sum_left,
                             size_t n_right, double sum_right,
                             double sigma, double tau_h);

// Draw routing indicators for all observations at a newly born internal node.
// Called after a birth is accepted.
// irs_mode: 1/2 = informed routing, 3 = uniform P=0.5.
void DrawRoutingIndicators(
    StanTree* node, size_t split_var, size_t cut_val,
    CutpointMatrix& cutpoints, StanTree& tree_root,
    DataInfo& data_info, double sigma, double tau_h,
    RoutingMap& routing_map, Random& random, int irs_mode = 1);

// Gibbs-redraw routing indicators at all nog nodes in the tree.
// Called once per tree per MCMC iteration (before computing residuals).
// irs_mode: 1/2 = informed routing, 3 = uniform P=0.5.
void RedrawNogRouting(
    StanTree& tree, CutpointMatrix& cutpoints,
    DataInfo& data_info, double sigma, double tau_h,
    RoutingMap& routing_map, Random& random, int irs_mode = 1);

// Remove routing indicators for a dying node. Called after death acceptance.
void RemoveRoutingIndicators(StanTree* dying_node, RoutingMap& routing_map);

// Routing-map variants of GetSufficientStatistics (birth).
void GetSufficientStatistics(StanTree& tree, StanTree* target_leaf,
                             size_t split_var, size_t cut_val,
                             CutpointMatrix& cutpoints, DataInfo& data_info,
                             size_t& left_count, double& left_sum,
                             size_t& right_count, double& right_sum,
                             RoutingMap& routing_map);

// Routing-map variant of GetSufficientStatistics (death).
void GetSufficientStatistics(StanTree& tree, StanTree* left_leaf,
                             StanTree* right_leaf,
                             CutpointMatrix& cutpoints, DataInfo& data_info,
                             size_t& left_count, double& left_sum,
                             size_t& right_count, double& right_sum,
                             RoutingMap& routing_map);

// Routing-map variant of GetAllLeafStatistics.
void GetAllLeafStatistics(StanTree& tree, CutpointMatrix& cutpoints,
                          DataInfo& data_info,
                          std::vector<StanTree*>& leaves,
                          std::vector<size_t>& observation_counts,
                          std::vector<double>& residual_sums,
                          RoutingMap& routing_map);

// Routing-map variant of DrawAllLeafMeans.
void DrawAllLeafMeans(StanTree& tree, CutpointMatrix& cutpoints,
                      DataInfo& data_info, PriorInfo& prior_info,
                      double sigma, Random& random,
                      RoutingMap& routing_map);

// Draw-then-decide: draw routing indicators for NaN observations at the
// proposed split_var BEFORE the birth, compute sufficient stats including
// routed NaN observations, and return the tentative routing indicators.
// If the birth is accepted, store the indicators in routing_map.
// If rejected, discard them.
void DrawRoutingAndGetSufficientStatistics(
    StanTree& tree, StanTree* target_leaf,
    size_t split_var, size_t cut_val,
    CutpointMatrix& cutpoints, DataInfo& data_info,
    double sigma, double tau_h,
    size_t& left_count, double& left_sum,
    size_t& right_count, double& right_sum,
    std::vector<int8_t>& tentative_indicators,
    RoutingMap& routing_map, Random& random);

#endif

