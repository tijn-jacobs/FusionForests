// This header file declares functions for non-reversible and reversible tree 
// modifications in a Bayesian Additive Regression Tree (BART) framework. 
// The functions handle tree growth, pruning, changes, and reversible jumps,
// managing the selection of split variables and cutpoints, as well as 
// computing necessary statistics and acceptance ratios for Metropolis-Hastings 
// sampling. 

#ifndef GUARD_TreeModifications_H
#define GUARD_TreeModifications_H

#include "Functions.h"

// Function to grow a tree at a good bottom node
bool Grow(Tree& tree, Cutpoints& cutpoints, Data& data, TreePrior& tree_prior, 
          const double& sigma, std::vector<size_t>& variable_inclusion_count, 
          std::vector<double>& variable_inclusion_prob, 
          ScaleMixture& scale_mixture, Random& random);

// Function to prune a selected NOG node from the tree
bool Prune(Tree& tree, Cutpoints& cutpoints, Data& data, TreePrior& tree_prior, 
           const double& sigma, std::vector<size_t>& variable_inclusion_count, 
           std::vector<double>& variable_inclusion_prob, 
           ScaleMixture& scale_mixture, Random& random);

// Function to change the split variable and cut value of a NOG node
bool Change(Tree& tree, Cutpoints& cutpoints, Data& data, TreePrior& tree_prior, 
            const double& sigma, std::vector<size_t>& variable_inclusion_count, 
            std::vector<double>& variable_inclusion_prob, 
            ScaleMixture& scale_mixture, Random& random);

// Function to perform a reversible jump grow operation
bool RJ_Grow(Tree& tree, Cutpoints& cutpoints, Data& data, TreePrior& tree_prior, 
             const double& sigma, std::vector<size_t>& variable_inclusion_count, 
             std::vector<double>& variable_inclusion_prob, 
             ScaleMixture& scale_mixture, Random& random);

// Function to perform a reversible jump prune operation
bool RJ_Prune(Tree& tree, Cutpoints& cutpoints, Data& data, TreePrior& tree_prior, 
              const double& sigma, std::vector<size_t>& variable_inclusion_count, 
              std::vector<double>& variable_inclusion_prob, 
              ScaleMixture& scale_mixture, Random& random);

// Function to perform a reversible jump change operation
bool RJ_Change(Tree& tree, Cutpoints& cutpoints, Data& data, TreePrior& tree_prior, 
               const double& sigma, std::vector<size_t>& variable_inclusion_count, 
               std::vector<double>& variable_inclusion_prob, 
               ScaleMixture& scale_mixture, Random& random);

// Function to perform reversible jumps with specific probabilities for each action
bool ReversibleJump(Tree& tree, Cutpoints& cutpoints, Data& data, 
                    TreePrior& tree_prior, const double& sigma, const double& omega, 
                    std::vector<size_t>& nodes, 
                    std::vector<double>& variable_inclusion_prob, 
                    const size_t& delayed_proposal,
                    bool reversible_jump, ScaleMixture& scale_mixture, 
                    Random& random);

#endif // GUARD_TreeModifications_H
