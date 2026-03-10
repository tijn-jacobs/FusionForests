#include "StanTreeFunctions.h"

// Evaluate tree at all n observations; x[i,j] is *(x + p*i + j).
void FitTree(StanTree& tree, CutpointMatrix& cutpoints, size_t p, size_t n,
             double* x, double* fit_values)
{
  for (size_t i = 0; i < n; i++) {
    StanTree* leaf = tree.FindLeaf(x + i * p, cutpoints);
    fit_values[i]  = leaf->GetStepHeight();
  }
}

// Return true iff the leaf node can be split on at least one variable.
bool CanSplit(StanTree* node, CutpointMatrix& cutpoints)
{
  bool split_var_found = false;
  size_t split_var = 0;
  while (!split_var_found && (split_var < cutpoints.size())) {
    int lower_bound = 0;
    int upper_bound = static_cast<int>(cutpoints[split_var].size()) - 1;
    node->FindRegionBounds(split_var, &lower_bound, &upper_bound);
    if (upper_bound >= lower_bound) split_var_found = true;
    split_var++;
  }
  return split_var_found;
}

// Populate good_vars with the indices of all variables that node can split on.
void GetSplittableVariables(StanTree* node, CutpointMatrix& cutpoints,
                            std::vector<size_t>& good_vars)
{
  good_vars.clear();
  for (size_t split_var = 0; split_var < cutpoints.size(); split_var++) {
    int lower_bound = 0;
    int upper_bound = static_cast<int>(cutpoints[split_var].size()) - 1;
    node->FindRegionBounds(split_var, &lower_bound, &upper_bound);
    if (upper_bound >= lower_bound) good_vars.push_back(split_var);
  }
}

