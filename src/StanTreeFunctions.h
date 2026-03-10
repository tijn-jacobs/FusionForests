#ifndef GUARD_StanTreeFunctions_h
#define GUARD_StanTreeFunctions_h

#include "StanTree.h"

// Evaluate a tree at all n observations in the feature matrix x (column-major,
// p predictors per row) and store the resulting step heights in fit_values.
void FitTree(StanTree& tree, CutpointMatrix& cutpoints, size_t p, size_t n,
             double* x, double* fit_values);

// Return true iff the given leaf node can be split on at least one variable.
bool CanSplit(StanTree* node, CutpointMatrix& cutpoints);

// Populate good_vars with the indices of all variables that node can split on.
void GetSplittableVariables(StanTree* node, CutpointMatrix& cutpoints,
                            std::vector<size_t>& good_vars);

#endif

