
#ifndef GUARD_info_h
#define GUARD_info_h
#include "Prerequisites.h"

// Data container passed to the birth-death and sufficient-statistics functions.
// X is stored column-major: the j-th predictor of the i-th observation is
// *(X + p * i + j).  residuals holds the current working residuals.
class DataInfo {
public:
  DataInfo() : p(0), n(0), X(nullptr), residuals(nullptr) {}
  size_t p;          // Number of predictors
  size_t n;          // Number of observations
  double* X;         // Feature matrix (n x p, column-major)
  double* residuals; // Working residuals (length n)
};

// Prior and MCMC tuning parameters used by the birth-death sampler.
class PriorInfo {
public:
  PriorInfo() : prob_birth_death(1.0), prob_birth(0.5),
                base(0.95), power(2.0), eta(1.0) {}
  // MCMC proposal probabilities
  double prob_birth_death; // Probability of proposing a birth or death step
  double prob_birth;       // Conditional probability of a birth (vs. death)
  // Tree topology prior: P(split at depth d) = base / (1 + d)^power
  double base;
  double power;
  // Prior standard deviation of leaf step heights
  double eta;
};

#endif

