#ifndef OUTERGIBBSFUNCTIONS_H
#define OUTERGIBBSFUNCTIONS_H

#include "Prerequisites.h"

// UpdateSigma
void UpdateSigma(
  bool sigma_known,
  double& sigma,
  Rcpp::NumericVector& store_sigma,
  const size_t i,
  const double* y,
  const size_t n,
  const double* prediction,
  const double nu,
  const double lambda,
  Random& random
);

// AugmentCensoredObservations
void AugmentCensoredObservations(
  bool is_survival,
  double* event_time, 
  const double* observed_time,
  const double* status_indicator,
  const double* predicted_time,
  const double& sigma,
  const size_t& n, 
  Random& random
);

// Second AugmentCensoredObservations with interval censoring
void AugmentCensoredObservations(
  bool is_survival,
  double* event_time,
  const double* observed_left_time,
  const double* status_indicator,
  const double* observed_right_time,
  const double* interval_censoring_indicator,
  const double* predicted_time,
  const double& sigma,
  const size_t& n,
  Random& random
);

// UpdateInvariantCoding
// Conjugate normal update for the parameter-expanded (invariant) treatment
// coding of Hahn, Murray & Carvalho (2020).  The model is
//   y_i = mu(x_i) + tilde_tau(x_i) * b_{z_i} + eps_i
// with priors  b_0 ~ N(0, 1/2),  b_1 ~ N(0, 1/2).
// Given current mu, tilde_tau and sigma, the full conditional for each b_j
// is a standard conjugate normal.
void UpdateInvariantCoding(
  double& b0,
  double& b1,
  double* b_train,
  double* b_test,
  const double* y,
  const double* prediction_control,
  const double* prediction_treat,
  const int* treatment_indicator,
  const int* treatment_indicator_test,
  const size_t n,
  const size_t n_test,
  const double sigma,
  Random& random
);

#endif // OUTERGIBBSFUNCTIONS_H

