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

#endif // OUTERGIBBSFUNCTIONS_H

