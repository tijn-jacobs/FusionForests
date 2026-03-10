#include "OuterGibbsFunctions.h"

void AugmentCensoredObservations(
  bool is_survival,
  double* event_time, 
  const double* observed_time,
  const double* status_indicator,
  const double* predicted_time,
  const double& sigma,
  const size_t& n, 
  Random& random
) {

  if (!is_survival) return;

  // Declare variables
  double prediction, x_i, temporary, U;                     
  
  // Loop over all observations 
  for (size_t i = 0; i < n; i++) {

    // Check if the event time is censored
    if (status_indicator[i] == 0) {

      // Sample from the truncated normal distribution
      U = random.uniform();
      prediction = predicted_time[i];
      x_i = (observed_time[i] - prediction)/sigma;  

      // x_i is often way too large on the first few iterations !!!
      // if the normalized difference is greater than 4, set x_i = 4.0
      if(x_i > 4)  x_i = 4.0;

      temporary = U + (1.0 - U) * standard_normal_cdf(x_i);

      event_time[i] = prediction + sigma * standard_normal_quantile(temporary);
    }
  }
}

void AugmentCensoredObservations(
  bool is_survival,
  double* event_time, 
  const double* observed_left_time,
  const double* status_indicator,
  const double* observed_right_time,
  const double* interval_censoring_indicator, // 1 if interval censored, 
  const double* predicted_time,               // 0 if observed or right censored
  const double& sigma,
  const size_t& n, 
  Random& random
) {

  if (!is_survival) return;

  // Declare variables
  double prediction, temporary, U;                     
  
  // Loop over all observations 
  for (size_t i = 0; i < n; i++) {

    // RIGHT CENSORING
    if (status_indicator[i] == 0 && interval_censoring_indicator[i] == 0) {

      // Draw from N(mu, sigma^2) truncated to [R, +inf)
      U = random.uniform();
      prediction = predicted_time[i];

      double a  = (observed_right_time[i] - prediction) / sigma; // R bound

      // Cap at 4 to avoid numerical issues (Fa too close to 1.0),
      // especially in early MCMC iterations
      if (a > 4.0) a = 4.0;

      const double Fa = standard_normal_cdf(a);

      // Uniform over [Fa, 1]
      temporary = Fa + U * (1.0 - Fa);

      // Clamp into (0, 1) for the quantile function
      temporary = std::max(temporary, std::nextafter(0.0, 1.0));
      temporary = std::min(temporary, std::nextafter(1.0, 0.0));

      event_time[i] = prediction + sigma * standard_normal_quantile(temporary);
    }


    // INTERVAL CENSORING
    if (status_indicator[i] == 0 && interval_censoring_indicator[i] == 1) {

      // Sample from the truncated normal distribution on [L, R]
      U = random.uniform();
      prediction = predicted_time[i];

      // Standardize bounds: a = (L - mu)/sigma, b = (R - mu)/sigma
      const double a = (observed_left_time[i]  - prediction) / sigma;
      const double b = (observed_right_time[i] - prediction) / sigma;

      // CDF at the bounds (handle one-sided cases if ever used)
      const double Fa = std::isfinite(a) ? standard_normal_cdf(a) : 0.0;
      const double Fb = std::isfinite(b) ? standard_normal_cdf(b) : 1.0;

      // Guard against degenerate/tiny intervals due to rounding
      const double width = Fb - Fa;
      if (width <= 0.0) {
        // fall back to the lower bound in CDF space
        temporary = Fa;
      } else {
        // uniform on [Fa, Fb]
        temporary = Fa + U * width;
      }

      // Clamp into (0, 1) for the quantile function
      temporary = std::max(temporary, std::nextafter(0.0, 1.0));
      temporary = std::min(temporary, std::nextafter(1.0, 0.0));

      // Map back to the original scale
      event_time[i] = prediction + sigma * standard_normal_quantile(temporary);
    }
  }
}

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
) {

  if (sigma_known) return;

  double residual_sum2 = 0.0;
  double residual_temp;

  for (size_t k = 0; k < n; k++) {
    // residual_temp = (y[k] - prediction[k])/sigma;
    residual_temp = (y[k] - prediction[k])/1.0;
    residual_sum2 += residual_temp * residual_temp;
  }
  
  sigma = std::sqrt((nu * lambda + residual_sum2) / random.chi_square(n + nu));

  if (sigma >= 10) sigma = 10;

  store_sigma[i] = sigma;

  if (std::isnan(sigma) || std::isinf(sigma)) {
    Rcpp::stop("Sigma became invalid (NaN or Inf) during MCMC. Stopping execution.");
  }
}

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
) {

  double sigma2 = sigma * sigma;

  // Sufficient statistics for b1 (treated: z_i = 1) and b0 (control: z_i = 0)
  double sum_tau2_treated = 0.0, sum_tau_resid_treated = 0.0;
  double sum_tau2_control = 0.0, sum_tau_resid_control = 0.0;

  for (size_t k = 0; k < n; k++) {
    double tau_k  = prediction_treat[k];
    double resid_k = y[k] - prediction_control[k];
    if (treatment_indicator[k] == 1) {
      sum_tau2_treated    += tau_k * tau_k;
      sum_tau_resid_treated += tau_k * resid_k;
    } else {
      sum_tau2_control    += tau_k * tau_k;
      sum_tau_resid_control += tau_k * resid_k;
    }
  }

  // Posterior for b1: prior precision = 1/(1/2) = 2
  double prec1 = 2.0 + sum_tau2_treated / sigma2;
  double mean1 = (sum_tau_resid_treated / sigma2) / prec1;
  b1 = mean1 + random.normal() / std::sqrt(prec1);

  // Posterior for b0: prior precision = 1/(1/2) = 2
  double prec0 = 2.0 + sum_tau2_control / sigma2;
  double mean0 = (sum_tau_resid_control / sigma2) / prec0;
  b0 = mean0 + random.normal() / std::sqrt(prec0);

  // Recompute b_train and b_test from the new draws
  for (size_t k = 0; k < n; k++)
    b_train[k] = (treatment_indicator[k] == 1) ? b1 : b0;
  for (size_t k = 0; k < n_test; k++)
    b_test[k] = (treatment_indicator_test[k] == 1) ? b1 : b0;
}



