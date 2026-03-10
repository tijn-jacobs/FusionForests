#include "TruncatedNormal.h"

double rtnorm(double mean, double truncation, double sd, Random& gen) {

  // Normalize the truncation point
  truncation = (truncation - mean) / sd;

  double z, lambda;

  if (truncation <= 0.0) {
    // Case 1: Truncation point is less than or equal to 0
    // Perform rejection sampling from a standard normal distribution
    do {
      z = gen.normal();
    } while (z < truncation);
  } else {
    // Case 2: Truncation point is greater than 0
    // Use exponential proposal distribution for efficient sampling

    // Calculate the optimal exponential rate parameter
    lambda = 0.5 * (truncation + sqrt(truncation * truncation + 4.0));

    // Perform rejection sampling
    do {
      z = gen.exp() / lambda + truncation;  // Sample from exponential distribution
    } while (gen.uniform() > exp(-0.5 * pow(z - lambda, 2.0)));  // Accept/reject step
  }

  // Scale back to the original distribution
  double sample = z * sd + mean;

  return sample;
}
