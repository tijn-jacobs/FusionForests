// This file serves as the "prerequisites" header, bringing together key includes
// and common constants used throughout the codebase. It provides standard C++
// headers for functionality such as input/output, mathematical operations, random
// number generation, and data structures like vectors and maps. It also includes
// the external Random library (which may be replaced by Rcpp Armadillo in future)
// and the GSL library for specific statistical functions, like chi-square quantiles.
// Additionally, some important mathematical constants, such as Pi, are defined here.

#ifndef GUARD_Prerequisites_h
#define GUARD_Prerequisites_h

#include <iostream>    // Standard input/output operations (e.g., cout, endl)
#include <vector>      // Vector data structure from the STL
#include <cmath>       // Common math functions (e.g., sqrt, pow)
#include <random>      // C++ random number generation library
#include <cstddef>     // Definitions for size_t
#include <algorithm>   // Algorithms like std::min, std::max, std::sort
#include <iomanip>     // Input/output manipulators (e.g., for formatting output)
#include <map>         // Associative containers (e.g., map)
#include <fstream>     // File stream operations for input/output with files
#include <ctime>       // Time functions (e.g., clock, time)
#include <stdexcept>   // 
#include <memory>      // For dynamic memory management (e.g., std::shared_ptr)
#include <limits>      // for std::numeric_limits<size_t>::max()

// New dependencies
#include <cstdint>   // for uint8_t (integers 0 to 255)
#include <algorithm> // for std::find

#include "Rcpp.h"

double LogSumExp(std::vector<double>& v);

// Random number generation library
#include "Random.h"

// Namespace declarations for commonly used entities
using namespace Rcpp;
using std::endl;
using std::string;
#define cout Rcpp::Rcout
#define printf Rprintf

// Define commonly used mathematical constants for convenience
#define PI 3.141592653589793238462643383280
#define log2pi 1.837877066409345483560659472811
#define sqrt2pi 2.506628274631000502415765284811

// Normal CDF
inline double standard_normal_cdf(double x) {
  return 0.5 * (1.0 + std::erf(x / std::sqrt(2.0)));
}

// Inverse normal CDF (approximation)
inline double standard_normal_quantile(double p) {
  if (p <= 0.0 || p >= 1.0) {
    throw std::invalid_argument("p must be in (0,1)");
  }

  // Approximation constants
  static const double a1 = -3.969683028665376e+01;
  static const double a2 =  2.209460984245205e+02;
  static const double a3 = -2.759285104469687e+02;
  static const double a4 =  1.383577518672690e+02;
  static const double a5 = -3.066479806614716e+01;
  static const double a6 =  2.506628277459239e+00;

  static const double b1 = -5.447609879822406e+01;
  static const double b2 =  1.615858368580409e+02;
  static const double b3 = -1.556989798598866e+02;
  static const double b4 =  6.680131188771972e+01;
  static const double b5 = -1.328068155288572e+01;

  static const double c1 = -7.784894002430293e-03;
  static const double c2 = -3.223964580411365e-01;
  static const double c3 = -2.400758277161838e+00;
  static const double c4 = -2.549732539343734e+00;
  static const double c5 =  4.374664141464968e+00;
  static const double c6 =  2.938163982698783e+00;

  static const double d1 =  7.784695709041462e-03;
  static const double d2 =  3.224671290700398e-01;
  static const double d3 =  2.445134137142996e+00;
  static const double d4 =  3.754408661907416e+00;

  // Define break-points
  const double plow  = 0.02425;
  const double phigh = 1 - plow;

  double q, r;

  if (p < plow) {
    // Rational approximation for lower region
    q = std::sqrt(-2 * std::log(p));
    return (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) /
           ((((d1 * q + d2) * q + d3) * q + d4) * q + 1);
  }
  if (phigh < p) {
    // Rational approximation for upper region
    q = std::sqrt(-2 * std::log(1 - p));
    return -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) /
            ((((d1 * q + d2) * q + d3) * q + d4) * q + 1);
  }

  // Rational approximation for central region
  q = p - 0.5;
  r = q * q;
  return (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q /
         (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1);
}


#endif

