#ifndef GUARD_Random_h
#define GUARD_Random_h

#include "Prerequisites.h"

// Abstract RNG interface
class Random {
public:
  Random() {}

  virtual double normal() = 0;
  virtual double uniform() = 0;
  virtual double chi_square(double df) = 0;
  virtual double exp() = 0;

  virtual double log_gamma(double shape) = 0;
  virtual double gamma(double shape, double rate) = 0;
  virtual double inv_gamma(double shape, double rate) = 0;
  virtual double beta(double a, double b) = 0;

  virtual size_t discrete() = 0;
  virtual size_t geometric(double p) = 0;

  virtual void SetInclusionWeights(std::vector<double>& _wts) = 0;
  virtual std::vector<double>
  log_dirichlet(std::vector<double>& alpha) = 0;

  virtual ~Random() {}
};

// Rcpp-based RNG implementation
class RandomGenerator : public Random {
public:
  RandomGenerator() {}

  virtual ~RandomGenerator() {}

  // Basic distributions
  virtual double normal() { return R::norm_rand(); }
  virtual double uniform() { return R::unif_rand(); }
  virtual double chi_square(double df) { return R::rchisq(df); }
  virtual double exp() { return R::exp_rand(); }

  // Gamma family
  virtual double log_gamma(double shape) {
    double y = std::log(R::rgamma(shape + 1.0, 1.0));
    double z = std::log(this->uniform()) / shape;
    return y + z;
  }

  virtual double gamma(double shape, double rate) {
    if (shape < 0.01)
      return std::exp(this->log_gamma(shape)) / rate;
    return R::rgamma(shape, 1.0) / rate;
  }

  virtual double inv_gamma(double shape, double rate) {
    double g = this->gamma(shape, rate);
    return 1.0 / g;
  }

  virtual double beta(double a, double b) {
    double x1 = this->gamma(a, 1.0);
    double x2 = this->gamma(b, 1.0);
    return x1 / (x1 + x2);
  }

  // Multinomial draw using R::rmultinom
  virtual size_t discrete() {
    size_t k = wts.size();
    if (k == 0) return 0;

    std::vector<int> out(k, 0);
    R::rmultinom(1, wts.data(), k, out.data());

    for (size_t j = 0; j < k; j++)
      if (out[j] == 1) return j;

    return 0;
  }

  virtual size_t geometric(double p) {
    return R::rgeom(p);
  }

  virtual void SetInclusionWeights(std::vector<double>& _wts) {
    double sumw = 0.0;
    wts.clear();

    for (double x : _wts) sumw += x;
    if (sumw <= 0.0) return;

    for (double x : _wts) wts.push_back(x / sumw);
  }

  virtual std::vector<double>
  log_dirichlet(std::vector<double>& alpha) {
    size_t k = alpha.size();
    std::vector<double> out(k);

    for (size_t j = 0; j < k; j++)
      out[j] = this->log_gamma(alpha[j]);

    double lse = LogSumExp(out);
    for (size_t j = 0; j < k; j++)
      out[j] -= lse;

    return out;
  }

private:
  std::vector<double> wts;
  Rcpp::RNGScope RNGstate;
};

#endif  // GUARD_Random_h

