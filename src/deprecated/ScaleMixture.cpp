#include "ScaleMixture.h"


//                //
// Fixed variance // 
//                // 


// Constructor for FixedVariance, initializing eta (fixed variance).
FixedVariance::FixedVariance(double _eta) : eta(_eta) {}

// Propose function for FixedVariance.
// This sets the second parameter (index 1) of `parameters` to `eta`.
// No sampling is needed since eta is fixed.
void FixedVariance::Propose(Parameters& parameters, 
                            const Parameters& global_parameters, 
                            const double& residual, 
                            const double& number_of_observations, 
                            const double& sigma,
                            const double& omega,  
                            Random& random) {
  parameters.SetParameters(1, eta);  // Set the fixed eta as variance
}

// GlobalUpdate for FixedVariance.
// Since eta is fixed, there is no need to update global global_parameters,
// so this function does nothing.
void FixedVariance::GlobalUpdate(Parameters& global_parameters, 
                                 const std::vector<Parameters*>& leaf_parameters, 
                                 const double& sigma,
                                 const double& omega,  
                                 Random& random) {
  // No global update required for fixed variance.
}

// LogProposeDensity for FixedVariance.
// Returns 0 because no proposal density is involved with a fixed variance.
double FixedVariance::LogProposeDensity(const Parameters& parameters, 
                                        const Parameters& global_parameters, 
                                        const double& residual, 
                                        const double& number_of_observations, 
                                        const double& sigma,
                                        const double& omega) {
  // No proposal density for fixed variance, return 0.
  return 0.0;
}

// LogPrior for FixedVariance.
// Returns 0 as the prior density for a fixed parameter is constant.
double FixedVariance::LogPrior(const Parameters& parameters, 
                               const Parameters& global_parameters,
                               const double& sigma,
                               const double& omega) {
  // No prior density computation for fixed variance, return 0.
  return 0.0;
}

// GetVariance for FixedVariance.
// Returns the fixed eta as the variance.
double FixedVariance::GetVariance(const Parameters& parameters, 
                                  const Parameters& global_parameters) {
  return eta;  // Fixed variance eta
}

// GetGlobal for FixedVariance.
// Returns false since no global parameters are involved.
bool FixedVariance::GetGlobal() {
  return global;  // No global update parameters
}


//                //
//  Half-Cauchy   //
//                //


HalfCauchy::HalfCauchy(double _alpha, double _tau) 
  : alpha(_alpha), tau(_tau) {}

// Propose new values for lambda and nu in Half-Cauchy distribution.
void HalfCauchy::Propose(Parameters& parameters, 
                         const Parameters& global_parameters, 
                         const double& residual, 
                         const double& number_of_observations, 
                         const double& sigma,
                         const double& omega, 
                         Random& random) {
  
  // Draw auxiliary variable nu from an inverse gamma distribution
  double scale_nu = 1 / (alpha * alpha) + 1 / parameters.GetParameters(1);
  double nu = 1 / random.gamma(1.0, scale_nu); 
  parameters.SetParameters(2, nu);  // Set the drawn nu in parameters
  
  // Draw lambda given nu from an inverse gamma distribution
  double h = parameters.GetParameters(0);
  double scale_lambda = 1.0 / nu + (h * h) / (2.0 * tau * omega);
  double lambda = 1 / random.gamma(1.0, scale_lambda);
  parameters.SetParameters(1, lambda);  // Set lambda in parameters
}

// Global update for HalfCauchy is not required, so this function does nothing.
void HalfCauchy::GlobalUpdate(Parameters& global_parameters, 
                              const std::vector<Parameters*>& leaf_parameters, 
                              const double& omega, const double& sigma, Random& random) {
  // No global update for Half-Cauchy
}

// Compute the log of the proposal density.
double HalfCauchy::LogProposeDensity(const Parameters& parameters, 
                                     const Parameters& global_parameters, 
                                     const double& residual, 
                                     const double& number_of_observations, 
                                     const double& sigma,
                                     const double& omega) {
  
  // Compute the scale for nu and retrieve current nu
  double scale_nu = 1 / (alpha * alpha) + 1 / parameters.GetParameters(1);
  double nu = parameters.GetParameters(2);
  
  // Compute the scale for lambda and retrieve current lambda
  double h = parameters.GetParameters(0);
  double scale_lambda = 1.0 / nu + (h * h) / (2.0 * tau * omega);
  double lambda = parameters.GetParameters(1);
  
  // Return the sum of the log of the inverse gamma densities
  return log_inverse_gamma_pdf(nu, 1.0, scale_nu) +
    log_inverse_gamma_pdf(lambda, 1.0, scale_lambda);
}

// Compute the log of the prior density using the Half-Cauchy prior.
double HalfCauchy::LogPrior(const Parameters& parameters, 
                            const Parameters& global_parameters,
                            const double& sigma,
                            const double& omega) {
  
  // Return the sum of the log densities for lambda and nu
  return log_inverse_gamma_pdf(parameters.GetParameters(1), 0.5, 
                               1 / parameters.GetParameters(2)) +
                                 log_inverse_gamma_pdf(parameters.GetParameters(2), 0.5, 
                                                       1 / std::pow(alpha, 2));
}

// Get the variance based on the Half-Cauchy prior.
double HalfCauchy::GetVariance(const Parameters& parameters, 
                               const Parameters& global_parameters) {
  return tau * parameters.GetParameters(1);  // Variance depends on tau
}

// Return whether global parameters are used in this prior.
bool HalfCauchy::GetGlobal() {
  return global;  // No global update for Half-Cauchy
}


//                //
//   Horseshoe    //
//                //


Horseshoe::Horseshoe(double _alpha_local, double _alpha_global)
  : alpha_local(_alpha_local), alpha_global(_alpha_global) {}

// Propose new values for lambda and nu in the Horseshoe prior
void Horseshoe::Propose(Parameters& parameters, 
                        const Parameters& global_parameters, 
                        const double& residual, 
                        const double& number_of_observations, 
                        const double& sigma,
                        const double& omega, 
                        Random& random) {
  
  // Draw the auxiliary variable nu from an inverse gamma distribution
  double scale_nu = 1 / (alpha_local * alpha_local) + 1 / parameters.GetParameters(1);
  double nu = 1 / random.gamma(1.0, scale_nu); 
  parameters.SetParameters(2, nu);  // Set nu in the parameters
  
  // Draw lambda given nu from an inverse gamma distribution
  double tau = global_parameters.GetGlobalParameters(0);
  double h = parameters.GetParameters(0);
  double scale_lambda = 1.0 / nu + (h * h) / (2.0 * tau * omega);
  double lambda = 1 / random.gamma(1.0, scale_lambda);
  parameters.SetParameters(1, lambda);  // Set lambda in the parameters
}

// Compute the log of the proposal density for the Horseshoe prior
double Horseshoe::LogProposeDensity(const Parameters& parameters, 
                                    const Parameters& global_parameters, 
                                    const double& residual, 
                                    const double& number_of_observations, 
                                    const double& sigma,
                                    const double& omega) {
  
  // Compute the scale for nu and retrieve current nu
  double scale_nu = 1 / (alpha_local * alpha_local) + 1 / parameters.GetParameters(1);
  double nu = parameters.GetParameters(2);
  
  // Compute the scale for lambda and retrieve current lambda
  double tau = global_parameters.GetGlobalParameters(0);
  double h = parameters.GetParameters(0);  
  double scale_lambda = 1.0 / nu + (h * h) / (2.0 * tau * omega);
  double lambda = parameters.GetParameters(1);
  
  // Return the sum of the log of the inverse gamma densities
  return log_inverse_gamma_pdf(nu, 1.0, scale_nu) +
    log_inverse_gamma_pdf(lambda, 1.0, scale_lambda);
}

// Compute the log of the prior density for the Horseshoe prior
double Horseshoe::LogPrior(const Parameters& parameters, 
                           const Parameters& global_parameters,
                           const double& sigma,
                           const double& omega) {
  
  // Return the sum of the log densities for lambda and nu
  return log_inverse_gamma_pdf(parameters.GetParameters(1), 0.5, 
                               1 / parameters.GetParameters(2)) +
                                 log_inverse_gamma_pdf(parameters.GetParameters(2), 0.5, 
                                                       1 / std::pow(alpha_local, 2));
}


// Get the variance based on the Horseshoe prior
double Horseshoe::GetVariance(const Parameters& parameters, 
                              const Parameters& global_parameters) {

  return global_parameters.GetGlobalParameters(0) * parameters.GetParameters(1);
}

// Return whether global parameters are used in this prior
bool Horseshoe::GetGlobal() {
  return global;
}

// Perform global update for the Horseshoe prior based on leaf parameters
void Horseshoe::GlobalUpdate(Parameters& global_parameters, 
                             const std::vector<Parameters*>& leaf_parameters, 
                             const double& sigma, 
                             const double& omega, 
                             Random& random) {
  
  // tau is a squared term
  double current_tau = global_parameters.GetGlobalParameters(0);
  
  // Draw a new auxiliary variable xi and store in the global parameters
  double xi = random.inv_gamma(1.0, 1.0 / alpha_global * alpha_global 
                                 + 1.0 / current_tau);
  global_parameters.SetGlobalParameters(1, xi);
  
  // Initialize the sum for h^2 / lambda
  double sum = 0.0;
  
  // Retrieve the number of parameters in the leaf_parameters vector
  size_t p = leaf_parameters.size();
  
  // Loop through each Parameters object in leaf_parameters
  for (const Parameters* par : leaf_parameters) {
    double h = par->GetParameters(0);        // Retrieve h (first parameter)
    double lambda = par->GetParameters(1);   // Retrieve lambda (second parameter)
    
    // Add h^2 / lambda to the sum
    sum += (h * h) / lambda;
  }
  
  // Compute new tau based on the sum and xi
  double new_tau = random.inv_gamma((p + 1.0) / 2.0, 
                                      1.0 / xi + (1 / (2 * omega)) * sum);
  global_parameters.SetGlobalParameters(0, new_tau);
}



//                            //
//   Horseshoe Forest Wide    //
//                            //


Horseshoe_fw::Horseshoe_fw(double _alpha_local)
  : alpha_local(_alpha_local) {}

// Propose new values for lambda and nu in the Horseshoe Forest Wide prior
void Horseshoe_fw::Propose(Parameters& parameters, 
                        const Parameters& global_parameters, 
                        const double& residual, 
                        const double& number_of_observations, 
                        const double& sigma,
                        const double& omega, 
                        Random& random) {
  
  // Draw the auxiliary variable nu from an inverse gamma distribution
  double scale_nu = 1 / (alpha_local * alpha_local) + 1 / parameters.GetParameters(1);
  double nu = 1 / random.gamma(1.0, scale_nu); 
  parameters.SetParameters(2, nu);  // Set nu in the parameters
  
  // Draw lambda given nu from an inverse gamma distribution
  double tau = global_parameters.GetGlobalParameters(0);
  double h = parameters.GetParameters(0);
  double scale_lambda = 1.0 / nu + (h * h) / (2.0 * tau * omega);
  double lambda = 1 / random.gamma(1.0, scale_lambda);
  parameters.SetParameters(1, lambda);  // Set lambda in the parameters
}

// Compute the log of the proposal density for the Horseshoe Forest Wide prior
double Horseshoe_fw::LogProposeDensity(const Parameters& parameters, 
                                    const Parameters& global_parameters, 
                                    const double& residual, 
                                    const double& number_of_observations, 
                                    const double& sigma,
                                    const double& omega) {
  
  // Compute the scale for nu and retrieve current nu
  double scale_nu = 1 / (alpha_local * alpha_local) + 1 / parameters.GetParameters(1);
  double nu = parameters.GetParameters(2);
  
  // Compute the scale for lambda and retrieve current lambda
  double tau = global_parameters.GetGlobalParameters(0);
  double h = parameters.GetParameters(0);  
  double scale_lambda = 1.0 / nu + (h * h) / (2.0 * tau * omega);
  double lambda = parameters.GetParameters(1);
  
  // Return the sum of the log of the inverse gamma densities
  return log_inverse_gamma_pdf(nu, 1.0, scale_nu) +
    log_inverse_gamma_pdf(lambda, 1.0, scale_lambda);
}

// Compute the log of the prior density for the Horseshoe Forest Wide prior
double Horseshoe_fw::LogPrior(const Parameters& parameters, 
                           const Parameters& global_parameters,
                           const double& sigma,
                           const double& omega) {
  
  // Return the sum of the log densities for lambda and nu
  return log_inverse_gamma_pdf(parameters.GetParameters(1), 0.5, 
                               1 / parameters.GetParameters(2)) +
                                 log_inverse_gamma_pdf(parameters.GetParameters(2), 0.5, 
                                                       1 / std::pow(alpha_local, 2));
}


// Get the variance based on the Horseshoe Forest Wide prior
double Horseshoe_fw::GetVariance(const Parameters& parameters, 
                              const Parameters& global_parameters) {
  return global_parameters.GetGlobalParameters(0) * parameters.GetParameters(1);
}

// Return whether global parameters are used in this prior
bool Horseshoe_fw::GetGlobal() {
  return global;
}

// Perform global update for the Horseshoe Forest Wide prior based on leaf parameters
void Horseshoe_fw::GlobalUpdate(Parameters& global_parameters, 
                             const std::vector<Parameters*>& leaf_parameters, 
                             const double& sigma, 
                             const double& omega, 
                             Random& random) {
  
  // No global update for the Horseshoe Forest Wide prior.
  // This is performed forest wide via an outer Gibbs step.
}


//                 //
// General objects //
//                 //


// Factory function implementation for etaPrior
EtaPrior* CreateEtaPrior(PriorType type, double param1, double param2) {
  switch (type) {
  case PriorType::FixedVariance:
    return new FixedVariance(param1);
  case PriorType::HalfCauchy:
    return new HalfCauchy(param1, param2);
  case PriorType::Horseshoe:
    return new Horseshoe(param1, param2);
  case PriorType::Horseshoe_fw:
    return new Horseshoe_fw(param1);
  default:
    throw std::invalid_argument("Unsupported prior type");
  }
}

// ScaleMixture class constructor
ScaleMixture::ScaleMixture(PriorType type, double param1, double param2) {
  eta_prior = CreateEtaPrior(type, param1, param2);
}

// Destructor to clean up allocated EtaPrior
ScaleMixture::~ScaleMixture() {
  delete eta_prior;
}

// Propose new step height and variance in the ScaleMixture class
void ScaleMixture::Propose(Parameters& parameters, 
                           const Parameters& global_parameters, 
                           const double& residual, 
                           const double& number_of_observations, 
                           const double& sigma, 
                           const double& omega, 
                           Random& random) {
  
  // Propose variance parameters using eta_prior
  eta_prior->Propose(parameters, global_parameters, residual, 
                     number_of_observations, sigma, omega, random);
  
  // Retrieve variance and compute conditional posterior mean and sd
  double precision = 1 / (omega * eta_prior->GetVariance(parameters, 
                                                global_parameters));
  double cond_post_mean = residual / (number_of_observations + precision);
  double cond_post_sd = 1 / std::sqrt(number_of_observations 
                                            + precision);
  
  // Draw normal and transform appropriately for step height
  double h = cond_post_mean + random.normal() * cond_post_sd;

  parameters.SetParameters(0, h);  // Save proposed step height
}

// Compute log density of proposed values
double ScaleMixture::LogProposeDensity(const Parameters& parameters, 
                                       const Parameters& global_parameters, 
                                       const double& residual, 
                                       const double& number_of_observations, 
                                       const double& sigma,
                                       const double& omega) {
  
  // Retrieve proposed height and variance
  double proposed_height = parameters.GetParameters(0);
  double precision = 1 / (omega * eta_prior->GetVariance(parameters, 
                                                global_parameters));
  
  // Compute conditional posterior mean and variance
  double cond_post_mean = residual / (number_of_observations + precision);
  double cond_post_variance = 1 / (number_of_observations + precision);
  
  double proposed_density = -0.5 * std::log(2.0 * M_PI * cond_post_variance)
    - 0.5 * std::pow((proposed_height 
                        - cond_post_mean), 2) / cond_post_variance;
                        
                        // Return total log density (height + variance)
                        return eta_prior->LogProposeDensity(parameters, 
                                                            global_parameters, 
                                                            residual,
                                                            number_of_observations, 
                                                            sigma,
                                                            omega) + 
                                                            proposed_density;
}

// Compute log prior density
double ScaleMixture::LogPrior(const Parameters& parameters, 
                              const Parameters& global_parameters,
                              const double& sigma,
                              const double& omega) {

  
  // Retrieve proposed height and variance
  double proposed_height = parameters.GetParameters(0);
  double variance = omega * eta_prior->GetVariance(parameters, global_parameters); 
  
  // Compute log-density for normal distribution
  double normal_density = -0.5 * std::log(2.0 * M_PI * variance)
    - 0.5 * std::pow(proposed_height, 2) / variance;
  
  // Return total log prior (height + variance)
  return eta_prior->LogPrior(parameters, global_parameters, sigma, omega) + normal_density;
}

// Compute log likelihood for the given parameters
double ScaleMixture::LogLikelihood(const Parameters& parameters, 
                                   const Parameters& global_parameters, 
                                   const double& sum_of_observations, 
                                   const double& number_of_observations, 
                                   const double& sigma) {
  
  // Retrieve proposed mean and variance
  double mean = parameters.GetParameters(0);
  double variance = sigma * sigma;
  
  // Compute squared deviation from the mean
  double squared_deviation = (sum_of_observations 
                                - number_of_observations * mean) 
    * (sum_of_observations 
    - number_of_observations * mean) 
    / number_of_observations;
    
    // Compute log-likelihood
    double log_likelihood = -0.5 * number_of_observations 
    * std::log(2 * M_PI * variance)
      - squared_deviation / (2 * variance);
      
      return log_likelihood;
}

// Global update for ScaleMixture class
void ScaleMixture::GlobalUpdate(Parameters& global_parameters, 
                                const std::vector<Parameters*>& leaf_parameters, 
                                const double& sigma,
                                const double& omega,  
                                Random& random) {
  
  eta_prior->GlobalUpdate(global_parameters, leaf_parameters, omega, sigma, random);
}

// Check if global parameters are required for ScaleMixture class
bool ScaleMixture::GlobalRequired() {
  return eta_prior->GetGlobal();
}

// Function to compute the log density of the Inverse Gamma distribution
// Given a value `x` and parameters `a` and `b`, this function computes the 
// log of the probability density function (PDF) of an Inverse Gamma(a, b) distribution.
// Note: x, a, and b must all be positive. If not, the function returns 0.
double log_inverse_gamma_pdf(double x, double a, double b) {
  if (x <= 0) {
    return 0;  // Return 0 for invalid inputs to avoid computation errors
  }

  if (a <= 0) {
    return 0;  // Return 0 for invalid inputs to avoid computation errors
  }

  if (b <= 0) {
    return 0;  // Return 0 for invalid inputs to avoid computation errors
  }
  
  // Compute the log of the PDF using the formula:
  // log(f(x; a, b)) = a * log(b) - log(Gamma(a)) - (a + 1) * log(x) - (b / x)
  double log_density = a * std::log(b) - std::lgamma(a) 
    - (a + 1) * std::log(x) - (b / x);
  return log_density;
}

