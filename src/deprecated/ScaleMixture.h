// This file contains the implementation of the ScaleMixture class and its 
// related prior types. It is used to manage scale mixture priors that are 
// assigned to the step height in the leaf nodes of the tree. The file provides 
// flexibility to define various types of scale mixture priors for step height 
// through a virtual class, allowing for easy extension and customization by 
// adding new priors if needed.

// The ScaleMixture class is used to wrap and manage these priors. It can 
// handle the process of proposing new step heights (step size updates), 
// calculating log densities, and updating global and local parameters.
// It also allows users to specify which prior they would like to use via a 
// factory function that creates the appropriate prior based on the user's input.

// It uses the EtaPrior class, which is a virtual base class that defines 
// the common interface for all scale mixture priors. It abstracts the key 
// functions needed to manage the prior, such as proposing new parameter 
// values, calculating log densities, and updating global or local parameters. 
// Derived classes like FixedVariance, HalfCauchy, and Horseshoe implement 
// the actual behavior for specific types of priors. EtaPrior serves as the 
// base structure for all scale mixture priors, providing a consistent 
// interface for the ScaleMixture class to interact with different priors 
// seamlessly.

// Functions defined in this file include:
// 1. Propose: This function generates a proposal for a new step height 
//    based on the current residuals, variance, and prior. It updates both 
//    local and global parameters as needed.
// 2. LogProposeDensity: Computes the log density of the proposed values 
//    (step height and variance) based on the selected prior's proposal 
//    distribution.
// 3. LogPrior: Computes the prior log density of the parameters (height 
//    and variance) using the selected scale mixture prior.
// 4. LogLikelihood: Computes the log-likelihood of the parameters (step 
//    height and variance) given the observed data and residuals.
// 5. GlobalUpdate: Updates the global parameters shared across all leaf 
//    nodes. This step is important when the prior involves both local 
//    (node-specific) and global (shared) components, such as in the 
//    Half-Cauchy or Horseshoe priors.
// 6. GetVariance: Retrieves the variance used for a particular node, based 
//    on the local or global parameters.
// 7. GlobalRequired: Indicates whether global parameters are required for 
//    the current scale mixture prior, e.g., if the prior has both local and 
//    global components like the Horseshoe or Half-Cauchy priors.

// Classes of Scale Mixture Priors:
// 1. FixedVariance: A simple model where the variance is fixed across all 
//    nodes and does not change.
// 2. HalfCauchy: A hierarchical prior that involves both local and global 
//    parameters. The local variance (lambda) is updated for each node, while 
//    the global parameter (tau) affects all nodes. This prior is often used 
//    to regularize the step heights by controlling the variance in a 
//    heavy-tailed distribution.
// 3. Horseshoe: Similar to the HalfCauchy, but with even stronger 
//    sparsity-inducing properties. It involves both local and global 
//    components, where the local parameters (lambda and nu) adapt based on 
//    the residuals, while the global parameter (tau) affects all nodes.
// 4. Horseshoe_fw: Similar to the HalfCauchy, but with global shrinkage
//    parameter defined for all trees in the forest. 
//
// Local vs. Global Parameters:
// In hierarchical models such as those implemented with these priors, there 
// is a distinction between local and global parameters:
// - Local parameters: These parameters are node-specific and capture 
//   characteristics of individual nodes in the tree. For example, each leaf 
//   node may have its own lambda, which is updated based on the data specific 
//   to that node.
// - Global parameters: These parameters are shared across all leaf nodes 
//   in the tree and help control the overall behavior of the model. For 
//   example, in the HalfCauchy prior, the global parameter tau regularizes 
//   the variance across all nodes.

// The implementation in this file follows the notation in the 
// Implementation_Notes.pdf, ensuring consistency between the theoretical 
// model and the codebase.


#ifndef GUARD_ScaleMixture_h
#define GUARD_ScaleMixture_h


#include "Information.h"

// Returns the log of the pdf of the inverse gamma distribution
double log_inverse_gamma_pdf(double x, double a, double b);

// Enum to define different types of priors
enum class PriorType {
  FixedVariance,
  HalfCauchy,
  Horseshoe,
  Horseshoe_fw
};

// Base class for EtaPrior
class EtaPrior {
public:
  // Propose new parameters for the scale mixture prior
  virtual void Propose(Parameters& parameters, 
                       const Parameters& global_parameters,
                       const double& residual, 
                       const double& number_of_observations,
                       const double& sigma,
                       const double& omega,
                       Random& random) = 0;
  
  // Update global parameters (e.g., for hierarchical priors)
  virtual void GlobalUpdate(Parameters& global_parameters,
                            const std::vector<Parameters*>& leaf_parameters,
                            const double &sigma, 
                            const double& omega,
                            Random& random) = 0;
  
  // Compute log-density of the proposed parameters
  virtual double LogProposeDensity(const Parameters& parameters,
                                   const Parameters& global_parameters,
                                   const double& residual,
                                   const double& number_of_observations,
                                   const double& sigma,
                                   const double& omega) = 0;
  
  // Compute the prior density of the parameters
  virtual double LogPrior(const Parameters& parameters,
                          const Parameters& global_parameters,
                          const double& sigma,
                          const double& omega) = 0;
  
  // Return the variance associated with the prior
  virtual double GetVariance(const Parameters& parameters,
                             const Parameters& global_parameters) = 0;
  
  // Return whether the prior requires global updates
  virtual bool GetGlobal() = 0;
  
  // Virtual destructor for base class
  virtual ~EtaPrior() {}  
};

// Factory function to create a EtaPrior based on the type and parameters
EtaPrior* CreateEtaPrior(PriorType type, 
                         double param1 = 1.0, 
                         double param2 = 1.0);

// Class for managing the scale mixture
class ScaleMixture {
private:
  EtaPrior* eta_prior;
  
public:
  // Constructor initializes with a prior type and optional parameters
  ScaleMixture(PriorType type, double param1 = 1.0, double param2 = 1.0);
  
  // Destructor ensures the EtaPrior object is deleted
  ~ScaleMixture();
  
  // Propose new parameters for the scale mixture and step height
  void Propose(Parameters& parameters, 
               const Parameters& global_parameters,
               const double& residual, 
               const double& number_of_observations,
               const double& sigma, 
               const double& omega,
               Random& random);
  
  // Update global parameters for hierarchical priors
  void GlobalUpdate(Parameters& global_parameters,
                    const std::vector<Parameters*>& leaf_parameters,
                    const double &sigma, 
                    const double& omega,
                    Random& random);
  
  // Compute the log-density of the proposed parameters
  double LogProposeDensity(const Parameters& parameters,
                           const Parameters& global_parameters,
                           const double& residual,
                           const double& number_of_observations,
                           const double& sigma,
                           const double& omega);
  
  // Compute the prior density of the parameters
  double LogPrior(const Parameters& parameters,
                  const Parameters& global_parameters,
                  const double& sigma,
                  const double& omega);
  
  // Compute the log-likelihood of the proposed step height 
  // (including the other parameters)
  double LogLikelihood(const Parameters& parameters,
                       const Parameters& global_parameters,
                       const double& sum_of_observations,
                       const double& number_of_observations,
                       const double& sigma);
  
  // Return whether the prior requires global updates
  bool GlobalRequired();
};


//                  //
// Derived classess //
//                  //


// Derived class for FixedVariance prior
class FixedVariance : public EtaPrior {
  
  // Indicates if global parameter updates are required (false for FixedVariance)
  bool global = false;
  
  // Fixed variance value (eta)
  double eta;
  
public:
  // Constructor to initialize eta (default value is 1.0)
  FixedVariance(double _eta = 1.0);
  
  // Propose new parameters based on the fixed variance prior
  void Propose(Parameters& parameters, 
               const Parameters& global_parameters,
               const double& residual, 
               const double& number_of_observations,
               const double& sigma,
               const double& omega,
              Random& random) override;
  
  // No global updates needed for FixedVariance (empty implementation)
  void GlobalUpdate(Parameters& global_parameters,
                    const std::vector<Parameters*>& leaf_parameters,
                    const double &sigma, 
                    const double& omega,
                    Random& random) override;
  
  // Log-density of the proposed parameters (returns 0 for FixedVariance)
  double LogProposeDensity(const Parameters& parameters,
                           const Parameters& global_parameters,
                           const double& residual,
                           const double& number_of_observations,
                           const double& sigma,
                           const double& omega) override;
  
  // Log-prior for the fixed variance (returns 0)
  double LogPrior(const Parameters& parameters,
                  const Parameters& global_parameters,
                  const double& sigma,
                  const double& omega) override;
  
  // Returns the fixed variance (eta)
  double GetVariance(const Parameters& parameters,
                     const Parameters& global_parameters) override;
  
  // Returns false, as FixedVariance doesn't require global updates
  bool GetGlobal() override;
};

// Derived class for Half-Cauchy prior
// Uses the auxiliary variables representation 
class HalfCauchy : public EtaPrior {
  
  // Indicates if global parameter updates are required (false for HalfCauchy)
  bool global = false;
  
  // Shape parameter (tau) and scaling factor (alpha) for Half-Cauchy prior
  double alpha;
  double tau;
  
public:
  // Constructor to initialize tau and alpha (default values are 1.0)
  HalfCauchy(double _tau = 1.0, double _alpha = 1.0);
  
  // Propose new parameters based on the Half-Cauchy prior
  void Propose(Parameters& parameters, 
               const Parameters& global_parameters,
               const double& residual, 
               const double& number_of_observations,
               const double& sigma, 
               const double& omega,
               Random& random) override;
  
  // No global updates needed for HalfCauchy (empty implementation)
  void GlobalUpdate(Parameters& global_parameters,
                    const std::vector<Parameters*>& leaf_parameters,
                    const double &sigma, 
                    const double& omega,
                    Random& random) override;
  
  // Log-density of the proposed parameters
  double LogProposeDensity(const Parameters& parameters,
                           const Parameters& global_parameters,
                           const double& residual,
                           const double& number_of_observations,
                           const double& sigma,
                           const double& omega) override;
  
  // Log-prior for the Half-Cauchy prior
  double LogPrior(const Parameters& parameters,
                  const Parameters& global_parameters,
                  const double& sigma,
                  const double& omega) override;
  
  // Returns the variance associated with the Half-Cauchy prior
  double GetVariance(const Parameters& parameters,
                     const Parameters& global_parameters) override;
  
  // Returns false, as Half-Cauchy doesn't require global updates
  bool GetGlobal() override;
};

// Derived class for Horseshoe prior
// Uses the auxiliary variables representation 
class Horseshoe : public EtaPrior {
  
  // Indicates if global parameter updates are required (true for Horseshoe)
  bool global = true;

  // Local and global scaling parameters for the Horseshoe prior
  double alpha_local;
  double alpha_global;
  
public:
  // Constructor to initialize alpha_local and alpha_global
  Horseshoe(double _alpha_local = 1.0, double _alpha_global = 1.0);
  
  // Propose new parameters based on the Horseshoe prior
  void Propose(Parameters& parameters, 
               const Parameters& global_parameters,
               const double& residual, 
               const double& number_of_observations,
               const double& sigma, 
               const double& omega,
               Random& random) override;
  
  // Update global parameters (tau) based on the Horseshoe prior
  void GlobalUpdate(Parameters& global_parameters,
                    const std::vector<Parameters*>& leaf_parameters,
                    const double &sigma, 
                    const double& omega,
                    Random& random) override;
  
  // Log-density of the proposed parameters
  double LogProposeDensity(const Parameters& parameters,
                           const Parameters& global_parameters,
                           const double& residual,
                           const double& number_of_observations,
                           const double& sigma,
                           const double& omega) override;
  
  // Log-prior for the Horseshoe prior
  double LogPrior(const Parameters& parameters,
                  const Parameters& global_parameters,
                  const double& sigma,
                  const double& omega) override;
  
  // Returns the variance associated with the Horseshoe prior
  double GetVariance(const Parameters& parameters,
                     const Parameters& global_parameters) override;
  
  // Returns true, as Horseshoe requires global updates
  bool GetGlobal() override;
};

// Derived class for Horseshoe Forest Wide prior
// Uses the auxiliary variables representation 
class Horseshoe_fw : public EtaPrior {
  
  // Indicates if global parameter updates are required (true for Horseshoe)
  bool global = false;

  // Local and global scaling parameters for the Horseshoe prior
  double alpha_local;
  
public:
  // Constructor to initialize alpha_local 
  Horseshoe_fw(double _alpha_local = 1.0);
  
  // Propose new parameters based on the Horseshoe prior
  void Propose(Parameters& parameters, 
               const Parameters& global_parameters,
               const double& residual, 
               const double& number_of_observations,
               const double& sigma, 
               const double& omega,
               Random& random) override;
  
  // Update global parameters (tau) based on the Horseshoe prior
  void GlobalUpdate(Parameters& global_parameters,
                    const std::vector<Parameters*>& leaf_parameters,
                    const double &sigma, 
                    const double& omega,
                    Random& random) override;
  
  // Log-density of the proposed parameters
  double LogProposeDensity(const Parameters& parameters,
                           const Parameters& global_parameters,
                           const double& residual,
                           const double& number_of_observations,
                           const double& sigma,
                           const double& omega) override;
  
  // Log-prior for the Horseshoe prior
  double LogPrior(const Parameters& parameters,
                  const Parameters& global_parameters,
                  const double& sigma,
                  const double& omega) override;
  
  // Returns the variance associated with the Horseshoe prior
  double GetVariance(const Parameters& parameters,
                     const Parameters& global_parameters) override;
  
  // Returns true, as Horseshoe requires global updates
  bool GetGlobal() override;
};


#endif // GUARD_ScaleMixture_h
