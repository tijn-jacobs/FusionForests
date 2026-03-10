#include "CausalHorseForest.h"
#include "ForestEngine.h"
#include <cmath>

// [[Rcpp::export]]
Rcpp::List CausalHorseForest_cpp(
  SEXP nSEXP, 
  SEXP p_treatSEXP, 
  SEXP p_controlSEXP, 
  SEXP X_train_treatSEXP,
  SEXP X_train_controlSEXP, 
  SEXP ySEXP, 
  SEXP status_indicatorSEXP,
  SEXP is_survivalSEXP,
  SEXP observed_left_timeSEXP,
  SEXP observed_right_timeSEXP,
  SEXP interval_censoring_indicatorSEXP,
  SEXP treatment_indicatorSEXP,
  SEXP n_testSEXP, 
  SEXP X_test_controlSEXP, 
  SEXP X_test_treatSEXP, 
  SEXP treatment_indicator_testSEXP,
  SEXP no_trees_treatSEXP, 
  SEXP power_treatSEXP, 
  SEXP base_treatSEXP,
  SEXP p_grow_treatSEXP, 
  SEXP p_prune_treatSEXP, 
  SEXP omega_treatSEXP,
  SEXP prior_type_treatSEXP, 
  SEXP param1_treatSEXP, 
  SEXP param2_treatSEXP,
  SEXP reversible_treatSEXP, 
  SEXP dirichlet_bool_treatSEXP,
  SEXP a_dirichlet_treatSEXP,
  SEXP b_dirichlet_treatSEXP, 
  SEXP rho_dirichlet_treatSEXP,  
  SEXP no_trees_controlSEXP,
  SEXP power_controlSEXP, 
  SEXP base_controlSEXP, 
  SEXP p_grow_controlSEXP,
  SEXP p_prune_controlSEXP, 
  SEXP omega_controlSEXP, 
  SEXP prior_type_controlSEXP,
  SEXP param1_controlSEXP, 
  SEXP param2_controlSEXP, 
  SEXP reversible_controlSEXP,
  SEXP dirichlet_bool_controlSEXP,
  SEXP a_dirichlet_controlSEXP,
  SEXP b_dirichlet_controlSEXP, 
  SEXP rho_dirichlet_controlSEXP,  
  SEXP sigma_knownSEXP, 
  SEXP sigmaSEXP, 
  SEXP lambdaSEXP,
  SEXP nuSEXP, 
  SEXP N_postSEXP, 
  SEXP N_burnSEXP, 
  SEXP delayed_proposalSEXP,
  SEXP store_posterior_sample_controlSEXP,
  SEXP store_posterior_sample_treatSEXP,
  SEXP verboseSEXP,
  SEXP treatment_codingSEXP,
  SEXP propensity_trainSEXP,
  SEXP propensity_testSEXP
) {

  // Conversion of function arguments //

  // Parameters and data for the training phase
  size_t n = Rcpp::as<size_t>(nSEXP);
  size_t p_treat = Rcpp::as<size_t>(p_treatSEXP);
  size_t p_control = Rcpp::as<size_t>(p_controlSEXP);
  Rcpp::NumericVector X_train_treat_vector(X_train_treatSEXP);   
  double* X_train_treat = &X_train_treat_vector[0];   
  Rcpp::NumericVector X_train_control_vector(X_train_controlSEXP);   
  double* X_train_control = &X_train_control_vector[0];   
  Rcpp::NumericVector y_vector(ySEXP);   
  double* y = &y_vector[0];  
  bool is_survival = Rcpp::as<bool>(is_survivalSEXP);
  Rcpp::IntegerVector treatment_indicator_vector(treatment_indicatorSEXP);
  int* treatment_indicator = &treatment_indicator_vector[0];
  Rcpp::NumericVector status_indicator_vector(status_indicatorSEXP);
  double* status_indicator = &status_indicator_vector[0];
  std::vector<double> y_observed_vector(y_vector.begin(), y_vector.end());  // Create an independent copy of y
  double* y_observed = y_observed_vector.data(); // Get pointer to use like an array

  // Objects for interval censoring
  Rcpp::NumericVector observed_left_time_vector(observed_left_timeSEXP);
  double* observed_left_time = &observed_left_time_vector[0];
  Rcpp::NumericVector observed_right_time_vector(observed_right_timeSEXP);
  double* observed_right_time = &observed_right_time_vector[0];
  Rcpp::NumericVector interval_censoring_indicator_vector(interval_censoring_indicatorSEXP);
  double* interval_censoring_indicator = &interval_censoring_indicator_vector[0];

  // Parameters and data for the test phase
  size_t n_test = Rcpp::as<size_t>(n_testSEXP);    
  Rcpp::NumericVector X_test_treat_vector(X_test_treatSEXP);   
  double* X_test_treat = &X_test_treat_vector[0];   
  Rcpp::NumericVector X_test_control_vector(X_test_controlSEXP);   
  double* X_test_control = &X_test_control_vector[0];   
  Rcpp::IntegerVector treatment_indicator_test_vector(treatment_indicator_testSEXP);
  int* treatment_indicator_test = &treatment_indicator_test_vector[0];

  // Hyperparameters treatment effect model
  size_t no_trees_treat = Rcpp::as<size_t>(no_trees_treatSEXP);    
  double power_treat = Rcpp::as<double>(power_treatSEXP);
  double base_treat = Rcpp::as<double>(base_treatSEXP);
  double p_grow_treat = Rcpp::as<double>(p_grow_treatSEXP);
  double p_prune_treat = Rcpp::as<double>(p_prune_treatSEXP);
  double omega_treat = Rcpp::as<double>(omega_treatSEXP);
  string prior_type_treat = Rcpp::as<string>(prior_type_treatSEXP);
  double param1_treat = Rcpp::as<double>(param1_treatSEXP);
  double param2_treat = Rcpp::as<double>(param2_treatSEXP);
  bool reversible_treat = Rcpp::as<bool>(reversible_treatSEXP);
  bool dirichlet_bool_treat = Rcpp::as<bool>(dirichlet_bool_treatSEXP);
  double a_dirichlet_treat = Rcpp::as<double>(a_dirichlet_treatSEXP);
  double b_dirichlet_treat = Rcpp::as<double>(b_dirichlet_treatSEXP); 
  double rho_dirichlet_treat = Rcpp::as<double>(rho_dirichlet_treatSEXP);

  // Hyperparameters prognostic model
  size_t no_trees_control = Rcpp::as<size_t>(no_trees_controlSEXP);
  double power_control = Rcpp::as<double>(power_controlSEXP);
  double base_control = Rcpp::as<double>(base_controlSEXP);
  double p_grow_control = Rcpp::as<double>(p_grow_controlSEXP);
  double p_prune_control = Rcpp::as<double>(p_prune_controlSEXP);
  double omega_control = Rcpp::as<double>(omega_controlSEXP);
  string prior_type_control = Rcpp::as<string>(prior_type_controlSEXP);
  double param1_control = Rcpp::as<double>(param1_controlSEXP);
  double param2_control = Rcpp::as<double>(param2_controlSEXP);
  bool reversible_control = Rcpp::as<bool>(reversible_controlSEXP);
  bool dirichlet_bool_control = Rcpp::as<bool>(dirichlet_bool_controlSEXP);
  double a_dirichlet_control = Rcpp::as<double>(a_dirichlet_controlSEXP);
  double b_dirichlet_control = Rcpp::as<double>(b_dirichlet_controlSEXP); 
  double rho_dirichlet_control = Rcpp::as<double>(rho_dirichlet_controlSEXP);

  // Hyperparameters error variance model
  bool sigma_known = Rcpp::as<bool>(sigma_knownSEXP);
  double sigma = Rcpp::as<double>(sigmaSEXP);
  double lambda = Rcpp::as<double>(lambdaSEXP);
  double nu = Rcpp::as<double>(nuSEXP);

  // Computational parameters
  size_t N_post = Rcpp::as<size_t>(N_postSEXP);
  size_t N_burn = Rcpp::as<size_t>(N_burnSEXP);
  size_t delayed_proposal = Rcpp::as<size_t>(delayed_proposalSEXP);

  // Storage parameters
  bool store_posterior_sample_control = Rcpp::as<bool>(store_posterior_sample_controlSEXP);
  bool store_posterior_sample_treat = Rcpp::as<bool>(store_posterior_sample_treatSEXP);

  // Verbose
  bool verbose = Rcpp::as<bool>(verboseSEXP);

  // Treatment coding
  string treatment_coding = Rcpp::as<string>(treatment_codingSEXP);
  Rcpp::NumericVector propensity_train_vector(propensity_trainSEXP);
  double* propensity_train = &propensity_train_vector[0];
  Rcpp::NumericVector propensity_test_vector(propensity_testSEXP);
  double* propensity_test = &propensity_test_vector[0];

  // Precompute treatment coding weights b_train[i] and b_test[i]
  double* b_train = new double[n];
  double* b_test = new double[n_test];

  bool invariant = (treatment_coding == "invariant");
  double b0 = -0.5, b1 = 0.5;  // initial values for invariant coding

  if (treatment_coding == "binary") {
    for (size_t k = 0; k < n; k++)
      b_train[k] = (treatment_indicator[k] == 1) ? 1.0 : 0.0;
    for (size_t k = 0; k < n_test; k++)
      b_test[k] = (treatment_indicator_test[k] == 1) ? 1.0 : 0.0;
  } else if (treatment_coding == "adaptive") {
    for (size_t k = 0; k < n; k++)
      b_train[k] = static_cast<double>(treatment_indicator[k]) - propensity_train[k];
    for (size_t k = 0; k < n_test; k++)
      b_test[k] = static_cast<double>(treatment_indicator_test[k]) - propensity_test[k];
  } else {
    // Default (centered) and invariant both start at (-0.5, 0.5)
    for (size_t k = 0; k < n; k++)
      b_train[k] = (treatment_indicator[k] == 1) ? 0.5 : -0.5;
    for (size_t k = 0; k < n_test; k++)
      b_test[k] = (treatment_indicator_test[k] == 1) ? 0.5 : -0.5;
  }

  // Storage for invariant coding posterior draws of b0 and b1
  Rcpp::NumericVector store_b0, store_b1;
  if (invariant) {
    store_b0 = Rcpp::NumericVector(N_post);
    store_b1 = Rcpp::NumericVector(N_post);
  }


  // Declare storage containers //

  // Storage for training and test predictions (posterior mean)
  Rcpp::NumericVector train_predictions_mean(n);
  Rcpp::NumericVector test_predictions_mean(n_test);
  train_predictions_mean.fill(0.0);
  test_predictions_mean.fill(0.0);  

  // Storage for training and test predictions (posterior sample) of the prognostic model
  Rcpp::NumericVector train_predictions_mean_control(n);
  Rcpp::NumericVector test_predictions_mean_control(n_test);
  train_predictions_mean_control.fill(0.0);
  test_predictions_mean_control.fill(0.0);  

  // Storage for training and test predictions (posterior sample) of the treatment effect model
  Rcpp::NumericVector train_predictions_mean_treat(n);
  Rcpp::NumericVector test_predictions_mean_treat(n_test);
  train_predictions_mean_treat.fill(0.0);
  test_predictions_mean_treat.fill(0.0);

  // Storage for training and test predictions (posterior sample) of the prognostic model
  Rcpp::NumericMatrix train_predictions_sample_control;
  Rcpp::NumericMatrix test_predictions_sample_control;
  if (store_posterior_sample_control) { // Allocate memory for posterior samples, if requested
    train_predictions_sample_control = Rcpp::NumericMatrix(N_post, n);
    test_predictions_sample_control = Rcpp::NumericMatrix(N_post, n_test);
  }

  // Storage for training and test predictions (posterior sample) of the treatment effect model
  Rcpp::NumericMatrix train_predictions_sample_treat;
  Rcpp::NumericMatrix test_predictions_sample_treat;
  if (store_posterior_sample_treat) { // Allocate memory for posterior samples, if requested
    train_predictions_sample_treat = Rcpp::NumericMatrix(N_post, n);
    test_predictions_sample_treat = Rcpp::NumericMatrix(N_post, n_test);
  }

  // Declare sigma storage based on whether sigma is known
  Rcpp::NumericVector store_sigma = sigma_known ? Rcpp::NumericVector::create(sigma) : Rcpp::NumericVector(N_post + N_burn);

  // Declare parameters for keeping track of the acceptance ratio
  bool* accepted_control = new bool[no_trees_control];  
  for (size_t j = 0; j < no_trees_control; ++j) accepted_control[j] = false;
  double sum_accept_control = 0;
  double acceptance_ratio_control; 
  bool* accepted_treat = new bool[no_trees_treat];
  for (size_t j = 0; j < no_trees_treat; ++j) accepted_treat[j] = false;
  double sum_accept_treat = 0;
  double acceptance_ratio_treat;

  // Allocate memory for the test predictions of both models
  double* testpred_treat = (n_test) ? new double[n_test] : nullptr;
  double* testpred_control = (n_test) ? new double[n_test] : nullptr;

  // Allocate a new array for the overall predictions
  double* total_predictions = new double[n]; // hat{y_i} = m(x_i) + b_i * tau(x_i)

  // Variables for the augmented outcome for both models
  // These need to be updated after each update of the other model
  double* augmented_outcome_treat = new double[n];      // (y_i - m(x_i))/b_i
  double* augmented_outcome_control = new double[n];    //  y_i - b_i * tau(x_i)
  double* treat_weights = new double[n];                // b_i^2 weights for treatment forest

  // Initialize the augmented outcome for both models
  for (size_t i = 0; i < n; i++) {
    augmented_outcome_treat[i] = y[i]/2;
    augmented_outcome_control[i] = y[i]/2;
  }

  // If prior_type = 6, we have a Horseshoe with forest wide shrinkage.
  // These updates are done in the outer Gibbs step.
  double forestwide_shrinkage_control = 1.0;
  double forestwide_shrinkage_treat = 1.0;
  double forestwide_auxiliary_control = 1.0;
  double forestwide_auxiliary_treat = 1.0;
  Rcpp::NumericVector store_forestwide_shrinkage_control;
  Rcpp::NumericVector store_forestwide_shrinkage_treat;

  if (prior_type_control == "horseshoe_fw") {
    store_forestwide_shrinkage_control = Rcpp::NumericVector(N_post);
    store_forestwide_shrinkage_control.fill(0);
  }  

  if (prior_type_treat == "horseshoe_fw") {
    store_forestwide_shrinkage_treat = Rcpp::NumericVector(N_post);
    store_forestwide_shrinkage_treat.fill(0);
  }

  // Declare storage for Dirichlet prior on variable selection
  Rcpp::NumericVector store_alpha_dirichlet_control;
  Rcpp::NumericVector store_alpha_dirichlet_treat;
  Rcpp::NumericMatrix store_split_probs_control;
  Rcpp::NumericMatrix store_split_probs_treat;
  Rcpp::NumericMatrix store_split_counts_control;
  Rcpp::NumericMatrix store_split_counts_treat;

  store_split_counts_control = Rcpp::NumericMatrix(N_post, p_control);
  store_split_counts_treat = Rcpp::NumericMatrix(N_post, p_treat);
  double alpha_dirichlet_control = 1.0;
  double alpha_dirichlet_treat = 1.0;

  if (dirichlet_bool_control) {
    store_alpha_dirichlet_control = Rcpp::NumericVector(N_post);
    store_split_probs_control     = Rcpp::NumericMatrix(N_post, p_control);
  }
  if (dirichlet_bool_treat) {
    store_alpha_dirichlet_treat = Rcpp::NumericVector(N_post);
    store_split_probs_treat     = Rcpp::NumericMatrix(N_post, p_treat);
  }
  
  Rcpp::NumericVector store_global_parameters_treat;
  Rcpp::NumericVector store_global_parameters_control;
  if (prior_type_treat == "standard-halfcauchy") {
    store_global_parameters_treat = Rcpp::NumericVector(N_post + N_burn);
  }
  if (prior_type_control == "standard-halfcauchy") {
    store_global_parameters_control = Rcpp::NumericVector(N_post + N_burn);
  }

  // Set-up the forest for the prognostic model // 

  PriorType prior_control;
  if (prior_type_control == "horseshoe") {
    prior_control = PriorType::Horseshoe;
  } else if (prior_type_control == "fixed" || prior_type_control == "standard") {
    prior_control = PriorType::FixedVariance;
  } else if (prior_type_control == "halfcauchy") {
    prior_control = PriorType::HalfCauchy;
  } else if (prior_type_control == "horseshoe_fw") {
    prior_control = PriorType::Horseshoe_fw;
  } else if (prior_type_control == "dirichlet") {
    prior_control = PriorType::FixedVariance; // make an empty one
  } else if (prior_type_control == "standard-halfcauchy") {
    prior_control = PriorType::FixedVariance; // make an empty one
  } else if (prior_type_control == "standard-halfnormal") {
    prior_control = PriorType::FixedVariance; // make an empty one
  } else {
    Rcpp::stop("Invalid prior type for prognostic forest. Choose one of: 'horseshoe', 'fixed', 'halfcauchy', 'horseshoe_fw', 'standard'. 'standard-halfcauchy, 'standard-halfnormal'.");
  }
  // Initialize the scale mixture prior on the step heights in the leaves for the prognostic model
  ScaleMixture scale_mixture_control(prior_control, param1_control, param2_control);

  // Build the forest 
  ForestEngineType engine_type_control;
  if (reversible_control) {
    engine_type_control = ForestEngineType::forest_type;
  } else {
    engine_type_control = ForestEngineType::stan_forest_type;
  }
  ForestEngine forest_control(engine_type_control, no_trees_control);
  forest_control.SetTreePrior(base_control, power_control, param1_control, p_grow_control, p_prune_control, a_dirichlet_control, b_dirichlet_control, rho_dirichlet_control, false, dirichlet_bool_control, alpha_dirichlet_control); // In case of NON-RJ; param1 = step height variance
  forest_control.SetUpForest(p_control, n, X_train_control, augmented_outcome_control, nullptr, omega_control); // Use augmented outcome for y

  // Setup a vector to access the tree
  // This is only needed (!) for the forestwide shrinkage update.
  // Consider removing this if not needed elsewhere, or empty initialization.
  // std::vector<Tree>* trees_control = forest_control.GetTreesPointer();


  // Set-up the forest for the treatment effect model // 

  PriorType prior_treat;
  if (prior_type_treat == "horseshoe") {
    prior_treat = PriorType::Horseshoe;
  } else if (prior_type_treat == "fixed" || prior_type_treat == "standard") {
    prior_treat = PriorType::FixedVariance;
  } else if (prior_type_treat == "halfcauchy") {
    prior_treat = PriorType::HalfCauchy;
  } else if (prior_type_treat == "horseshoe_fw") {
    prior_treat = PriorType::Horseshoe_fw;
  } else if (prior_type_treat == "dirichlet") {
    prior_treat = PriorType::FixedVariance; // make an empty one
  } else if (prior_type_treat == "standard-halfcauchy") {
    prior_treat = PriorType::FixedVariance; // make an empty one
  } else if (prior_type_treat == "standard-halfnormal") {
    prior_treat = PriorType::FixedVariance; // make an empty one
  } else {
    Rcpp::stop("Invalid prior type for treatment effect forest. Choose one of: 'horseshoe', 'fixed', 'halfcauchy', 'horseshoe_fw', 'standard'.");
  }

  // Initialize the scale mixture prior on the step heights in the leaves for the prognostic model
  ScaleMixture scale_mixture_treat(prior_treat, param1_treat, param2_treat);

  // Build the forest
  ForestEngineType engine_type_treat;
  if (reversible_treat) {
    engine_type_treat = ForestEngineType::forest_type;
  } else {
    engine_type_treat = ForestEngineType::stan_forest_type;
  }
  ForestEngine forest_treat(engine_type_treat, no_trees_treat);
  forest_treat.SetTreePrior(base_treat, power_treat, param1_treat, p_grow_treat, p_prune_treat, a_dirichlet_treat, b_dirichlet_treat, rho_dirichlet_treat, false, dirichlet_bool_treat, alpha_dirichlet_treat); // In case of NON-RJ; param1 = step height variance
  forest_treat.SetUpForest(p_treat, n, X_train_treat, augmented_outcome_treat, nullptr, omega_treat); // Use augmented outcome for y

  // Setup a vector to access the trees
  // std::vector<Tree>* trees_treat = forest_treat.GetTreesPointer();

  // Initialize the C-based random number generator
  RandomGenerator random;

  // Start the clock
  time_t time_stamp;
  int time_start = time(&time_stamp);

  // Initialize the progress bar
  int barWidth = 70;
  if(verbose) Rcpp::Rcout << "\nProgress of the MCMC sampler:\n\n";

  // Start the MCMC loop
  for (size_t i = 0; i < N_post + N_burn; i++) {

    // Print progress bar
    if (verbose) {

      // Calculate progress as a float between 0 and 1
      float progress = static_cast<float>(i) / static_cast<float>(N_post + N_burn);
    
      // Calculate the number of characters to fill in the bar
      int pos = static_cast<int>(barWidth * progress);
    
      // Begin drawing the progress bar
      Rcpp::Rcout << "|";
    
      // Fill the bar with '=' for completed, '>' for the current step, and ' ' for remaining
      for (int j = 0; j < barWidth; ++j) {
        if (j < pos) {
          Rcpp::Rcout << "=";
        } else if (j == pos) {
          Rcpp::Rcout << ">";
        } else {
          Rcpp::Rcout << " ";
        }
      }
    
      // Display the percentage completed and return carriage
      Rcpp::Rcout << "| " << static_cast<int>(progress * 100.0) << " %\r";
      Rcpp::Rcout.flush(); // Ensure the output is immediately written to the console
    }

    // Initialize the Dirichlet prior on variable selection, if applicable
    if(i == (N_burn/2) && dirichlet_bool_control) {
      forest_control.StartDirichlet();
    }
    if(i == (N_burn/2) && dirichlet_bool_treat) {
      forest_treat.StartDirichlet();
    }

    // Update the prognostic forest 
    forest_control.UpdateForest(
      sigma, 
      scale_mixture_control, 
      reversible_control, 
      delayed_proposal, 
      random, 
      accepted_control
    );

    /*
    // Update the forestwide shrinkage parameter of the prognostic forest, if applicable 
    UpdateForestwideShrinkage(
      prior_type_control,
      trees_control, 
      random, 
      forestwide_auxiliary_control, 
      forestwide_shrinkage_control, 
      param2_control
    );
    */

    // Update the augmented outcome and weights for the treatment effect model
    // The augmented outcome is (y - mu(x)) / b, which has noise sigma/|b|.
    // To correct for this heteroscedasticity, we use weights w_i = b_i^2
    // so that weighted sufficient statistics yield the correct posterior.
    for (size_t k = 0; k < n; k++) {
      double b = b_train[k];
      treat_weights[k] = b * b;
      if (std::abs(b) < 1e-10) {
        augmented_outcome_treat[k] = 0.0;
        treat_weights[k] = 0.0;
      } else {
        augmented_outcome_treat[k] = (y[k] - forest_control.GetPrediction(k)) / b;
      }
    }
    forest_treat.SetWeights(treat_weights);

    // Update the treatment effect forest
    forest_treat.UpdateForest(
      sigma, 
      scale_mixture_treat, 
      reversible_treat, 
      delayed_proposal, 
      random, 
      accepted_treat
    );
    forest_treat.SetWeights(nullptr);  // Clear weights after treatment forest update

    /*
    // Update the forestwide shrinkage parameter of the treatment effect forest, if applicable
    UpdateForestwideShrinkage(
      prior_type_treat,
      trees_treat, 
      random, 
      forestwide_auxiliary_treat, 
      forestwide_shrinkage_treat, 
      param2_treat
    );
    */

    // Update the augmented outcome for the prognostic model
    for (size_t k = 0; k < n; k++) {
      double b = b_train[k];
      augmented_outcome_control[k] = y[k] - b * forest_treat.GetPrediction(k);
    }

    // Update b0, b1 for invariant treatment coding (outer Gibbs step)
    if (invariant) {
      UpdateInvariantCoding(
        b0, b1, b_train, b_test,
        y,
        forest_control.GetPredictions(),
        forest_treat.GetPredictions(),
        treatment_indicator, treatment_indicator_test,
        n, n_test, sigma, random
      );
    }

    // Compute the current predictions of the (total) outcome
    for (size_t k = 0; k < n; k++) {
      double b = b_train[k];
      total_predictions[k] = forest_control.GetPredictions()[k] + b*forest_treat.GetPredictions()[k];
    }

    // Perform updates of the global eta parameters for both forest if STANDARD-HALFCAUCHY
    if (prior_type_treat == "standard-halfcauchy") {
      forest_treat.UpdateGlobalScaleParameters(
        prior_type_treat,
        param1_treat,
        store_global_parameters_treat[i],
        random
      );
    }
    if (prior_type_control == "standard-halfcauchy") {
      forest_control.UpdateGlobalScaleParameters(
        prior_type_control,
        param1_control,
        store_global_parameters_control[i],
        random
      );
    }

    // Update sigma (outer Gibbs step), if applicable
    UpdateSigma(
      sigma_known,
      sigma,
      store_sigma,
      i,
      y,
      n,
      total_predictions, 
      nu,
      lambda,
      random
    );

    // Augment the censored data (right-censored and interval-censored)
    AugmentCensoredObservations(
      is_survival,
      y,
      observed_left_time,
      status_indicator,
      observed_right_time,
      interval_censoring_indicator,
      total_predictions,
      sigma,
      n,
      random
    );

    // Storing the posterior info after burn-in
    if (i >= N_burn) {

      // For invariant coding, the actual treatment effect is (b1 - b0) * tau_tilde(x)
      double scale_treat = invariant ? (b1 - b0) : 1.0;

      // Store the posterior mean of the training predictions
      for (size_t k = 0; k < n; k++) {
        double b = b_train[k];
        train_predictions_mean[k] += forest_control.GetPrediction(k) + b * forest_treat.GetPrediction(k);
        train_predictions_mean_control[k] += forest_control.GetPrediction(k);
        train_predictions_mean_treat[k] += scale_treat * forest_treat.GetPrediction(k);
      }

      // Store the posterior sample of training predictions
      if (store_posterior_sample_control) {
        for (size_t k = 0; k < n; k++) {
          train_predictions_sample_control(i - N_burn, k) = forest_control.GetPrediction(k);
        }
      }

      // Store the posterior sample of training predictions
      if (store_posterior_sample_treat) {
        for (size_t k = 0; k < n; k++) {
          train_predictions_sample_treat(i - N_burn, k) = scale_treat * forest_treat.GetPrediction(k);
        }
      }

      // Predict test set outcomes
      if (n_test > 0) {
        forest_control.Predict(p_control, n_test, X_test_control, testpred_control);
        forest_treat.Predict(p_treat, n_test, X_test_treat, testpred_treat);
      }

      // Store posterior sample of test predictions
      if (store_posterior_sample_control && n_test > 0) {
        for (size_t k = 0; k < n_test; k++) {
          test_predictions_sample_control(i - N_burn, k) = testpred_control[k];
        }
      }

      // Store posterior sample of test predictions
      if (store_posterior_sample_treat && n_test > 0) {
        for (size_t k = 0; k < n_test; k++) {
          test_predictions_sample_treat(i - N_burn, k) = scale_treat * testpred_treat[k];
        }
      }

      // Store posterior mean of test predictions
      for (size_t k = 0; k < n_test; k++) {
        double b = b_test[k];
        test_predictions_mean[k] += testpred_control[k] + b * testpred_treat[k];
        test_predictions_mean_control[k] += testpred_control[k];
        test_predictions_mean_treat[k] += scale_treat * testpred_treat[k];
      }

      // Store b0, b1 posterior draws for invariant coding
      if (invariant) {
        store_b0[i - N_burn] = b0;
        store_b1[i - N_burn] = b1;
      }

      // Track acceptance of the prognostic model
      for (size_t j = 0; j < no_trees_control; j++) {
        sum_accept_control += accepted_control[j];
      }

      // Track acceptance of the treatment effect model
      for (size_t j = 0; j < no_trees_treat; j++) {
        sum_accept_treat += accepted_treat[j];
      }

      // Store the tree topology parameters of the prognostic model
      if (dirichlet_bool_control) {
        const std::vector<double>& probs_control = forest_control.GetVariableInclusionProb();
        for (size_t j = 0; j < p_control; ++j) {
          store_split_probs_control(i - N_burn, j) = probs_control[j];
        }
      }
      const std::vector<size_t>& counts_control = forest_control.GetVariableInclusionCount();
      for (size_t j = 0; j < p_control; ++j) store_split_counts_control(i - N_burn, j) = counts_control[j];

      // Store the tree topology parameters of the treatment effect model
      if (dirichlet_bool_treat) {
        const std::vector<double>& probs_treat = forest_treat.GetVariableInclusionProb();
        for (size_t j = 0; j < p_treat; ++j) {
          store_split_probs_treat(i - N_burn, j) = probs_treat[j];
        }
      }
      const std::vector<size_t>& counts_treat = forest_treat.GetVariableInclusionCount();
      for (size_t j = 0; j < p_treat; ++j) store_split_counts_treat(i - N_burn, j) = counts_treat[j];
      
      // Store the forestwide shrinkage parameters, if applicable
      if(prior_type_control == "horseshoe_fw") {
        store_forestwide_shrinkage_control(i - N_burn) = forestwide_shrinkage_control;
      }

      if(prior_type_treat == "horseshoe_fw") {
        store_forestwide_shrinkage_treat(i - N_burn) = forestwide_shrinkage_treat;
      }
    }

    
  } // End of MCMC loop

  // Calculate the acceptance ratio
  acceptance_ratio_control = sum_accept_control / (N_post * no_trees_control);
  acceptance_ratio_treat = sum_accept_treat / (N_post * no_trees_treat);

  // Register the end time
  int time_end = time(&time_stamp);

  if(verbose) {
    
    // Final update of progress bar to ensure it ends at 100%
    Rcpp::Rcout << "|";
    for (int j = 0; j < barWidth; ++j) Rcpp::Rcout << "=";
    Rcpp::Rcout << "| 100 %\r";
    Rcpp::Rcout.flush();
    Rcpp::Rcout << "\n" << std::endl;  // Move to the next line after the progress bar is done.
    Rcpp::Rcout << "Mean acceptance ratio (prognostic model): " << acceptance_ratio_control << std::endl;
    Rcpp::Rcout << "Mean acceptance ratio (treatment effect model): " << acceptance_ratio_treat << std::endl;
    Rcpp::Rcout << "\nDone in " << (time_end - time_start) << " seconds.\n";
    Rcpp::Rcout << std::endl;
  }

  // Rescale the predictions by the number of posterior samples
  for (size_t k = 0; k < n; k++) {
    train_predictions_mean[k] /= N_post;
    train_predictions_mean_control[k] /= N_post;
    train_predictions_mean_treat[k] /= N_post;
  }

  for (size_t k = 0; k < n_test; k++) {
    test_predictions_mean[k] /= N_post;
    test_predictions_mean_control[k] /= N_post;
    test_predictions_mean_treat[k] /= N_post;
  }

  // Declare the results to be returned
  Rcpp::List results;
  results["sigma"] = store_sigma;
  results["test_predictions"] = test_predictions_mean;
  results["train_predictions"] = train_predictions_mean;
  results["test_predictions_control"] = test_predictions_mean_control;
  results["train_predictions_control"] = train_predictions_mean_control;
  results["test_predictions_treat"] = test_predictions_mean_treat;
  results["train_predictions_treat"] = train_predictions_mean_treat;
  results["acceptance_ratio_control"] = acceptance_ratio_control;
  results["acceptance_ratio_treat"] = acceptance_ratio_treat;
  if (store_posterior_sample_control) {
    results["test_predictions_sample_control"] = test_predictions_sample_control;
    results["train_predictions_sample_control"] = train_predictions_sample_control;
  }
  if (store_posterior_sample_treat) {
    results["test_predictions_sample_treat"] = test_predictions_sample_treat;
    results["train_predictions_sample_treat"] = train_predictions_sample_treat;
  } 
  if (prior_type_control == "horseshoe_fw") {
    results["forestwide_shrinkage_control"] = store_forestwide_shrinkage_control;
  }
  if (prior_type_treat == "horseshoe_fw") {
    results["forestwide_shrinkage_treat"] = store_forestwide_shrinkage_treat;
  }
  if (dirichlet_bool_control) {
    results["alpha_dirichlet_control"] = store_alpha_dirichlet_control;
    results["split_probs_control"] = store_split_probs_control; 
  }
  if (dirichlet_bool_treat) {
    results["alpha_dirichlet_treat"] = store_alpha_dirichlet_treat;
    results["split_probs_treat"] = store_split_probs_treat; 
  }
  results["split_counts_control"] = store_split_counts_control;
  results["split_counts_treat"] = store_split_counts_treat;
  
  if (prior_type_treat == "standard-halfcauchy") {
    results["global_parameters_treat"] = store_global_parameters_treat;
  }
  if (prior_type_control == "standard-halfcauchy") {
    results["global_parameters_control"] = store_global_parameters_control;
  }
  if (invariant) {
    results["b0"] = store_b0;
    results["b1"] = store_b1;
  }

  // Clean up memory
  if (testpred_control) delete[] testpred_control;
  if (testpred_treat) delete[] testpred_treat;
  delete[] accepted_control;
  delete[] accepted_treat;
  delete[] total_predictions;
  delete[] augmented_outcome_control;
  delete[] augmented_outcome_treat;
  delete[] treat_weights;
  delete[] b_train;
  delete[] b_test;

  return results;
}
