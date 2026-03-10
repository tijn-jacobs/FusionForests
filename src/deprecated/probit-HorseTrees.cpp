#include "probit-HorseTrees.h"

// [[Rcpp::export]]
Rcpp::List probitHorseTrees_cpp(SEXP nSEXP,     
                      SEXP pSEXP,     
                      SEXP n_testSEXP,    
                      SEXP X_trainSEXP,    
                      SEXP ySEXP,     
                      SEXP X_testSEXP,    
                      SEXP number_of_treesSEXP,     
                      SEXP N_postSEXP,     
                      SEXP N_burnSEXP,   
                      SEXP delayed_proposalSEXP, 
                      SEXP powerSEXP,    
                      SEXP baseSEXP,
                      SEXP p_growSEXP,
                      SEXP p_pruneSEXP,     
                      SEXP omegaSEXP,
                      SEXP latent_thresholdSEXP, // Also binary_offset
                      SEXP param1SEXP,
                      SEXP param2SEXP,
                      SEXP prior_typeSEXP,
                      SEXP reversibleSEXP,
                      SEXP store_posterior_sampleSEXP, 
                      SEXP verboseSEXP
                      ) {     
      
  // Explicit conversion of SEXP to appropriate C++ types using Rcpp::as   
  size_t n = Rcpp::as<size_t>(nSEXP);   
  size_t p = Rcpp::as<size_t>(pSEXP);   
  size_t n_test = Rcpp::as<size_t>(n_testSEXP);    
  
  // Converting SEXP to Rcpp NumericVector for matrix-like data
  Rcpp::NumericVector X_train_vector(X_trainSEXP);   
  double* X_train = &X_train_vector[0];   
  
  Rcpp::IntegerVector y_vector(ySEXP);   
  int* y = &y_vector[0];   

  double* latent_z = new double[n];
  
  Rcpp::NumericVector X_test_vector(X_testSEXP);   
  double* X_test = &X_test_vector[0];    
  
  size_t number_of_trees = Rcpp::as<size_t>(number_of_treesSEXP);    
  
  size_t N_post = Rcpp::as<size_t>(N_postSEXP);   
  size_t N_burn = Rcpp::as<size_t>(N_burnSEXP);   
   
  size_t delayed_proposal = Rcpp::as<size_t>(delayed_proposalSEXP);  
  double power = Rcpp::as<double>(powerSEXP);   
  double base = Rcpp::as<double>(baseSEXP);  
  double p_grow = Rcpp::as<double>(p_growSEXP);   
  double p_prune = Rcpp::as<double>(p_pruneSEXP);    
  double omega = Rcpp::as<double>(omegaSEXP);
  double latent_threshold = Rcpp::as<double>(latent_thresholdSEXP);
  bool reversible =  Rcpp::as<bool>(reversibleSEXP);
  
  double param1 = Rcpp::as<double>(param1SEXP);   
  double param2 = Rcpp::as<double>(param2SEXP);
  string prior_type = Rcpp::as<string>(prior_typeSEXP);  
  
  bool print_progress = Rcpp::as<bool>(verboseSEXP);
  double sigma = 1;

  // Declare what has to be returned
  Rcpp::NumericVector train_predictions_mean(n);
  Rcpp::NumericVector test_predictions_mean(n_test);

  bool store_posterior_sample = Rcpp::as<bool>(store_posterior_sampleSEXP);
  Rcpp::NumericMatrix train_predictions_sample;
  Rcpp::NumericMatrix test_predictions_sample;

  if (store_posterior_sample) {
    train_predictions_sample = Rcpp::NumericMatrix(N_post, n);
    test_predictions_sample = Rcpp::NumericMatrix(N_post, n_test);
  }
  
  // Initialize the C-based random number generator
  RandomGenerator random;

  // Map the input to the corresponding PriorType
  PriorType prior;
  if (prior_type == "horseshoe") {
    prior = PriorType::Horseshoe;
  } else if (prior_type == "fixed" || prior_type == "standard") {
    prior = PriorType::FixedVariance;
  } else if (prior_type == "halfcauchy") {
    prior = PriorType::HalfCauchy;
  } else if (prior_type == "horseshoe_fw") {
    Rcpp::Rcout << "This prior has not been properly implemented for binary regression." << std::endl;
    Rcpp::stop("Invalid prior type provided.");
  } else {
    Rcpp::stop("Invalid prior type provided. Choose one of: 'horseshoe', 'half-cauchy', 'standard'.");
  }
  
  // Initialize the scale mixture prior on the step heights in the leaves
  ScaleMixture scale_mixture(prior, param1, param2);
  
  // Build the forest
  Forest forest(number_of_trees);
  forest.SetTreePrior(base, power, param1, p_grow, p_prune); // In case of NON-RJ; param1 = step height variance
  forest.SetUpForest(p, n, X_train, latent_z, nullptr, omega);

  // Initialize the latent observations
  for(size_t k = 0; k < n; k++) {
    if(y[k] == 0) {
      latent_z[k] = -rtnorm(0, latent_threshold, 1.0, random);
    } else {
      latent_z[k] = rtnorm(0, -latent_threshold, 1.0, random);
    }
  }
  
  for (size_t i = 0; i < n; i++) train_predictions_mean[i] = 0.0;
  for (size_t i = 0; i < n_test; i++) test_predictions_mean[i] = 0.0;
  
  
  //-----------------------------------------------------------
  // Temporary storage
  double* testpred = (n_test) ? new double[n_test] : nullptr;

  //-----------------------------------------------------------
  // MCMC
  if(print_progress) Rcpp::Rcout << "\nProgress of the MCMC sampler:\n\n";
  
  size_t total = N_post + N_burn;
  int barWidth = 70;
  
  bool* accepted = new bool[total * number_of_trees];
  
  time_t time_stamp;
  int time1 = time(&time_stamp);
  
  for (size_t i = 0; i < total; i++) {
    
    if(print_progress){
      // Progress bar
      float progress = static_cast<float>(i) / static_cast<float>(total);
      Rcpp::Rcout << "|";
      int pos = static_cast<int>(barWidth * progress);
      for (int j = 0; j < barWidth; ++j) {
        if (j < pos) Rcpp::Rcout << "=";
        else if (j == pos) Rcpp::Rcout << ">";
        else Rcpp::Rcout << " ";
      }
      Rcpp::Rcout << "| " << int(progress * 100.0) << " %\r";
      Rcpp::Rcout.flush();
    }

    
    // Update BART
    forest.UpdateForest(sigma, scale_mixture, reversible, delayed_proposal, random, &accepted[i * number_of_trees]);

    // Update latent variables
    for(size_t k = 0; k < n; k++) {
      if(y[k] == 0) {
        latent_z[k] = -rtnorm(-forest.GetPrediction(k), latent_threshold, 1.0, random);
      } else {
        latent_z[k] = rtnorm(forest.GetPrediction(k), -latent_threshold, 1.0, random);
      }
    }
    
    // Store the predictions at the correct sampling points
    if (i >= N_burn) {

      // Store posterior mean of training predictions
      for (size_t k = 0; k < n; k++) {
        train_predictions_mean[k] += forest.GetPrediction(k);
      }

      // Store posterior samples of training predictions
      if (store_posterior_sample) {
        for (size_t k = 0; k < n; k++) {
          train_predictions_sample(i - N_burn, k) = forest.GetPrediction(k);
        }
      }

      // Predict test set
      if (n_test > 0) {
        forest.Predict(p, n_test, X_test, testpred);
      }

      // Store posterior samples of test predictions
      if (store_posterior_sample && n_test > 0) {
        for (size_t k = 0; k < n_test; k++) {
          test_predictions_sample(i - N_burn, k) = testpred[k];
        }
      }

      // Store posterior mean of test predictions
      if (n_test > 0) {
        for (size_t k = 0; k < n_test; k++) {
          test_predictions_mean[k] += testpred[k];
        }
      }
    }

  }
    
  // Scale the predictions by the number of posterior samples
  for (size_t k = 0; k < n; k++) train_predictions_mean[k] /= N_post;
  for (size_t k = 0; k < n_test; k++) test_predictions_mean[k] /= N_post;

  // Calculate the mean acceptance ratio
  double sum_accept = 0.0;
  for (size_t i = 0; i < total * number_of_trees; ++i) {
    sum_accept += accepted[i];
  }
  double mean_accept = sum_accept / (total * number_of_trees);
  
  // Register the end time
  int time2 = time(&time_stamp);

  // Print the final progress bar 
  if(print_progress) {

    // Final update of progress bar to ensure it ends at 100%
    Rcpp::Rcout << "|";
    for (int j = 0; j < barWidth; ++j) Rcpp::Rcout << "=";
    Rcpp::Rcout << "| 100 %\r";
    Rcpp::Rcout.flush();
    Rcpp::Rcout << "\n" << std::endl;  // Move to the next line after the progress bar is done
    Rcpp::Rcout << "Mean acceptance ratio: " << mean_accept << std::endl;
    Rcpp::Rcout << "\nDone in " << (time2 - time1) << " seconds.\n";
    Rcpp::Rcout << std::endl;
  }
  
  // Store the results in a list
  Rcpp::List results;
  results["train_predictions"] = train_predictions_mean;
  results["test_predictions"] = test_predictions_mean;
  results["acceptance_ratio"] = mean_accept;
  if (store_posterior_sample) {
    results["test_predictions_sample"] = test_predictions_sample;
    results["train_predictions_sample"] = train_predictions_sample;
  }
  
  // Delete allocated memory
  if (testpred) delete[] testpred;
  delete[] accepted;
  delete[] latent_z;

  // Return the results
  return results;
}
