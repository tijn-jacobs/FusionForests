#include "FusionForest.h"

// [[Rcpp::export]]
Rcpp::List FusionForest_cpp(
  SEXP nSEXP, SEXP p_treatSEXP, SEXP p_controlSEXP, SEXP X_train_treatSEXP,
  SEXP X_train_controlSEXP, SEXP ySEXP, SEXP status_indicatorSEXP, SEXP is_survivalSEXP,
  SEXP treatment_indicatorSEXP, SEXP source_indicatorSEXP,
  SEXP n_testSEXP, SEXP X_test_controlSEXP, SEXP X_test_treatSEXP, SEXP X_test_deconfSEXP,
  SEXP treatment_indicator_testSEXP, SEXP source_indicator_testSEXP,
  SEXP n_deconfSEXP, SEXP p_deconfSEXP, SEXP X_train_deconfSEXP,
  SEXP no_trees_deconfSEXP, SEXP power_deconfSEXP, SEXP base_deconfSEXP,
  SEXP p_grow_deconfSEXP, SEXP p_prune_deconfSEXP, SEXP omega_deconfSEXP,
  SEXP no_trees_treatSEXP, SEXP power_treatSEXP, SEXP base_treatSEXP,
  SEXP p_grow_treatSEXP, SEXP p_prune_treatSEXP, SEXP omega_treatSEXP,
  SEXP no_trees_controlSEXP, SEXP power_controlSEXP, SEXP base_controlSEXP,
  SEXP p_grow_controlSEXP, SEXP p_prune_controlSEXP, SEXP omega_controlSEXP,
  SEXP sigma_knownSEXP, SEXP sigmaSEXP, SEXP lambdaSEXP, SEXP nuSEXP,
  SEXP N_postSEXP, SEXP N_burnSEXP,
  SEXP store_posterior_sampleSEXP,
  SEXP verboseSEXP
) {

  // ---- Argument conversion ----

  size_t n        = Rcpp::as<size_t>(nSEXP);
  size_t n_deconf = Rcpp::as<size_t>(n_deconfSEXP);
  size_t p_treat  = Rcpp::as<size_t>(p_treatSEXP);
  size_t p_control = Rcpp::as<size_t>(p_controlSEXP);
  size_t p_deconf  = Rcpp::as<size_t>(p_deconfSEXP);

  Rcpp::NumericVector X_train_treat_vector(X_train_treatSEXP);
  double* X_train_treat = &X_train_treat_vector[0];
  Rcpp::NumericVector X_train_control_vector(X_train_controlSEXP);
  double* X_train_control = &X_train_control_vector[0];
  Rcpp::NumericVector X_train_deconf_vector(X_train_deconfSEXP);
  double* X_train_deconf = &X_train_deconf_vector[0];
  Rcpp::NumericVector y_vector(ySEXP);
  double* y = &y_vector[0];

  bool is_survival = Rcpp::as<bool>(is_survivalSEXP);
  Rcpp::IntegerVector treatment_indicator_vector(treatment_indicatorSEXP);
  int* treatment_indicator = &treatment_indicator_vector[0];
  Rcpp::NumericVector status_indicator_vector(status_indicatorSEXP);
  double* status_indicator = &status_indicator_vector[0];
  std::vector<double> y_observed_vector(y_vector.begin(), y_vector.end());
  double* y_observed = y_observed_vector.data();
  Rcpp::IntegerVector source_indicator_vector(source_indicatorSEXP);
  int* source_indicator = &source_indicator_vector[0];

  size_t n_test = Rcpp::as<size_t>(n_testSEXP);
  Rcpp::NumericVector X_test_treat_vector(X_test_treatSEXP);
  double* X_test_treat = &X_test_treat_vector[0];
  Rcpp::NumericVector X_test_control_vector(X_test_controlSEXP);
  double* X_test_control = &X_test_control_vector[0];
  Rcpp::NumericVector X_test_deconf_vector(X_test_deconfSEXP);
  double* X_test_deconf = &X_test_deconf_vector[0];
  Rcpp::IntegerVector treatment_indicator_test_vector(treatment_indicator_testSEXP);
  int* treatment_indicator_test = &treatment_indicator_test_vector[0];
  Rcpp::IntegerVector source_indicator_test_vector(source_indicator_testSEXP);
  int* source_indicator_test = &source_indicator_test_vector[0];

  // Hyperparameters — treatment effect forest
  size_t no_trees_treat = Rcpp::as<size_t>(no_trees_treatSEXP);
  double power_treat    = Rcpp::as<double>(power_treatSEXP);
  double base_treat     = Rcpp::as<double>(base_treatSEXP);
  double p_grow_treat   = Rcpp::as<double>(p_grow_treatSEXP);
  double p_prune_treat  = Rcpp::as<double>(p_prune_treatSEXP);
  double omega_treat    = Rcpp::as<double>(omega_treatSEXP);

  // Hyperparameters — prognostic (control) forest
  size_t no_trees_control = Rcpp::as<size_t>(no_trees_controlSEXP);
  double power_control    = Rcpp::as<double>(power_controlSEXP);
  double base_control     = Rcpp::as<double>(base_controlSEXP);
  double p_grow_control   = Rcpp::as<double>(p_grow_controlSEXP);
  double p_prune_control  = Rcpp::as<double>(p_prune_controlSEXP);
  double omega_control    = Rcpp::as<double>(omega_controlSEXP);

  // Hyperparameters — deconfounding forest
  size_t no_trees_deconf = Rcpp::as<size_t>(no_trees_deconfSEXP);
  double power_deconf    = Rcpp::as<double>(power_deconfSEXP);
  double base_deconf     = Rcpp::as<double>(base_deconfSEXP);
  double p_grow_deconf   = Rcpp::as<double>(p_grow_deconfSEXP);
  double p_prune_deconf  = Rcpp::as<double>(p_prune_deconfSEXP);
  double omega_deconf    = Rcpp::as<double>(omega_deconfSEXP);

  // Error variance hyperparameters
  bool   sigma_known = Rcpp::as<bool>(sigma_knownSEXP);
  double sigma       = Rcpp::as<double>(sigmaSEXP);
  double lambda      = Rcpp::as<double>(lambdaSEXP);
  double nu          = Rcpp::as<double>(nuSEXP);

  // Commensurate shift fixed at zero (no inter-source mean adjustment)
  const double eta = 0.0;

  // MCMC dimensions
  size_t N_post = Rcpp::as<size_t>(N_postSEXP);
  size_t N_burn = Rcpp::as<size_t>(N_burnSEXP);

  // Storage flags
  bool store_posterior_sample = Rcpp::as<bool>(store_posterior_sampleSEXP);

  // Random number generation — uses R's RNG state (set via set.seed() in R)
  RandomGenerator random;

  bool verbose = Rcpp::as<bool>(verboseSEXP);


  // ---- Storage containers ----

  Rcpp::NumericVector train_predictions_mean(n,      0.0);
  Rcpp::NumericVector test_predictions_mean(n_test,  0.0);
  Rcpp::NumericVector train_predictions_mean_control(n,      0.0);
  Rcpp::NumericVector test_predictions_mean_control(n_test,  0.0);
  Rcpp::NumericVector train_predictions_mean_treat(n,      0.0);
  Rcpp::NumericVector test_predictions_mean_treat(n_test,  0.0);
  Rcpp::NumericVector train_predictions_mean_deconf(n_deconf, 0.0);
  Rcpp::NumericVector test_predictions_mean_deconf(n_test,   0.0);

  Rcpp::NumericMatrix train_predictions_sample_control;
  Rcpp::NumericMatrix test_predictions_sample_control;
  Rcpp::NumericMatrix train_predictions_sample_treat;
  Rcpp::NumericMatrix test_predictions_sample_treat;
  Rcpp::NumericMatrix train_predictions_sample_deconf;
  Rcpp::NumericMatrix test_predictions_sample_deconf;
  if (store_posterior_sample) {
    train_predictions_sample_control = Rcpp::NumericMatrix(N_post, n);
    test_predictions_sample_control  = Rcpp::NumericMatrix(N_post, n_test);
    train_predictions_sample_treat   = Rcpp::NumericMatrix(N_post, n);
    test_predictions_sample_treat    = Rcpp::NumericMatrix(N_post, n_test);
    train_predictions_sample_deconf  = Rcpp::NumericMatrix(N_post, n_deconf);
    test_predictions_sample_deconf   = Rcpp::NumericMatrix(N_post, n_test);
  }

  Rcpp::NumericVector store_sigma = sigma_known
    ? Rcpp::NumericVector::create(sigma)
    : Rcpp::NumericVector(N_post + N_burn);

  // Acceptance ratio trackers
  bool* accepted_control = new bool[no_trees_control]();
  bool* accepted_treat   = new bool[no_trees_treat]();
  bool* accepted_deconf  = new bool[no_trees_deconf]();
  double sum_accept_control = 0, sum_accept_treat = 0, sum_accept_deconf = 0;
  double acceptance_ratio_control, acceptance_ratio_treat, acceptance_ratio_deconf;

  // Working arrays
  double* testpred_treat   = n_test ? new double[n_test] : nullptr;
  double* testpred_control = n_test ? new double[n_test] : nullptr;
  double* testpred_deconf  = n_test ? new double[n_test] : nullptr;
  double* total_predictions       = new double[n];
  double* augmented_outcome_treat   = new double[n];
  double* augmented_outcome_control = new double[n];
  double* augmented_outcome_deconf  = new double[n_deconf];

  for (size_t i = 0; i < n; ++i) {
    augmented_outcome_treat[i]   = y[i] / 2.0;
    augmented_outcome_control[i] = y[i] / 2.0;
  }
  for (size_t i = 0; i < n_deconf; ++i) augmented_outcome_deconf[i] = 0.0;


  // ---- Set up forests ----

  // Prognostic (control) forest
  ForestEngine forest_control(no_trees_control);
  forest_control.SetTreePrior(base_control, power_control, omega_control,
                              p_grow_control, p_prune_control,
                              0.5, 1.0, static_cast<double>(p_control),
                              true, false, 1.0);
  forest_control.SetUpForest(p_control, n, X_train_control, augmented_outcome_control, nullptr, omega_control);

  // Treatment effect forest
  ForestEngine forest_treat(no_trees_treat);
  forest_treat.SetTreePrior(base_treat, power_treat, omega_treat,
                            p_grow_treat, p_prune_treat,
                            0.5, 1.0, static_cast<double>(p_treat),
                            true, false, 1.0);
  forest_treat.SetUpForest(p_treat, n, X_train_treat, augmented_outcome_treat, nullptr, omega_treat);

  // Deconfounding forest
  ForestEngine forest_deconf(no_trees_deconf);
  forest_deconf.SetTreePrior(base_deconf, power_deconf, omega_deconf,
                             p_grow_deconf, p_prune_deconf,
                             0.5, 1.0, static_cast<double>(p_deconf),
                             true, false, 1.0);
  forest_deconf.SetUpForest(p_deconf, n_deconf, X_train_deconf, augmented_outcome_deconf, nullptr, omega_deconf);


  // ---- Timing ----

  time_t time_stamp;
  int time_start = time(&time_stamp);
  int barWidth = 70;
  if (verbose) Rcpp::Rcout << "\nProgress of the MCMC sampler:\n\n";


  // ---- MCMC loop ----

  for (size_t i = 0; i < N_post + N_burn; ++i) {

    // Progress bar
    if (verbose) {
      float progress = static_cast<float>(i) / static_cast<float>(N_post + N_burn);
      int pos = static_cast<int>(barWidth * progress);
      Rcpp::Rcout << "|";
      for (int j = 0; j < barWidth; ++j)
        Rcpp::Rcout << (j < pos ? "=" : (j == pos ? ">" : " "));
      Rcpp::Rcout << "| " << static_cast<int>(progress * 100.0) << " %\r";
      Rcpp::Rcout.flush();
    }

    // -- Update prognostic forest --
    forest_control.UpdateForest(sigma, accepted_control, random);

    // After updating control: refresh treat and deconf augmented outcomes
    {
      size_t j = 0;
      for (size_t k = 0; k < n; ++k) {
        const double b = (treatment_indicator[k] == 1) ? 0.5 : -0.5;
        const double s = (source_indicator[k]   == 1) ? 1.0 :  0.0;

        double c_k = 0.0;
        if (source_indicator[k] == 0) {
          c_k = forest_deconf.GetPrediction(j);
          augmented_outcome_deconf[j] = (y[k] - forest_control.GetPrediction(k)
                                              - eta - b * forest_treat.GetPrediction(k)) / b;
          ++j;
        }
        augmented_outcome_treat[k] = (y[k] - (1.0 - s) * eta
                                           - forest_control.GetPrediction(k)
                                           - b * c_k) / b;
      }
    }

    // -- Update treatment effect forest --
    forest_treat.UpdateForest(sigma, accepted_treat, random);

    // After updating treat: refresh deconf and control augmented outcomes
    {
      size_t j = 0;
      for (size_t k = 0; k < n; ++k) {
        const double b = (treatment_indicator[k] == 1) ? 0.5 : -0.5;
        const double s = (source_indicator[k]   == 1) ? 1.0 :  0.0;

        double c_k = 0.0;
        if (source_indicator[k] == 0) {
          augmented_outcome_deconf[j] = (y[k] - forest_control.GetPrediction(k)
                                              - eta - b * forest_treat.GetPrediction(k)) / b;
          c_k = forest_deconf.GetPrediction(j);
          ++j;
        }
        augmented_outcome_control[k] = y[k] - (1.0 - s) * eta
                                            - b * forest_treat.GetPrediction(k)
                                            - b * c_k;
      }
    }

    // -- Update deconfounding forest --
    forest_deconf.UpdateForest(sigma, accepted_deconf, random);

    // After updating deconf: refresh treat and control augmented outcomes
    {
      size_t j = 0;
      for (size_t k = 0; k < n; ++k) {
        const double b = (treatment_indicator[k] == 1) ? 0.5 : -0.5;
        const double s = (source_indicator[k]   == 1) ? 1.0 :  0.0;

        double c_k = 0.0;
        if (source_indicator[k] == 0) {
          c_k = forest_deconf.GetPrediction(j);
          ++j;
        }
        augmented_outcome_treat[k]   = (y[k] - (1.0 - s) * eta
                                             - forest_control.GetPrediction(k)
                                             - b * c_k) / b;
        augmented_outcome_control[k] =  y[k] - (1.0 - s) * eta
                                             - b * forest_treat.GetPrediction(k)
                                             - b * c_k;
      }
    }

    // -- Compute total predictions --
    {
      size_t j = 0;
      for (size_t k = 0; k < n; ++k) {
        const double b = (treatment_indicator[k] == 1) ? 0.5 : -0.5;
        double c_k = 0.0, s = 1.0;
        if (source_indicator[k] == 0) {
          s   = 0.0;
          c_k = forest_deconf.GetPrediction(j);
          ++j;
        }
        total_predictions[k] = forest_control.GetPredictions()[k]
                              + (1.0 - s) * eta
                              + b * (forest_treat.GetPredictions()[k] + c_k);
      }
    }

    // -- Update sigma --
    UpdateSigma(sigma_known, sigma, store_sigma, i, y, n, total_predictions, nu, lambda, random);

    // -- Augment censored observations --
    AugmentCensoredObservations(is_survival, y, y_observed, status_indicator,
                                total_predictions, sigma, n, random);


    // -- Post-burn-in storage --
    if (i >= N_burn) {

      size_t j_os = 0;
      for (size_t k = 0; k < n; ++k) {
        const double b = (treatment_indicator[k] == 1) ? 0.5 : -0.5;
        const double m_k   = forest_control.GetPrediction(k);
        const double tau_k = forest_treat.GetPrediction(k);
        double c_k = 0.0, s = 1.0;
        if (source_indicator[k] == 0) {
          s   = 0.0;
          c_k = forest_deconf.GetPrediction(j_os);
          train_predictions_mean_deconf[j_os] += c_k;
          ++j_os;
        }
        train_predictions_mean[k]         += m_k + (1.0 - s) * eta + b * (tau_k + c_k);
        train_predictions_mean_control[k] += m_k;
        train_predictions_mean_treat[k]   += tau_k;
      }

      if (store_posterior_sample) {
        for (size_t k = 0; k < n; ++k) {
          train_predictions_sample_control(i - N_burn, k) = forest_control.GetPrediction(k);
          train_predictions_sample_treat(i - N_burn, k)   = forest_treat.GetPrediction(k);
        }
        size_t j_os2 = 0;
        for (size_t k = 0; k < n; ++k) {
          if (source_indicator[k] == 0) {
            train_predictions_sample_deconf(i - N_burn, j_os2) = forest_deconf.GetPrediction(j_os2);
            ++j_os2;
          }
        }
      }

      if (n_test > 0) {
        forest_control.Predict(p_control, n_test, X_test_control, testpred_control);
        forest_treat.Predict(p_treat,   n_test, X_test_treat,   testpred_treat);
        forest_deconf.Predict(p_deconf,  n_test, X_test_deconf,  testpred_deconf);

        if (store_posterior_sample) {
          for (size_t k = 0; k < n_test; ++k) {
            test_predictions_sample_control(i - N_burn, k) = testpred_control[k];
            test_predictions_sample_treat(i - N_burn, k)   = testpred_treat[k];
            test_predictions_sample_deconf(i - N_burn, k)  = testpred_deconf[k];
          }
        }

        for (size_t k = 0; k < n_test; ++k) {
          const double b = (treatment_indicator_test[k] == 1) ? 0.5 : -0.5;
          const double s = (source_indicator_test[k]    == 1) ? 1.0 : 0.0;
          const double c_k = testpred_deconf[k];
          test_predictions_mean[k]         += testpred_control[k] + (1.0 - s) * eta + b * (testpred_treat[k] + (1.0 - s) * c_k);
          test_predictions_mean_control[k] += testpred_control[k];
          test_predictions_mean_treat[k]   += testpred_treat[k];
          test_predictions_mean_deconf[k]  += c_k;
        }
      }

      // Acceptance ratios
      for (size_t j = 0; j < no_trees_control; ++j) sum_accept_control += accepted_control[j];
      for (size_t j = 0; j < no_trees_treat;   ++j) sum_accept_treat   += accepted_treat[j];
      for (size_t j = 0; j < no_trees_deconf;  ++j) sum_accept_deconf  += accepted_deconf[j];

    }

  } // end MCMC loop


  // ---- Post-loop ----

  acceptance_ratio_control = sum_accept_control / (N_post * no_trees_control);
  acceptance_ratio_treat   = sum_accept_treat   / (N_post * no_trees_treat);
  acceptance_ratio_deconf  = sum_accept_deconf  / (N_post * no_trees_deconf);

  int time_end = time(&time_stamp);

  if (verbose) {
    Rcpp::Rcout << "|";
    for (int j = 0; j < barWidth; ++j) Rcpp::Rcout << "=";
    Rcpp::Rcout << "| 100 %\r";
    Rcpp::Rcout.flush();
    Rcpp::Rcout << "\n" << std::endl;
    Rcpp::Rcout << "Mean acceptance ratio (prognostic model):    " << acceptance_ratio_control << std::endl;
    Rcpp::Rcout << "Mean acceptance ratio (treatment effect model): " << acceptance_ratio_treat << std::endl;
    Rcpp::Rcout << "Mean acceptance ratio (deconfounding model): " << acceptance_ratio_deconf  << std::endl;
    Rcpp::Rcout << "\nDone in " << (time_end - time_start) << " seconds.\n" << std::endl;
  }

  // Rescale posterior means
  for (size_t k = 0; k < n;       ++k) {
    train_predictions_mean[k]         /= N_post;
    train_predictions_mean_control[k] /= N_post;
    train_predictions_mean_treat[k]   /= N_post;
  }
  for (size_t k = 0; k < n_deconf; ++k) train_predictions_mean_deconf[k] /= N_post;
  for (size_t k = 0; k < n_test;   ++k) {
    test_predictions_mean[k]         /= N_post;
    test_predictions_mean_control[k] /= N_post;
    test_predictions_mean_treat[k]   /= N_post;
    test_predictions_mean_deconf[k]  /= N_post;
  }

  // ---- Build result list ----

  Rcpp::List results;
  results["sigma"]                   = store_sigma;
  results["train_predictions"]       = train_predictions_mean;
  results["test_predictions"]        = test_predictions_mean;
  results["train_predictions_control"] = train_predictions_mean_control;
  results["test_predictions_control"]  = test_predictions_mean_control;
  results["train_predictions_treat"]   = train_predictions_mean_treat;
  results["test_predictions_treat"]    = test_predictions_mean_treat;
  results["train_predictions_deconf"]  = train_predictions_mean_deconf;
  results["test_predictions_deconf"]   = test_predictions_mean_deconf;
  results["acceptance_ratio_control"]  = acceptance_ratio_control;
  results["acceptance_ratio_treat"]    = acceptance_ratio_treat;
  results["acceptance_ratio_deconf"]   = acceptance_ratio_deconf;
  if (store_posterior_sample) {
    results["train_predictions_sample_control"] = train_predictions_sample_control;
    results["test_predictions_sample_control"]  = test_predictions_sample_control;
    results["train_predictions_sample_treat"]   = train_predictions_sample_treat;
    results["test_predictions_sample_treat"]    = test_predictions_sample_treat;
    results["train_predictions_sample_deconf"]  = train_predictions_sample_deconf;
    results["test_predictions_sample_deconf"]   = test_predictions_sample_deconf;
  }

  // ---- Clean up ----

  delete[] testpred_control;
  delete[] testpred_treat;
  delete[] testpred_deconf;
  delete[] accepted_control;
  delete[] accepted_treat;
  delete[] accepted_deconf;
  delete[] total_predictions;
  delete[] augmented_outcome_control;
  delete[] augmented_outcome_treat;
  delete[] augmented_outcome_deconf;

  return results;
}
