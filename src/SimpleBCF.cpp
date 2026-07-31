#include "SimpleBCF.h"

// [[Rcpp::export]]
Rcpp::List SimpleBCF_cpp(
  SEXP nSEXP, SEXP p_progSEXP, SEXP p_treatSEXP,
  SEXP X_train_progSEXP, SEXP X_train_treatSEXP,
  SEXP ySEXP, SEXP treatment_indicatorSEXP,
  SEXP n_testSEXP,
  SEXP X_test_progSEXP, SEXP X_test_treatSEXP,
  SEXP treatment_indicator_testSEXP,
  SEXP no_trees_progSEXP, SEXP no_trees_treatSEXP,
  SEXP powerSEXP, SEXP baseSEXP,
  SEXP p_growSEXP, SEXP p_pruneSEXP,
  SEXP omega_progSEXP, SEXP omega_treatSEXP,
  SEXP sigma_knownSEXP, SEXP sigmaSEXP,
  SEXP lambdaSEXP, SEXP nuSEXP,
  SEXP N_postSEXP, SEXP N_burnSEXP,
  SEXP verboseSEXP, SEXP irsSEXP,
  SEXP store_posterior_sampleSEXP
) {

  // ---- Argument conversion ----

  size_t n       = Rcpp::as<size_t>(nSEXP);
  size_t p_prog  = Rcpp::as<size_t>(p_progSEXP);
  size_t p_treat = Rcpp::as<size_t>(p_treatSEXP);

  Rcpp::NumericVector X_prog_vec(X_train_progSEXP);
  double* X_train_prog = &X_prog_vec[0];
  Rcpp::NumericVector X_treat_vec(X_train_treatSEXP);
  double* X_train_treat = &X_treat_vec[0];

  Rcpp::NumericVector y_vector(ySEXP);
  double* y = &y_vector[0];

  Rcpp::IntegerVector A_vector(treatment_indicatorSEXP);
  int* A = &A_vector[0];

  size_t n_test = Rcpp::as<size_t>(n_testSEXP);
  Rcpp::NumericVector X_test_prog_vec(X_test_progSEXP);
  double* X_test_prog = &X_test_prog_vec[0];
  Rcpp::NumericVector X_test_treat_vec(X_test_treatSEXP);
  double* X_test_treat = &X_test_treat_vec[0];

  Rcpp::IntegerVector A_test_vector(treatment_indicator_testSEXP);
  int* A_test = &A_test_vector[0];

  size_t no_trees_prog  = Rcpp::as<size_t>(no_trees_progSEXP);
  size_t no_trees_treat = Rcpp::as<size_t>(no_trees_treatSEXP);
  double power   = Rcpp::as<double>(powerSEXP);
  double base    = Rcpp::as<double>(baseSEXP);
  double p_grow  = Rcpp::as<double>(p_growSEXP);
  double p_prune = Rcpp::as<double>(p_pruneSEXP);
  double omega_prog  = Rcpp::as<double>(omega_progSEXP);
  double omega_treat = Rcpp::as<double>(omega_treatSEXP);

  bool   sigma_known = Rcpp::as<bool>(sigma_knownSEXP);
  double sigma       = Rcpp::as<double>(sigmaSEXP);
  double lambda      = Rcpp::as<double>(lambdaSEXP);
  double nu          = Rcpp::as<double>(nuSEXP);

  size_t N_post = Rcpp::as<size_t>(N_postSEXP);
  size_t N_burn = Rcpp::as<size_t>(N_burnSEXP);
  bool verbose  = Rcpp::as<bool>(verboseSEXP);
  int irs       = Rcpp::as<int>(irsSEXP);
  bool store_posterior_sample =
    Rcpp::as<bool>(store_posterior_sampleSEXP);

  RandomGenerator random;


  // ---- Treatment coding: b = +0.5 (treated), -0.5 (control) ----

  std::vector<double> b(n);
  for (size_t k = 0; k < n; ++k)
    b[k] = (A[k] == 1) ? 0.5 : -0.5;

  std::vector<double> b_test(n_test);
  for (size_t k = 0; k < n_test; ++k)
    b_test[k] = (A_test[k] == 1) ? 0.5 : -0.5;


  // ---- Augmented outcomes (initialise to y/2 each) ----

  double* aug_prog  = new double[n];
  double* aug_treat = new double[n];
  for (size_t k = 0; k < n; ++k) {
    aug_prog[k]  = y[k] / 2.0;
    aug_treat[k] = y[k] / 2.0;
  }


  // ---- Storage ----

  Rcpp::NumericVector train_pred_mean(n, 0.0);
  Rcpp::NumericVector test_pred_mean(n_test, 0.0);
  Rcpp::NumericVector train_pred_prog_mean(n, 0.0);
  Rcpp::NumericVector test_pred_prog_mean(n_test, 0.0);
  Rcpp::NumericVector train_pred_treat_mean(n, 0.0);
  Rcpp::NumericVector test_pred_treat_mean(n_test, 0.0);

  Rcpp::NumericVector store_sigma = sigma_known
    ? Rcpp::NumericVector::create(sigma)
    : Rcpp::NumericVector(N_post + N_burn);

  bool* accepted_prog  = new bool[no_trees_prog]();
  bool* accepted_treat = new bool[no_trees_treat]();
  double sum_accept_prog  = 0.0;
  double sum_accept_treat = 0.0;

  Rcpp::NumericMatrix test_pred_treat_sample;
  Rcpp::NumericMatrix test_pred_total_sample;
  if (store_posterior_sample) {
    test_pred_treat_sample =
      Rcpp::NumericMatrix(N_post, n_test);
    test_pred_total_sample =
      Rcpp::NumericMatrix(N_post, n_test);
  }

  double* testpred_prog  = n_test ?
    new double[n_test] : nullptr;
  double* testpred_treat = n_test ?
    new double[n_test] : nullptr;


  // ---- Set up forests ----

  ForestEngine forest_prog(no_trees_prog);
  forest_prog.SetTreePrior(
    base, power, omega_prog, p_grow, p_prune,
    0.5, 1.0, static_cast<double>(p_prog),
    true, false, 1.0);
  forest_prog.SetUpForest(
    p_prog, n, X_train_prog, aug_prog,
<<<<<<< HEAD
    static_cast<size_t>(100), omega_prog);
  if (irs > 0) forest_prog.SetIRS(irs);
=======
    nullptr, omega_prog);
  (void) irs;  // IRS modes not currently wired into ForestEngine; ignored.
>>>>>>> borrow

  ForestEngine forest_treat(no_trees_treat);
  forest_treat.SetTreePrior(
    base, power, omega_treat, p_grow, p_prune,
    0.5, 1.0, static_cast<double>(p_treat),
    true, false, 1.0);
  forest_treat.SetUpForest(
    p_treat, n, X_train_treat, aug_treat,
<<<<<<< HEAD
    static_cast<size_t>(100), omega_treat);
  if (irs > 0) forest_treat.SetIRS(irs);
=======
    nullptr, omega_treat);
>>>>>>> borrow


  // ---- Timing ----

  time_t time_stamp;
  int time_start = time(&time_stamp);
  int barWidth = 70;
  if (verbose)
    Rcpp::Rcout << "\nSimpleBCF — Progress of the "
                << "MCMC sampler:\n\n";


  // ---- MCMC loop ----

  for (size_t i = 0; i < N_post + N_burn; ++i) {

    // Progress bar
    if (verbose) {
      float progress = static_cast<float>(i) /
        static_cast<float>(N_post + N_burn);
      int pos = static_cast<int>(barWidth * progress);
      Rcpp::Rcout << "|";
      for (int j = 0; j < barWidth; ++j)
        Rcpp::Rcout << (j < pos ? "=" :
                        (j == pos ? ">" : " "));
      Rcpp::Rcout << "| "
                  << static_cast<int>(progress * 100.0)
                  << " %\r";
      Rcpp::Rcout.flush();
    }

    // -- Update prognostic forest --
    // aug_prog[k] = y[k] - b[k] * tau_hat[k]
    for (size_t k = 0; k < n; ++k)
      aug_prog[k] = y[k] -
        b[k] * forest_treat.GetPrediction(k);

    forest_prog.UpdateForest(sigma, accepted_prog, random);

    // -- Update treatment forest --
    // aug_treat[k] = (y[k] - mu_hat[k]) / b[k]
    for (size_t k = 0; k < n; ++k)
      aug_treat[k] = (y[k] -
        forest_prog.GetPrediction(k)) / b[k];

    forest_treat.UpdateForest(sigma, accepted_treat,
                              random);

    // -- Update sigma --
    // total[k] = mu_hat[k] + b[k] * tau_hat[k]
    // Reuse aug_prog as temporary total prediction
    double* total = aug_prog;
    for (size_t k = 0; k < n; ++k)
      total[k] = forest_prog.GetPrediction(k) +
        b[k] * forest_treat.GetPrediction(k);

    UpdateSigma(sigma_known, sigma, store_sigma, i,
                y, n, total, nu, lambda, random);


    // -- Post-burn-in storage --
    if (i >= N_burn) {

      for (size_t k = 0; k < n; ++k) {
        double mu_k  = forest_prog.GetPrediction(k);
        double tau_k = forest_treat.GetPrediction(k);
        train_pred_prog_mean[k]  += mu_k;
        train_pred_treat_mean[k] += tau_k;
        train_pred_mean[k] += mu_k + b[k] * tau_k;
      }

      if (n_test > 0) {
<<<<<<< HEAD
        if (irs > 0) {
          forest_prog.Predict(
            p_prog, n_test, X_test_prog,
            testpred_prog, random);
          forest_treat.Predict(
            p_treat, n_test, X_test_treat,
            testpred_treat, random);
        } else {
          forest_prog.Predict(
            p_prog, n_test, X_test_prog,
            testpred_prog);
          forest_treat.Predict(
            p_treat, n_test, X_test_treat,
            testpred_treat);
        }
=======
        forest_prog.Predict(
          p_prog, n_test, X_test_prog,
          testpred_prog);
        forest_treat.Predict(
          p_treat, n_test, X_test_treat,
          testpred_treat);
>>>>>>> borrow
        for (size_t k = 0; k < n_test; ++k) {
          test_pred_prog_mean[k]  += testpred_prog[k];
          test_pred_treat_mean[k] += testpred_treat[k];
          test_pred_mean[k] += testpred_prog[k] +
            b_test[k] * testpred_treat[k];
        }

        if (store_posterior_sample) {
          for (size_t k = 0; k < n_test; ++k) {
            test_pred_treat_sample(i - N_burn, k) =
              testpred_treat[k];
            test_pred_total_sample(i - N_burn, k) =
              testpred_prog[k] +
              b_test[k] * testpred_treat[k];
          }
        }
      }

      for (size_t j = 0; j < no_trees_prog; ++j)
        sum_accept_prog += accepted_prog[j];
      for (size_t j = 0; j < no_trees_treat; ++j)
        sum_accept_treat += accepted_treat[j];
    }

  } // end MCMC loop


  // ---- Post-loop ----

  double ar_prog  = sum_accept_prog /
    (N_post * no_trees_prog);
  double ar_treat = sum_accept_treat /
    (N_post * no_trees_treat);

  int time_end = time(&time_stamp);

  if (verbose) {
    Rcpp::Rcout << "|";
    for (int j = 0; j < barWidth; ++j)
      Rcpp::Rcout << "=";
    Rcpp::Rcout << "| 100 %\r";
    Rcpp::Rcout.flush();
    Rcpp::Rcout << "\n" << std::endl;
    Rcpp::Rcout << "Acceptance ratio (prog):  "
                << ar_prog << std::endl;
    Rcpp::Rcout << "Acceptance ratio (treat): "
                << ar_treat << std::endl;
    Rcpp::Rcout << "Done in "
                << (time_end - time_start)
                << " seconds.\n" << std::endl;
  }

  for (size_t k = 0; k < n; ++k) {
    train_pred_mean[k]       /= N_post;
    train_pred_prog_mean[k]  /= N_post;
    train_pred_treat_mean[k] /= N_post;
  }
  for (size_t k = 0; k < n_test; ++k) {
    test_pred_mean[k]       /= N_post;
    test_pred_prog_mean[k]  /= N_post;
    test_pred_treat_mean[k] /= N_post;
  }


  // ---- Build result ----

  Rcpp::List results;
  results["train_predictions"]       = train_pred_mean;
  results["test_predictions"]        = test_pred_mean;
  results["train_predictions_prog"]  = train_pred_prog_mean;
  results["test_predictions_prog"]   = test_pred_prog_mean;
  results["train_predictions_treat"] = train_pred_treat_mean;
  results["test_predictions_treat"]  = test_pred_treat_mean;
  results["sigma"]                   = store_sigma;
  results["acceptance_ratio_prog"]   = ar_prog;
  results["acceptance_ratio_treat"]  = ar_treat;
  if (store_posterior_sample) {
    results["test_predictions_treat_sample"] =
      test_pred_treat_sample;
    results["test_predictions_sample"] =
      test_pred_total_sample;
  }


  // ---- Clean up ----

  delete[] aug_prog;
  delete[] aug_treat;
  delete[] accepted_prog;
  delete[] accepted_treat;
  delete[] testpred_prog;
  delete[] testpred_treat;

  return results;
}
