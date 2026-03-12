#include "SimpleBART.h"

// [[Rcpp::export]]
Rcpp::List SimpleBART_cpp(
  SEXP nSEXP, SEXP pSEXP, SEXP X_trainSEXP, SEXP ySEXP,
  SEXP n_testSEXP, SEXP X_testSEXP,
  SEXP no_treesSEXP, SEXP powerSEXP, SEXP baseSEXP,
  SEXP p_growSEXP, SEXP p_pruneSEXP, SEXP omegaSEXP,
  SEXP sigma_knownSEXP, SEXP sigmaSEXP, SEXP lambdaSEXP, SEXP nuSEXP,
  SEXP N_postSEXP, SEXP N_burnSEXP,
  SEXP verboseSEXP,
  SEXP irsSEXP
) {

  // ---- Argument conversion ----

  size_t n = Rcpp::as<size_t>(nSEXP);
  size_t p = Rcpp::as<size_t>(pSEXP);

  Rcpp::NumericVector X_train_vector(X_trainSEXP);
  double* X_train = &X_train_vector[0];
  Rcpp::NumericVector y_vector(ySEXP);
  double* y = &y_vector[0];

  size_t n_test = Rcpp::as<size_t>(n_testSEXP);
  Rcpp::NumericVector X_test_vector(X_testSEXP);
  double* X_test = &X_test_vector[0];

  size_t no_trees = Rcpp::as<size_t>(no_treesSEXP);
  double power    = Rcpp::as<double>(powerSEXP);
  double base     = Rcpp::as<double>(baseSEXP);
  double p_grow   = Rcpp::as<double>(p_growSEXP);
  double p_prune  = Rcpp::as<double>(p_pruneSEXP);
  double omega    = Rcpp::as<double>(omegaSEXP);

  bool   sigma_known = Rcpp::as<bool>(sigma_knownSEXP);
  double sigma       = Rcpp::as<double>(sigmaSEXP);
  double lambda      = Rcpp::as<double>(lambdaSEXP);
  double nu          = Rcpp::as<double>(nuSEXP);

  size_t N_post = Rcpp::as<size_t>(N_postSEXP);
  size_t N_burn = Rcpp::as<size_t>(N_burnSEXP);
  bool verbose  = Rcpp::as<bool>(verboseSEXP);
  int irs       = Rcpp::as<int>(irsSEXP);

  RandomGenerator random;


  // ---- Storage ----

  Rcpp::NumericVector train_predictions_mean(n,      0.0);
  Rcpp::NumericVector test_predictions_mean(n_test,  0.0);

  Rcpp::NumericVector store_sigma = sigma_known
    ? Rcpp::NumericVector::create(sigma)
    : Rcpp::NumericVector(N_post + N_burn);

  bool* accepted = new bool[no_trees]();
  double sum_accept = 0.0;

  double* testpred = n_test ? new double[n_test] : nullptr;


  // ---- Set up forest ----

  ForestEngine forest(no_trees);
  forest.SetTreePrior(base, power, omega,
                      p_grow, p_prune,
                      0.5, 1.0, static_cast<double>(p),
                      true, false, 1.0);
  forest.SetUpForest(p, n, X_train, y, static_cast<size_t>(100), omega);
  if (irs > 0) forest.SetIRS(irs);


  // ---- Timing ----

  time_t time_stamp;
  int time_start = time(&time_stamp);
  int barWidth = 70;
  if (verbose) Rcpp::Rcout << "\nSimpleBART — Progress of the MCMC sampler:\n\n";


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

    // Update the forest
    forest.UpdateForest(sigma, accepted, random);

    // Update sigma
    UpdateSigma(sigma_known, sigma, store_sigma, i, y, n,
                forest.GetPredictions(), nu, lambda, random);

    // Post-burn-in storage
    if (i >= N_burn) {

      for (size_t k = 0; k < n; ++k)
        train_predictions_mean[k] += forest.GetPrediction(k);

      if (n_test > 0) {
        if (irs > 0) {
          forest.Predict(p, n_test, X_test, testpred, random);
        } else {
          forest.Predict(p, n_test, X_test, testpred);
        }
        for (size_t k = 0; k < n_test; ++k)
          test_predictions_mean[k] += testpred[k];
      }

      for (size_t j = 0; j < no_trees; ++j)
        sum_accept += accepted[j];
    }

  } // end MCMC loop


  // ---- Post-loop ----

  double acceptance_ratio = sum_accept / (N_post * no_trees);

  int time_end = time(&time_stamp);

  if (verbose) {
    Rcpp::Rcout << "|";
    for (int j = 0; j < barWidth; ++j) Rcpp::Rcout << "=";
    Rcpp::Rcout << "| 100 %\r";
    Rcpp::Rcout.flush();
    Rcpp::Rcout << "\n" << std::endl;
    Rcpp::Rcout << "Mean acceptance ratio: " << acceptance_ratio << std::endl;
    Rcpp::Rcout << "Done in " << (time_end - time_start) << " seconds.\n" << std::endl;
  }

  for (size_t k = 0; k < n;      ++k) train_predictions_mean[k] /= N_post;
  for (size_t k = 0; k < n_test; ++k) test_predictions_mean[k]  /= N_post;


  // ---- Build result ----

  Rcpp::List results;
  results["train_predictions"] = train_predictions_mean;
  results["test_predictions"]  = test_predictions_mean;
  results["sigma"]             = store_sigma;
  results["acceptance_ratio"]  = acceptance_ratio;


  // ---- Clean up ----

  delete[] accepted;
  delete[] testpred;

  return results;
}
