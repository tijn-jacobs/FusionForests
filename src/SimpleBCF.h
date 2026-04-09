#ifndef SIMPLEBCF_H
#define SIMPLEBCF_H

#include "ForestEngine.h"
#include "OuterGibbsFunctions.h"

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
);

#endif // SIMPLEBCF_H
