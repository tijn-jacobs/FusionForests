#ifndef SIMPLEBART_H
#define SIMPLEBART_H

#include "ForestEngine.h"
#include "OuterGibbsFunctions.h"

Rcpp::List SimpleBART_cpp(
  SEXP nSEXP, SEXP pSEXP, SEXP X_trainSEXP, SEXP ySEXP,
  SEXP n_testSEXP, SEXP X_testSEXP,
  SEXP no_treesSEXP, SEXP powerSEXP, SEXP baseSEXP,
  SEXP p_growSEXP, SEXP p_pruneSEXP, SEXP omegaSEXP,
  SEXP sigma_knownSEXP, SEXP sigmaSEXP, SEXP lambdaSEXP, SEXP nuSEXP,
  SEXP N_postSEXP, SEXP N_burnSEXP,
  SEXP verboseSEXP,
  SEXP irsSEXP,
  SEXP store_posterior_sampleSEXP
);

#endif // SIMPLEBART_H
