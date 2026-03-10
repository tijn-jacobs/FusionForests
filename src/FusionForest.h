#ifndef FUSIONFOREST_H
#define FUSIONFOREST_H

#include "ForestEngine.h"
#include "CommensurateParameters.h"
#include "OuterGibbsFunctions.h"

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
  SEXP eta_commensurateSEXP,
  SEXP N_postSEXP, SEXP N_burnSEXP,
  SEXP store_posterior_sampleSEXP,
  SEXP verboseSEXP
);

#endif // FUSIONFOREST_H
