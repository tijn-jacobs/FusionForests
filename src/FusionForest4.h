#ifndef FUSIONFOREST4_H
#define FUSIONFOREST4_H

#include "ForestEngine.h"
#include "OuterGibbsFunctions.h"

Rcpp::List FusionForest4_cpp(
  SEXP nSEXP, SEXP p_treatSEXP, SEXP p_controlSEXP, SEXP X_train_treatSEXP,
  SEXP X_train_controlSEXP, SEXP ySEXP, SEXP status_indicatorSEXP, SEXP is_survivalSEXP,
  SEXP observed_left_timeSEXP, SEXP observed_right_timeSEXP,
  SEXP interval_censoring_indicatorSEXP,
  SEXP treatment_indicatorSEXP, SEXP source_indicatorSEXP,
  SEXP n_testSEXP, SEXP X_test_controlSEXP, SEXP X_test_treatSEXP, SEXP X_test_deconfSEXP,
  SEXP X_test_deviationSEXP,
  SEXP treatment_indicator_testSEXP, SEXP source_indicator_testSEXP,
  SEXP n_deconfSEXP, SEXP p_deconfSEXP, SEXP X_train_deconfSEXP,
  SEXP no_trees_deconfSEXP, SEXP power_deconfSEXP, SEXP base_deconfSEXP,
  SEXP p_grow_deconfSEXP, SEXP p_prune_deconfSEXP, SEXP omega_deconfSEXP,
  SEXP n_deviationSEXP, SEXP p_deviationSEXP, SEXP X_train_deviationSEXP,
  SEXP no_trees_deviationSEXP, SEXP power_deviationSEXP, SEXP base_deviationSEXP,
  SEXP p_grow_deviationSEXP, SEXP p_prune_deviationSEXP, SEXP omega_deviationSEXP,
  SEXP no_trees_treatSEXP, SEXP power_treatSEXP, SEXP base_treatSEXP,
  SEXP p_grow_treatSEXP, SEXP p_prune_treatSEXP, SEXP omega_treatSEXP,
  SEXP no_trees_controlSEXP, SEXP power_controlSEXP, SEXP base_controlSEXP,
  SEXP p_grow_controlSEXP, SEXP p_prune_controlSEXP, SEXP omega_controlSEXP,
  SEXP sigma_knownSEXP, SEXP sigmaSEXP, SEXP lambdaSEXP, SEXP nuSEXP,
  SEXP N_postSEXP, SEXP N_burnSEXP,
  SEXP store_posterior_sampleSEXP,
  SEXP verboseSEXP,
  SEXP treatment_codingSEXP,
  SEXP propensity_trainSEXP,
  SEXP propensity_testSEXP,
  SEXP mixture_modeSEXP,
  SEXP mixture_KSEXP,
  SEXP mixture_prior_atom_varianceSEXP,
  SEXP mixture_mass_initSEXP
);

#endif // FUSIONFOREST4_H
