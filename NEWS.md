# FusionForests 1.1.0

- New data sets `breast_fusion` and `breast_fusion_truth`.
- The `k_*` arguments now follow the paper's scaling
  σ_h = k / (2√J), with defaults `k = 1` and `k_treat = 0.5`. Double old
  `k_*` values to reproduce earlier fits.
- The default `error_dist` is now `"source_hdp_scale"`.
- `fusion_estimand()` now respects `treatment_coding` and per-source
  error scales.
- Removed `SimpleBART()`, `SimpleBCF()`, the `decomposition` argument
  and the `"RMST"` estimand.
- `FusionForest()` now errors on missing covariate values.
- Documentation in the paper's notation, code cleanup and more tests.

# FusionForests 1.0.1

- Internal cleanup of the C++ layer for CRAN portability: removed
  unused standard-library includes, routed all console output
  explicitly through `Rcpp::Rcout`, and restricted the shared
  library's symbol visibility to the registered entry points.

# FusionForests 1.0.0

First release of **FusionForests**.

## New model: `FusionForest()`

`FusionForest()` is a Bayesian tree ensemble for combining data from a
randomised controlled trial (RCT) and real-world data (RWD).

The outcome (continuous, or log survival time) is decomposed over
separate tree forests: a control forest, a treatment forest, a
deconfounding forest and a deviation forest that captures how the
treatment effect in the RWD deviates from the RCT. The observational
data are not assumed to be unconfounded.

Continuous, right-censored and interval-censored outcomes are
supported via an accelerated failure time formulation, with Gaussian
or Dirichlet-process mixture error distributions (`error_dist`).

## Posterior summaries

- `fusion_estimand()` — posterior draws of causal survival estimands:
  survival difference and acceleration factor.
- `fusion_projection()` — interpretable linear projections of the
  posterior treatment effect surface.
- `print()` and `summary()` methods for `FusionForest` fits.

## Additional models

- `SimpleBART()` — single-forest BART.
- `SimpleBCF()` — Bayesian causal forest with prognostic and treatment
  forests.

## Re-exports

The single-study causal and survival models from the
[ShrinkageTrees](https://cran.r-project.org/package=ShrinkageTrees)
package (`ShrinkageTrees()`, `HorseTrees()`, `CausalShrinkageForest()`,
`CausalHorseForest()`, `SurvivalBART()`, `SurvivalDART()`,
`SurvivalBCF()`, `SurvivalShrinkageBCF()`) are re-exported, so
`library(FusionForests)` provides every model from the accompanying
paper in one namespace.
