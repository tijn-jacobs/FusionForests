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
