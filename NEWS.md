# FusionForests 1.1.2

- Censored event times are now augmented with the error scale of their own
  source under the error models with a scale per source (`source_hdp_scale`,
  `source_dp_scale`, `source_gaussian`).  They used to get the pooled scale
  for every observation, which pulled the per-source scales towards each
  other.  The pooled scale is still the forests' reference scale and is what
  `sigma` returns.  Fits under these error models change; the other error
  models are unaffected.
- The default `k_treat` is now 1, so every forest defaults to `k = 1`.  Pass
  `k_treat = 0.5` to reproduce earlier default fits.
- The preliminary fit that centres and scales a censored outcome now uses the
  interval bounds.  It used to read interval-censored rows as right-censored at
  `y`, which overstated the location and scale, and with them `sigma_hat` and
  the size of every forest's prior on the response scale.  Fits with
  interval-censored rows change, by about a factor 0.6 in `sigma_hat` in our
  simulation design.
- When the `survival` package is missing, or `survreg()` fails or does not
  converge, the preliminary fit now solves the same censored-normal likelihood
  in base R.  It used to fall back on a cruder estimate when `survival` was
  missing, so results could differ between machines.  Fits without interval
  bounds are unchanged when `survival` is installed.

# FusionForests 1.1.1

- The sampler for the centred error mixture is now exact.  It samples the
  uncentred (parameter-expanded) mixture, whose full conditionals are all
  conjugate, and centres every stored draw afterwards (Yang et al., 2010):
  the RCT mixture mean is added to `m0_sh` and the difference of the two
  source means to `d`; the atoms are shifted to mean zero.  Previously the
  forests conditioned on the centred atoms while the atoms and weights were
  updated as in the uncentred model.  `tau` and `c` are unaffected; `m0_sh`
  and `d` now carry the source-specific error means.  Draws are not
  bit-identical to 1.1.0.
- The baseline forest now fits the current augmented outcome; it used to see
  the outcome from one sweep earlier.
- `dp_mu_g` is returned for every mixture error model, and the mixture
  outputs are documented in full.
- With `k_deviation = 0` and a source-specific error model, `d` equals the
  constant difference of the source-specific error means.

# FusionForests 1.1.0

- New data sets `breast_fusion` and `breast_fusion_truth`.
- The `k_*` arguments now follow the paper's scaling
  σ_h = k / (2√J), with defaults `k = 1` and `k_treat = 0.5`. Double old
  `k_*` values to reproduce earlier fits.
- The default `error_dist` is now `"source_hdp_scale"`.
- New `error_dist = "source_gaussian"`: a normal error with a separate
  scale per source.
- `fusion_estimand()` now respects `treatment_coding` and per-source
  error scales.
- The survival difference for `target_source = "rwd"` is now causal: it
  uses `tau` only and no longer adds the confounding function `c`.
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
