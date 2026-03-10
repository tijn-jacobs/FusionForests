# FusionForests 1.0.0

First release of **FusionForests**.

## New model: `FusionForest`

Added `FusionForest()`, a 3-forest Bayesian tree ensemble for combining
data from a randomised controlled trial (RCT) and an observational study (OS).

The model decomposes the outcome as:

  ŷᵢ = m(xᵢ) + bᵢ · τ₀(xᵢ) + bᵢ · (1 − sᵢ) · τ₁(xᵢ)

where `m(x)` is the control forest, `τ₀(x)` is the RCT treatment forest,
and `τ₁(x)` is the OS deconfounding forest. The binary variable `sᵢ` indicates
whether observation `i` comes from the RCT (1) or the OS (0).

A **commensurate prior** with spike-and-slab structure controls borrowing
strength from the observational data. The `eta_commensurate` argument
configures this prior.

All three forests default to standard BART with leaf hyperparameters
automatically scaled to `1 / sqrt(number_of_trees_*)`.

## Inherited models from FusionForests

The package also includes the full suite of models from the FusionForests
framework for single-study causal inference and survival analysis:

- `HorseTrees()`, `ShrinkageTrees()` — single-forest models with Horseshoe
  and flexible shrinkage priors
- `CausalHorseForest()`, `CausalShrinkageForest()` — BCF-style causal models
- `SurvivalBART()`, `SurvivalDART()`, `SurvivalBCF()`, `SurvivalShrinkageBCF()`
  — survival wrappers supporting right-censored and interval-censored outcomes

All models support continuous, binary, and censored survival outcomes via
an AFT framework, multi-chain MCMC, and S3 methods for `print`, `summary`,
`predict`, and `plot`.
