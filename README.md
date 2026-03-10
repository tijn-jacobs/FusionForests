# FusionForests <img src="https://img.shields.io/badge/R%3E%3D-4.1-blue" alt="R >= 4.1"> ![License: MIT](https://img.shields.io/badge/license-MIT-green) ![Status: Development](https://img.shields.io/badge/status-in%20development-orange)

<img src="sticker/FusionForests_hex.png" align="right" width="150"/>

> **This package is under active development. The API may change without notice and features may be incomplete or unstable.**

## Overview

**FusionForests** is an R package for Bayesian tree ensemble models focused on **data fusion** and **causal inference**. The flagship model, `FusionForest`, combines data from a randomised controlled trial (RCT) and an observational study into a single Bayesian framework using commensurate priors for automatic information borrowing.

The package also includes the full **FusionForests** suite for single-study causal inference and survival analysis.

## Reference

> _Horseshoe Forests for High-Dimensional Causal Survival Analysis_
> T. Jacobs, W.N. van Wieringen, S.L. van der Pas
> https://arxiv.org/abs/2507.22004

## Development notes

The C++ backend contains two historically parallel implementations:

1. **Stan-based forest** (`Stan*` files) — standard BART and Dirichlet BART (DART). This is the active implementation used going forward.
2. **RJMCMC forest** (`Forest`, `Tree`, `TreeModifications`, etc.) — BART with global-local shrinkage priors (horseshoe / half-Cauchy) on the leaf step heights, requiring a reversible-jump MCMC sampler. **This implementation has been deprecated during development** and the source files have been moved to `src/deprecated/` for reference. The associated R entry points (`HorseTrees`, `CausalHorseForest`, `probitHorseTrees`) are no longer exported.

All new development targets the Stan-based path.

## License

[MIT License](https://cran.r-project.org/web/licenses/MIT)

## Funding

Funded by the European Research Council (ERC) under Horizon Europe (Grant No. 101074082).
