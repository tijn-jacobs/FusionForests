# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Package Overview

FusionForests is an R package for Bayesian tree ensemble models focused on data fusion and causal inference. The flagship model `FusionForest` combines RCT and observational study data using commensurate priors. It also includes `SimpleBART` (single-forest BART) and `SimpleBCF` (Bayesian causal forest), posterior summaries via `fusion_estimand`/`fusion_projection`, and re-exports the single-study models from the CRAN `ShrinkageTrees` package. Accompanies the paper "Bayesian fusion forests for heterogeneous treatment effects on survival from randomised and real-world data".

## Build & Development Commands

```bash
# Install package (from R)
devtools::install()

# Run all tests
devtools::test()

# Run a single test file
testthat::test_file("tests/testthat/test-FusionForest.R")

# Full R CMD check (build + tests + examples + documentation)
devtools::check()

# Rebuild documentation (roxygen2 → man/ and NAMESPACE)
devtools::document()

# Recompile C++ after changes (without full reinstall)
Rcpp::compileAttributes()  # regenerates R/RcppExports.R and src/RcppExports.cpp
devtools::load_all()        # recompiles and loads
```

## Architecture

### Two C++ backends (only one is active)

1. **Stan-based forest** (`src/Stan*.h/.cpp`) — Active implementation. Standard BART and Dirichlet BART (DART). All new development targets this path.
2. **RJMCMC forest** (`src/deprecated/`) — Deprecated. BART with global-local shrinkage priors (horseshoe/half-Cauchy) via reversible-jump MCMC. Source retained for reference only; not exported.

### C++ layer (`src/`)

- `StanForest.h/cpp` — Forest ensemble (collection of trees)
- `StanTree.h/cpp` — Individual tree nodes
- `StanTreeFunctions.h/cpp` — Tree operations (grow/prune)
- `StanBirthDeath.h/cpp` — Birth-death MCMC proposal moves
- `StanForestFunctions.h/cpp` — Forest-level MCMC operations
- `ForestEngine.h` — Wrapper around StanForest for BART/DART modes
- `FusionForest.h/cpp` — Rcpp entry point for the data fusion model (3-forest ensemble)
- `SimpleBART.h/cpp` — Rcpp entry point for single-forest BART (supports IRS modes 0-3)
- `OuterGibbsFunctions.h/cpp` — Outer Gibbs sampler orchestration
- `CommensurateParameters.h/cpp` — Commensurate prior for information borrowing
- `Prerequisites.h` — Common includes, constants (PI, log2pi), utility math functions

Compiled with OpenMP via `src/Makevars` (portable flags only — CRAN rejects `-O3 -march=native -ffast-math`).

### R layer (`R/`)

- `FusionForest.R` — Main export: data fusion model
- `FusionForest-methods.R` — S3 methods (print, summary) for FusionForest fits
- `SimpleBART.R`, `SimpleBCF.R` — Single-forest BART and Bayesian causal forest
- `estimands.R` — `fusion_estimand()`: posterior causal survival estimands
- `projection.R` — `fusion_projection()`: posterior linear projections
- `reexports.R` — Re-exports of single-study models from the CRAN `ShrinkageTrees` package (Imports)
- `helpers.R` — Utility functions
- `RcppExports.R` — Auto-generated Rcpp bindings (do not edit manually)

Exported: `FusionForest`, `SimpleBART`, `SimpleBCF`, `fusion_estimand`, `fusion_projection`, plus the ShrinkageTrees re-exports.

### Tests (`tests/testthat/`)

Uses testthat edition 3. Tests use small datasets (n≈60) and minimal MCMC iterations (N_post=50, N_burn=20) for speed. All tests use `set.seed(1)` for reproducibility.

## Coding Conventions

- R and C++ lines under 80 characters where possible
- Documentation via roxygen2 comments (`#'`)
- NAMESPACE is auto-generated — run `devtools::document()` after changing exports
- C++11 standard (unique_ptr, vector, random library)
- No external C++ math libraries — statistical functions implemented in-house
