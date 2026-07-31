# FusionForests <img src="https://img.shields.io/badge/R%3E%3D-3.5-blue" alt="R >= 3.5"> ![License: MIT](https://img.shields.io/badge/license-MIT-green)

<img src="sticker/FusionForests_hex.png" align="right" width="150"/>

**FusionForests** is an R package for Bayesian tree ensemble models for
**data fusion** and **causal inference**. The flagship model,
`FusionForest()`, combines data from a randomised controlled trial (RCT)
and an observational study (real-world data, RWD) in a single Bayesian
framework, using a commensurate prior for adaptive information borrowing.

The model targets heterogeneous treatment effects on continuous and
(interval-)censored survival outcomes via an accelerated failure time
formulation. The outcome is decomposed over separate tree forests — a
control forest, a treatment forest, a deconfounding forest and,
optionally, a deviation forest — so that the RWD can sharpen the RCT
treatment effect estimate without importing its confounding.

## Installation

The development version can be installed from GitHub:

```r
# install.packages("remotes")
remotes::install_github("tijn-jacobs/FusionForests")
```

Installation compiles C++ source code, so a working toolchain
(Rtools on Windows, Xcode command line tools on macOS) is required.

## Quick example

```r
library(FusionForests)

# Simulated fusion data: an RCT and an RWD cohort
set.seed(1)
n <- 500
X <- matrix(rnorm(n * 3), n, 3)
s <- rbinom(n, 1, 0.5)  # 1 = RCT, 0 = RWD
a <- rbinom(n, 1, 0.5)  # treatment
true_time <- exp(1 + X[, 1] + 0.5 * a + 0.3 * rnorm(n))
cens_time <- rexp(n, rate = 1 / (2 * mean(true_time)))
time   <- pmin(true_time, cens_time)
status <- as.integer(true_time <= cens_time)

fit <- FusionForest(
  y = time, status = status,
  X_train_control = X, X_train_treat = X,
  treatment_indicator_train = a, source_indicator_train = s,
  X_test_control = X, X_test_treat = X,
  treatment_indicator_test = a, source_indicator_test = s,
  outcome_type = "right-censored",
  store_posterior_sample = TRUE
)
print(fit)

# Posterior draws of causal survival estimands at the test points
af   <- fusion_estimand(fit, estimand = "AF")            # acceleration factor
rmst <- fusion_estimand(fit, estimand = "RMST", time = 5) # RMST difference

# Interpretable linear projection of the treatment effect surface
colnames(X) <- paste0("x", 1:3)
proj <- fusion_projection(fit, basis = ~ x1 + x2 + x3,
                          X_eval = as.data.frame(X))
```

## Models

| Function              | Description                                                                          |
| --------------------- | ------------------------------------------------------------------------------------ |
| `FusionForest()`      | RCT + RWD data fusion with commensurate-prior borrowing                              |
| `fusion_estimand()`   | Posterior causal survival estimands (survival difference, RMST, acceleration factor) |
| `fusion_projection()` | Posterior linear projections of the treatment effect surface                         |
| `SimpleBART()`        | Single-forest BART                                                                   |
| `SimpleBCF()`         | Bayesian causal forest (prognostic + treatment forest)                               |

The single-study causal and survival models from the
[ShrinkageTrees](https://cran.r-project.org/package=ShrinkageTrees)
package (`CausalShrinkageForest()`, `SurvivalBART()`, and friends) are
re-exported, so `library(FusionForests)` provides every model from the
accompanying paper in one namespace.

## Reference

The methodology is described in:

> _Bayesian fusion forests for heterogeneous treatment effects on survival from randomised and real-world data_
> T. Jacobs, S.L. van der Pas, W.N. van Wieringen
> arXiv preprint (2026)

The single-study horseshoe models are described in:

> _Horseshoe Forests for High-Dimensional Causal Survival Analysis_
> T. Jacobs, W.N. van Wieringen, S.L. van der Pas
> [arXiv:2507.22004](https://arxiv.org/abs/2507.22004)

## License

[MIT License](https://cran.r-project.org/web/licenses/MIT)

## Funding

This project has received funding from the European Research Council (ERC) under the European Union’s Horizon Europe program under Grant agreement No. 101074802. Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the European Union or the European Research Council Executive Agency. Neither the European Union nor the granting authority can be held responsible for them. This work used the Dutch national e-infrastructure with the support of the SURF Cooperative using grant no. EINF-18803.
