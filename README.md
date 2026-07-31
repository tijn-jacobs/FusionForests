# FusionForests <img src="https://img.shields.io/badge/R%3E%3D-3.5-blue" alt="R >= 3.5"> ![License: MIT](https://img.shields.io/badge/license-MIT-green)

<img src="sticker/FusionForests_hex.png" align="right" width="150"/>

**FusionForests** is an R package for Bayesian tree ensemble models for
**data fusion** and **causal inference**. The flagship model,
`FusionForest()`, combines data from a randomised controlled trial (RCT)
and an observational study (real-world data, RWD) in a single Bayesian
framework, using a commensurate prior for adaptive information borrowing.

The model targets heterogeneous treatment effects on continuous and
(interval-)censored survival outcomes. The outcome is decomposed over
separate tree forests:

$$Y = \mu(X) + (1-S)\ g(X) + A\ \tau(X) + (1-S)\ A\ c(X) + \sigma\varepsilon,$$

where $Y$ is the log survival time (accelerated failure time
formulation) or a continuous outcome, $A$ is the treatment and $S$
indicates the source ($S = 1$ for the RCT, $S = 0$ for the RWD). Each
term is modelled by its own forest:

- $\mu(X)$ — **control forest**: the prognostic surface under control;
- $g(X)$ — **deconfounding forest**: an RWD-only shift that absorbs
  confounding in the observational data;
- $\tau(X)$ — **treatment forest**: the heterogeneous treatment effect,
  identified by the RCT;
- $c(X)$ — **deviation forest**: how the RWD treatment effect deviates
  from the RCT one.

A commensurate prior shrinks the deviation forest towards zero, so the
RWD sharpens the RCT treatment effect estimate without importing its
confounding.

## Installation

The development version can be installed from GitHub:

```r
# install.packages("remotes")
remotes::install_github("tijn-jacobs/FusionForests")
```

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

## The FusionForest model

`FusionForest()` is the heart of the package. It fits the decomposition
above with standard BART priors on each forest and returns posterior
draws of every component, so treatment effects, borrowing strength and
uncertainty are all available directly from one fit. Right-censored and
interval-censored survival outcomes are handled through data
augmentation, and the error distribution can be Gaussian or a
Dirichlet-process mixture (`error_dist`) for added robustness.

Two companion functions summarise a fit: `fusion_estimand()` returns
posterior draws of causal survival estimands (survival difference,
RMST difference, acceleration factor), and `fusion_projection()`
projects the treatment effect surface onto an interpretable linear
basis.

The single-study causal and survival models from the
[ShrinkageTrees](https://cran.r-project.org/package=ShrinkageTrees)
package are re-exported, so `library(FusionForests)` provides every
model from the accompanying paper in one namespace.

## Reference

The methodology is described in:

> _Bayesian fusion forests for heterogeneous treatment effects on survival from randomised and real-world data_
> T. Jacobs, S.L. van der Pas, W.N. van Wieringen
> arXiv preprint (2026)

## License

[MIT License](https://cran.r-project.org/web/licenses/MIT)

## Funding

This project has received funding from the European Research Council (ERC) under the European Union’s Horizon Europe program under Grant agreement No. 101074802. Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the European Union or the European Research Council Executive Agency. Neither the European Union nor the granting authority can be held responsible for them. This work used the Dutch national e-infrastructure with the support of the SURF Cooperative using grant no. EINF-18803.
