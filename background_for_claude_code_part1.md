# Background: Combining RCT and RWD for Heterogeneous Treatment Effects in Survival Data

## Project overview

We are building a Bayesian nonparametric model that combines randomized controlled trial (RCT) data with real-world/observational data (RWD) to estimate heterogeneous treatment effects (HTEs) in survival settings. The target journal is *Biostatistics*.

The key innovation is that we **do not assume unconfoundedness of the observational data**. Instead, we explicitly model unmeasured confounding through a confounding function. The model uses Bayesian Additive Regression Trees (BART) as the nonparametric engine for all three model components.

---

## Notation

- `A ∈ {0, 1}`: treatment assignment
- `S ∈ {0, 1}`: data source indicator (`S = 1` for RCT, `S = 0` for observational study)
- `X ∈ ℝ^p`: high-dimensional baseline covariates
- `T`: survival (time-to-event) outcome
- `Y(a)`: potential outcome under treatment `a`
- `τ(x)`: conditional average treatment effect (CATE)
- `c(x)`: confounding function (bias due to unmeasured confounding in the OS)
- `m_0(x, s)`: baseline/prognostic function (expected outcome under control)

---

## Core model: outcome decomposition

Under an accelerated failure time (AFT) formulation, the outcome is decomposed as:

```
log(T) = m_0(X, S) + A * τ(X) + A * (1 - S) * c(X) + σ * ε
```

where:
- `m_0(X, S)`: prognostic/baseline function — the expected log-survival under control
- `τ(X)`: the conditional average treatment effect (CATE) — the causal effect of treatment
- `c(X)`: the confounding function — captures the bias due to unmeasured confounding in the observational data. This term is only active when `S = 0` (observational) and `A = 1` (treated)
- `ε ~ p(ε)`: mean-zero error with a nonparametric prior (centered stick-breaking mixture)

### Interpretation of each component

The decomposition separates three distinct sources of variation:
1. **m_0(X, S)**: What would happen without treatment? Can differ by data source (different populations).
2. **τ(X)**: The true causal treatment effect, assumed transportable across sources (by exchangeability assumption A4).
3. **c(X)**: The confounding bias. Only present in the observational treated group. Captures the difference between the observed treatment-control contrast in the OS and the true causal effect.

### Why three separate BART models

Each of `m_0`, `τ`, and `c` is modeled by its own BART ensemble. This gives:
- Transparent decomposition: each component is interpretable on its own
- Separate regularization: each function can have its own prior (e.g., stronger shrinkage on `c` to encourage borrowing)
- Identifiability: `τ` is identified from the RCT; `c` captures the residual discrepancy in the OS

---

## Identification and assumptions

The following assumptions are required:

1. **(A1) SUTVA**: No interference, observed outcome equals potential outcome for received treatment.
2. **(A2) Unconfoundedness of the RCT**: `Y(a) ⊥ A | X, S = 1`. Holds by design.
3. **(A3) Positivity of the RCT**: `0 < P(A = a | X, S = 1) < 1`.
4. **(A4) Exchangeability across sources**: `Y(a) | X, S = 1 =d Y(a) | X, S = 0`. The CATE is the same in both populations conditional on X. This is the transportability assumption.
5. **(A5) Unconfoundedness of OS**: This is **not assumed**. Instead, the confounding function `c(x)` captures the violation.

Under (A1)–(A4) and the decomposition, the confounding function is defined as:

```
c(x) = E[Y | X=x, A=1, S=0] - E[Y | X=x, A=0, S=0] - τ(x)
```

---

## Survival estimands

Given the AFT decomposition, the potential survival curves are:

```
S_1(t | x, s) = 1 - Φ((log(t) - m_0(x,s) - τ(x) - (1-s)*c(x)) / σ)
S_0(t | x, s) = 1 - Φ((log(t) - m_0(x,s)) / σ)
```

where `Φ` is the CDF of the error distribution (nonparametric, estimated via CSBM prior).

From these, we compute:
- **Risk difference**: `S_1(t|x) - S_0(t|x)` at a fixed time horizon `t`
- **RMST difference**: `∫_0^T [S_1(u|x) - S_0(u|x)] du`

Posterior estimates are obtained by averaging over MCMC draws and the empirical covariate distribution.

---

## Error distribution: Centered Stick-Breaking Mixture (CSBM) prior

The error term `ε` is given a nonparametric prior to allow flexible, non-Gaussian error distributions:

```
ε_i | μ_i ~ N(μ_i, 1)
μ_i | P ~ P
P = Σ_h V_h * Π_{l<h} (1 - V_l) * δ_{θ_h}
```

where the atoms are centered to ensure mean zero:
```
θ_h = θ*_h - μ_{G*}
V_h ~ Beta(1, α)
```

This prior places support on absolutely continuous distributions with mean zero and identity covariance.

---

## Handling missing covariates in BART

### The problem: block-wise missingness

In practice, RCT and RWD often measure different sets of covariates. A covariate may be measured in the RCT but entirely absent from the RWD (or vice versa). This is **block-wise missingness**: all observations from one data source are missing a particular variable.

Standard approaches (e.g., pre-imputation with MICE) are problematic because:
- The missingness is structural, not random
- A full imputation model is hard to specify well for block-wise patterns
- It separates imputation from the causal model, losing coherence

### Existing approach: MIA (Missing Incorporated in Attributes)

The only current missingness-handling method within BART is MIA, which treats "missing" as a separate category at each split. All missing observations are deterministically sent to one learned side (left or right). This is crude: it doesn't use information about where the missing value likely falls relative to the split point.

### Our approach: stochastic routing within BART

We propose a novel alternative that handles missing covariates **within the BART MCMC sampler** by making stochastic left/right routing decisions at each split node where the split variable is missing.

#### Core idea

When a tree proposes a split on a covariate that is missing for some observation:
1. The observation cannot be deterministically routed left or right.
2. Instead, we make a **stochastic binary decision** (go left or right) for that observation at that split.
3. This decision is made **at each MCMC iteration**, so uncertainty about the routing is propagated through the posterior.
4. Because BART is a sum-of-trees model, different trees may split on the missing variable at different points. The observation may go left in some trees and right in others, creating a natural **multiple imputation effect** across the ensemble.

#### Current implementation: uniform routing (Phase 1)

For the initial implementation, use a **fair coin flip** (probability 0.5) to decide left vs. right when a covariate is missing at a split node:

```
P(go left | covariate missing) = 0.5
```

This is the simplest version. It ensures:
- Missing observations still contribute to the model
- The sum-of-trees structure provides averaging over different routings
- The MCMC explores different routing configurations across iterations
- No additional modeling assumptions are needed

#### Implementation details

- **When does routing happen?** At each MCMC iteration, whenever a tree is grown/modified and a split is proposed on a variable that is missing for some observations.
- **Scope**: This applies independently to each of the three BART ensembles (m_0, τ, c). Each ensemble handles its own missing data routing.
- **Leaf parameters**: After routing decisions are made, the leaf parameters are drawn from the standard conjugate normal posterior, conditioned on whichever observations landed in that leaf (including the stochastically routed ones).
- **Effect on tree acceptance**: If routing missing observations randomly doesn't improve the fit, the BART sampler will tend to prefer splits on observed covariates — the model self-regulates.

#### Future extension: informed routing (Phase 2)

In later work, we will replace the 0.5 coin flip with an **outcome-informed routing probability** based on the posterior predictive densities of the leaf parameters.

For a missing observation with partial residual `R_i` at a split node with left child data `R_L` and right child data `R_R`:

```
P(go left) = p(R_i | R_L) / [p(R_i | R_L) + p(R_i | R_R)]
```

where `p(R_i | R_d)` is the posterior predictive density under child `d`:

```
R_i | R_d ~ N(posterior_mean_d, σ² + posterior_variance_d)
```

with:
```
posterior_mean_d = (σ⁻² * Σ_{j ∈ d} R_j) / (n_d * σ⁻² + τ⁻²)
posterior_variance_d = 1 / (n_d * σ⁻² + τ⁻²)
```

This uses the machinery already inside BART (conjugate normal leaf model) to make outcome-informed routing decisions, without needing an external imputation model. Observations whose residuals are more consistent with one child's leaf distribution get routed there with higher probability.

---

## Key references

- **Ye et al. (2025)**: Frequentist AFT-based integrative analysis of RCT + RWD with censoring and hidden confounding. Penalized weighted least squares. Synthetic data: https://github.com/ke-zhu/intFRT
- **Yozova et al. (2025)**: Bayesian data fusion under weak identifiability using BCF with tempering
- **Yang et al. (2025)**: Semiparametric framework for combining RCT + OS through confounding function, efficiency bounds for HTE
- **Kallus et al. (2018)**: Experimental grounding — using small RCT to correct hidden confounding in larger OS
- **Zhou and Ji (2021)**: BART for incorporating external data into clinical trials (T-learner approach)
- **Hahn et al. (2020)**: Bayesian Causal Forest (BCF) — regularization, confounding, heterogeneous effects
- **Lin et al. (2024)**: Shows bias correction helps HTE but not marginal ATE — motivates our HTE focus
- **Yang et al. (2010)**: Centered stick-breaking mixture prior for the error distribution
