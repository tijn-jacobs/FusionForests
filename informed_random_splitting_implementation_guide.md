# Informed Random Splitting: Implementation Guide

## What we are implementing

We are adding a novel missing data handling method called **informed random splitting** to a BART (Bayesian Additive Regression Trees) implementation. This method handles missing covariates *within* the BART MCMC sampler by making stochastic left/right routing decisions at tree nodes where the split variable is missing.

## Background: how BART works (brief)

BART models an outcome as a sum of trees:

```
f(X_i) = sum_{g=1}^{G} rho_g(X_i)
```

where `rho_g` is the routing function of tree `g`: it sends observation `X_i` down the tree according to splitting rules and returns the step height `h_ℓ` of the terminal node (leaf) it reaches.

Each tree `g` has:
- A tree structure `T_g` (topology + splitting rules `(j, c)` at each internal node: go left if `X_j < c`, right if `X_j >= c`)
- A set of step heights `H_g = {h_ℓ : ℓ in leaves}`, one scalar per leaf

The BART MCMC (Bayesian backfitting) cycles through trees `g = 1, ..., G`. For each tree:
1. Compute partial residuals: `R_i = Y_i - sum_{g' ≠ g} rho_{g'}(X_i)`
2. Propose a structural change (grow/prune) and accept/reject via Metropolis-Hastings
3. Draw leaf (step height) parameters from their conjugate normal posterior

The conjugate leaf posterior is: given observations with residuals `R_1, ..., R_n` in a leaf, and prior `h ~ N(0, τ_h²)`:

```
h | R_1,...,R_n ~ N(σ⁻² * sum(R_j) / (σ⁻² * n + τ_h⁻²), 1 / (σ⁻² * n + τ_h⁻²))
```

## The problem: missing covariates

When an observation reaches a split node on variable `X_j` and `X_j` is missing, we cannot route it left or right. This is especially relevant in **block-wise missingness**: e.g. a covariate measured in the RCT but entirely absent from the observational data.

## Our solution: informed random splitting

### Phase 1: Uniform random splitting (implement first)

When an observation hits a split node where its split variable is missing, flip a fair coin:

```
P(go left | covariate missing) = 0.5
```

This is the simplest baseline. At each MCMC iteration, for each tree, when routing observations to compute partial residuals and leaf assignments, missing observations get a fresh coin flip at each ambiguous node.

**Implementation points:**
- This happens during the tree traversal step when assigning observations to leaves
- The coin flip is redrawn every MCMC iteration (not fixed across iterations)
- After routing, the leaf parameter updates proceed as normal using whatever observations ended up in each leaf
- When proposing grow/prune moves, the MH acceptance ratio is computed with the current routing

### Phase 2: Informed random splitting (implement after Phase 1 works)

Replace the 0.5 coin flip with an outcome-informed routing probability based on the posterior predictive densities of the child nodes.

At an ambiguous node `η` with left child containing observed residuals `R_L` and right child containing `R_R`:

**Step 1:** Compute the posterior predictive under each child. The posterior predictive for a new residual under child `d ∈ {L, R}` is:

```
R_i | R_d ~ N(hat_h_d, σ² + hat_v_d)
```

where:
```
hat_h_d = (σ⁻² * sum(R_j in d)) / (σ⁻² * n_d + τ_h⁻²)     # posterior mean of step height
hat_v_d = 1 / (σ⁻² * n_d + τ_h⁻²)                            # posterior variance of step height
```

**Step 2:** Compute the routing probability:

```
π = φ(R_i; hat_h_L, σ² + hat_v_L) / [φ(R_i; hat_h_L, σ² + hat_v_L) + φ(R_i; hat_h_R, σ² + hat_v_R)]
```

where `φ(x; m, v)` is the Gaussian density with mean `m` and variance `v` evaluated at `x`.

**Step 3:** Draw routing: `Z ~ Bernoulli(π)`. Go left if `Z = 1`, right if `Z = 0`.

**Key property:** This is NOT an approximation. It is the exact conditional posterior of the routing indicator under a uniform prior on the routing. So it corresponds to valid Gibbs sampling.

### Practical notes for implementation

- The routing probability `π` uses only quantities already available in the BART sampler: partial residuals, leaf sufficient statistics, and `σ²`
- When both children have similar step heights, `π ≈ 0.5` (reverts to uniform — harmless since the routing doesn't matter much)
- When step heights differ substantially, the routing has strong signal to send the observation to the correct side
- For numerical stability, compute the log densities and use the log-sum-exp trick:
  ```
  log_φ_L = -0.5 * log(v_L) - 0.5 * (R_i - hat_h_L)² / v_L
  log_φ_R = -0.5 * log(v_R) - 0.5 * (R_i - hat_h_R)² / v_R
  π = 1 / (1 + exp(log_φ_R - log_φ_L))
  ```
  where `v_L = σ² + hat_v_L` and `v_R = σ² + hat_v_R`.

### Where in the code this needs to happen

The routing decision occurs during **tree traversal** — the function that sends an observation down a tree to determine its leaf assignment. This is typically called:
1. When computing partial residuals (need to know each observation's current leaf in every other tree)
2. When evaluating the MH ratio for a proposed grow/prune (need to know how observations split)
3. When drawing leaf parameters (need to know which observations are in each leaf)

Look for the function that traverses an observation down a tree (something like `get_leaf`, `predict_node`, `traverse`, or similar). This is where you add the missing data check and the stochastic routing.

### What NOT to change

- The tree proposal mechanism (grow/prune) does not need to change
- The leaf parameter conjugate update does not need to change (it just uses whatever observations ended up in each leaf after routing)
- The `σ²` update does not need to change
- The prior on tree structure does not need to change

### Testing strategy

1. **No missingness**: with complete data, the method should reproduce standard BART exactly
2. **MCAR missingness**: randomly mask some covariate values. Compare predictions to standard BART run on complete data. The informed version should recover reasonable predictions
3. **Block-wise missingness**: mask an entire covariate for a subset of observations (simulating the data fusion setting). Check that the model still estimates the outcome function well
4. **Simple DGP for sanity check**: `Y = f(X1) + ε` where `X1` is observed, but `X2` (noise variable) has missingness. The method should basically ignore `X2` and focus on `X1`
5. **Informative covariate with missingness**: `Y = X1 + X2 + ε`, mask `X2` for half the observations. Check that informed routing outperforms uniform routing in prediction accuracy
