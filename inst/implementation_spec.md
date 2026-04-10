# Implementation Specification: Joint Fusion Model with MAP-Prior Borrowing and Missing Covariate Imputation

## Context

We have an existing BART implementation and a fusion model with three separate forests for the decomposition

$$\log T = m_0(X, S) + \tau(X) A + (1 - S) A\, c(X) + \sigma \varepsilon.$$

This document specifies two extensions:

1. **Replacing the single $m_0$ forest with a two-forest MAP-prior decomposition.**
2. **Adding an imputation component for missing covariates with the same MAP-prior structure**, embedded in the joint Gibbs sampler.

The current three-forest implementation models $m_0(X, S)$ with a single BART that takes $S$ as a covariate. We replace this with an asymmetric decomposition that gives a controllable borrowing knob.

---

## Part 1: MAP-prior decomposition for $m_0$

### Mathematical structure

Replace the single $m_0$ forest with

$$m_0(X, S) = \mu(X) + (1 - S)\, g(X)$$

where:

- $\mu(X)$ is a BART forest fit using **all observations** (RCT and RWD). It represents the shared baseline log-survival surface and equals $m_0(X, 1)$, the RCT baseline.
- $g(X)$ is a BART forest fit using **only RWD observations** ($S = 0$). It captures the systematic difference between the RWD baseline and the RCT baseline, so that $m_0(X, 0) = \mu(X) + g(X)$.

The model now has **four outcome forests**: $\mu$, $g$, $\tau$, $c$ (plus the residual variance $\sigma^2$).

### BART priors

Each forest uses the standard BART prior with its own hyperparameters:

- $\mu$: $m_\mu$ trees, leaf prior scale $k_\mu$ (default $k_\mu = 2$).
- $g$: $m_g$ trees, leaf prior scale $k_g$ (controls borrowing — see below).
- $\tau$: $m_\tau$ trees, leaf prior scale $k_\tau$.
- $c$: $m_c$ trees, leaf prior scale $k_c$.

The standard BART leaf prior is $\mathcal{N}(0, \sigma_{\mu,\bullet}^2)$ where $\sigma_{\mu,\bullet}^2 = 1 / (4 k_\bullet^2 m_\bullet)$ on the standardized response scale.

**Borrowing parameter $k_g$:**

- $k_g \to \infty$ → $g \equiv 0$ → full pooling, $m_0(X, 0) = m_0(X, 1) = \mu(X)$.
- $k_g = k_\mu$ → no shrinkage of $g$ relative to a standalone BART.
- Intermediate values give partial pooling.

For initial experiments, **fix $k_g$ to a chosen value** (e.g., $k_g = 2, 5, 10, \infty$) so we can study sensitivity. Later we will give it a hyperprior.

### Fitting the forests in the Gibbs sampler

The four forests are updated sequentially using **Bayesian backfitting** (the standard BART MCMC). For each forest, the partial residual is computed by subtracting the contributions of all other components from the response.

Let $r_i = \log T_i$ be the response for observation $i$. The partial residuals are:

**For $\mu$:** Use all observations.
$$r_i^{(\mu)} = \log T_i - (1 - S_i) g(X_i) - \tau(X_i) A_i - (1 - S_i) A_i c(X_i)$$
Fit BART for $\mu$ to $r_i^{(\mu)}$ on all $n$ observations.

**For $g$:** Use only RWD observations ($S_i = 0$).
$$r_i^{(g)} = \log T_i - \mu(X_i) - \tau(X_i) A_i - A_i c(X_i), \quad i: S_i = 0$$
Fit BART for $g$ to $r_i^{(g)}$ on the $n_{\text{RWD}}$ RWD observations only.

**For $\tau$:** Use all observations.
$$r_i^{(\tau)} = \log T_i - \mu(X_i) - (1 - S_i) g(X_i) - (1 - S_i) A_i c(X_i)$$
Fit BART for $\tau$ to $r_i^{(\tau)} / A_i$ on observations with $A_i = 1$, or use the standard BART trick for fitting $A \cdot \tau(X)$ as a regression on $X$ with $A$ as a multiplier. (This depends on the existing implementation. If $\tau$ is currently fit as a standard forest with $A$ as a covariate, keep that approach and use the appropriate partial residual.)

**For $c$:** Use only RWD observations with $A = 1$ (the only place $c$ appears in the likelihood).
$$r_i^{(c)} = \log T_i - \mu(X_i) - g(X_i) - \tau(X_i), \quad i: S_i = 0, A_i = 1$$
Fit BART for $c$ to $r_i^{(c)}$ on these observations.

**For $\sigma^2$:** Update from the inverse-gamma full conditional based on the residuals after subtracting all four forest contributions.

> **Implementation note:** If the current three-forest implementation has a clean abstraction for "fit a forest to a partial residual on a subset of observations", the addition of $g$ should slot in by following the same pattern. The only new thing is the subset restriction (RWD only) and the leaf-prior hyperparameter $k_g$.

---

## Part 2: Imputation component

### Mathematical structure

Let $X_j$ be the covariate subject to missingness. Specify its conditional distribution given the other covariates $X_{-j}$ as

$$X_j \mid X_{-j}, S \sim h(X_{-j}) + (1 - S)\, \delta(X_{-j}) + \eta, \quad \eta \sim \mathcal{N}(0, \sigma_{X_j}^2)$$

where:

- $h(X_{-j})$ is a BART forest fit on **all observations with $X_j$ observed**, regardless of source.
- $\delta(X_{-j})$ is a BART forest fit on **RWD observations with $X_j$ observed**. Under block_rwd missingness, this set is empty, and $\delta$ is updated only through its prior (centered at zero) and the outcome-likelihood feedback in the Metropolis step below.
- $\sigma_{X_j}^2$ has an inverse-gamma prior.

This mirrors the $\mu / g$ structure of the outcome decomposition. The borrowing parameter $k_\delta$ controls how strongly $\delta$ is shrunk toward zero.

For multiple missing covariates, the simplest approach is to handle them one at a time (sequentially), each with its own pair of imputation forests $(h_j, \delta_j)$. We start with the single-missing-covariate case.

### Treating missing values as latent variables

Let $X_j^{\text{mis}}$ denote the vector of missing $X_j$ values. These are **latent variables** in the joint model. At each Gibbs iteration:

1. The imputation forests $h$ and $\delta$ are updated treating the current values of $X_j^{\text{mis}}$ (from the previous iteration) as observed.
2. New values of $X_j^{\text{mis}}$ are then sampled from the joint full conditional, which combines the imputation model and the outcome model.

This is the standard data-augmentation approach to missing data in Bayesian models.

### Initialization

At the start of the sampler, initialize $X_j^{\text{mis}}$ with reasonable starting values. Two simple options:

- **Mean imputation:** Set each missing value to the mean of the observed $X_j$ values (in either the same source or the combined data).
- **Random sampling:** Draw each missing value from $\mathcal{N}(\bar{X}_j, \text{var}(X_j))$ using the empirical mean and variance of the observed values.

A short warm-up period (e.g., 100–500 iterations) where only the imputation forests are updated, before the outcome forests start contributing, can stabilize the early MCMC.

### Fitting the imputation forests

**For $h$:** Use all observations where $X_j$ is observed (in the dataset, not the imputed values).
Treat $X_{j,i}$ as the response and $X_{-j,i}$ as the predictors. Compute the partial residual
$$r_i^{(h)} = X_{j,i} - (1 - S_i) \delta(X_{-j,i})$$
and fit BART for $h$ to this.

Wait — this is subtle. There are two design choices about which observations to use:

**Design A (purist):** Fit $h$ and $\delta$ using only observations with $X_j$ actually observed. The latent values $X_j^{\text{mis}}$ do not contribute to fitting the imputation forests — they only get drawn from the imputation distribution. Under block_rwd, this means $\delta$ has no observations to fit on and is updated only through its prior.

**Design B (data-augmented):** Fit $h$ and $\delta$ using all observations, treating $X_j^{\text{mis}}$ as if they were observed (using the current imputed values). This is the standard data-augmentation approach. Under block_rwd, $\delta$ now has 350 "pseudo-observations" (the imputed RWD values), but these were generated from the imputation model itself, so the posterior for $\delta$ degenerates without strong outcome-likelihood feedback.

**Use Design B (data-augmented).** This is the standard approach and what makes the joint Gibbs sampler coherent. The imputed values $X_j^{\text{mis}}$ are treated as latent variables; the imputation forests are conditioned on them; the outcome model provides the feedback that prevents degeneracy.

So the partial residuals for the imputation forests are:

**For $h$:** Use **all** observations (with $X_j$ from the data when observed, and from the current imputed values when missing).
$$r_i^{(h)} = X_{j,i}^{\text{current}} - (1 - S_i) \delta(X_{-j,i})$$
Fit BART for $h$ to $r_i^{(h)}$ on all $n$ observations.

**For $\delta$:** Use **only RWD observations**.
$$r_i^{(\delta)} = X_{j,i}^{\text{current}} - h(X_{-j,i}), \quad i: S_i = 0$$
Fit BART for $\delta$ to $r_i^{(\delta)}$ on the $n_{\text{RWD}}$ RWD observations only.

**For $\sigma_{X_j}^2$:** Update from the inverse-gamma full conditional based on the residuals
$$X_{j,i}^{\text{current}} - h(X_{-j,i}) - (1 - S_i) \delta(X_{-j,i})$$
on all $n$ observations.

### Sampling missing $X_j$ values (the Metropolis step)

For each observation $i$ with $X_{j,i}$ missing, draw a new value $X_{j,i}^{\text{new}}$ from the joint full conditional

$$p(X_{j,i} \mid \text{rest}) \propto \underbrace{p(\log T_i \mid X_i, A_i, S_i, \Theta)}_{\text{outcome likelihood}} \times \underbrace{\mathcal{N}(X_{j,i}; \hat{X}_{j,i}, \sigma_{X_j}^2)}_{\text{imputation model}}$$

where $\hat{X}_{j,i} = h(X_{-j,i}) + (1 - S_i) \delta(X_{-j,i})$ is the imputation model's prediction.

The imputation factor is Gaussian. The outcome factor is generally not conjugate (it depends on BART evaluations of $\mu$, $g$, $\tau$, $c$ at $X_i$ with the new $X_{j,i}$ value). Use a **Metropolis-Hastings step** with the imputation Gaussian as the proposal distribution:

1. Propose $X_{j,i}^\star \sim \mathcal{N}(\hat{X}_{j,i}, \sigma_{X_j}^2)$.
2. Construct $X_i^\star$ by replacing the $j$-th coordinate of $X_i$ with $X_{j,i}^\star$.
3. Compute the outcome model predictions $\hat{Y}_i = \mu(X_i) + (1 - S_i) g(X_i) + \tau(X_i) A_i + (1 - S_i) A_i c(X_i)$ and $\hat{Y}_i^\star = \mu(X_i^\star) + (1 - S_i) g(X_i^\star) + \tau(X_i^\star) A_i + (1 - S_i) A_i c(X_i^\star)$.
4. Compute the acceptance probability
   $$\alpha = \min\left\{1, \frac{\mathcal{N}(\log T_i; \hat{Y}_i^\star, \sigma^2)}{\mathcal{N}(\log T_i; \hat{Y}_i, \sigma^2)}\right\}$$
   This is the ratio of outcome-likelihood densities; the proposal density and the prior cancel because the proposal is the imputation model.
5. With probability $\alpha$, accept the proposal: $X_{j,i}^{\text{new}} = X_{j,i}^\star$. Otherwise, keep the current value.

> **Why the proposal density cancels:** The MH acceptance ratio is
> $$\alpha = \frac{p(X_{j,i}^\star \mid \text{rest})}{p(X_{j,i} \mid \text{rest})} \cdot \frac{q(X_{j,i} \mid X_{j,i}^\star)}{q(X_{j,i}^\star \mid X_{j,i})}$$
> where $q$ is the proposal. With a symmetric proposal (Gaussian centered at the imputation mean), the proposal ratio becomes the ratio of the imputation densities at $X_{j,i}$ and $X_{j,i}^\star$. But the target ratio $p(X_{j,i}^\star \mid \text{rest}) / p(X_{j,i} \mid \text{rest})$ also includes the imputation density factor (which is the prior in the full conditional). These cancel, leaving only the outcome-likelihood ratio.
>
> **More precisely:** the proposal is *not* symmetric around the current value (it's centered at $\hat{X}_{j,i}$, not at $X_{j,i}$). So we need an independence Metropolis ratio:
> $$\alpha = \min\left\{1, \frac{p(\log T_i \mid X_i^\star, \dots)}{p(\log T_i \mid X_i, \dots)} \cdot \frac{\mathcal{N}(X_{j,i}^\star; \hat{X}_{j,i}, \sigma_{X_j}^2)}{\mathcal{N}(X_{j,i}; \hat{X}_{j,i}, \sigma_{X_j}^2)} \cdot \frac{\mathcal{N}(X_{j,i}; \hat{X}_{j,i}, \sigma_{X_j}^2)}{\mathcal{N}(X_{j,i}^\star; \hat{X}_{j,i}, \sigma_{X_j}^2)}\right\}$$
> The two imputation density factors are the prior contribution to the target and the proposal density. They cancel, and the acceptance ratio reduces to the outcome-likelihood ratio. **Use this simpler form in implementation.**

---

## Part 3: Full Gibbs sampler

One iteration of the joint Gibbs sampler:

```
INPUT: current state of all forests, leaf parameters, variance components,
       and imputed values X_j^mis
OUTPUT: updated state

# --- Outcome model updates ---
1. Update mu (shared baseline forest):
     compute partial residual r^(mu) on all n observations
     run one BART backfitting sweep for mu

2. Update g (RWD-specific deviation forest):
     compute partial residual r^(g) on RWD observations only (S=0)
     run one BART backfitting sweep for g, using only RWD observations

3. Update tau (treatment effect forest):
     compute partial residual r^(tau) on all n observations
     run one BART backfitting sweep for tau

4. Update c (confounding forest):
     compute partial residual r^(c) on RWD treated observations (S=0, A=1)
     run one BART backfitting sweep for c, using these observations

5. Update sigma^2 (residual variance):
     draw from inverse-gamma full conditional based on outcome residuals

# --- Imputation model updates ---
6. Update h (shared imputation forest):
     compute partial residual r^(h) on all n observations
     (using current imputed X_j values where missing)
     run one BART backfitting sweep for h

7. Update delta (RWD-specific imputation deviation forest):
     compute partial residual r^(delta) on RWD observations only
     run one BART backfitting sweep for delta, using only RWD observations

8. Update sigma_Xj^2 (imputation residual variance):
     draw from inverse-gamma full conditional based on imputation residuals

# --- Imputation step ---
9. For each observation i with X_{j,i} missing:
     compute imputation prediction: hat_X_{j,i} = h(X_{-j,i}) + (1-S_i)*delta(X_{-j,i})
     propose X_{j,i}^star ~ N(hat_X_{j,i}, sigma_Xj^2)
     compute current and proposed outcome predictions
     compute MH acceptance ratio (outcome likelihood ratio only)
     accept or reject
```

---

## Part 4: Hyperparameters to expose

The implementation should expose these as user-settable parameters:

| Parameter | Description | Suggested default |
|-----------|-------------|-------------------|
| $m_\mu, m_g, m_\tau, m_c$ | Number of trees per outcome forest | 50, 50, 50, 50 |
| $k_\mu$ | Leaf prior scale for $\mu$ | 2 |
| $k_g$ | Leaf prior scale for $g$ (outcome borrowing) | **Variable for sensitivity studies; e.g., 2, 5, 10, $\infty$** |
| $k_\tau$ | Leaf prior scale for $\tau$ | 2 |
| $k_c$ | Leaf prior scale for $c$ | 2 |
| $\nu, \lambda$ | Inverse-gamma prior on $\sigma^2$ | BART default |
| $m_h, m_\delta$ | Number of trees per imputation forest | 50, 50 |
| $k_h$ | Leaf prior scale for $h$ | 2 |
| $k_\delta$ | Leaf prior scale for $\delta$ (imputation borrowing) | **Variable for sensitivity studies** |
| $\nu_{X_j}, \lambda_{X_j}$ | Inverse-gamma prior on $\sigma_{X_j}^2$ | BART default |
| Initialization for $X_j^{\text{mis}}$ | Mean / random / user-supplied | Mean of observed $X_j$ |
| Number of MCMC iterations | Total iterations | 2000 |
| Burn-in | Discarded iterations | 500 |
| Warm-up for imputation only | Iterations where outcome forests are not updated | 0 (optional) |

The "$k_g \to \infty$" case should be implemented by **disabling the $g$ forest entirely** (skipping its update step and treating $g \equiv 0$). Same for $k_\delta \to \infty$ disabling $\delta$.

---

## Part 5: Outputs to save

For each MCMC iteration (after burn-in):

- Predictions of $\mu(X_i)$, $g(X_i)$, $\tau(X_i)$, $c(X_i)$ at each training observation (or at a fixed evaluation grid).
- Predictions of the same forests at test observations.
- Posterior draws of $\sigma^2$ and $\sigma_{X_j}^2$.
- Imputed values $X_{j,i}^{\text{new}}$ for each missing observation, at each iteration. (Useful for inspecting the imputation distribution.)
- Acceptance rate of the MH step (per iteration or aggregated). Should be in a reasonable range (20–60%); if it's near 0% or near 100%, the proposal scale needs tuning.

The posterior CATE at observation $i$ is computed at each iteration as
$$\hat{\tau}_i = \tau(X_i)$$
since $\tau$ is identified directly from the model and does not require subtracting outcome predictions.

---

## Part 6: Validation tests before running simulations

Before running the full simulation study, verify:

1. **Oracle case (no missingness):** With no missing data, the joint sampler should reproduce the existing three-forest implementation's results. The imputation forests $h, \delta$ are still updated but never get used for imputation (no missing values to draw).

2. **Full pooling for $m_0$ (large $k_g$):** Setting $k_g$ very large should give results very close to the original three-forest implementation that uses a single $m_0$ forest with $S$ as a covariate. (Not exactly identical, because the parameterization differs.)

3. **No borrowing for $m_0$ (small $k_g$):** Setting $k_g = k_\mu$ should let the deviation forest fit RWD-specific structure freely. Verify that $g$ recovers a non-trivial function in scenarios where the true RWD baseline differs from the RCT baseline.

4. **MH acceptance rate:** Run a small simulation with missingness and check that the acceptance rate is in the 20–60% range. If not, the variance of the proposal (which is $\sigma_{X_j}^2$) may need adjustment, or there may be a bug in the likelihood ratio computation.

5. **Block_rwd recovery:** Run a single simulation with $X_j$ missing in the RWD and verify that the joint sampler produces CATE estimates substantially better than the IRS / Complete-covariates baselines, when $k_\delta$ is set to a borrowing value (small or moderate, not infinite).

---

## Notes on implementation choices

- **Single vs multiple missing covariates:** Start with a single missing covariate. The extension to multiple missing covariates is sequential — one pair of imputation forests per missing covariate, updated in turn. Each missing variable is imputed conditionally on the others.

- **Categorical missing covariates:** Out of scope for now. The method as specified assumes continuous $X_j$.

- **Censoring:** The outcome likelihood for censored observations is the survival function rather than the density. The MH acceptance ratio in step 9 should use whichever form is appropriate for each observation. The existing fusion implementation already handles censoring; the imputation step should call into the same likelihood functions.

- **Code structure:** The main change to the existing fusion sampler is (a) adding the $g$ forest with subset-restricted updates, (b) adding the imputation forests $h, \delta$ with their own update steps, and (c) adding the MH step that draws missing values. Each of these is local to a single Gibbs step and should not require restructuring the existing code.
