# Review notes — simulation study (`sim_surv_v12.R`)

Concerns and observations from reviewing the main simulation study against the
manuscript (`manuscripts/manuscript_v2/simulation/SIMULATION.tex` and `manuscripts/manuscript_v2/methodology/`).
Recorded for later; nothing here has been changed in the code or the text.

Assume **the code is the source of truth** unless noted otherwise.

## 1. What the setup does (for reference)

- **DGP** (`sim_surv_v12.R:1157-1197`): `p=10` MVN AR(1) covariates (rho=0.3,
  5 active + 5 noise).
  - RCT: randomised `A ~ Bern(0.5)`, `logT = m0(X) + A*tau(X) + N(0, 0.75^2)`.
  - RWD: `A ~ Bern(expit(X1 + U))`, unobserved `U ~ Unif(0,1)`,
    `logT = m0(X) + lambda_d*dev(X) + A*tau(X) + lambda_u*A*U + eps`,
    `eps` mean-0 / unit-variance reversed Gumbel (Weibull times, skewed).
- **Censoring**: RCT exponential right-censoring ~35%; RWD interval-censored at
  8 decile visits q10..q80, ~20% right-censored past q80. Per-cell calibration
  fixed once on 2e5 draws.
- **Fits**: fusion via `FusionForest_temp` (four-forest, `source_hdp_scale`);
  single-source RCT (right-censored) and RWD (interval-censored) via
  `CausalShrinkageForest_temp`.
- **Standardisation trick**: the `*_temp` copies strip internal outcome scaling
  so the caller pre-standardises the pooled *latent* log-survival to mean-0 /
  unit-variance and lets the sampler draw sigma even for the all-censored RWD.
  Predicted CATEs are rescaled by `scale_sd` before evaluation.
- **Metrics**: in-sample CATE RMSE / bias / coverage / width / posterior
  variance, split by RCT / RWD / All; cross design sweeping `lambda_u`
  (at `lambda_d=1`) and `lambda_d` (at `lambda_u=1`).

## 2. Strengths

- DGP matches the model's structural decomposition cleanly: `dev(X)` is purely
  prognostic (cancels in the CATE, captured by `d`); the RWD treatment contrast
  carries the confounding, captured by `c`.
- Error misspecification (Gaussian RCT vs skewed Gumbel RWD) genuinely stresses
  the HDPM error model.
- Reproducible per-replication seeds, non-overlapping across cells; censoring
  calibration fixed within a cell.
- Exercises both right- and interval-censoring — the feature the paper sells.

## 3. Manuscript <-> code inconsistencies (actionable)

1. **DGP coefficients differ by 2x.** `SIMULATION.tex:24` states
   `m0 = 4x1 - 2x2x3 + x4^2`, `tau = 1/2 + 2x1 - x2^2`, `d = 2x4 - x5`.
   Code uses exactly half: `m0 = 2x1 - x2x3 + 0.5x4^2`,
   `tau = 1/2 + x1 - 0.5x2^2`, `dev = x4 - 0.5x5`. Noise SDs are fixed, so this
   is a different SNR regime, not a reparametrisation — a reader reproducing
   from the text would not match the figures.
2. **MCMC length.** `SIMULATION.tex:46` says "5000 posterior samples after
   5000 burn-in"; code uses `N_post=3000, N_burn=2000` (`:1508-1509`).
3. **`k` values.** Methodology recommends `k_sh=k_d=k_c=1, k_tau=1/2`; the
   fusion fit sets all four to `0.5` (`:1317-1320`). If described as "default
   parameters," this needs reconciling.
4. **Stale banner.** `:1500` prints "k_treat = 0.25" but the fit uses
   `k_treat = 0.5`. Cosmetic, but misleading in logs.

## 4. Methodological points

- **Confounding structure is gentle; transportability is technically
  violated.** `U` enters only the *treated* RWD arm (`lambda_u*A*U`) — it is a
  treatment-effect modifier + selection driver with **no main effect**, so the
  control-arm baseline is never confounded by `U`. Consequently the RWD's own
  causal CATE is `tau(X) + 0.5*lambda_u`, so the paper's CATE-transport
  condition (`METHODOLOGY.tex:70`) is strictly violated for `lambda_u > 0`.
  The method still recovers the RCT `tau(X)` because the four-forest
  decomposition can *always* represent the RWD-specific component as `c(X)` —
  which is the right thing to demonstrate (borrowing does not import the bias),
  but it also means this DGP **cannot exhibit a transportability failure**.
  A stronger battery would add:
  - (a) a confounder with a **main effect in both arms** (to stress the
    shared-baseline borrowing under baseline confounding);
  - (b) a regime where finite-sample borrowing could plausibly drag `tau_hat`
    toward the contaminated RWD signal.
- **Two advantages are conflated.** Single-source baselines use a single
  Gaussian likelihood, while fusion gets both information borrowing *and* the
  flexible HDPM error model against skewed RWD errors. The headline lower
  RMSE / variance mixes the two. A fusion-with-Gaussian-error ablation would
  isolate the borrowing contribution.
- **Oracle standardisation.** `scale_mu` / `scale_sd` come from the *true
  uncensored* latent log-survival (`:1225-1228`) — information a real analyst
  cannot access (the package normally estimates scale from censored data).
  Applied identically to all three estimators, so the comparison is fair, but
  every method gets a small oracle edge over the deployed package. Worth a
  caveat sentence or a robustness check estimating scale from data.
- **RWD-only gets the oracle propensity.** `fit_single_ic` appends the *true*
  propensity `e = expit(X1 + U)` (which encodes the unobserved `U`) as a
  control covariate (`:1367, :1449`). Generous to the competitor (so it does
  not undermine fusion's win), but asymmetric — fusion uses binary coding with
  no propensity.
- **In-sample evaluation.** `Xev = d$X`; test points are the training rows.
  Common in BART-CATE work, but an independent evaluation grid would make the
  generalisation claims cleaner.
- **Calibration RNG not seeded.** `calibrate()` runs before the seeded
  `foreach` (`:1474`) on ambient RNG state, so censoring rate / visit times are
  not exactly reproducible run-to-run (stable to ~3 digits at n=2e5; easy to
  fix with a seed).
- **Limited robustness axes.** Only `lambda_u` and `lambda_d` are swept, one at
  a time. Sample sizes, censoring level, `p`, and rho are fixed. The "robust to
  both nuisances" claim would be stronger with at least one sensitivity sweep.

## 5. Checked and consistent

- `n_rct=150`, `n_rwd=350`, 1000 reps, 8 visits, ~35% / ~20% censoring.
- Per-forest tree counts and base/power match the methodology
  (control 200 / treat 100 / deconf 50 / deviation 50; tau depth-penalised
  power=3/base=0.95; c strongly regularised power=3/base=0.25).
- 5 active + 5 noise covariates; Weibull (RWD) / log-normal (RCT) times.
- The deepAFT competitor experiment is **not** in this file (presumably the
  untracked `simulations/exp2_competitor/`).

## 6. Proposed experiment — efficiency vs dimensionality trade-off

Motivation: the HTE efficiency gain comes from "more observations for fewer
covariates." Pin down what happens at **fixed `n`, increasing `p`**.

Mechanism (two separable levers):
- Sample size acts on the **nuisances**: fusion pools `n_rct + n_rwd` for the
  shared baseline `m0_sh` and the error model; `tau` is identified from the RCT
  contrast only (Prop. 2). The HTE gain is indirect — a better baseline
  de-noises the residual and sharpens the RCT contrast — so it scales with how
  hard the baseline is to learn.
- Dimension `p` acts on **every** forest: BART splits uniformly over `p`, so
  noise covariates dilute splits and inflate variance for `m0`, `d`, `tau`, `c`.
- Tension: pooling helps `m0` *more* in high dim (RCT-only has only `n_rct` for
  a `p`-dim baseline), but `d` and `c` are estimated from `n_rwd` alone and
  their burden also grows with `p`.

Hypotheses:
- H1: absolute RMSE / posterior variance rise with `p` for all estimators.
- H2: fusion's *relative* variance advantage over RCT-only is preserved or grows
  over a moderate `p` range (pooling rescues the high-dim baseline).
- H3: crossover at large `p` — estimating high-dim `d`, `c` from `n_rwd` becomes
  the bottleneck and poorly-identified `c` can leak into `tau` in finite
  samples; the advantage peaks then shrinks. Locating this is the key insight.
- H4: coverage degrades with `p` for all; sparsity priors (DART / horseshoe,
  cf. `jacobs2025horseshoe`) mitigate and restore the fusion advantage.

Designs (the script makes `p` a free axis: active functions use only X[,1:5],
so raising `p_total` just appends noise covariates):
- **A — dimension sweep at fixed `n`.** `n_rct=150, n_rwd=350, lambda_d=lambda_u=1`;
  `p in {5,10,20,50,100}`, active set fixed at 5. Report fusion-vs-RCT-only
  ratios of RMSE and posterior variance, and coverage, as functions of `p`.
- **B — (n, p) frontier.** `n_rwd in {150,350,700,1400} x p in {10,50,100}`.
  Iso-RMSE contours: how much RWD sample buys back a given increase in `p`.
- **C — signal vs noise dimension.** Fix `p=50`, vary active-covariate count
  `k in {2,5,10}`. Separates noise-covariate cost from intrinsic complexity.
- **D — prior ablation at high `p`.** At `p=100`, compare uniform-split BART vs
  DART / horseshoe step-heights.

Caveat: `CausalShrinkageForest_temp` exposes `a_dirichlet_*`, but
`FusionForest_temp` does **not** — Experiment D for the fusion fit may need
four-forest backend support for a sparsity prior. Confirm before scoping D.

### High-p slowdown investigation (sim_hd_v1.R)

Symptom: at p >= 100 the sim runs far too long (no error). Diagnosis so far:
- Ruled out `calibrate()`: its `mvrnorm(2e5, p)` is ~3.4s at p=200 (measured).
- Suspected the fit's per-sweep cost: every birth/death proposal rescans all p
  variables across all leaves (`GetBirthProbability` -> `CanSplit`,
  `GetSplittableVariables`), i.e. O(num_leaves * p * depth) per tree per sweep,
  and `use_augmentation = true` is hardcoded on for all four fusion forests
  (`FusionForest4.cpp`). Could not profile the real package locally (not built
  in the scratch env).
- Isolation in progress: (1) dropped MASS, used a base-R Cholesky draw;
  (2) now using **independent covariates** (Sigma = I, `draw_X = rnorm` matrix)
  to rule out the covariate-draw machinery entirely; grid cut to p = 25, 100
  for a fast contrast.

**Remember for later:** when switching back to correlated covariates, revisit
whether to build the draw via `chol` (user flagged a reservation about `chol`).
Options then: cache the Cholesky factor per p; an AR(1) direct recursion
(x_1 ~ N(0,1); x_j = rho * x_{j-1} + sqrt(1 - rho^2) * e_j) which avoids any
p-by-p factorisation; or a banded/sparse construction.
