# Non-parametric error priors in FusionForest — simulation findings

**Setup.** 25 replicates, 3 RWD-residual scenarios, 3 priors (`gaussian` / `shared_dp` / `source_dp`). n_RCT = 100, n_RWD = 200. RCT residuals always Gaussian; RWD residuals misspecified. Metric: CATE RMSE.

## Results

| scenario | error_dist | σ (sd)        | RMSE_All (sd)      |
|----------|------------|---------------|--------------------|
| bimodal  | gaussian   | 0.720 (0.025) | 0.239 (0.050)      |
| bimodal  | shared_dp  | 0.435 (0.072) | **0.312 (0.264)** ← unstable |
| bimodal  | source_dp  | 0.313 (0.021) | **0.204 (0.047)** ← best   |
| gumbel   | gaussian   | 0.548 (0.035) | 0.219 (0.044)      |
| gumbel   | shared_dp  | 0.383 (0.030) | 0.219 (0.046)      |
| gumbel   | source_dp  | 0.369 (0.032) | 0.218 (0.045)      |
| logistic | gaussian   | 0.556 (0.023) | 0.206 (0.046)      |
| logistic | shared_dp  | 0.404 (0.021) | 0.208 (0.047)      |
| logistic | source_dp  | 0.384 (0.020) | 0.206 (0.045)      |

## Findings

1. **`source_dp` wins under shape misspecification.** Bimodal RWD noise: 15% RMSE reduction over Gaussian with the same SD across reps. σ drops 0.72 → 0.31, exactly the misspecification cost paid by Gaussian.

2. **`shared_dp` is *unstable* when sources differ in residual shape.** RMSE 0.312 with SD 0.264 (CV ≈ 0.85) — heavy-tailed across replicates. Cause: global centring couples a RCT-near-zero atom to RWD atoms at ±1; with n_RCT = 100 some reps land in mislabelled regimes. Motivates the full HDP-CDP (Stage B): shared atoms with *per-source* centring.

3. **Skew or heavy tails alone — no CATE gain.** Gumbel and logistic: DPs cut σ by 25–35% but RMSE is identical across the three priors. BART's structural flexibility already absorbs symmetric / mildly-skewed residual shape; the DP only earns its keep on *multi-modality* the means can't fit away.

## Takeaways for next steps

- Default `error_dist = "gaussian"` remains safe for most regimes; flip to `source_dp` when residual diagnostics suggest source-specific multi-modality.
- `shared_dp` should be used with caution under per-source heterogeneity. The HDP-CDP extension (shared atoms, per-source weights and centring) is the principled fix and is the next implementation milestone.
