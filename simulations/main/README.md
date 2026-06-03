# Main simulation study — data fusion for heterogeneous survival effects

Code and results for the main simulation of the paper. The written design lives
in [`notes/simulation/SIMULATION.tex`](../../notes/simulation/SIMULATION.tex);
this folder holds the R implementation, the inspection output, and (later) the
HPC driver and Monte Carlo results.

## Goal

Quantify the efficiency gain from combining a randomised trial (RCT) with a
real-world data source (RWD) for estimating the conditional average treatment
effect (CATE). The combination should lower CATE estimation error relative to
the trial alone, without acquiring bias when the RWD is confounded and without
losing credible-interval coverage. Two features of the data-generating process
are varied: the strength of unmeasured confounding in the RWD, and the degree
of between-study heterogeneity (baseline drift plus a source-specific residual
law).

## Estimand and metrics

The estimand is the CATE on the log-time scale,
`tau(x) = E[log T(1) - log T(0) | X = x]`, equivalently the log acceleration
factor. The treatment effect is assumed constant over time, so this single
estimand captures the contrast at every horizon. Evaluation is in-sample on the
combined covariates. Per replication we record RMSE, signed bias, pointwise 95%
credible-interval coverage, and mean interval width, averaged over the
evaluation points.

## Data-generating process

Covariates `X ~ U(-1/2, 1/2)^10` in both sources (5 active, 5 noise). An
unmeasured confounder `U ~ N(0,1)` is independent of `X`. The log-survival time
follows the AFT structural model

```
log T = m0_sh(X) + (1-S) * lambda_d * g(X) + beta_U * U
        + A * { tau(X) + gamma_U * U } + eps
m0_sh(X) = sin(pi X1) + X2 X3 + (X4^2 - 1/12)
g(X)     = X1 - 0.5 X5
tau(X)   = 1 + 0.5 X1 + (1/3) 1{X2 > 0}
```

Trial treatment is randomised; RWD treatment is `Bernoulli(expit(alpha_0 + X1 +
X2 + alpha_U U))` with `alpha_0` calibrated to ~50% prevalence.

- **Confounding scenarios** `(alpha_U, beta_U, gamma_U)`:
  S0 `(0,1,0)` none, S1 `(0.5,1,0)` mild, S2 `(1.5,1.5,0)` strong,
  S3 `(1,1,0.5)` effect modification.
- **Heterogeneity regimes**: *matched* (`lambda_d=0`, `eps ~ N(0,1)` in both
  sources); *heterogeneous* (`lambda_d=0.5`; RCT `eps ~ N(0,0.6^2)`, log-normal
  event times; RWD `eps` a recentred smallest-extreme-value variate, Weibull
  event times, SD 1).
- **Censoring**: RCT right-censored (`Exp` rate tuned to ~35%) and
  administratively censored at a short horizon `t_RCT = vmax/3`; RWD
  interval-censored on 10 equally spaced inspection times over `[0, vmax]`.
  `vmax` and the `Exp` rate are calibrated once on a pre-simulation.

## Estimators compared

All three run through the `FusionForest` AFT-BART backend, so the comparison
isolates the fusion design, not the implementation. Forests use 200 (prognostic)
/ 100 (treatment) / 50 (deviation) / 50 (confounding) trees and a uniform
leaf-scale `k = 0.5`.

1. **Bayesian fusion forests** — four-forest, both sources, confounding function
   `c`, source-specific HDP residual prior (`error_dist = "source_hdp_scale"`).
2. **RCT-only** — single-source AFT causal forest on the trial.
3. **Naive pooling** — sources pooled, confounding forest neutralised, one shared
   residual.

RCT-only and naive pooling use the established codebase idiom for a two-forest
BCF out of `FusionForest`: one dummy `source = 0` row plus
`number_of_trees_deconf = 1` makes `c` inert (`error_dist = "gaussian"`).

## Files

| File | Purpose |
|------|---------|
| [`simulation_main_single.R`](simulation_main_single.R) | One simulated dataset, all three estimators, metrics + diagnostic plots. The development/inspection tool. |
| `single_run_diagnostics.pdf` | Plots from the last single run (regenerated each run). |
| `single_run_results.rds`     | Metrics + calibration from the last single run. |
| `simulation_main_hpc.R`      | *Pending* — M = 1000 Monte Carlo driver for the HPC (4 scenarios x 2 regimes), modelled on `../prior/simulation_censored_hpc.R`. |

## Running

```bash
# one inspection run: Rscript simulation_main_single.R [N_post] [N_burn] [scenario] [regime]
Rscript simulations/main/simulation_main_single.R                       # defaults: 2000 2000 S1 heterogeneous
Rscript simulations/main/simulation_main_single.R 2000 2000 S2 heterogeneous
```

A single fit takes ~10–25 s at `N = 2000 + 2000` on one core (n = 500).

## Single-run results (indicative)

CATE metrics from **one** replication per cell (`n1 = 150`, `n0 = 350`,
`N = 2000 + 2000`, `seed = 1`), in-sample on the 500 combined covariates,
log-AF scale.

| Scenario | Regime | Method | RMSE | Bias | Coverage | Width |
|----------|--------|--------|------|------|----------|-------|
| S0 (none)   | matched | Fusion     | 0.271 | 0.155 | 1.00 | 1.84 |
| S0 (none)   | matched | RCT-only   | 0.254 | 0.082 | 1.00 | 1.90 |
| S0 (none)   | matched | Naive pool | 0.273 | 0.175 | 1.00 | 1.82 |
| S1 (mild)   | hetero. | Fusion     | 0.265 | 0.141 | 1.00 | 1.65 |
| S1 (mild)   | hetero. | RCT-only   | 0.253 | 0.069 | 1.00 | 1.73 |
| S1 (mild)   | hetero. | Naive pool | 0.448 | 0.396 | 0.99 | 1.67 |
| S2 (strong) | hetero. | Fusion     | 0.324 | 0.240 | 1.00 | 2.07 |
| S2 (strong) | hetero. | RCT-only   | 0.241 | 0.031 | 1.00 | 2.11 |
| S2 (strong) | hetero. | Naive pool | 0.590 | 0.548 | 1.00 | 2.01 |
| S3 (eff.mod)| hetero. | Fusion     | 0.259 | 0.141 | 1.00 | 1.77 |
| S3 (eff.mod)| hetero. | RCT-only   | 0.258 | -0.084| 1.00 | 1.79 |
| S3 (eff.mod)| hetero. | Naive pool | 0.562 | 0.520 | 0.98 | 1.79 |

**Reading these numbers.** They validate the pipeline and the qualitative DGP
behaviour, not the headline comparison.

- Naive pooling is biased in proportion to the confounding strength
  (0.18 → 0.40 → 0.55 → 0.52 across S0–S3). Pooling without a confounding
  correction transmits the RWD bias straight into `tau`.
- RCT-only stays near-unbiased and stable. Randomisation protects it.
- Coverage is conservative (≈ 1) and intervals are wide. This matches the
  over-coverage of the variance calibration seen in the prior-specification
  study.
- With a **single** replication the bias column is one fitted-surface offset,
  correlated across the 500 evaluation points. Fusion-vs-RCT-only *differences*
  of ~0.02–0.08 in RMSE and bias are within Monte Carlo noise. The headline
  question — does fusion beat RCT-only, and does it stay unbiased under
  confounding — is answered only by averaging over many replications.

## Status and next steps

- [x] Single-run inspection script, runs end to end (interval censoring, HDP
  residual, dummy-row two-forest idiom all verified).
- [ ] HPC driver for M = 1000 across the 4 x 2 grid (on hold).
- [ ] Aggregate and report the CATE block once the Monte Carlo runs finish.

A recurring signal already visible at one replication: under **strong**
confounding (S2) the small RCT (`n1 = 150`) does not fully disambiguate `tau`
from `c`, so some bias remains in the fusion estimate. This is the
small-trial-anchor tension. Whether it persists on average, and how the
fusion–RCT gap moves with confounding strength and sample size, is exactly what
the full study will quantify.
