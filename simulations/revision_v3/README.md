# Simulations, revision round (manuscript_v3)

The simulations behind the revised simulation study. The first four experiments
and the folder layout follow `../submission_v2/`, from which those scripts are
derived. Experiments 5 and 6 are new.

| Folder | Experiment | HPC script(s) | Local analysis |
|--------|------------|---------------|----------------|
| `exp1_confounding_heterogeneity/` | Confounding & between-source heterogeneity sweeps, plus a covariate-dependent confounding sweep | `sim_surv_v13.R` | `analyse_surv_v13.R` |
| `exp2_competitor/` | Ye et al. (2025) and machine-learning competitors on the right-censored design | `sim_surv_v13_{bff,ye,deepaft,xgbaft,bjelm,bjtree}.R` | `analyse_competitors_v13.R` |
| `exp3_covariate_dimension/` | Growing covariate dimension `p` | `sim_hd_v2{a,b,c,d}.R` | `analyse_hd_v2.R` |
| `exp4_error_distribution/` | Residual priors across six error settings and a small trial | `sim_err_v3.R` | `analyse_err_v3.R` |
| `exp5_prior_sensitivity/` | Sensitivity to the prior settings, one at a time around the default | `sim_prior_v1.R` | `analyse_prior_v1.R` |
| `exp6_assumption_violations/` | Scenarios in which one assumption fails, for Web Appendix D.6 | `sim_viol_v1.R` | `analyse_viol_v1.R` |
| `exp7_interval_handling/` | The cost of ignoring the interval bounds, for Web Appendix D.7 | `sim_ic_v1.R` | `analyse_ic_v1.R` |
| `pilot_ktau/` | Check of `k_tau` on the final package, before the full runs; not in the paper | `sim_ktau_v1.R` | `analyse_ktau_v1.R` |

## Folder layout and workflow

Each experiment folder holds only R scripts at its top level. The HPC
script(s) compute the replications and end at `saveRDS()` of
`<script>_output.rds`; they do no summaries or plotting. Download the outputs
into the folder's `results/`, then run the one local analysis script, from the
repository root, e.g.

```bash
Rscript simulations/revision_v3/exp1_confounding_heterogeneity/analyse_surv_v13.R
```

Each analysis script reads only the stored output (base R) and writes

* figures to `results/figures/`. Until the results are final the paper reads
  them from there through `\graphicspath` (exp1: `sim_vary_lambda_{u,d}.pdf`,
  `sim_vary_lambda_u_hetero.pdf`; exp3: `sim_hd_metrics_vs_p.pdf`,
  `sim_hd_ratio_vs_p.pdf`); they are copied to one figures folder at the end;
* summary CSVs and the LaTeX of the supplementary tables to
  `results/tables/` (exp2: `competitors_table.tex`, `ye_table.tex`,
  `deepaft_table.tex`; exp4: `sim_err_table.tex`; exp5: `prior_table.tex`;
  exp6: `viol_table.tex`, `viol_table_detail.tex`).
  The supplement types these tables in by hand, so paste them over.

`analyse_err_v3.R` has one optional part, `run_density <- TRUE`, which refits
the four residual priors on one dataset for an illustrative density figure;
it needs FusionForests and loads the DGP from `sim_err_v3.R` without running
the HPC part.

## What changed relative to the submission

1. **Confounder design (referee 1, comments 2 and 4).** The unmeasured
   confounder `U` enters the RWD outcome in both arms, so
   `T(1) - T(0) = tau(X)` for every unit and the CATE is common to the two
   sources (Assumption 4 holds). In the RWD, `A ~ Bernoulli{expit(X1)}` and
   `U | A, X ~ N(A delta(X), 1)`. By Bayes' rule
   `logit P(A = 1 | X, U) = X1 + delta(X) U - delta(X)^2 / 2`, so `U` drives
   treatment. The residual of `log T` given `(X, A)` has a law free of `(X, A)`,
   so Assumption 6 holds exactly. The confounding function is
   `c(X) = lambda_u delta(X)` and the RWD baseline is `m0(X) + dev(X)`. The
   constants are set once per script: `delta_u = 0.84` and, for experiment 1's
   third slice `design = "heterogeneous"`, `delta_het = c(0.78, 0.31)`, i.e.
   `delta(X) = 0.78 + 0.31 X2`, with `c(X)` from `0.27 lambda_u` to
   `1.29 lambda_u` (5th to 95th percentile; figure
   `sim_vary_lambda_u_hetero.pdf`). These match the mean and spread of the
   previous logistic design, which now serves as a violation of Assumption 6
   in experiment 6. In the trial, treatment is randomized, so `U` would only be
   part of the error; it is left out of the trial outcome to keep the
   trial-only baseline fixed along the `lambda_u` sweep. Experiments 2, 3 and 5
   carry the same `U` design; experiment 4 has no confounding and is unchanged.
2. **Fits use the exported functions.** `FusionForest()` and
   `CausalShrinkageForest()` are called directly. The fusion forest runs with
   `k = 1` for every forest, including `k_tau = 1` (set explicitly; chosen by
   a pilot), on `sigma_h = k / (2 sqrt J)`.
   The submission scripts' frozen copy used `sigma_h = k / sqrt J` with all
   `k = 0.5`, i.e. `k_tau = 1` on the paper's scale, and skipped the package's
   outcome standardization; both are gone. The single-source comparators keep
   their submission settings (`local_hp = 1 / sqrt(J)`, `beta = 2`).
3. **RWD-only comparator without the true propensity.** The submission scripts
   handed the RWD-only causal forest the true propensity score, which is a
   function of the unmeasured `U`. It now sees `X` only.
4. **Experiment 4** (referee 1, comment 7; associate editor, comment 1):
   * adds `source_dp_scale`, the independent per-source mixture, as the
     comparator that separates source-specific residual shapes from
     hierarchical sharing;
   * adds setting 6, where the two error laws share no components (RCT
     unimodal and skewed around zero, RWD bimodal at about +-1.7), to
     measure the cost of the HDP link when it is wrong;
   * reruns settings 5 (shared components) and 6 with a small trial,
     `n_rct = 50`, where sharing should pay off or cost the most;
   * records the error-law recovery per source (`w1_rct`, `w1_rwd`: the
     Wasserstein-1 distance of the fitted to the true error CDF), the direct
     measure of what the error prior does; the CATE barely depends on it.
   Eight cells in total (settings 1-6 at `n_rct = 150`, 5 and 6 at 50).
5. Output files are `sim_surv_v13*`, `sim_hd_v2*`, `sim_err_v3*`.

## Still to add

* The package's preliminary standardization ignores interval-censoring
  bounds (`censored_info()`); decide whether to fix it before these runs.

## Fusion competitor in experiment 2

`exp2_competitor/sim_surv_v13_ye.R` reimplements the integrative estimator of
Ye, Yang, Wang and Liu (2025), which has no public code: the same AFT model,
estimand and confounding function as the fusion forest, fitted by a
Stute-weighted penalized R-loss with a linear (as published) and a quadratic
basis. Its header lists every implementation choice. The integrative
R-learner of Wu and Yang (2022) is not run: it is built for uncensored
outcomes, and Ye et al. is its censoring-adjusted counterpart. Mao et al.
(2025) is not run either: it targets the conditional RMST, a different
estimand.

## Before the full runs

`smoke_test.R` runs every fusion script and its analysis at tiny settings (M = 2,
50/50 draws) in a temporary folder, without editing the scripts or touching
`results/`. Run it from the repository root; `--all` adds the competitors and
`--dry-run` only shows the edits. The fit functions catch errors, so it also
reports methods whose fits all failed.

`pilot_ktau/sim_ktau_v1.R` varies `k_tau` over 0.75, 1, 1.25, 1.5, 2 in the two
cells of experiment 5, at M = 200 and 2000/2000 draws (2000 fits). The decision
rule is in `analyse_ktau_v1.R`, which also checks that `sigma_hat` is near 1:
a value near 1.6 means the HPC runs the old package. Decided 2026-11-02 after the pilot on package 1.1.2 and the final design: k_tau = 1. At M = 200, 2000/2000 draws, it gave RMSE 0.713 and 0.756, bias 0.003 and 0.011, coverage 0.958 and 0.968 (uniform lambda_u = 1, covariate-dependent lambda_u = 2). The pre-set rule flagged k_tau = 1.25 in the uniform cell (RMSE lower by 0.023, SE 0.002), but it was worse in the other cell (by 0.009, SE 0.003), with more bias and wider intervals, so k_tau = 1 was kept. Monitor coverage, which is still somewhat above nominal, in the full runs.

## Assumption violations (experiment 6)

`exp6_assumption_violations/sim_viol_v1.R` (referee 1, comments 3, 6 and 7;
referee 2, comments 4 and 5). Starts from experiment 1 at `lambda_d = lambda_u
= 1` with `delta = 0.84` and changes one feature at a time, in 15 scenarios:
a reference; the error law, Assumption 6 (the previous logistic design at
`lambda_u = 1, 2`, a treated-arm error scale of 1.5 or 2, a covariate-dependent
error scale); a short, administratively censored trial, 75% censored, with and
without the treated-arm scale; transportability, Assumption 4 (`tau(X) + eta(X)`
in the RWD, `eta(X) = kappa (1 + X2) / 2`, `kappa = 0.5, 1`, with `tau-hat` also
scored against `tau + eta` and `c-hat` against `eta + lambda_u delta`); overlap
(trial restricted to `|X1| <= 1` or `X1 <= 0`, RWD rows scored inside and outside
the trial support; the one-sided scenario is run but not reported, see
`analyse_viol_v1.R`); informative censoring, Assumption 5 (censoring and dropout
depend on the outcome residual, `kappa = 0` as reference, `kappa = 1` in both
arms and in the treated arm only). Fusion, RCT-only and RWD-only fits as in
experiment 1. M = 1000, 5000/5000; 45,000 fits.

## Interval handling (experiment 7)

`exp7_interval_handling/sim_ic_v1.R` (referee 2, comment 1). On the same datasets
at `lambda_d = lambda_u = 1`, the RWD records enter the fusion forest through the
interval bounds, right-censored at the last visit, as a midpoint event, or with
the true event time; a trial-only fit is the reference. Eight visits and three
visits. Also records effective sample sizes. M = 1000, 5000/5000; 10,000 fits.

## Prior sensitivity (experiment 5)

`exp5_prior_sensitivity/sim_prior_v1.R` (referee 1, comment 4; associate
editor, comment 2). Nineteen prior configurations, one setting changed at a
time around the default (`k_tau`, `beta_tau`, `k_d`, `k_c`, `(alpha_c,
beta_c)`, `k_m0`, error-atom scale), on paired data in two cells: uniform
confounding at `lambda_u = 1` and covariate-dependent confounding at
`lambda_u = 2`. Besides the CATE it scores `c(x)` and `d(x)` against their true
values and the average treatment effect. Two further configurations give `tau`
or `c` the baseline's prior, (0.95, 2) with 200 trees, so no preference for
homogeneity (associate editor, comment 2). The default uses `k_tau = 1`, checked
by the pilot in `pilot_ktau/`. M = 1000, 5000/5000; 38,000 fits.
