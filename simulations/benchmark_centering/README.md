# Benchmark: centred-mixture sampler, 1.1.0 vs 1.1.1

One script, `benchmark_centering.R`.  It compares the FusionForests sampler
before and after the parameter-expansion fix of the centred error mixture
(referee 1, comment 1), paired on identical data, in three settings:

* `exp1`   -- experiment 1 at lambda_d = lambda_u = 1.
* `stress` -- skewed RCT error, wide asymmetric bimodal RWD error, heavy and
  coarse censoring; the fitted mixture means are far from zero, so the fix is
  most visible.
* `gauss`  -- both errors Gaussian; the two versions should coincide.

Per replicate it records the paper's CATE metrics, the baseline levels
against the truth per source, the Wasserstein-1 distance of the fitted error
law per source, the mean of the stored mixture, effective sample sizes,
per-source scales and runtime, plus the per-observation `tau` and baseline
estimates for same-data correlations.

```bash
Rscript benchmark_centering.R --smoke   # builds lib/old + lib/new, 4 reps at N = 200/200
Rscript benchmark_centering.R           # 200 reps at N = 2000/2000 on 4 cores, then summary.csv
```

For 1.1.1 vs 1.1.2 (the per-source scale in the data augmentation), keep the
earlier results and libraries apart with the environment overrides:

```bash
OLD_REF=ba678e6 BENCH_KTREAT=1 BENCH_OUT=$PWD/results_pooled_sigma \
LIB_OLD=$PWD/lib/v111 LIB_NEW=$PWD/lib/v112 Rscript benchmark_centering.R
```

This run also fits a `control` setting, the `stress` data under
`error_dist = "gaussian"`. That model has one scale, so it must come out
identical in the two versions, and it gives the Gaussian's W1 on the same data.

Sub-commands `setup`, `run`, `summarize` run the steps on their own; see the
script header for the environment overrides (`BENCH_REPS=100` halves the
runtime). Workers run under `nice -n 19`, so the machine stays usable.  Each worker is a separate R
process, so the two package versions never share a session.  Needs Rcpp
(>= 1.0.11) and ShrinkageTrees in the default library.  `lib/` and
`results*/` are git-ignored.
