# Feature notes from writing the vignette

Temporary file (excluded from the build). Gaps and issues found while
illustrating the package on `breast_fusion`, roughly by priority.

## 1. Check first (possible bugs or wrong docs)

1. **Survival difference for `target_source = "rwd"` is not causal.**
   `.linear_predictor()` uses `tau + c` for the OS target, so
   `fusion_estimand(fit, "SD", target_source = "rwd")` returns the
   observational contrast S(t | A = 1, x, S = 0) - S(t | A = 0, x, S = 0),
   bias included. The help page calls these causal estimands. Either drop
   `c` for the OS target (and state the transport assumption on the
   baseline) or document that only `"rct"` gives a causal difference. The
   vignette uses `"rct"` only for this reason.
2. **Wrong reference in `?fusion_projection`.** It cites "Bayesian
   inference with posterior projection", JCGS 29(4), 798-808 (2020). The
   paper's bib has Woody, Carvalho and Murray (2021), "Model interpretation
   through lower-dimensional posterior summarization", JCGS 30(1), 144-161.
3. **`summary()` reports a pooled residual SD** for the default
   `"source_hdp_scale"` error model. It should report sigma_s per source
   (`fit$dp_sigma_g`).
4. **`timescale = "time"` stores exp(posterior mean)**, not the posterior
   mean of exp(.), in `*_predictions_treat`/`_deconf`/`_deviation`.
   `summary()` labels these "survival time ratios". Worth stating, or
   storing the posterior mean of the AF instead.

## 2. Estimands and summaries

5. **Target population.** `fusion_estimand()` averages over all test rows.
   Targeting one source needs `population_average = FALSE` plus a manual
   `rowMeans()`, which loses the Bayesian bootstrap. Add a `rows` (or
   `target_weights`) argument. The paper's hierarchical Bayesian bootstrap
   with pi ~ Dirichlet(alpha) over the two sources (Web Appendix,
   eq. ate-bb) is not implemented.
6. **`which = "train"` in `fusion_estimand()`**, as `fusion_projection()`
   already has. The vignette passes the training set again as test set,
   which doubles the stored draws.
7. **Vectorised `time`** for the survival difference. The vignette loops
   over 14 time points. Per-arm survival curves S_a(t | x) would also
   allow a model check against Kaplan-Meier curves.
8. **Posterior summary helper.** Every estimand needs the same
   mean / 2.5% / 97.5% code. A `summary()` method for estimand draws would
   remove it.
9. **Component accessor.** Users must know the names
   `test_predictions_sample_{control,treat,deconf,deviation}` and their
   scale conventions. A `fusion_components(fit, which, scale)` returning
   m0, d, tau and c on the log scale would help.
10. **Error distribution.** No function evaluates the posterior error
    density or CDF per source. The group order of `dp_sigma_g` and
    `dp_mix_prop` (1 = OS, 2 = RCT) is not documented under `@return`.

## 3. Interface

11. **`predict()` method** for new covariates. The test set is now fixed
    at fitting time.
12. **Default `X_train_treat = X_train_control`** (and the same for the
    test matrices). Most analyses pass the same matrix twice. A formula /
    data-frame interface would go further.
13. **Memory.** `store_posterior_sample = TRUE` keeps train and test draws
    for all four forests. With n = 3668 and 2000 draws that is about
    8 x 59 MB. Allow storing test draws only, or selected forests.

## 4. Diagnostics

14. **Convergence.** No multiple chains (`CausalShrinkageForest()` has
    `n_chains`), no export to `coda`/`posterior`, no `plot()` method. The
    vignette builds a trace plot by hand.
15. **Overlap.** No helper for positivity between arms (OS propensity) or
    between sources (source membership), although the paper discusses
    both.

## 5. Single-source comparators

16. `fusion_estimand()` and `fusion_projection()` do not accept
    `CausalShrinkageForest()` fits. The vignette exponentiates
    `test_predictions_sample_treat` by hand. Shared post-processing would
    make the RCT-only comparison one line.
17. Matching priors is fiddly: `FusionForest()` uses `k / (2 sqrt(J))`
    while `CausalShrinkageForest()` takes `local_hp` directly.

## 6. Documentation

18. Add `inst/CITATION` for the paper; the vignette now cites it as
    "Manuscript".
19. Link the vignette from the README; consider a pkgdown site.
