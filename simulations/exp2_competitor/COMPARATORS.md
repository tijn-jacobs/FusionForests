# Comparator methods — simulation experiment 2

Status overview for the machine-learning comparison in `manuscripts/manuscript_v2/simulation/SIMULATION.tex`,
experiment 2. Last updated 29 July 2026.

**Scope.** Experiment 2 compares the Bayesian fusion forest with flexible machine-learning
alternatives on **right-censored data only**. Both sources are right-censored at circa 35%,
so every method sees identical data and the comparison isolates the method rather than the
censoring design. Interval censoring belongs to experiments 1 and 3.

---

## 1. The methods

| # | Method | Reference | Software | Censoring |
|---|--------|-----------|----------|-----------|
| 1 | Bayesian fusion forest | this paper | `FusionForests` | right + interval |
| 2 | AFT Bayesian causal forest, RCT-only | Jacobs (2026) | `ShrinkageTrees` | right + interval |
| 3 | AFT Bayesian causal forest, RWD-only | Jacobs (2026) | `ShrinkageTrees` | right + interval |
| 4 | deepAFT | Norman, Chen & Li (2024) | `dnn` (CRAN) | right |
| 5 | XGBoost-AFT | Barnwal, Cho & Hocking, *JCGS* | `xgboost` (CRAN) | right + **interval** |
| 6 | BJ-ELM | Kong & Zhang, *Biometrical Journal* | supplementary code, vendored | right |
| 7 | BJ-trees | Wang & Wang (2010), *SAGMB* 9(1) art. 24 | `bujar` (CRAN) | right |

Methods 4–7 are each fitted as four estimators: S-learner and T-learner
\citep{kunzel2019metalearners}, on the trial alone and on a naive pool of both sources.
That is twelve competitor rows against one fusion row and two single-source rows.

### Notes per method

**deepAFT.** The CRAN package is called `dnn`, not `deepAFT`; the fitted object carries
class `deepAFT`. The manuscript should name the package, since "deepAFT" alone is
ambiguous. Norman et al. offer three censoring strategies (one Buckley-James style
imputation, two IPCW variants); the current run uses `method = "BuckleyJames"` inside
`hyperTuning`. Which one was used should be stated, or a second variant reported.

**XGBoost-AFT.** Gradient boosting over trees with the AFT likelihood as the loss
(`objective = "survival:aft"`) instead of squared error. It is the only comparator that
accepts interval-censored labels, supplied as `label_lower_bound` / `label_upper_bound`
ranges. That capability is *not* exercised in experiment 2, but it does mean the
manuscript sentence "To our knowledge none supports interval censoring" is now false and
must be rewritten. The weaker claim — that no alternative combines a causal target, data
fusion *and* interval censoring — still holds.

**BJ-ELM.** Buckley-James boosting over extreme learning machines. Not part of `bujar`:
version 0.2-11 implements `linear.regression`, `pspline`, `tree`, `enet`, `enet2`, `mars`
and MCP/SCAD, but has **no `elm` learner**. The implementation is the authors'
supplementary material, vendored under `bimj2456-sup-0001-suppmat/`.

In Kong & Zhang's own benchmark, random survival forests appear as a *separate* competitor
(`rfsrc`) and as an imputation device, **not** as a Buckley-James base learner. There is
therefore no "BJ-RF" to run.

**BJ-trees.** `bujar` with `learner = "tree"`, one of the three base learners Wang & Wang
benchmark. Boosted trees against a Bayesian tree ensemble puts both methods in the same
function class, so the experiment isolates the fusion and the uncertainty quantification
rather than the base learner. `degree = 2` is essential: the default of 1 is an additive
model, and the DGP baseline contains `x2*x3`, which an additive fit cannot represent —
leaving it at 1 would make bujar lose on a restriction we imposed ourselves.

---

## 2. Implementation status

| Method | Script | Design | Bootstrap | Run |
|--------|--------|--------|-----------|-----|
| Fusion + single-source | `sim_surv_v12_bff.R` | — | posterior | M = 1000, 3×3 grid |
| deepAFT | `sim_surv_v12_deepaft.R` | RCT/pool × S/T | B = 100 | M = 1000, 3×3 grid |
| XGBoost-AFT | `sim_surv_v12_xgbaft.R` | RCT/pool × S/T | B = 100 | **running locally**, M = 1000 |
| BJ-ELM | `sim_surv_v12_bjelm.R` | RCT/pool × S/T | B = 10 | M = 10, cell (1,1) — pilot only |
| BJ-trees | `sim_surv_v12_bjtree.R` | RCT/pool × S/T | B = 100 | M = 1000, cell (1,1) |

All scripts share the DGP, calibration, metrics (`eval_cate`, `make_rows`), stratified
bootstrap and output columns of `sim_surv_v12_deepaft.R`, so their rows row-bind directly.
Each writes to `TMPDIR` when that is set (HPC) and to this directory otherwise.

**Still to do:** the two Buckley-James methods need full runs, and both are slow enough
that they belong on the HPC. See §5.

### Prototypes

`prototypes/` holds one-replication smoke tests used to get each method fitting before the
full simulation was written: `dgp_prototype.R` (shared DGP with a `censoring` switch),
`proto_xgboost_aft.R`, `proto_bjelm.R`, `proto_bujar.R`. None of these produce paper
results.

### Implementation details worth remembering

- **XGBoost `predict()` needs `outputmargin = TRUE`.** For `survival:aft` the default
  returns `exp(margin)`, a predicted survival *time*. The CATE is a difference on the log
  scale, so without the flag you silently get a difference of times instead of a log-ratio.
- **The S-learner must fit once and predict twice.** Fitting separately at A = 1 and A = 0
  doubles the cost and, for ELM, leaks initialisation noise into the contrast: the
  hidden-layer weights are drawn from U(−1, 1) at every fit.
- **BJ_ELM.R opens with a hardcoded `source()` path** that does not exist here. The scripts
  load `ELM_Boosting.R` directly and evaluate `BJ_ELM.R` with that one line filtered out,
  leaving the authors' code untouched.
- **`emplik` must be attached** — `BJ_ELM.R` calls the `cumsumsurv` C routine through `.C()`
  without a `PACKAGE` argument.
- **`nthread = 1` for XGBoost** under `doParallel`, or its internal threads fight the
  workers for cores.
- **No seeds** in the prototypes, so repeated runs show the run-to-run spread.

---

## 3. Results so far

At λ_d = λ_u = 1, population "All". Every method uses M = 1000 replications and, where applicable, B = 100
bootstrap resamples.

| Method | RMSE | Bias | Coverage | Width |
|--------|------|------|----------|-------|
| **Bayesian fusion forest** | **0.68** | −0.03 | **0.96** | 2.82 |
| deepAFT RCT-S | 1.11 | −0.19 | 0.38 | 0.95 |
| deepAFT pooled-S | 1.11 | 0.11 | 0.44 | 0.99 |
| deepAFT RCT-T | 1.25 | −0.25 | 0.91 | 3.73 |
| deepAFT pooled-T | 1.35 | 0.14 | 0.99 | 4.92 |
| XGBoost-AFT RCT-S | 1.12 | 0.05 | 0.44 | 1.10 |
| XGBoost-AFT pooled-S | 1.08 | 0.46 | 0.49 | 1.21 |
| XGBoost-AFT pooled-T | 2.94 | 0.79 | 0.91 | 7.97 |
| XGBoost-AFT RCT-T | 3.55 | 0.56 | 0.95 | 10.29 |
| BJ-ELM RCT-S | 1.29 | −0.06 | 0.35 | 1.10 |
| BJ-ELM pooled-S | 1.29 | 0.29 | 0.23 | 0.70 |
| BJ-ELM pooled-T | 1.15 | 0.44 | 0.77 | 2.46 |
| BJ-ELM RCT-T | 1.59 | 0.26 | 0.84 | 4.58 |

### What the three comparators agree on

**S-learners undercover badly.** Coverage 0.23–0.49 against a nominal 0.95, with the
narrowest intervals of any method. Narrow and wrong, in all three method families.

**T-learners buy coverage with width.** Coverage 0.77–0.99, but intervals two to ten times
wider than the fusion forest's, on a CATE whose standard deviation is circa 1.2.

**Neither configuration is selectable in advance.** The two learners give opposite answers
and no principled rule picks between them. That is the substance of the "no principled rule
selects among these configurations" argument in §4.

The consistency across three unrelated method families is the strongest result here: the
failure is a property of meta-learner plus bootstrap, not an artefact of one network.

### Differences between the comparators

**Naive pooling transfers confounding bias.** XGBoost-AFT RCT-S is nearly unbiased (0.05)
while pooled-S has bias 0.46. BJ-ELM shows the same move (−0.06 → 0.29). Pooling without a
confounding function imports the bias it should have removed.

**BJ-ELM is the strongest competitor.** Its T-learners reach RMSE 1.15–1.59 with widths
2.5–4.6, against XGBoost's 2.94–3.55 and 8.0–10.3. If only one booster survives into the
paper, BJ-ELM is the one to keep: beating the stronger opponent answers the "easy contest"
objection better than adding a weak one.

---

## 4. Excluded, and why

| Method | Reason |
|--------|--------|
| Ye et al. (2025), integrative AFT fusion | No implementation released. Their treatment effect and confounding function are both linear, so a comparison against a nonlinear DGP would test the implementation rather than the method. **Closest existing method — the exclusion must be justified in §3, which it currently is not.** |
| KAN-AFT (Jose et al., 2025) | No implementation released; would need a Python build on `pykan`. Unrefereed preprint. |
| Survival SVM | Ranking or ε-insensitive loss, not an AFT model; no conditional mean on the time scale. |
| Causal survival forest (`grf`), `survlearners` | RMST or survival-probability scale; different estimand. |
| Cox-based methods, CoxKAN | Hazard scale; different estimand. |
| `bujar` with pspline / linear / mars / enet learners | `tree` is kept (same function class as the fusion forest); the others would add variants without adding an argument. |
| Random survival forest | Not a Buckley-James variant; a separate competitor in Kong & Zhang. Could be added as a standalone method if a hazard-free comparison is wanted. |

---

## 5. Running the Buckley-James methods on the HPC

Both are far slower than XGBoost, and BJ-trees is the worst of the set. One `bujar` fit
runs a Buckley-James loop of up to `iter.bj` iterations, each fitting `mstop` boosted
trees, so a single fit is order 10³ tree fits. BJ-ELM is lighter but still up to 30 BJ
iterations × `nbase` ELMs per fit.

**The cost multiplies fast.** Each replication needs 6 fits — S = 1 and T = 2, for each of
the RCT and pooled designs — and the bootstrap multiplies that by `1 + B_boot`. At
B = 100 that is 606 fits per replication, and at M = 1000 it is 606,000 fits per grid cell.
At 5 seconds a fit that is 840 core-hours for one cell.

**Measure before committing.** `sim_surv_v12_bjtree.R` has a `TIMING <- TRUE` switch that
fits one model, prints the elapsed time and projects the total for the configured M,
B_boot, grid and core count, then exits. Run that first.

**Levers, in the order worth pulling:**

1. **One grid cell.** The main text reports experiment 2 at λ_d = λ_u = 1 only; the grid
   goes to the supplementary materials. Dropping to one cell is a 9× saving, and the
   supplementary grid can stay deepAFT-only with a sentence saying so.
2. **B_boot.** The largest multiplier after the grid. B = 50 halves the cost and still
   gives a usable percentile interval; the paper would need to say the bootstrap size
   differs per method.
3. **mstop.** Linear in the fit cost. The default of 50 is generous.
4. **M.** Last resort — it is what makes the Monte Carlo error small, and the other
   methods use 1000.

### Package installation

`hpc_install_packages.R` installs all 15 packages into a user library and then **load
tests** every one, plus the vendored BJ-ELM source, including a live call to `cumsumsurv()`
to confirm emplik's C routine resolves. Installing and loading are different things on a
cluster: a package can build against the wrong system library and only fail at
`dyn.load()`.

```
Rscript simulations/exp2_competitor/hpc_install_packages.R
```

Run it from the repository root, against the same R module you will run the simulations
with — packages are version-specific. The heavy ones are `rms` and `caret` (large compiled
dependency trees, needed only by BJ-ELM), `xgboost` (C++14 and OpenMP) and `scorecard`
(pulls in much of the tidyverse).

The vendored Kong & Zhang code under `bimj2456-sup-0001-suppmat/` must be copied to the
cluster with the rest of the repository; it is not installable.

---

## 6. Open items

1. Finish the XGBoost-AFT run (M = 1000, B = 100), then run both Buckley-James methods.
2. Decide the grid: one cell for the main text, or the full 3×3 to match deepAFT.
3. Rewrite the "none supports interval censoring" sentence in §3 — XGBoost-AFT falsifies it.
4. Name `dnn` as the package for deepAFT, and state which censoring strategy is used.
5. Add one or two sentences in §3 on why Ye et al. is not a comparator.
6. Fill the `% TODO` in `SIMULATION.tex` naming the individual comparator methods.
7. Decide whether the design summary table (method × censoring × estimand scale ×
   uncertainty) goes in the supplementary materials with a pointer from §3.1.
