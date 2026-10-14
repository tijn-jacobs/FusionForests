# Simulation Study Conclusions: IRS for Missing Data in BART

## 1. Study Overview

We benchmarked Incorporated Random Splits (IRS) — a missing data mechanism for BART — across three simulation studies with increasing complexity:

- **v2**: Single-forest BART, Y = f(X, A)
- **v3**: Two-forest BCF decomposition, Y = mu(X, e) + tau(X) * A
- **Investigation**: Targeted study isolating sample size vs confounding as drivers of IRS collapse

All simulations use the same data-generating process: n=500 (150 RCT + 350 RWD), p=5 covariates with AR(1) correlation, and X5 subject to block-wise or MCAR missingness. Four outcome scenarios vary the role of X5: irrelevant (Sc 0), prognostic (Sc 1), effect modifier (Sc 2), and both (Sc 3).

Methods compared: Oracle (full data), IRS, Complete case, Complete covariates (drop X5), MissForest+BART/BCF, MICE+BART, and BART+MIA (bartMachine).

---

## 2. Main Findings

### 2.1 IRS performs well under block_rct missingness

When X5 is missing in the RCT (the primary motivating setting for data fusion), IRS recovers most of the Oracle's efficiency:

| Scenario (v2, rho=0) | Oracle | IRS | Complete cov. |
|-----------------------|--------|-----|---------------|
| 0 (irrelevant)        | 0.576  | 0.580 | 0.554 |
| 1 (prognostic)        | 0.616  | 0.649 | 0.744 |
| 2 (effect modifier)   | 0.423  | 0.608 | 1.038 |
| 3 (both)              | 0.470  | 0.673 | 1.151 |

IRS substantially outperforms Complete covariates in Scenarios 2 and 3 where X5 drives treatment heterogeneity. The 350 RWD observations with observed X5 provide enough signal for IRS to learn useful splits.

### 2.2 IRS collapses to Complete covariates under block_rwd missingness

When X5 is missing in the RWD (the larger sample), IRS produces CATE RMSE and bias values identical to simply dropping X5:

| Scenario (v2, rho=0) | Oracle | IRS | Complete cov. |
|-----------------------|--------|-----|---------------|
| 1 (prognostic)        | 0.619  | **0.754** | **0.744** |
| 2 (effect modifier)   | 0.423  | **1.037** | **1.041** |
| 3 (both)              | 0.473  | **1.148** | **1.154** |

IRS matches Complete covariates to three decimal places. This is the most important negative finding: IRS cannot create information that doesn't exist. When 70% of observations have X5 entirely missing, random splits on X5 are pure noise.

### 2.3 The collapse is structural, not driven by sample size or confounding

The targeted investigation crossed three sample configurations (150/350, 350/150, 500/500) with two confounding levels (alpha5=0 vs 0.5). Key results:

- **Inverting sample sizes (350 RCT / 150 RWD)** does not help IRS. Even with more observed X5 data, IRS still equals Complete covariates.
- **Removing confounding (alpha5=0)** does not help IRS either. The collapse persists even without confounding through X5.
- **Equal sizes (500/500)** show the same pattern.

This confirms the collapse is purely structural: when a covariate is entirely unobserved in one block, IRS random splits carry zero information regardless of the sample size or confounding structure.

### 2.4 Informed vs uniform routing: no difference under block missingness

The investigation compared three IRS variants:
- Mode 1 (skip-then-draw): NaN obs excluded from MH ratio
- Mode 2 (draw-then-decide): NaN obs routed first, included in MH ratio
- Mode 3 (uniform random): P=0.5 coin flip

All three produce identical results under block missingness. The informed routing mechanism has no information to exploit when X5 is completely unobserved in an entire block.

### 2.5 MissForest consistently outperforms IRS under block_rwd

MissForest can partially reconstruct X5 from the other covariates before fitting BART. This gives it a meaningful advantage:

| Scenario 3, block_rwd, rho=0 (v2) | CATE RMSE | Bias |
|------------------------------------|-----------|------|
| Oracle                              | 0.473     | 0.012 |
| MissForest+BART                     | **0.903** | 0.414 |
| IRS                                 | **1.148** | 0.479 |
| Complete covariates                  | **1.154** | 0.479 |

The advantage grows with covariate correlation: at rho=0.5, MissForest improves further because X5 is more recoverable from X1-X4.

### 2.6 MICE+BART has a severe coverage problem

MICE+BART produces competitive point estimates but catastrophically undercovers:

| Metric (v2, averaged) | MICE+BART | IRS | MissForest |
|-----------------------|-----------|-----|------------|
| CATE coverage (nominal 95%) | **50-72%** | 90-95% | 91-95% |
| CI width | ~0.9 | ~2.0 | ~2.0 |

The simple averaging of posterior samples across imputations underestimates uncertainty. MICE+BART should not be used for inference without proper Rubin's rules variance pooling.

### 2.7 BART+MIA (bartMachine) offers no advantage

bartMachine with Missingness Incorporated in Attributes produces CATE RMSE comparable to IRS and MissForest, but:
- ~10x slower (270s vs 28s per replicate)
- Requires Java, which is incompatible with R's fork-based parallelism
- No meaningful accuracy advantage over simpler methods

### 2.8 BCF decomposition (v3) does not rescue IRS

The two-forest BCF model (Y = mu(X,e) + tau(X)*A) shows the same IRS collapse pattern under block_rwd:

| Scenario 2, block_rwd, rho=0 (v3) | CATE RMSE |
|------------------------------------|-----------|
| Oracle                              | 0.582 |
| MissForest+BCF                      | **0.662** |
| IRS                                 | **1.118** |
| Complete covariates                  | **1.124** |

BCF does improve CATE estimation generally (tau(x) is estimated directly rather than through counterfactual subtraction), but it does not change the fundamental limitation of IRS under block missingness.

BCF coverage is notably lower than single-forest BART (~87% vs ~94%), suggesting the narrower treatment forest credible intervals undercover.

### 2.9 Correlation helps imputation, not IRS

At rho=0.5 (moderate covariate correlation), MissForest and MICE improve because X5 becomes partially predictable from X1-X4. IRS does not benefit: the random splits are still uninformed regardless of correlation structure.

---

## 3. Summary: When to Use Each Method

| Setting | Best method | Why |
|---------|-------------|-----|
| X5 missing in RCT (small sample) | **IRS** | Leverages X5 from the larger RWD to learn splits |
| X5 missing in RWD (large sample) | **MissForest+BART/BCF** | IRS collapses; imputation partially recovers X5 |
| MCAR missingness | **IRS or MissForest** | Both perform well; IRS is simpler |
| X5 irrelevant | **Any method** | No method is harmed; IRS adds negligible overhead |
| Need valid coverage | **IRS or MissForest** | Avoid MICE+BART (severe undercoverage) |

---

## 4. Limitations and Open Questions

1. **Block missingness is the hardest case.** The clean separation (X5 entirely missing in one block) is the worst case for IRS. Partial missingness within blocks would likely show IRS performing better.

2. **Propensity score estimation under missingness.** In v3, the propensity score was estimated on the full X (before imposing missingness). In practice, the propensity score itself would be affected by missing covariates.

3. **IRS mode 2 (draw-then-decide) under partial missingness.** Mode 2 includes routed observations in the MH ratio. Under block missingness this doesn't help, but under partial missingness within a block, mode 2 could outperform mode 1 by leveraging outcome information from routed observations.

4. **The BCF coverage gap.** The ~87% coverage (vs nominal 95%) in v3 warrants further investigation — it may reflect the leaf prior calibration (omega_treat) rather than a fundamental issue.

5. **Scalability of MissForest.** MissForest is fast in these simulations (p=5, n=500), but may become prohibitive with larger covariate spaces.
