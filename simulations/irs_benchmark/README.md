# IRS Benchmark Simulation

Benchmarks **Imputation at Random Splitting (IRS)** — a BART-native strategy for handling missing covariates — against standard alternatives.

## Methods compared

| Method                | Description                                                              |
| --------------------- | ------------------------------------------------------------------------ |
| **Oracle**            | BART on fully observed data (no missingness); lower bound                |
| **IRS (informed)**    | `SimpleBART` with `irs=2`; split probabilities informed by observed data |
| **IRS (uniform)**     | `SimpleBART` with `irs=3`; uniform random split at missing values        |
| **Complete case**     | Drop rows with any `NaN`, fit standard BART on survivors                 |
| **bartMachine MIA**   | `bartMachine` with `use_missing_data=TRUE` + missingness dummies         |
| **missForest + BART** | Impute with `missForest`, then fit standard BART                         |

## Data-generating processes

**Covariates:** 10 dimensions. For `quadratic`/`linear`/`friedman`: multivariate Gaussian with mean 1 and equi-correlation ρ=0.5. For `nonlinear`: 10 nonlinear transforms of a single hidden uniform variable on [-3, 0] (manifold structure).

**Regression functions:**

| Model       | f(X)                                                      |
| ----------- | --------------------------------------------------------- |
| `quadratic` | X₁² + X₂² + X₃²                                           |
| `linear`    | β′X with β = (1, 2, −1, 3, −0.5, −1, 0.3, 1.7, 0.4, −0.3) |
| `friedman`  | 10 sin(πX₁X₂) + 20(X₃−0.5)² + 10X₄ + 5X₅                  |
| `nonlinear` | sin(πX₁X₂) + 2(X₃−0.5)² + X₄ + 0.5X₅                      |

**Noise:** y = f(X) + ε, ε ~ N(0, σ²), σ = 0.1.

## Missingness mechanisms

Missingness is imposed on columns 1–3 only.

| Pattern      | Mechanism                                                                                                                           |
| ------------ | ----------------------------------------------------------------------------------------------------------------------------------- |
| `mcar`       | Each entry missing independently with probability p_miss                                                                            |
| `mnar`       | Values above the (1 − p_miss) quantile are missing (large values masked)                                                            |
| `predictive` | MCAR missingness, but the response is regenerated as y = Σ(Xⱼ² + 2Mⱼ) + ε, so the missingness indicator M itself is predictive of y |

## Scenario grid

Full factorial: **4 models × 3 patterns × 4 p_miss × 4 n_train = 192 scenarios**, each with 100 replications.

| Factor  | Levels                                 |
| ------- | -------------------------------------- |
| model   | quadratic, linear, friedman, nonlinear |
| pattern | mcar, mnar, predictive                 |
| p_miss  | 0.2, 0.4, 0.6, 0.8                     |
| n_train | 100, 200, 500, 1000                    |

## BART settings

- 200 trees, 2000 posterior draws, 1000 burn-in
- Test set: n = 1000 per replication

## Evaluation metrics

- RMSE (train & test)
- Bias (test)
- MAE (test)
- 95% credible interval coverage
- 95% credible interval width

## Usage

```bash
Rscript irs_benchmark.R [num_cores]
```

Output saved to `$TMPDIR/` as per-scenario `.rds` files and a combined `irs_benchmark_combined.rds`.
