# Simulation Design: Benchmarking Missing Data Methods for BART

## 1. Overview

We benchmark missing data handling strategies for Bayesian Additive Regression Trees (BART) in a standard regression setting. The simulation is designed to reflect the structure of our data fusion problem — combining a small RCT with a larger observational dataset — where **block-wise missingness** (entire covariates unobserved in one data source) is the primary missing data pattern.

The methods under comparison are:

| Method | Description |
|--------|-------------|
| **IRS** | Incorporated Random Splits — our proposed missing data mechanism for BART |
| **BART + MIA** | BART with Missingness Incorporated in Attributes |
| **MissForest + BART** | MissForest imputation as preprocessing, then standard BART |
| **Oracle** | BART fitted on the full data with no missingness (upper bound) |
| **Complete case** | BART fitted only on observations with no missing values |
| **Complete covariates** | BART fitted on all observations, but using only covariates that are fully observed |

---

## 2. Data-Generating Process

### 2.1 Sample structure

| Quantity | Value |
|----------|-------|
| Total sample size | $n = 500$ |
| RCT sample ($S = 1$) | $n_{\text{RCT}} = 150$ |
| RWD sample ($S = 0$) | $n_{\text{RWD}} = 350$ |
| Number of covariates | $p = 5$ |

### 2.2 Covariates

Draw $X = (X_1, X_2, X_3, X_4, X_5) \sim \mathcal{N}_5(0, \Sigma)$ where

$$\Sigma_{jk} = \rho^{|j - k|}$$

for some correlation parameter $\rho \in \{0, 0.5\}$. The parameter $\rho$ controls how recoverable $X_5$ is from the remaining covariates — this directly affects how much imputation-based methods can compensate for the missing covariate.

> **Note:** The correlation structure matters because it determines the information content of the observed covariates about the missing one. When $\rho = 0$, the missing covariate is unrecoverable from the rest; when $\rho = 0.5$, partial recovery is possible.

### 2.3 Treatment assignment

- **RCT** ($S = 1$): Treatment is randomized, $A \mid S = 1 \sim \text{Bernoulli}(0.5)$.
- **RWD** ($S = 0$): Treatment depends on covariates (confounding):
$$P(A = 1 \mid X, S = 0) = \text{expit}(\alpha_0 + \alpha_1 X_1 + \alpha_5 X_5),$$
where $\text{expit}(z) = 1/(1 + e^{-z})$. The parameter $\alpha_5$ controls whether $X_5$ is associated with treatment assignment in the RWD (confounding).

> **Key property:** By design, $X_5$ is independent of $A$ in the RCT (randomization), but may be associated with $A$ in the RWD (confounding). This mirrors the motivating data fusion setting.

### 2.4 Outcome model

The fitted model is a standard regression:

$$Y = m(X, A) + \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, \sigma^2)$$

The true function $m(X, A)$ varies by scenario (see Section 3). We set $\sigma^2$ to achieve a signal-to-noise ratio of approximately 2 (to be calibrated).

### 2.5 Missing covariate

$X_5$ is the covariate subject to missingness. Let $M \in \{0, 1\}$ indicate whether $X_5$ is missing ($M = 1$) for a given observation.

---

## 3. Simulation Scenarios

We cross two design factors: the **role of $X_5$** in the outcome model (Sections 3.1–3.4) and the **missingness pattern** (Section 3.5). Each combination defines one simulation scenario.

### 3.1 Scenario 0: $X_5$ is irrelevant

$$m(X, A) = 1 + X_1 + X_2 + 2X_3 A$$

$X_5$ does not appear in the outcome model.

**Why:** Sanity check. No method should gain from recovering $X_5$, and no method should be harmed much by its absence. Any performance difference reflects overhead or artifacts of the missing data strategy.

### 3.2 Scenario 1: $X_5$ is prognostic only

$$m(X, A) = 1 + X_1 + X_2 + \beta_5 X_5 + 2X_3 A$$

$X_5$ affects the outcome but does not interact with treatment.

**Why:** Missing $X_5$ increases residual variance (hurts prediction), but should not introduce bias in treatment effect estimation. This scenario tests whether methods maintain unbiased CATE estimation when the missing covariate is purely prognostic, and whether recovering it yields efficiency gains.

### 3.3 Scenario 2: $X_5$ is an effect modifier

$$m(X, A) = 1 + X_1 + X_2 + (\gamma_0 + \gamma_1 X_5) A$$

$X_5$ interacts with treatment but has no prognostic (main) effect.

**Why:** This is the critical scenario. Missing $X_5$ directly impairs CATE estimation because the treatment effect varies with $X_5$. Methods that can recover or work around the missing effect modifier will have a clear advantage.

### 3.4 Scenario 3: $X_5$ is prognostic and an effect modifier

$$m(X, A) = 1 + X_1 + X_2 + \beta_5 X_5 + (\gamma_0 + \gamma_1 X_5) A$$

$X_5$ has both a main effect and a treatment interaction.

**Why:** The most demanding case. Missing $X_5$ affects both prediction accuracy and CATE estimation. This scenario tests whether methods can simultaneously recover prognostic and effect-modifying information.

### 3.5 Missingness patterns

For each outcome scenario above, we consider three missingness structures for $X_5$:

**(a) Block-wise: $X_5$ missing in the RCT.** $P(M = 1 \mid S = 1) = 1$, $P(M = 1 \mid S = 0) = 0$. The covariate is entirely unobserved in the RCT but fully available in the RWD.

> This is the primary motivating pattern: RCTs often collect fewer covariates than observational databases. Note that $X_5$ is independent of $A$ in the RCT by randomization, but may be associated with $A$ in the RWD.

**(b) Block-wise: $X_5$ missing in the RWD.** $P(M = 1 \mid S = 1) = 0$, $P(M = 1 \mid S = 0) = 1$. Reversed direction.

> In the data fusion context, this means the variable needed for confounding adjustment in the RWD is unavailable there. This has different implications depending on whether $X_5$ drives confounding.

**(c) MCAR.** $P(M = 1 \mid S) = \pi$ for some $\pi$ (e.g., $\pi = 0.3$), regardless of source.

> This is a special case of the block-wise framework (missingness independent of $S$). Serves as a standard reference to compare structured vs. unstructured missingness.

---

## 4. Outcome Measures

All metrics are computed on both the **training sample** (relevant for treatment effect estimation on the study population) and a **held-out test sample** (with no missingness, to isolate the effect of the missing data strategy from overfitting).

### 4.1 Prediction accuracy

| Metric | Definition |
|--------|------------|
| RMSE of $\hat{m}$ | $\sqrt{\frac{1}{n}\sum_{i=1}^{n} (\hat{m}(X_i, A_i) - m(X_i, A_i))^2}$ |
| MAE of $\hat{m}$ | $\frac{1}{n}\sum_{i=1}^{n} |\hat{m}(X_i, A_i) - m(X_i, A_i)|$ |

### 4.2 Treatment effect estimation

| Metric | Definition |
|--------|------------|
| Bias of $\hat{\tau}(x_i)$ | $\frac{1}{n}\sum_{i=1}^{n} [\hat{\tau}(x_i) - \tau(x_i)]$ |
| RMSE of $\hat{\tau}(x_i)$ | $\sqrt{\frac{1}{n}\sum_{i=1}^{n} (\hat{\tau}(x_i) - \tau(x_i))^2}$ |
| Bias of $\hat{\tau}_{\text{ATE}}$ | $\hat{\tau}_{\text{ATE}} - \tau_{\text{ATE}}$ where $\hat{\tau}_{\text{ATE}} = \frac{1}{n}\sum_i \hat{\tau}(x_i)$ |
| RMSE of $\hat{\tau}_{\text{ATE}}$ | Across simulation replicates |

Here $\hat{\tau}(x) = \hat{m}(x, 1) - \hat{m}(x, 0)$ and $\tau(x) = m(x, 1) - m(x, 0)$.

### 4.3 Uncertainty quantification

| Metric | Definition |
|--------|------------|
| Coverage of $\hat{\tau}(x_i)$ | Proportion of pointwise 95% credible intervals containing $\tau(x_i)$ |
| Average CI width of $\hat{\tau}(x_i)$ | Mean width of 95% credible intervals across observations |
| Coverage of $\hat{\tau}_{\text{ATE}}$ | Proportion of simulations where the 95% credible interval contains $\tau_{\text{ATE}}$ |
| Average CI width of $\hat{\tau}_{\text{ATE}}$ | Mean width of the ATE credible interval across simulations |

### 4.4 Variance decomposition

| Metric | Definition |
|--------|------------|
| Posterior variance of $\hat{\tau}(x_i)$ | $\frac{1}{n}\sum_{i=1}^{n} \text{Var}[\tau(x_i) \mid \mathcal{D}]$ — average within-simulation uncertainty |
| Frequentist variance of $\hat{\tau}_{\text{ATE}}$ | $\text{Var}_{\text{sim}}[\hat{\tau}_{\text{ATE}}]$ — variability of the point estimate across simulation replicates |

### 4.5 Computational cost

| Metric | Definition |
|--------|------------|
| Wall-clock time | Total time per method per simulation replicate |

---

## 5. Simulation Parameters

| Parameter | Values | Notes |
|-----------|--------|-------|
| $n_{\text{RCT}}$ | 150 | |
| $n_{\text{RWD}}$ | 350 | |
| $p$ | 5 | |
| $\rho$ | 0, 0.5 | Covariate correlation |
| $\beta_5$ | 1 | Prognostic effect of $X_5$ (Scenarios 1, 3) |
| $\gamma_0$ | 2 | Baseline treatment effect |
| $\gamma_1$ | 1 | Effect modification by $X_5$ (Scenarios 2, 3) |
| $\alpha_0$ | 0 | Intercept for RWD propensity score |
| $\alpha_1$ | 0.5 | Confounding through $X_1$ in RWD |
| $\alpha_5$ | 0.5 | Confounding through $X_5$ in RWD |
| $\sigma^2$ | TBD | Calibrate for SNR $\approx 2$ |
| $\pi$ | 0.3 | MCAR missingness probability |
| $n_{\text{test}}$ | 500 | Test set size (complete data, no missingness) |
| $n_{\text{sim}}$ | 200 | Number of simulation replicates |

---

## 6. Scenario Overview Table

| Scenario | Role of $X_5$ | Missingness | Key question |
|----------|--------------|-------------|--------------|
| 0a | Irrelevant | Block: missing in RCT | Does the method waste effort on a noise variable? |
| 0b | Irrelevant | Block: missing in RWD | Same, reversed direction |
| 0c | Irrelevant | MCAR ($\pi = 0.3$) | Sanity check under standard missingness |
| 1a | Prognostic | Block: missing in RCT | Can efficiency be recovered without bias? |
| 1b | Prognostic | Block: missing in RWD | Same, reversed |
| 1c | Prognostic | MCAR ($\pi = 0.3$) | Reference |
| 2a | Effect modifier | Block: missing in RCT | Can CATE be estimated when the effect modifier is missing in the RCT? |
| 2b | Effect modifier | Block: missing in RWD | Same, reversed |
| 2c | Effect modifier | MCAR ($\pi = 0.3$) | Reference |
| 3a | Prognostic + effect modifier | Block: missing in RCT | Hardest case, block-wise |
| 3b | Prognostic + effect modifier | Block: missing in RWD | Hardest case, reversed |
| 3c | Prognostic + effect modifier | MCAR ($\pi = 0.3$) | Hardest case, MCAR reference |

Total: 12 scenarios $\times$ 2 correlation levels ($\rho$) = 24 configurations.
