# Simulation study: prior specification for the data-fusion AFT–BART model

> **Outcome of the study (current methodology).** The five phases below
> tested an asymmetric variance-budget parametrisation against several
> simpler alternatives. The final adopted default in the methodology is the
> **uniform Chipman calibration** $\sigma_{h,f} = k_f/(2\sqrt{m_f})$ with
> $k_f = 1$ for every forest, tree counts
> $(m_{\mathrm{sh}}, m_d, m_\tau, m_c) = (200, 50, 100, 50)$, and the
> per-forest depth priors $(\alpha_\tau, \beta_\tau) = (0.95, 3)$ and
> $(\alpha_c, \beta_c) = (0.25, 3)$ (other forests use the BART defaults
> $(0.95, 2)$). The variance-budget framework and per-forest shrinkage
> factors $(k_d, k_\tau, k_c)$ described below were the *experimental*
> parametrisations under which the simulations were run; the uniform
> Chipman default emerged from the v4 and v5 evidence. The phase-by-phase
> notes are kept as a historical record of what was tested.

**Original purpose.** We derived a variance-budget prior for the four
forests $(m_0^{\mathrm{sh}}, d, \tau, c)$ and parameterised it by per-forest
shrinkage factors $k_f$. This study checked that derivation empirically: it
selected the default $k_f$ (and the tree/depth settings) and confirmed that
the method delivers the promised efficiency gain on the CATE **without**
importing real-world confounding bias.

Two variants of the study are run:

- **Censored variant** — outcomes are right-censored at a fixed administrative
  rate of $35\%$. This is the headline study.
- **Uncensored variant** — identical DGP and configs, no censoring step. Used as
  a clean reference: with all outcomes observed we can standardise $\log T$
  cleanly to unit variance and remove the wobble introduced by the preliminary
  AFT-based scale estimate that the censored fit relies on.

---

## 1. Questions the study must answer

1. **Which prior gives the best CATE estimate?** Lowest RMSE of $\tau(x)$ at
   nominal (95%) credible-interval coverage, across confounding scenarios.
2. **Is the efficiency gain real?** Does borrowing beat the RCT-only estimator
   when the real-world data (RWD) are informative, and at least match it when
   they are not?
3. **Is the bias controlled?** When confounding is present, does a too-tight
   prior on $c$ leak bias into $\tau$? Where is the safe operating point for
   $k_c$?

---

## 2. Estimand

On the log-time scale, with the **target population = RWD** ($S=0$):

$$\tau(x) \;=\; \mathbb{E}[\log T(1) - \log T(0)\mid X=x].$$

The true $\tau(x)$ is known from the DGP and used directly to score
$\hat\tau(x)$. The confounding function $c(x)$ is an internal nuisance
parameter; it is not scored as a primary outcome.

---

## 3. Data-generating process

### 3.1 Covariates and unmeasured confounder

- $p = 10$: $X_1,\dots,X_5$ are _active_, $X_6,\dots,X_{10}$ are noise.
- $X_j \stackrel{\mathrm{iid}}{\sim} U(0,1)$ for every $j$ and both sources. No
  source-specific covariate shift.
- Unmeasured confounder $U\sim N(0,1)$, independent of $X$ and identical in
  distribution across sources (satisfies cross-source exchangeability).

For convenience the DGP forms below use the centred covariates
$Z_j = X_j - 0.5 \sim U(-0.5, 0.5)$.

### 3.2 Outcome model (structural)

$$
\log T \;=\; f_0(X,U) \;+\; A\,\tau^*(X,U) \;+\; \sigma\,\varepsilon,
\qquad \varepsilon \sim N(0,1).
$$

- **Prognostic baseline:**
  $f_0(X,U) = 0.7\sin(\pi Z_1) + 1.0\,Z_2 Z_3 + 1.0\,(Z_4^2 - \tfrac{1}{12}) + \beta_U\,U$
  (the $-\tfrac{1}{12}$ centres $Z_4^2$ at mean zero under $U(-0.5,0.5)$).
- **Unit-level effect:** $\tau^*(X,U) = \tau(X) + \gamma_U\,U$. Since
  $U\perp X$, $\tau(x)=\mathbb{E}[\tau^*\mid X{=}x]$, so the CATE is $\tau(X)$.
- **CATE:**
  $\tau(X) = \tau_0 + \tau_1 Z_1 + \tau_2\,\mathbf 1\{X_2>0.5\}$.
  Homogeneous: $(\tau_0,\tau_1,\tau_2)=(0.5,0,0)$.
  Heterogeneous: $(\tau_0,\tau_1,\tau_2)=(0.5,0.4,0.3)$.
- **Noise scale $\sigma$.** Chosen so that the marginal residual variance
  $\mathrm{Var}(\sigma\varepsilon)=\sigma^2$ approximately matches the
  marginal variance of the structural component
  $f_0(X,U)+A\,\tau^*(X,U)$ under the heterogeneous CATE and the default
  confounding (S1). Concretely we use $\sigma = 1$, which gives
  $\mathrm{SNR}\approx 1$ on the standardised log-time scale.

### 3.3 Treatment assignment

- **RCT ($S=1$):** $A\sim\mathrm{Bernoulli}(0.5)$.
- **RWD ($S=0$):**
  $A\sim\mathrm{Bernoulli}\big(\mathrm{expit}(\alpha_0+\alpha_X(Z_1+Z_2)+\alpha_U U)\big)$,
  $\alpha_X=1.0$, $\alpha_0$ tuned so the marginal
  $\Pr(A{=}1\mid S{=}0)\approx 0.5$.
  **$\alpha_U$ is the confounding strength**.

### 3.4 Censoring (censored variant only)

Administrative censoring at a fixed log-time threshold $c^\star$:
$\delta_i = \mathbf{1}\{\log T_i \le c^\star\}$, $\tilde Y_i = \min(\log T_i, c^\star)$.
The threshold $c^\star$ is tuned once at script start (via a large
pre-simulation) so that the expected censoring rate equals $35\%$ under the
central cell (heterogeneous CATE, scenario S1, $(n_1,n_0) = (250,1000)$).
The same $c^\star$ is then used across all cells.

In the **uncensored variant** no censoring is applied: $\delta_i\equiv 1$ and
$\tilde Y_i = \log T_i$ for every observation. $\log T$ is standardised to
unit variance by dividing by its sample SD before fitting.

### 3.5 Sample sizes

$(n_1,n_0)\in\{(250,1000),\,(100,400)\}$.

---

## 4. Simulation factors

| Factor                                    | Levels                                                                                                         |
| ----------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| Confounding $(\alpha_U,\beta_U,\gamma_U)$ | **S0** none $(0,1,0)$; **S1** mild $(0.5,1,0)$; **S2** strong $(1.5,1.5,0)$; **S3** effect-mod $(1.0,1.0,0.5)$ |
| CATE heterogeneity                        | homogeneous; heterogeneous                                                                                     |
| Sample size $(n_1,n_0)$                   | $(250,1000)$; $(100,400)$                                                                                      |

**Grid:** $4 \times 2 \times 2 = 16$ cells.

---

## 5. Prior configurations to compare

> **Historical note.** The configurations in this section are the
> experimental parametrisations tested in Phase 2. They use the
> variance-budget calibration with per-forest shrinkage factors
> $(k_d, k_\tau, k_c)$ in the tightness convention. The current methodology
> has since dropped this parametrisation in favour of the uniform Chipman
> calibration $\sigma_{h,f} = k_f/(2\sqrt{m_f})$ with $k_f = 1$ across
> forests, on the basis of the v4 and v5 evidence. The table below records
> what was actually fitted in v1; current production fits use
> $(m_{\mathrm{sh}}, m_d, m_\tau, m_c) = (200, 50, 100, 50)$ and the
> uniform $k_f = 1$ default.

All Phase-2 configs use the variance-budget calibration in the code
convention $\omega_f = k_f^{\mathrm{code}}/\sqrt{m_f}$, with the setup-side
$(k_d,k_\tau,k_c)$ in the _tightness_ convention (larger ⇒ tighter prior).
The translation is $s_0=(1+k_d^{-2}+k_\tau^{-2}+k_c^{-2})^{-1/2}$ and
$k_f^{\mathrm{code}}=s_0/k_f^{\mathrm{setup}}$, with
$k_{\mathrm{sh}}^{\mathrm{setup}}=1$ so $k_{\mathrm{sh}}^{\mathrm{code}}=s_0$.

| Config      | $k_d$ | $k_\tau$ | $k_c$ | trees $(m_{\mathrm{sh}},m_d,m_\tau,m_c)$ | depth $\tau$ / $c$      | role              |
| ----------- | ----- | -------- | ----- | ---------------------------------------- | ----------------------- | ----------------- |
| **Default** | 2.5   | 1.7      | 4     | $(200, 50, 200, 200)$                    | $(0.95,3)$ / $(0.25,3)$ | proposed          |
| Efficiency  | 4     | 1.7      | 10    | $(200, 50, 200, 200)$                    | $(0.95,3)$ / $(0.25,3)$ | trust RWD         |
| Robust      | 2     | 1.7      | 2     | $(200, 50, 200, 200)$                    | $(0.95,3)$ / $(0.25,3)$ | discount RWD      |
| Equal-share | 1     | 1        | 1     | $(200, 50, 200, 200)$                    | $(0.95,3)$ / $(0.25,3)$ | agnostic baseline |
| $\tau$-deep | 2.5   | 1.7      | 4     | $(200, 50, 200, 200)$                    | $(0.95,2)$ / $(0.25,3)$ | richer HTE        |

Depth columns are written as $(\mathrm{base},\mathrm{power})$, i.e. the
non-terminal probability $\mathrm{base}\,(1+d)^{-\mathrm{power}}$ of
\citet{chipman1998bayesian}. Default $\beta_\tau=3$ enforces shallow $\tau$
trees (BCF-style); $\tau$-deep relaxes this to the BART default $\beta_\tau=2$.
The confounding-forest base $\alpha_c=0.25$ enforces very shallow $c$ trees.

No benchmarks (RCT-only, Naive-pool) are included in this study. They will be
added later — RCT-only via the `ShrinkageTrees` package and Naive-pool via a
dedicated comparison run if needed. The present study is purely a
hyperparameter comparison among the five prior configs.

---

## 6. Outcome metrics

Evaluated **in-sample on the combined training data** ($n = n_1 + n_0$
points per replication: RCT plus RWD). For each replication we take the
posterior mean and the $95\%$ equal-tailed credible interval of $\tau$ at
each training point. Bayesian shrinkage from the BART prior takes the
role that a hold-out set would play under maximum-likelihood; an external
test set drawn from the same $U(0,1)^{10}$ distribution adds Monte Carlo
noise without changing the qualitative ranking of configs.

- **Integrated RMSE:**
  $\mathrm{RMSE}_\tau=\sqrt{\frac{1}{n}\sum_{i=1}^{n}(\hat\tau(X_i)-\tau(X_i))^2}$,
  averaged over replications.
- **Pointwise coverage:** mean over training points of
  $\mathbf 1\{\tau(X_i)\in\mathrm{CI}_{95}(X_i)\}$; target $0.95$.
- **Mean CI width** for $\tau(X_i)$ (efficiency proxy).
- **Integrated bias:** $\frac{1}{n}\sum_{i=1}^{n}(\hat\tau(X_i)-\tau(X_i))$
  (signed; detects confounding leakage).

---

## 7. Protocol

- **Replications:** $R=300$ Monte Carlo datasets per cell. Report Monte Carlo
  standard errors on all summaries.
- **MCMC:** **one chain**, $2000$ warm-up + $2000$ retained iterations,
  no thinning.
- **Common random numbers** across configurations within a replication: the
  same simulated dataset is fed to every prior config and benchmark, so RMSE
  differences are not inflated by dataset-to-dataset noise.
- **Standardisation** of $\log T$ is handled inside each replication on its
  own data: empirical SD for the uncensored variant, the FusionForest internal
  preliminary AFT estimate for the censored variant.
- **Variance budget** $s_0$ is computed inside each replication from the
  config's $(k_d, k_\tau, k_c)$ with $\pi_f \equiv 1$ and then mapped to the
  code-side $k_f^{\mathrm{code}}$.

---

## 8. Analysis and decision rule

For each prior config, aggregate the four primary metrics by confounding
scenario. The default is chosen by a Pareto argument:

1. **Hard constraint:** coverage $\ge 0.90$ at every cell.
2. **Among configs meeting (1):** minimise average $\mathrm{RMSE}_\tau$ across
   S0–S3 (equal weight), i.e. best efficiency that is robust to confounding.
3. **Tie-break:** narrower CI width at nominal coverage.

Expected pattern (the study confirms or refutes it):

- **S0 (no confounding):** Efficiency ($k_c=10$) wins on RMSE; Robust is the
  most conservative.
- **S2/S3 (strong confounding):** Efficiency shows large bias and
  under-coverage; Robust and Default stay near nominal.
- **Default** should sit on the efficiency/robustness frontier across all
  scenarios — that is the hypothesis under test.

---

## 9. Reporting

- **F1.** $\mathrm{RMSE}_\tau$ by config × confounding scenario (grouped bars).
- **F2.** Coverage by config × scenario (with the $0.95$ target line).
- **F3.** CI width (efficiency) vs RMSE scatter, one point per config, faceted
  by scenario (the frontier).
- **T1.** Full metric table (RMSE, bias, coverage, width) for the full
  $4\times 2\times 2$ grid, with Monte Carlo SEs.

Each figure and table is produced for both the censored and the uncensored
variant. The uncensored set is the appendix-grade clean reference; the
censored set is the headline.

---

## 10. Phasing

- **Phase 1 (pilot):** configs = {Default, Efficiency, Robust},
  $R=100$, $(250,1000)$ only, both variants. Confirms the frontier story.
- **Phase 2 (full):** all five prior configs, full $4\times 2\times 2$ grid,
  $R=300$, both sample sizes, both variants.
- **Phase 2b (v2 refinement, uncensored only):** the Phase 2 results
  identified Robust $(k_d, k_\tau, k_c) = (2, 1.7, 2)$ as the winner but
  flagged systematic over-coverage ($\geq 0.94$ in every non-Efficiency
  cell). v2 fine-tunes two knobs around Robust on the same
  $4\times 2\times 2$ DGP grid:
  - $k_\tau \in \{1.5, 1.7, 2.0\}$ (holding $k_d = k_c = 2$ fixed);
  - $\omega^2 \in \{0.95, 1.00\}$, where $\omega^2$ is the total prior
    variance budget on standardised $\log T$. Tightening to $\omega^2 = 0.95$
    proportionally tightens every forest's leaf-prior SD, with the goal of
    pulling coverage back toward nominal.

  Six configs total ($3 \times 2$), uncensored variant only, $R=300$, single
  chain. Script: `simulation_uncensored_hpc_v2.R`. Output:
  `simulation_uncensored_hpc_v2_output.rds`.
- **Phase 2c (v3 tree-count sensitivity, uncensored only):** holds
  $(k_d, k_\tau, k_c) = (2, 1.5, 2)$, $\omega = 1.0$, and $\sigma = 1$
  fixed and sweeps the tree counts of the $\tau$ and $c$ forests:
  - $m_\tau \in \{50, 100, 200\}$;
  - $m_c \in \{50, 100, 200\}$;
  - $m_{\mathrm{sh}} = 200$ and $m_d = 50$ are held fixed.

  Nine configs total ($3 \times 3$), uncensored variant only, $R=300$,
  single chain, same $4\times 2\times 2$ DGP grid. Script:
  `simulation_uncensored_hpc_v3.R`. Output:
  `simulation_uncensored_hpc_v3_output.rds`.
- **Phase 2d (v4 aggressive $\omega$ sweep, uncensored only):** v2 showed
  that $\omega = 0.95$ does not pull coverage back to nominal (it stays
  $\geq 0.99$ everywhere). v3 showed that tree counts in $\{50, 100, 200\}$
  for $\tau$ and $c$ do not move CATE metrics meaningfully. v4 anchors at
  the v2 winner $(k_d, k_\tau, k_c) = (2, 2.0, 2)$ with
  $(m_{\mathrm{sh}}, m_d, m_\tau, m_c) = (200, 100, 100, 100)$ and sweeps
  $\omega$ more aggressively to locate where coverage drops back toward
  $0.95$:
  - $\omega \in \{0.50, 0.75, 1.00\}$.

  In addition to the four CATE metrics, v4 records the posterior mean of
  the residual scale $\sigma$ as a diagnostic for the residual-prior
  hypothesis (is $\sigma$ being allowed to drift above the truth
  $\sigma = 1$ under the default $(\nu, q) = (3, 0.9)$ inverse-$\chi^2$
  prior?).

  Three configs total, uncensored variant only, $R=300$, single chain,
  same $4\times 2\times 2$ DGP grid. Script:
  `simulation_uncensored_hpc_v4.R`. Output:
  `simulation_uncensored_hpc_v4_output.rds`.
- **Phase 2e (v5 parametrisation comparison, uncensored only, 3 scenarios):**
  pits the variance-budget anchor at $\rho \in \{0.25, 0.50, 0.75, 1.00\}$
  with $(k_{\mathrm{sh}}, k_d, k_\tau, k_c) = (1, 2, 2, 2)$ against an
  alternative activation-weighted parametrisation
  $\sigma_{h,f} = k/(2\sqrt{m_f})$ with a single per-forest $k = \gamma/2$
  (i.e. all forests share the same leaf prior). Scenarios restricted to
  S0, S2, S3. Eight configs (4 anchor + 4 new). Script:
  `simulation_uncensored_hpc_v5.R`. Output:
  `simulation_uncensored_hpc_v5_output.rds`.

### Outcome and adopted methodology

The simulation phases converge on a single recommendation that the
**methodology now adopts**: the **uniform Chipman calibration**
$\sigma_{h,f} = k_f/(2\sqrt{m_f})$ with $k_f = 1$ for every forest, tree
counts $(m_{\mathrm{sh}}, m_d, m_\tau, m_c) = (200, 50, 100, 50)$, depth
priors $(\alpha_\tau, \beta_\tau) = (0.95, 3)$ and
$(\alpha_c, \beta_c) = (0.25, 3)$, others at the BART default $(0.95, 2)$.
The variance-budget framework and the per-forest $(k_d, k_\tau, k_c)$
tuples were valuable for navigating the design space but did not deliver
an RMSE gain large enough to justify the calibration overhead; the uniform
Chipman default matches the BART/BCF defaults of the implementation and
is coherent with the inverse-$\chi^2$ residual prior. See the methodology
section for the corresponding writeup.
