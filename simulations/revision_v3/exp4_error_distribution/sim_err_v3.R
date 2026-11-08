# ══════════════════════════════════════════════════════════════════════════════
# sim_err_v3.R -- nonparametric error distribution: residual priors across five error settings, CATE and outcome
# Revision round (manuscript_v3).  Derived from submission_v2/exp4_error_distribution/sim_err_v2.R.
#
# Changes for the revision:
#   * Unmeasured confounder U ~ N(0, 1) enters the RWD outcome in BOTH arms, so
#     T(1) - T(0) = tau(X) for every unit and the CATE is common to the two
#     sources (referee 1, comment 2).  U drives RWD treatment choice with
#     strength gamma_u (optionally covariate-dependent: comment 4).
#   * Fits call the exported FusionForest() / CausalShrinkageForest() with the
#     priors k = 1 for every forest, including k_tau = 1, on
#     sigma_h = k / (2 sqrt J); the
#     frozen package copies of the submission scripts are gone, so the package's
#     own outcome standardisation is used, as the paper describes.
#   * The RWD-only comparator no longer receives the true propensity score,
#     which was a function of the unmeasured U.
# Output: $TMPDIR/sim_err_v3_output.rds (HPC contract, see ../../hpc_job_skeleton.sh).
# ══════════════════════════════════════════════════════════════════════════════
library(FusionForests)
library(ShrinkageTrees)
library(doParallel)
library(foreach)
library(evd)
library(MASS)

# ── DGP ───────────────────────────────────────────────────────────────────────
p_total     <- 10L               # observed covariates

# This experiment isolates the residual model.  We set lambda_u = lambda_d = 0,
# so the two sources share the same structural mean (no unmeasured confounding,
# no between-source prognostic deviation) and differ ONLY in the distribution of
# their error term, which is set by `error_setting` (1 = easy ... 5 = hard,
# plus 6 = no shared components).  Settings 5 and 6 are also run with a small
# trial (n_rct = 50): the regime where sharing mixture components across the
# sources should pay off (5) or cost (6) the most (associate editor, comment 1).
# v2 pushes the error model harder: heavier, coarser censoring so a large share
# of observations enter the likelihood only through the error CDF over wide
# intervals -- where getting the residual shape/scale right (HDPM) matters most.
target_cens <- 0.50              # heavier RCT right-censoring
visit_probs <- c(0.25, 0.5, 0.75)# coarse RWD visits: quartile bins, ~25% right-censored beyond

# Covariates: multivariate normal with AR(1) dependence, Sigma_ij = rho^|i-j|.
rho_X    <- 0.3
Sigma_X  <- rho_X ^ abs(outer(seq_len(p_total), seq_len(p_total), "-"))
draw_X   <- function(n) MASS::mvrnorm(n, mu = rep(0, p_total), Sigma = Sigma_X)

m0   <- function(X)  (2 * X[, 1] - X[, 2] * X[, 3] + 0.5 * X[, 4]^2)
tau  <- function(X)  (1/2 + X[, 1] - 0.5 * X[, 2]^2)         # true CATE

# ── Error settings (v3) ───────────────────────────────────────────────────────
# Six mean-zero error laws per source.  Every non-null rung pairs a per-source
# SCALE gap (sigma_RCT << sigma_RWD, which the single-sigma gaussian/shared_dp
# priors cannot represent) with growing non-Gaussianity and source-specific
# shape -- the regime the source-specific HDP mixture is built for.  All
# generators return n mean-zero draws; the per-source sd is noted in [].
gamma_em   <- 0.5772156649
std_t      <- function(n, df) rt(n, df) * sqrt((df - 2) / df)        # variance 1
std_gumbel <- function(n) {                                          # skewed, var 1
  bs <- sqrt(6) / pi
  -rgumbel(n, loc = -bs * gamma_em, scale = bs)
}
# Normal mixture with shared atoms `mu`, weights `w`, common within-component sd
# `s`, standardised to mean zero and unit variance (so an external scale factor
# sets the per-source sd directly while the modal shape is preserved).
mix_norm <- function(n, mu, w, s) {
  mbar <- sum(w * mu)
  v    <- sum(w * ((mu - mbar)^2 + s^2))      # mixture variance
  k    <- sample.int(length(mu), n, replace = TRUE, prob = w)
  (rnorm(n, mu[k], s) - mbar) / sqrt(v)
}
# Mean-zero error draw for a source ("rct"/"rwd") under one of the five settings.
draw_eps <- function(n, setting, source) {
  s <- as.character(setting); rct <- source == "rct"
  # 1) Gaussian, identical -- null (gaussian likelihood correct).
  if (s == "1") return(rnorm(n))                                          # [1.0 / 1.0]
  # 2) Gaussian, pure scale gap -- isolates the per-source-scale advantage.
  if (s == "2") return(if (rct) rnorm(n) * 0.5 else rnorm(n) * 1.5)       # [0.5 / 1.5]
  # 3) Skew + scale gap (RCT tight normal, RWD wide skewed).
  if (s == "3") return(if (rct) rnorm(n) * 0.5 else std_gumbel(n) * 1.5)  # [0.5 / 1.5]
  # 4) Bimodal RWD + scale gap (RCT tight light-tailed, RWD wide bimodal).
  if (s == "4") return(if (rct) std_t(n, 8) * 0.5                         # [0.5 / ~1.6]
                       else mix_norm(n, c(-0.9, 0.9), c(0.5, 0.5), 0.4) * 1.6)
  # 5) Multimodal both, shared atoms, large weight AND scale gap (showcase).
  if (s == "5") return(if (rct)                                          # [~0.6 / ~1.7]
                         mix_norm(n, c(-2, 0, 2), c(0.15, 0.70, 0.15), 0.3) * 0.6
                       else
                         mix_norm(n, c(-2, 0, 2), c(0.45, 0.10, 0.45), 0.3) * 1.7)
  # 6) No shared components: the opposite of 5.  The RCT error is unimodal and
  #    skewed with its mass around zero; the RWD error is bimodal with modes at
  #    about +-1.7 and essentially no mass near zero.  The atoms one source needs
  #    are of no use to the other, so the HDP link can only cost here.
  if (s == "6") return(if (rct) std_gumbel(n) * 0.6                     # [0.6 / ~1.7]
                       else mix_norm(n, c(-1, 1), c(0.5, 0.5), 0.2) * 1.7)
  stop("unknown error_setting: ", s)
}

# Latent log-survival generators (no censoring), one per source.  With lambda = 0
# both sources share m0 + A*tau and differ only through the error term.
latent_rct <- function(n, setting) {
  X <- draw_X(n)
  A <- rbinom(n, 1L, 0.5)                              # randomised
  list(X = X, A = A,
       logT = m0(X) + A * tau(X) + draw_eps(n, setting, "rct"))
}
latent_rwd <- function(n, setting) {
  X <- draw_X(n)
  U <- runif(n)                                        # RWD-only (no effect, lu = 0)
  e <- plogis(X[, 1] + U)
  A <- rbinom(n, 1L, e)
  list(X = X, A = A, e = e,
       logT = m0(X) + A * tau(X) + draw_eps(n, setting, "rwd"))
}

# Calibrate the RCT exponential censoring rate (time scale) to ~target.
# P(cens) = E[1 - exp(-rate * T)], monotone increasing in rate.
solve_cens_rate <- function(logT, target = target_cens) {
  Tt <- exp(logT)
  uniroot(function(r) mean(1 - exp(-r * Tt)) - target, c(1e-10, 1e6))$root
}

# Per-cell calibration, fixed once on a large pre-simulation:
#   rate_rct -- RCT exponential censoring rate.
#   visits   -- RWD inspection times (time scale) at the deciles q10..q80.
calibrate <- function(setting, n_cal = 2e5) {
  rate_rct <- solve_cens_rate(latent_rct(n_cal, setting)$logT)
  visits   <- as.numeric(quantile(exp(latent_rwd(n_cal, setting)$logT),
                                  probs = visit_probs))
  list(rate_rct = rate_rct, visits = visits)
}

# Simulate one full dataset (RCT right-censored, RWD interval-censored).
# All time bounds are returned on the LOG scale (timescale = "log").
make_data <- function(n_rct, n_rwd, setting, rate_rct, visits) {
  r <- latent_rct(n_rct, setting)
  w <- latent_rwd(n_rwd, setting)
  # Standardise the true (latent) log-survival of the pooled sources to mean 0,
  # unit variance.  The same affine map `std` is applied to every log-time bound
  # constructed below, so all fits operate on a common scale.  Predicted CATEs
  # are on this scale and are rescaled by `scale_sd` before evaluation.
  logT_true <- c(r$logT, w$logT)
  scale_mu  <- mean(logT_true)
  scale_sd  <- sd(logT_true)
  std       <- function(z) (z - scale_mu) / scale_sd
  logfloor <- min(c(r$logT, w$logT)) - 5    # finite stand-in for log(0)
  
  # --- RCT: exponential right censoring ---
  logC   <- log(rexp(n_rct, rate = rate_rct))
  y_rct  <- pmin(r$logT, logC)
  st_rct <- as.integer(r$logT <= logC)
  icc_rct   <- rep(0L, n_rct)               # 0 = not interval-censored
  left_rct  <- y_rct                        # ignored for status==1 / right-cens
  right_rct <- y_rct
  
  # --- RWD: interval censoring on the decile visits ---
  T0         <- exp(w$logT)
  last_visit <- visits[length(visits)]      # q80
  breaks     <- c(0, visits)                # bin edges: 0, q10, ..., q80
  j          <- findInterval(T0, breaks)    # 1..length(visits) below q80
  beyond     <- T0 >= last_visit            # event after the last visit
  j[!beyond] <- pmax(j[!beyond], 1L)
  
  left_rwd  <- ifelse(j == 1L, logfloor, log(breaks[pmin(j, length(breaks))]))
  right_rwd <- log(breaks[pmin(j + 1L, length(breaks))])
  st_rwd    <- rep(0L, n_rwd)               # no exact event times in the RWD
  icc_rwd   <- rep(1L, n_rwd)               # interval-censored by default
  # events beyond the last visit: right-censored at q80 (icc = 0)
  left_rwd[beyond]  <- log(last_visit)
  right_rwd[beyond] <- log(last_visit)
  icc_rwd[beyond]   <- 0L
  y_rwd <- right_rwd                        # placeholder; bounds carry the info

  # --- standardise all log-time bounds to the common scale ---
  y_rct     <- std(y_rct)
  left_rct  <- std(left_rct)
  right_rct <- std(right_rct)
  left_rwd  <- std(left_rwd)
  right_rwd <- std(right_rwd)
  y_rwd     <- std(y_rwd)

  list(
    X_rct = r$X, X_rwd = w$X,
    A_rct = r$A, A_rwd = w$A,
    y_rct = y_rct, y_rwd = y_rwd,
    st_rct = st_rct, st_rwd = st_rwd,
    left_rct = left_rct, right_rct = right_rct, icc_rct = icc_rct,
    left_rwd = left_rwd, right_rwd = right_rwd, icc_rwd = icc_rwd,
    X      = rbind(r$X, w$X),
    A      = c(r$A, w$A),
    S      = c(rep(1L, n_rct), rep(0L, n_rwd)),
    y      = c(y_rct, y_rwd),
    status = c(st_rct, st_rwd),
    left   = c(left_rct, left_rwd),
    right  = c(right_rct, right_rwd),
    icc    = c(icc_rct, icc_rwd),
    # True RWD propensity score (uses unobserved U).  e_rwd: training-row
    # propensity for the RWD-only fit.  e_all: propensity on the combined
    # evaluation set, with 0.5 for the randomised RCT rows.
    e_rwd  = w$e,
    e_all  = c(rep(0.5, n_rct), w$e),
    # Mean and SD used to standardise the pooled latent log-survival.  Predicted
    # CATEs are rescaled by scale_sd; full-outcome predictions by scale_sd and
    # shifted by scale_mu, before evaluation on the log-time scale.
    scale_sd = scale_sd, scale_mu = scale_mu,
    n_rct = n_rct, n_rwd = n_rwd
  )
}

# ── Fit (FusionForest, log-time, interval-censoring args; error_dist varies) ──
fit_fusion <- function(d, N_post, N_burn, error_dist) {
  # Step-height scales k_f = 1 for every forest, including k_tau = 1 (chosen
  # by a pilot; sigma_h = k / (2 sqrt J)).
  FusionForest(
    y                            = d$y,
    status                       = d$status,
    observed_left_time           = d$left,
    observed_right_time          = d$right,
    interval_censoring_indicator = d$icc,
    X_train_control           = d$X,
    X_train_treat             = d$X,
    treatment_indicator_train = d$A,
    source_indicator_train    = d$S,
    X_test_control            = d$X,
    X_test_treat              = d$X,
    X_test_deconf             = d$X,
    # Predict at the OBSERVED (A, S) so test_predictions is the full conditional
    # mean E[log T | X, A, S] (sum of forests) for each subject.  The treat-forest
    # prediction (the CATE) is A-independent, so CATE metrics are unaffected.
    treatment_indicator_test  = d$A,
    source_indicator_test     = d$S,
    outcome_type              = "right-censored",
    timescale                 = "log",
    error_dist                = error_dist,
    number_of_trees_control = 200,
    number_of_trees_treat = 100,
    number_of_trees_deconf = 50,
    number_of_trees_deviation = 50,
    k_treat                   = 1,     # k_tau = 1, the package default
    N_post = N_post, N_burn = N_burn,
    treatment_coding = "binary",
    store_posterior_sample = TRUE, verbose = FALSE)
}

# ── Single-source fits (CausalShrinkageForest) ───────────────────────────────
# FusionForest is reserved for the multi-source fusion fit; a single-source
# FusionForest is ill-defined.  The RCT is right-censored; the RWD is interval-
# censored, so each baseline uses the matching CausalShrinkageForest outcome
# type.  Both predict the CATE on the combined covariate set Xev so that metrics
# can be split by population.
# Single-source priors as in the submission: leaf SD local_hp = 1 / sqrt(J)
# (k = 2 on the paper's scale) and (alpha, beta) = (0.95, 2) for both forests.
# These are separate models with their own settings; the fusion forest uses
# k = 1 for every forest, including k_tau = 1, and beta_tau = 3.
n_trees_csf <- 200L

# RCT-only: right-censored on the log-time scale.
fit_single_rc <- function(y, status, X, A, Xev, N_post, N_burn) {
  CausalShrinkageForest(
    y                         = y,
    status                    = status,
    X_train_control           = X,
    X_train_treat             = X,
    treatment_indicator_train = A,
    X_test_control            = Xev,
    X_test_treat              = Xev,
    treatment_indicator_test  = rep(1L, nrow(Xev)),
    outcome_type              = "right-censored",
    timescale                 = "log",
    prior_type_control        = "standard",
    prior_type_treat          = "standard",
    local_hp_control          = 1 / sqrt(n_trees_csf),
    local_hp_treat            = 1 / sqrt(100),
    number_of_trees_control   = n_trees_csf,
    number_of_trees_treat     = 100,
    power_control = 2,
    power_treat = 2,
    base_control = 0.95,
    base_treat = 0.95,
    N_post = N_post, N_burn = N_burn,
    treatment_coding = "binary",
    store_posterior_sample = TRUE, verbose = FALSE)
}

# RWD-only: interval-censored on the log-time scale.  Bounds use the interval2
# convention: interval-censored rows have finite left < right; rows censored
# beyond the last visit have right = Inf (right-censored).
fit_single_ic <- function(left_time, right_time, X, A, Xev, N_post, N_burn) {
  # No propensity covariate: the true score is a function of the unmeasured U
  # (the submission scripts handed it to this fit), and an estimated score
  # from X alone adds nothing the forests do not already see.
  Xc    <- X
  Xev_c <- Xev
  CausalShrinkageForest(
    left_time                 = left_time,
    right_time                = right_time,
    X_train_control           = Xc,
    X_train_treat             = X,
    treatment_indicator_train = A,
    X_test_control            = Xev_c,
    X_test_treat              = Xev,
    treatment_indicator_test  = rep(1L, nrow(Xev)),
    outcome_type              = "interval-censored",
    timescale                 = "log",
    prior_type_control        = "standard",
    prior_type_treat          = "standard",
    local_hp_control          = 1 / sqrt(n_trees_csf),
    local_hp_treat            = 1 / sqrt(100),
    number_of_trees_control   = n_trees_csf,
    number_of_trees_treat     = 100,
    power_control = 2,
    power_treat = 2,
    base_control = 0.95,
    base_treat = 0.95,
    N_post = N_post, N_burn = N_burn,
    treatment_coding = "binary",
    store_posterior_sample = TRUE, verbose = FALSE)
}

# ── Metrics ───────────────────────────────────────────────────────────────────
# point : length-n posterior-mean CATE on the combined set.
# samp  : posterior CATE draws (FusionForest returns N_post x n_obs).  We coerce
#         to n_obs x N_post by matching the axis whose length equals length(point).
# truth : length-n true CATE.  idx : rows of the target population.
as_obs_by_draw <- function(samp, n_obs) {
  if (nrow(samp) == n_obs) samp else t(samp)
}
eval_cate <- function(point, samp, truth, idx) {
  p  <- point[idx]; tr <- truth[idx]
  M  <- as_obs_by_draw(samp, length(point))[idx, , drop = FALSE]  # n_idx x draws
  ci <- apply(M, 1, quantile, probs = c(0.025, 0.975))            # 2 x n_idx
  c(rmse     = sqrt(mean((p - tr)^2)),
    bias     = mean(p - tr),
    coverage = mean(tr >= ci[1, ] & tr <= ci[2, ]),
    width    = mean(ci[2, ] - ci[1, ]),
    # mean posterior variance of the CATE over evaluation points: the model's
    # own uncertainty per estimate (within-fit), distinct from the across-
    # replication estimator variance computed in post-processing.
    postvar  = mean(apply(M, 1, var)))
}

metric_cols <- c("rmse", "bias", "coverage", "width", "postvar")

# Error-law recovery: Wasserstein-1 distance between the posterior-mean fitted
# error CDF of one source and the true error CDF, both on the standardised data
# scale (the scale of the y passed to the fit).  The CATE barely depends on the
# residual model, so this is the direct measure of what the error prior does.
#   eps_std : large sample of the true error of that source, divided by the
#             dataset's scale_sd.
#   Groups in the fit: 1 = RWD, 2 = RCT; "shared_dp" has one group for both;
#   "gaussian" has no mixture fields.  dp_locations and dp_sigma_g are already on
#   the response scale; fit$sigma is on the package's internal scale and needs
#   fit$meta$sigma_hat.  Draws are thinned to at most `max_draws`.
w1_error <- function(fit, source, eps_std, grid_n = 300L, max_draws = 250L) {
  if (is.null(fit)) return(NA_real_)
  sh <- fit$meta$sigma_hat
  if (is.null(fit$dp_mix_prop)) {                      # gaussian
    sg <- fit$sigma * sh
    W  <- matrix(1, length(sg), 1); L <- matrix(0, length(sg), 1)
  } else {
    g  <- if (length(fit$dp_mix_prop) == 1L) 1L else if (source == "rwd") 1L else 2L
    W  <- fit$dp_mix_prop[[g]]; L <- fit$dp_locations[[g]]
    sg <- if (!is.null(fit$dp_sigma_g)) fit$dp_sigma_g[[g]] else fit$sigma * sh
  }
  idx <- unique(round(seq(1, nrow(W), length.out = min(nrow(W), max_draws))))
  rng <- quantile(eps_std, c(0.0005, 0.9995))
  x   <- seq(rng[1] - 1, rng[2] + 1, length.out = grid_n)
  Ffit <- rowMeans(vapply(idx, function(b)
    colSums(W[b, ] * pnorm(outer(L[b, ], x, function(t, xx) (xx - t) / sg[b]))),
    numeric(grid_n)))
  sum(abs(Ffit - ecdf(eps_std)(x))) * (x[2] - x[1])
}

make_rows <- function(method, point, samp, truth, n_rct, n_all) {
  pops <- list(RCT = seq_len(n_rct),
               RWD = (n_rct + 1L):n_all,
               All = seq_len(n_all))
  do.call(rbind, lapply(names(pops), function(pop) {
    vals <- if (is.null(point)) setNames(rep(NA_real_, length(metric_cols)), metric_cols)
    else eval_cate(point, samp, truth, pops[[pop]])
    data.frame(method = method, population = pop, as.list(vals),
               stringsAsFactors = FALSE)
  }))
}

# Error-distribution priors compared (the HDPM protagonist plus two baselines).
# "source_dp_scale" is the independent per-source mixture (own atoms, weights
# and scale per source, no sharing): the comparator that separates source-
# specific shapes from hierarchical sharing (referee 1, comment 7).
error_priors <- c("gaussian", "shared_dp", "source_dp_scale", "source_hdp_scale")

# Reconstruct the full-outcome posterior sample (draws x n) at the observed
# (A, S) from the stored component sample arrays, mirroring the backend's
# total = control + (1 - S) * deviation + A * (treat + (1 - S) * deconf).
# Returns NULL if any component sample is missing.
total_sample <- function(ff, A, S) {
  need <- c("test_predictions_sample_control", "test_predictions_sample_treat",
            "test_predictions_sample_deconf",  "test_predictions_sample_deviation")
  if (!all(need %in% names(ff))) return(NULL)
  oms <- 1 - S
  ff$test_predictions_sample_control +
    sweep(ff$test_predictions_sample_deviation, 2, oms, `*`) +
    sweep(ff$test_predictions_sample_treat +
            sweep(ff$test_predictions_sample_deconf, 2, oms, `*`), 2, A, `*`)
}

# ── One replication: the fusion forest under each error prior, one dataset ────
# For each prior we record two targets: the CATE tau(X) (treatment contrast) and
# the full conditional-mean outcome E[log T | X, A, S] (sum of forests).  The
# outcome target does not cancel the shared baseline m0, so it is the more
# sensitive probe of the residual model.
run_one_sim <- function(seed, n_rct, n_rwd, N_post, N_burn, setting,
                        rate_rct, visits, eps_true) {
  set.seed(seed)
  d     <- make_data(n_rct, n_rwd, setting, rate_rct, visits)
  n_all <- n_rct + n_rwd
  safe  <- function(expr) tryCatch(expr, error = function(e) NULL)

  # Fits run on the standardised log scale: CATE = treat-forest * scale_sd; the
  # full outcome = total prediction * scale_sd + scale_mu.
  s  <- d$scale_sd; mu <- d$scale_mu; A <- d$A; S <- d$S
  truth_cate <- tau(d$X)
  truth_out  <- m0(d$X) + A * tau(d$X)   # full conditional mean at observed (A, S); lambda = 0

  res <- do.call(rbind, lapply(error_priors, function(ed) {
    ff <- safe(fit_fusion(d, N_post, N_burn, ed))
    # CATE (treatment-effect contrast; A-independent).
    rc <- make_rows(ed,
            if (!is.null(ff)) ff$test_predictions_treat * s,
            if (!is.null(ff)) ff$test_predictions_sample_treat * s,
            truth_cate, n_rct, n_all)
    rc$target <- "cate"
    # Full-outcome conditional mean (sum of forests) at observed (A, S).
    op <- os <- NULL
    if (!is.null(ff)) {
      ts <- total_sample(ff, A, S)
      if (!is.null(ts)) { op <- ff$test_predictions * s + mu; os <- ts * s + mu }
    }
    ro <- make_rows(ed, op, os, truth_out, n_rct, n_all)
    ro$target <- "outcome"
    out <- rbind(rc, ro)
    # Error-law recovery per source (same value on every row of this prior).
    out$w1_rct <- w1_error(ff, "rct", eps_true$rct / s)
    out$w1_rwd <- w1_error(ff, "rwd", eps_true$rwd / s)
    out
  }))
  # realised censoring sanity-checks
  res$cens_rct    <- mean(d$st_rct == 0)            # RCT right-censored
  res$ic_rwd      <- mean(d$icc_rwd == 1)           # RWD interval-censored
  res$rc_rwd_tail <- mean(d$icc_rwd == 0)           # RWD right-censored past last visit
  res
}

# ── Parallel runner over replications (one error setting) ────────────────────
run_simulation_tidy <- function(n_rep, n_rct, n_rwd, N_post, N_burn,
                                setting, seed_offset = 1000L) {
  cal <- calibrate(setting)       # per-setting RCT rate + RWD quartile visits (fixed)
  cat(sprintf("  RCT cens rate = %.4g | RWD visits (q25..q75) = %s\n",
              cal$rate_rct, paste(round(cal$visits, 2), collapse = ", ")))
  # Reference samples of the true errors (log-time scale), fixed per setting
  # and drawn before the workers fork, so they do not touch the replication
  # seeds.  Each replication divides by its own scale_sd.
  set.seed(seed_offset)
  eps_true <- list(rct = draw_eps(1e5, setting, "rct"),
                   rwd = draw_eps(1e5, setting, "rwd"))
  foreach(
    i = seq_len(n_rep),
    .combine = "rbind",
    .packages = c("FusionForests", "ShrinkageTrees", "evd", "MASS")
  ) %dopar% {
    res <- run_one_sim(seed_offset + i, n_rct, n_rwd, N_post, N_burn, setting,
                       cal$rate_rct, cal$visits, eps_true)
    res$setting <- setting
    res$n_rct   <- n_rct
    res$Iter    <- i
    res
  }
}

# ── Main (HPC) ────────────────────────────────────────────────────────────────
args <- commandArgs(trailingOnly = TRUE)
if (length(args) > 0) {
  num_cores <- as.integer(args[1]) - 1L
} else {
  num_cores <- parallel::detectCores() - 1L
}
registerDoParallel(cores = num_cores)
cat("Number of cores being used (1 free):", num_cores, "\n")
cat("SIMULATION: error-distribution prior (sim_err_v3): FusionForest under",
    "gaussian / shared_dp / source_dp_scale / source_hdp_scale -- HPC\n")
cat("lambda_u = lambda_d = 0 (sources differ only in error distribution);",
    "per-source scale gap + heavy coarse censoring to stress the error model.\n")
cat("RWD interval-censored at quartiles q25/q50/q75; RCT right-censored ~50%.\n")
cat("Covariates: MVN AR(1), rho =", rho_X, "\n")

# Simulation settings.
M       <- 1000             # replications per setting (start small)
n_rct   <- 150L                  # default trial size (cells below override)
n_rwd   <- 350L
N_post  <- 5000L
N_burn  <- 5000L
cat(sprintf("M = %d reps/setting, n_rct = %d, n_rwd = %d, N_post = %d, N_burn = %d\n",
            M, n_rct, n_rwd, N_post, N_burn))

# Cells: the six error settings at n_rct = 150, plus settings 5 (shared
# components) and 6 (no shared components) with a small trial, n_rct = 50.
cells <- data.frame(setting = c(1:6, 5L, 6L),
                    n_rct   = c(rep(n_rct, 6L), 50L, 50L))
cat(sprintf("Cells: %s\n", paste(sprintf("%d/n=%d", cells$setting, cells$n_rct),
                                 collapse = ", ")))

all_cells <- vector("list", nrow(cells))
for (g in seq_len(nrow(cells))) {
  setting <- cells$setting[g]
  cat(sprintf("\nCell %d/%d (error_setting = %d, n_rct = %d)\n",
              g, nrow(cells), setting, cells$n_rct[g]))
  all_cells[[g]] <- run_simulation_tidy(
    n_rep = M, n_rct = cells$n_rct[g], n_rwd = n_rwd,
    N_post = N_post, N_burn = N_burn,
    setting = setting, seed_offset = 1000L * g)
}
final_flat_df <- do.call(rbind, all_cells)

# Output file path (! NAME MUST BE FILENAME_output.rds !).
output_file <- file.path(Sys.getenv("TMPDIR"), "sim_err_v3_output.rds")
cat("\nSaving all results to:", output_file, "\n")
saveRDS(final_flat_df, file = output_file)
cat("All results successfully saved in one file.\n")

# Local analysis (tables, figures): analyse_err_v3.R in this folder.
