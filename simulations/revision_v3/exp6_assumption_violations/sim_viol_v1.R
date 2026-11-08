# ══════════════════════════════════════════════════════════════════════════════
# sim_viol_v1.R -- experiment 6: the Bayesian fusion forest when its assumptions
# fail (referee 1, comments 3, 6 and 7; referee 2, comments 4 and 5).
#
# The main designs satisfy Assumptions 1-6 by construction.  This experiment
# starts from the design of experiment 1 (sim_surv_v13.R) at lambda_d =
# lambda_u = 1 with constant delta(X) and changes one feature at a time:
#
#   Error law, Assumption 6
#     logistic_lu1, logistic_lu2  U ~ N(0, 1) marginally with logistic selection
#                                 A ~ Bernoulli[expit{X1 + (1 + X2 / 2) U}], at
#                                 lambda_u = 1, 2: the law of U given (A, X)
#                                 differs by more than a shift (mild departure);
#     trt_scale_1.5, trt_scale_2  error scale multiplied by 1.5 or 2 in the
#                                 treated arm of both sources: the mean contrast
#                                 stays tau(X), treatment is no location shift;
#     x_scale                     error scale multiplied by exp(X1 / 2 - 1 / 4)
#                                 in both sources (unit mean square).
#   Short follow-up
#     short_followup              administrative trial censoring at a fixed time,
#                                 75% censored; all assumptions hold, so this
#                                 tests that the shift is identified from the
#                                 follow-up window;
#     short_trt_scale_1.5         the same with the treated-arm scale 1.5: the
#                                 early effect then differs from the mean effect.
#   Transportability, Assumption 4
#     transport_0.5, transport_1  the RWD effect is tau(X) + eta(X), with
#                                 eta(X) = kappa (1 + X2) / 2, kappa = 0.5, 1.
#                                 tau-hat is scored against tau (the trial CATE)
#                                 and against tau + eta on the RWD rows; c-hat
#                                 against eta + lu delta (Web Appendix A.3).
#   Overlap, Assumption 3
#     overlap_two_sided           trial restricted to |X1| <= 1;
#     overlap_one_sided           trial restricted to X1 <= 0.  The RWD is
#                                 unrestricted.  RWD rows are also scored inside
#                                 and outside the trial support.
#   Informative censoring, Assumption 5.  r is the outcome residual, standardized
#   to unit variance.  The trial censoring rate is multiplied by exp(-kappa r);
#   an RWD patient drops out after each visit with probability
#   expit{logit(0.05) - kappa r} and is right-censored at the last attended visit.
#     dropout_null                kappa = 0: dropout that is not informative;
#     dropout_informative         kappa = 1 in both arms;
#     dropout_differential        kappa = 1 in the treated arm only.
#
# Fits and priors as in experiment 1: fusion forest, RCT-only and RWD-only.
# Recorded (long format, one row per cell x rep x method x target x population):
# rmse, bias, coverage, width, postvar.  Targets: "tau1" (tau, all methods),
# "tau0" (tau + eta on the RWD rows, transport cells only), "c" and "d" (fusion
# only, on the RWD rows).  Populations: RCT, RWD, All, and RWD-in, RWD-out in the
# overlap cells.
#
# The HPC job only stores the output (HPC contract: <base_name>_output.rds);
# analysis is local (analyse_viol_v1.R).
# ══════════════════════════════════════════════════════════════════════════════
library(FusionForests)
library(ShrinkageTrees)
library(doParallel)
library(foreach)
library(evd)
library(MASS)

# ── DGP ───────────────────────────────────────────────────────────────────────
p_total      <- 10L              # observed covariates
lambda_d     <- 1.0              # between-source heterogeneity
sigma_rct    <- 0.75             # RCT residual SD
target_cens  <- 0.35             # target right-censoring fraction, trial
target_admin <- 0.75             # censored fraction, short follow-up cells
visit_probs  <- seq(0.1, 0.8, 0.1) # RWD inspection visits: deciles q10..q80
drop_p0      <- 0.05             # RWD dropout probability per visit at r = 0

# Covariates: multivariate normal with AR(1) dependence, Sigma_ij = rho^|i-j|.
rho_X    <- 0.3
Sigma_X  <- rho_X ^ abs(outer(seq_len(p_total), seq_len(p_total), "-"))
draw_X   <- function(n) MASS::mvrnorm(n, mu = rep(0, p_total), Sigma = Sigma_X)

m0   <- function(X)  (2 * X[, 1] - X[, 2] * X[, 3] + 0.5 * X[, 4]^2)
dev  <- function(X, ld) ld * (X[, 4] - 0.5 * X[, 5])  # RWD baseline deviation
tau  <- function(X)  (1/2 + X[, 1] - 0.5 * X[, 2]^2)  # true CATE (trial)
delta_u <- 0.84                                # main design: c(X) = 0.84 * lu

# ── Scenarios ─────────────────────────────────────────────────────────────────
new_cell <- function(cell, block, lu = 1, u_design = "normal", trt_scale = 1,
                     x_scale = 0, admin = FALSE, eta = 0, support = "none",
                     drop_kappa = NA_real_, drop_diff = FALSE) {
  list(cell = cell, block = block, lu = lu, u_design = u_design,
       trt_scale = trt_scale, x_scale = x_scale, admin = admin, eta = eta,
       support = support, drop_kappa = drop_kappa, drop_diff = drop_diff)
}
cells <- list(
  new_cell("reference", "Reference"),
  new_cell("logistic_lu1", "Error law", u_design = "logistic"),
  new_cell("logistic_lu2", "Error law", lu = 2, u_design = "logistic"),
  new_cell("trt_scale_1.5", "Error law", trt_scale = 1.5),
  new_cell("trt_scale_2", "Error law", trt_scale = 2),
  new_cell("x_scale", "Error law", x_scale = 0.5),
  new_cell("short_followup", "Short follow-up", admin = TRUE),
  new_cell("short_trt_scale_1.5", "Short follow-up", admin = TRUE,
           trt_scale = 1.5),
  new_cell("transport_0.5", "Transportability", eta = 0.5),
  new_cell("transport_1", "Transportability", eta = 1),
  new_cell("overlap_two_sided", "Overlap", support = "two_sided"),
  new_cell("overlap_one_sided", "Overlap", support = "one_sided"),
  new_cell("dropout_null", "Informative censoring", drop_kappa = 0),
  new_cell("dropout_informative", "Informative censoring", drop_kappa = 1),
  new_cell("dropout_differential", "Informative censoring", drop_kappa = 1,
           drop_diff = TRUE)
)

# Multiplier of the error scale.  x_scale gives unit mean square over X1.
err_factor <- function(X, A, cl) {
  cl$trt_scale^A * exp(cl$x_scale * X[, 1] - cl$x_scale^2)
}
# Effect difference between the sources (Assumption 4 fails if eta != 0).
eta_x <- function(X, cl) cl$eta * (1 + X[, 2]) / 2
# Trial support.  The trial is restricted by redrawing the rows outside it.
in_support <- function(X, support) {
  switch(support,
         none      = rep(TRUE, nrow(X)),
         two_sided = abs(X[, 1]) <= 1,
         one_sided = X[, 1] <= 0)
}
draw_X_trial <- function(n, support) {
  X   <- draw_X(n)
  out <- !in_support(X, support)
  while (any(out)) {
    X[out, ] <- draw_X(sum(out) + 1L)[seq_len(sum(out)), ]
    out <- !in_support(X, support)
  }
  X
}
# Strength of the dependence of censoring on the residual, per patient.
kappa_i <- function(A, cl) {
  if (is.na(cl$drop_kappa)) return(rep(0, length(A)))
  if (cl$drop_diff) cl$drop_kappa * A else rep(cl$drop_kappa, length(A))
}

# Latent log-survival generators (no censoring), one per source.  `res` is the
# outcome residual standardized to unit variance before err_factor; only the
# dropout cells use it, and these have u_design = "normal".
latent_rct <- function(n, cl) {
  X <- draw_X_trial(n, cl$support)
  A <- rbinom(n, 1L, 0.5)                              # randomised
  z <- rnorm(n)
  list(X = X, A = A, res = z,
       logT = m0(X) + A * tau(X) + sigma_rct * err_factor(X, A, cl) * z)
}
latent_rwd <- function(n, cl) {
  gamma_em   <- 0.5772156649
  beta_scale <- sqrt(6) / pi
  mu_loc     <- -beta_scale * gamma_em
  X <- draw_X(n)
  if (cl$u_design == "logistic") {
    U <- rnorm(n)
    A <- rbinom(n, 1L, plogis(X[, 1] + (1 + 0.5 * X[, 2]) * U))
  } else {
    A <- rbinom(n, 1L, plogis(X[, 1]))
    U <- rnorm(n, mean = A * delta_u)
  }
  eps <- -rgumbel(n, loc = mu_loc, scale = beta_scale) # mean-0 residual (SD 1)
  list(X = X, A = A,
       res  = (cl$lu * (U - A * delta_u) + eps) / sqrt(cl$lu^2 + 1),
       logT = m0(X) + dev(X, lambda_d) + A * (tau(X) + eta_x(X, cl)) +
              cl$lu * U + err_factor(X, A, cl) * eps)  # U in both arms
}

# Calibrate the RCT exponential censoring rate (time scale) to ~target.
# P(cens) = E[1 - exp(-rate_i * T)], monotone increasing in rate.
solve_cens_rate <- function(logT, res, kappa, target = target_cens) {
  Tt <- exp(logT)
  uniroot(function(r) mean(1 - exp(-r * exp(-kappa * res) * Tt)) - target,
          c(1e-10, 1e6))$root
}

# Per-cell calibration, fixed once on a large pre-simulation:
#   rate_rct   -- RCT exponential censoring rate (not in the short follow-up cells).
#   admin_time -- RCT administrative censoring time (short follow-up cells only).
#   visits     -- RWD inspection times (time scale) at the deciles q10..q80.
calibrate <- function(cl, n_cal = 2e5) {
  r <- latent_rct(n_cal, cl)
  w <- latent_rwd(n_cal, cl)
  list(
    rate_rct   = if (cl$admin) NA_real_
                 else solve_cens_rate(r$logT, r$res, kappa_i(r$A, cl)),
    admin_time = if (cl$admin) as.numeric(quantile(exp(r$logT), 1 - target_admin))
                 else NA_real_,
    visits     = as.numeric(quantile(exp(w$logT), probs = visit_probs))
  )
}

# Index of the last visit an RWD patient attends.
last_visit_index <- function(A, res, cl, n_visits) {
  if (is.na(cl$drop_kappa)) return(rep(n_visits, length(A)))
  p <- plogis(qlogis(drop_p0) - kappa_i(A, cl) * res)
  pmin(rgeom(length(A), p) + 1L, n_visits)
}

# Simulate one full dataset (RCT right-censored, RWD interval-censored).
# All time bounds are returned on the LOG scale (timescale = "log").
make_data <- function(n_rct, n_rwd, cl, cal) {
  r <- latent_rct(n_rct, cl)
  w <- latent_rwd(n_rwd, cl)
  # Standardise the true (latent) log-survival of the pooled sources to mean 0,
  # unit variance.  The same affine map `std` is applied to every log-time bound
  # constructed below, so all fits operate on a common scale.  Predicted CATEs
  # are on this scale and are rescaled by `scale_sd` before evaluation.
  logT_true <- c(r$logT, w$logT)
  scale_mu  <- mean(logT_true)
  scale_sd  <- sd(logT_true)
  std       <- function(z) (z - scale_mu) / scale_sd
  logfloor <- min(c(r$logT, w$logT)) - 5    # finite stand-in for log(0)

  # --- RCT: exponential or administrative right censoring ---
  logC <- if (cl$admin) {
    rep(log(cal$admin_time), n_rct)
  } else {
    log(rexp(n_rct, rate = cal$rate_rct * exp(-kappa_i(r$A, cl) * r$res)))
  }
  y_rct  <- pmin(r$logT, logC)
  st_rct <- as.integer(r$logT <= logC)
  icc_rct   <- rep(0L, n_rct)               # 0 = not interval-censored
  left_rct  <- y_rct                        # ignored for status==1 / right-cens
  right_rct <- y_rct

  # --- RWD: interval censoring on the decile visits, up to the last attended ---
  visits     <- cal$visits
  T0         <- exp(w$logT)
  last_visit <- visits[last_visit_index(w$A, w$res, cl, length(visits))]
  breaks     <- c(0, visits)                # bin edges: 0, q10, ..., q80
  j          <- findInterval(T0, breaks)    # 1..length(visits) below q80
  beyond     <- T0 >= last_visit            # event after the last attended visit
  j[!beyond] <- pmax(j[!beyond], 1L)

  left_rwd  <- ifelse(j == 1L, logfloor, log(breaks[pmin(j, length(breaks))]))
  right_rwd <- log(breaks[pmin(j + 1L, length(breaks))])
  st_rwd    <- rep(0L, n_rwd)               # no exact event times in the RWD
  icc_rwd   <- rep(1L, n_rwd)               # interval-censored by default
  # events beyond the last attended visit: right-censored there (icc = 0)
  left_rwd[beyond]  <- log(last_visit[beyond])
  right_rwd[beyond] <- log(last_visit[beyond])
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
    # SD used to standardise the pooled latent log-survival.  Predicted CATEs
    # come back on the standardised scale and are multiplied by this to recover
    # the log-time scale before evaluation against tau(X).
    scale_sd = scale_sd,
    n_rct = n_rct, n_rwd = n_rwd
  )
}

# ── True c(x) and d(x) on the model's scale ───────────────────────────────────
# Closed form under u_design = "normal", where E[U | A = a, x] = a delta_u.  By
# quadrature over U ~ N(0, 1) under u_design = "logistic".
EU_given_A <- function(x1, g, a, ngrid = 801L) {
  u <- seq(-8, 8, length.out = ngrid); w <- dnorm(u)
  P <- plogis(x1 + outer(g, u))               # n x ngrid
  if (a == 0L) P <- 1 - P
  as.numeric((P %*% (u * w)) / (P %*% w))
}
true_c <- function(X, cl) {
  if (cl$u_design == "logistic") {
    g <- 1 + 0.5 * X[, 2]
    return(cl$lu * (EU_given_A(X[, 1], g, 1L) - EU_given_A(X[, 1], g, 0L)))
  }
  cl$lu * delta_u + eta_x(X, cl)              # c absorbs the effect difference
}
true_d <- function(X, cl) {
  if (cl$u_design == "logistic") {
    return(dev(X, lambda_d) +
             cl$lu * EU_given_A(X[, 1], 1 + 0.5 * X[, 2], 0L))
  }
  dev(X, lambda_d)
}

# ── Fits (all via FusionForest, log-time, with interval-censoring args) ───────
fit_fusion <- function(d, N_post, N_burn) {
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
    treatment_indicator_test  = rep(1L, nrow(d$X)),
    source_indicator_test     = rep(1L, nrow(d$X)),
    outcome_type              = "right-censored",
    timescale                 = "log",
    error_dist                = "source_hdp_scale",
    number_of_trees_control = 200,
    number_of_trees_treat = 100,
    number_of_trees_deconf = 50,
    number_of_trees_deviation = 50,
    k_treat                   = 1,     # k_tau = 1, the package default
    N_post = N_post, N_burn = N_burn,
    treatment_coding = "binary",
    store_posterior_sample = TRUE, verbose = FALSE)
}

# ── Single-source fits (CausalShrinkageForest), as in experiment 1 ────────────
# Single-source priors as in the submission: leaf SD local_hp = 1 / sqrt(J)
# (k = 2 on the paper's scale) and (alpha, beta) = (0.95, 2) for both forests.
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
# beyond the last attended visit have right = Inf (right-censored).
fit_single_ic <- function(left_time, right_time, X, A, Xev, N_post, N_burn) {
  CausalShrinkageForest(
    left_time                 = left_time,
    right_time                = right_time,
    X_train_control           = X,
    X_train_treat             = X,
    treatment_indicator_train = A,
    X_test_control            = Xev,
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
# point : length-n posterior mean on the combined set.
# samp  : posterior draws.  We coerce to n_obs x N_post by matching the axis
#         whose length equals length(point).
# truth : length-n true function.  idx : rows of the target population.
as_obs_by_draw <- function(samp, n_obs) {
  if (nrow(samp) == n_obs) samp else t(samp)
}
eval_fun <- function(point, samp, truth, idx) {
  p  <- point[idx]; tr <- truth[idx]
  M  <- as_obs_by_draw(samp, length(point))[idx, , drop = FALSE]  # n_idx x draws
  ci <- apply(M, 1, quantile, probs = c(0.025, 0.975))            # 2 x n_idx
  c(rmse     = sqrt(mean((p - tr)^2)),
    bias     = mean(p - tr),
    coverage = mean(tr >= ci[1, ] & tr <= ci[2, ]),
    width    = mean(ci[2, ] - ci[1, ]),
    postvar  = mean(apply(M, 1, var)))
}

metric_cols <- c("rmse", "bias", "coverage", "width", "postvar")

make_rows <- function(method, target, point, samp, truth, pops) {
  do.call(rbind, lapply(names(pops), function(pop) {
    vals <- if (is.null(point)) setNames(rep(NA_real_, length(metric_cols)), metric_cols)
    else eval_fun(point, samp, truth, pops[[pop]])
    data.frame(method = method, target = target, population = pop, as.list(vals),
               stringsAsFactors = FALSE)
  }))
}

# ── One replication: all three estimators on one dataset ─────────────────────
run_one_sim <- function(seed, n_rct, n_rwd, N_post, N_burn, cl, cal) {
  set.seed(seed)
  d     <- make_data(n_rct, n_rwd, cl, cal)
  n_all <- n_rct + n_rwd
  i_rct <- seq_len(n_rct)
  i_rwd <- n_rct + seq_len(n_rwd)
  safe  <- function(expr) tryCatch(expr, error = function(e) NULL)

  # Populations.  The overlap cells split the RWD rows by the trial support.
  pops_rwd <- list(RWD = i_rwd)
  if (cl$support != "none") {
    inside   <- in_support(d$X_rwd, cl$support)
    pops_rwd <- c(pops_rwd, list("RWD-in"  = i_rwd[inside],
                                 "RWD-out" = i_rwd[!inside]))
  }
  pops <- c(list(RCT = i_rct, All = seq_len(n_all)), pops_rwd)

  # RWD interval2 bounds: keep finite (left, right] for interval-censored rows;
  # set right = Inf for right-censored rows.
  lt_rwd <- d$left_rwd
  rt_rwd <- d$right_rwd
  rt_rwd[d$icc_rwd == 0] <- Inf

  ff  <- safe(fit_fusion(d, N_post, N_burn))
  rct <- safe(fit_single_rc(d$y_rct, d$st_rct, d$X_rct, d$A_rct, d$X,
                            N_post, N_burn))
  rwd <- safe(fit_single_ic(lt_rwd, rt_rwd, d$X_rwd, d$A_rwd, d$X,
                            N_post, N_burn))

  # The fits run on the standardised log scale.  Multiply back by scale_sd to
  # recover the log-time scale before comparing with the truth.
  s  <- d$scale_sd
  sc <- function(x) if (is.null(x)) NULL else x * s
  tau_t <- tau(d$X)
  est <- list("Fusion"   = list(sc(ff$test_predictions_treat),
                                sc(ff$test_predictions_sample_treat)),
              "RCT-only" = list(sc(rct$test_predictions_treat),
                                sc(rct$test_predictions_sample_treat)),
              "RWD-only" = list(sc(rwd$test_predictions_treat),
                                sc(rwd$test_predictions_sample_treat)))
  res <- do.call(rbind, lapply(names(est), function(m) {
    rows <- make_rows(m, "tau1", est[[m]][[1]], est[[m]][[2]], tau_t, pops)
    if (cl$eta != 0) {                         # the RWD effect differs from tau
      rows <- rbind(rows, make_rows(m, "tau0", est[[m]][[1]], est[[m]][[2]],
                                    tau_t + eta_x(d$X, cl), pops_rwd))
    }
    rows
  }))
  res <- rbind(
    res,
    make_rows("Fusion", "c", sc(ff$test_predictions_deconf),
              sc(ff$test_predictions_sample_deconf), true_c(d$X, cl), pops_rwd),
    make_rows("Fusion", "d", sc(ff$test_predictions_deviation),
              sc(ff$test_predictions_sample_deviation), true_d(d$X, cl), pops_rwd)
  )
  # realised censoring sanity-checks
  res$cens_rct    <- mean(d$st_rct == 0)            # RCT right-censored
  res$ic_rwd      <- mean(d$icc_rwd == 1)           # RWD interval-censored
  res$rc_rwd_tail <- mean(d$icc_rwd == 0)           # RWD right-censored
  res
}

# ── Parallel runner over replications (one scenario) ─────────────────────────
run_simulation_tidy <- function(n_rep, n_rct, n_rwd, N_post, N_burn, cl,
                                seed_offset) {
  set.seed(seed_offset)
  cal <- calibrate(cl)            # per-cell censoring and RWD visits (fixed)
  cat(sprintf("  RCT cens rate = %s | admin time = %s | RWD visits = %s\n",
              format(cal$rate_rct, digits = 4), format(cal$admin_time, digits = 4),
              paste(round(cal$visits, 2), collapse = ", ")))
  foreach(
    i = seq_len(n_rep),
    .combine = "rbind",
    .packages = c("FusionForests", "ShrinkageTrees", "evd", "MASS")
  ) %dopar% {
    res <- run_one_sim(seed_offset + i, n_rct, n_rwd, N_post, N_burn, cl, cal)
    res$cell  <- cl$cell
    res$block <- cl$block
    res$Iter  <- i
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
cat("SIMULATION: violations of the assumptions (sim_viol_v1, FusionForests",
    as.character(packageVersion("FusionForests")), ") -- HPC\n")

M      <- 1000L                  # replications per scenario
n_rct  <- 150L
n_rwd  <- 350L
N_post <- 5000L
N_burn <- 5000L
cat(sprintf("M = %d reps/cell, %d cells, n_rct = %d, n_rwd = %d, N_post = %d, N_burn = %d\n",
            M, length(cells), n_rct, n_rwd, N_post, N_burn))

all_cells <- vector("list", length(cells))
for (g in seq_along(cells)) {
  cat(sprintf("\nCell %d/%d: %s\n", g, length(cells), cells[[g]]$cell))
  all_cells[[g]] <- run_simulation_tidy(
    n_rep = M, n_rct = n_rct, n_rwd = n_rwd, N_post = N_post, N_burn = N_burn,
    cl = cells[[g]], seed_offset = 100000L + 1000L * g)
}
final_flat_df <- do.call(rbind, all_cells)

# Output file path (! NAME MUST BE <base_name>_output.rds !).
out_dir <- Sys.getenv("TMPDIR")
if (!nzchar(out_dir)) out_dir <- "."   # interactive run: current directory
output_file <- file.path(out_dir, "sim_viol_v1_output.rds")
cat("\nSaving all results to:", output_file, "\n")
saveRDS(final_flat_df, file = output_file)
cat("All results successfully saved in one file.\n")

# Local analysis (tables): analyse_viol_v1.R in this folder.
