# ══════════════════════════════════════════════════════════════════════════════
# sim_hd_v2c.R -- growing covariate dimension p (tile c of the p grid), CATE
# Revision round (manuscript_v3).  Derived from submission_v2/exp3_covariate_dimension/sim_hd_v1c.R.
#
# Changes for the revision:
#   * Unmeasured confounder U enters the RWD outcome in BOTH arms, so
#     T(1) - T(0) = tau(X) for every unit and the CATE is common to the two
#     sources (referee 1, comment 2).  In the RWD U | A, X ~ N(A delta(X), 1),
#     so U drives treatment choice and the residual law is free of (X, A),
#     with delta optionally covariate-dependent (comment 4).
#   * Fits call the exported FusionForest() / CausalShrinkageForest() with the
#     priors k = 1 for every forest, including k_tau = 1, on
#     sigma_h = k / (2 sqrt J); the
#     frozen package copies of the submission scripts are gone, so the package's
#     own outcome standardisation is used, as the paper describes.
#   * The RWD-only comparator no longer receives the true propensity score,
#     which was a function of the unmeasured U.
# Output: $TMPDIR/sim_hd_v2c_output.rds (HPC contract, see ../../hpc_job_skeleton.sh).
# ══════════════════════════════════════════════════════════════════════════════
library(FusionForests)
library(ShrinkageTrees)
library(doParallel)
library(foreach)
library(evd)
library(MASS)

# ── DGP ───────────────────────────────────────────────────────────────────────
# High-dimensional experiment (sim_hd_v2): hold the signal fixed and grow the
# number of observed covariates p.  The active functions below touch only the
# first `p_active` columns, so appending columns adds pure noise covariates --
# the outcome distribution, SNR, censoring rates, and the standardisation scale
# are all invariant to p.  The only thing that changes is the number of
# irrelevant split candidates each forest must search.
p_active <- 5L                                # signal lives in X[, 1:5]
p_grid   <- unique(round(10 ^ seq(log10(5), log10(500), length.out = 20)))[14:19]
# swept covariate dimension:
# 10 log-spaced points, 5 -> 250

# Nuisances held at their worst-case value while p is swept.
lambda_d_fixed <- 1.0
lambda_u_fixed <- 1.0
sigma_rct   <- 0.75              # RCT residual SD
target_cens <- 0.35              # target right-censoring fraction per source
visit_probs <- seq(0.1, 0.8, 0.1) # RWD inspection visits: deciles q10..q80


# Covariates: BLOCK-dependent standard normals.  Sigma is block-diagonal with
# blocks of size `block_size`; within a block the correlation is AR(1) with
# parameter rho_X, and covariates in different blocks are independent.  The
# signal lives in X[, 1:5], so the active covariates are correlated with each
# other and with the rest of the first block.  Because each block is at most
# block_size x block_size, the per-block Cholesky factor stays small as p grows
# -- the draw never builds a full p x p Cholesky, which keeps it cheap at high p.
# Marginals remain standard normal (diag(Sigma) = 1), so SNR, censoring
# calibration, and the standardisation scale are unchanged from the independent
# run; only the dependence structure differs.
rho_X      <- 0.3
block_size <- 10L
make_Sigma <- function(p) rho_X ^ abs(outer(seq_len(p), seq_len(p), "-"))  # full AR(1); unused
make_block <- function(b) rho_X ^ abs(outer(seq_len(b), seq_len(b), "-"))  # one AR(1) block
chol_full  <- chol(make_block(block_size))   # reused for every full-size block

draw_X <- function(n, p) {
  X <- matrix(rnorm(n * p), n, p)            # iid N(0,1) start
  for (s in seq.int(1L, p, by = block_size)) {
    idx <- s:min(s + block_size - 1L, p)
    L   <- if (length(idx) == block_size) chol_full else chol(make_block(length(idx)))
    X[, idx] <- X[, idx, drop = FALSE] %*% L # rows now N(0, block Sigma)
  }
  X
}

m0   <- function(X)  (2 * X[, 1] - X[, 2] * X[, 3] + 0.5 * X[, 4]^2)
dev  <- function(X, ld) ld * (X[, 4] - 0.5 * X[, 5])  # RWD baseline deviation
tau  <- function(X)  (1/2 + X[, 1] - 0.5 * X[, 2]^2)         # true CATE
# Unmeasured confounder U: a prognostic factor that enters the RWD outcome in
# BOTH arms (no A x U interaction), so T(1) - T(0) = tau(X) for every unit and
# the CATE is common to the two sources (Assumption 4).  In the RWD we draw
#   A | X ~ Bernoulli{expit(X1)},   U | A, X ~ N(A delta(X), 1).
# By Bayes' rule logit P(A = 1 | X, U) = X1 + delta(X) U - delta(X)^2 / 2, so U
# drives treatment choice and is a confounder there; in the trial treatment is
# randomised and U would only be part of the error.  The RWD contrast is
# tau(X) + c(X) with c(X) = lu * delta(X), pure confounding bias, and the RWD
# baseline is m0(X) + dev(X, ld).  The residual of log T given (X, A) is
# lu * {U - A delta(X)} + eps, whose law is free of (X, A) (Assumption 6).
conf      <- function(U, lu) lu * U
delta_u   <- 0.84                              # uniform design: c(X) = 0.84 * lu
delta_het <- c(0.78, 0.31)                     # covariate-dependent design
# conf_design: "uniform" -> delta(X) = delta_u, so c is constant in X;
# "heterogeneous" -> delta(X) = 0.78 + 0.31 X2, so c(X) has mean 0.78 * lu, SD
# 0.31 * lu and runs from 0.27 * lu to 1.29 * lu (5th to 95th percentile).
# Set per grid cell in the main loop; forked workers inherit the global.
conf_design <- "uniform"
delta_x <- function(X) {
  if (conf_design == "heterogeneous") delta_het[1] + delta_het[2] * X[, 2]
  else delta_u
}

# Latent log-survival generators (no censoring), one per source.
latent_rct <- function(n, p) {
  X <- draw_X(n, p)
  A <- rbinom(n, 1L, 0.5)                              # randomised
  list(X = X, A = A,
       logT = m0(X) + A * tau(X) + rnorm(n, 0, sigma_rct))
}
latent_rwd <- function(n, p, ld, lu) {
  gamma_em   <- 0.5772156649
  beta_scale <- sqrt(6) / pi
  mu_loc     <- -beta_scale * gamma_em
  X <- draw_X(n, p)
  e <- plogis(X[, 1])                                  # P(A = 1 | X): selection on X1
  A <- rbinom(n, 1L, e)
  U <- rnorm(n, mean = A * delta_x(X))                 # unmeasured prognostic factor
  eps <- -rgumbel(n, loc = mu_loc, scale = beta_scale) # mean-0 residual (SD 1)
  list(X = X, A = A, e = e, eps = eps,
       logT = m0(X) + dev(X, ld) + A * tau(X) + conf(U, lu) + eps)  # U in both arms
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
calibrate <- function(ld, lu, p, n_cal = 2e5) {
  rate_rct <- solve_cens_rate(latent_rct(n_cal, p)$logT)
  visits   <- as.numeric(quantile(exp(latent_rwd(n_cal, p, ld, lu)$logT),
                                  probs = visit_probs))
  list(rate_rct = rate_rct, visits = visits)
}

# Simulate one full dataset (RCT right-censored, RWD interval-censored).
# All time bounds are returned on the LOG scale (timescale = "log").
make_data <- function(n_rct, n_rwd, p, ld, lu, rate_rct, visits) {
  r <- latent_rct(n_rct, p)
  w <- latent_rwd(n_rwd, p, ld, lu)
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
    # True RWD propensity score P(A = 1 | X).  e_rwd: training-row
    # propensity for the RWD-only fit.  e_all: propensity on the combined
    # evaluation set, with 0.5 for the randomised RCT rows.
    e_rwd  = w$e,
    e_all  = c(rep(0.5, n_rct), w$e),
    # SD used to standardise the pooled latent log-survival.  Predicted CATEs
    # come back on the standardised scale and are multiplied by this to recover
    # the log-time scale before evaluation against tau(X).
    scale_sd = scale_sd,
    n_rct = n_rct, n_rwd = n_rwd
  )
}

# ── Fits (all via FusionForest, log-time, with interval-censoring args) ───────
# `cols` selects which covariate columns to hand to every forest.  cols = NULL
# uses all p; cols = 1:p_active gives the active-only oracle (the floor that
# removes the dimensionality penalty).
fit_fusion <- function(d, N_post, N_burn, cols = NULL) {
  Xf <- if (is.null(cols)) d$X else d$X[, cols, drop = FALSE]
  # Step-height scales k_f = 1 for every forest, including k_tau = 1 (chosen
  # by a pilot; sigma_h = k / (2 sqrt J)).
  FusionForest(
    y                            = d$y,
    status                       = d$status,
    observed_left_time           = d$left,
    observed_right_time          = d$right,
    interval_censoring_indicator = d$icc,
    X_train_control           = Xf,
    X_train_treat             = Xf,
    treatment_indicator_train = d$A,
    source_indicator_train    = d$S,
    X_test_control            = Xf,
    X_test_treat              = Xf,
    X_test_deconf             = Xf,
    treatment_indicator_test  = rep(1L, nrow(Xf)),
    source_indicator_test     = rep(1L, nrow(Xf)),
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
  # No propensity covariate: the score is unknown in practice (the submission
  # scripts handed the true one to this fit), and an estimated score from X
  # alone adds nothing the forests do not already see.
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

# ── One replication: all estimators on one dataset ───────────────────────────
run_one_sim <- function(seed, n_rct, n_rwd, p, N_post, N_burn, ld, lu,
                        rate_rct, visits) {
  set.seed(seed)
  d     <- make_data(n_rct, n_rwd, p, ld, lu, rate_rct, visits)
  truth <- tau(d$X)
  n_all <- n_rct + n_rwd
  safe  <- function(expr) tryCatch(expr, error = function(e) NULL)
  
  # RWD interval2 bounds: keep finite (left, right] for interval-censored rows;
  # set right = Inf for rows censored beyond the last visit (right-censored).
  lt_rwd <- d$left_rwd
  rt_rwd <- d$right_rwd
  rt_rwd[d$icc_rwd == 0] <- Inf
  
  # The fits run on the standardised log scale, so their treatment-effect
  # predictions are tau(X) / scale_sd.  Multiply back by scale_sd to recover the
  # log-time CATE before comparing with the true tau(X).
  s <- d$scale_sd
  # Fit, score and free one model at a time: each fit carries its full
  # posterior draws, and holding all four at once ran the node out of memory
  # at p = 500.
  score <- function(name, fit) {
    rows <- make_rows(name, if (!is.null(fit)) fit$test_predictions_treat * s,
                      if (!is.null(fit)) fit$test_predictions_sample_treat * s,
                      truth, n_rct, n_all)
    rm(fit); invisible(gc(verbose = FALSE))
    rows
  }
  res <- rbind(
    score("Fusion",        safe(fit_fusion(d, N_post, N_burn))),
    # Active-only oracle: fusion handed just the p_active signal covariates.
    # This removes the dimensionality penalty and traces the achievable floor.
    score("Fusion-oracle", safe(fit_fusion(d, N_post, N_burn, cols = seq_len(p_active)))),
    score("RCT-only",      safe(fit_single_rc(d$y_rct, d$st_rct, d$X_rct, d$A_rct, d$X,
                                              N_post, N_burn))),
    score("RWD-only",      safe(fit_single_ic(lt_rwd, rt_rwd, d$X_rwd, d$A_rwd, d$X,
                                              N_post, N_burn)))
  )
  # realised censoring sanity-checks
  res$cens_rct    <- mean(d$st_rct == 0)            # RCT right-censored
  res$ic_rwd      <- mean(d$icc_rwd == 1)           # RWD interval-censored
  res$rc_rwd_tail <- mean(d$icc_rwd == 0)           # RWD right-censored past q80
  res
}

# ── Parallel runner over replications (one grid cell) ────────────────────────
run_simulation_tidy <- function(n_rep, n_rct, n_rwd, p, N_post, N_burn,
                                ld, lu, seed_offset = 1000L) {
  cal <- calibrate(ld, lu, p)     # per-cell RCT rate + RWD decile visits (fixed)
  cat(sprintf("  RCT cens rate = %.4g | RWD visits (q10..q80) = %s\n",
              cal$rate_rct, paste(round(cal$visits, 2), collapse = ", ")))
  foreach(
    i = seq_len(n_rep),
    .combine = "rbind",
    .packages = c("FusionForests", "ShrinkageTrees", "evd")
  ) %dopar% {
    res <- run_one_sim(seed_offset + i, n_rct, n_rwd, p, N_post, N_burn, ld, lu,
                       cal$rate_rct, cal$visits)
    res$p        <- p
    res$lambda_d <- ld
    res$lambda_u <- lu
    res$Iter     <- i
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
# Workers in parallel.  Each worker needs memory for one fit with its full
# posterior draws.  Lower max_workers if a node runs out of memory; tile d
# (p = 500) runs on a fat_genoa node with all cores.
max_workers <- num_cores
num_cores   <- min(num_cores, max_workers)
registerDoParallel(cores = num_cores)
cat("Number of cores being used (1 free):", num_cores, "\n")
cat("SIMULATION: high-dimensional sweep (sim_hd_v2): Fusion vs Fusion-oracle vs RCT-only vs RWD-only -- HPC\n")
cat("Grow covariate dimension p at fixed n; lambda_d = lambda_u = 1.\n")
cat("RWD interval-censored at deciles q10..q80; RCT right-censored ~35%.\n")
cat(sprintf("Covariates: MVN, AR(1) rho = %g within independent blocks of %d; signal in X[, 1:%d].\n",
            rho_X, block_size, p_active))

# Simulation settings.
M       <- 1000                  # replications per grid cell
n_rct   <- 150
n_rwd   <- 350L
N_post  <- 5000L
N_burn  <- 5000L
cat(sprintf("M = %d reps/cell, n_rct = %d, n_rwd = %d, N_post = %d, N_burn = %d\n",
            M, n_rct, n_rwd, N_post, N_burn))

# Dimension grid: sweep p at the worst-case nuisance setting (lambda_d = lambda_u = 1).
grid <- data.frame(p = p_grid, lambda_d = lambda_d_fixed, lambda_u = lambda_u_fixed)
rownames(grid) <- NULL
cat(sprintf("Grid: %d p cells (%s)\n", nrow(grid), paste(p_grid, collapse = ", ")))

all_cells <- vector("list", nrow(grid))
for (g in seq_len(nrow(grid))) {
  pg <- grid$p[g]; ld <- grid$lambda_d[g]; lu <- grid$lambda_u[g]
  cat(sprintf("\nCell %d/%d: p = %d, lambda_d = %.1f, lambda_u = %.1f\n",
              g, nrow(grid), pg, ld, lu))
  all_cells[[g]] <- run_simulation_tidy(
    n_rep = M, n_rct = n_rct, n_rwd = n_rwd, p = pg,
    N_post = N_post, N_burn = N_burn,
    ld = ld, lu = lu, seed_offset = 1000L * g)
}
final_flat_df <- do.call(rbind, all_cells)

# Output file path (! NAME MUST BE FILENAME_output.rds !).
output_file <- file.path(Sys.getenv("TMPDIR"), "sim_hd_v2c_output.rds")
cat("\nSaving all results to:", output_file, "\n")
saveRDS(final_flat_df, file = output_file)
cat("All results successfully saved in one file.\n")

# Local analysis (tables, figures): analyse_hd_v2.R in this folder.
