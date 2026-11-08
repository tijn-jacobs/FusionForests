# ══════════════════════════════════════════════════════════════════════════════
# sim_surv_v13_bff.R -- Bayesian fusion forest on the right-censored competitor design (3 x 3 lambda grid), CATE
# Revision round (manuscript_v3).  Derived from submission_v2/exp2_competitor/sim_surv_v12_bff.R.
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
# Output: $TMPDIR/sim_surv_v13_bff_output.rds (HPC contract, see ../../hpc_job_skeleton.sh).
# ══════════════════════════════════════════════════════════════════════════════
library(FusionForests)
library(ShrinkageTrees)
library(doParallel)
library(foreach)
library(evd)
library(MASS)

# ── DGP ───────────────────────────────────────────── (verbatim from v12) ──────
p_total     <- 10L               # observed covariates

# Cross design: sweep one lambda over `lambda_vals` while holding the other at 1.
lambda_vals    <- c(0.0, 0.5, 1.0, 1.5, 2.0)  # swept values for each axis
lambda_d_fixed <- 1.0                          # held when sweeping lambda_u
lambda_u_fixed <- 1.0                          # held when sweeping lambda_d
sigma_rct   <- 0.75              # RCT residual SD
target_cens <- 0.35              # target right-censoring fraction (both sources)

# Covariates: multivariate normal with AR(1) dependence, Sigma_ij = rho^|i-j|.
rho_X    <- 0.3
Sigma_X  <- rho_X ^ abs(outer(seq_len(p_total), seq_len(p_total), "-"))
draw_X   <- function(n) MASS::mvrnorm(n, mu = rep(0, p_total), Sigma = Sigma_X)

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
latent_rct <- function(n) {
  X <- draw_X(n)
  A <- rbinom(n, 1L, 0.5)                              # randomised
  list(X = X, A = A,
       logT = m0(X) + A * tau(X) + rnorm(n, 0, sigma_rct))
}
latent_rwd <- function(n, ld, lu) {
  gamma_em   <- 0.5772156649
  beta_scale <- sqrt(6) / pi
  mu_loc     <- -beta_scale * gamma_em
  X <- draw_X(n)
  e <- plogis(X[, 1])                                  # P(A = 1 | X): selection on X1
  A <- rbinom(n, 1L, e)
  U <- rnorm(n, mean = A * delta_x(X))                 # unmeasured prognostic factor
  eps <- -rgumbel(n, loc = mu_loc, scale = beta_scale) # mean-0 residual (SD 1)
  list(X = X, A = A, e = e, eps = eps,
       logT = m0(X) + dev(X, ld) + A * tau(X) + conf(U, lu) + eps)  # U in both arms
}

# Calibrate the RCT exponential censoring rate (time scale) to ~target.
solve_cens_rate <- function(logT, target = target_cens) {
  Tt <- exp(logT)
  uniroot(function(r) mean(1 - exp(-r * Tt)) - target, c(1e-10, 1e6))$root
}

# Per-cell calibration, fixed once on a large pre-simulation: the RCT and the RWD
# each get an independent exponential right-censoring rate tuned to ~target_cens.
calibrate <- function(ld, lu, n_cal = 1e5) {
  rate_rct <- solve_cens_rate(latent_rct(n_cal)$logT)
  rate_rwd <- solve_cens_rate(latent_rwd(n_cal, ld, lu)$logT)
  list(rate_rct = rate_rct, rate_rwd = rate_rwd)
}

# Simulate one full dataset.  Both sources are right-censored: the RCT and the
# RWD each get independent exponential censoring on the time scale, tuned to
# ~target_cens.  Outcomes are returned on the standardised LOG-time scale.
make_data <- function(n_rct, n_rwd, ld, lu, rate_rct, rate_rwd) {
  r <- latent_rct(n_rct)
  w <- latent_rwd(n_rwd, ld, lu)
  # Standardise the pooled latent log-survival to mean 0, unit variance; the same
  # affine map is applied to both sources.  Predicted CATEs come back on this
  # scale and are rescaled by `scale_sd` before evaluation.
  logT_true <- c(r$logT, w$logT)
  scale_mu  <- mean(logT_true)
  scale_sd  <- sd(logT_true)
  std       <- function(z) (z - scale_mu) / scale_sd

  # --- RCT: exponential right censoring ---
  logC_rct <- log(rexp(n_rct, rate = rate_rct))
  y_rct    <- pmin(r$logT, logC_rct)
  st_rct   <- as.integer(r$logT <= logC_rct)

  # --- RWD: exponential right censoring (same mechanism as the RCT) ---
  logC_rwd <- log(rexp(n_rwd, rate = rate_rwd))
  y_rwd    <- pmin(w$logT, logC_rwd)
  st_rwd   <- as.integer(w$logT <= logC_rwd)

  # --- standardise the log-time outcomes to the common scale ---
  y_rct <- std(y_rct)
  y_rwd <- std(y_rwd)

  list(
    X_rct = r$X, X_rwd = w$X,
    A_rct = r$A, A_rwd = w$A,
    y_rct = y_rct, y_rwd = y_rwd,
    st_rct = st_rct, st_rwd = st_rwd,
    X      = rbind(r$X, w$X),
    A      = c(r$A, w$A),
    S      = c(rep(1L, n_rct), rep(0L, n_rwd)),
    y      = c(y_rct, y_rwd),
    status = c(st_rct, st_rwd),
    e_rwd  = w$e,
    e_all  = c(rep(0.5, n_rct), w$e),
    scale_sd = scale_sd,
    n_rct = n_rct, n_rwd = n_rwd
  )
}

# ── Bayesian Fusion Forest fit (combined RCT + RWD) ───────────────────────────
# sim_surv_v13.R's fit_fusion, minus the interval-censoring arguments: both
# sources are right-censored, so we pass only (y, status).
fit_fusion <- function(d, N_post, N_burn) {
  # Step-height scales k_f = 1 for every forest, including k_tau = 1 (chosen
  # by a pilot; sigma_h = k / (2 sqrt J)).
  FusionForest(
    y                         = d$y,
    status                    = d$status,
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

# ── Metrics (posterior CATE sample) ───────────────────────────────────────────
# FusionForest returns posterior CATE draws (N_post x n_obs); coverage / width /
# postvar come from the 95% credible interval and the mean posterior variance.
as_obs_by_draw <- function(samp, n_obs) {
  if (nrow(samp) == n_obs) samp else t(samp)
}
eval_cate <- function(point, samp, truth, idx) {
  p  <- point[idx]; tr <- truth[idx]
  M  <- as_obs_by_draw(samp, length(point))[idx, , drop = FALSE]
  ci <- apply(M, 1, quantile, probs = c(0.025, 0.975))
  c(rmse     = sqrt(mean((p - tr)^2)),
    bias     = mean(p - tr),
    coverage = mean(tr >= ci[1, ] & tr <= ci[2, ]),
    width    = mean(ci[2, ] - ci[1, ]),
    postvar  = mean(apply(M, 1, var)))
}

metric_cols <- c("rmse", "bias", "coverage", "width", "postvar")

make_rows <- function(method, point, samp, truth, n_rct, n_all) {
  pops <- list(RCT = seq_len(n_rct),
               RWD = (n_rct + 1L):n_all,
               All = seq_len(n_all))
  do.call(rbind, lapply(names(pops), function(pop) {
    vals <- if (is.null(point))
      setNames(rep(NA_real_, length(metric_cols)), metric_cols)
    else eval_cate(point, samp, truth, pops[[pop]])
    data.frame(method = method, population = pop, as.list(vals),
               stringsAsFactors = FALSE)
  }))
}

# ── One replication: BFF on the combined dataset ──────────────────────────────
run_one_sim <- function(seed, n_rct, n_rwd, N_post, N_burn, ld, lu,
                        rate_rct, rate_rwd) {
  set.seed(seed)
  d     <- make_data(n_rct, n_rwd, ld, lu, rate_rct, rate_rwd)
  truth <- tau(d$X)
  n_all <- n_rct + n_rwd
  safe  <- function(expr) tryCatch(expr, error = function(e) {
    message("    BFF fit failed: ", conditionMessage(e)); NULL })

  s  <- d$scale_sd          # standardised-log CATE -> log-time scale
  ff <- safe(fit_fusion(d, N_post, N_burn))

  res <- make_rows("Fusion",
                   if (!is.null(ff)) ff$test_predictions_treat * s,
                   if (!is.null(ff)) ff$test_predictions_sample_treat * s,
                   truth, n_rct, n_all)
  res$cens_rct <- mean(d$st_rct == 0)
  res$cens_rwd <- mean(d$st_rwd == 0)
  res
}

# ── Parallel runner over replications (one grid cell) ─────────────────────────
run_simulation_tidy <- function(n_rep, n_rct, n_rwd, N_post, N_burn,
                                ld, lu, seed_offset = 1000L) {
  cal <- calibrate(ld, lu)        # per-cell RCT + RWD exponential censoring rates
  cat(sprintf("  RCT cens rate = %.4g | RWD cens rate = %.4g\n",
              cal$rate_rct, cal$rate_rwd))
  foreach(
    i = seq_len(n_rep),
    .combine = "rbind",
    .packages = c("FusionForests", "ShrinkageTrees", "evd", "MASS")
  ) %dopar% {
    res <- run_one_sim(seed_offset + i, n_rct, n_rwd, N_post, N_burn, ld, lu,
                       cal$rate_rct, cal$rate_rwd)
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
registerDoParallel(cores = num_cores)
cat("Number of cores being used (1 free):", num_cores, "\n")
cat("SIMULATION: Bayesian Fusion Forest on combined RCT+RWD (right-censored) -- HPC\n")
cat("RCT and RWD both right-censored ~35%.\n")
cat("Covariates: MVN AR(1), rho =", rho_X, "\n")

# Simulation settings.
M       <- 1000L                 # replications per grid cell
n_rct   <- 150L                  # v12 settings
n_rwd   <- 350L                  # v12 settings
N_post  <- 5000L                 # v12 settings
N_burn  <- 5000L                 # v12 settings
cat(sprintf("M = %d reps/cell, n_rct = %d, n_rwd = %d, N_post = %d, N_burn = %d\n",
            M, n_rct, n_rwd, N_post, N_burn))

# Full 3x3 grid: lambda_d and lambda_u each in {0, 1, 2} (nine cells).
grid <- expand.grid(lambda_d = c(0, 1, 2), lambda_u = c(0, 1, 2))
cat(sprintf("Grid: %d (lambda_d, lambda_u) cells\n", nrow(grid)))

all_cells <- vector("list", nrow(grid))
for (g in seq_len(nrow(grid))) {
  ld <- grid$lambda_d[g]; lu <- grid$lambda_u[g]
  cat(sprintf("\nCell %d/%d: lambda_d = %.1f, lambda_u = %.1f\n",
              g, nrow(grid), ld, lu))
  all_cells[[g]] <- run_simulation_tidy(
    n_rep = M, n_rct = n_rct, n_rwd = n_rwd,
    N_post = N_post, N_burn = N_burn,
    ld = ld, lu = lu, seed_offset = 1000L * g)
}
final_flat_df <- do.call(rbind, all_cells)

# Output file path (! NAME MUST BE FILENAME_output.rds !).
output_file <- file.path(Sys.getenv("TMPDIR"), "sim_surv_v13_bff_output.rds")
cat("\nSaving all results to:", output_file, "\n")
saveRDS(final_flat_df, file = output_file)
cat("All results successfully saved in one file.\n")

# Local analysis (tables, figures): analyse_competitors_v13.R in this folder.
