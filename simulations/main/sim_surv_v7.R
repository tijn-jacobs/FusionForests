library(FusionForests)
library(ShrinkageTrees)
library(doParallel)
library(foreach)
library(evd)
library(MASS)

# ── DGP ───────────────────────────────────────────────────────────────────────
p_total     <- 10L               # observed covariates
lambda_d    <- c(1.0)       # grid: RWD baseline-deviation strength
lambda_u    <- c(0.0, 1.0, 2.0)       # grid: unmeasured-confounding strength
sigma_rct   <- 0.75              # RCT residual SD
target_cens <- 0.35              # target right-censoring fraction per source
visit_probs <- seq(0.1, 0.9, 0.1) # RWD inspection visits: deciles q10..q90
factor <- 2

# Covariates: multivariate normal with AR(1) dependence, Sigma_ij = rho^|i-j|.
rho_X    <- 0.5
Sigma_X  <- rho_X ^ abs(outer(seq_len(p_total), seq_len(p_total), "-"))
draw_X   <- function(n) MASS::mvrnorm(n, mu = rep(0, p_total), Sigma = Sigma_X)

# The factor 2 is folded into each structural function (was an outer factor * (...)
# on log T), so tau(X) is the true CATE on the log-time scale directly.
m0   <- function(X) factor * (2 * X[, 1] - X[, 2] * X[, 3] + 0.5 * X[, 4]^2)
dev  <- function(X, ld) factor * ld * (X[, 4] - 0.5 * X[, 5])  # RWD baseline deviation
tau  <- function(X) factor * (X[, 1] + 0.5 * X[, 2]^2)         # true CATE
conf <- function(U, lu) factor * lu * (U)               # confounding via unobs. U

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
  U <- rnorm(n)                                        # RWD-only, unobserved
  A <- rbinom(n, 1L, plogis(X[, 1] + U))               # selection on X1, U
  list(X = X, A = A,
       logT = m0(X) + dev(X, ld) + A * tau(X) + A * conf(U, lu) -
         rgumbel(n, loc = mu_loc, scale = beta_scale))
}

# Calibrate the RCT exponential censoring rate (time scale) to ~target.
# P(cens) = E[1 - exp(-rate * T)], monotone increasing in rate.
solve_cens_rate <- function(logT, target = target_cens) {
  Tt <- exp(logT)
  uniroot(function(r) mean(1 - exp(-r * Tt)) - target, c(1e-10, 1e6))$root
}

# Per-cell calibration, fixed once on a large pre-simulation:
#   rate_rct -- RCT exponential censoring rate.
#   visits   -- RWD inspection times (time scale) at the deciles q10..q90.
calibrate <- function(ld, lu, n_cal = 2e5) {
  rate_rct <- solve_cens_rate(latent_rct(n_cal)$logT)
  visits   <- as.numeric(quantile(exp(latent_rwd(n_cal, ld, lu)$logT),
                                  probs = visit_probs))
  list(rate_rct = rate_rct, visits = visits)
}

# Simulate one full dataset (RCT right-censored, RWD interval-censored).
# All time bounds are returned on the LOG scale (timescale = "log").
make_data <- function(n_rct, n_rwd, ld, lu, rate_rct, visits) {
  r <- latent_rct(n_rct)
  w <- latent_rwd(n_rwd, ld, lu)
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
  last_visit <- visits[length(visits)]      # q90
  breaks     <- c(0, visits)                # bin edges: 0, q10, ..., q90
  j          <- findInterval(T0, breaks)    # 1..length(visits) below q90
  beyond     <- T0 >= last_visit            # event after the last visit
  j[!beyond] <- pmax(j[!beyond], 1L)

  left_rwd  <- ifelse(j == 1L, logfloor, log(breaks[pmin(j, length(breaks))]))
  right_rwd <- log(breaks[pmin(j + 1L, length(breaks))])
  st_rwd    <- rep(0L, n_rwd)               # no exact event times in the RWD
  icc_rwd   <- rep(1L, n_rwd)               # interval-censored by default
  # events beyond the last visit: right-censored at q90 (icc = 0)
  left_rwd[beyond]  <- log(last_visit)
  right_rwd[beyond] <- log(last_visit)
  icc_rwd[beyond]   <- 0L
  y_rwd <- right_rwd                        # placeholder; bounds carry the info

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
    n_rct = n_rct, n_rwd = n_rwd
  )
}

# ── Fits (all via FusionForest, log-time, with interval-censoring args) ───────
fit_fusion <- function(d, N_post, N_burn) {
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
    k_control = 0.5,
    k_treat = 0.5,
    k_deconf = 0.5,
    k_deviation = 0.5,
    power_control = 2,
    base_control = 0.95,
    power_deviation = 2,
    base_deviation = 0.95,
    power_treat = 2,
    base_treat = 0.95,
    power_deconf = 2,
    base_deconf = 0.95,
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
n_trees_csf <- 200L

# RCT-only: right-censored on the log-time scale.
fit_single_rc <- function(y, status, X, A, Xev, N_post, N_burn) {
  ShrinkageTrees::CausalShrinkageForest(
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
    local_hp_control          = 0.5 / sqrt(n_trees_csf),
    local_hp_treat            = 0.5 / sqrt(n_trees_csf),
    number_of_trees_control   = n_trees_csf,
    number_of_trees_treat     = n_trees_csf,
    N_post = N_post, N_burn = N_burn,
    treatment_coding = "binary",
    store_posterior_sample = TRUE, verbose = FALSE)
}

# RWD-only: interval-censored on the log-time scale.  Bounds use the interval2
# convention: interval-censored rows have finite left < right; rows censored
# beyond the last visit have right = Inf (right-censored).
fit_single_ic <- function(left_time, right_time, X, A, Xev, N_post, N_burn) {
  ShrinkageTrees::CausalShrinkageForest(
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
    local_hp_control          = 0.5 / sqrt(n_trees_csf),
    local_hp_treat            = 0.5 / sqrt(n_trees_csf),
    number_of_trees_control   = n_trees_csf,
    number_of_trees_treat     = n_trees_csf,
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

# ── One replication: all three estimators on one dataset ─────────────────────
run_one_sim <- function(seed, n_rct, n_rwd, N_post, N_burn, ld, lu,
                        rate_rct, visits) {
  set.seed(seed)
  d     <- make_data(n_rct, n_rwd, ld, lu, rate_rct, visits)
  truth <- tau(d$X)
  n_all <- n_rct + n_rwd
  safe  <- function(expr) tryCatch(expr, error = function(e) NULL)

  # RWD interval2 bounds: keep finite (left, right] for interval-censored rows;
  # set right = Inf for rows censored beyond the last visit (right-censored).
  lt_rwd <- d$left_rwd
  rt_rwd <- d$right_rwd
  rt_rwd[d$icc_rwd == 0] <- Inf

  ff  <- safe(fit_fusion(d, N_post, N_burn))
  rct <- safe(fit_single_rc(d$y_rct, d$st_rct, d$X_rct, d$A_rct, d$X,
                            N_post, N_burn))
  rwd <- safe(fit_single_ic(lt_rwd, rt_rwd, d$X_rwd, d$A_rwd, d$X,
                            N_post, N_burn))

  res <- rbind(
    make_rows("Fusion",   if (!is.null(ff))  ff$test_predictions_treat,
              if (!is.null(ff))  ff$test_predictions_sample_treat,  truth, n_rct, n_all),
    make_rows("RCT-only", if (!is.null(rct)) rct$test_predictions_treat,
              if (!is.null(rct)) rct$test_predictions_sample_treat, truth, n_rct, n_all),
    make_rows("RWD-only", if (!is.null(rwd)) rwd$test_predictions_treat,
              if (!is.null(rwd)) rwd$test_predictions_sample_treat, truth, n_rct, n_all)
  )
  # realised censoring sanity-checks
  res$cens_rct    <- mean(d$st_rct == 0)            # RCT right-censored
  res$ic_rwd      <- mean(d$icc_rwd == 1)           # RWD interval-censored
  res$rc_rwd_tail <- mean(d$icc_rwd == 0)           # RWD right-censored past q90
  res
}

# ── Parallel runner over replications (one grid cell) ────────────────────────
run_simulation_tidy <- function(n_rep, n_rct, n_rwd, N_post, N_burn,
                                ld, lu, seed_offset = 1000L) {
  cal <- calibrate(ld, lu)        # per-cell RCT rate + RWD decile visits (fixed)
  cat(sprintf("  RCT cens rate = %.4g | RWD visits (q10..q90) = %s\n",
              cal$rate_rct, paste(round(cal$visits, 2), collapse = ", ")))
  foreach(
    i = seq_len(n_rep),
    .combine = "rbind",
    .packages = c("FusionForests", "ShrinkageTrees", "evd", "MASS")
  ) %dopar% {
    res <- run_one_sim(seed_offset + i, n_rct, n_rwd, N_post, N_burn, ld, lu,
                       cal$rate_rct, cal$visits)
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
cat("SIMULATION: FusionForest vs RCT-only vs RWD-only (SURVIVAL v7) -- HPC\n")
cat("RWD interval-censored at deciles q10..q90; RCT right-censored ~35%.\n")
cat("Covariates: MVN AR(1), rho =", rho_X, "\n")

# Simulation settings.
M       <- num_cores * 1L        # replications per grid cell
n_rct   <- 150
n_rwd   <- 3500L
N_post  <- 2000L
N_burn  <- 1000L
cat(sprintf("M = %d reps/cell, n_rct = %d, n_rwd = %d, N_post = %d, N_burn = %d\n",
            M, n_rct, n_rwd, N_post, N_burn))

# Grid over (lambda_d, lambda_u).
grid <- expand.grid(lambda_d = lambda_d, lambda_u = lambda_u)
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
output_file <- file.path(Sys.getenv("TMPDIR"), "sim_surv_v7_output.rds")
cat("\nSaving all results to:", output_file, "\n")
saveRDS(final_flat_df, file = output_file)
cat("All results successfully saved in one file.\n")


# ──────────────────────────────────────────────────────────────────────────────
# LOCAL POST-PROCESSING (uncomment after downloading the .rds from the HPC).
# Place this script and sim_surv_v7_output.rds in the same folder and run from the repo
# root.  Prints, per (lambda_d, lambda_u, method, population):
#   rmse     -- mean pointwise CATE RMSE over replications
#   bias     -- mean integrated (averaged-over-points) signed bias
#   variance -- across-replication variance of the integrated CATE, var(bias_r)
#   sd       -- sqrt(variance)
#   coverage, width, postvar -- as recorded
# ──────────────────────────────────────────────────────────────────────────────
# res <- readRDS("simulations/main/sim_surv_v7_output.rds")
# 
# grp <- cbind(rmse, bias, coverage, width, postvar) ~
#        lambda_d + lambda_u + method + population
# mean_tbl <- aggregate(grp, data = res,
#   FUN = function(x) mean(x, na.rm = TRUE), na.action = na.pass)
# var_tbl  <- aggregate(bias ~ lambda_d + lambda_u + method + population,
#   data = res, FUN = function(x) var(x, na.rm = TRUE), na.action = na.pass)
# names(var_tbl)[ncol(var_tbl)] <- "variance"
# summary_tbl <- merge(mean_tbl, var_tbl)
# summary_tbl$sd <- sqrt(summary_tbl$variance)
# 
# summary_tbl$method     <- factor(summary_tbl$method,
#                                  levels = c("Fusion", "RCT-only", "RWD-only"))
# summary_tbl$population  <- factor(summary_tbl$population,
#                                  levels = c("RCT", "RWD", "All"))
# ord <- order(summary_tbl$lambda_d, summary_tbl$lambda_u,
#              summary_tbl$population, summary_tbl$method)
# 
# cat("\n=== CATE metrics (mean over replications) ===\n")
# print(summary_tbl[ord, c("lambda_d", "lambda_u", "population", "method",
#                          "rmse", "bias", "variance", "sd", "coverage", "width",
#                          "postvar")],
#       row.names = FALSE, digits = 3)
# 
# # Realised censoring fractions per cell:
# # cens_tbl <- aggregate(cbind(cens_rct, ic_rwd, rc_rwd_tail) ~ lambda_d + lambda_u,
# #   data = res, FUN = function(x) mean(x, na.rm = TRUE))
# # print(cens_tbl, row.names = FALSE, digits = 3)
# 
# # Boxplots over replications (combined "All" population).  One panel per
# # lambda_d (left = smallest, right = largest); within a panel the boxes are
# # grouped by lambda_u from small to high, three estimators side by side per
# # group.  Metric: pick "rmse", "bias", "coverage", "width" or "postvar".
# a <- res[res$population == "All", ]
# meth <- c("Fusion", "RCT-only", "RWD-only"); cols <- c(2, 3, 4)
# a$method   <- factor(a$method, levels = meth)
# a$lambda_u <- factor(a$lambda_u, levels = sort(unique(a$lambda_u)))
# metric <- "rmse"
# lds <- sort(unique(a$lambda_d))
# par(mfrow = c(1, length(lds)))
# for (ld in lds) {
#   b <- a[a$lambda_d == ld, ]
#   boxplot(b[[metric]] ~ b$method + b$lambda_u, col = cols, xaxt = "n",
#           ylim = range(a[[metric]], na.rm = TRUE),
#           main = paste("lambda_d =", ld, "m:", metric), ylab = metric,
#           xlab = "lambda_u", sep = " ")
#   nlu <- nlevels(b$lambda_u)
#   axis(1, at = (seq_len(nlu) - 0.5) * length(meth) + 0.5,
#        labels = levels(b$lambda_u))
#   abline(v = seq_len(nlu - 1) * length(meth) + 0.5, lty = 3, col = "grey")
#   if (metric == "coverage") abline(h = 0.95, lty = 2)
#   if (metric == "bias")     abline(h = 0,    lty = 2)
#   legend("topright", meth, fill = cols, bty = "n", cex = 0.8)
# }
