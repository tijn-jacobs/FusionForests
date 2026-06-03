library(FusionForests)
library(ShrinkageTrees)
library(doParallel)
library(foreach)
library(evd)
library(MASS)

# ── DGP ───────────────────────────────────────────────────────────────────────
p_total     <- 10L               # observed covariates

# Fix both lambdas at 1; optimise the fusion forest over k_treat.
lambda_d_fixed <- 1.0            # RWD baseline-deviation strength (fixed)
lambda_u_fixed <- 1.0            # unmeasured-confounding strength (fixed)
k_treat_vals   <- seq(0.2, 0.6, 0.1)  # k_treat grid for the fusion forest
sigma_rct   <- 0.75              # RCT residual SD
target_cens <- 0.35              # target right-censoring fraction per source
visit_probs <- seq(0.1, 0.8, 0.1) # RWD inspection visits: deciles q10..q80


# Covariates: multivariate normal with AR(1) dependence, Sigma_ij = rho^|i-j|.
rho_X    <- 0.3
Sigma_X  <- rho_X ^ abs(outer(seq_len(p_total), seq_len(p_total), "-"))
draw_X   <- function(n) MASS::mvrnorm(n, mu = rep(0, p_total), Sigma = Sigma_X)

m0   <- function(X)  (2 * X[, 1] - X[, 2] * X[, 3] + 0.5 * X[, 4]^2)
dev  <- function(X, ld) ld * (X[, 4] - 0.5 * X[, 5])  # RWD baseline deviation
tau  <- function(X)  (1/2 + X[, 1] - 0.5 * X[, 2]^2)         # true CATE
conf <- function(U, lu) lu * (U)               # confounding via unobs. U

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
  U <- runif(n)                                        # RWD-only, unobserved
  e <- plogis(X[, 1] + U)                              # true propensity score
  A <- rbinom(n, 1L, e)                                # selection on X1, U
  list(X = X, A = A, e = e,
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
#   visits   -- RWD inspection times (time scale) at the deciles q10..q80.
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
    # e_rwd  = w$e,
    e_all  = c(rep(0.5, n_rct), w$e),
    n_rct = n_rct, n_rwd = n_rwd
  )
}

# ── Fits (all via FusionForest, log-time, with interval-censoring args) ───────
fit_fusion <- function(d, N_post, N_burn, k_treat = 0.5) {
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
    k_treat = k_treat,
    k_deconf = 0.5,
    k_deviation = 0.5,
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
  
  # k_treat optimisation: fit the fusion forest once per k_treat value.
  # RCT-only / RWD-only baselines are not run here (results obtained elsewhere).
  res <- do.call(rbind, lapply(k_treat_vals, function(kt) {
    ff   <- safe(fit_fusion(d, N_post, N_burn, k_treat = kt))
    rows <- make_rows("Fusion",
                      if (!is.null(ff)) ff$test_predictions_treat,
                      if (!is.null(ff)) ff$test_predictions_sample_treat,
                      truth, n_rct, n_all)
    rows$k_treat <- kt
    rows
  }))
  # realised censoring sanity-checks
  res$cens_rct    <- mean(d$st_rct == 0)            # RCT right-censored
  res$ic_rwd      <- mean(d$icc_rwd == 1)           # RWD interval-censored
  res$rc_rwd_tail <- mean(d$icc_rwd == 0)           # RWD right-censored past q80
  res
}

# ── Parallel runner over replications (one grid cell) ────────────────────────
run_simulation_tidy <- function(n_rep, n_rct, n_rwd, N_post, N_burn,
                                ld, lu, seed_offset = 1000L) {
  cal <- calibrate(ld, lu)        # per-cell RCT rate + RWD decile visits (fixed)
  cat(sprintf("  RCT cens rate = %.4g | RWD visits (q10..q80) = %s\n",
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
cat("SIMULATION: FusionForest k_treat optimisation (SURVIVAL v10) -- HPC\n")
cat("RWD interval-censored at deciles q10..q80; RCT right-censored ~35%.\n")
cat("Covariates: MVN AR(1), rho =", rho_X, "\n")

# Simulation settings.
M       <- num_cores * 3L        # replications per grid cell
n_rct   <- 150
n_rwd   <- 350L
N_post  <- 3000L
N_burn  <- 2000L
cat(sprintf("M = %d reps/cell, n_rct = %d, n_rwd = %d, N_post = %d, N_burn = %d\n",
            M, n_rct, n_rwd, N_post, N_burn))

# Single cell: both lambdas fixed at 1.  The k_treat sweep happens inside
# run_one_sim, so each replication yields one row per k_treat value.
grid <- data.frame(lambda_d = lambda_d_fixed, lambda_u = lambda_u_fixed)
cat(sprintf("Grid: %d (lambda_d, lambda_u) cell, k_treat grid of %d\n",
            nrow(grid), length(k_treat_vals)))

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
output_file <- file.path(Sys.getenv("TMPDIR"), "sim_surv_v10_output.rds")
cat("\nSaving all results to:", output_file, "\n")
saveRDS(final_flat_df, file = output_file)
cat("All results successfully saved in one file.\n")


# ──────────────────────────────────────────────────────────────────────────────
# LOCAL POST-PROCESSING (uncomment after downloading the .rds from the HPC).
# Place this script and sim_surv_v10_output.rds in the same folder and run from
# the repo root.  This run optimises the fusion forest over k_treat at fixed
# lambda_d = lambda_u = 1, so only the Fusion estimator is present and the
# results vary over k_treat.  Prints, per (k_treat, population):
#   rmse     -- mean pointwise CATE RMSE over replications
#   bias     -- mean integrated (averaged-over-points) signed bias
#   variance -- across-replication variance of the integrated CATE, var(bias_r)
#   sd       -- sqrt(variance)
#   coverage, width, postvar -- as recorded
# ──────────────────────────────────────────────────────────────────────────────
# res <- readRDS("simulations/main/sim_surv_v10_output.rds")
# 
# grp <- cbind(rmse, bias, coverage, width, postvar) ~ k_treat + population
# mean_tbl <- aggregate(grp, data = res,
#                       FUN = function(x) mean(x, na.rm = TRUE), na.action = na.pass)
# var_tbl  <- aggregate(bias ~ k_treat + population,
#                       data = res, FUN = function(x) var(x, na.rm = TRUE), na.action = na.pass)
# names(var_tbl)[ncol(var_tbl)] <- "variance"
# summary_tbl <- merge(mean_tbl, var_tbl)
# summary_tbl$sd <- sqrt(summary_tbl$variance)
# summary_tbl$population <- factor(summary_tbl$population,
#                                  levels = c("RCT", "RWD", "All"))
# ord <- order(summary_tbl$population, summary_tbl$k_treat)
# 
# cat("\n=== Fusion CATE metrics by k_treat (mean over replications) ===\n")
# print(summary_tbl[ord, c("k_treat", "population", "rmse", "bias",
#                          "variance", "sd", "coverage", "width", "postvar")],
#       row.names = FALSE, digits = 3)
# 
# # Realised censoring fractions (target ~35% RCT, interval-censored RWD):
# # cens_tbl <- aggregate(cbind(cens_rct, ic_rwd, rc_rwd_tail) ~ 1,
# #   data = res, FUN = function(x) mean(x, na.rm = TRUE))
# # print(cens_tbl, row.names = FALSE, digits = 3)
# 
# # Boxplots over replications (combined "All" population), one box per k_treat.
# # 2x2 grid of metrics; serif font, larger text, no per-panel titles.
# a <- res[res$population == "All", ]
# a$k_treat <- factor(a$k_treat, levels = sort(unique(a$k_treat)))
# metrics  <- c("rmse", "bias", "coverage", "postvar")
# metric_labs <- c(rmse = "RMSE", bias = "bias", coverage = "coverage",
#                  width = "CI width", postvar = "posterior variance")
# op <- par(mfrow = c(2, 2),
#           family = "serif",          # serif font (Times / Computer Modern)
#           mar = c(4, 4.5, 2, 1.5),   # per-panel margins (space between panels)
#           mgp = c(2.6, 0.8, 0),
#           cex.lab = 1.3, cex.axis = 1.1)
# for (metric in metrics) {
#   boxplot(a[[metric]] ~ a$k_treat, col = "goldenrod1",
#           main = "", ylab = metric_labs[metric], xlab = "k_treat")
#   if (metric == "coverage") abline(h = 0.95, lty = 2)
#   if (metric == "bias")     abline(h = 0,    lty = 2)
# }
# par(op)
# 
