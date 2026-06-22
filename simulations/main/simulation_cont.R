################################################################################
# CATE estimation: FusionForest (RCT + RWD) vs. RCT-only and RWD-only.
# CONTINUOUS / UNCENSORED variant -- HPC parallel Monte Carlo driver.
#
# Sweeps a 2 x 2 grid over (lambda_d, lambda_u):
#   lambda_d : RWD baseline-deviation strength (between-study heterogeneity).
#   lambda_u : unmeasured-confounding strength in the RWD.
# For each grid cell, M replications are run in parallel; each replicate fits
# all three estimators and records RMSE, bias, 95% coverage and CI width of the
# CATE, on three populations (RCT rows, RWD rows, All rows).
#
# Data-generating process
# -----------------------
#   y = m0(X) + (1-S) dev(X) + A tau(X) + A (1-S) conf(U) + sigma_S eps
#   m0(X)   = 2*(2*X1 - X2 + 0.5*X3)    (prognostic function, observed X)
#   dev(X)  = 2*lambda_d*(X4 - 0.5*X5)  (RWD baseline deviation)
#   tau(X)  = 2*(X1 + 0.5*X2^2)         (true CATE, observed X)
#   conf(U) = 2*lambda_u*(-0.5*U)       (confounding driven by UNOBSERVED U)
#   RCT: A ~ Bernoulli(0.5),            eps ~ N(0, 0.75^2)
#   RWD: A ~ Bernoulli(plogis(X1 + U)), eps ~ N(0, 1.25^2)
#   U ~ N(0,1) seen by no model.  Source coding: S = 1 (RCT), S = 0 (RWD).
#
# Estimators (all predict the CATE on the combined covariate set)
# ---------------------------------------------------------------
#   (1) FusionForest : RCT + RWD, four-forest with confounding function c,
#                      error_dist = "source_dp_scale".
#   (2) RCT-only     : CausalShrinkageForest on the RCT (standard BART priors).
#   (3) RWD-only     : CausalShrinkageForest on the RWD (ignores confounding).
#
# Conventions (mirroring simulations/prior/simulation_censored_hpc.R)
# ------------------------------------------------------------------
#   * Cores read from CLI arg, e.g.  Rscript simulation_cont.R 192
#   * Parallelisation is across replications within a grid cell.
#   * Output is one tidy long data.frame saved to
#     $TMPDIR/simulation_cont_output.rds  (NAME MUST BE *_output.rds).
#
# Requires: FusionForests, ShrinkageTrees, doParallel, foreach.
################################################################################

library(FusionForests)
library(ShrinkageTrees)
library(doParallel)
library(foreach)

# ── DGP ───────────────────────────────────────────────────────────────────────
p_total   <- 10L                 # observed covariates
lambda_d  <- c(0.0, 1.0)         # grid: RWD baseline-deviation strength
lambda_u  <- c(0.0, 1.0)         # grid: unmeasured-confounding strength
sigma_rct <- 0.75               # RCT residual SD
sigma_rwd <- 1.25               # RWD residual SD (messier source)

# The factor 2 is folded into each structural function (was an outer 2 * (...)
# on y), so tau(X) is the true CATE on the y-scale directly.
m0   <- function(X) 2 * (2 * X[, 1] - X[, 2] + 0.5 * X[, 3])
dev  <- function(X, ld) 2 * ld * (X[, 4] - 0.5 * X[, 5])  # RWD baseline deviation
tau  <- function(X) 2 * (X[, 1] + 0.5 * X[, 2]^2)         # true CATE
conf <- function(U, lu) 2 * lu * (-0.5 * U)               # confounding via unobs. U

# Simulate one full dataset (RCT + RWD), continuous outcome, no censoring.
# ld, lu are the cell's deviation / confounding strengths.
make_data <- function(n_rct, n_rwd, ld, lu) {
  X_rct <- matrix(runif(n_rct * p_total), n_rct, p_total)
  X_rwd <- matrix(runif(n_rwd * p_total), n_rwd, p_total)
  U     <- rnorm(n_rwd)                              # RWD-only, unobserved

  A_rct <- rbinom(n_rct, 1L, 0.5)                    # randomised
  A_rwd <- rbinom(n_rwd, 1L, plogis(X_rwd[, 1] + U)) # selection driven by X1, U

  y_rct <- m0(X_rct) + A_rct * tau(X_rct) +
           rnorm(n_rct, 0, sigma_rct)
  y_rwd <- m0(X_rwd) + dev(X_rwd, ld) + A_rwd * tau(X_rwd) +
           A_rwd * conf(U, lu) +
           rnorm(n_rwd, 0, sigma_rwd)

  list(
    X_rct = X_rct, X_rwd = X_rwd,
    A_rct = A_rct, A_rwd = A_rwd,
    y_rct = y_rct, y_rwd = y_rwd,
    X     = rbind(X_rct, X_rwd),
    A     = c(A_rct, A_rwd),
    S     = c(rep(1L, n_rct), rep(0L, n_rwd)),
    y     = c(y_rct, y_rwd),
    n_rct = n_rct, n_rwd = n_rwd
  )
}

# ── Fits ──────────────────────────────────────────────────────────────────────
n_trees_csf <- 200L

fit_fusion <- function(d, N_post, N_burn) {
  FusionForest(
    y                         = d$y,
    X_train_control           = d$X,
    X_train_treat             = d$X,
    treatment_indicator_train = d$A,
    source_indicator_train    = d$S,
    X_test_control            = d$X,
    X_test_treat              = d$X,
    X_test_deconf             = d$X,
    treatment_indicator_test  = rep(1L, nrow(d$X)),
    source_indicator_test     = rep(1L, nrow(d$X)),
    error_dist                = "source_dp_scale",
    N_post = N_post, N_burn = N_burn,
    store_posterior_sample = TRUE, verbose = FALSE)
}

# Single-source causal forest (CausalShrinkageForest), predicting on the
# combined covariate set so that metrics can be split by population.
fit_csf <- function(y, X, A, Xev, N_post, N_burn) {
  ShrinkageTrees::CausalShrinkageForest(
    y                         = y,
    X_train_control           = X,
    X_train_treat             = X,
    treatment_indicator_train = A,
    X_test_control            = Xev,
    X_test_treat              = Xev,
    treatment_indicator_test  = rep(1L, nrow(Xev)),
    prior_type_control        = "standard",
    prior_type_treat          = "standard",
    local_hp_control          = 0.5 / sqrt(n_trees_csf),
    local_hp_treat            = 0.5 / sqrt(n_trees_csf),
    number_of_trees_control   = n_trees_csf,
    number_of_trees_treat     = n_trees_csf,
    N_post = N_post, N_burn = N_burn,
    store_posterior_sample = TRUE, verbose = FALSE)
}

# ── Metrics ───────────────────────────────────────────────────────────────────
# point : length-n posterior-mean CATE on the combined set.
# samp  : posterior CATE draws.  FusionForest returns N_post x n_obs;
#         CausalShrinkageForest returns n_obs x N_post.  We coerce to
#         n_obs x N_post by matching the axis whose length equals length(point).
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
    vals <- if (is.null(point)) setNames(rep(NA_real_, 4), metric_cols)
            else eval_cate(point, samp, truth, pops[[pop]])
    data.frame(method = method, population = pop, as.list(vals),
               stringsAsFactors = FALSE)
  }))
}

# ── One replication: all three estimators on one dataset ─────────────────────
run_one_sim <- function(seed, n_rct, n_rwd, N_post, N_burn, ld, lu) {
  set.seed(seed)
  d     <- make_data(n_rct, n_rwd, ld, lu)
  truth <- tau(d$X)
  n_all <- n_rct + n_rwd
  safe  <- function(expr) tryCatch(expr, error = function(e) NULL)

  ff  <- safe(fit_fusion(d, N_post, N_burn))
  rct <- safe(fit_csf(d$y_rct, d$X_rct, d$A_rct, d$X, N_post, N_burn))
  rwd <- safe(fit_csf(d$y_rwd, d$X_rwd, d$A_rwd, d$X, N_post, N_burn))

  rbind(
    make_rows("Fusion",   if (!is.null(ff))  ff$test_predictions_treat,
              if (!is.null(ff))  ff$test_predictions_sample_treat,  truth, n_rct, n_all),
    make_rows("RCT-only", if (!is.null(rct)) rct$test_predictions_treat,
              if (!is.null(rct)) rct$test_predictions_sample_treat, truth, n_rct, n_all),
    make_rows("RWD-only", if (!is.null(rwd)) rwd$test_predictions_treat,
              if (!is.null(rwd)) rwd$test_predictions_sample_treat, truth, n_rct, n_all)
  )
}

# ── Parallel runner over replications (one grid cell) ────────────────────────
run_simulation_tidy <- function(n_rep, n_rct, n_rwd, N_post, N_burn,
                                ld, lu, seed_offset = 1000L) {
  foreach(
    i = seq_len(n_rep),
    .combine = "rbind",
    .packages = c("FusionForests", "ShrinkageTrees")
  ) %dopar% {
    res <- run_one_sim(seed_offset + i, n_rct, n_rwd, N_post, N_burn, ld, lu)
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
cat("SIMULATION: FusionForest vs RCT-only vs RWD-only (CONTINUOUS) -- HPC\n")

# Simulation settings.
M       <- num_cores*5          # replications per grid cell
n_rct   <- 100L
n_rwd   <- 400L
N_post  <- 5000L
N_burn  <- 5000L
cat(sprintf("M = %d reps/cell, n_rct = %d, n_rwd = %d, N_post = %d, N_burn = %d\n",
            M, n_rct, n_rwd, N_post, N_burn))

# Grid over (lambda_d, lambda_u): 2 x 2 = 4 cells.
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
output_file <- file.path(Sys.getenv("TMPDIR"), "simulation_cont_output.rds")
cat("\nSaving all results to:", output_file, "\n")
saveRDS(final_flat_df, file = output_file)
cat("All results successfully saved in one file.\n")


# ──────────────────────────────────────────────────────────────────────────────
# LOCAL POST-PROCESSING (uncomment after downloading the .rds from the HPC).
# Place this script and simulation_cont_output.rds in the same folder and run
# from the repo root.  Prints, per (lambda_d, lambda_u, method, population):
#   rmse     -- mean pointwise CATE RMSE over replications
#   bias     -- mean integrated (averaged-over-points) signed bias
#   variance -- Monte Carlo variance of the integrated CATE estimate across
#               replications, i.e. var(bias_r).  This is the spread of the
#               estimator around its own average.  It pairs with `bias` via the
#               exact identity  mean(bias_r^2) = bias^2 + variance, so it is the
#               variance term of the *integrated* error (NOT of the pointwise
#               `rmse` column, which lives at a different level).
#   sd       -- sqrt(variance)
#   coverage, width -- as before
# A pointwise bias-variance decomposition var_r(tauhat(x)) at fixed x is not
# estimable here because X is redrawn every replication; it would need a fixed
# evaluation grid shared across replications.
# ──────────────────────────────────────────────────────────────────────────────
# res <- readRDS("simulations/main/simulation_cont_output.rds")
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
# # `variance` = across-replication variance of the estimator (var of bias_r);
# # `postvar`  = mean within-fit posterior variance of the CATE (boxable, below).
# 
# # Boxplots over replications (combined "All" population).  One panel per
# # lambda_d (left = smallest, right = largest); within a panel the boxes are
# # grouped by lambda_u from small to high, three estimators side by side per
# # group.  Generalises to any number of lambda_d / lambda_u values.
# # Metric: pick "rmse", "bias", "coverage", "width" or "postvar".
# a <- res[res$population == "All", ]
# meth <- c("Fusion", "RCT-only", "RWD-only"); cols <- c(2, 3, 4)
# a$method   <- factor(a$method, levels = meth)
# a$lambda_u <- factor(a$lambda_u, levels = sort(unique(a$lambda_u)))
# metric <- "coverage"
# lds <- sort(unique(a$lambda_d))
# par(mfrow = c(1, length(lds)))
# for (ld in lds) {
#   b <- a[a$lambda_d == ld, ]
#   # boxes ordered method-within-lambda_u; coloured by estimator
#   boxplot(b[[metric]] ~ b$method + b$lambda_u, col = cols, xaxt = "n",
#           ylim = range(a[[metric]], na.rm = TRUE),
#           main = paste("lambda_d =", ld), ylab = metric,
#           xlab = "lambda_u", sep = " ")
#   nlu <- nlevels(b$lambda_u)
#   axis(1, at = (seq_len(nlu) - 0.5) * length(meth) + 0.5,
#        labels = levels(b$lambda_u))
#   abline(v = seq_len(nlu - 1) * length(meth) + 0.5, lty = 3, col = "grey")
#   if (metric == "coverage") abline(h = 0.95, lty = 2)
#   if (metric == "bias")     abline(h = 0,    lty = 2)
#   legend("topright", meth, fill = cols, bty = "n", cex = 0.8)
# }
