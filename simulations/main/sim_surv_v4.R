library(FusionForests)
library(ShrinkageTrees)
library(doParallel)
library(foreach)

# ── DGP ───────────────────────────────────────────────────────────────────────
p_total     <- 10L               # observed covariates
lambda_d    <- c(0.0, 1.0, 2.0)       # grid: RWD baseline-deviation strength
lambda_u    <- c(0.0, 0.5, 1.0, 1.5, 2.0)       # grid: unmeasured-confounding strength
sigma_rct   <- 0.75              # RCT residual SD
sigma_rwd   <- 1.25              # RWD residual SD (messier source)
target_cens <- 0.35              # target right-censoring fraction per source
factor <- 2

# The factor 2 is folded into each structural function (was an outer factor * (...)
# on log T), so tau(X) is the true CATE on the log-time scale directly.
m0   <- function(X) factor * (2 * X[, 1] - X[, 2] * X[, 3] + 0.5 * X[, 4]^2)
dev  <- function(X, ld) factor * ld * (X[, 4] - 0.5 * X[, 5])  # RWD baseline deviation
tau  <- function(X) factor * (X[, 1] + 0.5 * X[, 2]^2)         # true CATE
conf <- function(U, lu) factor * lu * (U)               # confounding via unobs. U

# Latent log-survival generators (no censoring), one per source.
latent_rct <- function(n) {
  X <- matrix(runif(n * p_total), n, p_total)
  A <- rbinom(n, 1L, 0.5)                              # randomised
  list(X = X, A = A,
       logT = m0(X) + A * tau(X) + rnorm(n, 0, sigma_rct))
}
latent_rwd <- function(n, ld, lu) {
  X <- matrix(runif(n * p_total), n, p_total)
  U <- rnorm(n)                                        # RWD-only, unobserved
  A <- rbinom(n, 1L, plogis(X[, 1] + U))               # selection on X1, U
  list(X = X, A = A,
       logT = m0(X) + dev(X, ld) + A * tau(X) + A * conf(U, lu) +
         rnorm(n, 0, sigma_rwd))
}

# Calibrate the exponential censoring rate (original time scale) so the
# right-censoring fraction is ~target in a source with latent log-times logT.
# T = exp(logT); C ~ Exp(rate); right-censored when C < T, i.e.
# P(cens) = E[1 - exp(-rate * T)], monotone increasing in rate.
solve_cens_rate <- function(logT, target = target_cens) {
  Tt <- exp(logT)
  uniroot(function(r) mean(1 - exp(-r * Tt)) - target, c(1e-10, 1e6))$root
}

# Per-source censoring rates for one grid cell, fixed once on a large pre-sim.
cens_rates <- function(ld, lu, n_cal = 2e5) {
  c(rct = solve_cens_rate(latent_rct(n_cal)$logT),
    rwd = solve_cens_rate(latent_rwd(n_cal, ld, lu)$logT))
}

# Simulate one full dataset (RCT + RWD) with right censoring.
# rate_rct, rate_rwd are the per-source exponential censoring rates.
make_data <- function(n_rct, n_rwd, ld, lu, rate_rct, rate_rwd) {
  r <- latent_rct(n_rct)
  w <- latent_rwd(n_rwd, ld, lu)
  
  # Exponential censoring on the time scale, then log-transformed.
  logC_rct <- log(rexp(n_rct, rate = rate_rct))
  logC_rwd <- log(rexp(n_rwd, rate = rate_rwd))
  y_rct  <- pmin(r$logT, logC_rct); st_rct <- as.integer(r$logT <= logC_rct)
  y_rwd  <- pmin(w$logT, logC_rwd); st_rwd <- as.integer(w$logT <= logC_rwd)
  
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
    n_rct = n_rct, n_rwd = n_rwd
  )
}

# ── Fits (all via FusionForest, right-censored, log-time) ─────────────────────
fit_fusion <- function(d, N_post, N_burn) {
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
    N_post = N_post, N_burn = N_burn,
    treatment_coding = "binary",
    store_posterior_sample = TRUE, verbose = FALSE)
}

# Single-source causal forest (CausalShrinkageForest), right-censored on the
# log-time scale.  FusionForest is reserved for the multi-source fusion fit; a
# single-source FusionForest is ill-defined.  Predicts the CATE on the combined
# covariate set so that metrics can be split by population.
n_trees_csf <- 200L

fit_single <- function(y, status, X, A, Xev, N_post, N_burn) {
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
    vals <- if (is.null(point)) setNames(rep(NA_real_, 4), metric_cols)
    else eval_cate(point, samp, truth, pops[[pop]])
    data.frame(method = method, population = pop, as.list(vals),
               stringsAsFactors = FALSE)
  }))
}

# ── One replication: all three estimators on one dataset ─────────────────────
run_one_sim <- function(seed, n_rct, n_rwd, N_post, N_burn, ld, lu,
                        rate_rct, rate_rwd) {
  set.seed(seed)
  d     <- make_data(n_rct, n_rwd, ld, lu, rate_rct, rate_rwd)
  truth <- tau(d$X)
  n_all <- n_rct + n_rwd
  safe  <- function(expr) tryCatch(expr, error = function(e) NULL)
  
  ff  <- safe(fit_fusion(d, N_post, N_burn))
  rct <- safe(fit_single(d$y_rct, d$st_rct, d$X_rct, d$A_rct, d$X, N_post, N_burn))
  rwd <- safe(fit_single(d$y_rwd, d$st_rwd, d$X_rwd, d$A_rwd, d$X, N_post, N_burn))
  
  res <- rbind(
    make_rows("Fusion",   if (!is.null(ff))  ff$test_predictions_treat,
              if (!is.null(ff))  ff$test_predictions_sample_treat,  truth, n_rct, n_all),
    make_rows("RCT-only", if (!is.null(rct)) rct$test_predictions_treat,
              if (!is.null(rct)) rct$test_predictions_sample_treat, truth, n_rct, n_all),
    make_rows("RWD-only", if (!is.null(rwd)) rwd$test_predictions_treat,
              if (!is.null(rwd)) rwd$test_predictions_sample_treat, truth, n_rct, n_all)
  )
  # realised censoring fractions, for sanity-checking the ~35% target
  res$cens_rct <- mean(d$st_rct == 0)
  res$cens_rwd <- mean(d$st_rwd == 0)
  res
}

# ── Parallel runner over replications (one grid cell) ────────────────────────
run_simulation_tidy <- function(n_rep, n_rct, n_rwd, N_post, N_burn,
                                ld, lu, seed_offset = 1000L) {
  rates <- cens_rates(ld, lu)   # per-cell exponential censoring rates (fixed)
  cat(sprintf("  censoring rates: rct = %.4g, rwd = %.4g\n",
              rates["rct"], rates["rwd"]))
  foreach(
    i = seq_len(n_rep),
    .combine = "rbind",
    .packages = c("FusionForests", "ShrinkageTrees")
  ) %dopar% {
    res <- run_one_sim(seed_offset + i, n_rct, n_rwd, N_post, N_burn, ld, lu,
                       rates["rct"], rates["rwd"])
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
cat("SIMULATION: FusionForest vs RCT-only vs RWD-only (SURVIVAL) -- HPC\n")

# Simulation settings.
M       <- 1000 #num_cores*3          # replications per grid cell
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
output_file <- file.path(Sys.getenv("TMPDIR"), "sim_surv_v4_output.rds")
cat("\nSaving all results to:", output_file, "\n")
saveRDS(final_flat_df, file = output_file)
cat("All results successfully saved in one file.\n")


# ──────────────────────────────────────────────────────────────────────────────
# LOCAL POST-PROCESSING (uncomment after downloading the .rds from the HPC).
# Place this script and sim_surv_v4_output.rds in the same folder and run
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
res <- readRDS("simulations/main/sim_surv_v4_output.rds")

grp <- cbind(rmse, bias, coverage, width, postvar) ~
       lambda_d + lambda_u + method + population
mean_tbl <- aggregate(grp, data = res,
  FUN = function(x) mean(x, na.rm = TRUE), na.action = na.pass)
var_tbl  <- aggregate(bias ~ lambda_d + lambda_u + method + population,
  data = res, FUN = function(x) var(x, na.rm = TRUE), na.action = na.pass)
names(var_tbl)[ncol(var_tbl)] <- "variance"
summary_tbl <- merge(mean_tbl, var_tbl)
summary_tbl$sd <- sqrt(summary_tbl$variance)

summary_tbl$method     <- factor(summary_tbl$method,
                                 levels = c("Fusion", "RCT-only", "RWD-only"))
summary_tbl$population  <- factor(summary_tbl$population,
                                 levels = c("RCT", "RWD", "All"))
ord <- order(summary_tbl$lambda_d, summary_tbl$lambda_u,
             summary_tbl$population, summary_tbl$method)

cat("\n=== CATE metrics (mean over replications) ===\n")
print(summary_tbl[ord, c("lambda_d", "lambda_u", "population", "method",
                         "rmse", "bias", "variance", "sd", "coverage", "width",
                         "postvar")],
      row.names = FALSE, digits = 3)

# `variance` = across-replication variance of the estimator (var of bias_r);
# `postvar`  = mean within-fit posterior variance of the CATE (boxable, below).

# Realised censoring fractions per cell (target ~0.35 in both sources):
# cens_tbl <- aggregate(cbind(cens_rct, cens_rwd) ~ lambda_d + lambda_u,
#   data = res, FUN = function(x) mean(x, na.rm = TRUE))
# print(cens_tbl, row.names = FALSE, digits = 3)

# Boxplots over replications (combined "All" population).  One panel per
# lambda_d (left = smallest, right = largest); within a panel the boxes are
# grouped by lambda_u from small to high, three estimators side by side per
# group.  Generalises to any number of lambda_d / lambda_u values.
# Metric: pick "rmse", "bias", "coverage", "width" or "postvar".
a <- res[res$population == "All", ]
meth <- c("Fusion", "RCT-only", "RWD-only"); cols <- c(2, 3, 4)
a$method   <- factor(a$method, levels = meth)
a$lambda_u <- factor(a$lambda_u, levels = sort(unique(a$lambda_u)))
metric <- "rmse"
lds <- sort(unique(a$lambda_d))
par(mfrow = c(1, length(lds)))
for (ld in lds) {
  b <- a[a$lambda_d == ld, ]
  # boxes ordered method-within-lambda_u; coloured by estimator
  boxplot(b[[metric]] ~ b$method + b$lambda_u, col = cols, xaxt = "n",
          ylim = range(a[[metric]], na.rm = TRUE),
          main = paste("lambda_d =", ld), ylab = metric,
          xlab = "lambda_u", sep = " ")
  nlu <- nlevels(b$lambda_u)
  axis(1, at = (seq_len(nlu) - 0.5) * length(meth) + 0.5,
       labels = levels(b$lambda_u))
  abline(v = seq_len(nlu - 1) * length(meth) + 0.5, lty = 3, col = "grey")
  if (metric == "coverage") abline(h = 0.95, lty = 2)
  if (metric == "bias")     abline(h = 0,    lty = 2)
  legend("topright", meth, fill = cols, bty = "n", cex = 0.8)
}
