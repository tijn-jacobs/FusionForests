# ══════════════════════════════════════════════════════════════════════════════
# <base_name>.R — HPC simulation skeleton
#
# One-line description of the experiment, the estimators compared, and the
# quantity evaluated (e.g. "FusionForest vs RCT-only vs RWD-only, CATE").
#
# FILE-NAMING CONTRACT (the SLURM job script depends on it):
#   * this script is called   <base_name>.R
#   * it writes its results to $TMPDIR/<base_name>_output.rds
#   The job script (hpc_job_skeleton.sh) copies <base_name>.R to $TMPDIR, runs
#   it with the core count as the single command-line argument, and copies
#   <base_name>_output.rds back to $HOME/sim.  Change `base_name` in ONE place
#   in the job script; the R script must match.
#
# LAYOUT (keep the section order — every simulation in this project uses it):
#   1. Libraries
#   2. (optional) Local temp copies of package functions
#   3. DGP        — constants, truth functions, make_data()
#   4. Fits       — one fit_<method>() wrapper per estimator
#   5. Metrics    — eval + tidy-row builders
#   6. One replication — run_one_sim(seed, ...)
#   7. Parallel runner  — run_simulation_tidy() (foreach %dopar%)
#   8. Main (HPC)       — cores, settings, grid loop, saveRDS
#   9. LOCAL POST-PROCESSING (commented out; run after downloading the .rds)
# ══════════════════════════════════════════════════════════════════════════════

# ── 1. Libraries ─────────────────────────────────────────────────────────────
library(doParallel)
library(foreach)
# library(<model package(s)>)     # e.g. FusionForests, ShrinkageTrees
# library(MASS)                   # mvrnorm for correlated covariates

# ── 2. (Optional) local temp copies of package functions ─────────────────────
# If the experiment needs a modified version of a package function, paste the
# full function here as <fun>_temp and bind it to the package namespace so its
# unexported internals still resolve:
#
# MyModel_temp <- function(...) { ... }
# environment(MyModel_temp) <- asNamespace("<package>")

# ── 3. DGP ───────────────────────────────────────────────────────────────────
# All DGP constants live here, at the top, each with a short comment.
p_total <- 10L                    # number of observed covariates
# sigma_eps   <- 0.75             # residual SD
# target_cens <- 0.35             # target censoring fraction

# Covariate draw (example: MVN with AR(1) correlation, Sigma_ij = rho^|i-j|).
# rho_X   <- 0.3
# Sigma_X <- rho_X ^ abs(outer(seq_len(p_total), seq_len(p_total), "-"))
draw_X <- function(n) matrix(rnorm(n * p_total), n, p_total)

# Truth functions: prognostic surface, treatment effect, etc.
m0  <- function(X) 2 * X[, 1] - X[, 2] * X[, 3]        # baseline
tau <- function(X) 0.5 + X[, 1]                        # true CATE

# If any DGP quantity must be calibrated (censoring rates, visit times, SNR),
# do it ONCE per grid cell on a large pre-simulation, not per replication:
# calibrate <- function(<cell parameters>, n_cal = 2e5) { ... }

# Simulate one full dataset.  Return a plain list carrying everything a fit or
# metric later needs (X, treatment, outcome, censoring info, true propensity,
# any standardisation constants such as scale_sd, and the sample sizes).
make_data <- function(n, ...) {
  X <- draw_X(n)
  A <- rbinom(n, 1L, 0.5)
  y <- m0(X) + A * tau(X) + rnorm(n)
  list(X = X, A = A, y = y, n = n)
}

# ── 4. Fits ──────────────────────────────────────────────────────────────────
# One thin wrapper per estimator.  Each takes the data list `d` plus MCMC /
# tuning settings, fixes all method hyperparameters HERE (so a reader can see
# every setting in one place), and returns the fitted object.
# Keep verbose = FALSE inside %dopar%.
fit_method_a <- function(d, N_post, N_burn) {
  # MyModel(y = d$y, X = d$X, ..., N_post = N_post, N_burn = N_burn,
  #         verbose = FALSE)
  stop("replace with a model fit")
}
# fit_method_b <- function(d, N_post, N_burn) { ... }

# ── 5. Metrics ───────────────────────────────────────────────────────────────
# eval_*(): compute the metric vector for one fit on one (sub)population.
#   point : length-n posterior-mean estimate
#   samp  : posterior draws, coerced to n_obs x n_draws
#   truth : length-n true value;  idx : rows of the target population
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

# Build tidy rows: one row per (method, population), NA metrics if the fit
# failed (point is NULL).  Add / drop populations as the experiment needs.
make_rows <- function(method, point, samp, truth, n_all) {
  pops <- list(All = seq_len(n_all))
  # pops <- list(RCT = seq_len(n_rct), RWD = (n_rct + 1L):n_all,
  #              All = seq_len(n_all))
  do.call(rbind, lapply(names(pops), function(pop) {
    vals <- if (is.null(point))
      setNames(rep(NA_real_, length(metric_cols)), metric_cols)
    else eval_cate(point, samp, truth, pops[[pop]])
    data.frame(method = method, population = pop, as.list(vals),
               stringsAsFactors = FALSE)
  }))
}

# ── 6. One replication: all estimators on one dataset ────────────────────────
# The seed is the ONLY source of randomness; everything else is a fixed input.
# Wrap each fit in `safe()` so one MCMC failure yields NA rows, not a dead
# worker.  Return a tidy data.frame; append any realised sanity checks
# (censoring fractions, overlap, ...) as extra columns.
run_one_sim <- function(seed, n, N_post, N_burn) {
  set.seed(seed)
  d     <- make_data(n)
  truth <- tau(d$X)
  safe  <- function(expr) tryCatch(expr, error = function(e) NULL)

  fa <- safe(fit_method_a(d, N_post, N_burn))
  # fb <- safe(fit_method_b(d, N_post, N_burn))

  res <- rbind(
    make_rows("MethodA",
              if (!is.null(fa)) fa$test_predictions_treat,
              if (!is.null(fa)) fa$test_predictions_sample_treat,
              truth, d$n)
    # , make_rows("MethodB", ...)
  )
  # res$cens <- mean(d$status == 0)          # realised sanity checks
  res
}

# ── 7. Parallel runner over replications (one grid cell) ─────────────────────
# foreach over replications; .combine = "rbind" keeps the result tidy.  List
# every package the workers need in .packages.  Per-cell calibration happens
# here, once, before the loop.  Distinct seed_offset per cell (set in the main
# loop) keeps all seeds unique across the whole grid.
run_simulation_tidy <- function(n_rep, n, N_post, N_burn,
                                seed_offset = 1000L) {
  # cal <- calibrate(<cell parameters>)
  foreach(
    i = seq_len(n_rep),
    .combine = "rbind",
    .packages = c()               # e.g. c("FusionForests", "ShrinkageTrees")
  ) %dopar% {
    res <- run_one_sim(seed_offset + i, n, N_post, N_burn)
    res$Iter <- i                 # tag replication + cell parameters
    # res$<cell_param> <- ...
    res
  }
}

# ── 8. Main (HPC) ────────────────────────────────────────────────────────────
# The job script passes the node's core count as the single argument; keep one
# core free.  Falls back to the local machine when run interactively.
args <- commandArgs(trailingOnly = TRUE)
if (length(args) > 0) {
  num_cores <- as.integer(args[1]) - 1L
} else {
  num_cores <- parallel::detectCores() - 1L
}
registerDoParallel(cores = num_cores)
cat("Number of cores being used (1 free):", num_cores, "\n")

# Banner: say WHAT this simulation is, so the SLURM log is self-describing.
cat("SIMULATION: <description / version> -- HPC\n")

# Simulation settings — echo them all to the log.
M      <- 1000L                   # replications per grid cell
n      <- 500L                    # sample size(s)
N_post <- 3000L                   # posterior draws
N_burn <- 2000L                   # burn-in
cat(sprintf("M = %d reps/cell, n = %d, N_post = %d, N_burn = %d\n",
            M, n, N_post, N_burn))

# Grid of simulation cells (one row per scenario); drop duplicate cells.
grid <- data.frame(cell = 1L)
# grid <- unique(rbind(
#   data.frame(lambda_d = 1,    lambda_u = lambda_vals),
#   data.frame(lambda_d = lambda_vals, lambda_u = 1)))
cat(sprintf("Grid: %d cells\n", nrow(grid)))

all_cells <- vector("list", nrow(grid))
for (g in seq_len(nrow(grid))) {
  cat(sprintf("\nCell %d/%d\n", g, nrow(grid)))
  all_cells[[g]] <- run_simulation_tidy(
    n_rep = M, n = n, N_post = N_post, N_burn = N_burn,
    seed_offset = 1000L * g)      # unique seeds per cell
}
final_flat_df <- do.call(rbind, all_cells)

# Output file path (! NAME MUST BE <base_name>_output.rds !).
out_dir <- Sys.getenv("TMPDIR")
if (!nzchar(out_dir)) out_dir <- "."   # interactive run: current directory
output_file <- file.path(out_dir, "<base_name>_output.rds")
cat("\nSaving all results to:", output_file, "\n")
saveRDS(final_flat_df, file = output_file)
cat("All results successfully saved in one file.\n")


# ──────────────────────────────────────────────────────────────────────────────
# 9. LOCAL POST-PROCESSING (uncomment after downloading the .rds from the HPC).
# Place this script and <base_name>_output.rds in the same folder and run from
# the repo root.  Prints, per (cell, method, population):
#   rmse     -- mean pointwise RMSE over replications
#   bias     -- mean integrated signed bias
#   variance -- across-replication variance of the integrated estimate
#   coverage, width, postvar -- as recorded
# ──────────────────────────────────────────────────────────────────────────────
# res <- readRDS("simulations/<experiment_dir>/<base_name>_output.rds")
#
# grp <- cbind(rmse, bias, coverage, width, postvar) ~ method + population
# mean_tbl <- aggregate(grp, data = res,
#                       FUN = function(x) mean(x, na.rm = TRUE),
#                       na.action = na.pass)
# var_tbl  <- aggregate(bias ~ method + population, data = res,
#                       FUN = function(x) var(x, na.rm = TRUE),
#                       na.action = na.pass)
# names(var_tbl)[ncol(var_tbl)] <- "variance"
# summary_tbl <- merge(mean_tbl, var_tbl)
# summary_tbl$sd <- sqrt(summary_tbl$variance)
# print(summary_tbl, row.names = FALSE, digits = 3)
#
# # Figures: write straight to the manuscript figures directory so a re-run
# # updates the paper.  Okabe-Ito colours, shared across all paper figures:
# #   Fusion #E69F00 | RCT-only #009E73 | RWD-only #D55E00
# # fig_dir <- "manuscripts/manuscript_v2/general/figures"
# # dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)
# # pdf(file.path(fig_dir, "<figure>.pdf"), width = 18, height = 10,
# #     family = "Times")
# # ...boxplots over replications, grouped by cell, methods side by side...
# # dev.off()
