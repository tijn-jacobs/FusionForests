# =============================================================================
# Simulation study: error_dist = gaussian / shared_dp / source_dp /
# source_dp_scale under three misspecified residual distributions in the RWD arm.
#
# Design
#   nRep replicates per scenario, 3 scenarios, 4 methods =>
#   3 * 4 * nRep model fits in total.
#   Parallelised over nCores cores via parallel::mclapply (forks; macOS / Linux).
#
# Scenarios (RWD residuals; RCT residuals are always Gaussian N(0, 0.5^2))
#   bimodal  : 0.5 N(-1, 0.2^2) + 0.5 N(+1, 0.2^2)         — symmetric, two modes
#   gumbel   : centred standard Gumbel scaled to SD 1.0    — right-skewed (Weibull
#              event times on the log scale).  SD=1.0 gives a noise-to-signal
#              ratio large enough that the right-tail outliers actually drag the
#              BART leaf means; with the earlier SD=0.5 the misspecification was
#              too mild to show in CATE RMSE.
#   logistic : standard logistic scaled to SD 1.0          — symmetric, heavy tails
#              (log-logistic event times on the log scale).  Same rationale.
#
# Each residual is mean-zero so the structural mean components remain identified.
#
# Outputs
#   - examples/NP_error_simulation_results.rds   (raw per-fit table)
#   - aggregated mean (sd) summary printed to console
#
# Notes
#   OMP_NUM_THREADS is forced to 1 below to avoid oversubscription when the
#   underlying BLAS / OpenMP runtime would otherwise fan out inside each forked
#   worker.  Without it you can end up with `nCores * omp_max_threads`
#   software threads competing for ~`nCores` physical ones.
# =============================================================================

setwd("~/Library/CloudStorage/OneDrive-VrijeUniversiteitAmsterdam/Documents/GitHub/FusionForests")

library(FusionForests)
library(parallel)

# ---- Config ---------------------------------------------------------
nRep       <- 25L
nCores     <- 5L
scenarios   <- c("bimodal", "gumbel", "logistic")
errorDists <- c("gaussian", "shared_dp", "source_dp", "source_dp_scale", "source_hdp")

n_rct  <- 100L
n_rwd   <- 200L
p      <- 5L
q      <- 2L
N_post <- 2500L
N_burn <- 1500L

outFile <- "examples/NP_error_simulation_results.rds"

# ---- Data-generating process ----------------------------------------
gen_data <- function(scenario) {
  X_rct <- matrix(runif(n_rct * p), n_rct, p)
  X_rwd  <- matrix(runif(n_rwd  * p), n_rwd,  p)
  U_rwd  <- matrix(rnorm(n_rwd * q), n_rwd, q)

  m0   <- function(X) 2*X[,1] - X[,2] + 0.5*X[,3]
  tau  <- function(X) X[,1] + 0.5 * X[,2]^2
  conf <- function(U) -0.5*U[,1] + 0.25*U[,2]

  A_rct <- rbinom(n_rct, 1, 0.5)
  A_rwd  <- rbinom(n_rwd,  1, plogis(X_rwd[,1] + U_rwd[,1] + U_rwd[,2]))

  eps_rct <- rnorm(n_rct, 0, 0.5)

  eps_rwd <- switch(scenario,
    bimodal = {
      m <- rbinom(n_rwd, 1, 0.5)
      ifelse(m == 1, rnorm(n_rwd,  1, 0.2),
                     rnorm(n_rwd, -1, 0.2))
    },
    gumbel = {
      g       <- -log(-log(runif(n_rwd)))               # standard Gumbel(0, 1)
      gamma_E <- 0.5772156649015329                    # mean of standard Gumbel
      sd_g    <- pi / sqrt(6)                          # SD of standard Gumbel
      (g - gamma_E) * (1.0 / sd_g)                     # mean-zero, SD = 1.0
    },
    logistic = {
      qlogis(runif(n_rwd)) * (1.0 / (pi / sqrt(3)))     # mean-zero, SD = 1.0
    },
    stop("unknown scenario: ", scenario)
  )

  y_rct <- m0(X_rct) + A_rct * tau(X_rct) + eps_rct
  y_rwd  <- m0(X_rwd)  + A_rwd  * tau(X_rwd) + A_rwd * conf(U_rwd) + eps_rwd

  list(
    X_train       = rbind(X_rct, X_rwd),
    A_train       = c(A_rct, A_rwd),
    S_train       = c(rep(1L, n_rct), rep(0L, n_rwd)),
    y_train       = c(y_rct, y_rwd),
    true_cate_rct = tau(X_rct),
    true_cate_rwd  = tau(X_rwd)
  )
}

# ---- Single fit ------------------------------------------------------
fit_one <- function(dat, error_dist) {
  t0  <- proc.time()[["elapsed"]]
  fit <- FusionForest(
    y                         = dat$y_train,
    X_train_control           = dat$X_train,
    X_train_treat             = dat$X_train,
    treatment_indicator_train = dat$A_train,
    source_indicator_train    = dat$S_train,
    outcome_type              = "continuous",
    decomposition             = "four-forest",
    error_dist                = error_dist,
    error_truncation_K        = 50L,
    error_atom_scale          = 0.5,
    error_mass_init           = 1.0,
    N_post                    = N_post,
    N_burn                    = N_burn,
    verbose                   = FALSE
  )
  elapsed   <- proc.time()[["elapsed"]] - t0

  cate_hat  <- fit$train_predictions_treat
  idx_rct   <- seq_len(n_rct)
  idx_rwd    <- (n_rct + 1L):(n_rct + n_rwd)
  truth_all <- c(dat$true_cate_rct, dat$true_cate_rwd)

  data.frame(
    error_dist = error_dist,
    runtime_s  = elapsed,
    sigma_mean = mean(fit$sigma),
    rmse_rct   = sqrt(mean((cate_hat[idx_rct] - dat$true_cate_rct)^2)),
    rmse_rwd   = sqrt(mean((cate_hat[idx_rwd]  - dat$true_cate_rwd )^2)),
    rmse_all   = sqrt(mean((cate_hat          - truth_all       )^2)),
    stringsAsFactors = FALSE
  )
}

# ---- One (scenario, rep) task ---------------------------------------
run_task <- function(task) {
  msg <- sprintf("[pid %5d] scenario=%-8s rep=%2d  start\n",
                 Sys.getpid(), task$scenario, task$rep)
  cat(msg)
  dat <- gen_data(task$scenario)
  out <- do.call(rbind, lapply(errorDists, function(ed) fit_one(dat, ed)))
  cbind(scenario = task$scenario, rep = task$rep, out)
}

# ---- Task list -------------------------------------------------------
tasks <- list()
for (sc in scenarios)
  for (r in seq_len(nRep))
    tasks[[length(tasks) + 1L]] <- list(scenario = sc, rep = r)

cat(sprintf("Running %d tasks on %d cores  (%d reps x %d scenarios)\n",
            length(tasks), nCores, nRep, length(scenarios)))

# Force-realize all lazy-loaded bindings in the parent BEFORE forking.
# Without this, `mclapply` workers inherit stale handles to
# FusionForests.rdb and the first call into a not-yet-realized binding
# (e.g. FusionForest_cpp) dies with
#   "lazy-load database '.../FusionForests.rdb' is corrupt".
local({
  ns <- asNamespace("FusionForests")
  invisible(eapply(ns, force, all.names = TRUE))
})

# ---- Run -------------------------------------------------------------
t_global <- proc.time()[["elapsed"]]
res_list <- mclapply(tasks, run_task,
                    mc.cores         = nCores,
                    mc.preschedule   = FALSE,
                    mc.allow.recursive = FALSE)
elapsed_global <- proc.time()[["elapsed"]] - t_global

# Surface any silent worker failures
errs <- vapply(res_list, inherits, logical(1), what = "try-error")
if (any(errs)) {
  cat("\n!! Worker errors:\n")
  for (i in which(errs)) {
    cat(sprintf("  task %d (scenario=%s, rep=%d):\n",
                i, tasks[[i]]$scenario, tasks[[i]]$rep))
    print(res_list[[i]])
  }
  res_list <- res_list[!errs]
}

if (length(res_list) == 0L)
  stop("All tasks failed — see worker errors above. ",
       "No results to aggregate.")

results <- do.call(rbind, res_list)
results <- results[order(results$scenario, results$rep, results$error_dist), ]

dir.create(dirname(outFile), showWarnings = FALSE, recursive = TRUE)
saveRDS(results, outFile)
cat(sprintf("\nTotal wall time: %.1f s   (%.1f min)\n",
            elapsed_global, elapsed_global / 60))
cat(sprintf("Raw per-fit results written to %s\n", outFile))

# ---- Aggregate -------------------------------------------------------
agg <- aggregate(
  cbind(runtime_s, sigma_mean, rmse_rct, rmse_rwd, rmse_all) ~ scenario + error_dist,
  data = results,
  FUN  = function(x) c(mean = mean(x), sd = sd(x))
)

fmt_msd <- function(m, s) sprintf("%.3f (%.3f)", m, s)
summary_tab <- with(agg, data.frame(
  scenario   = scenario,
  error_dist = error_dist,
  runtime_s  = sprintf("%.2f", runtime_s[, "mean"]),
  sigma      = fmt_msd(sigma_mean[, "mean"], sigma_mean[, "sd"]),
  RMSE_RCT   = fmt_msd(rmse_rct[, "mean"], rmse_rct[, "sd"]),
  RMSE_RWD   = fmt_msd(rmse_rwd[, "mean"], rmse_rwd[, "sd"]),
  RMSE_All   = fmt_msd(rmse_all[, "mean"], rmse_all[, "sd"]),
  stringsAsFactors = FALSE,
  check.names = FALSE
))

# order: scenario then method, with methods in the natural order
summary_tab$error_dist <- factor(summary_tab$error_dist, levels = errorDists)
summary_tab$scenario   <- factor(summary_tab$scenario,   levels = scenarios)
summary_tab <- summary_tab[order(summary_tab$scenario, summary_tab$error_dist), ]

cat(sprintf("\n========== Aggregated results  (mean (sd) over %d replicates) ==========\n",
            nRep))
print(summary_tab, row.names = FALSE)
