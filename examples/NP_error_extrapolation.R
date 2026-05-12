# =============================================================================
# Simulation study: extrapolation under shorter RCT follow-up.
#
# Same DGP family as examples/NP_error_simulation.R, but here we
#   - fix a single OS residual scenario (configurable; default = gumbel),
#   - compare only two methods: error_dist = gaussian vs source_hdp,
#   - administratively right-censor the RCT at 70% of the largest
#     non-censored event time in the OS (per replicate).
#
# Motivation
#   The OS has longer follow-up than the RCT.  Under AFT, the treatment
#   effect tau(X) is a log-time shift; the question is whether a heavily
#   admin-censored RCT can still anchor tau (and disentangle it from c)
#   when the OS error tail is misspecified by the gaussian working model
#   but absorbed by the HDP working model.
#
# Outputs
#   - examples/NP_error_extrapolation_results.rds  (raw per-fit table)
#   - aggregated mean (sd) summary printed to console
# =============================================================================

setwd("~/Library/CloudStorage/OneDrive-VrijeUniversiteitAmsterdam/Documents/GitHub/FusionForests")

library(FusionForests)
library(parallel)

# ---- Config ---------------------------------------------------------
scenario   <- "gumbel"                            # "bimodal" | "gumbel" | "logistic"
errorDists <- c("gaussian", "source_hdp")

nRep   <- 25L
nCores <- 5L

n_rct  <- 100L
n_os   <- 200L
p      <- 5L
q      <- 2L
N_post <- 2500L
N_burn <- 1500L

cens_frac        <- 0.70                          # RCT admin cutoff: 70% of max OS event
random_cens_rate <- 0.25                          # per-source random censoring rate

outFile <- "examples/NP_error_extrapolation_results.rds"

# ---- Helpers --------------------------------------------------------
# Random censoring independent of the event time, applied per source.
# Each obs is censored w.p. `rate`; the censoring time on the log-time
# scale is set to y_event - Exp(1), which guarantees observed < event
# without requiring an additional distributional knob.
random_censor <- function(y_event, rate) {
  n     <- length(y_event)
  cens  <- rbinom(n, 1, rate)
  shift <- rexp(n, rate = 1)
  list(
    y_obs  = y_event - cens * shift,
    status = as.integer(cens == 0L)
  )
}

# ---- Data-generating process ----------------------------------------
gen_data <- function(scenario) {
  X_rct <- matrix(runif(n_rct * p), n_rct, p)
  X_os  <- matrix(runif(n_os  * p), n_os,  p)
  U_os  <- matrix(rnorm(n_os * q), n_os, q)

  m0   <- function(X) 2*X[,1] - X[,2] + 0.5*X[,3]
  tau  <- function(X) X[,1] + 0.5 * X[,2]^2
  conf <- function(U) -0.5*U[,1] + 0.25*U[,2]

  A_rct <- rbinom(n_rct, 1, 0.5)
  A_os  <- rbinom(n_os,  1, plogis(X_os[,1] + U_os[,1] + U_os[,2]))

  eps_rct <- rnorm(n_rct, 0, 0.5)

  eps_os <- switch(scenario,
    bimodal = {
      m <- rbinom(n_os, 1, 0.5)
      ifelse(m == 1, rnorm(n_os,  1, 0.2),
                     rnorm(n_os, -1, 0.2))
    },
    gumbel = {
      g       <- -log(-log(runif(n_os)))
      gamma_E <- 0.5772156649015329
      sd_g    <- pi / sqrt(6)
      (g - gamma_E) * (1.0 / sd_g)
    },
    logistic = {
      qlogis(runif(n_os)) * (1.0 / (pi / sqrt(3)))
    },
    stop("unknown scenario: ", scenario)
  )

  y_rct_event <- m0(X_rct) + A_rct * tau(X_rct) + eps_rct
  y_os_event  <- m0(X_os)  + A_os  * tau(X_os) + A_os * conf(U_os) + eps_os

  # Random censoring per source (independent of event time).
  os_rc  <- random_censor(y_os_event,  random_cens_rate)
  rct_rc <- random_censor(y_rct_event, random_cens_rate)

  # Admin censoring threshold: cens_frac * max non-censored OS event.
  t_cut <- cens_frac * max(os_rc$y_obs[os_rc$status == 1L])

  # Admin censoring of the RCT, applied on top of the random censoring:
  # any obs (event or random-censored) sitting above t_cut is replaced
  # by t_cut with status forced to 0.
  admin_cens <- rct_rc$y_obs > t_cut
  y_rct      <- ifelse(admin_cens, t_cut, rct_rc$y_obs)
  status_rct <- ifelse(admin_cens, 0L,    rct_rc$status)

  list(
    X_train          = rbind(X_rct, X_os),
    A_train          = c(A_rct, A_os),
    S_train          = c(rep(1L, n_rct), rep(0L, n_os)),
    y_train          = c(y_rct, os_rc$y_obs),
    status_train     = c(status_rct, os_rc$status),
    true_cate_rct    = tau(X_rct),
    true_cate_os     = tau(X_os),
    t_cut            = t_cut,
    rct_cens_rate    = mean(status_rct == 0L),
    rct_admin_rate   = mean(admin_cens),
    os_cens_rate     = mean(os_rc$status == 0L)
  )
}

# ---- Single fit ------------------------------------------------------
fit_one <- function(dat, error_dist) {
  t0  <- proc.time()[["elapsed"]]
  fit <- FusionForest(
    y                         = dat$y_train,
    status                    = dat$status_train,
    X_train_control           = dat$X_train,
    X_train_treat             = dat$X_train,
    treatment_indicator_train = dat$A_train,
    source_indicator_train    = dat$S_train,
    outcome_type              = "right-censored",
    timescale                 = "log",
    decomposition             = "three-forest",
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
  idx_os    <- (n_rct + 1L):(n_rct + n_os)
  truth_all <- c(dat$true_cate_rct, dat$true_cate_os)

  data.frame(
    error_dist     = error_dist,
    runtime_s      = elapsed,
    sigma_mean     = mean(fit$sigma),
    t_cut          = dat$t_cut,
    rct_cens_rate  = dat$rct_cens_rate,
    rct_admin_rate = dat$rct_admin_rate,
    os_cens_rate   = dat$os_cens_rate,
    rmse_rct       = sqrt(mean((cate_hat[idx_rct] - dat$true_cate_rct)^2)),
    rmse_rwd       = sqrt(mean((cate_hat[idx_os]  - dat$true_cate_os )^2)),
    rmse_all       = sqrt(mean((cate_hat          - truth_all       )^2)),
    stringsAsFactors = FALSE
  )
}

# ---- One rep task ----------------------------------------------------
run_task <- function(task) {
  msg <- sprintf("[pid %5d] scenario=%-8s rep=%2d  start\n",
                 Sys.getpid(), scenario, task$rep)
  cat(msg)
  dat <- gen_data(scenario)
  out <- do.call(rbind, lapply(errorDists, function(ed) fit_one(dat, ed)))
  cbind(rep = task$rep, out)
}

# ---- Task list -------------------------------------------------------
tasks <- list()
for (r in seq_len(nRep))
  tasks[[length(tasks) + 1L]] <- list(rep = r)

cat(sprintf("Running %d tasks on %d cores  (scenario = %s, %d reps)\n",
            length(tasks), nCores, scenario, nRep))

# Force-realize all lazy-loaded bindings in the parent BEFORE forking.
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

errs <- vapply(res_list, inherits, logical(1), what = "try-error")
if (any(errs)) {
  cat("\n!! Worker errors:\n")
  for (i in which(errs)) {
    cat(sprintf("  task %d (rep=%d):\n", i, tasks[[i]]$rep))
    print(res_list[[i]])
  }
  res_list <- res_list[!errs]
}

if (length(res_list) == 0L)
  stop("All tasks failed — see worker errors above. ",
       "No results to aggregate.")

results <- do.call(rbind, res_list)
results <- results[order(results$rep, results$error_dist), ]

dir.create(dirname(outFile), showWarnings = FALSE, recursive = TRUE)
saveRDS(results, outFile)
cat(sprintf("\nTotal wall time: %.1f s   (%.1f min)\n",
            elapsed_global, elapsed_global / 60))
cat(sprintf("Raw per-fit results written to %s\n", outFile))

# ---- Aggregate -------------------------------------------------------
agg <- aggregate(
  cbind(runtime_s, sigma_mean, t_cut,
        rct_cens_rate, rct_admin_rate, os_cens_rate,
        rmse_rct, rmse_rwd, rmse_all) ~ error_dist,
  data = results,
  FUN  = function(x) c(mean = mean(x), sd = sd(x))
)

fmt_msd <- function(m, s) sprintf("%.3f (%.3f)", m, s)
summary_tab <- with(agg, data.frame(
  error_dist     = error_dist,
  runtime_s      = sprintf("%.2f", runtime_s[, "mean"]),
  sigma          = fmt_msd(sigma_mean[, "mean"], sigma_mean[, "sd"]),
  t_cut          = fmt_msd(t_cut[, "mean"], t_cut[, "sd"]),
  rct_cens_rate  = fmt_msd(rct_cens_rate[, "mean"], rct_cens_rate[, "sd"]),
  rct_admin_rate = fmt_msd(rct_admin_rate[, "mean"], rct_admin_rate[, "sd"]),
  os_cens_rate   = fmt_msd(os_cens_rate[, "mean"], os_cens_rate[, "sd"]),
  RMSE_RCT       = fmt_msd(rmse_rct[, "mean"], rmse_rct[, "sd"]),
  RMSE_RWD       = fmt_msd(rmse_rwd[, "mean"], rmse_rwd[, "sd"]),
  RMSE_All       = fmt_msd(rmse_all[, "mean"], rmse_all[, "sd"]),
  stringsAsFactors = FALSE,
  check.names = FALSE
))

summary_tab$error_dist <- factor(summary_tab$error_dist, levels = errorDists)
summary_tab <- summary_tab[order(summary_tab$error_dist), ]

cat(sprintf("\n========== Aggregated results  (mean (sd) over %d replicates, scenario = %s) ==========\n",
            nRep, scenario))
print(summary_tab, row.names = FALSE)
