################################################################################
# Main simulation study (data fusion for heterogeneous survival effects).
# SINGLE-RUN INSPECTION script.
#
# Purpose: simulate ONE dataset under one (scenario, regime) cell, fit the three
# estimators of the main simulation, and dump a rich set of statistics and
# diagnostic plots so the design can be eyeballed before launching the full
# Monte Carlo study on the HPC (see simulation_main_hpc.R).
#
# Estimators (all via the FusionForest AFT-BART backend, so the comparison
# isolates the fusion design, not the implementation):
#   (1) Bayesian fusion forests  : four-forest, both sources, confounding
#                                  function c, source-specific HDP residual.
#   (2) RCT-only                 : CausalShrinkageForest on the trial alone.
#   (3) Naive pooling            : CausalShrinkageForest on the pooled data.
# RCT-only and Naive-pool use CausalShrinkageForest: FusionForest is reserved
# for the multi-source fusion fit, so a single-source FusionForest is
# ill-defined.
#
# DGP follows notes/simulation/SIMULATION.tex.  X ~ U(-1/2, 1/2)^10 (5 active),
# log T = m0_sh(X) + (1-S) lambda_d g(X) + beta_U U + A {tau(X) + gamma_U U} + eps.
# RCT: right + administrative censoring.  RWD: interval censoring.  Heterogeneous
# regime: log-normal RCT residual, Weibull (recentred smallest-extreme-value)
# RWD residual.
#
# Usage:
#   Rscript simulation_main_single.R [N_post] [N_burn] [scenario] [regime]
#   e.g.    Rscript simulation_main_single.R 2000 2000 S2 heterogeneous
################################################################################

setwd("~/Library/CloudStorage/OneDrive-VrijeUniversiteitAmsterdam/Documents/GitHub/FusionForests")

.libPaths(c("~/Rlib", .libPaths()))
suppressMessages(library(FusionForests))
suppressMessages(library(ShrinkageTrees))

# ── CLI / settings ───────────────────────────────────────────────────────────
args        <- commandArgs(trailingOnly = TRUE)
N_post      <- if (length(args) >= 1) as.integer(args[1]) else 2000L
N_burn      <- if (length(args) >= 2) as.integer(args[2]) else 2000L
scenario_nm <- if (length(args) >= 3) args[3] else "S0"
regime_nm   <- if (length(args) >= 4) args[4] else "heterogeneous"

n_rct   <- 100L
n_rwd   <- 400L
p_total <- 10L
n_bins  <- 10L          # RWD interval-censoring inspection bins
out_dir <- "simulations/main"
use_censoring <- FALSE  # FALSE = fully observed (diagnostic); TRUE = censored design

# ── DGP ────────────────────────────────────────────────────────────────────--
scenarios <- list(
  S0 = c(alpha_U = 0.0, beta_U = 1.0, gamma_U = 0.0),
  S1 = c(alpha_U = 0.5, beta_U = 1.0, gamma_U = 0.0),
  S2 = c(alpha_U = 1.5, beta_U = 1.5, gamma_U = 0.0),
  S3 = c(alpha_U = 1.0, beta_U = 1.0, gamma_U = 0.5)
)
gamma_E <- 0.5772156649

true_tau <- function(X) 1 + 0.5 * X[, 1] + (1 / 3) * (X[, 2] > 0)
m0_sh    <- function(X) sin(pi * X[, 1]) + X[, 2] * X[, 3] + (X[, 4]^2 - 1 / 12)
g_dev    <- function(X) X[, 1] - 0.5 * X[, 5]

structural_logT <- function(X, A, S, U, scn, lambda_d) {
  m0_sh(X) + (1 - S) * lambda_d * g_dev(X) + scn["beta_U"] * U +
    A * (true_tau(X) + scn["gamma_U"] * U)
}

rwd_propensity <- function(X, U, scn, a0)
  plogis(a0 + X[, 1] + X[, 2] + scn["alpha_U"] * U)

# Residual draws.  matched: N(0,1) both.  heterogeneous: RCT ~ N(0,0.6^2)
# (log-normal event times); RWD ~ recentred smallest-extreme-value with SD 1
# (Weibull event times).
draw_eps <- function(n, source, regime, sigma0 = 1.0, sigma_rct = 0.6) {
  # if (regime == "matched") return(rnorm(n, 0, 1))
  # if (source == "rct")     return(rnorm(n, 0, sigma_rct))
  # sigma_W <- sigma0 * sqrt(6) / pi
  # sigma_W * (log(rexp(n)) + gamma_E)            # SEV, mean 0, SD sigma0
  return(rnorm(n, 0, 0.5))   # test: single Gaussian N(0, 1/2) for both sources
}

solve_alpha_0 <- function(scn, n_search = 1e5) {
  X <- matrix(runif(n_search * p_total, -0.5, 0.5), n_search, p_total)
  U <- rnorm(n_search)
  uniroot(function(a0) mean(rwd_propensity(X, U, scn, a0)) - 0.5,
          c(-10, 10))$root
}

# Calibrate, on a large pre-simulation of the chosen cell:
#   vmax = max event time, tRCT = vmax / 3 (short trial horizon),
#   eta  = exponential censoring rate giving ~35% random censoring in the RCT.
calibrate_censoring <- function(scn, regime, a0, n_search = 2e5,
                                target_cens = 0.35) {
  lambda_d <- if (regime == "heterogeneous") 0.5 else 0.0
  X  <- matrix(runif(n_search * p_total, -0.5, 0.5), n_search, p_total)
  U  <- rnorm(n_search)
  A  <- rbinom(n_search, 1L, 0.5)
  e  <- draw_eps(n_search, "rct", regime)
  T  <- exp(structural_logT(X, A, S = 1, U, scn, lambda_d) + e)
  vmax <- as.numeric(quantile(T, 0.999))        # robust max
  tRCT <- vmax / 3
  eta  <- uniroot(function(r) mean(1 - exp(-r * T)) - target_cens,
                  c(1e-4, 50))$root
  # RWD inspection visits at the event-time deciles up to and including q90;
  # events beyond the last visit (q90) are right-censored there.
  visits <- as.numeric(quantile(T, seq(0.1, 0.9, 0.1)))
  list(vmax = vmax, tRCT = tRCT, eta = eta, visits = visits)
}

make_data <- function(scn, regime, n1, n0, a0, cal, use_censoring = TRUE) {
  lambda_d <- if (regime == "heterogeneous") 0.5 else 0.0

  # RCT: randomised, right + administrative censoring.
  X1 <- matrix(runif(n1 * p_total, -0.5, 0.5), n1, p_total)
  U1 <- rnorm(n1); A1 <- rbinom(n1, 1L, 0.5)
  lT1 <- structural_logT(X1, A1, S = 1, U1, scn, lambda_d) +
         draw_eps(n1, "rct", regime)
  T1  <- exp(lT1)
  if (use_censoring) {
    C1    <- rexp(n1, rate = cal$eta)
    obs1  <- pmin(T1, C1, cal$tRCT)
    stat1 <- as.integer(T1 <= pmin(C1, cal$tRCT))
  } else {
    obs1  <- T1
    stat1 <- rep(1L, n1)
  }

  # RWD: confounded assignment, interval censoring on a 0..vmax grid.
  X0 <- matrix(runif(n0 * p_total, -0.5, 0.5), n0, p_total)
  U0 <- rnorm(n0)
  A0 <- rbinom(n0, 1L, rwd_propensity(X0, U0, scn, a0))
  lT0 <- structural_logT(X0, A0, S = 0, U0, scn, lambda_d) +
         draw_eps(n0, "rwd", regime)
  T0  <- exp(lT0)
  if (use_censoring) {
    # RWD interval grid: visits at the event-time deciles up to and including q90.
    brks       <- c(0, cal$visits)                 # edges 0, q10, ..., q90 (9 bins)
    last_visit <- cal$visits[length(cal$visits)]   # q90 = final follow-up visit
    j      <- findInterval(T0, brks, rightmost.closed = TRUE)
    beyond <- T0 > last_visit                       # no visit after q90
    j[j < 1L] <- 1L
    logfloor <- min(c(lT1, lT0)) - 5
    left0  <- ifelse(j == 1L, logfloor, log(brks[j]))
    right0 <- log(brks[pmin(j + 1L, length(brks))])
    stat0  <- rep(0L, n0)
    icc0   <- rep(1L, n0)
    # events past the last visit are right-censored at q90
    left0[beyond]  <- log(last_visit)
    right0[beyond] <- log(last_visit)
    icc0[beyond]   <- 0L
    y0 <- right0
  } else {
    # fully observed (diagnostic: no censoring)
    left0 <- lT0; right0 <- lT0
    stat0 <- rep(1L, n0); icc0 <- rep(0L, n0); y0 <- lT0
  }

  list(
    X      = rbind(X1, X0),
    A      = c(A1, A0),
    S      = c(rep(1L, n1), rep(0L, n0)),
    y      = c(log(obs1), y0),
    status = c(stat1, stat0),
    icc    = c(rep(0L, n1), icc0),
    left   = c(log(obs1), left0),
    right  = c(log(obs1), right0),
    n1 = n1, n0 = n0,
    eps_rct = lT1 - structural_logT(X1, A1, 1, U1, scn, lambda_d),
    eps_rwd = lT0 - structural_logT(X0, A0, 0, U0, scn, lambda_d),
    cens_rct = mean(stat1 == 0), cens_rwd_interval = mean(icc0 == 1)
  )
}

# ── Fits (all three through FusionForest) ─────────────────────────────────────
# Default FusionForest hyperparameters (matching examples/cate_rmse_comparison.R):
# 200 trees per forest, k = 0.5 (k_g = 0.25), gaussian residual.
trees <- list(control = 200L, treat = 200L, deconf = 200L, deviation = 200L)
ks    <- list(control = 0.5, treat = 0.5, deconf = 0.5, g = 0.25)

fit_fusion <- function(d, Xev) {
  FusionForest(
    y = d$y, status = d$status,
    observed_left_time = d$left, observed_right_time = d$right,
    interval_censoring_indicator = d$icc,
    X_train_control = d$X, X_train_treat = d$X,
    treatment_indicator_train = d$A, source_indicator_train = d$S,
    X_test_control = Xev, X_test_treat = Xev, X_test_deconf = Xev,
    treatment_indicator_test = rep(1L, nrow(Xev)),
    source_indicator_test = rep(1L, nrow(Xev)),
    outcome_type = "right-censored", timescale = "log",
    decomposition = "four-forest",
    number_of_trees_control = trees$control, number_of_trees_treat = trees$treat,
    number_of_trees_deconf = trees$deconf,
    number_of_trees_deviation = trees$deviation,
    k_control = ks$control, k_treat = ks$treat,
    k_deconf = ks$deconf, k_g = ks$g,
    N_post = N_post, N_burn = N_burn, treatment_coding = "centered",
    error_dist = "gaussian", store_posterior_sample = TRUE,
    verbose = FALSE)
}

# Single-source / pooled causal forest (CausalShrinkageForest).  Used for the
# RCT-only and naive-pool baselines: FusionForest is reserved for the
# multi-source fusion fit, so a single-source FusionForest is ill-defined.
# Right-censored when no interval-censored rows are present; otherwise
# interval-censored via the interval2 convention (exact: left == right;
# right-censored: right = Inf; interval: finite left < right).
fit_single_source <- function(y, status, X, A, Xev, icc = NULL,
                              left = NULL, right = NULL) {
  common <- list(
    X_train_control = X, X_train_treat = X,
    treatment_indicator_train = A,
    X_test_control = Xev, X_test_treat = Xev,
    treatment_indicator_test = rep(1L, nrow(Xev)),
    timescale = "log",
    prior_type_control = "standard", prior_type_treat = "standard",
    local_hp_control = 0.5 / sqrt(trees$control),
    local_hp_treat   = 0.5 / sqrt(trees$treat),
    number_of_trees_control = trees$control,
    number_of_trees_treat   = trees$treat,
    N_post = N_post, N_burn = N_burn, treatment_coding = "centered",
    store_posterior_sample = TRUE, verbose = FALSE)

  if (is.null(icc) || all(icc == 0)) {
    do.call(ShrinkageTrees::CausalShrinkageForest,
            c(list(y = y, status = status, outcome_type = "right-censored"),
              common))
  } else {
    lt <- left; rt <- right
    ev <- status == 1;            lt[ev] <- y[ev]; rt[ev] <- y[ev]  # exact
    rc <- status == 0 & icc == 0; rt[rc] <- Inf                     # right-cens
    do.call(ShrinkageTrees::CausalShrinkageForest,
            c(list(left_time = lt, right_time = rt,
                   outcome_type = "interval-censored"), common))
  }
}

# ── Metric helper ─────────────────────────────────────────────────────────────
eval_cate <- function(fit, truth) {
  tau_hat  <- fit$test_predictions_treat
  samp     <- fit$test_predictions_sample_treat
  ci       <- apply(samp, 2, quantile, probs = c(0.025, 0.975))
  c(rmse     = sqrt(mean((tau_hat - truth)^2)),
    bias     = mean(tau_hat - truth),
    coverage = mean(truth >= ci[1, ] & truth <= ci[2, ]),
    width    = mean(ci[2, ] - ci[1, ]))
}

# ── Run ───────────────────────────────────────────────────────────────────────
cat(sprintf("MAIN SIM single run | scenario=%s regime=%s | n1=%d n0=%d | N=%d+%d\n",
            scenario_nm, regime_nm, n_rct, n_rwd, N_burn, N_post))
scn  <- scenarios[[scenario_nm]]
a0   <- solve_alpha_0(scn)
cal  <- calibrate_censoring(scn, regime_nm, a0)
cat(sprintf("  alpha_0=%+.3f  eta=%.4f  tRCT=%.3f  vmax=%.3f\n",
            a0, cal$eta, cal$tRCT, cal$vmax))

d    <- make_data(scn, regime_nm, n_rct, n_rwd, a0, cal, use_censoring)
Xev  <- d$X
truth <- true_tau(Xev)
cat(sprintf("  treated: RCT=%.2f RWD=%.2f | RCT right-cens=%.2f | RWD interval=%.2f\n",
            mean(d$A[d$S == 1]), mean(d$A[d$S == 0]),
            d$cens_rct, d$cens_rwd_interval))

cat("Fitting (1/3) Bayesian fusion forests ...\n"); t <- Sys.time()
fit_fus  <- fit_fusion(d, Xev)
cat("  ", round(difftime(Sys.time(), t, units = "secs"), 1), "s\n")

cat("Fitting (2/3) RCT-only ...\n"); t <- Sys.time()
rct <- d$S == 1
fit_rct  <- fit_single_source(d$y[rct], d$status[rct], d$X[rct, ], d$A[rct], Xev)
cat("  ", round(difftime(Sys.time(), t, units = "secs"), 1), "s\n")

cat("Fitting (3/3) Naive pooling ...\n"); t <- Sys.time()
fit_pool <- fit_single_source(d$y, d$status, d$X, d$A, Xev,
                              icc = d$icc, left = d$left, right = d$right)
cat("  ", round(difftime(Sys.time(), t, units = "secs"), 1), "s\n")

fits    <- list(`Fusion` = fit_fus, `RCT-only` = fit_rct, `Naive pool` = fit_pool)
metrics <- t(sapply(fits, eval_cate, truth = truth))

cat("\n=== CATE metrics (in-sample on combined covariates, log-AF scale) ===\n")
print(round(metrics, 4))

# ── Diagnostics PDF ───────────────────────────────────────────────────────────
pdf(file.path(out_dir, "single_run_diagnostics.pdf"), width = 10, height = 7)
cols <- c(Fusion = "#1b9e77", `RCT-only` = "#d95f02", `Naive pool` = "#7570b3")

# (1) estimated vs true CATE
par(mfrow = c(2, 3), mar = c(4, 4, 3, 1))
for (m in names(fits)) {
  th <- fits[[m]]$test_predictions_treat
  plot(truth, th, pch = 19, col = adjustcolor(cols[m], 0.4), cex = 0.5,
       xlab = "true CATE", ylab = "posterior-mean CATE",
       main = sprintf("%s\nRMSE=%.3f bias=%+.3f", m, metrics[m, "rmse"],
                      metrics[m, "bias"]))
  abline(0, 1, lwd = 2, lty = 2)
}

# (2) CATE vs X1, with 95% CI, split by the X2>0 effect modifier
for (m in names(fits)) {
  th  <- fits[[m]]$test_predictions_treat
  ci  <- apply(fits[[m]]$test_predictions_sample_treat, 2,
               quantile, probs = c(0.025, 0.975))
  o   <- order(Xev[, 1])
  plot(Xev[o, 1], th[o], type = "n", ylim = range(ci),
       xlab = "X1", ylab = "CATE", main = paste(m, "(CI vs X1)"))
  polygon(c(Xev[o, 1], rev(Xev[o, 1])), c(ci[1, o], rev(ci[2, o])),
          col = adjustcolor(cols[m], 0.2), border = NA)
  points(Xev[, 1], th, pch = 19, cex = 0.4, col = adjustcolor(cols[m], 0.6))
  points(Xev[, 1], truth, pch = 19, cex = 0.4, col = "black")
}

# (3) residual heterogeneity (the two source error laws) + metric bars
par(mfrow = c(1, 2), mar = c(4, 4, 3, 1))
xr <- range(c(d$eps_rct, d$eps_rwd))
hist(d$eps_rwd, breaks = 30, col = adjustcolor("#7570b3", 0.5), border = NA,
     freq = FALSE, xlim = xr, xlab = "true residual",
     main = sprintf("Source residuals (%s)\nRCT lognormal / RWD Weibull",
                    regime_nm))
hist(d$eps_rct, breaks = 30, col = adjustcolor("#d95f02", 0.5), border = NA,
     freq = FALSE, add = TRUE)
legend("topright", c("RWD (eps)", "RCT (eps)"), fill = c("#7570b3", "#d95f02"),
       bty = "n")

bp <- barplot(metrics[, "rmse"], col = cols[rownames(metrics)],
              ylab = "CATE RMSE", main = "CATE RMSE by method",
              ylim = c(0, max(metrics[, "rmse"]) * 1.2))
text(bp, metrics[, "rmse"], round(metrics[, "rmse"], 3), pos = 3, cex = 0.9)

# (4) Kaplan-Meier curves by source x treatment.
# Base-R KM (no survival dependency).  The RWD is interval-censored, so its
# curves use interval-midpoint imputation and are approximate.
km_fit <- function(time, event) {
  o <- order(time); time <- time[o]; event <- event[o]
  ut <- sort(unique(time[event == 1]))
  if (length(ut) == 0L) return(list(time = c(0, max(time)), surv = c(1, 1)))
  atrisk <- vapply(ut, function(tt) sum(time >= tt), numeric(1))
  devs   <- vapply(ut, function(tt) sum(time == tt & event == 1), numeric(1))
  list(time = c(0, ut), surv = c(1, cumprod(1 - devs / atrisk)))
}
km_time  <- ifelse(d$icc == 1,
                   (exp(pmax(d$left, log(1e-8))) + exp(d$right)) / 2, exp(d$y))
km_event <- ifelse(d$icc == 1, 1L, d$status)
par(mfrow = c(1, 2), mar = c(4, 4, 3, 1))
arm_col <- c(`1` = "#1b9e77", `0` = "#d95f02")
for (s in c(1L, 0L)) {
  lab  <- if (s == 1L) "RCT (right-censored)" else "RWD (interval-censored, approx.)"
  xmax <- as.numeric(quantile(km_time[d$S == s], 0.95))
  plot(NA, xlim = c(0, xmax), ylim = c(0, 1), xlab = "time", ylab = "survival",
       main = paste("Kaplan-Meier:", lab))
  for (a in c(1L, 0L)) {
    idx <- d$S == s & d$A == a
    k   <- km_fit(km_time[idx], km_event[idx])
    lines(k$time, k$surv, type = "s", lwd = 2, col = arm_col[as.character(a)])
  }
  legend("topright", c("treated", "control"), lwd = 2,
         col = arm_col[c("1", "0")], bty = "n")
}
invisible(dev.off())

saveRDS(list(metrics = metrics, scenario = scenario_nm, regime = regime_nm,
             calibration = cal, alpha_0 = a0,
             cens = c(rct = d$cens_rct, rwd_interval = d$cens_rwd_interval)),
        file.path(out_dir, "single_run_results.rds"))

cat("\nWrote", file.path(out_dir, "single_run_diagnostics.pdf"), "and",
    file.path(out_dir, "single_run_results.rds"), "\n")
cat("DONE\n")
