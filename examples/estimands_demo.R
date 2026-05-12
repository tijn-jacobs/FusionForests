# =============================================================================
# Demonstration: causal survival estimands from FusionForest fits.
#
# Steps
#   1. Simulate an RCT + larger confounded OS with a heavy-tailed OS error
#      and admin censoring on the RCT.
#   2. Plot Kaplan-Meier curves by arm and source.
#   3. Fit three models (all four-forest unless noted):
#        - "Gaussian"   : error_dist = "gaussian"
#        - "Source-HDP" : error_dist = "source_hdp"
#        - "Three-Gss"  : decomposition = "three-forest", gaussian error
#   4. Evaluate every estimand (AF, SD, RMST) at both target populations
#      (OS and RCT), pointwise and population-averaged with the Bayesian
#      bootstrap.  Print a summary table and save plots to examples/plots/.
#
# This is a small-N, short-MCMC demo intended to run in a few minutes on a
# laptop.  For reproducible benchmarking use the simulation scripts in
# examples/NP_error_*.R with more replicates and longer chains.
# =============================================================================

suppressPackageStartupMessages({
  library(FusionForests)
  library(survival)
})

set.seed(2026)

# ---- Configuration ---------------------------------------------------------
n_rct  <- 120L
n_os   <- 240L
p      <- 4L
q      <- 2L                              # number of unmeasured confounders
N_post <- 1500L
N_burn <- 1500L
n_test <- 200L                            # eval grid for the estimands

cens_frac        <- 0.70                  # RCT admin cutoff: frac of OS max
random_cens_rate <- 0.20

out_dir <- "examples/plots"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)


# ---- Data-generating process ----------------------------------------------
m0_fn   <- function(X) 2 * X[, 1] - X[, 2] + 0.5 * X[, 3]
tau_fn  <- function(X) X[, 1] + 0.5 * X[, 2]
conf_fn <- function(U) -0.5 * U[, 1] + 0.25 * U[, 2]

# Standardised Gumbel: heavy right tail; used as the OS error to stress the
# tail-misspecified Gaussian fit.
gumbel_std <- function(n) {
  g <- -log(-log(runif(n)))
  (g - 0.5772156649015329) / (pi / sqrt(6))
}

random_censor <- function(y_event, rate) {
  cens  <- rbinom(length(y_event), 1L, rate)
  shift <- rexp(length(y_event), 1)
  list(y_obs = y_event - cens * shift,
       status = as.integer(cens == 0L))
}

gen_data <- function() {
  X_rct <- matrix(runif(n_rct * p), n_rct, p)
  X_os  <- matrix(runif(n_os  * p), n_os,  p)
  U_os  <- matrix(rnorm(n_os * q), n_os, q)

  A_rct <- rbinom(n_rct, 1L, 0.5)
  A_os  <- rbinom(n_os,  1L, plogis(X_os[, 1] + U_os[, 1] + U_os[, 2]))

  eps_rct <- rnorm(n_rct, 0, 0.5)
  eps_os  <- gumbel_std(n_os)

  logT_rct <- m0_fn(X_rct) + A_rct * tau_fn(X_rct) + eps_rct
  logT_os  <- m0_fn(X_os)  + A_os  * tau_fn(X_os)  +
              A_os * conf_fn(U_os) + eps_os

  rct_rc <- random_censor(logT_rct, random_cens_rate)
  os_rc  <- random_censor(logT_os,  random_cens_rate)

  # Admin censoring on the RCT at cens_frac * max event in the OS.
  t_cut <- cens_frac * max(os_rc$y_obs[os_rc$status == 1L])
  admin <- rct_rc$y_obs > t_cut
  y_rct  <- ifelse(admin, t_cut, rct_rc$y_obs)
  st_rct <- ifelse(admin, 0L,    rct_rc$status)

  X_test <- matrix(runif(n_test * p), n_test, p)

  list(
    X_train = rbind(X_rct, X_os),
    y_train = c(y_rct, os_rc$y_obs),        # log-time scale
    status  = c(st_rct, os_rc$status),
    A       = c(A_rct, A_os),
    S       = c(rep(1L, n_rct), rep(0L, n_os)),
    t_cut   = t_cut,
    X_test  = X_test,
    true_tau_test = tau_fn(X_test)
  )
}

dat <- gen_data()
cat(sprintf("Simulated %d RCT + %d OS observations.\n",
            n_rct, n_os))
cat(sprintf("RCT admin cutoff (log time):       %.3f  (time scale: %.3f)\n",
            dat$t_cut, exp(dat$t_cut)))
cat(sprintf("Censoring rates: RCT %.2f, OS %.2f\n",
            mean(dat$status[dat$S == 1L] == 0L),
            mean(dat$status[dat$S == 0L] == 0L)))


# ---- Kaplan-Meier curves --------------------------------------------------
# Convert to the time scale for KM plotting.
T_obs <- exp(dat$y_train)
arm   <- ifelse(dat$S == 1L,
                ifelse(dat$A == 1L, "RCT/Trt", "RCT/Ctl"),
                ifelse(dat$A == 1L, "OS/Trt",  "OS/Ctl"))
arm   <- factor(arm, levels = c("RCT/Ctl", "RCT/Trt", "OS/Ctl", "OS/Trt"))

km <- survfit(Surv(T_obs, dat$status) ~ arm)

pdf(file.path(out_dir, "km_curves.pdf"), width = 7, height = 5)
op <- par(mar = c(4, 4, 2, 1) + 0.1)
plot(km, col = c("steelblue", "navy", "tomato", "firebrick"),
     lty = c(1, 1, 2, 2), lwd = 2,
     xlab = "time", ylab = "S(t)",
     main = "Kaplan-Meier by source and arm")
legend("topright", legend = levels(arm), bty = "n",
       col = c("steelblue", "navy", "tomato", "firebrick"),
       lty = c(1, 1, 2, 2), lwd = 2)
abline(v = exp(dat$t_cut), lty = 3, col = "grey50")
text(exp(dat$t_cut), 0.05, "RCT admin cutoff", pos = 2, cex = 0.8,
     col = "grey40")
par(op); dev.off()
cat("KM plot saved to ", file.path(out_dir, "km_curves.pdf"), "\n", sep = "")


# ---- Fit three models -----------------------------------------------------
common_args <- list(
  y                         = dat$y_train,
  status                    = dat$status,
  X_train_control           = dat$X_train,
  X_train_treat             = dat$X_train,
  treatment_indicator_train = dat$A,
  source_indicator_train    = dat$S,
  X_test_control            = dat$X_test,
  X_test_treat              = dat$X_test,
  treatment_indicator_test  = rep(1L, n_test),
  source_indicator_test     = rep(0L, n_test),
  outcome_type              = "right-censored",
  timescale                 = "log",
  N_post                    = N_post,
  N_burn                    = N_burn,
  store_posterior_sample    = TRUE,
  verbose                   = FALSE
)

model_specs <- list(
  Gaussian   = list(decomposition = "four-forest",  error_dist = "gaussian"),
  `Source-HDP` = list(decomposition = "four-forest",  error_dist = "source_hdp"),
  `Three-Gss`  = list(decomposition = "three-forest", error_dist = "gaussian")
)

fits <- vector("list", length(model_specs))
names(fits) <- names(model_specs)
for (nm in names(model_specs)) {
  cat(sprintf("Fitting model: %-12s ", nm))
  t0 <- proc.time()[["elapsed"]]
  fits[[nm]] <- do.call(FusionForest, c(common_args, model_specs[[nm]]))
  cat(sprintf("done (%.1fs)\n", proc.time()[["elapsed"]] - t0))
}


# ---- Estimand evaluation --------------------------------------------------
# In-window horizon: 60% of max RCT event time (well-supported).
# Extrapolation horizon: 150% of max RCT event time.
t_in_log  <- quantile(dat$y_train[dat$status == 1L & dat$S == 1L], 0.85)
t_out_log <- max(dat$y_train) + 0.5            # outside RCT support
t_in      <- exp(t_in_log)
t_out     <- exp(t_out_log)
cat(sprintf("\nHorizons:   t_in  = %.3f   t_out = %.3f\n", t_in, t_out))

# Helper: posterior summary (mean + 95% credible interval) of a draws vector.
psum <- function(v) {
  ci <- quantile(v, c(0.025, 0.975), names = FALSE)
  c(mean = mean(v), lo = ci[1], hi = ci[2])
}

# Build the result table.  For each (model, estimand, source) we report the
# population-averaged posterior mean and 95% credible interval, using the
# Bayesian bootstrap to inject covariate-distribution uncertainty.
specs <- list(
  list(estimand = "AF",   time = NA,    source = "rct"),
  list(estimand = "AF",   time = NA,    source = "os"),
  list(estimand = "SD",   time = t_in,  source = "rct"),
  list(estimand = "SD",   time = t_in,  source = "os"),
  list(estimand = "RMST", time = t_in,  source = "rct"),
  list(estimand = "RMST", time = t_in,  source = "os"),
  list(estimand = "RMST", time = t_out, source = "rct"),
  list(estimand = "RMST", time = t_out, source = "os")
)

rows <- list()
for (mname in names(fits)) {
  fit <- fits[[mname]]
  for (s in specs) {
    draws <- if (s$estimand == "AF")
      fusion_estimand(fit, "AF",
                      target_source      = s$source,
                      population_average = TRUE,
                      bayesian_bootstrap = TRUE,
                      seed               = 1L)
    else
      fusion_estimand(fit, s$estimand,
                      time               = s$time,
                      target_source      = s$source,
                      population_average = TRUE,
                      bayesian_bootstrap = TRUE,
                      seed               = 1L)

    sm <- psum(draws)
    rows[[length(rows) + 1L]] <- data.frame(
      model    = mname,
      estimand = s$estimand,
      source   = s$source,
      time     = if (is.na(s$time)) NA_real_ else s$time,
      mean     = sm[["mean"]],
      lo       = sm[["lo"]],
      hi       = sm[["hi"]],
      stringsAsFactors = FALSE
    )
  }
}
summary_tab <- do.call(rbind, rows)

# Pretty-print
fmt <- function(m, lo, hi) sprintf("%6.3f  [%6.3f, %6.3f]", m, lo, hi)
summary_tab$estimate <- with(summary_tab, fmt(mean, lo, hi))
print_tab <- summary_tab[, c("model", "estimand", "source", "time", "estimate")]
cat("\n========== Population-averaged estimands (mean [95% CI], BB) ==========\n")
print(print_tab, row.names = FALSE)

saveRDS(summary_tab, file.path(out_dir, "estimand_summary.rds"))


# ---- Per-individual plot --------------------------------------------------
# Visualise the heterogeneity of two estimands across a slice of the test set
# under one chosen target (target_source = "os") and the in-window horizon.

slice <- order(dat$true_tau_test)[seq(1, n_test, length.out = 25L)]

af_draws <- lapply(fits, function(f)
  fusion_estimand(f, "AF", target_source = "rct")[, slice, drop = FALSE])
rmst_draws <- lapply(fits, function(f)
  fusion_estimand(f, "RMST", time = t_in, target_source = "os")[, slice, drop = FALSE])

summarise_by_x <- function(M) {
  m  <- colMeans(M)
  lo <- apply(M, 2, quantile, 0.025)
  hi <- apply(M, 2, quantile, 0.975)
  data.frame(mean = m, lo = lo, hi = hi)
}

af_summ   <- lapply(af_draws,   summarise_by_x)
rmst_summ <- lapply(rmst_draws, summarise_by_x)

cols <- c(Gaussian = "steelblue", `Source-HDP` = "darkorange",
          `Three-Gss` = "darkgreen")

plot_pointwise <- function(summ_list, true_curve, ylab, main, file) {
  pdf(file, width = 7, height = 5)
  op <- par(mar = c(4, 4, 2, 1) + 0.1)
  y_all <- unlist(lapply(summ_list, function(d) c(d$lo, d$hi)))
  plot(seq_along(slice), true_curve, type = "n",
       xlim = c(1, length(slice)),
       ylim = range(c(y_all, true_curve), na.rm = TRUE),
       xlab = "test row (sorted by true tau)", ylab = ylab, main = main)
  lines(seq_along(slice), true_curve, lwd = 2, col = "black", lty = 2)
  for (nm in names(summ_list)) {
    d <- summ_list[[nm]]
    arrows(seq_along(slice), d$lo, seq_along(slice), d$hi,
           length = 0.02, angle = 90, code = 3, col = cols[nm])
    points(seq_along(slice), d$mean, pch = 19, col = cols[nm])
  }
  legend("topleft", bty = "n",
         legend = c("truth", names(summ_list)),
         col    = c("black", unname(cols[names(summ_list)])),
         lty    = c(2,        rep(1, length(summ_list))),
         pch    = c(NA,       rep(19, length(summ_list))))
  par(op); dev.off()
}

plot_pointwise(af_summ, exp(dat$true_tau_test[slice]),
               ylab = "AF(x)", main = "Per-x acceleration factor",
               file = file.path(out_dir, "af_pointwise.pdf"))

# Truth for RMST(t_in; x, OS = 0) would need integrating the true survival
# function under the simulated DGP.  We omit the truth line in this plot
# and only compare model fits visually.
plot_pointwise(rmst_summ, rep(NA_real_, length(slice)),
               ylab = sprintf("RMST diff at t* = %.2f", t_in),
               main = "Per-x RMST contrast (target = OS)",
               file = file.path(out_dir, "rmst_pointwise.pdf"))

cat("Pointwise plots saved to ", out_dir, "/{af,rmst}_pointwise.pdf\n", sep = "")
cat("Done.\n")
