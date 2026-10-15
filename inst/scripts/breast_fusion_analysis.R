# =============================================================================
# Data-fusion analysis of breast_fusion, written as a real application.
#
# Question: what is the effect of treatment A on relapse-free survival in
# the real-world (RWD) population?  The RCT is unconfounded but small and
# has short follow-up; the RWD are large but their treatment assignment is
# confounded.  FusionForest combines both.
#
# breast_fusion_truth is NOT used here: a real analysis would not have it.
#
# Steps
#   1. Describe the two sources: sizes, covariate shift, follow-up, KM.
#   2. Overlap: RWD propensity score and source-membership score.
#   3. RCT-only analysis with CausalShrinkageForest, transported to the RWD.
#   4. FusionForest on both sources, evaluated on the RWD covariates.
#   5. Estimands: acceleration factor and survival differences.
#   6. Effect modification: CATE plots and posterior linear projection.
#   7. Confounding function in the RWD.
# =============================================================================

suppressPackageStartupMessages({
  library(FusionForests)
  library(survival)
})

# ---- Configuration ---------------------------------------------------------
seed       <- 1L
N_post     <- 2000L
N_burn     <- 2000L
sd_times   <- c(2, 5)      # years at which to report survival differences

data("breast_fusion", package = "FusionForests")
d       <- breast_fusion
x_names <- c("age", "meno", "size", "grade", "nodes", "pgr", "er")
is_rct  <- d$S == 1L
is_os   <- d$S == 0L


# ---- 1. Describe the sources -----------------------------------------------
cat("---- Patients by source and arm ----\n")
print(addmargins(table(source = d$source,
                       arm = ifelse(d$A == 1, "treated", "control"))))

cat("\n---- Covariate means by source ----\n")
print(round(sapply(split(d[, x_names], d$source), colMeans), 2))

cat("\n---- Events and follow-up by source ----\n")
fu <- survfit(Surv(time, 1 - status) ~ source, data = d)  # reverse KM
print(round(cbind(
  events      = tapply(d$status, d$source, sum),
  censored    = tapply(1 - d$status, d$source, mean),
  median_fu   = summary(fu)$table[, "median"]
), 3))

grp <- factor(paste(d$source, ifelse(d$A == 1, "treated", "control")),
              levels = c("RCT control", "RCT treated",
                         "RWD control", "RWD treated"))
km      <- survfit(Surv(time, status) ~ grp, data = d)
km_cols <- c("steelblue", "navy", "tomato", "firebrick")

op <- par(mar = c(4, 4, 2, 1) + 0.1)
plot(km, col = km_cols, lty = c(1, 1, 2, 2), lwd = 2,
     xlab = "Years", ylab = "Relapse-free survival",
     main = "Kaplan-Meier by source and arm")
legend("bottomleft", legend = levels(grp), col = km_cols,
       lty = c(1, 1, 2, 2), lwd = 2, bty = "n")
par(op)


# ---- 2. Overlap --------------------------------------------------------------
# RWD propensity score: who gets treated outside the trial?
form_x <- ~ age + meno + size + grade + log1p(nodes) + log1p(pgr) +
  log1p(er)
ps_fit <- glm(update(form_x, A ~ .), family = binomial, data = d[is_os, ])
ps_os  <- fitted(ps_fit)

cat("\n---- RWD propensity model ----\n")
print(round(summary(ps_fit)$coefficients, 3))

# Source-membership score: how RCT-like is each patient?
mem_fit <- glm(update(form_x, S ~ .), family = binomial, data = d)
p_rct   <- fitted(mem_fit)

op <- par(mfrow = c(1, 2), mar = c(4, 4, 2, 1) + 0.1)
brk <- seq(0, 1, by = 0.025)
hist(ps_os[d$A[is_os] == 0], breaks = brk, freq = FALSE,
     col = adjustcolor("tomato", 0.5), border = NA,
     xlab = "P(treated | x)", main = "RWD propensity score")
hist(ps_os[d$A[is_os] == 1], breaks = brk, freq = FALSE,
     col = adjustcolor("firebrick", 0.5), border = NA, add = TRUE)
legend("topright", c("control", "treated"), bty = "n",
       fill = adjustcolor(c("tomato", "firebrick"), 0.5), border = NA)
hist(p_rct[is_os], breaks = brk, freq = FALSE,
     col = adjustcolor("tomato", 0.5), border = NA,
     xlab = "P(RCT | x)", main = "Source membership")
hist(p_rct[is_rct], breaks = brk, freq = FALSE,
     col = adjustcolor("navy", 0.5), border = NA, add = TRUE)
legend("topright", c("RWD", "RCT"), bty = "n",
       fill = adjustcolor(c("tomato", "navy"), 0.5), border = NA)
par(op)


# ---- 3. RCT-only analysis --------------------------------------------------
# Unconfounded, but uses only 686 patients and transports the CATE to the
# RWD covariates by extrapolation.
X_all <- as.matrix(d[, x_names])
X_rct <- X_all[is_rct, , drop = FALSE]
X_os  <- X_all[is_os,  , drop = FALSE]
n_os  <- nrow(X_os)

set.seed(seed)
fit_rct <- CausalShrinkageForest(
  y                         = log(d$time[is_rct]),
  status                    = d$status[is_rct],
  X_train_control           = X_rct,
  X_train_treat             = X_rct,
  treatment_indicator_train = d$A[is_rct],
  X_test_control            = X_os,
  X_test_treat              = X_os,
  treatment_indicator_test  = rep(1L, n_os),
  outcome_type              = "right-censored",
  timescale                 = "log",
  number_of_trees_control   = 200,
  number_of_trees_treat     = 100,
  prior_type_control        = "standard",
  prior_type_treat          = "standard",
  local_hp_control          = 0.5 / sqrt(200),
  local_hp_treat            = 0.5 / sqrt(100),
  N_post                    = N_post,
  N_burn                    = N_burn,
  store_posterior_sample    = TRUE,
  verbose                   = FALSE
)
cate_rct_os <- fit_rct$test_predictions_treat   # log-time scale, RWD rows


# ---- 4. FusionForest -------------------------------------------------------
# Trained on both sources; the test set is the RWD population, so the
# estimands below target the RWD.
set.seed(seed)
t0  <- proc.time()[["elapsed"]]
fit <- FusionForest(
  y                         = log(d$time),
  status                    = d$status,
  X_train_control           = X_all,
  X_train_treat             = X_all,
  treatment_indicator_train = d$A,
  source_indicator_train    = d$S,
  X_test_control            = X_os,
  X_test_treat              = X_os,
  X_test_deconf             = X_os,
  treatment_indicator_test  = d$A[is_os],
  source_indicator_test     = rep(0L, n_os),
  outcome_type              = "right-censored",
  timescale                 = "log",
  error_dist                = "source_hdp",
  N_post                    = N_post,
  N_burn                    = N_burn,
  store_posterior_sample    = TRUE,
  verbose                   = FALSE
)
cat(sprintf("\nFusionForest fitted in %.1f s\n",
            proc.time()[["elapsed"]] - t0))
print(fit)
summary(fit)


# ---- 5. Estimands in the RWD population -----------------------------------
psum <- function(v) {
  ci <- quantile(v, c(0.025, 0.975), names = FALSE)
  c(mean = mean(v), lo = ci[1], hi = ci[2])
}

af_pop <- fusion_estimand(fit, "AF", population_average = TRUE,
                          seed = seed)
sd_pop <- lapply(sd_times, function(t)
  fusion_estimand(fit, "SD", time = t, target_source = "rwd",
                  population_average = TRUE, seed = seed))

est_tab <- rbind(
  AF = psum(af_pop),
  do.call(rbind, lapply(seq_along(sd_times), function(k)
    psum(sd_pop[[k]])))
)
rownames(est_tab) <- c("Acceleration factor",
                       sprintf("Survival difference, %g y", sd_times))
cat("\n---- Population-averaged estimands, RWD (mean [95% CI]) ----\n")
print(round(est_tab, 3))

# Comparison: average AF from the RCT-only fit, transported to the RWD.
cat(sprintf("\nRCT-only average AF in the RWD: %.3f\n",
            mean(exp(cate_rct_os))))


# ---- 6. Effect modification ------------------------------------------------
af_draws <- fusion_estimand(fit, "AF")            # N_post x n_os
af_mean  <- colMeans(af_draws)
af_lo    <- apply(af_draws, 2, quantile, 0.025)
af_hi    <- apply(af_draws, 2, quantile, 0.975)

# Individual AFs, ordered, with credible intervals.
ord <- order(af_mean)
op  <- par(mar = c(4, 4, 2, 1) + 0.1)
plot(seq_len(n_os), af_mean[ord], type = "n", log = "y",
     ylim = range(af_lo, af_hi),
     xlab = "RWD patients (ordered)", ylab = "Acceleration factor",
     main = "Individual acceleration factors, RWD")
segments(seq_len(n_os), af_lo[ord], seq_len(n_os), af_hi[ord],
         col = adjustcolor("grey50", 0.3))
points(seq_len(n_os), af_mean[ord], pch = 19, cex = 0.3)
abline(h = 1, lty = 2)
par(op)

# AF against the two strongest-looking modifiers.
op <- par(mfrow = c(1, 2), mar = c(4, 4, 2, 1) + 0.1)
grade_col <- ifelse(d$grade[is_os] == 1, "firebrick", "steelblue")
plot(log1p(d$pgr[is_os]), af_mean, pch = 19, cex = 0.4, col = grade_col,
     log = "y", xlab = "log(1 + pgr)", ylab = "Posterior mean AF",
     main = "AF by pgr")
abline(h = 1, lty = 2)
legend("topleft", c("grade 1-2", "grade 3"), pch = 19, bty = "n",
       col = c("steelblue", "firebrick"))
boxplot(af_mean ~ d$grade[is_os], log = "y",
        names = c("grade 1-2", "grade 3"),
        xlab = "", ylab = "Posterior mean AF", main = "AF by grade")
abline(h = 1, lty = 2)
par(op)

# Posterior linear projection of the CATE onto interpretable terms.
X_os_df <- as.data.frame(X_os)
proj <- fusion_projection(
  fit,
  basis   = ~ age + meno + size + grade + log1p(nodes) + log1p(pgr) +
    log1p(er),
  X_eval  = X_os_df,
  target  = "cate",
  which   = "test",
  weights = "bayesian_bootstrap",
  seed    = seed
)
cat("\n---- Posterior projection of the CATE (log-time scale) ----\n")
print(round(t(apply(proj, 2, psum)), 3))


# ---- 7. Confounding in the RWD ---------------------------------------------
# c(x): the bias of the naive RWD treatment contrast on the log-time scale.
c_os <- fit$test_predictions_deconf
cat("\n---- Estimated confounding function c(x), RWD ----\n")
print(round(summary(c_os), 3))

op <- par(mar = c(4, 4, 2, 1) + 0.1)
hist(c_os, breaks = 40, col = "grey80", border = "white",
     xlab = "Posterior mean c(x)", main = "Confounding function, RWD")
abline(v = 0, lty = 2)
par(op)

cat("Done.\n")
