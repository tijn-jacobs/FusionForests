## Example: Comparing BART (RCT only) vs FusionForest (RCT + RWD)
## for estimating Heterogeneous Treatment Effects (HTE)
##
## Simulation design:
##   - True CATE varies by covariate X1 (effect modifier)
##   - RCT: small, randomised, no unmeasured confounding
##   - RWD: large, observational, with confounding on X2
##
## We compare:
##   1. CausalShrinkageForest fitted on RCT data only  (from ShrinkageTrees)
##   2. FusionForest combining RCT + RWD               (from FusionForests)

library(FusionForests)
library(ShrinkageTrees)

set.seed(42)

# -------------------------------------------------------------------------
# 1. DATA GENERATING PROCESS
# -------------------------------------------------------------------------

n_rct <- 200   # RCT sample size (small, as is typical)
n_rwd <- 800   # RWD sample size (larger observational dataset)
p     <- 10    # number of covariates

# --- True functions -------------------------------------------------------
mu0       <- function(X) 2 * X[, 1] - X[, 3]^2
true_cate <- function(X) 1.5 + 2 * X[, 1]

sigma_true <- 1.0

# --- Generate RCT ---------------------------------------------------------
X_rct <- matrix(rnorm(n_rct * p), nrow = n_rct, ncol = p)
colnames(X_rct) <- paste0("X", seq_len(p))

trt_rct <- rbinom(n_rct, 1, 0.5)   # balanced randomisation
y_rct   <- mu0(X_rct) + true_cate(X_rct) * trt_rct + rnorm(n_rct, sd = sigma_true)

# --- Generate RWD ---------------------------------------------------------
X_rwd <- matrix(rnorm(n_rwd * p), nrow = n_rwd, ncol = p)
colnames(X_rwd) <- paste0("X", seq_len(p))

prop_score_rwd <- plogis(0.8 * X_rwd[, 2])   # confounded on X2
trt_rwd <- rbinom(n_rwd, 1, prop_score_rwd)
y_rwd   <- mu0(X_rwd) + true_cate(X_rwd) * trt_rwd + rnorm(n_rwd, sd = sigma_true)

# --- Pool RCT + RWD -------------------------------------------------------
X_all      <- rbind(X_rct, X_rwd)
y_all      <- c(y_rct, y_rwd)
trt_all    <- c(trt_rct, trt_rwd)
source_all <- c(rep(1L, n_rct), rep(0L, n_rwd))   # 1 = RCT, 0 = RWD

true_cate_all <- true_cate(X_all)
true_cate_rct <- true_cate(X_rct)

cat("--------------------------------------------------------------\n")
cat("Data summary\n")
cat("  RCT:  n =", n_rct, " | treated:", sum(trt_rct),
    "| control:", sum(trt_rct == 0), "\n")
cat("  RWD:  n =", n_rwd, " | treated:", sum(trt_rwd),
    "| control:", sum(trt_rwd == 0), "\n")
cat("  True CATE range: [", round(min(true_cate_all), 2),
    ",", round(max(true_cate_all), 2), "]\n")
cat("--------------------------------------------------------------\n\n")

# -------------------------------------------------------------------------
# 2. HYPERPARAMETER SETUP
# -------------------------------------------------------------------------
n_trees  <- 50    # use 200+ in practice
N_post   <- 2500
N_burn   <- 1500

# CausalShrinkageForest horseshoe scale
hp_local  <- 0.5 / sqrt(n_trees)
hp_global <- 0.5 / sqrt(n_trees)

# -------------------------------------------------------------------------
# 3. MODEL 1: CausalShrinkageForest on RCT data only (horseshoe BART)
# -------------------------------------------------------------------------
cat("Fitting Model 1: CausalShrinkageForest (RCT only)...\n")

fit_rct <- CausalShrinkageForest(
  y                         = y_rct,
  X_train_control           = X_rct,
  X_train_treat             = X_rct,
  treatment_indicator_train = trt_rct,
  X_test_control            = X_all,
  X_test_treat              = X_all,
  treatment_indicator_test  = trt_all,
  outcome_type              = "continuous",
  number_of_trees_control   = n_trees,
  number_of_trees_treat     = n_trees,
  prior_type_control        = "horseshoe",
  prior_type_treat          = "horseshoe",
  local_hp_control          = hp_local,
  local_hp_treat            = hp_local,
  global_hp_control         = hp_global,
  global_hp_treat           = hp_global,
  store_posterior_sample    = TRUE,
  N_post                    = N_post,
  N_burn                    = N_burn,
  verbose                   = FALSE
)

cat("  Done.\n\n")

# -------------------------------------------------------------------------
# 4. MODEL 2: FusionForest on RCT + RWD
#
# API changes from FusionShrinkageForest:
#   - Function renamed to FusionForest
#   - prior_type_*/local_hp_*/global_hp_* removed; standard BART leaf prior
#     (omega = 0.5/sqrt(trees)) is used internally for all three forests
#   - eta_commensurate removed; eta is fixed at 0 (no inter-source mean shift)
# -------------------------------------------------------------------------
cat("Fitting Model 2: FusionForest (RCT + RWD)...\n")

fit_fusion <- FusionForest(
  y                         = y_all,
  X_train_control           = X_all,
  X_train_treat             = X_all,
  treatment_indicator_train = trt_all,
  source_indicator_train    = source_all,
  X_test_control            = X_all,
  X_test_treat              = X_all,
  treatment_indicator_test  = trt_all,
  source_indicator_test     = source_all,
  outcome_type              = "continuous",
  number_of_trees_control   = n_trees,
  number_of_trees_treat     = n_trees,
  number_of_trees_deconf    = n_trees,
  store_posterior_sample    = TRUE,
  N_post                    = N_post,
  N_burn                    = N_burn,
  verbose                   = FALSE
)

cat("  Done.\n\n")

# -------------------------------------------------------------------------
# 5. EXTRACT CATE ESTIMATES
# -------------------------------------------------------------------------
cate_rct_only <- fit_rct$test_predictions_treat
cate_fusion   <- fit_fusion$test_predictions_treat

cate_rct_samples    <- fit_rct$test_predictions_sample_treat      # N_post x n_all
cate_fusion_samples <- fit_fusion$test_predictions_sample_treat   # N_post x n_all

# -------------------------------------------------------------------------
# 6. EVALUATE PERFORMANCE
# -------------------------------------------------------------------------
rmse <- function(pred, truth) sqrt(mean((pred - truth)^2))
mae  <- function(pred, truth) mean(abs(pred - truth))

rmse_rct    <- rmse(cate_rct_only, true_cate_all)
rmse_fusion <- rmse(cate_fusion,   true_cate_all)
mae_rct     <- mae(cate_rct_only,  true_cate_all)
mae_fusion  <- mae(cate_fusion,    true_cate_all)

ci_rct    <- apply(cate_rct_samples,    2, quantile, probs = c(0.025, 0.975))
ci_fusion <- apply(cate_fusion_samples, 2, quantile, probs = c(0.025, 0.975))

coverage_rct    <- mean(true_cate_all >= ci_rct[1, ]    & true_cate_all <= ci_rct[2, ])
coverage_fusion <- mean(true_cate_all >= ci_fusion[1, ] & true_cate_all <= ci_fusion[2, ])

width_rct    <- mean(ci_rct[2, ]    - ci_rct[1, ])
width_fusion <- mean(ci_fusion[2, ] - ci_fusion[1, ])

cat("==============================================================\n")
cat("HTE Estimation Performance (evaluated on all n =", nrow(X_all), "obs)\n")
cat("==============================================================\n")
cat(sprintf("%-35s %10s %10s\n", "Metric", "RCT only", "Fusion"))
cat(sprintf("%-35s %10.4f %10.4f\n", "RMSE (CATE)",        rmse_rct,     rmse_fusion))
cat(sprintf("%-35s %10.4f %10.4f\n", "MAE  (CATE)",        mae_rct,      mae_fusion))
cat(sprintf("%-35s %10.4f %10.4f\n", "95%% CI coverage",   coverage_rct, coverage_fusion))
cat(sprintf("%-35s %10.4f %10.4f\n", "Mean 95%% CI width", width_rct,    width_fusion))
cat("==============================================================\n\n")

# -------------------------------------------------------------------------
# 7. PLOTS
# -------------------------------------------------------------------------
old_par <- par(mfrow = c(2, 3), mar = c(4, 4, 3, 1))
on.exit(par(old_par), add = TRUE)

col_rct    <- rgb(0.2, 0.4, 0.8, 0.6)
col_fusion <- rgb(0.8, 0.3, 0.1, 0.6)
col_truth  <- "black"
n_all      <- nrow(X_all)

# --- Panel 1: Estimated vs True CATE (RCT only) --------------------------
plot(true_cate_all, cate_rct_only,
     pch = 16, cex = 0.5, col = col_rct,
     xlab = "True CATE", ylab = "Estimated CATE",
     main = "RCT only: Estimated vs True CATE")
abline(0, 1, col = col_truth, lwd = 2)

# --- Panel 2: Estimated vs True CATE (Fusion) ----------------------------
plot(true_cate_all, cate_fusion,
     pch = 16, cex = 0.5, col = col_fusion,
     xlab = "True CATE", ylab = "Estimated CATE",
     main = "Fusion: Estimated vs True CATE")
abline(0, 1, col = col_truth, lwd = 2)

# --- Panel 3: CATE bias by X1 (the true effect modifier) -----------------
x1_vals <- X_all[, 1]
plot(x1_vals, cate_rct_only - true_cate_all,
     pch = 16, cex = 0.4, col = col_rct,
     xlab = "X1 (effect modifier)", ylab = "CATE bias (estimate - truth)",
     main = "Bias by effect modifier X1",
     ylim = range(c(cate_rct_only - true_cate_all,
                    cate_fusion   - true_cate_all)))
points(x1_vals, cate_fusion - true_cate_all,
       pch = 16, cex = 0.4, col = col_fusion)
abline(h = 0, lwd = 2)
legend("topleft", legend = c("RCT only", "Fusion"),
       col = c(col_rct, col_fusion), pch = 16, bty = "n")

# --- Panel 4: Sorted CATE with 95% CI (RCT only) -------------------------
ord <- order(true_cate_all)

plot(seq_len(n_all), true_cate_all[ord],
     type = "l", lwd = 2, col = col_truth,
     xlab = "Individual (sorted by true CATE)", ylab = "CATE",
     main = "RCT only: CATE estimates with 95% CI",
     ylim = range(ci_rct))
polygon(c(seq_len(n_all), rev(seq_len(n_all))),
        c(ci_rct[1, ord], rev(ci_rct[2, ord])),
        col = adjustcolor(col_rct, alpha.f = 0.3), border = NA)
lines(seq_len(n_all), cate_rct_only[ord], col = col_rct, lwd = 2)
legend("topleft", legend = c("Truth", "Estimate", "95% CI"),
       col = c(col_truth, col_rct, col_rct),
       lty = c(1, 1, NA), fill = c(NA, NA, adjustcolor(col_rct, 0.3)),
       border = NA, bty = "n")

# --- Panel 5: Sorted CATE with 95% CI (Fusion) ---------------------------
plot(seq_len(n_all), true_cate_all[ord],
     type = "l", lwd = 2, col = col_truth,
     xlab = "Individual (sorted by true CATE)", ylab = "CATE",
     main = "Fusion: CATE estimates with 95% CI",
     ylim = range(ci_fusion))
polygon(c(seq_len(n_all), rev(seq_len(n_all))),
        c(ci_fusion[1, ord], rev(ci_fusion[2, ord])),
        col = adjustcolor(col_fusion, alpha.f = 0.3), border = NA)
lines(seq_len(n_all), cate_fusion[ord], col = col_fusion, lwd = 2)
legend("topleft", legend = c("Truth", "Estimate", "95% CI"),
       col = c(col_truth, col_fusion, col_fusion),
       lty = c(1, 1, NA), fill = c(NA, NA, adjustcolor(col_fusion, 0.3)),
       border = NA, bty = "n")

# --- Panel 6: Sigma traceplots -------------------------------------------
plot(fit_rct$sigma, type = "l", col = col_rct,
     xlab = "Posterior iteration", ylab = expression(sigma),
     main = "Sigma traceplots",
     ylim = range(c(fit_rct$sigma, fit_fusion$sigma)))
lines(fit_fusion$sigma, col = col_fusion)
abline(h = sigma_true, lwd = 2, lty = 2)
legend("topright",
       legend = c("RCT only", "Fusion", "True sigma"),
       col    = c(col_rct, col_fusion, "black"),
       lty    = c(1, 1, 2), lwd = 2, bty = "n")
