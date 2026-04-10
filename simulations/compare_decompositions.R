# =============================================================================
# Simulation: Three-forest vs Four-forest FusionForest
#             + BCF-style RCT-only and RWD-only baselines
#
# DGP
# ---
#   RCT:  y = mu(X) + A * tau(X) + eps
#   RWD:  y = mu(X) + g(X) + gamma(U) + A * (tau(X) + c(U)) + eps
#
#   mu(X)    = 2*X1 - X2 + 0.5*X3^2        shared baseline (observed X)
#   g(X)     = 0.5*X1 - 0.3*X4             RWD-specific deviation (observed X)
#   gamma(U) = U1 - 0.5*U2                 unmeasured prognostic effect
#   tau(X)   = X1 + 0.5*X2^2               heterogeneous treatment effect
#   c(U)     = -0.5*U1 + 0.25*U2           unmeasured confounding bias
#   eps      ~ N(0, sigma^2)
#
#   X1-X5: relevant covariates
#   X6-X8: irrelevant noise covariates
#   U1-U2: unobserved confounders (RWD only, never passed to models)
#
#   Unmeasured confounding: U affects both treatment assignment
#   (P(A=1) = expit(X1 + U1 + U2)) AND outcome (via gamma(U) in baseline
#   and c(U) in treatment effect), making U a proper confounder.
#
# Methods compared
# ----------------
#   1. Three-forest FusionForest (RCT + RWD, single m0)
#   2. Four-forest  FusionForest (RCT + RWD, mu + g decomposition, k_g = 2)
#   3. Four-forest  FusionForest with large k_g (k_g = 10, strong borrowing)
#   4. BCF-style RCT-only (FusionForest on RCT data + minimal OS dummy)
#   5. BCF-style RWD-only (FusionForest on RWD data + minimal RCT dummy)
#
# Metrics: Bias, RMSE, 95% CI coverage on training and test sets
# =============================================================================

library(FusionForests)

set.seed(2024)

# ---- Dimensions -----------------------------------------------------------
n_rct  <- 200
n_rwd  <- 500
n_test <- 300
p_rel  <- 5    # relevant covariates
p_irr  <- 3    # irrelevant covariates
p      <- p_rel + p_irr
q_unobs <- 2   # unobserved confounders

# ---- MCMC settings --------------------------------------------------------
n_trees_main  <- 200   # mu (control) and tau (treat) forests
n_trees_sub   <- 100   # g (deviation) and c (deconf) forests
N_post        <- 2000
N_burn        <- 1000

# ---- True functions -------------------------------------------------------
mu_fn    <- function(X) 2 * X[, 1] - X[, 2] + 0.5 * X[, 3]^2
g_fn     <- function(X) 0.5 * X[, 1] - 0.3 * X[, 4]
gamma_fn <- function(U) U[, 1] - 0.5 * U[, 2]       # unmeasured prognostic
tau_fn   <- function(X) X[, 1] + 0.5 * X[, 2]^2
conf_fn  <- function(U) -0.5 * U[, 1] + 0.25 * U[, 2]  # unmeasured trt bias

sigma_true <- 1.0

# ---- Generate covariates from multivariate normal -------------------------
# Joint distribution: (X1, ..., X8, U1, U2) ~ MVN(0, Sigma)
# Correlation structure: AR(1)-like within X, cross-correlation between X and U
d_total <- p + q_unobs  # 10 total variables

# Build correlation matrix: AR(1) base with rho = 0.3
rho <- 0.3
Sigma <- matrix(0, d_total, d_total)
for (i in 1:d_total) {
  for (j in 1:d_total) {
    Sigma[i, j] <- rho^abs(i - j)
  }
}
# Boost correlation between X1 and U1/U2 (confounding pathway)
Sigma[1, p + 1] <- Sigma[p + 1, 1] <- 0.4   # cor(X1, U1)
Sigma[1, p + 2] <- Sigma[p + 2, 1] <- 0.3   # cor(X1, U2)
# Boost correlation between X4 and U1 (makes g(X) and gamma(U) correlated)
Sigma[4, p + 1] <- Sigma[p + 1, 4] <- 0.35  # cor(X4, U1)

# Cholesky factorisation for sampling
L <- chol(Sigma)

# RCT: draw (X, U) jointly, but U is unused in outcome
Z_rct  <- matrix(rnorm(n_rct  * d_total), n_rct,  d_total) %*% L
X_rct  <- Z_rct[, 1:p]

# RWD: draw (X, U) jointly; U drives confounding
Z_rwd  <- matrix(rnorm(n_rwd  * d_total), n_rwd,  d_total) %*% L
X_rwd  <- Z_rwd[, 1:p]
U_rwd  <- Z_rwd[, (p + 1):d_total]

# Test: draw X only (marginal of X from the joint)
L_x    <- chol(Sigma[1:p, 1:p])
X_test <- matrix(rnorm(n_test * p), n_test, p) %*% L_x

colnames(X_rct)  <- paste0("X", 1:p)
colnames(X_rwd)  <- paste0("X", 1:p)
colnames(X_test) <- paste0("X", 1:p)

# ---- Treatment assignment -------------------------------------------------
A_rct <- rbinom(n_rct, 1, 0.5)
A_rwd <- rbinom(n_rwd, 1, plogis(X_rwd[, 1] + U_rwd[, 1] + U_rwd[, 2]))

# ---- Outcomes -------------------------------------------------------------
y_rct <- mu_fn(X_rct) +
         A_rct * tau_fn(X_rct) +
         rnorm(n_rct, 0, sigma_true)

y_rwd <- mu_fn(X_rwd) + g_fn(X_rwd) + gamma_fn(U_rwd) +
         A_rwd * (tau_fn(X_rwd) + conf_fn(U_rwd)) +
         rnorm(n_rwd, 0, sigma_true)

# ---- Pooled training set --------------------------------------------------
X_train <- rbind(X_rct, X_rwd)
A_train <- c(A_rct, A_rwd)
S_train <- c(rep(1L, n_rct), rep(0L, n_rwd))
y_train <- c(y_rct, y_rwd)
n_train <- n_rct + n_rwd

# ---- True CATE at training and test points --------------------------------
true_cate_train <- tau_fn(X_train)
true_cate_test  <- tau_fn(X_test)

# ---- Test set indicators --------------------------------------------------
A_test <- rep(1L, n_test)
S_test <- rep(1L, n_test)

# ---- Summary --------------------------------------------------------------
cat("=== Data summary ===\n")
cat(sprintf("  RCT:  n = %d  (treated: %d, control: %d)\n",
            n_rct, sum(A_rct), n_rct - sum(A_rct)))
cat(sprintf("  RWD:  n = %d  (treated: %d, control: %d)\n",
            n_rwd, sum(A_rwd), n_rwd - sum(A_rwd)))
cat(sprintf("  Test: n = %d\n", n_test))
cat(sprintf("  Covariates: %d (%d relevant + %d noise)\n",
            p, p_rel, p_irr))
cat(sprintf("  True CATE range: [%.2f, %.2f]\n",
            min(true_cate_train), max(true_cate_train)))
cat(sprintf("  True sigma: %.2f\n\n", sigma_true))


# =============================================================================
# Fit all models
# =============================================================================

# --- Shared FusionForest arguments -----------------------------------------
ff_shared <- list(
  number_of_trees_control = n_trees_main,
  number_of_trees_treat   = n_trees_main,
  number_of_trees_deconf  = n_trees_sub,
  store_posterior_sample   = TRUE,
  N_post                  = N_post,
  N_burn                  = N_burn,
  verbose                 = FALSE
)

# ---- Model 1: Three-forest FusionForest (RCT + RWD) ----------------------
cat("Fitting Model 1: Three-forest FusionForest...\n")
t1 <- system.time({
  fit_3f <- do.call(FusionForest, c(list(
    y                         = y_train,
    X_train_control           = X_train,
    X_train_treat             = X_train,
    treatment_indicator_train = A_train,
    source_indicator_train    = S_train,
    X_test_control            = X_test,
    X_test_treat              = X_test,
    X_test_deconf             = X_test,
    treatment_indicator_test  = A_test,
    source_indicator_test     = S_test,
    decomposition             = "three-forest"
  ), ff_shared))
})
cat(sprintf("  Done in %.1f seconds.\n", t1["elapsed"]))


# ---- Model 2: Four-forest FusionForest, k_g = 2 (default) ----------------
cat("Fitting Model 2: Four-forest FusionForest (k_g = 2)...\n")
t2 <- system.time({
  fit_4f <- do.call(FusionForest, c(list(
    y                         = y_train,
    X_train_control           = X_train,
    X_train_treat             = X_train,
    treatment_indicator_train = A_train,
    source_indicator_train    = S_train,
    X_test_control            = X_test,
    X_test_treat              = X_test,
    X_test_deconf             = X_test,
    treatment_indicator_test  = A_test,
    source_indicator_test     = S_test,
    decomposition             = "four-forest",
    number_of_trees_deviation = n_trees_sub,
    k_g                       = 2
  ), ff_shared))
})
cat(sprintf("  Done in %.1f seconds.\n", t2["elapsed"]))


# ---- Model 3: Four-forest FusionForest, k_g = 10 (strong borrowing) ------
cat("Fitting Model 3: Four-forest FusionForest (k_g = 10)...\n")
t3 <- system.time({
  fit_4f_shrink <- do.call(FusionForest, c(list(
    y                         = y_train,
    X_train_control           = X_train,
    X_train_treat             = X_train,
    treatment_indicator_train = A_train,
    source_indicator_train    = S_train,
    X_test_control            = X_test,
    X_test_treat              = X_test,
    X_test_deconf             = X_test,
    treatment_indicator_test  = A_test,
    source_indicator_test     = S_test,
    decomposition             = "four-forest",
    number_of_trees_deviation = n_trees_sub,
    k_g                       = 10
  ), ff_shared))
})
cat(sprintf("  Done in %.1f seconds.\n", t3["elapsed"]))


# ---- Model 4: RCT-only BCF-style -----------------------------------------
# Use FusionForest on RCT data only.  The function requires >= 1 OS row,
# so we add 2 dummy OS observations at the RCT column means with y at
# the RCT mean.  These carry negligible information.
cat("Fitting Model 4: RCT-only BCF...\n")
n_dummy  <- 2L
X_dummy  <- matrix(colMeans(X_rct), nrow = n_dummy, ncol = p, byrow = TRUE)
A_dummy  <- c(1L, 0L)
y_dummy  <- rep(mean(y_rct), n_dummy)

X_rct_aug <- rbind(X_rct, X_dummy)
A_rct_aug <- c(A_rct, A_dummy)
S_rct_aug <- c(rep(1L, n_rct), rep(0L, n_dummy))
y_rct_aug <- c(y_rct, y_dummy)

t4 <- system.time({
  fit_rct_only <- do.call(FusionForest, c(list(
    y                         = y_rct_aug,
    X_train_control           = X_rct_aug,
    X_train_treat             = X_rct_aug,
    treatment_indicator_train = A_rct_aug,
    source_indicator_train    = S_rct_aug,
    X_test_control            = X_test,
    X_test_treat              = X_test,
    X_test_deconf             = X_test,
    treatment_indicator_test  = A_test,
    source_indicator_test     = S_test,
    decomposition             = "three-forest"
  ), ff_shared))
})
cat(sprintf("  Done in %.1f seconds.\n", t4["elapsed"]))


# ---- Model 5: RWD-only BCF-style -----------------------------------------
# Use FusionForest on RWD data only, adding 2 dummy RCT observations.
cat("Fitting Model 5: RWD-only BCF...\n")
X_dummy_rwd <- matrix(colMeans(X_rwd), nrow = n_dummy, ncol = p, byrow = TRUE)
A_dummy_rwd <- c(1L, 0L)
y_dummy_rwd <- rep(mean(y_rwd), n_dummy)

X_rwd_aug <- rbind(X_dummy_rwd, X_rwd)
A_rwd_aug <- c(A_dummy_rwd, A_rwd)
S_rwd_aug <- c(rep(1L, n_dummy), rep(0L, n_rwd))
y_rwd_aug <- c(y_dummy_rwd, y_rwd)

t5 <- system.time({
  fit_rwd_only <- do.call(FusionForest, c(list(
    y                         = y_rwd_aug,
    X_train_control           = X_rwd_aug,
    X_train_treat             = X_rwd_aug,
    treatment_indicator_train = A_rwd_aug,
    source_indicator_train    = S_rwd_aug,
    X_test_control            = X_test,
    X_test_treat              = X_test,
    X_test_deconf             = X_test,
    treatment_indicator_test  = A_test,
    source_indicator_test     = S_test,
    decomposition             = "three-forest"
  ), ff_shared))
})
cat(sprintf("  Done in %.1f seconds.\n\n", t5["elapsed"]))


# =============================================================================
# Extract CATE estimates and posterior samples
# =============================================================================

# Training-set CATE (posterior mean)
cate_train <- list(
  `3-forest`         = fit_3f$train_predictions_treat,
  `4-forest (k=2)`   = fit_4f$train_predictions_treat,
  `4-forest (k=10)`  = fit_4f_shrink$train_predictions_treat,
  # RCT-only: first n_rct rows are real, last n_dummy are dummies
  `RCT-only`         = fit_rct_only$train_predictions_treat[1:n_rct],
  # RWD-only: first n_dummy are dummies, rest are real
  `RWD-only`         = fit_rwd_only$train_predictions_treat[(n_dummy + 1):(n_dummy + n_rwd)]
)

# Test-set CATE (posterior mean)
cate_test <- list(
  `3-forest`         = fit_3f$test_predictions_treat,
  `4-forest (k=2)`   = fit_4f$test_predictions_treat,
  `4-forest (k=10)`  = fit_4f_shrink$test_predictions_treat,
  `RCT-only`         = fit_rct_only$test_predictions_treat,
  `RWD-only`         = fit_rwd_only$test_predictions_treat
)

# Posterior samples for credible intervals (N_post x n)
samples_train <- list(
  `3-forest`         = fit_3f$train_predictions_sample_treat,
  `4-forest (k=2)`   = fit_4f$train_predictions_sample_treat,
  `4-forest (k=10)`  = fit_4f_shrink$train_predictions_sample_treat,
  `RCT-only`         = fit_rct_only$train_predictions_sample_treat[, 1:n_rct],
  `RWD-only`         = fit_rwd_only$train_predictions_sample_treat[, (n_dummy + 1):(n_dummy + n_rwd)]
)

samples_test <- list(
  `3-forest`         = fit_3f$test_predictions_sample_treat,
  `4-forest (k=2)`   = fit_4f$test_predictions_sample_treat,
  `4-forest (k=10)`  = fit_4f_shrink$test_predictions_sample_treat,
  `RCT-only`         = fit_rct_only$test_predictions_sample_treat,
  `RWD-only`         = fit_rwd_only$test_predictions_sample_treat
)

# True CATE at the subsets used by each model's training set
true_cate_train_list <- list(
  `3-forest`         = true_cate_train,
  `4-forest (k=2)`   = true_cate_train,
  `4-forest (k=10)`  = true_cate_train,
  `RCT-only`         = true_cate_train[1:n_rct],
  `RWD-only`         = true_cate_train[(n_rct + 1):n_train]
)


# =============================================================================
# Compute metrics
# =============================================================================

compute_metrics <- function(pred, truth, posterior_samples) {
  bias     <- mean(pred - truth)
  rmse     <- sqrt(mean((pred - truth)^2))
  ci       <- apply(posterior_samples, 2, quantile,
                    probs = c(0.025, 0.975))
  coverage <- mean(truth >= ci[1, ] & truth <= ci[2, ])
  width    <- mean(ci[2, ] - ci[1, ])
  c(Bias = bias, RMSE = rmse, Coverage = coverage, CI_width = width)
}


# ---- Training-set metrics -------------------------------------------------
cat("=================================================================\n")
cat("CATE performance on TRAINING set\n")
cat("=================================================================\n")

train_results <- sapply(names(cate_train), function(nm) {
  compute_metrics(cate_train[[nm]],
                  true_cate_train_list[[nm]],
                  samples_train[[nm]])
})
print(round(train_results, 4))

# ---- Test-set metrics -----------------------------------------------------
cat("\n=================================================================\n")
cat("CATE performance on TEST set\n")
cat("=================================================================\n")

test_results <- sapply(names(cate_test), function(nm) {
  compute_metrics(cate_test[[nm]],
                  true_cate_test,
                  samples_test[[nm]])
})
print(round(test_results, 4))


# =============================================================================
# Four-forest specific: inspect the deviation forest g(X)
# =============================================================================
cat("\n=================================================================\n")
cat("Four-forest deviation g(X): posterior mean summary (RWD rows)\n")
cat("=================================================================\n")

cat("\nk_g = 2:\n")
cat(sprintf("  Range of g_hat: [%.4f, %.4f]\n",
            min(fit_4f$train_predictions_deviation),
            max(fit_4f$train_predictions_deviation)))
cat(sprintf("  Mean |g_hat|:   %.4f\n",
            mean(abs(fit_4f$train_predictions_deviation))))

cat("\nk_g = 10 (strong borrowing):\n")
cat(sprintf("  Range of g_hat: [%.4f, %.4f]\n",
            min(fit_4f_shrink$train_predictions_deviation),
            max(fit_4f_shrink$train_predictions_deviation)))
cat(sprintf("  Mean |g_hat|:   %.4f\n",
            mean(abs(fit_4f_shrink$train_predictions_deviation))))

# True g at RWD observations
true_g_rwd <- g_fn(X_train[S_train == 0L, , drop = FALSE])
cat(sprintf("\nTrue g range:     [%.4f, %.4f]\n",
            min(true_g_rwd), max(true_g_rwd)))
cat(sprintf("True mean |g|:    %.4f\n", mean(abs(true_g_rwd))))

g_rmse_k2  <- sqrt(mean((fit_4f$train_predictions_deviation - true_g_rwd)^2))
g_rmse_k10 <- sqrt(mean((fit_4f_shrink$train_predictions_deviation - true_g_rwd)^2))
cat(sprintf("\nRMSE(g_hat, g_true):  k_g=2: %.4f  |  k_g=10: %.4f\n",
            g_rmse_k2, g_rmse_k10))


# =============================================================================
# Sigma posterior
# =============================================================================
cat("\n=================================================================\n")
cat("Sigma posterior summary\n")
cat("=================================================================\n")
cat(sprintf("  True sigma:         %.4f\n", sigma_true))
cat(sprintf("  3-forest mean:      %.4f\n", mean(fit_3f$sigma)))
cat(sprintf("  4-forest (k=2):     %.4f\n", mean(fit_4f$sigma)))
cat(sprintf("  4-forest (k=10):    %.4f\n", mean(fit_4f_shrink$sigma)))
cat(sprintf("  RCT-only:           %.4f\n", mean(fit_rct_only$sigma)))
cat(sprintf("  RWD-only:           %.4f\n", mean(fit_rwd_only$sigma)))


# =============================================================================
# Plots
# =============================================================================

pdf("simulations/compare_decompositions.pdf", width = 14, height = 10)
old_par <- par(mfrow = c(2, 3), mar = c(4, 4, 3, 1))

col_3f     <- rgb(0.2, 0.4, 0.8, 0.6)
col_4f     <- rgb(0.8, 0.3, 0.1, 0.6)
col_4f_s   <- rgb(0.1, 0.7, 0.3, 0.6)
col_rct    <- rgb(0.5, 0.5, 0.5, 0.6)
col_rwd    <- rgb(0.7, 0.2, 0.7, 0.6)
col_truth  <- "black"

# --- Panel 1: Estimated vs True CATE (test set) ----------------------------
ylim_all <- range(c(cate_test[["3-forest"]],
                    cate_test[["4-forest (k=2)"]],
                    cate_test[["RCT-only"]]))

plot(true_cate_test, cate_test[["3-forest"]],
     pch = 16, cex = 0.4, col = col_3f,
     xlab = "True CATE", ylab = "Estimated CATE",
     main = "Test: Estimated vs True CATE",
     ylim = ylim_all)
points(true_cate_test, cate_test[["4-forest (k=2)"]],
       pch = 16, cex = 0.4, col = col_4f)
points(true_cate_test, cate_test[["RCT-only"]],
       pch = 16, cex = 0.4, col = col_rct)
abline(0, 1, lwd = 2)
legend("topleft",
       legend = c("3-forest", "4-forest (k=2)", "RCT-only"),
       col = c(col_3f, col_4f, col_rct), pch = 16, bty = "n", cex = 0.8)

# --- Panel 2: CATE bias by X1 (test set) -----------------------------------
bias_3f  <- cate_test[["3-forest"]]        - true_cate_test
bias_4f  <- cate_test[["4-forest (k=2)"]]  - true_cate_test
bias_rct <- cate_test[["RCT-only"]]        - true_cate_test

plot(X_test[, 1], bias_3f,
     pch = 16, cex = 0.3, col = col_3f,
     xlab = "X1 (effect modifier)", ylab = "CATE bias",
     main = "Test: Bias by X1",
     ylim = range(c(bias_3f, bias_4f, bias_rct)))
points(X_test[, 1], bias_4f,  pch = 16, cex = 0.3, col = col_4f)
points(X_test[, 1], bias_rct, pch = 16, cex = 0.3, col = col_rct)
abline(h = 0, lwd = 2)
legend("topleft",
       legend = c("3-forest", "4-forest (k=2)", "RCT-only"),
       col = c(col_3f, col_4f, col_rct), pch = 16, bty = "n", cex = 0.8)

# --- Panel 3: Deviation g(X) recovery (four-forest) ------------------------
plot(true_g_rwd, fit_4f$train_predictions_deviation,
     pch = 16, cex = 0.5, col = col_4f,
     xlab = "True g(X)", ylab = "Estimated g(X)",
     main = "Four-forest: g(X) recovery")
points(true_g_rwd, fit_4f_shrink$train_predictions_deviation,
       pch = 16, cex = 0.5, col = col_4f_s)
abline(0, 1, lwd = 2)
legend("topleft",
       legend = c("k_g = 2", "k_g = 10"),
       col = c(col_4f, col_4f_s), pch = 16, bty = "n")

# --- Panel 4: Sorted CATE with 95% CI (3-forest, test) --------------------
ord <- order(true_cate_test)
ci_3f <- apply(samples_test[["3-forest"]], 2, quantile,
               probs = c(0.025, 0.975))

plot(seq_len(n_test), true_cate_test[ord],
     type = "l", lwd = 2, col = col_truth,
     xlab = "Individual (sorted by true CATE)", ylab = "CATE",
     main = "3-forest: CATE with 95% CI (test)",
     ylim = range(ci_3f))
polygon(c(seq_len(n_test), rev(seq_len(n_test))),
        c(ci_3f[1, ord], rev(ci_3f[2, ord])),
        col = adjustcolor(col_3f, alpha.f = 0.3), border = NA)
lines(seq_len(n_test), cate_test[["3-forest"]][ord],
      col = col_3f, lwd = 2)
legend("topleft", legend = c("Truth", "Estimate", "95% CI"),
       col = c(col_truth, col_3f, col_3f),
       lty = c(1, 1, NA),
       fill = c(NA, NA, adjustcolor(col_3f, 0.3)),
       border = NA, bty = "n", cex = 0.8)

# --- Panel 5: Sorted CATE with 95% CI (4-forest k=2, test) ----------------
ci_4f <- apply(samples_test[["4-forest (k=2)"]], 2, quantile,
               probs = c(0.025, 0.975))

plot(seq_len(n_test), true_cate_test[ord],
     type = "l", lwd = 2, col = col_truth,
     xlab = "Individual (sorted by true CATE)", ylab = "CATE",
     main = "4-forest (k_g=2): CATE with 95% CI (test)",
     ylim = range(ci_4f))
polygon(c(seq_len(n_test), rev(seq_len(n_test))),
        c(ci_4f[1, ord], rev(ci_4f[2, ord])),
        col = adjustcolor(col_4f, alpha.f = 0.3), border = NA)
lines(seq_len(n_test), cate_test[["4-forest (k=2)"]][ord],
      col = col_4f, lwd = 2)
legend("topleft", legend = c("Truth", "Estimate", "95% CI"),
       col = c(col_truth, col_4f, col_4f),
       lty = c(1, 1, NA),
       fill = c(NA, NA, adjustcolor(col_4f, 0.3)),
       border = NA, bty = "n", cex = 0.8)

# --- Panel 6: Sigma traceplots --------------------------------------------
plot(fit_3f$sigma, type = "l", col = col_3f,
     xlab = "Posterior iteration", ylab = expression(sigma),
     main = "Sigma traceplots",
     ylim = range(c(fit_3f$sigma, fit_4f$sigma,
                    fit_rct_only$sigma, fit_rwd_only$sigma)))
lines(fit_4f$sigma,       col = col_4f)
lines(fit_rct_only$sigma, col = col_rct)
lines(fit_rwd_only$sigma, col = col_rwd)
abline(h = sigma_true, lwd = 2, lty = 2)
legend("topright",
       legend = c("3-forest", "4-forest", "RCT-only",
                  "RWD-only", "True"),
       col = c(col_3f, col_4f, col_rct, col_rwd, "black"),
       lty = c(1, 1, 1, 1, 2), lwd = 2, bty = "n", cex = 0.7)

par(old_par)
dev.off()

cat("\nPlots saved to simulations/compare_decompositions.pdf\n")
