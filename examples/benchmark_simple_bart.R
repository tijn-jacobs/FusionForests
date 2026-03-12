# ============================================================================
# Benchmark: SimpleBART — complete vs missing data with IRS
#
# DGP: Step function (piecewise constant — ideal for trees)
#   Y = 3 * I(X1 > 0.5) - 2 * I(X2 > 0.5) + eps
#   X3 is pure noise
#
# Compares:
#   - Complete data (baseline, irs = 0)
#   - IRS mode 1: skip-then-draw (NaN excluded from MH ratio)
#   - IRS mode 2: draw-then-decide (NaN routed before MH, included in ratio)
#
# Run after: R CMD INSTALL .
# ============================================================================

library(FusionForests)

set.seed(42)

# ----------------------------------------------------------------------------
# Data generation
# ----------------------------------------------------------------------------

n_train <- 250
n_test  <- 250
p       <- 3

N_post <- 2000
N_burn <- 1000

sigma_true <- 0.3

X_train <- matrix(runif(n_train * p), ncol = p)
X_test  <- matrix(runif(n_test  * p), ncol = p)

f_true <- function(X) 3 * (X[, 1] > 0.5) - 2 * (X[, 2] > 0.5)

y_train <- f_true(X_train) + rnorm(n_train, sd = sigma_true)
f_train <- f_true(X_train)
f_test  <- f_true(X_test)

# ----------------------------------------------------------------------------
# 1. Complete data (baseline)
# ----------------------------------------------------------------------------

cat("=== Complete data (no missingness) ===\n\n")

fit_complete <- SimpleBART(
  y       = y_train,
  X_train = X_train,
  X_test  = X_test,
  N_post  = N_post,
  N_burn  = N_burn,
  verbose = TRUE,
  irs     = 0L
)

rmse_train_complete <- sqrt(mean((fit_complete$train_predictions - f_train)^2))
rmse_test_complete  <- sqrt(mean((fit_complete$test_predictions  - f_test)^2))
cat("Train RMSE (vs true f):", round(rmse_train_complete, 3), "\n")
cat("Test  RMSE (vs true f):", round(rmse_test_complete, 3), "\n")
cat("Posterior mean sigma:  ", round(mean(fit_complete$sigma), 3),
    " (true:", sigma_true, ")\n")
cat("Acceptance ratio:      ", round(fit_complete$acceptance_ratio, 3), "\n\n")

# ----------------------------------------------------------------------------
# 2. Missing data: compare IRS mode 1 vs mode 2
# ----------------------------------------------------------------------------

miss_rates <- c(0.10, 0.20, 0.30, 0.50)
irs_modes  <- c(1L, 2L)
mode_names <- c("skip-then-draw", "draw-then-decide")

# Results: one row per (mode, missingness) combination + baseline
n_rows  <- 1 + length(miss_rates) * length(irs_modes)
results <- data.frame(
  mode        = character(n_rows),
  missingness = numeric(n_rows),
  rmse_train  = numeric(n_rows),
  rmse_test   = numeric(n_rows),
  sigma_hat   = numeric(n_rows),
  accept      = numeric(n_rows),
  stringsAsFactors = FALSE
)

results[1, ] <- list("Complete", 0,
                      rmse_train_complete, rmse_test_complete,
                      mean(fit_complete$sigma), fit_complete$acceptance_ratio)

row <- 2
for (rate in miss_rates) {

  # Create the same missingness pattern for both modes
  miss_idx      <- sample(n_train, size = floor(rate * n_train))
  miss_idx_test <- sample(n_test,  size = floor(rate * n_test))

  X_train_miss <- X_train
  X_train_miss[miss_idx, 2] <- NaN

  X_test_miss <- X_test
  X_test_miss[miss_idx_test, 2] <- NaN

  for (m in seq_along(irs_modes)) {
    mode <- irs_modes[m]
    cat(sprintf("=== %s | %.0f%% missingness in X2 ===\n\n",
                mode_names[m], rate * 100))

    fit <- SimpleBART(
      y       = y_train,
      X_train = X_train_miss,
      X_test  = X_test_miss,
      N_post  = N_post,
      N_burn  = N_burn,
      verbose = TRUE,
      irs     = mode
    )

    rmse_tr <- sqrt(mean((fit$train_predictions - f_train)^2))
    rmse_te <- sqrt(mean((fit$test_predictions  - f_test)^2))
    cat("Train RMSE (vs true f):", round(rmse_tr, 3), "\n")
    cat("Test  RMSE (vs true f):", round(rmse_te, 3), "\n")
    cat("Posterior mean sigma:  ", round(mean(fit$sigma), 3),
        " (true:", sigma_true, ")\n")
    cat("Acceptance ratio:      ", round(fit$acceptance_ratio, 3), "\n\n")

    results[row, ] <- list(mode_names[m], rate, rmse_tr, rmse_te,
                           mean(fit$sigma), fit$acceptance_ratio)
    row <- row + 1
  }
}

# ----------------------------------------------------------------------------
# Summary table
# ----------------------------------------------------------------------------

cat("=== Summary ===\n\n")
cat(sprintf("%-18s  %6s  %11s  %11s  %10s  %10s\n",
            "Mode", "Miss%", "Train RMSE", "Test RMSE", "Sigma hat", "Accept"))
cat(paste(rep("-", 72), collapse = ""), "\n")
for (j in seq_len(nrow(results))) {
  miss_label <- if (results$missingness[j] == 0) "  -" else
    sprintf("%4.0f%%", results$missingness[j] * 100)
  cat(sprintf("%-18s  %6s  %11.3f  %11.3f  %10.3f  %10.3f\n",
              results$mode[j], miss_label,
              results$rmse_train[j], results$rmse_test[j],
              results$sigma_hat[j], results$accept[j]))
}
