# ============================================================================
# Evaluation functions for the IRS simulation study
#
# Reports: train RMSE, test RMSE, and coverage/width on training set.
# ============================================================================

# --- Point estimation metrics ------------------------------------------------

compute_rmse <- function(f_hat, f_true) {
  sqrt(mean((f_hat - f_true)^2))
}

compute_bias <- function(f_hat, f_true) {
  mean(f_hat - f_true)
}

compute_mae <- function(f_hat, f_true) {
  mean(abs(f_hat - f_true))
}

# --- Uncertainty quantification metrics --------------------------------------

# Coverage: proportion of points where the credible interval contains f_true.
# posterior_matrix: N_post x n matrix of posterior draws.
compute_coverage <- function(posterior_matrix, f_true, level = 0.95) {
  alpha <- 1 - level
  lower <- apply(posterior_matrix, 2, quantile, probs = alpha / 2)
  upper <- apply(posterior_matrix, 2, quantile, probs = 1 - alpha / 2)
  mean(f_true >= lower & f_true <= upper)
}

# Average width of credible intervals.
compute_interval_width <- function(posterior_matrix, level = 0.95) {
  alpha <- 1 - level
  lower <- apply(posterior_matrix, 2, quantile, probs = alpha / 2)
  upper <- apply(posterior_matrix, 2, quantile, probs = 1 - alpha / 2)
  mean(upper - lower)
}

# --- Aggregator --------------------------------------------------------------

# Evaluate a SimpleBART fit.
#   f_train, f_test: true function values on train/test sets.
#   Coverage and interval width are computed on the TRAINING set
#   (requires fit$train_predictions_sample).
evaluate_fit <- function(fit, f_train, f_test, level = 0.95) {
  metrics <- list(
    rmse_train = compute_rmse(fit$train_predictions, f_train),
    rmse_test  = compute_rmse(fit$test_predictions,  f_test)
  )

  if (!is.null(fit$train_predictions_sample)) {
    metrics$coverage <- compute_coverage(fit$train_predictions_sample, f_train, level)
    metrics$width    <- compute_interval_width(fit$train_predictions_sample, level)
  }

  metrics
}
