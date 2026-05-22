# ============================================================================
# Evaluation functions for the IRS benchmark simulation
# ============================================================================

compute_rmse <- function(f_hat, f_true) {
  sqrt(mean((f_hat - f_true)^2))
}

compute_bias <- function(f_hat, f_true) {
  mean(f_hat - f_true)
}

compute_mae <- function(f_hat, f_true) {
  mean(abs(f_hat - f_true))
}

compute_coverage <- function(posterior_matrix, f_true, level = 0.95) {
  alpha <- 1 - level
  lower <- apply(posterior_matrix, 2, quantile, probs = alpha / 2)
  upper <- apply(posterior_matrix, 2, quantile, probs = 1 - alpha / 2)
  mean(f_true >= lower & f_true <= upper)
}

compute_interval_width <- function(posterior_matrix, level = 0.95) {
  alpha <- 1 - level
  lower <- apply(posterior_matrix, 2, quantile, probs = alpha / 2)
  upper <- apply(posterior_matrix, 2, quantile, probs = 1 - alpha / 2)
  mean(upper - lower)
}

evaluate_fit <- function(fit, f_train, f_test, level = 0.95) {
  metrics <- list(
    rmse_train = compute_rmse(fit$train_predictions, f_train),
    rmse_test  = compute_rmse(fit$test_predictions,  f_test),
    bias_test  = compute_bias(fit$test_predictions,  f_test),
    mae_test   = compute_mae(fit$test_predictions,   f_test)
  )

  if (!is.null(fit$train_predictions_sample)) {
    metrics$coverage <- compute_coverage(fit$train_predictions_sample, f_train, level)
    metrics$width    <- compute_interval_width(fit$train_predictions_sample, level)
  }

  metrics
}
