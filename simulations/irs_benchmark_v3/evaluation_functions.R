# ============================================================================
# Evaluation functions for IRS benchmark v2
#
# Metrics for prediction accuracy, CATE estimation, uncertainty
# quantification, and variance decomposition.
# ============================================================================

# --- Prediction accuracy ----------------------------------------------------

compute_rmse <- function(f_hat, f_true) {
  sqrt(mean((f_hat - f_true)^2))
}

compute_mae <- function(f_hat, f_true) {
  mean(abs(f_hat - f_true))
}

# --- CATE metrics -----------------------------------------------------------

compute_cate_bias <- function(tau_hat, tau_true) {
  mean(tau_hat - tau_true)
}

compute_cate_rmse <- function(tau_hat, tau_true) {
  sqrt(mean((tau_hat - tau_true)^2))
}

# --- ATE metrics (point estimate) ------------------------------------------

compute_ate_bias <- function(tau_hat, tau_true) {
  mean(tau_hat) - mean(tau_true)
}

# --- Uncertainty quantification ---------------------------------------------

#' Compute pointwise coverage of credible intervals
#' @param tau_samples N_post x n matrix of posterior CATE draws
#' @param tau_true    length-n vector of true CATEs
compute_coverage <- function(tau_samples, tau_true,
                             level = 0.95) {
  alpha <- 1 - level
  lower <- apply(tau_samples, 2, quantile,
                 probs = alpha / 2)
  upper <- apply(tau_samples, 2, quantile,
                 probs = 1 - alpha / 2)
  mean(tau_true >= lower & tau_true <= upper)
}

#' Compute average width of credible intervals
compute_interval_width <- function(tau_samples,
                                   level = 0.95) {
  alpha <- 1 - level
  lower <- apply(tau_samples, 2, quantile,
                 probs = alpha / 2)
  upper <- apply(tau_samples, 2, quantile,
                 probs = 1 - alpha / 2)
  mean(upper - lower)
}

# --- Variance decomposition -------------------------------------------------

#' Average posterior variance of tau(x_i) across observations
compute_avg_posterior_var <- function(tau_samples) {
  mean(apply(tau_samples, 2, var))
}

# --- Composite evaluation ---------------------------------------------------

#' Evaluate a fitted model on prediction and CATE metrics
#'
#' @param fit           SimpleBART fit object
#' @param fit_1         SimpleBART fit at A=1 (for CATE)
#' @param fit_0         SimpleBART fit at A=0 (for CATE)
#' @param m_true_train  True m(X,A) on training set
#' @param m_true_test   True m(X,A) on test set
#' @param tau_true_train True CATE on training set
#' @param tau_true_test  True CATE on test set
#' @param has_posterior  Whether posterior samples are available
evaluate_fit <- function(fit, fit_1, fit_0,
                         m_true_train, m_true_test,
                         tau_true_train, tau_true_test,
                         has_posterior = TRUE) {

  # Point predictions for m
  m_hat_train <- fit$train_predictions
  m_hat_test  <- fit$test_predictions

  # CATE: tau_hat = m_hat(x,1) - m_hat(x,0)
  tau_hat_test  <- fit_1$test_predictions -
                   fit_0$test_predictions
  tau_hat_train <- fit_1$train_predictions -
                   fit_0$train_predictions

  metrics <- list(
    # Prediction
    rmse_m_train = compute_rmse(m_hat_train, m_true_train),
    rmse_m_test  = compute_rmse(m_hat_test, m_true_test),
    mae_m_test   = compute_mae(m_hat_test, m_true_test),
    # CATE
    cate_bias_test  = compute_cate_bias(tau_hat_test,
                                        tau_true_test),
    cate_rmse_test  = compute_cate_rmse(tau_hat_test,
                                        tau_true_test),
    cate_bias_train = compute_cate_bias(tau_hat_train,
                                        tau_true_train),
    cate_rmse_train = compute_cate_rmse(tau_hat_train,
                                        tau_true_train),
    # ATE
    ate_bias = compute_ate_bias(tau_hat_test, tau_true_test),
    ate_hat  = mean(tau_hat_test),
    ate_true = mean(tau_true_test)
  )

  # Uncertainty quantification (if posterior samples available)
  if (has_posterior &&
      !is.null(fit_1$test_predictions_sample) &&
      !is.null(fit_0$test_predictions_sample)) {
    tau_samples_test <- fit_1$test_predictions_sample -
                        fit_0$test_predictions_sample
    metrics$cate_coverage <- compute_coverage(
      tau_samples_test, tau_true_test
    )
    metrics$cate_ci_width <- compute_interval_width(
      tau_samples_test
    )
    metrics$avg_post_var <- compute_avg_posterior_var(
      tau_samples_test
    )
    # ATE posterior
    ate_samples <- rowMeans(tau_samples_test)
    metrics$ate_coverage <- as.numeric(
      quantile(ate_samples, 0.025) <= mean(tau_true_test) &
      mean(tau_true_test) <= quantile(ate_samples, 0.975)
    )
    metrics$ate_ci_width <- as.numeric(
      quantile(ate_samples, 0.975) -
      quantile(ate_samples, 0.025)
    )
  }

  metrics
}
