# ============================================================================
# Method wrappers for the IRS benchmark simulation
#
# Each function takes (y, X_train, X_test, bart_settings) and returns a
# list with: train_predictions, test_predictions, train_predictions_sample.
# ============================================================================

source("simulations/irs_benchmark/evaluation.R")

# --- Shared BART args -------------------------------------------------------

bart_args <- function(s) {
  list(
    number_of_trees       = s$number_of_trees,
    N_post                = s$N_post,
    N_burn                = s$N_burn,
    verbose               = FALSE,
    store_posterior_sample = TRUE
  )
}

fit_simple_bart <- function(y, X_train, X_test, irs, s) {
  do.call(SimpleBART, c(
    list(y = y, X_train = X_train, X_test = X_test, irs = as.integer(irs)),
    bart_args(s)
  ))
}

# --- Method: Oracle ----------------------------------------------------------

run_oracle <- function(data, miss, scenario, s) {
  if (scenario$pattern == "predictive") {
    X_oracle <- cbind(data$X_train,
                      miss$M[, scenario$miss_cols, drop = FALSE])
    X_oracle_test <- cbind(data$X_test,
      matrix(0L, nrow = nrow(data$X_test),
             ncol = length(scenario$miss_cols)))
    f_train <- miss$f_true_predictive
    fit <- fit_simple_bart(miss$y, X_oracle, X_oracle_test, irs = 0L, s)
  } else {
    f_train <- data$f_train
    fit <- fit_simple_bart(data$y_train, data$X_train, data$X_test,
                           irs = 0L, s)
  }
  evaluate_fit(fit, f_train, data$f_test)
}

# --- Method: IRS (informed) -------------------------------------------------

run_irs_informed <- function(data, miss, scenario, s) {
  y_fit   <- if (scenario$pattern == "predictive") miss$y else data$y_train
  f_train <- if (scenario$pattern == "predictive") miss$f_true_predictive else data$f_train
  fit <- fit_simple_bart(y_fit, miss$X_miss, data$X_test, irs = 2L, s)
  evaluate_fit(fit, f_train, data$f_test)
}

# --- Method: IRS (uniform) --------------------------------------------------

run_irs_uniform <- function(data, miss, scenario, s) {
  y_fit   <- if (scenario$pattern == "predictive") miss$y else data$y_train
  f_train <- if (scenario$pattern == "predictive") miss$f_true_predictive else data$f_train
  fit <- fit_simple_bart(y_fit, miss$X_miss, data$X_test, irs = 3L, s)
  evaluate_fit(fit, f_train, data$f_test)
}

# --- Method: Complete case ---------------------------------------------------

run_complete_case <- function(data, miss, scenario, s) {
  y_fit   <- if (scenario$pattern == "predictive") miss$y else data$y_train
  f_train <- if (scenario$pattern == "predictive") miss$f_true_predictive else data$f_train

  obs_rows <- apply(miss$X_miss, 1, function(r) !any(is.nan(r)))
  n_obs <- sum(obs_rows)

  if (n_obs < 30) {
    return(list(rmse_train = NA, rmse_test = NA, bias_test = NA,
                mae_test = NA, coverage = NA, width = NA))
  }

  X_cc       <- miss$X_miss[obs_rows, , drop = FALSE]
  y_cc       <- y_fit[obs_rows]
  f_train_cc <- f_train[obs_rows]

  fit <- fit_simple_bart(y_cc, X_cc, data$X_test, irs = 0L, s)

  ev <- list(
    rmse_train = compute_rmse(fit$train_predictions, f_train_cc),
    rmse_test  = compute_rmse(fit$test_predictions, data$f_test),
    bias_test  = compute_bias(fit$test_predictions, data$f_test),
    mae_test   = compute_mae(fit$test_predictions, data$f_test)
  )
  if (!is.null(fit$train_predictions_sample)) {
    ev$coverage <- compute_coverage(fit$train_predictions_sample, f_train_cc)
    ev$width    <- compute_interval_width(fit$train_predictions_sample)
  }
  ev
}

# --- Method: bartMachine MIA ------------------------------------------------

run_bartmachine <- function(data, miss, scenario, s) {
  y_fit   <- if (scenario$pattern == "predictive") miss$y else data$y_train
  f_train <- if (scenario$pattern == "predictive") miss$f_true_predictive else data$f_train

  X_train_na <- miss$X_miss
  X_train_na[is.nan(X_train_na)] <- NA

  bm <- bartMachine::bartMachine(
    as.data.frame(X_train_na), y_fit,
    num_trees                          = s$number_of_trees,
    num_burn_in                        = s$N_burn,
    num_iterations_after_burn_in       = s$N_post,
    use_missing_data                   = TRUE,
    use_missing_data_dummies_as_covars = TRUE,
    verbose                            = FALSE
  )

  pred_train <- bartMachine::bart_machine_get_posterior(bm, as.data.frame(X_train_na))
  pred_test  <- bartMachine::bart_machine_get_posterior(bm, as.data.frame(data$X_test))

  ev <- list(
    rmse_train = compute_rmse(pred_train$y_hat, f_train),
    rmse_test  = compute_rmse(pred_test$y_hat, data$f_test),
    bias_test  = compute_bias(pred_test$y_hat, data$f_test),
    mae_test   = compute_mae(pred_test$y_hat, data$f_test),
    coverage   = compute_coverage(t(pred_test$y_hat_posterior_samples), data$f_test),
    width      = compute_interval_width(t(pred_test$y_hat_posterior_samples))
  )
  rm(bm); gc()
  ev
}

# --- Method: missForest + BART -----------------------------------------------

run_missforest <- function(data, miss, scenario, s) {
  y_fit   <- if (scenario$pattern == "predictive") miss$y else data$y_train
  f_train <- if (scenario$pattern == "predictive") miss$f_true_predictive else data$f_train

  X_train_na <- miss$X_miss
  X_train_na[is.nan(X_train_na)] <- NA

  imputed   <- missForest::missForest(as.data.frame(X_train_na), verbose = FALSE)
  X_imputed <- as.matrix(imputed$ximp)

  fit <- fit_simple_bart(y_fit, X_imputed, data$X_test, irs = 0L, s)
  evaluate_fit(fit, f_train, data$f_test)
}
