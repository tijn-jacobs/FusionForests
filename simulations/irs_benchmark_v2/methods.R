# ============================================================================
# Method wrappers for IRS benchmark v2
#
# Each method fits Y ~ m(X, A), then computes CATE by predicting
# at (X, A=1) and (X, A=0) and taking the difference.
#
# All methods return the output of evaluate_fit().
# ============================================================================

source("evaluation.R")

# --- Shared BART settings ---------------------------------------------------

bart_args <- function(s) {
  list(
    number_of_trees       = s$number_of_trees,
    N_post                = s$N_post,
    N_burn                = s$N_burn,
    verbose               = FALSE,
    store_posterior_sample = TRUE
  )
}

#' Fit SimpleBART and also predict at counterfactual A=1, A=0
fit_bart_with_cate <- function(y, XA_train, XA_test,
                               XA_test_1, XA_test_0,
                               XA_train_1, XA_train_0,
                               irs, s) {
  # Main fit: predict at observed (X, A)
  fit <- do.call(SimpleBART, c(
    list(y = y, X_train = XA_train, X_test = XA_test,
         irs = as.integer(irs)),
    bart_args(s)
  ))

  # Counterfactual predictions for CATE
  # Refit is unnecessary — use predict on the same forest

  # But SimpleBART doesn't have a predict method separate from
  # fitting, so we fit twice with counterfactual test sets.
  # To avoid refitting, we do one fit with all three test sets
  # stacked, then split the predictions.
  n_test <- nrow(XA_test)
  n_train <- nrow(XA_train)
  XA_all_test <- rbind(XA_test, XA_test_1, XA_test_0)
  XA_all_train_cf <- rbind(XA_train_1, XA_train_0)

  # Re-fit with stacked test matrices
  fit_all <- do.call(SimpleBART, c(
    list(y = y, X_train = XA_train,
         X_test = rbind(XA_all_test, XA_all_train_cf),
         irs = as.integer(irs)),
    bart_args(s)
  ))

  # Split predictions
  idx1 <- 1:n_test
  idx2 <- (n_test + 1):(2 * n_test)
  idx3 <- (2 * n_test + 1):(3 * n_test)
  idx4 <- (3 * n_test + 1):(3 * n_test + n_train)
  idx5 <- (3 * n_test + n_train + 1):(3 * n_test +
                                       2 * n_train)

  fit_obs <- list(
    train_predictions = fit_all$train_predictions,
    test_predictions  = fit_all$test_predictions[idx1]
  )
  fit_1 <- list(
    train_predictions = fit_all$test_predictions[idx4],
    test_predictions  = fit_all$test_predictions[idx2]
  )
  fit_0 <- list(
    train_predictions = fit_all$test_predictions[idx5],
    test_predictions  = fit_all$test_predictions[idx3]
  )

  if (!is.null(fit_all$test_predictions_sample)) {
    fit_obs$test_predictions_sample <-
      fit_all$test_predictions_sample[, idx1, drop = FALSE]
    fit_1$test_predictions_sample <-
      fit_all$test_predictions_sample[, idx2, drop = FALSE]
    fit_0$test_predictions_sample <-
      fit_all$test_predictions_sample[, idx3, drop = FALSE]
    fit_1$train_predictions_sample <-
      fit_all$test_predictions_sample[, idx4, drop = FALSE]
    fit_0$train_predictions_sample <-
      fit_all$test_predictions_sample[, idx5, drop = FALSE]
  }

  list(fit = fit_obs, fit_1 = fit_1, fit_0 = fit_0)
}

# --- Method: Oracle ---------------------------------------------------------
# Fit on full data with no missingness

run_oracle <- function(data, s) {
  fits <- fit_bart_with_cate(
    y = data$y,
    XA_train   = data$XA_train,
    XA_test    = data$XA_test,
    XA_test_1  = data$XA_test_1,
    XA_test_0  = data$XA_test_0,
    XA_train_1 = data$XA_train_1,
    XA_train_0 = data$XA_train_0,
    irs = 0L, s = s
  )
  evaluate_fit(
    fits$fit, fits$fit_1, fits$fit_0,
    data$m_train, data$m_test,
    data$tau_train, data$tau_test
  )
}

# --- Method: IRS (informed, mode 2) ----------------------------------------

run_irs <- function(data, miss, s, irs_mode = 2L) {
  fits <- fit_bart_with_cate(
    y = data$y,
    XA_train   = miss$XA_miss,
    XA_test    = data$XA_test,
    XA_test_1  = data$XA_test_1,
    XA_test_0  = data$XA_test_0,
    XA_train_1 = data$XA_train_1,
    XA_train_0 = data$XA_train_0,
    irs = irs_mode, s = s
  )
  evaluate_fit(
    fits$fit, fits$fit_1, fits$fit_0,
    data$m_train, data$m_test,
    data$tau_train, data$tau_test
  )
}

# --- Method: Complete case --------------------------------------------------
# Only use observations with no missing values

run_complete_case <- function(data, miss, s) {
  obs_rows <- !is.nan(miss$XA_miss[, 5])
  n_obs <- sum(obs_rows)

  if (n_obs < 30) {
    return(null_result())
  }

  XA_cc <- miss$XA_miss[obs_rows, , drop = FALSE]
  y_cc  <- data$y[obs_rows]
  m_cc  <- data$m_train[obs_rows]
  tau_cc <- data$tau_train[obs_rows]

  # Counterfactual train matrices for complete cases
  XA_cc_1 <- XA_cc; XA_cc_1[, 6] <- 1
  XA_cc_0 <- XA_cc; XA_cc_0[, 6] <- 0

  fits <- fit_bart_with_cate(
    y = y_cc,
    XA_train   = XA_cc,
    XA_test    = data$XA_test,
    XA_test_1  = data$XA_test_1,
    XA_test_0  = data$XA_test_0,
    XA_train_1 = XA_cc_1,
    XA_train_0 = XA_cc_0,
    irs = 0L, s = s
  )
  evaluate_fit(
    fits$fit, fits$fit_1, fits$fit_0,
    m_cc, data$m_test,
    tau_cc, data$tau_test
  )
}

# --- Method: Complete covariates --------------------------------------------
# Use all observations but drop X5 entirely

run_complete_covariates <- function(data, s) {
  # Remove X5 (column 5), keep [X1..X4, A]
  drop5 <- function(M) M[, -5, drop = FALSE]

  XA_no5       <- drop5(data$XA_train)
  XA_test_no5  <- drop5(data$XA_test)
  XA_test1_no5 <- drop5(data$XA_test_1)
  XA_test0_no5 <- drop5(data$XA_test_0)
  XA_tr1_no5   <- drop5(data$XA_train_1)
  XA_tr0_no5   <- drop5(data$XA_train_0)

  fits <- fit_bart_with_cate(
    y = data$y,
    XA_train   = XA_no5,
    XA_test    = XA_test_no5,
    XA_test_1  = XA_test1_no5,
    XA_test_0  = XA_test0_no5,
    XA_train_1 = XA_tr1_no5,
    XA_train_0 = XA_tr0_no5,
    irs = 0L, s = s
  )
  evaluate_fit(
    fits$fit, fits$fit_1, fits$fit_0,
    data$m_train, data$m_test,
    data$tau_train, data$tau_test
  )
}

# --- Method: bartMachine MIA -----------------------------------------------

run_bartmachine <- function(data, miss, s) {
  XA_na <- miss$XA_miss
  XA_na[is.nan(XA_na)] <- NA

  bm <- bartMachine::bartMachine(
    as.data.frame(XA_na), data$y,
    num_trees                          = s$number_of_trees,
    num_burn_in                        = s$N_burn,
    num_iterations_after_burn_in       = s$N_post,
    use_missing_data                   = TRUE,
    use_missing_data_dummies_as_covars = TRUE,
    verbose                            = FALSE
  )

  # Predict at observed, A=1, A=0
  pred_obs <- bartMachine::bart_machine_get_posterior(
    bm, as.data.frame(data$XA_test)
  )
  pred_1 <- bartMachine::bart_machine_get_posterior(
    bm, as.data.frame(data$XA_test_1)
  )
  pred_0 <- bartMachine::bart_machine_get_posterior(
    bm, as.data.frame(data$XA_test_0)
  )
  pred_tr <- bartMachine::bart_machine_get_posterior(
    bm, as.data.frame(XA_na)
  )
  pred_tr1 <- bartMachine::bart_machine_get_posterior(
    bm, as.data.frame(data$XA_train_1)
  )
  pred_tr0 <- bartMachine::bart_machine_get_posterior(
    bm, as.data.frame(data$XA_train_0)
  )

  fit_obs <- list(
    train_predictions = pred_tr$y_hat,
    test_predictions  = pred_obs$y_hat
  )
  fit_1 <- list(
    train_predictions = pred_tr1$y_hat,
    test_predictions  = pred_1$y_hat,
    test_predictions_sample = t(pred_1$y_hat_posterior_samples)
  )
  fit_0 <- list(
    train_predictions = pred_tr0$y_hat,
    test_predictions  = pred_0$y_hat,
    test_predictions_sample = t(pred_0$y_hat_posterior_samples)
  )

  rm(bm); gc()

  evaluate_fit(
    fit_obs, fit_1, fit_0,
    data$m_train, data$m_test,
    data$tau_train, data$tau_test
  )
}

# --- Method: missForest + BART ----------------------------------------------

run_missforest <- function(data, miss, s) {
  XA_na <- miss$XA_miss
  XA_na[is.nan(XA_na)] <- NA

  imputed <- missForest::missForest(
    as.data.frame(XA_na), verbose = FALSE
  )
  XA_imp <- as.matrix(imputed$ximp)

  # Counterfactual train from imputed data
  XA_imp_1 <- XA_imp; XA_imp_1[, 6] <- 1
  XA_imp_0 <- XA_imp; XA_imp_0[, 6] <- 0

  fits <- fit_bart_with_cate(
    y = data$y,
    XA_train   = XA_imp,
    XA_test    = data$XA_test,
    XA_test_1  = data$XA_test_1,
    XA_test_0  = data$XA_test_0,
    XA_train_1 = XA_imp_1,
    XA_train_0 = XA_imp_0,
    irs = 0L, s = s
  )
  evaluate_fit(
    fits$fit, fits$fit_1, fits$fit_0,
    data$m_train, data$m_test,
    data$tau_train, data$tau_test
  )
}

# --- Method: MICE + BART ----------------------------------------------------

run_mice <- function(data, miss, s, m_imp = 5) {
  XA_na <- miss$XA_miss
  XA_na[is.nan(XA_na)] <- NA

  imp <- mice::mice(
    as.data.frame(XA_na),
    m       = m_imp,
    method  = "pmm",
    printFlag = FALSE
  )

  # Average predictions across multiply-imputed datasets
  all_fits <- lapply(seq_len(m_imp), function(k) {
    XA_imp <- as.matrix(mice::complete(imp, k))
    XA_imp_1 <- XA_imp; XA_imp_1[, 6] <- 1
    XA_imp_0 <- XA_imp; XA_imp_0[, 6] <- 0

    fit_bart_with_cate(
      y = data$y,
      XA_train   = XA_imp,
      XA_test    = data$XA_test,
      XA_test_1  = data$XA_test_1,
      XA_test_0  = data$XA_test_0,
      XA_train_1 = XA_imp_1,
      XA_train_0 = XA_imp_0,
      irs = 0L, s = s
    )
  })

  # Pool: average point predictions across imputations
  pool_pred <- function(field, component) {
    preds <- lapply(all_fits,
                    function(f) f[[component]][[field]])
    Reduce("+", preds) / m_imp
  }

  fit_obs <- list(
    train_predictions = pool_pred("train_predictions", "fit"),
    test_predictions  = pool_pred("test_predictions", "fit")
  )
  fit_1 <- list(
    train_predictions = pool_pred("train_predictions",
                                  "fit_1"),
    test_predictions  = pool_pred("test_predictions", "fit_1")
  )
  fit_0 <- list(
    train_predictions = pool_pred("train_predictions",
                                  "fit_0"),
    test_predictions  = pool_pred("test_predictions", "fit_0")
  )

  # Pool posterior samples if available
  if (!is.null(all_fits[[1]]$fit_1$test_predictions_sample)) {
    pool_samples <- function(field, component) {
      samps <- lapply(all_fits,
                      function(f) f[[component]][[field]])
      Reduce("+", samps) / m_imp
    }
    fit_1$test_predictions_sample <- pool_samples(
      "test_predictions_sample", "fit_1"
    )
    fit_0$test_predictions_sample <- pool_samples(
      "test_predictions_sample", "fit_0"
    )
  }

  evaluate_fit(
    fit_obs, fit_1, fit_0,
    data$m_train, data$m_test,
    data$tau_train, data$tau_test
  )
}

# --- Null result placeholder ------------------------------------------------

null_result <- function() {
  list(
    rmse_m_train    = NA, rmse_m_test    = NA,
    mae_m_test      = NA,
    cate_bias_test  = NA, cate_rmse_test  = NA,
    cate_bias_train = NA, cate_rmse_train = NA,
    ate_bias        = NA, ate_hat         = NA,
    ate_true        = NA,
    cate_coverage   = NA, cate_ci_width   = NA,
    avg_post_var    = NA,
    ate_coverage    = NA, ate_ci_width    = NA
  )
}
