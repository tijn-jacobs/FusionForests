test_that("CausalHorseForest works for continuous outcome", {

  # Generate data
  n <- 50
  p <- 3
  X_control <- matrix(runif(n * p), ncol = p)
  X_treat   <- matrix(runif(n * p), ncol = p)
  treatment <- rbinom(n, 1, 0.5)

  # True treatment effect
  tau <- 2

  # Outcome: baseline + treatment effect + noise
  y <- X_control[, 1] + treatment * tau + rnorm(n)

  # Test data
  X_test_control <- matrix(runif(n * p), ncol = p)
  X_test_treat   <- matrix(runif(n * p), ncol = p)
  treatment_test <- rbinom(n, 1, 0.5)

  # Fit the model
  set.seed(1)
  fit <- CausalHorseForest(
    y = y,
    X_train_control = X_control,
    X_train_treat = X_treat,
    treatment_indicator_train = treatment,
    X_test_control = X_test_control,
    X_test_treat = X_test_treat,
    treatment_indicator_test = treatment_test,
    outcome_type = "continuous",
    number_of_trees = 5,
    N_post = 10,
    N_burn = 5,
    store_posterior_sample = TRUE,
    verbose = FALSE
  )
  
  # --- Basic list checks ---
  expect_type(fit, "list")
  
  # Check scalar outputs
  expect_length(fit$acceptance_ratio_control, 1)
  expect_length(fit$acceptance_ratio_treat, 1)
  expect_true(all(fit$acceptance_ratio_control >= 0 & fit$acceptance_ratio_control <= 1))
  expect_true(all(fit$acceptance_ratio_treat >= 0 & fit$acceptance_ratio_treat <= 1))
  
  # --- Check main predictions ---
  expect_length(fit$train_predictions, n)
  expect_length(fit$test_predictions, n)
  expect_length(fit$train_predictions_control, n)
  expect_length(fit$test_predictions_control, n)
  expect_length(fit$train_predictions_treat, n)
  expect_length(fit$test_predictions_treat, n)
  
  # --- Check sigma vector ---
  expect_length(fit$sigma, 10)
  expect_true(all(fit$sigma > 0))
  
  # --- Posterior samples (control) ---
  expect_equal(dim(fit$train_predictions_sample_control), c(10, n))
  expect_equal(dim(fit$test_predictions_sample_control), c(10, n))
  
  # --- Posterior samples (treat) ---
  expect_equal(dim(fit$train_predictions_sample_treat), c(10, n))
  expect_equal(dim(fit$test_predictions_sample_treat), c(10, n))
  
  # --- Numerical sanity checks ---
  fit_numeric <- unlist(fit[vapply(fit, is.numeric, logical(1))])
  expect_false(any(is.na(fit_numeric)))
  expect_false(any(is.nan(fit_numeric)))
  expect_true(all(is.finite(fit_numeric)))
  
  # Predictions are not all zero
  expect_true(sd(fit$train_predictions) > 0)
  expect_true(sd(fit$test_predictions) > 0)
  expect_true(sd(fit$train_predictions_control) > 0)
  expect_true(sd(fit$test_predictions_control) > 0)
  expect_true(sd(fit$train_predictions_treat) > 0)
  expect_true(sd(fit$test_predictions_treat) > 0)
  
  # Posterior samples non-constant
  expect_true(sd(fit$train_predictions_sample_control) > 0)
  expect_true(sd(fit$test_predictions_sample_control) > 0)
  expect_true(sd(fit$train_predictions_sample_treat) > 0)
  expect_true(sd(fit$test_predictions_sample_treat) > 0)
  
  # --- Check reproducibility ---
  set.seed(1)
  fit2 <- CausalHorseForest(
    y = y,
    X_train_control = X_control,
    X_train_treat = X_treat,
    treatment_indicator_train = treatment,
    outcome_type = "continuous",
    number_of_trees = 5,
    N_post = 10,
    N_burn = 5,
    store_posterior_sample = TRUE,
    verbose = FALSE
  )
  expect_equal(fit$train_predictions, fit2$train_predictions)
})


test_that("CausalHorseForest works for survival outcome", {
  
  # Generate data
  n <- 50
  p <- 3
  X_control <- matrix(runif(n * p), ncol = p)
  X_treat   <- matrix(runif(n * p), ncol = p)
  treatment <- rbinom(n, 1, 0.5)
  
  # True treatment effect on log-time (AFT interpretation)
  tau <- -0.5  # Negative tau => accelerates survival (protective)
  
  # Baseline log survival time
  log_T0 <- X_control[, 1] + rnorm(n)
  
  # Treatment effect
  log_T <- log_T0 + treatment * tau
  
  # Transform to actual survival time
  Ti <- exp(log_T)
  
  # Administrative censoring times (e.g., follow-up cutoff)
  C <- rexp(n, rate = 0.1)  # example: random censoring times
  
  # Observed time and status indicator
  time <- pmin(Ti, C)
  status <- as.integer(T <= C)  # 1 = event (uncensored), 0 = censored
  
  # Test data
  X_test_control <- matrix(runif(n * p), ncol = p)
  X_test_treat   <- matrix(runif(n * p), ncol = p)
  treatment_test <- rbinom(n, 1, 0.5)
  
  # Fit the model
  set.seed(1)
  fit <- CausalHorseForest(
    y = time,
    status = status,
    X_train_control = X_control,
    X_train_treat = X_treat,
    treatment_indicator_train = treatment,
    X_test_control = X_test_control,
    X_test_treat = X_test_treat,
    treatment_indicator_test = treatment_test,
    outcome_type = "right-censored",
    number_of_trees = 5,
    N_post = 10,
    N_burn = 5,
    store_posterior_sample = TRUE,
    verbose = FALSE
  )
  
  # --- Basic list checks ---
  expect_type(fit, "list")
  
  # Check scalar outputs
  expect_length(fit$acceptance_ratio_control, 1)
  expect_length(fit$acceptance_ratio_treat, 1)
  expect_true(all(fit$acceptance_ratio_control >= 0 & fit$acceptance_ratio_control <= 1))
  expect_true(all(fit$acceptance_ratio_treat >= 0 & fit$acceptance_ratio_treat <= 1))
  
  # --- Check main predictions ---
  expect_length(fit$train_predictions, n)
  expect_length(fit$test_predictions, n)
  expect_length(fit$train_predictions_control, n)
  expect_length(fit$test_predictions_control, n)
  expect_length(fit$train_predictions_treat, n)
  expect_length(fit$test_predictions_treat, n)
  
  # --- Check sigma vector ---
  expect_length(fit$sigma, 10)
  expect_true(all(fit$sigma > 0))
  
  # --- Posterior samples (control) ---
  expect_equal(dim(fit$train_predictions_sample_control), c(10, n))
  expect_equal(dim(fit$test_predictions_sample_control), c(10, n))
  
  # --- Posterior samples (treat) ---
  expect_equal(dim(fit$train_predictions_sample_treat), c(10, n))
  expect_equal(dim(fit$test_predictions_sample_treat), c(10, n))
  
  # --- Numerical sanity checks ---
  fit_numeric <- unlist(fit[vapply(fit, is.numeric, logical(1))])
  expect_false(any(is.na(fit_numeric)))
  expect_false(any(is.nan(fit_numeric)))
  expect_true(all(is.finite(fit_numeric)))
  
  # Predictions are not all zero
  expect_true(sd(fit$train_predictions) > 0)
  expect_true(sd(fit$test_predictions) > 0)
  expect_true(sd(fit$train_predictions_control) > 0)
  expect_true(sd(fit$test_predictions_control) > 0)
  expect_true(sd(fit$train_predictions_treat) > 0)
  expect_true(sd(fit$test_predictions_treat) > 0)
  
  # Posterior samples non-constant
  expect_true(sd(fit$train_predictions_sample_control) > 0)
  expect_true(sd(fit$test_predictions_sample_control) > 0)
  expect_true(sd(fit$train_predictions_sample_treat) > 0)
  expect_true(sd(fit$test_predictions_sample_treat) > 0)
  
  # --- Check reproducibility ---
  set.seed(1)
  fit2 <- CausalHorseForest(
    y = time,
    status = status,
    X_train_control = X_control,
    X_train_treat = X_treat,
    treatment_indicator_train = treatment,
    outcome_type = "right-censored",
    number_of_trees = 5,
    N_post = 10,
    N_burn = 5,
    store_posterior_sample = TRUE,
    verbose = FALSE
  )
  expect_equal(fit$train_predictions, fit2$train_predictions)
})


# ── treatment_coding ──────────────────────────────────────────────────────────

test_that("CausalHorseForest works with treatment_coding = 'binary'", {
  n <- 50; p <- 3
  X <- matrix(runif(n * p), ncol = p)
  treatment <- rbinom(n, 1, 0.5)
  y <- X[, 1] + treatment * 2 + rnorm(n)

  set.seed(1)
  fit <- CausalHorseForest(
    y = y,
    X_train_control = X, X_train_treat = X,
    treatment_indicator_train = treatment,
    outcome_type = "continuous",
    treatment_coding = "binary",
    number_of_trees = 5,
    N_post = 10, N_burn = 5,
    store_posterior_sample = TRUE,
    verbose = FALSE
  )

  expect_s3_class(fit, "CausalShrinkageForest")
  expect_equal(fit$treatment_coding, "binary")
  expect_length(fit$train_predictions, n)
  fit_numeric <- unlist(fit[vapply(fit, is.numeric, logical(1))])
  expect_false(any(is.na(fit_numeric)))
  expect_true(all(is.finite(fit_numeric)))
  expect_true(sd(fit$train_predictions) > 0)
})

test_that("CausalHorseForest works with treatment_coding = 'adaptive'", {
  n <- 50; p <- 3
  X <- matrix(runif(n * p), ncol = p)
  treatment <- rbinom(n, 1, X[, 1])
  y <- X[, 1] + treatment * 2 + rnorm(n)
  propensity <- X[, 1]  # true propensity

  set.seed(1)
  fit <- CausalHorseForest(
    y = y,
    X_train_control = X, X_train_treat = X,
    treatment_indicator_train = treatment,
    outcome_type = "continuous",
    treatment_coding = "adaptive",
    propensity = propensity,
    number_of_trees = 5,
    N_post = 10, N_burn = 5,
    store_posterior_sample = TRUE,
    verbose = FALSE
  )

  expect_s3_class(fit, "CausalShrinkageForest")
  expect_equal(fit$treatment_coding, "adaptive")
  expect_length(fit$train_predictions, n)
  fit_numeric <- unlist(fit[vapply(fit, is.numeric, logical(1))])
  expect_false(any(is.na(fit_numeric)))
  expect_true(all(is.finite(fit_numeric)))
  expect_true(sd(fit$train_predictions) > 0)
})

test_that("CausalHorseForest errors when adaptive coding used without propensity", {
  n <- 30; p <- 3
  X <- matrix(runif(n * p), ncol = p)
  treatment <- rbinom(n, 1, 0.5)
  y <- X[, 1] + treatment * 2 + rnorm(n)

  expect_error(
    CausalHorseForest(
      y = y,
      X_train_control = X, X_train_treat = X,
      treatment_indicator_train = treatment,
      outcome_type = "continuous",
      treatment_coding = "adaptive",
      number_of_trees = 5,
      N_post = 10, N_burn = 5,
      verbose = FALSE
    ),
    "propensity"
  )
})

test_that("CausalHorseForest works with treatment_coding = 'invariant'", {
  n <- 50; p <- 3
  X <- matrix(runif(n * p), ncol = p)
  treatment <- rbinom(n, 1, 0.5)
  y <- X[, 1] + treatment * 2 + rnorm(n)

  set.seed(1)
  fit <- CausalHorseForest(
    y = y,
    X_train_control = X, X_train_treat = X,
    treatment_indicator_train = treatment,
    outcome_type = "continuous",
    treatment_coding = "invariant",
    number_of_trees = 5,
    N_post = 10, N_burn = 5,
    store_posterior_sample = TRUE,
    verbose = FALSE
  )

  expect_s3_class(fit, "CausalShrinkageForest")
  expect_equal(fit$treatment_coding, "invariant")
  expect_length(fit$train_predictions, n)

  # b0 and b1 posterior draws should be returned
  expect_length(fit$b0, 10)
  expect_length(fit$b1, 10)
  expect_true(all(is.finite(fit$b0)))
  expect_true(all(is.finite(fit$b1)))

  fit_numeric <- unlist(fit[vapply(fit, is.numeric, logical(1))])
  expect_false(any(is.na(fit_numeric)))
  expect_true(all(is.finite(fit_numeric)))
  expect_true(sd(fit$train_predictions) > 0)
})


# ── S3 class and methods ──────────────────────────────────────────────────────

test_that("CausalHorseForest returns CausalShrinkageForest S3 object with working methods", {
  n <- 30; p <- 3
  X <- matrix(runif(n * p), ncol = p)
  treatment <- rbinom(n, 1, 0.5)
  y <- X[, 1] + treatment * 2 + rnorm(n)

  set.seed(1)
  fit <- CausalHorseForest(
    y = y,
    X_train_control = X, X_train_treat = X,
    treatment_indicator_train = treatment,
    outcome_type = "continuous",
    number_of_trees = 5,
    N_post = 10, N_burn = 5,
    store_posterior_sample = TRUE,
    verbose = FALSE
  )

  expect_s3_class(fit, "CausalShrinkageForest")
  expect_no_error(capture.output(print(fit)))
  expect_no_error(smry <- summary(fit))
  expect_type(smry, "list")
  expect_true(!is.null(smry$sigma))
})


# ── Multi-chain (n_chains) ────────────────────────────────────────────────────

test_that("CausalHorseForest n_chains > 1 pools chains correctly", {
  n <- 30; p <- 3
  X <- matrix(runif(n * p), ncol = p)
  treatment <- rbinom(n, 1, 0.5)
  y <- X[, 1] + treatment * 2 + rnorm(n)

  set.seed(1)
  fit <- CausalHorseForest(
    y = y,
    X_train_control = X, X_train_treat = X,
    treatment_indicator_train = treatment,
    outcome_type = "continuous",
    number_of_trees = 5,
    N_post = 10, N_burn = 5,
    n_chains = 2,
    verbose = FALSE
  )

  expect_s3_class(fit, "CausalShrinkageForest")
  expect_length(fit$sigma, 20)
  expect_equal(fit$mcmc$n_chains, 2)
  expect_length(fit$chains$acceptance_ratios_control, 2)
  expect_length(fit$chains$acceptance_ratios_treat,   2)
})

