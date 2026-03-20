test_that("SurvivalBART works for right-censored survival outcome", {
  
  # Generate covariates
  set.seed(123)
  X <- matrix(rnorm(50 * 3), ncol = 3)
  X_test <- matrix(rnorm(50 * 3), ncol = 3)
  
  # Generate survival times (AFT-style exponential model)
  linpred <- X[, 1]
  true_time <- rexp(50, rate = exp(linpred))
  
  # Generate censoring
  censor_time <- rexp(50, rate = 0.5)
  
  # Observed time + status
  time_obs <- pmin(true_time, censor_time)
  status <- as.numeric(true_time <= censor_time)
  
  # Fit model
  set.seed(1)
  fit <- SurvivalBART(
    time = time_obs,
    status = status,
    X_train = X,
    X_test = X_test,
    timescale = "time",
    number_of_trees = 50,
    k = 2.0,
    N_post = 10,
    N_burn = 5,
    verbose = FALSE
  )
  
  # Basic structure checks
  expect_type(fit, "list")
  expect_true("train_predictions" %in% names(fit))
  expect_true("test_predictions" %in% names(fit))
  expect_true("sigma" %in% names(fit))
  
  # Dimensions
  expect_length(fit$train_predictions, 50)
  expect_length(fit$test_predictions, 50)
  expect_length(fit$sigma, 10)
  
  # Numerical sanity
  expect_false(any(is.na(fit$train_predictions)))
  expect_false(any(is.nan(fit$train_predictions)))
  expect_true(all(is.finite(fit$train_predictions)))
  expect_true(all(fit$sigma > 0))
  
  # Posterior samples exist
  expect_true("train_predictions_sample" %in% names(fit))
  expect_true("test_predictions_sample" %in% names(fit))
  expect_equal(dim(fit$train_predictions_sample), c(10, 50))
  expect_equal(dim(fit$test_predictions_sample), c(10, 50))
  
  # Predictions not constant
  expect_true(sd(fit$train_predictions) > 0)
  expect_true(sd(fit$test_predictions) > 0)
  expect_true(sd(fit$train_predictions_sample) > 0)
  expect_true(sd(fit$test_predictions_sample) > 0)
  
  # Reproducibility
  set.seed(1)
  fit2 <- SurvivalBART(
    time = time_obs,
    status = status,
    X_train = X,
    X_test = X_test,
    timescale = "time",
    number_of_trees = 50,
    k = 2.0,
    N_post = 10,
    N_burn = 5,
    verbose = FALSE
  )
  
  expect_equal(fit$train_predictions, fit2$train_predictions)
})

test_that("SurvivalBART errors without status", {
  X <- matrix(rnorm(20 * 2), ncol = 2)
  time <- rexp(20)
  
  expect_error(
    SurvivalBART(
      time = time,
      status = NULL,
      X_train = X
    )
  )
})
