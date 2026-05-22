test_that("SurvivalBCF works for right-censored survival outcome", {
  
  n <- 50
  p <- 3
  
  # Generate covariates
  set.seed(123)
  X <- matrix(rnorm(n * p), ncol = p)
  treatment <- rbinom(n, 1, 0.5)
  
  # Generate AFT-style log survival times
  tau <- -0.5   # protective effect
  log_T0 <- 0.5 * X[, 1] + rnorm(n)
  log_T  <- log_T0 + treatment * tau
  true_time <- exp(log_T)
  
  # Censoring
  censor_time <- rexp(n, rate = 0.2)
  time_obs <- pmin(true_time, censor_time)
  status <- as.integer(true_time <= censor_time)
  
  # Fit model
  set.seed(1)
  fit <- SurvivalBCF(
    time = time_obs,
    status = status,
    X_train = X,
    treatment = treatment,
    timescale = "log",
    N_post = 10,
    N_burn = 5,
    verbose = FALSE
  )
  
  # --------------------
  # Basic structure
  # --------------------
  expect_type(fit, "list")
  
  expect_true("train_predictions" %in% names(fit))
  expect_true("train_predictions_control" %in% names(fit))
  expect_true("train_predictions_treat" %in% names(fit))
  expect_true("sigma" %in% names(fit))
  
  # --------------------
  # Dimensions
  # --------------------
  expect_length(fit$train_predictions, n)
  expect_length(fit$train_predictions_control, n)
  expect_length(fit$train_predictions_treat, n)
  expect_length(fit$sigma, 10)
  
  # --------------------
  # Numerical sanity
  # --------------------
  fit_numeric <- unlist(fit[vapply(fit, is.numeric, logical(1))])
  expect_false(any(is.na(fit_numeric)))
  expect_true(all(is.finite(fit_numeric)))
  expect_true(all(fit$sigma > 0))
  
  # --------------------
  # Posterior samples
  # --------------------
  expect_true("train_predictions_sample_control" %in% names(fit))
  expect_true("train_predictions_sample_treat" %in% names(fit))
  
  expect_equal(dim(fit$train_predictions_sample_control), c(10, n))
  expect_equal(dim(fit$train_predictions_sample_treat), c(10, n))
  
  # --------------------
  # Non-degeneracy
  # --------------------
  expect_true(sd(fit$train_predictions) > 0)
  expect_true(sd(fit$train_predictions_control) > 0)
  expect_true(sd(fit$train_predictions_treat) > 0)
  
  # --------------------
  # Reproducibility
  # --------------------
  set.seed(1)
  fit2 <- SurvivalBCF(
    time = time_obs,
    status = status,
    X_train = X,
    treatment = treatment,
    timescale = "log",
    N_post = 10,
    N_burn = 5,
    verbose = FALSE
  )
  
  expect_equal(fit$train_predictions, fit2$train_predictions)
})
