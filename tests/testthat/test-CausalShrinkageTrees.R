test_that("CausalShrinkageForest works for continuous outcome", {
  
  n <- 50
  p <- 3
  X <- matrix(runif(n * p), ncol = p)
  X_treat <- X_control <- X
  treatment <- rbinom(n, 1, X[, 1])
  tau <- 2
  y <- X[, 1] + (0.5 - treatment) * tau + rnorm(n)
  
  # Fit the model
  set.seed(1)
  fit <- CausalShrinkageForest(
    y = y,
    X_train_control = X_control,
    X_train_treat = X_treat,
    treatment_indicator_train = treatment,
    outcome_type = "continuous",
    number_of_trees_treat = 5,
    number_of_trees_control = 5,
    prior_type_control = "horseshoe",
    prior_type_treat = "horseshoe",
    local_hp_control = 0.1 / sqrt(5),
    local_hp_treat = 0.1 / sqrt(5),
    global_hp_control = 0.1 / sqrt(5),
    global_hp_treat = 0.1 / sqrt(5),
    N_post = 10,
    N_burn = 5,
    store_posterior_sample = TRUE,
    verbose = FALSE
  )
  
  # Basic checks
  expect_type(fit, "list")
  expect_true("train_predictions" %in% names(fit))
  expect_length(fit$train_predictions, n)
  expect_length(fit$sigma, 10)  # N_post + N_burn = 10 + 5
  
  # Numerical checks
  fit_numeric <- unlist(fit[vapply(fit, is.numeric, logical(1))])
  expect_false(any(is.na(fit_numeric)))
  expect_true(all(is.finite(fit_numeric)))
  
  # Check acceptance ratios
  expect_length(fit$acceptance_ratio_control, 1)
  expect_length(fit$acceptance_ratio_treat, 1)
  expect_true(fit$acceptance_ratio_control >= 0 & fit$acceptance_ratio_control <= 1)
  expect_true(fit$acceptance_ratio_treat >= 0 & fit$acceptance_ratio_treat <= 1)
  
  # Check posterior samples
  expect_equal(dim(fit$train_predictions_sample_control), c(10, n))
  expect_equal(dim(fit$train_predictions_sample_treat), c(10, n))
  
  # Check predictions are not all constant
  expect_true(sd(fit$train_predictions) > 0)
  expect_true(sd(fit$train_predictions_control) > 0)
  expect_true(sd(fit$train_predictions_treat) > 0)
  
  # Check reproducibility
  set.seed(1)
  fit2 <- CausalShrinkageForest(
    y = y,
    X_train_control = X_control,
    X_train_treat = X_treat,
    treatment_indicator_train = treatment,
    outcome_type = "continuous",
    number_of_trees_treat = 5,
    number_of_trees_control = 5,
    prior_type_control = "horseshoe",
    prior_type_treat = "horseshoe",
    local_hp_control = 0.1 / sqrt(5),
    local_hp_treat = 0.1 / sqrt(5),
    global_hp_control = 0.1 / sqrt(5),
    global_hp_treat = 0.1 / sqrt(5),
    N_post = 10,
    N_burn = 5,
    store_posterior_sample = TRUE,
    verbose = FALSE
  )
  expect_equal(fit$train_predictions, fit2$train_predictions)
})


test_that("CausalShrinkageForest works for right-censored survival outcome", {
  
  n <- 50
  p <- 3
  X <- matrix(runif(n * p), ncol = p)
  X_treat <- X_control <- X
  treatment <- rbinom(n, 1, 0.5)
  tau <- -0.5  # protective effect
  
  log_T0 <- X[, 1] + rnorm(n)
  log_T <- log_T0 + treatment * tau
  true_time <- exp(log_T)
  
  censor_time <- rexp(n, rate = 0.1)
  time_obs <- pmin(true_time, censor_time)
  status <- as.integer(true_time <= censor_time)
  
  # Fit the model
  set.seed(1)
  fit <- CausalShrinkageForest(
    y = time_obs,
    status = status,
    X_train_control = X_control,
    X_train_treat = X_treat,
    treatment_indicator_train = treatment,
    outcome_type = "right-censored",
    timescale = "log",
    number_of_trees_treat = 5,
    number_of_trees_control = 5,
    prior_type_control = "horseshoe",
    prior_type_treat = "horseshoe",
    local_hp_control = 0.1 / sqrt(5),
    local_hp_treat = 0.1 / sqrt(5),
    global_hp_control = 0.1 / sqrt(5),
    global_hp_treat = 0.1 / sqrt(5),
    N_post = 10,
    N_burn = 5,
    store_posterior_sample = TRUE,
    verbose = FALSE
  )
  
  # Basic checks
  expect_type(fit, "list")
  expect_true("train_predictions" %in% names(fit))
  expect_length(fit$train_predictions, n)
  expect_length(fit$sigma, 10)
  
  # Numerical checks
  fit_numeric <- unlist(fit[vapply(fit, is.numeric, logical(1))])
  expect_false(any(is.na(fit_numeric)))
  expect_true(all(is.finite(fit_numeric)))
  
  # Check acceptance ratios
  expect_length(fit$acceptance_ratio_control, 1)
  expect_length(fit$acceptance_ratio_treat, 1)
  expect_true(fit$acceptance_ratio_control >= 0 & fit$acceptance_ratio_control <= 1)
  expect_true(fit$acceptance_ratio_treat >= 0 & fit$acceptance_ratio_treat <= 1)
  
  # Check posterior samples
  expect_equal(dim(fit$train_predictions_sample_control), c(10, n))
  expect_equal(dim(fit$train_predictions_sample_treat), c(10, n))
  
  # Check predictions are not all constant
  expect_true(sd(fit$train_predictions) > 0)
  expect_true(sd(fit$train_predictions_control) > 0)
  expect_true(sd(fit$train_predictions_treat) > 0)
  
  # Check reproducibility
  set.seed(1)
  fit2 <- CausalShrinkageForest(
    y = time_obs,
    status = status,
    X_train_control = X_control,
    X_train_treat = X_treat,
    treatment_indicator_train = treatment,
    outcome_type = "right-censored",
    timescale = "log",
    number_of_trees_treat = 5,
    number_of_trees_control = 5,
    prior_type_control = "horseshoe",
    prior_type_treat = "horseshoe",
    local_hp_control = 0.1 / sqrt(5),
    local_hp_treat = 0.1 / sqrt(5),
    global_hp_control = 0.1 / sqrt(5),
    global_hp_treat = 0.1 / sqrt(5),
    N_post = 10,
    N_burn = 5,
    store_posterior_sample = TRUE,
    verbose = FALSE
  )
  expect_equal(fit$train_predictions, fit2$train_predictions)
})


test_that("CausalShrinkageForest works for continuous outcome with half-cauchy prior", {
  
  n <- 50
  p <- 3
  X <- matrix(runif(n * p), ncol = p)
  X_treat <- X_control <- X
  treatment <- rbinom(n, 1, X[, 1])
  tau <- 2
  y <- X[, 1] + (0.5 - treatment) * tau + rnorm(n)

  set.seed(1)
  fit <- CausalShrinkageForest(
    y = y,
    X_train_control = X_control,
    X_train_treat = X_treat,
    treatment_indicator_train = treatment,
    outcome_type = "continuous",
    number_of_trees_treat = 5,
    number_of_trees_control = 5,
    prior_type_control = "half-cauchy",
    prior_type_treat = "half-cauchy",
    local_hp_control = 1 / sqrt(5),
    local_hp_treat = 1 / sqrt(5),
    N_post = 10,
    N_burn = 5,
    store_posterior_sample = TRUE,
    verbose = FALSE
  )
  
  expect_type(fit, "list")
  expect_true("train_predictions" %in% names(fit))
  expect_length(fit$train_predictions, n)
  expect_length(fit$sigma, 10)
  fit_numeric <- unlist(fit[vapply(fit, is.numeric, logical(1))])
  expect_false(any(is.na(fit_numeric)))
  expect_true(all(is.finite(fit_numeric)))
  expect_true(sd(fit$train_predictions) > 0)
  expect_true(sd(fit$train_predictions_control) > 0)
  expect_true(sd(fit$train_predictions_treat) > 0)
  expect_equal(dim(fit$train_predictions_sample_control), c(10, n))
  expect_equal(dim(fit$train_predictions_sample_treat), c(10, n))
})

test_that("CausalShrinkageForest works for continuous outcome with horseshoe_fw prior", {
  
  n <- 50
  p <- 3
  X <- matrix(runif(n * p), ncol = p)
  X_treat <- X_control <- X
  treatment <- rbinom(n, 1, X[, 1])
  tau <- 2
  y <- X[, 1] + (0.5 - treatment) * tau + rnorm(n)
  
  set.seed(1)
  fit <- CausalShrinkageForest(
    y = y,
    X_train_control = X_control,
    X_train_treat = X_treat,
    treatment_indicator_train = treatment,
    outcome_type = "continuous",
    number_of_trees_treat = 5,
    number_of_trees_control = 5,
    prior_type_control = "horseshoe_fw",
    prior_type_treat = "horseshoe_fw",
    local_hp_control = 0.1 / sqrt(5),
    local_hp_treat = 0.1 / sqrt(5),
    global_hp_control = 0.1 / sqrt(5),
    global_hp_treat = 0.1 / sqrt(5),
    N_post = 10,
    N_burn = 5,
    store_posterior_sample = TRUE,
    verbose = FALSE
  )
  
  expect_type(fit, "list")
  expect_true("train_predictions" %in% names(fit))
  expect_true("forestwide_shrinkage_control" %in% names(fit))
  expect_true("forestwide_shrinkage_treat" %in% names(fit))
  
  expect_length(fit$train_predictions, n)
  expect_length(fit$sigma, 10)
  
  # Check forestwide shrinkage vectors
  expect_true(is.numeric(fit$forestwide_shrinkage_control))
  expect_true(is.numeric(fit$forestwide_shrinkage_treat))
  expect_length(fit$forestwide_shrinkage_control, 10)
  expect_length(fit$forestwide_shrinkage_treat, 10)
  
  # Sanity checks on numerical values
  fit_numeric <- unlist(fit[vapply(fit, is.numeric, logical(1))])
  expect_false(any(is.na(fit_numeric)))
  expect_true(all(is.finite(fit_numeric)))
  
  expect_true(sd(fit$train_predictions) > 0)
  expect_true(sd(fit$train_predictions_control) > 0)
  expect_true(sd(fit$train_predictions_treat) > 0)
  
  expect_equal(dim(fit$train_predictions_sample_control), c(10, n))
  expect_equal(dim(fit$train_predictions_sample_treat), c(10, n))
  
  # Forestwide shrinkage values should be non-negative
  expect_true(all(fit$forestwide_shrinkage_control >= 0))
  expect_true(all(fit$forestwide_shrinkage_treat >= 0))
})


# ── treatment_coding ──────────────────────────────────────────────────────────

test_that("CausalShrinkageForest works with treatment_coding = 'binary'", {
  n <- 50; p <- 3
  X <- matrix(runif(n * p), ncol = p)
  treatment <- rbinom(n, 1, 0.5)
  y <- X[, 1] + treatment * 2 + rnorm(n)
  lh <- 0.1 / sqrt(5)

  set.seed(1)
  fit <- CausalShrinkageForest(
    y = y,
    X_train_control = X, X_train_treat = X,
    treatment_indicator_train = treatment,
    outcome_type = "continuous",
    treatment_coding = "binary",
    number_of_trees_control = 5, number_of_trees_treat = 5,
    prior_type_control = "horseshoe", prior_type_treat = "horseshoe",
    local_hp_control = lh, global_hp_control = lh,
    local_hp_treat  = lh, global_hp_treat  = lh,
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

test_that("CausalShrinkageForest works with treatment_coding = 'adaptive'", {
  n <- 50; p <- 3
  X <- matrix(runif(n * p), ncol = p)
  treatment <- rbinom(n, 1, X[, 1])
  y <- X[, 1] + treatment * 2 + rnorm(n)
  propensity <- X[, 1]  # true propensity
  lh <- 0.1 / sqrt(5)

  set.seed(1)
  fit <- CausalShrinkageForest(
    y = y,
    X_train_control = X, X_train_treat = X,
    treatment_indicator_train = treatment,
    outcome_type = "continuous",
    treatment_coding = "adaptive",
    propensity = propensity,
    number_of_trees_control = 5, number_of_trees_treat = 5,
    prior_type_control = "horseshoe", prior_type_treat = "horseshoe",
    local_hp_control = lh, global_hp_control = lh,
    local_hp_treat  = lh, global_hp_treat  = lh,
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

test_that("CausalShrinkageForest errors when adaptive coding used without propensity", {
  n <- 30; p <- 3
  X <- matrix(runif(n * p), ncol = p)
  treatment <- rbinom(n, 1, 0.5)
  y <- X[, 1] + treatment * 2 + rnorm(n)
  lh <- 0.1 / sqrt(5)

  expect_error(
    CausalShrinkageForest(
      y = y,
      X_train_control = X, X_train_treat = X,
      treatment_indicator_train = treatment,
      outcome_type = "continuous",
      treatment_coding = "adaptive",
      number_of_trees_control = 5, number_of_trees_treat = 5,
      prior_type_control = "horseshoe", prior_type_treat = "horseshoe",
      local_hp_control = lh, global_hp_control = lh,
      local_hp_treat  = lh, global_hp_treat  = lh,
      N_post = 10, N_burn = 5,
      verbose = FALSE
    ),
    "propensity"
  )
})

test_that("CausalShrinkageForest works with treatment_coding = 'invariant'", {
  n <- 50; p <- 3
  X <- matrix(runif(n * p), ncol = p)
  treatment <- rbinom(n, 1, 0.5)
  y <- X[, 1] + treatment * 2 + rnorm(n)
  lh <- 0.1 / sqrt(5)

  set.seed(1)
  fit <- CausalShrinkageForest(
    y = y,
    X_train_control = X, X_train_treat = X,
    treatment_indicator_train = treatment,
    outcome_type = "continuous",
    treatment_coding = "invariant",
    number_of_trees_control = 5, number_of_trees_treat = 5,
    prior_type_control = "horseshoe", prior_type_treat = "horseshoe",
    local_hp_control = lh, global_hp_control = lh,
    local_hp_treat  = lh, global_hp_treat  = lh,
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

test_that("CausalShrinkageForest returns CausalShrinkageForest S3 object with working methods", {
  n <- 30; p <- 3
  X <- matrix(runif(n * p), ncol = p)
  treatment <- rbinom(n, 1, X[, 1])
  y <- X[, 1] + (0.5 - treatment) * 2 + rnorm(n)
  lh <- 0.1 / sqrt(5)

  set.seed(1)
  fit <- CausalShrinkageForest(
    y = y,
    X_train_control = X, X_train_treat = X,
    treatment_indicator_train = treatment,
    outcome_type = "continuous",
    number_of_trees_control = 5, number_of_trees_treat = 5,
    prior_type_control = "horseshoe", prior_type_treat = "horseshoe",
    local_hp_control = lh, global_hp_control = lh,
    local_hp_treat  = lh, global_hp_treat  = lh,
    N_post = 10, N_burn = 5,
    store_posterior_sample = TRUE,
    verbose = FALSE
  )

  expect_s3_class(fit, "CausalShrinkageForest")
  expect_no_error(capture.output(print(fit)))
  expect_no_error(smry <- summary(fit))
  expect_type(smry, "list")
  expect_true(!is.null(smry$sigma))
  # ATE should be available since store_posterior_sample = TRUE
  expect_true(!is.null(smry$treatment_effect$ate))
  expect_true(!is.null(smry$treatment_effect$ate_lower))
})


# ── Multi-chain (n_chains) ────────────────────────────────────────────────────

test_that("CausalShrinkageForest n_chains > 1 pools chains correctly", {
  n <- 30; p <- 3
  X <- matrix(runif(n * p), ncol = p)
  treatment <- rbinom(n, 1, X[, 1])
  y <- X[, 1] + (0.5 - treatment) * 2 + rnorm(n)
  lh <- 0.1 / sqrt(5)

  set.seed(1)
  fit <- CausalShrinkageForest(
    y = y,
    X_train_control = X, X_train_treat = X,
    treatment_indicator_train = treatment,
    outcome_type = "continuous",
    number_of_trees_control = 5, number_of_trees_treat = 5,
    prior_type_control = "horseshoe", prior_type_treat = "horseshoe",
    local_hp_control = lh, global_hp_control = lh,
    local_hp_treat  = lh, global_hp_treat  = lh,
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

