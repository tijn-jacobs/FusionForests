# High-dimensional regression tests (p >> n).
#
# These tests use n = 50, p = 200 with only 3 active predictors.
# They guard against C++ crashes, numerical blow-up, and wrong dimension
# handling in the p > n regime — scenarios that the standard p = 3 tests
# cannot detect.
#
# MCMC settings are kept minimal (N_post = 10, N_burn = 5, trees = 10)
# to keep runtime acceptable on CI.

# ---------------------------------------------------------------------------
# Shared sparse-AFT data (n = 50, p = 200, 3 active predictors)
# ---------------------------------------------------------------------------
.hd_data <- local({
  set.seed(42)
  n <- 50L; p <- 200L
  X      <- matrix(rnorm(n * p), nrow = n)
  log_T  <- 1.5 * X[, 1] - 1.0 * X[, 2] + 0.5 * X[, 3] + rnorm(n)
  C      <- rexp(n, rate = 0.4)
  time   <- pmin(exp(log_T), C)
  status <- as.integer(exp(log_T) <= C)
  X_test <- matrix(rnorm(20L * p), nrow = 20L)
  list(X = X, log_T = log_T, time = time, status = status,
       X_test = X_test, n = n, p = p)
})


# ---------------------------------------------------------------------------
# HorseTrees — survival, p >> n
# ---------------------------------------------------------------------------

test_that("HorseTrees handles p >> n survival without error", {
  d <- .hd_data
  set.seed(1)
  fit <- HorseTrees(
    y               = d$time,
    status          = d$status,
    X_train         = d$X,
    X_test          = d$X_test,
    outcome_type    = "right-censored",
    number_of_trees = 10,
    N_post          = 10,
    N_burn          = 5,
    store_posterior_sample = TRUE,
    verbose         = FALSE
  )

  # Dimensions
  expect_length(fit$train_predictions, d$n)
  expect_length(fit$test_predictions,  nrow(d$X_test))
  expect_length(fit$sigma, 10)
  expect_equal(dim(fit$train_predictions_sample), c(10L, d$n))
  expect_equal(dim(fit$test_predictions_sample),  c(10L, nrow(d$X_test)))

  # Numerical sanity
  expect_true(all(is.finite(fit$train_predictions)))
  expect_true(all(is.finite(fit$test_predictions)))
  expect_true(all(is.finite(fit$train_predictions_sample)))
  expect_true(all(fit$sigma > 0))

  # Predictions are not all identical (shrinkage didn't collapse to zero)
  expect_gt(sd(fit$train_predictions), 0)
  expect_gt(sd(fit$test_predictions),  0)
})


# ---------------------------------------------------------------------------
# ShrinkageTrees (horseshoe) — survival, p >> n
# ---------------------------------------------------------------------------

test_that("ShrinkageTrees horseshoe handles p >> n survival without error", {
  d  <- .hd_data
  lh <- 0.1 / sqrt(10)
  set.seed(2)
  fit <- ShrinkageTrees(
    y               = d$time,
    status          = d$status,
    X_train         = d$X,
    X_test          = d$X_test,
    outcome_type    = "right-censored",
    prior_type      = "horseshoe",
    local_hp        = lh,
    global_hp       = lh,
    number_of_trees = 10,
    N_post          = 10,
    N_burn          = 5,
    store_posterior_sample = TRUE,
    verbose         = FALSE
  )

  expect_length(fit$train_predictions, d$n)
  expect_length(fit$test_predictions,  nrow(d$X_test))
  expect_length(fit$sigma, 10)
  expect_equal(dim(fit$train_predictions_sample), c(10L, d$n))
  expect_equal(dim(fit$test_predictions_sample),  c(10L, nrow(d$X_test)))

  expect_true(all(is.finite(fit$train_predictions)))
  expect_true(all(is.finite(fit$test_predictions)))
  expect_true(all(is.finite(fit$train_predictions_sample)))
  expect_true(all(fit$sigma > 0))
  expect_gt(sd(fit$train_predictions), 0)
  expect_gt(sd(fit$test_predictions),  0)
})


# ---------------------------------------------------------------------------
# SurvivalDART — survival, p >> n
# ---------------------------------------------------------------------------

test_that("SurvivalDART handles p >> n survival without error", {
  d <- .hd_data
  set.seed(3)
  fit <- SurvivalDART(
    time            = d$time,
    status          = d$status,
    X_train         = d$X,
    X_test          = d$X_test,
    number_of_trees = 10,
    rho_dirichlet   = 3,      # prior: ~3 active predictors
    N_post          = 10,
    N_burn          = 5,
    verbose         = FALSE
  )

  expect_length(fit$train_predictions, d$n)
  expect_length(fit$test_predictions,  nrow(d$X_test))
  expect_length(fit$sigma, 10)
  expect_equal(dim(fit$train_predictions_sample), c(10L, d$n))
  expect_equal(dim(fit$test_predictions_sample),  c(10L, nrow(d$X_test)))

  expect_true(all(is.finite(fit$train_predictions)))
  expect_true(all(is.finite(fit$test_predictions)))
  expect_true(all(is.finite(fit$train_predictions_sample)))
  expect_true(all(fit$sigma > 0))
  expect_gt(sd(fit$train_predictions), 0)
  expect_gt(sd(fit$test_predictions),  0)

  # DART should record split probabilities for variable importance
  expect_false(is.null(fit$split_probs))
  expect_equal(ncol(fit$split_probs), d$p)
})


# ---------------------------------------------------------------------------
# predict() after high-dimensional fit
# ---------------------------------------------------------------------------

test_that("predict() works after high-dimensional ShrinkageTrees fit", {
  d  <- .hd_data
  lh <- 0.1 / sqrt(10)
  set.seed(4)
  fit <- ShrinkageTrees(
    y               = d$time,
    status          = d$status,
    X_train         = d$X,
    outcome_type    = "right-censored",
    prior_type      = "horseshoe",
    local_hp        = lh,
    global_hp       = lh,
    number_of_trees = 10,
    N_post          = 10,
    N_burn          = 5,
    store_posterior_sample = TRUE,
    verbose         = FALSE
  )

  pred <- predict(fit, newdata = d$X_test)
  expect_s3_class(pred, "ShrinkageTreesPrediction")
  expect_length(pred$mean,  nrow(d$X_test))
  expect_length(pred$lower, nrow(d$X_test))
  expect_length(pred$upper, nrow(d$X_test))
  expect_true(all(is.finite(pred$mean)))
  expect_true(all(pred$lower <= pred$mean))
  expect_true(all(pred$mean  <= pred$upper))
})


# ---------------------------------------------------------------------------
# ShrinkageTrees (horseshoe_fw) — survival, p >> n
# ---------------------------------------------------------------------------

test_that("ShrinkageTrees horseshoe_fw handles p >> n survival without error", {
  d  <- .hd_data
  lh <- 0.1 / sqrt(10)
  set.seed(5)
  fit <- ShrinkageTrees(
    y               = d$time,
    status          = d$status,
    X_train         = d$X,
    outcome_type    = "right-censored",
    prior_type      = "horseshoe_fw",
    local_hp        = lh,
    global_hp       = lh,
    number_of_trees = 10,
    N_post          = 10,
    N_burn          = 5,
    verbose         = FALSE
  )

  expect_length(fit$train_predictions, d$n)
  expect_true(all(is.finite(fit$train_predictions)))
  expect_true(all(fit$sigma > 0))
  expect_true(is.numeric(fit$forestwide_shrinkage))
  expect_true(all(fit$forestwide_shrinkage >= 0))
})

