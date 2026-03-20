# ── predict.ShrinkageTrees ────────────────────────────────────────────────────

test_that("predict.ShrinkageTrees works for continuous outcome", {
  n <- 50; p <- 3
  X <- matrix(runif(n * p), ncol = p)
  y <- X[, 1] + rnorm(n)

  set.seed(1)
  fit <- HorseTrees(
    y = y, X_train = X,
    outcome_type = "continuous",
    number_of_trees = 5,
    N_post = 10, N_burn = 5,
    store_posterior_sample = TRUE,
    verbose = FALSE
  )

  X_new <- matrix(runif(20 * p), ncol = p)
  pred <- predict(fit, newdata = X_new)

  expect_s3_class(pred, "ShrinkageTreesPrediction")
  expect_length(pred$mean, 20)
  expect_length(pred$lower, 20)
  expect_length(pred$upper, 20)
  expect_true(all(pred$lower <= pred$mean))
  expect_true(all(pred$mean <= pred$upper))
  expect_false(any(is.na(pred$mean)))
  expect_true(all(is.finite(pred$mean)))

  # print and summary should work
  expect_no_error(capture.output(print(pred)))
  expect_no_error(capture.output(summary(pred)))
})


test_that("predict.ShrinkageTrees works for binary outcome", {
  n <- 50; p <- 3
  X <- matrix(rnorm(n * p), ncol = p)
  y <- ifelse(X[, 1] + rnorm(n) > 0, 1, 0)

  set.seed(1)
  fit <- HorseTrees(
    y = y, X_train = X,
    outcome_type = "binary",
    number_of_trees = 5,
    N_post = 10, N_burn = 5,
    store_posterior_sample = TRUE,
    verbose = FALSE
  )

  X_new <- matrix(rnorm(15 * p), ncol = p)
  pred <- predict(fit, newdata = X_new)

  expect_s3_class(pred, "ShrinkageTreesPrediction")
  expect_length(pred$mean, 15)
  # Probabilities should be in [0, 1]
  expect_true(all(pred$mean >= 0 & pred$mean <= 1))
  expect_true(all(pred$lower >= 0 & pred$lower <= 1))
  expect_true(all(pred$upper >= 0 & pred$upper <= 1))
  expect_true(all(pred$lower <= pred$upper))
})


test_that("predict.ShrinkageTrees works for right-censored survival", {
  n <- 50; p <- 3
  X <- matrix(rnorm(n * p), ncol = p)

  true_time <- rexp(n, rate = exp(0.3 * X[, 1]))
  censor_time <- rexp(n, rate = 0.5)
  time_obs <- pmin(true_time, censor_time)
  status <- as.numeric(true_time <= censor_time)

  set.seed(1)
  fit <- HorseTrees(
    y = time_obs, X_train = X,
    status = status,
    outcome_type = "right-censored",
    number_of_trees = 5,
    N_post = 10, N_burn = 5,
    store_posterior_sample = TRUE,
    verbose = FALSE
  )

  X_new <- matrix(rnorm(10 * p), ncol = p)
  pred <- predict(fit, newdata = X_new)

  expect_s3_class(pred, "ShrinkageTreesPrediction")
  expect_length(pred$mean, 10)
  expect_true(all(pred$mean > 0))
  expect_true(all(is.finite(pred$mean)))
  expect_true(all(pred$lower <= pred$upper))

  # Survival predictions should include posterior samples and sigma
  expect_true(!is.null(pred$predictions_sample))
  expect_equal(nrow(pred$predictions_sample), 10)  # N_post
  expect_equal(ncol(pred$predictions_sample), 10)   # n_new
  expect_true(!is.null(pred$sigma))
  expect_length(pred$sigma, 10)
  expect_true(all(pred$sigma > 0))
})


test_that("predict.ShrinkageTrees errors on dimension mismatch", {
  n <- 50; p <- 3
  X <- matrix(runif(n * p), ncol = p)
  y <- rnorm(n)

  set.seed(1)
  fit <- HorseTrees(
    y = y, X_train = X,
    outcome_type = "continuous",
    number_of_trees = 5,
    N_post = 10, N_burn = 5,
    store_posterior_sample = TRUE,
    verbose = FALSE
  )

  X_wrong <- matrix(runif(20 * 5), ncol = 5)  # wrong number of columns
  expect_error(predict(fit, newdata = X_wrong), "columns")
})


# ── predict.CausalShrinkageForest ────────────────────────────────────────────

test_that("predict.CausalShrinkageForest works for continuous outcome", {
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
    number_of_trees_control = 5, number_of_trees_treat = 5,
    prior_type_control = "horseshoe", prior_type_treat = "horseshoe",
    local_hp_control = lh, global_hp_control = lh,
    local_hp_treat = lh, global_hp_treat = lh,
    N_post = 10, N_burn = 5,
    store_posterior_sample = TRUE,
    verbose = FALSE
  )

  X_new <- matrix(runif(15 * p), ncol = p)
  pred <- predict(fit, newdata_control = X_new, newdata_treat = X_new)

  expect_s3_class(pred, "CausalShrinkageForestPrediction")
  expect_equal(pred$n, 15)

  # Prognostic component
  expect_length(pred$prognostic$mean, 15)
  expect_length(pred$prognostic$lower, 15)
  expect_length(pred$prognostic$upper, 15)
  expect_true(all(pred$prognostic$lower <= pred$prognostic$upper))

  # CATE component
  expect_length(pred$cate$mean, 15)
  expect_true(all(pred$cate$lower <= pred$cate$upper))

  # Total component
  expect_length(pred$total$mean, 15)
  expect_true(all(pred$total$lower <= pred$total$upper))

  # No NAs
  expect_false(any(is.na(pred$prognostic$mean)))
  expect_false(any(is.na(pred$cate$mean)))
  expect_false(any(is.na(pred$total$mean)))

  # print and summary
  expect_no_error(capture.output(print(pred)))
  expect_no_error(capture.output(summary(pred)))
})


test_that("predict.CausalShrinkageForest works for right-censored survival", {
  n <- 50; p <- 3
  X <- matrix(runif(n * p), ncol = p)
  treatment <- rbinom(n, 1, 0.5)

  log_T0 <- X[, 1] + rnorm(n)
  log_T <- log_T0 + treatment * (-0.5)
  true_time <- exp(log_T)
  censor_time <- rexp(n, rate = 0.1)
  time_obs <- pmin(true_time, censor_time)
  status <- as.integer(true_time <= censor_time)

  lh <- 0.1 / sqrt(5)

  set.seed(1)
  fit <- CausalShrinkageForest(
    y = time_obs, status = status,
    X_train_control = X, X_train_treat = X,
    treatment_indicator_train = treatment,
    outcome_type = "right-censored", timescale = "log",
    number_of_trees_control = 5, number_of_trees_treat = 5,
    prior_type_control = "horseshoe", prior_type_treat = "horseshoe",
    local_hp_control = lh, global_hp_control = lh,
    local_hp_treat = lh, global_hp_treat = lh,
    N_post = 10, N_burn = 5,
    store_posterior_sample = TRUE,
    verbose = FALSE
  )

  X_new <- matrix(runif(10 * p), ncol = p)
  pred <- predict(fit, newdata_control = X_new, newdata_treat = X_new)

  expect_s3_class(pred, "CausalShrinkageForestPrediction")
  expect_equal(pred$n, 10)
  expect_length(pred$prognostic$mean, 10)
  expect_length(pred$cate$mean, 10)
  expect_length(pred$total$mean, 10)
  expect_false(any(is.na(pred$prognostic$mean)))
  expect_true(all(is.finite(pred$cate$mean)))
})


test_that("predict.CausalShrinkageForest errors on dimension mismatch", {
  n <- 30; p <- 3
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
    number_of_trees_control = 5, number_of_trees_treat = 5,
    prior_type_control = "horseshoe", prior_type_treat = "horseshoe",
    local_hp_control = lh, global_hp_control = lh,
    local_hp_treat = lh, global_hp_treat = lh,
    N_post = 10, N_burn = 5,
    store_posterior_sample = TRUE,
    verbose = FALSE
  )

  X_wrong <- matrix(runif(10 * 5), ncol = 5)
  expect_error(predict(fit, newdata_control = X_wrong, newdata_treat = X_wrong),
               "columns")

  # Row mismatch
  X_a <- matrix(runif(10 * p), ncol = p)
  X_b <- matrix(runif(5 * p), ncol = p)
  expect_error(predict(fit, newdata_control = X_a, newdata_treat = X_b),
               "same number of rows")
})

