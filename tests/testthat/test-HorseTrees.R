test_that("HorseTrees for continuous outcome", {
  
  # Generate data
  X <- matrix(runif(50 * 3), ncol = 3)
  X_test <- matrix(runif(50 * 3), ncol = 3)
  y <- X[, 1] + rnorm(50)
  
  # Fit the model
  set.seed(1)
  fit <- HorseTrees(y = y, 
                    X_train = X,
                    X_test = X_test,
                    outcome_type = "continuous",
                    number_of_trees = 5,
                    N_post = 10,
                    N_burn = 5,
                    store_posterior_sample = TRUE,
                    verbose = FALSE)
  
  # Basic checks
  expect_type(fit, "list")
  expect_true("train_predictions" %in% names(fit))
  expect_true("test_predictions" %in% names(fit))
  expect_length(fit$train_predictions, 50)
  expect_length(fit$sigma, 10)
  
  # Numerical checks
  expect_false(any(is.na(fit$train_predictions)))
  expect_false(any(is.nan(fit$train_predictions)))
  expect_true(all(is.finite(fit$train_predictions)))
  
  # Check posterior samples
  expect_true("train_predictions_sample" %in% names(fit))
  expect_true("test_predictions_sample" %in% names(fit))
  expect_equal(dim(fit$train_predictions_sample), c(10, 50))
  expect_equal(dim(fit$test_predictions_sample), c(10, 50))
  
  # Check sigma positivity
  expect_true(all(fit$sigma > 0))
  
  # Check predictions are not all zero
  expect_true(sd(fit$train_predictions) > 0)
  expect_true(sd(fit$train_predictions_sample) > 0)
  expect_true(sd(fit$test_predictions) > 0)
  expect_true(sd(fit$test_predictions_sample) > 0)
  
  # Check reproducibility
  set.seed(1)
  fit2 <- HorseTrees(y = y, X_train = X, outcome_type = "continuous", number_of_trees = 5, 
                     N_post = 10, N_burn = 5, store_posterior_sample = TRUE, 
                     verbose = FALSE)
  expect_equal(fit$train_predictions, fit2$train_predictions)
})


test_that("HorseTrees for binary outcome", {
  
  # Generate data
  X <- matrix(rnorm(50 * 3), ncol = 3)
  X_test <- matrix(runif(50 * 3), ncol = 3)
  y <- ifelse(X[, 1] + rnorm(50) > 0, 1, 0)
  
  # Fit the model
  set.seed(1)
  fit <- HorseTrees(y = y, 
                    X_train = X,
                    X_test = X_test,
                    outcome_type = "binary",
                    number_of_trees = 5,
                    N_post = 10,
                    N_burn = 5,
                    store_posterior_sample = TRUE,
                    verbose = FALSE)
  
  # Basic checks
  expect_type(fit, "list")
  expect_true("train_predictions" %in% names(fit))
  expect_length(fit$train_predictions, 50)

  # Numerical checks
  expect_false(any(is.na(fit$train_predictions)))
  expect_false(any(is.nan(fit$train_predictions)))
  expect_true(all(is.finite(fit$train_predictions)))
  
  # Check posterior samples
  expect_true("train_predictions_sample" %in% names(fit))
  expect_true("test_predictions_sample" %in% names(fit))
  expect_true("train_probabilities_sample" %in% names(fit))
  expect_true("test_probabilities_sample" %in% names(fit))
  expect_equal(dim(fit$train_predictions_sample), c(10, 50))
  expect_equal(dim(fit$test_predictions_sample), c(10, 50))
  expect_equal(dim(fit$train_probabilities_sample), c(10, 50))
  expect_equal(dim(fit$test_probabilities_sample), c(10, 50))
  
  # Check estimated probabilities are within [0,1]
  expect_true(all(fit$train_probabilities >= 0 & fit$train_probabilities <= 1))
  expect_true(all(fit$test_probabilities >= 0 & fit$test_probabilities <= 1))
  
  # Check predictions are not all constant
  expect_true(sd(fit$train_predictions) > 0)
  expect_true(sd(fit$train_predictions_sample) > 0)
  expect_true(sd(fit$test_predictions) > 0)
  expect_true(sd(fit$test_predictions_sample) > 0)
  
  # Check reproducibility
  set.seed(1)
  fit2 <- HorseTrees(y = y, X_train = X, outcome_type = "binary", number_of_trees = 5, 
                     N_post = 10, N_burn = 5, store_posterior_sample = TRUE, 
                     verbose = FALSE)
  expect_equal(fit$train_predictions, fit2$train_predictions)
})

test_that("HorseTrees for right-censored survival outcome", {
  
  # Generate covariates
  X <- matrix(rnorm(50 * 3), ncol = 3)
  X_test <- matrix(runif(50 * 3), ncol = 3)
  
  # Generate survival times (exponential baseline hazard depending on X[,1])
  linpred <- X[, 1]
  true_time <- rexp(50, rate = exp(linpred))
  
  # Generate censoring times
  censor_time <- rexp(50, rate = 0.5)
  
  # Observed time and status indicator
  time_obs <- pmin(true_time, censor_time)
  status <- as.numeric(true_time <= censor_time)
  
  # Fit the model
  set.seed(1)
  fit <- HorseTrees(y = time_obs, 
                    X_train = X,
                    X_test = X_test,
                    outcome_type = "right-censored",
                    status = status,
                    number_of_trees = 5,
                    N_post = 10,
                    N_burn = 5,
                    store_posterior_sample = TRUE,
                    verbose = FALSE)
  
  # Basic checks
  expect_type(fit, "list")
  expect_true("train_predictions" %in% names(fit))
  expect_true("test_predictions" %in% names(fit))
  expect_true("test_predictions" %in% names(fit))
  expect_length(fit$train_predictions, 50)
  expect_length(fit$sigma, 10)
  
  # Numerical checks
  expect_false(any(is.na(fit$train_predictions)))
  expect_false(any(is.nan(fit$train_predictions)))
  expect_true(all(is.finite(fit$train_predictions)))
  
  # Check posterior samples
  expect_true("train_predictions_sample" %in% names(fit))
  expect_true("test_predictions_sample" %in% names(fit))
  expect_equal(dim(fit$train_predictions_sample), c(10, 50))
  expect_equal(dim(fit$test_predictions_sample), c(10, 50))
  
  # Check sigma positivity
  expect_true(all(fit$sigma > 0))
  
  # Check predictions are not all constant
  expect_true(sd(fit$train_predictions) > 0)
  expect_true(sd(fit$train_predictions_sample) > 0)
  expect_true(sd(fit$test_predictions) > 0)
  expect_true(sd(fit$test_predictions_sample) > 0)
  
  # Check reproducibility
  set.seed(1)
  fit2 <- HorseTrees(y = time_obs, X_train = X, outcome_type = "right-censored",
                     status = status, number_of_trees = 5, N_post = 10,
                     N_burn = 5, store_posterior_sample = TRUE, verbose = FALSE)
  expect_equal(fit$train_predictions, fit2$train_predictions)
})


# ── S3 class and methods ──────────────────────────────────────────────────────

test_that("HorseTrees returns ShrinkageTrees S3 object with working methods", {
  X <- matrix(runif(50 * 3), ncol = 3)
  y <- X[, 1] + rnorm(50)

  set.seed(1)
  fit <- HorseTrees(
    y = y, X_train = X,
    outcome_type = "continuous",
    number_of_trees = 5,
    N_post = 10, N_burn = 5,
    store_posterior_sample = TRUE,
    verbose = FALSE
  )

  expect_s3_class(fit, "ShrinkageTrees")
  expect_no_error(capture.output(print(fit)))
  expect_no_error(smry <- summary(fit))
  expect_type(smry, "list")
  expect_true(!is.null(smry$sigma))
})


# ── Multi-chain (n_chains) ────────────────────────────────────────────────────

test_that("HorseTrees n_chains > 1 pools chains correctly", {
  X <- matrix(runif(50 * 3), ncol = 3)
  y <- X[, 1] + rnorm(50)

  set.seed(1)
  fit <- HorseTrees(
    y = y, X_train = X,
    outcome_type = "continuous",
    number_of_trees = 5,
    N_post = 10, N_burn = 5,
    n_chains = 2,
    verbose = FALSE
  )

  expect_s3_class(fit, "ShrinkageTrees")
  expect_length(fit$sigma, 20)
  expect_equal(fit$mcmc$n_chains, 2)
  expect_length(fit$chains$acceptance_ratios, 2)
})


# ── Interval-censored survival ───────────────────────────────────────────────

test_that("HorseTrees for interval-censored survival outcome", {

  n <- 50; p <- 3
  X <- matrix(rnorm(n * p), ncol = p)
  X_test <- matrix(rnorm(n * p), ncol = p)

  # Generate true event times
  true_time <- rexp(n, rate = exp(0.3 * X[, 1]))

  # Create mixed censoring:
  #   ~40% exact events, ~30% interval-censored, ~30% right-censored
  set.seed(42)
  obs_type <- sample(c("exact", "interval", "right"),
                     n, replace = TRUE, prob = c(0.4, 0.3, 0.3))

  left_time  <- numeric(n)
  right_time <- numeric(n)

  for (i in seq_len(n)) {
    if (obs_type[i] == "exact") {
      left_time[i]  <- true_time[i]
      right_time[i] <- true_time[i]
    } else if (obs_type[i] == "interval") {
      width <- runif(1, 0.1, 0.5) * true_time[i]
      left_time[i]  <- true_time[i] - width / 2
      right_time[i] <- true_time[i] + width / 2
    } else {
      left_time[i]  <- true_time[i] * runif(1, 0.3, 0.9)
      right_time[i] <- Inf
    }
  }

  # Fit the model
  set.seed(1)
  fit <- HorseTrees(left_time = left_time,
                    right_time = right_time,
                    X_train = X,
                    X_test = X_test,
                    outcome_type = "interval-censored",
                    number_of_trees = 5,
                    N_post = 10,
                    N_burn = 5,
                    store_posterior_sample = TRUE,
                    verbose = FALSE)

  # Basic checks
  expect_type(fit, "list")
  expect_s3_class(fit, "ShrinkageTrees")
  expect_true("train_predictions" %in% names(fit))
  expect_true("test_predictions" %in% names(fit))
  expect_length(fit$train_predictions, n)
  expect_length(fit$sigma, 10)

  # Predictions should be positive (back-transformed survival times)
  expect_true(all(fit$train_predictions > 0))
  expect_true(all(fit$test_predictions > 0))

  # Numerical checks
  expect_false(any(is.na(fit$train_predictions)))
  expect_false(any(is.nan(fit$train_predictions)))
  expect_true(all(is.finite(fit$train_predictions)))

  # Check posterior samples
  expect_true("train_predictions_sample" %in% names(fit))
  expect_true("test_predictions_sample" %in% names(fit))
  expect_equal(dim(fit$train_predictions_sample), c(10, n))
  expect_equal(dim(fit$test_predictions_sample), c(10, n))

  # Check sigma positivity
  expect_true(all(fit$sigma > 0))

  # Check predictions are not all constant
  expect_true(sd(fit$train_predictions) > 0)
  expect_true(sd(fit$test_predictions) > 0)

  # Check stored data fields
  expect_true(!is.null(fit$data$left_time_train))
  expect_true(!is.null(fit$data$right_time_train))
  expect_true(!is.null(fit$data$ic_indicator_train))
  expect_length(fit$data$left_time_train, n)
  expect_length(fit$data$right_time_train, n)
  expect_length(fit$data$ic_indicator_train, n)

  # Check print/summary work
  expect_no_error(capture.output(print(fit)))
  expect_no_error(smry <- summary(fit))
  expect_type(smry, "list")

  # Check reproducibility
  set.seed(1)
  fit2 <- HorseTrees(left_time = left_time,
                     right_time = right_time,
                     X_train = X,
                     X_test = X_test,
                     outcome_type = "interval-censored",
                     number_of_trees = 5,
                     N_post = 10,
                     N_burn = 5,
                     store_posterior_sample = TRUE,
                     verbose = FALSE)
  expect_equal(fit$train_predictions, fit2$train_predictions)
})


test_that("HorseTrees interval-censored predict on new data works", {

  n <- 50; p <- 3
  X <- matrix(rnorm(n * p), ncol = p)

  true_time <- rexp(n, rate = exp(0.3 * X[, 1]))
  set.seed(42)
  obs_type <- sample(c("exact", "interval", "right"),
                     n, replace = TRUE, prob = c(0.4, 0.3, 0.3))

  left_time  <- numeric(n)
  right_time <- numeric(n)
  for (i in seq_len(n)) {
    if (obs_type[i] == "exact") {
      left_time[i] <- true_time[i]; right_time[i] <- true_time[i]
    } else if (obs_type[i] == "interval") {
      w <- runif(1, 0.1, 0.5) * true_time[i]
      left_time[i] <- true_time[i] - w / 2
      right_time[i] <- true_time[i] + w / 2
    } else {
      left_time[i] <- true_time[i] * runif(1, 0.3, 0.9)
      right_time[i] <- Inf
    }
  }

  set.seed(1)
  fit <- HorseTrees(left_time = left_time,
                    right_time = right_time,
                    X_train = X,
                    outcome_type = "interval-censored",
                    number_of_trees = 5,
                    N_post = 10, N_burn = 5,
                    store_posterior_sample = TRUE,
                    verbose = FALSE)

  # Predict on new data
  X_new <- matrix(rnorm(20 * p), ncol = p)
  pred <- predict(fit, newdata = X_new)

  expect_true(all(pred$mean > 0))
  expect_true(all(is.finite(pred$mean)))
  expect_length(pred$mean, 20)
  expect_no_error(capture.output(print(pred)))
})


test_that("HorseTrees interval-censored validation errors", {

  n <- 50; p <- 3
  X <- matrix(rnorm(n * p), ncol = p)
  lt <- rexp(n); rt <- lt + runif(n, 0.1, 1)

  # Missing right_time
  expect_error(
    HorseTrees(left_time = lt, X_train = X,
               outcome_type = "interval-censored",
               number_of_trees = 5, N_post = 5, N_burn = 5, verbose = FALSE),
    "left_time.*right_time"
  )

  # Missing left_time
  expect_error(
    HorseTrees(right_time = rt, X_train = X,
               outcome_type = "interval-censored",
               number_of_trees = 5, N_post = 5, N_burn = 5, verbose = FALSE),
    "left_time.*right_time"
  )

  # left > right
  bad_lt <- rt + 1
  expect_error(
    HorseTrees(left_time = bad_lt, right_time = rt, X_train = X,
               outcome_type = "interval-censored",
               number_of_trees = 5, N_post = 5, N_burn = 5, verbose = FALSE),
    "left_time.*right_time"
  )

  # Negative values with timescale = "time"
  neg_lt <- lt; neg_lt[1] <- -1
  expect_error(
    HorseTrees(left_time = neg_lt, right_time = rt, X_train = X,
               outcome_type = "interval-censored",
               number_of_trees = 5, N_post = 5, N_burn = 5, verbose = FALSE),
    "positive"
  )
})

