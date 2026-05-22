# ── Helper: generate interval-censored data ──────────────────────────────────

generate_ic_data <- function(n = 50, p = 3) {
  X <- matrix(rnorm(n * p), ncol = p)
  true_time <- rexp(n, rate = exp(0.3 * X[, 1]))

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
      w <- runif(1, 0.1, 0.5) * true_time[i]
      left_time[i]  <- true_time[i] - w / 2
      right_time[i] <- true_time[i] + w / 2
    } else {
      left_time[i]  <- true_time[i] * runif(1, 0.3, 0.9)
      right_time[i] <- Inf
    }
  }

  list(X = X, left_time = left_time, right_time = right_time,
       true_time = true_time, obs_type = obs_type)
}


# ── ShrinkageTrees with interval-censored survival ──────────────────────────

test_that("ShrinkageTrees works for interval-censored survival", {
  dat <- generate_ic_data(n = 50, p = 3)

  set.seed(1)
  fit <- ShrinkageTrees(
    left_time = dat$left_time,
    right_time = dat$right_time,
    X_train = dat$X,
    outcome_type = "interval-censored",
    number_of_trees = 5,
    prior_type = "horseshoe",
    local_hp = 0.1 / sqrt(5),
    global_hp = 0.1 / sqrt(5),
    N_post = 10, N_burn = 5,
    store_posterior_sample = TRUE,
    verbose = FALSE
  )

  expect_s3_class(fit, "ShrinkageTrees")
  expect_length(fit$train_predictions, 50)
  expect_length(fit$sigma, 10)
  expect_true(all(fit$train_predictions > 0))
  expect_true(all(fit$sigma > 0))
  expect_false(any(is.na(fit$train_predictions)))

  # Posterior samples
  expect_equal(dim(fit$train_predictions_sample), c(10, 50))

  # Stored interval data
  expect_length(fit$data$left_time_train, 50)
  expect_length(fit$data$right_time_train, 50)
  expect_length(fit$data$ic_indicator_train, 50)

  # S3 methods
  expect_no_error(capture.output(print(fit)))
  expect_no_error(smry <- summary(fit))
  expect_type(smry, "list")
})


test_that("ShrinkageTrees interval-censored predict works", {
  dat <- generate_ic_data(n = 50, p = 3)

  set.seed(1)
  fit <- ShrinkageTrees(
    left_time = dat$left_time,
    right_time = dat$right_time,
    X_train = dat$X,
    outcome_type = "interval-censored",
    number_of_trees = 5,
    prior_type = "horseshoe",
    local_hp = 0.1 / sqrt(5),
    global_hp = 0.1 / sqrt(5),
    N_post = 10, N_burn = 5,
    store_posterior_sample = TRUE,
    verbose = FALSE
  )

  X_new <- matrix(rnorm(20 * 3), ncol = 3)
  pred <- predict(fit, newdata = X_new)

  expect_s3_class(pred, "ShrinkageTreesPrediction")
  expect_length(pred$mean, 20)
  expect_true(all(pred$mean > 0))
  expect_true(all(is.finite(pred$mean)))
  expect_true(all(pred$lower <= pred$upper))
  expect_true(!is.null(pred$predictions_sample))
  expect_true(!is.null(pred$sigma))
})


# ── CausalShrinkageForest with interval-censored survival ───────────────────

test_that("CausalShrinkageForest works for interval-censored survival", {
  n <- 50; p <- 3
  X <- matrix(rnorm(n * p), ncol = p)
  treatment <- rbinom(n, 1, 0.5)

  true_time <- rexp(n, rate = exp(0.3 * X[, 1] + treatment * (-0.3)))

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
      w <- runif(1, 0.1, 0.5) * true_time[i]
      left_time[i]  <- true_time[i] - w / 2
      right_time[i] <- true_time[i] + w / 2
    } else {
      left_time[i]  <- true_time[i] * runif(1, 0.3, 0.9)
      right_time[i] <- Inf
    }
  }

  lh <- 0.1 / sqrt(5)

  set.seed(1)
  fit <- CausalShrinkageForest(
    left_time = left_time,
    right_time = right_time,
    X_train_control = X, X_train_treat = X,
    treatment_indicator_train = treatment,
    outcome_type = "interval-censored",
    number_of_trees_control = 5, number_of_trees_treat = 5,
    prior_type_control = "horseshoe", prior_type_treat = "horseshoe",
    local_hp_control = lh, global_hp_control = lh,
    local_hp_treat = lh, global_hp_treat = lh,
    N_post = 10, N_burn = 5,
    store_posterior_sample = TRUE,
    verbose = FALSE
  )

  expect_s3_class(fit, "CausalShrinkageForest")
  expect_length(fit$train_predictions, n)
  expect_length(fit$sigma, 10)
  expect_true(all(fit$sigma > 0))
  expect_false(any(is.na(fit$train_predictions)))
  expect_true(all(is.finite(fit$train_predictions)))

  # CATE should be available
  expect_length(fit$train_predictions_treat, n)
  expect_length(fit$train_predictions_control, n)

  # S3 methods
  expect_no_error(capture.output(print(fit)))
  expect_no_error(smry <- summary(fit))
  expect_type(smry, "list")
})


test_that("CausalHorseForest works for interval-censored survival", {
  n <- 50; p <- 3
  X <- matrix(rnorm(n * p), ncol = p)
  treatment <- rbinom(n, 1, 0.5)

  true_time <- rexp(n, rate = exp(0.3 * X[, 1] + treatment * (-0.3)))

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
      w <- runif(1, 0.1, 0.5) * true_time[i]
      left_time[i]  <- true_time[i] - w / 2
      right_time[i] <- true_time[i] + w / 2
    } else {
      left_time[i]  <- true_time[i] * runif(1, 0.3, 0.9)
      right_time[i] <- Inf
    }
  }

  set.seed(1)
  fit <- CausalHorseForest(
    left_time = left_time,
    right_time = right_time,
    X_train_control = X, X_train_treat = X,
    treatment_indicator_train = treatment,
    outcome_type = "interval-censored",
    number_of_trees = 5,
    N_post = 10, N_burn = 5,
    store_posterior_sample = TRUE,
    verbose = FALSE
  )

  expect_s3_class(fit, "CausalShrinkageForest")
  expect_length(fit$train_predictions, n)
  expect_length(fit$sigma, 10)
  expect_true(all(fit$sigma > 0))
  expect_false(any(is.na(fit$train_predictions)))
  expect_true(all(is.finite(fit$train_predictions)))

  # S3 methods
  expect_no_error(capture.output(print(fit)))
  expect_no_error(smry <- summary(fit))
})

