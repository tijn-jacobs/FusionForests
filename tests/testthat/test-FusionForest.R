set.seed(1)

# Minimal shared data (small n so tests run fast)
n     <- 60
p     <- 3
X     <- matrix(rnorm(n * p), n, p)
trt   <- as.integer(rbinom(n, 1, 0.5))
src   <- as.integer(c(rep(1L, 30), rep(0L, 30)))   # first 30 RCT, last 30 RWD
y_cnt <- 1 + X[, 1] + trt * X[, 2] + rnorm(n)
y_srv <- exp(y_cnt)
stat  <- as.integer(rbinom(n, 1, 0.8))

# Small MCMC dimensions to keep runtime short
N_post <- 50L
N_burn <- 20L

# ---------------------------------------------------------------------------
test_that("FusionForest runs on continuous outcomes and returns named list", {
  fit <- FusionForest(
    y                         = y_cnt,
    X_train_control           = X,
    X_train_treat             = X,
    treatment_indicator_train = trt,
    source_indicator_train    = src,
    N_post = N_post, N_burn = N_burn, verbose = FALSE
  )

  expect_type(fit, "list")
  expect_true("train_predictions"         %in% names(fit))
  expect_true("train_predictions_control" %in% names(fit))
  expect_true("train_predictions_treat"   %in% names(fit))
  expect_true("train_predictions_deconf"  %in% names(fit))
  expect_true("sigma"                     %in% names(fit))
})

test_that("FusionForest train_predictions has length n", {
  fit <- FusionForest(
    y                         = y_cnt,
    X_train_control           = X,
    X_train_treat             = X,
    treatment_indicator_train = trt,
    source_indicator_train    = src,
    N_post = N_post, N_burn = N_burn, verbose = FALSE
  )

  expect_length(fit$train_predictions, n)
  expect_true(all(is.finite(fit$train_predictions)))
})

test_that("FusionForest runs on right-censored survival outcomes", {
  fit <- FusionForest(
    y                         = y_srv,
    status                    = stat,
    X_train_control           = X,
    X_train_treat             = X,
    treatment_indicator_train = trt,
    source_indicator_train    = src,
    outcome_type              = "right-censored",
    N_post = N_post, N_burn = N_burn, verbose = FALSE
  )

  expect_type(fit, "list")
  expect_length(fit$train_predictions, n)
  expect_true(all(is.finite(fit$train_predictions)))
  expect_true(all(fit$train_predictions > 0))   # back-transformed to time scale
})

test_that("FusionForest returns test predictions when test data supplied", {
  n_test <- 10L
  X_test <- matrix(rnorm(n_test * p), n_test, p)

  fit <- FusionForest(
    y                         = y_cnt,
    X_train_control           = X,
    X_train_treat             = X,
    treatment_indicator_train = trt,
    source_indicator_train    = src,
    X_test_control            = X_test,
    X_test_treat              = X_test,
    treatment_indicator_test  = rep(1L, n_test),
    source_indicator_test     = rep(1L, n_test),
    N_post = N_post, N_burn = N_burn, verbose = FALSE
  )

  expect_length(fit$test_predictions, n_test)
  expect_true(all(is.finite(fit$test_predictions)))
})

test_that("FusionForest posterior_sample matrices present when requested", {
  fit <- FusionForest(
    y                         = y_cnt,
    X_train_control           = X,
    X_train_treat             = X,
    treatment_indicator_train = trt,
    source_indicator_train    = src,
    store_posterior_sample    = TRUE,
    N_post = N_post, N_burn = N_burn, verbose = FALSE
  )

  expect_true("train_predictions_sample_control" %in% names(fit))
  expect_equal(nrow(fit$train_predictions_sample_control), N_post)
  expect_equal(ncol(fit$train_predictions_sample_control), n)
})

test_that("FusionForest errors on invalid outcome_type", {
  expect_error(
    FusionForest(
      y                         = y_cnt,
      X_train_control           = X,
      X_train_treat             = X,
      treatment_indicator_train = trt,
      source_indicator_train    = src,
      outcome_type              = "wrong"
    ),
    regexp = "outcome_type"
  )
})

test_that("FusionForest errors when no RWD rows present", {
  src_rct_only <- rep(1L, n)
  expect_error(
    FusionForest(
      y                         = y_cnt,
      X_train_control           = X,
      X_train_treat             = X,
      treatment_indicator_train = trt,
      source_indicator_train    = src_rct_only
    ),
    regexp = "real-world-data"
  )
})

# ---------------------------------------------------------------------------
# Four-forest decomposition tests
# ---------------------------------------------------------------------------

test_that("FusionForest four-forest runs on continuous outcomes", {
  fit <- FusionForest(
    y                         = y_cnt,
    X_train_control           = X,
    X_train_treat             = X,
    treatment_indicator_train = trt,
    source_indicator_train    = src,
    decomposition             = "four-forest",
    N_post = N_post, N_burn = N_burn, verbose = FALSE
  )

  expect_type(fit, "list")
  expect_true("train_predictions"           %in% names(fit))
  expect_true("train_predictions_control"   %in% names(fit))
  expect_true("train_predictions_treat"     %in% names(fit))
  expect_true("train_predictions_deconf"    %in% names(fit))
  expect_true("train_predictions_deviation" %in% names(fit))
  expect_true("sigma"                       %in% names(fit))
  expect_true("acceptance_ratio_deviation"  %in% names(fit))
  expect_length(fit$train_predictions, n)
  expect_true(all(is.finite(fit$train_predictions)))
})

test_that("FusionForest four-forest deviation length equals n_deconf", {
  n_rwd <- sum(src == 0L)
  fit <- FusionForest(
    y                         = y_cnt,
    X_train_control           = X,
    X_train_treat             = X,
    treatment_indicator_train = trt,
    source_indicator_train    = src,
    decomposition             = "four-forest",
    N_post = N_post, N_burn = N_burn, verbose = FALSE
  )

  expect_length(fit$train_predictions_deviation, n_rwd)
  expect_true(all(is.finite(fit$train_predictions_deviation)))
})

test_that("FusionForest four-forest runs on right-censored outcomes", {
  fit <- FusionForest(
    y                         = y_srv,
    status                    = stat,
    X_train_control           = X,
    X_train_treat             = X,
    treatment_indicator_train = trt,
    source_indicator_train    = src,
    outcome_type              = "right-censored",
    decomposition             = "four-forest",
    N_post = N_post, N_burn = N_burn, verbose = FALSE
  )

  expect_type(fit, "list")
  expect_length(fit$train_predictions, n)
  expect_true(all(is.finite(fit$train_predictions)))
  expect_true(all(fit$train_predictions > 0))
  expect_true("train_predictions_deviation" %in% names(fit))
})

test_that("FusionForest three-forest still works with explicit arg", {
  fit <- FusionForest(
    y                         = y_cnt,
    X_train_control           = X,
    X_train_treat             = X,
    treatment_indicator_train = trt,
    source_indicator_train    = src,
    decomposition             = "three-forest",
    N_post = N_post, N_burn = N_burn, verbose = FALSE
  )

  expect_type(fit, "list")
  expect_length(fit$train_predictions, n)
  expect_true(all(is.finite(fit$train_predictions)))
  expect_false("train_predictions_deviation" %in% names(fit))
})

test_that("FusionForest four-forest posterior samples when requested", {
  n_rwd <- sum(src == 0L)
  fit <- FusionForest(
    y                         = y_cnt,
    X_train_control           = X,
    X_train_treat             = X,
    treatment_indicator_train = trt,
    source_indicator_train    = src,
    decomposition             = "four-forest",
    store_posterior_sample     = TRUE,
    N_post = N_post, N_burn = N_burn, verbose = FALSE
  )

  expect_true("train_predictions_sample_deviation" %in% names(fit))
  expect_equal(nrow(fit$train_predictions_sample_deviation), N_post)
  expect_equal(ncol(fit$train_predictions_sample_deviation), n_rwd)
})

test_that("FusionForest errors on invalid decomposition", {
  expect_error(
    FusionForest(
      y                         = y_cnt,
      X_train_control           = X,
      X_train_treat             = X,
      treatment_indicator_train = trt,
      source_indicator_train    = src,
      decomposition             = "five-forest"
    ),
    regexp = "decomposition"
  )
})
