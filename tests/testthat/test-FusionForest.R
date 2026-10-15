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
# Four-forest model tests
# ---------------------------------------------------------------------------

test_that("FusionForest four-forest runs on continuous outcomes", {
  fit <- FusionForest(
    y                         = y_cnt,
    X_train_control           = X,
    X_train_treat             = X,
    treatment_indicator_train = trt,
    source_indicator_train    = src,
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
    N_post = N_post, N_burn = N_burn, verbose = FALSE
  )

  expect_type(fit, "list")
  expect_length(fit$train_predictions, n)
  expect_true(all(is.finite(fit$train_predictions)))
  expect_true(all(fit$train_predictions > 0))
  expect_true("train_predictions_deviation" %in% names(fit))
})

test_that("FusionForest four-forest posterior samples when requested", {
  n_rwd <- sum(src == 0L)
  fit <- FusionForest(
    y                         = y_cnt,
    X_train_control           = X,
    X_train_treat             = X,
    treatment_indicator_train = trt,
    source_indicator_train    = src,
    store_posterior_sample     = TRUE,
    N_post = N_post, N_burn = N_burn, verbose = FALSE
  )

  expect_true("train_predictions_sample_deviation" %in% names(fit))
  expect_equal(nrow(fit$train_predictions_sample_deviation), N_post)
  expect_equal(ncol(fit$train_predictions_sample_deviation), n_rwd)
})

test_that("FusionForest errors on missing covariate values", {
  X_na <- X
  X_na[1, 2] <- NA
  expect_error(
    FusionForest(
      y                         = y_cnt,
      X_train_control           = X_na,
      X_train_treat             = X,
      treatment_indicator_train = trt,
      source_indicator_train    = src
    ),
    regexp = "missing values"
  )
})

test_that("FusionForest defaults match the paper", {
  f <- formals(FusionForest)
  expect_equal(f$k_control, 1)
  expect_equal(f$k_deviation, 1)
  expect_equal(f$k_deconf, 1)
  expect_equal(f$k_treat, 0.5)
  expect_equal(eval(f$treatment_coding)[1], "centered")
  expect_equal(eval(f$error_dist)[1], "source_hdp_scale")
  expect_equal(f$error_truncation_K, 50L)
  expect_equal(c(f$number_of_trees_control, f$number_of_trees_deviation,
                 f$number_of_trees_treat, f$number_of_trees_deconf),
               c(200, 50, 100, 50))
  expect_equal(c(f$base_control, f$power_control, f$base_deviation,
                 f$power_deviation, f$base_treat, f$power_treat,
                 f$base_deconf, f$power_deconf),
               c(0.95, 2, 0.95, 2, 0.95, 3, 0.25, 3))
  expect_equal(f$nu, 3)
})

# ---------------------------------------------------------------------------
# Reproducibility, censoring and input validation
# ---------------------------------------------------------------------------

test_that("FusionForest is reproducible under a fixed seed", {
  fit_once <- function() {
    set.seed(123)
    FusionForest(
      y                         = y_srv,
      status                    = stat,
      X_train_control           = X,
      X_train_treat             = X,
      treatment_indicator_train = trt,
      source_indicator_train    = src,
      outcome_type              = "right-censored",
      store_posterior_sample    = TRUE,
      N_post = N_post, N_burn = N_burn, verbose = FALSE
    )
  }
  expect_identical(fit_once(), fit_once())
})

test_that("FusionForest handles interval-censored outcomes", {
  # Exact events, right-censored rows and interval-censored rows.
  ic    <- as.integer(stat == 0 & seq_len(n) %% 2 == 0)
  left  <- ifelse(ic == 1, y_srv * 0.7, y_srv)
  right <- ifelse(ic == 1, y_srv * 1.3, y_srv)
  fit <- FusionForest(
    y                            = y_srv,
    status                       = stat,
    observed_left_time           = left,
    observed_right_time          = right,
    interval_censoring_indicator = ic,
    X_train_control              = X,
    X_train_treat                = X,
    treatment_indicator_train    = trt,
    source_indicator_train       = src,
    outcome_type                 = "right-censored",
    N_post = N_post, N_burn = N_burn, verbose = FALSE
  )
  expect_length(fit$train_predictions, n)
  expect_true(all(is.finite(fit$train_predictions)))
  expect_true(all(fit$train_predictions > 0))
})

test_that("FusionForest validates the interval-censoring bounds", {
  ic <- as.integer(stat == 0)
  expect_error(
    FusionForest(
      y = y_srv, status = stat,
      observed_left_time = y_srv,
      X_train_control = X, X_train_treat = X,
      treatment_indicator_train = trt, source_indicator_train = src,
      outcome_type = "right-censored", verbose = FALSE
    ),
    regexp = "all three must be"
  )
  expect_error(
    FusionForest(
      y = y_srv, status = stat,
      observed_left_time = y_srv, observed_right_time = y_srv,
      interval_censoring_indicator = ic,
      X_train_control = X, X_train_treat = X,
      treatment_indicator_train = trt, source_indicator_train = src,
      outcome_type = "right-censored", verbose = FALSE
    ),
    regexp = "observed_left_time < observed_right_time"
  )
})

test_that("FusionForest validates its inputs", {
  fit_with <- function(...) {
    args <- list(y = y_cnt, X_train_control = X, X_train_treat = X,
                 treatment_indicator_train = trt,
                 source_indicator_train = src, verbose = FALSE)
    args[names(list(...))] <- list(...)
    do.call(FusionForest, args)
  }
  expect_error(fit_with(outcome_type = "binary"), "Invalid outcome_type")
  expect_error(fit_with(y = y_srv, outcome_type = "right-censored"),
               "requires a 'status' vector")
  expect_error(fit_with(y = -y_srv, status = stat,
                        outcome_type = "right-censored"),
               "Negative values in y")
  expect_error(fit_with(source_indicator_train = src + 1L),
               "must be 0 \\(RWD\\) or 1 \\(RCT\\)")
  expect_error(fit_with(treatment_indicator_train = trt[-1]),
               "must match length\\(y\\)")
  expect_error(fit_with(treatment_coding = "adaptive"),
               "requires propensity_train")
})

test_that("FusionForest runs under every error distribution", {
  for (ed in c("gaussian", "shared_dp", "source_dp", "source_dp_scale",
               "source_hdp", "source_hdp_scale")) {
    fit <- FusionForest(
      y                         = y_srv,
      status                    = stat,
      X_train_control           = X,
      X_train_treat             = X,
      treatment_indicator_train = trt,
      source_indicator_train    = src,
      outcome_type              = "right-censored",
      error_dist                = ed,
      N_post = N_post, N_burn = N_burn, verbose = FALSE
    )
    expect_true(all(is.finite(fit$train_predictions)), info = ed)
    expect_true(all(is.finite(fit$sigma) & fit$sigma > 0), info = ed)
    expect_identical(fit$meta$error_dist, ed)
    if (ed != "gaussian")
      expect_false(is.null(fit$dp_mix_prop), info = ed)
    if (ed %in% c("source_dp_scale", "source_hdp_scale"))
      expect_length(fit$dp_sigma_g, 2L)
  }
})

test_that("k_deviation = 0 switches off the deviation forest", {
  fit <- FusionForest(
    y                         = y_cnt,
    X_train_control           = X,
    X_train_treat             = X,
    treatment_indicator_train = trt,
    source_indicator_train    = src,
    k_deviation               = 0,
    N_post = N_post, N_burn = N_burn, verbose = FALSE
  )
  expect_true(all(abs(fit$train_predictions_deviation) < 1e-6))
})
