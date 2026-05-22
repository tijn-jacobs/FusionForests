# Tests for plot.ShrinkageTrees and plot.CausalShrinkageForest.
# All tests are skipped when ggplot2 is not installed.

# -- Helper -------------------------------------------------------------------

.skip_plots <- function() {
  skip_if_not_installed("ggplot2")
}

# Helper to build a tiny CausalShrinkageForest fit (horseshoe priors).
.causal_fit <- function(n_chains = 1L, store = TRUE) {
  n <- 30L; p <- 3L
  set.seed(1)
  X         <- matrix(runif(n * p), ncol = p)
  treatment <- rbinom(n, 1, 0.5)
  y         <- X[, 1] + (0.5 - treatment) * 2 + rnorm(n)
  lh        <- 0.1 / sqrt(5)
  CausalShrinkageForest(
    y = y, X_train_control = X, X_train_treat = X,
    treatment_indicator_train = treatment,
    outcome_type = "continuous",
    number_of_trees_control = 5, number_of_trees_treat = 5,
    prior_type_control = "horseshoe", prior_type_treat = "horseshoe",
    local_hp_control = lh, global_hp_control = lh,
    local_hp_treat   = lh, global_hp_treat   = lh,
    N_post = 10, N_burn = 5,
    store_posterior_sample = store,
    n_chains = n_chains,
    verbose = FALSE
  )
}


# -- plot.ShrinkageTrees: trace -----------------------------------------------

test_that("plot.ShrinkageTrees type='trace' returns a ggplot", {
  .skip_plots()
  n <- 30L; p <- 3L
  set.seed(1)
  X <- matrix(runif(n * p), ncol = p)
  y <- X[, 1] + rnorm(n)
  fit <- HorseTrees(y, X_train = X, outcome_type = "continuous",
                    number_of_trees = 5, N_post = 10, N_burn = 5,
                    verbose = FALSE)
  out <- expect_no_error(plot(fit, type = "trace"))
  expect_s3_class(out, "gg")
})


# -- plot.ShrinkageTrees: density ---------------------------------------------

test_that("plot.ShrinkageTrees type='density' works with n_chains > 1", {
  .skip_plots()
  n <- 30L; p <- 3L
  set.seed(1)
  X <- matrix(runif(n * p), ncol = p)
  y <- X[, 1] + rnorm(n)
  fit <- HorseTrees(y, X_train = X, outcome_type = "continuous",
                    number_of_trees = 5, N_post = 10, N_burn = 5,
                    n_chains = 2, verbose = FALSE)
  out <- expect_no_error(plot(fit, type = "density"))
  expect_s3_class(out, "gg")
})

test_that("plot.ShrinkageTrees type='density' works with n_chains == 1", {
  .skip_plots()
  n <- 30L; p <- 3L
  set.seed(1)
  X <- matrix(runif(n * p), ncol = p)
  y <- X[, 1] + rnorm(n)
  fit <- HorseTrees(y, X_train = X, outcome_type = "continuous",
                    number_of_trees = 5, N_post = 10, N_burn = 5,
                    verbose = FALSE)
  out <- expect_no_error(plot(fit, type = "density"))
  expect_s3_class(out, "gg")
})


# -- plot.ShrinkageTrees: vi --------------------------------------------------

test_that("plot.ShrinkageTrees type='vi' works for Dirichlet model", {
  .skip_plots()
  n <- 30L; p <- 3L
  set.seed(1)
  X <- matrix(runif(n * p), ncol = p)
  y <- X[, 1] + rnorm(n)
  fit <- ShrinkageTrees(y, X_train = X, outcome_type = "continuous",
                        number_of_trees = 5, prior_type = "dirichlet",
                        local_hp = 1 / sqrt(5),
                        N_post = 10, N_burn = 5, verbose = FALSE)
  out <- expect_no_error(plot(fit, type = "vi", n_vi = 3))
  expect_s3_class(out, "gg")
})

test_that("plot.ShrinkageTrees type='vi' errors for non-Dirichlet model", {
  .skip_plots()
  n <- 30L; p <- 3L
  set.seed(1)
  X <- matrix(runif(n * p), ncol = p)
  y <- X[, 1] + rnorm(n)
  fit <- HorseTrees(y, X_train = X, outcome_type = "continuous",
                    number_of_trees = 5, N_post = 10, N_burn = 5,
                    verbose = FALSE)
  expect_error(plot(fit, type = "vi"), "split_probs is NULL")
})


# -- plot.CausalShrinkageForest: trace ----------------------------------------

test_that("plot.CausalShrinkageForest type='trace' returns a ggplot", {
  .skip_plots()
  fit <- .causal_fit()
  out <- expect_no_error(plot(fit, type = "trace"))
  expect_s3_class(out, "gg")
})


# -- plot.CausalShrinkageForest: density --------------------------------------

test_that("plot.CausalShrinkageForest type='density' works with n_chains > 1", {
  .skip_plots()
  fit <- .causal_fit(n_chains = 2L)
  out <- expect_no_error(plot(fit, type = "density"))
  expect_s3_class(out, "gg")
})

test_that("plot.CausalShrinkageForest type='density' works with n_chains == 1", {
  .skip_plots()
  fit <- .causal_fit()
  out <- expect_no_error(plot(fit, type = "density"))
  expect_s3_class(out, "gg")
})


# -- plot.CausalShrinkageForest: ate ------------------------------------------

test_that("plot.CausalShrinkageForest type='ate' returns a ggplot", {
  .skip_plots()
  fit <- .causal_fit(store = TRUE)
  out <- expect_no_error(plot(fit, type = "ate"))
  expect_s3_class(out, "gg")
})

test_that("plot.CausalShrinkageForest type='ate' errors without posterior samples", {
  .skip_plots()
  fit <- .causal_fit(store = FALSE)
  expect_error(plot(fit, type = "ate"), "store_posterior_sample")
})


# -- plot.CausalShrinkageForest: cate -----------------------------------------

test_that("plot.CausalShrinkageForest type='cate' returns a ggplot", {
  .skip_plots()
  fit <- .causal_fit(store = TRUE)
  out <- expect_no_error(plot(fit, type = "cate"))
  expect_s3_class(out, "gg")
})

test_that("plot.CausalShrinkageForest type='cate' errors without posterior samples", {
  .skip_plots()
  fit <- .causal_fit(store = FALSE)
  expect_error(plot(fit, type = "cate"), "store_posterior_sample")
})


# -- plot.CausalShrinkageForest: vi -------------------------------------------

test_that("plot.CausalShrinkageForest type='vi' works for Dirichlet model", {
  .skip_plots()
  n <- 30L; p <- 3L
  set.seed(1)
  X         <- matrix(runif(n * p), ncol = p)
  treatment <- rbinom(n, 1, 0.5)
  y         <- X[, 1] + (0.5 - treatment) * 2 + rnorm(n)
  lh        <- 1 / sqrt(5)
  fit <- CausalShrinkageForest(
    y = y, X_train_control = X, X_train_treat = X,
    treatment_indicator_train = treatment,
    outcome_type = "continuous",
    number_of_trees_control = 5, number_of_trees_treat = 5,
    prior_type_control = "dirichlet", prior_type_treat = "dirichlet",
    local_hp_control = lh, local_hp_treat = lh,
    N_post = 10, N_burn = 5, verbose = FALSE
  )

  # forest = "both" returns a named list of two ggplots
  out_both <- expect_no_error(plot(fit, type = "vi", forest = "both", n_vi = 3))
  expect_type(out_both, "list")
  expect_s3_class(out_both$control, "gg")
  expect_s3_class(out_both$treat,   "gg")

  # forest = "control" returns a single ggplot
  out_c <- expect_no_error(plot(fit, type = "vi", forest = "control", n_vi = 3))
  expect_s3_class(out_c, "gg")

  # forest = "treat" returns a single ggplot
  out_t <- expect_no_error(plot(fit, type = "vi", forest = "treat", n_vi = 3))
  expect_s3_class(out_t, "gg")
})

test_that("plot.CausalShrinkageForest type='vi' errors for non-Dirichlet model", {
  .skip_plots()
  fit <- .causal_fit()
  expect_error(plot(fit, type = "vi"), "not available")
})


# -- plot.ShrinkageTreesPrediction: survival -----------------------------------

test_that("plot.ShrinkageTreesPrediction type='survival' returns a ggplot", {
  .skip_plots()
  set.seed(1)
  n <- 30L; p <- 3L
  X      <- matrix(runif(n * p), ncol = p)
  X_test <- matrix(runif(10 * p), ncol = p)
  time   <- rexp(n, rate = exp(0.5 * X[, 1]))
  status <- rbinom(n, 1, 0.7)

  fit <- HorseTrees(
    y = time, status = status, X_train = X, X_test = X_test,
    outcome_type = "right-censored",
    number_of_trees = 5, N_post = 10, N_burn = 5,
    store_posterior_sample = TRUE, verbose = FALSE
  )
  pred <- predict(fit, newdata = X_test)

  # Population-averaged
  out <- expect_no_error(plot(pred, type = "survival"))
  expect_s3_class(out, "gg")

  # Individual curves
  out2 <- expect_no_error(plot(pred, type = "survival", obs = c(1, 3)))
  expect_s3_class(out2, "gg")
})

