set.seed(1)

# Minimal shared data (small n so tests run fast)
n     <- 60
p     <- 3
X     <- matrix(rnorm(n * p), n, p)
trt   <- as.integer(rbinom(n, 1, 0.5))
src   <- as.integer(c(rep(1L, 30), rep(0L, 30)))   # first 30 RCT, last 30 RWD
y_cnt <- 1 + X[, 1] + trt * X[, 2] + rnorm(n)

N_post <- 50L
N_burn <- 20L

dp_modes <- c("shared_dp", "source_dp", "source_dp_scale",
              "source_hdp", "source_hdp_scale")

fit_small <- function(...) {
  FusionForest(
    y                         = y_cnt,
    X_train_control           = X,
    X_train_treat             = X,
    treatment_indicator_train = trt,
    source_indicator_train    = src,
    N_post = N_post, N_burn = N_burn, verbose = FALSE, ...
  )
}

test_that("the stored error mixture has mean zero in every source", {
  for (ed in dp_modes) {
    set.seed(1)
    fit <- fit_small(error_dist = ed)
    expect_length(fit$dp_mu_g, length(fit$dp_locations))
    for (g in seq_along(fit$dp_locations)) {
      w   <- fit$dp_mix_prop[[g]]
      th  <- fit$dp_locations[[g]]
      expect_true(all(abs(rowSums(w) - 1) < 1e-10), info = ed)
      expect_true(all(abs(rowSums(w * th)) < 1e-8), info = ed)
    }
    if (ed %in% c("source_hdp", "source_hdp_scale")) {
      for (g in 1:2) {
        expect_equal(fit$dp_locations[[g]],
                     fit$dp_locations_shared - fit$dp_mu_g[[g]],
                     tolerance = 1e-10)
        expect_equal(fit$dp_mu_g[[g]],
                     rowSums(fit$dp_mix_prop[[g]] * fit$dp_locations_shared),
                     tolerance = 1e-10)
      }
    }
  }
})

test_that("with k_deviation = 0, d equals the difference of the error means", {
  for (ed in c("source_dp", "source_hdp_scale")) {
    set.seed(1)
    fit <- fit_small(error_dist = ed, k_deviation = 0,
                     store_posterior_sample = TRUE)
    d_const <- fit$dp_mu_g[[1]] - fit$dp_mu_g[[2]]   # RWD minus RCT
    dev     <- fit$train_predictions_sample_deviation
    expect_true(all(abs(sweep(dev, 1, d_const)) < 1e-5), info = ed)
    expect_equal(fit$train_predictions_deviation,
                 rep(mean(d_const), sum(src == 0)), tolerance = 1e-6)
  }
  set.seed(1)
  fit <- fit_small(error_dist = "shared_dp", k_deviation = 0)
  expect_true(all(abs(fit$train_predictions_deviation) < 1e-5))
})

test_that("the stored components compose to the fitted values", {
  set.seed(1)
  fit <- fit_small(store_posterior_sample = TRUE)
  b   <- trt - 0.5
  s   <- src
  dev <- rep(0, n); dev[s == 0] <- fit$train_predictions_deviation
  dec <- rep(0, n); dec[s == 0] <- fit$train_predictions_deconf
  composed <- fit$train_predictions_control + (1 - s) * dev +
    b * (fit$train_predictions_treat + (1 - s) * dec)
  expect_equal(composed, fit$train_predictions, tolerance = 1e-8)
  expect_equal(colMeans(fit$train_predictions_sample_control),
               fit$train_predictions_control, tolerance = 1e-8)
})

test_that("the baseline is recovered under a skewed mean-zero error", {
  # The error has mean zero but a mean of the fitted working mixture that is
  # well away from zero, so an omitted or wrong shift of the baseline would
  # show up as a bias of train_predictions_control.
  set.seed(1)
  n_r <- 300L
  X_r <- matrix(runif(n_r * 2), n_r, 2)
  a_r <- rbinom(n_r, 1, 0.5)
  s_r <- rep(c(1L, 0L), each = n_r / 2)
  # 0.8 N(-0.5, 0.2^2) + 0.2 N(2, 0.2^2): mean zero, bimodal
  eps <- ifelse(runif(n_r) < 0.8, rnorm(n_r, -0.5, 0.2), rnorm(n_r, 2.0, 0.2))
  m0  <- 1 + X_r[, 1]
  tau_true <- 1
  y_r <- m0 + tau_true * a_r + eps
  tau_hat <- numeric(0)
  for (ed in dp_modes) {
    set.seed(1)
    fit <- FusionForest(
      y                         = y_r,
      X_train_control           = X_r,
      X_train_treat             = X_r,
      treatment_indicator_train = a_r,
      source_indicator_train    = s_r,
      treatment_coding          = "binary",
      error_dist                = ed,
      N_post = 100L, N_burn = 100L, verbose = FALSE
    )
    expect_lt(abs(mean(fit$train_predictions_control - m0)), 0.2, label = ed)
    expect_gt(cor(fit$train_predictions_control, m0), 0.9, label = ed)
    expect_lt(abs(mean(fit$train_predictions_treat) - tau_true), 0.2,
              label = ed)
    expect_true(all(abs(fit$train_predictions_deviation) < 0.5), info = ed)
    tau_hat[ed] <- mean(fit$train_predictions_treat)
  }
  # tau is invariant to the centring, so the error models agree on it
  expect_lt(diff(range(tau_hat)), 0.15)
})
