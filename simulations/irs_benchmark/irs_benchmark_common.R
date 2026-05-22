# ============================================================================
# IRS Benchmark — Shared DGP, missingness, and evaluation functions
#
# Source this file from both irs_benchmark.R and irs_benchmark_bartm.R
# to ensure identical data generation.
# ============================================================================

# ============================================================================
# Scenarios
# ============================================================================

build_scenario_grid <- function() {
  grid <- expand.grid(
    model   = c("quadratic", "linear", "friedman", "nonlinear"),
    pattern = c("mcar", "mnar", "predictive"),
    p_miss  = c(0.25, 0.6, 0.75),
    n_train = c(100, 200, 400),
    stringsAsFactors = FALSE
  )
  grid$d <- 10L
  grid$miss_cols <- replicate(nrow(grid), 1:3, simplify = FALSE)
  grid$scenario_id <- seq_len(nrow(grid))
  grid
}

get_scenario <- function(id) {
  grid <- build_scenario_grid()
  if (id < 1 || id > nrow(grid)) {
    stop(sprintf("Scenario ID %d out of range [1, %d]", id, nrow(grid)))
  }
  out <- as.list(grid[id, ])
  out$miss_cols <- unlist(out$miss_cols)
  out
}

get_n_scenarios <- function() {
  nrow(build_scenario_grid())
}

# ============================================================================
# DGP — Covariate generation
# ============================================================================

generate_covariates_gaussian <- function(n, d, rho = 0.5) {
  mu    <- rep(1, d)
  Sigma <- rho * tcrossprod(rep(1, d)) + (1 - rho) * diag(d)
  MASS::mvrnorm(n, mu = mu, Sigma = Sigma)
}

generate_covariates_nonlinear <- function(n, sd_eps = 0.05) {
  X_hidden <- runif(n, min = -3, max = 0)
  eps <- matrix(rnorm(n * 10, sd = sd_eps), nrow = n, ncol = 10)
  X <- cbind(
    X_hidden^2                                        + eps[, 1],
    sin(X_hidden)                                     + eps[, 2],
    tanh(X_hidden) * exp(X_hidden) * sin(X_hidden)   + eps[, 3],
    sin(X_hidden - 1) + cos(X_hidden - 3)^3          + eps[, 4],
    (1 - X_hidden)^3                                  + eps[, 5],
    sqrt(sin(X_hidden^2) + 2)                         + eps[, 6],
    X_hidden - 3                                      + eps[, 7],
    (1 - X_hidden) * sin(X_hidden) * cosh(X_hidden)  + eps[, 8],
    1 / (sin(2 * X_hidden) - 2)                       + eps[, 9],
    X_hidden^4                                        + eps[, 10]
  )
  list(X = X, X_hidden = X_hidden)
}

# ============================================================================
# DGP — True regression functions
# ============================================================================

f_quadratic <- function(X) {
  X[, 1]^2 + X[, 2]^2 + X[, 3]^2
}

f_linear <- function(X) {
  beta <- c(1, 2, -1, 3, -0.5, -1, 0.3, 1.7, 0.4, -0.3)
  d <- min(ncol(X), length(beta))
  as.numeric(X[, 1:d, drop = FALSE] %*% beta[1:d])
}

f_friedman <- function(X) {
  10 * sin(pi * X[, 1] * X[, 2]) + 20 * (X[, 3] - 0.5)^2 +
    10 * X[, 4] + 5 * X[, 5]
}

f_nonlinear <- function(X) {
  sin(pi * X[, 1] * X[, 2]) + 2 * (X[, 3] - 0.5)^2 +
    X[, 4] + 0.5 * X[, 5]
}

# ============================================================================
# DGP — Data generation
# ============================================================================

generate_data <- function(n_train = 200, n_test = 1000,
                          model = c("quadratic", "linear",
                                    "friedman", "nonlinear"),
                          d = 10, rho = 0.5, sigma = 0.1,
                          seed = NULL) {
  model <- match.arg(model)
  if (!is.null(seed)) set.seed(seed)

  f_true <- switch(model,
    quadratic = f_quadratic,
    linear    = f_linear,
    friedman  = f_friedman,
    nonlinear = f_nonlinear
  )

  if (model == "nonlinear") {
    train_cov <- generate_covariates_nonlinear(n_train)
    test_cov  <- generate_covariates_nonlinear(n_test)
    X_train   <- train_cov$X
    X_test    <- test_cov$X
  } else {
    X_train <- generate_covariates_gaussian(n_train, d = d, rho = rho)
    X_test  <- generate_covariates_gaussian(n_test,  d = d, rho = rho)
  }

  f_train <- as.numeric(f_true(X_train))
  f_test  <- as.numeric(f_true(X_test))
  y_train <- f_train + rnorm(n_train, sd = sigma)

  list(
    X_train = X_train,
    X_test  = X_test,
    y_train = y_train,
    f_train = f_train,
    f_test  = f_test,
    sigma   = sigma
  )
}

# ============================================================================
# DGP — Missingness mechanisms
# ============================================================================

impose_missingness <- function(X, y = NULL,
                               pattern = c("mcar", "mnar", "predictive"),
                               miss_cols = 1:3,
                               p_miss = 0.3,
                               sigma = 0.1) {
  pattern <- match.arg(pattern)
  n <- nrow(X)
  M <- matrix(0L, nrow = n, ncol = ncol(X))
  X_miss <- X

  if (pattern == "mcar") {
    for (j in miss_cols) {
      idx <- which(rbinom(n, 1, p_miss) == 1)
      M[idx, j]      <- 1L
      X_miss[idx, j]  <- NaN
    }
  } else if (pattern == "mnar") {
    for (j in miss_cols) {
      threshold <- quantile(X[, j], probs = 1 - p_miss)
      idx <- which(X[, j] > threshold)
      M[idx, j]      <- 1L
      X_miss[idx, j]  <- NaN
    }
  } else if (pattern == "predictive") {
    for (j in miss_cols) {
      idx <- which(rbinom(n, 1, p_miss) == 1)
      M[idx, j]      <- 1L
      X_miss[idx, j]  <- NaN
    }
    f_new <- rowSums(X[, miss_cols, drop = FALSE]^2 +
                       2 * M[, miss_cols, drop = FALSE])
    y <- f_new + rnorm(n, sd = sigma)
  }

  list(X_miss = X_miss, M = M, y = y,
       f_true_predictive = if (pattern == "predictive") {
         rowSums(X[, miss_cols, drop = FALSE]^2 +
                   2 * M[, miss_cols, drop = FALSE])
       } else NULL)
}

# ============================================================================
# Evaluation
# ============================================================================

compute_rmse <- function(f_hat, f_true) {
  sqrt(mean((f_hat - f_true)^2))
}

compute_bias <- function(f_hat, f_true) {
  mean(f_hat - f_true)
}

compute_mae <- function(f_hat, f_true) {
  mean(abs(f_hat - f_true))
}

compute_coverage <- function(posterior_matrix, f_true,
                             level = 0.95) {
  alpha <- 1 - level
  lower <- apply(posterior_matrix, 2, quantile,
                 probs = alpha / 2)
  upper <- apply(posterior_matrix, 2, quantile,
                 probs = 1 - alpha / 2)
  mean(f_true >= lower & f_true <= upper)
}

compute_interval_width <- function(posterior_matrix,
                                   level = 0.95) {
  alpha <- 1 - level
  lower <- apply(posterior_matrix, 2, quantile,
                 probs = alpha / 2)
  upper <- apply(posterior_matrix, 2, quantile,
                 probs = 1 - alpha / 2)
  mean(upper - lower)
}

evaluate_fit <- function(fit, f_train, f_test,
                         level = 0.95) {
  metrics <- list(
    rmse_train = compute_rmse(fit$train_predictions, f_train),
    rmse_test  = compute_rmse(fit$test_predictions,  f_test),
    bias_test  = compute_bias(fit$test_predictions,  f_test),
    mae_test   = compute_mae(fit$test_predictions,   f_test)
  )
  if (!is.null(fit$train_predictions_sample)) {
    metrics$coverage <- compute_coverage(
      fit$train_predictions_sample, f_train, level
    )
    metrics$width <- compute_interval_width(
      fit$train_predictions_sample, level
    )
  }
  metrics
}

# ============================================================================
# Shared settings
# ============================================================================

n_test <- 1000L
rho    <- 0.5
sigma  <- 0.5
seed0  <- 2026L

bart_settings <- list(
  number_of_trees = 200,
  N_post          = 3000L,
  N_burn          = 2000L
)

# Helper to build a result row
make_row <- function(rep, method_name, ev) {
  data.frame(
    rep        = rep,
    method     = method_name,
    rmse_train = ev$rmse_train,
    rmse_test  = ev$rmse_test,
    bias_test  = ifelse(is.null(ev$bias_test), NA,
                        ev$bias_test),
    mae_test   = ifelse(is.null(ev$mae_test), NA,
                        ev$mae_test),
    coverage   = ifelse(is.null(ev$coverage), NA,
                        ev$coverage),
    width      = ifelse(is.null(ev$width), NA,
                        ev$width),
    stringsAsFactors = FALSE
  )
}
