# ============================================================================
# DGP and missingness functions for the IRS benchmark simulation
#
# Based on the simulation settings of Josse, Prost, Scornet & Varoquaux:
#   - 4 regression models (Quadratic, Linear, Friedman, Nonlinear)
#   - 3 missing patterns (MCAR, Censoring MNAR, Predictive Missingness)
#   - Multiple missingness fractions and sample sizes
# ============================================================================

# --- Covariate generation ---------------------------------------------------

# Models 1-3: X ~ N(1_d, Sigma) with Sigma = rho*1*1' + (1-rho)*I
generate_covariates_gaussian <- function(n, d, rho = 0.5) {
  mu    <- rep(1, d)
  Sigma <- rho * tcrossprod(rep(1, d)) + (1 - rho) * diag(d)
  MASS::mvrnorm(n, mu = mu, Sigma = Sigma)
}

# Model 4: covariates are nonlinear transformations of a hidden uniform X
generate_covariates_nonlinear <- function(n, sd_eps = 0.05) {
  X_hidden <- runif(n, min = -3, max = 0)
  eps <- matrix(rnorm(n * 10, sd = sd_eps), nrow = n, ncol = 10)

  X <- cbind(
    X_hidden^2                                          + eps[, 1],
    sin(X_hidden)                                       + eps[, 2],
    tanh(X_hidden) * exp(X_hidden) * sin(X_hidden)     + eps[, 3],
    sin(X_hidden - 1) + cos(X_hidden - 3)^3            + eps[, 4],
    (1 - X_hidden)^3                                    + eps[, 5],
    sqrt(sin(X_hidden^2) + 2)                           + eps[, 6],
    X_hidden - 3                                        + eps[, 7],
    (1 - X_hidden) * sin(X_hidden) * cosh(X_hidden)    + eps[, 8],
    1 / (sin(2 * X_hidden) - 2)                         + eps[, 9],
    X_hidden^4                                          + eps[, 10]
  )

  list(X = X, X_hidden = X_hidden)
}

# --- True regression functions ----------------------------------------------

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

# --- Master DGP function ----------------------------------------------------

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

# --- Missingness mechanisms -------------------------------------------------

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
    # Overwrite y: Y = sum_{j}(X_j^2 + 2*M_j) + eps
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
