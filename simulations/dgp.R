# ============================================================================
# DGP and missingness functions for the IRS simulation study
#
# See info/IRS_simulation_plan.md for the full design.
# ============================================================================

# Generate correlated covariates X ~ N(0, Sigma)
#
# Sigma has unit diagonal, Cor(X_{miss_col}, X1:X3) = rho, other off-diag = 0.1.
# Additional covariates beyond the first 4 are independent noise (corr 0.1).
generate_covariates <- function(n, p = 5, rho = 0.5, miss_col = 4) {
  Sigma <- matrix(0.1, nrow = p, ncol = p)
  diag(Sigma) <- 1.0
  # Set correlation between X_{miss_col} and X1:X3 (excluding miss_col itself)
  signal_cols <- setdiff(1:min(3, p), miss_col)
  if (length(signal_cols) > 0 && miss_col <= p) {
    Sigma[miss_col, signal_cols] <- rho
    Sigma[signal_cols, miss_col] <- rho
  }
  MASS::mvrnorm(n, mu = rep(0, p), Sigma = Sigma)
}

# True regression function
#
# type = "additive":  f(x) = 2*x1 + x2 + 0.5*x3 + beta4*x4
# type = "nonlinear": f(x) = x1*x2 + sin(pi*x3) + beta4*x4^2
true_function <- function(X, type = c("additive", "nonlinear"), beta4 = 1.0) {
  type <- match.arg(type)
  if (type == "additive") {
    2 * X[, 1] + X[, 2] + 0.5 * X[, 3] + beta4 * X[, 4]
  } else {
    X[, 1] * X[, 2] + sin(pi * X[, 3]) + beta4 * X[, 4]^2
  }
}

# Master DGP function
#
# Generates training and test data with noise calibrated to the target SNR.
# Returns a list with all components needed for fitting and evaluation.
generate_data <- function(n_train = 200, n_test = 1000, p = 5, rho = 0.5,
                          f_type = "additive", beta4 = 1.0, snr = 3,
                          miss_col = 4, seed = NULL) {
  if (!is.null(seed)) set.seed(seed)

  X_train <- generate_covariates(n_train, p = p, rho = rho, miss_col = miss_col)
  X_test  <- generate_covariates(n_test,  p = p, rho = rho, miss_col = miss_col)

  f_train <- true_function(X_train, type = f_type, beta4 = beta4)
  f_test  <- true_function(X_test,  type = f_type, beta4 = beta4)

  # Calibrate noise: Var(f) / sigma^2 = snr^2
  sigma_noise <- sqrt(var(f_train) / snr^2)
  y_train     <- f_train + rnorm(n_train, sd = sigma_noise)

  list(
    X_train     = X_train,
    X_test      = X_test,
    y_train     = y_train,
    f_train     = f_train,
    f_test      = f_test,
    sigma_noise = sigma_noise
  )
}

# Impose missingness on a single covariate
#
# Returns the modified training matrix and the indices of missing observations.
# Uses NaN (not NA) because SimpleBART uses NaN to signal missingness.
#
# mechanism:
#   "block"     — second half of observations are missing
#   "mcar_low"  — 20% MCAR (default)
#   "mcar_high" — 60% MCAR
#   "mcar"      — custom MCAR with miss_frac
#   "mar"       — missingness depends on X[,1] via logit
impose_missingness <- function(X_train,
                               mechanism = c("block", "mcar_low", "mcar_high",
                                             "mcar", "mar"),
                               miss_col = 4,
                               miss_frac = 0.3) {
  mechanism <- match.arg(mechanism)
  X <- X_train
  n <- nrow(X)

  miss_idx <- switch(mechanism,
    "block"     = seq(ceiling(n / 2) + 1, n),
    "mcar_low"  = sample(n, size = floor(0.2 * n)),
    "mcar_high" = sample(n, size = floor(0.6 * n)),
    "mcar"      = sample(n, size = floor(miss_frac * n)),
    "mar"       = which(runif(n) < plogis(X[, 1]))
  )

  X[miss_idx, miss_col] <- NaN

  list(X_train_miss = X, miss_idx = miss_idx)
}
