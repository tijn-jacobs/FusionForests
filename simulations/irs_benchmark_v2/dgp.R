# ============================================================================
# DGP for the IRS benchmark v2: causal inference with missing covariates
#
# RCT + RWD structure, treatment effects, block-wise missingness on X5
# ============================================================================

library(MASS)

# --- Covariate generation ---------------------------------------------------

#' Generate p=5 covariates with AR(1) correlation: Sigma_{jk} = rho^|j-k|
generate_covariates <- function(n, p = 5, rho = 0) {
  Sigma <- rho^abs(outer(1:p, 1:p, "-"))
  mvrnorm(n, mu = rep(0, p), Sigma = Sigma)
}

# --- Treatment assignment ---------------------------------------------------

#' Assign treatment
#' RCT (S=1): randomized, P(A=1) = 0.5
#' RWD (S=0): confounded, logit P(A=1) = alpha0 + alpha1*X1 + alpha5*X5
assign_treatment <- function(X, S, alpha0 = 0, alpha1 = 0.5,
                             alpha5 = 0.5) {
  n <- nrow(X)
  A <- integer(n)
  rct <- which(S == 1)
  rwd <- which(S == 0)
  A[rct] <- rbinom(length(rct), 1, 0.5)
  lp <- alpha0 + alpha1 * X[rwd, 1] + alpha5 * X[rwd, 5]
  A[rwd] <- rbinom(length(rwd), 1, plogis(lp))
  A
}

# --- Outcome models ---------------------------------------------------------

#' Scenario 0: X5 irrelevant
#' m(X, A) = 1 + X1 + X2 + 2*X3*A
m_scenario0 <- function(X, A) {
  1 + X[, 1] + X[, 2] + 2 * X[, 3] * A
}

#' Scenario 1: X5 prognostic only
#' m(X, A) = 1 + X1 + X2 + beta5*X5 + 2*X3*A
m_scenario1 <- function(X, A, beta5 = 1) {
  1 + X[, 1] + X[, 2] + beta5 * X[, 5] + 2 * X[, 3] * A
}

#' Scenario 2: X5 is an effect modifier
#' m(X, A) = 1 + X1 + X2 + (gamma0 + gamma1*X5)*A
m_scenario2 <- function(X, A, gamma0 = 2, gamma1 = 1) {
  1 + X[, 1] + X[, 2] + (gamma0 + gamma1 * X[, 5]) * A
}

#' Scenario 3: X5 prognostic + effect modifier
#' m(X, A) = 1 + X1 + X2 + beta5*X5 + (gamma0 + gamma1*X5)*A
m_scenario3 <- function(X, A, beta5 = 1, gamma0 = 2,
                        gamma1 = 1) {
  1 + X[, 1] + X[, 2] + beta5 * X[, 5] +
    (gamma0 + gamma1 * X[, 5]) * A
}

# --- CATE functions (tau(x) = m(x,1) - m(x,0)) ----------------------------

tau_scenario0 <- function(X) {
  2 * X[, 3]
}

tau_scenario1 <- function(X) {
  2 * X[, 3]
}

tau_scenario2 <- function(X, gamma0 = 2, gamma1 = 1) {
  gamma0 + gamma1 * X[, 5]
}

tau_scenario3 <- function(X, gamma0 = 2, gamma1 = 1) {
  gamma0 + gamma1 * X[, 5]
}

# --- Calibrate sigma for SNR ~ 2 -------------------------------------------

#' Calibrate sigma so that Var(m) / sigma^2 ~ target_snr
calibrate_sigma <- function(outcome_scenario, rho = 0,
                            target_snr = 2,
                            n_cal = 10000, seed = 999) {
  set.seed(seed)
  X <- generate_covariates(n_cal, rho = rho)
  A <- rbinom(n_cal, 1, 0.5)

  m_fn <- switch(as.character(outcome_scenario),
    "0" = m_scenario0,
    "1" = m_scenario1,
    "2" = m_scenario2,
    "3" = m_scenario3
  )
  m_vals <- m_fn(X, A)
  sigma <- sqrt(var(m_vals) / target_snr)
  as.numeric(sigma)
}

# --- Master DGP function ---------------------------------------------------

#' Generate a complete dataset for one simulation replicate
#'
#' @param n_rct  Number of RCT observations
#' @param n_rwd  Number of RWD observations
#' @param n_test Number of test observations (complete, no missingness)
#' @param outcome_scenario Integer 0-3
#' @param rho Covariate correlation parameter
#' @param sigma Noise standard deviation (NULL = auto-calibrate)
#' @param seed Random seed
#' @return List with training data, test data, true values
generate_data <- function(n_rct = 150, n_rwd = 350,
                          n_test = 500,
                          outcome_scenario = 0,
                          rho = 0, sigma = NULL,
                          alpha0 = 0, alpha1 = 0.5,
                          alpha5 = 0.5,
                          beta5 = 1, gamma0 = 2,
                          gamma1 = 1,
                          seed = NULL) {
  if (!is.null(seed)) set.seed(seed)

  n <- n_rct + n_rwd

  # Source indicator
  S <- c(rep(1L, n_rct), rep(0L, n_rwd))

  # Covariates
  X <- generate_covariates(n, rho = rho)
  X_test <- generate_covariates(n_test, rho = rho)

  # Treatment
  A <- assign_treatment(X, S, alpha0, alpha1, alpha5)
  A_test <- rbinom(n_test, 1, 0.5)

  # Outcome function
  m_fn <- switch(as.character(outcome_scenario),
    "0" = function(X, A) m_scenario0(X, A),
    "1" = function(X, A) m_scenario1(X, A, beta5),
    "2" = function(X, A) m_scenario2(X, A, gamma0, gamma1),
    "3" = function(X, A) m_scenario3(X, A, beta5, gamma0,
                                     gamma1)
  )

  tau_fn <- switch(as.character(outcome_scenario),
    "0" = function(X) tau_scenario0(X),
    "1" = function(X) tau_scenario1(X),
    "2" = function(X) tau_scenario2(X, gamma0, gamma1),
    "3" = function(X) tau_scenario3(X, gamma0, gamma1)
  )

  # Calibrate sigma if not provided
  if (is.null(sigma)) {
    sigma <- calibrate_sigma(outcome_scenario, rho)
  }

  # True regression and CATE values
  m_train <- m_fn(X, A)
  m_test  <- m_fn(X_test, A_test)
  tau_train <- tau_fn(X)
  tau_test  <- tau_fn(X_test)

  # Observed outcome
  y <- m_train + rnorm(n, sd = sigma)

  # Build covariate matrix with treatment: [X, A]
  XA_train <- cbind(X, A)
  XA_test  <- cbind(X_test, A_test)
  colnames(XA_train) <- c(paste0("X", 1:5), "A")
  colnames(XA_test)  <- c(paste0("X", 1:5), "A")

  # Counterfactual test matrices (for CATE)
  XA_test_1 <- cbind(X_test, 1)
  XA_test_0 <- cbind(X_test, 0)
  colnames(XA_test_1) <- colnames(XA_train)
  colnames(XA_test_0) <- colnames(XA_train)

  # Same for training
  XA_train_1 <- cbind(X, 1)
  XA_train_0 <- cbind(X, 0)
  colnames(XA_train_1) <- colnames(XA_train)
  colnames(XA_train_0) <- colnames(XA_train)

  list(
    # Training
    y         = y,
    XA_train  = XA_train,
    X_train   = X,
    A_train   = A,
    S_train   = S,
    m_train   = m_train,
    tau_train = tau_train,
    # Test (complete data)
    XA_test   = XA_test,
    XA_test_1 = XA_test_1,
    XA_test_0 = XA_test_0,
    X_test    = X_test,
    A_test    = A_test,
    m_test    = m_test,
    tau_test  = tau_test,
    # Counterfactual train matrices
    XA_train_1 = XA_train_1,
    XA_train_0 = XA_train_0,
    # Parameters
    sigma     = sigma,
    n_rct     = n_rct,
    n_rwd     = n_rwd
  )
}

# --- Missingness mechanisms -------------------------------------------------

#' Impose missingness on X5 (column 5 of XA, which has A as col 6)
#'
#' @param XA  n x 6 matrix [X1..X5, A]
#' @param S   Source indicator (1 = RCT, 0 = RWD)
#' @param miss_pattern "block_rct", "block_rwd", or "mcar"
#' @param pi_mcar MCAR probability (used only if pattern = "mcar")
#' @return List with XA_miss (NaN for missing), M (indicator)
impose_missingness <- function(XA, S,
                               miss_pattern = c("block_rct",
                                                "block_rwd",
                                                "mcar"),
                               pi_mcar = 0.3) {
  miss_pattern <- match.arg(miss_pattern)
  n <- nrow(XA)
  XA_miss <- XA
  M <- integer(n)  # 1 = X5 missing

  if (miss_pattern == "block_rct") {
    # X5 missing in RCT (S=1), observed in RWD (S=0)
    idx <- which(S == 1)
    M[idx] <- 1L
    XA_miss[idx, 5] <- NaN
  } else if (miss_pattern == "block_rwd") {
    # X5 missing in RWD (S=0), observed in RCT (S=1)
    idx <- which(S == 0)
    M[idx] <- 1L
    XA_miss[idx, 5] <- NaN
  } else if (miss_pattern == "mcar") {
    # MCAR with probability pi
    idx <- which(rbinom(n, 1, pi_mcar) == 1)
    M[idx] <- 1L
    XA_miss[idx, 5] <- NaN
  }

  list(XA_miss = XA_miss, M = M)
}
