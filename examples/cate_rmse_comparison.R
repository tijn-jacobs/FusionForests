# =============================================================================
# CATE estimation: FusionForest (RCT + OS) vs. BART T-learner (RCT only)
#
# Data-generating process
# -----------------------
#   y = m0(X) + A * tau(X) + A * (1-S) * c(X) + sigma * eps
#
#   m0(X)   = 2*X1 - X2 + 0.5*X3        (prognostic function)
#   tau(X)  = X1 + 0.5*X2^2             (true CATE — nonlinear)
#   c(X)    = -X1 + X3                  (confounding, OS treated only)
#
# Source coding: S = 1 (RCT), S = 0 (OS)
#
# Requires:
#   devtools::load_all()   for FusionForest
#   install.packages("BART")
# =============================================================================

library(FusionForests)
library(BART)

set.seed(42)

# ---- Dimensions -------------------------------------------------------
n_rct  <- 200
n_os   <- 400
n_test <- 500
p      <- 5

# ---- Covariates -------------------------------------------------------
X_rct  <- matrix(rnorm(n_rct * p), n_rct,  p)
X_os   <- matrix(rnorm(n_os  * p), n_os,   p)
X_test <- matrix(rnorm(n_test * p), n_test, p)

# ---- True functions ---------------------------------------------------
m0   <- function(X) 2*X[,1] - X[,2] + 0.5*X[,3]
tau  <- function(X) X[,1] + 0.5*X[,2]^2          # true CATE (nonlinear)
conf <- function(X) -X[,1] + X[,3]               # confounding (OS only)

true_cate_test <- tau(X_test)

# ---- Treatment assignment ---------------------------------------------
A_rct <- rbinom(n_rct, 1, 0.5)                    # RCT: balanced
A_os  <- rbinom(n_os,  1, plogis(X_os[,1]))       # OS: confounded

# ---- Outcomes ---------------------------------------------------------
sigma <- 1.0

y_rct <- m0(X_rct) + A_rct * tau(X_rct)                       + rnorm(n_rct, 0, sigma)
y_os  <- m0(X_os)  + A_os  * tau(X_os) + A_os * conf(X_os)   + rnorm(n_os,  0, sigma)

# ---- Combined training set (RCT first) --------------------------------
X_train <- rbind(X_rct, X_os)
A_train <- c(A_rct, A_os)
S_train <- c(rep(1L, n_rct), rep(0L, n_os))
y_train <- c(y_rct, y_os)

# =============================================================================
# Model 1: FusionForest — RCT + OS, three-forest data fusion
# =============================================================================
cat("Fitting FusionForest (RCT + OS)...\n")

fit_ff <- FusionForest(
  y                         = y_train,
  X_train_control           = X_train,
  X_train_treat             = X_train,
  treatment_indicator_train = A_train,
  source_indicator_train    = S_train,
  X_test_control            = X_test,
  X_test_treat              = X_test,
  treatment_indicator_test  = rep(1L, n_test),
  source_indicator_test     = rep(1L, n_test),
  N_post = 1000, N_burn = 500,
  verbose = FALSE
)

cate_ff   <- fit_ff$test_predictions_treat
rmse_ff   <- sqrt(mean((cate_ff - true_cate_test)^2))

# =============================================================================
# Model 2: BART T-learner — RCT only, standard BART priors
#
# Two separate BART forests fitted on the treated and control RCT arms.
# CATE(x) = E[Y(1)|X=x] - E[Y(0)|X=x]
#          = mu_1(x) - mu_0(x)
#
# This is the equivalent of CausalShrinkageForest with standard (non-horseshoe)
# BART priors, using only the RCT data.
# =============================================================================
cat("Fitting BART T-learner (RCT only)...\n")

rct_ctrl_idx <- which(A_rct == 0)
rct_trt_idx  <- which(A_rct == 1)

fit_ctrl <- wbart(
  x.train = X_rct[rct_ctrl_idx, ],
  y.train = y_rct[rct_ctrl_idx],
  x.test  = X_test,
  nskip   = 500,
  ndpost  = 1000,
  printevery = 0L
)

fit_trt <- wbart(
  x.train = X_rct[rct_trt_idx, ],
  y.train = y_rct[rct_trt_idx],
  x.test  = X_test,
  nskip   = 500,
  ndpost  = 1000,
  printevery = 0L
)

cate_tl   <- fit_trt$yhat.test.mean - fit_ctrl$yhat.test.mean
rmse_tl   <- sqrt(mean((cate_tl - true_cate_test)^2))

# =============================================================================
# Results
# =============================================================================
cat("\n===== CATE RMSE on test set =====\n")
cat(sprintf("  FusionForest (RCT + OS)      : %.4f\n", rmse_ff))
cat(sprintf("  BART T-learner (RCT only)    : %.4f\n", rmse_tl))
cat(sprintf("  Ratio (T-learner / FF)       : %.2fx\n", rmse_tl / rmse_ff))
