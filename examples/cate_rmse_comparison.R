# =============================================================================
# CATE estimation: FusionForest (RCT + OS) vs. CausalShrinkageForest (RCT only)
#
# Data-generating process
# -----------------------
#   y = m0(X) + A * tau(X) + A * (1-S) * c(U) + sigma * eps
#
#   m0(X)   = 2*X1 - X2 + 0.5*X3        (prognostic function, observed X)
#   tau(X)  = X1 + 0.5*X2^2             (true CATE, observed X)
#   c(U)    = -U1 + U2                  (confounding driven by UNOBSERVED U)
#
# In the RWD, treatment selection is confounded by both unobserved U1 and U2:
#   P(A=1 | X, U, S=0) = logistic(X1 + U1 + U2)
# Without U2 in the propensity score, U2 would only shift the outcome, not
# confound treatment selection — U2 must appear in both places to be a true
# confounder.
#
# U is never passed to either model — it is a true unobserved confounder.
#
# Source coding: S = 1 (RCT), S = 0 (OS)
# CATE RMSE is evaluated on three populations:
#   - RCT  : first n_rct rows of training data
#   - RWD  : last n_os rows of training data
#   - All  : all n_rct + n_os rows
#
# Requires:
#   devtools::load_all()         for FusionForest
#   library(ShrinkageTrees)      for CausalShrinkageForest (tau-learner baseline)
# =============================================================================

library(FusionForests)
library(ShrinkageTrees)

# ---- Dimensions -------------------------------------------------------
n_rct <- 100
n_os  <- 400
p     <- 5     # number of OBSERVED covariates
q     <- 2     # number of UNOBSERVED confounders (OS only)

# ---- MCMC settings ---------------------------------------------------
N_post <- 2500
N_burn <- 1500

# ---- Observed covariates (passed to models) ---------------------------
X_rct <- matrix(runif(n_rct * p), n_rct, p)
X_os  <- matrix(runif(n_os  * p), n_os,  p)

# ---- Unobserved confounders (OS only, never seen by any model) --------
U_os  <- matrix(rnorm(n_os * q), n_os, q)

# ---- True functions ---------------------------------------------------
m0   <- function(X) 2*X[,1] - X[,2] + 0.5*X[,3]
tau  <- function(X) X[,1] + 0.5 * X[,2]^2          # true CATE (nonlinear, depends on observed X)
conf <- function(U) -1/2*U[,1] + 1/4*U[,2]                  # confounding driven by unobserved U

true_cate_rct <- tau(X_rct)
true_cate_os  <- tau(X_os)
true_cate_all <- c(true_cate_rct, true_cate_os)

# ---- Treatment assignment ---------------------------------------------
A_rct <- rbinom(n_rct, 1, 0.5)                                    # RCT: balanced randomisation
A_os  <- rbinom(n_os,  1, plogis(X_os[,1] + U_os[,1] + U_os[,2])) # OS: confounded by X1, U1, and U2

# ---- Outcomes ---------------------------------------------------------
sigma <- 0.5

y_rct <- m0(X_rct) + A_rct * tau(X_rct)                        + rnorm(n_rct, 0, sigma)
y_os  <- m0(X_os)  + A_os  * tau(X_os) + A_os * conf(U_os)    + rnorm(n_os,  0, sigma)

# ---- Combined training set — only observed X is passed to models ------
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
  N_post = N_post, N_burn = N_burn,
  verbose = FALSE
)

cate_ff_rct <- fit_ff$train_predictions_treat[1:n_rct]
cate_ff_os  <- fit_ff$train_predictions_treat[(n_rct + 1):(n_rct + n_os)]
cate_ff_all <- fit_ff$train_predictions_treat

# =============================================================================
# Model 2: CausalShrinkageForest — RCT only, standard BART priors
#
# Two-forest BCF-style model fitted on the RCT data only.
# prior_type = "standard" uses the standard BART leaf prior (half-normal
# with scale omega = 0.5/sqrt(trees)), equivalent to FusionForest's omega.
# To evaluate on all populations, supply X_train as test data.
# =============================================================================
cat("Fitting CausalShrinkageForest (RCT only, standard BART priors)...\n")

n_trees_csf <- 200

fit_csf <- CausalShrinkageForest(
  y                         = y_rct,
  X_train_control           = X_rct,
  X_train_treat             = X_rct,
  treatment_indicator_train = A_rct,
  X_test_control            = X_train,
  X_test_treat              = X_train,
  treatment_indicator_test  = A_train,
  prior_type_control        = "standard",
  prior_type_treat          = "standard",
  local_hp_control          = 0.5 / sqrt(n_trees_csf),
  local_hp_treat            = 0.5 / sqrt(n_trees_csf),
  number_of_trees_control   = n_trees_csf,
  number_of_trees_treat     = n_trees_csf,
  N_post = N_post, N_burn = N_burn,
  verbose = FALSE
)

cate_csf_rct <- fit_csf$train_predictions_treat          # in-sample RCT predictions
cate_csf_os  <- fit_csf$test_predictions_treat[(n_rct + 1):(n_rct + n_os)]
cate_csf_all <- c(cate_csf_rct, cate_csf_os)

# =============================================================================
# Results
# =============================================================================
rmse <- function(pred, truth) sqrt(mean((pred - truth)^2))

cat("\n===== CATE RMSE by population =====\n")
cat(sprintf("%-12s  %10s  %10s\n", "Population", "FusionForest", "CSF (RCT)"))
cat(sprintf("%-12s  %10.4f  %10.4f\n", "RCT",  rmse(cate_ff_rct, true_cate_rct), rmse(cate_csf_rct, true_cate_rct)))
cat(sprintf("%-12s  %10.4f  %10.4f\n", "RWD",  rmse(cate_ff_os,  true_cate_os),  rmse(cate_csf_os,  true_cate_os)))
cat(sprintf("%-12s  %10.4f  %10.4f\n", "All",  rmse(cate_ff_all, true_cate_all), rmse(cate_csf_all, true_cate_all)))
