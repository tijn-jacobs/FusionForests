# =============================================================================
# CATE estimation: FusionForest (RCT + RWD) vs. CausalShrinkageForest (RCT only)
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
# Source coding: S = 1 (RCT), S = 0 (RWD)
# CATE RMSE is evaluated on three populations:
#   - RCT  : first n_rct rows of training data
#   - RWD  : last n_rwd rows of training data
#   - All  : all n_rct + n_rwd rows
#
# Requires:
#   devtools::load_all()         for FusionForest
#   library(ShrinkageTrees)      for CausalShrinkageForest (tau-learner baseline)
# =============================================================================

library(FusionForests)
library(ShrinkageTrees)

# ---- Dimensions -------------------------------------------------------
n_rct <- 100
n_rwd  <- 400
p     <- 5     # number of OBSERVED covariates

# ---- MCMC settings ---------------------------------------------------
N_post <- 2500
N_burn <- 1500

# ---- Observed covariates (passed to models) ---------------------------
X_rct <- matrix(runif(n_rct * p), n_rct, p)
X_rwd  <- matrix(runif(n_rwd  * p), n_rwd,  p)

# ---- Unobserved confounders (RWD only, never seen by any model) -------
U  <- rnorm(n_rwd)

# ---- True functions ---------------------------------------------------
m0   <- function(X) 2*X[,1] - X[,2] + 0.5*X[,3]
dev <- function(X, lambda) lambda * (X[,4] - 1/2 * X[,5]) 
tau  <- function(X) X[,1] + 0.5 * X[,2]^2          # true CATE (nonlinear, depends on observed X)
conf <- function(U) -1/2*U

true_cate_rct <- tau(X_rct)
true_cate_rwd  <- tau(X_rwd)
true_cate_all <- c(true_cate_rct, true_cate_rwd)

# ---- Treatment assignment ---------------------------------------------
A_rct <- rbinom(n_rct, 1, 0.5)                                    # RCT: balanced randomisation
A_rwd  <- rbinom(n_rwd,  1, plogis(X_rwd[,1] + U)) # RWD: confounded by X1 and U

# ---- Outcomes ---------------------------------------------------------
y_rct <- 2 * (m0(X_rct) + A_rct * tau(X_rct))                        + rnorm(n_rct, 0, 0.75)
y_rwd  <- 2 * (m0(X_rwd) + dev(X_rwd, 1) + A_rwd  * tau(X_rwd) + A_rwd * conf(U))    + rnorm(n_rwd,  0, 1.25)

# ---- Combined training set — only observed X is passed to models ------
X_train <- rbind(X_rct, X_rwd)
A_train <- c(A_rct, A_rwd)
S_train <- c(rep(1L, n_rct), rep(0L, n_rwd))
y_train <- c(y_rct, y_rwd)

# =============================================================================
# Model 1: FusionForest — RCT + RWD
# =============================================================================
fit_ff <- FusionForest(
  y                         = y_train,
  X_train_control           = X_train,
  X_train_treat             = X_train,
  treatment_indicator_train = A_train,
  source_indicator_train    = S_train,
  error_dist = "source_dp_scale",
  N_post = N_post, N_burn = N_burn,
  verbose = FALSE
)

cate_ff_rct <- fit_ff$train_predictions_treat[1:n_rct]
cate_ff_rwd  <- fit_ff$train_predictions_treat[(n_rct + 1):(n_rct + n_rwd)]
cate_ff_all <- fit_ff$train_predictions_treat

# =============================================================================
# Model 2: CausalShrinkageForest — RCT only, standard BART priors
#
# Two-forest BCF-style model fitted on the RCT data only.
# prior_type = "standard" uses the standard BART leaf prior (half-normal
# with scale omega = 0.5/sqrt(trees)), equivalent to FusionForest's omega.
# To evaluate on all populations, supply X_train as test data.
# =============================================================================
n_trees_csf <- 200

fit_csf <- ShrinkageTrees::CausalShrinkageForest(
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
cate_csf_rwd  <- fit_csf$test_predictions_treat[(n_rct + 1):(n_rct + n_rwd)]
cate_csf_all <- c(cate_csf_rct, cate_csf_rwd)

# =============================================================================
# Model 2: CausalShrinkageForest — RCT only, standard BART priors
#
# Two-forest BCF-style model fitted on the RCT data only.
# prior_type = "standard" uses the standard BART leaf prior (half-normal
# with scale omega = 0.5/sqrt(trees)), equivalent to FusionForest's omega.
# To evaluate on all populations, supply X_train as test data.
# =============================================================================

n_trees_csf <- 200

fit_rwd <- ShrinkageTrees::CausalShrinkageForest(
  y                         = y_rwd,
  X_train_control           = X_rwd,
  X_train_treat             = X_rwd,
  treatment_indicator_train = A_rwd,
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

cate_rwd_rwd <- fit_rwd$train_predictions_treat          # in-sample RCT predictions
cate_rwd_rct  <- fit_rwd$test_predictions_treat[1:n_rct]
cate_rwd_all <- c(cate_rwd_rct, cate_rwd_rwd)

# =============================================================================
# Results
# =============================================================================
rmse <- function(pred, truth) sqrt(mean((pred - truth)^2))

results <- data.frame(
  Population   = c("RCT", "RWD", "All"),
  FusionForest = c(rmse(cate_ff_rct,  true_cate_rct),
                   rmse(cate_ff_rwd,  true_cate_rwd),
                   rmse(cate_ff_all,  true_cate_all)),
  `RCT`        = c(rmse(cate_csf_rct, true_cate_rct),
                   rmse(cate_csf_rwd, true_cate_rwd),
                   rmse(cate_csf_all, true_cate_all)),
  `RWD`        = c(rmse(cate_rwd_rct, true_cate_rct),
                   rmse(cate_rwd_rwd, true_cate_rwd),
                   rmse(cate_rwd_all, true_cate_all)),
  check.names  = FALSE
)
results[, -1] <- round(results[, -1], 4)
print(results, row.names = FALSE)
