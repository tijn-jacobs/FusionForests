## Example: Comparing FusionForest decompositions for HTE estimation
##
## Simulation design:
##   - True CATE varies by covariate X1 (effect modifier)
##   - RCT: small, randomised, no unmeasured confounding
##   - RWD: large, observational, with unmeasured confounding (U)
##
## We compare four models, all using the FusionForest engine so that
## tree priors (power, base, p_grow, p_prune), error-variance prior
## (nu, q), and MCMC settings (N_post, N_burn) are identical:
##
##   1. Two-forest BCF on RCT data only   (unconfounded, low power)
##   2. Two-forest BCF on RWD data only   (naive — ignores confounding)
##   3. Three-forest FusionForest (RCT+RWD, with deconfounding)
##   4. Four-forest FusionForest  (RCT+RWD, deconfounding + deviation)
##
## For the two-forest models, the deconfounding forest is neutralised
## by setting number_of_trees_deconf = 1 and flagging a single training
## observation as source = 0 (the minimum the API requires). All test
## predictions use source = 1, so c(X) contributes nothing.


setwd("~/Library/CloudStorage/OneDrive-VrijeUniversiteitAmsterdam/Documents/GitHub/FusionForests")
devtools::document()
devtools::load_all()

# -------------------------------------------------------------------------
# 1. DATA GENERATING PROCESS
# -------------------------------------------------------------------------

n_rct <- 100   # RCT sample size (small, as is typical)
n_rwd <- 500   # RWD sample size (larger observational dataset)
p     <- 10    # number of covariates

# --- True functions -------------------------------------------------------
mu0       <- function(X) 2 * X[, 1] - X[, 3]^2
true_cate <- function(X) 1.5 + 2 * X[, 1]

sigma_true <- 2

# --- Generate RCT ---------------------------------------------------------
X_rct <- matrix(runif(n_rct * p), nrow = n_rct, ncol = p)
colnames(X_rct) <- paste0("X", seq_len(p))

trt_rct <- rbinom(n_rct, 1, 0.5)   # balanced randomisation
y_rct   <- mu0(X_rct) + true_cate(X_rct) * trt_rct +
           rnorm(n_rct, sd = sigma_true)

# --- Generate RWD ---------------------------------------------------------
X_rwd <- matrix(runif(n_rwd * p), nrow = n_rwd, ncol = p)
colnames(X_rwd) <- paste0("X", seq_len(p))

# U is an unmeasured confounder: it drives both treatment assignment
# and the outcome but is NOT included in the covariate matrix X_rwd.
U_rwd          <- rnorm(n_rwd)
prop_score_rwd <- plogis(0.8 * U_rwd)
trt_rwd <- rbinom(n_rwd, 1, prop_score_rwd)
y_rwd   <- mu0(X_rwd) + 1.0*U_rwd + true_cate(X_rwd) * trt_rwd +
           rnorm(n_rwd, sd = sigma_true)

# --- Pool RCT + RWD -------------------------------------------------------
X_all      <- rbind(X_rct, X_rwd)
y_all      <- c(y_rct, y_rwd)
trt_all    <- c(trt_rct, trt_rwd)
source_all <- c(rep(1L, n_rct), rep(0L, n_rwd))   # 1 = RCT, 0 = RWD

true_cate_all <- true_cate(X_all)
n_all         <- nrow(X_all)

cat("--------------------------------------------------------------\n")
cat("Data summary\n")
cat("  RCT:  n =", n_rct, " | treated:", sum(trt_rct),
    "| control:", sum(trt_rct == 0), "\n")
cat("  RWD:  n =", n_rwd, " | treated:", sum(trt_rwd),
    "| control:", sum(trt_rwd == 0), "\n")
cat("  True CATE range: [", round(min(true_cate_all), 2),
    ",", round(max(true_cate_all), 2), "]\n")
cat("--------------------------------------------------------------\n\n")

# -------------------------------------------------------------------------
# 2. HYPERPARAMETERS
#
# Tree counts and k (leaf-prior scale) vary by forest type but are
# kept identical across the four models for a fair comparison.
# The remaining tree-prior and error-variance parameters are shared.
# -------------------------------------------------------------------------

# Per-forest-type tree counts
n_trees_prog   <- 200   # prognostic forest (mu / m0)
n_trees_treat  <- 100   # treatment-effect forest (tau)
n_trees_deconf <- 200   # deconfounding forest (c)
n_trees_dev    <- 50    # deviation forest (g, four-forest only)

# Leaf-prior scale k for the deviation forest.
# Larger k_g => stronger shrinkage of g toward zero (more borrowing).
# The control/treat/deconf forests use the package default (k = 1).
k_g <- 2

# MCMC
N_post <- 3000
N_burn <- 2000

# Shared across all forests
power   <- 2.0
base    <- 0.95
p_grow  <- 0.4
p_prune <- 0.4
nu      <- 3
q       <- 0.90

# -------------------------------------------------------------------------
# 3. MODEL 1: Two-forest BCF --- RCT only
#
# FusionForest requires >= 1 RWD row. We flag one training
# observation as source = 0 and set number_of_trees_deconf = 1
# so the deconfounding forest is effectively inert.
# -------------------------------------------------------------------------
cat("Fitting Model 1: Two-forest BCF (RCT only)...\n")

src_rct      <- rep(1L, n_rct)
src_rct[n_rct] <- 0L                    # single dummy RWD row

fit_rct <- FusionForest(
  y                         = y_rct,
  X_train_control           = X_rct,
  X_train_treat             = X_rct,
  treatment_indicator_train  = trt_rct,
  source_indicator_train     = src_rct,
  X_test_control            = X_all,
  X_test_treat              = X_all,
  treatment_indicator_test   = trt_all,
  source_indicator_test      = rep(1L, n_all),
  outcome_type              = "continuous",
  decomposition             = "three-forest",
  treatment_coding          = "centered",
  number_of_trees_control   = n_trees_prog,
  number_of_trees_treat     = n_trees_treat,
  number_of_trees_deconf    = 1,
  power                     = power,
  base                      = base,
  p_grow                    = p_grow,
  p_prune                   = p_prune,
  nu                        = nu,
  q                         = q,
  N_post                    = N_post,
  N_burn                    = N_burn,
  store_posterior_sample     = TRUE,
  verbose                   = FALSE
)

cat("  Done.\n\n")

# -------------------------------------------------------------------------
# 4. MODEL 2: Two-forest BCF --- RWD only (naive, ignores confounding)
#
# Same trick: all observations labelled source = 1 except one dummy
# source = 0 row, so c(X) is inert. By treating RWD as if it
# were randomised we get a naive BCF that does not adjust for
# unmeasured confounding.
# -------------------------------------------------------------------------
cat("Fitting Model 2: Two-forest BCF (RWD only)...\n")

src_rwd       <- rep(1L, n_rwd)
src_rwd[n_rwd] <- 0L                     # single dummy RWD row

fit_rwd <- FusionForest(
  y                         = y_rwd,
  X_train_control           = X_rwd,
  X_train_treat             = X_rwd,
  treatment_indicator_train  = trt_rwd,
  source_indicator_train     = src_rwd,
  X_test_control            = X_all,
  X_test_treat              = X_all,
  treatment_indicator_test   = trt_all,
  source_indicator_test      = rep(1L, n_all),
  outcome_type              = "continuous",
  decomposition             = "three-forest",
  treatment_coding          = "centered",
  number_of_trees_control   = n_trees_prog,
  number_of_trees_treat     = n_trees_treat,
  number_of_trees_deconf    = 1,
  power                     = power,
  base                      = base,
  p_grow                    = p_grow,
  p_prune                   = p_prune,
  nu                        = nu,
  q                         = q,
  N_post                    = N_post,
  N_burn                    = N_burn,
  store_posterior_sample     = TRUE,
  verbose                   = FALSE
)

cat("  Done.\n\n")

# -------------------------------------------------------------------------
# 5. MODEL 3: Three-forest FusionForest (RCT + RWD)
# -------------------------------------------------------------------------
cat("Fitting Model 3: Three-forest FusionForest (RCT + RWD)...\n")

fit_three <- FusionForest(
  y                         = y_all,
  X_train_control           = X_all,
  X_train_treat             = X_all,
  treatment_indicator_train  = trt_all,
  source_indicator_train     = source_all,
  X_test_control            = X_all,
  X_test_treat              = X_all,
  treatment_indicator_test   = trt_all,
  source_indicator_test      = source_all,
  outcome_type              = "continuous",
  decomposition             = "three-forest",
  treatment_coding          = "centered",
  number_of_trees_control   = n_trees_prog,
  number_of_trees_treat     = n_trees_treat,
  number_of_trees_deconf    = n_trees_deconf,
  power                     = power,
  base                      = base,
  p_grow                    = p_grow,
  p_prune                   = p_prune,
  nu                        = nu,
  q                         = q,
  N_post                    = N_post,
  N_burn                    = N_burn,
  store_posterior_sample    = TRUE,
  verbose                   = FALSE
)

cat("  Done.\n\n")

# -------------------------------------------------------------------------
# 6. MODEL 4: Four-forest FusionForest (RCT + RWD)
# -------------------------------------------------------------------------
cat("Fitting Model 4: Four-forest FusionForest (RCT + RWD)...\n")

fit_four <- FusionForest(
  y                         = y_all,
  X_train_control           = X_all,
  X_train_treat             = X_all,
  treatment_indicator_train  = trt_all,
  source_indicator_train     = source_all,
  X_test_control            = X_all,
  X_test_treat              = X_all,
  treatment_indicator_test   = trt_all,
  source_indicator_test      = source_all,
  outcome_type              = "continuous",
  decomposition             = "four-forest",
  treatment_coding          = "centered",
  number_of_trees_control   = n_trees_prog,
  number_of_trees_treat     = n_trees_treat,
  number_of_trees_deconf    = n_trees_deconf,
  number_of_trees_deviation = n_trees_dev,
  k_g                       = k_g,
  power                     = power,
  base                      = base,
  p_grow                    = p_grow,
  p_prune                   = p_prune,
  nu                        = nu,
  q                         = q,
  N_post                    = N_post,
  N_burn                    = N_burn,
  store_posterior_sample     = TRUE,
  verbose                   = FALSE
)

cat("  Done.\n\n")

# -------------------------------------------------------------------------
# 7. EXTRACT CATE ESTIMATES
# -------------------------------------------------------------------------
cate_rct   <- fit_rct$test_predictions_treat
cate_rwd    <- fit_rwd$test_predictions_treat
cate_three <- fit_three$test_predictions_treat
cate_four  <- fit_four$test_predictions_treat

cate_rct_samples   <- fit_rct$test_predictions_sample_treat
cate_rwd_samples    <- fit_rwd$test_predictions_sample_treat
cate_three_samples <- fit_three$test_predictions_sample_treat
cate_four_samples  <- fit_four$test_predictions_sample_treat

# -------------------------------------------------------------------------
# 8. EVALUATE PERFORMANCE
# -------------------------------------------------------------------------
model_names <- c("RCT only", "RWD only", "Three-forest", "Four-forest")
cate_list   <- list(cate_rct, cate_rwd, cate_three, cate_four)
sample_list <- list(cate_rct_samples, cate_rwd_samples,
                    cate_three_samples, cate_four_samples)

rmse_vec <- cov_vec <- width_vec <- numeric(4)

for (i in seq_len(4)) {
  ci <- apply(sample_list[[i]], 2, quantile, probs = c(0.025, 0.975))
  rmse_vec[i]  <- sqrt(mean((cate_list[[i]] - true_cate_all)^2))
  cov_vec[i]   <- mean(true_cate_all >= ci[1, ] & true_cate_all <= ci[2, ])
  width_vec[i] <- mean(ci[2, ] - ci[1, ])
}

results <- data.frame(
  Model    = model_names,
  RMSE     = round(rmse_vec, 4),
  Coverage = round(cov_vec, 4),
  Width    = round(width_vec, 4)
)
print(results, row.names = FALSE)
