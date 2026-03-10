# =============================================================================
# CATE estimation: FusionForest (RCT + OS) vs. T-learner (RCT only)
#
# Data-generating process
# -----------------------
#   y = m0(X) + A * tau(X) + A * (1-S) * c(X) + sigma * eps
#
#   m0(X)   = 2*X1 - X2 + 0.5*X3          (prognostic function)
#   tau(X)  = X1 + 0.5*X2^2               (true CATE — nonlinear)
#   c(X)    = -X1 + X3                     (confounding, OS treated only)
#
# Source coding: S = 1 (RCT), S = 0 (OS)
# =============================================================================

library(FusionForests)

set.seed(42)

# ---- Dimensions -------------------------------------------------------
n_rct  <- 200   # RCT sample size
n_os   <- 400   # OS sample size (larger, confounded)
n_test <- 500   # test set (true CATE known)
p      <- 5

# ---- Covariate draws --------------------------------------------------
X_rct  <- matrix(rnorm(n_rct * p), n_rct,  p)
X_os   <- matrix(rnorm(n_os  * p), n_os,   p)
X_test <- matrix(rnorm(n_test * p), n_test, p)

# ---- True functions ---------------------------------------------------
m0   <- function(X) 2*X[,1] - X[,2] + 0.5*X[,3]
tau  <- function(X) X[,1] + 0.5*X[,2]^2          # true CATE
conf <- function(X) -X[,1] + X[,3]               # confounding

true_cate_test <- tau(X_test)

# ---- Treatment assignment ---------------------------------------------
# RCT: balanced randomisation
A_rct <- rbinom(n_rct, 1, 0.5)

# OS: confounded — higher treatment probability for X1 > 0
ps_os <- plogis(X_os[,1])
A_os  <- rbinom(n_os, 1, ps_os)

# ---- Outcomes ---------------------------------------------------------
sigma <- 1.0

y_rct <- m0(X_rct) + A_rct * tau(X_rct)                          + rnorm(n_rct, 0, sigma)
y_os  <- m0(X_os)  + A_os  * tau(X_os)  + A_os * conf(X_os)     + rnorm(n_os,  0, sigma)

# ---- Combine into a single training set (RCT first) ------------------
X_train   <- rbind(X_rct, X_os)
A_train   <- c(A_rct, A_os)
S_train   <- c(rep(1L, n_rct), rep(0L, n_os))   # 1 = RCT, 0 = OS
y_train   <- c(y_rct, y_os)

# =============================================================================
# Model 1: FusionForest (RCT + OS)
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
  source_indicator_test     = rep(1L, n_test),   # treat test obs as RCT
  N_post = 1000, N_burn = 500,
  verbose = FALSE
)

cate_ff   <- fit_ff$test_predictions_treat
rmse_ff   <- sqrt(mean((cate_ff - true_cate_test)^2))

# =============================================================================
# Model 2: T-learner on RCT only (linear, to keep it simple)
#
# Fit two OLS models on treated and control RCT observations separately.
# CATE = mu_1(X) - mu_0(X)
# =============================================================================
cat("Fitting T-learner (RCT only, linear)...\n")

rct_df  <- as.data.frame(X_rct)
names(rct_df) <- paste0("V", seq_len(p))
rct_df$y <- y_rct
rct_df$A <- A_rct

fit_t1   <- lm(y ~ ., data = rct_df[rct_df$A == 1, setdiff(names(rct_df), "A")])
fit_t0   <- lm(y ~ ., data = rct_df[rct_df$A == 0, setdiff(names(rct_df), "A")])

test_df        <- as.data.frame(X_test)
names(test_df) <- paste0("V", seq_len(p))

cate_tl  <- predict(fit_t1, newdata = test_df) - predict(fit_t0, newdata = test_df)
rmse_tl  <- sqrt(mean((cate_tl - true_cate_test)^2))

# =============================================================================
# Results
# =============================================================================
cat("\n===== CATE RMSE on test set =====\n")
cat(sprintf("  FusionForest (RCT + OS) : %.4f\n", rmse_ff))
cat(sprintf("  T-learner   (RCT only)  : %.4f\n", rmse_tl))
cat(sprintf("  Ratio (TL / FF)         : %.2fx\n", rmse_tl / rmse_ff))
