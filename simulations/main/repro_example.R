################################################################################
# Reproduction of examples/cate_rmse_comparison.R in this environment.
# Goal: confirm the "fusion beats RCT-only" result the older file shows, so we
# have a known-good reference DGP to compare the main sim against.
#
# Faithful to the original: continuous outcome, sigma = 0.5, p = 5 observed
# covariates, q = 2 UNOBSERVED confounders that enter the outcome ONLY through
# the treated RWD arm (A * conf(U)); the baseline m0 is a clean function of X.
# Fusion = FusionForest defaults (four-forest, gaussian, k_g = 0.25, 200 trees).
# RCT-only = CausalShrinkageForest (genuine two-forest BCF) on the RCT.
# No seed (single-run inspection).
################################################################################
.libPaths(c("~/Rlib", .libPaths()))
suppressMessages(library(FusionForests))

n_rct <- 100; n_rwd <- 400; p <- 5; q <- 2
N_post <- 2500; N_burn <- 1500

X_rct <- matrix(runif(n_rct * p), n_rct, p)
X_rwd <- matrix(runif(n_rwd * p), n_rwd, p)
U_rwd <- matrix(rnorm(n_rwd * q), n_rwd, q)          # unobserved, RWD only

m0   <- function(X) 2*X[,1] - X[,2] + 0.5*X[,3]
tau  <- function(X) X[,1] + 0.5 * X[,2]^2            # true CATE
conf <- function(U) -0.5*U[,1] + 0.25*U[,2]          # confounding via unobserved U

A_rct <- rbinom(n_rct, 1, 0.5)
A_rwd <- rbinom(n_rwd, 1, plogis(X_rwd[,1] + U_rwd[,1] + U_rwd[,2]))

sigma <- 0.5
y_rct <- m0(X_rct) + A_rct * tau(X_rct)                         + rnorm(n_rct, 0, sigma)
y_rwd <- m0(X_rwd) + A_rwd * tau(X_rwd) + A_rwd * conf(U_rwd)   + rnorm(n_rwd, 0, sigma)

X_train <- rbind(X_rct, X_rwd); A_train <- c(A_rct, A_rwd)
S_train <- c(rep(1L, n_rct), rep(0L, n_rwd)); y_train <- c(y_rct, y_rwd)
true_cate_rct <- tau(X_rct); true_cate_rwd <- tau(X_rwd)
true_cate_all <- c(true_cate_rct, true_cate_rwd)

cat("Fitting FusionForest (RCT + RWD)...\n")
fit_ff <- FusionForest(
  y = y_train, X_train_control = X_train, X_train_treat = X_train,
  treatment_indicator_train = A_train, source_indicator_train = S_train,
  N_post = N_post, N_burn = N_burn, verbose = FALSE)
cate_ff_rct <- fit_ff$train_predictions_treat[1:n_rct]
cate_ff_rwd <- fit_ff$train_predictions_treat[(n_rct+1):(n_rct+n_rwd)]
cate_ff_all <- fit_ff$train_predictions_treat

# RCT-only via FusionForest dummy-row two-forest (CausalShrinkageForest's C++
# backend is not compiled in this build).  Same fitting machinery as fusion.
cat("Fitting RCT-only (FusionForest, dummy-row two-forest)...\n")
src <- rep(1L, n_rct); src[n_rct] <- 0L
fit_rct <- FusionForest(
  y = y_rct, X_train_control = X_rct, X_train_treat = X_rct,
  treatment_indicator_train = A_rct, source_indicator_train = src,
  X_test_control = X_train, X_test_treat = X_train,
  treatment_indicator_test = A_train, source_indicator_test = rep(1L, n_rct + n_rwd),
  number_of_trees_deconf = 1, number_of_trees_deviation = 1,
  N_post = N_post, N_burn = N_burn, verbose = FALSE)
cate_csf_rct <- fit_rct$test_predictions_treat[1:n_rct]
cate_csf_rwd <- fit_rct$test_predictions_treat[(n_rct+1):(n_rct+n_rwd)]
cate_csf_all <- fit_rct$test_predictions_treat

rmse <- function(p, t) sqrt(mean((p - t)^2))
cat("\n===== CATE RMSE by population =====\n")
cat(sprintf("%-12s %12s %12s\n", "Population", "FusionForest", "CSF (RCT)"))
cat(sprintf("%-12s %12.4f %12.4f\n", "RCT", rmse(cate_ff_rct, true_cate_rct), rmse(cate_csf_rct, true_cate_rct)))
cat(sprintf("%-12s %12.4f %12.4f\n", "RWD", rmse(cate_ff_rwd, true_cate_rwd), rmse(cate_csf_rwd, true_cate_rwd)))
cat(sprintf("%-12s %12.4f %12.4f\n", "All", rmse(cate_ff_all, true_cate_all), rmse(cate_csf_all, true_cate_all)))
