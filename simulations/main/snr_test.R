################################################################################
# SNR test: does fusion beat RCT-only on OUR survival DGP once the unobserved
# baseline frailty (beta_U * U) is small relative to the learnable signal?
# Uncensored, Gaussian error for all fits (isolates the DGP / signal-to-noise).
# RMSE reported split by population (RCT / RWD / All), as in the example.
# Usage: Rscript snr_test.R [beta_U]   (default 0.3).  No seeds.
################################################################################
.libPaths(c("~/Rlib", .libPaths()))
suppressMessages(library(FusionForests))

args   <- commandArgs(trailingOnly = TRUE)
beta_U <- if (length(args) >= 1) as.numeric(args[1]) else 0.3
alpha_U <- 0.5; gamma_U <- 0.0; lambda_d <- 0.5      # S1, heterogeneous baseline drift
n1 <- 100L; n0 <- 400L; p <- 10L; sigma <- 0.5
NP <- 1500L; NB <- 1500L

true_tau <- function(X) 1 + 0.5 * X[,1] + (1/3) * (X[,2] > 0)
m0_sh    <- function(X) sin(pi*X[,1]) + X[,2]*X[,3] + (X[,4]^2 - 1/12)
g_dev    <- function(X) X[,1] - 0.5*X[,5]
slogT <- function(X,A,S,U) m0_sh(X) + (1-S)*lambda_d*g_dev(X) + beta_U*U +
  A*(true_tau(X) + gamma_U*U)

X1 <- matrix(runif(n1*p,-.5,.5),n1,p); U1 <- rnorm(n1); A1 <- rbinom(n1,1,.5)
lT1 <- slogT(X1,A1,1,U1) + rnorm(n1,0,sigma)
X0 <- matrix(runif(n0*p,-.5,.5),n0,p); U0 <- rnorm(n0)
A0 <- rbinom(n0,1,plogis(X0[,1] + X0[,2] + alpha_U*U0))
lT0 <- slogT(X0,A0,0,U0) + rnorm(n0,0,sigma)

X <- rbind(X1,X0); A <- c(A1,A0); S <- c(rep(1L,n1),rep(0L,n0))
y <- c(lT1,lT0); truth <- true_tau(X)
var_m0 <- var(m0_sh(X)); var_Unoise <- beta_U^2
cat(sprintf("beta_U=%.2f | Var(m0_sh)=%.2f  Var(beta_U*U)=%.2f  sigma^2=%.2f\n",
            beta_U, var_m0, var_Unoise, sigma^2))

# Fusion (four-forest, gaussian, both sources)
fit_f <- FusionForest(y=y, X_train_control=X, X_train_treat=X,
  treatment_indicator_train=A, source_indicator_train=S,
  X_test_control=X, X_test_treat=X, treatment_indicator_test=A,
  source_indicator_test=rep(1L,nrow(X)),
  number_of_trees_control=200, number_of_trees_treat=100,
  number_of_trees_deconf=50, number_of_trees_deviation=50,
  k_control=.5,k_treat=.5,k_deconf=.5,k_g=.5,
  error_dist="gaussian", N_post=NP, N_burn=NB, verbose=FALSE)
tf <- fit_f$test_predictions_treat

# RCT-only (dummy-row two-forest, gaussian)
src <- rep(1L,n1); src[n1] <- 0L
fit_r <- FusionForest(y=lT1, X_train_control=X1, X_train_treat=X1,
  treatment_indicator_train=A1, source_indicator_train=src,
  X_test_control=X, X_test_treat=X, treatment_indicator_test=A,
  source_indicator_test=rep(1L,nrow(X)),
  number_of_trees_deconf=1, number_of_trees_deviation=1,
  error_dist="gaussian", N_post=NP, N_burn=NB, verbose=FALSE)
tr <- fit_r$test_predictions_treat

rmse <- function(p,t) sqrt(mean((p-t)^2))
idxR <- 1:n1; idxW <- (n1+1):(n1+n0)
cat("\n         Fusion    RCT-only\n")
cat(sprintf("RCT     %.4f    %.4f\n", rmse(tf[idxR],truth[idxR]), rmse(tr[idxR],truth[idxR])))
cat(sprintf("RWD     %.4f    %.4f\n", rmse(tf[idxW],truth[idxW]), rmse(tr[idxW],truth[idxW])))
cat(sprintf("All     %.4f    %.4f\n", rmse(tf,truth), rmse(tr,truth)))
