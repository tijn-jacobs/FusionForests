# =============================================================================
# Compare three residual-distribution priors in FusionForest:
#   error_dist = "gaussian"   — single normal residual (baseline)
#   error_dist = "shared_dp"  — one DP mixture pooled across sources
#   error_dist = "source_dp"  — independent DP mixtures per source
#
# Metrics
#   1. Wall-clock runtime per fit (sec)
#   2. CATE (HTE) RMSE on three populations: RCT, RWD, All
#
# DGP — designed so the DP options have something to learn:
#   m0(X)   = 2*X1 - X2 + 0.5*X3
#   tau(X)  = X1 + 0.5*X2^2
#   c(U)    = -1/2 U1 + 1/4 U2          (unobserved confounding, RWD only)
#
#   RCT residuals: Gaussian, N(0, 0.5^2)
#   RWD residuals: BIMODAL, 0.5 N(-1, 0.2^2) + 0.5 N(+1, 0.2^2), mean-zero
#                  source_dp should pick this up; gaussian must absorb it
#                  into an inflated sigma.
#
# Source coding: S = 1 (RCT), S = 0 (RWD)
# =============================================================================

library(FusionForests)

# ---- Dimensions ------------------------------------------------------
n_rct <- 100
n_rwd  <- 200
p     <- 5     # observed covariates
q     <- 2     # unobserved confounders (RWD only)

# ---- MCMC settings ---------------------------------------------------
N_post <- 2500
N_burn <- 1500

# ---- Covariates ------------------------------------------------------
X_rct <- matrix(runif(n_rct * p), n_rct, p)
X_rwd  <- matrix(runif(n_rwd  * p), n_rwd,  p)
U_rwd  <- matrix(rnorm(n_rwd * q), n_rwd, q)

# ---- True functions --------------------------------------------------
m0   <- function(X) 2*X[,1] - X[,2] + 0.5*X[,3]
tau  <- function(X) X[,1] + 0.5 * X[,2]^2
conf <- function(U) -0.5*U[,1] + 0.25*U[,2]

true_cate_rct <- tau(X_rct)
true_cate_rwd  <- tau(X_rwd)
true_cate_all <- c(true_cate_rct, true_cate_rwd)

# ---- Treatment assignment --------------------------------------------
A_rct <- rbinom(n_rct, 1, 0.5)
A_rwd  <- rbinom(n_rwd,  1, plogis(X_rwd[,1] + U_rwd[,1] + U_rwd[,2]))

# ---- Residuals -------------------------------------------------------
# RCT: standard Gaussian.
eps_rct <- rnorm(n_rct, 0, 0.5)

# RWD: 50/50 mixture of N(-1, 0.2^2) and N(+1, 0.2^2). Mean zero by symmetry.
mix_label <- rbinom(n_rwd, 1, 0.5)
eps_rwd    <- ifelse(mix_label == 1,
                    rnorm(n_rwd,  1, 0.2),
                    rnorm(n_rwd, -1, 0.2))

# ---- Outcomes --------------------------------------------------------
y_rct <- m0(X_rct) + A_rct * tau(X_rct) + eps_rct
y_rwd  <- m0(X_rwd)  + A_rwd  * tau(X_rwd) + A_rwd * conf(U_rwd) + eps_rwd

# ---- Combined training set -------------------------------------------
X_train <- rbind(X_rct, X_rwd)
A_train <- c(A_rct, A_rwd)
S_train <- c(rep(1L, n_rct), rep(0L, n_rwd))
y_train <- c(y_rct, y_rwd)

idx_rct <- 1:n_rct
idx_rwd  <- (n_rct + 1):(n_rct + n_rwd)

# ---- Fit-and-time helper --------------------------------------------
fit_one <- function(error_dist) {
  cat(sprintf("\n--- Fitting FusionForest [error_dist = %s] ---\n", error_dist))
  t0 <- proc.time()[["elapsed"]]
  fit <- FusionForest(
    y                         = y_train,
    X_train_control           = X_train,
    X_train_treat             = X_train,
    treatment_indicator_train = A_train,
    source_indicator_train    = S_train,
    outcome_type              = "continuous",
    decomposition             = "three-forest",
    error_dist                = error_dist,
    error_truncation_K        = 50L,
    error_atom_scale          = 0.5,
    error_mass_init           = 1.0,
    N_post                    = N_post,
    N_burn                    = N_burn,
    verbose                   = FALSE
  )
  elapsed <- proc.time()[["elapsed"]] - t0

  cate_hat <- fit$train_predictions_treat
  list(
    fit         = fit,
    elapsed_sec = elapsed,
    rmse_rct    = sqrt(mean((cate_hat[idx_rct] - true_cate_rct)^2)),
    rmse_rwd     = sqrt(mean((cate_hat[idx_rwd]  - true_cate_rwd )^2)),
    rmse_all    = sqrt(mean((cate_hat          - true_cate_all)^2)),
    sigma_mean  = mean(fit$sigma)
  )
}

results <- list(
  gaussian  = fit_one("gaussian"),
  shared_dp = fit_one("shared_dp"),
  source_dp = fit_one("source_dp")
)

# ---- Summary table ---------------------------------------------------
tab <- data.frame(
  error_dist = names(results),
  runtime_s  = vapply(results, function(r) r$elapsed_sec, numeric(1)),
  sigma_mean = vapply(results, function(r) r$sigma_mean,  numeric(1)),
  RMSE_RCT   = vapply(results, function(r) r$rmse_rct,    numeric(1)),
  RMSE_RWD   = vapply(results, function(r) r$rmse_rwd,     numeric(1)),
  RMSE_All   = vapply(results, function(r) r$rmse_all,    numeric(1)),
  row.names  = NULL,
  check.names = FALSE
)

cat("\n========== CATE RMSE & runtime ==========\n")
print(tab, row.names = FALSE, digits = 4)

# ---- DP diagnostics --------------------------------------------------
cat("\n========== DP diagnostics ==========\n")
shared_mass <- results$shared_dp$fit$dp_mass
if (!is.null(shared_mass)) {
  cat(sprintf("shared_dp:   posterior mean concentration alpha = %.3f\n",
              mean(shared_mass[[1]])))
}
source_mass <- results$source_dp$fit$dp_mass
if (!is.null(source_mass)) {
  # group 0 = RWD (S=0), group 1 = RCT (S=1) by MixtureDP convention
  cat(sprintf("source_dp:   posterior mean alpha_RWD = %.3f (group 0)\n",
              mean(source_mass[[1]])))
  cat(sprintf("source_dp:   posterior mean alpha_RCT = %.3f (group 1)\n",
              mean(source_mass[[2]])))
}

# Average number of occupied components per group (over MCMC draws).
occupied <- function(mp_mat, threshold = 1 / (2 * ncol(mp_mat))) {
  mean(rowSums(mp_mat > threshold))
}
shared_mp <- results$shared_dp$fit$dp_mix_prop
if (!is.null(shared_mp)) {
  cat(sprintf("shared_dp:   avg occupied clusters       = %.2f / %d\n",
              occupied(shared_mp[[1]]), ncol(shared_mp[[1]])))
}
source_mp <- results$source_dp$fit$dp_mix_prop
if (!is.null(source_mp)) {
  cat(sprintf("source_dp:   avg occupied RWD clusters   = %.2f / %d\n",
              occupied(source_mp[[1]]), ncol(source_mp[[1]])))
  cat(sprintf("source_dp:   avg occupied RCT clusters   = %.2f / %d\n",
              occupied(source_mp[[2]]), ncol(source_mp[[2]])))
}
