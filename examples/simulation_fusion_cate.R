## Simulation study: CATE estimation across FusionForest decompositions
##
## Replicates fusion_vs_rct_hte.R over n_rep runs, comparing:
##   1. Two-forest BCF (RCT only)
##   2. Two-forest BCF (RWD only, naive)
##   3. Three-forest FusionForest (RCT + RWD)
##   4. Four-forest FusionForest  (RCT + RWD)
##
## Metrics (training sample): RMSE of CATE, 95% CI coverage, CI width
## Parallelised with foreach + doParallel over 6 cores.

setwd("~/Library/CloudStorage/OneDrive-VrijeUniversiteitAmsterdam/Documents/GitHub/FusionForests")

library(FusionForests)
library(foreach)
library(doParallel)

# -------------------------------------------------------------------------
# 1. SETTINGS
# -------------------------------------------------------------------------
n_rep   <- 36
n_cores <- 6

# DGP — matches examples/fusion_vs_rct_hte.R
n_rct      <- 100
n_rwd      <- 100
p          <- 10
sigma_true <- 2

mu0       <- function(X) 2 * X[, 1] - X[, 3]^2
true_cate <- function(X) 1.5 + 2 * X[, 1]

# Per-forest-type tree counts
n_trees_prog   <- 200
n_trees_treat  <- 100
n_trees_deconf <- 200
n_trees_dev    <- 50

# Leaf-prior scale for deviation forest (omega_g = k_g / sqrt(m_g))
k_g <- 0.25

# MCMC
N_post <- 2000
N_burn <- 1000

# Shared tree-prior and error-variance parameters
power   <- 2.0
base    <- 0.95
p_grow  <- 0.4
p_prune <- 0.4
nu      <- 3
q_sig   <- 0.90

# -------------------------------------------------------------------------
# 2. SINGLE-REPLICATION FUNCTION
# -------------------------------------------------------------------------
run_one_rep <- function(rep_id) {

  set.seed(rep_id)

  # --- Generate data -----------------------------------------------------
  X_rct   <- matrix(runif(n_rct * p), nrow = n_rct)
  trt_rct <- rbinom(n_rct, 1, 0.5)
  y_rct   <- mu0(X_rct) + true_cate(X_rct) * trt_rct +
             rnorm(n_rct, sd = sigma_true)

  X_rwd   <- matrix(runif(n_rwd * p), nrow = n_rwd)
  U_rwd   <- rnorm(n_rwd)
  trt_rwd <- rbinom(n_rwd, 1, plogis(0.8 * U_rwd))
  y_rwd   <- mu0(X_rwd) + 1.0 * U_rwd +
             true_cate(X_rwd) * trt_rwd +
             rnorm(n_rwd, sd = sigma_true)

  X_all      <- rbind(X_rct, X_rwd)
  y_all      <- c(y_rct, y_rwd)
  trt_all    <- c(trt_rct, trt_rwd)
  source_all <- c(rep(1L, n_rct), rep(0L, n_rwd))

  truth_rct <- true_cate(X_rct)
  truth_rwd <- true_cate(X_rwd)
  truth_all <- true_cate(X_all)

  # --- Metrics helper ----------------------------------------------------
  eval_cate <- function(cate_hat, cate_samples, truth) {
    rmse <- sqrt(mean((cate_hat - truth)^2))
    bias <- mean(cate_hat - truth)
    ci   <- apply(cate_samples, 2, quantile, probs = c(0.025, 0.975))
    cov  <- mean(truth >= ci[1, ] & truth <= ci[2, ])
    wid  <- mean(ci[2, ] - ci[1, ])
    c(rmse = rmse, bias = bias, coverage = cov, width = wid)
  }

  # --- Model 1: Two-forest BCF (RCT only) -------------------------------
  src <- rep(1L, n_rct); src[n_rct] <- 0L
  fit <- FusionForest(
    y                         = y_rct,
    X_train_control           = X_rct,
    X_train_treat             = X_rct,
    treatment_indicator_train = trt_rct,
    source_indicator_train    = src,
    outcome_type              = "continuous",
    decomposition             = "four-forest",
    treatment_coding          = "centered",
    number_of_trees_control   = n_trees_prog,
    number_of_trees_treat     = n_trees_treat,
    number_of_trees_deconf    = 1,
    power = power, base = base,
    p_grow = p_grow, p_prune = p_prune,
    nu = nu, q = q_sig,
    N_post = N_post, N_burn = N_burn,
    store_posterior_sample = TRUE, verbose = FALSE
  )
  m1 <- eval_cate(fit$train_predictions_treat,
                  fit$train_predictions_sample_treat, truth_rct)

  # --- Model 2: Two-forest BCF (RWD only, naive) ------------------------
  src <- rep(1L, n_rwd); src[n_rwd] <- 0L
  fit <- FusionForest(
    y                         = y_rwd,
    X_train_control           = X_rwd,
    X_train_treat             = X_rwd,
    treatment_indicator_train = trt_rwd,
    source_indicator_train    = src,
    outcome_type              = "continuous",
    decomposition             = "four-forest",
    treatment_coding          = "centered",
    number_of_trees_control   = n_trees_prog,
    number_of_trees_treat     = n_trees_treat,
    number_of_trees_deconf    = 1,
    power = power, base = base,
    p_grow = p_grow, p_prune = p_prune,
    nu = nu, q = q_sig,
    N_post = N_post, N_burn = N_burn,
    store_posterior_sample = TRUE, verbose = FALSE
  )
  m2 <- eval_cate(fit$train_predictions_treat,
                  fit$train_predictions_sample_treat, truth_rwd)

  # --- Model 3: Three-forest FusionForest (RCT + RWD) -------------------
  fit <- FusionForest(
    y                         = y_all,
    X_train_control           = X_all,
    X_train_treat             = X_all,
    treatment_indicator_train = trt_all,
    source_indicator_train    = source_all,
    outcome_type              = "continuous",
    decomposition             = "four-forest",
    treatment_coding          = "centered",
    number_of_trees_control   = n_trees_prog,
    number_of_trees_treat     = n_trees_treat,
    number_of_trees_deconf    = n_trees_deconf,
    power = power, base = base,
    p_grow = p_grow, p_prune = p_prune,
    nu = nu, q = q_sig,
    N_post = N_post, N_burn = N_burn,
    store_posterior_sample = TRUE, verbose = FALSE
  )
  m3 <- eval_cate(fit$train_predictions_treat,
                  fit$train_predictions_sample_treat, truth_all)

  # --- Model 4: Four-forest FusionForest (RCT + RWD) --------------------
  fit <- FusionForest(
    y                         = y_all,
    X_train_control           = X_all,
    X_train_treat             = X_all,
    treatment_indicator_train = trt_all,
    source_indicator_train    = source_all,
    outcome_type              = "continuous",
    decomposition             = "four-forest",
    treatment_coding          = "centered",
    number_of_trees_control   = n_trees_prog,
    number_of_trees_treat     = n_trees_treat,
    number_of_trees_deconf    = n_trees_deconf,
    number_of_trees_deviation = n_trees_dev,
    k_g = k_g,
    power = power, base = base,
    p_grow = p_grow, p_prune = p_prune,
    nu = nu, q = q_sig,
    N_post = N_post, N_burn = N_burn,
    store_posterior_sample = TRUE, verbose = FALSE
  )
  m4 <- eval_cate(fit$train_predictions_treat,
                  fit$train_predictions_sample_treat, truth_all)

  c(rct = m1, rwd = m2, three = m3, four = m4)
}

# -------------------------------------------------------------------------
# 3. RUN SIMULATION (foreach + doParallel)
# -------------------------------------------------------------------------
cl <- makeCluster(n_cores)
registerDoParallel(cl)
on.exit(stopCluster(cl), add = TRUE)

# Load the dev version of FusionForests on each worker so changes made via
# devtools::load_all() in the main session are available without reinstalling.
pkg_path <- getwd()
clusterCall(cl, function(p) {
  suppressMessages(devtools::load_all(p, quiet = TRUE))
}, p = pkg_path)

cat(sprintf("Running %d replications on %d cores...\n", n_rep, n_cores))
t0 <- proc.time()

res_list <- foreach(rep_id = seq_len(n_rep)) %dopar% run_one_rep(rep_id)

elapsed <- (proc.time() - t0)["elapsed"]
cat(sprintf("Done in %.1f s (%.1f s / rep).\n\n", elapsed, elapsed / n_rep))

res <- do.call(rbind, res_list)   # n_rep x 16

# -------------------------------------------------------------------------
# 4. SUMMARY TABLE
# -------------------------------------------------------------------------
models   <- c("RCT only", "RWD only", "Three-forest", "Four-forest")
prefixes <- c("rct", "rwd", "three", "four")
met_keys <- c("rmse", "bias", "coverage", "width")
met_labs <- c("RMSE", "Bias", "Coverage", "CI Width")

summary_tab <- data.frame(Model = models)
for (k in seq_along(met_keys)) {
  cols <- paste0(prefixes, ".", met_keys[k])
  vals <- res[, cols, drop = FALSE]
  summary_tab[[paste0(met_labs[k], "_mean")]] <- round(colMeans(vals), 4)
  summary_tab[[paste0(met_labs[k], "_sd")]]   <- round(apply(vals, 2, sd), 4)
}

cat(sprintf("Simulation: %d reps | n_rct = %d, n_rwd = %d, p = %d\n",
            n_rep, n_rct, n_rwd, p))
print(summary_tab, row.names = FALSE)
