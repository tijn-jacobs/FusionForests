## Simulation: compare residual-distribution priors in FusionForest.
##
## Goal: verify that source_hdp_scale (HDP shared atoms + per-source sigma)
## recovers the data-generating per-source error scales and matches the
## CATE-estimation quality of the best competing prior on each scenario.
##
## Three scenarios, varying what is true about the per-source residuals:
##
##   A. SHARED  -- sigma_RCT = sigma_RWD = sigma_true; Gaussian errors.
##                 All priors should perform similarly (null case).
##
##   B. SCALE   -- sigma_RCT != sigma_RWD; Gaussian errors otherwise.
##                 source_dp_scale and source_hdp_scale should win on sigma
##                 recovery; equal-sigma modes pay a price.
##
##   C. HDP-SCALE -- sigma_RCT != sigma_RWD AND residuals are a 2-component
##                   skewed mixture with shared atom locations but different
##                   mixture weights per source.  source_hdp_scale should be
##                   the only mode that recovers all three features.
##
## For each scenario / mode / rep we record:
##   * CATE RMSE, bias, 95% coverage, mean CI width
##   * Per-source sigma posterior mean (for *_scale modes)
##   * Per-source residual KS distance against the true source-specific
##     residual sample (i.e. predictive replication of residuals)
##
## Defaults are small so the script runs in minutes on a laptop; scale up
## n_rep and N_post for a publication-grade run.

setwd("~/Library/CloudStorage/OneDrive-VrijeUniversiteitAmsterdam/Documents/GitHub/FusionForests")

suppressMessages({
  library(FusionForests)
  library(foreach)
  library(doParallel)
})

# -------------------------------------------------------------------------
# 1. SETTINGS
# -------------------------------------------------------------------------
n_rep   <- 3
n_cores <- 4

n_rct   <- 150
n_rwd   <- 300
p       <- 5

mu0       <- function(X) 2 * X[, 1] - X[, 3]^2
true_cate <- function(X) 1.5 + 2 * X[, 1]

# Scenario parameters
sigma_shared <- 1.0
sigma_rct    <- 0.5
sigma_rwd    <- 1.5

# Mixture atoms used by scenario C (in standardised-residual units of sd_shared)
atoms_C        <- c(-1.5, 1.0)
weights_rct_C  <- c(0.20, 0.80)   # RCT mostly upper atom
weights_rwd_C  <- c(0.75, 0.25)   # RWD mostly lower atom

# MCMC
N_post <- 1500
N_burn <- 1500

modes <- c("gaussian", "shared_dp", "source_dp",
           "source_dp_scale", "source_hdp", "source_hdp_scale")

# -------------------------------------------------------------------------
# 2. DGPs
# -------------------------------------------------------------------------
draw_errors <- function(n, scenario, source_label) {
  if (scenario == "A") {
    return(rnorm(n, sd = sigma_shared))
  }
  if (scenario == "B") {
    sd_s <- if (source_label == "rct") sigma_rct else sigma_rwd
    return(rnorm(n, sd = sd_s))
  }
  if (scenario == "C") {
    sd_s <- if (source_label == "rct") sigma_rct else sigma_rwd
    w    <- if (source_label == "rct") weights_rct_C else weights_rwd_C
    k    <- sample.int(2, n, replace = TRUE, prob = w)
    mu_centered <- atoms_C - sum(w * atoms_C)   # mean-zero shift
    return(mu_centered[k] + rnorm(n, sd = sd_s))
  }
  stop("unknown scenario")
}

make_data <- function(scenario, seed) {
  set.seed(seed)
  X_rct   <- matrix(runif(n_rct * p), n_rct, p)
  trt_rct <- rbinom(n_rct, 1, 0.5)
  eps_rct <- draw_errors(n_rct, scenario, "rct")
  y_rct   <- mu0(X_rct) + true_cate(X_rct) * trt_rct + eps_rct

  X_rwd   <- matrix(runif(n_rwd * p), n_rwd, p)
  U_rwd   <- rnorm(n_rwd)
  trt_rwd <- rbinom(n_rwd, 1, plogis(0.8 * U_rwd))
  eps_rwd <- draw_errors(n_rwd, scenario, "rwd")
  y_rwd   <- mu0(X_rwd) + 0.8 * U_rwd + true_cate(X_rwd) * trt_rwd + eps_rwd

  list(
    X      = rbind(X_rct, X_rwd),
    y      = c(y_rct, y_rwd),
    trt    = c(trt_rct, trt_rwd),
    src    = c(rep(1L, n_rct), rep(0L, n_rwd)),
    truth  = true_cate(rbind(X_rct, X_rwd)),
    eps_rct = eps_rct,
    eps_rwd = eps_rwd
  )
}

# -------------------------------------------------------------------------
# 3. METRICS
# -------------------------------------------------------------------------
eval_cate <- function(cate_hat, cate_samples, truth) {
  rmse <- sqrt(mean((cate_hat - truth)^2))
  bias <- mean(cate_hat - truth)
  ci   <- apply(cate_samples, 2, quantile, probs = c(0.025, 0.975))
  cov  <- mean(truth >= ci[1, ] & truth <= ci[2, ])
  wid  <- mean(ci[2, ] - ci[1, ])
  c(rmse = rmse, bias = bias, coverage = cov, width = wid)
}

eval_sigma <- function(fit, scenario) {
  # Returns posterior-mean sigma_rwd and sigma_rct (NA if mode doesn't
  # expose per-source sigma).  Group 0 = RWD, group 1 = RCT.
  truth_rwd <- if (scenario == "A") sigma_shared else sigma_rwd
  truth_rct <- if (scenario == "A") sigma_shared else sigma_rct
  if (!is.null(fit$dp_sigma_g)) {
    sg_rwd <- mean(fit$dp_sigma_g[[1]])
    sg_rct <- mean(fit$dp_sigma_g[[2]])
  } else {
    sg_rwd <- mean(fit$sigma)
    sg_rct <- mean(fit$sigma)
  }
  c(sig_rwd_hat = sg_rwd, sig_rct_hat = sg_rct,
    sig_rwd_err = sg_rwd - truth_rwd, sig_rct_err = sg_rct - truth_rct)
}

eval_resid_fit <- function(fit, data) {
  # Posterior mean of residuals e_i = y_i - total_predictions[i]; compare
  # source-specific empirical CDF to the truth.
  resid <- data$y - fit$train_predictions
  ks_rct <- ks.test(resid[data$src == 1L], data$eps_rct,
                    exact = FALSE)$statistic
  ks_rwd <- ks.test(resid[data$src == 0L], data$eps_rwd,
                    exact = FALSE)$statistic
  c(ks_rct = unname(ks_rct), ks_rwd = unname(ks_rwd))
}

# Mixture-prior diagnostics: effective and occupied component counts per
# source, per-source concentration M_s, top-level gamma (HDP), |mu_0 - mu_1|
# (HDP), plus tree-update acceptance ratios.  NA where the mode doesn't
# expose the corresponding state.  shared_dp duplicates the single shared
# state into both per-source slots so columns stay comparable.
eval_diagnostics <- function(fit, eps = 0.01) {
  out <- c(
    K_eff_rwd       = NA_real_,
    K_eff_rct       = NA_real_,
    n_occupied_rwd  = NA_real_,
    n_occupied_rct  = NA_real_,
    n_shared        = NA_real_,
    M_rwd           = NA_real_,
    M_rct           = NA_real_,
    gamma_top       = NA_real_,
    mu_diff         = NA_real_,
    accept_control  = unname(fit$acceptance_ratio_control),
    accept_treat    = unname(fit$acceptance_ratio_treat),
    accept_deconf   = unname(fit$acceptance_ratio_deconf)
  )
  if (is.null(fit$dp_mix_prop)) return(out)

  K_eff <- function(P) mean(1 / rowSums(P^2))
  n_occ <- function(P) mean(rowSums(P > eps))

  if (length(fit$dp_mix_prop) == 1L) {
    P <- fit$dp_mix_prop[[1]]
    out["K_eff_rwd"]      <- K_eff(P)
    out["K_eff_rct"]      <- K_eff(P)
    out["n_occupied_rwd"] <- n_occ(P)
    out["n_occupied_rct"] <- n_occ(P)
    out["M_rwd"]          <- mean(fit$dp_mass[[1]])
    out["M_rct"]          <- mean(fit$dp_mass[[1]])
  } else {
    P_rwd <- fit$dp_mix_prop[[1]]
    P_rct <- fit$dp_mix_prop[[2]]
    out["K_eff_rwd"]      <- K_eff(P_rwd)
    out["K_eff_rct"]      <- K_eff(P_rct)
    out["n_occupied_rwd"] <- n_occ(P_rwd)
    out["n_occupied_rct"] <- n_occ(P_rct)
    out["M_rwd"]          <- mean(fit$dp_mass[[1]])
    out["M_rct"]          <- mean(fit$dp_mass[[2]])
    out["n_shared"]       <- mean(rowSums((P_rwd > eps) & (P_rct > eps)))
  }

  if (!is.null(fit$dp_gamma)) {
    out["gamma_top"] <- mean(fit$dp_gamma)
    out["mu_diff"]   <- mean(abs(fit$dp_mu_g[[1]] - fit$dp_mu_g[[2]]))
  }

  out
}

# -------------------------------------------------------------------------
# 4. ONE FIT
# -------------------------------------------------------------------------
fit_one <- function(data, error_dist) {
  FusionForest(
    y                         = data$y,
    X_train_control           = data$X,
    X_train_treat             = data$X,
    treatment_indicator_train = data$trt,
    source_indicator_train    = data$src,
    outcome_type              = "continuous",
    decomposition             = "four-forest",
    treatment_coding          = "centered",
    number_of_trees_control   = 200,
    number_of_trees_treat     = 100,
    number_of_trees_deconf    = 200,
    N_post = N_post, N_burn = N_burn,
    error_dist               = error_dist,
    store_posterior_sample   = TRUE,
    verbose                  = FALSE
  )
}

# Ordered metric names used for both success and failure rows so the row
# schema is identical across modes.
metric_cols <- c(
  "rmse", "bias", "coverage", "width",
  "sig_rwd_hat", "sig_rct_hat", "sig_rwd_err", "sig_rct_err",
  "ks_rct", "ks_rwd",
  "K_eff_rwd", "K_eff_rct", "n_occupied_rwd", "n_occupied_rct",
  "n_shared", "M_rwd", "M_rct", "gamma_top", "mu_diff",
  "accept_control", "accept_treat", "accept_deconf"
)

# -------------------------------------------------------------------------
# 5. ONE REPLICATION
# -------------------------------------------------------------------------
run_one_rep <- function(rep_id, scenario) {
  data <- make_data(scenario, seed = 1000 * which(scenario == c("A","B","C")) + rep_id)
  out <- list()
  for (m in modes) {
    fit <- tryCatch(fit_one(data, m), error = function(e) NULL)
    if (is.null(fit)) {
      vals <- setNames(rep(NA_real_, length(metric_cols)), metric_cols)
    } else {
      metrics <- c(
        eval_cate(fit$train_predictions_treat,
                  fit$train_predictions_sample_treat, data$truth),
        eval_sigma(fit, scenario),
        eval_resid_fit(fit, data),
        eval_diagnostics(fit)
      )
      vals <- setNames(rep(NA_real_, length(metric_cols)), metric_cols)
      vals[names(metrics)] <- unname(metrics)
    }
    out[[m]] <- vals
  }
  do.call(rbind, lapply(modes, function(m) c(scenario = scenario, mode = m,
                                              as.list(out[[m]]))))
}

# -------------------------------------------------------------------------
# 6. RUN
# -------------------------------------------------------------------------
cl <- makeCluster(n_cores)
registerDoParallel(cl)
on.exit(stopCluster(cl), add = TRUE)

pkg_path <- getwd()
clusterCall(cl, function(p) suppressMessages(devtools::load_all(p, quiet = TRUE)),
            p = pkg_path)
clusterExport(cl, c("n_rct","n_rwd","p","sigma_shared","sigma_rct","sigma_rwd",
                    "atoms_C","weights_rct_C","weights_rwd_C","N_post","N_burn",
                    "modes","mu0","true_cate","draw_errors","make_data",
                    "eval_cate","eval_sigma","eval_resid_fit","eval_diagnostics",
                    "fit_one","metric_cols"))

cat(sprintf("Running %d reps x 3 scenarios x %d modes on %d cores...\n",
            n_rep, length(modes), n_cores))
t0 <- proc.time()

grid <- expand.grid(scenario = c("A","B","C"), rep_id = seq_len(n_rep),
                    stringsAsFactors = FALSE)
res_list <- foreach(i = seq_len(nrow(grid))) %dopar% {
  run_one_rep(grid$rep_id[i], grid$scenario[i])
}
elapsed <- (proc.time() - t0)["elapsed"]
cat(sprintf("Done in %.1f s.\n\n", elapsed))

res <- do.call(rbind, lapply(res_list, as.data.frame))
res[] <- lapply(res, function(x) if (is.list(x)) unlist(x) else x)
for (col in setdiff(names(res), c("scenario","mode")))
  res[[col]] <- as.numeric(res[[col]])

# -------------------------------------------------------------------------
# 7. SUMMARY
# -------------------------------------------------------------------------
summary_tbl <- aggregate(
  cbind(rmse, bias, coverage, width,
        sig_rwd_hat, sig_rct_hat, sig_rwd_err, sig_rct_err,
        ks_rct, ks_rwd,
        K_eff_rwd, K_eff_rct, n_occupied_rwd, n_occupied_rct,
        n_shared, M_rwd, M_rct, gamma_top, mu_diff,
        accept_control, accept_treat, accept_deconf) ~ scenario + mode,
  data = res, FUN = function(x) mean(x, na.rm = TRUE), na.action = na.pass)

cat("\n=== Mean across reps (scenario x mode): CATE + residual fit ===\n")
print(summary_tbl[order(summary_tbl$scenario, summary_tbl$mode),
                  c("scenario","mode","rmse","bias","coverage","width",
                    "sig_rwd_err","sig_rct_err","ks_rwd","ks_rct")],
      row.names = FALSE, digits = 3)

cat("\n=== Mixture diagnostics (scenario x mode) ===\n")
print(summary_tbl[order(summary_tbl$scenario, summary_tbl$mode),
                  c("scenario","mode",
                    "K_eff_rwd","K_eff_rct","n_occupied_rwd","n_occupied_rct",
                    "n_shared","M_rwd","M_rct","gamma_top","mu_diff")],
      row.names = FALSE, digits = 3)

saveRDS(list(per_rep = res, summary = summary_tbl),
        file = "examples/sim_error_dist_results.rds")
cat("\nSaved: examples/sim_error_dist_results.rds\n")

# Quick sanity prints
cat("\n=== Expected: source_hdp_scale wins on sigma_rwd_err & sigma_rct_err",
    "in scenarios B and C ===\n")
sub <- summary_tbl[summary_tbl$scenario %in% c("B","C"),
                   c("scenario","mode","sig_rwd_err","sig_rct_err","rmse","ks_rwd","ks_rct")]
print(sub[order(sub$scenario, sub$mode), ], row.names = FALSE, digits = 3)
