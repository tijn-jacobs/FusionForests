################################################################################
# FusionForest residual-prior simulation (HPC / Snellius version).
#
# Appendix-grade study of the source_hdp_scale prior: it should match or beat
# every other residual prior on CATE estimation across data-generating regimes
# and become a *strict* winner when (i) the two sources differ in residual
# spread AND in shape and (ii) the RCT is small enough that borrowing atom
# locations from the larger RWD helps.
#
# Conventions:
#   * Cores read from CLI arg (e.g. `Rscript simulation_error_dist_hpc.R 192`).
#   * Output saved to $TMPDIR with the *_output.rds convention used by the
#     simulations/Main sim/ HPC scripts (download afterwards, evaluate locally).
#   * Parallelisation is across replications within each cell; each replicate
#     fits all six error-distribution modes serially on one core.
#
# Scenarios:
#   A. SHARED          -- sigma_RCT = sigma_RWD; Gaussian errors (null case).
#   B. SCALE           -- sigma_RCT != sigma_RWD; Gaussian otherwise.
#   C. SHAPE+SCALE     -- sigma_RCT != sigma_RWD + per-source 2-component
#                         mixture with shared atom locations and different
#                         per-source weights (mild separation).
#   D. TAIL SHARING    -- sigma_RCT != sigma_RWD + 2-component mixture where
#                         one atom is rare in the RCT but common in the RWD,
#                         so the RCT alone barely sees the heavy tail.  This
#                         is the regime where atom-borrowing pays off and
#                         where source_hdp_scale should win cleanly.
#
# Sample size scan: each scenario is run with n_rct in {50, 150, 300} while
# n_rwd is fixed at 300.  Smaller RCT amplifies the value of HDP borrowing.
#
# Modes compared:
#   gaussian, shared_dp, source_dp, source_dp_scale, source_hdp, source_hdp_scale
################################################################################

library(FusionForests)
library(doParallel)
library(foreach)

# ── Data-generating process ─────────────────────────────────────────────────
mu0       <- function(X) 2 * X[, 1] - X[, 3]^2
true_cate <- function(X) 1.5 + 2 * X[, 1]

# Scenario parameters
sigma_shared  <- 1.0
sigma_rct     <- 0.5
sigma_rwd     <- 1.5

# Scenario C: shape+scale, moderate separation
atoms_C       <- c(-1.5, 1.0)
weights_rct_C <- c(0.20, 0.80)
weights_rwd_C <- c(0.75, 0.25)

# Scenario D: tail-sharing.  The lower atom is rare in the RCT (5%) but
# common in the RWD (40%), so the RCT alone barely sees the heavy tail.
# This is the regime where HDP's atom-borrowing should pay off the most.
sigma_rct_D   <- 0.4
sigma_rwd_D   <- 0.8
atoms_D       <- c(-3.0, 0.5)
weights_rct_D <- c(0.05, 0.95)
weights_rwd_D <- c(0.40, 0.60)

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
  if (scenario == "D") {
    sd_s <- if (source_label == "rct") sigma_rct_D else sigma_rwd_D
    w    <- if (source_label == "rct") weights_rct_D else weights_rwd_D
    k    <- sample.int(2, n, replace = TRUE, prob = w)
    mu_centered <- atoms_D - sum(w * atoms_D)
    return(mu_centered[k] + rnorm(n, sd = sd_s))
  }
  stop("unknown scenario")
}

# True per-source within-component sigma (used by eval_sigma).  For mixture
# scenarios this is the sd of the Gaussian convolved with the atoms, NOT the
# total residual sd -- see eval_total_sigma for the latter.
true_sigma_within <- function(scenario, source_label) {
  if (scenario == "A") return(sigma_shared)
  if (scenario == "B") return(if (source_label == "rct") sigma_rct  else sigma_rwd)
  if (scenario == "C") return(if (source_label == "rct") sigma_rct  else sigma_rwd)
  if (scenario == "D") return(if (source_label == "rct") sigma_rct_D else sigma_rwd_D)
  stop("unknown scenario")
}

make_data <- function(scenario, seed, n_rct, n_rwd, p) {
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
    X       = rbind(X_rct, X_rwd),
    y       = c(y_rct, y_rwd),
    trt     = c(trt_rct, trt_rwd),
    src     = c(rep(1L, n_rct), rep(0L, n_rwd)),
    truth   = true_cate(rbind(X_rct, X_rwd)),
    eps_rct = eps_rct,
    eps_rwd = eps_rwd
  )
}

# ── Metrics ─────────────────────────────────────────────────────────────────
eval_cate <- function(cate_hat, cate_samples, truth) {
  rmse <- sqrt(mean((cate_hat - truth)^2))
  bias <- mean(cate_hat - truth)
  ci   <- apply(cate_samples, 2, quantile, probs = c(0.025, 0.975))
  cov  <- mean(truth >= ci[1, ] & truth <= ci[2, ])
  wid  <- mean(ci[2, ] - ci[1, ])
  c(rmse = rmse, bias = bias, coverage = cov, width = wid)
}

eval_sigma <- function(fit, scenario) {
  # Within-component sigma recovery.  Only directly comparable across modes
  # when the truth is Gaussian (scenarios A, B); in C and D the mixture
  # absorbs part of the variance so within-component sigma is a misleading
  # standalone metric -- eval_total_sigma is the apples-to-apples version.
  truth_rwd <- true_sigma_within(scenario, "rwd")
  truth_rct <- true_sigma_within(scenario, "rct")
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

# Total residual sd per source: sqrt(sigma_s^2 + sum_k pi_sk (theta_sk - mu_s)^2).
# The fit stores already-centered locations in fit$dp_locations (per source for
# source_*; shared single column for shared_dp), so the second term is simply
# rowSums(P * L^2) per iteration.  Compared against the empirical sd of the
# DGP residuals for that source -- the apples-to-apples metric across modes.
eval_total_sigma <- function(fit, data) {
  true_rwd <- sd(data$eps_rwd)
  true_rct <- sd(data$eps_rct)

  if (is.null(fit$dp_mix_prop)) {
    sig_tot_rwd <- mean(fit$sigma)
    sig_tot_rct <- mean(fit$sigma)
  } else {
    n_g     <- length(fit$dp_mix_prop)
    sg_rwd  <- if (!is.null(fit$dp_sigma_g)) fit$dp_sigma_g[[1]] else fit$sigma
    sg_rct  <- if (!is.null(fit$dp_sigma_g)) fit$dp_sigma_g[[if (n_g == 2L) 2L else 1L]]
               else fit$sigma
    P_rwd <- fit$dp_mix_prop[[1]]
    P_rct <- fit$dp_mix_prop[[if (n_g == 2L) 2L else 1L]]
    L_rwd <- fit$dp_locations[[1]]
    L_rct <- fit$dp_locations[[if (n_g == 2L) 2L else 1L]]
    var_atoms_rwd <- rowSums(P_rwd * L_rwd^2)
    var_atoms_rct <- rowSums(P_rct * L_rct^2)
    sig_tot_rwd <- mean(sqrt(sg_rwd^2 + var_atoms_rwd))
    sig_tot_rct <- mean(sqrt(sg_rct^2 + var_atoms_rct))
  }
  c(sig_total_rwd_hat = sig_tot_rwd, sig_total_rct_hat = sig_tot_rct,
    sig_total_rwd_err = sig_tot_rwd - true_rwd,
    sig_total_rct_err = sig_tot_rct - true_rct)
}

eval_resid_fit <- function(fit, data) {
  resid <- data$y - fit$train_predictions
  ks_rct <- ks.test(resid[data$src == 1L], data$eps_rct,
                    exact = FALSE)$statistic
  ks_rwd <- ks.test(resid[data$src == 0L], data$eps_rwd,
                    exact = FALSE)$statistic
  c(ks_rct = unname(ks_rct), ks_rwd = unname(ks_rwd))
}

# Mixture-prior diagnostics: effective and occupied component counts per
# source, per-source concentration M_s, top-level concentration gamma (HDP),
# absolute posterior-mean discrepancy of source means mu_0 vs mu_1 (HDP), and
# the three tree-update acceptance ratios.  Returns NA where a mode doesn't
# expose the corresponding state (dp_gamma is HDP-only; dp_mix_prop is absent
# under "gaussian").  For "shared_dp" the per-source slots receive the shared
# value so the columns remain comparable across modes.
eval_diagnostics <- function(fit, eps = 0.01) {
  out <- c(
    K_eff_rwd        = NA_real_,
    K_eff_rct        = NA_real_,
    n_occupied_rwd   = NA_real_,
    n_occupied_rct   = NA_real_,
    n_shared         = NA_real_,
    M_rwd            = NA_real_,
    M_rct            = NA_real_,
    gamma_top        = NA_real_,
    mu_diff          = NA_real_,
    atoms_max_shared = NA_real_,
    accept_control   = unname(fit$acceptance_ratio_control),
    accept_treat     = unname(fit$acceptance_ratio_treat),
    accept_deconf    = unname(fit$acceptance_ratio_deconf)
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
  if (!is.null(fit$dp_locations_shared)) {
    # Drift of the unconstrained shared atoms theta_k* away from zero.  Large
    # values flag the mu_s vs theta_k* identifiability tension (atoms drift
    # far from the prior mean and the per-source centering shifts absorb the
    # gap).  Posterior mean of the per-iteration max |theta_k*|.
    out["atoms_max_shared"] <-
      mean(apply(abs(fit$dp_locations_shared), 1, max))
  }

  out
}

# ── One fit ─────────────────────────────────────────────────────────────────
fit_one <- function(data, error_dist, N_post, N_burn) {
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

# Ordered metric names used for both the success and failure rows so the
# returned data.frame schema is identical.
metric_cols <- c(
  "rmse", "bias", "coverage", "width",
  "sig_rwd_hat", "sig_rct_hat", "sig_rwd_err", "sig_rct_err",
  "sig_total_rwd_hat", "sig_total_rct_hat",
  "sig_total_rwd_err", "sig_total_rct_err",
  "ks_rct", "ks_rwd",
  "K_eff_rwd", "K_eff_rct", "n_occupied_rwd", "n_occupied_rct",
  "n_shared", "M_rwd", "M_rct", "gamma_top", "mu_diff",
  "atoms_max_shared",
  "accept_control", "accept_treat", "accept_deconf"
)

make_row <- function(scenario, mode, metrics = NULL) {
  vals <- setNames(rep(NA_real_, length(metric_cols)), metric_cols)
  if (!is.null(metrics)) vals[names(metrics)] <- unname(metrics)
  df <- as.data.frame(as.list(vals), stringsAsFactors = FALSE)
  cbind(scenario = scenario, mode = mode, df, stringsAsFactors = FALSE)
}

# ── One replication: fit all six modes, return tidy data.frame ──────────────
run_one_sim <- function(
    seed, scenario,
    n_rct, n_rwd, p,
    N_post, N_burn,
    modes
) {
  data <- make_data(scenario = scenario, seed = seed,
                    n_rct = n_rct, n_rwd = n_rwd, p = p)

  do.call(rbind, lapply(modes, function(m) {
    fit <- tryCatch(fit_one(data, m, N_post, N_burn),
                    error = function(e) NULL)
    if (is.null(fit)) return(make_row(scenario, m))
    metrics <- c(
      eval_cate(fit$train_predictions_treat,
                fit$train_predictions_sample_treat, data$truth),
      eval_sigma(fit, scenario),
      eval_total_sigma(fit, data),
      eval_resid_fit(fit, data),
      eval_diagnostics(fit)
    )
    make_row(scenario, m, metrics)
  }))
}

# ── Parallel runner over replications (one scenario) ────────────────────────
run_simulation_tidy <- function(
    n_rep, scenario,
    n_rct, n_rwd, p,
    N_post, N_burn, modes,
    seed_offset = 1000L
) {
  foreach(
    i = seq_len(n_rep),
    .combine = "rbind",
    .packages = "FusionForests"
  ) %dopar% {
    res <- run_one_sim(
      seed     = seed_offset + i,
      scenario = scenario,
      n_rct    = n_rct,
      n_rwd    = n_rwd,
      p        = p,
      N_post   = N_post,
      N_burn   = N_burn,
      modes    = modes
    )
    res$Iter <- i
    res
  }
}

# ── Main (HPC) ──────────────────────────────────────────────────────────────
args <- commandArgs(trailingOnly = TRUE)

if (length(args) > 0) {
  num_cores <- as.integer(args[1]) - 1L
} else {
  num_cores <- parallel::detectCores() - 1L
}

registerDoParallel(cores = num_cores)
cat("Number of cores being used (1 free):", num_cores, "\n")
cat("SIMULATION: FusionForest residual-prior comparison -- HPC\n")

set.seed(42)

# Simulation grid
M               <- num_cores   # replications per cell
n_rwd_value     <- 300L
n_rct_values    <- c(50L, 150L, 300L)
p               <- 5L
N_post          <- 1000L
N_burn          <- 1000L
modes           <- c("gaussian", "shared_dp", "source_dp",
                     "source_dp_scale", "source_hdp", "source_hdp_scale")
scenarios       <- c("A", "B", "C", "D")

cells <- expand.grid(scenario = scenarios, n_rct = n_rct_values,
                     stringsAsFactors = FALSE)
cells <- cells[order(cells$scenario, cells$n_rct), ]
rownames(cells) <- NULL

cat(sprintf("Grid: %d scenarios x %d n_rct values = %d cells, %d reps each\n",
            length(scenarios), length(n_rct_values), nrow(cells), M))

all_sim_data <- vector("list", nrow(cells))
for (cell_idx in seq_len(nrow(cells))) {
  sc    <- cells$scenario[cell_idx]
  n_rct <- cells$n_rct[cell_idx]
  cat(sprintf("\nCell %d/%d: scenario=%s, n_rct=%d, n_rwd=%d\n",
              cell_idx, nrow(cells), sc, n_rct, n_rwd_value))
  sim_df <- run_simulation_tidy(
    n_rep       = M,
    scenario    = sc,
    n_rct       = n_rct,
    n_rwd       = n_rwd_value,
    p           = p,
    N_post      = N_post,
    N_burn      = N_burn,
    modes       = modes,
    seed_offset = 1000L * cell_idx
  )
  sim_df$n_rct <- n_rct
  sim_df$n_rwd <- n_rwd_value
  all_sim_data[[cell_idx]] <- sim_df
}

final_flat_df <- do.call(rbind, all_sim_data)

# Define output file path
# (! NAME MUST BE FILENAME_output.rds !)
output_file <- file.path(Sys.getenv("TMPDIR"),
                         "simulation_error_dist_hpc_output.rds")

cat("Saving all settings results to:", output_file, "\n")
saveRDS(final_flat_df, file = output_file)
cat("All results successfully saved in one file.\n")


# ──────────────────────────────────────────────────────────────────────────────
# LOCAL POST-PROCESSING (uncomment after downloading the .rds from HPC).
# Place this script and simulation_error_dist_hpc_output.rds in the same folder
# and run from the repo root.  Prints four summary blocks per (scenario, n_rct,
# mode) cell:
#   1. CATE estimation (RMSE, bias, coverage, width)
#   2. Total-residual-sd recovery (the apples-to-apples sigma metric)
#   3. Mixture diagnostics (K_eff, occupancy, sharing, concentration, mu_diff,
#      atom drift)
#   4. Sample-size sensitivity for scenarios C and D (where HDP_SCALE should
#      improve relative to alternatives as n_rct shrinks)
# ──────────────────────────────────────────────────────────────────────────────
# res <- readRDS("examples/simulation_error_dist_hpc_output.rds")
#
# summary_tbl <- aggregate(
#   cbind(rmse, bias, coverage, width,
#         sig_rwd_hat, sig_rct_hat, sig_rwd_err, sig_rct_err,
#         sig_total_rwd_hat, sig_total_rct_hat,
#         sig_total_rwd_err, sig_total_rct_err,
#         ks_rct, ks_rwd,
#         K_eff_rwd, K_eff_rct, n_occupied_rwd, n_occupied_rct,
#         n_shared, M_rwd, M_rct, gamma_top, mu_diff, atoms_max_shared,
#         accept_control, accept_treat, accept_deconf) ~ scenario + n_rct + mode,
#   data = res, FUN = function(x) mean(x, na.rm = TRUE), na.action = na.pass)
#
# ord <- order(summary_tbl$scenario, summary_tbl$n_rct, summary_tbl$mode)
#
# cat("\n=== CATE estimation (scenario x n_rct x mode) ===\n")
# print(summary_tbl[ord,
#                   c("scenario","n_rct","mode",
#                     "rmse","bias","coverage","width")],
#       row.names = FALSE, digits = 3)
#
# cat("\n=== Total-residual-sd recovery (scenario x n_rct x mode) ===\n")
# print(summary_tbl[ord,
#                   c("scenario","n_rct","mode",
#                     "sig_total_rwd_hat","sig_total_rct_hat",
#                     "sig_total_rwd_err","sig_total_rct_err",
#                     "ks_rwd","ks_rct")],
#       row.names = FALSE, digits = 3)
#
# cat("\n=== Mixture diagnostics (scenario x n_rct x mode) ===\n")
# print(summary_tbl[ord,
#                   c("scenario","n_rct","mode",
#                     "K_eff_rwd","K_eff_rct",
#                     "n_occupied_rwd","n_occupied_rct","n_shared",
#                     "M_rwd","M_rct","gamma_top",
#                     "mu_diff","atoms_max_shared")],
#       row.names = FALSE, digits = 3)
#
# cat("\n=== Sample-size sensitivity: CATE RMSE in scenarios C and D ===\n")
# sub <- summary_tbl[summary_tbl$scenario %in% c("C","D"),
#                    c("scenario","n_rct","mode","rmse","width",
#                      "sig_total_rct_err","sig_total_rwd_err")]
# print(sub[order(sub$scenario, sub$n_rct, sub$mode), ],
#       row.names = FALSE, digits = 3)

# cat("\n=== Per-source sigma recovery + CATE RMSE (scenarios B and C) ===\n")
# sub <- summary_tbl[summary_tbl$scenario %in% c("B", "C"),
#                    c("scenario", "mode",
#                      "sig_rwd_err", "sig_rct_err",
#                      "rmse", "ks_rwd", "ks_rct")]
# print(sub[order(sub$scenario, sub$mode), ], row.names = FALSE, digits = 3)
