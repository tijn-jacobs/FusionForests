################################################################################
# FusionForest prior-specification simulation — UNCENSORED variant, v4 (HPC).
#
# Aggressive omega sweep around the v2 winner k_tau = 2.0.  v2 showed that
# omega = 0.95 is not enough to fix the systematic over-coverage; v4 tests
# whether substantially smaller omega brings coverage to nominal.
#
# Fixed:
#   * (k_d, k_tau, k_c) = (2, 2.0, 2)
#   * (m_sh, m_d, m_tau, m_c) = (200, 100, 100, 100)
#   * sigma = 1.0 (DGP noise scale)
#   * error_dist = "gaussian"
#
# Swept:
#   * omega in {0.50, 0.75, 1.00}
#
# That gives 3 configurations.  The DGP, depth priors, MCMC budget, and grid
# (4 scenarios x 2 CATE regimes x 2 sample-size pairs) are unchanged.
#
# Conventions:
#   * Cores read from CLI arg (e.g. `Rscript simulation_uncensored_hpc_v4.R 192`).
#   * Output saved to $TMPDIR with the *_output.rds convention used by the
#     simulations/Main sim/ HPC scripts (download afterwards, evaluate locally).
#   * Parallelisation is across replications within each cell; each replicate
#     fits the three v4 configs serially on one core.
#   * Single MCMC chain per fit (N_burn + N_post = 2000 + 2000).
#   * Metric is in-sample on the combined RCT + RWD training data.
################################################################################

library(FusionForests)
library(doParallel)
library(foreach)

# ── DGP (mirrors simulation_uncensored_hpc.R) ───────────────────────────────
scenarios <- list(
  S0 = c(alpha_U = 0.0, beta_U = 1.0, gamma_U = 0.0),
  S1 = c(alpha_U = 0.5, beta_U = 1.0, gamma_U = 0.0),
  S2 = c(alpha_U = 1.5, beta_U = 1.5, gamma_U = 0.0),
  S3 = c(alpha_U = 1.0, beta_U = 1.0, gamma_U = 0.5)
)

cate_coefs <- list(
  homogeneous   = c(tau0 = 0.5, tau1 = 0.0, tau2 = 0.0),
  heterogeneous = c(tau0 = 0.5, tau1 = 0.4, tau2 = 0.3)
)

p_total      <- 10L
alpha_X      <- 1.0
alpha_X_cols <- c(1L, 2L)
sigma_noise  <- 1.0

true_tau <- function(X, cate) {
  Z1 <- X[, 1] - 0.5
  cate["tau0"] + cate["tau1"] * Z1 + cate["tau2"] * (X[, 2] > 0.5)
}

structural_logT <- function(X, A, U, scn, cate) {
  Z      <- X - 0.5
  f0     <- 0.7 * sin(pi * Z[, 1]) +
            1.0 * Z[, 2] * Z[, 3] +
            1.0 * (Z[, 4]^2 - 1 / 12) +
            scn["beta_U"] * U
  tau_x  <- true_tau(X, cate)
  tau_xU <- tau_x + scn["gamma_U"] * U
  f0 + A * tau_xU
}

rwd_propensity <- function(X, U, scn, alpha_0) {
  Z <- X - 0.5
  lin <- alpha_0 +
         alpha_X * (Z[, alpha_X_cols[1]] + Z[, alpha_X_cols[2]]) +
         scn["alpha_U"] * U
  plogis(lin)
}

solve_alpha_0 <- function(scn, n_search = 2e5) {
  set.seed(101)
  X <- matrix(runif(n_search * p_total), n_search, p_total)
  U <- rnorm(n_search)
  obj <- function(a0) mean(rwd_propensity(X, U, scn, a0)) - 0.5
  uniroot(obj, c(-10, 10))$root
}

make_data <- function(seed, scenario_name, cate_name,
                      n_rct, n_rwd, alpha_0) {
  set.seed(seed)
  scn  <- scenarios[[scenario_name]]
  cate <- cate_coefs[[cate_name]]

  X_rct <- matrix(runif(n_rct * p_total), n_rct, p_total)
  U_rct <- rnorm(n_rct)
  A_rct <- rbinom(n_rct, 1L, 0.5)
  e_rct <- sigma_noise * rnorm(n_rct)
  logT_rct <- structural_logT(X_rct, A_rct, U_rct, scn, cate) + e_rct

  X_rwd <- matrix(runif(n_rwd * p_total), n_rwd, p_total)
  U_rwd <- rnorm(n_rwd)
  pi_rwd <- rwd_propensity(X_rwd, U_rwd, scn, alpha_0)
  A_rwd  <- rbinom(n_rwd, 1L, pi_rwd)
  e_rwd  <- sigma_noise * rnorm(n_rwd)
  logT_rwd <- structural_logT(X_rwd, A_rwd, U_rwd, scn, cate) + e_rwd

  list(
    X     = rbind(X_rct, X_rwd),
    y     = c(logT_rct, logT_rwd),
    delta = rep(1L, n_rct + n_rwd),
    A     = c(A_rct, A_rwd),
    S     = c(rep(1L, n_rct), rep(0L, n_rwd)),
    n_rct = n_rct,
    n_rwd = n_rwd
  )
}

# ── Prior configurations ────────────────────────────────────────────────────
K_D     <- 2.0
K_TAU   <- 2.0
K_C     <- 2.0
M_SH    <- 200L
M_D     <- 100L
M_TAU   <- 100L
M_C     <- 100L
OMEGA_GRID <- c(0.50, 0.75, 1.00)

to_code_ks <- function(k_d, k_tau, k_c, omega = 1.0) {
  s0 <- omega / sqrt(1 + k_d^(-2) + k_tau^(-2) + k_c^(-2))
  list(
    k_sh    = s0,
    k_d     = s0 / k_d,
    k_tau   = s0 / k_tau,
    k_c     = s0 / k_c
  )
}

build_configs <- function() {
  base_depth <- list(power_treat = 3.0, base_treat = 0.95,
                     power_deconf = 3.0, base_deconf = 0.25)

  cfg <- function(omega) {
    label <- sprintf("o%03d", as.integer(round(omega * 100)))
    ks    <- to_code_ks(K_D, K_TAU, K_C, omega)
    c(list(label     = label,
           k_d_setup = K_D, k_tau_setup = K_TAU, k_c_setup = K_C,
           omega     = omega,
           k_sh_code = ks$k_sh, k_d_code = ks$k_d,
           k_tau_code = ks$k_tau, k_c_code = ks$k_c,
           m_sh = M_SH, m_d = M_D, m_tau = M_TAU, m_c = M_C),
      base_depth)
  }

  out <- lapply(OMEGA_GRID, cfg)
  names(out) <- vapply(out, `[[`, character(1), "label")
  out
}

# ── Fit functions ───────────────────────────────────────────────────────────
fit_config <- function(cfg, data, N_post, N_burn) {
  FusionForest(
    y                         = data$y,
    X_train_control           = data$X,
    X_train_treat             = data$X,
    treatment_indicator_train = data$A,
    source_indicator_train    = data$S,
    outcome_type              = "continuous",
    decomposition             = "four-forest",
    number_of_trees_control   = cfg$m_sh,
    number_of_trees_treat     = cfg$m_tau,
    number_of_trees_deconf    = cfg$m_c,
    number_of_trees_deviation = cfg$m_d,
    k_control                 = cfg$k_sh_code,
    k_treat                   = cfg$k_tau_code,
    k_deconf                  = cfg$k_c_code,
    k_g                       = cfg$k_d_code,
    power                     = 2.0,
    base                      = 0.95,
    power_treat               = cfg$power_treat,
    base_treat                = cfg$base_treat,
    power_deconf              = cfg$power_deconf,
    base_deconf               = cfg$base_deconf,
    N_post                    = N_post,
    N_burn                    = N_burn,
    treatment_coding          = "centered",
    error_dist                = "gaussian",
    store_posterior_sample    = TRUE,
    verbose                   = FALSE
  )
}

# ── Metrics ─────────────────────────────────────────────────────────────────
# In-sample on the combined training data (n = n_rct + n_rwd).  Also returns
# the posterior mean of sigma so we can diagnose whether the over-coverage
# is a residual-prior effect.
eval_cate <- function(fit, truth) {
  tau_hat  <- fit$train_predictions_treat
  tau_samp <- fit$train_predictions_sample_treat
  ci       <- apply(tau_samp, 2, quantile, probs = c(0.025, 0.975))
  rmse <- sqrt(mean((tau_hat - truth)^2))
  bias <- mean(tau_hat - truth)
  cov  <- mean(truth >= ci[1, ] & truth <= ci[2, ])
  wid  <- mean(ci[2, ] - ci[1, ])
  sig  <- mean(fit$sigma)
  c(rmse = rmse, bias = bias, coverage = cov, width = wid, sigma_hat = sig)
}

metric_cols <- c("rmse", "bias", "coverage", "width", "sigma_hat")

make_row <- function(scenario, cate_name, n_rct, n_rwd, config,
                     metrics = NULL) {
  vals <- setNames(rep(NA_real_, length(metric_cols)), metric_cols)
  if (!is.null(metrics)) vals[names(metrics)] <- unname(metrics)
  df <- as.data.frame(as.list(vals), stringsAsFactors = FALSE)
  cbind(scenario = scenario, cate = cate_name,
        n_rct = n_rct, n_rwd = n_rwd, config = config,
        df, stringsAsFactors = FALSE)
}

# ── One replication ─────────────────────────────────────────────────────────
run_one_sim <- function(seed, scenario_name, cate_name,
                        n_rct, n_rwd, alpha_0,
                        configs, N_post, N_burn) {
  data  <- make_data(seed, scenario_name, cate_name,
                     n_rct, n_rwd, alpha_0)
  truth <- true_tau(data$X, cate_coefs[[cate_name]])

  rows <- list()
  for (label in names(configs)) {
    cfg <- configs[[label]]
    fit <- tryCatch(fit_config(cfg, data, N_post, N_burn),
                    error = function(e) NULL)
    if (!is.null(fit)) {
      rows[[label]] <- make_row(
        scenario_name, cate_name, n_rct, n_rwd, label,
        eval_cate(fit, truth))
    } else {
      rows[[label]] <- make_row(scenario_name, cate_name, n_rct, n_rwd, label)
    }
  }
  do.call(rbind, rows)
}

# ── Parallel runner over replications (one cell) ────────────────────────────
run_simulation_tidy <- function(n_rep, scenario_name, cate_name,
                                n_rct, n_rwd, alpha_0_map,
                                configs, N_post, N_burn,
                                seed_offset = 1000L) {
  alpha_0 <- alpha_0_map[[scenario_name]]
  foreach(
    i = seq_len(n_rep),
    .combine = "rbind",
    .packages = "FusionForests"
  ) %dopar% {
    res <- run_one_sim(
      seed          = seed_offset + i,
      scenario_name = scenario_name,
      cate_name     = cate_name,
      n_rct         = n_rct,
      n_rwd         = n_rwd,
      alpha_0       = alpha_0,
      configs       = configs,
      N_post        = N_post,
      N_burn        = N_burn
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
cat("SIMULATION: FusionForest prior spec, v4 (UNCENSORED) — HPC\n")

set.seed(42)

M               <- num_cores
n_pairs         <- list(c(250L, 1000L), c(100L, 400L))
scenario_names  <- names(scenarios)
cate_names      <- names(cate_coefs)
N_post          <- 3000L
N_burn          <- 2000L

cat("Calibrating RWD propensity intercept per scenario...\n")
alpha_0_map  <- setNames(
  lapply(scenario_names, function(s) solve_alpha_0(scenarios[[s]])),
  scenario_names
)
for (s in scenario_names) {
  cat(sprintf("  %s: alpha_0 = %+0.4f\n", s, alpha_0_map[[s]]))
}

configs <- build_configs()

cells <- expand.grid(
  scenario = scenario_names,
  cate     = cate_names,
  n_idx    = seq_along(n_pairs),
  stringsAsFactors = FALSE
)
cells <- cells[order(cells$scenario, cells$cate, cells$n_idx), ]
rownames(cells) <- NULL

cat(sprintf("Grid: %d scenarios x %d CATE x %d size pairs = %d cells, %d reps each, %d configs\n",
            length(scenario_names), length(cate_names), length(n_pairs),
            nrow(cells), M, length(configs)))

all_sim_data <- vector("list", nrow(cells))
for (cell_idx in seq_len(nrow(cells))) {
  sn      <- cells$scenario[cell_idx]
  cn      <- cells$cate[cell_idx]
  pair    <- n_pairs[[cells$n_idx[cell_idx]]]
  n_rct   <- pair[1]
  n_rwd   <- pair[2]
  cat(sprintf("\nCell %d/%d: scenario=%s, cate=%s, n_rct=%d, n_rwd=%d\n",
              cell_idx, nrow(cells), sn, cn, n_rct, n_rwd))
  sim_df <- run_simulation_tidy(
    n_rep         = M,
    scenario_name = sn,
    cate_name     = cn,
    n_rct         = n_rct,
    n_rwd         = n_rwd,
    alpha_0_map   = alpha_0_map,
    configs       = configs,
    N_post        = N_post,
    N_burn        = N_burn,
    seed_offset   = 1000L * cell_idx
  )
  all_sim_data[[cell_idx]] <- sim_df
}

final_flat_df <- do.call(rbind, all_sim_data)

# Define output file path
# (! NAME MUST BE FILENAME_output.rds !)
output_file <- file.path(Sys.getenv("TMPDIR"),
                         "simulation_uncensored_hpc_v4_output.rds")

cat("Saving all settings results to:", output_file, "\n")
saveRDS(final_flat_df, file = output_file)
cat("All results successfully saved in one file.\n")


# ──────────────────────────────────────────────────────────────────────────────
# LOCAL POST-PROCESSING (uncomment after downloading the .rds from HPC).
# Place this script and simulation_uncensored_hpc_v4_output.rds in the same
# folder and run from the repo root.  Prints two summary blocks per
# (scenario, cate, size, config) cell:
#   1. CATE estimation (RMSE, bias, coverage, width, posterior sigma)
#   2. Coverage vs omega:  is the over-coverage fixed by tightening omega?
# ──────────────────────────────────────────────────────────────────────────────
# res <- readRDS("simulations/prior/simulation_uncensored_hpc_v4_output.rds")
#
# summary_tbl <- aggregate(
#   cbind(rmse, bias, coverage, width, sigma_hat) ~
#     scenario + cate + n_rct + n_rwd + config,
#   data = res, FUN = function(x) mean(x, na.rm = TRUE), na.action = na.pass)
#
# ord <- order(summary_tbl$scenario, summary_tbl$cate,
#              summary_tbl$n_rct, summary_tbl$config)
#
# cat("\n=== CATE estimation + posterior sigma (scenario x cate x size x config) ===\n")
# print(summary_tbl[ord,
#                   c("scenario","cate","n_rct","n_rwd","config",
#                     "rmse","bias","coverage","width","sigma_hat")],
#       row.names = FALSE, digits = 3)
#
# cat("\n=== Coverage and width sweep across omega ===\n")
# print(summary_tbl[ord,
#                   c("scenario","cate","n_rct","config","coverage","width")],
#       row.names = FALSE, digits = 3)
