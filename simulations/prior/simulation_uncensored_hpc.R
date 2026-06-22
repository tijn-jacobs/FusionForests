################################################################################
# FusionForest prior-specification simulation — UNCENSORED variant (HPC).
#
# Companion to simulation_censored_hpc.R.  Identical DGP and configurations but
# no administrative censoring is applied (delta == 1 for every observation),
# so log T can be standardised cleanly by FusionForest's internal continuous
# branch (subtract mean, divide by sample SD).  Used as a clean reference for
# the headline censored study.
#
# Conventions:
#   * Cores read from CLI arg (e.g. `Rscript simulation_uncensored_hpc.R 192`).
#   * Output saved to $TMPDIR with the *_output.rds convention used by the
#     simulations/Main sim/ HPC scripts (download afterwards, evaluate locally).
#   * Parallelisation is across replications within each cell; each replicate
#     fits the five prior configs serially on one core.
#   * Single MCMC chain per fit (N_burn + N_post = 2000 + 2000).
#
# Benchmarks (RCT-only, Naive-pool) are out of scope. Only the five
# four-forest prior configs are fitted here.
################################################################################

library(FusionForests)
library(doParallel)
library(foreach)

# ── DGP (mirrors simulation_censored_hpc.R) ─────────────────────────────────
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
    y     = c(logT_rct, logT_rwd),         # log T directly
    delta = rep(1L, n_rct + n_rwd),
    A     = c(A_rct, A_rwd),
    S     = c(rep(1L, n_rct), rep(0L, n_rwd)),
    n_rct = n_rct,
    n_rwd = n_rwd
  )
}

make_test_set <- function(n_test = 1000L, seed = 7L) {
  set.seed(seed)
  X_test <- matrix(runif(n_test * p_total), n_test, p_total)
  list(X = X_test, n = n_test)
}

# ── Prior configurations (same as censored variant) ─────────────────────────
to_code_ks <- function(k_d, k_tau, k_c) {
  s0 <- 1 / sqrt(1 + k_d^(-2) + k_tau^(-2) + k_c^(-2))
  list(
    k_sh    = s0,
    k_d     = s0 / k_d,
    k_tau   = s0 / k_tau,
    k_c     = s0 / k_c
  )
}

build_configs <- function() {
  base_trees <- list(m_sh = 200L, m_d = 50L, m_tau = 200L, m_c = 200L)
  base_depth <- list(power_treat = 3.0, base_treat = 0.95,
                     power_deconf = 3.0, base_deconf = 0.25)

  cfg <- function(label, k_d, k_tau, k_c, depth = base_depth) {
    ks <- to_code_ks(k_d, k_tau, k_c)
    c(list(label = label,
           k_d_setup = k_d, k_tau_setup = k_tau, k_c_setup = k_c,
           k_sh_code = ks$k_sh, k_d_code = ks$k_d,
           k_tau_code = ks$k_tau, k_c_code = ks$k_c),
      base_trees, depth)
  }

  list(
    Default     = cfg("Default",     2.5, 1.7, 4),
    Efficiency  = cfg("Efficiency",  4.0, 1.7, 10),
    Robust      = cfg("Robust",      2.0, 1.7, 2),
    Equal_share = cfg("Equal_share", 1.0, 1.0, 1),
    Tau_deep    = cfg("Tau_deep",    2.5, 1.7, 4,
                      depth = list(power_treat = 2.0, base_treat = 0.95,
                                   power_deconf = 3.0, base_deconf = 0.25))
  )
}

# ── Fit functions ───────────────────────────────────────────────────────────
fit_config <- function(cfg, data, X_test, N_post, N_burn) {
  FusionForest(
    y                         = data$y,
    X_train_control           = data$X,
    X_train_treat             = data$X,
    treatment_indicator_train = data$A,
    source_indicator_train    = data$S,
    X_test_control            = X_test,
    X_test_treat              = X_test,
    X_test_deconf             = X_test,
    treatment_indicator_test  = rep(1L, nrow(X_test)),
    source_indicator_test     = rep(0L, nrow(X_test)),
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
eval_cate <- function(fit, truth) {
  tau_hat  <- fit$train_predictions_treat
  tau_samp <- fit$train_predictions_sample_treat
  ci       <- apply(tau_samp, 2, quantile, probs = c(0.025, 0.975))
  rmse <- sqrt(mean((tau_hat - truth)^2))
  bias <- mean(tau_hat - truth)
  cov  <- mean(truth >= ci[1, ] & truth <= ci[2, ])
  wid  <- mean(ci[2, ] - ci[1, ])
  c(rmse = rmse, bias = bias, coverage = cov, width = wid)
}

metric_cols <- c("rmse", "bias", "coverage", "width")

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
                        configs, test_set, N_post, N_burn) {
  data  <- make_data(seed, scenario_name, cate_name,
                     n_rct, n_rwd, alpha_0)
  truth <- true_tau(data$X, cate_coefs[[cate_name]])

  rows <- list()

  # Prior configs
  for (label in names(configs)) {
    cfg <- configs[[label]]
    fit <- tryCatch(fit_config(cfg, data, test_set$X, N_post, N_burn),
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
                                configs, test_set, N_post, N_burn,
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
      test_set      = test_set,
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
cat("SIMULATION: FusionForest prior specification (UNCENSORED) — HPC\n")

set.seed(42)

M               <- 1000
n_pairs         <- list(c(250L, 1000L), c(100L, 400L))
scenario_names  <- names(scenarios)
cate_names      <- names(cate_coefs)
N_post          <- 5000L
N_burn          <- 5000L

cat("Calibrating RWD propensity intercept per scenario...\n")
alpha_0_map  <- setNames(
  lapply(scenario_names, function(s) solve_alpha_0(scenarios[[s]])),
  scenario_names
)
for (s in scenario_names) {
  cat(sprintf("  %s: alpha_0 = %+0.4f\n", s, alpha_0_map[[s]]))
}

configs  <- build_configs()
test_set <- make_test_set(n_test = 1000L, seed = 7L)

cells <- expand.grid(
  scenario = scenario_names,
  cate     = cate_names,
  n_idx    = seq_along(n_pairs),
  stringsAsFactors = FALSE
)
cells <- cells[order(cells$scenario, cells$cate, cells$n_idx), ]
rownames(cells) <- NULL

cat(sprintf("Grid: %d scenarios x %d CATE x %d size pairs = %d cells, %d reps each\n",
            length(scenario_names), length(cate_names), length(n_pairs),
            nrow(cells), M))

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
    test_set      = test_set,
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
                         "simulation_uncensored_hpc_output.rds")

cat("Saving all settings results to:", output_file, "\n")
saveRDS(final_flat_df, file = output_file)
cat("All results successfully saved in one file.\n")


# ──────────────────────────────────────────────────────────────────────────────
# LOCAL POST-PROCESSING (uncomment after downloading the .rds from HPC).
# Place this script and simulation_uncensored_hpc_output.rds in the same folder
# and run from the repo root.  Prints two summary blocks per
# (scenario, cate, size, config) cell:
#   1. CATE estimation (RMSE, bias, coverage, width)
#   2. Strong-confounding bias check (S2, S3)
# ──────────────────────────────────────────────────────────────────────────────
# res <- readRDS("simulations/prior/simulation_uncensored_hpc_output.rds")
# 
# summary_tbl <- aggregate(
#   cbind(rmse, bias, coverage, width) ~
#     scenario + cate + n_rct + n_rwd + config,
#   data = res, FUN = function(x) mean(x, na.rm = TRUE), na.action = na.pass)
# 
# ord <- order(summary_tbl$scenario, summary_tbl$cate,
#              summary_tbl$n_rct, summary_tbl$config)
# 
# cat("\n=== CATE estimation (scenario x cate x size x config) ===\n")
# print(summary_tbl[ord,
#                   c("scenario","cate","n_rct","n_rwd","config",
#                     "rmse","bias","coverage","width")],
#       row.names = FALSE, digits = 3)
# 
# cat("\n=== Strong-confounding bias check (S2, S3) ===\n")
# sub <- summary_tbl[summary_tbl$scenario %in% c("S2","S3"),
#                    c("scenario","cate","n_rct","config",
#                      "rmse","bias","coverage")]
# print(sub[order(sub$scenario, sub$cate, sub$n_rct, sub$config), ],
#       row.names = FALSE, digits = 3)
