# ============================================================================
# IRS Benchmark v2 — BART+MIA (bartMachine) only
#
# Usage:
#   Rscript irs_v2_bartm.R [num_cores]
#
# Output:
#   $TMPDIR/irs_v2_bartm_output.rds
# ============================================================================

# Force Java options BEFORE anything loads Java
options(java.parameters = c("-Xmx20g", "--add-modules=jdk.incubator.vector", "-XX:+UseZGC")) 

library(FusionForests)
library(MASS)
library(bartMachine)

source("evaluation_functions.R")

# ============================================================================
# Parse arguments
# ============================================================================

args <- commandArgs(trailingOnly = TRUE)

if (length(args) >= 1) {
  num_cores <- as.integer(args[1]) - 2
} else {
  num_cores <- parallel::detectCores() - 2
}

# bartMachine uses Java threads internally — let it use
# all available cores per fit. No R-level parallelism.
bartMachine::set_bart_machine_num_cores(num_cores)

# ============================================================================
# Settings
# ============================================================================

n_reps  <- 200L
n_rct   <- 150L
n_rwd   <- 350L
n_test  <- 500L
seed0   <- 2026L

# DGP parameters
beta5   <- 1
gamma0  <- 2
gamma1  <- 1
alpha0  <- 0
alpha1  <- 0.5
alpha5  <- 0.5
pi_mcar <- 0.3

# BART settings
bart_settings <- list(
  number_of_trees = 200L,
  N_post          = 2000L,
  N_burn          = 1000L
)

# Output directory
output_dir <- if (nzchar(Sys.getenv("TMPDIR"))) {
  Sys.getenv("TMPDIR")
} else {
  "results"
}
if (!dir.exists(output_dir)) dir.create(output_dir,
                                        recursive = TRUE)

cat(sprintf("Cores: %d | Reps per scenario: %d\n",
            num_cores, n_reps))

# ============================================================================
# DGP functions
# ============================================================================

generate_covariates <- function(n, p = 5, rho = 0) {
  Sigma <- rho^abs(outer(1:p, 1:p, "-"))
  mvrnorm(n, mu = rep(0, p), Sigma = Sigma)
}

assign_treatment <- function(X, S, alpha0 = 0,
                             alpha1 = 0.5,
                             alpha5 = 0.5) {
  n <- nrow(X)
  A <- integer(n)
  rct <- which(S == 1)
  rwd <- which(S == 0)
  A[rct] <- rbinom(length(rct), 1, 0.5)
  lp <- alpha0 + alpha1 * X[rwd, 1] + alpha5 * X[rwd, 5]
  A[rwd] <- rbinom(length(rwd), 1, plogis(lp))
  A
}

# Outcome models
m_scenario0 <- function(X, A) {
  1 + X[, 1] + X[, 2] + 2 * X[, 3] * A
}

m_scenario1 <- function(X, A, beta5 = 1) {
  1 + X[, 1] + X[, 2] + beta5 * X[, 5] + 2 * X[, 3] * A
}

m_scenario2 <- function(X, A, gamma0 = 2, gamma1 = 1) {
  1 + X[, 1] + X[, 2] + (gamma0 + gamma1 * X[, 5]) * A
}

m_scenario3 <- function(X, A, beta5 = 1, gamma0 = 2,
                        gamma1 = 1) {
  1 + X[, 1] + X[, 2] + beta5 * X[, 5] +
    (gamma0 + gamma1 * X[, 5]) * A
}

# CATE functions
tau_scenario0 <- function(X) { 2 * X[, 3] }
tau_scenario1 <- function(X) { 2 * X[, 3] }
tau_scenario2 <- function(X, gamma0 = 2, gamma1 = 1) {
  gamma0 + gamma1 * X[, 5]
}
tau_scenario3 <- function(X, gamma0 = 2, gamma1 = 1) {
  gamma0 + gamma1 * X[, 5]
}

calibrate_sigma <- function(outcome_scenario, rho = 0,
                            target_snr = 2,
                            n_cal = 10000, seed = 999) {
  set.seed(seed)
  X <- generate_covariates(n_cal, rho = rho)
  A <- rbinom(n_cal, 1, 0.5)
  m_fn <- switch(as.character(outcome_scenario),
    "0" = m_scenario0, "1" = m_scenario1,
    "2" = m_scenario2, "3" = m_scenario3
  )
  m_vals <- m_fn(X, A)
  as.numeric(sqrt(var(m_vals) / target_snr))
}

generate_data <- function(n_rct = 150, n_rwd = 350,
                          n_test = 500,
                          outcome_scenario = 0,
                          rho = 0, sigma = NULL,
                          alpha0 = 0, alpha1 = 0.5,
                          alpha5 = 0.5,
                          beta5 = 1, gamma0 = 2,
                          gamma1 = 1,
                          seed = NULL) {
  if (!is.null(seed)) set.seed(seed)

  n <- n_rct + n_rwd
  S <- c(rep(1L, n_rct), rep(0L, n_rwd))

  X      <- generate_covariates(n, rho = rho)
  X_test <- generate_covariates(n_test, rho = rho)

  A      <- assign_treatment(X, S, alpha0, alpha1, alpha5)
  A_test <- rbinom(n_test, 1, 0.5)

  m_fn <- switch(as.character(outcome_scenario),
    "0" = function(X, A) m_scenario0(X, A),
    "1" = function(X, A) m_scenario1(X, A, beta5),
    "2" = function(X, A) m_scenario2(X, A, gamma0, gamma1),
    "3" = function(X, A) m_scenario3(X, A, beta5, gamma0,
                                     gamma1)
  )
  tau_fn <- switch(as.character(outcome_scenario),
    "0" = function(X) tau_scenario0(X),
    "1" = function(X) tau_scenario1(X),
    "2" = function(X) tau_scenario2(X, gamma0, gamma1),
    "3" = function(X) tau_scenario3(X, gamma0, gamma1)
  )

  if (is.null(sigma)) {
    sigma <- calibrate_sigma(outcome_scenario, rho)
  }

  m_train   <- m_fn(X, A)
  m_test    <- m_fn(X_test, A_test)
  tau_train <- tau_fn(X)
  tau_test  <- tau_fn(X_test)

  y <- m_train + rnorm(n, sd = sigma)

  XA_train <- cbind(X, A)
  XA_test  <- cbind(X_test, A_test)
  colnames(XA_train) <- c(paste0("X", 1:5), "A")
  colnames(XA_test)  <- c(paste0("X", 1:5), "A")

  XA_test_1 <- cbind(X_test, 1)
  XA_test_0 <- cbind(X_test, 0)
  colnames(XA_test_1) <- colnames(XA_train)
  colnames(XA_test_0) <- colnames(XA_train)

  XA_train_1 <- cbind(X, 1)
  XA_train_0 <- cbind(X, 0)
  colnames(XA_train_1) <- colnames(XA_train)
  colnames(XA_train_0) <- colnames(XA_train)

  list(
    y = y, XA_train = XA_train, X_train = X,
    A_train = A, S_train = S,
    m_train = m_train, tau_train = tau_train,
    XA_test = XA_test, XA_test_1 = XA_test_1,
    XA_test_0 = XA_test_0,
    X_test = X_test, A_test = A_test,
    m_test = m_test, tau_test = tau_test,
    XA_train_1 = XA_train_1, XA_train_0 = XA_train_0,
    sigma = sigma, n_rct = n_rct, n_rwd = n_rwd
  )
}

impose_missingness <- function(XA, S,
                               miss_pattern = c("block_rct",
                                                "block_rwd",
                                                "mcar"),
                               pi_mcar = 0.3) {
  miss_pattern <- match.arg(miss_pattern)
  n <- nrow(XA)
  XA_miss <- XA
  M <- integer(n)

  if (miss_pattern == "block_rct") {
    idx <- which(S == 1)
  } else if (miss_pattern == "block_rwd") {
    idx <- which(S == 0)
  } else {
    idx <- which(rbinom(n, 1, pi_mcar) == 1)
  }
  M[idx] <- 1L
  XA_miss[idx, 5] <- NaN

  list(XA_miss = XA_miss, M = M)
}

# ============================================================================
# Scenario grid
# ============================================================================

build_scenario_grid <- function() {
  grid <- expand.grid(
    outcome_scenario = 0:3,
    miss_pattern     = c("block_rct", "block_rwd", "mcar"),
    rho              = c(0, 0.5),
    stringsAsFactors = FALSE
  )
  grid$scenario_id <- seq_len(nrow(grid))
  grid$outcome_label <- c(
    "irrelevant", "prognostic",
    "effect_modifier", "prognostic_em"
  )[grid$outcome_scenario + 1]
  grid
}

get_scenario <- function(id) {
  grid <- build_scenario_grid()
  as.list(grid[id, ])
}

get_n_scenarios <- function() {
  nrow(build_scenario_grid())
}

# ============================================================================
# Method wrapper
# ============================================================================

# --- bartMachine MIA ---

run_bartmachine <- function(data, miss, s) {
  XA_na <- miss$XA_miss
  XA_na[is.nan(XA_na)] <- NA

  bm <- bartMachine::bartMachine(
    as.data.frame(XA_na), data$y,
    num_trees                          = s$number_of_trees,
    num_burn_in                        = s$N_burn,
    num_iterations_after_burn_in       = s$N_post,
    use_missing_data                   = TRUE,
    use_missing_data_dummies_as_covars = TRUE,
    verbose                            = FALSE
  )

  pred_obs <- bartMachine::bart_machine_get_posterior(
    bm, as.data.frame(data$XA_test)
  )
  pred_1 <- bartMachine::bart_machine_get_posterior(
    bm, as.data.frame(data$XA_test_1)
  )
  pred_0 <- bartMachine::bart_machine_get_posterior(
    bm, as.data.frame(data$XA_test_0)
  )
  pred_tr <- bartMachine::bart_machine_get_posterior(
    bm, as.data.frame(XA_na)
  )
  pred_tr1 <- bartMachine::bart_machine_get_posterior(
    bm, as.data.frame(data$XA_train_1)
  )
  pred_tr0 <- bartMachine::bart_machine_get_posterior(
    bm, as.data.frame(data$XA_train_0)
  )

  fit_obs <- list(
    train_predictions = pred_tr$y_hat,
    test_predictions  = pred_obs$y_hat
  )
  fit_1 <- list(
    train_predictions = pred_tr1$y_hat,
    test_predictions  = pred_1$y_hat,
    test_predictions_sample =
      t(pred_1$y_hat_posterior_samples)
  )
  fit_0 <- list(
    train_predictions = pred_tr0$y_hat,
    test_predictions  = pred_0$y_hat,
    test_predictions_sample =
      t(pred_0$y_hat_posterior_samples)
  )

  rm(bm); gc()

  evaluate_fit(
    fit_obs, fit_1, fit_0,
    data$m_train, data$m_test,
    data$tau_train, data$tau_test
  )
}

# --- Null result ---

null_result <- function() {
  list(
    rmse_m_train    = NA, rmse_m_test    = NA,
    mae_m_test      = NA,
    cate_bias_test  = NA, cate_rmse_test  = NA,
    cate_bias_train = NA, cate_rmse_train = NA,
    ate_bias        = NA, ate_hat         = NA,
    ate_true        = NA,
    cate_coverage   = NA, cate_ci_width   = NA,
    avg_post_var    = NA,
    ate_coverage    = NA, ate_ci_width    = NA
  )
}

# ============================================================================
# Main simulation loop
# ============================================================================

n_scenarios <- get_n_scenarios()

cat(sprintf("\nTotal scenarios: %d\n", n_scenarios))
cat(sprintf("Output directory: %s\n\n", output_dir))

all_scenario_results <- list()

for (scenario_id in seq_len(n_scenarios)) {

  scenario <- get_scenario(scenario_id)

  cat(sprintf(
    "=== Scenario %d / %d: outcome=%d (%s) | miss=%s | rho=%.1f ===\n",
    scenario_id, n_scenarios,
    scenario$outcome_scenario, scenario$outcome_label,
    scenario$miss_pattern, scenario$rho
  ))

  rep_list <- vector("list", n_reps)

  for (rep in seq_len(n_reps)) {

    seed <- seed0 + (scenario_id - 1) * n_reps + rep

    data <- generate_data(
      n_rct  = n_rct, n_rwd  = n_rwd, n_test = n_test,
      outcome_scenario = scenario$outcome_scenario,
      rho = scenario$rho, sigma = NULL,
      alpha0 = alpha0, alpha1 = alpha1, alpha5 = alpha5,
      beta5 = beta5, gamma0 = gamma0, gamma1 = gamma1,
      seed = seed
    )

    miss <- impose_missingness(
      XA           = data$XA_train,
      S            = data$S_train,
      miss_pattern = scenario$miss_pattern,
      pi_mcar      = pi_mcar
    )

    t0 <- proc.time()
    ev <- tryCatch(
      run_bartmachine(data, miss, bart_settings),
      error = function(e) {
        cat(sprintf("  Rep %d error: %s\n", rep,
                    conditionMessage(e)))
        null_result()
      }
    )
    elapsed <- (proc.time() - t0)[3]

    row <- data.frame(
      rep = rep, method = "BART+MIA", time = elapsed,
      stringsAsFactors = FALSE
    )
    for (nm in names(ev)) {
      row[[nm]] <- ifelse(is.null(ev[[nm]]), NA, ev[[nm]])
    }
    rep_list[[rep]] <- row

    if (rep %% 10 == 0) {
      cat(sprintf("  Completed %d / %d reps\n",
                  rep, n_reps))
    }
  }

  results <- do.call(rbind, rep_list)

  # Handle error results
  if (is.data.frame(results)) {
    valid <- results
  } else if (is.list(results)) {
    keep <- vapply(results, is.data.frame, logical(1))
    if (any(keep)) {
      valid <- do.call(rbind, results[keep])
    } else {
      valid <- NULL
    }
  } else {
    valid <- NULL
  }

  if (is.null(valid) || nrow(valid) == 0) {
    cat("  ** All replications failed — skipping.\n\n")
    all_scenario_results[[scenario_id]] <- NULL
    next
  }
  results <- valid

  # Quick summary
  summary_df <- aggregate(
    cbind(rmse_m_test, cate_rmse_test, cate_bias_test,
          ate_bias, cate_coverage, cate_ci_width,
          ate_coverage) ~ method,
    data = results,
    FUN = function(x) mean(x, na.rm = TRUE),
    na.action = na.pass
  )
  cat("\n")
  print(summary_df, digits = 3, row.names = FALSE)
  cat("\n")

  results$scenario_id      <- scenario_id
  results$outcome_scenario <- scenario$outcome_scenario
  results$miss_pattern     <- scenario$miss_pattern
  results$rho              <- scenario$rho
  all_scenario_results[[scenario_id]] <- results
}

# ============================================================================
# Save combined results
# ============================================================================

combined <- do.call(rbind, all_scenario_results)
rownames(combined) <- NULL

combined_file <- file.path(output_dir,
                           "irs_v2_bartm_output.rds")
saveRDS(combined, file = combined_file)

cat(sprintf("All %d scenarios complete.\n", n_scenarios))
cat(sprintf("Combined results: %s (%d rows)\n",
            combined_file, nrow(combined)))

