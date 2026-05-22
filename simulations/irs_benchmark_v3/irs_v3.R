# ============================================================================
# IRS Benchmark v3 — BCF decomposition: Y = mu(X,e) + tau(X)*A + eps
#
# Same DGP and scenarios as v2, but using SimpleBCF (two-forest)
# instead of SimpleBART (single forest).
#
# Methods: Oracle, IRS, Complete case, Complete covariates,
#          MissForest+BCF
#
# Usage:
#   Rscript irs_v3.R [num_cores]
#
# Output:
#   $TMPDIR/irs_v3_output.rds
# ============================================================================

library(FusionForests)
library(doParallel)
library(foreach)
library(MASS)
library(missForest)

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

registerDoParallel(cores = num_cores)

# ============================================================================
# Settings
# ============================================================================

n_reps  <- num_cores
n_rct   <- 150L
n_rwd   <- 350L
n_test  <- 500L
seed0   <- 3030L

# DGP parameters
beta5   <- 1
gamma0  <- 2
gamma1  <- 1
alpha0  <- 0
alpha1  <- 0.5
alpha5  <- 0.5
pi_mcar <- 0.3

# BCF settings
bcf_settings <- list(
  number_of_trees_prog  = 200L,
  number_of_trees_treat = 100L,
  N_post                = 2000L,
  N_burn                = 2000L
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
# DGP functions (same as v2)
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
                          gamma1 = 1, seed = NULL) {
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

  # Estimate propensity score (logistic on X1..X5)
  ps_fit <- glm(A ~ X, family = binomial)
  e_train <- fitted(ps_fit)
  e_test  <- plogis(cbind(1, X_test) %*% coef(ps_fit))

  list(
    y = y, X_train = X, A_train = A, S_train = S,
    m_train = m_train, tau_train = tau_train,
    X_test = X_test, A_test = A_test,
    m_test = m_test, tau_test = tau_test,
    e_train = as.numeric(e_train),
    e_test  = as.numeric(e_test),
    sigma = sigma, n_rct = n_rct, n_rwd = n_rwd
  )
}

impose_missingness <- function(X, S,
                               miss_pattern = c("block_rct",
                                                "block_rwd",
                                                "mcar"),
                               pi_mcar = 0.3) {
  miss_pattern <- match.arg(miss_pattern)
  n <- nrow(X)
  X_miss <- X
  M <- integer(n)

  if (miss_pattern == "block_rct") {
    idx <- which(S == 1)
  } else if (miss_pattern == "block_rwd") {
    idx <- which(S == 0)
  } else {
    idx <- which(rbinom(n, 1, pi_mcar) == 1)
  }
  M[idx] <- 1L
  X_miss[idx, 5] <- NaN

  list(X_miss = X_miss, M = M)
}

# ============================================================================
# Scenario grid (same as v2)
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
# BCF method wrappers
# ============================================================================

#' Fit SimpleBCF and return structured results for evaluation
#'
#' SimpleBCF directly gives tau(x) from the treatment forest,
#' so no counterfactual trick is needed.
fit_bcf <- function(y, X_train, A_train, e_train,
                    X_test, A_test, e_test,
                    irs, s) {
  fit <- SimpleBCF(
    y = y, X_train = X_train,
    treatment_indicator = as.integer(A_train),
    propensity_score = e_train,
    X_test = X_test,
    treatment_indicator_test = as.integer(A_test),
    propensity_score_test = e_test,
    number_of_trees_prog  = s$number_of_trees_prog,
    number_of_trees_treat = s$number_of_trees_treat,
    N_post  = s$N_post,
    N_burn  = s$N_burn,
    verbose = FALSE,
    irs     = as.integer(irs),
    store_posterior_sample = TRUE
  )

  # Build objects compatible with evaluate_fit
  fit_obs <- list(
    train_predictions = fit$train_predictions,
    test_predictions  = fit$test_predictions
  )
  # tau(x) comes directly from the treatment forest
  fit_1 <- list(
    train_predictions = fit$train_predictions_treat,
    test_predictions  = fit$test_predictions_treat
  )
  # For the "fit_0" slot we store zeros (tau = fit_1 - 0)
  fit_0 <- list(
    train_predictions = rep(0, length(fit$train_predictions)),
    test_predictions  = rep(0, length(fit$test_predictions))
  )

  if (!is.null(fit$test_predictions_treat_sample)) {
    fit_1$test_predictions_sample <-
      fit$test_predictions_treat_sample
    fit_0$test_predictions_sample <-
      matrix(0, nrow = nrow(fit$test_predictions_treat_sample),
                ncol = ncol(fit$test_predictions_treat_sample))
  }

  list(fit = fit_obs, fit_1 = fit_1, fit_0 = fit_0)
}

# --- Oracle BCF ---

run_oracle <- function(data, s) {
  fits <- fit_bcf(
    y = data$y, X_train = data$X_train,
    A_train = data$A_train, e_train = data$e_train,
    X_test = data$X_test, A_test = data$A_test,
    e_test = data$e_test,
    irs = 0L, s = s
  )
  evaluate_fit(
    fits$fit, fits$fit_1, fits$fit_0,
    data$m_train, data$m_test,
    data$tau_train, data$tau_test
  )
}

# --- IRS BCF ---

run_irs <- function(data, miss, s, irs_mode = 2L) {
  fits <- fit_bcf(
    y = data$y, X_train = miss$X_miss,
    A_train = data$A_train, e_train = data$e_train,
    X_test = data$X_test, A_test = data$A_test,
    e_test = data$e_test,
    irs = irs_mode, s = s
  )
  evaluate_fit(
    fits$fit, fits$fit_1, fits$fit_0,
    data$m_train, data$m_test,
    data$tau_train, data$tau_test
  )
}

# --- Complete case BCF ---

run_complete_case <- function(data, miss, s) {
  obs_rows <- !is.nan(miss$X_miss[, 5])
  n_obs <- sum(obs_rows)
  if (n_obs < 30) return(null_result())

  fits <- fit_bcf(
    y = data$y[obs_rows],
    X_train = miss$X_miss[obs_rows, , drop = FALSE],
    A_train = data$A_train[obs_rows],
    e_train = data$e_train[obs_rows],
    X_test = data$X_test, A_test = data$A_test,
    e_test = data$e_test,
    irs = 0L, s = s
  )
  evaluate_fit(
    fits$fit, fits$fit_1, fits$fit_0,
    data$m_train[obs_rows], data$m_test,
    data$tau_train[obs_rows], data$tau_test
  )
}

# --- Complete covariates BCF (drop X5) ---

run_complete_covariates <- function(data, s) {
  fits <- fit_bcf(
    y = data$y,
    X_train = data$X_train[, -5, drop = FALSE],
    A_train = data$A_train, e_train = data$e_train,
    X_test = data$X_test[, -5, drop = FALSE],
    A_test = data$A_test, e_test = data$e_test,
    irs = 0L, s = s
  )
  evaluate_fit(
    fits$fit, fits$fit_1, fits$fit_0,
    data$m_train, data$m_test,
    data$tau_train, data$tau_test
  )
}

# --- MissForest + BCF ---

run_missforest <- function(data, miss, s) {
  X_na <- miss$X_miss
  X_na[is.nan(X_na)] <- NA

  imputed <- missForest::missForest(
    as.data.frame(X_na), verbose = FALSE
  )
  X_imp <- as.matrix(imputed$ximp)

  fits <- fit_bcf(
    y = data$y, X_train = X_imp,
    A_train = data$A_train, e_train = data$e_train,
    X_test = data$X_test, A_test = data$A_test,
    e_test = data$e_test,
    irs = 0L, s = s
  )
  evaluate_fit(
    fits$fit, fits$fit_1, fits$fit_0,
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

  results <- foreach(
    rep = seq_len(n_reps),
    .combine  = rbind,
    .packages = c("FusionForests", "MASS", "missForest"),
    .errorhandling = "stop"
  ) %dopar% {

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
      X            = data$X_train,
      S            = data$S_train,
      miss_pattern = scenario$miss_pattern,
      pi_mcar      = pi_mcar
    )

    make_row <- function(method_name, ev, elapsed) {
      row <- data.frame(
        rep = rep, method = method_name, time = elapsed,
        stringsAsFactors = FALSE
      )
      for (nm in names(ev)) {
        row[[nm]] <- ifelse(is.null(ev[[nm]]), NA,
                            ev[[nm]])
      }
      row
    }

    rep_rows <- list()
    idx <- 0L

    # Oracle BCF
    t0 <- proc.time()
    ev <- tryCatch(run_oracle(data, bcf_settings),
                   error = function(e) null_result())
    idx <- idx + 1L
    rep_rows[[idx]] <- make_row("Oracle", ev,
                                (proc.time() - t0)[3])

    # IRS BCF
    t0 <- proc.time()
    ev <- tryCatch(
      run_irs(data, miss, bcf_settings, irs_mode = 2L),
      error = function(e) null_result()
    )
    idx <- idx + 1L
    rep_rows[[idx]] <- make_row("IRS", ev,
                                (proc.time() - t0)[3])

    # Complete case BCF
    t0 <- proc.time()
    ev <- tryCatch(
      run_complete_case(data, miss, bcf_settings),
      error = function(e) null_result()
    )
    idx <- idx + 1L
    rep_rows[[idx]] <- make_row("Complete case", ev,
                                (proc.time() - t0)[3])

    # Complete covariates BCF
    t0 <- proc.time()
    ev <- tryCatch(
      run_complete_covariates(data, bcf_settings),
      error = function(e) null_result()
    )
    idx <- idx + 1L
    rep_rows[[idx]] <- make_row("Complete covariates", ev,
                                (proc.time() - t0)[3])

    # MissForest + BCF
    t0 <- proc.time()
    ev <- tryCatch(
      run_missforest(data, miss, bcf_settings),
      error = function(e) null_result()
    )
    idx <- idx + 1L
    rep_rows[[idx]] <- make_row("MissForest+BCF", ev,
                                (proc.time() - t0)[3])

    do.call(rbind, rep_rows)
  }

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

combined_file <- file.path(output_dir, "irs_v3_output.rds")
saveRDS(combined, file = combined_file)

cat(sprintf("All %d scenarios complete.\n", n_scenarios))
cat(sprintf("Combined results: %s (%d rows)\n",
            combined_file, nrow(combined)))
