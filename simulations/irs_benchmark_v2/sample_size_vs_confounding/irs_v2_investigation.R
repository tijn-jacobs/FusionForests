# ============================================================================
# Investigation: Is IRS collapse under block_rwd due to
# sample size asymmetry or confounding?
#
# Usage:
#   Rscript irs_v2_investigation.R [num_cores]
#
# Output:
#   $TMPDIR/irs_v2_investigation_output.rds
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
seed0   <- 7777L
n_test  <- 500L

# Fixed DGP parameters
beta5   <- 1
gamma0  <- 2
gamma1  <- 1
alpha0  <- 0
alpha1  <- 0.5
pi_mcar <- 0.3

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

# ============================================================================
# DGP
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
    "1" = m_scenario1, "2" = m_scenario2,
    "3" = m_scenario3
  )
  m_vals <- m_fn(X, A)
  as.numeric(sqrt(var(m_vals) / target_snr))
}

generate_data <- function(n_rct, n_rwd, n_test = 500,
                          outcome_scenario,
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
    "1" = function(X, A) m_scenario1(X, A, beta5),
    "2" = function(X, A) m_scenario2(X, A, gamma0, gamma1),
    "3" = function(X, A) m_scenario3(X, A, beta5, gamma0,
                                     gamma1)
  )
  tau_fn <- switch(as.character(outcome_scenario),
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

impose_block_rwd <- function(XA, S) {
  n <- nrow(XA)
  XA_miss <- XA
  M <- integer(n)
  idx <- which(S == 0)
  M[idx] <- 1L
  XA_miss[idx, 5] <- NaN
  list(XA_miss = XA_miss, M = M)
}

# ============================================================================
# Method wrappers
# ============================================================================

bart_args <- function(s) {
  list(
    number_of_trees       = s$number_of_trees,
    N_post                = s$N_post,
    N_burn                = s$N_burn,
    verbose               = FALSE,
    store_posterior_sample = TRUE
  )
}

fit_bart_with_cate <- function(y, XA_train, XA_test,
                               XA_test_1, XA_test_0,
                               XA_train_1, XA_train_0,
                               irs, s) {
  n_test  <- nrow(XA_test)
  n_train <- nrow(XA_train)

  fit_all <- do.call(SimpleBART, c(
    list(y = y, X_train = XA_train,
         X_test = rbind(XA_test, XA_test_1, XA_test_0,
                        XA_train_1, XA_train_0),
         irs = as.integer(irs)),
    bart_args(s)
  ))

  idx1 <- 1:n_test
  idx2 <- (n_test + 1):(2 * n_test)
  idx3 <- (2 * n_test + 1):(3 * n_test)
  idx4 <- (3 * n_test + 1):(3 * n_test + n_train)
  idx5 <- (3 * n_test + n_train + 1):(3 * n_test +
                                       2 * n_train)

  fit_obs <- list(
    train_predictions = fit_all$train_predictions,
    test_predictions  = fit_all$test_predictions[idx1]
  )
  fit_1 <- list(
    train_predictions = fit_all$test_predictions[idx4],
    test_predictions  = fit_all$test_predictions[idx2]
  )
  fit_0 <- list(
    train_predictions = fit_all$test_predictions[idx5],
    test_predictions  = fit_all$test_predictions[idx3]
  )

  if (!is.null(fit_all$test_predictions_sample)) {
    fit_1$test_predictions_sample <-
      fit_all$test_predictions_sample[, idx2, drop = FALSE]
    fit_0$test_predictions_sample <-
      fit_all$test_predictions_sample[, idx3, drop = FALSE]
  }

  list(fit = fit_obs, fit_1 = fit_1, fit_0 = fit_0)
}

run_oracle <- function(data, s) {
  fits <- fit_bart_with_cate(
    y = data$y,
    XA_train = data$XA_train, XA_test = data$XA_test,
    XA_test_1 = data$XA_test_1,
    XA_test_0 = data$XA_test_0,
    XA_train_1 = data$XA_train_1,
    XA_train_0 = data$XA_train_0,
    irs = 0L, s = s
  )
  evaluate_fit(fits$fit, fits$fit_1, fits$fit_0,
               data$m_train, data$m_test,
               data$tau_train, data$tau_test)
}

run_irs <- function(data, miss, s, irs_mode = 2L) {
  fits <- fit_bart_with_cate(
    y = data$y,
    XA_train = miss$XA_miss, XA_test = data$XA_test,
    XA_test_1 = data$XA_test_1,
    XA_test_0 = data$XA_test_0,
    XA_train_1 = data$XA_train_1,
    XA_train_0 = data$XA_train_0,
    irs = irs_mode, s = s
  )
  evaluate_fit(fits$fit, fits$fit_1, fits$fit_0,
               data$m_train, data$m_test,
               data$tau_train, data$tau_test)
}

run_complete_covariates <- function(data, s) {
  drop5 <- function(M) M[, -5, drop = FALSE]
  fits <- fit_bart_with_cate(
    y = data$y,
    XA_train = drop5(data$XA_train),
    XA_test = drop5(data$XA_test),
    XA_test_1 = drop5(data$XA_test_1),
    XA_test_0 = drop5(data$XA_test_0),
    XA_train_1 = drop5(data$XA_train_1),
    XA_train_0 = drop5(data$XA_train_0),
    irs = 0L, s = s
  )
  evaluate_fit(fits$fit, fits$fit_1, fits$fit_0,
               data$m_train, data$m_test,
               data$tau_train, data$tau_test)
}

run_missforest <- function(data, miss, s) {
  XA_na <- miss$XA_miss
  XA_na[is.nan(XA_na)] <- NA

  imputed <- missForest::missForest(
    as.data.frame(XA_na), verbose = FALSE
  )
  XA_imp <- as.matrix(imputed$ximp)

  XA_imp_1 <- XA_imp; XA_imp_1[, 6] <- 1
  XA_imp_0 <- XA_imp; XA_imp_0[, 6] <- 0

  fits <- fit_bart_with_cate(
    y = data$y,
    XA_train = XA_imp, XA_test = data$XA_test,
    XA_test_1 = data$XA_test_1,
    XA_test_0 = data$XA_test_0,
    XA_train_1 = XA_imp_1,
    XA_train_0 = XA_imp_0,
    irs = 0L, s = s
  )
  evaluate_fit(fits$fit, fits$fit_1, fits$fit_0,
               data$m_train, data$m_test,
               data$tau_train, data$tau_test)
}

null_result <- function() {
  list(
    rmse_m_train = NA, rmse_m_test = NA, mae_m_test = NA,
    cate_bias_test = NA, cate_rmse_test = NA,
    cate_bias_train = NA, cate_rmse_train = NA,
    ate_bias = NA, ate_hat = NA, ate_true = NA,
    cate_coverage = NA, cate_ci_width = NA,
    avg_post_var = NA,
    ate_coverage = NA, ate_ci_width = NA
  )
}

# ============================================================================
# Scenario grid
# ============================================================================

grid <- expand.grid(
  outcome_scenario = 1:3,
  sample_config    = c("150_350", "350_150", "500_500"),
  alpha5           = c(0, 0.5),
  stringsAsFactors = FALSE
)
grid$scenario_id <- seq_len(nrow(grid))

parse_config <- function(cfg) {
  parts <- as.integer(strsplit(cfg, "_")[[1]])
  list(n_rct = parts[1], n_rwd = parts[2])
}

cat(sprintf("Cores: %d | Reps: %d | Scenarios: %d\n",
            num_cores, n_reps, nrow(grid)))
cat(sprintf("Output directory: %s\n\n", output_dir))

# ============================================================================
# Simulation loop
# ============================================================================

all_results <- list()

for (i in seq_len(nrow(grid))) {

  sc    <- grid[i, ]
  sizes <- parse_config(sc$sample_config)

  cat(sprintf(
    "=== %d/%d: outcome=%d | n_rct=%d n_rwd=%d | alpha5=%.1f ===\n",
    i, nrow(grid), sc$outcome_scenario,
    sizes$n_rct, sizes$n_rwd, sc$alpha5
  ))

  results <- foreach(
    rep = seq_len(n_reps),
    .combine  = rbind,
    .packages = c("FusionForests", "MASS", "missForest"),
    .errorhandling = "pass"
  ) %dopar% {

    seed <- seed0 + (i - 1) * n_reps + rep

    data <- generate_data(
      n_rct = sizes$n_rct, n_rwd = sizes$n_rwd,
      n_test = n_test,
      outcome_scenario = sc$outcome_scenario,
      rho = 0, sigma = NULL,
      alpha0 = alpha0, alpha1 = alpha1,
      alpha5 = sc$alpha5,
      beta5 = beta5, gamma0 = gamma0, gamma1 = gamma1,
      seed = seed
    )

    miss <- impose_block_rwd(data$XA_train, data$S_train)

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

    # Oracle
    t0 <- proc.time()
    ev <- tryCatch(run_oracle(data, bart_settings),
                   error = function(e) null_result())
    idx <- idx + 1L
    rep_rows[[idx]] <- make_row("Oracle", ev,
                                (proc.time() - t0)[3])

    # IRS (mode 1: informed, skip-then-draw)
    t0 <- proc.time()
    ev <- tryCatch(
      run_irs(data, miss, bart_settings, irs_mode = 1L),
      error = function(e) null_result())
    idx <- idx + 1L
    rep_rows[[idx]] <- make_row("IRS (informed)", ev,
                                (proc.time() - t0)[3])

    # IRS (mode 2: informed, draw-then-decide)
    t0 <- proc.time()
    ev <- tryCatch(
      run_irs(data, miss, bart_settings, irs_mode = 2L),
      error = function(e) null_result())
    idx <- idx + 1L
    rep_rows[[idx]] <- make_row("IRS (draw-then-decide)", ev,
                                (proc.time() - t0)[3])

    # IRS (mode 3: uniform random routing)
    t0 <- proc.time()
    ev <- tryCatch(
      run_irs(data, miss, bart_settings, irs_mode = 3L),
      error = function(e) null_result())
    idx <- idx + 1L
    rep_rows[[idx]] <- make_row("IRS (uniform)", ev,
                                (proc.time() - t0)[3])

    # IRS (mode 4: skip-then-draw, exclude NaN from mu)
    t0 <- proc.time()
    ev <- tryCatch(
      run_irs(data, miss, bart_settings, irs_mode = 4L),
      error = function(e) null_result())
    idx <- idx + 1L
    rep_rows[[idx]] <- make_row("IRS (informed, excl mu)",
                                ev, (proc.time() - t0)[3])

    # IRS (mode 5: draw-then-decide, exclude NaN from mu)
    t0 <- proc.time()
    ev <- tryCatch(
      run_irs(data, miss, bart_settings, irs_mode = 5L),
      error = function(e) null_result())
    idx <- idx + 1L
    rep_rows[[idx]] <- make_row("IRS (dtd, excl mu)",
                                ev, (proc.time() - t0)[3])

    # IRS (mode 6: uniform, exclude NaN from mu)
    t0 <- proc.time()
    ev <- tryCatch(
      run_irs(data, miss, bart_settings, irs_mode = 6L),
      error = function(e) null_result())
    idx <- idx + 1L
    rep_rows[[idx]] <- make_row("IRS (uniform, excl mu)",
                                ev, (proc.time() - t0)[3])

    # Complete covariates
    t0 <- proc.time()
    ev <- tryCatch(
      run_complete_covariates(data, bart_settings),
      error = function(e) null_result()
    )
    idx <- idx + 1L
    rep_rows[[idx]] <- make_row("Complete covariates", ev,
                                (proc.time() - t0)[3])

    # MissForest + BART
    t0 <- proc.time()
    ev <- tryCatch(
      run_missforest(data, miss, bart_settings),
      error = function(e) null_result()
    )
    idx <- idx + 1L
    rep_rows[[idx]] <- make_row("MissForest+BART", ev,
                                (proc.time() - t0)[3])

    do.call(rbind, rep_rows)
  }

  # Filter valid results
  if (is.data.frame(results)) {
    valid <- results
  } else if (is.list(results)) {
    keep <- vapply(results, is.data.frame, logical(1))
    valid <- if (any(keep)) do.call(rbind, results[keep])
             else NULL
  } else {
    valid <- NULL
  }

  if (is.null(valid) || nrow(valid) == 0) {
    cat("  ** All reps failed.\n\n")
    next
  }

  valid$outcome_scenario <- sc$outcome_scenario
  valid$sample_config    <- sc$sample_config
  valid$n_rct            <- sizes$n_rct
  valid$n_rwd            <- sizes$n_rwd
  valid$alpha5           <- sc$alpha5
  valid$confounding      <- ifelse(sc$alpha5 > 0,
                                   "Confounded",
                                   "No confounding")

  # Quick summary
  summary_df <- aggregate(
    cbind(cate_rmse_test, cate_bias_test,
          ate_bias, cate_coverage) ~ method,
    data = valid,
    FUN = function(x) mean(x, na.rm = TRUE),
    na.action = na.pass
  )
  cat("\n")
  print(summary_df, digits = 3, row.names = FALSE)
  cat("\n")

  all_results[[i]] <- valid
}

# ============================================================================
# Save results
# ============================================================================

combined <- do.call(rbind, all_results)
rownames(combined) <- NULL

out_file <- file.path(output_dir,
                      "irs_v2_investigation_output.rds")
saveRDS(combined, file = out_file)
cat(sprintf("\nResults saved to %s (%d rows)\n",
            out_file, nrow(combined)))

# ============================================================================
# Summary table
# ============================================================================

agg <- aggregate(
  cbind(rmse_m_test, cate_rmse_test, cate_bias_test,
        ate_bias, cate_coverage, cate_ci_width,
        ate_coverage) ~
    method + outcome_scenario + sample_config +
    confounding,
  data = combined,
  FUN = function(x) mean(x, na.rm = TRUE),
  na.action = na.pass
)

num_cols <- sapply(agg, is.numeric)
agg[num_cols] <- round(agg[num_cols], 3)

csv_file <- file.path(output_dir,
                      "irs_v2_investigation_summary.csv")
write.csv(agg, file = csv_file, row.names = FALSE)
cat(sprintf("Summary saved to %s\n", csv_file))

# ============================================================================
# Print key comparison table
# ============================================================================

cat("\n")
cat("============================================================\n")
cat("  IRS: CATE RMSE (test) by condition\n")
cat("============================================================\n\n")

irs_data <- agg[agg$method == "IRS",
  c("outcome_scenario", "sample_config", "confounding",
    "cate_rmse_test", "cate_bias_test", "ate_bias",
    "cate_coverage")
]
irs_data <- irs_data[order(irs_data$outcome_scenario,
                           irs_data$confounding,
                           irs_data$sample_config), ]
print(irs_data, row.names = FALSE)

cat("\n")
cat("============================================================\n")
cat("  Complete covariates: CATE RMSE (test) for reference\n")
cat("============================================================\n\n")

cc_data <- agg[agg$method == "Complete covariates",
  c("outcome_scenario", "sample_config", "confounding",
    "cate_rmse_test", "cate_bias_test")
]
cc_data <- cc_data[order(cc_data$outcome_scenario,
                         cc_data$confounding,
                         cc_data$sample_config), ]
print(cc_data, row.names = FALSE)
