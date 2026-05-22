# ============================================================================
# Benchmark: IRS vs competitors for BART with missing data
#
# Methods compared:
#   1. Oracle         — complete data, no missingness (ceiling)
#   2. IRS (informed) — draw-then-decide (irs = 2)
#   3. IRS (uniform)  — uniform P=0.5 random routing (irs = 3)
#   4. Complete case   — drop rows with NaN, fit standard BART
#   5. bartMachine MIA — Missing Incorporated in Attributes (Kapelner & Bleich)
#   6. missForest+BART — impute with missForest, then fit standard BART
#
# Run after: R CMD INSTALL .
# Usage:     Rscript simulations/run_simulation.R
#            Rscript simulations/run_simulation.R --n_train 500 --p 10
# ============================================================================

library(FusionForests)
# source("simulations/dgp.R")
# source("simulations/evaluation.R")

# ============================================================================
# Default settings (override via command-line args or by editing this section)
# ============================================================================

settings <- list(
  # DGP
  n_train   = 200,
  n_test    = 1000,
  p         = 5,
  f_type    = "additive",
  beta4     = 1.0,
  rho       = 0.5,
  snr       = 3,
  miss_col  = 4,

  # Missingness
  mechanism = "mar",
  miss_frac = 0.8,

  # BART hyperparameters
  number_of_trees = 200,
  N_post          = 2000,
  N_burn          = 1000,

  # Simulation
  n_reps  = 20,
  seed0   = floor(10^3*runif(1)),

  # Which methods to run (set FALSE to skip)
  run_oracle         = TRUE,
  run_irs_informed   = TRUE,
  run_irs_uniform    = TRUE,
  run_complete_case  = TRUE,
  run_bartmachine    = TRUE,
  run_missforest     = TRUE
)

# ============================================================================
# Parse command-line arguments (--key value pairs override defaults)
# ============================================================================

parse_cli_args <- function(defaults) {
  args <- commandArgs(trailingOnly = TRUE)
  if (length(args) == 0) return(defaults)

  i <- 1
  while (i <= length(args)) {
    key <- sub("^--", "", args[i])
    if (key %in% names(defaults)) {
      val <- args[i + 1]
      # Coerce to the same type as the default
      if (is.logical(defaults[[key]])) {
        defaults[[key]] <- as.logical(val)
      } else if (is.integer(defaults[[key]])) {
        defaults[[key]] <- as.integer(val)
      } else if (is.numeric(defaults[[key]])) {
        defaults[[key]] <- as.numeric(val)
      } else {
        defaults[[key]] <- val
      }
      i <- i + 2
    } else {
      i <- i + 1
    }
  }
  defaults
}

settings <- parse_cli_args(settings)

# ============================================================================
# Helper: fit a method and evaluate it
# ============================================================================

# Shared BART settings for SimpleBART calls
bart_args <- function(s) {
  list(
    number_of_trees       = s$number_of_trees,
    N_post                = s$N_post,
    N_burn                = s$N_burn,
    verbose               = FALSE,
    store_posterior_sample = TRUE
  )
}

# Fit SimpleBART wrapper
fit_simple_bart <- function(y, X_train, X_test, irs, s) {
  do.call(SimpleBART, c(
    list(y = y, X_train = X_train, X_test = X_test, irs = as.integer(irs)),
    bart_args(s)
  ))
}

# ============================================================================
# Check for optional packages
# ============================================================================

options(java.parameters = c("-Xmx4g", "--add-modules=jdk.incubator.vector"))
has_bartmachine <- requireNamespace("bartMachine", quietly = TRUE)
has_missforest  <- requireNamespace("missForest", quietly = TRUE)

if (settings$run_bartmachine && !has_bartmachine) {
  message("bartMachine not installed — skipping bartMachine MIA.")
  settings$run_bartmachine <- FALSE
}
if (settings$run_missforest && !has_missforest) {
  message("missForest not installed — skipping missForest + BART.")
  settings$run_missforest <- FALSE
}

# ============================================================================
# Print settings
# ============================================================================

cat("=== Simulation settings ===\n")
cat(sprintf("  DGP: %s, p=%d, n_train=%d, n_test=%d, beta4=%g, rho=%g, snr=%g\n",
            settings$f_type, settings$p, settings$n_train, settings$n_test,
            settings$beta4, settings$rho, settings$snr))
cat(sprintf("  Missingness: %s (frac=%g) on X%d\n",
            settings$mechanism, settings$miss_frac, settings$miss_col))
cat(sprintf("  BART: %d trees, %d post, %d burn\n",
            settings$number_of_trees, settings$N_post, settings$N_burn))
cat(sprintf("  Replications: %d\n\n", settings$n_reps))

methods_to_run <- c(
  if (settings$run_oracle)        "Oracle",
  if (settings$run_irs_informed)  "IRS (informed)",
  if (settings$run_irs_uniform)   "IRS (uniform)",
  if (settings$run_complete_case) "Complete case",
  if (settings$run_bartmachine)   "bartMachine MIA",
  if (settings$run_missforest)    "missForest+BART"
)
cat("Methods:", paste(methods_to_run, collapse = ", "), "\n\n")

# ============================================================================
# Run replications
# ============================================================================

results <- vector("list", settings$n_reps)

for (rep in seq_len(settings$n_reps)) {

  seed <- settings$seed0 + rep
  cat(sprintf("--- Replication %d/%d (seed %d) ---\n", rep, settings$n_reps, seed))

  # Generate data
  data <- generate_data(
    n_train  = settings$n_train,
    n_test   = settings$n_test,
    p        = settings$p,
    rho      = settings$rho,
    f_type   = settings$f_type,
    beta4    = settings$beta4,
    snr      = settings$snr,
    miss_col = settings$miss_col,
    seed     = seed
  )

  # Impose missingness on training data
  miss <- impose_missingness(
    data$X_train,
    mechanism = settings$mechanism,
    miss_col  = settings$miss_col,
    miss_frac = settings$miss_frac
  )

  rep_results <- list()

  # --- 1. Oracle (complete data) ---
  if (settings$run_oracle) {
    fit <- fit_simple_bart(data$y_train, data$X_train, data$X_test, irs = 0L, settings)
    ev  <- evaluate_fit(fit, data$f_train, data$f_test)
    rep_results[["Oracle"]] <- ev
  }

  # --- 2. IRS informed (draw-then-decide) ---
  if (settings$run_irs_informed) {
    fit <- fit_simple_bart(data$y_train, miss$X_train_miss, data$X_test, irs = 2L, settings)
    ev  <- evaluate_fit(fit, data$f_train, data$f_test)
    rep_results[["IRS (informed)"]] <- ev
  }

  # --- 3. IRS uniform ---
  if (settings$run_irs_uniform) {
    fit <- fit_simple_bart(data$y_train, miss$X_train_miss, data$X_test, irs = 3L, settings)
    ev  <- evaluate_fit(fit, data$f_train, data$f_test)
    rep_results[["IRS (uniform)"]] <- ev
  }

  # --- 4. Complete case ---
  if (settings$run_complete_case) {
    obs_rows <- apply(miss$X_train_miss, 1, function(r) !any(is.nan(r)))
    X_cc <- miss$X_train_miss[obs_rows, , drop = FALSE]
    y_cc <- data$y_train[obs_rows]
    f_train_cc <- data$f_train[obs_rows]

    if (sum(obs_rows) >= 30) {
      fit <- fit_simple_bart(y_cc, X_cc, data$X_test, irs = 0L, settings)
      ev  <- list(
        rmse_train = compute_rmse(fit$train_predictions, f_train_cc),
        rmse_test  = compute_rmse(fit$test_predictions, data$f_test)
      )
      if (!is.null(fit$train_predictions_sample)) {
        ev$coverage <- compute_coverage(fit$train_predictions_sample, f_train_cc)
        ev$width    <- compute_interval_width(fit$train_predictions_sample)
      }
    } else {
      ev <- list(rmse_train = NA, rmse_test = NA, coverage = NA, width = NA)
    }
    rep_results[["Complete case"]] <- ev
  }

  # --- 5. bartMachine MIA ---
  if (settings$run_bartmachine) {
    # bartMachine uses NA (not NaN) for missingness and handles it via MIA
    X_train_na <- miss$X_train_miss
    X_train_na[is.nan(X_train_na)] <- NA
    X_train_df <- as.data.frame(X_train_na)
    X_test_df  <- as.data.frame(data$X_test)

    bm <- bartMachine::bartMachine(
      X_train_df, data$y_train,
      num_trees            = settings$number_of_trees,
      num_burn_in          = settings$N_burn,
      num_iterations_after_burn_in = settings$N_post,
      use_missing_data     = TRUE,
      use_missing_data_dummies_as_covars = TRUE,
      verbose              = FALSE
    )

    pred_train <- bartMachine::bart_machine_get_posterior(bm, X_train_df)
    pred_test  <- bartMachine::bart_machine_get_posterior(bm, X_test_df)

    ev <- list(
      rmse_train = compute_rmse(pred_train$y_hat, data$f_train),
      rmse_test  = compute_rmse(pred_test$y_hat, data$f_test),
      coverage   = compute_coverage(t(pred_test$y_hat_posterior_samples), data$f_test),
      width      = compute_interval_width(t(pred_test$y_hat_posterior_samples))
    )

    rm(bm)
    gc()
    rep_results[["bartMachine MIA"]] <- ev
  }

  # --- 6. missForest + BART ---
  if (settings$run_missforest) {
    X_train_na <- miss$X_train_miss
    X_train_na[is.nan(X_train_na)] <- NA

    imputed <- missForest::missForest(as.data.frame(X_train_na), verbose = FALSE)
    X_imputed <- as.matrix(imputed$ximp)

    fit <- fit_simple_bart(data$y_train, X_imputed, data$X_test, irs = 0L, settings)
    ev  <- evaluate_fit(fit, data$f_train, data$f_test)
    rep_results[["missForest+BART"]] <- ev
  }

  # Collect into a data.frame
  rep_df <- do.call(rbind, lapply(names(rep_results), function(m) {
    ev <- rep_results[[m]]
    data.frame(
      rep        = rep,
      method     = m,
      rmse_train = ev$rmse_train,
      rmse_test  = ev$rmse_test,
      coverage   = ifelse(is.null(ev$coverage), NA, ev$coverage),
      width      = ifelse(is.null(ev$width),    NA, ev$width),
      stringsAsFactors = FALSE
    )
  }))

  results[[rep]] <- rep_df
}

# ============================================================================
# Summary
# ============================================================================

results_df <- do.call(rbind, results)

cat("\n=== Per-replication results ===\n\n")
print(results_df, digits = 3, row.names = FALSE)

cat("\n=== Summary (mean +/- SE across replications) ===\n\n")

summary_df <- aggregate(
  cbind(rmse_train, rmse_test, coverage, width) ~ method,
  data = results_df, FUN = function(x) mean(x, na.rm = TRUE)
)

se_df <- aggregate(
  cbind(rmse_train, rmse_test, coverage, width) ~ method,
  data = results_df,
  FUN = function(x) {
    x <- x[!is.na(x)]
    if (length(x) > 1) sd(x) / sqrt(length(x)) else NA
  }
)

# Merge mean and SE
names(se_df)[-1] <- paste0(names(se_df)[-1], "_se")
summary_out <- merge(summary_df, se_df, by = "method")

# Reorder columns for readability
col_order <- c("method",
               "rmse_train", "rmse_train_se",
               "rmse_test",  "rmse_test_se",
               "coverage",   "coverage_se",
               "width",      "width_se")
summary_out <- summary_out[, col_order]

print(summary_out, digits = 3, row.names = FALSE)

# # Save results
# outfile <- sprintf("simulations/results_%s_p%d_n%d_%s_frac%02d.rds",
#                    settings$f_type, settings$p, settings$n_train,
#                    settings$mechanism, settings$miss_frac * 100)
# saveRDS(list(settings = settings, results = results_df, summary = summary_out),
#         file = outfile)
# cat(sprintf("\nResults saved to %s\n", outfile))
