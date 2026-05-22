# ============================================================================
# IRS Benchmark — Main simulation
#
# Methods: Oracle, IRS (informed), IRS (uniform), Complete case,
#          missForest+BART
#
# Usage:
#   Rscript irs_benchmark.R [num_cores]
#
# Output:
#   $TMPDIR/irs_benchmark_output.rds
# ============================================================================

library(FusionForests)
library(doParallel)
library(foreach)
library(missForest)

# source("irs_benchmark_common.R")
source("/Users/tijnjacobs/Library/CloudStorage/OneDrive-VrijeUniversiteitAmsterdam/Documents/GitHub/FusionForests/simulations/irs_benchmark/irs_benchmark_common.R")

# ============================================================================
# Methods
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

fit_simple_bart <- function(y, X_train, X_test, irs, s) {
  do.call(SimpleBART, c(
    list(y = y, X_train = X_train, X_test = X_test,
         irs = as.integer(irs)),
    bart_args(s)
  ))
}

# --- Oracle -----------------------------------------------------------------

run_oracle <- function(data, miss, scenario, s) {
  if (scenario$pattern == "predictive") {
    X_oracle <- cbind(data$X_train,
                      miss$M[, scenario$miss_cols, drop = FALSE])
    X_oracle_test <- cbind(data$X_test,
      matrix(0L, nrow = nrow(data$X_test),
             ncol = length(scenario$miss_cols)))
    f_train <- miss$f_true_predictive
    fit <- fit_simple_bart(miss$y, X_oracle, X_oracle_test,
                           irs = 0L, s)
  } else {
    f_train <- data$f_train
    fit <- fit_simple_bart(data$y_train, data$X_train,
                           data$X_test, irs = 0L, s)
  }
  evaluate_fit(fit, f_train, data$f_test)
}

# --- IRS (informed) ---------------------------------------------------------

run_irs_informed <- function(data, miss, scenario, s) {
  y_fit <- if (scenario$pattern == "predictive") {
    miss$y
  } else {
    data$y_train
  }
  f_train <- if (scenario$pattern == "predictive") {
    miss$f_true_predictive
  } else {
    data$f_train
  }
  fit <- fit_simple_bart(y_fit, miss$X_miss, data$X_test,
                         irs = 2L, s)
  evaluate_fit(fit, f_train, data$f_test)
}

# --- IRS (uniform) ----------------------------------------------------------

run_irs_uniform <- function(data, miss, scenario, s) {
  y_fit <- if (scenario$pattern == "predictive") {
    miss$y
  } else {
    data$y_train
  }
  f_train <- if (scenario$pattern == "predictive") {
    miss$f_true_predictive
  } else {
    data$f_train
  }
  fit <- fit_simple_bart(y_fit, miss$X_miss, data$X_test,
                         irs = 3L, s)
  evaluate_fit(fit, f_train, data$f_test)
}

# --- Complete case ----------------------------------------------------------

run_complete_case <- function(data, miss, scenario, s) {
  y_fit <- if (scenario$pattern == "predictive") {
    miss$y
  } else {
    data$y_train
  }
  f_train <- if (scenario$pattern == "predictive") {
    miss$f_true_predictive
  } else {
    data$f_train
  }

  obs_rows <- apply(miss$X_miss, 1,
                    function(r) !any(is.nan(r)))
  n_obs <- sum(obs_rows)

  if (n_obs < 30) {
    return(list(rmse_train = NA, rmse_test = NA,
                bias_test = NA, mae_test = NA,
                coverage = NA, width = NA))
  }

  X_cc       <- miss$X_miss[obs_rows, , drop = FALSE]
  y_cc       <- y_fit[obs_rows]
  f_train_cc <- f_train[obs_rows]

  fit <- fit_simple_bart(y_cc, X_cc, data$X_test,
                         irs = 0L, s)

  ev <- list(
    rmse_train = compute_rmse(fit$train_predictions,
                              f_train_cc),
    rmse_test  = compute_rmse(fit$test_predictions,
                              data$f_test),
    bias_test  = compute_bias(fit$test_predictions,
                              data$f_test),
    mae_test   = compute_mae(fit$test_predictions,
                              data$f_test)
  )
  if (!is.null(fit$train_predictions_sample)) {
    ev$coverage <- compute_coverage(
      fit$train_predictions_sample, f_train_cc
    )
    ev$width <- compute_interval_width(
      fit$train_predictions_sample
    )
  }
  ev
}

# --- missForest + BART ------------------------------------------------------

run_missforest <- function(data, miss, scenario, s) {
  y_fit <- if (scenario$pattern == "predictive") {
    miss$y
  } else {
    data$y_train
  }
  f_train <- if (scenario$pattern == "predictive") {
    miss$f_true_predictive
  } else {
    data$f_train
  }

  X_train_na <- miss$X_miss
  X_train_na[is.nan(X_train_na)] <- NA

  imputed   <- missForest::missForest(
    as.data.frame(X_train_na), verbose = FALSE
  )
  X_imputed <- as.matrix(imputed$ximp)

  fit <- fit_simple_bart(y_fit, X_imputed, data$X_test,
                         irs = 0L, s)
  evaluate_fit(fit, f_train, data$f_test)
}

# ============================================================================
# Settings
# ============================================================================

args <- commandArgs(trailingOnly = TRUE)

if (length(args) >= 1) {
  num_cores <- as.integer(args[1]) - 1
} else {
  num_cores <- parallel::detectCores() - 1
}

registerDoParallel(cores = 5) # ADJUST

n_reps <- num_cores

output_dir <- Sys.getenv("TMPDIR", unset = ".")

n_scenarios <- get_n_scenarios()

cat("Number of cores being used (1 free):", num_cores, "\n")
cat(sprintf("Replications per scenario: %d\n", n_reps))
cat(sprintf("Total scenarios: %d\n", n_scenarios))
cat(sprintf("Output directory: %s\n\n", output_dir))

# ============================================================================
# Loop over all scenarios
# ============================================================================

all_scenario_results <- list()

for (scenario_id in seq_len(n_scenarios)) {

  scenario <- get_scenario(scenario_id)

  cat(sprintf(
    "=== Scenario %d / %d: %s | %s | p=%.2f | n=%d ===\n",
    scenario_id, n_scenarios, scenario$model,
    scenario$pattern, scenario$p_miss, scenario$n_train
  ))

  results <- foreach(
    rep = seq_len(n_reps),
    .combine  = rbind,
    .packages = c("FusionForests", "MASS", "missForest"),
    .errorhandling = "pass"
  ) %dopar% {

    seed <- seed0 + (scenario_id - 1) * n_reps + rep

    data <- generate_data(
      n_train = scenario$n_train,
      n_test  = n_test,
      model   = scenario$model,
      d       = scenario$d,
      rho     = rho,
      sigma   = sigma,
      seed    = seed
    )

    miss <- impose_missingness(
      X         = data$X_train,
      y         = data$y_train,
      pattern   = scenario$pattern,
      miss_cols = scenario$miss_cols,
      p_miss    = scenario$p_miss,
      sigma     = sigma
    )

    rep_rows <- list()

    ev <- tryCatch(
      run_oracle(data, miss, scenario, bart_settings),
      error = function(e) list(rmse_train = NA,
                               rmse_test = NA)
    )
    rep_rows[[1]] <- make_row(rep, "Oracle", ev)

    ev <- tryCatch(
      run_irs_informed(data, miss, scenario, bart_settings),
      error = function(e) list(rmse_train = NA,
                               rmse_test = NA)
    )
    rep_rows[[2]] <- make_row(rep, "IRS (informed)", ev)

    ev <- tryCatch(
      run_irs_uniform(data, miss, scenario, bart_settings),
      error = function(e) list(rmse_train = NA,
                               rmse_test = NA)
    )
    rep_rows[[3]] <- make_row(rep, "IRS (uniform)", ev)

    ev <- tryCatch(
      run_complete_case(data, miss, scenario, bart_settings),
      error = function(e) list(rmse_train = NA,
                               rmse_test = NA)
    )
    rep_rows[[4]] <- make_row(rep, "Complete case", ev)

    ev <- tryCatch(
      run_missforest(data, miss, scenario, bart_settings),
      error = function(e) list(rmse_train = NA,
                               rmse_test = NA)
    )
    rep_rows[[5]] <- make_row(rep, "missForest+BART", ev)

    do.call(rbind, rep_rows)
  }

  # --- Summarise this scenario ----------------------------------------------

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
    cat("  ** All replications failed — skipping.\n")
    cat("  ** results class:", paste(class(results)), "\n")
    str(results, max.level = 1)
    if (is.list(results)) {
      for (r in results) {
        if (is.character(r)) {
          cat("  ** First error: ", r[1], "\n")
          break
        }
      }
    }
    cat("\n")
    all_scenario_results[[scenario_id]] <- NULL
    next
  }
  results <- valid

  summary_df <- aggregate(
    cbind(rmse_train, rmse_test, bias_test, mae_test,
          coverage, width) ~ method,
    data = results,
    FUN = function(x) mean(x, na.rm = TRUE)
  )

  se_df <- aggregate(
    cbind(rmse_train, rmse_test, bias_test, mae_test,
          coverage, width) ~ method,
    data = results,
    FUN = function(x) {
      x <- x[!is.na(x)]
      if (length(x) > 1) sd(x) / sqrt(length(x)) else NA
    }
  )
  names(se_df)[-1] <- paste0(names(se_df)[-1], "_se")

  summary_out <- merge(summary_df, se_df, by = "method")

  cat("\n")
  print(summary_out, digits = 3, row.names = FALSE)
  cat("\n")

  results$scenario_id <- scenario_id
  results$model       <- scenario$model
  results$pattern     <- scenario$pattern
  results$p_miss      <- scenario$p_miss
  results$n_train     <- scenario$n_train
  all_scenario_results[[scenario_id]] <- results
}

# ============================================================================
# Save combined results
# ============================================================================

combined <- do.call(rbind, all_scenario_results)
rownames(combined) <- NULL

combined_file <- file.path(output_dir,
                           "irs_benchmark_output.rds")
saveRDS(combined, file = combined_file)

cat(sprintf("All %d scenarios complete.\n", n_scenarios))
cat(sprintf("Combined results: %s (%d rows)\n",
            combined_file, nrow(combined)))
