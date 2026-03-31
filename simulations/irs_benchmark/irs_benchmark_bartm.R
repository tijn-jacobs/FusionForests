# ============================================================================
# IRS Benchmark — bartMachine MIA
#
# Runs the same scenarios and seeds as irs_benchmark.R but only fits
# bartMachine with MIA (missing-in-attributes). Runs sequentially
# because bartMachine uses Java which cannot fork.
#
# Usage:
#   Rscript irs_benchmark_bartm.R [num_cores_for_bartmachine]
#
# Output:
#   $TMPDIR/irs_benchmark_bartm_output.rds
# ============================================================================

options(java.parameters = c("-Xmx5g",
                            "--add-modules=jdk.incubator.vector"))

library(bartMachine)

#source("~/irs_benchmark_common.R")
 source("/Users/tijnjacobs/Library/CloudStorage/OneDrive-VrijeUniversiteitAmsterdam/Documents/GitHub/FusionForests/simulations/irs_benchmark/irs_benchmark_common.R")


# ============================================================================
# Settings
# ============================================================================

args <- commandArgs(trailingOnly = TRUE)

if (length(args) >= 1) {
  bm_cores <- as.integer(args[1])
} else {
  bm_cores <- parallel::detectCores() - 1
}

bartMachine::set_bart_machine_num_cores(1)

# n_reps must match irs_benchmark.R for seed alignment.
# In irs_benchmark.R: n_reps <- num_cores (= args[1] - 1).
# Pass the same value here, or set manually.
if (length(args) >= 2) {
  n_reps <- as.integer(args[2])
} else {
  n_reps <- bm_cores
}

output_dir <- Sys.getenv("TMPDIR", unset = ".")

n_scenarios <- get_n_scenarios()

cat(sprintf("bartMachine cores: %d\n", bm_cores))
cat(sprintf("Replications per scenario: %d\n", n_reps))
cat(sprintf("Total scenarios: %d\n", n_scenarios))
cat(sprintf("Output directory: %s\n\n", output_dir))

# ============================================================================
# bartMachine method
# ============================================================================

run_bartmachine <- function(data, miss, scenario, s) {
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

  bm <- bartMachine::bartMachine(
    as.data.frame(X_train_na), y_fit,
    num_trees                          = s$number_of_trees,
    num_burn_in                        = s$N_burn,
    num_iterations_after_burn_in       = s$N_post,
    use_missing_data                   = TRUE,
    use_missing_data_dummies_as_covars = TRUE,
    verbose                            = FALSE
  )

  pred_train <- bartMachine::bart_machine_get_posterior(
    bm, as.data.frame(X_train_na)
  )
  pred_test <- bartMachine::bart_machine_get_posterior(
    bm, as.data.frame(data$X_test)
  )

  ev <- list(
    rmse_train = compute_rmse(pred_train$y_hat, f_train),
    rmse_test  = compute_rmse(pred_test$y_hat, data$f_test),
    bias_test  = compute_bias(pred_test$y_hat, data$f_test),
    mae_test   = compute_mae(pred_test$y_hat, data$f_test),
    coverage   = compute_coverage(
      t(pred_test$y_hat_posterior_samples), data$f_test
    ),
    width = compute_interval_width(
      t(pred_test$y_hat_posterior_samples)
    )
  )
  rm(bm); gc()
  ev
}

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

  rep_rows <- list()

  for (rep in seq_len(n_reps)) {

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

    ev <- tryCatch(
      run_bartmachine(data, miss, scenario, bart_settings),
      error = function(e) {
        cat(sprintf("  rep %d error: %s\n", rep,
                    conditionMessage(e)))
        list(rmse_train = NA, rmse_test = NA)
      }
    )

    rep_rows[[rep]] <- make_row(rep, "bartMachine MIA", ev)
  }

  results <- do.call(rbind, rep_rows)

  summary_df <- aggregate(
    cbind(rmse_train, rmse_test, bias_test, mae_test,
          coverage, width) ~ method,
    data = results,
    FUN = function(x) mean(x, na.rm = TRUE)
  )

  cat("\n")
  print(summary_df, digits = 3, row.names = FALSE)
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
                           "irs_benchmark_bartm_output.rds")
saveRDS(combined, file = combined_file)

cat(sprintf("All %d scenarios complete.\n", n_scenarios))
cat(sprintf("Combined results: %s (%d rows)\n",
            combined_file, nrow(combined)))
