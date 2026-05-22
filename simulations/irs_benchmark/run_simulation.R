# ============================================================================
# IRS Benchmark — HPC simulation runner
#
# Usage:
#   Rscript simulations/irs_benchmark/run_simulation.R <num_cores> <scenario_id>
#
# Arguments:
#   1. num_cores    — number of cores available (1 is kept free)
#   2. scenario_id  — integer index into the scenario grid (see scenarios.R)
#
# Output:
#   $TMPDIR/irs_benchmark_<scenario_id>_output.rds
#
# Methods compared:
#   Oracle, IRS (informed), IRS (uniform), Complete case,
#   bartMachine MIA, missForest+BART
# ============================================================================

library(FusionForests)
library(doParallel)
library(foreach)

source("simulations/irs_benchmark/dgp.R")
source("simulations/irs_benchmark/evaluation.R")
source("simulations/irs_benchmark/methods.R")
source("simulations/irs_benchmark/scenarios.R")

# ============================================================================
# Parse command-line arguments
# ============================================================================

args <- commandArgs(trailingOnly = TRUE)

if (length(args) >= 1) {
  num_cores <- as.integer(args[1]) - 1
} else {
  num_cores <- parallel::detectCores() - 1
}

if (length(args) >= 2) {
  scenario_id <- as.integer(args[2])
} else {
  scenario_id <- 1L
}

registerDoParallel(cores = num_cores)

# ============================================================================
# Load scenario
# ============================================================================

scenario <- get_scenario(scenario_id)

# Simulation settings
n_reps <- 100L
n_test <- 1000L
rho    <- 0.5
sigma  <- 0.1
seed0  <- 2026L

# BART settings
bart_settings <- list(
  number_of_trees = 200L,
  N_post          = 2000L,
  N_burn          = 1000L
)

# Check optional packages
options(java.parameters = c("-Xmx4g", "--add-modules=jdk.incubator.vector"))
has_bartmachine <- requireNamespace("bartMachine", quietly = TRUE)
has_missforest  <- requireNamespace("missForest",  quietly = TRUE)

# ============================================================================
# Print info
# ============================================================================

cat("Number of cores being used (1 free):", num_cores, "\n")
cat(sprintf("SIMULATION: IRS Benchmark — Scenario %d / %d\n",
            scenario_id, get_n_scenarios()))
cat(sprintf("  Model:     %s\n", scenario$model))
cat(sprintf("  Pattern:   %s\n", scenario$pattern))
cat(sprintf("  p_miss:    %g\n", scenario$p_miss))
cat(sprintf("  n_train:   %d\n", scenario$n_train))
cat(sprintf("  d:         %d\n", scenario$d))
cat(sprintf("  n_reps:    %d\n", n_reps))
cat(sprintf("  bartMachine available: %s\n", has_bartmachine))
cat(sprintf("  missForest available:  %s\n", has_missforest))
cat("\n")

# ============================================================================
# Parallel replications
# ============================================================================

results <- foreach(
  rep = seq_len(n_reps),
  .combine  = rbind,
  .packages = c("FusionForests", "MASS"),
  .errorhandling = "pass"
) %dopar% {

  seed <- seed0 + (scenario_id - 1) * n_reps + rep

  # Generate data
  data <- generate_data(
    n_train = scenario$n_train,
    n_test  = n_test,
    model   = scenario$model,
    d       = scenario$d,
    rho     = rho,
    sigma   = sigma,
    seed    = seed
  )

  # Impose missingness
  miss <- impose_missingness(
    X         = data$X_train,
    y         = data$y_train,
    pattern   = scenario$pattern,
    miss_cols = scenario$miss_cols,
    p_miss    = scenario$p_miss,
    sigma     = sigma
  )

  # Run all methods and collect results
  rep_rows <- list()

  make_row <- function(method_name, ev) {
    data.frame(
      rep        = rep,
      method     = method_name,
      rmse_train = ev$rmse_train,
      rmse_test  = ev$rmse_test,
      bias_test  = ifelse(is.null(ev$bias_test), NA, ev$bias_test),
      mae_test   = ifelse(is.null(ev$mae_test),  NA, ev$mae_test),
      coverage   = ifelse(is.null(ev$coverage),  NA, ev$coverage),
      width      = ifelse(is.null(ev$width),     NA, ev$width),
      stringsAsFactors = FALSE
    )
  }

  # 1. Oracle
  ev <- tryCatch(run_oracle(data, miss, scenario, bart_settings),
                 error = function(e) list(rmse_train = NA, rmse_test = NA))
  rep_rows[[1]] <- make_row("Oracle", ev)

  # 2. IRS (informed)
  ev <- tryCatch(run_irs_informed(data, miss, scenario, bart_settings),
                 error = function(e) list(rmse_train = NA, rmse_test = NA))
  rep_rows[[2]] <- make_row("IRS (informed)", ev)

  # 3. IRS (uniform)
  ev <- tryCatch(run_irs_uniform(data, miss, scenario, bart_settings),
                 error = function(e) list(rmse_train = NA, rmse_test = NA))
  rep_rows[[3]] <- make_row("IRS (uniform)", ev)

  # 4. Complete case
  ev <- tryCatch(run_complete_case(data, miss, scenario, bart_settings),
                 error = function(e) list(rmse_train = NA, rmse_test = NA))
  rep_rows[[4]] <- make_row("Complete case", ev)

  # 5. bartMachine MIA
  if (has_bartmachine) {
    ev <- tryCatch(run_bartmachine(data, miss, scenario, bart_settings),
                   error = function(e) list(rmse_train = NA, rmse_test = NA))
    rep_rows[[5]] <- make_row("bartMachine MIA", ev)
  }

  # 6. missForest + BART
  if (has_missforest) {
    ev <- tryCatch(run_missforest(data, miss, scenario, bart_settings),
                   error = function(e) list(rmse_train = NA, rmse_test = NA))
    rep_rows[[6]] <- make_row("missForest+BART", ev)
  }

  do.call(rbind, rep_rows)
}

# ============================================================================
# Summarise and save
# ============================================================================

info_object <- list(
  scenario_id = scenario_id,
  scenario    = scenario,
  n_reps      = n_reps,
  n_test      = n_test,
  rho         = rho,
  sigma       = sigma,
  seed0       = seed0,
  bart        = bart_settings,
  R_version   = R.version.string,
  timestamp   = Sys.time()
)

# Summary table
summary_df <- aggregate(
  cbind(rmse_train, rmse_test, bias_test, mae_test, coverage, width) ~ method,
  data = results, FUN = function(x) mean(x, na.rm = TRUE)
)

se_df <- aggregate(
  cbind(rmse_train, rmse_test, bias_test, mae_test, coverage, width) ~ method,
  data = results,
  FUN = function(x) {
    x <- x[!is.na(x)]
    if (length(x) > 1) sd(x) / sqrt(length(x)) else NA
  }
)
names(se_df)[-1] <- paste0(names(se_df)[-1], "_se")

summary_out <- merge(summary_df, se_df, by = "method")

cat("\n=== Summary ===\n\n")
print(summary_out, digits = 3, row.names = FALSE)

# Combine all results
combined_results <- list(
  info    = info_object,
  results = results,
  summary = summary_out
)

# Define output file path
# (! NAME MUST BE FILENAME_output.rds !)
output_file <- file.path(
  Sys.getenv("TMPDIR"),
  sprintf("irs_benchmark_%03d_output.rds", scenario_id)
)

cat("\nSaving all settings results to:", output_file, "\n")

saveRDS(combined_results, file = output_file)

cat("All results successfully saved in one file.\n")
