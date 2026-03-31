# ============================================================================
# IRS Benchmark v2 — HPC simulation runner
#
# Usage:
#   Rscript simulations/irs_benchmark_v2/run_simulation.R \
#     <num_cores> <scenario_id>
#
# Arguments:
#   1. num_cores    — number of cores available (1 kept free)
#   2. scenario_id  — integer index into scenario grid (see scenarios.R)
#
# Output:
#   $TMPDIR/irs_v2_<scenario_id>_output.rds
# ============================================================================

library(FusionForests)
library(doParallel)
library(foreach)

setwd("~/Library/CloudStorage/OneDrive-VrijeUniversiteitAmsterdam/Documents/GitHub/FusionForests/simulations/irs_benchmark_v2")
source("dgp.R")
source("methods.R")
source("scenarios.R")

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
# Load scenario and settings
# ============================================================================

scenario <- get_scenario(scenario_id)

# Simulation settings
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

# Check optional packages
options(java.parameters = c("-Xmx4g",
                            "--add-modules=jdk.incubator.vector"))
has_bartmachine <- requireNamespace("bartMachine", quietly = TRUE)
has_missforest  <- requireNamespace("missForest",  quietly = TRUE)
has_mice        <- requireNamespace("mice",        quietly = TRUE)

# ============================================================================
# Print info
# ============================================================================

cat("Number of cores being used (1 free):", num_cores, "\n")
cat(sprintf(
  "SIMULATION: IRS Benchmark v2 — Scenario %d / %d\n",
  scenario_id, get_n_scenarios()
))
cat(sprintf("  Outcome:   %d (%s)\n",
            scenario$outcome_scenario,
            scenario$outcome_label))
cat(sprintf("  Missing:   %s\n", scenario$miss_pattern))
cat(sprintf("  rho:       %g\n", scenario$rho))
cat(sprintf("  n_reps:    %d\n", n_reps))
cat(sprintf("  bartMachine: %s\n", has_bartmachine))
cat(sprintf("  missForest:  %s\n", has_missforest))
cat(sprintf("  mice:        %s\n", has_mice))
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
    n_rct  = n_rct,
    n_rwd  = n_rwd,
    n_test = n_test,
    outcome_scenario = scenario$outcome_scenario,
    rho    = scenario$rho,
    sigma  = NULL,  # auto-calibrate
    alpha0 = alpha0, alpha1 = alpha1, alpha5 = alpha5,
    beta5  = beta5, gamma0 = gamma0, gamma1 = gamma1,
    seed   = seed
  )

  # Impose missingness
  miss <- impose_missingness(
    XA           = data$XA_train,
    S            = data$S_train,
    miss_pattern = scenario$miss_pattern,
    pi_mcar      = pi_mcar
  )

  # Helper to build result row
  make_row <- function(method_name, ev, elapsed) {
    row <- data.frame(
      rep    = rep,
      method = method_name,
      time   = elapsed,
      stringsAsFactors = FALSE
    )
    for (nm in names(ev)) {
      row[[nm]] <- ifelse(is.null(ev[[nm]]), NA, ev[[nm]])
    }
    row
  }

  rep_rows <- list()
  idx <- 0L

  # 1. Oracle
  t0 <- proc.time()
  ev <- tryCatch(
    run_oracle(data, bart_settings),
    error = function(e) null_result()
  )
  elapsed <- (proc.time() - t0)[3]
  idx <- idx + 1L
  rep_rows[[idx]] <- make_row("Oracle", ev, elapsed)

  # 2. IRS (informed, mode 2)
  t0 <- proc.time()
  ev <- tryCatch(
    run_irs(data, miss, bart_settings, irs_mode = 2L),
    error = function(e) null_result()
  )
  elapsed <- (proc.time() - t0)[3]
  idx <- idx + 1L
  rep_rows[[idx]] <- make_row("IRS", ev, elapsed)

  # 3. Complete case
  t0 <- proc.time()
  ev <- tryCatch(
    run_complete_case(data, miss, bart_settings),
    error = function(e) null_result()
  )
  elapsed <- (proc.time() - t0)[3]
  idx <- idx + 1L
  rep_rows[[idx]] <- make_row("Complete case", ev, elapsed)

  # 4. Complete covariates
  t0 <- proc.time()
  ev <- tryCatch(
    run_complete_covariates(data, bart_settings),
    error = function(e) null_result()
  )
  elapsed <- (proc.time() - t0)[3]
  idx <- idx + 1L
  rep_rows[[idx]] <- make_row("Complete covariates", ev,
                              elapsed)

  # 5. bartMachine MIA
  if (has_bartmachine) {
    t0 <- proc.time()
    ev <- tryCatch(
      run_bartmachine(data, miss, bart_settings),
      error = function(e) null_result()
    )
    elapsed <- (proc.time() - t0)[3]
    idx <- idx + 1L
    rep_rows[[idx]] <- make_row("BART+MIA", ev, elapsed)
  }

  # 6. missForest + BART
  if (has_missforest) {
    t0 <- proc.time()
    ev <- tryCatch(
      run_missforest(data, miss, bart_settings),
      error = function(e) null_result()
    )
    elapsed <- (proc.time() - t0)[3]
    idx <- idx + 1L
    rep_rows[[idx]] <- make_row("MissForest+BART", ev,
                                elapsed)
  }

  # 7. MICE + BART
  if (has_mice) {
    t0 <- proc.time()
    ev <- tryCatch(
      run_mice(data, miss, bart_settings),
      error = function(e) null_result()
    )
    elapsed <- (proc.time() - t0)[3]
    idx <- idx + 1L
    rep_rows[[idx]] <- make_row("MICE+BART", ev, elapsed)
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
  n_rct       = n_rct,
  n_rwd       = n_rwd,
  n_test      = n_test,
  seed0       = seed0,
  bart        = bart_settings,
  params      = list(beta5 = beta5, gamma0 = gamma0,
                     gamma1 = gamma1, alpha0 = alpha0,
                     alpha1 = alpha1, alpha5 = alpha5,
                     pi_mcar = pi_mcar),
  R_version   = R.version.string,
  timestamp   = Sys.time()
)

# Summarise key metrics
metric_cols <- c("rmse_m_test", "cate_bias_test",
                 "cate_rmse_test", "ate_bias",
                 "cate_coverage", "cate_ci_width", "time")

summary_df <- aggregate(
  cbind(rmse_m_test, cate_bias_test, cate_rmse_test,
        ate_bias, cate_coverage, cate_ci_width,
        time) ~ method,
  data = results,
  FUN = function(x) mean(x, na.rm = TRUE)
)

se_df <- aggregate(
  cbind(rmse_m_test, cate_bias_test, cate_rmse_test,
        ate_bias, cate_coverage, cate_ci_width,
        time) ~ method,
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

# Save
combined_results <- list(
  info    = info_object,
  results = results,
  summary = summary_out
)

output_file <- file.path(
  Sys.getenv("TMPDIR"),
  sprintf("irs_v2_%03d_output.rds", scenario_id)
)

cat("\nSaving results to:", output_file, "\n")
saveRDS(combined_results, file = output_file)
cat("Done.\n")
