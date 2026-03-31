# ============================================================================
# IRS Benchmark v2 — Local runner (all scenarios)
#
# Runs all 24 scenarios sequentially with parallel replications.
# For HPC use run_simulation.R with SLURM array jobs instead.
#
# Usage:
#   Rscript simulations/irs_benchmark_v2/irs_benchmark_v2.R [num_cores]
# ============================================================================

library(FusionForests)
library(doParallel)
library(foreach)

setwd("~/Library/CloudStorage/OneDrive-VrijeUniversiteitAmsterdam/Documents/GitHub/FusionForests/simulations/irs_benchmark_v2")
source("dgp.R")
source("methods.R")
source("scenarios.R")

# ============================================================================
# Settings
# ============================================================================

args <- commandArgs(trailingOnly = TRUE)

if (length(args) >= 1) {
  num_cores <- as.integer(args[1]) - 2
} else {
  num_cores <- parallel::detectCores() - 2
}

registerDoParallel(cores = num_cores)

n_reps  <- num_cores
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

bart_settings <- list(
  number_of_trees = 200L,
  N_post          = 1000L,
  N_burn          = 1000L
)

output_dir <- Sys.getenv("TMPDIR", unset = ".")

has_bartmachine <- requireNamespace("bartMachine", quietly = TRUE)
has_missforest  <- requireNamespace("missForest",  quietly = TRUE)
has_mice        <- requireNamespace("mice",        quietly = TRUE)

n_scenarios <- get_n_scenarios()

cat("Number of cores:", num_cores, "\n")
cat(sprintf("Replications per scenario: %d\n", n_reps))
cat(sprintf("Total scenarios: %d\n", n_scenarios))
cat(sprintf("Output directory: %s\n\n", output_dir))

# ============================================================================
# Loop over all scenarios
# ============================================================================

all_scenario_results <- list()
tic("all sims")
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
    .packages = c("FusionForests", "MASS"),
    .errorhandling = "pass"
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
      XA           = data$XA_train,
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

    # Oracle
    t0 <- proc.time()
    ev <- tryCatch(run_oracle(data, bart_settings),
                   error = function(e) null_result())
    idx <- idx + 1L
    rep_rows[[idx]] <- make_row("Oracle", ev,
                                (proc.time() - t0)[3])

    # IRS
    t0 <- proc.time()
    ev <- tryCatch(
      run_irs(data, miss, bart_settings, irs_mode = 2L),
      error = function(e) null_result()
    )
    idx <- idx + 1L
    rep_rows[[idx]] <- make_row("IRS", ev,
                                (proc.time() - t0)[3])

    # Complete case
    t0 <- proc.time()
    ev <- tryCatch(
      run_complete_case(data, miss, bart_settings),
      error = function(e) null_result()
    )
    idx <- idx + 1L
    rep_rows[[idx]] <- make_row("Complete case", ev,
                                (proc.time() - t0)[3])

    # Complete covariates
    t0 <- proc.time()
    ev <- tryCatch(
      run_complete_covariates(data, bart_settings),
      error = function(e) null_result()
    )
    idx <- idx + 1L
    rep_rows[[idx]] <- make_row("Complete covariates", ev,
                                (proc.time() - t0)[3])

    # bartMachine MIA
    if (has_bartmachine) {
      t0 <- proc.time()
      ev <- tryCatch(
        run_bartmachine(data, miss, bart_settings),
        error = function(e) null_result()
      )
      idx <- idx + 1L
      rep_rows[[idx]] <- make_row("BART+MIA", ev,
                                  (proc.time() - t0)[3])
    }

    # missForest + BART
    if (has_missforest) {
      t0 <- proc.time()
      ev <- tryCatch(
        run_missforest(data, miss, bart_settings),
        error = function(e) null_result()
      )
      idx <- idx + 1L
      rep_rows[[idx]] <- make_row("MissForest+BART", ev,
                                  (proc.time() - t0)[3])
    }

    # MICE + BART
    if (has_mice) {
      t0 <- proc.time()
      ev <- tryCatch(
        run_mice(data, miss, bart_settings),
        error = function(e) null_result()
      )
      idx <- idx + 1L
      rep_rows[[idx]] <- make_row("MICE+BART", ev,
                                  (proc.time() - t0)[3])
    }

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
    cbind(cate_rmse_test, cate_bias_test,
          ate_bias) ~ method,
    data = results,
    FUN = function(x) mean(x, na.rm = TRUE)
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
toc()
# ============================================================================
# Save combined results
# ============================================================================

combined <- do.call(rbind, all_scenario_results)
rownames(combined) <- NULL

combined_file <- file.path(output_dir,
                           "irs_v2_all_output.rds")
saveRDS(combined, file = combined_file)

cat(sprintf("All %d scenarios complete.\n", n_scenarios))
cat(sprintf("Combined results: %s (%d rows)\n",
            combined_file, nrow(combined)))
