# ============================================================================
# Collect all scenario results into a single combined file
#
# Usage:
#   Rscript simulations/irs_benchmark_v2/collect_results.R <results_dir>
#
# Reads all irs_v2_*_output.rds files and merges them.
# ============================================================================

args <- commandArgs(trailingOnly = TRUE)
results_dir <- if (length(args) >= 1) args[1] else "."

files <- list.files(results_dir,
                    pattern = "^irs_v2_.*_output\\.rds$",
                    full.names = TRUE)

cat(sprintf("Found %d result files in %s\n",
            length(files), results_dir))

all_results   <- list()
all_summaries <- list()

for (f in files) {
  obj <- readRDS(f)
  sc  <- obj$info$scenario

  res <- obj$results
  res$outcome_scenario <- sc$outcome_scenario
  res$outcome_label    <- sc$outcome_label
  res$miss_pattern     <- sc$miss_pattern
  res$rho              <- sc$rho
  res$scenario_id      <- obj$info$scenario_id

  all_results[[f]] <- res

  summ <- obj$summary
  summ$outcome_scenario <- sc$outcome_scenario
  summ$outcome_label    <- sc$outcome_label
  summ$miss_pattern     <- sc$miss_pattern
  summ$rho              <- sc$rho
  summ$scenario_id      <- obj$info$scenario_id

  all_summaries[[f]] <- summ
}

combined_results   <- do.call(rbind, all_results)
combined_summaries <- do.call(rbind, all_summaries)

rownames(combined_results)   <- NULL
rownames(combined_summaries) <- NULL

outfile <- file.path(results_dir, "irs_v2_combined.rds")
saveRDS(list(results = combined_results,
             summaries = combined_summaries),
        file = outfile)

cat(sprintf("Combined results saved to %s\n", outfile))
cat(sprintf("  Total rows: %d\n", nrow(combined_results)))
cat(sprintf("  Scenarios:  %d\n",
            length(unique(combined_results$scenario_id))))

cat("\n=== Scenarios completed ===\n")
print(table(combined_results$outcome_scenario,
            combined_results$miss_pattern))
