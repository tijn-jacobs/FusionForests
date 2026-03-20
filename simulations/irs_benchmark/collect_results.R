# ============================================================================
# Collect all scenario results into a single combined file
#
# Usage:
#   Rscript simulations/irs_benchmark/collect_results.R <results_dir>
#
# Reads all irs_benchmark_*_output.rds files from the given directory
# (defaults to current directory) and merges them into one data frame.
# ============================================================================

args <- commandArgs(trailingOnly = TRUE)
results_dir <- if (length(args) >= 1) args[1] else "."

files <- list.files(results_dir,
                    pattern = "^irs_benchmark_.*_output\\.rds$",
                    full.names = TRUE)

cat(sprintf("Found %d result files in %s\n", length(files), results_dir))

all_results  <- list()
all_summaries <- list()

for (f in files) {
  obj <- readRDS(f)
  sc  <- obj$info$scenario

  # Tag results with scenario info
  res <- obj$results
  res$model     <- sc$model
  res$pattern   <- sc$pattern
  res$p_miss    <- sc$p_miss
  res$n_train   <- sc$n_train
  res$d         <- sc$d
  res$scenario_id <- obj$info$scenario_id

  all_results[[f]] <- res

  # Tag summary
  summ <- obj$summary
  summ$model       <- sc$model
  summ$pattern     <- sc$pattern
  summ$p_miss      <- sc$p_miss
  summ$n_train     <- sc$n_train
  summ$scenario_id <- obj$info$scenario_id

  all_summaries[[f]] <- summ
}

combined_results   <- do.call(rbind, all_results)
combined_summaries <- do.call(rbind, all_summaries)

rownames(combined_results)   <- NULL
rownames(combined_summaries) <- NULL

outfile <- file.path(results_dir, "irs_benchmark_combined.rds")
saveRDS(list(results = combined_results, summaries = combined_summaries),
        file = outfile)

cat(sprintf("Combined results saved to %s\n", outfile))
cat(sprintf("  Total rows: %d\n", nrow(combined_results)))
cat(sprintf("  Scenarios:  %d\n", length(unique(combined_results$scenario_id))))

# Print overview
cat("\n=== Scenarios completed ===\n")
print(table(combined_results$model, combined_results$pattern))
