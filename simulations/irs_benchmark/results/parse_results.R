# Parse slurm output files and create preliminary results tables

parse_slurm_output <- function(filepath) {
  lines <- readLines(filepath)

  results <- list()
  i <- 1

  while (i <= length(lines)) {
    # Find scenario headers
    m <- regmatches(lines[i], regexpr(
      "=== Scenario (\\d+) / \\d+: (\\w+) \\| (\\w+) \\| p=([0-9.]+) \\| n=(\\d+) ===",
      lines[i], perl = TRUE
    ))

    if (length(m) > 0 && nchar(m) > 0) {
      # Extract scenario info
      parts <- regmatches(lines[i], gregexpr(
        "Scenario (\\d+) / \\d+: (\\w+) \\| (\\w+) \\| p=([0-9.]+) \\| n=(\\d+)",
        lines[i], perl = TRUE
      ))[[1]]

      caps <- regmatches(parts, regexec(
        "Scenario (\\d+) / \\d+: (\\w+) \\| (\\w+) \\| p=([0-9.]+) \\| n=(\\d+)",
        parts
      ))[[1]]

      scenario_id <- as.integer(caps[2])
      model <- caps[3]
      pattern <- caps[4]
      p_miss <- as.numeric(caps[5])
      n_train <- as.integer(caps[6])

      # Skip to the data table (next non-empty line after header)
      i <- i + 1
      while (i <= length(lines) && trimws(lines[i]) == "") i <- i + 1

      # Check if it's a failure message
      if (i <= length(lines) && grepl("failed", lines[i])) {
        i <- i + 1
        next
      }

      # Read the header line and data lines
      if (i <= length(lines)) {
        header_line <- trimws(lines[i])
        i <- i + 1

        # Read method rows until empty line or next scenario
        while (i <= length(lines) && trimws(lines[i]) != "" &&
               !grepl("^===", lines[i]) && !grepl("rmse_train_se", lines[i])) {
          vals <- strsplit(trimws(lines[i]), "\\s+")[[1]]

          # Method name can be multi-word, metrics are last 6 values
          n_vals <- length(vals)
          metrics <- as.numeric(vals[(n_vals-5):n_vals])
          method <- paste(vals[1:(n_vals-6)], collapse = " ")

          results[[length(results) + 1]] <- data.frame(
            scenario_id = scenario_id,
            model = model,
            pattern = pattern,
            p_miss = p_miss,
            n_train = n_train,
            method = method,
            rmse_train = metrics[1],
            rmse_test = metrics[2],
            bias_test = metrics[3],
            mae_test = metrics[4],
            coverage = metrics[5],
            width = metrics[6],
            stringsAsFactors = FALSE
          )
          i <- i + 1
        }
      }
    } else {
      i <- i + 1
    }
  }

  do.call(rbind, results)
}

# Parse both files
main_res <- parse_slurm_output("slurm-20949129.out")
bartm_res <- parse_slurm_output("slurm-20960424.out")

cat("Main simulation:\n")
cat(sprintf("  Scenarios completed: %d\n",
            length(unique(main_res$scenario_id))))
cat(sprintf("  Methods: %s\n",
            paste(unique(main_res$method), collapse = ", ")))
cat(sprintf("  n_train levels: %s\n",
            paste(sort(unique(main_res$n_train)), collapse = ", ")))

cat("\nbartMachine:\n")
cat(sprintf("  Scenarios completed: %d\n",
            length(unique(bartm_res$scenario_id))))

# Merge on overlapping scenarios
combined <- rbind(main_res, bartm_res)

cat("\n\nCombined results preview (MCAR, n=100):\n\n")
sub <- combined[combined$pattern == "mcar" &
                combined$n_train == 100 &
                combined$p_miss == 0.25, ]
print(sub[, c("model", "method", "rmse_test", "coverage")],
      row.names = FALSE)

cat("\n\nRMSE test by method and pattern (n=100, p=0.25):\n\n")
sub2 <- combined[combined$n_train == 100 &
                 combined$p_miss == 0.25, ]
tab <- reshape(sub2[, c("model", "pattern", "method", "rmse_test")],
               idvar = c("model", "method"),
               timevar = "pattern",
               direction = "wide")
names(tab) <- gsub("rmse_test\\.", "", names(tab))
print(tab[order(tab$model, tab$method), ], row.names = FALSE)

# Save parsed results
saveRDS(combined, "preliminary_results.rds")
cat("\n\nSaved to preliminary_results.rds\n")
