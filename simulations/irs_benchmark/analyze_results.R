# ============================================================================
# IRS Benchmark — Results analysis
#
# Usage:
#   source("simulations/irs_benchmark/analyze_results.R")
#
# Expects:
#   simulations/results/irs_benchmark_combined.rds
#   (or individual irs_benchmark_*_output.rds files)
# ============================================================================

library(ggplot2)

results_dir <- "simulations/results"

# ============================================================================
# Load data
# ============================================================================

combined_file <- file.path(results_dir,
                           "irs_benchmark_combined.rds")

if (file.exists(combined_file)) {
  cat("Loading combined results file...\n")
  combined <- readRDS(combined_file)
} else {
  cat("No combined file found. Reading individual .rds files...\n")
  files <- list.files(results_dir,
                      pattern = "^irs_benchmark_.*_output\\.rds$",
                      full.names = TRUE)
  if (length(files) == 0) {
    stop("No result files found in ", results_dir)
  }
  cat(sprintf("Found %d result files.\n", length(files)))

  all_results <- list()
  for (f in files) {
    obj <- readRDS(f)
    sc  <- obj$info$scenario
    res <- obj$results
    res$scenario_id <- obj$info$scenario_id
    res$model       <- sc$model
    res$pattern     <- sc$pattern
    res$p_miss      <- sc$p_miss
    res$n_train     <- sc$n_train
    all_results[[f]] <- res
  }
  combined <- do.call(rbind, all_results)
  rownames(combined) <- NULL
}

cat(sprintf("Loaded %d rows across %d scenarios.\n",
            nrow(combined),
            length(unique(combined$scenario_id))))

# ============================================================================
# Aggregate: mean and SE per (method, model, pattern, p_miss, n_train)
# ============================================================================

group_vars <- c("method", "model", "pattern", "p_miss", "n_train")

agg_mean <- aggregate(
  cbind(rmse_test, bias_test, mae_test, coverage, width) ~
    method + model + pattern + p_miss + n_train,
  data = combined,
  FUN = function(x) mean(x, na.rm = TRUE)
)

agg_se <- aggregate(
  cbind(rmse_test, bias_test, mae_test, coverage, width) ~
    method + model + pattern + p_miss + n_train,
  data = combined,
  FUN = function(x) {
    x <- x[!is.na(x)]
    if (length(x) > 1) sd(x) / sqrt(length(x)) else NA
  }
)
names(agg_se)[-(1:5)] <- paste0(names(agg_se)[-(1:5)], "_se")

agg <- merge(agg_mean, agg_se, by = group_vars)

# ============================================================================
# Table 1: Overall RMSE by method (averaged across all scenarios)
# ============================================================================

cat("\n")
cat("============================================================\n")
cat("  Table 1: Overall test RMSE by method\n")
cat("============================================================\n\n")

overall <- aggregate(
  rmse_test ~ method,
  data = combined,
  FUN = function(x) c(mean = mean(x, na.rm = TRUE),
                      se = sd(x, na.rm = TRUE) / sqrt(sum(!is.na(x))))
)
overall <- do.call(data.frame, overall)
names(overall) <- c("method", "rmse_test", "se")
overall <- overall[order(overall$rmse_test), ]
overall$rank <- seq_len(nrow(overall))
print(overall, digits = 4, row.names = FALSE)

# ============================================================================
# Table 2: Test RMSE by method and missingness pattern
# ============================================================================

cat("\n")
cat("============================================================\n")
cat("  Table 2: Test RMSE by method x pattern\n")
cat("============================================================\n\n")

by_pattern <- aggregate(
  rmse_test ~ method + pattern,
  data = combined,
  FUN = function(x) mean(x, na.rm = TRUE)
)
tab_pattern <- reshape(by_pattern,
                       idvar = "method",
                       timevar = "pattern",
                       direction = "wide")
names(tab_pattern) <- gsub("rmse_test\\.", "", names(tab_pattern))
print(tab_pattern, digits = 4, row.names = FALSE)

# ============================================================================
# Table 3: Test RMSE by method and model
# ============================================================================

cat("\n")
cat("============================================================\n")
cat("  Table 3: Test RMSE by method x model\n")
cat("============================================================\n\n")

by_model <- aggregate(
  rmse_test ~ method + model,
  data = combined,
  FUN = function(x) mean(x, na.rm = TRUE)
)
tab_model <- reshape(by_model,
                     idvar = "method",
                     timevar = "model",
                     direction = "wide")
names(tab_model) <- gsub("rmse_test\\.", "", names(tab_model))
print(tab_model, digits = 4, row.names = FALSE)

# ============================================================================
# Table 4: Coverage by method (where available)
# ============================================================================

cat("\n")
cat("============================================================\n")
cat("  Table 4: 95% credible interval coverage (nominal = 0.95)\n")
cat("============================================================\n\n")

cov_data <- combined[!is.na(combined$coverage), ]
if (nrow(cov_data) > 0) {
  cov_tbl <- aggregate(
    cbind(coverage, width) ~ method,
    data = cov_data,
    FUN = function(x) mean(x, na.rm = TRUE)
  )
  cov_tbl <- cov_tbl[order(-cov_tbl$coverage), ]
  print(cov_tbl, digits = 4, row.names = FALSE)
} else {
  cat("No coverage data available.\n")
}

# ============================================================================
# Plot 1: RMSE vs p_miss by method, faceted by pattern
# ============================================================================

cat("\n\nGenerating plots...\n")

theme_bench <- theme_bw(base_size = 11) +
  theme(
    legend.position  = "bottom",
    strip.background = element_rect(fill = "grey90"),
    panel.grid.minor = element_blank()
  )

method_colors <- c(
  "Oracle"          = "#333333",
  "IRS (informed)"  = "#E41A1C",
  "IRS (uniform)"   = "#FF7F00",
  "Complete case"   = "#377EB8",
  "bartMachine MIA" = "#4DAF4A",
  "missForest+BART" = "#984EA3"
)

# Aggregate across models and n_train for this plot
rmse_by_pmiss <- aggregate(
  rmse_test ~ method + pattern + p_miss,
  data = combined,
  FUN = function(x) mean(x, na.rm = TRUE)
)
rmse_by_pmiss_se <- aggregate(
  rmse_test ~ method + pattern + p_miss,
  data = combined,
  FUN = function(x) {
    x <- x[!is.na(x)]
    if (length(x) > 1) sd(x) / sqrt(length(x)) else NA
  }
)
names(rmse_by_pmiss_se)[4] <- "se"
rmse_by_pmiss <- merge(rmse_by_pmiss, rmse_by_pmiss_se,
                       by = c("method", "pattern", "p_miss"))

p1 <- ggplot(rmse_by_pmiss,
             aes(x = p_miss, y = rmse_test,
                 color = method, shape = method)) +
  geom_line(linewidth = 0.6) +
  geom_point(size = 2) +
  geom_errorbar(aes(ymin = rmse_test - 2 * se,
                    ymax = rmse_test + 2 * se),
                width = 0.02, linewidth = 0.4) +
  facet_wrap(~ pattern, scales = "free_y") +
  scale_color_manual(values = method_colors) +
  labs(x = "Missingness fraction",
       y = "Test RMSE (mean +/- 2 SE)",
       title = "Test RMSE vs missingness fraction",
       color = "Method", shape = "Method") +
  theme_bench

# ============================================================================
# Plot 2: RMSE vs p_miss, faceted by model x pattern
# ============================================================================

rmse_full <- aggregate(
  rmse_test ~ method + model + pattern + p_miss,
  data = combined,
  FUN = function(x) mean(x, na.rm = TRUE)
)

p2 <- ggplot(rmse_full,
             aes(x = p_miss, y = rmse_test,
                 color = method, shape = method)) +
  geom_line(linewidth = 0.5) +
  geom_point(size = 1.5) +
  facet_grid(model ~ pattern, scales = "free_y") +
  scale_color_manual(values = method_colors) +
  labs(x = "Missingness fraction",
       y = "Test RMSE",
       title = "Test RMSE by model and pattern",
       color = "Method", shape = "Method") +
  theme_bench

# ============================================================================
# Plot 3: RMSE vs n_train (sample size effect)
# ============================================================================

rmse_by_n <- aggregate(
  rmse_test ~ method + pattern + n_train,
  data = combined,
  FUN = function(x) mean(x, na.rm = TRUE)
)

p3 <- ggplot(rmse_by_n,
             aes(x = n_train, y = rmse_test,
                 color = method, shape = method)) +
  geom_line(linewidth = 0.6) +
  geom_point(size = 2) +
  facet_wrap(~ pattern, scales = "free_y") +
  scale_color_manual(values = method_colors) +
  scale_x_continuous(breaks = c(100, 200, 500, 1000)) +
  labs(x = "Training sample size",
       y = "Test RMSE",
       title = "Test RMSE vs sample size",
       color = "Method", shape = "Method") +
  theme_bench

# ============================================================================
# Plot 4: Coverage vs p_miss
# ============================================================================

cov_by_pmiss <- aggregate(
  coverage ~ method + pattern + p_miss,
  data = combined[!is.na(combined$coverage), ],
  FUN = function(x) mean(x, na.rm = TRUE)
)

p4 <- ggplot(cov_by_pmiss,
             aes(x = p_miss, y = coverage,
                 color = method, shape = method)) +
  geom_line(linewidth = 0.6) +
  geom_point(size = 2) +
  geom_hline(yintercept = 0.95, linetype = "dashed",
             color = "grey50") +
  facet_wrap(~ pattern) +
  scale_color_manual(values = method_colors) +
  coord_cartesian(ylim = c(0.5, 1.0)) +
  labs(x = "Missingness fraction",
       y = "95% CI coverage",
       title = "Credible interval coverage",
       color = "Method", shape = "Method") +
  theme_bench

# ============================================================================
# Plot 5: Relative RMSE to Oracle
# ============================================================================

oracle_rmse <- aggregate(
  rmse_test ~ model + pattern + p_miss + n_train,
  data = combined[combined$method == "Oracle", ],
  FUN = function(x) mean(x, na.rm = TRUE)
)
names(oracle_rmse)[5] <- "oracle_rmse"

rel_data <- merge(
  aggregate(
    rmse_test ~ method + model + pattern + p_miss + n_train,
    data = combined,
    FUN = function(x) mean(x, na.rm = TRUE)
  ),
  oracle_rmse,
  by = c("model", "pattern", "p_miss", "n_train")
)
rel_data$relative_rmse <- rel_data$rmse_test / rel_data$oracle_rmse
rel_data <- rel_data[rel_data$method != "Oracle", ]

rel_agg <- aggregate(
  relative_rmse ~ method + pattern + p_miss,
  data = rel_data,
  FUN = function(x) mean(x, na.rm = TRUE)
)

p5 <- ggplot(rel_agg,
             aes(x = p_miss, y = relative_rmse,
                 color = method, shape = method)) +
  geom_line(linewidth = 0.6) +
  geom_point(size = 2) +
  geom_hline(yintercept = 1, linetype = "dashed",
             color = "grey50") +
  facet_wrap(~ pattern) +
  scale_color_manual(values = method_colors) +
  labs(x = "Missingness fraction",
       y = "RMSE / Oracle RMSE",
       title = "Relative efficiency vs Oracle",
       color = "Method", shape = "Method") +
  theme_bench

# ============================================================================
# Save plots
# ============================================================================

plot_dir <- file.path(results_dir, "figures")
dir.create(plot_dir, showWarnings = FALSE, recursive = TRUE)

ggsave(file.path(plot_dir, "rmse_vs_pmiss.pdf"),
       p1, width = 10, height = 4)
ggsave(file.path(plot_dir, "rmse_full_grid.pdf"),
       p2, width = 10, height = 10)
ggsave(file.path(plot_dir, "rmse_vs_n.pdf"),
       p3, width = 10, height = 4)
ggsave(file.path(plot_dir, "coverage_vs_pmiss.pdf"),
       p4, width = 10, height = 4)
ggsave(file.path(plot_dir, "relative_rmse.pdf"),
       p5, width = 10, height = 4)

cat(sprintf("\nPlots saved to %s/\n", plot_dir))

# ============================================================================
# Store aggregated table for easy access
# ============================================================================

agg_file <- file.path(results_dir, "irs_benchmark_summary.csv")
write.csv(agg, file = agg_file, row.names = FALSE)
cat(sprintf("Summary table saved to %s\n", agg_file))
