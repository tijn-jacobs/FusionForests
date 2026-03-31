# ============================================================================
# IRS Benchmark v2 — Results analysis
#
# Usage:
#   source("simulations/irs_benchmark_v2/analyze_results.R")
#
# Expects: irs_v2_combined.rds or irs_v2_all_output.rds
# ============================================================================

library(ggplot2)

results_dir <- "simulations/irs_benchmark_v2/results"

# ============================================================================
# Load data
# ============================================================================

combined_file <- file.path(results_dir, "irs_v2_combined.rds")
all_file      <- file.path(results_dir, "irs_v2_all_output.rds")

if (file.exists(combined_file)) {
  cat("Loading combined results file...\n")
  obj <- readRDS(combined_file)
  combined <- obj$results
} else if (file.exists(all_file)) {
  cat("Loading all-output results file...\n")
  combined <- readRDS(all_file)
} else {
  stop("No result files found in ", results_dir)
}

cat(sprintf("Loaded %d rows across %d scenarios.\n",
            nrow(combined),
            length(unique(combined$scenario_id))))

# Outcome labels
outcome_labels <- c(
  "0" = "Irrelevant",
  "1" = "Prognostic",
  "2" = "Effect modifier",
  "3" = "Prognostic + EM"
)

miss_labels <- c(
  "block_rct" = "Block: missing in RCT",
  "block_rwd" = "Block: missing in RWD",
  "mcar"      = "MCAR (30%)"
)

# ============================================================================
# Theme and colours
# ============================================================================

theme_bench <- theme_bw(base_size = 11) +
  theme(
    legend.position  = "bottom",
    strip.background = element_rect(fill = "grey90"),
    panel.grid.minor = element_blank()
  )

method_colors <- c(
  "Oracle"              = "#333333",
  "IRS"                 = "#E41A1C",
  "BART+MIA"            = "#4DAF4A",
  "MissForest+BART"     = "#984EA3",
  "MICE+BART"           = "#A65628",
  "Complete case"       = "#377EB8",
  "Complete covariates" = "#FF7F00"
)

# ============================================================================
# Table 1: CATE RMSE by method x outcome scenario
# ============================================================================

cat("\n")
cat("============================================================\n")
cat("  Table 1: CATE RMSE (test) by method x outcome scenario\n")
cat("============================================================\n\n")

tab1 <- aggregate(
  cate_rmse_test ~ method + outcome_scenario,
  data = combined,
  FUN = function(x) mean(x, na.rm = TRUE)
)
tab1_wide <- reshape(tab1,
  idvar   = "method",
  timevar = "outcome_scenario",
  direction = "wide"
)
names(tab1_wide) <- gsub("cate_rmse_test\\.", "Sc",
                          names(tab1_wide))
print(tab1_wide, digits = 4, row.names = FALSE)

# ============================================================================
# Table 2: CATE bias by method x outcome scenario
# ============================================================================

cat("\n")
cat("============================================================\n")
cat("  Table 2: CATE bias (test) by method x outcome scenario\n")
cat("============================================================\n\n")

tab2 <- aggregate(
  cate_bias_test ~ method + outcome_scenario,
  data = combined,
  FUN = function(x) mean(x, na.rm = TRUE)
)
tab2_wide <- reshape(tab2,
  idvar   = "method",
  timevar = "outcome_scenario",
  direction = "wide"
)
names(tab2_wide) <- gsub("cate_bias_test\\.", "Sc",
                          names(tab2_wide))
print(tab2_wide, digits = 4, row.names = FALSE)

# ============================================================================
# Table 3: ATE bias by method x missingness pattern
# ============================================================================

cat("\n")
cat("============================================================\n")
cat("  Table 3: ATE bias by method x missingness pattern\n")
cat("============================================================\n\n")

tab3 <- aggregate(
  ate_bias ~ method + miss_pattern,
  data = combined,
  FUN = function(x) mean(x, na.rm = TRUE)
)
tab3_wide <- reshape(tab3,
  idvar   = "method",
  timevar = "miss_pattern",
  direction = "wide"
)
names(tab3_wide) <- gsub("ate_bias\\.", "",
                          names(tab3_wide))
print(tab3_wide, digits = 4, row.names = FALSE)

# ============================================================================
# Table 4: CATE coverage
# ============================================================================

cat("\n")
cat("============================================================\n")
cat("  Table 4: 95% CATE credible interval coverage\n")
cat("============================================================\n\n")

cov_data <- combined[!is.na(combined$cate_coverage), ]
if (nrow(cov_data) > 0) {
  tab4 <- aggregate(
    cbind(cate_coverage, cate_ci_width) ~ method,
    data = cov_data,
    FUN = function(x) mean(x, na.rm = TRUE)
  )
  tab4 <- tab4[order(-tab4$cate_coverage), ]
  print(tab4, digits = 4, row.names = FALSE)
} else {
  cat("No coverage data available.\n")
}

# ============================================================================
# Table 5: Wall-clock time
# ============================================================================

cat("\n")
cat("============================================================\n")
cat("  Table 5: Average wall-clock time (seconds)\n")
cat("============================================================\n\n")

tab5 <- aggregate(
  time ~ method,
  data = combined,
  FUN = function(x) mean(x, na.rm = TRUE)
)
tab5 <- tab5[order(tab5$time), ]
print(tab5, digits = 2, row.names = FALSE)

# ============================================================================
# Plot 1: CATE RMSE by outcome scenario, faceted by missingness
# ============================================================================

cat("\n\nGenerating plots...\n")

plot_data <- aggregate(
  cate_rmse_test ~ method + outcome_scenario + miss_pattern,
  data = combined,
  FUN = function(x) mean(x, na.rm = TRUE)
)
plot_data$outcome_label <- outcome_labels[
  as.character(plot_data$outcome_scenario)
]
plot_data$miss_label <- miss_labels[plot_data$miss_pattern]

p1 <- ggplot(
  plot_data,
  aes(x = outcome_label, y = cate_rmse_test,
      fill = method)
) +
  geom_col(position = position_dodge(0.8), width = 0.7) +
  facet_wrap(~ miss_label) +
  scale_fill_manual(values = method_colors) +
  labs(x = "Outcome scenario",
       y = "CATE RMSE (test)",
       title = "CATE estimation accuracy",
       fill = "Method") +
  theme_bench +
  theme(axis.text.x = element_text(angle = 30, hjust = 1))

# ============================================================================
# Plot 2: CATE RMSE by rho (effect of correlation)
# ============================================================================

plot_data2 <- aggregate(
  cate_rmse_test ~ method + outcome_scenario + rho,
  data = combined,
  FUN = function(x) mean(x, na.rm = TRUE)
)
plot_data2$outcome_label <- outcome_labels[
  as.character(plot_data2$outcome_scenario)
]

p2 <- ggplot(
  plot_data2,
  aes(x = factor(rho), y = cate_rmse_test,
      fill = method)
) +
  geom_col(position = position_dodge(0.8), width = 0.7) +
  facet_wrap(~ outcome_label, scales = "free_y") +
  scale_fill_manual(values = method_colors) +
  labs(x = expression(rho ~ "(covariate correlation)"),
       y = "CATE RMSE (test)",
       title = "Effect of covariate correlation on CATE",
       fill = "Method") +
  theme_bench

# ============================================================================
# Plot 3: Coverage by scenario
# ============================================================================

cov_plot <- aggregate(
  cate_coverage ~ method + outcome_scenario + miss_pattern,
  data = combined[!is.na(combined$cate_coverage), ],
  FUN = function(x) mean(x, na.rm = TRUE)
)
cov_plot$outcome_label <- outcome_labels[
  as.character(cov_plot$outcome_scenario)
]
cov_plot$miss_label <- miss_labels[cov_plot$miss_pattern]

p3 <- ggplot(
  cov_plot,
  aes(x = outcome_label, y = cate_coverage,
      fill = method)
) +
  geom_col(position = position_dodge(0.8), width = 0.7) +
  geom_hline(yintercept = 0.95, linetype = "dashed",
             color = "grey50") +
  facet_wrap(~ miss_label) +
  scale_fill_manual(values = method_colors) +
  coord_cartesian(ylim = c(0.5, 1.0)) +
  labs(x = "Outcome scenario",
       y = "95% CI coverage",
       title = "Credible interval coverage for CATE",
       fill = "Method") +
  theme_bench +
  theme(axis.text.x = element_text(angle = 30, hjust = 1))

# ============================================================================
# Plot 4: ATE bias by scenario
# ============================================================================

ate_plot <- aggregate(
  ate_bias ~ method + outcome_scenario + miss_pattern,
  data = combined,
  FUN = function(x) mean(x, na.rm = TRUE)
)
ate_plot$outcome_label <- outcome_labels[
  as.character(ate_plot$outcome_scenario)
]
ate_plot$miss_label <- miss_labels[ate_plot$miss_pattern]

p4 <- ggplot(
  ate_plot,
  aes(x = outcome_label, y = ate_bias, fill = method)
) +
  geom_col(position = position_dodge(0.8), width = 0.7) +
  geom_hline(yintercept = 0, linetype = "dashed",
             color = "grey50") +
  facet_wrap(~ miss_label) +
  scale_fill_manual(values = method_colors) +
  labs(x = "Outcome scenario",
       y = "ATE bias",
       title = "Bias in average treatment effect estimation",
       fill = "Method") +
  theme_bench +
  theme(axis.text.x = element_text(angle = 30, hjust = 1))

# ============================================================================
# Plot 5: Relative CATE RMSE to Oracle
# ============================================================================

oracle_cate <- aggregate(
  cate_rmse_test ~ outcome_scenario + miss_pattern + rho,
  data = combined[combined$method == "Oracle", ],
  FUN = function(x) mean(x, na.rm = TRUE)
)
names(oracle_cate)[4] <- "oracle_cate_rmse"

rel_data <- merge(
  aggregate(
    cate_rmse_test ~ method + outcome_scenario +
      miss_pattern + rho,
    data = combined,
    FUN = function(x) mean(x, na.rm = TRUE)
  ),
  oracle_cate,
  by = c("outcome_scenario", "miss_pattern", "rho")
)
rel_data$relative_cate_rmse <- rel_data$cate_rmse_test /
  rel_data$oracle_cate_rmse
rel_data <- rel_data[rel_data$method != "Oracle", ]

rel_data$outcome_label <- outcome_labels[
  as.character(rel_data$outcome_scenario)
]
rel_data$miss_label <- miss_labels[rel_data$miss_pattern]

p5 <- ggplot(
  rel_data,
  aes(x = outcome_label, y = relative_cate_rmse,
      fill = method)
) +
  geom_col(position = position_dodge(0.8), width = 0.7) +
  geom_hline(yintercept = 1, linetype = "dashed",
             color = "grey50") +
  facet_wrap(~ miss_label) +
  scale_fill_manual(values = method_colors) +
  labs(x = "Outcome scenario",
       y = "CATE RMSE / Oracle CATE RMSE",
       title = "Relative CATE estimation efficiency",
       fill = "Method") +
  theme_bench +
  theme(axis.text.x = element_text(angle = 30, hjust = 1))

# ============================================================================
# Save plots
# ============================================================================

plot_dir <- file.path(results_dir, "figures")
dir.create(plot_dir, showWarnings = FALSE, recursive = TRUE)

ggsave(file.path(plot_dir, "cate_rmse_by_scenario.pdf"),
       p1, width = 12, height = 5)
ggsave(file.path(plot_dir, "cate_rmse_by_rho.pdf"),
       p2, width = 10, height = 7)
ggsave(file.path(plot_dir, "cate_coverage.pdf"),
       p3, width = 12, height = 5)
ggsave(file.path(plot_dir, "ate_bias.pdf"),
       p4, width = 12, height = 5)
ggsave(file.path(plot_dir, "relative_cate_rmse.pdf"),
       p5, width = 12, height = 5)

cat(sprintf("\nPlots saved to %s/\n", plot_dir))

# ============================================================================
# Save summary CSV
# ============================================================================

agg_full <- aggregate(
  cbind(rmse_m_test, cate_bias_test, cate_rmse_test,
        ate_bias, cate_coverage, cate_ci_width,
        time) ~ method + outcome_scenario + miss_pattern +
    rho,
  data = combined,
  FUN = function(x) mean(x, na.rm = TRUE)
)

agg_file <- file.path(results_dir, "irs_v2_summary.csv")
write.csv(agg_full, file = agg_file, row.names = FALSE)
cat(sprintf("Summary table saved to %s\n", agg_file))
