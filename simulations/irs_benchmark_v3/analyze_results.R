# ============================================================================
# IRS Benchmark v3 (BCF) — Results analysis
# ============================================================================

setwd("~/Library/CloudStorage/OneDrive-VrijeUniversiteitAmsterdam/Documents/GitHub/FusionForests/simulations/irs_benchmark_v3")

library(ggplot2)

results_dir <- "results"

# ============================================================================
# Load data
# ============================================================================

rds_files <- list.files(results_dir, pattern = "\\.rds$",
                        full.names = TRUE)
if (length(rds_files) == 0)
  stop("No result files found in ", results_dir)

parts <- lapply(rds_files, function(f) {
  cat(sprintf("Loading %s\n", basename(f)))
  readRDS(f)
})
combined <- do.call(rbind, parts)

cat(sprintf("Loaded %d rows across %d scenarios.\n",
            nrow(combined),
            length(unique(combined$scenario_id))))

# Labels
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
  "MissForest+BCF"      = "#984EA3",
  "Complete case"       = "#377EB8",
  "Complete covariates" = "#FF7F00"
)

# ============================================================================
# Helper plots
# ============================================================================

make_scenario_plot <- function(data, yvar, ylabel, title,
                               hline = NULL,
                               ylim = NULL) {
  p <- ggplot(
    data,
    aes(x = outcome_label, y = .data[[yvar]],
        fill = method)
  ) +
    geom_col(position = position_dodge(0.8), width = 0.7) +
    facet_wrap(~ miss_label) +
    scale_fill_manual(values = method_colors) +
    labs(x = "Outcome scenario", y = ylabel,
         title = title, fill = "Method") +
    theme_bench +
    theme(axis.text.x = element_text(angle = 30, hjust = 1))
  if (!is.null(hline))
    p <- p + geom_hline(yintercept = hline,
                        linetype = "dashed",
                        color = "grey50")
  if (!is.null(ylim)) p <- p + coord_cartesian(ylim = ylim)
  p
}

make_rho_plot <- function(data, yvar, ylabel, title,
                          hline = NULL) {
  p <- ggplot(
    data,
    aes(x = factor(rho), y = .data[[yvar]],
        fill = method)
  ) +
    geom_col(position = position_dodge(0.8), width = 0.7) +
    facet_wrap(~ outcome_label, scales = "free_y") +
    scale_fill_manual(values = method_colors) +
    labs(x = expression(rho ~ "(covariate correlation)"),
         y = ylabel, title = title, fill = "Method") +
    theme_bench
  if (!is.null(hline))
    p <- p + geom_hline(yintercept = hline,
                        linetype = "dashed",
                        color = "grey50")
  p
}

# ============================================================================
# Aggregate
# ============================================================================

metrics_scen <- aggregate(
  cbind(cate_rmse_test, cate_bias_test, rmse_m_test,
        cate_rmse_train, cate_bias_train, rmse_m_train,
        ate_bias, cate_coverage, cate_ci_width,
        avg_post_var) ~
    method + outcome_scenario + miss_pattern,
  data = combined,
  FUN = function(x) mean(x, na.rm = TRUE),
  na.action = na.pass
)
metrics_scen$outcome_label <- outcome_labels[
  as.character(metrics_scen$outcome_scenario)
]
metrics_scen$miss_label <- miss_labels[
  metrics_scen$miss_pattern
]

metrics_rho <- aggregate(
  cbind(cate_rmse_test, cate_bias_test, rmse_m_test,
        cate_rmse_train, cate_bias_train, rmse_m_train,
        ate_bias, cate_coverage, cate_ci_width,
        avg_post_var) ~
    method + outcome_scenario + rho,
  data = combined,
  FUN = function(x) mean(x, na.rm = TRUE),
  na.action = na.pass
)
metrics_rho$outcome_label <- outcome_labels[
  as.character(metrics_rho$outcome_scenario)
]

# ============================================================================
# Plot specifications
# ============================================================================

plot_specs <- list(
  list(yvar = "cate_rmse_test",
       ylabel = "CATE RMSE (test)",
       title = "CATE estimation accuracy — test set"),
  list(yvar = "cate_rmse_train",
       ylabel = "CATE RMSE (train)",
       title = "CATE estimation accuracy — training set"),
  list(yvar = "cate_bias_test",
       ylabel = "CATE bias (test)",
       title = "CATE estimation bias — test set",
       hline = 0),
  list(yvar = "cate_bias_train",
       ylabel = "CATE bias (train)",
       title = "CATE estimation bias — training set",
       hline = 0),
  list(yvar = "rmse_m_test",
       ylabel = "RMSE of m (test)",
       title = "Prediction accuracy — test set"),
  list(yvar = "rmse_m_train",
       ylabel = "RMSE of m (train)",
       title = "Prediction accuracy — training set"),
  list(yvar = "ate_bias",
       ylabel = "ATE bias",
       title = "Bias in average treatment effect",
       hline = 0),
  list(yvar = "cate_coverage",
       ylabel = "95% CI coverage",
       title = "Credible interval coverage for CATE",
       hline = 0.95, ylim = c(0.5, 1.0)),
  list(yvar = "cate_ci_width",
       ylabel = "Average CI width",
       title = "Credible interval width for CATE"),
  list(yvar = "avg_post_var",
       ylabel = "Avg posterior variance",
       title = "Average posterior variance of CATE")
)

# ============================================================================
# Save plots
# ============================================================================

plot_dir <- file.path(results_dir, "figures")
dir.create(plot_dir, showWarnings = FALSE, recursive = TRUE)

pdf(file.path(plot_dir, "results_by_scenario.pdf"),
    width = 12, height = 5)
for (spec in plot_specs) {
  p <- make_scenario_plot(
    metrics_scen, spec$yvar, spec$ylabel, spec$title,
    hline = spec$hline, ylim = spec$ylim
  )
  print(p)
}
dev.off()

pdf(file.path(plot_dir, "results_by_rho.pdf"),
    width = 10, height = 7)
for (spec in plot_specs) {
  p <- make_rho_plot(
    metrics_rho, spec$yvar, spec$ylabel, spec$title,
    hline = spec$hline
  )
  print(p)
}
dev.off()

cat(sprintf("\nPlots saved to %s/\n", plot_dir))

# ============================================================================
# Tables
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
# Save summary CSV
# ============================================================================

agg_means <- aggregate(
  cbind(rmse_m_train, rmse_m_test, mae_m_test,
        cate_bias_train, cate_rmse_train,
        cate_bias_test, cate_rmse_test,
        ate_bias, ate_hat, ate_true,
        cate_coverage, cate_ci_width, avg_post_var,
        ate_coverage, ate_ci_width
        ) ~ method + outcome_scenario +
    miss_pattern + rho,
  data = combined,
  FUN = function(x) mean(x, na.rm = TRUE),
  na.action = na.pass
)

ate_var <- aggregate(
  ate_hat ~ method + outcome_scenario +
    miss_pattern + rho,
  data = combined,
  FUN = function(x) var(x, na.rm = TRUE),
  na.action = na.pass
)
names(ate_var)[ncol(ate_var)] <- "ate_freq_var"

ate_rmse <- aggregate(
  ate_bias ~ method + outcome_scenario +
    miss_pattern + rho,
  data = combined,
  FUN = function(x) sqrt(mean(x^2, na.rm = TRUE)),
  na.action = na.pass
)
names(ate_rmse)[ncol(ate_rmse)] <- "ate_rmse"

agg_full <- merge(agg_means, ate_var,
  by = c("method", "outcome_scenario",
         "miss_pattern", "rho")
)
agg_full <- merge(agg_full, ate_rmse,
  by = c("method", "outcome_scenario",
         "miss_pattern", "rho")
)

num_cols <- sapply(agg_full, is.numeric)
agg_full[num_cols] <- round(agg_full[num_cols], 3)

agg_file <- file.path(results_dir, "irs_v3_summary.csv")
write.csv(agg_full, file = agg_file, row.names = FALSE)
cat(sprintf("Summary table saved to %s\n", agg_file))

