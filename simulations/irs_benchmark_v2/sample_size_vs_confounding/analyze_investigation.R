# ============================================================================
# Analysis: Sample size vs confounding investigation
# ============================================================================

setwd("~/Library/CloudStorage/OneDrive-VrijeUniversiteitAmsterdam/Documents/GitHub/FusionForests/simulations/irs_benchmark_v2/sample_size_vs_confounding")

library(ggplot2)

# ============================================================================
# Load data
# ============================================================================

rds_file <- "investigation_results.rds"
if (!file.exists(rds_file)) {
  rds_file <- "irs_v2_investigation_output.rds"
}
if (!file.exists(rds_file)) {
  stop("No results file found.")
}

combined <- readRDS(rds_file)
cat(sprintf("Loaded %d rows.\n", nrow(combined)))

# ============================================================================
# Labels
# ============================================================================

outcome_labels <- c(
  "1" = "Sc 1: Prognostic",
  "2" = "Sc 2: Effect modifier",
  "3" = "Sc 3: Prognostic + EM"
)

combined$outcome_label <- outcome_labels[
  as.character(combined$outcome_scenario)
]
combined$size_label <- factor(
  combined$sample_config,
  levels = c("150_350", "500_500", "350_150"),
  labels = c("n_RCT=150\nn_RWD=350",
             "n_RCT=500\nn_RWD=500",
             "n_RCT=350\nn_RWD=150")
)

# ============================================================================
# Aggregate
# ============================================================================

agg <- aggregate(
  cbind(rmse_m_test, cate_rmse_test, cate_bias_test,
        ate_bias, cate_coverage, cate_ci_width,
        ate_coverage) ~
    method + outcome_scenario + outcome_label +
    sample_config + size_label + confounding,
  data = combined,
  FUN = function(x) mean(x, na.rm = TRUE),
  na.action = na.pass
)

# ============================================================================
# Theme and colours
# ============================================================================

theme_inv <- theme_bw(base_size = 11) +
  theme(
    legend.position  = "bottom",
    strip.background = element_rect(fill = "grey90"),
    panel.grid.minor = element_blank()
  )

method_colors <- c(
  "Oracle"                  = "#333333",
  "IRS (informed)"          = "#E41A1C",
  "IRS (draw-then-decide)"  = "#377EB8",
  "IRS (uniform)"           = "#FB9A99",
  "IRS (informed, excl mu)" = "#B2182B",
  "IRS (dtd, excl mu)"      = "#2166AC",
  "IRS (uniform, excl mu)"  = "#D6604D",
  "MissForest+BART"         = "#984EA3",
  "Complete covariates"     = "#FF7F00"
)

# ============================================================================
# Helper: grouped bar plot
# ============================================================================

make_plot <- function(data, yvar, ylabel, title,
                      hline = NULL, ylim = NULL) {
  p <- ggplot(
    data,
    aes(x = size_label, y = .data[[yvar]],
        fill = method)
  ) +
    geom_col(position = position_dodge(0.8), width = 0.7) +
    facet_grid(outcome_label ~ confounding) +
    scale_fill_manual(values = method_colors) +
    labs(x = "Sample configuration",
         y = ylabel,
         title = title,
         fill = "Method") +
    theme_inv
  if (!is.null(hline)) {
    p <- p + geom_hline(yintercept = hline,
                        linetype = "dashed",
                        color = "grey50")
  }
  if (!is.null(ylim)) p <- p + coord_cartesian(ylim = ylim)
  p
}

# ============================================================================
# Multi-page PDF
# ============================================================================

plot_specs <- list(
  list(yvar = "cate_rmse_test",
       ylabel = "CATE RMSE (test)",
       title = "CATE estimation accuracy"),
  list(yvar = "cate_bias_test",
       ylabel = "CATE bias (test)",
       title = "CATE estimation bias",
       hline = 0),
  list(yvar = "rmse_m_test",
       ylabel = "RMSE of m (test)",
       title = "Prediction accuracy"),
  list(yvar = "ate_bias",
       ylabel = "ATE bias",
       title = "Bias in ATE estimation",
       hline = 0),
  list(yvar = "cate_coverage",
       ylabel = "95% CI coverage",
       title = "Credible interval coverage",
       hline = 0.95, ylim = c(0.3, 1.0)),
  list(yvar = "cate_ci_width",
       ylabel = "Average CI width",
       title = "Credible interval width"),
  list(yvar = "ate_coverage",
       ylabel = "ATE 95% CI coverage",
       title = "ATE credible interval coverage",
       hline = 0.95, ylim = c(0.0, 1.0))
)

pdf("investigation_plots.pdf", width = 10, height = 10)
for (spec in plot_specs) {
  p <- make_plot(agg, spec$yvar, spec$ylabel, spec$title,
                 hline = spec$hline, ylim = spec$ylim)
  print(p)
}
dev.off()

cat("Plots saved to investigation_plots.pdf\n")

# ============================================================================
# Print key tables
# ============================================================================

cat("\n")
cat("============================================================\n")
cat("  IRS vs Complete covariates: CATE RMSE (test)\n")
cat("  Rows: sample config x confounding\n")
cat("  Columns: outcome scenario\n")
cat("============================================================\n\n")

for (m in c("IRS (informed)", "IRS (draw-then-decide)",
            "IRS (uniform)",
            "IRS (informed, excl mu)",
            "IRS (dtd, excl mu)",
            "IRS (uniform, excl mu)",
            "Complete covariates", "MissForest+BART",
            "Oracle")) {
  cat(sprintf("--- %s ---\n", m))
  sub <- agg[agg$method == m,
    c("outcome_scenario", "sample_config", "confounding",
      "cate_rmse_test", "cate_bias_test", "ate_bias",
      "cate_coverage")
  ]
  sub <- sub[order(sub$outcome_scenario,
                   sub$confounding,
                   sub$sample_config), ]
  num_cols <- sapply(sub, is.numeric)
  sub[num_cols] <- round(sub[num_cols], 3)
  print(sub, row.names = FALSE)
  cat("\n")
}

# ============================================================================
# Key diagnostic: IRS gap from Oracle vs Complete covariates
# ============================================================================

cat("============================================================\n")
cat("  Diagnostic: Where does each IRS variant sit?\n")
cat("  gap_to_oracle = (method - Oracle) / Oracle\n")
cat("  gap_to_compcov = (method - CompCov) / CompCov\n")
cat("  If gap_to_compcov ~ 0: method collapsed to CompCov\n")
cat("============================================================\n\n")

orc <- agg[agg$method == "Oracle", ]
cc  <- agg[agg$method == "Complete covariates", ]

keys <- c("outcome_scenario", "sample_config",
          "confounding")

for (irs_name in c("IRS (informed)",
                    "IRS (draw-then-decide)",
                    "IRS (uniform)",
                    "IRS (informed, excl mu)",
                    "IRS (dtd, excl mu)",
                    "IRS (uniform, excl mu)")) {

  cat(sprintf("--- %s ---\n", irs_name))

  irs <- agg[agg$method == irs_name, ]

  diag <- merge(
    irs[, c(keys, "cate_rmse_test")],
    orc[, c(keys, "cate_rmse_test")],
    by = keys, suffixes = c("_irs", "_oracle")
  )
  diag <- merge(
    diag,
    cc[, c(keys, "cate_rmse_test")],
    by = keys
  )
  names(diag)[ncol(diag)] <- "cate_rmse_test_compcov"

  diag$gap_to_oracle <- round(
    (diag$cate_rmse_test_irs -
       diag$cate_rmse_test_oracle) /
      diag$cate_rmse_test_oracle, 3
  )
  diag$gap_to_compcov <- round(
    (diag$cate_rmse_test_irs -
       diag$cate_rmse_test_compcov) /
      diag$cate_rmse_test_compcov, 3
  )

  diag <- diag[order(diag$outcome_scenario,
                     diag$confounding,
                     diag$sample_config), ]
  print(diag, row.names = FALSE)
  cat("\n")
}

# ============================================================================
# Save summary CSV
# ============================================================================

num_cols <- sapply(agg, is.numeric)
agg[num_cols] <- round(agg[num_cols], 3)

write.csv(agg, file = "investigation_summary.csv",
          row.names = FALSE)
cat("\nSummary saved to investigation_summary.csv\n")
