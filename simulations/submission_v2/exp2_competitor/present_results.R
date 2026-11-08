# ──────────────────────────────────────────────────────────────────────────────
# Present the right-censored competitor results: load the saved outputs, build a
# summary table, and draw boxplot grids of the CATE metrics over the 3x3 lambda
# grid.  Base R only (no ggplot2 dependency).  Run from the repo root.
# ──────────────────────────────────────────────────────────────────────────────

res_dir <- "simulations/exp2_competitor"
fig_dir <- file.path(res_dir, "figures")
dir.create(fig_dir, showWarnings = FALSE, recursive = TRUE)

# Which result files to load (only those that exist are used).  Add the RSF line
# once you have a fresh RSF run.
files <- c(
  "sim_surv_v12_bff_output.rds",
  "sim_surv_v12_deepaft_output.rds"
  # , "sim_surv_v12_rsf_output.rds"
)
files <- files[file.exists(file.path(res_dir, files))]
stopifnot(length(files) > 0)

res <- do.call(rbind, lapply(files, function(f) readRDS(file.path(res_dir, f))))

cat("Loaded result files:\n")
for (f in files) {
  d <- readRDS(file.path(res_dir, f))
  cat(sprintf("  %-34s %5d reps x %d methods: %s\n", f, max(d$Iter),
              length(unique(d$method)), paste(unique(d$method), collapse = ", ")))
}

# ── factor ordering (Fusion first, then deepAFT, then RSF) ────────────────────
method_order <- c("Fusion",
                  "deepAFT-RCT-S", "deepAFT-RCT-T",
                  "deepAFT-Pooled-S", "deepAFT-Pooled-T",
                  "RSF-RCT-S", "RSF-RCT-T", "RSF-Pooled-S", "RSF-Pooled-T",
                  "RSF-RCT", "RSF-Pooled")
lv <- method_order[method_order %in% unique(res$method)]
lv <- c(lv, setdiff(unique(res$method), lv))     # keep any unexpected labels
res$method     <- factor(res$method, levels = lv)
res$population  <- factor(res$population, levels = c("RCT", "RWD", "All"))

metrics     <- c("rmse", "bias", "coverage", "width", "postvar")
metric_labs <- c(rmse = "RMSE", bias = "Bias", coverage = "Coverage",
                 width = "CI width", postvar = "Posterior variance")
ref_lines   <- c(rmse = NA, bias = 0, coverage = 0.95, width = NA, postvar = NA)

# ── summary table: mean over replications per method x population x cell ───────
summ <- aggregate(cbind(rmse, bias, coverage, width, postvar) ~
                    method + population + lambda_d + lambda_u,
                  data = res, FUN = function(x) mean(x, na.rm = TRUE),
                  na.action = na.pass)
# across-replication variance of the (integrated) bias -- estimator variance
vtab <- aggregate(bias ~ method + population + lambda_d + lambda_u,
                  data = res, FUN = function(x) var(x, na.rm = TRUE),
                  na.action = na.pass)
names(vtab)[ncol(vtab)] <- "bias_var"
summ <- merge(summ, vtab)
summ <- summ[order(summ$population, summ$lambda_d, summ$lambda_u, summ$method), ]

csv_path <- file.path(res_dir, "results_summary.csv")
write.csv(summ, csv_path, row.names = FALSE)
cat("\nSummary table written to:", csv_path, "\n")

# Compact console view: mean RMSE by method (rows) x lambda cell (cols), All pop.
allp <- summ[summ$population == "All", ]
cell <- sprintf("d%g/u%g", allp$lambda_d, allp$lambda_u)
rmse_wide <- tapply(allp$rmse, list(method = allp$method, cell = cell), FUN = identity)
cat("\n=== mean CATE RMSE, population = All  (rows: method, cols: lambda_d/lambda_u) ===\n")
print(round(rmse_wide, 3))

# ── boxplot grid: one figure per metric, 3x3 panels over the lambda grid ──────
# Within each panel the estimators sit side by side (coloured); a shared legend
# sits beneath.  Panels are arranged with lambda_d down the rows and lambda_u
# across the columns.
plot_metric_grid <- function(metric, population, file = NULL) {
  d    <- droplevels(res[res$population == population, ])
  meth <- levels(d$method)
  cols <- setNames(hcl.colors(length(meth), "Dark 3"), meth)
  lds  <- sort(unique(d$lambda_d))
  lus  <- sort(unique(d$lambda_u))
  ref  <- ref_lines[[metric]]

  # shared y-limits across panels (whisker range, hide outliers), incl. ref line
  st   <- boxplot(d[[metric]] ~ d$method + interaction(d$lambda_d, d$lambda_u),
                  plot = FALSE)$stats
  ylim <- range(c(as.vector(st), ref), na.rm = TRUE)
  ylim <- ylim + c(-1, 1) * 0.04 * diff(ylim)

  render <- function() {
    op <- par(mfrow = c(length(lds), length(lus)),
              oma = c(7, 1, 3, 1), mar = c(1.5, 4, 2.2, 1),
              family = "serif", cex.axis = 0.9)
    for (ld in lds) for (lu in lus) {
      dd <- d[d$lambda_d == ld & d$lambda_u == lu, ]
      boxplot(dd[[metric]] ~ factor(dd$method, levels = meth),
              col = cols, xaxt = "n", outline = FALSE, ylim = ylim,
              ylab = metric_labs[[metric]], xlab = "",
              main = bquote(lambda[d] == .(ld) ~ "," ~ lambda[u] == .(lu)))
      if (!is.na(ref)) abline(h = ref, lty = 2, col = "grey40")
    }
    mtext(sprintf("%s  -  population: %s", metric_labs[[metric]], population),
          outer = TRUE, line = 1, cex = 1.3, font = 2)
    # shared legend across the bottom
    par(fig = c(0, 1, 0, 1), oma = c(0, 0, 0, 0), mar = c(0, 0, 0, 0),
        new = TRUE)
    plot.new()
    legend("bottom", legend = meth, fill = cols,
           ncol = min(length(meth), 5), bty = "n", xpd = TRUE, cex = 1.0)
    par(op)
  }

  render()                                   # draw to the active device
  if (!is.null(file)) {                       # and write a PDF
    pdf(file, width = 11, height = 9, family = "Times")
    render()
    dev.off()
    cat("  wrote", file, "\n")
  }
}

# Generate figures for the "All" and "RCT" populations, all metrics.
cat("\nDrawing figures:\n")
for (pop in c("All", "RCT")) {
  for (m in metrics) {
    plot_metric_grid(m, pop,
                     file = file.path(fig_dir, sprintf("%s_%s.pdf", m, pop)))
  }
}
cat("\nDone. Figures in:", fig_dir, "\n")
