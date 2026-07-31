# ──────────────────────────────────────────────────────────────────────────────
# Figures for the BFF-vs-deepAFT experiment (right-censored RWD).
# Methods: Bayesian fusion forest (Fusion) and the deep-AFT S-/T-learners fitted
# on the RCT alone and on the naive RCT+RWD pool.  All metrics are evaluated on
# the full pooled sample (population = "All").
#
#   * Main text : 2x2 panel (RMSE, Bias, Coverage, Posterior variance) at the
#                 single cell lambda_d = lambda_u = 1.
#   * SM        : the full 3x3 lambda grid, one figure per metric.
#
# Base R only.  Run from the repo root.  Writes PDFs to notes/general/figures/.
# ──────────────────────────────────────────────────────────────────────────────

res_dir <- "simulations/exp2_competitor"
fig_dir <- "notes/general/figures"   # write straight to the manuscript figures dir
dir.create(fig_dir, showWarnings = FALSE, recursive = TRUE)

res <- rbind(
  readRDS(file.path(res_dir, "sim_surv_v12_bff_output.rds")),
  readRDS(file.path(res_dir, "sim_surv_v12_deepaft_output.rds"))
)
res <- res[res$population == "All", ]          # evaluate on the full pooled sample

# Methods, display labels and colours.  RCT learners in blue, pooled in red,
# Fusion in gold; S = light shade, T = dark shade.
meth      <- c("Fusion", "deepAFT-RCT-S", "deepAFT-RCT-T",
               "deepAFT-Pooled-S", "deepAFT-Pooled-T")
meth_labs <- c("Bayesian fusion forest", "DNN (RCT, S)", "DNN (RCT, T)",
               "DNN (pool, S)", "DNN (pool, T)")
# Okabe-Ito colourblind-safe palette. Fusion keeps the orange it has in every
# other figure; the DNN variants keep the cool/warm split, with the light shade
# for the S-learner and the dark shade for the T-learner.
#   orange #E69F00 | sky blue #56B4E9 | blue #0072B2
#   reddish purple #CC79A7 | vermillion #D55E00
cols      <- c("#E69F00", "#56B4E9", "#0072B2", "#CC79A7", "#D55E00")
stopifnot(all(meth %in% unique(res$method)))
res$method <- factor(res$method, levels = meth)

metrics     <- c("rmse", "bias", "coverage", "postvar")
metric_labs <- c(rmse = "RMSE", bias = "Bias",
                 coverage = "Coverage", postvar = "Posterior variance")
ref_lines   <- c(rmse = NA, bias = 0, coverage = 0.95, postvar = NA)

# One boxplot panel: the five methods (coloured) for `metric` on data `dd`.
panel <- function(dd, metric, ylim = NULL, lab = metric_labs[[metric]]) {
  if (is.null(ylim)) {
    st   <- boxplot(dd[[metric]] ~ dd$method, plot = FALSE)$stats
    ylim <- range(c(as.vector(st), ref_lines[[metric]]), na.rm = TRUE)
    ylim <- ylim + c(-1, 1) * 0.04 * diff(ylim)
  }
  boxplot(dd[[metric]] ~ dd$method, col = cols, xaxt = "n", outline = FALSE,
          ylim = ylim, ylab = lab, xlab = "")
  if (!is.na(ref_lines[[metric]])) abline(h = ref_lines[[metric]], lty = 2,
                                          col = "grey40")
}

# ── Main-text figure: 2x2 metrics at lambda_d = lambda_u = 1 ───────────────────
d11 <- res[res$lambda_d == 1 & res$lambda_u == 1, ]
main_file <- file.path(fig_dir, "sim_deepaft_main.pdf")
draw_main <- function() {
  # All four metrics in a single row (5 boxes each -> 20 bars across), with one
  # shared legend beneath.  Wide-strip canvas in Times to match the manuscript;
  # large fonts so the figure stays legible when scaled to \textwidth.
  main_labs <- c(rmse = "RMSE", bias = "Bias",
                 coverage = "Coverage", postvar = "Posterior var.")
  op <- par(mfrow = c(1, 4), family = "serif", oma = c(6, 0, 0, 0),
            mar = c(3, 10.5, 2.5, 3.5), mgp = c(7.0, 1.5, 0),
            cex.lab = 4.3, cex.axis = 3.2,
            las = 1)                     # upright (horizontal) y-axis numbers
  for (m in metrics) panel(d11, m, lab = main_labs[[m]])
  par(fig = c(0, 1, 0, 1), oma = c(0, 0, 0, 0), mar = c(0, 0, 0, 0),
      family = "serif", new = TRUE)
  plot.new()
  legend("bottom", legend = meth_labs, fill = cols, horiz = TRUE, bty = "n",
         xpd = TRUE, cex = 3.3)
  par(op)
}
if (interactive()) draw_main()              # preview only in an interactive session
pdf(main_file, width = 24, height = 7.5, family = "Times"); draw_main(); dev.off()
cat("wrote", main_file, "\n")

# ── SM figures: full 3x3 lambda grid, one PDF per metric ──────────────────────
lds <- sort(unique(res$lambda_d)); lus <- sort(unique(res$lambda_u))
draw_grid <- function(metric) {
  # shared y-limits across all nine cells
  st   <- boxplot(res[[metric]] ~ res$method + interaction(res$lambda_d, res$lambda_u),
                  plot = FALSE)$stats
  ylim <- range(c(as.vector(st), ref_lines[[metric]]), na.rm = TRUE)
  ylim <- ylim + c(-1, 1) * 0.04 * diff(ylim)
  render <- function() {
    op <- par(mfrow = c(length(lds), length(lus)), family = "serif",
              oma = c(7, 1, 3, 1), mar = c(1.5, 4.2, 2.2, 1), cex.axis = 0.9)
    for (ld in lds) for (lu in lus) {
      dd <- res[res$lambda_d == ld & res$lambda_u == lu, ]
      panel(dd, metric, ylim = ylim)
      title(main = bquote(lambda[d] == .(ld) ~ "," ~ lambda[u] == .(lu)))
    }
    mtext(sprintf("%s  -  population: pooled (All)", metric_labs[[metric]]),
          outer = TRUE, line = 1, cex = 1.3, font = 2)
    par(fig = c(0, 1, 0, 1), oma = c(0, 0, 0, 0), mar = c(0, 0, 0, 0),
        new = TRUE)
    plot.new()
    legend("bottom", legend = meth_labs, fill = cols, ncol = 3, bty = "n",
           xpd = TRUE, cex = 1.0)
    par(op)
  }
  f <- file.path(fig_dir, sprintf("sim_deepaft_%s.pdf", metric))
  pdf(f, width = 12, height = 11, family = "Times"); render(); dev.off()
  cat("wrote", f, "\n")
}
for (m in metrics) draw_grid(m)

# ── Numeric summary to the console ────────────────────────────────────────────
# For each metric: a 3x3 (lambda_d x lambda_u) grid; each cell is the mean over
# replications with the 2.5%-97.5% replication interval in brackets (pooled).
meth_tex <- c("BFF", "DNN R/S", "DNN R/T", "DNN P/S", "DNN P/T")
# Drop non-finite values and rare divergent blow-ups (a diverged deepAFT bootstrap
# refit can give a posterior variance ~1e6).  The Q3 + 30*IQR rule is extreme
# enough to catch only genuine numerical artefacts, leaving heavy tails intact.
trim_div <- function(v) {
  v  <- v[is.finite(v)]
  if (length(v) < 4) return(v)
  qs <- quantile(v, c(0.25, 0.75), names = FALSE); iqr <- qs[2] - qs[1]
  if (iqr <= 0) return(v)
  v[v >= qs[1] - 30 * iqr & v <= qs[2] + 30 * iqr]
}
fmt_ci <- function(v) {
  v <- trim_div(v)
  sprintf("%.3f [%.3f, %.3f]", mean(v),
          quantile(v, 0.025, names = FALSE), quantile(v, 0.975, names = FALSE))
}
for (mt in metrics) {
  grid <- expand.grid(lambda_d = lds, lambda_u = lus)
  tab  <- grid
  for (k in seq_along(meth)) {
    tab[[meth_tex[k]]] <- vapply(seq_len(nrow(grid)), function(i) {
      fmt_ci(res[res$lambda_d == grid$lambda_d[i] &
                   res$lambda_u == grid$lambda_u[i] &
                   res$method   == meth[k], mt])
    }, character(1))
  }
  cat(sprintf("\n=== %s  (mean [2.5%%, 97.5%%] over reps; pooled sample) ===\n",
              metric_labs[[mt]]))
  print(tab, row.names = FALSE)
}

cat("\nDone. Figures in:", fig_dir, "\n")
