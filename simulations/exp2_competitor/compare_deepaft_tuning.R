# ──────────────────────────────────────────────────────────────────────────────
# Compare the deepAFT tuning variants on the right-censored experiment:
#   k=7 CV    -- fixed 8-6 architecture, 7-fold CV  (old)
#   k=10 CV   -- fixed 8-6 architecture, 10-fold CV
#   arch-tuned-- architecture + hyperparameters searched by 10-fold CV
# Same four estimators (RCT/pool x S/T), pooled evaluation sample.
# Prints per-metric tables (rows = method, cols = variant) and saves a 2x2
# grouped-boxplot figure at lambda_d = lambda_u = 1.  Base R only.
# ──────────────────────────────────────────────────────────────────────────────

res_dir <- "simulations/exp2_competitor"
fig_dir <- file.path(res_dir, "figures")
dir.create(fig_dir, showWarnings = FALSE, recursive = TRUE)

# variant label -> file (only existing files are used)
files <- c("k=7 CV"     = "sim_surv_v12_deepaft_k7_output.rds",
           "k=10 CV"    = "sim_surv_v12_deepaft_output.rds",
           "arch-tuned" = "sim_surv_v12_deepaft_archtune_output.rds")
files <- files[file.exists(file.path(res_dir, files))]
stopifnot(length(files) >= 2)

res <- do.call(rbind, Map(function(f, lab) {
  d <- readRDS(file.path(res_dir, f)); d$variant <- lab; d
}, files, names(files)))
res <- res[res$population == "All", ]                 # pooled evaluation sample

variants <- names(files)
meth      <- c("deepAFT-RCT-S", "deepAFT-RCT-T", "deepAFT-Pooled-S", "deepAFT-Pooled-T")
meth_short<- c("RCT-S", "RCT-T", "Pool-S", "Pool-T")
res$variant <- factor(res$variant, levels = variants)
res$method  <- factor(res$method,  levels = meth)

metrics     <- c("rmse", "bias", "coverage", "postvar")
metric_labs <- c(rmse = "RMSE", bias = "Bias",
                 coverage = "Coverage", postvar = "Posterior variance")
ref_lines   <- c(rmse = NA, bias = 0, coverage = 0.95, postvar = NA)

# Drop non-finite values and rare divergent blow-ups (a diverged deepAFT bootstrap
# refit can give a posterior variance ~1e6); Q3 + 30*IQR catches only artefacts.
trim_div <- function(v) {
  v  <- v[is.finite(v)]
  if (length(v) < 4) return(v)
  qs <- quantile(v, c(0.25, 0.75), names = FALSE); iqr <- qs[2] - qs[1]
  if (iqr <= 0) return(v)
  v[v >= qs[1] - 30 * iqr & v <= qs[2] + 30 * iqr]
}
trimmed_mean <- function(v) mean(trim_div(v))

# ── tables: rows = method, cols = variant ─────────────────────────────────────
print_table <- function(dat, title) {
  cat("\n###", title, "\n")
  for (mt in metrics) {
    m <- tapply(dat[[mt]], list(method = dat$method, variant = dat$variant),
                trimmed_mean)
    m <- m[meth, variants, drop = FALSE]               # fix row/col order
    cat(sprintf("\n-- %s --\n", metric_labs[[mt]]))
    print(round(m, 3))
  }
}
print_table(res, "Mean over replications AND the full lambda grid (pooled sample)")
print_table(res[res$lambda_d == 1 & res$lambda_u == 1, ],
            "Mean over replications at lambda_d = lambda_u = 1 (pooled sample)")

# full per-cell summary to CSV
summ <- aggregate(cbind(rmse, bias, coverage, postvar) ~
                    variant + method + lambda_d + lambda_u,
                  data = res, FUN = trimmed_mean, na.action = na.pass)
csv_path <- file.path(res_dir, "deepaft_tuning_comparison.csv")
write.csv(summ, csv_path, row.names = FALSE)
cat("\nWrote", csv_path, "\n")

# ── figure: 2x2 metrics, grouped boxplots (method groups x variant) at (1,1) ──
vcols <- setNames(hcl.colors(length(variants), "Dark 3"), variants)
d11   <- res[res$lambda_d == 1 & res$lambda_u == 1, ]
nb <- length(variants); ng <- length(meth); gap <- 1
at      <- as.vector(outer(seq_len(nb), (seq_len(ng) - 1) * (nb + gap), "+"))
centers <- (seq_len(ng) - 1) * (nb + gap) + (nb + 1) / 2

draw <- function() {
  op <- par(mfrow = c(2, 2), family = "serif", oma = c(4, 0, 2, 0),
            mar = c(3, 5, 2, 1), mgp = c(3, 1, 0), cex.lab = 1.4, cex.axis = 1.1)
  for (mt in metrics) {
    st   <- boxplot(d11[[mt]] ~ d11$variant + d11$method, plot = FALSE)$stats
    ylim <- range(c(as.vector(st), ref_lines[[mt]]), na.rm = TRUE)
    ylim <- ylim + c(-1, 1) * 0.04 * diff(ylim)
    boxplot(d11[[mt]] ~ d11$variant + d11$method, at = at, col = vcols,
            xaxt = "n", outline = FALSE, ylim = ylim,
            xlim = range(at) + c(-0.5, 0.5),
            ylab = metric_labs[[mt]], xlab = "")
    axis(1, at = centers, labels = meth_short)
    if (!is.na(ref_lines[[mt]])) abline(h = ref_lines[[mt]], lty = 2, col = "grey40")
  }
  mtext("deepAFT tuning comparison  -  pooled sample, lambda_d = lambda_u = 1",
        outer = TRUE, line = 0.2, cex = 1.1, font = 2)
  par(fig = c(0, 1, 0, 1), oma = c(0, 0, 0, 0), mar = c(0, 0, 0, 0), new = TRUE)
  plot.new()
  legend("bottom", legend = variants, fill = vcols, horiz = TRUE, bty = "n",
         xpd = TRUE, cex = 1.2)
  par(op)
}
if (interactive()) draw()
f <- file.path(fig_dir, "deepaft_tuning_comparison.pdf")
pdf(f, width = 12, height = 9, family = "Times"); draw(); dev.off()
cat("Wrote", f, "\n")
