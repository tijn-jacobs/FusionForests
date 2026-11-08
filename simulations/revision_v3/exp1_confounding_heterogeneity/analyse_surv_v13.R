# ══════════════════════════════════════════════════════════════════════════════
# analyse_surv_v13.R -- local analysis of experiment 1 (confounding and
# between-source heterogeneity).
#
# Reads results/sim_surv_v13_output.rds (one row per replication x method x
# population, produced on the HPC by sim_surv_v13.R) and writes:
#   results/figures/ (copy to manuscripts/manuscript_v3/figures/ for the paper)
#     sim_vary_lambda_u.pdf         main text, Figure sim-lambda-u
#     sim_vary_lambda_d.pdf         Web Figure sim-lambda-d
#     sim_vary_lambda_u_hetero.pdf  covariate-dependent confounding
#   results/tables/
#     summary.csv                   mean metrics per (design, lambda, method,
#                                   population), plus across-rep variance/sd
#     censoring.csv                 realised censoring fractions per cell
# and prints the summary.  Base R only.  Run from the repository root:
#   Rscript simulations/revision_v3/exp1_confounding_heterogeneity/analyse_surv_v13.R
#
# Metrics per (design, lambda_d, lambda_u, method, population):
#   rmse     -- mean pointwise CATE RMSE over replications
#   bias     -- mean integrated (averaged-over-points) signed bias
#   variance -- across-replication variance of the integrated bias, var(bias_r)
#   sd       -- sqrt(variance)
#   coverage, width, postvar -- as recorded
# ══════════════════════════════════════════════════════════════════════════════

exp_dir <- "simulations/revision_v3/exp1_confounding_heterogeneity"
res_dir <- file.path(exp_dir, "results")
tab_dir <- file.path(res_dir, "tables")
fig_dir <- file.path(res_dir, "figures")
dir.create(tab_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)

res <- readRDS(file.path(res_dir, "sim_surv_v13_output.rds"))
cat(sprintf("Loaded %d rows; designs: %s; %d reps per cell\n", nrow(res),
            paste(unique(res$design), collapse = ", "), max(res$Iter)))

meth <- c("Fusion", "RCT-only", "RWD-only")
pops <- c("RCT", "RWD", "All")

# ── Summary table ─────────────────────────────────────────────────────────────
grp <- cbind(rmse, bias, coverage, width, postvar) ~
  design + lambda_d + lambda_u + method + population
mean_tbl <- aggregate(grp, data = res,
                      FUN = function(x) mean(x, na.rm = TRUE),
                      na.action = na.pass)
var_tbl  <- aggregate(bias ~ design + lambda_d + lambda_u + method + population,
                      data = res, FUN = function(x) var(x, na.rm = TRUE),
                      na.action = na.pass)
names(var_tbl)[ncol(var_tbl)] <- "variance"
n_tbl    <- aggregate(rmse ~ design + lambda_d + lambda_u + method + population,
                      data = res, FUN = function(x) sum(is.finite(x)),
                      na.action = na.pass)
names(n_tbl)[ncol(n_tbl)] <- "n"
summary_tbl <- merge(merge(mean_tbl, var_tbl), n_tbl)
summary_tbl$sd <- sqrt(summary_tbl$variance)

summary_tbl$design     <- factor(summary_tbl$design,
                                 levels = c("uniform", "heterogeneous"))
summary_tbl$method     <- factor(summary_tbl$method, levels = meth)
summary_tbl$population <- factor(summary_tbl$population, levels = pops)
ord <- order(summary_tbl$design, summary_tbl$lambda_d, summary_tbl$lambda_u,
             summary_tbl$population, summary_tbl$method)
summary_tbl <- summary_tbl[ord, c("design", "lambda_d", "lambda_u",
                                  "population", "method", "n", "rmse", "bias",
                                  "variance", "sd", "coverage", "width",
                                  "postvar")]
write.csv(summary_tbl, file.path(tab_dir, "summary.csv"), row.names = FALSE)

cat("\n=== CATE metrics, population = All (mean over replications) ===\n")
print(summary_tbl[summary_tbl$population == "All", ], row.names = FALSE,
      digits = 3)

# Realised censoring fractions per cell (sanity check).
cens_tbl <- aggregate(cbind(cens_rct, ic_rwd, rc_rwd_tail) ~
                        design + lambda_d + lambda_u,
                      data = res[res$method == "Fusion" &
                                   res$population == "All", ],
                      FUN = function(x) mean(x, na.rm = TRUE))
write.csv(cens_tbl, file.path(tab_dir, "censoring.csv"), row.names = FALSE)
cat("\n=== Realised censoring fractions ===\n")
print(cens_tbl, row.names = FALSE, digits = 3)

# ── Figures ───────────────────────────────────────────────────────────────────
# Boxplots over replications (combined "All" population).  Each figure is a 2x2
# grid of metrics; within a panel the boxes are grouped by the swept lambda,
# with the three estimators side by side per group and one shared legend
# beneath.
a <- res[res$population == "All", ]
a$method <- factor(a$method, levels = meth)
# Okabe-Ito colourblind-safe palette, shared across every figure in the paper:
# Fusion orange, the trial bluish green, the real-world data vermillion.
cols <- c("#E69F00", "#009E73", "#D55E00")   # Fusion, RCT-only, RWD-only
metrics     <- c("rmse", "bias", "coverage", "postvar")
metric_labs <- c(rmse = "RMSE", bias = "Bias", coverage = "Coverage",
                 width = "CI width", postvar = "Posterior variance")

# Draws the 2x2 metric grid; previews on screen in an interactive session and
# writes the same figure to `file`.
plot_panel_grid <- function(dat, group, group_lab, file, ylims = NULL) {
  dat[[group]] <- factor(dat[[group]], levels = sort(unique(dat[[group]])))
  render <- function() {
    op <- par(mfrow = c(2, 2),
              family = "serif",
              oma = c(5, 0, 0, 0),       # outer bottom margin for shared legend
              mar = c(6.5, 8.8, 2, 2),
              mgp = c(5.6, 1.3, 0),
              # Font sizes ~10% above the manuscript KM figure: axis titles
              # ~33 pt (cex 2.75) and tick labels ~26 pt (cex 2.2).
              cex.lab = 2.75, cex.axis = 2.2,
              las = 1)                   # upright y-axis numbers
    nb  <- length(meth)
    ng  <- nlevels(dat[[group]])
    gap <- 1.5                           # extra space between groups
    at      <- as.vector(outer(seq_len(nb), (seq_len(ng) - 1) * (nb + gap), "+"))
    centers <- (seq_len(ng) - 1) * (nb + gap) + (nb + 1) / 2
    seps    <- centers[-ng] + diff(centers) / 2
    for (metric in metrics) {
      ylim_m <- if (is.null(ylims))
        range(boxplot(dat[[metric]] ~ dat$method + dat[[group]],
                      plot = FALSE)$stats, na.rm = TRUE)
      else ylims[[metric]]
      boxplot(dat[[metric]] ~ dat$method + dat[[group]], col = cols, xaxt = "n",
              at = at, xlim = range(at) + c(-0.5, 0.5), ylim = ylim_m,
              outline = FALSE,
              main = "", ylab = metric_labs[metric], xlab = group_lab, sep = " ")
      axis(1, at = centers, labels = levels(dat[[group]]))
      abline(v = seps, lty = 3, col = "grey")
      if (metric == "coverage") abline(h = 0.95, lty = 2)
      if (metric == "bias")     abline(h = 0,    lty = 2)
    }
    par(fig = c(0, 1, 0, 1), oma = c(0, 0, 0, 0), mar = c(0, 0, 0, 0),
        family = "serif", new = TRUE)
    plot.new()
    legend("bottom", legend = meth, fill = cols, horiz = TRUE, bty = "n",
           xpd = TRUE, cex = 2.75)
    par(op)
  }
  if (interactive()) render()
  pdf(file, width = 18, height = 10, family = "Times")  # as survival_curves_manuscript.pdf
  render()
  dev.off()
  cat("wrote", file, "\n")
}

u   <- a[a$design == "uniform", ]
d_u <- u[u$lambda_d == 1, ]   # sweep lambda_u, uniform confounding
d_d <- u[u$lambda_u == 1, ]   # sweep lambda_d
d_h <- rbind(u[u$lambda_d == 1 & u$lambda_u == 0, ],     # shared null cell
             a[a$design == "heterogeneous", ])           # covariate-dependent

# Shared per-metric y-limits so matching panels across the three figures use
# an identical y-axis.  Based on the boxplot whisker extents (outliers hidden),
# extended to include each panel's reference line and padded 4%.
ylims <- setNames(lapply(metrics, function(m) {
  s_u <- boxplot(d_u[[m]] ~ d_u$method + d_u$lambda_u, plot = FALSE)$stats
  s_d <- boxplot(d_d[[m]] ~ d_d$method + d_d$lambda_d, plot = FALSE)$stats
  s_h <- boxplot(d_h[[m]] ~ d_h$method + d_h$lambda_u, plot = FALSE)$stats
  vals <- c(s_u, s_d, s_h)
  if (m == "coverage") vals <- c(vals, 0.95)
  rng <- range(vals, na.rm = TRUE)
  # Bias: symmetric around zero, so over- and underestimation read alike.
  if (m == "bias") rng <- c(-1, 1) * max(abs(rng))
  rng + c(-1, 1) * 0.04 * diff(rng)
}), metrics)

cat("\n")
plot_panel_grid(d_u, "lambda_u", expression(lambda[u]),
                file = file.path(fig_dir, "sim_vary_lambda_u.pdf"), ylims = ylims)
plot_panel_grid(d_d, "lambda_d", expression(lambda[d]),
                file = file.path(fig_dir, "sim_vary_lambda_d.pdf"), ylims = ylims)
plot_panel_grid(d_h, "lambda_u", expression(lambda[u]),
                file = file.path(fig_dir, "sim_vary_lambda_u_hetero.pdf"),
                ylims = ylims)
cat("wrote", file.path(tab_dir, "summary.csv"), "and",
    file.path(tab_dir, "censoring.csv"), "\n")
