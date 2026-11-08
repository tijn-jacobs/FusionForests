# ══════════════════════════════════════════════════════════════════════════════
# analyse_prior_v1.R -- local analysis of experiment 5 (prior sensitivity).
#
# Reads sim_prior_v1_output.rds (one row per fit, produced on the HPC by
# sim_prior_v1.R) and writes, to results/figures/ and results/tables/:
#   tables/prior_means.csv        mean of every metric per (cell, config)
#   tables/prior_paired_diff.csv  paired difference to the default, with SE
#   tables/prior_table.tex        key metrics per (cell, config), LaTeX
#   figures/prior_sensitivity.pdf paired differences for the key metrics
# and prints the main table.  None of this is in the manuscript yet; the .tex
# file is pasted by hand.  Base R only.  Run from the repository root:
#   Rscript simulations/revision_v3/exp5_prior_sensitivity/analyse_prior_v1.R
# ══════════════════════════════════════════════════════════════════════════════

res_dir <- "simulations/revision_v3/exp5_prior_sensitivity/results"
res     <- readRDS(file.path(res_dir, "sim_prior_v1_output.rds"))
fig_dir <- file.path(res_dir, "figures")
tab_dir <- file.path(res_dir, "tables")
dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(tab_dir, recursive = TRUE, showWarnings = FALSE)

# Configurations in the order of sim_prior_v1.R; "default" first.
config_names <- unique(res$config)
cells        <- c("uniform", "heterogeneous")
cell_labs    <- c(uniform       = "Uniform confounding, lambda_u = 1",
                  heterogeneous = "Covariate-dependent confounding, lambda_u = 2")
cat(sprintf("%d fits, %d failed; %d configs x %d cells x %d reps\n",
            nrow(res), sum(res$failed), length(config_names),
            length(unique(res$cell)), max(res$Iter)))

# ── Means per (cell, config) ──────────────────────────────────────────────────
metrics <- setdiff(names(res), c("cell", "config", "Iter", "lambda_d",
                                 "lambda_u", "failed"))
means <- aggregate(res[, metrics], by = res[, c("cell", "config")],
                   FUN = function(x) mean(x, na.rm = TRUE))
# ATE as an RMSE over replications, and the number of successful fits.
ate_rmse <- aggregate(res$ate_err, by = res[, c("cell", "config")],
                      FUN = function(x) sqrt(mean(x^2, na.rm = TRUE)))
names(ate_rmse)[3] <- "ate_rmse"
n_ok <- aggregate(!res$failed, by = res[, c("cell", "config")], FUN = sum)
names(n_ok)[3] <- "n"
means <- merge(merge(means, ate_rmse), n_ok)
means$config <- factor(means$config, levels = config_names)
means$cell   <- factor(means$cell, levels = cells)
means <- means[order(means$cell, means$config), ]
write.csv(means, file.path(tab_dir, "prior_means.csv"), row.names = FALSE)

show <- c("n", "cate_all_rmse", "cate_all_bias", "cate_all_cover",
          "cate_all_width", "c_rmse", "c_cover", "d_rmse", "d_cover",
          "ate_rmse", "ate_cover", "ate_width")
for (cl in cells) {
  cat("\n===", cell_labs[[cl]], "===\n")
  # sd of the true c(x) over the RWD rows: a c(x) RMSE near this value means
  # the fit recovers no more of c's heterogeneity than a constant would.
  cat(sprintf("mean sd of the true c(x): %.3f\n",
              mean(res$sd_c_true[res$cell == cl], na.rm = TRUE)))
  m <- means[means$cell == cl, c("config", show)]
  names(m) <- sub("^cate_all_", "cate_", names(m))
  print(m, row.names = FALSE, digits = 3)
}

# ── Paired differences to the default ─────────────────────────────────────────
# Every configuration was fitted to the same datasets, so the difference to the
# default on the same replication removes the data noise.
key <- c(cate_all_rmse = "CATE RMSE", cate_all_cover = "CATE coverage",
         c_rmse = "c(x) RMSE", ate_width = "ATE interval width")
base <- res[res$config == "default", c("cell", "Iter", names(key))]
dd   <- merge(res[res$config != "default", c("cell", "config", "Iter", names(key))],
              base, by = c("cell", "Iter"), suffixes = c("", ".def"))
diffs <- do.call(rbind, lapply(names(key), function(k) {
  dv <- dd[[k]] - dd[[paste0(k, ".def")]]
  a  <- aggregate(dv, by = dd[, c("cell", "config")],
                  FUN = function(x) c(mean = mean(x, na.rm = TRUE),
                                      se = sd(x, na.rm = TRUE) / sqrt(sum(!is.na(x)))))
  data.frame(cell = a$cell, config = a$config, metric = k,
             diff = a$x[, "mean"], se = a$x[, "se"])
}))
diffs$config <- factor(diffs$config, levels = config_names[-1])
write.csv(diffs, file.path(tab_dir, "prior_paired_diff.csv"), row.names = FALSE)

# ── Figure: paired differences, one panel per (metric, cell) ──────────────────
plot_diffs <- function() {
  op <- par(mfrow = c(length(key), length(cells)), mar = c(3, 9, 2.5, 1),
            oma = c(1, 0, 0, 0), family = "serif", las = 1, cex.axis = 0.8)
  for (k in names(key)) for (cl in cells) {
    s <- diffs[diffs$metric == k & diffs$cell == cl, ]
    s <- s[order(s$config, decreasing = TRUE), ]
    y <- seq_len(nrow(s))
    lo <- s$diff - 2 * s$se; hi <- s$diff + 2 * s$se
    plot(s$diff, y, xlim = range(c(lo, hi, 0)), yaxt = "n", pch = 19,
         xlab = "", ylab = "", main = paste0(key[[k]], " -- ", cl), cex.main = 0.9)
    segments(lo, y, hi, y)
    abline(v = 0, lty = 2, col = "grey40")
    axis(2, at = y, labels = as.character(s$config))
  }
  mtext("Paired difference to the default prior (+- 2 SE)", side = 1, outer = TRUE)
  par(op)
}
if (interactive()) plot_diffs()
pdf(file.path(fig_dir, "prior_sensitivity.pdf"), width = 9, height = 13)
plot_diffs()
dev.off()

# ── LaTeX table: key metrics per (cell, config) ───────────────────────────────
# Bold: the default row.  Config labels in the paper's notation.
config_tex <- c(
  "default"         = "Default",
  "k_tau=0.5"       = "$k_\\tau = 0.5$",  "k_tau=0.75" = "$k_\\tau = 0.75$",
  "k_tau=1.5"       = "$k_\\tau = 1.5$",  "k_tau=2"    = "$k_\\tau = 2$",
  "beta_tau=2"      = "$\\beta_\\tau = 2$", "beta_tau=4" = "$\\beta_\\tau = 4$",
  "k_d=0.5"         = "$k_d = 0.5$",      "k_d=2"      = "$k_d = 2$",
  "k_c=0.5"         = "$k_c = 0.5$",      "k_c=2"      = "$k_c = 2$",
  "c_tree=(0.5,2)"  = "$(\\alpha_c, \\beta_c) = (0.5, 2)$",
  "c_tree=(0.95,2)" = "$(\\alpha_c, \\beta_c) = (0.95, 2)$",
  "k_m0=0.5"        = "$k_{m_0} = 0.5$",  "k_m0=2"     = "$k_{m_0} = 2$",
  "atom=0.25"       = "$\\sigma_\\theta = 0.25$", "atom=1" = "$\\sigma_\\theta = 1$",
  "tau_flexible"    = "$\\tau$ with the baseline prior",
  "c_flexible"      = "$c$ with the baseline prior")
num <- function(x, d = 3) {
  if (length(x) == 0 || is.na(x)) return("---")
  s <- formatC(x, format = "f", digits = d)
  if (x < 0) paste0("$", s, "$") else s
}
tex_cells <- c(uniform       = "Uniform confounding, $\\lambda_u = 1$",
               heterogeneous = "Covariate-dependent confounding, $\\lambda_u = 2$")
tex_cols  <- c("cate_all_rmse", "cate_all_bias", "cate_all_cover",
               "cate_all_width", "c_rmse", "d_rmse", "ate_rmse", "ate_cover")
rows <- character(0)
for (cl in cells) {
  rows <- c(rows, sprintf("    \\multicolumn{9}{l}{\\emph{%s}}\\\\", tex_cells[[cl]]))
  m <- means[means$cell == cl, ]
  for (i in seq_len(nrow(m))) {
    cfg <- as.character(m$config[i])
    lab <- if (cfg %in% names(config_tex)) config_tex[[cfg]] else cfg
    rows <- c(rows, sprintf("    \\quad %s & %s \\\\", lab,
                            paste(vapply(tex_cols, function(k) num(m[[k]][i]),
                                         ""), collapse = " & ")))
  }
  if (cl != tail(cells, 1)) rows <- c(rows, "    \\addlinespace")
}
writeLines(c(
  "% Generated by simulations/revision_v3/exp5_prior_sensitivity/analyse_prior_v1.R -- do not edit.",
  "\\begin{table}[!htbp]",
  "  \\centering",
  "  \\color{revisioncol}",
  "  \\caption{Sensitivity of the Bayesian fusion forest to its prior settings,",
  "    changed one at a time around the default and fitted to the same datasets.",
  "    CATE metrics over the combined population; root mean squared error of the",
  "    confounding function $c(x)$ and the deviation $d(x)$ over the real-world",
  "    rows; root mean squared error and $95\\%$ coverage of the average treatment",
  sprintf("    effect (ATE). Averages over $%d$ replications.}", max(means$n)),
  "  \\label{tab:sim-prior}",
  "  \\small",
  "  \\begin{tabular}{@{}l rrrr rr rr@{}}",
  "    \\toprule",
  "    & \\multicolumn{4}{c}{CATE} & $c(x)$ & $d(x)$ & \\multicolumn{2}{c}{ATE} \\\\",
  "    \\cmidrule(lr){2-5} \\cmidrule(lr){6-6} \\cmidrule(lr){7-7} \\cmidrule(l){8-9}",
  "    Prior & RMSE & Bias & Coverage & Width & RMSE & RMSE & RMSE & Coverage \\\\",
  "    \\midrule",
  rows,
  "    \\bottomrule",
  "  \\end{tabular}",
  "\\end{table}"), file.path(tab_dir, "prior_table.tex"))

cat("\nWrote", file.path(tab_dir, "prior_means.csv"), ",",
    file.path(tab_dir, "prior_paired_diff.csv"), ",",
    file.path(tab_dir, "prior_table.tex"), "and",
    file.path(fig_dir, "prior_sensitivity.pdf"), "\n")
