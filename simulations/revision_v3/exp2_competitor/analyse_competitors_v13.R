# ══════════════════════════════════════════════════════════════════════════════
# analyse_competitors_v13.R -- local analysis of experiment 2 (competitors on
# the right-censored design).
#
# Reads every results/sim_surv_v13_*_output.rds produced on the HPC by
# sim_surv_v13_{bff,ye,deepaft,xgbaft,bjelm,bjtree}.R (missing files are
# skipped) and writes:
#   results/tables/
#     results_summary.csv       mean metrics per (method, population, cell)
#     competitors_table.csv     every method at lambda_d = lambda_u = 1
#     competitors_table.tex     Web Table sim-competitors-full
#     ye_table.tex              Web Table sim-ye (fusion vs Ye et al., grid)
#     deepaft_table.csv         fusion vs DNN, mean and 2.5-97.5% range, grid
#     deepaft_table.tex         Web Table sim-deepaft
#   results/figures/
#     sim_competitors_main.pdf  the four metrics at lambda_d = lambda_u = 1
#     sim_grid_<metric>.pdf     fusion, Ye et al. and DNN over the 3x3 grid
# and prints the summaries plus the numbers quoted in Web Appendix D.1.  None
# of these figures is in the manuscript.  The .tex files are pasted into
# biometrics_sm.tex by hand.  Base R only.  Run from the repository root:
#   Rscript simulations/revision_v3/exp2_competitor/analyse_competitors_v13.R
# ══════════════════════════════════════════════════════════════════════════════

exp_dir <- "simulations/revision_v3/exp2_competitor"
res_dir <- file.path(exp_dir, "results")
tab_dir <- file.path(res_dir, "tables")
fig_dir <- file.path(res_dir, "figures")
dir.create(tab_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)

files <- sprintf("sim_surv_v13_%s_output.rds",
                 c("bff", "ye", "deepaft", "xgbaft", "bjelm", "bjtree"))
present <- file.exists(file.path(res_dir, files))
if (!any(present)) stop("No result files in ", res_dir)
if (!all(present))
  warning("Missing: ", paste(files[!present], collapse = ", "))
files <- files[present]

keep <- c("method", "population", "rmse", "bias", "coverage", "width",
          "postvar", "cens_rct", "cens_rwd", "lambda_d", "lambda_u", "Iter")
res <- do.call(rbind, lapply(files, function(f)
  readRDS(file.path(res_dir, f))[, keep]))

cat("Loaded:\n")
for (f in files) {
  d <- readRDS(file.path(res_dir, f))
  cat(sprintf("  %-34s M = %4d, %d cell(s), methods: %s\n", f, max(d$Iter),
              nrow(unique(d[c("lambda_d", "lambda_u")])),
              paste(unique(d$method), collapse = ", ")))
}

# ── Method order: fusion first, then the competitor families ─────────────────
families <- c(deepAFT = "DNN", xgbAFT = "XGBoost",
              bjelm = "BJ-ELM", bjtree = "BJ-trees")
learners <- c("RCT-S", "RCT-T", "Pooled-S", "Pooled-T")
meth_all <- c("Fusion", "Ye-Linear", "Ye-Quadratic",
              as.vector(t(outer(names(families), learners, paste, sep = "-"))))
meth_all <- c(meth_all[meth_all %in% res$method],
              setdiff(unique(res$method), meth_all))   # keep unexpected labels
res$method     <- factor(res$method, levels = meth_all)
res$population <- factor(res$population, levels = c("RCT", "RWD", "All"))

# ── Summary over the whole grid ───────────────────────────────────────────────
# Means over replications; bias_var is the across-replication variance of the
# integrated bias; n counts the replications with a finite RMSE (some
# competitor fits fail, so n can fall short of M).
by_vars  <- c("method", "population", "lambda_d", "lambda_u")
mean_na  <- function(x) mean(x, na.rm = TRUE)
summ <- aggregate(cbind(rmse, bias, coverage, width, postvar) ~
                    method + population + lambda_d + lambda_u,
                  data = res, FUN = mean_na, na.action = na.pass)
vtab <- aggregate(bias ~ method + population + lambda_d + lambda_u,
                  data = res, FUN = function(x) var(x, na.rm = TRUE),
                  na.action = na.pass)
names(vtab)[ncol(vtab)] <- "bias_var"
ntab <- aggregate(rmse ~ method + population + lambda_d + lambda_u,
                  data = res, FUN = function(x) sum(is.finite(x)),
                  na.action = na.pass)
names(ntab)[ncol(ntab)] <- "n"
summ <- merge(merge(summ, vtab), ntab)
summ$sd <- sqrt(summ$bias_var)
summ <- summ[order(summ$population, summ$lambda_d, summ$lambda_u,
                   summ$method), ]
write.csv(summ, file.path(tab_dir, "results_summary.csv"), row.names = FALSE)

allp <- summ[summ$population == "All", ]
cat("\n=== Mean CATE RMSE, population = All (rows: method, cols: lambda_d/lambda_u) ===\n")
print(round(tapply(allp$rmse, list(method = allp$method,
                                   cell = sprintf("d%g/u%g", allp$lambda_d,
                                                  allp$lambda_u)),
                   identity), 3))
cat(sprintf("\nMean censoring: RCT %.1f%%, RWD %.1f%%\n",
            100 * mean(res$cens_rct, na.rm = TRUE),
            100 * mean(res$cens_rwd, na.rm = TRUE)))

# ── LaTeX helpers ─────────────────────────────────────────────────────────────
num <- function(x, d = 3) {           # negatives in math mode, NA as an em-dash
  if (length(x) == 0 || is.na(x)) return("---")
  s <- formatC(x, format = "f", digits = d)
  if (x < 0) paste0("$", s, "$") else s
}
bold <- function(s) {                 # bold a formatted number, math-safe
  if (s == "---") return(s)
  paste0("$\\mathbf{", gsub("\\$", "", s), "}$")
}

# ── Web Table sim-competitors-full: every method at lambda_d = lambda_u = 1 ──
s11 <- summ[summ$population == "All" & summ$lambda_d == 1 & summ$lambda_u == 1, ]
write.csv(summ[summ$lambda_d == 1 & summ$lambda_u == 1, ],
          file.path(tab_dir, "competitors_table.csv"), row.names = FALSE)
cat("\n=== CATE metrics at lambda_d = lambda_u = 1, population = All ===\n")
print(s11[, c("method", "n", "rmse", "bias", "sd", "coverage", "width",
              "postvar")], row.names = FALSE, digits = 3)

get11 <- function(m, col) {
  v <- s11[[col]][s11$method == m]
  if (length(v)) v else NA_real_
}
best_rmse <- min(s11$rmse, na.rm = TRUE)
best_bias <- min(abs(s11$bias), na.rm = TRUE)
cells11 <- function(m) {
  r <- get11(m, "rmse"); b <- get11(m, "bias")
  rt <- num(r); bt <- num(b)
  if (!is.na(r) && isTRUE(all.equal(r, best_rmse)))      rt <- bold(rt)
  if (!is.na(b) && isTRUE(all.equal(abs(b), best_bias))) bt <- bold(bt)
  paste(c(rt, bt, num(get11(m, "coverage")), num(get11(m, "width"), 2)),
        collapse = " & ")
}
rows <- c(sprintf("    Bayesian fusion forest & Both & --- & %s \\\\", cells11("Fusion")),
          sprintf("    Ye et al.\\ (linear)    & Both & --- & %s \\\\", cells11("Ye-Linear")),
          sprintf("    Ye et al.\\ (quadratic) & Both & --- & %s \\\\", cells11("Ye-Quadratic")),
          "    \\midrule")
for (f in names(families)) {
  first <- TRUE
  for (src in c("RCT", "Pooled")) for (l in c("S", "T")) {
    rows <- c(rows, sprintf("    %-22s & %-5s & %s & %s \\\\",
                            if (first) families[[f]] else "",
                            if (l == "S") (if (src == "RCT") "Trial" else "Pool") else "",
                            l, cells11(paste(f, src, l, sep = "-"))))
    first <- FALSE
  }
  if (f != tail(names(families), 1)) rows <- c(rows, "    \\addlinespace")
}
M11 <- max(s11$n, na.rm = TRUE)
writeLines(c(
  "% Generated by simulations/revision_v3/exp2_competitor/analyse_competitors_v13.R -- do not edit.",
  "\\begin{table}[!tb]",
  "  \\centering",
  "  \\color{revisioncol}",
  "  \\caption{Comparison with the integrative estimator of \\citet{Ye2025} and the",
  "    machine-learning alternatives at $\\lambda_d = \\lambda_u = 1$: a deep neural",
  "    network (DNN), gradient boosting under an accelerated failure time loss",
  "    (XGBoost), and Buckley--James boosting over extreme learning machines",
  "    (BJ-ELM) and regression trees (BJ-trees), each fitted as an S- and a",
  "    T-learner on the trial alone and on a naive pool of both sources. The",
  "    estimator of \\citet{Ye2025} uses both sources, with a linear or a quadratic",
  "    basis (Web Appendix~\\ref{sm:sim-ye}). Both sources are right-censored at",
  "    circa $35\\%$. Coverage and width refer to $95\\%$ intervals: posterior for",
  "    the Bayesian fusion forest and percentile bootstrap over $100$ resamples",
  sprintf("    otherwise. Averages over $%d$ replications. The lowest root mean squared", M11),
  "    error and the bias closest to zero are in bold. The BJ-trees S-learners",
  "    never select the treatment indicator, so their estimated effect and",
  "    interval width are zero.",
  sprintf("    BJ-trees failed in $%d$ (trial) and $%d$ (pool) replications.}",
          M11 - get11("bjtree-RCT-S", "n"), M11 - get11("bjtree-Pooled-S", "n")),
  "  \\label{tab:sim-competitors-full}",
  "  \\small",
  "  \\begin{tabular}{@{}lll rrrr@{}}",
  "    \\toprule",
  "    Method & Source & Learner & RMSE & Bias & Coverage & Width \\\\",
  "    \\midrule",
  rows,
  "    \\bottomrule",
  "  \\end{tabular}",
  "\\end{table}"), file.path(tab_dir, "competitors_table.tex"))

# ── Web Table sim-ye: fusion vs Ye et al. across the grid ─────────────────────
ye_meth <- c(Fusion = "Fusion forest", `Ye-Linear` = "Ye et al., linear",
             `Ye-Quadratic` = "Ye et al., quadratic")
if (all(names(ye_meth) %in% allp$method)) {
  grid <- unique(allp[allp$method == "Ye-Linear", c("lambda_d", "lambda_u")])
  grid <- grid[order(grid$lambda_d, grid$lambda_u), ]
  rows <- character(0)
  for (i in seq_len(nrow(grid))) {
    sc <- allp[allp$lambda_d == grid$lambda_d[i] &
                 allp$lambda_u == grid$lambda_u[i] &
                 allp$method %in% names(ye_meth), ]
    best <- min(sc$rmse)
    for (k in seq_along(ye_meth)) {
      r  <- sc[sc$method == names(ye_meth)[k], ]
      rt <- num(r$rmse)
      if (isTRUE(all.equal(r$rmse, best))) rt <- bold(rt)
      lead <- if (k == 1) sprintf("%g & %g", grid$lambda_d[i], grid$lambda_u[i]) else " & "
      rows <- c(rows, sprintf("    %s & %-22s & %s & %s & %s & %s \\\\", lead,
                              ye_meth[[k]], rt, num(r$bias), num(r$coverage),
                              num(r$width, 2)))
    }
    if (i < nrow(grid)) rows <- c(rows, "    \\addlinespace")
  }
  Mye <- max(allp$n[allp$method %in% names(ye_meth)])
  writeLines(c(
    "% Generated by simulations/revision_v3/exp2_competitor/analyse_competitors_v13.R -- do not edit.",
    "\\begin{table}[!tb]",
    "  \\centering",
    "  \\color{revisioncol}",
    "  \\caption{The Bayesian fusion forest and the integrative estimator of",
    "    \\citet{Ye2025}, with a linear and a quadratic basis, across the",
    "    $(\\lambda_d, \\lambda_u)$ grid of the right-censored design. CATE metrics",
    sprintf("    over the combined population, averaged over $%d$ replications. Coverage", Mye),
    "    and width refer to $95\\%$ intervals: posterior for the fusion forest and",
    "    percentile bootstrap over $100$ resamples for the estimator of",
    "    \\citet{Ye2025}. The lowest root mean squared error per cell is in bold.}",
    "  \\label{tab:sim-ye}",
    "  \\small",
    "  \\begin{tabular}{@{}cc l rrrr@{}}",
    "    \\toprule",
    "    $\\lambda_d$ & $\\lambda_u$ & Method & RMSE & Bias & Coverage & Width \\\\",
    "    \\midrule",
    rows,
    "    \\bottomrule",
    "  \\end{tabular}",
    "\\end{table}"), file.path(tab_dir, "ye_table.tex"))
}

# ── Web Table sim-deepaft: fusion vs DNN across the grid ──────────────────────
# Each entry: mean over replications with the 2.5%-97.5% range across
# replications.  Non-finite values and rare divergent blow-ups (a diverged
# deepAFT bootstrap refit can give a posterior variance ~1e6) are dropped by a
# Q3 + 30 IQR rule, extreme enough to leave genuine heavy tails intact.
dnn_meth <- c("Fusion", paste("deepAFT", learners, sep = "-"))
dnn_tex  <- c("BFF", "DNN (RCT, S)", "DNN (RCT, T)", "DNN (pool, S)",
              "DNN (pool, T)")
dnn_metrics <- c(rmse = "RMSE", bias = "Bias", coverage = "Coverage",
                 postvar = "Posterior variance")
trim_div <- function(v) {
  v <- v[is.finite(v)]
  if (length(v) < 4) return(v)
  qs <- quantile(v, c(0.25, 0.75), names = FALSE); iqr <- qs[2] - qs[1]
  if (iqr <= 0) return(v)
  v[v >= qs[1] - 30 * iqr & v <= qs[2] + 30 * iqr]
}
a_all <- res[res$population == "All", ]
if (all(dnn_meth %in% a_all$method)) {
  grid <- unique(a_all[a_all$method == "deepAFT-RCT-S", c("lambda_d", "lambda_u")])
  grid <- grid[order(grid$lambda_u, grid$lambda_d), ]
  long <- do.call(rbind, lapply(names(dnn_metrics), function(mt)
    do.call(rbind, lapply(seq_len(nrow(grid)), function(i)
      do.call(rbind, lapply(dnn_meth, function(me) {
        v <- trim_div(a_all[a_all$lambda_d == grid$lambda_d[i] &
                              a_all$lambda_u == grid$lambda_u[i] &
                              a_all$method == me, mt])
        q <- quantile(v, c(0.025, 0.975), names = FALSE)
        data.frame(metric = mt, lambda_d = grid$lambda_d[i],
                   lambda_u = grid$lambda_u[i], method = me,
                   mean = mean(v), sd = sd(v), mcse = sd(v) / sqrt(length(v)),
                   q2.5 = q[1], q97.5 = q[2], n = length(v))
      }))))))
  write.csv(long, file.path(tab_dir, "deepaft_table.csv"), row.names = FALSE)

  ci2 <- function(m, lo, hi) sprintf("%s [%s;%s]", num(m, 2), num(lo, 2), num(hi, 2))
  rows <- character(0)
  for (mt in names(dnn_metrics)) {
    rows <- c(rows, sprintf("    \\multicolumn{7}{l}{\\emph{%s}}\\\\",
                            dnn_metrics[[mt]]))
    for (i in seq_len(nrow(grid))) {
      ent <- vapply(dnn_meth, function(me) {
        s <- long[long$metric == mt & long$lambda_d == grid$lambda_d[i] &
                    long$lambda_u == grid$lambda_u[i] & long$method == me, ]
        ci2(s$mean, s$q2.5, s$q97.5)
      }, character(1))
      rows <- c(rows, sprintf("    %g & %g & %s \\\\", grid$lambda_d[i],
                              grid$lambda_u[i], paste(ent, collapse = " & ")))
    }
    if (mt != tail(names(dnn_metrics), 1)) rows <- c(rows, "    \\midrule")
  }
  writeLines(c(
    "% Generated by simulations/revision_v3/exp2_competitor/analyse_competitors_v13.R -- do not edit.",
    "\\begin{sidewaystable}",
    "  \\centering",
    "  \\color{revisioncol}",
    "  \\caption{CATE metrics for the Bayesian fusion forest (BFF) and the four",
    "    deepAFT competitors across the $(\\lambda_d, \\lambda_u)$ grid. Each entry",
    "    is the mean over simulation replicates, with the 2.5--97.5\\% range in",
    "    brackets. R and P denote the RCT-only and naive-pool fits; S and T denote",
    "    the S-learner and T-learner.}",
    "  \\label{tab:sim-deepaft}",
    "  \\scriptsize",
    "  \\setlength{\\tabcolsep}{4pt}",
    "  \\begin{tabular}{cc ccccc}",
    "    \\toprule",
    paste0("    $\\lambda_d$ & $\\lambda_u$ & ", paste(dnn_tex, collapse = " & "),
           " \\\\"),
    "    \\midrule",
    rows,
    "    \\bottomrule",
    "  \\end{tabular}",
    "\\end{sidewaystable}"), file.path(tab_dir, "deepaft_table.tex"))
}

# ── Numbers quoted in the text of Web Appendix D.1 ────────────────────────────
cat("\n=== Text checks at lambda_d = lambda_u = 1 (All) ===\n")
fu <- s11[s11$method == "Fusion", ]
alt <- s11[s11$method != "Fusion", ]
cat(sprintf("Fusion: RMSE %.3f, bias %.3f, coverage %.3f, width %.2f\n",
            fu$rmse, fu$bias, fu$coverage, fu$width))
wide <- alt[alt$width >= fu$width, ]
narr <- alt[alt$width <  fu$width, ]
if (nrow(wide))
  cat(sprintf("Wider than fusion (%s): coverage %.2f-%.2f, width ratio %.2f-%.2f\n",
              paste(wide$method, collapse = ", "), min(wide$coverage),
              max(wide$coverage), min(wide$width / fu$width),
              max(wide$width / fu$width)))
if (nrow(narr))
  cat(sprintf("Narrower than fusion (%s): coverage %.2f-%.2f\n",
              paste(narr$method, collapse = ", "), min(narr$coverage),
              max(narr$coverage)))
cat("Lowest-RMSE alternative:", as.character(alt$method[which.min(alt$rmse)]),
    sprintf("(%.3f)\n", min(alt$rmse)))

cat("\n=== Text checks across the grid (All) ===\n")
rng <- function(m, col) {
  v <- allp[[col]][allp$method == m]
  if (length(v)) sprintf("%.3f to %.3f", min(v), max(v)) else "--"
}
for (m in intersect(c("Fusion", "Ye-Linear", "Ye-Quadratic"), allp$method))
  cat(sprintf("%-13s RMSE %s | bias %s | coverage %s\n", m, rng(m, "rmse"),
              rng(m, "bias"), rng(m, "coverage")))
win <- vapply(split(allp, interaction(allp$lambda_d, allp$lambda_u, drop = TRUE)),
              function(s) as.character(s$method[which.min(s$rmse)]), "")
cat("Lowest RMSE per cell:", paste(names(win), win, sep = ": ", collapse = "; "),
    "\n")

# ── Figures (results/figures; not in the manuscript) ──────────────────────────
# Okabe-Ito palette: fusion keeps its orange; the DNN variants a cool (trial)
# and warm (pool) pair, light for the S- and dark for the T-learner; Ye et al.
# grey (linear) and black (quadratic).
fig_meth <- c("Fusion", "Ye-Linear", "Ye-Quadratic",
              paste("deepAFT", learners, sep = "-"))
fig_labs <- c("Bayesian fusion forest", "Ye et al. (lin.)", "Ye et al. (quad.)",
              "DNN (RCT, S)", "DNN (RCT, T)", "DNN (pool, S)", "DNN (pool, T)")
fig_cols <- c("#E69F00", "#999999", "#000000",
              "#56B4E9", "#0072B2", "#CC79A7", "#D55E00")
ok <- fig_meth %in% a_all$method
fig_meth <- fig_meth[ok]; fig_labs <- fig_labs[ok]; fig_cols <- fig_cols[ok]
fa <- a_all[a_all$method %in% fig_meth, ]
fa$method <- factor(fa$method, levels = fig_meth)

fig_metrics <- c(rmse = "RMSE", bias = "Bias", coverage = "Coverage",
                 postvar = "Posterior var.")
ref_lines   <- c(rmse = NA, bias = 0, coverage = 0.95, postvar = NA)

panel <- function(dd, metric, ylim = NULL, lab = fig_metrics[[metric]]) {
  if (is.null(ylim)) {
    st   <- boxplot(dd[[metric]] ~ dd$method, plot = FALSE)$stats
    ylim <- range(c(as.vector(st), ref_lines[[metric]]), na.rm = TRUE)
    ylim <- ylim + c(-1, 1) * 0.04 * diff(ylim)
  }
  boxplot(dd[[metric]] ~ dd$method, col = fig_cols, xaxt = "n", outline = FALSE,
          ylim = ylim, ylab = lab, xlab = "")
  if (!is.na(ref_lines[[metric]]))
    abline(h = ref_lines[[metric]], lty = 2, col = "grey40")
}
save_fig <- function(render, file, width, height) {
  if (interactive()) render()
  pdf(file, width = width, height = height, family = "Times")
  render()
  dev.off()
  cat("wrote", file, "\n")
}

# The four metrics in one row at lambda_d = lambda_u = 1.
d11 <- fa[fa$lambda_d == 1 & fa$lambda_u == 1, ]
save_fig(function() {
  op <- par(mfrow = c(1, 4), family = "serif", oma = c(6, 0, 0, 0),
            mar = c(3, 10.5, 2.5, 3.5), mgp = c(7.0, 1.5, 0),
            cex.lab = 4.3, cex.axis = 3.2, las = 1)
  for (m in names(fig_metrics)) panel(d11, m)
  par(fig = c(0, 1, 0, 1), oma = c(0, 0, 0, 0), mar = c(0, 0, 0, 0),
      family = "serif", new = TRUE)
  plot.new()
  legend("bottom", legend = fig_labs, fill = fig_cols, horiz = TRUE, bty = "n",
         xpd = TRUE, cex = 2.6)
  par(op)
}, file.path(fig_dir, "sim_competitors_main.pdf"), width = 28, height = 7.5)

# One figure per metric over the 3x3 grid, shared y-limits across the cells.
lds <- sort(unique(fa$lambda_d)); lus <- sort(unique(fa$lambda_u))
for (m in names(fig_metrics)) {
  st   <- boxplot(fa[[m]] ~ fa$method + interaction(fa$lambda_d, fa$lambda_u),
                  plot = FALSE)$stats
  ylim <- range(c(as.vector(st), ref_lines[[m]]), na.rm = TRUE)
  ylim <- ylim + c(-1, 1) * 0.04 * diff(ylim)
  save_fig(function() {
    op <- par(mfrow = c(length(lds), length(lus)), family = "serif",
              oma = c(7, 1, 3, 1), mar = c(1.5, 4.2, 2.2, 1), cex.axis = 0.9)
    for (ld in lds) for (lu in lus) {
      panel(fa[fa$lambda_d == ld & fa$lambda_u == lu, ], m, ylim = ylim)
      title(main = bquote(lambda[d] == .(ld) ~ "," ~ lambda[u] == .(lu)))
    }
    mtext(sprintf("%s - population: pooled (All)", fig_metrics[[m]]),
          outer = TRUE, line = 1, cex = 1.3, font = 2)
    par(fig = c(0, 1, 0, 1), oma = c(0, 0, 0, 0), mar = c(0, 0, 0, 0),
        new = TRUE)
    plot.new()
    legend("bottom", legend = fig_labs, fill = fig_cols, ncol = 4, bty = "n",
           xpd = TRUE, cex = 1.0)
    par(op)
  }, file.path(fig_dir, sprintf("sim_grid_%s.pdf", m)), width = 12, height = 11)
}

cat("\nTables in", tab_dir, "\n")
