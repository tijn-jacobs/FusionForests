# ══════════════════════════════════════════════════════════════════════════════
# analyse_err_v3.R -- local analysis of experiment 4 (the nonparametric error
# distribution).
#
# Reads results/sim_err_v3_output.rds (one row per replication x prior x
# population x target, produced on the HPC by sim_err_v3.R) and writes:
#   results/tables/
#     summary.csv               mean metrics per (cell, prior, population, target)
#     w1.csv                    error-law recovery W1 per (cell, prior, source)
#     paired_contrasts.csv      source HDPM minus each other prior, paired on
#                               the same datasets (mean and MC SE)
#     sim_err_table.tex         Web Table sim-err
#   results/figures/
#     sim_err_metrics_cate.pdf     CATE metrics by cell
#     sim_err_metrics_outcome.pdf  full conditional-mean metrics by cell
#     sim_err_density.pdf          only with run_density = TRUE (below)
# and prints the summaries.  None of the figures is in the manuscript; the .tex
# file is pasted into biometrics_sm.tex by hand.  Base R only (the optional
# density figure also needs FusionForests).  Run from the repository root:
#   Rscript simulations/revision_v3/exp4_error_distribution/analyse_err_v3.R
#
# A cell is an error setting at a trial size; "5s" and "6s" are settings 5 and
# 6 with n_rct = 50.  All four priors are fitted to the same dataset in each
# replication, so differences between priors are paired by (cell, Iter).
# ══════════════════════════════════════════════════════════════════════════════

run_density <- FALSE   # TRUE refits the four priors on one dataset (minutes)

exp_dir <- "simulations/revision_v3/exp4_error_distribution"
res_dir <- file.path(exp_dir, "results")
tab_dir <- file.path(res_dir, "tables")
fig_dir <- file.path(res_dir, "figures")
dir.create(tab_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)

res <- readRDS(file.path(res_dir, "sim_err_v3_output.rds"))

prior_labs <- c(gaussian = "Gaussian", shared_dp = "Shared DP",
                source_dp_scale = "Source DP", source_hdp_scale = "Source HDPM")
prior_lvls <- names(prior_labs)
prior_cols <- c(gaussian = "grey55", shared_dp = "goldenrod1",
                source_dp_scale = "steelblue", source_hdp_scale = "olivedrab")
cell_lvls  <- c(1:6, "5s", "6s")
cell_labs  <- c(`1`  = "Setting 1 --- Gaussian, identical (null)",
                `2`  = "Setting 2 --- Gaussian, per-source scale gap",
                `3`  = "Setting 3 --- Skewed RWD, scale gap",
                `4`  = "Setting 4 --- Bimodal RWD, scale gap",
                `5`  = "Setting 5 --- Multimodal, shared atoms, weight + scale gap",
                `6`  = "Setting 6 --- No shared components",
                `5s` = "Setting 5, small trial ($n_1 = 50$)",
                `6s` = "Setting 6, small trial ($n_1 = 50$)")

res$cell <- factor(ifelse(res$n_rct == 50, paste0(res$setting, "s"),
                          res$setting), levels = cell_lvls)
res$method     <- factor(res$method, levels = prior_lvls)
res$population <- factor(res$population, levels = c("RCT", "RWD", "All"))
res$target     <- factor(res$target, levels = c("cate", "outcome"))
cat(sprintf("Loaded %d rows; cells: %s; %d reps per cell\n", nrow(res),
            paste(levels(droplevels(res$cell)), collapse = ", "), max(res$Iter)))

# ── Summary table ─────────────────────────────────────────────────────────────
# Two targets: the CATE tau(x) and the full conditional-mean outcome (sum of
# forests).  The outcome target does not cancel the shared baseline m0, so it
# is the more sensitive probe of the residual model.
grp <- cbind(rmse, bias, coverage, width, postvar) ~
  cell + method + population + target
mean_tbl <- aggregate(grp, data = res,
                      FUN = function(x) mean(x, na.rm = TRUE),
                      na.action = na.pass)
var_tbl  <- aggregate(bias ~ cell + method + population + target, data = res,
                      FUN = function(x) var(x, na.rm = TRUE),
                      na.action = na.pass)
names(var_tbl)[ncol(var_tbl)] <- "variance"
n_tbl    <- aggregate(rmse ~ cell + method + population + target, data = res,
                      FUN = function(x) sum(is.finite(x)), na.action = na.pass)
names(n_tbl)[ncol(n_tbl)] <- "n"
summary_tbl <- merge(merge(mean_tbl, var_tbl), n_tbl)
summary_tbl$sd <- sqrt(summary_tbl$variance)
summary_tbl <- summary_tbl[order(summary_tbl$target, summary_tbl$cell,
                                 summary_tbl$population, summary_tbl$method), ]
write.csv(summary_tbl, file.path(tab_dir, "summary.csv"), row.names = FALSE)

for (tg in c("cate", "outcome")) {
  cat(sprintf("\n=== %s metrics by cell, population = All ===\n",
              if (tg == "cate") "CATE" else "Outcome log(T)"))
  s <- summary_tbl[summary_tbl$target == tg & summary_tbl$population == "All", ]
  print(s[, c("cell", "method", "n", "rmse", "bias", "sd", "coverage",
              "width", "postvar")], row.names = FALSE, digits = 3)
}

# ── Error-law recovery (W1 to the true error CDF; lower is better) ───────────
# One value per (replication, prior); the rows of a fit repeat it, so take the
# CATE / All row.
fits <- res[res$target == "cate" & res$population == "All", ]
w1_tbl <- aggregate(cbind(w1_rct, w1_rwd) ~ cell + method, data = fits,
                    FUN = function(x) mean(x, na.rm = TRUE),
                    na.action = na.pass)
w1_tbl <- w1_tbl[order(w1_tbl$cell, w1_tbl$method), ]
write.csv(w1_tbl, file.path(tab_dir, "w1.csv"), row.names = FALSE)
cat("\n=== Error-law recovery (W1, mean over replications) ===\n")
print(w1_tbl, row.names = FALSE, digits = 3)

# ── Paired contrasts: source HDPM minus each other prior ──────────────────────
# The comparisons that answer the associate editor (comment 1) and referee 1
# (comment 7): HDPM against the independent per-source mixtures separates
# sharing from source-specific shapes; cells 5s (sharing should help) and 6 /
# 6s (it can only cost) are the key ones.  Negative = HDPM better.
outc <- res[res$target == "outcome" & res$population == "All",
            c("cell", "Iter", "method", "rmse")]
names(outc)[4] <- "rmse_outcome"
wide_in <- merge(fits[, c("cell", "Iter", "method", "rmse", "bias",
                          "w1_rct", "w1_rwd")],
                 outc, by = c("cell", "Iter", "method"))
names(wide_in)[names(wide_in) == "rmse"] <- "rmse_cate"
wide_in$abs_bias_cate <- abs(wide_in$bias)
contrast_vars <- c("rmse_cate", "abs_bias_cate", "rmse_outcome",
                   "w1_rct", "w1_rwd")
hdp <- wide_in[wide_in$method == "source_hdp_scale", ]
contrasts <- do.call(rbind, lapply(setdiff(prior_lvls, "source_hdp_scale"),
  function(ref) {
    m <- merge(hdp, wide_in[wide_in$method == ref, ], by = c("cell", "Iter"),
               suffixes = c("", ".ref"))
    do.call(rbind, lapply(contrast_vars, function(v) {
      dv <- m[[v]] - m[[paste0(v, ".ref")]]
      a  <- aggregate(dv, by = list(cell = m$cell), FUN = function(x)
        c(mean = mean(x, na.rm = TRUE),
          se = sd(x, na.rm = TRUE) / sqrt(sum(is.finite(x)))))
      data.frame(cell = a$cell, versus = ref, metric = v,
                 diff = a$x[, "mean"], se = a$x[, "se"])
    }))
  }))
contrasts$z <- contrasts$diff / contrasts$se
contrasts <- contrasts[order(contrasts$versus, contrasts$metric,
                             contrasts$cell), ]
write.csv(contrasts, file.path(tab_dir, "paired_contrasts.csv"),
          row.names = FALSE)
cat("\n=== Source HDPM minus Source DP (paired; negative = HDPM better) ===\n")
print(contrasts[contrasts$versus == "source_dp_scale",
                c("metric", "cell", "diff", "se", "z")],
      row.names = FALSE, digits = 3)

# Realised censoring fractions per cell (sanity check).
cens_tbl <- aggregate(cbind(cens_rct, ic_rwd, rc_rwd_tail) ~ cell, data = fits,
                      FUN = function(x) mean(x, na.rm = TRUE))
cat("\n=== Realised censoring fractions ===\n")
print(cens_tbl, row.names = FALSE, digits = 3)

# ── Web Table sim-err ─────────────────────────────────────────────────────────
# CATE metrics over the combined population and the per-source W1, per cell
# and prior.  Bold: lowest RMSE, bias closest to zero, lowest W1 per cell.
num <- function(x, d = 3) {
  if (length(x) == 0 || is.na(x)) return("---")
  s <- formatC(x, format = "f", digits = d)
  if (x < 0) paste0("$", s, "$") else s
}
bold_if <- function(s, cond) if (isTRUE(cond)) paste0("$\\mathbf{", gsub("\\$", "", s), "}$") else s
ca <- summary_tbl[summary_tbl$target == "cate" & summary_tbl$population == "All", ]
tb <- merge(ca, w1_tbl, by = c("cell", "method"))
rows <- character(0)
for (cl in intersect(cell_lvls, as.character(unique(tb$cell)))) {
  s <- tb[tb$cell == cl, ]
  s <- s[order(s$method), ]
  rows <- c(rows, sprintf("    \\multicolumn{7}{l}{\\emph{%s}}\\\\", cell_labs[[cl]]))
  for (i in seq_len(nrow(s))) {
    r <- s[i, ]
    rows <- c(rows, sprintf("    \\quad %-12s & %s & %s & %s & %s & %s & %s \\\\",
      prior_labs[[as.character(r$method)]],
      bold_if(num(r$rmse), r$rmse == min(s$rmse)),
      bold_if(num(r$bias), abs(r$bias) == min(abs(s$bias))),
      num(r$coverage), num(r$postvar),
      bold_if(num(r$w1_rct), r$w1_rct == min(s$w1_rct)),
      bold_if(num(r$w1_rwd), r$w1_rwd == min(s$w1_rwd))))
  }
  if (cl != tail(cell_lvls, 1)) rows <- c(rows, "    \\addlinespace")
}
writeLines(c(
  "% Generated by simulations/revision_v3/exp4_error_distribution/analyse_err_v3.R -- do not edit.",
  "\\begin{table}[!htbp]",
  "  \\centering",
  "  \\color{revisioncol}",
  "  \\caption{CATE metrics and error-law recovery by residual prior across the",
  "    six error settings and the two small-trial cells, over the combined",
  sprintf("    population and averaged over $%d$ replications. ``Gaussian'' is a", max(tb$n)),
  "    single normal residual; ``Shared DP'' pools one Dirichlet process mixture",
  "    across the sources; ``Source DP'' fits independent mixtures per source;",
  "    ``Source HDPM'' is the proposed source-specific hierarchical Dirichlet",
  "    process mixture. Coverage is the $95\\%$ credible-interval coverage and",
  "    ``Post.\\ var.'' the mean posterior variance of $\\hat\\tau(x)$. $W_1$ is the",
  "    Wasserstein-1 distance between the posterior-mean fitted and the true",
  "    error distribution of the trial (RCT) and the real-world (RWD) source, on",
  "    the standardized scale. Per cell, the lowest RMSE and $W_1$ and the bias",
  "    closest to zero are in bold.}",
  "  \\label{tab:sim-err}",
  "  \\small",
  "  \\begin{tabular}{@{}l rrrr rr@{}}",
  "    \\toprule",
  "    Residual prior & RMSE & Bias & Coverage & Post.\\ var. & $W_1$ RCT & $W_1$ RWD \\\\",
  "    \\midrule",
  rows,
  "    \\bottomrule",
  "  \\end{tabular}",
  "\\end{table}"), file.path(tab_dir, "sim_err_table.tex"))

# ── Figures (results/figures; not in the manuscript) ──────────────────────────
metrics     <- c("rmse", "bias", "coverage", "postvar")
metric_labs <- c(rmse = "RMSE", bias = "Bias", coverage = "Coverage",
                 width = "CI width", postvar = "Posterior variance")

# The four metrics by cell: boxes grouped by cell, the four priors side by
# side, one shared legend beneath.
plot_metrics_by_cell <- function(dat, pop = "All", target, file) {
  dat <- droplevels(dat[dat$population == pop & dat$target == target, ])
  cols <- prior_cols[prior_lvls]
  render <- function() {
    op <- par(mfrow = c(2, 2), family = "serif", oma = c(5, 0, 0, 0),
              mar = c(6.5, 7.5, 2, 2), mgp = c(4.3, 1.3, 0),
              cex.lab = 2.4, cex.axis = 2.0)
    nb <- length(prior_lvls); ng <- nlevels(dat$cell); gap <- 1.5
    at      <- as.vector(outer(seq_len(nb), (seq_len(ng) - 1) * (nb + gap), "+"))
    centers <- (seq_len(ng) - 1) * (nb + gap) + (nb + 1) / 2
    seps    <- centers[-ng] + diff(centers) / 2
    for (metric in metrics) {
      ylim_m <- range(boxplot(dat[[metric]] ~ dat$method + dat$cell,
                              plot = FALSE)$stats, na.rm = TRUE)
      if (metric == "coverage") ylim_m <- range(c(ylim_m, 0.95))
      if (metric == "bias")     ylim_m <- range(c(ylim_m, 0))
      boxplot(dat[[metric]] ~ dat$method + dat$cell, col = cols, xaxt = "n",
              at = at, xlim = range(at) + c(-0.5, 0.5), ylim = ylim_m,
              outline = FALSE, main = "", ylab = metric_labs[metric],
              xlab = "Error setting (s = small trial)", sep = " ")
      axis(1, at = centers, labels = levels(dat$cell))
      abline(v = seps, lty = 3, col = "grey")
      if (metric == "coverage") abline(h = 0.95, lty = 2)
      if (metric == "bias")     abline(h = 0,    lty = 2)
    }
    par(fig = c(0, 1, 0, 1), oma = c(0, 0, 0, 0), mar = c(0, 0, 0, 0),
        family = "serif", new = TRUE)
    plot.new()
    legend("bottom", legend = prior_labs[prior_lvls], fill = cols,
           horiz = TRUE, bty = "n", xpd = TRUE, cex = 2.4)
    par(op)
  }
  if (interactive()) render()
  pdf(file, width = 18, height = 10, family = "Times")
  render()
  dev.off()
  cat("wrote", file, "\n")
}
cat("\n")
plot_metrics_by_cell(res, target = "cate",
                     file = file.path(fig_dir, "sim_err_metrics_cate.pdf"))
plot_metrics_by_cell(res, target = "outcome",
                     file = file.path(fig_dir, "sim_err_metrics_outcome.pdf"))

# ── Optional: error-density recovery on one dataset (illustrative) ────────────
# Fits each prior once on a single dataset and overlays the posterior-mean
# fitted residual density per source on the true error density (both on the
# standardized residual scale; the truth is a large-sample KDE of
# eps / scale_sd).  Needs the DGP and fit functions of sim_err_v3.R: the
# definitions above its "Main (HPC)" marker are evaluated, the HPC run is not.
# Mixture fields per source group g (1 = RWD, 2 = RCT; one group for
# shared_dp): dp_mix_prop[[g]] (draws x K weights), dp_locations[[g]] (draws x
# K centred atoms, response scale), dp_sigma_g[[g]] (per-source sd, scale
# modes only).
if (run_density) {
  src  <- readLines(file.path(exp_dir, "sim_err_v3.R"))
  stop_at <- grep("^# ── Main \\(HPC\\)", src)[1] - 1L
  eval(parse(text = src[seq_len(stop_at)]), envir = globalenv())

  density_recovery <- function(setting = 5L, seed = 1L, file,
                               grid = seq(-4, 4, length.out = 400)) {
    set.seed(seed)
    cal <- calibrate(setting)
    d   <- make_data(150L, 350L, setting, cal$rate_rct, cal$visits)
    sc  <- d$scale_sd
    true_dens <- function(source) {
      e <- draw_eps(2e5, setting, source) / sc
      approxfun(density(e, n = 1024), rule = 2)
    }
    fd <- list(rct = true_dens("rct"), rwd = true_dens("rwd"))
    fitted_dens <- function(fit, source) {
      if (is.null(fit)) return(rep(NA_real_, length(grid)))
      sh <- fit$meta$sigma_hat                # fit$sigma is on the internal scale
      if (is.null(fit$dp_mix_prop))           # gaussian
        return(dnorm(grid, 0, mean(fit$sigma) * sh))
      g  <- if (length(fit$dp_mix_prop) == 1L) 1L else if (source == "rwd") 1L else 2L
      W  <- fit$dp_mix_prop[[g]]; L <- fit$dp_locations[[g]]
      sg <- if (!is.null(fit$dp_sigma_g)) fit$dp_sigma_g[[g]] else fit$sigma * sh
      dens <- numeric(length(grid))
      for (it in seq_len(nrow(W))) {
        M <- outer(L[it, ], grid, function(mu, w) dnorm(w, mu, sg[it]))
        dens <- dens + colSums(W[it, ] * M)
      }
      dens / nrow(W)
    }
    fits_d <- lapply(prior_lvls, function(ed)
      tryCatch(fit_fusion(d, N_post = 2000L, N_burn = 2000L, ed),
               error = function(e) NULL))
    names(fits_d) <- prior_lvls
    render <- function() {
      op <- par(mfrow = c(1, 2), family = "serif", oma = c(4, 0, 2, 0),
                mar = c(5, 5, 3, 1), mgp = c(3, 1, 0), cex.lab = 1.6,
                cex.axis = 1.4)
      for (source in c("rct", "rwd")) {
        plot(grid, fd[[source]](grid), type = "l", lwd = 4, col = "black",
             xlab = "standardized residual", ylab = "density",
             main = if (source == "rct") "RCT source" else "RWD source")
        for (ed in prior_lvls)
          lines(grid, fitted_dens(fits_d[[ed]], source), col = prior_cols[ed],
                lwd = 3)
      }
      par(fig = c(0, 1, 0, 1), oma = c(0, 0, 0, 0), mar = c(0, 0, 0, 0),
          family = "serif", new = TRUE)
      plot.new()
      legend("bottom", legend = c("Truth", prior_labs[prior_lvls]),
             col = c("black", prior_cols[prior_lvls]),
             lwd = c(4, rep(3, length(prior_lvls))), horiz = TRUE, bty = "n",
             xpd = TRUE, cex = 1.4)
      par(op)
    }
    if (interactive()) render()
    pdf(file, width = 14, height = 6, family = "Times")
    render()
    dev.off()
    cat("wrote", file, "\n")
  }
  density_recovery(setting = 5L, seed = 1L,
                   file = file.path(fig_dir, "sim_err_density.pdf"))
}

cat("Tables in", tab_dir, "\n")
