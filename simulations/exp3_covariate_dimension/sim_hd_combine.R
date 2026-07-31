# Combine the four high-dimensional sweep outputs (sim_hd_v1a..v1d) into one
# data set and produce the paper figures for the covariate-dimension experiment:
#   notes/general/figures/sim_hd_metrics_vs_p.pdf  (main text, Figure sim-highdim)
#   notes/general/figures/sim_hd_ratio_vs_p.pdf    (supplementary, Figure sim-hd-ratio)
#
# Each input is the per-replication `res` data frame saved by the corresponding
# sim_hd_v1*.R run; together v1a..v1d tile the full p grid (5 ... 500), disjointly.
# Base R only -- no package dependencies.  Run from the repository root.

# ── Inputs ────────────────────────────────────────────────────────────────────
# Point this at the downloaded .rds files (edit `result_dir` / `result_files`).
result_dir   <- "simulations/exp3_covariate_dimension"
result_files <- file.path(result_dir,
                          c("sim_hd_v1a_output.rds", "sim_hd_v1b_output.rds",
                            "sim_hd_v1c_output.rds", "sim_hd_v1d_output.rds"))

present <- file.exists(result_files)
if (!any(present))
  stop("No result files found. Put the sim_hd_v1*_output.rds files in '",
       result_dir, "' or edit `result_files`.")
if (!all(present))
  warning("Missing: ", paste(basename(result_files[!present]), collapse = ", "),
          " -- combining the ", sum(present), " present file(s).")

res <- do.call(rbind, lapply(result_files[present], readRDS))

# Guard against accidentally overlapping result files (e.g. an old slice that
# still includes p = 500): a replication is identified by (p, method, population,
# Iter), and the paired ratio below matches Fusion to RCT-only by Iter within p.
key <- with(res, paste(p, method, population, Iter, sep = "\r"))
if (any(duplicated(key))) {
  warning(sum(duplicated(key)), " duplicate (p, method, population, Iter) rows; ",
          "keeping the first of each -- check for overlapping result files.")
  res <- res[!duplicated(key), ]
}
cat(sprintf("Loaded %d file(s); %d rows; p = %s\n", sum(present), nrow(res),
            paste(sort(unique(res$p)), collapse = ", ")))

# ── Aggregate: mean + across-rep SE per metric, by (p, method, population) ────
metrics_all <- c("rmse", "bias", "coverage", "width", "postvar")
grp <- as.formula(paste0("cbind(", paste(metrics_all, collapse = ", "),
                         ") ~ p + method + population"))
mean_tbl <- aggregate(grp, data = res,
                      FUN = function(x) mean(x, na.rm = TRUE), na.action = na.pass)
# Across-replication standard error of each mean, for the 95% CI bands.
se_of_mean <- function(x) { x <- x[is.finite(x)]
  if (length(x) < 2L) NA_real_ else sd(x) / sqrt(length(x)) }
se_tbl <- aggregate(grp, data = res, FUN = se_of_mean, na.action = na.pass)
names(se_tbl)[match(metrics_all, names(se_tbl))] <- paste0(metrics_all, "_se")
var_tbl  <- aggregate(bias ~ p + method + population,
                      data = res, FUN = function(x) var(x, na.rm = TRUE), na.action = na.pass)
names(var_tbl)[ncol(var_tbl)] <- "variance"
summary_tbl <- merge(merge(mean_tbl, se_tbl), var_tbl)
summary_tbl$sd <- sqrt(summary_tbl$variance)

meth_levels <- c("Fusion", "Fusion-oracle", "RCT-only", "RWD-only")
summary_tbl$method     <- factor(summary_tbl$method, levels = meth_levels)
summary_tbl$population  <- factor(summary_tbl$population,
                                  levels = c("RCT", "RWD", "All"))
ord <- order(summary_tbl$p, summary_tbl$population, summary_tbl$method)

cat("\n=== CATE metrics vs p (mean over replications) ===\n")
print(summary_tbl[ord, c("p", "population", "method",
                         "rmse", "bias", "variance", "sd", "coverage", "width",
                         "postvar")],
      row.names = FALSE, digits = 3)

# ── Figures (written to the manuscript figures directory) ─────────────────────
fig_dir <- "notes/general/figures"   # write straight to the manuscript figures dir
dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)

meth        <- c("Fusion", "Fusion-oracle", "RCT-only", "RWD-only")
# Okabe-Ito colourblind-safe palette, matching the other figures: Fusion orange,
# trial green, real-world vermillion. NOTE this swaps the previous assignment --
# Fusion was green here and gold elsewhere, RCT-only the other way round. The
# oracle is a reference floor rather than an estimator, so it takes black.
#   orange #E69F00 | blue #0072B2 | reddish purple #CC79A7 | vermillion #D55E00
cols        <- c("#E69F00", "#0072B2", "#CC79A7", "#D55E00")
metrics     <- c("rmse", "bias", "coverage", "postvar")
metric_labs <- c(rmse = "RMSE", bias = "Bias", coverage = "Coverage",
                 width = "CI width", postvar = "Posterior variance")

# Each metric vs p (log-x), one solid line per method, for a chosen population.
# Mean curves are smoothed by a least-squares polynomial of degree `sm_degree`
# in log(p) -- a trend fit that does NOT interpolate every point.  The shaded
# band is the 95% Monte Carlo CI of the mean across replications (mean +/- z*SE),
# smoothed the same way.  Small markers show the observed mean at each p.
plot_vs_p <- function(stbl, pop = "All", file = NULL, z = 1.96, sm_degree = 3) {
  dat <- stbl[stbl$population == pop, ]
  pv  <- sort(unique(dat$p))
  xticks <- c(5, 50, 500)                      # fixed x-axis ticks
  xlim_p <- range(c(xticks, pv))
  smooth_xy <- function(x, y, nout = 200) {
    ok <- is.finite(x) & is.finite(y); x <- x[ok]; y <- y[ok]
    o <- order(x); x <- x[o]; y <- y[o]
    nu <- length(unique(x))
    if (nu < 2) return(list(x = x, y = y))
    lx  <- log(x)
    deg <- min(sm_degree, nu - 1)
    fit <- lm(y ~ poly(lx, deg, raw = TRUE))
    gx  <- seq(min(lx), max(lx), length.out = nout)
    list(x = exp(gx), y = as.numeric(predict(fit, data.frame(lx = gx))))
  }
  render <- function() {
    op <- par(mfrow = c(2, 2), family = "serif",
              oma = c(5, 0, 2, 0), mar = c(6.5, 8.8, 2, 2),
              mgp = c(5.6, 1.3, 0), cex.lab = 2.4, cex.axis = 2.0,
              las = 1)                   # upright (horizontal) y-axis numbers
    for (metric in metrics) {
      se_col <- paste0(metric, "_se")
      lo_all <- dat[[metric]] - z * dat[[se_col]]
      hi_all <- dat[[metric]] + z * dat[[se_col]]
      yl <-
        if (metric == "rmse")
          c(0.4,  max(c(hi_all, dat$rmse), na.rm = TRUE))
        else if (metric == "bias")     c(-0.35, 0.35)
        else if (metric == "coverage")
          c(0.80, max(c(hi_all, dat$coverage, 0.95), na.rm = TRUE))
        else if (metric == "postvar")
          c(0,    max(c(hi_all, dat$postvar), na.rm = TRUE))
        else range(c(dat[[metric]], lo_all, hi_all), na.rm = TRUE)
      plot(NA, xlim = xlim_p, ylim = yl, log = "x", xaxt = "n",
           xlab = expression(italic(p)), ylab = metric_labs[metric])
      axis(1, at = xticks, labels = xticks)
      for (mi in seq_along(meth)) {
        dm <- dat[dat$method == meth[mi], ]; dm <- dm[order(dm$p), ]
        lo <- dm[[metric]] - z * dm[[se_col]]
        hi <- dm[[metric]] + z * dm[[se_col]]
        if (any(is.finite(c(lo, hi)))) {           # 95% CI band
          sl <- smooth_xy(dm$p, lo); sh <- smooth_xy(dm$p, hi)
          polygon(c(sl$x, rev(sh$x)), c(sl$y, rev(sh$y)),
                  col = adjustcolor(cols[mi], alpha.f = 0.18), border = NA)
        }
        sm <- smooth_xy(dm$p, dm[[metric]])        # smoothed mean (solid)
        lines(sm$x, sm$y, col = cols[mi], lty = 1, lwd = 4)
        points(dm$p, dm[[metric]], col = cols[mi], pch = 16, cex = 1.1)
      }
      if (metric == "coverage") abline(h = 0.95, lty = 3, col = "grey40")
      if (metric == "bias")     abline(h = 0,    lty = 3, col = "grey40")
    }
    par(fig = c(0, 1, 0, 1), oma = c(0, 0, 0, 0), mar = c(0, 0, 0, 0),
        family = "serif", new = TRUE)
    plot.new()
    legend("bottom", legend = meth, col = cols, lty = 1, pch = 16,
           lwd = 4, pt.cex = 1.3, horiz = TRUE, bty = "n", xpd = TRUE, cex = 2.0)
    par(op)
  }
  render()
  if (!is.null(file)) {
    pdf(file, width = 18, height = 10, family = "Times")
    render(); dev.off()
  }
}

# Efficiency of Fusion relative to the RCT-only baseline (ratio < 1 favours
# Fusion), with the active-only oracle overlaid.  The band is the 95% MC CI of
# the mean per-replication ratio, matched to RCT-only by Iter within each p.
ratio_vs_p <- function(stbl, raw, pop = "All", file = NULL, z = 1.96,
                       sm_degree = 3) {
  pv <- sort(unique(stbl$p[stbl$population == pop]))
  xticks <- c(5, 50, 500); xlim_p <- range(c(xticks, pv))
  smooth_xy <- function(x, y, nout = 200) {
    ok <- is.finite(x) & is.finite(y); x <- x[ok]; y <- y[ok]
    o <- order(x); x <- x[o]; y <- y[o]; nu <- length(unique(x))
    if (nu < 2) return(list(x = x, y = y))
    lx <- log(x); deg <- min(sm_degree, nu - 1)
    fit <- lm(y ~ poly(lx, deg, raw = TRUE))
    gx  <- seq(min(lx), max(lx), length.out = nout)
    list(x = exp(gx), y = as.numeric(predict(fit, data.frame(lx = gx))))
  }
  ratio_summary <- function(metric, num_method) {
    d <- raw[raw$population == pop, ]
    do.call(rbind, lapply(pv, function(p) {
      dp <- d[d$p == p, ]
      m  <- merge(dp[dp$method == num_method, c("Iter", metric)],
                  dp[dp$method == "RCT-only", c("Iter", metric)],
                  by = "Iter", suffixes = c(".n", ".d"))
      r  <- m[[paste0(metric, ".n")]] / m[[paste0(metric, ".d")]]
      r  <- r[is.finite(r)]
      data.frame(p = p, mean = if (length(r)) mean(r) else NA_real_,
                 se = if (length(r) < 2) NA_real_ else sd(r) / sqrt(length(r)))
    }))
  }
  series <- list(
    list(num = "Fusion",        col = "#E69F00", lab = "Fusion / RCT-only"),
    list(num = "Fusion-oracle", col = "#0072B2", lab = "Fusion-oracle / RCT-only"))
  panels <- c(rmse    = "RMSE ratio (vs RCT-only)",
              postvar = "Posterior-variance ratio (vs RCT-only)")
  summ <- list()
  for (m in names(panels)) for (s in series)
    summ[[paste(m, s$num)]] <- ratio_summary(m, s$num)
  render <- function() {
    op <- par(mfrow = c(1, 2), family = "serif",
              oma = c(5, 0, 2, 0), mar = c(6.5, 9.8, 2, 2),
              mgp = c(6.0, 1.3, 0), cex.lab = 2.2, cex.axis = 1.9,
              las = 1)                   # upright (horizontal) y-axis numbers
    for (m in names(panels)) {
      ss <- lapply(series, function(s) summ[[paste(m, s$num)]])
      yl <- range(c(1, unlist(lapply(ss, function(d)
              c(d$mean - z * d$se, d$mean + z * d$se, d$mean)))), na.rm = TRUE)
      plot(NA, xlim = xlim_p, ylim = yl, log = "x", xaxt = "n",
           xlab = expression(italic(p)), ylab = panels[[m]])
      axis(1, at = xticks, labels = xticks)
      abline(h = 1, lty = 3, col = "grey40")
      for (k in seq_along(series)) {
        s <- series[[k]]; d <- ss[[k]][order(ss[[k]]$p), ]
        lo <- d$mean - z * d$se; hi <- d$mean + z * d$se
        if (any(is.finite(c(lo, hi)))) {           # 95% CI band
          sl <- smooth_xy(d$p, lo); sh <- smooth_xy(d$p, hi)
          polygon(c(sl$x, rev(sh$x)), c(sl$y, rev(sh$y)),
                  col = adjustcolor(s$col, alpha.f = 0.18), border = NA)
        }
        sm <- smooth_xy(d$p, d$mean)               # smoothed mean (solid)
        lines(sm$x, sm$y, col = s$col, lty = 1, lwd = 4)
        points(d$p, d$mean, col = s$col, pch = 16, cex = 1.1)
      }
    }
    par(fig = c(0, 1, 0, 1), oma = c(0, 0, 0, 0), mar = c(0, 0, 0, 0),
        family = "serif", new = TRUE)
    plot.new()
    legend("bottom", legend = vapply(series, `[[`, "", "lab"),
           col = vapply(series, `[[`, "", "col"), lty = 1, pch = 16,
           lwd = 4, pt.cex = 1.3, horiz = TRUE, bty = "n", xpd = TRUE, cex = 1.9)
    par(op)
  }
  render()
  if (!is.null(file)) {
    pdf(file, width = 18, height = 8, family = "Times")
    render(); dev.off()
  }
  data.frame(p = pv,
             rmse_ratio  = summ[["rmse Fusion"]]$mean,
             rmse_oracle = summ[["rmse Fusion-oracle"]]$mean,
             var_ratio   = summ[["postvar Fusion"]]$mean,
             var_oracle  = summ[["postvar Fusion-oracle"]]$mean)
}

# Plot 1: each metric vs p (combined "All" population).
plot_vs_p(summary_tbl, pop = "All",
          file = file.path(fig_dir, "sim_hd_metrics_vs_p.pdf"))
# Plot 2: Fusion / RCT-only efficiency ratios vs p.
ratio_tbl <- ratio_vs_p(summary_tbl, raw = res, pop = "All",
                        file = file.path(fig_dir, "sim_hd_ratio_vs_p.pdf"))
cat("\n=== Fusion / RCT-only efficiency ratios (All population) ===\n")
print(ratio_tbl, row.names = FALSE, digits = 3)

# --- Fusion / RCT-only ratio, p = 10 to p = 500 -------------------------------
rr_rng <- ratio_tbl[ratio_tbl$p >= 10 & ratio_tbl$p <= 500,
                    c("p", "rmse_ratio", "var_ratio")]
cat("\n=== Fusion / RCT-only ratio (p = 10 to 500) ===\n")
print(rr_rng, row.names = FALSE, digits = 3)
if (all(c(10, 500) %in% rr_rng$p)) {
  a <- rr_rng[rr_rng$p == 10, ]; b <- rr_rng[rr_rng$p == 500, ]
  cat(sprintf("RMSE ratio: %.3f at p=10  ->  %.3f at p=500  (x%.2f)\n",
              a$rmse_ratio, b$rmse_ratio, b$rmse_ratio / a$rmse_ratio))
  cat(sprintf("Var  ratio: %.3f at p=10  ->  %.3f at p=500  (x%.2f)\n",
              a$var_ratio,  b$var_ratio,  b$var_ratio  / a$var_ratio))
}

cat("\nWrote figures:\n  ", file.path(fig_dir, "sim_hd_metrics_vs_p.pdf"),
    "\n   ", file.path(fig_dir, "sim_hd_ratio_vs_p.pdf"), "\n", sep = "")
