################################################################################
## data/analysis 1/make_deepaft_caterpillars.R
##
## Standalone regeneration of the deepAFT subject-level caterpillar plots for
## the SM (Figure fig:cate-deepaft), straight from the saved
## analysis_deepaft.rds.  Use this when you re-ran ONLY the deepAFT section of
## analysis.R: the caterpillar loop there relies on make_caterpillar() and
## fig_dir from Section 9, so it does not run on its own.
##
## No model re-fitting.  It reads the bootstrap draws from the .rds and rebuilds
## only the source indicator (data prep, no MCMC).  Run from the project root
## (the FusionForests directory), e.g.
##   Rscript "data/analysis 1/make_deepaft_caterpillars.R"
## or source() it after setwd() to the project root.
################################################################################

suppressPackageStartupMessages(library(ggplot2))

proj    <- "."
out_dir <- file.path(proj, "data", "analysis 1")
fig_dir <- file.path(proj, "notes", "general", "figures")
dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)

## --- Source indicator over the combined cohort (rct then macs) -----------
## Rebuild the cohorts to recover the RCT/RWD source vector that aligns with
## the rows of the saved draws.  Data prep only -- no model fitting.
source(file.path(out_dir, "data_prep.R"))
rct    <- build_rct(male_only = TRUE)
macs   <- build_macs(cd4_band = c(200, 500))
d_fuse <- rbind(rct, macs)
d_fuse <- d_fuse[is.finite(d_fuse$log_time), ]
src_M2 <- as.integer(d_fuse$source == "RCT")

## --- Caterpillar helper (mirrors make_caterpillar() in analysis.R) -------
src_cols <- c(RCT = "#1f78b4", RWD = "#e31a1c")
make_caterpillar <- function(samples, src, out_path,
                             width = 14, height = 8, xlim = NULL) {
  af      <- exp(samples)
  mean_af <- colMeans(af)
  lo_af   <- apply(af, 2, quantile, 0.025)
  hi_af   <- apply(af, 2, quantile, 0.975)
  ord     <- order(mean_af)
  src_lab <- factor(ifelse(src == 1L, "RCT", "RWD"), levels = c("RCT", "RWD"))
  df <- data.frame(rank = seq_along(ord), mean = mean_af[ord],
                   lo = lo_af[ord], hi = hi_af[ord], source = src_lab[ord])
  p <- ggplot(df, aes(y = rank, xmin = lo, xmax = hi, colour = source)) +
    geom_linerange(alpha = 0.45, linewidth = 0.6) +
    geom_point(aes(x = mean), colour = "black", size = 0.6) +
    geom_vline(xintercept = 1, linetype = "dashed", colour = "grey40",
               linewidth = 0.6) +
    scale_colour_manual(values = src_cols) +
    coord_cartesian(xlim = xlim) +   # NULL = automatic; c(lo, hi) to fix the AF axis
    scale_y_continuous(breaks = NULL) +
    labs(y = "Patients (ordered by posterior mean)",
         x = expression("Acceleration factor"), colour = NULL) +
    guides(colour = guide_legend(
      override.aes = list(linewidth = 4, alpha = 1, size = 0))) +
    theme_minimal(base_size = 24) +
    theme(text              = element_text(family = "Times"),
          legend.position   = "top",
          legend.text       = element_text(size = 24, family = "Times"),
          legend.key.width  = unit(40, "pt"),
          legend.key.height = unit(16, "pt"),
          legend.spacing.x  = unit(8, "pt"),
          panel.grid.major.y = element_blank(),
          panel.grid.minor.y = element_blank(),
          panel.background  = element_rect(fill = "white", colour = NA))
  cairo_pdf(out_path, width = width, height = height)
  print(p)
  invisible(dev.off())
  cat("Wrote ", out_path, "\n", sep = "")
}

## --- Regenerate the four caterpillars from the saved draws ---------------
dd <- readRDS(file.path(out_dir, "analysis_deepaft.rds"))

## Per-plot x-axis limits (acceleration-factor scale); NULL = automatic.
## Tip: give the fusion and these the same window for a fair visual comparison.
dd_xlims <- list(
  deepaft_rct_s    = c(0.85, 2.5),
  deepaft_rct_t    = c(0, 9),
  deepaft_pooled_s = c(0.85, 2.5),
  deepaft_pooled_t = c(0, 9))

for (f in dd$fits) {
  if (is.null(f$draws) || !is.matrix(f$draws)) next
  if (nrow(f$draws) != length(src_M2))
    stop("draws rows (", nrow(f$draws), ") != cohort size (",
         length(src_M2), "): the cohort build does not match the saved fit.")
  tag <- gsub("[^a-z0-9]+", "_", tolower(f$name))   # e.g. deepaft_pooled_t
  make_caterpillar(
    t(f$draws), src_M2,
    file.path(fig_dir, paste0("cate_caterpillar_", tag, ".pdf")),
    xlim = dd_xlims[[tag]])
}
cat("Done: four deepAFT caterpillars written to ", fig_dir, "\n", sep = "")
