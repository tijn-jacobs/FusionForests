# ══════════════════════════════════════════════════════════════════════════════
# analyse_ktau_v1.R -- local analysis of the k_tau pilot.
#
# Reads results/sim_ktau_v1_output.rds (one row per fit, produced on the HPC by
# sim_ktau_v1.R), writes results/tables/ktau_means.csv and ktau_paired_diff.csv,
# and prints the table to decide on.  Base R only.  Run from the repository root:
#   Rscript simulations/revision_v3/pilot_ktau/analyse_ktau_v1.R
#
# Decision rule, fixed before seeing the results: keep k_tau = 1 unless, in a
# cell, its coverage is clearly below nominal or its RMSE is clearly worse than
# that of a neighbouring value.  "Clearly" is flagged below as coverage < 0.90,
# or a paired RMSE difference above 0.02 and above 2 standard errors.
# ══════════════════════════════════════════════════════════════════════════════

res_dir <- "simulations/revision_v3/pilot_ktau/results"
tab_dir <- file.path(res_dir, "tables")
dir.create(tab_dir, recursive = TRUE, showWarnings = FALSE)

res <- readRDS(file.path(res_dir, "sim_ktau_v1_output.rds"))
config_names <- unique(res$config)             # in the order of sim_ktau_v1.R
cells        <- unique(res$cell)
cat(sprintf("%d fits, %d failed; %d values of k_tau x %d cells x %d reps\n",
            nrow(res), sum(res$failed), length(config_names), length(cells),
            max(res$Iter)))

# ── The package check: is the fixed standardization running? ──────────────────
cat(sprintf("\nsigma_hat: mean %.3f (expected near 1, near 1.6 means the old package)\n",
            mean(res$sigma_hat, na.rm = TRUE)))
cat(sprintf("y_mean:    mean %.3f (expected near 0, near 1.6 means the old package)\n",
            mean(res$y_mean, na.rm = TRUE)))

# ── Means per (cell, k_tau) ───────────────────────────────────────────────────
show <- c("cate_all_rmse", "cate_all_bias", "cate_all_cover", "cate_all_width",
          "cate_all_postvar", "c_rmse", "c_cover", "d_rmse", "d_cover",
          "ate_cover", "ate_width", "sd_c_true")
means <- aggregate(res[, show], by = res[, c("cell", "config")],
                   FUN = function(x) mean(x, na.rm = TRUE))
n_ok <- aggregate(!res$failed, by = res[, c("cell", "config")], FUN = sum)
names(n_ok)[3] <- "n"
means <- merge(means, n_ok)
means$config <- factor(means$config, levels = config_names)
means$cell   <- factor(means$cell, levels = cells)
means <- means[order(means$cell, means$config), ]
write.csv(means, file.path(tab_dir, "ktau_means.csv"), row.names = FALSE)

# ── Paired RMSE differences to k_tau = 1 ──────────────────────────────────────
base <- res[res$config == "k_tau=1", c("cell", "Iter", "cate_all_rmse")]
dd   <- merge(res[res$config != "k_tau=1", c("cell", "config", "Iter", "cate_all_rmse")],
              base, by = c("cell", "Iter"), suffixes = c("", ".ref"))
dv   <- dd$cate_all_rmse - dd$cate_all_rmse.ref
a    <- aggregate(dv, by = dd[, c("cell", "config")],
                  FUN = function(x) c(mean = mean(x, na.rm = TRUE),
                                      se = sd(x, na.rm = TRUE) /
                                        sqrt(sum(!is.na(x)))))
diffs <- data.frame(cell = a$cell, config = a$config,
                    diff = a$x[, "mean"], se = a$x[, "se"])
diffs$config <- factor(diffs$config, levels = config_names)
diffs$cell   <- factor(diffs$cell, levels = cells)
diffs <- diffs[order(diffs$cell, diffs$config), ]
write.csv(diffs, file.path(tab_dir, "ktau_paired_diff.csv"), row.names = FALSE)

# ── Print the table to decide on ──────────────────────────────────────────────
for (cl in cells) {
  cat("\n===", cl, "===\n")
  m <- means[means$cell == cl, ]
  cat(sprintf("mean sd of the true c(x): %.3f\n", m$sd_c_true[1]))
  tbl <- m[, c("config", "n", "cate_all_rmse", "cate_all_bias", "cate_all_cover",
               "cate_all_width", "c_rmse", "d_rmse", "ate_cover")]
  names(tbl) <- sub("^cate_all_", "", names(tbl))
  print(tbl, row.names = FALSE, digits = 3)
  cat("\nPaired RMSE difference to k_tau = 1 (negative: better than k_tau = 1)\n")
  print(diffs[diffs$cell == cl, c("config", "diff", "se")], row.names = FALSE,
        digits = 3)

  cover_one <- m$cate_all_cover[m$config == "k_tau=1"]
  d_cl      <- diffs[diffs$cell == cl, ]
  better    <- d_cl[d_cl$diff < -0.02 & d_cl$diff < -2 * d_cl$se, ]
  if (cover_one < 0.90) {
    cat(sprintf("FLAG: coverage at k_tau = 1 is %.3f, below 0.90\n", cover_one))
  }
  if (nrow(better) > 0) {
    cat("FLAG: clearly lower RMSE than k_tau = 1 at:",
        paste(better$config, collapse = ", "), "\n")
  }
  if (cover_one >= 0.90 && nrow(better) == 0) {
    cat("No flag in this cell: the rule keeps k_tau = 1.\n")
  }
}

cat("\nWrote", file.path(tab_dir, "ktau_means.csv"), "and",
    file.path(tab_dir, "ktau_paired_diff.csv"), "\n")
