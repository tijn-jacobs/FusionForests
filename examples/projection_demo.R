# =============================================================================
# Demonstration: posterior linear projection of CATE and AF, and how the
# projection coefficients change when RWD is fused into an RCT fit.
#
# What this script shows
#   1. Simulate an RCT + confounded RWD where the true CATE is partly linear
#      and partly nonlinear in the observed covariates.
#   2. Fit two models, both with the same BART back-end and same leaf scale:
#        - "RCT-only" : two-forest CausalShrinkageForest (mu + tau) fitted
#                       on RCT data only.  Evaluated on the combined RCT +
#                       RWD training rows by supplying them as X_test.
#        - "Fusion"   : FusionForest (four-forest, RCT + RWD).
#   3. Project the CATE posterior of each fit onto the SAME basis at the
#      SAME evaluation rows (the combined training set), and compare.
#   4. Repeat for the acceleration factor AF = exp(tau).
#   5. Plot the marginal posteriors -- CATE and AF live on different scales,
#      so they get separate figures.
# =============================================================================

setwd("~/Library/CloudStorage/OneDrive-VrijeUniversiteitAmsterdam/Documents/GitHub/FusionForests")

suppressPackageStartupMessages({
  library(FusionForests)
  library(ShrinkageTrees)
})

# ---- Configuration --------------------------------------------------------
n_rct  <- 300L
n_rwd   <- 700L
p      <- 4L
q      <- 2L                              # unmeasured confounders
N_post <- 2000L
N_burn <- 1000L

out_dir <- "examples/plots"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)


# ---- Data-generating process ---------------------------------------------
# tau(X) is partly linear in X1, X2 and partly nonlinear in X3.  The
# projection on ~X1 + X2 recovers the linear part and absorbs the nonlinear
# X3 contribution into residual variation.
beta_true <- c(X1 = 1.0, X2 = 0.5)
m0_fn   <- function(X) 2 * X[, 1] - X[, 2] + 0.5 * X[, 3]
tau_fn  <- function(X) beta_true["X1"] * X[, 1] +
                       beta_true["X2"] * X[, 2] +
                       0.4 * sin(2 * pi * X[, 3])
conf_fn <- function(U) -0.5 * U[, 1] + 0.25 * U[, 2]
sigma   <- 0.5

X_rct <- matrix(runif(n_rct * p), n_rct, p)
X_rwd  <- matrix(runif(n_rwd  * p), n_rwd,  p)
U_rwd  <- matrix(rnorm(n_rwd * q), n_rwd, q)

A_rct <- rbinom(n_rct, 1L, 0.5)
A_rwd  <- rbinom(n_rwd,  1L, plogis(X_rwd[, 1] + U_rwd[, 1] + U_rwd[, 2]))

y_rct <- m0_fn(X_rct) + A_rct * tau_fn(X_rct) + rnorm(n_rct, 0, sigma)
y_rwd  <- m0_fn(X_rwd)  + A_rwd  * tau_fn(X_rwd)  +
         A_rwd * conf_fn(U_rwd) + rnorm(n_rwd, 0, sigma)

X_train <- rbind(X_rct, X_rwd)
A_train <- c(A_rct, A_rwd)
S_train <- c(rep(1L, n_rct), rep(0L, n_rwd))
y_train <- c(y_rct, y_rwd)
n_train <- n_rct + n_rwd

X_train_df <- as.data.frame(X_train)
names(X_train_df) <- paste0("X", seq_len(p))


# ---- Fit 1: CausalShrinkageForest on RCT only (two-forest BCF) -----------
# Evaluate on the combined training rows by supplying them as X_test, so
# the test posterior-sample matrix has ncol = n_train.
cat("Fitting CausalShrinkageForest (RCT only)... ")
t0 <- proc.time()[["elapsed"]]
n_trees_csf <- 200L
fit_csf <- CausalShrinkageForest(
  y                         = y_rct,
  X_train_control           = X_rct,
  X_train_treat             = X_rct,
  treatment_indicator_train = A_rct,
  X_test_control            = X_train,
  X_test_treat              = X_train,
  treatment_indicator_test  = A_train,
  prior_type_control        = "standard",
  prior_type_treat          = "standard",
  local_hp_control          = 0.5 / sqrt(n_trees_csf),
  local_hp_treat            = 0.5 / sqrt(n_trees_csf),
  number_of_trees_control   = n_trees_csf,
  number_of_trees_treat     = n_trees_csf,
  N_post                    = N_post,
  N_burn                    = N_burn,
  store_posterior_sample    = TRUE,
  verbose                   = FALSE
)
cat(sprintf("done (%.1fs)\n", proc.time()[["elapsed"]] - t0))


# ---- Fit 2: FusionForest on RCT + RWD ------------------------------------
cat("Fitting FusionForest (RCT + RWD)...          ")
t0 <- proc.time()[["elapsed"]]
fit_ff <- FusionForest(
  y                         = y_train,
  X_train_control           = X_train,
  X_train_treat             = X_train,
  treatment_indicator_train = A_train,
  source_indicator_train    = S_train,
  decomposition             = "four-forest",
  error_dist                = "gaussian",
  N_post                    = N_post,
  N_burn                    = N_burn,
  store_posterior_sample    = TRUE,
  verbose                   = FALSE
)
cat(sprintf("done (%.1fs)\n", proc.time()[["elapsed"]] - t0))


# ---- Projection on the training rows -------------------------------------
basis <- ~ X1 + X2 + X3 + X4

# (1) FusionForest: which = "train" pulls train_predictions_sample_treat.
g_cate_ff <- fusion_projection(
  fit_ff, basis, X_eval = X_train_df,
  which = "train", target = "cate",
  weights = "bayesian_bootstrap"
)
g_af_ff <- fusion_projection(
  fit_ff, basis, X_eval = X_train_df,
  which = "train", target = "af", af_scale = "multiplicative",
  weights = "bayesian_bootstrap"
)

# (2) CausalShrinkageForest: fusion_projection() is FusionForest-specific
#     (it requires fit$meta), so we run the same Woody-Carvalho-Murray
#     projection inline on the CSF tau posterior matrix.  Same basis, same
#     centring, same Bayesian-bootstrap recipe -- only the source of tau
#     differs.
project_tau_matrix <- function(tau_draws, X_eval, basis,
                                target = c("cate", "af")) {
  target <- match.arg(target)
  R    <- nrow(tau_draws)
  n_ev <- ncol(tau_draws)
  stopifnot(nrow(X_eval) == n_ev)
  theta <- if (target == "cate") tau_draws else exp(tau_draws)

  Phi <- model.matrix(basis, data = X_eval)
  non_int <- setdiff(colnames(Phi), "(Intercept)")
  Phi[, non_int] <- scale(Phi[, non_int], center = TRUE, scale = FALSE)

  g_w <- matrix(rgamma(R * n_ev, 1, 1), R, n_ev)
  w   <- g_w / rowSums(g_w)

  gamma <- matrix(NA_real_, R, ncol(Phi))
  for (r in seq_len(R)) {
    gamma[r, ] <- lm.wfit(Phi, theta[r, ], w = w[r, ])$coefficients
  }
  colnames(gamma) <- colnames(Phi)
  gamma
}

# CSF was given X_train as X_test, so the relevant matrix is the test one.
tau_csf    <- fit_csf$test_predictions_sample_treat
g_cate_csf <- project_tau_matrix(tau_csf, X_train_df, basis,
                                  target = "cate")
g_af_csf   <- project_tau_matrix(tau_csf, X_train_df, basis,
                                  target = "af")


# ---- Truth ---------------------------------------------------------------
# True projection of tau (or exp(tau)) onto ~X1 + X2 at the training rows,
# after the same centring the projection applies internally.
true_tau  <- tau_fn(X_train)
Phi_truth <- model.matrix(basis, data = X_train_df)
non_int   <- setdiff(colnames(Phi_truth), "(Intercept)")
Phi_truth[, non_int] <- scale(Phi_truth[, non_int], center = TRUE, scale = FALSE)
truth_cate <- coef(lm(true_tau      ~ Phi_truth + 0))
truth_af   <- coef(lm(exp(true_tau) ~ Phi_truth + 0))
names(truth_cate) <- colnames(Phi_truth)
names(truth_af)   <- colnames(Phi_truth)


# ---- Pointwise CATE diagnostics: RMSE and coverage -----------------------
# Compare the per-row posterior of tau(x) directly against the true tau.
# RMSE is on the posterior mean; coverage is the fraction of training rows
# whose true tau falls inside the 95% posterior credible interval.
tau_ff_train <- fit_ff$train_predictions_sample_treat   # R x n_train

cate_diag <- function(tau_draws, true_tau, label) {
  post_mean <- colMeans(tau_draws)
  ci_lo <- apply(tau_draws, 2, quantile, 0.025)
  ci_hi <- apply(tau_draws, 2, quantile, 0.975)
  data.frame(
    model    = label,
    rmse     = sqrt(mean((post_mean - true_tau)^2)),
    bias     = mean(post_mean - true_tau),
    coverage = mean(true_tau >= ci_lo & true_tau <= ci_hi),
    ci_width = mean(ci_hi - ci_lo),
    stringsAsFactors = FALSE
  )
}

cate_all <- rbind(
  cate_diag(tau_csf,      true_tau, "RCT-only (CSF)"),
  cate_diag(tau_ff_train, true_tau, "Fusion (FF)")
)
cate_rct_only <- rbind(
  cate_diag(tau_csf[,      S_train == 1L], true_tau[S_train == 1L],
            "RCT-only (CSF)"),
  cate_diag(tau_ff_train[, S_train == 1L], true_tau[S_train == 1L],
            "Fusion (FF)")
)
cate_rwd_only <- rbind(
  cate_diag(tau_csf[,      S_train == 0L], true_tau[S_train == 0L],
            "RCT-only (CSF)"),
  cate_diag(tau_ff_train[, S_train == 0L], true_tau[S_train == 0L],
            "Fusion (FF)")
)

print_cate <- function(d, header) {
  cat("\n----- ", header, " -----\n", sep = "")
  fmt <- d
  fmt$rmse     <- sprintf("%6.3f",   d$rmse)
  fmt$bias     <- sprintf("%+6.3f",  d$bias)
  fmt$coverage <- sprintf("%5.1f%%", 100 * d$coverage)
  fmt$ci_width <- sprintf("%6.3f",   d$ci_width)
  print(fmt, row.names = FALSE)
}
cat("\n===== Pointwise CATE diagnostics (per training row) =====")
print_cate(cate_all,      "All training rows")
print_cate(cate_rct_only, "RCT rows only")
print_cate(cate_rwd_only,  "RWD rows only")

d_rmse <- cate_all$rmse[cate_all$model == "Fusion (FF)"] -
          cate_all$rmse[cate_all$model == "RCT-only (CSF)"]
d_cov  <- cate_all$coverage[cate_all$model == "Fusion (FF)"] -
          cate_all$coverage[cate_all$model == "RCT-only (CSF)"]
cat(sprintf("\nFusion vs RCT-only:  delta RMSE = %+0.3f (%s),  delta coverage = %+0.1f pp\n",
            d_rmse,
            if (d_rmse < 0) "fusion better" else "RCT-only better",
            100 * d_cov))


# ---- Side-by-side summary, with shift and improvement -------------------
# For each coefficient we report CSF mean+CI, FF mean+CI, the shift
# (FF - CSF), the absolute bias against the projection truth before and
# after fusion, and whether fusion moved the coefficient closer to truth.
psum <- function(v) {
  ci <- quantile(v, c(0.025, 0.975), names = FALSE)
  c(mean = mean(v), lo = ci[1], hi = ci[2])
}

compare_fits <- function(G_csf, G_ff, truth, target_label) {
  coefs <- colnames(G_csf)
  rows <- lapply(coefs, function(nm) {
    s_csf <- psum(G_csf[, nm])
    s_ff  <- psum(G_ff[,  nm])
    tr    <- truth[nm]
    bias_csf <- abs(s_csf["mean"] - tr)
    bias_ff  <- abs(s_ff["mean"]  - tr)
    data.frame(
      target   = target_label,
      coef     = nm,
      truth    = tr,
      csf_mean = s_csf["mean"], csf_lo = s_csf["lo"], csf_hi = s_csf["hi"],
      ff_mean  = s_ff["mean"],  ff_lo  = s_ff["lo"],  ff_hi  = s_ff["hi"],
      shift          = s_ff["mean"] - s_csf["mean"],
      bias_csf       = bias_csf,
      bias_ff        = bias_ff,
      delta_bias     = bias_ff - bias_csf,   # negative = fusion improved
      improves       = bias_ff < bias_csf,
      stringsAsFactors = FALSE
    )
  })
  do.call(rbind, rows)
}

cmp_cate <- compare_fits(g_cate_csf, g_cate_ff, truth_cate, "CATE")
cmp_af   <- compare_fits(g_af_csf,   g_af_ff,   truth_af,   "AF")
cmp_all  <- rbind(cmp_cate, cmp_af)

# Pretty-print
cmp_all$CSF      <- sprintf("%6.3f [%6.3f,%6.3f]",
                            cmp_all$csf_mean, cmp_all$csf_lo, cmp_all$csf_hi)
cmp_all$FF       <- sprintf("%6.3f [%6.3f,%6.3f]",
                            cmp_all$ff_mean,  cmp_all$ff_lo,  cmp_all$ff_hi)
cmp_all$shift_f  <- sprintf("%+6.3f", cmp_all$shift)
cmp_all$dbias_f  <- sprintf("%+6.3f", cmp_all$delta_bias)
cmp_all$verdict  <- ifelse(cmp_all$improves, "improves",
                    ifelse(cmp_all$delta_bias == 0, "unchanged", "worsens"))

cat("\n===== Projection coefficients: RCT-only (CSF) vs Fusion (FF) =====\n")
print(cmp_all[, c("target", "coef", "truth", "CSF", "FF",
                  "shift_f", "dbias_f", "verdict")],
      row.names = FALSE, digits = 3)

# Aggregate fusion verdict per target
agg <- aggregate(cbind(bias_csf, bias_ff) ~ target, data = cmp_all, FUN = mean)
agg$delta_mean_abs_bias <- agg$bias_ff - agg$bias_csf
agg$global_verdict <- ifelse(agg$delta_mean_abs_bias < 0,
                             "Fusion improves on average",
                             "Fusion does not improve on average")
cat("\n----- Aggregate (mean |bias| across coefficients) -----\n")
print(agg, row.names = FALSE, digits = 3)


# ---- Plotting helpers ----------------------------------------------------
# Density plot with: posterior densities (CSF, FF), a vertical truth line, a
# horizontal arrow showing the mean shift (CSF -> FF), and a panel label
# stating whether fusion improved.
plot_marginals <- function(G_csf, G_ff, truth, cmp, main, file,
                           cols = c(CSF = "firebrick", FF = "steelblue")) {
  coefs <- colnames(G_csf)
  pdf(file, width = 8, height = 3 * length(coefs))
  op <- par(mfrow = c(length(coefs), 1L), mar = c(4.5, 4, 2.5, 1) + 0.1)
  for (j in seq_along(coefs)) {
    nm <- coefs[j]
    d_csf <- density(G_csf[, nm])
    d_ff  <- density(G_ff[,  nm])
    x_all <- c(d_csf$x, d_ff$x, truth[nm])
    y_max <- max(d_csf$y, d_ff$y)

    row <- cmp[cmp$coef == nm, ]
    arrow_y <- y_max * 0.05

    plot(NA, xlim = range(x_all), ylim = c(0, y_max * 1.18),
         xlab = paste0("gamma_{", nm, "}"), ylab = "density",
         main = if (j == 1L) main else "")
    polygon(c(d_csf$x, rev(d_csf$x)),
            c(d_csf$y, rep(0, length(d_csf$y))),
            col = adjustcolor(cols["CSF"], alpha.f = 0.20), border = NA)
    polygon(c(d_ff$x,  rev(d_ff$x)),
            c(d_ff$y,  rep(0, length(d_ff$y))),
            col = adjustcolor(cols["FF"],  alpha.f = 0.20), border = NA)
    lines(d_csf, col = cols["CSF"], lwd = 2)
    lines(d_ff,  col = cols["FF"],  lwd = 2)

    # truth line + zero reference
    abline(v = truth[nm], lty = 2, col = "black", lwd = 1.6)
    abline(v = 0,         lty = 3, col = "grey50")

    # posterior-mean markers + shift arrow
    points(row$csf_mean, arrow_y, pch = 19, col = cols["CSF"], cex = 1.2)
    points(row$ff_mean,  arrow_y, pch = 19, col = cols["FF"],  cex = 1.2)
    if (abs(row$shift) > 1e-6) {
      arrows(row$csf_mean, arrow_y, row$ff_mean, arrow_y,
             length = 0.10, angle = 25, col = "grey20", lwd = 2)
    }

    # text label: shift + improvement verdict
    verdict <- if (row$improves) "fusion improves"
               else if (abs(row$delta_bias) < 1e-9) "unchanged"
               else "fusion worsens"
    label   <- sprintf("shift = %+0.3f   |bias|: %.3f -> %.3f   (%s)",
                       row$shift, row$bias_csf, row$bias_ff, verdict)
    mtext(label, side = 3, line = 0.2, cex = 0.85,
          col = if (row$improves) "darkgreen" else "firebrick4")

    if (j == 1L)
      legend("topright", bty = "n",
             legend = c("RCT-only (CSF)", "Fusion (FF)", "truth",
                        "shift (CSF -> FF)"),
             col = c(cols["CSF"], cols["FF"], "black", "grey20"),
             lty = c(1, 1, 2, 1),
             pch = c(NA, NA, NA, NA), lwd = 2)
  }
  par(op); dev.off()
}

# Forest-plot view: one panel per coefficient, CSF and FF intervals stacked
# vertically with the truth as a vertical reference line.  Makes the shift
# (and whether it moves toward truth) visually obvious at a glance.
plot_forest <- function(cmp, main, file,
                        cols = c(CSF = "firebrick", FF = "steelblue")) {
  coefs <- unique(cmp$coef)
  pdf(file, width = 7, height = 1.1 + 0.9 * length(coefs))
  op <- par(mar = c(4, 6, 2.5, 1) + 0.1)
  y_pos <- seq_along(coefs)
  ylim  <- c(0.5, length(coefs) + 0.7)
  xlim  <- range(c(cmp$csf_lo, cmp$csf_hi, cmp$ff_lo, cmp$ff_hi,
                    cmp$truth, 0))
  plot(NA, xlim = xlim, ylim = ylim, yaxt = "n",
       xlab = "coefficient value", ylab = "", main = main)
  axis(2, at = y_pos, labels = coefs, las = 1)
  abline(v = 0, lty = 3, col = "grey60")
  for (j in seq_along(coefs)) {
    nm  <- coefs[j]
    row <- cmp[cmp$coef == nm, ]
    abline(v = row$truth, lty = 2, col = "black", lwd = 1.2)
    # CSF (slightly above the coef line)
    yh <- y_pos[j] + 0.18
    segments(row$csf_lo, yh, row$csf_hi, yh, col = cols["CSF"], lwd = 2)
    points(row$csf_mean, yh, pch = 19, col = cols["CSF"], cex = 1.4)
    # FF (slightly below)
    yl <- y_pos[j] - 0.18
    segments(row$ff_lo,  yl, row$ff_hi,  yl, col = cols["FF"],  lwd = 2)
    points(row$ff_mean,  yl, pch = 19, col = cols["FF"],  cex = 1.4)
    # Shift arrow between the two means
    arrows(row$csf_mean, yh - 0.05, row$ff_mean, yl + 0.05,
           length = 0.08, angle = 25, col = "grey25", lwd = 1.5)
    # Verdict text
    txt <- sprintf("%s  (Delta|bias| = %+0.3f)",
                   ifelse(row$improves, "improves", "worsens"),
                   row$delta_bias)
    mtext(txt, side = 4, at = y_pos[j], las = 1, cex = 0.7, line = -2,
          col = ifelse(row$improves, "darkgreen", "firebrick4"))
  }
  legend("topright", bty = "n",
         legend = c("RCT-only (CSF)", "Fusion (FF)", "truth"),
         col = c(cols["CSF"], cols["FF"], "black"),
         lty = c(1, 1, 2), pch = c(19, 19, NA), lwd = 2)
  par(op); dev.off()
}


# ---- Plots ---------------------------------------------------------------
plot_marginals(g_cate_csf, g_cate_ff, truth_cate, cmp_cate,
  main = "CATE projection: RCT-only vs Fusion",
  file = file.path(out_dir, "projection_cate_csf_vs_ff.pdf"))

plot_marginals(g_af_csf, g_af_ff, truth_af, cmp_af,
  main = "AF projection: RCT-only vs Fusion",
  file = file.path(out_dir, "projection_af_csf_vs_ff.pdf"))

plot_forest(cmp_cate,
  main = "CATE projection: shift from RCT-only to Fusion",
  file = file.path(out_dir, "projection_cate_forest.pdf"))

plot_forest(cmp_af,
  main = "AF projection: shift from RCT-only to Fusion",
  file = file.path(out_dir, "projection_af_forest.pdf"))

cat("\nPlots saved to ", out_dir, "/projection_{cate,af}_{csf_vs_ff,forest}.pdf\n",
    sep = "")
cat("Done.\n")
