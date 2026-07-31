################################################################################
## data/analysis 1/analysis.R
##
## Final analysis of the ACTG175 + MACS data fusion. Replaces the
## preliminary analysis in data/analysis 0/ with three updates:
##
##   1. Four-forest decomposition (mu, g, tau, c) instead of three-forest.
##   2. source_hdp_scale residual prior (HDP shared atoms + per-source
##      sigma), the headline residual prior of the methodology paper.
##   3. Leaf-prior and sampler hyperparameters aligned with the
##      simulation study simulations/main/sim_surv_v12.R:
##        - FusionForest: k = 0.5 for every forest, error_truncation_K
##          = 50.
##        - CausalShrinkageForest: local_hp = 1 / sqrt(m), power/base =
##          (2, 0.95) for both forests.
##      Treatment coding is kept "centered" (the sim uses "binary").
##
## Cohort   : page-2 trial-aligned (male RCT, MACS CD4 in [200, 500],
##            AIDS-free at anchor).
## Endpoint : RCT composite cens (published); MACS v5 = all-cause death
##            with last-lab-year as the event-time fallback.
## Covariates: (age, cd4, cd8, anchor_year, race, prior_art_years).
##
## Two fits:
##   M1 : RCT-only baseline, via ShrinkageTrees::CausalShrinkageForest.
##        Trained on the RCT rows only.  The MACS rows are passed as a
##        test set so M1 also produces a CATE prediction for every MACS
##        subject, which lets the combined-cohort caterpillar (page 3)
##        compare M1 against M2 on the full RCT + MACS sample.
##   M2 : RCT + MACS fusion, FusionForest four-forest, with the
##        interval-censored MACS encoding (dead MACS subjects pass
##        status = 0 and ic_indicator = 1 with the event known to lie in
##        (last_lab_year, t_max_macs]).  RCT rows are unchanged.
##
## Outputs (in this folder):
##   fits.rds                 - full posterior objects + cohort frames
##                              + run settings
##   analysis_summary.txt     - cohort sizes, ATE point + 95% CrI for M1
##                              and M2 (log-time and acceleration-factor)
##   analysis_figures.pdf     - page 1: ATE posterior densities (log-time)
##                              for M1 vs M2.
##                              page 2: subject-level CATE caterpillar on
##                              the RCT subjects only (M1, M2).
##                              page 3: subject-level CATE caterpillar on
##                              the combined RCT + MACS cohort.  M1's MACS
##                              predictions come from passing MACS as a
##                              test set; M2's are train-set predictions
##                              from the fusion fit.
##                              page 4: HDP-scale residual diagnostics for
##                              M2.
################################################################################

setwd("~/Library/CloudStorage/OneDrive-VrijeUniversiteitAmsterdam/Documents/GitHub/FusionForests/")

suppressPackageStartupMessages({
  library(FusionForests)
  library(ShrinkageTrees)
  library(ggplot2)
  library(patchwork)
})

proj    <- "."
macsDir <- file.path(proj, "data", "MACS PDS")
out_dir <- file.path(proj, "data", "analysis 1")
source(file.path(out_dir, "data_prep.R"))

################################################################################
## 1. Settings
################################################################################

seed   <- 1L
nBurn <- 5000L
nPost <- 5000L

## Methodology-default Chipman calibration: sigma_h = k / (2 sqrt(m))
## with k = 1 for every forest.  The wrapper-level argument is
## k_X = k / 2 = 0.5 (since FusionForest uses omega = k_X / sqrt(m) as
## the leaf SD).
kPaper       <- 1.0           # methodology default, uniform across forests
kCodeUniform <- kPaper / 2    # wrapper-level argument: 0.5

## Tree counts (methodology default).
nTreesSh  <- 200L  # shared baseline mu
nTreesD   <-  50L  # deviation g
nTreesTau <- 100L  # treatment tau
nTreesC   <-  50L  # confounding c

## M1 (CausalShrinkageForest, RCT-only): standard BCF-style counts.
nTreesM1Control <- 200L  # prognostic forest
nTreesM1Treat   <- 100L  # treatment forest

## HDP truncation
truncK <- 50L   # matches FusionForest default used in sim_surv_v12.R

cat(sprintf(
  "Chipman calibration: k = %.1f (uniform across forests) -> wrapper k_X = %.2f\n",
  kPaper, kCodeUniform))

################################################################################
## 2. Build cohorts
################################################################################

cat("\n", strrep("=", 70), "\n  COHORT CONSTRUCTION (trial-aligned)\n",
    strrep("=", 70), "\n", sep = "")

rct  <- build_rct(male_only = TRUE)
macs <- build_macs(cd4_band = c(200, 500))

cat(sprintf(
  "\nCohort sizes:\n  RCT  n = %d  (Z=0: %d, Z=1: %d; events: %d)\n  MACS n = %d  (Z=0: %d, Z=1: %d; events: %d)\n",
  nrow(rct),  sum(rct$treat  == 0), sum(rct$treat  == 1), sum(rct$status  == 1),
  nrow(macs), sum(macs$treat == 0), sum(macs$treat == 1), sum(macs$status == 1)))

################################################################################
## 3. Fit M1 (RCT-only, CausalShrinkageForest)
################################################################################

## d_train is the cohort the model is FIT on (here: RCT rows only).
## d_test is an optional cohort the model PREDICTS on without including
## those rows in the likelihood.  Passing MACS as the test set produces
## a CATE prediction for every MACS subject (fit$test_predictions_sample_treat)
## that can be concatenated with the RCT predictions to evaluate M1 on
## the full combined cohort.
run_fit_M1 <- function(d_train, d_test = NULL) {
  X_train <- as.matrix(d_train[, harmCovars])
  X_test  <- if (!is.null(d_test)) as.matrix(d_test[, harmCovars]) else NULL
  log_info <- ShrinkageTrees:::censored_info(d_train$log_time, d_train$status)
  local_hp_treat   <- 1 / sqrt(nTreesM1Treat)
  local_hp_control <- 1 / sqrt(nTreesM1Control)
  fit <- ShrinkageTrees::CausalShrinkageForest(
    y                         = d_train$log_time,
    status                    = d_train$status,
    X_train_control           = X_train,
    X_train_treat             = X_train,
    treatment_indicator_train = d_train$treat,
    X_test_control            = X_test,
    X_test_treat              = X_test,
    treatment_indicator_test  = if (!is.null(d_test)) d_test$treat else NULL,
    outcome_type              = "right-censored",
    timescale                 = "log",
    prior_type_control        = "standard",
    prior_type_treat          = "standard",
    local_hp_treat            = local_hp_treat,
    local_hp_control          = local_hp_control,
    number_of_trees_control   = nTreesM1Control,
    number_of_trees_treat     = nTreesM1Treat,
    power_control             = 2,    base_control = 0.95,
    power_treat               = 2,    base_treat   = 0.95,
    treatment_coding          = "centered",
    N_post                    = nPost,
    N_burn                    = nBurn,
    store_posterior_sample    = TRUE,
    verbose                   = FALSE
  )
  ## CSF stores sigma on the standardised internal scale; rescale to
  ## log-time so downstream survival overlays use the correct sigma.
  fit$sigma_scaled <- fit$sigma * log_info$sd
  fit
}

set.seed(seed)
cat("\n", strrep("=", 70),
    "\n  FIT M1: RCT trained, MACS as test set\n",
    strrep("=", 70), "\n", sep = "")
fit_M1 <- run_fit_M1(rct, d_test = macs)

################################################################################
## 4. Fit M2 (Fusion, FusionForest, four-forest, source_hdp_scale)
################################################################################

## v5 interval-censored MACS encoding: dead MACS subjects pass
## status = 0 and ic_indicator = 1 with the event known to lie in
## (last_year, t_max_macs] rather than a biased point-time at last_year.
## RCT rows are unaffected -- their ic_indicator is 0 and status_ic
## equals status.
run_fit_M2 <- function(d, src) {
  X <- as.matrix(d[, harmCovars])
  fit <- FusionForest(
    y                            = d$log_time,
    status                       = d$status_ic,
    observed_left_time           = d$obs_left_log_t,
    observed_right_time          = d$obs_right_log_t,
    interval_censoring_indicator = d$ic_indicator,
    X_train_control              = X,
    X_train_treat                = X,
    treatment_indicator_train    = d$treat,
    source_indicator_train       = src,
    outcome_type              = "right-censored",
    timescale                 = "log",
    decomposition             = "four-forest",
    treatment_coding          = "centered",
    number_of_trees_control   = nTreesSh,
    number_of_trees_deviation = nTreesD,
    number_of_trees_treat     = nTreesTau,
    number_of_trees_deconf    = nTreesC,
    k_control                 = kCodeUniform,
    k_deviation               = kCodeUniform,
    k_treat                   = 0.25,
    k_deconf                  = kCodeUniform,
    power_control             = 2,     base_control    = 0.95,  # shared baseline mu
    power_deviation           = 2,     base_deviation  = 0.95,  # deviation g
    power_treat               = 3,     base_treat      = 0.95,  # treatment tau
    power_deconf              = 3,     base_deconf     = 0.25,  # confounding c (stumps)
    error_dist                = "source_hdp_scale",
    error_truncation_K        = truncK,
    error_atom_scale          = 0.5,
    error_mass_init           = 1.0,
    N_post                    = nPost,
    N_burn                    = nBurn,
    store_posterior_sample    = TRUE,
    verbose                   = FALSE
  )
  fit$sigma_scaled <- fit$sigma
  fit
}

set.seed(seed)
cat("\n", strrep("=", 70),
    "\n  FIT M2: RCT + MACS fusion (FusionForest, four-forest)\n",
    strrep("=", 70), "\n", sep = "")
d_fuse <- rbind(rct, macs)
d_fuse <- d_fuse[is.finite(d_fuse$log_time), ]
src_M2 <- as.integer(d_fuse$source == "RCT")

## M2: fusion with the interval-censored MACS encoding (dead MACS
## subjects pass status = 0 and ic_indicator = 1 with the event known to
## lie in (last_year, t_max_macs]).
fit_M2 <- run_fit_M2(d_fuse, src_M2)

################################################################################
## 5. Posterior summaries (ATE via Bayesian bootstrap)
##
## We propagate the joint uncertainty over (tau, F, pi) by the hierarchical
## Bayesian bootstrap of METHODOLOGY.tex (Section "Posterior inference"):
##
##   M1 (single-source, RCT only):
##     Regular Rubin (1981) BB over the n_rct RCT subjects.  Per posterior
##     draw b, w_b ~ Dirichlet(1_{n_rct}) and
##       ate_M1[b] = sum_i w_b[i] * tau_b(X_i)     over RCT subjects.
##
##   M2 (combined, fusion):
##     Hierarchical BB with alpha = (n_0, n_1).  Per posterior draw b,
##     pi_b ~ Dirichlet(n_0, n_1) and per-source w_b^{(s)} ~ Dir(1_{n_s});
##       ate_M2[b] = sum_s pi_b[s] *
##                   sum_{i: S_i=s} w_b^{(s)}[i] * tau_b(X_i).
##     alpha = (n_0, n_1) approximately recovers a pooled BB and is the
##     methodology default.
################################################################################

## CATE samples on the log-time scale: N_post x n_eval matrix.
log_cate_samples <- function(fit) fit$train_predictions_sample_treat

eval_idx_M1 <- seq_len(nrow(rct))
eval_idx_M2 <- which(d_fuse$source == "RCT")

## Per-fit posterior matrices of subject-level log-time CATE.
##   - M1: train-set RCT subjects in train_predictions_sample_treat,
##         MACS predictions (passed as test set) in
##         test_predictions_sample_treat.  Combined cohort matrix is the
##         column-bind in the d_fuse row order (RCT first, MACS second).
##   - M2: trained on the combined cohort; train_predictions_sample_treat
##         already covers every row of d_fuse.
M1_rct_samples       <- fit_M1$train_predictions_sample_treat
M1_combined_samples  <- cbind(fit_M1$train_predictions_sample_treat,
                              fit_M1$test_predictions_sample_treat)
M2_combined_samples  <- log_cate_samples(fit_M2)
M2_rct_samples       <- M2_combined_samples[, eval_idx_M2, drop = FALSE]

## Regular Rubin (1981) Bayesian bootstrap: B x n flat-Dirichlet weight
## matrix.  Each row is one draw of w ~ Dirichlet(1_n), sampled via
## independent Exp(1) and row-normalised.
bb_weights_mat <- function(B, n) {
  E <- matrix(rexp(B * n), nrow = B, ncol = n)
  E / rowSums(E)
}

## Hierarchical Bayesian bootstrap (METHODOLOGY.tex eq:ate-bb).  Returns
## the B-vector of ATE draws with pi ~ Dir(alpha) over source-mixing
## fractions and inner per-source flat-Dirichlet weights.  `src` is the
## length-n_total source indicator with entries in {0, 1}, aligned with
## the columns of `samples`.  Default alpha = (n_0, n_1).
hbb_ate <- function(samples, src, alpha = NULL) {
  B  <- nrow(samples)
  i0 <- which(src == 0); n0 <- length(i0)
  i1 <- which(src == 1); n1 <- length(i1)
  if (is.null(alpha)) alpha <- c(n0, n1)
  W0 <- bb_weights_mat(B, n0)
  W1 <- bb_weights_mat(B, n1)
  a0 <- rowSums(W0 * samples[, i0, drop = FALSE])
  a1 <- rowSums(W1 * samples[, i1, drop = FALSE])
  G0 <- rgamma(B, alpha[1])
  G1 <- rgamma(B, alpha[2])
  pi0 <- G0 / (G0 + G1)
  pi0 * a0 + (1 - pi0) * a1
}

set.seed(seed)
## M1 (single-source, RCT-only): regular Rubin BB over the RCT covariate
## sample.  Target population is F_RCT.
W_M1   <- bb_weights_mat(nrow(M1_rct_samples), ncol(M1_rct_samples))
ate_M1 <- rowSums(W_M1 * M1_rct_samples)

## M2 (combined, fusion): hierarchical BB with alpha = (n_0, n_1) over
## the combined cohort.  Target population is F = pi_0 F_RWD + pi_1 F_RCT
## with pi ~ Dirichlet(n_0, n_1).
ate_M2 <- hbb_ate(M2_combined_samples, src_M2)

## --- Average width of the subject-level 95% CrI for the CATE ------------
## For each subject the CATE 95% credible interval is the 2.5%-97.5%
## posterior-quantile range of the log-time treatment effect; we report
## the mean width across subjects.  Both fits are evaluated on the full
## combined RCT + MACS cohort: M1's MACS predictions come from passing
## MACS as the test set (M1_combined_samples), M2's are the train-set
## predictions over d_fuse (M2_combined_samples).
## Reported on two scales: the log-time CATE itself, and the
## acceleration factor exp{tau(x)} (same exp() transform used in the
## caterpillar and projection sections).
cate_ci_width <- function(samples) {
  q <- apply(samples, 2, quantile, c(0.025, 0.975))
  q[2, ] - q[1, ]
}
mean_ci_M1_combined <- mean(cate_ci_width(M1_combined_samples))
mean_ci_M2_combined <- mean(cate_ci_width(M2_combined_samples))

mean_ci_M1_combined_af <- mean(cate_ci_width(exp(M1_combined_samples)))
mean_ci_M2_combined_af <- mean(cate_ci_width(exp(M2_combined_samples)))

## Posterior variance of the CATE: per-subject variance across posterior
## draws, averaged over the combined cohort.  Same two scales as the CrI
## widths.
cate_post_var <- function(samples) apply(samples, 2, var)
mean_var_M1_combined <- mean(cate_post_var(M1_combined_samples))
mean_var_M2_combined <- mean(cate_post_var(M2_combined_samples))

mean_var_M1_combined_af <- mean(cate_post_var(exp(M1_combined_samples)))
mean_var_M2_combined_af <- mean(cate_post_var(exp(M2_combined_samples)))

summarise_post <- function(x, label, scale = "log") {
  q <- quantile(x, c(0.025, 0.5, 0.975))
  sprintf("  %-22s  mean = %+.3f, median = %+.3f, 95%% CrI = [%+.3f, %+.3f]",
          label, mean(x), q[2], q[1], q[3])
}

summary_lines <- c(
  "============================================================",
  "  ATE posterior (hierarchical Bayesian bootstrap)",
  "    M1 target: F_RCT      (regular BB over RCT subjects)",
  "    M2 target: combined   (HBB with alpha = (n_0, n_1))",
  "============================================================",
  "",
  "Log-survival scale  (positive => treated longer log-survival):",
  summarise_post(ate_M1, "M1 (RCT only)"),
  summarise_post(ate_M2, "M2 (Fusion)"),
  "",
  "Acceleration-factor scale  (exp; 1.0 => no effect):",
  summarise_post(exp(ate_M1), "M1 (RCT only)"),
  summarise_post(exp(ate_M2), "M2 (Fusion)"),
  "",
  "Average 95% CrI width of subject-level CATE (log-time, combined cohort):",
  sprintf("  %-22s  = %.3f", "M1 (RCT only)", mean_ci_M1_combined),
  sprintf("  %-22s  = %.3f", "M2 (Fusion)",   mean_ci_M2_combined),
  sprintf("  %-22s  = %.1f%%", "width reduction (M2 vs M1)",
          100 * (1 - mean_ci_M2_combined / mean_ci_M1_combined)),
  "",
  "Average 95% CrI width of subject-level CATE (AF scale, combined cohort):",
  sprintf("  %-22s  = %.3f", "M1 (RCT only)", mean_ci_M1_combined_af),
  sprintf("  %-22s  = %.3f", "M2 (Fusion)",   mean_ci_M2_combined_af),
  sprintf("  %-22s  = %.1f%%", "width reduction (M2 vs M1)",
          100 * (1 - mean_ci_M2_combined_af / mean_ci_M1_combined_af)),
  "",
  "Average posterior variance of subject-level CATE (log-time, combined cohort):",
  sprintf("  %-22s  = %.4f", "M1 (RCT only)", mean_var_M1_combined),
  sprintf("  %-22s  = %.4f", "M2 (Fusion)",   mean_var_M2_combined),
  sprintf("  %-22s  = %.1f%%", "variance reduction (M2 vs M1)",
          100 * (1 - mean_var_M2_combined / mean_var_M1_combined)),
  "",
  "Average posterior variance of subject-level CATE (AF scale, combined cohort):",
  sprintf("  %-22s  = %.4f", "M1 (RCT only)", mean_var_M1_combined_af),
  sprintf("  %-22s  = %.4f", "M2 (Fusion)",   mean_var_M2_combined_af),
  sprintf("  %-22s  = %.1f%%", "variance reduction (M2 vs M1)",
          100 * (1 - mean_var_M2_combined_af / mean_var_M1_combined_af)),
  "",
  sprintf("Cohort: RCT n = %d  (Z=0: %d, Z=1: %d; events %d)",
          nrow(rct),  sum(rct$treat  == 0), sum(rct$treat  == 1), sum(rct$status  == 1)),
  sprintf("        MACS n = %d  (Z=0: %d, Z=1: %d; events %d)",
          nrow(macs), sum(macs$treat == 0), sum(macs$treat == 1), sum(macs$status == 1)),
  "",
  sprintf("Settings: Chipman k = %.1f (uniform), wrapper k_X = %.2f,",
          kPaper, kCodeUniform),
  sprintf("          trees (sh, d, tau, c) = (%d, %d, %d, %d),",
          nTreesSh, nTreesD, nTreesTau, nTreesC),
  sprintf("          N_burn = %d, N_post = %d, error_dist = source_hdp_scale",
          nBurn, nPost)
)
cat("\n"); writeLines(summary_lines)

writeLines(summary_lines, file.path(out_dir, "analysis_summary.txt"))

################################################################################
## 6. Plots
################################################################################

## --- (a) Average treatment effect posterior densities (log-time) ---------
model_levels <- c("M1 (RCT only)", "M2 (Fusion)")

ate_df <- rbind(
  data.frame(model = "M1 (RCT only)", ate = ate_M1),
  data.frame(model = "M2 (Fusion)",   ate = ate_M2))
ate_df$model <- factor(ate_df$model, levels = model_levels)

p_ate <- ggplot(ate_df, aes(x = ate, fill = model)) +
  geom_density(alpha = 0.45, colour = NA) +
  geom_vline(xintercept = 0, linetype = "dashed", colour = "grey40") +
  labs(x = "ATE (log time)", y = "Posterior density",
       title = paste0(
         "Average treatment effect (Bayesian bootstrap): ",
         "M1 over F_RCT, M2 over combined cohort")) +
  theme_minimal() + theme(legend.title = element_blank())

## --- (b, c) Subject-level CATE caterpillars ------------------------------
## summarise_subjects() takes a posterior sample matrix directly (rather
## than (fit, idx)) so we can pass an arbitrary combination, e.g. the
## column-bound train + test M1 predictions for the combined cohort.
summarise_subjects <- function(samples, label) {
  ord <- order(colMeans(samples))
  data.frame(
    rank  = seq_along(ord),
    mean  = colMeans(samples)[ord],
    lo    = apply(samples, 2, quantile, 0.025)[ord],
    hi    = apply(samples, 2, quantile, 0.975)[ord],
    model = label,
    stringsAsFactors = FALSE)
}

cate_rct_df <- rbind(
  summarise_subjects(M1_rct_samples, "M1 (RCT only)"),
  summarise_subjects(M2_rct_samples, "M2 (Fusion)"))
cate_rct_df$model <- factor(cate_rct_df$model, levels = model_levels)

cate_all_df <- rbind(
  summarise_subjects(M1_combined_samples, "M1 (RCT only)"),
  summarise_subjects(M2_combined_samples, "M2 (Fusion)"))
cate_all_df$model <- factor(cate_all_df$model, levels = model_levels)

## Caterpillar palette: both models share a single darkgreen family.
## CrI bars use a translucent darkgreen; the posterior-mean point is
## overlaid in a much darker green ("#0B2A0B", nearly black) and at full
## opacity so the central estimate stands out clearly against the bars.
band_colour <- "darkgreen"
mean_colour <- "#0B2A0B"

p_cate_rct <- ggplot(cate_rct_df, aes(x = rank, ymin = lo, ymax = hi)) +
  geom_linerange(colour = band_colour, alpha = 0.45) +
  geom_point(aes(y = mean), colour = mean_colour, size = 0.45) +
  geom_hline(yintercept = 0, linetype = "dashed", colour = "grey40") +
  facet_wrap(~ model, scales = "free_x") +
  labs(x = "RCT subjects (sorted by posterior mean CATE)",
       y = "CATE (log time)",
       title = "Subject-level CATE — RCT subjects only") +
  theme_minimal()

p_cate_all <- ggplot(cate_all_df, aes(x = rank, ymin = lo, ymax = hi)) +
  geom_linerange(colour = band_colour, alpha = 0.45) +
  geom_point(aes(y = mean), colour = mean_colour, size = 0.45) +
  geom_hline(yintercept = 0, linetype = "dashed", colour = "grey40") +
  facet_wrap(~ model, scales = "free_x") +
  labs(x = "All subjects, RCT + MACS combined (sorted by posterior mean CATE)",
       y = "CATE (log time)",
       title = "Subject-level CATE — combined RCT + MACS cohort") +
  theme_minimal()

## --- HDP residual diagnostics (M2) --------------------------------------
hdp_traces <- function(fit_M2) {
  patches <- list()
  if (!is.null(fit_M2$dp_gamma)) {
    df <- data.frame(iter = seq_along(fit_M2$dp_gamma),
                     gamma = fit_M2$dp_gamma)
    patches$gamma <- ggplot(df, aes(iter, gamma)) + geom_line(linewidth = 0.3) +
      labs(title = "Top-level concentration gamma",
           x = "MCMC iteration", y = expression(gamma)) +
      theme_minimal()
  }
  if (!is.null(fit_M2$dp_mass)) {
    n_g <- length(fit_M2$dp_mass)
    mass_df <- do.call(rbind, lapply(seq_len(n_g), function(g) {
      data.frame(iter = seq_along(fit_M2$dp_mass[[g]]),
                 mass = fit_M2$dp_mass[[g]],
                 source = if (g == 1L) "RWD" else "RCT")
    }))
    patches$mass <- ggplot(mass_df, aes(iter, mass, colour = source)) +
      geom_line(linewidth = 0.3) +
      labs(title = "Per-source concentration M_s",
           x = "MCMC iteration", y = expression(M[s])) +
      theme_minimal() + theme(legend.title = element_blank())
  }
  if (!is.null(fit_M2$dp_sigma_g)) {
    n_g <- length(fit_M2$dp_sigma_g)
    sg_df <- do.call(rbind, lapply(seq_len(n_g), function(g) {
      data.frame(iter = seq_along(fit_M2$dp_sigma_g[[g]]),
                 sigma_s = fit_M2$dp_sigma_g[[g]],
                 source = if (g == 1L) "RWD" else "RCT")
    }))
    patches$sigma_s <- ggplot(sg_df, aes(iter, sigma_s, colour = source)) +
      geom_line(linewidth = 0.3) +
      labs(title = "Per-source residual scale sigma_s",
           x = "MCMC iteration", y = expression(sigma[s])) +
      theme_minimal() + theme(legend.title = element_blank())
  }
  if (!is.null(fit_M2$dp_mix_prop)) {
    n_g <- length(fit_M2$dp_mix_prop)
    occ_df <- do.call(rbind, lapply(seq_len(n_g), function(g) {
      P <- fit_M2$dp_mix_prop[[g]]
      data.frame(iter = seq_len(nrow(P)),
                 n_occupied = rowSums(P > 0.01),
                 source = if (g == 1L) "RWD" else "RCT")
    }))
    patches$occ <- ggplot(occ_df, aes(iter, n_occupied, colour = source)) +
      geom_line(linewidth = 0.3) +
      labs(title = "Occupied HDP components per source (pi > 0.01)",
           x = "MCMC iteration", y = "# components") +
      theme_minimal() + theme(legend.title = element_blank())
  }
  patches
}
hdp <- hdp_traces(fit_M2)

## --- Assemble PDF -------------------------------------------------------
pdf_path <- file.path(out_dir, "analysis_figures.pdf")
cairo_pdf(pdf_path, width = 11, height = 7.5, onefile = TRUE)
## Page 1: ATE posterior density (log-time scale): M1 vs M2.
print(p_ate)
## Page 2: subject-level CATE on RCT subjects (M1, M2).
print(p_cate_rct)
## Page 3: subject-level CATE on the combined RCT + MACS cohort (M1, M2);
## M1's MACS predictions come from passing MACS as the test set.
print(p_cate_all)
## Page 4: HDP-scale residual diagnostics (M2)
if (length(hdp) > 0) {
  print(wrap_plots(hdp, ncol = 2) +
        plot_annotation(title = "HDP-scale residual diagnostics (M2)"))
}
invisible(dev.off())
cat("\nWrote ", pdf_path, "\n", sep = "")

################################################################################
## 7. Save full posterior + settings
################################################################################

saveRDS(
  list(
    fit_M1 = fit_M1,
    fit_M2 = fit_M2,
    rct    = rct,
    macs   = macs,
    d_fuse = d_fuse,
    src_M2 = src_M2,
    ate_M1 = ate_M1,
    ate_M2 = ate_M2,
    settings = list(
      cohort           = "trial-aligned (male RCT, MACS CD4 200-500)",
      endpoint         = "RCT composite EFS; MACS v5 (death-only)",
      harmCovars       = harmCovars,
      seed             = seed,
      N_burn           = nBurn,
      N_post           = nPost,
      trees            = c(sh = nTreesSh, d = nTreesD,
                           tau = nTreesTau, c = nTreesC,
                           M1_control = nTreesM1Control,
                           M1_treat   = nTreesM1Treat),
      paper_k          = kPaper,
      code_k           = kCodeUniform,
      error_dist_M2    = "source_hdp_scale",
      error_truncation = truncK
    )),
  file = file.path(out_dir, "fits.rds"))
cat("Wrote ", file.path(out_dir, "fits.rds"), "\n", sep = "")

################################################################################
## 8. Descriptive statistics (manuscript Table 1 + cohort paragraph)
##
## Prints to stdout only; nothing is saved.  Run interactively and copy
## numbers into ANALYSIS.tex.  Reports:
##   (a) median event-free survival (KM) by treatment and source, and
##       median follow-up (reverse-KM) per source;
##   (b) censoring fractions: right-censoring in the RCT; exact /
##       interval / right-censoring in the cohort;
##   (c) baseline covariate summaries by source (mean (SD), median [IQR]
##       for continuous; counts and % for race).
################################################################################

suppressPackageStartupMessages(library(survival))

cat("\n", strrep("=", 70),
    "\n  DESCRIPTIVE STATISTICS (Table 1 / cohort paragraph)\n",
    strrep("=", 70), "\n", sep = "")

## --- (a) Median EFS by arm and source, plus median follow-up -------------
rct_t  <- transform(rct,  t_years = exp(log_time))
macs_t <- transform(macs, t_years = exp(log_time))

median_efs_by_arm <- function(d, label) {
  fit <- survfit(Surv(t_years, status) ~ treat, data = d)
  tbl <- summary(fit)$table
  rownames(tbl) <- ifelse(grepl("=0$", rownames(tbl)),
                          "ZDV mono", "Combination")
  cat(sprintf("\n%s - median EFS (years) by treatment:\n", label))
  print(round(tbl[, c("records", "events", "median",
                      "0.95LCL", "0.95UCL")], 2))
}
median_efs_by_arm(rct_t,  "RCT  (ACTG175, trial-aligned)")
median_efs_by_arm(macs_t, "RWD  (MACS, CD4 200-500)")

## Reverse-KM (Schemper-Smith) gives median follow-up.
median_followup <- function(d, label) {
  fit <- survfit(Surv(t_years, 1 - status) ~ 1, data = d)
  m <- summary(fit)$table["median"]
  cat(sprintf("  %s: median follow-up = %.2f y (reverse-KM)\n", label, m))
}
cat("\nMedian follow-up:\n")
median_followup(rct_t,  "RCT")
median_followup(macs_t, "RWD")

## --- (b) Censoring fractions per source ----------------------------------
cat("\nCensoring fractions:\n")
cat(sprintf("  RCT  (right-censoring only):\n"))
cat(sprintf("       events       %4d / %d (%.1f%%)\n",
            sum(rct$status == 1), nrow(rct),
            100 * mean(rct$status == 1)))
cat(sprintf("       right-cens   %4d / %d (%.1f%%)\n",
            sum(rct$status == 0), nrow(rct),
            100 * mean(rct$status == 0)))

ex <- macs$ic_indicator == 0 & macs$status_ic == 1
ic <- macs$ic_indicator == 1
rc <- macs$ic_indicator == 0 & macs$status_ic == 0
cat(sprintf("  RWD  (deaths encoded as interval-censored under v5):\n"))
cat(sprintf("       exact events %4d / %d (%.1f%%)\n",
            sum(ex), nrow(macs), 100 * mean(ex)))
cat(sprintf("       int-cens     %4d / %d (%.1f%%)\n",
            sum(ic), nrow(macs), 100 * mean(ic)))
cat(sprintf("       right-cens   %4d / %d (%.1f%%)\n",
            sum(rc), nrow(macs), 100 * mean(rc)))

## --- (c) Baseline covariate summaries by source --------------------------
contvars <- intersect(c("age", "cd4", "cd8",
                        "anchor_year", "prior_art_years"),
                      harmCovars)
catvars  <- intersect("race", harmCovars)

cont_summary <- function(d, label) {
  m <- sapply(d[, contvars, drop = FALSE], function(x) {
    c(mean   = mean(x, na.rm = TRUE),
      sd     = sd(x, na.rm = TRUE),
      median = median(x, na.rm = TRUE),
      q25    = unname(quantile(x, 0.25, na.rm = TRUE)),
      q75    = unname(quantile(x, 0.75, na.rm = TRUE)))
  })
  cat(sprintf("\n%s - continuous covariates (mean (SD), median [IQR]):\n",
              label))
  print(round(t(m), 2))
}
cont_summary(rct,  "RCT")
cont_summary(macs, "RWD")

cat_summary <- function(d, label) {
  for (v in catvars) {
    tab  <- table(d[[v]])
    prop <- round(100 * prop.table(tab), 1)
    cat(sprintf("\n%s - %s (0 = white, 1 = non-white):\n", label, v))
    print(rbind(n   = as.integer(tab),
                pct = as.numeric(prop)))
  }
}
cat_summary(rct,  "RCT")
cat_summary(macs, "RWD")

################################################################################
## 9. Manuscript caterpillar (M2, subject-level acceleration factor)
##
## Fusion-fit subject-level acceleration factor exp{tau(x_i)} for every
## subject in the combined cohort, sorted by posterior mean, with 95%
## credible-interval bars and colour by source (RCT, RWD).  Style mirrors
## the KM manuscript figure (Times via cairo_pdf, large base_size, dark
## green for RCT to match the KM palette, firebrick for RWD as in the
## simulation panels).  Written to notes/general/figures/.
################################################################################

## Okabe-Ito colourblind-safe palette, shared with every other figure in the
## paper: blue #0072B2 for the trial, vermillion #D55E00 for the real-world
## data.  RCT carries the trial-aligned look; RWD stands apart.
src_cols <- c(RCT = "#0072B2", RWD = "#D55E00")

## Caterpillar helper -- transposed orientation (rank on the vertical
## axis, acceleration factor on the longer horizontal axis), Times font
## via cairo_pdf, source-coloured CrI bars with a black posterior-mean
## dot overlaid.  `samples` is a B x n posterior matrix on the log-time
## scale; `src` is the length-n source indicator (1 = RCT, 0 = RWD)
## aligned with the columns of `samples`.
make_caterpillar <- function(samples, src, out_path,
                             width = 14, height = 8, xlim = NULL) {
  af      <- exp(samples)
  mean_af <- colMeans(af)
  lo_af   <- apply(af, 2, quantile, 0.025)
  hi_af   <- apply(af, 2, quantile, 0.975)
  ord     <- order(mean_af)
  src_lab <- factor(ifelse(src == 1L, "RCT", "RWD"),
                    levels = c("RCT", "RWD"))
  df <- data.frame(
    rank   = seq_along(ord),
    mean   = mean_af[ord],
    lo     = lo_af[ord],
    hi     = hi_af[ord],
    source = src_lab[ord])
  p <- ggplot(df, aes(y = rank, xmin = lo, xmax = hi, colour = source)) +
    geom_linerange(alpha = 0.45, linewidth = 0.6) +
    geom_point(aes(x = mean), colour = "black", size = 0.6) +
    geom_vline(xintercept = 1, linetype = "dashed", colour = "grey40",
               linewidth = 0.6) +
    scale_colour_manual(values = src_cols) +
    coord_cartesian(xlim = xlim) +   # NULL = automatic; c(lo, hi) to fix the AF axis
    scale_y_continuous(breaks = NULL) +
    labs(y      = "Patients (ordered by posterior mean)",
         x      = expression("Acceleration factor"),
         colour = NULL) +
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
  print(p)                                          # active device (RStudio pane)
  cairo_pdf(out_path, width = width, height = height)
  print(p)                                          # also to PDF
  invisible(dev.off())
  cat("Wrote ", out_path, "\n", sep = "")
}

fig_dir <- file.path(proj, "notes", "general", "figures")
dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)

## M2 (fusion): caterpillar over the combined cohort; src_M2 aligns
## with the columns of M2_combined_samples.
make_caterpillar(
  M2_combined_samples, src_M2,
  file.path(fig_dir, "cate_caterpillar_manuscript.pdf"),
  xlim = NULL)   # e.g. xlim = c(0.8, 3) to fix the AF axis

## M1 (RCT-only): caterpillar over the same combined cohort.  M1_combined
## is the column-bind of train (RCT) and test (MACS) predictions, so the
## source indicator is the obvious block vector -- 1s for the RCT block,
## then 0s for the MACS block.  MACS bars come from the test-set pass
## (MACS rows are NOT in M1's likelihood).
src_M1 <- c(rep(1L, ncol(fit_M1$train_predictions_sample_treat)),
            rep(0L, ncol(fit_M1$test_predictions_sample_treat)))
make_caterpillar(
  M1_combined_samples, src_M1,
  file.path(fig_dir, "cate_caterpillar_M1_manuscript.pdf"),
  xlim = NULL)   # e.g. xlim = c(0.8, 3) to fix the AF axis

################################################################################
## 10. Linear projections of the CATE / acceleration factor
##
## Pushforward of the per-subject treatment effect onto a linear basis in
## the harmonised covariates (Woody, Carvalho & Murray 2020), with the
## Bayesian-bootstrap recipe (Rubin 1981) injecting covariate-distribution
## uncertainty per posterior draw.  Reused from examples/projection_demo.R:
##
##   - M2 (FusionForest): the exported FusionForests::fusion_projection().
##   - M1 (CSF): the inline project_tau_matrix() helper, since
##               fusion_projection() requires fit$meta (FusionForest-only).
##
## Both fits are projected on the *same* combined cohort (RCT + MACS), so
## coefficients are directly comparable: shift = M2 - M1 is the net
## effect of bringing the cohort into the likelihood.
################################################################################

## Inline CSF projection helper, copied verbatim from
## examples/projection_demo.R (lines ~139-160).  We keep a local copy so
## this script stays self-contained.
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

## Basis: linear in the six harmonised covariates.  Add interactions or
## splines here if the manuscript needs richer modifier shapes; the
## comparison below is invariant to the choice as long as M1 and M2 use
## the same basis.
proj_basis <- as.formula(paste("~", paste(harmCovars, collapse = " + ")))

## Evaluation rows -- same combined-cohort target for both fits.
##   M2: train_predictions_sample_treat covers every d_fuse row.
##   M1: predictions for RCT come from the train block, MACS from the
##       test block; the corresponding covariates are rbind(rct, macs)
##       in the same RCT-then-MACS order as M1_combined_samples.
X_eval_M2 <- d_fuse[,            harmCovars, drop = FALSE]
X_eval_M1 <- rbind(rct, macs)[,  harmCovars, drop = FALSE]

set.seed(seed)
g_af_M2   <- fusion_projection(
  fit_M2, proj_basis, X_eval = X_eval_M2,
  which = "train", target = "af", af_scale = "multiplicative",
  weights = "bayesian_bootstrap")
g_cate_M2 <- fusion_projection(
  fit_M2, proj_basis, X_eval = X_eval_M2,
  which = "train", target = "cate",
  weights = "bayesian_bootstrap")

set.seed(seed)
g_af_M1   <- project_tau_matrix(M1_combined_samples, X_eval_M1,
                                proj_basis, target = "af")
g_cate_M1 <- project_tau_matrix(M1_combined_samples, X_eval_M1,
                                proj_basis, target = "cate")

## Deviation forest g(x) -- four-forest only.  This is the RWD-specific
## shift of the control surface relative to the shared baseline mu(x);
## in the AFT decomposition the contribution at an RWD observation is
## (1 - S) * g(x).  Projecting it onto the covariates tells us which
## covariates drive the RCT-vs-RWD baseline gap.
##
## fit_M2$train_predictions_sample_deviation is stored over the RWD
## training subset only (the RCT rows zero out (1-S)*g and are not
## represented), so the matching X_eval is d_fuse[src_M2 == 0L, ].  We
## keep the same basis as the tau projections; the coefficients live
## on the log-time scale ("CATE-like" additive interpretation).
g_dev_samples <- fit_M2$train_predictions_sample_deviation
X_eval_dev    <- d_fuse[src_M2 == 0L, harmCovars, drop = FALSE]
if (!is.null(g_dev_samples) && ncol(g_dev_samples) == nrow(X_eval_dev)) {
  set.seed(seed)
  g_dev_M2 <- project_tau_matrix(g_dev_samples, X_eval_dev,
                                 proj_basis, target = "cate")
} else {
  g_dev_M2 <- NULL
  warning("fit_M2$train_predictions_sample_deviation is missing or its ",
          "column count does not match the RWD training rows; ",
          "skipping the deviation-forest projection.")
}

## Tidy summary: posterior mean and 95% CrI per coefficient per model.
proj_summary <- function(G, model, target) {
  data.frame(
    model  = model,
    target = target,
    coef   = colnames(G),
    mean   = colMeans(G),
    lo     = apply(G, 2, quantile, 0.025),
    hi     = apply(G, 2, quantile, 0.975),
    stringsAsFactors = FALSE)
}
proj_tbl <- rbind(
  proj_summary(g_cate_M1, "M1 (RCT only)", "CATE (log time)"),
  proj_summary(g_cate_M2, "M2 (Fusion)",   "CATE (log time)"),
  proj_summary(g_af_M1,   "M1 (RCT only)", "AF (multiplicative)"),
  proj_summary(g_af_M2,   "M2 (Fusion)",   "AF (multiplicative)"))
if (!is.null(g_dev_M2)) {
  proj_tbl <- rbind(
    proj_tbl,
    proj_summary(g_dev_M2, "M2 (Fusion)", "Deviation g(x) (log time, RWD)"))
}

cat("\n", strrep("=", 70),
    "\n  LINEAR PROJECTION COEFFICIENTS (Woody-Carvalho-Murray)\n",
    "  basis: ", deparse(proj_basis), "\n",
    strrep("=", 70), "\n", sep = "")
proj_tbl$pretty <- sprintf("%+6.3f  [%+6.3f, %+6.3f]",
                           proj_tbl$mean, proj_tbl$lo, proj_tbl$hi)
print(
  reshape(
    proj_tbl[, c("target", "coef", "model", "pretty")],
    idvar     = c("target", "coef"),
    timevar   = "model",
    direction = "wide"),
  row.names = FALSE)

writeLines(
  capture.output(print(proj_tbl[, c("model", "target", "coef",
                                    "mean", "lo", "hi")],
                       row.names = FALSE, digits = 3)),
  file.path(out_dir, "analysis_projection.txt"))

## Manuscript forest plot: AF-scale coefficients, M1 vs M2, with the
## intercept dropped (it absorbs the average level; the slopes are the
## modifier story).  Same Times / cairo style as the caterpillars.
g_plot <- rbind(
  proj_summary(g_af_M1, "M1 (RCT only)", "AF"),
  proj_summary(g_af_M2, "M2 (Fusion)",   "AF"))
g_plot <- g_plot[g_plot$coef != "(Intercept)", ]
g_plot$coef  <- factor(g_plot$coef, levels = rev(harmCovars))
g_plot$model <- factor(g_plot$model,
                       levels = c("M1 (RCT only)", "M2 (Fusion)"))

# Okabe-Ito, same assignment as every other figure: trial green, Fusion orange.
model_cols <- c("M1 (RCT only)" = "#009E73", "M2 (Fusion)" = "#E69F00")

p_proj <- ggplot(g_plot,
                 aes(y = coef, x = mean, xmin = lo, xmax = hi,
                     colour = model)) +
  geom_vline(xintercept = 0, linetype = "dashed", colour = "grey40",
             linewidth = 0.6) +
  geom_pointrange(position = position_dodge(width = 0.55),
                  size = 0.8, linewidth = 1.0) +
  scale_colour_manual(values = model_cols) +
  labs(x      = expression("Projection coefficient on " *
                            exp*"{"*tau*"(x)} (per unit covariate)"),
       y      = NULL,
       colour = NULL) +
  guides(colour = guide_legend(
    override.aes = list(linewidth = 4, size = 0.9))) +
  theme_minimal(base_size = 24) +
  theme(text             = element_text(family = "Times"),
        legend.position  = "top",
        legend.text      = element_text(size = 24, family = "Times"),
        legend.key.width = unit(40, "pt"),
        legend.key.height = unit(16, "pt"),
        legend.spacing.x = unit(8, "pt"),
        panel.background = element_rect(fill = "white", colour = NA))

out_proj <- file.path(fig_dir, "projection_manuscript.pdf")
print(p_proj)                                       # active device (RStudio pane)
cairo_pdf(out_proj, width = 14, height = 8)
print(p_proj)                                       # also to PDF
invisible(dev.off())
cat("Wrote ", out_proj, "\n", sep = "")

## Companion forest plot for the deviation-forest projection.  Different
## estimand (additive log-time shift, not a treatment effect) and
## different target population (RWD rows only), so it lives in its own
## figure rather than being dodged in with the tau coefficients.
if (!is.null(g_dev_M2)) {
  g_dev_plot <- proj_summary(g_dev_M2, "M2 deviation g(x)",
                             "Deviation (log time)")
  g_dev_plot <- g_dev_plot[g_dev_plot$coef != "(Intercept)", ]
  g_dev_plot$coef <- factor(g_dev_plot$coef, levels = rev(harmCovars))

  p_proj_dev <- ggplot(g_dev_plot,
                       aes(y = coef, x = mean, xmin = lo, xmax = hi)) +
    geom_vline(xintercept = 0, linetype = "dashed", colour = "grey40",
               linewidth = 0.6) +
    geom_pointrange(colour = "#33a02c",                 # dark green
                    size = 0.8, linewidth = 1.0) +
    labs(x = expression("Projection coefficient on g(x)"~
                        "(log-time shift per unit covariate)"),
         y = NULL) +
    theme_minimal(base_size = 24) +
    theme(text             = element_text(family = "Times"),
          panel.background = element_rect(fill = "white", colour = NA))

  out_proj_dev <- file.path(fig_dir, "projection_deviation_manuscript.pdf")
  print(p_proj_dev)                                       # active device
  cairo_pdf(out_proj_dev, width = 14, height = 8)
  print(p_proj_dev)                                       # also to PDF
  invisible(dev.off())
  cat("Wrote ", out_proj_dev, "\n", sep = "")
}

################################################################################
## 11. Fit-the-fit (Bargagli-Stoffi et al. 2023; Foster et al. 2011)
##
## Distil the FusionForest CATE posterior into an interpretable shallow
## tree:
##   (1) treat the posterior-mean log-time CATE as a pseudo-outcome and
##       fit a CART of depth 3 (minbucket 30) on the harmonised
##       covariates -- the partition is the "interpretable summary";
##   (2) for each leaf, pull the *posterior* of the within-leaf mean
##       CATE across draws, so leaf uncertainty comes from the original
##       Bayesian posterior, not a second-stage inference (no double
##       use of the data).
## We report both the log-time CATE and the acceleration factor
## exp{tau(x)} per leaf.  Manuscript PDF is rendered via rpart.plot.
################################################################################

suppressPackageStartupMessages(library(rpart))
## Tree-plot back-end:
##   - ggparty (preferred): ggplot-based, lets us put a real ggplot
##     panel inside each leaf (posterior CrI pointrange), Times font
##     via theme().  Depends on partykit; install with
##         install.packages(c("partykit", "ggparty"))
##   - rpart.plot (fallback): polished but base-graphics.
##   - plot.rpart() (last resort): always available, no styling.
have_ggparty    <- requireNamespace("partykit",   quietly = TRUE) &&
                   requireNamespace("ggparty",    quietly = TRUE)
have_rpart_plot <- requireNamespace("rpart.plot", quietly = TRUE)
if (!have_ggparty && !have_rpart_plot) {
  message("Neither ggparty nor rpart.plot installed -- using base ",
          "plot.rpart() fallback.  For the nicer figure run\n",
          "    install.packages(c(\"partykit\", \"ggparty\"))")
} else if (!have_ggparty) {
  message("ggparty not installed -- using rpart.plot fallback.  For ",
          "the nicer ggplot-style tree run\n",
          "    install.packages(c(\"partykit\", \"ggparty\"))")
}

## Pseudo-outcome: posterior-mean log-time CATE per subject (M2,
## combined cohort).  Covariates are harmCovars in the matched row
## order of d_fuse (same alignment used in Sections 9 and 10).
X_fit     <- d_fuse[, harmCovars, drop = FALSE]
tau_hat   <- colMeans(M2_combined_samples)
tree_data <- cbind(tau = tau_hat, X_fit)

set.seed(seed)
fit_tree <- rpart(tau ~ ., data = tree_data, method = "anova",
                  control = rpart.control(maxdepth = 3, cp = 0.01,
                                          minbucket = 30))

cat("\n", strrep("=", 70),
    "\n  FIT-THE-FIT: shallow CART on posterior-mean log-time CATE (M2)\n",
    strrep("=", 70), "\n", sep = "")
print(fit_tree)

## Per-leaf posterior summary.  fit_tree$where gives the row index of
## the leaf in fit_tree$frame for each observation; its row name is the
## actual node id used by rpart (so it lines up with the tree plot).
leaf_row <- fit_tree$where
node_id  <- as.integer(rownames(fit_tree$frame)[leaf_row])
leaves   <- sort(unique(node_id))
samp     <- M2_combined_samples         # B x n on the log-time scale
af_samp  <- exp(samp)                   # B x n on the AF scale

leaf_summary <- do.call(rbind, lapply(leaves, function(lf) {
  rows  <- which(node_id == lf)
  draws_tau <- rowMeans(samp[,    rows, drop = FALSE])  # B-vec: mean CATE in leaf
  draws_af  <- rowMeans(af_samp[, rows, drop = FALSE])
  data.frame(
    node      = lf,
    n         = length(rows),
    tau_mean  = mean(draws_tau),
    tau_lo    = quantile(draws_tau, 0.025, names = FALSE),
    tau_hi    = quantile(draws_tau, 0.975, names = FALSE),
    af_mean   = mean(draws_af),
    af_lo     = quantile(draws_af, 0.025, names = FALSE),
    af_hi     = quantile(draws_af, 0.975, names = FALSE),
    stringsAsFactors = FALSE)
}))
cat("\nLeaf summary (posterior mean and 95% CrI per leaf):\n")
print(leaf_summary, row.names = FALSE, digits = 3)

writeLines(
  capture.output(print(leaf_summary, row.names = FALSE, digits = 3)),
  file.path(out_dir, "analysis_fit_the_fit.txt"))

## Manuscript figure.  rpart.plot uses base graphics, so we set
## family = "Times" via par() for the active device and pass
## family = "Times" to cairo_pdf so the PDF embeds Times directly.
##
## Each leaf box shows:
##   line 1 : exp(mean tau_hat in leaf) -- acceleration factor
##   line 2 : "[lo, hi]" -- the 95% posterior CrI of the leaf-mean AF
##            (from leaf_summary, so it is the original posterior, not
##            a tree-driven recomputation)
##   line 3 : "n = ..." -- subjects in the leaf
## Internal split nodes show "var < cut" with rpart.plot defaults.
## Per-leaf posterior draws of the leaf-mean acceleration factor.
## leaf_summary collapses these to (mean, lo, hi); we keep the full
## B-vectors here so the ggparty terminal panels can show a real
## interval (and could show a density if we wanted).
leaf_draws_af <- lapply(leaves, function(lf) {
  rows <- which(node_id == lf)
  rowMeans(af_samp[, rows, drop = FALSE])     # B-vector
})
names(leaf_draws_af) <- as.character(leaves)

## --- (a) ggplot-style tree via partykit + ggparty ----------------------

build_ggparty_tree <- function() {
  party_tree <- partykit::as.party(fit_tree)
  ## Re-derive leaf assignment from the *party* object so the leaf ids
  ## match ggparty's `id` aesthetic (partykit re-numbers nodes
  ## depth-first; rpart's heap numbering does not survive as.party()).
  party_node <- predict(party_tree, newdata = tree_data, type = "node")
  leaf_ids   <- sort(unique(party_node))
  leaf_df <- do.call(rbind, lapply(leaf_ids, function(lf) {
    rows  <- which(party_node == lf)
    draws <- rowMeans(af_samp[, rows, drop = FALSE])
    data.frame(id = lf, n = length(rows),
               af = mean(draws),
               lo = quantile(draws, 0.025, names = FALSE),
               hi = quantile(draws, 0.975, names = FALSE))
  }))

  ## Clean, no-frills style: straight grey edges, plain split labels on
  ## the branches, white leaf boxes with a thin border and two lines
  ## (AF + CrI; sample size).
  ggparty::ggparty(party_tree, terminal_space = 0.35) +
    ggparty::geom_edge(colour = "grey55") +
    ggparty::geom_edge_label(size = 5, family = "Times",
                             colour = "grey30") +
    ggparty::geom_node_label(
      ggplot2::aes(label = splitvar),
      ids        = "inner",
      label.size = 0,
      fill       = "white",
      colour     = "black",
      size       = 7,
      family     = "Times") +
    ggparty::geom_node_label(
      line_list = list(
        ggplot2::aes(label = sprintf("AF = %.2f [%.2f, %.2f]",
                       leaf_df$af[match(id, leaf_df$id)],
                       leaf_df$lo[match(id, leaf_df$id)],
                       leaf_df$hi[match(id, leaf_df$id)])),
        ggplot2::aes(label = paste0("n = ",
                       leaf_df$n[match(id, leaf_df$id)]))),
      line_gpar = list(
        list(size = 6,                 family = "Times"),
        list(size = 5, col = "grey30", family = "Times")),
      ids        = "terminal",
      label.size = 0.5,
      label.r    = grid::unit(0, "lines"),  # square corners
      fill       = "white",
      colour     = "black") +
    ggplot2::theme(text = ggplot2::element_text(family = "Times"))
}

## --- (b) rpart.plot fallback (with the polished arg set from before) ---

leaf_label_lookup <- setNames(
  sprintf("AF = %.2f\n[%.2f, %.2f]\nn = %d",
          leaf_summary$af_mean, leaf_summary$af_lo,
          leaf_summary$af_hi,   leaf_summary$n),
  as.character(leaf_summary$node))

node_label_fun <- function(x, labs, digits, varlen) {
  node_ids <- as.integer(rownames(x$frame))
  is_leaf  <- x$frame$var == "<leaf>"
  out      <- labs
  out[is_leaf] <- leaf_label_lookup[as.character(node_ids[is_leaf])]
  out
}

draw_tree_rpart <- function() {
  if (have_rpart_plot) {
    rpart.plot::rpart.plot(
      fit_tree, type = 5, extra = 0, under = FALSE,
      fallen.leaves = TRUE, branch = 0.3, branch.lwd = 2.2,
      branch.col = "grey55", box.palette = "Blues",
      shadow.col = "grey80", tweak = 1.6, split.cex = 1.05,
      faclen = 0, varlen = 0, digits = 3, roundint = FALSE,
      node.fun = node_label_fun, main = NULL)
  } else {
    plot(fit_tree, uniform = TRUE, margin = 0.15)
    text(fit_tree, use.n = TRUE, all = TRUE, cex = 0.9)
  }
}

## --- (c) Render: active device first, then PDF ------------------------

out_tree <- file.path(fig_dir, "fit_the_fit_manuscript.pdf")
if (have_ggparty) {
  p_tree <- build_ggparty_tree()
  print(p_tree)                                # active device
  cairo_pdf(out_tree, width = 14, height = 8)
  print(p_tree)                                # also to PDF
  invisible(dev.off())
} else {
  op <- par(family = "Times")
  draw_tree_rpart()
  par(op)
  cairo_pdf(out_tree, width = 14, height = 8, family = "Times")
  op <- par(family = "Times")
  draw_tree_rpart()
  par(op)
  invisible(dev.off())
}
cat("Wrote ", out_tree, "\n", sep = "")

################################################################################
## 12. Posterior probability of benefit per patient
##     (Henderson, Louis, Rosner & Varadhan 2020 -- individualised benefit)
##
## pi_i = P{ tau(x_i) > 0 | data } -- the posterior probability that
## patient i gains from combination therapy.  Computed over the *trial*
## population (RCT subjects only) for both fits so the two columns of
## tab:hte-summary in ANALYSIS.tex are on the same target.
##
## Bins follow the manuscript table:
##   (0.99, 1]     strong evidence of benefit
##   (0.95, 0.99]
##   (0.75, 0.95]
##   (0.25, 0.75]  inconclusive
##   [0,    0.25]  evidence of harm
##
## M1 uses fit_M1$train_predictions_sample_treat (RCT rows = trained).
## M2 uses M2_combined_samples[, eval_idx_M2] (RCT rows of d_fuse;
##         already pre-extracted as M2_rct_samples in Section 5).
################################################################################

pi_M1 <- colMeans(M1_rct_samples > 0)
pi_M2 <- colMeans(M2_rct_samples > 0)

bins       <- c(-Inf, 0.25, 0.75, 0.95, 0.99, Inf)
bin_labels <- c("[0, 0.25]", "(0.25, 0.75]", "(0.75, 0.95]",
                "(0.95, 0.99]", "(0.99, 1]")
bin_M1 <- cut(pi_M1, breaks = bins, labels = bin_labels,
              include.lowest = TRUE, right = TRUE)
bin_M2 <- cut(pi_M2, breaks = bins, labels = bin_labels,
              include.lowest = TRUE, right = TRUE)

## Always tabulate against the full label set so a zero-count bin still
## appears as 0.00 instead of being silently dropped.
frac <- function(bin) {
  tab <- table(factor(bin, levels = bin_labels))
  prop.table(tab)
}
frac_M1 <- frac(bin_M1)
frac_M2 <- frac(bin_M2)

pi_tbl <- data.frame(
  bin        = bin_labels,
  fusion     = sprintf("%.3f", as.numeric(frac_M2)),
  trial_only = sprintf("%.3f", as.numeric(frac_M1)),
  n_fusion   = as.integer(table(factor(bin_M2, levels = bin_labels))),
  n_trial    = as.integer(table(factor(bin_M1, levels = bin_labels))),
  stringsAsFactors = FALSE)
## Reorder to match the manuscript table (highest pi first).
pi_tbl <- pi_tbl[match(c("(0.99, 1]", "(0.95, 0.99]", "(0.75, 0.95]",
                         "(0.25, 0.75]", "[0, 0.25]"), pi_tbl$bin), ]

cat("\n", strrep("=", 70),
    "\n  POSTERIOR PROBABILITY OF BENEFIT (RCT population)\n",
    "    pi_i = P{ tau(x_i) > 0 | data }; n_rct = ",
    ncol(M1_rct_samples), "\n",
    strrep("=", 70), "\n", sep = "")
print(pi_tbl, row.names = FALSE)

## LaTeX-ready snippet for tab:hte-summary in ANALYSIS.tex.
latex_lines <- c(
  "    $\\pi_i$        & Fusion & Trial-only \\\\",
  "    \\midrule",
  sprintf("    $%-12s$ & %s & %s \\\\",
          pi_tbl$bin, pi_tbl$fusion, pi_tbl$trial_only))
cat("\nLaTeX rows for tab:hte-summary (paste into ANALYSIS.tex):\n")
cat(paste(latex_lines, collapse = "\n"), "\n")

writeLines(
  c(capture.output(print(pi_tbl, row.names = FALSE)),
    "",
    "LaTeX rows:",
    latex_lines),
  file.path(out_dir, "analysis_pi_benefit.txt"))


################################################################################
## 13. Deep-learning competitor (deepAFT) -- black-box benchmark
##
## Black-box foil for the "structured statistical ML vs generic black-box"
## framing.  A deep accelerated failure time network (deepAFT, `dnn`
## package) is fitted to the same ACTG175 + MACS cohort as an S-learner on
##   (a) the RCT only          -- the trial-only analogue of M1, and
##   (b) the RCT + MACS pool    -- the fusion analogue of M2,
## with a source indicator S (1 = RCT, 0 = MACS) and the CATE evaluated at
## S = 1 (the transported / trial estimand).
##
## The architecture and the tuning protocol are IDENTICAL to the
## simulation competitor (simulations/main/competitors/sim_surv_v12_deepaft.R):
## a two-hidden-layer 8-6-1 net, ReLU on the hidden layers and identity
## ("idu") on the real-valued log-time output, with the learning rate,
## momentum (alpha) and L2 penalty (lambda) tuned by 10-fold cross-
## validation on the C-index (hyperTuning); the 8-6-1 architecture is held
## fixed (node = FALSE).
##
## deepAFT consumes only right-censored data, so the interval-censored MACS
## deaths are approximated by the right-censored last-lab-year encoding
## already held in `log_time` / `status` (the same encoding M1 sees).  This
## approximation is stated as a limitation in ANALYSIS_SM.tex.  The point
## estimate is the log-time score difference mu(x, 1) - mu(x, 0) = tau(x);
## intervals come from a stratified nonparametric bootstrap (deepAFT has no
## posterior), the frequentist analogue of the BART credible interval.
## Every subject of the combined cohort is evaluated, in the rct-then-macs
## order of M1_combined_samples / M2_combined_samples.
################################################################################

if (!requireNamespace("dnn", quietly = TRUE)) {
  message("Package 'dnn' not installed; skipping the deepAFT competitor. ",
          "Install with install.packages('dnn') to run this section.")
} else {
  library(dnn)
  library(survival)

  ## --- Settings (match the simulation competitor) ------------------------
  ddB <- 100L   # bootstrap refits per estimator (deepAFT is slow; lower to
                # speed up at the cost of noisier intervals)
  ddK <- 10L    # cross-validation folds for hyperparameter tuning
  ddR <- 20L    # random-search draws in hyperTuning

  ## --- deepAFT helpers (verbatim from the simulation competitor) ---------
  ## predict.deepAFT$predictors is the AFT log-time score mu (larger =>
  ## longer survival); differencing over A gives the CATE on the log-time
  ## scale, i.e. the log acceleration factor.
  dd_lp <- function(pr)
    if (is.list(pr) && !is.null(pr$predictors)) as.numeric(pr$predictors)
    else as.numeric(pr)
  ## Two hidden layers (8, 6) ReLU; identity ("idu") output.  Matches the
  ## deepAFT paper's application and the simulation competitor exactly.
  dd_model <- function(p)
    dNNmodel(units = c(8L, 6L, 1L),
             activation = c("relu", "relu", "idu"), input_shape = p)

  ## S-learner: one net with A as an input; CATE = mu(x, 1) - mu(x, 0).
  dd_cate_S <- function(des, control) {
    Z   <- cbind(des$X, A = des$A)
    fit <- deepAFT(as.matrix(Z), Surv(des$time, des$status),
                   model = dd_model(ncol(Z)), control = control)
    p1  <- dd_lp(predict(fit, newdata = cbind(des$Xev, A = 1)))
    p0  <- dd_lp(predict(fit, newdata = cbind(des$Xev, A = 0)))
    p1 - p0
  }

  ## T-learner: a separate net per arm (no A input); CATE = mu_1(x) - mu_0(x).
  dd_mu <- function(time, status, X, newX, control) {
    fit <- deepAFT(as.matrix(X), Surv(time, status),
                   model = dd_model(ncol(X)), control = control)
    dd_lp(predict(fit, newdata = as.matrix(newX)))
  }
  dd_cate_T <- function(des, control) {
    i1 <- des$A == 1; i0 <- des$A == 0
    mu1 <- dd_mu(des$time[i1], des$status[i1], des$X[i1, , drop = FALSE],
                 des$Xev, control)
    mu0 <- dd_mu(des$time[i0], des$status[i0], des$X[i0, , drop = FALSE],
                 des$Xev, control)
    mu1 - mu0
  }

  ## 10-fold CV tuning of lr / alpha / lambda; architecture fixed.  Bounds
  ## are identical to the simulation competitor.
  dd_tune <- function(x, time, status, K = ddK, R = ddR)
    hyperTuning(
      as.matrix(x), Surv(time, status), dd_model(ncol(x)),
      ER = "cindex", method = "BuckleyJames", node = FALSE, K = K, R = R,
      lower = dnnControl(epochs = 300, batch_size = 64, epsilon = 1e-2,
                         lr_rate = 1e-4, alpha = 0.5, lambda = 0),
      upper = dnnControl(lr_rate = 3e-3, alpha = 0.97, lambda = 10))$control

  ## --- Standardise inputs (deepAFT diverges on raw covariate scales) ------
  ## The simulation competitor fed ~N(0,1) covariates and a standardised
  ## log-time response, so the net trained fine.  The raw HIV covariates
  ## (anchor_year ~ 1991, cd8 ~ 1000, cd4 ~ 300, ...) blow up the gradients
  ## (NaN cost, C-index 0.50).  We z-score the covariates and the log-time
  ## response on the combined cohort, fit on that scale, and rescale the
  ## CATE back by the response SD (the centring cancels in the treated-minus-
  ## control contrast).  Scaling constants are fixed on the full cohort so
  ## they stay identical across the bootstrap refits.
  dd_cov_mu <- colMeans(as.matrix(d_fuse[, harmCovars]))
  dd_cov_sd <- apply(as.matrix(d_fuse[, harmCovars]), 2, sd)
  dd_cov_sd[dd_cov_sd == 0] <- 1
  dd_y_mu   <- mean(d_fuse$log_time)
  dd_y_sd   <- sd(d_fuse$log_time)
  dd_Z    <- function(df) scale(as.matrix(df[, harmCovars]),
                                center = dd_cov_mu, scale = dd_cov_sd)
  dd_time <- function(df) exp((df$log_time - dd_y_mu) / dd_y_sd)

  ## --- Designs on the real cohort (right-censored log_time / status) ------
  dd_Xev <- dd_Z(d_fuse)                       # combined cohort, rct then macs
  des_rct <- function(rdf) list(
    time = dd_time(rdf), status = rdf$status,
    X = dd_Z(rdf), A = rdf$treat, Xev = dd_Xev)
  des_pool <- function(fdf, src) list(
    time = dd_time(fdf), status = fdf$status,
    X = cbind(dd_Z(fdf), S = src), A = fdf$treat,
    Xev = cbind(dd_Xev, S = 1L))

  ## --- Tune once per design, reuse for the fit and the bootstrap ---------
  set.seed(seed)
  cat("\n", strrep("=", 70),
      "\n  deepAFT competitor: 10-fold CV tuning (hyperTuning)\n",
      strrep("=", 70), "\n", sep = "")
  dd_ctrl_rct  <- dd_tune(cbind(dd_Z(rct), A = rct$treat),
                          dd_time(rct), rct$status)
  dd_ctrl_pool <- dd_tune(cbind(dd_Z(d_fuse), S = src_M2, A = d_fuse$treat),
                          dd_time(d_fuse), d_fuse$status)
  cat(sprintf("  RCT-only : lr = %.2g, alpha = %.2g, lambda = %.2g\n",
              dd_ctrl_rct$lr_rate, dd_ctrl_rct$alpha, dd_ctrl_rct$lambda))
  cat(sprintf("  Pooled   : lr = %.2g, alpha = %.2g, lambda = %.2g\n",
              dd_ctrl_pool$lr_rate, dd_ctrl_pool$alpha, dd_ctrl_pool$lambda))

  ## --- Build a design for an estimator, with optional bootstrap resample --
  ## RCT design resamples the RCT rows; the pooled design resamples RCT and
  ## MACS rows separately (sizes fixed).  The evaluation set is held fixed.
  dd_n_rct <- nrow(rct); dd_n_macs <- nrow(macs)
  dd_make_des <- function(type, boot = FALSE) {
    if (type == "rct") {
      rdf <- if (boot) rct[sample.int(dd_n_rct, dd_n_rct, TRUE), ] else rct
      des_rct(rdf)
    } else {
      if (!boot) return(des_pool(d_fuse, src_M2))
      i_r <- sample.int(dd_n_rct,  dd_n_rct,  TRUE)
      i_m <- sample.int(dd_n_macs, dd_n_macs, TRUE)
      des_pool(rbind(rct[i_r, ], macs[i_m, ]),
               c(rep(1L, dd_n_rct), rep(0L, dd_n_macs)))
    }
  }

  ## Four estimators: RCT / Pooled design x S- / T-learner, matching the
  ## simulation competitor.  Each reuses its design's tuned control.
  dd_estimators <- list(
    list(name = "deepAFT-RCT-S",    type = "rct",  learner = dd_cate_S, ctrl = dd_ctrl_rct),
    list(name = "deepAFT-RCT-T",    type = "rct",  learner = dd_cate_T, ctrl = dd_ctrl_rct),
    list(name = "deepAFT-Pooled-S", type = "pool", learner = dd_cate_S, ctrl = dd_ctrl_pool),
    list(name = "deepAFT-Pooled-T", type = "pool", learner = dd_cate_T, ctrl = dd_ctrl_pool))

  ## --- Point estimate + stratified bootstrap per estimator ---------------
  ## Point estimate from the full fit; intervals from B refits on resampled
  ## training rows.  CATEs are returned on the standardised scale and
  ## rescaled to log-time by dd_y_sd (failed refits dropped).
  dd_safe <- function(expr) tryCatch(expr, error = function(e) {
    message("    deepAFT fit failed: ", conditionMessage(e)); NULL })
  dd_scl  <- function(z) if (is.null(z)) NULL else z * dd_y_sd
  set.seed(seed)
  cat("\n  deepAFT point + bootstrap (B =", ddB,
      ") for four estimators (RCT/Pooled x S/T) ...\n")
  dd_fits <- lapply(dd_estimators, function(est) {
    cat("    ", est$name, "\n", sep = "")
    point <- dd_scl(dd_safe(est$learner(dd_make_des(est$type), est$ctrl)))
    draws <- do.call(cbind, Filter(Negate(is.null),
      lapply(seq_len(ddB), function(b)
        dd_safe(est$learner(dd_make_des(est$type, TRUE), est$ctrl)))))
    draws <- if (is.matrix(draws)) draws * dd_y_sd else NULL
    list(name = est$name, point = point, draws = draws)
  })
  names(dd_fits) <- vapply(dd_estimators, `[[`, "", "name")

  ## --- Summaries on the acceleration-factor scale ------------------------
  ## Two columns: (1) mean width of the per-subject 95% interval (AF scale),
  ## and (2) the percentage of patients at least 95% certain of benefit, i.e.
  ## P(tau > 0) >= 0.95 (at least 95% of the posterior / bootstrap mass to the
  ## right of AF = 1).  For the deepAFT competitor column (2) is a bootstrap
  ## proportion, not a posterior probability.  BART samples are stored as
  ## (draws x subjects); the deepAFT draws as (subjects x bootstrap).
  dd_af_width <- function(draws) {
    if (is.null(draws) || !is.matrix(draws)) return(NA_real_)
    q <- apply(exp(draws), 1, quantile, c(0.025, 0.975), na.rm = TRUE)
    mean(q[2, ] - q[1, ])
  }
  pct_benefit <- function(cate, margin) {   # margin = subject axis (1 row, 2 col)
    if (is.null(cate) || !is.matrix(cate)) return(NA_real_)
    p <- apply(cate, margin, function(z) mean(z > 0, na.rm = TRUE))
    100 * mean(p >= 0.95)
  }

  dd_summary <- rbind(
    data.frame(
      method      = vapply(dd_fits, `[[`, "", "name"),
      af_ci_width = vapply(dd_fits, function(f) dd_af_width(f$draws), 0),
      pct_benefit = vapply(dd_fits, function(f) pct_benefit(f$draws, 1), 0),
      stringsAsFactors = FALSE),
    data.frame(
      method      = c("FusionForest (M2)", "Trial-only CSF (M1)"),
      af_ci_width = c(mean_ci_M2_combined_af, mean_ci_M1_combined_af),
      pct_benefit = c(pct_benefit(M2_combined_samples, 2),
                      pct_benefit(M1_combined_samples, 2)),
      stringsAsFactors = FALSE))
  rownames(dd_summary) <- NULL

  cat("\n=== deepAFT (S/T-learner) vs BART on the combined cohort (AF scale) ===\n")
  print(dd_summary, row.names = FALSE, digits = 3)

  ## --- Persist for the manuscript exhibit (ANALYSIS / ANALYSIS_SM) -------
  saveRDS(
    list(summary  = dd_summary,
         tuned    = list(rct = dd_ctrl_rct, pool = dd_ctrl_pool),
         fits     = dd_fits,   # per-estimator point + bootstrap draws
         settings = list(B = ddB, K = ddK, R = ddR, units = c(8L, 6L, 1L))),
    file = file.path(out_dir, "analysis_deepaft.rds"))
  writeLines(
    c("deepAFT competitor (S- and T-learner) vs BART (combined RCT + MACS cohort)",
      "Architecture 8-6-1 ReLU/idu; 10-fold CV tuning; B = 100 bootstrap.",
      "MACS interval-censored deaths approximated by the right-censored",
      "last-lab-year encoding (see ANALYSIS_SM.tex).",
      "",
      capture.output(print(dd_summary, row.names = FALSE, digits = 3))),
    file.path(out_dir, "analysis_deepaft.txt"))
  cat("Wrote ", file.path(out_dir, "analysis_deepaft.rds"),
      " and analysis_deepaft.txt\n", sep = "")

  ## --- Caterpillar plots for the SM (one per estimator) ------------------
  ## Subject-level acceleration factor from each deepAFT estimator over the
  ## combined cohort, in the same style as the fusion caterpillar (Section 9).
  ## Bars are the bootstrap 95% interval; the dot is the bootstrap mean.
  ## make_caterpillar() expects a (draws x subjects) matrix on the log-time
  ## scale, so we transpose the (subjects x bootstrap) draws.  The src_M2
  ## indicator aligns with the rows of d_fuse (= the columns after transpose).
  ## Filenames match the \includegraphics calls in ANALYSIS_SM.tex.
  ## Per-plot x-axis limits on the acceleration-factor scale.  Set an entry
  ## to c(lo, hi) to fix that caterpillar's axis; NULL (or missing) = automatic.
  dd_xlims <- list(
    deepaft_rct_s    = c(0.5,2.5),
    deepaft_rct_t    = c(0, 10),
    deepaft_pooled_s = c(0.5,2.5),
    deepaft_pooled_t = c(0,10))
  for (f in dd_fits) {
    if (is.null(f$draws) || !is.matrix(f$draws)) next
    tag <- gsub("[^a-z0-9]+", "_", tolower(f$name))      # e.g. deepaft_pooled_t
    make_caterpillar(
      t(f$draws), src_M2,
      file.path(fig_dir, paste0("cate_caterpillar_", tag, ".pdf")),
      xlim = dd_xlims[[tag]])
  }
}
