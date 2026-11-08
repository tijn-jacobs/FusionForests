# ──────────────────────────────────────────────────────────────────────────────
# sim_surv_v13_ye.R -- competitor simulation: the integrative estimator of
# Ye, Yang, Wang and Liu (2025, Lifetime Data Analysis 31, 473-497) on the
# right-censored competitor design (referee 1, comment 7).
# Revision round (manuscript_v3).  Same DGP, calibration, grid, metrics and
# output columns as sim_surv_v13_deepaft.R, so the rows row-bind straight onto
# sim_surv_v13_bff_output.rds.
#
# The method.  Ye et al. assume the same AFT model and estimand as the fusion
# forest,
#     log T = mu0(X, S) + A tau(X) + A (1 - S) c(X) + eps,
#     tau(X) = E[log T(1) - log T(0) | X],
# with c the confounding function, and estimate tau and c by a penalised R-loss
# (Robinson's residual-on-residual decomposition):
#     sum_i w_i [ {log T_i - mu(X_i, S_i)} - {A_i - e(X_i, S_i)}
#                 {tau(X_i) + (1 - S_i) c(X_i)} ]^2 + pen(tau) + pen(c),
# with mu(X, S) = E[log T | X, S] and e(X, S) = P(A = 1 | X, S) pre-estimated,
# and w_i Stute's Kaplan-Meier weights for right censoring.  This is the
# censoring-adjusted counterpart of the integrative R-learner of Wu and Yang
# (2022).  No public implementation exists, so this file reimplements it.
#
# Implementation choices (each one a decision the paper leaves open or one we
# had to adapt; all are fixed here, in one place):
#   * tau(X) = b(X)' alpha and c(X) = b(X)' beta with an unpenalised intercept
#     each.  Two bases: "Linear", b(X) = X, as in the paper (misspecified for
#     the quadratic tau of this DGP), and "Quadratic", b(X) = (X, X^2, X_j X_k),
#     65 terms for p = 10, which contains the true tau.
#   * Stute weights are computed within each source.  Ye et al. require the
#     censoring to depend on the failure time only; in this design the two
#     sources have different exponential censoring rates, so the condition holds
#     within a source but not across sources.  Each source's weights are
#     rescaled to sum to its sample size.
#   * Nuisances by 5-fold cross-fitting, stratified by source: e = 1/2 in the
#     RCT (known by design), a cross-validated logistic lasso on X in the RWD;
#     mu by a Stute-weighted, cross-validated lasso on the quadratic basis,
#     separately per source.
#   * Penalty: the adaptive lasso (initial ridge, glmnet with observation
#     weights) with separate penalty levels for tau and c through a relative
#     factor kappa = lambda_c / lambda_tau in {1/2, 1, 2}, chosen by weighted
#     5-fold cross-validation.  Ye et al. use MCP; ncvreg, which implements it,
#     takes no observation weights, which the Stute weighting needs.
#   * Intervals: pointwise 95% percentile intervals for tau(x) from a stratified
#     nonparametric bootstrap with B_boot = 100 resamples, the same bootstrap as
#     the other competitors.  The tuning parameters (lambdas, kappa) are fixed at
#     their full-sample values inside the bootstrap, which keeps the cost at a
#     few seconds per replication.  Ye et al. give a bootstrap variance for the
#     coefficients only, not an interval for tau(x).
# Output columns match sim_surv_v13.R (method, population, lambda_d, lambda_u,
# Iter, rmse, bias, coverage, width, postvar); postvar is the bootstrap
# variance.  Methods: "Ye-Linear", "Ye-Quadratic".
# ──────────────────────────────────────────────────────────────────────────────

library(glmnet)
library(doParallel)
library(foreach)
library(evd)
library(MASS)

# ── DGP ───────────────────────────────────────────── (verbatim from v12) ──────
p_total     <- 10L               # observed covariates

# Cross design: sweep one lambda over `lambda_vals` while holding the other at 1.
lambda_vals    <- c(0.0, 0.5, 1.0, 1.5, 2.0)  # swept values for each axis
lambda_d_fixed <- 1.0                          # held when sweeping lambda_u
lambda_u_fixed <- 1.0                          # held when sweeping lambda_d
sigma_rct   <- 0.75              # RCT residual SD
target_cens <- 0.35              # target right-censoring fraction (both sources)

# Covariates: multivariate normal with AR(1) dependence, Sigma_ij = rho^|i-j|.
rho_X    <- 0.3
Sigma_X  <- rho_X ^ abs(outer(seq_len(p_total), seq_len(p_total), "-"))
draw_X   <- function(n) MASS::mvrnorm(n, mu = rep(0, p_total), Sigma = Sigma_X)

m0   <- function(X)  (2 * X[, 1] - X[, 2] * X[, 3] + 0.5 * X[, 4]^2)
dev  <- function(X, ld) ld * (X[, 4] - 0.5 * X[, 5])  # RWD baseline deviation
tau  <- function(X)  (1/2 + X[, 1] - 0.5 * X[, 2]^2)         # true CATE
# Unmeasured confounder U: a prognostic factor that enters the RWD outcome in
# BOTH arms (no A x U interaction), so T(1) - T(0) = tau(X) for every unit and
# the CATE is common to the two sources (Assumption 4).  In the RWD we draw
#   A | X ~ Bernoulli{expit(X1)},   U | A, X ~ N(A delta(X), 1).
# By Bayes' rule logit P(A = 1 | X, U) = X1 + delta(X) U - delta(X)^2 / 2, so U
# drives treatment choice and is a confounder there; in the trial treatment is
# randomised and U would only be part of the error.  The RWD contrast is
# tau(X) + c(X) with c(X) = lu * delta(X), pure confounding bias, and the RWD
# baseline is m0(X) + dev(X, ld).  The residual of log T given (X, A) is
# lu * {U - A delta(X)} + eps, whose law is free of (X, A) (Assumption 6).
conf      <- function(U, lu) lu * U
delta_u   <- 0.84                              # uniform design: c(X) = 0.84 * lu
delta_het <- c(0.78, 0.31)                     # covariate-dependent design
# conf_design: "uniform" -> delta(X) = delta_u, so c is constant in X;
# "heterogeneous" -> delta(X) = 0.78 + 0.31 X2, so c(X) has mean 0.78 * lu, SD
# 0.31 * lu and runs from 0.27 * lu to 1.29 * lu (5th to 95th percentile).
# Set per grid cell in the main loop; forked workers inherit the global.
conf_design <- "uniform"
delta_x <- function(X) {
  if (conf_design == "heterogeneous") delta_het[1] + delta_het[2] * X[, 2]
  else delta_u
}

# Latent log-survival generators (no censoring), one per source.
latent_rct <- function(n) {
  X <- draw_X(n)
  A <- rbinom(n, 1L, 0.5)                              # randomised
  list(X = X, A = A,
       logT = m0(X) + A * tau(X) + rnorm(n, 0, sigma_rct))
}
latent_rwd <- function(n, ld, lu) {
  gamma_em   <- 0.5772156649
  beta_scale <- sqrt(6) / pi
  mu_loc     <- -beta_scale * gamma_em
  X <- draw_X(n)
  e <- plogis(X[, 1])                                  # P(A = 1 | X): selection on X1
  A <- rbinom(n, 1L, e)
  U <- rnorm(n, mean = A * delta_x(X))                 # unmeasured prognostic factor
  eps <- -rgumbel(n, loc = mu_loc, scale = beta_scale) # mean-0 residual (SD 1)
  list(X = X, A = A, e = e, eps = eps,
       logT = m0(X) + dev(X, ld) + A * tau(X) + conf(U, lu) + eps)  # U in both arms
}

# Calibrate the RCT exponential censoring rate (time scale) to ~target.
solve_cens_rate <- function(logT, target = target_cens) {
  Tt <- exp(logT)
  uniroot(function(r) mean(1 - exp(-r * Tt)) - target, c(1e-10, 1e6))$root
}

# Per-cell calibration, fixed once on a large pre-simulation: the RCT and the RWD
# each get an independent exponential right-censoring rate tuned to ~target_cens.
calibrate <- function(ld, lu, n_cal = 1e5) {
  rate_rct <- solve_cens_rate(latent_rct(n_cal)$logT)
  rate_rwd <- solve_cens_rate(latent_rwd(n_cal, ld, lu)$logT)
  list(rate_rct = rate_rct, rate_rwd = rate_rwd)
}

# Simulate one full dataset.  Both sources are right-censored: the RCT and the
# RWD each get independent exponential censoring on the time scale, tuned to
# ~target_cens.  Outcomes are returned on the standardised LOG-time scale.
make_data <- function(n_rct, n_rwd, ld, lu, rate_rct, rate_rwd) {
  r <- latent_rct(n_rct)
  w <- latent_rwd(n_rwd, ld, lu)
  # Standardise the pooled latent log-survival to mean 0, unit variance; the same
  # affine map is applied to both sources.  Predicted CATEs come back on this
  # scale and are rescaled by `scale_sd` before evaluation.
  logT_true <- c(r$logT, w$logT)
  scale_mu  <- mean(logT_true)
  scale_sd  <- sd(logT_true)
  std       <- function(z) (z - scale_mu) / scale_sd

  # --- RCT: exponential right censoring ---
  logC_rct <- log(rexp(n_rct, rate = rate_rct))
  y_rct    <- pmin(r$logT, logC_rct)
  st_rct   <- as.integer(r$logT <= logC_rct)

  # --- RWD: exponential right censoring (same mechanism as the RCT) ---
  logC_rwd <- log(rexp(n_rwd, rate = rate_rwd))
  y_rwd    <- pmin(w$logT, logC_rwd)
  st_rwd   <- as.integer(w$logT <= logC_rwd)

  # --- standardise the log-time outcomes to the common scale ---
  y_rct <- std(y_rct)
  y_rwd <- std(y_rwd)

  list(
    X_rct = r$X, X_rwd = w$X,
    A_rct = r$A, A_rwd = w$A,
    y_rct = y_rct, y_rwd = y_rwd,
    st_rct = st_rct, st_rwd = st_rwd,
    X      = rbind(r$X, w$X),
    A      = c(r$A, w$A),
    S      = c(rep(1L, n_rct), rep(0L, n_rwd)),
    y      = c(y_rct, y_rwd),
    status = c(st_rct, st_rwd),
    e_rwd  = w$e,
    e_all  = c(rep(0.5, n_rct), w$e),
    scale_sd = scale_sd,
    n_rct = n_rct, n_rwd = n_rwd
  )
}

# ── Ye et al. (2025) integrative estimator ────────────────────────────────────
basis_linear <- function(X) {
  colnames(X) <- paste0("x", seq_len(ncol(X)))
  X
}
basis_quad <- function(X) {
  p  <- ncol(X); nm <- paste0("x", seq_len(p))
  sq <- X^2
  colnames(sq) <- paste0(nm, "^2")
  pr <- combn(p, 2)
  ia <- X[, pr[1, ], drop = FALSE] * X[, pr[2, ], drop = FALSE]
  colnames(ia) <- paste0(nm[pr[1, ]], ":", nm[pr[2, ]])
  cbind(basis_linear(X), sq, ia)
}
bases <- list(Linear = basis_linear, Quadratic = basis_quad)

# Stute's weights: the jumps of the Kaplan-Meier estimator at the ordered
# observations, zero for censored ones.  Events precede censorings at ties.
stute_weights <- function(y, delta) {
  n   <- length(y)
  o   <- order(y, -delta)
  dl  <- delta[o]
  k   <- seq_len(n)
  fac <- ((n - k) / (n - k + 1))^dl
  lag <- c(1, cumprod(fac)[-n])
  w   <- numeric(n)
  w[o] <- dl / (n - k + 1) * lag
  w
}
# Within-source Stute weights, each source rescaled to total weight n_s.
source_weights <- function(y, delta, S) {
  w <- numeric(length(y))
  for (s in c(0L, 1L)) {
    i  <- which(S == s)
    ws <- stute_weights(y[i], delta[i])
    w[i] <- ws * length(i) / sum(ws)
  }
  w
}

# Folds stratified by source.
make_folds <- function(S, K) {
  f <- integer(length(S))
  for (s in unique(S)) {
    i <- which(S == s)
    f[i] <- sample(rep_len(seq_len(K), length(i)))
  }
  f
}

# glmnet fit on the default path, prediction at a fixed lambda.
glm_pred <- function(x, y, w, family, lam, xnew, alpha = 1) {
  fit <- glmnet(x, y, weights = w, family = family, alpha = alpha)
  as.numeric(predict(fit, xnew, s = lam, type = "response"))
}

# Nuisance tuning on the full sample: the lasso penalty of the RWD propensity
# model and of the two per-source outcome models.
tune_nuisance <- function(d, Bq) {
  i0 <- d$S == 0
  lam_e <- cv.glmnet(d$X[i0, ], d$A[i0], family = "binomial",
                     nfolds = 5)$lambda.min
  lam_mu <- vapply(c(0L, 1L), function(s) {
    i <- d$S == s
    cv.glmnet(Bq[i, ], d$y[i], weights = d$w[i], nfolds = 5)$lambda.min
  }, numeric(1))
  list(e = lam_e, mu = setNames(lam_mu, c("0", "1")))
}

# Cross-fitted nuisances at fixed tuning: e (1/2 in the RCT) and mu.
nuisance_oof <- function(d, Bq, folds, lam) {
  n  <- length(d$y)
  e  <- rep(0.5, n)
  mu <- numeric(n)
  for (k in sort(unique(folds))) {
    tr <- folds != k; te <- folds == k
    i_tr <- tr & d$S == 0; i_te <- te & d$S == 0
    if (any(i_te))
      e[i_te] <- glm_pred(d$X[i_tr, ], d$A[i_tr], rep(1, sum(i_tr)), "binomial",
                          lam$e, d$X[i_te, , drop = FALSE])
    for (s in c(0L, 1L)) {
      i_tr <- tr & d$S == s; i_te <- te & d$S == s
      if (any(i_te))
        mu[i_te] <- glm_pred(Bq[i_tr, ], d$y[i_tr], d$w[i_tr], "gaussian",
                             lam$mu[[as.character(s)]],
                             Bq[i_te, , drop = FALSE])
    }
  }
  list(e = pmin(pmax(e, 0.02), 0.98), mu = mu)
}

# R-loss design: unpenalised intercepts of tau and c, then the basis terms.
r_design <- function(B, A, S, e) {
  r   <- A - e
  oms <- 1 - S
  Z   <- cbind(r, oms * r, r * B, oms * r * B)
  colnames(Z) <- c("tau0", "c0", paste0("tau:", colnames(B)),
                   paste0("c:", colnames(B)))
  Z
}
n_basis <- function(Z) (ncol(Z) - 2L) / 2L

# Adaptive-lasso penalty factors: 0 for the intercepts, 1 / |ridge coef| for
# the basis terms, times kappa for the c block.
penalty_factors <- function(b_ridge, nb, kappa) {
  a <- 1 / pmax(abs(b_ridge[-(1:2)]), 1e-6)
  c(0, 0, a[seq_len(nb)], kappa * a[nb + seq_len(nb)])
}
ridge_coef <- function(Z, y, w, lam) {
  fit <- glmnet(Z, y, weights = w, alpha = 0, intercept = FALSE)
  as.numeric(coef(fit, s = lam))[-1]
}

# Final-stage tuning on the full sample: ridge penalty, kappa and lasso penalty.
kappa_grid <- c(0.5, 1, 2)
tune_final <- function(Z, y, w, folds) {
  lam_r <- cv.glmnet(Z, y, weights = w, alpha = 0, intercept = FALSE,
                     foldid = folds)$lambda.min
  b  <- ridge_coef(Z, y, w, lam_r)
  nb <- n_basis(Z)
  cvs <- lapply(kappa_grid, function(kap)
    cv.glmnet(Z, y, weights = w, alpha = 1, intercept = FALSE,
              penalty.factor = penalty_factors(b, nb, kap), foldid = folds))
  best <- which.min(vapply(cvs, function(cv) min(cv$cvm), numeric(1)))
  list(lam_ridge = lam_r, kappa = kappa_grid[best],
       lam = cvs[[best]]$lambda.min)
}
final_coef <- function(Z, y, w, tf) {
  b   <- ridge_coef(Z, y, w, tf$lam_ridge)
  fit <- glmnet(Z, y, weights = w, alpha = 1, intercept = FALSE,
                penalty.factor = penalty_factors(b, n_basis(Z), tf$kappa))
  cf  <- as.numeric(coef(fit, s = tf$lam))[-1]
  setNames(cf, colnames(Z))
}

# Full pipeline on one dataset; `tune` fixes the tuning (used in the bootstrap).
ye_fit <- function(d, basis, Xev, tune = NULL, K = 5L) {
  B   <- basis(d$X)
  Bq  <- basis_quad(d$X)
  d$w <- source_weights(d$y, d$status, d$S)
  folds <- make_folds(d$S, K)
  if (is.null(tune)) tune <- list(nuis = tune_nuisance(d, Bq))
  nu   <- nuisance_oof(d, Bq, folds, tune$nuis)
  Z    <- r_design(B, d$A, d$S, nu$e)
  ytil <- d$y - nu$mu
  if (is.null(tune$final)) tune$final <- tune_final(Z, ytil, d$w, folds)
  cf  <- final_coef(Z, ytil, d$w, tune$final)
  Bev <- basis(Xev)
  tau <- cf[["tau0"]] + as.numeric(Bev %*% cf[paste0("tau:", colnames(Bev))])
  list(tau = tau, tune = tune)
}

# ── Metrics (verbatim from sim_surv_v13_deepaft.R) ────────────────────────────
metric_cols <- c("rmse", "bias", "coverage", "width", "postvar")

eval_cate <- function(point, draws, truth, idx) {
  p <- point[idx]; tr <- truth[idx]
  if (is.null(draws)) {
    return(c(rmse = sqrt(mean((p - tr)^2)), bias = mean(p - tr),
             coverage = NA_real_, width = NA_real_, postvar = NA_real_))
  }
  M  <- draws[idx, , drop = FALSE]                       # n_idx x B
  ci <- apply(M, 1, quantile, probs = c(0.025, 0.975), na.rm = TRUE)
  c(rmse     = sqrt(mean((p - tr)^2)),
    bias     = mean(p - tr),
    coverage = mean(tr >= ci[1, ] & tr <= ci[2, ]),
    width    = mean(ci[2, ] - ci[1, ]),
    postvar  = mean(apply(M, 1, var, na.rm = TRUE)))
}

make_rows <- function(method, point, draws, truth, n_rct, n_all) {
  pops <- list(RCT = seq_len(n_rct),
               RWD = (n_rct + 1L):n_all,
               All = seq_len(n_all))
  do.call(rbind, lapply(names(pops), function(pop) {
    vals <- if (is.null(point))
      setNames(rep(NA_real_, length(metric_cols)), metric_cols)
    else eval_cate(point, draws, truth, pops[[pop]])
    data.frame(method = method, population = pop, as.list(vals),
               stringsAsFactors = FALSE)
  }))
}

# ── Bootstrap: stratified by source, evaluation set fixed ─────────────────────
resample_pooled <- function(d) {
  i <- c(sample(which(d$S == 1L), replace = TRUE),
         sample(which(d$S == 0L), replace = TRUE))
  d$X <- d$X[i, , drop = FALSE]; d$A <- d$A[i]; d$S <- d$S[i]
  d$y <- d$y[i];                 d$status <- d$status[i]
  d
}

# ── One replication: both bases on one dataset ────────────────────────────────
run_one_sim <- function(seed, n_rct, n_rwd, ld, lu, rate_rct, rate_rwd) {
  set.seed(seed)
  d     <- make_data(n_rct, n_rwd, ld, lu, rate_rct, rate_rwd)
  truth <- tau(d$X)
  n_all <- n_rct + n_rwd
  s     <- d$scale_sd                # standardised-log CATE -> log-time scale
  safe  <- function(expr) tryCatch(expr, error = function(e) {
    message("    Ye fit failed: ", conditionMessage(e)); NULL })

  res <- do.call(rbind, lapply(names(bases), function(bn) {
    full  <- safe(ye_fit(d, bases[[bn]], d$X))
    draws <- NULL
    if (!is.null(full) && B_boot > 0L) {
      cols <- lapply(seq_len(B_boot), function(b)
        tryCatch(ye_fit(resample_pooled(d), bases[[bn]], d$X,
                        tune = full$tune)$tau,
                 error = function(e) rep(NA_real_, n_all)))
      draws <- do.call(cbind, cols)
      keep  <- colSums(is.na(draws)) < n_all
      draws <- if (any(keep)) draws[, keep, drop = FALSE] else NULL
    }
    make_rows(paste0("Ye-", bn),
              if (!is.null(full))  full$tau * s,
              if (!is.null(draws)) draws * s, truth, n_rct, n_all)
  }))
  res$cens_rct <- mean(d$st_rct == 0)
  res$cens_rwd <- mean(d$st_rwd == 0)
  res
}

# ── Parallel runner over replications (one grid cell) ─────────────────────────
run_simulation_tidy <- function(n_rep, n_rct, n_rwd, ld, lu, seed_offset = 1000L) {
  cal <- calibrate(ld, lu)        # per-cell RCT + RWD exponential censoring rates
  cat(sprintf("  RCT cens rate = %.4g | RWD cens rate = %.4g\n",
              cal$rate_rct, cal$rate_rwd))
  foreach(
    i = seq_len(n_rep),
    .combine = "rbind",
    .packages = c("glmnet", "evd", "MASS")
  ) %dopar% {
    res <- run_one_sim(seed_offset + i, n_rct, n_rwd, ld, lu,
                       cal$rate_rct, cal$rate_rwd)
    res$lambda_d <- ld
    res$lambda_u <- lu
    res$Iter     <- i
    res
  }
}

# ── Main (HPC) ────────────────────────────────────────────────────────────────
args <- commandArgs(trailingOnly = TRUE)
if (length(args) > 0) {
  num_cores <- as.integer(args[1]) - 1L
} else {
  num_cores <- parallel::detectCores() - 1L
}
registerDoParallel(cores = num_cores)
cat("Number of cores being used (1 free):", num_cores, "\n")
cat("SIMULATION (competitors): Ye et al. (2025) integrative R-loss, linear and quadratic basis -- HPC\n")
cat("RCT and RWD both right-censored ~35%.\n")
cat("Covariates: MVN AR(1), rho =", rho_X, "\n")

# Simulation settings.
M       <- 1000L                 # replications per grid cell
n_rct   <- 150L
n_rwd   <- 350L
B_boot  <- 100L                  # bootstrap refits per fit for the tau(x) intervals
cat(sprintf("M = %d reps/cell, n_rct = %d, n_rwd = %d, B_boot = %d\n",
            M, n_rct, n_rwd, B_boot))

# Full 3x3 grid, as for the fusion forest: lambda_d, lambda_u in {0, 1, 2}.
grid <- expand.grid(lambda_d = c(0, 1, 2), lambda_u = c(0, 1, 2))
cat(sprintf("Grid: %d (lambda_d, lambda_u) cells\n", nrow(grid)))

all_cells <- vector("list", nrow(grid))
for (g in seq_len(nrow(grid))) {
  ld <- grid$lambda_d[g]; lu <- grid$lambda_u[g]
  cat(sprintf("\nCell %d/%d: lambda_d = %.1f, lambda_u = %.1f\n",
              g, nrow(grid), ld, lu))
  all_cells[[g]] <- run_simulation_tidy(
    n_rep = M, n_rct = n_rct, n_rwd = n_rwd,
    ld = ld, lu = lu, seed_offset = 1000L * g)
}
final_flat_df <- do.call(rbind, all_cells)

# Output file path (! NAME MUST BE FILENAME_output.rds !).
output_file <- file.path(Sys.getenv("TMPDIR"), "sim_surv_v13_ye_output.rds")
cat("\nSaving all results to:", output_file, "\n")
saveRDS(final_flat_df, file = output_file)
cat("All results successfully saved in one file.\n")

# Local analysis (tables, figures): analyse_competitors_v13.R in this folder.
