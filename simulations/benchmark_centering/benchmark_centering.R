# ══════════════════════════════════════════════════════════════════════════════
# benchmark_centering.R -- paired comparison of two FusionForests versions
#
# Written for 1.1.0 vs 1.1.1, the centred-mixture fix described below.  Also
# used for 1.1.1 vs 1.1.2, the per-source scale in the data augmentation: run it
# with OLD_REF=ba678e6 and BENCH_KTREAT=1 and new BENCH_OUT, LIB_OLD and LIB_NEW.
# The `control` setting then has to come out identical in the two versions.
#
# Compares the sampler before and after the parameter-expansion fix of the
# centred error mixture (referee 1, comment 1).  Version 1.1.0 conditioned the
# forests on the centred atoms while it updated the atoms and the weights as in
# the uncentred model; 1.1.1 samples the uncentred working model throughout and
# centres each stored draw.  The causal estimands cannot move under the exact
# posterior (the mixture means vary with the source only), so the benchmark
# checks that tau is unchanged beyond Monte Carlo error and that the baseline
# levels, the error law and the mixing behave at least as well.
#
# USAGE (from anywhere; the script locates itself):
#   Rscript benchmark_centering.R                # setup (if needed) + run + summarize
#   Rscript benchmark_centering.R --smoke        # same, 4 reps at N = 200/200
#   Rscript benchmark_centering.R setup          # build lib/old (git 01e1158) and lib/new (working tree)
#   Rscript benchmark_centering.R run [--smoke]
#   Rscript benchmark_centering.R summarize [results_dir]
# Environment overrides: BENCH_CORES (4), BENCH_REPS (200), BENCH_NPOST (2000),
#   BENCH_NBURN (2000), BENCH_KTREAT (0.5), BENCH_NICE (19), BENCH_OUT, LIB_OLD,
#   LIB_NEW, OLD_REF.
# Requires Rcpp (>= 1.0.11) and ShrinkageTrees in the default library.
# `lib/` and `results*/` next to this file are git-ignored.
#
# LAYOUT
#   1. Paths and settings
#   2. DGP        -- three settings, truth functions, make_data()
#   3. Fit        -- fit_fusion(): the exported FusionForest() of either version
#   4. Metrics    -- summarise_fit()
#   5. Worker     -- one R process per (version, setting, chunk); the script
#                    re-invokes itself, so the two versions never share a session
#   6. Setup      -- build the two package libraries
#   7. Driver     -- launch the workers in parallel
#   8. Summary    -- paired comparison, writes summary.csv
#   9. Main
# ══════════════════════════════════════════════════════════════════════════════

# ── 1. Paths and settings ────────────────────────────────────────────────────
self <- normalizePath(sub("--file=", "", grep("--file=", commandArgs(), value = TRUE)))
here <- dirname(self)
args <- commandArgs(trailingOnly = TRUE)
env  <- function(k, d) { v <- Sys.getenv(k, ""); if (nzchar(v)) v else d }

smoke   <- "--smoke" %in% args
lib_old <- env("LIB_OLD", file.path(here, "lib", "old"))
lib_new <- env("LIB_NEW", file.path(here, "lib", "new"))
old_ref <- env("OLD_REF", "01e1158")                 # v1.1.0 final commit
out_dir <- env("BENCH_OUT", file.path(here, if (smoke) "results_smoke" else "results"))
cores   <- as.integer(env("BENCH_CORES", 4L))
reps    <- as.integer(env("BENCH_REPS",  if (smoke) 4L else 200L))
N_post  <- as.integer(env("BENCH_NPOST", if (smoke) 200L else 2000L))
N_burn  <- as.integer(env("BENCH_NBURN", if (smoke) 200L else 2000L))
chunk   <- as.integer(env("BENCH_CHUNK", if (smoke) 2L else 5L))
niceness <- as.integer(env("BENCH_NICE", 19L))     # workers run at low priority
k_treat <- as.numeric(env("BENCH_KTREAT", 0.5))    # pinned: defaults differ by version
n_rct   <- 150L
n_rwd   <- 350L

# ── 2. DGP ───────────────────────────────────────────────────────────────────
p_total   <- 10L                      # observed covariates
lambda_d  <- 1.0                      # between-source baseline heterogeneity
lambda_u  <- 1.0                      # unmeasured confounding strength
sigma_rct <- 0.75                     # RCT residual SD in exp1 / gauss
rho_X     <- 0.3
Sigma_X   <- rho_X ^ abs(outer(seq_len(p_total), seq_len(p_total), "-"))
chol_X    <- chol(Sigma_X)
draw_X    <- function(n) matrix(rnorm(n * p_total), n, p_total) %*% chol_X

m0   <- function(X)  2 * X[, 1] - X[, 2] * X[, 3] + 0.5 * X[, 4]^2
dev  <- function(X)  lambda_d * (X[, 4] - 0.5 * X[, 5])
tau  <- function(X)  1/2 + X[, 1] - 0.5 * X[, 2]^2

# Per-setting censoring design: target RCT right-censoring and RWD visits.
#   exp1   : experiment 1 at lambda_d = lambda_u = 1
#   stress : skewed RCT error, wide asymmetric bimodal RWD error, heavy and
#            coarse censoring (experiment 4); the fitted mixture means are far
#            from zero, so the fix is most visible
#   gauss  : both errors N(0, 0.75^2); the versions should coincide
#   control: the stress data, fitted with error_dist = "gaussian"
setting_design <- list(
  exp1    = list(target_cens = 0.35, visit_probs = seq(0.1, 0.8, 0.1)),
  stress  = list(target_cens = 0.50, visit_probs = c(0.25, 0.50, 0.75)),
  gauss   = list(target_cens = 0.35, visit_probs = seq(0.1, 0.8, 0.1)),
  control = list(target_cens = 0.50, visit_probs = c(0.25, 0.50, 0.75))
)
settings <- names(setting_design)

gamma_em <- 0.5772156649
# Standardised Gumbel, as in experiment 1: mean 0, variance 1, left-skewed.
std_gumbel <- function(n) {
  bs <- sqrt(6) / pi
  -(-bs * gamma_em - bs * log(-log(runif(n))))
}
# Normal mixture standardised to mean zero and unit variance.
mix_norm <- function(n, mu, w, s) {
  mbar <- sum(w * mu)
  v    <- sum(w * ((mu - mbar)^2 + s^2))
  k    <- sample.int(length(mu), n, replace = TRUE, prob = w)
  (rnorm(n, mu[k], s) - mbar) / sqrt(v)
}
draw_eps <- function(n, setting, source) {
  rct <- source == "rct"
  switch(setting,
    exp1   = if (rct) rnorm(n, 0, sigma_rct) else std_gumbel(n),
    stress = ,
    control = if (rct) 0.6 * std_gumbel(n)
              else 1.5 * mix_norm(n, c(-0.5, 2.0), c(0.8, 0.2), 0.25),
    gauss  = rnorm(n, 0, sigma_rct),
    stop("unknown setting: ", setting))
}

latent_rct <- function(n, setting) {
  X <- draw_X(n)
  A <- rbinom(n, 1L, 0.5)
  list(X = X, A = A, base = m0(X),
       logT = m0(X) + A * tau(X) + draw_eps(n, setting, "rct"))
}
latent_rwd <- function(n, setting) {
  X <- draw_X(n)
  U <- runif(n)
  A <- rbinom(n, 1L, plogis(X[, 1] + U))
  list(X = X, A = A, base = m0(X) + dev(X),
       logT = m0(X) + dev(X) + A * tau(X) + lambda_u * A * U +
              draw_eps(n, setting, "rwd"))
}
solve_cens_rate <- function(logT, target) {
  Tt <- exp(logT)
  uniroot(function(r) mean(1 - exp(-r * Tt)) - target, c(1e-10, 1e6))$root
}
# Fixed once per setting on a large pre-simulation (seeded, so identical for
# both versions).
calibrate <- function(setting, n_cal = 2e5) {
  des <- setting_design[[setting]]
  set.seed(1)
  rate_rct <- solve_cens_rate(latent_rct(n_cal, setting)$logT, des$target_cens)
  visits   <- as.numeric(quantile(exp(latent_rwd(n_cal, setting)$logT),
                                  probs = des$visit_probs))
  list(rate_rct = rate_rct, visits = visits)
}

# One dataset.  RCT right-censored, RWD interval-censored; all bounds on the
# standardised log scale, as in experiment 1.  Also carries the true baseline
# so the fitted levels can be scored.
make_data <- function(n_rct, n_rwd, setting, cal) {
  r <- latent_rct(n_rct, setting)
  w <- latent_rwd(n_rwd, setting)
  logT_true <- c(r$logT, w$logT)
  scale_mu  <- mean(logT_true)
  scale_sd  <- sd(logT_true)
  std       <- function(z) (z - scale_mu) / scale_sd
  logfloor  <- min(logT_true) - 5

  logC   <- log(rexp(n_rct, rate = cal$rate_rct))
  y_rct  <- pmin(r$logT, logC)
  st_rct <- as.integer(r$logT <= logC)

  visits     <- cal$visits
  T0         <- exp(w$logT)
  last_visit <- visits[length(visits)]
  breaks     <- c(0, visits)
  j          <- findInterval(T0, breaks)
  beyond     <- T0 >= last_visit
  j[!beyond] <- pmax(j[!beyond], 1L)
  left_rwd  <- ifelse(j == 1L, logfloor, log(breaks[pmin(j, length(breaks))]))
  right_rwd <- log(breaks[pmin(j + 1L, length(breaks))])
  icc_rwd   <- rep(1L, n_rwd)
  left_rwd[beyond]  <- log(last_visit)
  right_rwd[beyond] <- log(last_visit)
  icc_rwd[beyond]   <- 0L

  list(
    X = rbind(r$X, w$X), A = c(r$A, w$A),
    S = c(rep(1L, n_rct), rep(0L, n_rwd)),
    y      = std(c(y_rct, right_rwd)),
    status = c(st_rct, rep(0L, n_rwd)),
    left   = std(c(y_rct, left_rwd)),
    right  = std(c(y_rct, right_rwd)),
    icc    = c(rep(0L, n_rct), icc_rwd),
    scale_mu = scale_mu, scale_sd = scale_sd,
    base_std = std(c(r$base, w$base)),     # true E[std log T | A = 0, X, S]
    n_rct = n_rct, n_rwd = n_rwd,
    cens_rct = mean(st_rct == 0), beyond_rwd = mean(beyond)
  )
}

# ── 3. Fit ───────────────────────────────────────────────────────────────────
# The exported FusionForest() of whichever version is loaded, with package
# defaults: J = (200, 50, 100, 50), k = 1, except k_treat as set above and
# error_dist as passed.  Binary coding so the baseline forests describe the
# control arm, which is what `base_std` is.
fit_fusion <- function(d, N_post, N_burn, error_dist) {
  FusionForest(
    y                            = d$y,
    status                       = d$status,
    observed_left_time           = d$left,
    observed_right_time          = d$right,
    interval_censoring_indicator = d$icc,
    X_train_control           = d$X,
    X_train_treat             = d$X,
    treatment_indicator_train = d$A,
    source_indicator_train    = d$S,
    X_test_control            = d$X,
    X_test_treat              = d$X,
    X_test_deconf             = d$X,
    treatment_indicator_test  = rep(1L, nrow(d$X)),
    source_indicator_test     = rep(1L, nrow(d$X)),
    outcome_type              = "right-censored",
    timescale                 = "log",
    treatment_coding          = "binary",
    k_treat                   = k_treat,
    error_dist                = error_dist,
    N_post = N_post, N_burn = N_burn,
    store_posterior_sample = TRUE, verbose = FALSE)
}

# ── 4. Metrics ───────────────────────────────────────────────────────────────
as_obs_by_draw <- function(samp, n_obs) if (nrow(samp) == n_obs) samp else t(samp)

eval_cate <- function(point, samp, truth) {
  M  <- as_obs_by_draw(samp, length(point))
  ci <- apply(M, 1, quantile, probs = c(0.025, 0.975))
  c(rmse = sqrt(mean((point - truth)^2)), bias = mean(point - truth),
    coverage = mean(truth >= ci[1, ] & truth <= ci[2, ]),
    width = mean(ci[2, ] - ci[1, ]), postvar = mean(apply(M, 1, var)))
}

# Effective sample size by the initial positive sequence estimator.
ess <- function(x) {
  n <- length(x); if (n < 10 || sd(x) == 0) return(NA_real_)
  r <- acf(x, lag.max = min(n - 1, 500), plot = FALSE)$acf[-1]
  s <- 0
  for (k in seq(1, length(r) - 1, by = 2)) {
    pair <- r[k] + r[k + 1]
    if (pair <= 0) break
    s <- s + pair
  }
  n / (1 + 2 * s)
}

# W1 distance between the posterior-mean fitted mixture CDF of group g and the
# empirical CDF of a large true-error sample, both on the standardised data
# scale; grid integration.
w1_error <- function(fit, g, eps_true_std, grid_n = 400L) {
  if (is.null(fit$dp_mix_prop)) {                      # error_dist = "gaussian"
    sg <- fit$sigma * fit$meta$sigma_hat
    w  <- matrix(1, length(sg), 1); th <- matrix(0, length(sg), 1)
  } else {
    w  <- fit$dp_mix_prop[[g]]; th <- fit$dp_locations[[g]]
    sg <- fit$dp_sigma_g[[g]]
  }
  lo <- min(eps_true_std, th - 3 * max(sg)); hi <- max(eps_true_std, th + 3 * max(sg))
  x  <- seq(lo, hi, length.out = grid_n)
  Ffit <- rowMeans(vapply(seq_len(nrow(w)), function(b)
    colSums(w[b, ] * pnorm(outer(th[b, ], x, function(t, xx) (xx - t) / sg[b]))),
    numeric(grid_n)))
  Ftrue <- ecdf(eps_true_std)(x)
  sum(abs(Ffit - Ftrue)) * (x[2] - x[1])
}
true_error_sample <- function(setting, source, scale_sd, n = 1e5) {
  set.seed(if (source == "rct") 11 else 13)
  draw_eps(n, setting, source) / scale_sd
}

summarise_fit <- function(fit, d, setting, secs) {
  n <- nrow(d$X); i_rct <- seq_len(d$n_rct); i_rwd <- d$n_rct + seq_len(d$n_rwd)
  tru <- tau(d$X)
  # CATE: standardised-scale draws back to the log-time scale
  point <- fit$test_predictions_treat * d$scale_sd
  samp  <- fit$test_predictions_sample_treat * d$scale_sd
  M_tau <- as_obs_by_draw(samp, n)
  cate_all <- eval_cate(point, samp, tru)
  cate_rct <- eval_cate(point[i_rct], M_tau[i_rct, , drop = FALSE], tru[i_rct])
  # Baseline m0sh + (1 - S) d on the standardised scale, versus the truth
  devs <- rep(0, n); devs[i_rwd] <- fit$train_predictions_deviation
  base <- fit$train_predictions_control + (1 - d$S) * devs
  err  <- base - d$base_std
  # Stored mixture: mean should be zero in every draw
  mix_mean <- if (is.null(fit$dp_locations)) NA_real_ else
    max(sapply(seq_along(fit$dp_locations), function(g)
      max(abs(rowSums(fit$dp_mix_prop[[g]] * fit$dp_locations[[g]])))))
  mu_g <- if (!is.null(fit$dp_mu_g)) sapply(fit$dp_mu_g, mean) else c(NA, NA)
  sg_g <- if (!is.null(fit$dp_sigma_g)) sapply(fit$dp_sigma_g, mean) else
    rep(mean(fit$sigma * fit$meta$sigma_hat), 2)
  # Error law per source (group 1 = RWD, 2 = RCT)
  w1_rwd <- w1_error(fit, 1, true_error_sample(setting, "rwd", d$scale_sd))
  w1_rct <- w1_error(fit, 2, true_error_sample(setting, "rct", d$scale_sd))
  # Mixing: ATE draw over the RCT rows and the RCT baseline level per draw
  ate_dr <- colMeans(M_tau[i_rct, , drop = FALSE])
  ctrl   <- fit$train_predictions_sample_control        # draws x observations
  if (nrow(ctrl) == n && ncol(ctrl) != n) ctrl <- t(ctrl)
  lvl_dr <- rowMeans(ctrl[, i_rct, drop = FALSE])
  list(
    row = data.frame(
      setting = setting,
      rmse_all = cate_all[["rmse"]], bias_all = cate_all[["bias"]],
      cover_all = cate_all[["coverage"]], width_all = cate_all[["width"]],
      postvar_all = cate_all[["postvar"]],
      rmse_rct = cate_rct[["rmse"]], bias_rct = cate_rct[["bias"]],
      cover_rct = cate_rct[["coverage"]],
      ate_rct = mean(ate_dr),
      base_rmse_rct = sqrt(mean(err[i_rct]^2)), base_bias_rct = mean(err[i_rct]),
      base_rmse_rwd = sqrt(mean(err[i_rwd]^2)), base_bias_rwd = mean(err[i_rwd]),
      mix_mean_max = mix_mean, mu_rwd = mu_g[1], mu_rct = mu_g[2],
      w1_rwd = w1_rwd, w1_rct = w1_rct,
      ess_ate = ess(ate_dr), ess_level = ess(lvl_dr),
      acf1_level = acf(lvl_dr, lag.max = 1, plot = FALSE)$acf[2],
      sigma_rwd = sg_g[1], sigma_rct = sg_g[2],
      cens_rct = d$cens_rct, beyond_rwd = d$beyond_rwd,
      secs = secs, row.names = NULL),
    tau_point  = point,
    base_point = base
  )
}

# ── 5. Worker ────────────────────────────────────────────────────────────────
# Both versions get identical data: the data seed depends on the setting and
# the replicate only; the fit reseeds separately.
run_worker <- function(version, lib, setting, rep_from, rep_to, N_post, N_burn,
                       out_dir) {
  suppressPackageStartupMessages(library(FusionForests, lib.loc = lib))
  cal   <- calibrate(setting)
  # `control` gets the same seeds as `stress`, so the same data
  s_idx <- match(if (setting == "control") "stress" else setting, settings)
  error_dist <- if (setting == "control") "gaussian" else "source_hdp_scale"
  rows  <- list(); tau_pts <- list(); base_pts <- list()
  for (r in rep_from:rep_to) {
    seed <- 1000L * s_idx + r
    set.seed(seed)
    d <- make_data(n_rct, n_rwd, setting, cal)
    set.seed(seed + 7919L)
    secs <- system.time(fit <- fit_fusion(d, N_post, N_burn, error_dist))[["elapsed"]]
    sm <- summarise_fit(fit, d, setting, secs)
    sm$row$version <- version; sm$row$rep <- r; sm$row$seed <- seed
    rows[[length(rows) + 1]] <- sm$row
    tau_pts[[as.character(r)]]  <- sm$tau_point
    base_pts[[as.character(r)]] <- sm$base_point
    cat(sprintf("[%s %s rep %d] rmse %.3f  base bias rct %+.3f  mu %s  %.0fs\n",
                version, setting, r, sm$row$rmse_all, sm$row$base_bias_rct,
                paste(round(c(sm$row$mu_rwd, sm$row$mu_rct), 3), collapse = "/"),
                secs))
  }
  dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
  out <- file.path(out_dir, sprintf("%s_%s_%04d-%04d.rds",
                                    version, setting, rep_from, rep_to))
  saveRDS(list(summary = do.call(rbind, rows), tau_point = tau_pts,
               base_point = base_pts,
               pkg_version = as.character(packageVersion("FusionForests")),
               N_post = N_post, N_burn = N_burn), out)
  cat("wrote", out, "\n")
}

# ── 6. Setup: build the two package libraries ────────────────────────────────
#   lib/old : the older version (git `old_ref`)
#   lib/new : the current working tree
# Compiled objects from another platform are dropped before installing.
pkg_parts <- c("DESCRIPTION", "NAMESPACE", "NEWS.md", "LICENSE", "LICENSE.md",
               "R", "src", "man", "inst", "data")
install_from <- function(src_dir, lib) {
  unlink(Sys.glob(file.path(src_dir, "src", c("*.o", "*.so", "*.dll"))))
  dir.create(lib, showWarnings = FALSE, recursive = TRUE)
  st <- system2("R", c("CMD", "INSTALL", "--no-docs", "--no-multiarch",
                       paste0("--library=", shQuote(lib)), shQuote(src_dir)))
  if (st != 0) stop("R CMD INSTALL failed for ", src_dir)
}
setup_libs <- function() {
  root <- system2("git", c("-C", shQuote(here), "rev-parse", "--show-toplevel"),
                  stdout = TRUE)
  tmp <- tempfile("bench_build_")
  on.exit(unlink(tmp, recursive = TRUE), add = TRUE)
  old_src <- file.path(tmp, "old"); new_src <- file.path(tmp, "new")
  dir.create(old_src, recursive = TRUE); dir.create(new_src, recursive = TRUE)

  cat("== old:", old_ref, "->", lib_old, "\n")
  st <- system(sprintf("git -C %s archive --format=tar %s | tar -xf - -C %s",
                       shQuote(root), old_ref, shQuote(old_src)))
  if (st != 0) stop("git archive failed for ", old_ref)
  install_from(old_src, lib_old)

  cat("== new: working tree ->", lib_new, "\n")
  for (part in pkg_parts) {
    p <- file.path(root, part)
    if (file.exists(p)) file.copy(p, new_src, recursive = TRUE)
  }
  install_from(new_src, lib_new)

  for (l in c(lib_old, lib_new))
    cat(sprintf("%s: FusionForests %s\n", l,
                as.character(packageVersion("FusionForests", lib.loc = l))))
}
libs_ready <- function()
  all(dir.exists(file.path(c(lib_old, lib_new), "FusionForests")))

# ── 7. Driver ────────────────────────────────────────────────────────────────
# Each worker is its own R process (this script, `worker` mode), so the two
# versions never share a session.  Jobs of both versions run interleaved so a
# drift in machine load affects them alike.
run_driver <- function() {
  if (!libs_ready()) stop("package libraries missing; run `setup` first")
  libs   <- c(old = lib_old, new = lib_new)
  starts <- seq(1L, reps, by = chunk)
  jobs   <- expand.grid(start = starts, setting = settings, version = names(libs),
                        stringsAsFactors = FALSE)
  jobs   <- jobs[order(jobs$start, jobs$setting, jobs$version), ]
  cat(sprintf("%d jobs on %d cores: reps = %d, N = %d/%d, settings = %s\n",
              nrow(jobs), cores, reps, N_post, N_burn,
              paste(settings, collapse = ", ")))
  dir.create(file.path(out_dir, "logs"), showWarnings = FALSE, recursive = TRUE)
  run_job <- function(i) {
    j <- jobs[i, ]
    a <- c(self, "worker", j$version, libs[[j$version]], j$setting, j$start,
           min(j$start + chunk - 1L, reps), N_post, N_burn, out_dir)
    log <- file.path(out_dir, "logs",
                     sprintf("%s_%s_%04d.log", j$version, j$setting, j$start))
    cmd <- if (niceness > 0 && nzchar(Sys.which("nice"))) "nice" else "Rscript"
    arg <- if (cmd == "nice") c("-n", niceness, "Rscript", shQuote(a)) else shQuote(a)
    st <- system2(cmd, arg, stdout = log, stderr = log,
                  env = c("OMP_NUM_THREADS=1", "OPENBLAS_NUM_THREADS=1"))
    if (st != 0) warning("job ", i, " failed; see ", log)
    st
  }
  status <- unlist(parallel::mclapply(seq_len(nrow(jobs)), run_job,
                                      mc.cores = cores, mc.preschedule = FALSE))
  cat(sprintf("done: %d of %d jobs succeeded; results in %s\n",
              sum(status == 0), length(status), out_dir))
}

# ── 8. Summary: paired comparison ────────────────────────────────────────────
# Per setting: mean of each metric by version, the paired difference new - old
# with its Monte Carlo standard error, and the correlation of the per-
# observation tau and baseline estimates between the versions on the same data.
metrics <- c("rmse_all", "bias_all", "cover_all", "width_all", "postvar_all",
             "rmse_rct", "bias_rct", "cover_rct", "ate_rct",
             "base_rmse_rct", "base_bias_rct", "base_rmse_rwd", "base_bias_rwd",
             "w1_rwd", "w1_rct", "mix_mean_max", "mu_rwd", "mu_rct",
             "ess_ate", "ess_level", "acf1_level", "sigma_rwd", "sigma_rct",
             "secs")
run_summary <- function(res_dir) {
  files <- list.files(res_dir, pattern = "\\.rds$", full.names = TRUE)
  if (!length(files)) stop("no .rds files in ", res_dir)
  parts <- lapply(files, readRDS)
  S <- do.call(rbind, lapply(parts, `[[`, "summary"))
  pts <- list()
  for (p in parts) for (r in names(p$tau_point)) {
    key <- paste(p$summary$version[1], p$summary$setting[1], r)
    pts[[key]] <- list(tau = p$tau_point[[r]], base = p$base_point[[r]])
  }
  out <- list()
  for (s in unique(S$setting)) {
    old <- S[S$setting == s & S$version == "old", ]
    new <- S[S$setting == s & S$version == "new", ]
    pair <- intersect(old$rep, new$rep)
    if (!length(pair)) {
      cat(sprintf("\n=== setting %s: no paired replicates (old %d, new %d) ===\n",
                  s, nrow(old), nrow(new)))
      next
    }
    old <- old[match(pair, old$rep), ]; new <- new[match(pair, new$rep), ]
    cat(sprintf("\n=== setting %s: %d paired replicates ===\n", s, length(pair)))
    tab <- t(sapply(metrics, function(m) {
      d <- new[[m]] - old[[m]]
      c(old = mean(old[[m]], na.rm = TRUE), new = mean(new[[m]], na.rm = TRUE),
        diff = mean(d, na.rm = TRUE),
        se = sd(d, na.rm = TRUE) / sqrt(sum(!is.na(d))))
    }))
    tab <- cbind(tab, z = tab[, "diff"] / tab[, "se"])
    print(round(tab, 4))
    cor_tau <- cor_base <- numeric(0)
    for (r in pair) {
      a <- pts[[paste("old", s, r)]]; b <- pts[[paste("new", s, r)]]
      if (is.null(a) || is.null(b)) next
      cor_tau  <- c(cor_tau,  cor(a$tau, b$tau))
      cor_base <- c(cor_base, cor(a$base, b$base))
    }
    if (length(cor_tau))
      cat(sprintf("same-data correlation old/new: tau %.4f (min %.4f), baseline %.4f (min %.4f)\n",
                  mean(cor_tau), min(cor_tau), mean(cor_base), min(cor_base)))
    out[[s]] <- data.frame(setting = s, metric = rownames(tab), tab,
                           n_pairs = length(pair), row.names = NULL)
  }
  if (length(out)) {
    write.csv(do.call(rbind, out), file.path(res_dir, "summary.csv"),
              row.names = FALSE)
    cat("\nwrote", file.path(res_dir, "summary.csv"), "\n")
  }
}

# ── 9. Main ──────────────────────────────────────────────────────────────────
pos <- args[args != "--smoke"]          # keep duplicates (e.g. N_post == N_burn)
cmd <- if (length(pos)) pos[1] else "all"
switch(cmd,
  worker = {
    a <- pos[-1]
    if (length(a) != 8) stop("worker expects 8 arguments")
    run_worker(a[1], a[2], a[3], as.integer(a[4]), as.integer(a[5]),
               as.integer(a[6]), as.integer(a[7]), a[8])
  },
  setup     = setup_libs(),
  run       = run_driver(),
  summarize = run_summary(if (length(pos) > 1) pos[2] else out_dir),
  all = {
    if (!libs_ready()) setup_libs()
    run_driver()
    run_summary(out_dir)
  },
  stop("unknown command: ", cmd)
)
