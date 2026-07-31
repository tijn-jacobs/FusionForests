# ──────────────────────────────────────────────────────────────────────────────
# Competitor simulation: BJ-trees (bujar) vs the main FusionForest survival
# simulation.
#
# Same shape as sim_surv_v12_deepaft.R and sim_surv_v12_bjelm.R: the DGP,
# calibration, metrics, bootstrap and output columns are the deepAFT script's,
# so the rows row-bind straight onto the other competitor outputs.
#
#   install.packages("bujar"); library(bujar); help("bujar")
#   Wang & Wang (2010), Statistical Applications in Genetics and Molecular
#   Biology 9(1), Article 24.
#
# Buckley-James boosting with regression trees as the base learner. Boosted
# trees against a Bayesian tree ensemble puts both methods in the same function
# class, so the experiment isolates the fusion and the uncertainty
# quantification rather than the base learner.
#
# Buckley-James handles RIGHT censoring only, so both sources are right-censored
# at ~35%, exactly the deepAFT design.
#
# WARNING -- this is the slowest competitor by a wide margin. One bujar fit runs
# a Buckley-James loop of up to `iter.bj` iterations, each fitting `mstop`
# boosted trees, so a single fit is order 10^3 tree fits. Time one fit before
# committing to a full run; see the arithmetic in COMPARATORS.md.
#
# BJ-trees has no posterior; the interval metrics come from a stratified
# nonparametric bootstrap, so `postvar` is the bootstrap variance rather than a
# posterior variance. Output columns match sim_surv_v12.R (method, population,
# lambda_d, lambda_u, Iter, rmse, bias, coverage, width, postvar).
# ──────────────────────────────────────────────────────────────────────────────

library(bujar)
library(doParallel)
library(foreach)
library(evd)
library(MASS)

# bujar's tree learner uses gbm, whose startup banner would otherwise print once
# per forked worker -- 191 copies of it on a full node, burying the real output.
# Attaching it quietly here means the forks inherit it already attached, so the
# library(bujar) that foreach runs in each worker is a no-op and stays silent.
suppressPackageStartupMessages(library(gbm))

# ── DGP ───────────────────────────────────────────── (verbatim from v12) ──────
p_total     <- 10L               # observed covariates
sigma_rct   <- 0.75              # RCT residual SD
target_cens <- 0.35              # target right-censoring fraction (both sources)

# Covariates: multivariate normal with AR(1) dependence, Sigma_ij = rho^|i-j|.
rho_X    <- 0.3
Sigma_X  <- rho_X ^ abs(outer(seq_len(p_total), seq_len(p_total), "-"))
draw_X   <- function(n) MASS::mvrnorm(n, mu = rep(0, p_total), Sigma = Sigma_X)

m0   <- function(X)  (2 * X[, 1] - X[, 2] * X[, 3] + 0.5 * X[, 4]^2)
dev  <- function(X, ld) ld * (X[, 4] - 0.5 * X[, 5])  # RWD baseline deviation
tau  <- function(X)  (1/2 + X[, 1] - 0.5 * X[, 2]^2)         # true CATE
conf <- function(U, lu) lu * (U)               # confounding via unobs. U

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
  U <- runif(n)                                        # RWD-only, unobserved
  e <- plogis(X[, 1] + U)                              # true propensity score
  A <- rbinom(n, 1L, e)                                # selection on X1, U
  eps <- -rgumbel(n, loc = mu_loc, scale = beta_scale) # mean-0 residual (SD 1)
  list(X = X, A = A, e = e,
       logT = m0(X) + dev(X, ld) + A * tau(X) + A * conf(U, lu) + eps)
}

# Calibrate an exponential censoring rate (time scale) to ~target.
solve_cens_rate <- function(logT, target = target_cens) {
  Tt <- exp(logT)
  uniroot(function(r) mean(1 - exp(-r * Tt)) - target, c(1e-10, 1e6))$root
}

calibrate <- function(ld, lu, n_cal = 1e5) {
  list(rate_rct = solve_cens_rate(latent_rct(n_cal)$logT),
       rate_rwd = solve_cens_rate(latent_rwd(n_cal, ld, lu)$logT))
}

# Simulate one full dataset. Both sources are right-censored. Outcomes come back
# on the standardised LOG-time scale: the pooled LATENT log-survival is centred
# and scaled to mean 0 / variance 1, and predicted CATEs are rescaled by
# `scale_sd` before evaluation.
make_data <- function(n_rct, n_rwd, ld, lu, cal) {
  r <- latent_rct(n_rct)
  w <- latent_rwd(n_rwd, ld, lu)

  logT_true <- c(r$logT, w$logT)
  scale_mu  <- mean(logT_true)
  scale_sd  <- sd(logT_true)
  std       <- function(z) (z - scale_mu) / scale_sd

  logC_rct <- log(rexp(n_rct, rate = cal$rate_rct))
  y_rct    <- pmin(r$logT, logC_rct)
  st_rct   <- as.integer(r$logT <= logC_rct)

  logC_rwd <- log(rexp(n_rwd, rate = cal$rate_rwd))
  y_rwd    <- pmin(w$logT, logC_rwd)
  st_rwd   <- as.integer(w$logT <= logC_rwd)

  list(
    X_rct = r$X, X_rwd = w$X,
    A_rct = r$A, A_rwd = w$A,
    y_rct = std(y_rct), y_rwd = std(y_rwd),
    st_rct = st_rct, st_rwd = st_rwd,
    X      = rbind(r$X, w$X),
    scale_sd = scale_sd,
    n_rct = n_rct, n_rwd = n_rwd
  )
}

# ── BJ-trees competitors (S- and T-learner) ───────────────────────────────────
# bujar wants a data frame with named columns, and predict() takes `newx`. Both
# the training and the evaluation frames are built with the same column names,
# so the tree splits line up.
as_frame <- function(X, extra = NULL) {
  df <- as.data.frame(X)
  names(df) <- paste0("x", seq_len(ncol(df)))
  if (!is.null(extra)) df <- cbind(df, extra)
  df
}

design_rct <- function(d, Xev) {
  list(y = d$y_rct, status = d$st_rct,
       X = as_frame(d$X_rct), A = d$A_rct, Xev = as_frame(Xev))
}
design_pool <- function(d, Xev) {
  S <- c(rep(1L, d$n_rct), rep(0L, d$n_rwd))
  list(y      = c(d$y_rct, d$y_rwd),
       status = c(d$st_rct, d$st_rwd),
       X      = as_frame(rbind(d$X_rct, d$X_rwd), data.frame(S = S)),
       A      = c(d$A_rct, d$A_rwd),
       Xev    = as_frame(Xev, data.frame(S = rep(1L, nrow(Xev)))))
}

# `y` is ALREADY the standardised LOG survival time, which is what bujar wants.
# Do not log it again. `cens` is the event indicator (1 = event, 0 = censored),
# matching bujar's convention (0 = alive, 1 = dead).
#
# degree = 2 matters: the default of 1 is an additive model, and the DGP
# baseline contains x2*x3, which an additive fit cannot represent. Leaving it at
# 1 would make bujar lose on a restriction we imposed ourselves.
#
# n.cores = 1 so bujar's internal parallelism does not fight doParallel.
bj_fit <- function(y, status, Z, degree, mstop, nu) {
  bujar(y = y, cens = status, x = Z,
        learner = "tree", degree = degree,
        mstop = mstop, nu = nu,
        tuning = FALSE, cv = FALSE, n.cores = 1L)
}
bj_pred <- function(fit, newZ) as.numeric(predict(fit, newx = newZ))

# S-learner: ONE model, predicted at A = 1 and at A = 0. Fitting twice would
# double the cost of the slowest competitor in the set for no gain.
cate_S <- function(des, degree, mstop, nu) {
  Z   <- cbind(des$X, A = des$A)
  fit <- bj_fit(des$y, des$status, Z, degree, mstop, nu)
  bj_pred(fit, cbind(des$Xev, A = 1L)) - bj_pred(fit, cbind(des$Xev, A = 0L))
}

# T-learner: a separate model per arm; CATE = mu_1(x) - mu_0(x).
cate_T <- function(des, degree, mstop, nu) {
  t1 <- des$A == 1L; t0 <- des$A == 0L
  f1 <- bj_fit(des$y[t1], des$status[t1], des$X[t1, , drop = FALSE], degree, mstop, nu)
  f0 <- bj_fit(des$y[t0], des$status[t0], des$X[t0, , drop = FALSE], degree, mstop, nu)
  bj_pred(f1, des$Xev) - bj_pred(f0, des$Xev)
}

# The four estimators (signature (d, Xev, degree, mstop, nu) for boot_cate).
fit_rct_S  <- function(d, Xev, dg, ms, nu) cate_S(design_rct(d, Xev),  dg, ms, nu)
fit_rct_T  <- function(d, Xev, dg, ms, nu) cate_T(design_rct(d, Xev),  dg, ms, nu)
fit_pool_S <- function(d, Xev, dg, ms, nu) cate_S(design_pool(d, Xev), dg, ms, nu)
fit_pool_T <- function(d, Xev, dg, ms, nu) cate_T(design_pool(d, Xev), dg, ms, nu)

# ── Metrics ───────────────────────────────────────────────────────────────────
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

# ── Bootstrap uncertainty ─────────────────────────────────────────────────────
resample_data <- function(d) {
  i_rct <- sample.int(d$n_rct, d$n_rct, replace = TRUE)
  i_rwd <- sample.int(d$n_rwd, d$n_rwd, replace = TRUE)
  d$X_rct  <- d$X_rct[i_rct, , drop = FALSE]; d$A_rct  <- d$A_rct[i_rct]
  d$y_rct  <- d$y_rct[i_rct];                 d$st_rct <- d$st_rct[i_rct]
  d$X_rwd  <- d$X_rwd[i_rwd, , drop = FALSE]; d$A_rwd  <- d$A_rwd[i_rwd]
  d$y_rwd  <- d$y_rwd[i_rwd];                 d$st_rwd <- d$st_rwd[i_rwd]
  d
}

boot_cate <- function(fit_one, d, Xev, B, ...) {
  n <- nrow(Xev)
  cols <- lapply(seq_len(B), function(b)
    tryCatch(fit_one(resample_data(d), Xev, ...),
             error = function(e) rep(NA_real_, n)))
  draws <- do.call(cbind, cols)
  keep  <- colSums(is.na(draws)) < n
  if (!any(keep)) NULL else draws[, keep, drop = FALSE]
}

# ── One replication: four BJ-trees estimators on one dataset ──────────────────
run_one_sim <- function(n_rct, n_rwd, ld, lu, cal, degree, mstop, nu, B_boot) {
  d     <- make_data(n_rct, n_rwd, ld, lu, cal)
  truth <- tau(d$X)
  n_all <- n_rct + n_rwd
  safe  <- function(expr) tryCatch(expr, error = function(e) {
    message("    BJ-trees fit failed: ", conditionMessage(e)); NULL })

  s   <- d$scale_sd                  # standardised-log CATE -> log-time scale
  scl <- function(z) if (is.null(z)) NULL else z * s

  estimators <- list(
    list(name = "bjtree-RCT-S",    fun = fit_rct_S),
    list(name = "bjtree-RCT-T",    fun = fit_rct_T),
    list(name = "bjtree-Pooled-S", fun = fit_pool_S),
    list(name = "bjtree-Pooled-T", fun = fit_pool_T)
  )
  res <- do.call(rbind, lapply(estimators, function(est) {
    point <- safe(est$fun(d, d$X, degree, mstop, nu))
    draws <- if (B_boot > 0L && !is.null(point))
      safe(boot_cate(est$fun, d, d$X, B_boot, degree, mstop, nu)) else NULL
    make_rows(est$name, scl(point), scl(draws), truth, n_rct, n_all)
  }))
  res$cens_rct <- mean(d$st_rct == 0)
  res$cens_rwd <- mean(d$st_rwd == 0)
  res
}

# ── Parallel runner over replications (one grid cell) ─────────────────────────
run_simulation_tidy <- function(n_rep, n_rct, n_rwd, ld, lu,
                                degree, mstop, nu, B_boot) {
  cal <- calibrate(ld, lu)
  cat(sprintf("  RCT cens rate = %.4g | RWD cens rate = %.4g\n",
              cal$rate_rct, cal$rate_rwd))
  foreach(
    i = seq_len(n_rep),
    .combine = "rbind",
    .packages = c("bujar", "evd", "MASS")
  ) %dopar% {
    res <- run_one_sim(n_rct, n_rwd, ld, lu, cal, degree, mstop, nu, B_boot)
    res$lambda_d <- ld
    res$lambda_u <- lu
    res$Iter     <- i
    res
  }
}

# ── CONFIG ────────────────────────────────────────────────────────────────────
# TIME ONE FIT before scaling up. With TIMING = TRUE the script fits a single
# bujar model, prints the elapsed time and the projected total, and exits.
TIMING <- FALSE

M       <- 1000L                 # replications per grid cell
B_boot  <- 100L                  # bootstrap refits per fit

n_rct  <- 150L
n_rwd  <- 350L

DEGREE <- 2L                     # 2 = second-order interactions; see bj_fit
MSTOP  <- 50L                    # boosting iterations (bujar default)
NU     <- 0.1                    # boosting step length (bujar default)

# The main text reports experiment 2 at lambda_d = lambda_u = 1 only; the full
# grid goes to the supplementary materials. Running one cell is 9x cheaper.
grid <- expand.grid(lambda_d = 1, lambda_u = 1)
# grid <- expand.grid(lambda_d = c(0, 1, 2), lambda_u = c(0, 1, 2))

# ── Main ──────────────────────────────────────────────────────────────────────
args <- commandArgs(trailingOnly = TRUE)
num_cores <- if (length(args) > 0) {
  as.integer(args[1]) - 1L
} else {
  max(1L, parallel::detectCores() - 1L)
}

if (TIMING) {
  cat("TIMING RUN: one bujar fit, then projected totals.\n")
  cal <- calibrate(1, 1)
  d   <- make_data(n_rct, n_rwd, 1, 1, cal)
  el  <- system.time(fit_rct_S(d, d$X, DEGREE, MSTOP, NU))[["elapsed"]]
  # 6 fits per replication: S = 1, T = 2, for each of RCT and pooled.
  per_rep <- 6 * (1 + B_boot) * el
  cat(sprintf("  one S-learner fit      : %.1f s\n", el))
  cat(sprintf("  per replication (B=%d) : %.1f min\n", B_boot, per_rep / 60))
  cat(sprintf("  M=%d, %d cell(s)       : %.1f core-hours\n",
              M, nrow(grid), M * nrow(grid) * per_rep / 3600))
  cat(sprintf("  on %d cores            : %.1f hours\n",
              num_cores, M * nrow(grid) * per_rep / 3600 / num_cores))
  quit(save = "no")
}

registerDoParallel(cores = num_cores)
cat("SIMULATION (competitors): BJ-trees RCT/Pooled x S/T-learner\n")
cat("Cores in use (1 free):", num_cores, "\n")
cat("RCT and RWD both right-censored ~35% (Buckley-James takes right censoring only).\n")
cat("Covariates: MVN AR(1), rho =", rho_X, "\n")
cat(sprintf("M = %d reps/cell, n_rct = %d, n_rwd = %d, B_boot = %d, degree = %d, mstop = %d\n",
            M, n_rct, n_rwd, B_boot, DEGREE, MSTOP))
cat(sprintf("Grid: %d (lambda_d, lambda_u) cells\n", nrow(grid)))

run_started <- Sys.time()

all_cells <- vector("list", nrow(grid))
for (g in seq_len(nrow(grid))) {
  ld <- grid$lambda_d[g]; lu <- grid$lambda_u[g]
  cat(sprintf("\nCell %d/%d: lambda_d = %.1f, lambda_u = %.1f\n",
              g, nrow(grid), ld, lu))
  all_cells[[g]] <- run_simulation_tidy(
    n_rep = M, n_rct = n_rct, n_rwd = n_rwd, ld = ld, lu = lu,
    degree = DEGREE, mstop = MSTOP, nu = NU, B_boot = B_boot)
}
final_flat_df <- do.call(rbind, all_cells)

elapsed <- as.numeric(difftime(Sys.time(), run_started, units = "secs"))
cat(sprintf("\nReplications took %.1f min on %d workers.\n", elapsed / 60, num_cores))

# On the HPC write to TMPDIR; locally write next to the other outputs.
out_dir <- if (nzchar(Sys.getenv("TMPDIR"))) Sys.getenv("TMPDIR") else
  file.path("simulations", "exp2_competitor")
output_file <- file.path(out_dir, "sim_surv_v12_bjtree_output.rds")
cat("\nSaving all results to:", output_file, "\n")
saveRDS(final_flat_df, file = output_file)

# ── Quick summary ─────────────────────────────────────────────────────────────
res <- final_flat_df
mean_tbl <- aggregate(cbind(rmse, bias, coverage, width, postvar) ~
                        lambda_d + lambda_u + method + population,
                      data = res, FUN = function(x) mean(x, na.rm = TRUE),
                      na.action = na.pass)
var_tbl  <- aggregate(bias ~ lambda_d + lambda_u + method + population,
                      data = res, FUN = function(x) var(x, na.rm = TRUE),
                      na.action = na.pass)
names(var_tbl)[ncol(var_tbl)] <- "bias_var"
summary_tbl <- merge(mean_tbl, var_tbl)
summary_tbl$sd <- sqrt(summary_tbl$bias_var)

summary_tbl$population <- factor(summary_tbl$population,
                                 levels = c("RCT", "RWD", "All"))
ord <- order(summary_tbl$lambda_d, summary_tbl$lambda_u,
             summary_tbl$population, summary_tbl$method)
tbl <- summary_tbl[ord, c("lambda_d", "lambda_u", "population", "method",
                          "rmse", "bias", "bias_var", "sd",
                          "coverage", "width", "postvar")]

cat("\n=== BJ-trees CATE metrics (mean over replications) ===\n")
print(tbl, row.names = FALSE, digits = 3)
cat(sprintf("\nMean censoring: RCT %.1f%%, RWD %.1f%%\n",
            100 * mean(res$cens_rct), 100 * mean(res$cens_rwd)))

csv_file <- sub("_output\\.rds$", "_summary.csv", output_file)
txt_file <- sub("_output\\.rds$", "_summary.txt", output_file)

write.csv(summary_tbl[ord, c("method", "population", "lambda_d", "lambda_u",
                             "rmse", "bias", "coverage", "width", "postvar",
                             "bias_var")],
          file = csv_file, row.names = FALSE)

writeLines(c(
  "BJ-trees (bujar) competitor simulation",
  sprintf("finished         : %s", format(Sys.time(), "%Y-%m-%d %H:%M:%S")),
  "censoring design : right (both sources right-censored)",
  sprintf("replications (M) : %d", M),
  sprintf("bootstrap (B)    : %d", B_boot),
  sprintf("n_rct / n_rwd    : %d / %d", n_rct, n_rwd),
  sprintf("degree / mstop   : %d / %d", DEGREE, MSTOP),
  sprintf("grid cells       : %d", nrow(grid)),
  sprintf("mean censoring   : RCT %.1f%%, RWD %.1f%%",
          100 * mean(res$cens_rct), 100 * mean(res$cens_rwd)),
  "",
  capture.output(print(tbl, row.names = FALSE, digits = 3))
), con = txt_file)

cat("Summary written to:\n  ", csv_file, "\n  ", txt_file, "\n", sep = "")
