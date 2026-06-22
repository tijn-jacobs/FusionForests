# ──────────────────────────────────────────────────────────────────────────────
# Competitor simulation: Random Survival Forest (RSF) vs the main FusionForest
# survival simulation.
#
# Reuses the DGP, calibration and evaluation of simulations/exp1_confounding_heterogeneity/sim_surv_v12.R
# (MVN AR(1) covariates), but (a) makes BOTH sources right-censored at ~35% --
# no interval censoring -- for a fair comparison, and (b) swaps the estimators
# for a random survival forest (rfsrc from the `randomForestSRC` package).
#
#   install.packages("randomForestSRC")
#   library(randomForestSRC); ?rfsrc
#
# RSF predicts a survival curve, not a log-time score.  To get a CATE comparable
# to tau(X) (the additive effect on log T), we compute the expected log survival
# time E[log T | x, a] from each predicted curve and difference the A = 1 / A = 0
# counterfactuals (S-learner).  RSF has no posterior, so only point metrics
# (rmse, bias) are reported; coverage / width / postvar are NA.
#
# Output columns match sim_surv_v12.R so the rows can be row-bound onto the main
# results (and onto the deepAFT competitor) for a side-by-side comparison.
# ──────────────────────────────────────────────────────────────────────────────

library(randomForestSRC)
library(survival)
library(doParallel)
library(foreach)
library(evd)
library(MASS)

# IMPORTANT (HPC): rfsrc uses OpenMP and defaults to detectCores() threads,
# ignoring OMP_NUM_THREADS.  We parallelise across replications with foreach, so
# each rfsrc call MUST be single-threaded -- otherwise the forked workers x rfsrc
# threads oversubscribe the node (e.g. 191 forks x 191 threads) and it hangs.
options(rf.cores = 1, mc.cores = 1)

# ── DGP ───────────────────────────────────────────── (verbatim from v12) ──────
p_total     <- 10L               # observed covariates

lambda_vals    <- c(0.0, 0.5, 1.0, 1.5, 2.0)
lambda_d_fixed <- 1.0
lambda_u_fixed <- 1.0
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
  list(X = X, A = A, e = e, eps = eps,
       logT = m0(X) + dev(X, ld) + A * tau(X) + A * conf(U, lu) + eps)
}

solve_cens_rate <- function(logT, target = target_cens) {
  Tt <- exp(logT)
  uniroot(function(r) mean(1 - exp(-r * Tt)) - target, c(1e-10, 1e6))$root
}

# Both sources get an independent exponential right-censoring rate tuned to
# ~target_cens.
calibrate <- function(ld, lu, n_cal = 1e5) {
  rate_rct <- solve_cens_rate(latent_rct(n_cal)$logT)
  rate_rwd <- solve_cens_rate(latent_rwd(n_cal, ld, lu)$logT)
  list(rate_rct = rate_rct, rate_rwd = rate_rwd)
}

# Both sources are right-censored: independent exponential censoring on the time
# scale tuned to ~target_cens.  Outcomes are on the standardised LOG-time scale.
make_data <- function(n_rct, n_rwd, ld, lu, rate_rct, rate_rwd) {
  r <- latent_rct(n_rct)
  w <- latent_rwd(n_rwd, ld, lu)
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

# ── RSF competitors (S- and T-learner) ───────────────────────────────────────
# rfsrc predicts a survival curve S(t | z).  We turn that into a log-time score
#
#   mu(z) = E[log T | z] ~ sum_k log(t_k) * (S_{k-1} - S_k) + log(t_K) * S_K,
#
# i.e. the mean of log T under the predicted distribution, with the residual mass
# beyond the last event time t_K lumped at t_K (a winsorised mean -- the survival
# curve is undefined past the last event, and the bias largely cancels in the
# arm difference).  Two meta-learners turn mu into a CATE:
#   S-learner: one forest with A as a covariate, tau(x) = mu(x,A=1) - mu(x,A=0).
#   T-learner: a separate forest per arm,        tau(x) = mu_1(x) - mu_0(x).

# Expected log survival time per subject from a survival matrix.
# surv: n x K, columns = S(t_k | x_i); times: length-K positive event grid.
expected_logtime <- function(surv, times) {
  surv  <- as.matrix(surv)
  K     <- ncol(surv)
  logt  <- log(times)
  S_prev <- cbind(1, surv[, -K, drop = FALSE])    # S_{k-1}, with S_0 = 1
  dF     <- S_prev - surv                          # probability mass at t_k
  as.numeric(dF %*% logt) + logt[K] * surv[, K]    # winsorise tail at t_K
}

# Each "design" packages the right-censored training data (time, status, X, A)
# and the evaluation covariates Xev for one population; the S- and T-learners
# below consume a design.
#   RCT-only : covariates X, treatment A.
#   Pooled   : RCT + RWD, with the source indicator S (1 = RCT, 0 = RWD) as an
#              extra covariate; the CATE is evaluated at S = 1 (the trial
#              estimand / RCT population).
design_rct <- function(d, Xev) {
  list(time = exp(d$y_rct), status = d$st_rct,
       X = d$X_rct, A = d$A_rct, Xev = Xev)
}
design_pool <- function(d, Xev) {
  S <- c(rep(1L, d$n_rct), rep(0L, d$n_rwd))
  list(time   = c(exp(d$y_rct), exp(d$y_rwd)),
       status = c(d$st_rct, d$st_rwd),
       X      = cbind(rbind(d$X_rct, d$X_rwd), S = S),
       A      = c(d$A_rct, d$A_rwd),
       Xev    = cbind(Xev, S = 1L))
}

# Named covariate data.frame (identical column construction for train and
# evaluation so rfsrc's predict() lines the columns up).
rsf_frame <- function(X) {
  fr <- as.data.frame(as.matrix(X))
  names(fr) <- paste0("x", seq_len(ncol(fr)))
  fr
}

# Fit one RSF and return E[log T] on newX (no treatment counterfactual).
rsf_mu <- function(time, status, X, newX) {
  tr <- rsf_frame(X); tr$time <- time; tr$status <- status
  fit <- rfsrc(Surv(time, status) ~ ., data = tr, ntree = 500)
  p   <- predict(fit, newdata = rsf_frame(newX))
  expected_logtime(p$survival, p$time.interest)
}

# S-learner: one forest with A as a covariate; CATE = mu(x, A=1) - mu(x, A=0).
cate_S <- function(des) {
  tr <- rsf_frame(cbind(des$X, A = des$A))
  tr$time <- des$time; tr$status <- des$status
  fit <- rfsrc(Surv(time, status) ~ ., data = tr, ntree = 500)
  pred_a <- function(a) {
    p <- predict(fit, newdata = rsf_frame(cbind(des$Xev, A = a)))
    expected_logtime(p$survival, p$time.interest)
  }
  pred_a(1) - pred_a(0)
}

# T-learner: a separate forest per arm; CATE = mu_1(x) - mu_0(x).
cate_T <- function(des) {
  arm <- function(keep) rsf_mu(des$time[keep], des$status[keep],
                               des$X[keep, , drop = FALSE], des$Xev)
  arm(des$A == 1) - arm(des$A == 0)
}

# The four estimators (signature (d, Xev) so boot_cate can refit them).
fit_rct_S  <- function(d, Xev) cate_S(design_rct(d, Xev))
fit_rct_T  <- function(d, Xev) cate_T(design_rct(d, Xev))
fit_pool_S <- function(d, Xev) cate_S(design_pool(d, Xev))
fit_pool_T <- function(d, Xev) cate_T(design_pool(d, Xev))

# ── Metrics ───────────────────────────────────────────────────────────────────
# With bootstrap draws (n_eval x B) the interval metrics mirror the main sim's
# eval_cate: the bootstrap percentile CI gives coverage / width, and the per-
# point bootstrap variance fills `postvar` (the frequentist analog of the
# Bayesian posterior variance).  With draws = NULL only rmse / bias are computed.
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

# ── Bootstrap uncertainty (S-learner) ─────────────────────────────────────────
# Stratified nonparametric bootstrap: resample the RCT and RWD training rows
# separately (keeping n_rct / n_rwd fixed), refit, and predict the CATE on the
# fixed evaluation set Xev.  B refits give a draws matrix (n_eval x B) for the
# interval metrics.  The evaluation covariates d$X and the truth are NOT
# resampled.
resample_data <- function(d) {
  i_rct <- sample.int(d$n_rct, d$n_rct, replace = TRUE)
  i_rwd <- sample.int(d$n_rwd, d$n_rwd, replace = TRUE)
  d$X_rct  <- d$X_rct[i_rct, , drop = FALSE]; d$A_rct <- d$A_rct[i_rct]
  d$y_rct  <- d$y_rct[i_rct];                 d$st_rct <- d$st_rct[i_rct]
  d$X_rwd  <- d$X_rwd[i_rwd, , drop = FALSE]; d$A_rwd <- d$A_rwd[i_rwd]
  d$y_rwd  <- d$y_rwd[i_rwd];                 d$st_rwd <- d$st_rwd[i_rwd]
  d
}

# B bootstrap CATE vectors -> n_eval x B (failed refits dropped).
boot_cate <- function(fit_one, d, Xev, B, ...) {
  n <- nrow(Xev)
  cols <- lapply(seq_len(B), function(b)
    tryCatch(fit_one(resample_data(d), Xev, ...),
             error = function(e) rep(NA_real_, n)))
  draws <- do.call(cbind, cols)
  keep  <- colSums(is.na(draws)) < n
  if (!any(keep)) NULL else draws[, keep, drop = FALSE]
}

# ── One replication: four RSF estimators (RCT/Pooled x S/T) on one dataset ─────
run_one_sim <- function(seed, n_rct, n_rwd, ld, lu, rate_rct, rate_rwd) {
  set.seed(seed)
  d     <- make_data(n_rct, n_rwd, ld, lu, rate_rct, rate_rwd)
  truth <- tau(d$X)
  n_all <- n_rct + n_rwd
  safe  <- function(expr) tryCatch(expr, error = function(e) {
    message("    RSF fit failed: ", conditionMessage(e)); NULL })

  s   <- d$scale_sd                  # standardised-log CATE -> log-time scale
  scl <- function(z) if (is.null(z)) NULL else z * s

  # RCT and pooled designs each get an S-learner and a T-learner.  Point estimate
  # from the full fit; interval metrics from a stratified bootstrap (skipped when
  # B_boot == 0).
  estimators <- list(
    list(name = "RSF-RCT-S",    fun = fit_rct_S),
    list(name = "RSF-RCT-T",    fun = fit_rct_T),
    list(name = "RSF-Pooled-S", fun = fit_pool_S),
    list(name = "RSF-Pooled-T", fun = fit_pool_T)
  )
  res <- do.call(rbind, lapply(estimators, function(est) {
    point <- safe(est$fun(d, d$X))
    draws <- if (B_boot > 0L && !is.null(point))
      safe(boot_cate(est$fun, d, d$X, B_boot)) else NULL
    make_rows(est$name, scl(point), scl(draws), truth, n_rct, n_all)
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
    .packages = c("randomForestSRC", "survival", "evd", "MASS")
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
cat("SIMULATION (competitors): RSF RCT/Pooled x S/T-learner -- HPC\n")
cat("RCT and RWD both right-censored ~35%.\n")
cat("Covariates: MVN AR(1), rho =", rho_X, "\n")

# Simulation settings.
M       <- 1000L                 # replications per grid cell
n_rct   <- 150L                  # v12 settings
n_rwd   <- 350L                  # v12 settings
B_boot  <- 100L                   # bootstrap refits per fit for CATE intervals
                                 # (0 = point only).  RSF fits are fast, so this
                                 # can be larger than the deepAFT competitor.
cat(sprintf("M = %d reps/cell, n_rct = %d, n_rwd = %d, B_boot = %d\n",
            M, n_rct, n_rwd, B_boot))

# Full 3x3 grid: lambda_d and lambda_u each in {0, 1, 2} (nine cells).
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
output_file <- file.path(Sys.getenv("TMPDIR"), "sim_surv_v12_rsf_output.rds")
cat("\nSaving all results to:", output_file, "\n")
saveRDS(final_flat_df, file = output_file)
cat("All results successfully saved in one file.\n")

# ── Quick summary ─────────────────────────────────────────────────────────────
# rmse/bias/coverage/width/postvar are means over replications; `variance` is the
# across-replication variance of the integrated bias (sd = sqrt of that).
res <- final_flat_df
mean_tbl <- aggregate(cbind(rmse, bias, coverage, width, postvar) ~
                        lambda_d + lambda_u + method + population,
                      data = res, FUN = function(x) mean(x, na.rm = TRUE),
                      na.action = na.pass)
var_tbl  <- aggregate(bias ~ lambda_d + lambda_u + method + population,
                      data = res, FUN = function(x) var(x, na.rm = TRUE),
                      na.action = na.pass)
names(var_tbl)[ncol(var_tbl)] <- "variance"
summary_tbl <- merge(mean_tbl, var_tbl)
summary_tbl$sd <- sqrt(summary_tbl$variance)

summary_tbl$population <- factor(summary_tbl$population,
                                 levels = c("RCT", "RWD", "All"))
ord <- order(summary_tbl$lambda_d, summary_tbl$lambda_u,
             summary_tbl$population, summary_tbl$method)
cat("\n=== RSF CATE metrics (mean over replications) ===\n")
print(summary_tbl[ord, c("lambda_d", "lambda_u", "population", "method",
                         "rmse", "bias", "variance", "sd",
                         "coverage", "width", "postvar")],
      row.names = FALSE, digits = 3)
