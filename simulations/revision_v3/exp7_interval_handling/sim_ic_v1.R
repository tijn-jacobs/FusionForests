# ══════════════════════════════════════════════════════════════════════════════
# sim_ic_v1.R -- experiment 7: what handling interval censoring correctly is worth
# (referee 2, comment 1).
#
# DGP of experiment 1 (sim_surv_v13.R, copied verbatim) at lambda_d = lambda_u = 1
# with constant delta.  On the same datasets, the real-world records enter the
# fusion forest in four ways:
#   "bounds"   the interval bounds, as the method intends;
#   "right"    right-censored at the last visit before the event, the encoding the
#              machine-learning alternatives need in the application;
#   "midpoint" an exact event at the midpoint of the interval on the time scale;
#   "exact"    the true event time when it falls before the last visit, the floor
#              of what interval censoring costs.
# Events after the last visit stay right-censored in all four.  A trial-only fit
# is the reference.  Two inspection schemes: the eight visits of experiment 1
# (deciles q10..q80) and three coarse visits (quartiles q25..q75).
#
# Recorded (long format): rmse, bias, coverage, width, postvar per target and
# population, sigma_hat, seconds, and the effective sample sizes of the average
# tau draws and of the residual scale, as evidence on mixing.  Targets: "tau",
# and for the fusion fits "c" and "d" on the RWD rows.
#
# The HPC job only stores the output (HPC contract: <base_name>_output.rds);
# analysis is local (analyse_ic_v1.R).
# ══════════════════════════════════════════════════════════════════════════════
library(FusionForests)
library(ShrinkageTrees)
library(doParallel)
library(foreach)
library(evd)
library(MASS)

# ── DGP ───────────────────────────────────────────────────────────────────────
p_total     <- 10L               # observed covariates

# Cross design: sweep one lambda over `lambda_vals` while holding the other at 1.
lambda_vals    <- c(0.0, 0.5, 1.0, 1.5, 2.0)  # swept values for each axis
lambda_d_fixed <- 1.0                          # held when sweeping lambda_u
lambda_u_fixed <- 1.0                          # held when sweeping lambda_d
sigma_rct   <- 0.75              # RCT residual SD
target_cens <- 0.35              # target right-censoring fraction per source


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

# Scenarios: the inspection scheme.
ld <- 1
lu <- 1
schemes <- list(visits_8 = seq(0.1, 0.8, 0.1), visits_3 = c(0.25, 0.5, 0.75))
handlings <- c("bounds", "right", "midpoint", "exact")

# Calibrate the RCT exponential censoring rate (time scale) to ~target.
solve_cens_rate <- function(logT, target = target_cens) {
  Tt <- exp(logT)
  uniroot(function(r) mean(1 - exp(-r * Tt)) - target, c(1e-10, 1e6))$root
}
calibrate <- function(probs, n_cal = 2e5) {
  list(rate_rct = solve_cens_rate(latent_rct(n_cal)$logT),
       visits   = as.numeric(quantile(exp(latent_rwd(n_cal, ld, lu)$logT),
                                      probs = probs)))
}

# One dataset (RCT right-censored, RWD interval-censored), on the common
# standardized log scale of experiment 1.  Also returns what the other handlings
# need: the interval midpoints and the true event times of the RWD.
make_data <- function(n_rct, n_rwd, rate_rct, visits) {
  r <- latent_rct(n_rct)
  w <- latent_rwd(n_rwd, ld, lu)
  logT_true <- c(r$logT, w$logT)
  scale_mu  <- mean(logT_true)
  scale_sd  <- sd(logT_true)
  std       <- function(z) (z - scale_mu) / scale_sd
  logfloor  <- min(logT_true) - 5           # finite stand-in for log(0)

  # --- RCT: exponential right censoring ---
  logC   <- log(rexp(n_rct, rate = rate_rct))
  y_rct  <- pmin(r$logT, logC)
  st_rct <- as.integer(r$logT <= logC)

  # --- RWD: interval censoring on the visits ---
  T0         <- exp(w$logT)
  last_visit <- visits[length(visits)]
  breaks     <- c(0, visits)
  j          <- findInterval(T0, breaks)
  beyond     <- T0 >= last_visit
  j[!beyond] <- pmax(j[!beyond], 1L)
  left_rwd   <- ifelse(j == 1L, logfloor, log(breaks[pmin(j, length(breaks))]))
  right_rwd  <- log(breaks[pmin(j + 1L, length(breaks))])
  left_rwd[beyond]  <- log(last_visit)
  right_rwd[beyond] <- log(last_visit)
  mid_rwd    <- log((breaks[pmin(j, length(breaks))] +
                       breaks[pmin(j + 1L, length(breaks))]) / 2)

  list(X = rbind(r$X, w$X), A = c(r$A, w$A),
       S = c(rep(1L, n_rct), rep(0L, n_rwd)),
       X_rct = r$X, A_rct = r$A, y_rct = std(y_rct), st_rct = st_rct,
       left_rwd = std(left_rwd), right_rwd = std(right_rwd),
       mid_rwd = std(mid_rwd), logT_rwd = std(w$logT), beyond = beyond,
       scale_sd = scale_sd, n_rct = n_rct, n_rwd = n_rwd)
}

# The outcome arguments of the fusion fit under one handling.  Events after the
# last visit are right-censored at it throughout; the trial rows never change.
encode <- function(d, handling) {
  inside <- !d$beyond
  y      <- d$right_rwd
  status <- rep(0L, d$n_rwd)
  left   <- d$left_rwd
  right  <- d$right_rwd
  icc    <- as.integer(inside)
  if (handling == "right") {
    right[inside] <- left[inside]
    icc[inside]   <- 0L
  } else if (handling %in% c("midpoint", "exact")) {
    z <- if (handling == "midpoint") d$mid_rwd else d$logT_rwd
    y[inside]      <- z[inside]
    left[inside]   <- z[inside]
    right[inside]  <- z[inside]
    status[inside] <- 1L
    icc[inside]    <- 0L
  }
  list(y = c(d$y_rct, y), status = c(d$st_rct, status),
       left = c(d$y_rct, left), right = c(d$y_rct, right),
       icc = c(rep(0L, d$n_rct), icc))
}

# ── Fits ──────────────────────────────────────────────────────────────────────
fit_fusion <- function(d, e, N_post, N_burn) {
  FusionForest(
    y                            = e$y,
    status                       = e$status,
    observed_left_time           = e$left,
    observed_right_time          = e$right,
    interval_censoring_indicator = e$icc,
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
    error_dist                = "source_hdp_scale",
    number_of_trees_control = 200,
    number_of_trees_treat = 100,
    number_of_trees_deconf = 50,
    number_of_trees_deviation = 50,
    k_treat                   = 1,
    N_post = N_post, N_burn = N_burn,
    treatment_coding = "binary",
    store_posterior_sample = TRUE, verbose = FALSE)
}

# Single-source priors as in experiment 1.
n_trees_csf <- 200L
fit_trial <- function(d, N_post, N_burn) {
  CausalShrinkageForest(
    y                         = d$y_rct,
    status                    = d$st_rct,
    X_train_control           = d$X_rct,
    X_train_treat             = d$X_rct,
    treatment_indicator_train = d$A_rct,
    X_test_control            = d$X,
    X_test_treat              = d$X,
    treatment_indicator_test  = rep(1L, nrow(d$X)),
    outcome_type              = "right-censored",
    timescale                 = "log",
    prior_type_control        = "standard",
    prior_type_treat          = "standard",
    local_hp_control          = 1 / sqrt(n_trees_csf),
    local_hp_treat            = 1 / sqrt(100),
    number_of_trees_control   = n_trees_csf,
    number_of_trees_treat     = 100,
    power_control = 2,
    power_treat = 2,
    base_control = 0.95,
    base_treat = 0.95,
    N_post = N_post, N_burn = N_burn,
    treatment_coding = "binary",
    store_posterior_sample = TRUE, verbose = FALSE)
}

# ── Metrics ───────────────────────────────────────────────────────────────────
as_obs_by_draw <- function(samp, n_obs) {
  if (nrow(samp) == n_obs) samp else t(samp)
}
eval_fun <- function(point, samp, truth, idx) {
  p  <- point[idx]; tr <- truth[idx]
  M  <- as_obs_by_draw(samp, length(point))[idx, , drop = FALSE]
  ci <- apply(M, 1, quantile, probs = c(0.025, 0.975))
  c(rmse     = sqrt(mean((p - tr)^2)),
    bias     = mean(p - tr),
    coverage = mean(tr >= ci[1, ] & tr <= ci[2, ]),
    width    = mean(ci[2, ] - ci[1, ]),
    postvar  = mean(apply(M, 1, var)))
}
metric_cols <- c("rmse", "bias", "coverage", "width", "postvar")
make_rows <- function(method, target, point, samp, truth, pops) {
  do.call(rbind, lapply(names(pops), function(pop) {
    vals <- if (is.null(point)) setNames(rep(NA_real_, length(metric_cols)), metric_cols)
    else eval_fun(point, samp, truth, pops[[pop]])
    data.frame(method = method, target = target, population = pop, as.list(vals),
               stringsAsFactors = FALSE)
  }))
}
# Effective sample size by the initial positive sequence of autocorrelations.
ess <- function(x) {
  n <- length(x)
  if (n < 10L || !is.finite(var(x)) || var(x) == 0) return(NA_real_)
  rho  <- acf(x, lag.max = min(n - 1L, 1000L), plot = FALSE)$acf[-1]
  stop <- which(rho <= 0)
  k    <- if (length(stop)) stop[1] - 1L else length(rho)
  n / (1 + 2 * sum(rho[seq_len(k)]))
}

# ── One replication: all fits on one dataset ─────────────────────────────────
run_one_sim <- function(seed, n_rct, n_rwd, N_post, N_burn, cal) {
  set.seed(seed)
  d     <- make_data(n_rct, n_rwd, cal$rate_rct, cal$visits)
  n_all <- n_rct + n_rwd
  i_rwd <- n_rct + seq_len(n_rwd)
  pops  <- list(RCT = seq_len(n_rct), RWD = i_rwd, All = seq_len(n_all))
  safe  <- function(expr) tryCatch(expr, error = function(e) NULL)
  s     <- d$scale_sd
  sc    <- function(x) if (is.null(x)) NULL else x * s
  tau_t <- tau(d$X)

  res <- NULL
  for (h in handlings) {
    secs <- system.time(
      ff <- safe(fit_fusion(d, encode(d, h), N_post, N_burn))
    )[["elapsed"]]
    ts   <- sc(ff$test_predictions_sample_treat)
    rows <- rbind(
      make_rows(h, "tau", sc(ff$test_predictions_treat), ts, tau_t, pops),
      make_rows(h, "c", sc(ff$test_predictions_deconf),
                sc(ff$test_predictions_sample_deconf),
                rep(lu * delta_u, n_all), list(RWD = i_rwd)),
      make_rows(h, "d", sc(ff$test_predictions_deviation),
                sc(ff$test_predictions_sample_deviation),
                dev(d$X, ld), list(RWD = i_rwd))
    )
    rows$seconds   <- secs
    rows$sigma_hat <- if (is.null(ff)) NA_real_ else ff$meta$sigma_hat
    rows$ess_tau   <- if (is.null(ts)) NA_real_ else
      ess(colMeans(as_obs_by_draw(ts, n_all)))
    rows$ess_sigma <- if (is.null(ff)) NA_real_ else
      ess(ff$sigma[(length(ff$sigma) - N_post + 1L):length(ff$sigma)])
    res <- rbind(res, rows)
  }
  rct  <- safe(fit_trial(d, N_post, N_burn))
  rows <- make_rows("trial", "tau", sc(rct$test_predictions_treat),
                    sc(rct$test_predictions_sample_treat), tau_t, pops)
  rows$seconds <- NA_real_; rows$sigma_hat <- NA_real_
  rows$ess_tau <- NA_real_; rows$ess_sigma <- NA_real_
  res <- rbind(res, rows)
  res$ic_rwd <- mean(!d$beyond)                # RWD share with a finite interval
  res
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
cat("SIMULATION: handling of interval censoring (sim_ic_v1, FusionForests",
    as.character(packageVersion("FusionForests")), ") -- HPC\n")

M      <- 1000L                  # replications per scheme (paired across handlings)
n_rct  <- 150L
n_rwd  <- 350L
N_post <- 5000L
N_burn <- 5000L
cat(sprintf("M = %d reps/scheme, %d schemes, N_post = %d, N_burn = %d\n",
            M, length(schemes), N_post, N_burn))

all_cells <- vector("list", length(schemes))
for (g in seq_along(schemes)) {
  cat(sprintf("\nScheme %d/%d: %s\n", g, length(schemes), names(schemes)[g]))
  set.seed(200000L + 1000L * g)
  cal <- calibrate(schemes[[g]])
  all_cells[[g]] <- foreach(
    i = seq_len(M), .combine = "rbind",
    .packages = c("FusionForests", "ShrinkageTrees", "evd", "MASS")
  ) %dopar% {
    res <- run_one_sim(200000L + 1000L * g + i, n_rct, n_rwd, N_post, N_burn, cal)
    res$scheme <- names(schemes)[g]
    res$Iter   <- i
    res
  }
}
final_flat_df <- do.call(rbind, all_cells)

# Output file path (! NAME MUST BE <base_name>_output.rds !).
out_dir <- Sys.getenv("TMPDIR")
if (!nzchar(out_dir)) out_dir <- "."   # interactive run: current directory
output_file <- file.path(out_dir, "sim_ic_v1_output.rds")
cat("\nSaving all results to:", output_file, "\n")
saveRDS(final_flat_df, file = output_file)
cat("All results successfully saved in one file.\n")

# Local analysis (tables): analyse_ic_v1.R in this folder.
