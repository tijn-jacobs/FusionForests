# ══════════════════════════════════════════════════════════════════════════════
# sim_ktau_v1.R -- pilot for the treatment-effect step-height scale k_tau, on the
# final package and the final confounder design, before the full reruns.  Not
# reported in the paper.
#
# Why: k_tau = 1 was chosen under the old censored_info(), when sigma_hat was
# about 1.6 on these data.  With sigma_hat about 1 the same k gives a prior about
# 40% smaller on the log-time scale, and the old prior size needs k of about 1.6.
#
# The DGP, fit and metrics are those of experiment 5 (sim_prior_v1.R), copied
# verbatim.  Only k_tau varies, on paired data, in the two cells of experiment 5:
#   "uniform"       lambda_d = 1, lambda_u = 1, constant confounding;
#   "heterogeneous" lambda_d = 1, lambda_u = 2, covariate-dependent confounding.
# 5 values x 2 cells x M = 200 at 2000/2000 draws: 2000 fits.
#
# Each row also records sigma_hat and y_mean.  They should be close to 1 and 0;
# values near 1.6 mean the HPC is not running the fixed package.
# The HPC job only stores the output (HPC contract: <base_name>_output.rds);
# analysis is local (analyse_ktau_v1.R).
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
visit_probs <- seq(0.1, 0.8, 0.1) # RWD inspection visits: deciles q10..q80


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
# P(cens) = E[1 - exp(-rate * T)], monotone increasing in rate.
solve_cens_rate <- function(logT, target = target_cens) {
  Tt <- exp(logT)
  uniroot(function(r) mean(1 - exp(-r * Tt)) - target, c(1e-10, 1e6))$root
}

# Per-cell calibration, fixed once on a large pre-simulation:
#   rate_rct -- RCT exponential censoring rate.
#   visits   -- RWD inspection times (time scale) at the deciles q10..q80.
calibrate <- function(ld, lu, n_cal = 2e5) {
  rate_rct <- solve_cens_rate(latent_rct(n_cal)$logT)
  visits   <- as.numeric(quantile(exp(latent_rwd(n_cal, ld, lu)$logT),
                                  probs = visit_probs))
  list(rate_rct = rate_rct, visits = visits)
}

# Simulate one full dataset (RCT right-censored, RWD interval-censored).
# All time bounds are returned on the LOG scale (timescale = "log").
make_data <- function(n_rct, n_rwd, ld, lu, rate_rct, visits) {
  r <- latent_rct(n_rct)
  w <- latent_rwd(n_rwd, ld, lu)
  # Standardise the true (latent) log-survival of the pooled sources to mean 0,
  # unit variance.  The same affine map `std` is applied to every log-time bound
  # constructed below, so all fits operate on a common scale.  Predicted CATEs
  # are on this scale and are rescaled by `scale_sd` before evaluation.
  logT_true <- c(r$logT, w$logT)
  scale_mu  <- mean(logT_true)
  scale_sd  <- sd(logT_true)
  std       <- function(z) (z - scale_mu) / scale_sd
  logfloor <- min(c(r$logT, w$logT)) - 5    # finite stand-in for log(0)
  
  # --- RCT: exponential right censoring ---
  logC   <- log(rexp(n_rct, rate = rate_rct))
  y_rct  <- pmin(r$logT, logC)
  st_rct <- as.integer(r$logT <= logC)
  icc_rct   <- rep(0L, n_rct)               # 0 = not interval-censored
  left_rct  <- y_rct                        # ignored for status==1 / right-cens
  right_rct <- y_rct
  
  # --- RWD: interval censoring on the decile visits ---
  T0         <- exp(w$logT)
  last_visit <- visits[length(visits)]      # q80
  breaks     <- c(0, visits)                # bin edges: 0, q10, ..., q80
  j          <- findInterval(T0, breaks)    # 1..length(visits) below q80
  beyond     <- T0 >= last_visit            # event after the last visit
  j[!beyond] <- pmax(j[!beyond], 1L)
  
  left_rwd  <- ifelse(j == 1L, logfloor, log(breaks[pmin(j, length(breaks))]))
  right_rwd <- log(breaks[pmin(j + 1L, length(breaks))])
  st_rwd    <- rep(0L, n_rwd)               # no exact event times in the RWD
  icc_rwd   <- rep(1L, n_rwd)               # interval-censored by default
  # events beyond the last visit: right-censored at q80 (icc = 0)
  left_rwd[beyond]  <- log(last_visit)
  right_rwd[beyond] <- log(last_visit)
  icc_rwd[beyond]   <- 0L
  y_rwd <- right_rwd                        # placeholder; bounds carry the info

  # --- standardise all log-time bounds to the common scale ---
  y_rct     <- std(y_rct)
  left_rct  <- std(left_rct)
  right_rct <- std(right_rct)
  left_rwd  <- std(left_rwd)
  right_rwd <- std(right_rwd)
  y_rwd     <- std(y_rwd)

  list(
    X_rct = r$X, X_rwd = w$X,
    A_rct = r$A, A_rwd = w$A,
    y_rct = y_rct, y_rwd = y_rwd,
    st_rct = st_rct, st_rwd = st_rwd,
    left_rct = left_rct, right_rct = right_rct, icc_rct = icc_rct,
    left_rwd = left_rwd, right_rwd = right_rwd, icc_rwd = icc_rwd,
    X      = rbind(r$X, w$X),
    A      = c(r$A, w$A),
    S      = c(rep(1L, n_rct), rep(0L, n_rwd)),
    y      = c(y_rct, y_rwd),
    status = c(st_rct, st_rwd),
    left   = c(left_rct, left_rwd),
    right  = c(right_rct, right_rwd),
    icc    = c(icc_rct, icc_rwd),
    # True RWD propensity score P(A = 1 | X).  e_rwd: training-row
    # propensity for the RWD-only fit.  e_all: propensity on the combined
    # evaluation set, with 0.5 for the randomised RCT rows.
    e_rwd  = w$e,
    e_all  = c(rep(0.5, n_rct), w$e),
    # SD used to standardise the pooled latent log-survival.  Predicted CATEs
    # come back on the standardised scale and are multiplied by this to recover
    # the log-time scale before evaluation against tau(X).
    scale_sd = scale_sd,
    n_rct = n_rct, n_rwd = n_rwd
  )
}

# ── Prior configurations: only k_tau varies ──────────────────────────────────
default_prior <- list(
  k_control = 1, k_deviation = 1, k_treat = 1, k_deconf = 1,
  base_control = 0.95, power_control = 2,
  base_deviation = 0.95, power_deviation = 2,
  base_treat = 0.95, power_treat = 3,
  base_deconf = 0.25, power_deconf = 3,
  error_atom_scale = 0.5
)
one_at_a_time <- function(name, ...) {
  p <- modifyList(default_prior, list(...))
  list(name = name, prior = p)
}
configs <- list(
  one_at_a_time("k_tau=0.75", k_treat = 0.75),
  one_at_a_time("k_tau=1",    k_treat = 1),
  one_at_a_time("k_tau=1.25", k_treat = 1.25),
  one_at_a_time("k_tau=1.5",  k_treat = 1.5),
  one_at_a_time("k_tau=2",    k_treat = 2)
)
config_names <- vapply(configs, `[[`, "", "name")

cells <- data.frame(cell        = c("uniform", "heterogeneous"),
                    design      = c("uniform", "heterogeneous"),
                    lambda_d    = c(1, 1),
                    lambda_u    = c(1, 2),
                    seed_offset = c(50000L, 60000L),
                    stringsAsFactors = FALSE)

# ── Fusion fit with an explicit prior ─────────────────────────────────────────
fit_fusion_prior <- function(d, N_post, N_burn, pr) {
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
    error_dist                = "source_hdp_scale",
    number_of_trees_control   = 200,
    number_of_trees_treat     = 100,
    number_of_trees_deconf    = 50,
    number_of_trees_deviation = 50,
    k_control = pr$k_control, k_deviation = pr$k_deviation,
    k_treat   = pr$k_treat,   k_deconf    = pr$k_deconf,
    base_control   = pr$base_control,   power_control   = pr$power_control,
    base_deviation = pr$base_deviation, power_deviation = pr$power_deviation,
    base_treat     = pr$base_treat,     power_treat     = pr$power_treat,
    base_deconf    = pr$base_deconf,    power_deconf    = pr$power_deconf,
    error_atom_scale = pr$error_atom_scale,
    N_post = N_post, N_burn = N_burn,
    treatment_coding = "binary",
    store_posterior_sample = TRUE, verbose = FALSE)
}

# ── True c(x) and d(x) on the model's scale ───────────────────────────────────
# E[U | A = a, x] = a delta(x), so the RWD baseline has no contribution from U.
true_c <- function(X, lu) lu * rep_len(delta_x(X), nrow(X))
true_d <- function(X, ld, lu) dev(X, ld)

# ── Metrics ───────────────────────────────────────────────────────────────────
as_draws_by_obs <- function(samp, n_obs) if (ncol(samp) == n_obs) samp else t(samp)
fun_metrics <- function(point, samp, truth, idx, prefix) {
  if (is.null(point) || is.null(samp)) {
    out <- rep(NA_real_, 5)
  } else {
    M  <- as_draws_by_obs(samp, length(point))[, idx, drop = FALSE]   # draws x n_idx
    ci <- apply(M, 2, quantile, probs = c(0.025, 0.975))
    p  <- point[idx]; tr <- truth[idx]
    out <- c(sqrt(mean((p - tr)^2)), mean(p - tr),
             mean(tr >= ci[1, ] & tr <= ci[2, ]), mean(ci[2, ] - ci[1, ]),
             mean(apply(M, 2, var)))
  }
  setNames(as.list(out), paste0(prefix, c("rmse", "bias", "cover", "width", "postvar")))
}
ate_metrics <- function(samp, truth, n_obs) {
  if (is.null(samp)) return(list(ate_err = NA_real_, ate_cover = NA_real_, ate_width = NA_real_))
  a  <- rowMeans(as_draws_by_obs(samp, n_obs))
  ci <- quantile(a, c(0.025, 0.975)); t0 <- mean(truth)
  list(ate_err = mean(a) - t0, ate_cover = as.numeric(t0 >= ci[1] & t0 <= ci[2]),
       ate_width = unname(ci[2] - ci[1]))
}

# ── One task: one prior configuration on one dataset ─────────────────────────
run_one_fit <- function(cell_row, cfg_idx, i, N_post, N_burn, n_rct, n_rwd, cal) {
  cl <- cells[cell_row, ]
  conf_design <<- cl$design                    # read by delta_x()
  seed <- cl$seed_offset + i
  set.seed(seed)
  d     <- make_data(n_rct, n_rwd, cl$lambda_d, cl$lambda_u, cal$rate_rct, cal$visits)
  n_all <- n_rct + n_rwd
  i_rct <- seq_len(n_rct); i_rwd <- n_rct + seq_len(n_rwd)
  s     <- d$scale_sd
  tau_t <- tau(d$X)
  c_t   <- true_c(d$X, cl$lambda_u)
  d_t   <- true_d(d$X, cl$lambda_d, cl$lambda_u)

  set.seed(seed + 7919L * cfg_idx)
  ff <- tryCatch(fit_fusion_prior(d, N_post, N_burn, configs[[cfg_idx]]$prior),
                 error = function(e) {
                   message("fit failed (", cl$cell, ", ", config_names[cfg_idx],
                           ", rep ", i, "): ", conditionMessage(e)); NULL })
  sc <- function(x) if (is.null(x)) NULL else x * s
  tp <- sc(ff$test_predictions_treat);     ts <- sc(ff$test_predictions_sample_treat)
  cp <- sc(ff$test_predictions_deconf);    cs <- sc(ff$test_predictions_sample_deconf)
  dp <- sc(ff$test_predictions_deviation); ds <- sc(ff$test_predictions_sample_deviation)

  data.frame(c(
    list(cell = cl$cell, config = config_names[cfg_idx], Iter = i,
         lambda_d = cl$lambda_d, lambda_u = cl$lambda_u, failed = is.null(ff)),
    fun_metrics(tp, ts, tau_t, seq_len(n_all), "cate_all_"),
    fun_metrics(tp, ts, tau_t, i_rct,          "cate_rct_"),
    fun_metrics(tp, ts, tau_t, i_rwd,          "cate_rwd_"),
    fun_metrics(cp, cs, c_t,   i_rwd,          "c_"),
    fun_metrics(dp, ds, d_t,   i_rwd,          "d_"),
    ate_metrics(ts, tau_t, n_all),
    list(sd_c_true = sd(c_t[i_rwd]),           # how heterogeneous c is in this dataset
         sigma_hat = if (is.null(ff)) NA_real_ else ff$meta$sigma_hat,
         y_mean    = if (is.null(ff)) NA_real_ else ff$meta$y_mean,
         cens_rct = mean(d$st_rct == 0), ic_rwd = mean(d$icc_rwd == 1))
  ), stringsAsFactors = FALSE)
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
cat("SIMULATION: k_tau pilot (sim_ktau_v1, FusionForests",
    as.character(packageVersion("FusionForests")), ") -- HPC\n")

M      <- 200L                   # replications per cell (paired across configs)
n_rct  <- 150L
n_rwd  <- 350L
N_post <- 2000L
N_burn <- 2000L
cat(sprintf("M = %d reps/cell, %d configs, %d cells, N_post = %d, N_burn = %d\n",
            M, length(configs), nrow(cells), N_post, N_burn))
cat("Configs:", paste(config_names, collapse = ", "), "\n")

# Calibration once per cell (depends on the confounding design).
cals <- lapply(seq_len(nrow(cells)), function(r) {
  conf_design <<- cells$design[r]
  set.seed(cells$seed_offset[r])
  cal <- calibrate(cells$lambda_d[r], cells$lambda_u[r])
  cat(sprintf("  %s: RCT cens rate = %.4g | RWD visits = %s\n", cells$cell[r],
              cal$rate_rct, paste(round(cal$visits, 2), collapse = ", ")))
  cal
})

tasks <- expand.grid(i = seq_len(M), cfg = seq_along(configs),
                     cell = seq_len(nrow(cells)))
cat(sprintf("Tasks: %d single fusion fits\n", nrow(tasks)))

res <- foreach(
  t = seq_len(nrow(tasks)), .combine = "rbind",
  .packages = c("FusionForests", "evd", "MASS")
) %dopar% {
  run_one_fit(tasks$cell[t], tasks$cfg[t], tasks$i[t], N_post, N_burn,
              n_rct, n_rwd, cals[[tasks$cell[t]]])
}

# Output file path (! NAME MUST BE <base_name>_output.rds !).
out_dir <- Sys.getenv("TMPDIR")
if (!nzchar(out_dir)) out_dir <- "."   # interactive run: current directory
output_file <- file.path(out_dir, "sim_ktau_v1_output.rds")
cat("\nSaving all results to:", output_file, "\n")
saveRDS(res, file = output_file)
cat("All results successfully saved in one file.\n")

# Local analysis: analyse_ktau_v1.R in this folder.
