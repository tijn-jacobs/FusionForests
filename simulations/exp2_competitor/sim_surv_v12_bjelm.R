# ──────────────────────────────────────────────────────────────────────────────
# Competitor simulation: BJ-ELM vs the main FusionForest survival simulation.
#
# Same shape as sim_surv_v12_deepaft.R: the DGP, calibration, metrics, bootstrap
# and output columns are the deepAFT script's, so the rows row-bind straight
# onto sim_surv_v12_bff_output.rds and sim_surv_v12_deepaft_output.rds.
#
# Buckley-James boosting over extreme learning machines, from the supplementary
# material of
#   Kong & Zhang, "Buckley-James Boosting Model based on Extreme Learning
#   Machine and Random Survival Forests", Biometrical Journal.
# It is a standalone implementation, NOT part of the bujar package -- bujar
# 0.2-11 has no "elm" learner. Their fitting code is inlined below, so this
# script is self-contained and needs no companion files.
#
# Buckley-James handles RIGHT censoring only, so both sources are right-censored
# at ~35% here, exactly the deepAFT design. That makes these rows directly
# comparable with sim_surv_v12_bff.R and sim_surv_v12_deepaft.R, and it is also
# the reason this method cannot be run on the paper's actual mixed design --
# which is the point the comparator paragraph in Section 3 makes.
#
# Sized to run LOCALLY (small M, single grid cell). Bump M, B_boot and `grid` at
# the CONFIG block to scale up. BJ-ELM is slower than XGBoost: each fit runs a
# Buckley-James loop with up to 30 iterations, each fitting `nbase` ELMs.
#
# BJ-ELM has no posterior; the interval metrics come from a stratified
# nonparametric bootstrap, so `postvar` is the bootstrap variance rather than a
# posterior variance. Output columns match sim_surv_v12.R (method, population,
# lambda_d, lambda_u, Iter, rmse, bias, coverage, width, postvar).
# ──────────────────────────────────────────────────────────────────────────────

library(doParallel)
library(foreach)
library(evd)
library(MASS)

# ── BJ-ELM implementation ─────────────────────────────────────────────────────
# Verbatim from the supplementary material of
#   Kong, J. & Zhang, S., "Buckley-James Boosting Model based on Extreme
#   Learning Machine and Random Survival Forests", Biometrical Journal,
#   files R/Main_Functions/ELM_Boosting.R and R/Main_Functions/BJ_ELM.R.
#
# Inlined so this script is self-contained: the cluster job copies the .R file
# into $TMPDIR and runs from there, where the vendored folder is not available.
# Only the functions on the fitting path are reproduced. CV_BJ_ELM() is left
# out -- it selects nhid/nbase by CV on the C-index and needs compareC::estC(),
# which we do not use -- and with it go the rms, caret, rootSolve, scorecard and
# dplyr dependencies. What remains needs MASS (ginv) and emplik (WKM, and the
# cumsumsurv C routine reached through .C() without a PACKAGE argument, so emplik
# must be attached rather than merely installed). purrr is no longer needed: the
# one map_dfc() call is replaced below, which also removes the warning about
# purrr masking foreach's accumulate() and when().
library(emplik)

# --- ELM_Boosting.R ----------------------------------------------------------

actfun = function(x){
  # Activation function (sigmoid).
  y = 1 / (1 + exp(-1*x))
  return(y)
}

randomMatrix = function(nCols, nRows) {
  # Weights of the hidden layer in ELM.
  myMat = matrix(runif(nCols*nRows, min = -1, max = 1), ncol = nCols)
  myMat
}

elmtrain.default = function(x, y, nhid) {
  # Train one ELM. Returns the hidden weights, biases, output weights and fit.
  if(nhid < 1) stop("ERROR: number of hidden neurons must be >= 1")

  T = t(y)
  P = t(x)

  inpweight = randomMatrix(nrow(P), nhid)  ##w-U(-1,1)
  tempH = inpweight %*% P
  biashid = runif(nhid, min = -1, max = 1)
  biasMatrix = matrix(rep(biashid, ncol(P)), nrow = nhid, ncol = ncol(P), byrow = F)

  tempH = tempH + biasMatrix
  H = 1 / (1 + exp(-1*tempH))
  outweight = ginv(t(H), tol = sqrt(.Machine$double.eps)) %*% t(T)
  Y = t(t(H) %*% outweight)
  fitted.values = t(Y)
  residuals = y - fitted.values
  model = list(inpweight = inpweight, biashid = biashid, outweight = outweight,
               nhid = nhid, fitted.values = fitted.values)
  return(model)
}

ELM_boosting = function(x, y, nhid = 20, nbase = 25, step = 0.1){
  # Boosting over ELM base learners.
  bias = matrix(NA, ncol = nbase, nrow = nhid)
  hid_weights = vector('list', length = nbase)
  output_weights = matrix(NA, ncol = nbase, nrow = nhid)
  target = matrix(NA, ncol = nbase + 1, nrow = nrow(x))
  target[ , 1] = y
  fit_value = matrix(NA, ncol = nbase, nrow = nrow(x))

  for(k in 1:nbase){
    base_learner = elmtrain.default(x = x, y = target[, k], nhid = nhid)
    bias[, k] = c(base_learner$biashid)
    hid_weights[[k]] = t(base_learner$inpweight)
    output_weights[ , k] = c(base_learner$outweight)
    target[, k + 1] = c(target[, k]) - c(step*base_learner$fitted.values)
    fit_value[, k] = c(base_learner$fitted.values)
  }
  y_hat = step*(apply(fit_value, 1, sum))
  out = list('y_hat' = y_hat, 'hid_weights' = hid_weights, 'bias' = bias,
             'output_weights' = output_weights, 'residuals' = target, 'step' = step)
  return(out)
}

# --- BJ_ELM.R ----------------------------------------------------------------

cumsumsurv = function (x){
  # Jumps of the Kaplan-Meier estimator; C routine from emplik.
  if (any(is.na(x)))
    stop("NaNs")
  s = x
  .C("cumsumsurv", x = as.numeric(x), s = as.numeric(s), LLL = length(x))$s
}

iter = function(x, y, status, y_hat, nhid, nbase){
  # One Buckley-James iteration: impute the censored responses from the
  # Kaplan-Meier of the residuals, then refit the boosted ELM.
  N = length(status)
  u = y_hat
  res = y - u
  niceorder = order(res, - status)
  resorder = res[niceorder]
  dorder = status[niceorder]
  dorder[N] = 1
  uorder = u[niceorder]
  ystar = y[niceorder]
  xorder = as.matrix(x[niceorder, ])

  temp = WKM(x = resorder, d = dorder, zc = 1:N)

  jifen = cumsumsurv(resorder * temp$jump)
  Sresorder = temp$surv
  for (i in 1:N) if (dorder[i] == 0) {
    ystar[i] = uorder[i] + jifen[i]/Sresorder[i]
  }

  model = ELM_boosting(x = xorder, y = ystar, nhid = nhid, nbase = nbase, step = 0.1)
  return(model)
}

BJ_ELM = function (x, y, status, nhid = 20, nbase = 25) {
  # Fit the BJ-ELM model. Iterates until the fitted values stop moving or
  # maxiter is reached.
  maxiter = 30
  error = 1e-05
  x = as.matrix(x)
  model = vector('list', length = 3)
  model[[1]] = ELM_boosting(x = x, y = y, nhid = nhid, nbase = nbase, step = 0.1)
  f_x_hat = matrix(NA, ncol = nrow(x), nrow = 3)
  f_x_hat[1, ] = model[[1]]$y_hat
  for (i in 2:3) {
    model[[i]] = iter(x = x, y = y, status = status, y_hat = f_x_hat[i-1, ],
                      nhid = nhid, nbase = nbase)
    f_x_hat[i, ] = model[[i]]$y_hat
  }
  k = 2
  while (k <= maxiter && error <= sum(abs(f_x_hat[2, ] -
                                          f_x_hat[3, ]))) {
    f_x_hat[2, ] = f_x_hat[3, ]
    model[[2]] = model[[3]]
    model[[3]] = iter(x = x, y = y, status = status, y_hat = f_x_hat[2, ],
                      nhid = nhid, nbase = nbase)
    f_x_hat[3, ] = model[[3]]$y_hat
    k = k + 1
  }

  list(model = model[[3]])
}

pre_BJ_ELM = function(model, newx){
  # Predicted log survival times from a fitted BJ-ELM model.
  hid_weights = model$hid_weights
  bias = model$bias
  beta = model$output_weights
  outweight = model$outweight
  a = model$a
  step = model$step
  # CHANGED from the original, which used purrr::map_dfc() here. That builds a
  # tibble from unnamed columns, so it emitted `nbase` "New names: .. -> ...1"
  # messages on every call -- and this runs on every bootstrap prediction, which
  # flooded the cluster log. do.call(cbind, lapply(...)) produces exactly the
  # same n x nbase matrix, silently, and skips the tibble construction in what
  # is the hottest loop of the method. The arithmetic inside is untouched.
  y_pre_row = do.call(cbind, lapply(1:ncol(beta),
                      function(x) actfun(newx%*%as.matrix(hid_weights[[x]]) +
                                           matrix(rep(bias[, x], nrow(newx)), nrow = nrow(newx), byrow = TRUE))%*%matrix(beta[, x], ncol = 1)))
  y_pre_row = as.matrix(y_pre_row)
  y_pre = step*apply(y_pre_row, 1, sum)
  return(y_pre)
}

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

# Per-cell calibration, fixed once on a large pre-simulation: the RCT and the
# RWD each get an independent exponential right-censoring rate tuned to ~target.
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

# ── BJ-ELM competitors (S- and T-learner) ─────────────────────────────────────
# BJ_ELM models  log(T) = mu(Z) + error,  with mu the boosted-ELM output (larger
# => longer survival). Two meta-learners turn it into a CATE on the
# (standardised) log-time scale:
#   S-learner: one model with A as an input, tau(x) = mu(x, A=1) - mu(x, A=0).
#   T-learner: a separate model per arm,     tau(x) = mu_1(x) - mu_0(x).

# Each "design" packages the right-censored training data and the evaluation
# covariates Xev for one population.
#   RCT-only : covariates X, treatment A.
#   Pooled   : RCT + RWD, with the source indicator S (1 = RCT, 0 = RWD) as an
#              extra covariate; the CATE is evaluated at S = 1 (the trial
#              estimand / RCT population).
design_rct <- function(d, Xev) {
  list(y = d$y_rct, status = d$st_rct,
       X = d$X_rct, A = d$A_rct, Xev = Xev)
}
design_pool <- function(d, Xev) {
  S <- c(rep(1L, d$n_rct), rep(0L, d$n_rwd))
  list(y      = c(d$y_rct, d$y_rwd),
       status = c(d$st_rct, d$st_rwd),
       X      = cbind(rbind(d$X_rct, d$X_rwd), S = S),
       A      = c(d$A_rct, d$A_rwd),
       Xev    = cbind(Xev, S = 1L))
}

# `y` is ALREADY the standardised LOG survival time, which is what BJ_ELM wants.
# Do not log it again. `status` is 1 = event, 0 = right-censored, matching the
# implementation's convention (0 = alive, 1 = dead).
bjelm_fit <- function(y, status, Z, nhid, nbase) {
  BJ_ELM(x = as.matrix(Z), y = y, status = status,
         nhid = nhid, nbase = nbase)$model
}
bjelm_pred <- function(model, newZ) {
  as.numeric(pre_BJ_ELM(model = model, newx = as.matrix(newZ)))
}

# S-learner: ONE model, predicted at A = 1 and at A = 0. This matters here: the
# ELM hidden-layer weights are drawn at random, so fitting twice and
# differencing would put initialisation noise into the contrast.
cate_S <- function(des, nhid, nbase) {
  Z <- cbind(des$X, A = des$A)
  m <- bjelm_fit(des$y, des$status, Z, nhid, nbase)
  bjelm_pred(m, cbind(des$Xev, A = 1)) - bjelm_pred(m, cbind(des$Xev, A = 0))
}

# T-learner: a separate model per arm; CATE = mu_1(x) - mu_0(x).
cate_T <- function(des, nhid, nbase) {
  t1 <- des$A == 1L; t0 <- des$A == 0L
  m1 <- bjelm_fit(des$y[t1], des$status[t1], des$X[t1, , drop = FALSE], nhid, nbase)
  m0 <- bjelm_fit(des$y[t0], des$status[t0], des$X[t0, , drop = FALSE], nhid, nbase)
  bjelm_pred(m1, des$Xev) - bjelm_pred(m0, des$Xev)
}

# The four estimators (signature (d, Xev, nhid, nbase) so boot_cate can refit).
fit_rct_S  <- function(d, Xev, nh, nb) cate_S(design_rct(d, Xev),  nh, nb)
fit_rct_T  <- function(d, Xev, nh, nb) cate_T(design_rct(d, Xev),  nh, nb)
fit_pool_S <- function(d, Xev, nh, nb) cate_S(design_pool(d, Xev), nh, nb)
fit_pool_T <- function(d, Xev, nh, nb) cate_T(design_pool(d, Xev), nh, nb)

# ── Metrics ───────────────────────────────────────────────────────────────────
# Identical to sim_surv_v12_deepaft.R: the bootstrap percentile CI gives
# coverage / width, and the per-point bootstrap variance fills `postvar` (the
# frequentist analogue of the Bayesian posterior variance). With draws = NULL
# only rmse / bias are computed.
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
# Stratified nonparametric bootstrap: resample the RCT and RWD training rows
# separately (keeping n_rct / n_rwd fixed), refit, and predict the CATE on the
# fixed evaluation set Xev. B refits give a draws matrix (n_eval x B). The
# evaluation covariates d$X and the truth are NOT resampled.
resample_data <- function(d) {
  i_rct <- sample.int(d$n_rct, d$n_rct, replace = TRUE)
  i_rwd <- sample.int(d$n_rwd, d$n_rwd, replace = TRUE)
  d$X_rct  <- d$X_rct[i_rct, , drop = FALSE]; d$A_rct  <- d$A_rct[i_rct]
  d$y_rct  <- d$y_rct[i_rct];                 d$st_rct <- d$st_rct[i_rct]
  d$X_rwd  <- d$X_rwd[i_rwd, , drop = FALSE]; d$A_rwd  <- d$A_rwd[i_rwd]
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

# ── One replication: four BJ-ELM estimators on one dataset ────────────────────
run_one_sim <- function(n_rct, n_rwd, ld, lu, cal, nhid, nbase, B_boot) {
  d     <- make_data(n_rct, n_rwd, ld, lu, cal)
  truth <- tau(d$X)
  n_all <- n_rct + n_rwd
  safe  <- function(expr) tryCatch(expr, error = function(e) {
    message("    BJ-ELM fit failed: ", conditionMessage(e)); NULL })

  s   <- d$scale_sd                  # standardised-log CATE -> log-time scale
  scl <- function(z) if (is.null(z)) NULL else z * s

  estimators <- list(
    list(name = "bjelm-RCT-S",    fun = fit_rct_S),
    list(name = "bjelm-RCT-T",    fun = fit_rct_T),
    list(name = "bjelm-Pooled-S", fun = fit_pool_S),
    list(name = "bjelm-Pooled-T", fun = fit_pool_T)
  )
  res <- do.call(rbind, lapply(estimators, function(est) {
    point <- safe(est$fun(d, d$X, nhid, nbase))
    draws <- if (B_boot > 0L && !is.null(point))
      safe(boot_cate(est$fun, d, d$X, B_boot, nhid, nbase)) else NULL
    make_rows(est$name, scl(point), scl(draws), truth, n_rct, n_all)
  }))
  res$cens_rct <- mean(d$st_rct == 0)
  res$cens_rwd <- mean(d$st_rwd == 0)
  res
}

# ── Parallel runner over replications (one grid cell) ─────────────────────────
run_simulation_tidy <- function(n_rep, n_rct, n_rwd, ld, lu, nhid, nbase, B_boot) {
  cal <- calibrate(ld, lu)
  cat(sprintf("  RCT cens rate = %.4g | RWD cens rate = %.4g\n",
              cal$rate_rct, cal$rate_rwd))
  foreach(
    i = seq_len(n_rep),
    .combine = "rbind",
    .packages = c("evd", "MASS", "emplik")
  ) %dopar% {
    res <- run_one_sim(n_rct, n_rwd, ld, lu, cal, nhid, nbase, B_boot)
    res$lambda_d <- ld
    res$lambda_u <- lu
    res$Iter     <- i
    res
  }
}

# ── CONFIG ────────────────────────────────────────────────────────────────────
M       <- 1000L                 # replications per grid cell
B_boot  <- 100L                  # bootstrap refits per fit

n_rct  <- 150L                   # v12 settings
n_rwd  <- 350L                   # v12 settings

# Kong & Zhang's defaults; they suggest nhid in [5, 40] and nbase in [10, 40].
# CV_BJ_ELM() selects both by 5-fold CV on the C-index, but needs a data frame
# with `time` and `status` columns and compareC::estC(). Tune once on a
# representative dataset and reuse, as the deepAFT script does for its control.
NHID  <- 20L                     # hidden neurons per ELM base learner
NBASE <- 25L                     # number of base learners (boosting Mstop)

# One cell for a first local look. For the full sweep use:
#   grid <- expand.grid(lambda_d = c(0, 1, 2), lambda_u = c(0, 1, 2))
grid <- expand.grid(lambda_d = 1, lambda_u = 1)

# ── Main ──────────────────────────────────────────────────────────────────────
args <- commandArgs(trailingOnly = TRUE)
num_cores <- if (length(args) > 0) {
  as.integer(args[1]) - 1L
} else {
  max(1L, parallel::detectCores() - 1L)
}
registerDoParallel(cores = num_cores)

cat("SIMULATION (competitors): BJ-ELM RCT/Pooled x S/T-learner -- LOCAL\n")
cat("Cores in use (1 free):", num_cores, "\n")
cat("RCT and RWD both right-censored ~35% (Buckley-James takes right censoring only).\n")
cat("Covariates: MVN AR(1), rho =", rho_X, "\n")
cat(sprintf("M = %d reps/cell, n_rct = %d, n_rwd = %d, B_boot = %d, nhid = %d, nbase = %d\n",
            M, n_rct, n_rwd, B_boot, NHID, NBASE))
cat(sprintf("Grid: %d (lambda_d, lambda_u) cells\n", nrow(grid)))

run_started <- Sys.time()

all_cells <- vector("list", nrow(grid))
for (g in seq_len(nrow(grid))) {
  ld <- grid$lambda_d[g]; lu <- grid$lambda_u[g]
  cat(sprintf("\nCell %d/%d: lambda_d = %.1f, lambda_u = %.1f\n",
              g, nrow(grid), ld, lu))
  all_cells[[g]] <- run_simulation_tidy(
    n_rep = M, n_rct = n_rct, n_rwd = n_rwd, ld = ld, lu = lu,
    nhid = NHID, nbase = NBASE, B_boot = B_boot)
}
final_flat_df <- do.call(rbind, all_cells)

elapsed <- as.numeric(difftime(Sys.time(), run_started, units = "secs"))
cat(sprintf("\nReplications took %.1f min on %d workers.\n", elapsed / 60, num_cores))

# On the HPC write to TMPDIR; locally write next to the other outputs.
out_dir <- if (nzchar(Sys.getenv("TMPDIR"))) Sys.getenv("TMPDIR") else
  file.path("simulations", "exp2_competitor")
output_file <- file.path(out_dir, "sim_surv_v12_bjelm_output.rds")
cat("\nSaving all results to:", output_file, "\n")
saveRDS(final_flat_df, file = output_file)

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
names(var_tbl)[ncol(var_tbl)] <- "bias_var"   # column name of results_summary.csv
summary_tbl <- merge(mean_tbl, var_tbl)
summary_tbl$sd <- sqrt(summary_tbl$bias_var)

summary_tbl$population <- factor(summary_tbl$population,
                                 levels = c("RCT", "RWD", "All"))
ord <- order(summary_tbl$lambda_d, summary_tbl$lambda_u,
             summary_tbl$population, summary_tbl$method)
tbl <- summary_tbl[ord, c("lambda_d", "lambda_u", "population", "method",
                          "rmse", "bias", "bias_var", "sd",
                          "coverage", "width", "postvar")]

cat("\n=== BJ-ELM CATE metrics (mean over replications) ===\n")
print(tbl, row.names = FALSE, digits = 3)
cat(sprintf("\nMean censoring: RCT %.1f%%, RWD %.1f%%\n",
            100 * mean(res$cens_rct), 100 * mean(res$cens_rwd)))

# ── Write the summary to disk ─────────────────────────────────────────────────
# Two files beside the .rds. The CSV carries the same columns as
# results_summary.csv so the downstream table and figure scripts can read it
# unchanged. The TXT is the same table plus the settings that produced it, so a
# result is never separated from the configuration behind it.
csv_file <- sub("_output\\.rds$", "_summary.csv", output_file)
txt_file <- sub("_output\\.rds$", "_summary.txt", output_file)

write.csv(summary_tbl[ord, c("method", "population", "lambda_d", "lambda_u",
                             "rmse", "bias", "coverage", "width", "postvar",
                             "bias_var")],
          file = csv_file, row.names = FALSE)

writeLines(c(
  "BJ-ELM competitor simulation",
  sprintf("finished         : %s", format(Sys.time(), "%Y-%m-%d %H:%M:%S")),
  "censoring design : right (both sources right-censored)",
  sprintf("replications (M) : %d", M),
  sprintf("bootstrap (B)    : %d", B_boot),
  sprintf("n_rct / n_rwd    : %d / %d", n_rct, n_rwd),
  sprintf("nhid / nbase     : %d / %d", NHID, NBASE),
  sprintf("grid cells       : %d", nrow(grid)),
  sprintf("mean censoring   : RCT %.1f%%, RWD %.1f%%",
          100 * mean(res$cens_rct), 100 * mean(res$cens_rwd)),
  "",
  capture.output(print(tbl, row.names = FALSE, digits = 3))
), con = txt_file)

cat("Summary written to:\n  ", csv_file, "\n  ", txt_file, "\n", sep = "")
