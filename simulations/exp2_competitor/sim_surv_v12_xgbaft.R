# ──────────────────────────────────────────────────────────────────────────────
# Competitor simulation: XGBoost-AFT vs the main FusionForest survival
# simulation.
#
# Same shape as sim_surv_v12_deepaft.R: the DGP, calibration, metrics, bootstrap
# and output columns are the deepAFT script's, so the rows row-bind straight
# onto sim_surv_v12_bff_output.rds and sim_surv_v12_deepaft_output.rds.
#
#   install.packages("xgboost"); library(xgboost)
#   Barnwal, Cho & Hocking (JCGS), "Survival regression with accelerated
#   failure time model in XGBoost".
#
# CENSORING is the one real choice here, and it is not cosmetic:
#
#   "right"  Both sources right-censored at ~35%, exactly the deepAFT design.
#            Use this first: it puts XGBoost, deepAFT and the fusion forest on
#            the same data, so the three sets of rows are directly comparable.
#   "mixed"  RCT right-censored, RWD interval-censored -- the paper's actual
#            design. XGBoost is the only comparator that can be fitted on it,
#            which is the reason for including the method at all. Compare these
#            rows against exp1 (sim_surv_v12.R), which uses the same design.
#
# Sized to run LOCALLY (small M, single grid cell, sequential-friendly). Bump M,
# B_boot and `grid` at the CONFIG block to scale up.
#
# XGBoost has no posterior; the interval metrics come from a stratified
# nonparametric bootstrap, so `postvar` is the bootstrap variance rather than a
# posterior variance. Output columns match sim_surv_v12.R (method, population,
# lambda_d, lambda_u, Iter, rmse, bias, coverage, width, postvar).
# ──────────────────────────────────────────────────────────────────────────────

library(xgboost)
library(doParallel)
library(foreach)
library(evd)
library(MASS)

# ── DGP ───────────────────────────────────────────── (verbatim from v12) ──────
p_total     <- 10L               # observed covariates
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

# Per-cell calibration, fixed once on a large pre-simulation:
#   rate_rct / rate_rwd -- exponential right-censoring rates, one per source.
#   visits              -- RWD inspection times at the deciles q10..q80.
calibrate <- function(ld, lu, n_cal = 1e5) {
  rwd <- latent_rwd(n_cal, ld, lu)$logT
  list(rate_rct = solve_cens_rate(latent_rct(n_cal)$logT),
       rate_rwd = solve_cens_rate(rwd),
       visits   = as.numeric(quantile(exp(rwd), probs = visit_probs)))
}

# Simulate one full dataset.  Labels are returned as (lower, upper) bounds on
# the standardised LOG-time scale, with upper = Inf for right-censored rows:
#   exact           lower = upper = log t
#   right-censored  lower = log c,  upper = Inf
#   interval        lower = log a,  upper = log b
# The pooled LATENT log-survival is centred and scaled to mean 0 / variance 1;
# predicted CATEs come back on that scale and are rescaled by `scale_sd`.
make_data <- function(n_rct, n_rwd, ld, lu, cal, censoring) {
  r <- latent_rct(n_rct)
  w <- latent_rwd(n_rwd, ld, lu)

  logT_true <- c(r$logT, w$logT)
  scale_mu  <- mean(logT_true)
  scale_sd  <- sd(logT_true)
  std       <- function(z) (z - scale_mu) / scale_sd   # std(Inf) stays Inf
  logfloor  <- min(logT_true) - 5                      # finite stand-in for log(0)

  # --- RCT: exponential right censoring (both designs) ---
  logC   <- log(rexp(n_rct, rate = cal$rate_rct))
  y_rct  <- pmin(r$logT, logC)
  st_rct <- as.integer(r$logT <= logC)
  lo_rct <- y_rct
  up_rct <- ifelse(st_rct == 1L, y_rct, Inf)

  if (censoring == "right") {
    # --- RWD: exponential right censoring, same mechanism as the RCT ---
    logC_w <- log(rexp(n_rwd, rate = cal$rate_rwd))
    y_rwd  <- pmin(w$logT, logC_w)
    st_rwd <- as.integer(w$logT <= logC_w)
    lo_rwd <- y_rwd
    up_rwd <- ifelse(st_rwd == 1L, y_rwd, Inf)
  } else {
    # --- RWD: interval censoring on the decile visits ---
    T0         <- exp(w$logT)
    last_visit <- cal$visits[length(cal$visits)]        # q80
    breaks     <- c(0, cal$visits)                      # bin edges 0, q10, ..., q80
    j          <- pmax(findInterval(T0, breaks), 1L)
    beyond     <- T0 >= last_visit                      # event after the last visit

    lo_rwd <- ifelse(j == 1L, logfloor, log(breaks[pmin(j, length(breaks))]))
    up_rwd <- log(breaks[pmin(j + 1L, length(breaks))])
    lo_rwd[beyond] <- log(last_visit)                   # right-censored at q80
    up_rwd[beyond] <- Inf
    st_rwd <- rep(0L, n_rwd)                            # no exact RWD event times
  }

  list(
    X_rct = r$X, X_rwd = w$X,
    A_rct = r$A, A_rwd = w$A,
    lo_rct = std(lo_rct), up_rct = std(up_rct), st_rct = st_rct,
    lo_rwd = std(lo_rwd), up_rwd = std(up_rwd), st_rwd = st_rwd,
    X      = rbind(r$X, w$X),
    scale_sd = scale_sd,
    n_rct = n_rct, n_rwd = n_rwd
  )
}

# ── XGBoost-AFT competitors (S- and T-learner) ────────────────────────────────
# XGBoost-AFT models  log(T) = mu(Z) + sigma * error,  with mu the boosted-tree
# output (larger => longer survival).  Two meta-learners turn it into a CATE on
# the (standardised) log-time scale:
#   S-learner: one model with A as an input, tau(x) = mu(x, A=1) - mu(x, A=0).
#   T-learner: a separate model per arm,     tau(x) = mu_1(x) - mu_0(x).

# Each "design" packages the training labels, covariates and treatment for one
# population, plus the evaluation covariates Xev.
#   RCT-only : covariates X, treatment A.
#   Pooled   : RCT + RWD, with the source indicator S (1 = RCT, 0 = RWD) as an
#              extra covariate; the CATE is evaluated at S = 1 (the trial
#              estimand / RCT population).
design_rct <- function(d, Xev) {
  list(lo = d$lo_rct, up = d$up_rct,
       X = d$X_rct, A = d$A_rct, Xev = Xev)
}
design_pool <- function(d, Xev) {
  S <- c(rep(1L, d$n_rct), rep(0L, d$n_rwd))
  list(lo  = c(d$lo_rct, d$lo_rwd),
       up  = c(d$up_rct, d$up_rwd),
       X   = cbind(rbind(d$X_rct, d$X_rwd), S = S),
       A   = c(d$A_rct, d$A_rwd),
       Xev = cbind(Xev, S = 1L))
}

# Labels go in on the TIME scale; the AFT loss takes the log internally.  Our
# bounds are standardised log-times, so exp() puts them where xgboost wants them
# and the fit works on the standardised log scale throughout.  exp(Inf) = Inf is
# the right-censored upper bound and is accepted as-is.
xgb_fit <- function(lo, up, Z, params, nrounds) {
  dtrain <- xgb.DMatrix(as.matrix(Z))
  setinfo(dtrain, "label_lower_bound", exp(lo))
  setinfo(dtrain, "label_upper_bound", exp(up))
  xgb.train(params, dtrain, nrounds = nrounds, verbose = 0)
}

# For survival:aft, predict() returns exp(margin), i.e. a predicted survival
# TIME.  The CATE is a difference on the log scale, so we need the raw margin --
# hence outputmargin = TRUE.  Dropping it silently returns a difference of times
# instead of a log-ratio.
xgb_pred <- function(bst, newZ) {
  predict(bst, xgb.DMatrix(as.matrix(newZ)), outputmargin = TRUE)
}

# ── Hyperparameter tuning ─────────────────────────────────────────────────────
# Random search with k-fold CV on the AFT negative log-likelihood, following
# Barnwal, Cho & Hocking. They treat six hyperparameters as relevant --
# learning_rate, max_depth, min_child_weight, reg_alpha, reg_lambda and
# aft_loss_distribution_scale -- and report that roughly 100 combinations
# suffice: random search with 1000 trials gave the best validation accuracy but
# did not improve test accuracy over searches with fewer than 100 trials.
#
# aft_loss_distribution is NOT searched. Barnwal et al. fit normal, logistic and
# extreme as three separate models and report them separately; we fit the normal
# only, which is the same treatment restricted to one law.
#
# The CV also sets the number of boosting rounds, through early stopping on the
# mean validation metric. Returns the winning params and that round count.
#
# As with deepAFT, tuning every replication would dominate the runtime, so this
# runs once on a representative dataset and the result is reused throughout.
tune_xgb <- function(lo, up, Z, n_trials, nfold, max_rounds, early_stop) {
  # The trials are independent, so they run over the same workers as the
  # replications. Run serially this is n_trials x nfold single-threaded fits on
  # one core while the rest of the node idles -- with 100 trials, 5 folds and up
  # to 500 rounds it dominates the job.
  #
  # The parameter sets are drawn HERE, in the main process, not inside the
  # parallel loop: forked workers inherit the RNG state, so drawing in the loop
  # would hand every trial the same combination.
  draws <- lapply(seq_len(n_trials), function(t)
    c(BASE_PARAMS, lapply(TUNE_GRID, function(v) v[sample.int(length(v), 1L)])))

  Zm <- as.matrix(Z); lo_e <- exp(lo); up_e <- exp(up)

  trials <- foreach(p = draws, .packages = "xgboost") %dopar% {
    # Built inside the worker: a DMatrix is an external pointer and does not
    # survive being shared across processes.
    dtrain <- xgb.DMatrix(Zm)
    setinfo(dtrain, "label_lower_bound", lo_e)
    setinfo(dtrain, "label_upper_bound", up_e)
    cv <- tryCatch(
      xgb.cv(params = p, data = dtrain, nrounds = max_rounds, nfold = nfold,
             early_stopping_rounds = early_stop, verbose = 0, showsd = FALSE),
      error = function(e) NULL)
    if (is.null(cv)) return(NULL)
    # The metric column is named after eval_metric and contains a dash, so it is
    # looked up by pattern rather than by a literal name.
    elog <- as.data.frame(cv$evaluation_log)
    mcol <- grep("^test_.*_mean$", names(elog), value = TRUE)[1]
    it   <- if (!is.null(cv$best_iteration)) cv$best_iteration
            else which.min(elog[[mcol]])
    sc   <- elog[[mcol]][it]
    if (!is.finite(sc)) return(NULL)
    list(score = sc, params = p, nrounds = it)
  }

  trials <- Filter(Negate(is.null), trials)
  if (!length(trials)) stop("every tuning trial failed")
  trials[[which.min(vapply(trials, `[[`, numeric(1), "score"))]]
}

# Compact one-line report of a tuning result.
describe_tuning <- function(lab, tuned) {
  p <- tuned$params
  cat(sprintf(
    "  %-8s nloglik = %.4f | nrounds = %d | lr = %g, depth = %d, mcw = %g,\n",
    lab, tuned$score, tuned$nrounds, p$learning_rate, p$max_depth,
    p$min_child_weight))
  cat(sprintf("           alpha = %g, lambda = %g, scale = %g\n",
              p$reg_alpha, p$reg_lambda, p$aft_loss_distribution_scale))
}

# S-learner: ONE model, predicted at A = 1 and at A = 0.  Fitting twice would
# double the cost and, for any learner with a random component, would leak
# initialisation noise into the contrast.
cate_S <- function(des, params, nrounds) {
  Z   <- cbind(des$X, A = des$A)
  bst <- xgb_fit(des$lo, des$up, Z, params, nrounds)
  xgb_pred(bst, cbind(des$Xev, A = 1)) - xgb_pred(bst, cbind(des$Xev, A = 0))
}

# T-learner: a separate model per arm; CATE = mu_1(x) - mu_0(x).
cate_T <- function(des, params, nrounds) {
  t1 <- des$A == 1L; t0 <- des$A == 0L
  b1 <- xgb_fit(des$lo[t1], des$up[t1], des$X[t1, , drop = FALSE], params, nrounds)
  b0 <- xgb_fit(des$lo[t0], des$up[t0], des$X[t0, , drop = FALSE], params, nrounds)
  xgb_pred(b1, des$Xev) - xgb_pred(b0, des$Xev)
}

# The four estimators (signature (d, Xev, params, nrounds) so boot_cate can
# refit them).
fit_rct_S  <- function(d, Xev, p, n) cate_S(design_rct(d, Xev),  p, n)
fit_rct_T  <- function(d, Xev, p, n) cate_T(design_rct(d, Xev),  p, n)
fit_pool_S <- function(d, Xev, p, n) cate_S(design_pool(d, Xev), p, n)
fit_pool_T <- function(d, Xev, p, n) cate_T(design_pool(d, Xev), p, n)

# ── Metrics ───────────────────────────────────────────────────────────────────
# Identical to sim_surv_v12_deepaft.R: the bootstrap percentile CI gives
# coverage / width, and the per-point bootstrap variance fills `postvar` (the
# frequentist analogue of the Bayesian posterior variance).  With draws = NULL
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
# fixed evaluation set Xev.  B refits give a draws matrix (n_eval x B).  The
# evaluation covariates d$X and the truth are NOT resampled.
resample_data <- function(d) {
  i_rct <- sample.int(d$n_rct, d$n_rct, replace = TRUE)
  i_rwd <- sample.int(d$n_rwd, d$n_rwd, replace = TRUE)
  d$X_rct  <- d$X_rct[i_rct, , drop = FALSE]; d$A_rct  <- d$A_rct[i_rct]
  d$lo_rct <- d$lo_rct[i_rct];                d$up_rct <- d$up_rct[i_rct]
  d$st_rct <- d$st_rct[i_rct]
  d$X_rwd  <- d$X_rwd[i_rwd, , drop = FALSE]; d$A_rwd  <- d$A_rwd[i_rwd]
  d$lo_rwd <- d$lo_rwd[i_rwd];                d$up_rwd <- d$up_rwd[i_rwd]
  d$st_rwd <- d$st_rwd[i_rwd]
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

# ── One replication: four XGBoost-AFT estimators on one dataset ───────────────
run_one_sim <- function(n_rct, n_rwd, ld, lu, cal, censoring,
                        tuned_rct, tuned_pool, B_boot) {
  d     <- make_data(n_rct, n_rwd, ld, lu, cal, censoring)
  truth <- tau(d$X)
  n_all <- n_rct + n_rwd
  safe  <- function(expr) tryCatch(expr, error = function(e) {
    message("    XGBoost-AFT fit failed: ", conditionMessage(e)); NULL })

  s   <- d$scale_sd                  # standardised-log CATE -> log-time scale
  scl <- function(z) if (is.null(z)) NULL else z * s

  # Each design reuses the control tuned for it, as in the deepAFT script.
  estimators <- list(
    list(name = "xgbAFT-RCT-S",    fun = fit_rct_S,  tp = tuned_rct),
    list(name = "xgbAFT-RCT-T",    fun = fit_rct_T,  tp = tuned_rct),
    list(name = "xgbAFT-Pooled-S", fun = fit_pool_S, tp = tuned_pool),
    list(name = "xgbAFT-Pooled-T", fun = fit_pool_T, tp = tuned_pool)
  )
  res <- do.call(rbind, lapply(estimators, function(est) {
    point <- safe(est$fun(d, d$X, est$tp$params, est$tp$nrounds))
    draws <- if (B_boot > 0L && !is.null(point))
      safe(boot_cate(est$fun, d, d$X, B_boot, est$tp$params, est$tp$nrounds))
      else NULL
    make_rows(est$name, scl(point), scl(draws), truth, n_rct, n_all)
  }))
  res$cens_rct <- mean(d$st_rct == 0)
  res$cens_rwd <- mean(d$st_rwd == 0)
  res
}

# ── Parallel runner over replications (one grid cell) ─────────────────────────
run_simulation_tidy <- function(n_rep, n_rct, n_rwd, ld, lu, censoring,
                                tuned_rct, tuned_pool, B_boot) {
  cal <- calibrate(ld, lu)        # per-cell censoring rates and visit times
  cat(sprintf("  RCT cens rate = %.4g | RWD cens rate = %.4g | q80 = %.3g\n",
              cal$rate_rct, cal$rate_rwd, cal$visits[length(cal$visits)]))
  foreach(
    i = seq_len(n_rep),
    .combine = "rbind",
    .packages = c("xgboost", "evd", "MASS")
  ) %dopar% {
    res <- run_one_sim(n_rct, n_rwd, ld, lu, cal, censoring,
                       tuned_rct, tuned_pool, B_boot)
    res$lambda_d <- ld
    res$lambda_u <- lu
    res$Iter     <- i
    res
  }
}

# ── CONFIG ────────────────────────────────────────────────────────────────────
CENSORING <- "right"             # "right" (deepAFT design) or "mixed" (paper's)

M       <- 1000L                 # replications per grid cell
B_boot  <- 100L                  # bootstrap refits per fit

n_rct   <- 150L                  # v12 settings
n_rwd   <- 350L                  # v12 settings

# Only lambda_d = lambda_u = 1 is reported; that is the cell the main text uses.
grid <- expand.grid(lambda_d = 1, lambda_u = 1)

# Fixed across every fit. nthread = 1 so that xgboost does not fight doParallel
# for cores. aft_loss_distribution is fixed to the normal: Barnwal et al. fit
# normal, logistic and extreme as three separate models rather than searching
# over them, and we report the normal only.
BASE_PARAMS <- list(objective             = "survival:aft",
                    eval_metric           = "aft-nloglik",
                    aft_loss_distribution = "normal",
                    tree_method           = "hist",
                    nthread               = 1L)

# The six hyperparameters Barnwal et al. treat as relevant, over their grid.
TUNE_GRID <- list(
  learning_rate               = c(0.001, 0.01, 0.1, 1.0),
  max_depth                   = 2:10,
  min_child_weight            = c(0.001, 0.1, 1, 10, 100),
  reg_alpha                   = c(0.001, 0.01, 0.1, 1, 10, 100),
  reg_lambda                  = c(0.001, 0.01, 0.1, 1, 10, 100),
  aft_loss_distribution_scale = c(0.5, 0.8, 1.1, 1.4, 1.7, 2.0)
)

TUNE       <- TRUE               # FALSE uses DEFAULT_PARAMS below instead
N_TRIALS   <- 100L               # Barnwal et al.: ~100 combinations suffice
NFOLD      <- 5L                 # inner CV folds
MAX_ROUNDS <- 500L               # ceiling; early stopping picks the actual count
EARLY_STOP <- 20L

# Fallback when TUNE = FALSE: the defaults from Barnwal et al.'s tutorial.
DEFAULT_PARAMS  <- c(BASE_PARAMS,
                     list(learning_rate               = 0.05,
                          max_depth                   = 2L,
                          min_child_weight            = 1,
                          reg_alpha                   = 0,
                          reg_lambda                  = 1,
                          aft_loss_distribution_scale = 1.20))
DEFAULT_NROUNDS <- 200L

# ── Main ──────────────────────────────────────────────────────────────────────
args <- commandArgs(trailingOnly = TRUE)
num_cores <- if (length(args) > 0) {
  as.integer(args[1]) - 1L
} else {
  max(1L, parallel::detectCores() - 1L)
}
registerDoParallel(cores = num_cores)

cat("SIMULATION (competitors): XGBoost-AFT RCT/Pooled x S/T-learner -- LOCAL\n")
cat("Cores in use (1 free):", num_cores, "\n")
cat("Censoring design:", CENSORING,
    if (CENSORING == "right") "(both sources right-censored, deepAFT design)"
    else "(RCT right, RWD interval -- the paper's design)", "\n")
cat("Covariates: MVN AR(1), rho =", rho_X, "\n")
cat(sprintf("M = %d reps/cell, n_rct = %d, n_rwd = %d, B_boot = %d\n",
            M, n_rct, n_rwd, B_boot))
cat(sprintf("Grid: %d (lambda_d, lambda_u) cells\n", nrow(grid)))

# ── Tune once, reuse across replications ──────────────────────────────────────
# Tuning every replication (N_TRIALS x NFOLD fits each) would dominate the
# runtime, so we tune once on a representative dataset from the (1,1) cell and
# reuse the result everywhere, exactly as the deepAFT script does. Seeded so the
# reported hyperparameters are reproducible; the replications themselves are not
# seeded.
if (TUNE) {
  cat(sprintf("\nTuning XGBoost-AFT (random search, %d trials, %d-fold CV) ...\n",
              N_TRIALS, NFOLD))
  set.seed(1L)
  cal0 <- calibrate(1.0, 1.0)
  d0   <- make_data(n_rct, n_rwd, 1.0, 1.0, cal0, CENSORING)
  # Tuned on the S-learner design (the fuller matrix) and reused for the
  # T-learner of the same design, as deepAFT does with its control.
  tuned_rct <- tune_xgb(d0$lo_rct, d0$up_rct,
                        cbind(d0$X_rct, A = d0$A_rct),
                        N_TRIALS, NFOLD, MAX_ROUNDS, EARLY_STOP)
  Sp <- c(rep(1L, d0$n_rct), rep(0L, d0$n_rwd))
  tuned_pool <- tune_xgb(c(d0$lo_rct, d0$lo_rwd), c(d0$up_rct, d0$up_rwd),
                         cbind(rbind(d0$X_rct, d0$X_rwd), S = Sp,
                               A = c(d0$A_rct, d0$A_rwd)),
                         N_TRIALS, NFOLD, MAX_ROUNDS, EARLY_STOP)
  describe_tuning("RCT", tuned_rct)
  describe_tuning("Pooled", tuned_pool)
} else {
  cat("\nTuning skipped; using Barnwal et al.'s tutorial defaults.\n")
  tuned_rct <- tuned_pool <- list(params = DEFAULT_PARAMS,
                                  nrounds = DEFAULT_NROUNDS, score = NA_real_)
}

run_started <- Sys.time()

all_cells <- vector("list", nrow(grid))
for (g in seq_len(nrow(grid))) {
  ld <- grid$lambda_d[g]; lu <- grid$lambda_u[g]
  cat(sprintf("\nCell %d/%d: lambda_d = %.1f, lambda_u = %.1f\n",
              g, nrow(grid), ld, lu))
  all_cells[[g]] <- run_simulation_tidy(
    n_rep = M, n_rct = n_rct, n_rwd = n_rwd, ld = ld, lu = lu,
    censoring = CENSORING, tuned_rct = tuned_rct, tuned_pool = tuned_pool,
    B_boot = B_boot)
}
final_flat_df <- do.call(rbind, all_cells)

elapsed <- as.numeric(difftime(Sys.time(), run_started, units = "secs"))
cat(sprintf("\nReplications took %.1f min on %d workers.\n", elapsed / 60, num_cores))

# On the HPC write to TMPDIR; locally write next to the other outputs.
out_dir <- if (nzchar(Sys.getenv("TMPDIR"))) Sys.getenv("TMPDIR") else
  file.path("simulations", "exp2_competitor")
# Named after the script, so a Slurm wrapper looking for ${base_name}_output.rds
# finds it. The censoring design is recorded in the summary and as an attribute
# rather than in the filename; rename by hand if you keep both designs.
attr(final_flat_df, "censoring") <- CENSORING
output_file <- file.path(out_dir, "sim_surv_v12_xgbaft_output.rds")
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

cat("\n=== XGBoost-AFT CATE metrics (mean over replications) ===\n")
print(tbl, row.names = FALSE, digits = 3)
cat(sprintf("\nMean censoring: RCT %.1f%%, RWD %.1f%%\n",
            100 * mean(res$cens_rct), 100 * mean(res$cens_rwd)))

# ── Write the summary to disk ─────────────────────────────────────────────────
# The CSV carries the columns of results_summary.csv so the downstream table and
# figure scripts read it unchanged. The TXT is the same table plus the settings
# that produced it, including the selected hyperparameters -- the whole point of
# tuning is lost if the winning values are not recorded next to the result.
csv_file <- sub("_output\\.rds$", "_summary.csv", output_file)
txt_file <- sub("_output\\.rds$", "_summary.txt", output_file)

write.csv(summary_tbl[ord, c("method", "population", "lambda_d", "lambda_u",
                             "rmse", "bias", "coverage", "width", "postvar",
                             "bias_var")],
          file = csv_file, row.names = FALSE)

fmt_tuned <- function(lab, t) {
  p <- t$params
  sprintf(paste0("%-16s nloglik = %s, nrounds = %d, lr = %g, max_depth = %d,\n",
                 "                 min_child_weight = %g, reg_alpha = %g, ",
                 "reg_lambda = %g,\n                 scale = %g"),
          paste0(lab, ":"),
          if (is.na(t$score)) "--" else sprintf("%.4f", t$score),
          t$nrounds, p$learning_rate, p$max_depth, p$min_child_weight,
          p$reg_alpha, p$reg_lambda, p$aft_loss_distribution_scale)
}

writeLines(c(
  "XGBoost-AFT competitor simulation",
  sprintf("finished         : %s", format(Sys.time(), "%Y-%m-%d %H:%M:%S")),
  sprintf("censoring design : %s", CENSORING),
  sprintf("replications (M) : %d", M),
  sprintf("bootstrap (B)    : %d", B_boot),
  sprintf("n_rct / n_rwd    : %d / %d", n_rct, n_rwd),
  sprintf("error law        : %s (fixed, not searched)",
          BASE_PARAMS$aft_loss_distribution),
  sprintf("tuning           : %s",
          if (TUNE) sprintf("random search, %d trials, %d-fold CV, tuned once",
                            N_TRIALS, NFOLD)
          else "off (Barnwal et al. tutorial defaults)"),
  fmt_tuned("RCT design", tuned_rct),
  fmt_tuned("Pooled design", tuned_pool),
  sprintf("grid cells       : %d", nrow(grid)),
  sprintf("mean censoring   : RCT %.1f%%, RWD %.1f%%",
          100 * mean(res$cens_rct), 100 * mean(res$cens_rwd)),
  "",
  capture.output(print(tbl, row.names = FALSE, digits = 3))
), con = txt_file)

cat("Summary written to:\n  ", csv_file, "\n  ", txt_file, "\n", sep = "")
