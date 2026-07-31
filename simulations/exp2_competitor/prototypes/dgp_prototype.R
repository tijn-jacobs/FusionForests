# ──────────────────────────────────────────────────────────────────────────────
# Shared DGP for the comparator prototypes.
#
# Lifted from simulations/exp1_confounding_heterogeneity/sim_surv_v12.R so the
# prototypes see exactly the data the main simulation produces.  The only
# addition is the `censoring` switch:
#
#   "mixed"  RCT right-censored, RWD interval-censored -- the paper's design.
#   "right"  both sources right-censored               -- for comparators that
#                                                         cannot take intervals
#                                                         (deepAFT, bujar).
#
# Labels come back in ONE representation for every row, on the standardised
# LOG-time scale:
#
#   exact           lower = upper = log t
#   right-censored  lower = log c,  upper = Inf
#   interval        lower = log a,  upper = log b
#
# plus `y` (= lower) and `status` (1 = exact) for the right-censored-only case.
# A comparator reads whichever pair it needs; nothing has to be converted.
#
# Usage:
#   source("simulations/exp2_competitor/prototypes/dgp_prototype.R")
#   cal <- calibrate(ld = 1, lu = 1)
#   d   <- make_data(150, 350, ld = 1, lu = 1, cal, censoring = "right")
# ──────────────────────────────────────────────────────────────────────────────

library(MASS)
library(evd)

# ── DGP constants ───────────────────────────────────── (verbatim from v12) ────
p_total     <- 10L                # observed covariates
sigma_rct   <- 0.75               # RCT residual SD
target_cens <- 0.35               # target right-censoring fraction per source
visit_probs <- seq(0.1, 0.8, 0.1) # RWD inspection visits: deciles q10..q80

rho_X   <- 0.3
Sigma_X <- rho_X ^ abs(outer(seq_len(p_total), seq_len(p_total), "-"))
draw_X  <- function(n) MASS::mvrnorm(n, mu = rep(0, p_total), Sigma = Sigma_X)

m0   <- function(X)     2 * X[, 1] - X[, 2] * X[, 3] + 0.5 * X[, 4]^2
dev  <- function(X, ld) ld * (X[, 4] - 0.5 * X[, 5])   # RWD baseline deviation
tau  <- function(X)     1/2 + X[, 1] - 0.5 * X[, 2]^2  # true CATE
conf <- function(U, lu) lu * U                          # confounding via unobs. U

# ── Latent log-survival, one generator per source ──────────────────────────────
latent_rct <- function(n) {
  X <- draw_X(n)
  A <- rbinom(n, 1L, 0.5)                                # randomised
  list(X = X, A = A,
       logT = m0(X) + A * tau(X) + rnorm(n, 0, sigma_rct))
}

latent_rwd <- function(n, ld, lu) {
  gamma_em   <- 0.5772156649
  beta_scale <- sqrt(6) / pi
  mu_loc     <- -beta_scale * gamma_em
  X <- draw_X(n)
  U <- runif(n)                                          # RWD-only, unobserved
  e <- plogis(X[, 1] + U)                                # true propensity score
  A <- rbinom(n, 1L, e)                                  # selection on X1, U
  eps <- -rgumbel(n, loc = mu_loc, scale = beta_scale)   # mean-0 residual (SD 1)
  list(X = X, A = A,
       logT = m0(X) + dev(X, ld) + A * tau(X) + A * conf(U, lu) + eps)
}

# ── Calibration ───────────────────────────────────────────────────────────────
# P(cens) = E[1 - exp(-rate * T)], monotone increasing in rate.
solve_cens_rate <- function(logT, target = target_cens) {
  Tt <- exp(logT)
  uniroot(function(r) mean(1 - exp(-r * Tt)) - target, c(1e-10, 1e6))$root
}

# rate_rct / rate_rwd -- exponential right-censoring rates, one per source.
# visits              -- RWD inspection times (time scale) at deciles q10..q80.
# n_cal is smaller than the main simulation's 2e5; these are smoke tests.
calibrate <- function(ld, lu, n_cal = 5e4) {
  rwd <- latent_rwd(n_cal, ld, lu)$logT
  list(rate_rct = solve_cens_rate(latent_rct(n_cal)$logT),
       rate_rwd = solve_cens_rate(rwd),
       visits   = as.numeric(quantile(exp(rwd), probs = visit_probs)))
}

# ── One dataset ───────────────────────────────────────────────────────────────
# All bounds are standardised by the same affine map as the main simulation:
# the pooled LATENT log-survival is centred and scaled to mean 0 / variance 1.
# CATEs therefore come out on the standardised scale and must be multiplied by
# `scale_sd` before they are compared with tau(X).
make_data <- function(n_rct, n_rwd, ld, lu, cal,
                      censoring = c("mixed", "right")) {
  censoring <- match.arg(censoring)
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
    # events beyond the last visit are right-censored at q80
    lo_rwd[beyond] <- log(last_visit)
    up_rwd[beyond] <- Inf
    st_rwd <- rep(0L, n_rwd)                            # no exact RWD event times
    y_rwd  <- lo_rwd
  }

  list(
    X      = rbind(r$X, w$X),
    A      = c(r$A, w$A),
    S      = c(rep(1L, n_rct), rep(0L, n_rwd)),         # 1 = RCT, 0 = RWD
    lower  = std(c(lo_rct, lo_rwd)),
    upper  = std(c(up_rct, up_rwd)),
    y      = std(c(y_rct, y_rwd)),
    status = c(st_rct, st_rwd),
    truth  = tau(rbind(r$X, w$X)),                      # true CATE, log-time scale
    scale_sd = scale_sd,
    n_rct = n_rct, n_rwd = n_rwd,
    censoring = censoring
  )
}

# ── Reporting ─────────────────────────────────────────────────────────────────
# `cate` is on the standardised log scale; it is rescaled here before being
# compared with d$truth.  One row per evaluation population.
report <- function(d, cate, method) {
  cate <- cate * d$scale_sd
  pops <- list(RCT = which(d$S == 1L),
               RWD = which(d$S == 0L),
               All = seq_along(d$truth))
  do.call(rbind, lapply(names(pops), function(pop) {
    i <- pops[[pop]]
    data.frame(method     = method,
               population = pop,
               rmse       = sqrt(mean((cate[i] - d$truth[i])^2)),
               bias       = mean(cate[i] - d$truth[i]),
               stringsAsFactors = FALSE)
  }))
}

# Censoring summary, to confirm the design came out as intended.
describe <- function(d) {
  cat(sprintf("censoring = %-5s | n_rct = %d, n_rwd = %d\n",
              d$censoring, d$n_rct, d$n_rwd))
  cat(sprintf("  exact          %5.1f%%\n",
              100 * mean(d$lower == d$upper)))
  cat(sprintf("  right-censored %5.1f%%\n",
              100 * mean(is.infinite(d$upper))))
  cat(sprintf("  interval       %5.1f%%\n",
              100 * mean(d$lower < d$upper & is.finite(d$upper))))
}
