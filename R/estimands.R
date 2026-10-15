#' Causal survival estimands from a FusionForest fit
#'
#' Computes posterior draws of the survival difference or the acceleration
#' factor at the test rows of a \code{\link{FusionForest}} fit.
#'
#' @details
#' The survival function under treatment \eqn{a} in target source \eqn{s_t}
#' is
#' \deqn{S_a(t \mid x, s_t) = 1 - F_{s_t}\{\log t - \eta_a(x, s_t)\},}{
#'   S_a(t | x, s_t) = 1 - F_{s_t}(log t - eta_a(x, s_t)),}
#' with \eqn{F_{s_t}} the error distribution in source \eqn{s_t} and
#' \deqn{\eta_a(x, s_t) = m_0^{\mathrm{sh}}(x) + (1 - s_t)\,d(x) +
#'   a\,\tau(x) + (1 - s_t)\,a\,c(x).}{eta_a(x, s_t) = m_0^sh(x) +
#'   (1 - s_t) d(x) + a tau(x) + (1 - s_t) a c(x).}
#' For fits with the \code{"centered"} or \code{"adaptive"} treatment
#' coding, the function uses the coded treatment instead of \eqn{a}.
#'
#' Two estimands are available:
#' \describe{
#'   \item{\code{"SD"}}{Survival difference
#'     \eqn{S_1(t \mid x, s_t) - S_0(t \mid x, s_t)}{S_1(t | x, s_t) -
#'     S_0(t | x, s_t)}.}
#'   \item{\code{"AF"}}{Acceleration factor \eqn{\exp\{\tau(x)\}}{exp(tau(x))}.
#'     It does not depend on the error distribution or the target source.}
#' }
#'
#' The population average weights the test rows.  By default, the weights
#' are drawn from a Dirichlet(1, ..., 1) distribution at each iteration: the
#' Bayesian bootstrap (Rubin, 1981).
#'
#' @param fit A \code{FusionForest} fit with
#'   \code{store_posterior_sample = TRUE}.
#' @param estimand \code{"SD"} or \code{"AF"}.
#' @param time Time point for \code{"SD"}, on the time scale.  Ignored for
#'   \code{"AF"}.
#' @param target_source Target population: \code{"rwd"} (default) or
#'   \code{"rct"}.  Ignored for \code{"AF"}.
#' @param population_average If \code{TRUE}, average over the test rows.
#'   Default \code{FALSE}.
#' @param bayesian_bootstrap If \code{TRUE} (default), the population
#'   average uses Bayesian bootstrap weights.  If \code{FALSE}, it uses
#'   equal weights.
#' @param seed Optional seed for the Bayesian bootstrap weights.
#'
#' @return A matrix of posterior draws with one row per iteration and one
#'   column per test row.  With \code{population_average = TRUE}, a vector
#'   with one draw per iteration.  The result carries the attributes
#'   \code{estimand}, \code{time} and \code{target_source}.
#'
#' @references
#' Rubin, D.~B. (1981).  The Bayesian bootstrap.
#' \emph{The Annals of Statistics}, 9, 130--134.
#'
#' @examples
#' # Right-censored fusion fit that stores the posterior draws
#' set.seed(1)
#' n <- 100
#' X <- matrix(rnorm(n * 3), n, 3)
#' s <- rbinom(n, 1, 0.5)
#' a <- rbinom(n, 1, 0.5)
#' true_time <- exp(1 + X[, 1] + 0.5 * a + 0.3 * rnorm(n))
#' cens_time <- rexp(n, rate = 1 / (2 * mean(true_time)))
#' time   <- pmin(true_time, cens_time)
#' status <- as.integer(true_time <= cens_time)
#'
#' fit <- FusionForest(
#'   y = time, status = status,
#'   X_train_control = X, X_train_treat = X,
#'   treatment_indicator_train = a, source_indicator_train = s,
#'   X_test_control = X, X_test_treat = X,
#'   treatment_indicator_test = a, source_indicator_test = s,
#'   outcome_type = "right-censored",
#'   N_post = 50, N_burn = 25,
#'   store_posterior_sample = TRUE, verbose = FALSE
#' )
#'
#' # Posterior draws of the acceleration factor per test row
#' af <- fusion_estimand(fit, estimand = "AF")
#' quantile(colMeans(af), c(0.25, 0.5, 0.75))
#'
#' @importFrom stats pnorm rgamma
#' @export
fusion_estimand <- function(
  fit,
  estimand           = c("SD", "AF"),
  time               = NULL,
  target_source      = c("rwd", "rct"),
  population_average = FALSE,
  bayesian_bootstrap = TRUE,
  seed               = NULL
) {

  estimand      <- match.arg(estimand)
  target_source <- match.arg(target_source)

  meta <- fit$meta
  if (is.null(meta))
    stop("fit$meta is missing.  Refit with FusionForest() using the current ",
         "package version, which attaches scale metadata to the result.")

  if (estimand == "SD") {
    if (is.null(time))
      stop("'time' is required for estimand = '", estimand, "'.")
    if (!is.numeric(time) || length(time) != 1L || !is.finite(time) || time <= 0)
      stop("'time' must be a single positive numeric value.")
  }

  ## -----------------------------------------------------------------
  ## Recover per-iteration test predictions on the user log-time scale
  ## -----------------------------------------------------------------

  comp <- .log_scale_components(fit, meta, which = "test")  # list(mu, tau, c, g) of R*n
  R    <- nrow(comp$mu)
  n_ev <- ncol(comp$mu)

  ## -----------------------------------------------------------------
  ## Branch by estimand
  ## -----------------------------------------------------------------

  if (estimand == "AF") {
    # AF_causal(x) = exp(tau(x)); independent of source and error dist.
    draws <- exp(comp$tau)
    attrs <- list(estimand = "AF", time = NA_real_,
                  target_source = target_source)
  } else {
    s_t  <- if (target_source == "rct") 1L else 0L
    eta0 <- .linear_predictor(comp, meta, a = 0, s_t = s_t)
    eta1 <- .linear_predictor(comp, meta, a = 1, s_t = s_t)

    mix   <- .residual_mixture(fit, meta, s_t)    # list(pi, theta, sigma)
    draws <- .sd_draws(log(time), eta0, eta1, mix)
    attrs <- list(estimand = "SD", time = time,
                  target_source = target_source)
  }

  ## -----------------------------------------------------------------
  ## Population average
  ## -----------------------------------------------------------------

  if (!population_average) {
    out <- draws
  } else {
    if (!is.null(seed)) set.seed(seed)
    if (bayesian_bootstrap) {
      g <- matrix(rgamma(R * n_ev, shape = 1, rate = 1), R, n_ev)
      w <- g / rowSums(g)
      out <- rowSums(draws * w)
    } else {
      out <- rowMeans(draws)
    }
    attrs$bayesian_bootstrap <- bayesian_bootstrap
  }

  for (nm in names(attrs)) attr(out, nm) <- attrs[[nm]]
  out
}


# -----------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------

# Return mu(X), tau(X), c(X), g(X) per iteration on the user log-time scale
# (i.e. the scale of log T after centring is undone).  When timescale =
# "time" the back-transform applied exp(.), so we take log() to recover
# the log-time scale predictions used in the eta_a formula.  For "log"
# and continuous outcomes the predictions are already on the right scale.
# `which` selects test or train rows.
.log_scale_components <- function(fit, meta, which = c("test", "train")) {

  which <- match.arg(which)
  prefix <- paste0(which, "_predictions_sample_")

  need <- paste0(prefix, c("control", "treat", "deconf"))
  for (nm in need) {
    if (is.null(fit[[nm]]))
      stop("fit$", nm, " is missing.  Refit with store_posterior_sample = TRUE.")
  }

  to_log <- (meta$outcome_type == "right-censored" && meta$timescale == "time")
  unlog  <- function(x) if (to_log) log(x) else x

  list(
    mu  = unlog(fit[[paste0(prefix, "control")]]),
    tau = unlog(fit[[paste0(prefix, "treat")]]),
    c   = unlog(fit[[paste0(prefix, "deconf")]]),
    g   = if (!is.null(fit[[paste0(prefix, "deviation")]]))
            unlog(fit[[paste0(prefix, "deviation")]]) else NULL
  )
}

# Linear predictor of arm a in target source s_t, per iteration (R x n):
#   eta_a = m_0^sh(x) + (1 - s_t) d(x) + b(a) [tau(x) + (1 - s_t) c(x)],
# where b(a) is the coded treatment used by FusionForest().  Under the
# "centered" and "adaptive" codings the baseline forests do not describe the
# control arm, so the coding must be applied here as in the fit.
.linear_predictor <- function(comp, meta, a, s_t) {
  base <- comp$mu + (if (s_t == 0L && !is.null(comp$g)) comp$g else 0)
  eff  <- comp$tau + (if (s_t == 0L) comp$c else 0)
  b    <- .coded_treatment(meta, a, ncol(eff))
  base + eff * rep(b, each = nrow(eff))
}

# Coded treatment b(a) for each evaluation row, matching FusionForest().
.coded_treatment <- function(meta, a, n_ev) {
  coding <- meta$treatment_coding
  if (is.null(coding))
    stop("fit$meta$treatment_coding is missing.  Refit with the current ",
         "package version, or set it to the coding used for this fit ",
         "(\"centered\", \"binary\" or \"adaptive\").")
  switch(coding,
    binary   = rep(a, n_ev),
    centered = rep(a - 0.5, n_ev),
    adaptive = {
      pi_test <- meta$propensity_test
      if (length(pi_test) != n_ev)
        stop("fit$meta$propensity_test must have one entry per test row ",
             "for treatment_coding = \"adaptive\".")
      a - pi_test
    },
    stop("Unknown treatment_coding: \"", coding, "\"."))
}

# Return per-iteration mixture weights, atoms, and kernel SD for source s_t.
# All quantities are on the user log-time (response) scale.  For Gaussian
# error this is a degenerate one-component "mixture".  For shared_dp the
# same atoms/weights are used regardless of s_t.
.residual_mixture <- function(fit, meta, s_t) {

  err <- meta$error_dist
  R   <- length(fit$sigma)

  if (is.null(err) || err == "gaussian") {
    sigma_resp <- fit$sigma * meta$sigma_hat       # R-vector
    return(list(
      pi    = matrix(1.0, R, 1L),
      theta = matrix(0.0, R, 1L),
      sigma = sigma_resp                            # R-vector (constant in k)
    ))
  }

  # dp_locations is rescaled to response scale in the wrapper; dp_mix_prop
  # is scale-free.  For shared_dp there is a single group (index 1) used
  # for both sources; otherwise group 1 = RWD (s = 0), group 2 = RCT
  # (s = 1).
  g_idx <- if (err == "shared_dp") 1L else (s_t + 1L)

  pi_mat    <- fit$dp_mix_prop[[g_idx]]
  theta_mat <- fit$dp_locations[[g_idx]]

  # Per-source scales ("source_dp_scale", "source_hdp_scale"): use the
  # target source's own sigma_s.  fit$sigma then holds a pooled scale.
  if (!is.null(fit$dp_sigma_g)) {
    sigma_resp <- fit$dp_sigma_g[[g_idx]]           # already on response
  } else {
    sigma_resp <- fit$sigma * meta$sigma_hat
  }

  list(pi = pi_mat, theta = theta_mat, sigma = sigma_resp)
}

# Survival difference draws: R x n_ev matrix.
#   eta0, eta1 : R x n_ev
#   mix$pi, mix$theta : R x K
#   mix$sigma : length R
.sd_draws <- function(log_t, eta0, eta1, mix) {
  K <- ncol(mix$pi)
  R <- nrow(eta0)
  out <- matrix(0.0, R, ncol(eta0))
  inv_sigma <- 1.0 / mix$sigma
  for (k in seq_len(K)) {
    # xi_a,k(x) = eta_a(x) + theta_{s_t,k}; broadcast theta over columns
    theta_k <- mix$theta[, k]                       # length R
    # (log_t - xi_{a,k}) / sigma  -- both eta and sigma vary across r
    z0 <- (log_t - eta0 - theta_k) * inv_sigma
    z1 <- (log_t - eta1 - theta_k) * inv_sigma
    out <- out + mix$pi[, k] * (pnorm(z0) - pnorm(z1))
  }
  out
}
