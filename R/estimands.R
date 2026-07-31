#' Causal survival estimands from a FusionForest fit
#'
#' Compute posterior draws of the survival difference, restricted mean
#' survival time (RMST) difference, or acceleration factor (AF) at the
#' covariate values supplied to \code{FusionForest()} as the test set,
#' using the closed-form expressions of the AFT decomposition with
#' (H)DP error.  See \code{notes/estimands.tex} for the derivations.
#'
#' @details
#' Under \deqn{\log T = m_0(X, S) + A \tau(X) + (1-S) A c(X) + \sigma
#' \varepsilon, \quad \varepsilon \sim \sum_k \pi_{S,k} N(\theta_{S,k},
#' 1),} the survival function under treatment \eqn{a} in target
#' population \eqn{s_t} is
#' \deqn{S_a(t | x, s_t) = 1 - \sum_k \pi_{s_t,k} \Phi\!\bigl((\log t -
#' \eta_a(x, s_t) - \theta_{s_t,k})/\sigma\bigr),}
#' with \eqn{\eta_a(x, s_t) = m_0(x, s_t) + a \tau(x) + (1-s_t) a c(x)}
#' and \eqn{m_0(x, s_t) = \mu(x) + (1 - s_t) g(x)} in four-forest mode.
#' The three estimands implemented here are
#' \describe{
#'   \item{\code{"SD"}}{Survival difference
#'     \eqn{\Delta_{SD}(t; x, s_t) = S_1(t|x, s_t) - S_0(t|x, s_t)}.}
#'   \item{\code{"RMST"}}{Restricted mean survival time difference at
#'     horizon \eqn{t^{*}}, computed in closed form as the source-weighted
#'     sum of lognormal RMSTs.}
#'   \item{\code{"AF"}}{Causal acceleration factor
#'     \eqn{\exp(\tau(x))}.  Does not depend on the error distribution
#'     or on \code{target_source}.}
#' }
#'
#' Population-averaged versions use Bayesian-bootstrap weights drawn fresh
#' at each MCMC iteration (\eqn{w^{(r)} \sim Dirichlet(1, \ldots, 1)}), as
#' in Rubin (1981); set \code{bayesian_bootstrap = FALSE} for equal
#' weights.
#'
#' @param fit A fitted \code{FusionForest} object obtained with
#'   \code{store_posterior_sample = TRUE}.  Must include posterior sample
#'   matrices for all relevant component forests.  When
#'   \code{estimand \%in\% c("SD", "RMST")}, the fit must use one of the
#'   supported \code{error_dist} families.
#' @param estimand Character, one of \code{"SD"}, \code{"RMST"},
#'   \code{"AF"}.
#' @param time Numeric scalar (positive).  Required for \code{"SD"} and
#'   \code{"RMST"}; ignored for \code{"AF"}.  Interpreted on the
#'   \emph{time} (not log-time) scale.
#' @param target_source Character, \code{"rwd"} (default) or \code{"rct"}.
#'   Selects the target population \eqn{s_t} appearing in the survival
#'   formula (eqs.\ 2.6--2.8 of \code{estimands.tex}).  Ignored for
#'   \code{"AF"} since the causal AF is shared across sources.
#' @param population_average Logical; if \code{TRUE} the function returns
#'   posterior draws of the population-averaged estimand (a length
#'   \eqn{R} vector).  Otherwise returns the \eqn{R \times n_{ev}} matrix
#'   of per-\eqn{x} draws.  Default \code{FALSE}.
#' @param bayesian_bootstrap Logical; if \code{TRUE} (default) the
#'   population average uses Dirichlet weights drawn fresh at each
#'   iteration.  If \code{FALSE} the average uses equal weights
#'   \eqn{1/n_{ev}}.  Ignored when \code{population_average = FALSE}.
#' @param seed Integer or \code{NULL}.  Sets the RNG seed used to draw
#'   the Bayesian-bootstrap weights, for reproducibility.
#'
#' @return If \code{population_average = FALSE}, an \eqn{R \times n_{ev}}
#'   matrix of posterior draws of the estimand at each evaluation point
#'   (test rows of the fit).  Otherwise a length-\eqn{R} numeric vector
#'   of population-averaged draws.  In both cases the result carries
#'   attributes \code{estimand}, \code{time}, \code{target_source}, and
#'   (when applicable) \code{bayesian_bootstrap}.
#'
#' @references
#' Rubin, D.~B. (1981).  The Bayesian bootstrap.
#' \emph{The Annals of Statistics}, 9, 130--134.
#'
#' Royston, P.\ and Parmar, M.~K.~B. (2013).  Restricted mean survival
#' time: an alternative to the hazard ratio for the design and analysis
#' of randomized trials with a time-to-event outcome.
#' \emph{BMC Medical Research Methodology}, 13, 152.
#'
#' @examples
#' # Right-censored survival fusion fit with stored posterior samples
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
#' # Posterior draws of the acceleration factor at each test point
#' af <- fusion_estimand(fit, estimand = "AF")
#' quantile(colMeans(af), c(0.25, 0.5, 0.75))
#'
#' @importFrom stats pnorm rgamma
#' @export
fusion_estimand <- function(
  fit,
  estimand           = c("SD", "RMST", "AF"),
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

  if (estimand %in% c("SD", "RMST")) {
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
    # eta_a(x, s_t) = m_0(x, s_t) + a tau(x) + (1 - s_t) a c(x)
    s_t <- if (target_source == "rct") 1L else 0L
    m0  <- comp$mu + (if (!is.null(comp$g) && s_t == 0L) comp$g else 0)
    eta0 <- m0
    eta1 <- m0 + comp$tau + (if (s_t == 0L) comp$c else 0)

    mix  <- .residual_mixture(fit, meta, s_t)     # list(pi, theta, sigma)
    log_t <- log(time)

    if (estimand == "SD") {
      draws <- .sd_draws(log_t, eta0, eta1, mix)
      attrs <- list(estimand = "SD", time = time,
                    target_source = target_source)
    } else {                                       # "RMST"
      draws <- .rmst_draws(time, log_t, eta0, eta1, mix)
      attrs <- list(estimand = "RMST", time = time,
                    target_source = target_source)
    }
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
# g(X) is NULL in three-forest mode.  `which` selects test or train rows.
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
    g   = if (meta$decomposition == "four-forest")
            unlog(fit[[paste0(prefix, "deviation")]]) else NULL
  )
}

# Return per-iteration mixture weights, atoms, and kernel SD for source s_t.
# All quantities are on the user log-time (response) scale.  For Gaussian
# error this is a degenerate one-component "mixture".  For shared_dp the
# same atoms/weights are used regardless of s_t.
.residual_mixture <- function(fit, meta, s_t) {

  err <- meta$error_dist
  R   <- length(fit$sigma)

  if (err == "gaussian" || is.null(err)) {
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

  if (err == "source_dp_scale") {
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

# RMST contrast draws: R x n_ev matrix.  Per-component lognormal RMST,
# weighted by the source mixture.
.rmst_draws <- function(t_star, log_t, eta0, eta1, mix) {
  K <- ncol(mix$pi)
  R <- nrow(eta0)
  out <- matrix(0.0, R, ncol(eta0))
  sigma_r <- mix$sigma                              # R-vector
  sigma2  <- sigma_r * sigma_r                      # R-vector
  for (k in seq_len(K)) {
    theta_k <- mix$theta[, k]                       # R-vector
    xi0 <- eta0 + theta_k
    xi1 <- eta1 + theta_k
    out <- out + mix$pi[, k] *
      (.rmst_component(t_star, log_t, xi1, sigma_r, sigma2) -
       .rmst_component(t_star, log_t, xi0, sigma_r, sigma2))
  }
  out
}

# Lognormal RMST: E[min(T, t_star)] for log T ~ N(xi, sigma^2).
# Returns an R x n_ev matrix.  Standard closed form:
#   exp(xi + sigma^2/2) Phi((log t_star - xi - sigma^2)/sigma)
#   + t_star (1 - Phi((log t_star - xi)/sigma)).
.rmst_component <- function(t_star, log_t, xi, sigma_r, sigma2) {
  inv_sigma <- 1.0 / sigma_r
  a <- (log_t - xi - sigma2) * inv_sigma
  b <- (log_t - xi) * inv_sigma
  exp(xi + 0.5 * sigma2) * pnorm(a) + t_star * (1.0 - pnorm(b))
}
