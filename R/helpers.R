#' Compute mean estimate for censored data
#'
#' Estimates the mean and standard deviation for right-censored or
#' interval-censored survival data by the maximum likelihood fit of a normal
#' distribution.  Uses `survival::survreg()` if available.  If `survival` is
#' missing, or `survreg()` fails or does not converge, it solves the same
#' likelihood in base R with [censored_normal_fit()], so the result does not
#' depend on the optional package beyond the tolerance of the optimizer.
#'
#' Rows follow the convention of [FusionForest()]: `status = 1` is an event at
#' `y`; `status = 0` with `interval = 0` is right-censored at `right_time`;
#' `status = 0` with `interval = 1` is an event in (`left_time`, `right_time`].
#' Without `interval`, rows with `status = 0` are right-censored at `y`.
#'
#' @param y Numeric vector of (log-transformed) survival times.
#' @param status Numeric vector; event indicator (1 = event, 0 = censored).
#' @param left_time Optional numeric vector of left boundaries (for interval censoring).
#' @param right_time Optional numeric vector of right boundaries (for interval censoring).
#' @param interval Optional indicator of interval-censored rows (1 = interval-censored).
#'
#' @return A list with elements:
#'   \item{mu}{Estimated mean of survival times.}
#'   \item{sd}{Estimated standard deviation of survival times.}
#'
#' @noRd
censored_info <- function(y, status, left_time = NULL, right_time = NULL,
                          interval = NULL) {

  # Bounds per row.  Equal bounds mark an event, an infinite bound an open end.
  lo <- y
  hi <- ifelse(status == 1, y, Inf)
  if (!is.null(interval)) {
    rc <- status == 0 & interval == 0
    ic <- status == 0 & interval == 1
    lo[rc] <- right_time[rc]
    lo[ic] <- left_time[ic]
    hi[ic] <- right_time[ic]
  }

  if (requireNamespace("survival", quietly = TRUE)) {
    surv_obj <- if (is.null(interval)) {
      survival::Surv(y, status)
    } else {
      # Surv(type = "interval2") reads NA as an open end.
      survival::Surv(ifelse(is.finite(lo), lo, NA),
                     ifelse(is.finite(hi), hi, NA), type = "interval2")
    }
    fit <- tryCatch(survival::survreg(surv_obj ~ 1, dist = "gaussian"),
                    error = function(e) NULL, warning = function(w) NULL)
    if (!is.null(fit)) {
      return(list(mu = as.numeric(fit$coefficients[1]),
                  sd = as.numeric(fit$scale)))
    }
  }

  censored_normal_fit(lo, hi)
}

#' Maximum likelihood fit of a normal distribution to censored data
#'
#' The model of `survival::survreg(Surv(...) ~ 1, dist = "gaussian")`, on the
#' parameters mean and log standard deviation, solved in base R.
#'
#' @param lo,hi Bounds per row.  Equal bounds mark an event, an infinite bound
#'   an open end.
#'
#' @return A list with elements `mu` and `sd`.
#'
#' @importFrom stats dnorm optim
#' @noRd
censored_normal_fit <- function(lo, hi) {

  keep  <- !is.na(lo) & !is.na(hi)
  lo    <- lo[keep]
  hi    <- hi[keep]
  event <- lo == hi

  # Starting values from the midpoints of finite intervals, the events and the
  # bounds of one-sided rows.
  finite_row <- is.finite(lo) & is.finite(hi)
  z  <- ifelse(finite_row, (lo + hi) / 2, ifelse(is.finite(lo), lo, hi))
  z  <- z[is.finite(z)]
  mu <- mean(z)
  s  <- sd(z)
  if (!is.finite(s) || s <= 0) s <- 1

  # One-sided rows alone cannot locate the distribution.
  if (!any(finite_row)) return(list(mu = mu, sd = s))

  neg_loglik <- function(par) {
    m  <- par[1]
    sg <- exp(par[2])
    a  <- (lo[!event] - m) / sg
    b  <- (hi[!event] - m) / sg
    # Probabilities in the lower tail, which keeps precision far above m.
    flip  <- a > 0
    lower <- ifelse(flip, -b, a)
    upper <- ifelse(flip, -a, b)
    lp_up <- pnorm(upper, log.p = TRUE)
    lp_lo <- pnorm(lower, log.p = TRUE)
    -sum(dnorm(lo[event], m, sg, log = TRUE)) -
      sum(lp_up + log1p(-pmin(exp(lp_lo - lp_up), 1 - 1e-16)))
  }
  fit <- optim(c(mu, log(s)), neg_loglik, method = "BFGS",
               control = list(maxit = 500, reltol = 1e-10))
  if (!all(is.finite(fit$par))) return(list(mu = mu, sd = s))

  list(mu = fit$par[1], sd = exp(fit$par[2]))
}
