#' Compute mean estimate for censored data
#'
#' Estimates the mean and standard deviation for right-censored or
#' interval-censored survival data. Uses the `survival` package if available,
#' otherwise falls back to the naive mean among observed events.
#'
#' @param y Numeric vector of (log-transformed) survival times.
#' @param status Numeric vector; event indicator (1 = event, 0 = censored).
#' @param left_time Optional numeric vector of left boundaries (for interval censoring).
#' @param right_time Optional numeric vector of right boundaries (for interval censoring).
#'
#' @return A list with elements:
#'   \item{mu}{Estimated mean of survival times.}
#'   \item{sd}{Estimated standard deviation of survival times.}
#'
#' @noRd
censored_info <- function(y, status, left_time = NULL, right_time = NULL) {

  if (requireNamespace("survival", quietly = TRUE)) {

    if (!is.null(left_time) && !is.null(right_time)) {
      # Interval-censored path: use Surv(time, time2, type = "interval2")
      surv_obj <- survival::Surv(left_time, right_time, type = "interval2")
    } else {
      # Right-censored path
      surv_obj <- survival::Surv(y, status)
    }
    fit <- survival::survreg(surv_obj ~ 1, dist = "gaussian")
    mu  <- as.numeric(fit$coefficients[1])
    sd  <- as.numeric(fit$scale)

  } else {

    # Fallback: use only uncensored observations
    mu <- mean(y[status == 1], na.rm = TRUE)
    sd <- sd(y[status == 1], na.rm = TRUE)
  }

  list(mu = mu, sd = sd)
}
