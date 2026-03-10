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
#' @param ic_indicator Optional numeric vector; 1 = interval-censored, 0 = right-censored/exact.
#'
#' @return A list with elements:
#'   \item{mu}{Estimated mean of survival times.}
#'   \item{sd}{Estimated standard deviation of survival times.}
#'   \item{min}{Estimated minimum of survival times.}
#'   \item{max}{Estimated maximum of survival times.}
#'
#' @importFrom stats dnorm pnorm
#' @noRd
censored_info <- function(y, status, left_time = NULL, right_time = NULL,
                          ic_indicator = NULL) {

  if (requireNamespace("survival", quietly = TRUE)) {

    if (!is.null(left_time) && !is.null(right_time)) {
      # Interval-censored path: use Surv(time, time2, type = "interval2")
      surv_obj <- survival::Surv(left_time, right_time, type = "interval2")
      fit <- survival::survreg(surv_obj ~ 1, dist = "gaussian")

      mu <- as.numeric(fit$coefficients[1])
      sd <- as.numeric(fit$scale)

      # Impute censored values for min/max estimation
      imputed_y <- y

      # Right-censored observations: impute with Mills ratio
      if (!is.null(ic_indicator)) {
        rc_idx <- which(status == 0 & ic_indicator == 0)
        if (length(rc_idx) > 0) {
          a <- (y[rc_idx] - mu) / sd
          lambda_val <- dnorm(a) / (1 - pnorm(a))
          imputed_y[rc_idx] <- mu + sd * lambda_val
        }
      }

      min_val <- min(imputed_y, na.rm = TRUE)
      max_val <- max(imputed_y, na.rm = TRUE)

    } else {
      # Right-censored path
      fit <- survival::survreg(survival::Surv(y, status) ~ 1, dist = "gaussian")

      mu <- as.numeric(fit$coefficients[1])
      sd <- as.numeric(fit$scale)

      # Impute censored values
      cens_idx <- which(status == 0)
      imputed_y <- y

      if (length(cens_idx) > 0) {
        a <- (y[cens_idx] - mu) / sd
        lambda <- dnorm(a) / (1 - pnorm(a))
        imputed_y[cens_idx] <- mu + sd * lambda
      }

      min_val <- min(imputed_y, na.rm = TRUE)
      max_val <- max(imputed_y, na.rm = TRUE)
    }

  } else {

    # Fallback: use only uncensored observations
    min_val <- min(y, na.rm = TRUE)
    max_val <- max(y, na.rm = TRUE)

    mu <- mean(y[status == 1], na.rm = TRUE)
    sd <- sd(y[status == 1], na.rm = TRUE)
  }

  return(list(
    mu = mu,
    sd = sd,
    min = min_val,
    max = max_val
  ))
}
