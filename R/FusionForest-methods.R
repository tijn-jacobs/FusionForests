#' Print a FusionForest fit
#'
#' Displays a compact overview of a fitted [FusionForest()] model:
#' outcome type, forest decomposition, error model, sample sizes and
#' MCMC settings.
#'
#' @param x A `FusionForest` object.
#' @param ... Ignored.
#'
#' @return `x`, invisibly.
#'
#' @export
print.FusionForest <- function(x, ...) {
  meta <- x$meta
  cat("FusionForest fit\n")
  if (identical(meta$outcome_type, "continuous")) {
    cat("  Outcome type:    continuous\n")
  } else {
    cat("  Outcome type:    ", meta$outcome_type,
        " (timescale: ", meta$timescale, ")\n", sep = "")
  }
  cat("  Decomposition:   ", meta$decomposition, "\n", sep = "")
  cat("  Error model:     ", meta$error_dist, "\n", sep = "")
  cat("  Observations:    ", meta$n_train,
      " (RCT: ", meta$n_rct, ", RWD: ", meta$n_rwd, ")\n", sep = "")
  cat("  Covariates:      ", meta$p_control, " (control forest), ",
      meta$p_treat, " (treatment forest)\n", sep = "")
  cat("  Posterior draws: ", meta$N_post,
      " (burn-in: ", meta$N_burn, ")\n", sep = "")
  invisible(x)
}

#' Summarise a FusionForest fit
#'
#' Computes posterior summaries of a fitted [FusionForest()] model:
#' the residual standard deviation and the distribution of the
#' estimated individual treatment effects (posterior means of the
#' treatment forest evaluated at the training covariates).
#'
#' @param object A `FusionForest` object.
#' @param ... Ignored.
#'
#' @return An object of class `summary.FusionForest`: a list with
#'   elements `meta`, `sigma` (posterior draws of the residual SD on
#'   the standardised scale, `NULL` if `sigma` was fixed),
#'   `treatment_effects` (posterior-mean treatment forest predictions
#'   for the training data) and `acceptance_ratios`.
#'
#' @export
summary.FusionForest <- function(object, ...) {
  out <- list(
    meta = object$meta,
    sigma = object$sigma,
    treatment_effects = object$train_predictions_treat,
    acceptance_ratios = c(
      control   = mean(object$acceptance_ratio_control),
      treat     = mean(object$acceptance_ratio_treat),
      deconf    = mean(object$acceptance_ratio_deconf),
      deviation = if (!is.null(object$acceptance_ratio_deviation))
        mean(object$acceptance_ratio_deviation) else NA_real_
    )
  )
  class(out) <- "summary.FusionForest"
  out
}

#' Print a FusionForest summary
#'
#' @param x A `summary.FusionForest` object.
#' @param digits Number of significant digits to print.
#' @param ... Ignored.
#'
#' @return `x`, invisibly.
#'
#' @export
print.summary.FusionForest <- function(x, digits = 3, ...) {
  meta <- x$meta
  cat("Summary of FusionForest fit\n\n")
  cat("Model\n")
  cat("  Outcome type:    ", meta$outcome_type, "\n", sep = "")
  cat("  Decomposition:   ", meta$decomposition, "\n", sep = "")
  cat("  Error model:     ", meta$error_dist, "\n", sep = "")
  cat("  Observations:    ", meta$n_train,
      " (RCT: ", meta$n_rct, ", RWD: ", meta$n_rwd, ")\n", sep = "")
  cat("  Posterior draws: ", meta$N_post,
      " (burn-in: ", meta$N_burn, ")\n\n", sep = "")

  if (!is.null(x$sigma) && length(x$sigma) > 0) {
    sig <- x$sigma * meta$sigma_hat
    qs  <- quantile(sig, c(0.025, 0.5, 0.975))
    cat("Residual standard deviation (posterior)\n")
    cat("  Mean: ", signif(mean(sig), digits),
        "   Median: ", signif(qs[2], digits),
        "   95% CI: [", signif(qs[1], digits), ", ",
        signif(qs[3], digits), "]\n\n", sep = "")
  }

  te <- x$treatment_effects
  if (!is.null(te) && length(te) > 0) {
    qs <- quantile(te, c(0.025, 0.25, 0.5, 0.75, 0.975))
    label <- if (identical(meta$outcome_type, "continuous")) {
      "Estimated individual treatment effects (posterior means)"
    } else if (identical(meta$timescale, "time")) {
      "Estimated individual survival time ratios (posterior means)"
    } else {
      "Estimated individual treatment effects on log time (posterior means)"
    }
    cat(label, "\n", sep = "")
    print(signif(qs, digits))
    cat("\n")
  }

  cat("Average Metropolis-Hastings acceptance ratios\n")
  ar <- x$acceptance_ratios[!is.na(x$acceptance_ratios)]
  print(signif(ar, digits))
  invisible(x)
}
