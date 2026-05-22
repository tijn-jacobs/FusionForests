#' Posterior predictive inference for a ShrinkageTrees model
#'
#' Re-runs the MCMC sampler on new covariate data using the stored training
#' data and hyperparameters, returning posterior mean predictions and credible
#' interval bounds.
#'
#' @param object A fitted \code{ShrinkageTrees} model object.
#' @param newdata A matrix (or object coercible to one) of new covariates with
#'   the same number of columns as the training data.
#' @param level Credible interval width. Default \code{0.95}.
#' @param ... Currently unused.
#' @return A \code{ShrinkageTreesPrediction} object with elements:
#'   \describe{
#'     \item{mean}{Posterior mean predictions (length \code{nrow(newdata)}).}
#'     \item{lower}{Lower credible interval bound.}
#'     \item{upper}{Upper credible interval bound.}
#'     \item{n}{Number of test observations.}
#'     \item{level}{Credible level used.}
#'     \item{outcome_type}{Outcome type inherited from the fitted model.}
#'     \item{timescale}{Timescale inherited from the fitted model (survival only).}
#'     \item{predictions_sample}{(Survival only) \code{N_post} x \code{n}
#'       matrix of posterior predictive draws on the original scale.}
#'     \item{sigma}{(Survival only) Posterior draws of sigma on the log-time
#'       scale (length \code{N_post}).}
#'   }
#' @seealso \code{\link{HorseTrees}}, \code{\link{ShrinkageTrees}},
#'   \code{\link{print.ShrinkageTreesPrediction}},
#'   \code{\link{summary.ShrinkageTreesPrediction}},
#'   \code{\link{plot.ShrinkageTreesPrediction}}
#' @export
predict.ShrinkageTrees <- function(object, newdata, level = 0.95, ...) {

  if (!is.matrix(newdata)) newdata <- as.matrix(newdata)
  if (ncol(newdata) != object$data_info$p_features)
    stop("newdata has ", ncol(newdata), " columns but the model expects ",
         object$data_info$p_features, ".")

  n_new   <- nrow(newdata)
  X_new   <- as.numeric(t(newdata))
  n_train <- object$data_info$n_train
  p       <- object$data_info$p_features
  pre     <- object$preprocess
  args    <- object$args
  alpha   <- (1 - level) / 2

  if (object$outcome_type == "binary") {

    fit <- probitHorseTrees_cpp(
      nSEXP                    = n_train,
      pSEXP                    = p,
      n_testSEXP               = n_new,
      X_trainSEXP              = object$data$X_train,
      ySEXP                    = object$data$y_train,
      X_testSEXP               = X_new,
      number_of_treesSEXP      = object$mcmc$number_of_trees,
      N_postSEXP               = object$mcmc$N_post,
      N_burnSEXP               = object$mcmc$N_burn,
      delayed_proposalSEXP     = args$delayed_proposal,
      powerSEXP                = args$power,
      baseSEXP                 = args$base,
      p_growSEXP               = args$p_grow,
      p_pruneSEXP              = args$p_prune,
      omegaSEXP                = args$omega,
      latent_thresholdSEXP     = pre$latent_threshold,
      param1SEXP               = args$param1,
      param2SEXP               = args$param2,
      prior_typeSEXP           = object$prior$prior_type_cpp,
      reversibleSEXP           = args$reversible,
      store_posterior_sampleSEXP = TRUE,
      verboseSEXP              = FALSE
    )

    out <- list(
      mean         = pnorm(fit$test_predictions),
      lower        = pnorm(apply(fit$test_predictions_sample, 2, quantile, alpha)),
      upper        = pnorm(apply(fit$test_predictions_sample, 2, quantile, 1 - alpha)),
      n            = n_new,
      level        = level,
      outcome_type = object$outcome_type,
      timescale    = object$timescale
    )
    class(out) <- "ShrinkageTreesPrediction"
    return(out)

  } else {

    y_std    <- (object$data$y_train - pre$y_mean) / pre$sigma_hat
    survival <- object$outcome_type %in% c("right-censored", "interval-censored")
    status   <- if (survival) object$data$status_train else rep(1L, n_train)

    # Reconstruct interval-censored boundaries if applicable
    if (object$outcome_type == "interval-censored") {
      left_std  <- (object$data$left_time_train  - pre$y_mean) / pre$sigma_hat
      right_std <- (object$data$right_time_train - pre$y_mean) / pre$sigma_hat
      ic_ind    <- object$data$ic_indicator_train
    } else {
      left_std  <- numeric(n_train)
      right_std <- y_std + 0
      ic_ind    <- numeric(n_train)
    }

    fit <- HorseTrees_cpp(
      nSEXP                    = n_train,
      pSEXP                    = p,
      n_testSEXP               = n_new,
      X_trainSEXP              = object$data$X_train,
      ySEXP                    = y_std,
      status_indicatorSEXP     = status,
      is_survivalSEXP          = survival,
      observed_left_timeSEXP   = left_std,
      observed_right_timeSEXP  = right_std,
      interval_censoring_indicatorSEXP = ic_ind,
      X_testSEXP               = X_new,
      number_of_treesSEXP      = object$mcmc$number_of_trees,
      N_postSEXP               = object$mcmc$N_post,
      N_burnSEXP               = object$mcmc$N_burn,
      delayed_proposalSEXP     = args$delayed_proposal,
      powerSEXP                = args$power,
      baseSEXP                 = args$base,
      p_growSEXP               = args$p_grow,
      p_pruneSEXP              = args$p_prune,
      nuSEXP                   = args$nu,
      lambdaSEXP               = args$lambda,
      dirichlet_boolSEXP       = isTRUE(object$prior$dirichlet),
      a_dirichletSEXP          = args$a_dirichlet,
      b_dirichletSEXP          = args$b_dirichlet,
      rho_dirichletSEXP        = args$rho_dirichlet,
      sigmaSEXP                = pre$sigma_hat,
      sigma_knownSEXP          = object$preprocess$sigma_known,
      omegaSEXP                = args$omega,
      param1SEXP               = args$param1,
      param2SEXP               = args$param2,
      prior_typeSEXP           = object$prior$prior_type_cpp,
      reversibleSEXP           = args$reversible,
      store_parametersSEXP     = FALSE,
      store_posterior_sampleSEXP = TRUE,
      verboseSEXP              = FALSE
    )

    # De-standardize
    if (survival && object$timescale == "time") {
      mean_pred   <- exp(fit$test_predictions       * pre$sigma_hat + pre$y_mean)
      sample_pred <- exp(fit$test_predictions_sample * pre$sigma_hat + pre$y_mean)
    } else {
      mean_pred   <- fit$test_predictions       * pre$sigma_hat + pre$y_mean
      sample_pred <- fit$test_predictions_sample * pre$sigma_hat + pre$y_mean
    }

    out <- list(
      mean         = mean_pred,
      lower        = apply(sample_pred, 2, quantile, alpha),
      upper        = apply(sample_pred, 2, quantile, 1 - alpha),
      n            = n_new,
      level        = level,
      outcome_type = object$outcome_type,
      timescale    = object$timescale
    )

    # Store posterior samples for survival curve plotting
    if (survival) {
      out$predictions_sample <- sample_pred   # N_post x n_new
      # sigma is jointly sampled with the trees (sigma_known = FALSE for
      # survival); remove burn-in to match predictions_sample rows.
      out$sigma <- fit$sigma[-(1:object$mcmc$N_burn)] * pre$sigma_hat
    }

    class(out) <- "ShrinkageTreesPrediction"
    out
  }
}

#' Print a ShrinkageTreesPrediction object
#'
#' Displays a formatted table of posterior mean predictions and credible
#' interval bounds for the first \code{n_head} observations.
#'
#' @param x A \code{ShrinkageTreesPrediction} object.
#' @param n_head Number of observations to display. Default \code{6}.
#' @param digits Number of decimal places. Default \code{3}.
#' @param ... Currently unused.
#' @return Invisibly returns \code{x}.
#' @seealso \code{\link{predict.ShrinkageTrees}},
#'   \code{\link{summary.ShrinkageTreesPrediction}}
#' @export
print.ShrinkageTreesPrediction <- function(x, n_head = 6, digits = 3, ...) {

  ci_pct <- paste0(round(x$level * 100), "%")
  scale_label <- switch(
    x$outcome_type,
    "binary"            = "probability",
    "right-censored"    = if (x$timescale == "time") "survival time" else "log survival time",
    "interval-censored" = if (x$timescale == "time") "survival time" else "log survival time",
    "fitted value"
  )

  cat("\n")
  cat("ShrinkageTrees predictions\n")
  cat("--------------------------\n")
  lbl <- function(s) sprintf("%-22s", s)
  cat(lbl("Observations:"),      x$n,         "\n", sep = "")
  cat(lbl("Credible interval:"), ci_pct,      "\n", sep = "")
  cat(lbl("Scale:"),             scale_label, "\n", sep = "")

  n_show <- min(n_head, x$n)
  cat("\n")
  cat(sprintf("  %5s  %8s  %8s  %8s\n", "", "mean", "lower", "upper"))
  cat(sprintf("  %5s  %8s  %8s  %8s\n", "", strrep("-", 8), strrep("-", 8), strrep("-", 8)))
  for (i in seq_len(n_show)) {
    cat(sprintf("  [%3d]  %8.*f  %8.*f  %8.*f\n",
                i, digits, x$mean[i], digits, x$lower[i], digits, x$upper[i]))
  }
  if (x$n > n_head)
    cat("  ... (", x$n - n_head, " more)\n", sep = "")

  cat("\n")
  invisible(x)
}

#' Summarise a ShrinkageTreesPrediction object
#'
#' Returns distributional summaries (min, Q1, median, max) of the posterior
#' mean predictions and credible interval bounds across all observations.
#'
#' @param object A \code{ShrinkageTreesPrediction} object.
#' @param ... Currently unused.
#' @return A \code{summary.ShrinkageTreesPrediction} object.
#' @seealso \code{\link{predict.ShrinkageTrees}},
#'   \code{\link{print.summary.ShrinkageTreesPrediction}}
#' @export
summary.ShrinkageTreesPrediction <- function(object, ...) {
  out <- list(
    n            = object$n,
    level        = object$level,
    outcome_type = object$outcome_type,
    timescale    = object$timescale,
    mean         = quantile(object$mean,  c(0, 0.25, 0.5, 1)),
    lower        = quantile(object$lower, c(0, 0.25, 0.5, 1)),
    upper        = quantile(object$upper, c(0, 0.25, 0.5, 1))
  )
  class(out) <- "summary.ShrinkageTreesPrediction"
  out
}

#' Print a ShrinkageTreesPrediction summary
#'
#' Displays distributional summaries (min, Q1, median, max) of the posterior
#' mean predictions and credible interval bounds.
#'
#' @param x A \code{summary.ShrinkageTreesPrediction} object.
#' @param digits Number of decimal places. Default \code{3}.
#' @param ... Currently unused.
#' @return Invisibly returns \code{x}.
#' @seealso \code{\link{summary.ShrinkageTreesPrediction}}
#' @export
print.summary.ShrinkageTreesPrediction <- function(x, digits = 3, ...) {

  ci_pct <- paste0(round(x$level * 100), "%")

  cat("\n")
  cat("ShrinkageTrees prediction summary\n")
  cat("----------------------------------\n")
  lbl <- function(s) sprintf("%-22s", s)
  cat(lbl("Observations:"),      x$n,    "\n", sep = "")
  cat(lbl("Credible interval:"), ci_pct, "\n", sep = "")

  cat("\n")
  cat(sprintf("  %-8s  %8s  %8s  %8s\n", "", "mean", "lower", "upper"))
  cat(sprintf("  %-8s  %8s  %8s  %8s\n", "", strrep("-", 8), strrep("-", 8), strrep("-", 8)))
  nms <- c("Min.", "Q1", "Median", "Max.")
  for (i in seq_along(nms)) {
    cat(sprintf("  %-8s  %8.*f  %8.*f  %8.*f\n",
                nms[i], digits, x$mean[i], digits, x$lower[i], digits, x$upper[i]))
  }

  cat("\n")
  invisible(x)
}

#' Print a ShrinkageTrees model
#'
#' Displays a concise summary of a fitted \code{ShrinkageTrees} model,
#' including outcome type, prior, MCMC settings, acceptance ratio, and
#' posterior mean sigma.
#'
#' @param x A fitted \code{ShrinkageTrees} model object.
#' @param ... Currently unused.
#' @return Invisibly returns \code{x}.
#' @seealso \code{\link{summary.ShrinkageTrees}}, \code{\link{HorseTrees}},
#'   \code{\link{ShrinkageTrees}}
#' @export
print.ShrinkageTrees <- function(x, ...) {
  cat("\n")
  cat("ShrinkageTrees model\n")
  cat("---------------------\n")

  # Outcome type
  outcome_label <- switch(
    x$outcome_type,
    "continuous" = "Continuous",
    "binary" = "Binary (probit)",
    "right-censored" = paste0("Right-censored (AFT, timescale = ",
                              x$timescale, ")"),
    "interval-censored" = paste0("Interval-censored (AFT, timescale = ",
                                  x$timescale, ")"),
    x$outcome_type
  )

  lbl <- function(s) sprintf("%-22s", s)

  cat(lbl("Outcome type:"), outcome_label, "\n", sep = "")
  cat(lbl("Prior:"), x$prior$prior_type_user, "\n", sep = "")
  cat(lbl("Number of trees:"), x$mcmc$number_of_trees, "\n", sep = "")
  cat(lbl("Training size (n):"), x$data_info$n_train, "\n", sep = "")
  cat(lbl("Number of features:"), x$data_info$p_features, "\n", sep = "")
  n_ch <- x$mcmc$n_chains
  if (!is.null(n_ch) && n_ch > 1) {
    cat(lbl("Chains:"),          n_ch,         "\n", sep = "")
    cat(lbl("Draws per chain:"), x$mcmc$N_post,
        " (burn-in ", x$mcmc$N_burn, ")\n", sep = "")
  } else {
    cat(lbl("Posterior draws:"), x$mcmc$N_post,
        " (burn-in ", x$mcmc$N_burn, ")\n", sep = "")
  }

  if (!is.null(x$chains)) {
    cat(lbl("Acceptance ratio:"),
        paste(round(x$chains$acceptance_ratios, 3), collapse = ", "),
        " (per chain)\n", sep = "")
  } else if (!is.null(x$acceptance_ratio)) {
    cat(lbl("Acceptance ratio:"), round(mean(x$acceptance_ratio), 3), "\n",
        sep = "")
  }

  if (!x$preprocess$sigma_known && !is.null(x$sigma))
    cat(lbl("Posterior mean sigma:"), round(mean(x$sigma), 3), "\n", sep = "")

  cat("\n")
  invisible(x)
}

# -- summary ------------------------------------------------------------------

.outcome_label <- function(outcome_type, timescale) {
  switch(
    outcome_type,
    "continuous"        = "Continuous",
    "binary"            = "Binary (probit)",
    "right-censored"    = paste0("Right-censored (AFT, timescale = ",
                                  timescale, ")"),
    "interval-censored" = paste0("Interval-censored (AFT, timescale = ",
                                  timescale, ")"),
    outcome_type
  )
}

.sigma_summary <- function(sigma) {
  c(
    mean  = mean(sigma),
    sd    = sd(sigma),
    lower = unname(quantile(sigma, 0.025)),
    upper = unname(quantile(sigma, 0.975))
  )
}

.pred_summary <- function(preds) {
  c(mean = mean(preds), sd = sd(preds),
    min  = min(preds),  max = max(preds))
}

.vi_from_counts <- function(counts) {
  vi <- colMeans(counts / rowSums(counts))
  names(vi) <- if (!is.null(colnames(counts))) colnames(counts)
               else paste0("X", seq_along(vi))
  sort(vi, decreasing = TRUE)
}

.vi_from_probs <- function(probs) {
  vi <- colMeans(probs)
  names(vi) <- if (!is.null(colnames(probs))) colnames(probs)
               else paste0("X", seq_along(vi))
  sort(vi, decreasing = TRUE)
}

#' Summarise a ShrinkageTrees model
#'
#' Returns an inspectable list with posterior sigma summaries, prediction
#' summaries, variable importance (posterior inclusion probabilities), and
#' MCMC diagnostics.
#'
#' @param object A fitted \code{ShrinkageTrees} model object.
#' @param ... Currently unused.
#' @return A \code{summary.ShrinkageTrees} object with elements:
#'   \describe{
#'     \item{call}{The original model call.}
#'     \item{outcome_type}{Outcome type (\code{"continuous"}, \code{"binary"},
#'       \code{"right-censored"}, or \code{"interval-censored"}).}
#'     \item{timescale}{Timescale for survival outcomes (\code{"time"} or
#'       \code{"log"}).}
#'     \item{prior}{Prior specification.}
#'     \item{mcmc}{MCMC settings (trees, draws, burn-in).}
#'     \item{data_info}{Training and test data dimensions.}
#'     \item{sigma}{Named vector with posterior mean, SD, and 95\% CI of sigma
#'       (continuous and survival outcomes only).}
#'     \item{predictions}{List with \code{train} (and optionally \code{test})
#'       prediction summaries (mean, SD, range).}
#'     \item{variable_importance}{Named vector of posterior inclusion
#'       probabilities, sorted decreasingly (if available).}
#'     \item{acceptance_ratio}{MCMC acceptance ratio vector.}
#'     \item{diagnostics}{(When \pkg{coda} is installed) A list with
#'       \code{ess} (effective sample size) and, for multi-chain fits,
#'       \code{rhat} (Gelman--Rubin \eqn{\hat{R}}).}
#'   }
#' @seealso \code{\link{print.summary.ShrinkageTrees}},
#'   \code{\link{as.mcmc.list.ShrinkageTrees}},
#'   \code{\link{HorseTrees}}, \code{\link{ShrinkageTrees}}
#' @export
summary.ShrinkageTrees <- function(object, ...) {

  out <- list(
    call         = object$call,
    outcome_type = object$outcome_type,
    timescale    = object$timescale,
    prior        = object$prior,
    mcmc         = object$mcmc,
    data_info    = object$data_info
  )

  if (!object$preprocess$sigma_known && !is.null(object$sigma))
    out$sigma <- .sigma_summary(object$sigma)

  out$predictions <- list(
    train = .pred_summary(object$train_predictions),
    test  = if (isTRUE(object$data_info$test_provided))
              .pred_summary(object$test_predictions)
  )

  if (isTRUE(object$prior$dirichlet) && !is.null(object$split_probs))
    out$variable_importance <- .vi_from_probs(object$split_probs)
  else if (!is.null(object$store_split_counts))
    out$variable_importance <- .vi_from_counts(object$store_split_counts)

  out$acceptance_ratio <- object$acceptance_ratio
  if (!is.null(object$chains))
    out$chains <- object$chains

  out$diagnostics <- .mcmc_diagnostics(object)

  class(out) <- "summary.ShrinkageTrees"
  out
}

#' Print a ShrinkageTrees model summary
#'
#' Displays a detailed summary of a \code{ShrinkageTrees} model, including
#' model specification, posterior sigma, prediction summaries, variable
#' importance, and MCMC diagnostics.
#'
#' @param x A \code{summary.ShrinkageTrees} object.
#' @param n_vi Maximum number of variables to display in the variable
#'   importance table. Default \code{10}.
#' @param ... Currently unused.
#' @return Invisibly returns \code{x}.
#' @seealso \code{\link{summary.ShrinkageTrees}}
#' @export
print.summary.ShrinkageTrees <- function(x, n_vi = 10, ...) {

  cat("\n")
  cat("ShrinkageTrees model summary\n")
  cat("============================\n")
  cat("Call: "); print(x$call)
  cat("\n")
  cat("Outcome: ", .outcome_label(x$outcome_type, x$timescale),
      " | Prior: ", x$prior$prior_type_user,
      " | Trees: ", x$mcmc$number_of_trees, "\n", sep = "")
  n_ch <- x$mcmc$n_chains
  draws_str <- if (!is.null(n_ch) && n_ch > 1)
    paste0(x$mcmc$N_post, " x ", n_ch, " chains (burn-in ", x$mcmc$N_burn, ")")
  else
    paste0(x$mcmc$N_post, " (burn-in ", x$mcmc$N_burn, ")")
  cat("Data:    n = ", x$data_info$n_train,
      ", p = ", x$data_info$p_features,
      " | Draws: ", draws_str, "\n", sep = "")

  if (!is.null(x$sigma)) {
    cat("\nPosterior sigma:\n")
    cat("  Mean: ", round(x$sigma["mean"],  3),
        "  SD: ",   round(x$sigma["sd"],    3),
        "  95% CI: [", round(x$sigma["lower"], 3),
        ", ",          round(x$sigma["upper"], 3), "]\n", sep = "")
  }

  cat("\nPredictions (posterior mean):\n")
  fmt_pred <- function(p) {
    paste0("mean = ", round(p["mean"], 3),
           ", sd = ",  round(p["sd"],   3),
           ", range = [", round(p["min"], 3), ", ", round(p["max"], 3), "]")
  }
  cat("  Train: ", fmt_pred(x$predictions$train), "\n", sep = "")
  if (!is.null(x$predictions$test))
    cat("  Test:  ", fmt_pred(x$predictions$test), "\n", sep = "")

  if (!is.null(x$variable_importance)) {
    cat("\nVariable importance (posterior inclusion probability):\n")
    vi <- head(x$variable_importance, n_vi)
    cat(" ", paste(names(vi), round(vi, 3), sep = ": ", collapse = "   "), "\n")
  }

  if (!is.null(x$chains)) {
    cat("\nMCMC acceptance ratio (per chain): ",
        paste(round(x$chains$acceptance_ratios, 3), collapse = ", "), "\n",
        sep = "")
  } else if (!is.null(x$acceptance_ratio)) {
    cat("\nMCMC acceptance ratio: ", round(mean(x$acceptance_ratio), 3), "\n",
        sep = "")
  }

  if (!is.null(x$diagnostics)) {
    cat("\nConvergence diagnostics (coda):\n")
    if (!is.null(x$diagnostics$rhat))
      cat("  Gelman-Rubin R-hat: ",
          paste(names(x$diagnostics$rhat), round(x$diagnostics$rhat, 3),
                sep = " = ", collapse = ", "), "\n", sep = "")
    cat("  Effective sample size: ",
        paste(names(x$diagnostics$ess), round(x$diagnostics$ess, 0),
              sep = " = ", collapse = ", "), "\n", sep = "")
  }

  cat("\n")
  invisible(x)
}

#' Summarise a CausalShrinkageForest model
#'
#' Returns an inspectable list with treatment effect estimates, prognostic
#' function summaries, posterior sigma, variable importance for each forest,
#' and MCMC diagnostics.
#'
#' @param object A fitted \code{CausalShrinkageForest} model object.
#' @param ... Currently unused.
#' @return A \code{summary.CausalShrinkageForest} object with elements:
#'   \describe{
#'     \item{call}{The original model call.}
#'     \item{outcome_type}{Outcome type.}
#'     \item{timescale}{Timescale for survival outcomes.}
#'     \item{prior}{Prior specification for control and treatment forests.}
#'     \item{mcmc}{MCMC settings.}
#'     \item{data_info}{Training and test data dimensions.}
#'     \item{treatment_effect}{List with \code{ate} (posterior mean ATE),
#'       \code{cate_sd} (SD of individual CATEs), and optionally
#'       \code{ate_lower} and \code{ate_upper} (95\% CI; requires
#'       \code{store_posterior_sample = TRUE}).}
#'     \item{prognostic}{Summary of the prognostic function (mean, SD, range).}
#'     \item{sigma}{Named vector with posterior mean, SD, and 95\% CI of sigma
#'       (if estimated).}
#'     \item{variable_importance_control}{Variable importance for the control
#'       forest (if available).}
#'     \item{variable_importance_treat}{Variable importance for the treatment
#'       forest (if available).}
#'     \item{acceptance_ratios}{List with acceptance ratios for each forest.}
#'   }
#' @seealso \code{\link{print.summary.CausalShrinkageForest}},
#'   \code{\link{CausalHorseForest}}, \code{\link{CausalShrinkageForest}}
#' @export
summary.CausalShrinkageForest <- function(object, ...) {


  out <- list(
    call         = object$call,
    outcome_type = object$outcome_type,
    timescale    = object$timescale,
    prior        = object$prior,
    mcmc         = object$mcmc,
    data_info    = object$data_info
  )

  tau <- object$train_predictions_treat
  te  <- list(ate = mean(tau), cate_sd = sd(tau))

  if (!is.null(object$train_predictions_sample_treat)) {
    ate_samples  <- rowMeans(object$train_predictions_sample_treat)
    te$ate_lower <- unname(quantile(ate_samples, 0.025))
    te$ate_upper <- unname(quantile(ate_samples, 0.975))
  }
  out$treatment_effect <- te

  out$prognostic <- .pred_summary(object$train_predictions_control)

  if (!object$preprocess$sigma_known && !is.null(object$sigma))
    out$sigma <- .sigma_summary(object$sigma)

  if (isTRUE(object$prior$control$dirichlet) && !is.null(object$split_probs_control))
    out$variable_importance_control <- .vi_from_probs(object$split_probs_control)
  else if (!is.null(object$split_counts_control))
    out$variable_importance_control <- .vi_from_counts(object$split_counts_control)

  if (isTRUE(object$prior$treat$dirichlet) && !is.null(object$split_probs_treat))
    out$variable_importance_treat <- .vi_from_probs(object$split_probs_treat)
  else if (!is.null(object$split_counts_treat))
    out$variable_importance_treat <- .vi_from_counts(object$split_counts_treat)

  out$acceptance_ratios <- list(
    control = object$acceptance_ratio_control,
    treat   = object$acceptance_ratio_treat
  )
  if (!is.null(object$chains))
    out$acceptance_ratios$per_chain <- object$chains

  class(out) <- "summary.CausalShrinkageForest"
  out
}

#' Print a CausalShrinkageForest model summary
#'
#' Displays a detailed summary of a \code{CausalShrinkageForest} model,
#' including model specification, treatment effect estimates, prognostic
#' function, posterior sigma, variable importance for each forest, and MCMC
#' diagnostics.
#'
#' @param x A \code{summary.CausalShrinkageForest} object.
#' @param n_vi Maximum number of variables to display per variable importance
#'   table. Default \code{10}.
#' @param ... Currently unused.
#' @return Invisibly returns \code{x}.
#' @seealso \code{\link{summary.CausalShrinkageForest}}
#' @export
print.summary.CausalShrinkageForest <- function(x, n_vi = 10, ...) {
  
  cat("\n")
  cat("CausalShrinkageForest model summary\n")
  cat("=====================================\n")
  cat("Call: "); print(x$call)
  cat("\n")
  cat("Outcome: ", .outcome_label(x$outcome_type, x$timescale), "\n", sep = "")
  cat("Prior:   control = ", x$prior$control$prior_type_user,
      ", treatment = ", x$prior$treat$prior_type_user, "\n", sep = "")
  cat("Trees:   control = ", x$mcmc$number_of_trees_control,
      ", treatment = ", x$mcmc$number_of_trees_treat, "\n", sep = "")
  n_ch <- x$mcmc$n_chains
  draws_str <- if (!is.null(n_ch) && n_ch > 1)
    paste0(x$mcmc$N_post, " x ", n_ch, " chains (burn-in ", x$mcmc$N_burn, ")")
  else
    paste0(x$mcmc$N_post, " (burn-in ", x$mcmc$N_burn, ")")
  cat("Data:    n = ", x$data_info$n_train,
      ", p_control = ", x$data_info$p_control,
      ", p_treat = ", x$data_info$p_treat,
      " | Draws: ", draws_str, "\n", sep = "")

  cat("\nTreatment effect (CATE):\n")
  te <- x$treatment_effect
  if (!is.null(te$ate_lower)) {
    cat("  ATE:     ", round(te$ate, 4),
        "  95% CI: [", round(te$ate_lower, 4), ", ",
                       round(te$ate_upper, 4), "]\n", sep = "")
  } else {
    cat("  ATE:     ", round(te$ate, 4),
        "  (no CI -- refit with store_posterior_sample = TRUE)\n", sep = "")
  }
  cat("  CATE SD: ", round(te$cate_sd, 4), "\n", sep = "")

  cat("\nPrognostic function (mu):\n")
  p <- x$prognostic
  cat("  Mean: ", round(p["mean"], 3),
      "  SD: ",   round(p["sd"],   3),
      "  Range: [", round(p["min"], 3), ", ", round(p["max"], 3), "]\n",
      sep = "")

  if (!is.null(x$sigma)) {
    cat("\nPosterior sigma:\n")
    cat("  Mean: ", round(x$sigma["mean"],  3),
        "  SD: ",   round(x$sigma["sd"],    3),
        "  95% CI: [", round(x$sigma["lower"], 3),
        ", ",          round(x$sigma["upper"], 3), "]\n", sep = "")
  }

  fmt_vi <- function(vi) {
    vi <- head(vi, n_vi)
    paste(names(vi), round(vi, 3), sep = ": ", collapse = "   ")
  }
  if (!is.null(x$variable_importance_control)) {
    cat("\nVariable importance - control forest (posterior inclusion probability):\n")
    cat(" ", fmt_vi(x$variable_importance_control), "\n")
  }
  if (!is.null(x$variable_importance_treat)) {
    cat("\nVariable importance - treatment forest (posterior inclusion probability):\n")
    cat(" ", fmt_vi(x$variable_importance_treat), "\n")
  }

  if (!is.null(x$acceptance_ratios$per_chain)) {
    pc <- x$acceptance_ratios$per_chain
    cat("\nMCMC acceptance ratios (per chain):\n")
    cat("  control:   ",
        paste(round(pc$acceptance_ratios_control, 3), collapse = ", "), "\n",
        sep = "")
    cat("  treatment: ",
        paste(round(pc$acceptance_ratios_treat,   3), collapse = ", "), "\n",
        sep = "")
  } else {
    cat("\nMCMC acceptance ratios:",
        " control = ", round(mean(x$acceptance_ratios$control), 3),
        ", treatment = ", round(mean(x$acceptance_ratios$treat), 3), "\n",
        sep = "")
  }

  cat("\n")
  invisible(x)
}

#' Print a CausalShrinkageForest model
#'
#' Displays a concise summary of a fitted \code{CausalShrinkageForest} model
#' with per-forest columns for priors, tree counts, feature counts, and MCMC
#' acceptance ratios.
#'
#' @param x A fitted \code{CausalShrinkageForest} model object.
#' @param ... Currently unused.
#' @return Invisibly returns \code{x}.
#' @seealso \code{\link{summary.CausalShrinkageForest}},
#'   \code{\link{CausalHorseForest}}, \code{\link{CausalShrinkageForest}}
#' @export
print.CausalShrinkageForest <- function(x, ...) {

  cat("\n")
  cat("CausalShrinkageForest model\n")
  cat("---------------------------\n")

  lbl <- function(s) sprintf("%-22s", s)

  # -- General info ----------------------------------------------------------
  cat(lbl("Outcome type:"), .outcome_label(x$outcome_type, x$timescale),
      "\n", sep = "")
  cat(lbl("Training size (n):"), x$data_info$n_train, "\n", sep = "")
  n_ch <- x$mcmc$n_chains
  if (!is.null(n_ch) && n_ch > 1) {
    cat(lbl("Chains:"),          n_ch,          "\n", sep = "")
    cat(lbl("Draws per chain:"), x$mcmc$N_post,
        " (burn-in ", x$mcmc$N_burn, ")\n", sep = "")
  } else {
    cat(lbl("Posterior draws:"), x$mcmc$N_post,
        " (burn-in ", x$mcmc$N_burn, ")\n", sep = "")
  }

  if (!x$preprocess$sigma_known && !is.null(x$sigma))
    cat(lbl("Posterior mean sigma:"), round(mean(x$sigma), 3), "\n", sep = "")

  # -- Per-forest info (side by side) ----------------------------------------
  col  <- function(s) sprintf("%-20s", s)
  hdr  <- function(s) sprintf("%-20s", s)

  cat("\n")
  cat(sprintf("%-22s", ""), hdr("Control"), hdr("Treatment"), "\n", sep = "")
  cat(sprintf("%-22s", ""), strrep("-", 19), " ", strrep("-", 19), "\n",
      sep = "")

  cat(lbl("Prior:"),
      col(x$prior$control$prior_type_user),
      col(x$prior$treat$prior_type_user), "\n", sep = "")
  cat(lbl("Number of trees:"),
      col(x$mcmc$number_of_trees_control),
      col(x$mcmc$number_of_trees_treat), "\n", sep = "")
  cat(lbl("Number of features:"),
      col(x$data_info$p_control),
      col(x$data_info$p_treat), "\n", sep = "")

  if (!is.null(x$chains)) {
    acc_c <- paste(round(x$chains$acceptance_ratios_control, 3), collapse = ", ")
    acc_t <- paste(round(x$chains$acceptance_ratios_treat,   3), collapse = ", ")
    cat(lbl("Acceptance ratio:"), col(acc_c), col(acc_t), "(per chain)\n",
        sep = "")
  } else if (!is.null(x$acceptance_ratio_control) ||
             !is.null(x$acceptance_ratio_treat)) {
    acc_c <- if (!is.null(x$acceptance_ratio_control))
               round(mean(x$acceptance_ratio_control), 3) else "-"
    acc_t <- if (!is.null(x$acceptance_ratio_treat))
               round(mean(x$acceptance_ratio_treat), 3) else "-"
    cat(lbl("Acceptance ratio:"), col(acc_c), col(acc_t), "\n", sep = "")
  }

  cat("\n")
  invisible(x)
}

# ── as.mcmc.list.ShrinkageTrees ───────────────────────────────────────────────

#' Convert MCMC output to a coda mcmc.list
#'
#' Converts the posterior draws stored in a \code{ShrinkageTrees} object into
#' a \code{\link[coda]{mcmc.list}} for use with the \pkg{coda} package's
#' convergence diagnostics (Gelman--Rubin \eqn{\hat{R}}, effective sample
#' size, Geweke test, etc.).
#'
#' @param x A fitted \code{ShrinkageTrees} object.
#' @param ... Currently unused.
#' @return A \code{\link[coda]{mcmc.list}} object.
#'   Each chain is an \code{\link[coda]{mcmc}} object whose columns include:
#'   \describe{
#'     \item{sigma}{Posterior draws of the residual standard deviation
#'       (continuous and survival outcomes only).}
#'   }
#' @details
#' Requires the suggested package \pkg{coda}.
#' For single-chain fits the returned object contains one chain.
#' @seealso \code{\link{summary.ShrinkageTrees}} which reports R-hat and ESS
#'   automatically when coda is available.
#' @examples
#' \donttest{
#' fit <- HorseTrees(y = rnorm(50), X_train = matrix(rnorm(250), 50, 5),
#'                   N_post = 200, N_burn = 100, n_chains = 2)
#' if (requireNamespace("coda", quietly = TRUE)) {
#'   mcmc_obj <- as.mcmc.list(fit)
#'   coda::gelman.diag(mcmc_obj)
#'   coda::effectiveSize(mcmc_obj)
#' }
#' }
#' @exportS3Method coda::as.mcmc.list
as.mcmc.list.ShrinkageTrees <- function(x, ...) {

  if (!requireNamespace("coda", quietly = TRUE))
    stop("Package 'coda' is needed for as.mcmc.list(). ",
         "Install it with: install.packages('coda')", call. = FALSE)

  N_post   <- x$mcmc$N_post
  n_chains <- if (!is.null(x$mcmc$n_chains)) x$mcmc$n_chains else 1L

  # Build a matrix of parameters for each chain
  chain_list <- vector("list", n_chains)
  for (ch in seq_len(n_chains)) {
    idx <- ((ch - 1L) * N_post + 1L):(ch * N_post)
    params <- list()

    if (!is.null(x$sigma))
      params$sigma <- x$sigma[idx]

    mat <- do.call(cbind, params)
    if (!is.matrix(mat)) mat <- matrix(mat, ncol = 1, dimnames = list(NULL, names(params)))
    chain_list[[ch]] <- coda::mcmc(mat, start = 1, end = N_post, thin = 1)
  }

  coda::mcmc.list(chain_list)
}

# ── MCMC diagnostics helpers ─────────────────────────────────────────────────

# Compute Gelman-Rubin R-hat and bulk/tail ESS using coda, if available.
# Returns NULL when coda is not installed or only one chain was run.
.mcmc_diagnostics <- function(object) {
  if (!requireNamespace("coda", quietly = TRUE)) return(NULL)
  n_chains <- if (!is.null(object$mcmc$n_chains)) object$mcmc$n_chains else 1L
  if (is.null(object$sigma)) return(NULL)

  mcmc_obj <- as.mcmc.list.ShrinkageTrees(object)
  ess <- coda::effectiveSize(mcmc_obj)

  rhat <- if (n_chains >= 2L) {
    gr <- coda::gelman.diag(mcmc_obj, autoburnin = FALSE, multivariate = FALSE)
    gr$psrf[, "Point est."]
  }

  list(ess = ess, rhat = rhat)
}

# ── predict.CausalShrinkageForest ─────────────────────────────────────────────

#' Posterior predictive inference for a CausalShrinkageForest model
#'
#' Re-runs the MCMC sampler on new covariate data using the stored training
#' data and hyperparameters, returning posterior mean predictions and credible
#' interval bounds for three quantities: the \strong{prognostic function}
#' (control-forest prediction \eqn{\mu(X)}), the \strong{Conditional Average
#' Treatment Effect} (CATE, \eqn{\tau(X)}), and the \strong{total predicted
#' outcome} (\eqn{\mu(X) + \tau(X)}).
#'
#' @details
#' The causal forest decomposes the expected outcome as
#' \deqn{E[Y \mid X] = \mu(X) + \tau(X) \cdot W,}
#' where \eqn{\mu(X)} is the prognostic function (control forest),
#' \eqn{\tau(X)} is the CATE (treatment forest), and \eqn{W} is the
#' treatment indicator.
#'
#' For \strong{continuous} outcomes and survival with
#' \code{timescale = "log"}, all three components are on the response scale:
#' \code{prognostic} and \code{total} include the intercept shift
#' (\eqn{+ \bar{y}}), while \code{cate} is the pure additive treatment
#' effect with no intercept.
#'
#' For \strong{survival with \code{timescale = "time"}}, predictions are
#' back-transformed to the original time scale:
#' \itemize{
#'   \item \code{prognostic}: posterior expected baseline survival time
#'     \eqn{E[\exp(\mu(X))]}.
#'   \item \code{cate}: multiplicative effect on survival time
#'     \eqn{\exp(\tau(X))}; a value greater than 1 means treatment prolongs
#'     survival.
#'   \item \code{total}: posterior expected survival time under the observed
#'     treatment \eqn{E[\exp(\mu(X) + \tau(X))]}.
#' }
#'
#' @param object A fitted \code{CausalShrinkageForest} model object.
#' @param newdata_control A matrix of new covariates for the control forest,
#'   with the same number of columns as \code{X_train_control} at fit time.
#' @param newdata_treat A matrix of new covariates for the treatment forest,
#'   with the same number of columns as \code{X_train_treat} at fit time.
#'   Must have the same number of rows as \code{newdata_control}.
#' @param level Credible interval width. Default \code{0.95}.
#' @param ... Currently unused.
#' @return A \code{CausalShrinkageForestPrediction} object with elements:
#'   \describe{
#'     \item{prognostic}{List with \code{mean}, \code{lower}, \code{upper}:
#'       posterior summaries of the prognostic function
#'       \eqn{\mu(X_{\text{new}})}.}
#'     \item{cate}{List with \code{mean}, \code{lower}, \code{upper}:
#'       posterior summaries of the CATE \eqn{\tau(X_{\text{new}})}.}
#'     \item{total}{List with \code{mean}, \code{lower}, \code{upper}:
#'       posterior summaries of the total outcome
#'       \eqn{\mu(X_{\text{new}}) + \tau(X_{\text{new}})}.}
#'     \item{n}{Number of test observations.}
#'     \item{level}{Credible level used.}
#'     \item{outcome_type}{Outcome type inherited from the fitted model.}
#'     \item{timescale}{Timescale inherited from the fitted model.}
#'   }
#' @seealso \code{\link{CausalHorseForest}}, \code{\link{CausalShrinkageForest}},
#'   \code{\link{print.CausalShrinkageForestPrediction}},
#'   \code{\link{summary.CausalShrinkageForestPrediction}}
#' @export
predict.CausalShrinkageForest <- function(object, newdata_control, newdata_treat,
                                          level = 0.95, ...) {

  if (!is.matrix(newdata_control)) newdata_control <- as.matrix(newdata_control)
  if (!is.matrix(newdata_treat))   newdata_treat   <- as.matrix(newdata_treat)

  p_control <- object$data_info$p_control
  p_treat   <- object$data_info$p_treat

  if (ncol(newdata_control) != p_control)
    stop("newdata_control has ", ncol(newdata_control),
         " columns but the model expects ", p_control, ".")
  if (ncol(newdata_treat) != p_treat)
    stop("newdata_treat has ", ncol(newdata_treat),
         " columns but the model expects ", p_treat, ".")

  n_new <- nrow(newdata_control)
  if (nrow(newdata_treat) != n_new)
    stop("newdata_control and newdata_treat must have the same number of rows.")

  n_train  <- object$data_info$n_train
  pre      <- object$preprocess
  args     <- object$args
  survival <- object$outcome_type %in% c("right-censored", "interval-censored")

  # Reconstruct standardised training response (inverse of fit-time transforms)
  y      <- as.numeric(object$data$y_train)
  status <- object$data$status_train
  if (survival && object$timescale == "time") y <- log(y)
  y <- (y - pre$y_mean) / pre$sigma_hat
  if (!survival) status <- rep(1L, n_train)

  # Format matrices for C++ (row-major: each observation is a contiguous block)
  X_train_control <- as.numeric(t(object$data$X_train_control))
  X_train_treat   <- as.numeric(t(object$data$X_train_treat))
  X_test_control  <- as.numeric(t(newdata_control))
  X_test_treat    <- as.numeric(t(newdata_treat))
  trt_train       <- as.integer(object$data$treatment_indicator_train)
  trt_test        <- as.integer(rep(1L, n_new))  # placeholder; not used in predictions

  # Treatment coding and propensity for C++
  tc <- if (!is.null(object$treatment_coding)) object$treatment_coding else "centered"
  prop_train_cpp <- if (!is.null(object$data$propensity_train)) {
    as.numeric(object$data$propensity_train)
  } else {
    numeric(n_train)
  }
  prop_test_cpp <- rep(0.5, n_new)

  alpha <- (1 - level) / 2

  # Reconstruct interval-censored boundaries if applicable
  if (object$outcome_type == "interval-censored") {
    # For causal functions, stored raw values are on original scale
    lt_raw <- object$data$left_time_train
    rt_raw <- object$data$right_time_train
    if (object$timescale == "time") {
      lt_raw <- log(lt_raw)
      rt_raw <- log(rt_raw)
    }
    left_std_c  <- (lt_raw - pre$y_mean) / pre$sigma_hat
    right_std_c <- (rt_raw - pre$y_mean) / pre$sigma_hat
    ic_ind_c    <- object$data$ic_indicator_train
  } else {
    left_std_c  <- numeric(n_train)
    right_std_c <- y + 0
    ic_ind_c    <- numeric(n_train)
  }

  fit <- CausalHorseForest_cpp(
    nSEXP                              = n_train,
    p_treatSEXP                        = p_treat,
    p_controlSEXP                      = p_control,
    X_train_treatSEXP                  = X_train_treat,
    X_train_controlSEXP                = X_train_control,
    ySEXP                              = y,
    status_indicatorSEXP               = status,
    is_survivalSEXP                    = survival,
    observed_left_timeSEXP             = left_std_c,
    observed_right_timeSEXP            = right_std_c,
    interval_censoring_indicatorSEXP   = ic_ind_c,
    treatment_indicatorSEXP            = trt_train,
    n_testSEXP                         = n_new,
    X_test_controlSEXP                 = X_test_control,
    X_test_treatSEXP                   = X_test_treat,
    treatment_indicator_testSEXP       = trt_test,
    no_trees_treatSEXP                 = object$mcmc$number_of_trees_treat,
    power_treatSEXP                    = args$treat$power,
    base_treatSEXP                     = args$treat$base,
    p_grow_treatSEXP                   = args$p_grow,
    p_prune_treatSEXP                  = args$p_prune,
    omega_treatSEXP                    = args$treat$omega,
    prior_type_treatSEXP               = object$prior$treat$prior_type_cpp,
    param1_treatSEXP                   = args$treat$param1,
    param2_treatSEXP                   = args$treat$param2,
    reversible_treatSEXP               = args$treat$reversible,
    dirichlet_bool_treatSEXP           = isTRUE(object$prior$treat$dirichlet),
    a_dirichlet_treatSEXP              = args$treat$a_dirichlet,
    b_dirichlet_treatSEXP              = args$treat$b_dirichlet,
    rho_dirichlet_treatSEXP            = args$treat$rho_dirichlet,
    no_trees_controlSEXP               = object$mcmc$number_of_trees_control,
    power_controlSEXP                  = args$control$power,
    base_controlSEXP                   = args$control$base,
    p_grow_controlSEXP                 = args$p_grow,
    p_prune_controlSEXP                = args$p_prune,
    omega_controlSEXP                  = args$control$omega,
    prior_type_controlSEXP             = object$prior$control$prior_type_cpp,
    param1_controlSEXP                 = args$control$param1,
    param2_controlSEXP                 = args$control$param2,
    reversible_controlSEXP             = args$control$reversible,
    dirichlet_bool_controlSEXP         = isTRUE(object$prior$control$dirichlet),
    a_dirichlet_controlSEXP            = args$control$a_dirichlet,
    b_dirichlet_controlSEXP            = args$control$b_dirichlet,
    rho_dirichlet_controlSEXP          = args$control$rho_dirichlet,
    sigma_knownSEXP                    = TRUE,
    sigmaSEXP                          = pre$sigma_hat,
    lambdaSEXP                         = args$lambda,
    nuSEXP                             = args$nu,
    N_postSEXP                         = object$mcmc$N_post,
    N_burnSEXP                         = object$mcmc$N_burn,
    delayed_proposalSEXP               = args$delayed_proposal,
    store_posterior_sample_controlSEXP = TRUE,
    store_posterior_sample_treatSEXP   = TRUE,
    verboseSEXP                        = FALSE,
    treatment_codingSEXP               = tc,
    propensity_trainSEXP               = prop_train_cpp,
    propensity_testSEXP                = prop_test_cpp
  )

  # C++ does not return a combined total sample; reconstruct from components.
  control_std <- fit$test_predictions_sample_control  # N_post x n_new
  treat_std   <- fit$test_predictions_sample_treat    # N_post x n_new
  total_std   <- control_std + treat_std

  sigma_hat <- pre$sigma_hat
  y_mean    <- pre$y_mean

  if (survival && object$timescale == "time") {
    # Back-transform to original time scale
    prognostic_sample <- exp(control_std * sigma_hat + y_mean)
    cate_sample       <- exp(treat_std   * sigma_hat)          # multiplicative ratio
    total_sample      <- exp(total_std   * sigma_hat + y_mean)
  } else {
    # Continuous or log-scale survival: additive de-standardisation
    prognostic_sample <- control_std * sigma_hat + y_mean
    cate_sample       <- treat_std   * sigma_hat               # no intercept shift
    total_sample      <- total_std   * sigma_hat + y_mean
  }

  ci <- function(mat) list(
    mean  = colMeans(mat),
    lower = apply(mat, 2, quantile, alpha),
    upper = apply(mat, 2, quantile, 1 - alpha)
  )

  out <- list(
    prognostic   = ci(prognostic_sample),
    cate         = ci(cate_sample),
    total        = ci(total_sample),
    n            = n_new,
    level        = level,
    outcome_type = object$outcome_type,
    timescale    = object$timescale
  )
  class(out) <- "CausalShrinkageForestPrediction"
  out
}

# ── print / summary for CausalShrinkageForestPrediction ───────────────────────

#' Print a CausalShrinkageForestPrediction object
#'
#' Displays a formatted table of posterior mean predictions and credible
#' interval bounds for the first \code{n_head} observations, with separate
#' sections for the prognostic function \eqn{\mu(X)}, the CATE \eqn{\tau(X)},
#' and the total outcome \eqn{\mu(X) + \tau(X)}.
#'
#' @param x A \code{CausalShrinkageForestPrediction} object.
#' @param n_head Number of observations to display per section. Default \code{6}.
#' @param digits Number of decimal places. Default \code{3}.
#' @param ... Currently unused.
#' @return Invisibly returns \code{x}.
#' @seealso \code{\link{predict.CausalShrinkageForest}},
#'   \code{\link{summary.CausalShrinkageForestPrediction}}
#' @export
print.CausalShrinkageForestPrediction <- function(x, n_head = 6, digits = 3, ...) {

  ci_pct     <- paste0(round(x$level * 100), "%")
  time_scale <- x$outcome_type %in% c("right-censored", "interval-censored") && x$timescale == "time"

  cat("\n")
  cat("CausalShrinkageForest predictions\n")
  cat("----------------------------------\n")
  lbl <- function(s) sprintf("%-22s", s)
  cat(lbl("Observations:"),      x$n,            "\n", sep = "")
  cat(lbl("Credible interval:"), ci_pct,         "\n", sep = "")
  cat(lbl("Outcome type:"),      x$outcome_type, "\n", sep = "")

  n_show <- min(n_head, x$n)
  hdr    <- sprintf("  %5s  %8s  %8s  %8s\n", "", "mean", "lower", "upper")
  sep    <- sprintf("  %5s  %8s  %8s  %8s\n", "",
                    strrep("-", 8), strrep("-", 8), strrep("-", 8))
  row_fn <- function(i, lst)
    sprintf("  [%3d]  %8.*f  %8.*f  %8.*f\n",
            i, digits, lst$mean[i], digits, lst$lower[i], digits, lst$upper[i])

  print_block <- function(lst, label) {
    cat("\n", label, ":\n", sep = "")
    cat(hdr); cat(sep)
    for (i in seq_len(n_show)) cat(row_fn(i, lst))
    if (x$n > n_head) cat("  ... (", x$n - n_head, " more)\n", sep = "")
  }

  print_block(x$prognostic,
              if (time_scale) "Prognostic (E[T0])" else "Prognostic (mu)")
  print_block(x$cate,
              if (time_scale) "CATE (time ratio, exp(tau))" else "CATE (tau)")
  print_block(x$total,
              if (time_scale) "Total (E[T])" else "Total (mu + tau)")

  cat("\n")
  invisible(x)
}

#' Summarise a CausalShrinkageForestPrediction object
#'
#' Returns distributional summaries (min, Q1, median, max) of the posterior
#' mean predictions and credible interval bounds across all test observations,
#' separately for the prognostic function, CATE, and total outcome.
#'
#' @param object A \code{CausalShrinkageForestPrediction} object.
#' @param ... Currently unused.
#' @return A \code{summary.CausalShrinkageForestPrediction} object.
#' @seealso \code{\link{predict.CausalShrinkageForest}},
#'   \code{\link{print.summary.CausalShrinkageForestPrediction}}
#' @export
summary.CausalShrinkageForestPrediction <- function(object, ...) {
  summ_pred <- function(lst) list(
    mean  = quantile(lst$mean,  c(0, 0.25, 0.5, 1)),
    lower = quantile(lst$lower, c(0, 0.25, 0.5, 1)),
    upper = quantile(lst$upper, c(0, 0.25, 0.5, 1))
  )
  out <- list(
    n            = object$n,
    level        = object$level,
    outcome_type = object$outcome_type,
    timescale    = object$timescale,
    prognostic   = summ_pred(object$prognostic),
    cate         = summ_pred(object$cate),
    total        = summ_pred(object$total)
  )
  class(out) <- "summary.CausalShrinkageForestPrediction"
  out
}

#' Print a CausalShrinkageForestPrediction summary
#'
#' Displays distributional summaries (min, Q1, median, max) of the posterior
#' mean predictions and credible interval bounds, separately for the prognostic
#' function, CATE, and total outcome.
#'
#' @param x A \code{summary.CausalShrinkageForestPrediction} object.
#' @param digits Number of decimal places. Default \code{3}.
#' @param ... Currently unused.
#' @return Invisibly returns \code{x}.
#' @seealso \code{\link{summary.CausalShrinkageForestPrediction}}
#' @export
print.summary.CausalShrinkageForestPrediction <- function(x, digits = 3, ...) {

  ci_pct     <- paste0(round(x$level * 100), "%")
  time_scale <- x$outcome_type %in% c("right-censored", "interval-censored") && x$timescale == "time"

  cat("\n")
  cat("CausalShrinkageForest prediction summary\n")
  cat("-----------------------------------------\n")
  lbl <- function(s) sprintf("%-22s", s)
  cat(lbl("Observations:"),      x$n,    "\n", sep = "")
  cat(lbl("Credible interval:"), ci_pct, "\n", sep = "")

  hdr <- sprintf("  %-8s  %8s  %8s  %8s\n", "", "mean", "lower", "upper")
  sep <- sprintf("  %-8s  %8s  %8s  %8s\n", "",
                 strrep("-", 8), strrep("-", 8), strrep("-", 8))

  print_block <- function(lst, label) {
    cat("\n", label, ":\n", sep = "")
    cat(hdr); cat(sep)
    nms <- c("Min.", "Q1", "Median", "Max.")
    for (i in seq_along(nms))
      cat(sprintf("  %-8s  %8.*f  %8.*f  %8.*f\n",
                  nms[i], digits, lst$mean[i],
                  digits, lst$lower[i], digits, lst$upper[i]))
  }

  print_block(x$prognostic,
              if (time_scale) "Prognostic (E[T0])" else "Prognostic (mu)")
  print_block(x$cate,
              if (time_scale) "CATE (time ratio, exp(tau))" else "CATE (tau)")
  print_block(x$total,
              if (time_scale) "Total (E[T])" else "Total (mu + tau)")

  cat("\n")
  invisible(x)
}
