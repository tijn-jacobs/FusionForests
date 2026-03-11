#' FusionForest
#'
#' Bayesian data-fusion model combining an RCT and an observational study to
#' estimate heterogeneous treatment effects.  Three BART ensembles are fitted
#' jointly under an AFT decomposition:
#' \deqn{\log(T) = m_0(X,S) + A\,\tau(X) + A\,(1-S)\,c(X) + \sigma\varepsilon}
#' where \eqn{m_0} is the prognostic function, \eqn{\tau} is the CATE,
#' \eqn{c} is the confounding function (active only in the OS treated arm),
#' and \eqn{\eta = 0} (sources share a common baseline; no commensurate shift).
#'
#' @param y Numeric vector of outcomes (survival times or continuous responses).
#' @param status Integer vector of event indicators (\code{1} = event observed,
#'   \code{0} = censored).  Required when \code{outcome_type = "right-censored"}.
#' @param X_train_control Numeric matrix of covariates for the prognostic
#'   (\eqn{m_0}) and deconfounding (\eqn{c}) forests.  One row per training
#'   observation.
#' @param X_train_treat Numeric matrix of covariates for the treatment-effect
#'   (\eqn{\tau}) forest.  Must have the same number of rows as
#'   \code{X_train_control}.
#' @param treatment_indicator_train Integer vector of treatment assignments
#'   (\code{1} = treated, \code{0} = control) for training observations.
#' @param source_indicator_train Integer vector of data-source labels
#'   (\code{1} = RCT, \code{0} = observational study) for training observations.
#' @param X_test_control,X_test_treat,X_test_deconf Optional test matrices.
#'   If omitted, predictions are returned for a single row at the column means
#'   of the training data.
#' @param treatment_indicator_test,source_indicator_test Optional integer vectors
#'   for the test set (same coding as the training-set equivalents).
#' @param outcome_type Character; either \code{"continuous"} or
#'   \code{"right-censored"}.
#' @param timescale Character; \code{"time"} (raw survival times, will be
#'   log-transformed internally) or \code{"log"} (already on log scale).
#' @param number_of_trees_control,number_of_trees_treat,number_of_trees_deconf
#'   Number of trees in each of the three BART ensembles.  Default 200.
#' @param power,base Tree topology prior parameters.  The probability that a
#'   node at depth \eqn{d} is non-terminal is
#'   \eqn{\texttt{base} / (1 + d)^{\texttt{power}}}.
#' @param p_grow,p_prune Probabilities of proposing a grow or prune move at
#'   each MCMC step.
#' @param nu Degrees of freedom for the inverse-chi-squared prior on
#'   \eqn{\sigma^2}.
#' @param q Quantile used to set the scale parameter \eqn{\lambda} of the
#'   error-variance prior.
#' @param sigma Optional fixed value for \eqn{\sigma}.  If \code{NULL} (default),
#'   \eqn{\sigma} is estimated from the data and updated each MCMC iteration.
#' @param N_post,N_burn Number of posterior and burn-in MCMC iterations.
#' @param store_posterior_sample Logical; if \code{TRUE}, the full
#'   \eqn{N_{\text{post}} \times n} posterior sample matrices are returned for
#'   all three component forests.
#' @param verbose Logical; print a progress bar and summary statistics.
#'
#' @return A named list with components:
#' \describe{
#'   \item{train_predictions, test_predictions}{Posterior mean of the total
#'     fitted values \eqn{\hat{y} = m_0 + (1-S)\eta + b[\tau + (1-S)c]}.}
#'   \item{train_predictions_control, test_predictions_control}{Posterior mean
#'     of the prognostic component \eqn{m_0(X)}.}
#'   \item{train_predictions_treat, test_predictions_treat}{Posterior mean of
#'     the CATE \eqn{\tau(X)}.}
#'   \item{train_predictions_deconf, test_predictions_deconf}{Posterior mean of
#'     the confounding function \eqn{c(X)} (OS rows only for training).}
#'   \item{sigma}{Posterior sample of \eqn{\sigma} (or the fixed value if
#'     \code{sigma} was supplied).}
#'   \item{acceptance_ratio_control, acceptance_ratio_treat,
#'     acceptance_ratio_deconf}{Tree-update acceptance rates.}
#'   \item{train_predictions_sample_control, ...}{Full posterior sample matrices
#'     (only present when \code{store_posterior_sample = TRUE}).}
#' }
#'
#' @importFrom Rcpp evalCpp
#' @useDynLib FusionForests, .registration = TRUE
#' @importFrom stats sd qchisq qnorm runif
#' @export
FusionForest <- function(
  y,
  status                    = NULL,
  X_train_control,
  X_train_treat,
  treatment_indicator_train,
  source_indicator_train,
  X_test_control            = NULL,
  X_test_treat              = NULL,
  X_test_deconf             = NULL,
  treatment_indicator_test  = NULL,
  source_indicator_test     = NULL,
  outcome_type              = "continuous",
  timescale                 = "time",
  number_of_trees_control   = 200,
  number_of_trees_treat     = 200,
  number_of_trees_deconf    = 200,
  power                     = 2.0,
  base                      = 0.95,
  p_grow                    = 0.4,
  p_prune                   = 0.4,
  nu                        = 3,
  q                         = 0.90,
  sigma                     = NULL,
  N_post                    = 5000,
  N_burn                    = 5000,
  store_posterior_sample    = FALSE,
  verbose                   = TRUE
) {

  ## ------------------------------------------------------------------
  ## Input validation
  ## ------------------------------------------------------------------

  allowed_types <- c("continuous", "right-censored")
  if (!outcome_type %in% allowed_types)
    stop("Invalid outcome_type. Choose 'continuous' or 'right-censored'.")

  if (outcome_type == "right-censored" && is.null(status))
    stop("outcome_type = 'right-censored' requires a 'status' vector.")

  if (outcome_type != "right-censored" && !is.null(status))
    warning("'status' is ignored for outcome_type = '", outcome_type, "'.")

  if (outcome_type == "right-censored" && timescale == "time" && any(y < 0))
    stop("Negative values in y with timescale = 'time': survival times must be non-negative.")

  ## ------------------------------------------------------------------
  ## Data preparation
  ## ------------------------------------------------------------------

  n_train   <- nrow(X_train_control)
  p_control <- ncol(X_train_control)
  p_treat   <- ncol(X_train_treat)

  if (nrow(X_train_control)          != length(y)) stop("X_train_control rows must match length(y).")
  if (nrow(X_train_treat)            != length(y)) stop("X_train_treat rows must match length(y).")
  if (length(treatment_indicator_train) != length(y)) stop("treatment_indicator_train must match length(y).")
  if (length(source_indicator_train)    != length(y)) stop("source_indicator_train must match length(y).")

  treatment_indicator_train <- as.integer(treatment_indicator_train)
  if (!all(source_indicator_train %in% c(0L, 1L)))
    stop("source_indicator_train must be 0 (OS) or 1 (RCT).")
  source_indicator_train <- as.integer(source_indicator_train)

  # OS subset for the deconfounding forest
  n_deconf <- sum(source_indicator_train == 0L)
  if (n_deconf <= 0L) stop("At least one observational-study (source = 0) row is required.")
  X_train_deconf <- X_train_control[source_indicator_train == 0L, , drop = FALSE]
  p_deconf       <- ncol(X_train_deconf)

  # Test data
  if (!is.null(X_test_control) && !is.null(X_test_treat)) {

    if (!is.matrix(X_test_control) || !is.matrix(X_test_treat))
      stop("X_test_control and X_test_treat must be matrices.")
    n_test <- nrow(X_test_control)
    if (nrow(X_test_treat) != n_test)
      stop("X_test_control and X_test_treat must have the same number of rows.")
    if (ncol(X_test_control) != p_control || ncol(X_test_treat) != p_treat)
      stop("Test matrix column counts must match training: p_control and p_treat.")

    source_indicator_test <- if (is.null(source_indicator_test)) {
      rep.int(1L, n_test)
    } else {
      s <- as.integer(source_indicator_test)
      if (length(s) != n_test) stop("source_indicator_test length must match number of test rows.")
      if (!all(s %in% c(0L, 1L))) stop("source_indicator_test must be 0 (OS) or 1 (RCT).")
      s
    }

    treatment_indicator_test <- if (is.null(treatment_indicator_test)) {
      rep.int(1L, n_test)
    } else {
      t <- as.integer(treatment_indicator_test)
      if (length(t) != n_test) stop("treatment_indicator_test length must match number of test rows.")
      t
    }

    X_test_deconf <- if (is.null(X_test_deconf)) {
      X_test_control
    } else {
      if (!is.matrix(X_test_deconf)) stop("X_test_deconf must be a matrix.")
      if (nrow(X_test_deconf) != n_test) stop("X_test_deconf rows must match X_test_control.")
      if (ncol(X_test_deconf) != p_deconf) stop("X_test_deconf must have ", p_deconf, " columns.")
      X_test_deconf
    }

    X_test_control <- as.numeric(t(X_test_control))
    X_test_treat   <- as.numeric(t(X_test_treat))
    X_test_deconf  <- as.numeric(t(X_test_deconf))

  } else {
    n_test                   <- 1L
    X_test_control           <- as.numeric(colMeans(X_train_control))
    X_test_treat             <- as.numeric(colMeans(X_train_treat))
    X_test_deconf            <- X_test_control
    treatment_indicator_test <- 1L
    source_indicator_test    <- 1L
  }

  # Flatten training matrices
  X_train_control <- as.numeric(t(X_train_control))
  X_train_treat   <- as.numeric(t(X_train_treat))
  X_train_deconf  <- as.numeric(t(X_train_deconf))

  # Scalar coercions
  N_post  <- as.integer(N_post)[1L]
  N_burn  <- as.integer(N_burn)[1L]
  power   <- as.numeric(power)[1L]
  base    <- as.numeric(base)[1L]
  p_grow  <- as.numeric(p_grow)[1L]
  p_prune <- as.numeric(p_prune)[1L]

  ## ------------------------------------------------------------------
  ## Outcome-specific setup and C++ call
  ## ------------------------------------------------------------------

  if (outcome_type == "right-censored") {

    y <- as.numeric(y)
    if (timescale == "time") y <- log(y)

    cens_inf  <- censored_info(y, status)
    y_mean    <- cens_inf$mu
    y         <- y - y_mean

    if (is.null(sigma)) {
      sigma_hat  <- cens_inf$sd
      sigma_known <- FALSE
    } else {
      sigma_hat  <- sigma
      sigma_known <- TRUE
    }

    y       <- y / sigma_hat
    survival <- TRUE
    qchi    <- qchisq(1.0 - q, nu)
    lambda  <- (sigma_hat^2 * qchi) / nu

    fit <- FusionForest_cpp(
      nSEXP                       = n_train,
      p_treatSEXP                 = p_treat,
      p_controlSEXP               = p_control,
      X_train_treatSEXP           = X_train_treat,
      X_train_controlSEXP         = X_train_control,
      ySEXP                       = y,
      status_indicatorSEXP        = status,
      is_survivalSEXP             = survival,
      treatment_indicatorSEXP     = treatment_indicator_train,
      source_indicatorSEXP        = source_indicator_train,
      n_testSEXP                  = n_test,
      X_test_controlSEXP          = X_test_control,
      X_test_treatSEXP            = X_test_treat,
      X_test_deconfSEXP           = X_test_deconf,
      treatment_indicator_testSEXP = treatment_indicator_test,
      source_indicator_testSEXP   = source_indicator_test,
      n_deconfSEXP                = n_deconf,
      p_deconfSEXP                = p_deconf,
      X_train_deconfSEXP          = X_train_deconf,
      no_trees_deconfSEXP         = number_of_trees_deconf,
      power_deconfSEXP            = power,
      base_deconfSEXP             = base,
      p_grow_deconfSEXP           = p_grow,
      p_prune_deconfSEXP          = p_prune,
      omega_deconfSEXP            = 0.5 / sqrt(number_of_trees_deconf),
      no_trees_treatSEXP          = number_of_trees_treat,
      power_treatSEXP             = power,
      base_treatSEXP              = base,
      p_grow_treatSEXP            = p_grow,
      p_prune_treatSEXP           = p_prune,
      omega_treatSEXP             = 0.5 / sqrt(number_of_trees_treat),
      no_trees_controlSEXP        = number_of_trees_control,
      power_controlSEXP           = power,
      base_controlSEXP            = base,
      p_grow_controlSEXP          = p_grow,
      p_prune_controlSEXP         = p_prune,
      omega_controlSEXP           = 0.5 / sqrt(number_of_trees_control),
      sigma_knownSEXP             = sigma_known,
      sigmaSEXP                   = sigma_hat,
      lambdaSEXP                  = lambda,
      nuSEXP                      = nu,
      N_postSEXP                  = N_post,
      N_burnSEXP                  = N_burn,
      store_posterior_sampleSEXP  = store_posterior_sample,
      verboseSEXP                 = verbose
    )

    # Back-transform predictions
    if (timescale == "time") {
      fit$train_predictions         <- exp(fit$train_predictions         * sigma_hat + y_mean)
      fit$test_predictions          <- exp(fit$test_predictions          * sigma_hat + y_mean)
      fit$train_predictions_control <- exp(fit$train_predictions_control * sigma_hat + y_mean)
      fit$test_predictions_control  <- exp(fit$test_predictions_control  * sigma_hat + y_mean)
      fit$train_predictions_treat   <- exp(fit$train_predictions_treat   * sigma_hat)
      fit$test_predictions_treat    <- exp(fit$test_predictions_treat    * sigma_hat)
      fit$train_predictions_deconf  <- exp(fit$train_predictions_deconf  * sigma_hat)
      fit$test_predictions_deconf   <- exp(fit$test_predictions_deconf   * sigma_hat)
      if (store_posterior_sample) {
        fit$train_predictions_sample_control <- exp(fit$train_predictions_sample_control * sigma_hat + y_mean)
        fit$test_predictions_sample_control  <- exp(fit$test_predictions_sample_control  * sigma_hat + y_mean)
        fit$train_predictions_sample_treat   <- exp(fit$train_predictions_sample_treat   * sigma_hat)
        fit$test_predictions_sample_treat    <- exp(fit$test_predictions_sample_treat    * sigma_hat)
        fit$train_predictions_sample_deconf  <- exp(fit$train_predictions_sample_deconf  * sigma_hat)
        fit$test_predictions_sample_deconf   <- exp(fit$test_predictions_sample_deconf   * sigma_hat)
      }
    } else {
      fit$train_predictions         <- fit$train_predictions         * sigma_hat + y_mean
      fit$test_predictions          <- fit$test_predictions          * sigma_hat + y_mean
      fit$train_predictions_control <- fit$train_predictions_control * sigma_hat + y_mean
      fit$test_predictions_control  <- fit$test_predictions_control  * sigma_hat + y_mean
      fit$train_predictions_treat   <- fit$train_predictions_treat   * sigma_hat
      fit$test_predictions_treat    <- fit$test_predictions_treat    * sigma_hat
      fit$train_predictions_deconf  <- fit$train_predictions_deconf  * sigma_hat
      fit$test_predictions_deconf   <- fit$test_predictions_deconf   * sigma_hat
      if (store_posterior_sample) {
        fit$train_predictions_sample_control <- fit$train_predictions_sample_control * sigma_hat + y_mean
        fit$test_predictions_sample_control  <- fit$test_predictions_sample_control  * sigma_hat + y_mean
        fit$train_predictions_sample_treat   <- fit$train_predictions_sample_treat   * sigma_hat
        fit$test_predictions_sample_treat    <- fit$test_predictions_sample_treat    * sigma_hat
        fit$train_predictions_sample_deconf  <- fit$train_predictions_sample_deconf  * sigma_hat
        fit$test_predictions_sample_deconf   <- fit$test_predictions_sample_deconf   * sigma_hat
      }
    }

  } else {

    # Continuous outcome
    y       <- as.numeric(y)
    status  <- rep(1L, n_train)
    survival <- FALSE

    if (is.null(sigma)) {
      sigma_hat   <- sd(y)
      sigma_known <- FALSE
    } else {
      sigma_hat   <- sigma
      sigma_known <- TRUE
    }

    qchi   <- qchisq(1.0 - q, nu)
    lambda <- (sigma_hat^2 * qchi) / nu
    y_mean <- mean(y)
    y      <- (y - y_mean) / sigma_hat

    fit <- FusionForest_cpp(
      nSEXP                       = n_train,
      p_treatSEXP                 = p_treat,
      p_controlSEXP               = p_control,
      X_train_treatSEXP           = X_train_treat,
      X_train_controlSEXP         = X_train_control,
      ySEXP                       = y,
      status_indicatorSEXP        = status,
      is_survivalSEXP             = FALSE,
      treatment_indicatorSEXP     = treatment_indicator_train,
      source_indicatorSEXP        = source_indicator_train,
      n_testSEXP                  = n_test,
      X_test_controlSEXP          = X_test_control,
      X_test_treatSEXP            = X_test_treat,
      X_test_deconfSEXP           = X_test_deconf,
      treatment_indicator_testSEXP = treatment_indicator_test,
      source_indicator_testSEXP   = source_indicator_test,
      n_deconfSEXP                = n_deconf,
      p_deconfSEXP                = p_deconf,
      X_train_deconfSEXP          = X_train_deconf,
      no_trees_deconfSEXP         = number_of_trees_deconf,
      power_deconfSEXP            = power,
      base_deconfSEXP             = base,
      p_grow_deconfSEXP           = p_grow,
      p_prune_deconfSEXP          = p_prune,
      omega_deconfSEXP            = 0.5 / sqrt(number_of_trees_deconf),
      no_trees_treatSEXP          = number_of_trees_treat,
      power_treatSEXP             = power,
      base_treatSEXP              = base,
      p_grow_treatSEXP            = p_grow,
      p_prune_treatSEXP           = p_prune,
      omega_treatSEXP             = 0.5 / sqrt(number_of_trees_treat),
      no_trees_controlSEXP        = number_of_trees_control,
      power_controlSEXP           = power,
      base_controlSEXP            = base,
      p_grow_controlSEXP          = p_grow,
      p_prune_controlSEXP         = p_prune,
      omega_controlSEXP           = 0.5 / sqrt(number_of_trees_control),
      sigma_knownSEXP             = sigma_known,
      sigmaSEXP                   = sigma_hat,
      lambdaSEXP                  = lambda,
      nuSEXP                      = nu,
      N_postSEXP                  = N_post,
      N_burnSEXP                  = N_burn,
      store_posterior_sampleSEXP  = store_posterior_sample,
      verboseSEXP                 = verbose
    )

    # Back-transform
    fit$train_predictions         <- fit$train_predictions         * sigma_hat + y_mean
    fit$test_predictions          <- fit$test_predictions          * sigma_hat + y_mean
    fit$train_predictions_control <- fit$train_predictions_control * sigma_hat + y_mean
    fit$test_predictions_control  <- fit$test_predictions_control  * sigma_hat + y_mean
    fit$train_predictions_treat   <- fit$train_predictions_treat   * sigma_hat
    fit$test_predictions_treat    <- fit$test_predictions_treat    * sigma_hat
    fit$train_predictions_deconf  <- fit$train_predictions_deconf  * sigma_hat
    fit$test_predictions_deconf   <- fit$test_predictions_deconf   * sigma_hat
    if (store_posterior_sample) {
      fit$train_predictions_sample_control <- fit$train_predictions_sample_control * sigma_hat + y_mean
      fit$test_predictions_sample_control  <- fit$test_predictions_sample_control  * sigma_hat + y_mean
      fit$train_predictions_sample_treat   <- fit$train_predictions_sample_treat   * sigma_hat
      fit$test_predictions_sample_treat    <- fit$test_predictions_sample_treat    * sigma_hat
      fit$train_predictions_sample_deconf  <- fit$train_predictions_sample_deconf  * sigma_hat
      fit$test_predictions_sample_deconf   <- fit$test_predictions_sample_deconf   * sigma_hat
    }
  }

  # Discard burn-in sigma draws
  if (!sigma_known) fit$sigma <- fit$sigma[-(1:N_burn)]

  return(fit)
}
