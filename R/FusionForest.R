#' FusionForest
#'
#' Bayesian data-fusion model combining an RCT and an observational study to
#' estimate heterogeneous treatment effects.
#'
#' Two decompositions are available, selected via
#' \code{decomposition}:
#' \describe{
#'   \item{\code{"three-forest"}}{The original model with a single prognostic
#'     forest:
#'     \deqn{\log(T) = m_0(X) + b\,[\tau(X) + (1-S)\,c(X)] +
#'       \sigma\varepsilon}}
#'   \item{\code{"four-forest"} (default)}{Replaces \eqn{m_0} with a
#'     MAP-prior decomposition:
#'     \deqn{\log(T) = \mu(X) + (1-S)\,g(X) + b\,[\tau(X) + (1-S)\,c(X)] +
#'       \sigma\varepsilon}
#'     where \eqn{\mu(X)} is a shared baseline (all data) and \eqn{g(X)}
#'     captures the RWD-specific deviation (RWD only).  The borrowing
#'     parameter \code{k_g} controls shrinkage of \eqn{g} toward zero.}
#' }
#'
#' @param y Numeric vector of outcomes (survival times or continuous responses).
#' @param status Integer vector of event indicators (\code{1} = event observed,
#'   \code{0} = censored).  Required when \code{outcome_type = "right-censored"}.
#' @param X_train_control Numeric matrix of covariates for the prognostic
#'   (\eqn{m_0} or \eqn{\mu}) and deconfounding (\eqn{c}) forests.  One row
#'   per training observation.
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
#' @param decomposition Character; \code{"four-forest"} (default) for the
#'   \eqn{\mu + g} decomposition or \code{"three-forest"} for the original
#'   single-\eqn{m_0} model.
#' @param number_of_trees_control,number_of_trees_treat,number_of_trees_deconf
#'   Number of trees in each BART ensemble.  Default 200.
#' @param number_of_trees_deviation Number of trees for the \eqn{g} forest
#'   (four-forest mode only).  Default 200.
#' @param k_g Leaf-prior scale for the deviation forest \eqn{g}.
#'   Larger values shrink \eqn{g} toward zero (stronger borrowing).
#'   \code{k_g = Inf} disables \eqn{g} entirely (full pooling).
#'   Default 2 (standard BART scale).  Ignored in three-forest mode.
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
#' @param treatment_coding Character; how the binary treatment indicator
#'   \eqn{Z \in \{0,1\}} is mapped to the regression weight \eqn{b_i} in the
#'   BCF parameterisation \eqn{y = \mu(X) + b\,\tau(X) + \dots} (and similarly
#'   for the deconfounding term).  One of:
#'   \describe{
#'     \item{\code{"centered"} (default)}{\eqn{b = 0.5} if treated,
#'       \eqn{b = -0.5} if control.  Both arms inform \eqn{\tau}; \eqn{\mu}
#'       is the average-arm mean function.}
#'     \item{\code{"binary"}}{\eqn{b = 1} if treated, \eqn{b = 0} if control.
#'       Only treated rows inform \eqn{\tau}; \eqn{\mu} is the control-arm
#'       mean function.}
#'     \item{\code{"adaptive"}}{\eqn{b_i = z_i - \pi_i}, the propensity-score
#'       residual (Hahn et al., 2020).  Requires \code{propensity_train} and
#'       \code{propensity_test}.}
#'   }
#'   The \eqn{\tau} and deconfounding (\eqn{c}) forests are fit with
#'   per-observation weights \eqn{w_i = b_i^2} so that the weighted
#'   sufficient statistics match the full-data likelihood; rows with
#'   \eqn{|b_i|} essentially zero are excluded from those updates.
#' @param propensity_train,propensity_test Numeric vectors of estimated
#'   propensity scores in (0, 1) for the training and test rows.  Required
#'   when \code{treatment_coding = "adaptive"}; ignored otherwise.
#' @param error_dist Character; residual-distribution prior.  One of:
#'   \describe{
#'     \item{\code{"gaussian"} (default)}{Single normal residual,
#'       \eqn{\varepsilon_i \sim N(0, \sigma^2)}.}
#'     \item{\code{"shared_dp"}}{Centred Dirichlet-process mixture of
#'       Gaussians on the residual, pooled across data sources (AFTrees-style;
#'       Henderson et al., 2020).  Atoms are recentred to weighted mean zero
#'       to identify the structural mean components.}
#'     \item{\code{"source_dp"}}{Independent centred Dirichlet-process
#'       mixtures per source (Stage A of the HDP-CDP plan).  RCT and OS get
#'       their own atoms, weights, and concentration; \eqn{\sigma} is shared.}
#'     \item{\code{"source_dp_scale"}}{Same as \code{"source_dp"} but with
#'       per-source error scales \eqn{\sigma_s}.  Each source gets its own
#'       inverse-gamma conjugate update for \eqn{\sigma_s}, and the per-source
#'       precision is propagated to the forest backfitting via per-observation
#'       precision weights.}
#'     \item{\code{"source_hdp"}}{Hierarchical centred Dirichlet process
#'       (Stage B): atoms \eqn{\theta_k^*} are shared across sources via a
#'       top-level \eqn{DP(\gamma, H)}; per-source weights \eqn{\pi_{sk}} and
#'       means \eqn{\mu_s} restore identifiability of the structural
#'       components.  Concentration parameters \eqn{\gamma, M_0, M_1} are
#'       updated via Escobar-West auxiliary-variable steps; the top-level
#'       sticks \eqn{\beta} use the Antoniak-Teh table-count augmentation.}
#'   }
#' @param error_truncation_K Integer; truncation level for the stick-breaking
#'   representation.  Default 50.  Increase if the number of occupied components
#'   approaches the bound during MCMC.
#' @param error_atom_scale Numeric; prior standard deviation for each atom on
#'   the standardised response scale, so each atom has prior
#'   \eqn{N(0, \texttt{error\_atom\_scale}^2)}.  Default 0.5 (so atoms are
#'   expected within roughly \eqn{\pm 1} on the standardised scale).
#' @param error_mass_init Numeric; starting value for the concentration
#'   parameter \eqn{\alpha} (or each \eqn{M_s} under \code{"source_dp"}).
#'   Updated by the sampler via an AFTrees-style Gamma conjugate step with
#'   fixed hyperprior \eqn{\mathrm{Gamma}(2, 0.1)}.  Default 1.
#' @param store_posterior_sample Logical; if \code{TRUE}, the full
#'   \eqn{N_{\text{post}} \times n} posterior sample matrices are returned for
#'   all component forests.
#' @param verbose Logical; print a progress bar and summary statistics.
#'
#' @return A named list with components:
#' \describe{
#'   \item{train_predictions, test_predictions}{Posterior mean of the total
#'     fitted values.}
#'   \item{train_predictions_control, test_predictions_control}{Posterior mean
#'     of the shared baseline \eqn{\mu(X)} (four-forest) or \eqn{m_0(X)}
#'     (three-forest).}
#'   \item{train_predictions_treat, test_predictions_treat}{Posterior mean of
#'     the CATE \eqn{\tau(X)}.}
#'   \item{train_predictions_deconf, test_predictions_deconf}{Posterior mean of
#'     the confounding function \eqn{c(X)} (OS rows only for training).}
#'   \item{train_predictions_deviation, test_predictions_deviation}{Posterior
#'     mean of the RWD deviation \eqn{g(X)} (four-forest only; OS rows only
#'     for training).}
#'   \item{sigma}{Posterior sample of \eqn{\sigma} (or the fixed value if
#'     \code{sigma} was supplied).}
#'   \item{acceptance_ratio_control, acceptance_ratio_treat,
#'     acceptance_ratio_deconf}{Tree-update acceptance rates.}
#'   \item{acceptance_ratio_deviation}{Acceptance rate for \eqn{g}
#'     (four-forest only).}
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
  status                      = NULL,
  X_train_control,
  X_train_treat,
  treatment_indicator_train,
  source_indicator_train,
  X_test_control              = NULL,
  X_test_treat                = NULL,
  X_test_deconf               = NULL,
  treatment_indicator_test    = NULL,
  source_indicator_test       = NULL,
  outcome_type                = "continuous",
  timescale                   = "time",
  decomposition               = "four-forest",
  number_of_trees_control     = 200,
  number_of_trees_treat       = 200,
  number_of_trees_deconf      = 200,
  number_of_trees_deviation   = 200,
  k_g                         = 2,
  power                       = 2.0,
  base                        = 0.95,
  p_grow                      = 0.4,
  p_prune                     = 0.4,
  nu                          = 3,
  q                           = 0.90,
  sigma                       = NULL,
  N_post                      = 5000,
  N_burn                      = 5000,
  treatment_coding            = c("centered", "binary", "adaptive"),
  propensity_train            = NULL,
  propensity_test             = NULL,
  error_dist                  = c("gaussian", "shared_dp", "source_dp",
                                  "source_dp_scale", "source_hdp"),
  error_truncation_K          = 50L,
  error_atom_scale            = 0.5,
  error_mass_init             = 1.0,
  store_posterior_sample       = FALSE,
  verbose                     = TRUE
) {

  ## ------------------------------------------------------------------
  ## Input validation
  ## ------------------------------------------------------------------

  allowed_types <- c("continuous", "right-censored")
  if (!outcome_type %in% allowed_types)
    stop("Invalid outcome_type. Choose 'continuous' or 'right-censored'.")

  allowed_decomp <- c("three-forest", "four-forest")
  if (!decomposition %in% allowed_decomp)
    stop("Invalid decomposition. Choose 'three-forest' or 'four-forest'.")

  treatment_coding <- match.arg(treatment_coding,
                                c("centered", "binary", "adaptive"))

  error_dist <- match.arg(error_dist,
                          c("gaussian", "shared_dp", "source_dp",
                            "source_dp_scale", "source_hdp"))
  mixture_mode <- switch(error_dist,
                         "gaussian"         = 0L,
                         "shared_dp"        = 1L,
                         "source_dp"        = 2L,
                         "source_dp_scale"  = 3L,
                         "source_hdp"       = 4L)
  error_truncation_K <- as.integer(error_truncation_K)[1L]
  if (error_truncation_K < 2L)
    stop("error_truncation_K must be at least 2.")
  if (!is.numeric(error_atom_scale) || length(error_atom_scale) != 1L ||
      !is.finite(error_atom_scale) || error_atom_scale <= 0)
    stop("error_atom_scale must be a single positive number.")
  if (!is.numeric(error_mass_init) || length(error_mass_init) != 1L ||
      !is.finite(error_mass_init) || error_mass_init <= 0)
    stop("error_mass_init must be a single positive number.")
  mixture_prior_atom_variance <- as.numeric(error_atom_scale) ^ 2
  mixture_mass_init           <- as.numeric(error_mass_init)

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

  if (treatment_coding == "adaptive") {
    if (is.null(propensity_train))
      stop("treatment_coding = 'adaptive' requires propensity_train.")
    if (length(propensity_train) != length(y))
      stop("propensity_train must match length(y).")
    if (any(propensity_train <= 0 | propensity_train >= 1))
      stop("propensity_train must lie strictly in (0, 1).")
    propensity_train <- as.numeric(propensity_train)
  } else {
    propensity_train <- numeric(0)
  }

  # OS subset for the deconfounding forest
  n_deconf <- sum(source_indicator_train == 0L)
  if (n_deconf <= 0L) stop("At least one observational-study (source = 0) row is required.")
  X_train_deconf <- X_train_control[source_indicator_train == 0L, , drop = FALSE]
  p_deconf       <- ncol(X_train_deconf)

  # OS subset for the deviation (g) forest (same rows as deconf)
  use_four_forest <- (decomposition == "four-forest")
  n_deviation     <- n_deconf
  X_train_deviation <- X_train_deconf  # same subset, same covariates
  p_deviation       <- p_deconf

  # Compute omega for deviation forest via k_g
  if (use_four_forest && is.infinite(k_g)) {
    omega_deviation <- 1e-10  # effectively disable g
  } else if (use_four_forest) {
    omega_deviation <- 0.5 / (k_g * sqrt(number_of_trees_deviation))
  } else {
    omega_deviation <- NULL  # unused in three-forest mode
  }

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

    if (treatment_coding == "adaptive") {
      if (is.null(propensity_test))
        stop("treatment_coding = 'adaptive' requires propensity_test.")
      if (length(propensity_test) != n_test)
        stop("propensity_test must match the number of test rows.")
      if (any(propensity_test <= 0 | propensity_test >= 1))
        stop("propensity_test must lie strictly in (0, 1).")
      propensity_test <- as.numeric(propensity_test)
    } else {
      propensity_test <- numeric(0)
    }

    X_test_deconf <- if (is.null(X_test_deconf)) {
      X_test_control
    } else {
      if (!is.matrix(X_test_deconf)) stop("X_test_deconf must be a matrix.")
      if (nrow(X_test_deconf) != n_test) stop("X_test_deconf rows must match X_test_control.")
      if (ncol(X_test_deconf) != p_deconf) stop("X_test_deconf must have ", p_deconf, " columns.")
      X_test_deconf
    }

    X_test_deviation <- X_test_deconf  # same covariates for g and c

    X_test_control   <- as.numeric(t(X_test_control))
    X_test_treat     <- as.numeric(t(X_test_treat))
    X_test_deconf    <- as.numeric(t(X_test_deconf))
    X_test_deviation <- as.numeric(t(X_test_deviation))

  } else {
    n_test                   <- 1L
    X_test_control           <- as.numeric(colMeans(X_train_control))
    X_test_treat             <- as.numeric(colMeans(X_train_treat))
    X_test_deconf            <- X_test_control
    X_test_deviation         <- X_test_control
    treatment_indicator_test <- 1L
    source_indicator_test    <- 1L
    if (treatment_coding == "adaptive") {
      if (is.null(propensity_test))
        stop("treatment_coding = 'adaptive' requires propensity_test.")
      propensity_test <- as.numeric(propensity_test)[1L]
      if (length(propensity_test) != 1L)
        stop("propensity_test must have one entry when no test data is given.")
    } else {
      propensity_test <- numeric(0)
    }
  }

  # Flatten training matrices
  X_train_control   <- as.numeric(t(X_train_control))
  X_train_treat     <- as.numeric(t(X_train_treat))
  X_train_deconf    <- as.numeric(t(X_train_deconf))
  X_train_deviation <- as.numeric(t(X_train_deviation))

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

    fit <- .call_cpp_backend(
      use_four_forest, n_train, p_treat, p_control, X_train_treat,
      X_train_control, y, status, survival, treatment_indicator_train,
      source_indicator_train, n_test, X_test_control, X_test_treat,
      X_test_deconf, X_test_deviation, treatment_indicator_test,
      source_indicator_test, n_deconf, p_deconf, X_train_deconf,
      number_of_trees_deconf, n_deviation, p_deviation,
      X_train_deviation, number_of_trees_deviation, omega_deviation,
      number_of_trees_treat, number_of_trees_control,
      power, base, p_grow, p_prune, sigma_known, sigma_hat, lambda,
      nu, N_post, N_burn, store_posterior_sample, verbose,
      treatment_coding, propensity_train, propensity_test,
      mixture_mode, error_truncation_K,
      mixture_prior_atom_variance, mixture_mass_init
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
      if (use_four_forest) {
        fit$train_predictions_deviation <- exp(fit$train_predictions_deviation * sigma_hat)
        fit$test_predictions_deviation  <- exp(fit$test_predictions_deviation  * sigma_hat)
      }
      if (store_posterior_sample) {
        fit$train_predictions_sample_control <- exp(fit$train_predictions_sample_control * sigma_hat + y_mean)
        fit$test_predictions_sample_control  <- exp(fit$test_predictions_sample_control  * sigma_hat + y_mean)
        fit$train_predictions_sample_treat   <- exp(fit$train_predictions_sample_treat   * sigma_hat)
        fit$test_predictions_sample_treat    <- exp(fit$test_predictions_sample_treat    * sigma_hat)
        fit$train_predictions_sample_deconf  <- exp(fit$train_predictions_sample_deconf  * sigma_hat)
        fit$test_predictions_sample_deconf   <- exp(fit$test_predictions_sample_deconf   * sigma_hat)
        if (use_four_forest) {
          fit$train_predictions_sample_deviation <- exp(fit$train_predictions_sample_deviation * sigma_hat)
          fit$test_predictions_sample_deviation  <- exp(fit$test_predictions_sample_deviation  * sigma_hat)
        }
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
      if (use_four_forest) {
        fit$train_predictions_deviation <- fit$train_predictions_deviation * sigma_hat
        fit$test_predictions_deviation  <- fit$test_predictions_deviation  * sigma_hat
      }
      if (store_posterior_sample) {
        fit$train_predictions_sample_control <- fit$train_predictions_sample_control * sigma_hat + y_mean
        fit$test_predictions_sample_control  <- fit$test_predictions_sample_control  * sigma_hat + y_mean
        fit$train_predictions_sample_treat   <- fit$train_predictions_sample_treat   * sigma_hat
        fit$test_predictions_sample_treat    <- fit$test_predictions_sample_treat    * sigma_hat
        fit$train_predictions_sample_deconf  <- fit$train_predictions_sample_deconf  * sigma_hat
        fit$test_predictions_sample_deconf   <- fit$test_predictions_sample_deconf   * sigma_hat
        if (use_four_forest) {
          fit$train_predictions_sample_deviation <- fit$train_predictions_sample_deviation * sigma_hat
          fit$test_predictions_sample_deviation  <- fit$test_predictions_sample_deviation  * sigma_hat
        }
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

    fit <- .call_cpp_backend(
      use_four_forest, n_train, p_treat, p_control, X_train_treat,
      X_train_control, y, status, survival, treatment_indicator_train,
      source_indicator_train, n_test, X_test_control, X_test_treat,
      X_test_deconf, X_test_deviation, treatment_indicator_test,
      source_indicator_test, n_deconf, p_deconf, X_train_deconf,
      number_of_trees_deconf, n_deviation, p_deviation,
      X_train_deviation, number_of_trees_deviation, omega_deviation,
      number_of_trees_treat, number_of_trees_control,
      power, base, p_grow, p_prune, sigma_known, sigma_hat, lambda,
      nu, N_post, N_burn, store_posterior_sample, verbose,
      treatment_coding, propensity_train, propensity_test,
      mixture_mode, error_truncation_K,
      mixture_prior_atom_variance, mixture_mass_init
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
    if (use_four_forest) {
      fit$train_predictions_deviation <- fit$train_predictions_deviation * sigma_hat
      fit$test_predictions_deviation  <- fit$test_predictions_deviation  * sigma_hat
    }
    if (store_posterior_sample) {
      fit$train_predictions_sample_control <- fit$train_predictions_sample_control * sigma_hat + y_mean
      fit$test_predictions_sample_control  <- fit$test_predictions_sample_control  * sigma_hat + y_mean
      fit$train_predictions_sample_treat   <- fit$train_predictions_sample_treat   * sigma_hat
      fit$test_predictions_sample_treat    <- fit$test_predictions_sample_treat    * sigma_hat
      fit$train_predictions_sample_deconf  <- fit$train_predictions_sample_deconf  * sigma_hat
      fit$test_predictions_sample_deconf   <- fit$test_predictions_sample_deconf   * sigma_hat
      if (use_four_forest) {
        fit$train_predictions_sample_deviation <- fit$train_predictions_sample_deviation * sigma_hat
        fit$test_predictions_sample_deviation  <- fit$test_predictions_sample_deviation  * sigma_hat
      }
    }
  }

  # Discard burn-in sigma draws
  if (!sigma_known) fit$sigma <- fit$sigma[-(1:N_burn)]

  # Back-transform DP atoms onto the response scale (multiply by sigma_hat).
  # Atoms are mean-zero shifts so no y_mean offset is needed.  Mass and stick
  # weights are scale-free and pass through unchanged.  Per-source sigma_g
  # (SOURCE_DP_SCALE) is also rescaled to the response scale.
  if (!is.null(fit$dp_locations)) {
    fit$dp_locations <- lapply(fit$dp_locations, function(m) m * sigma_hat)
    fit$error_dist   <- error_dist
  }
  if (!is.null(fit$dp_sigma_g)) {
    fit$dp_sigma_g <- lapply(fit$dp_sigma_g, function(v) v * sigma_hat)
  }
  # HDP-only posterior fields: shared atoms and per-source means are on the
  # standardised residual scale and need rescaling.  beta (stick weights) and
  # gamma (concentration) are scale-free.
  if (!is.null(fit$dp_locations_shared))
    fit$dp_locations_shared <- fit$dp_locations_shared * sigma_hat
  if (!is.null(fit$dp_mu_g))
    fit$dp_mu_g <- lapply(fit$dp_mu_g, function(v) v * sigma_hat)

  # Metadata consumed by fusion_estimand().  sigma is stored on the standardised
  # scale; sigma * sigma_hat recovers the response (log-time) scale residual SD.
  fit$meta <- list(
    sigma_hat     = sigma_hat,
    y_mean        = y_mean,
    timescale     = timescale,
    outcome_type  = outcome_type,
    decomposition = decomposition,
    error_dist    = error_dist
  )
  class(fit) <- c("FusionForest", class(fit))

  return(fit)
}


# Internal helper: dispatch to three-forest or four-forest C++ backend
.call_cpp_backend <- function(
  use_four_forest, n_train, p_treat, p_control, X_train_treat,
  X_train_control, y, status, is_survival, treatment_indicator_train,
  source_indicator_train, n_test, X_test_control, X_test_treat,
  X_test_deconf, X_test_deviation, treatment_indicator_test,
  source_indicator_test, n_deconf, p_deconf, X_train_deconf,
  number_of_trees_deconf, n_deviation, p_deviation,
  X_train_deviation, number_of_trees_deviation, omega_deviation,
  number_of_trees_treat, number_of_trees_control,
  power, base, p_grow, p_prune, sigma_known, sigma_hat, lambda,
  nu, N_post, N_burn, store_posterior_sample, verbose,
  treatment_coding, propensity_train, propensity_test,
  mixture_mode, mixture_K, mixture_prior_atom_variance, mixture_mass_init
) {

  # Shared arguments for both backends
  shared <- list(
    nSEXP                        = n_train,
    p_treatSEXP                  = p_treat,
    p_controlSEXP                = p_control,
    X_train_treatSEXP            = X_train_treat,
    X_train_controlSEXP          = X_train_control,
    ySEXP                        = y,
    status_indicatorSEXP         = status,
    is_survivalSEXP              = is_survival,
    treatment_indicatorSEXP      = treatment_indicator_train,
    source_indicatorSEXP         = source_indicator_train,
    n_testSEXP                   = n_test,
    X_test_controlSEXP           = X_test_control,
    X_test_treatSEXP             = X_test_treat,
    X_test_deconfSEXP            = X_test_deconf,
    treatment_indicator_testSEXP = treatment_indicator_test,
    source_indicator_testSEXP    = source_indicator_test,
    n_deconfSEXP                 = n_deconf,
    p_deconfSEXP                 = p_deconf,
    X_train_deconfSEXP           = X_train_deconf,
    no_trees_deconfSEXP          = number_of_trees_deconf,
    power_deconfSEXP             = power,
    base_deconfSEXP              = base,
    p_grow_deconfSEXP            = p_grow,
    p_prune_deconfSEXP           = p_prune,
    omega_deconfSEXP             = 0.5 / sqrt(number_of_trees_deconf),
    no_trees_treatSEXP           = number_of_trees_treat,
    power_treatSEXP              = power,
    base_treatSEXP               = base,
    p_grow_treatSEXP             = p_grow,
    p_prune_treatSEXP            = p_prune,
    omega_treatSEXP              = 0.5 / sqrt(number_of_trees_treat),
    no_trees_controlSEXP         = number_of_trees_control,
    power_controlSEXP            = power,
    base_controlSEXP             = base,
    p_grow_controlSEXP           = p_grow,
    p_prune_controlSEXP          = p_prune,
    omega_controlSEXP            = 0.5 / sqrt(number_of_trees_control),
    sigma_knownSEXP              = sigma_known,
    sigmaSEXP                    = sigma_hat,
    lambdaSEXP                   = lambda,
    nuSEXP                       = nu,
    N_postSEXP                   = N_post,
    N_burnSEXP                   = N_burn,
    store_posterior_sampleSEXP   = store_posterior_sample,
    verboseSEXP                  = verbose,
    treatment_codingSEXP         = treatment_coding,
    propensity_trainSEXP         = propensity_train,
    propensity_testSEXP          = propensity_test,
    mixture_modeSEXP             = as.integer(mixture_mode),
    mixture_KSEXP                = as.integer(mixture_K),
    mixture_prior_atom_varianceSEXP = as.numeric(mixture_prior_atom_variance),
    mixture_mass_initSEXP        = as.numeric(mixture_mass_init)
  )

  if (use_four_forest) {
    extra <- list(
      X_test_deviationSEXP     = X_test_deviation,
      n_deviationSEXP          = n_deviation,
      p_deviationSEXP          = p_deviation,
      X_train_deviationSEXP    = X_train_deviation,
      no_trees_deviationSEXP   = number_of_trees_deviation,
      power_deviationSEXP      = power,
      base_deviationSEXP       = base,
      p_grow_deviationSEXP     = p_grow,
      p_prune_deviationSEXP    = p_prune,
      omega_deviationSEXP      = omega_deviation
    )
    do.call(FusionForest4_cpp, c(shared, extra))
  } else {
    do.call(FusionForest_cpp, shared)
  }
}
