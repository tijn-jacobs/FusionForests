#' Bayesian fusion forest
#'
#' Fits the Bayesian fusion forest.  The model combines a randomised
#' controlled trial (RCT) with real-world data (RWD) to estimate
#' heterogeneous treatment effects.  The RWD may be confounded.
#'
#' @details
#' The model decomposes the log survival time as
#' \deqn{\log T = m_0^{\mathrm{sh}}(X) + (1-S)\,d(X) + A\,\tau(X) +
#'   (1-S)\,A\,c(X) + \varepsilon,}{log T = m_0^sh(X) + (1-S) d(X) +
#'   A tau(X) + (1-S) A c(X) + epsilon,}
#' with treatment \eqn{A} (1 = treated, 0 = control) and source \eqn{S}
#' (1 = RCT, 0 = RWD).  The error \eqn{\varepsilon}{epsilon} has mean zero;
#' see \code{error_dist}.  For a continuous outcome, the outcome replaces
#' \eqn{\log T}{log T}.
#'
#' Each component has its own BART forest.  The arguments of a forest carry
#' the suffix in brackets.
#' \describe{
#'   \item{\eqn{m_0^{\mathrm{sh}}(X)}{m_0^sh(X)} (\code{control})}{Baseline
#'     log survival time under control, shared by both sources.}
#'   \item{\eqn{d(X)} (\code{deviation})}{Difference between the RWD and
#'     RCT baselines.  The baseline is
#'     \eqn{m_0(X, S) = m_0^{\mathrm{sh}}(X) + (1-S)\,d(X)}{m_0(X, S) =
#'     m_0^sh(X) + (1-S) d(X)}.}
#'   \item{\eqn{\tau(X)} (\code{treat})}{Treatment effect on the log-time
#'     scale.}
#'   \item{\eqn{c(X)} (\code{deconf})}{Confounding function: the bias in
#'     the RWD treatment contrast.}
#' }
#'
#' The step heights of forest
#' \eqn{f \in \{\mathrm{sh}, d, \tau, c\}}{f in {sh, d, tau, c}} have prior
#' \eqn{N(0, \sigma_{h,f}^2)} with
#' \deqn{\sigma_{h,f} = k_f / (2\sqrt{J_f}),}
#' where \eqn{J_f} is the number of trees and \eqn{k_f} the scale.  A smaller
#' \eqn{k_f} shrinks the forest more strongly toward zero.  All defaults
#' follow the paper.
#'
#' The model above uses the binary treatment indicator \eqn{A}, as in the
#' paper (\code{treatment_coding = "binary"}).  The default \code{"centered"}
#' coding replaces \eqn{A} by \eqn{A - 1/2}.  The functions \eqn{\tau} and
#' \eqn{c} keep their meaning.  The baseline forests then describe the
#' average of the two arms instead of the control arm.
#'
#' @param y Numeric vector of outcomes: survival times or a continuous
#'   response.
#' @param status Event indicator (1 = event, 0 = censored).  Required for
#'   right-censored outcomes.  Use 0 for interval-censored rows.
#' @param observed_left_time,observed_right_time,interval_censoring_indicator
#'   Optional vectors for interval censoring, on the same scale as \code{y}.
#'   Each row is one of three types:
#'   \describe{
#'     \item{\code{status = 1}}{Event at \code{y}.  The bounds are ignored.}
#'     \item{\code{status = 0}, indicator 0}{Right-censored at
#'       \code{observed_right_time}.}
#'     \item{\code{status = 0}, indicator 1}{Event in the interval
#'       (\code{observed_left_time}, \code{observed_right_time}].}
#'   }
#'   Leave all three \code{NULL} for plain right-censoring at \code{y}.  Only
#'   used when \code{outcome_type = "right-censored"}.
#' @param X_train_control Covariate matrix for the
#'   \eqn{m_0^{\mathrm{sh}}}{m_0^sh}, \eqn{d} and \eqn{c} forests.  The
#'   \eqn{d} and \eqn{c} forests use the RWD rows only.
#' @param X_train_treat Covariate matrix for the \eqn{\tau} forest, with the
#'   same rows as \code{X_train_control}.
#' @param treatment_indicator_train Treatment (1 = treated, 0 = control).
#' @param source_indicator_train Source (1 = RCT, 0 = RWD).
#' @param X_test_control,X_test_treat,X_test_deconf Optional covariate
#'   matrices for prediction.  \code{X_test_deconf} defaults to
#'   \code{X_test_control}.  Without test matrices, the function predicts at
#'   the column means of the training data.
#' @param treatment_indicator_test,source_indicator_test Treatment and
#'   source of the test rows.  Both default to 1.
#' @param outcome_type \code{"continuous"} or \code{"right-censored"}.  Use
#'   \code{"right-censored"} also for interval-censored data.
#' @param timescale \code{"time"} if \code{y} holds survival times,
#'   \code{"log"} if it holds log survival times.  With \code{"time"}, the
#'   predictions are returned on the time scale.
#' @param number_of_trees_control,number_of_trees_treat,number_of_trees_deconf
#'   Number of trees \eqn{J_f} in the \eqn{m_0^{\mathrm{sh}}}{m_0^sh},
#'   \eqn{\tau} and \eqn{c} forests.  Defaults: 200, 100 and 50.
#' @param number_of_trees_deviation Number of trees in the \eqn{d} forest.
#'   Default 50.
#' @param k_control,k_treat,k_deconf,k_deviation Step-height scales
#'   \eqn{k_f} of the \eqn{m_0^{\mathrm{sh}}}{m_0^sh}, \eqn{\tau}, \eqn{c}
#'   and \eqn{d} forests; see Details.  Larger values shrink less.  The
#'   defaults are 1, except \code{k_treat = 0.5}.  Set
#'   \code{k_deviation = 0} to switch off \eqn{d}, so that both sources share
#'   one baseline.
#' @param power_control,base_control,power_deviation,base_deviation,power_treat,base_treat,power_deconf,base_deconf
#'   Tree-structure prior of each forest.  A node at depth \eqn{\delta}
#'   splits with probability \eqn{\alpha_f (1 + \delta)^{-\beta_f}}, with
#'   \eqn{\alpha_f} = \code{base} and \eqn{\beta_f} = \code{power}.  A larger
#'   \code{power} or a smaller \code{base} gives shallower trees.  Defaults:
#'   \eqn{(\alpha_f, \beta_f) = (0.95, 2)} for
#'   \eqn{m_0^{\mathrm{sh}}}{m_0^sh} and \eqn{d}, \eqn{(0.95, 3)} for
#'   \eqn{\tau} and \eqn{(0.25, 3)} for \eqn{c}.
#' @param p_grow,p_prune Probabilities of a grow and a prune proposal.
#' @param nu Degrees of freedom of the scaled inverse chi-squared prior on
#'   \eqn{\sigma^2}.
#' @param q Quantile that calibrates the scale \eqn{\lambda} of this prior.
#' @param sigma Fixed error standard deviation.  If \code{NULL} (default),
#'   \eqn{\sigma} is estimated.
#' @param N_post,N_burn Number of posterior and burn-in iterations.
#' @param treatment_coding How the treatment enters the model.  The coded
#'   treatment \eqn{b_i} replaces \eqn{A} in front of \eqn{\tau} and
#'   \eqn{c}.  One of:
#'   \describe{
#'     \item{\code{"centered"} (default)}{\eqn{b_i = \pm 1/2}{b_i = +/- 1/2}.
#'       Both arms inform \eqn{\tau}.  The baseline forests describe the
#'       average of the two arms.}
#'     \item{\code{"binary"}}{\eqn{b_i = A_i}, as in the paper.  The baseline
#'       forests describe the control arm.}
#'     \item{\code{"adaptive"}}{\eqn{b_i = A_i - \pi_i}, the propensity score
#'       residual (Hahn et al., 2020).  Requires propensity scores.}
#'   }
#' @param propensity_train,propensity_test Propensity scores in (0, 1) for
#'   the training and test rows.  Only used with
#'   \code{treatment_coding = "adaptive"}.
#' @param error_dist Prior for the error distribution.  All mixtures are
#'   centred, so the error has mean zero in each source.  One of:
#'   \describe{
#'     \item{\code{"gaussian"}}{Normal errors with one scale \eqn{\sigma}.}
#'     \item{\code{"shared_dp"}}{One Dirichlet process mixture of normals
#'       for both sources (Henderson et al., 2020).}
#'     \item{\code{"source_dp"}}{A separate Dirichlet process mixture per
#'       source, with a shared scale \eqn{\sigma}.}
#'     \item{\code{"source_dp_scale"}}{As \code{"source_dp"}, with a
#'       separate scale \eqn{\sigma_s} per source.}
#'     \item{\code{"source_hdp"}}{Hierarchical Dirichlet process mixture.
#'       The sources share the mixture components but have their own
#'       weights.  Shared scale \eqn{\sigma}.}
#'     \item{\code{"source_hdp_scale"} (default)}{As \code{"source_hdp"},
#'       with a separate scale \eqn{\sigma_s} per source.  This is the error
#'       model of the paper.}
#'   }
#' @param error_truncation_K Truncation level of the mixture.  Default 50.
#'   Increase it if the number of occupied components gets close to this
#'   level.
#' @param error_atom_scale Prior standard deviation of the mixture
#'   locations, on the standardised scale.  Default 0.5.
#' @param error_mass_init Starting value of the concentration parameters.
#'   They have a \eqn{\mathrm{Gamma}(2, 0.1)}{Gamma(2, 0.1)} prior.
#'   Default 1.
#' @param store_posterior_sample If \code{TRUE}, return all posterior draws
#'   of each forest.  Required by \code{\link{fusion_estimand}} and
#'   \code{\link{fusion_projection}}.
#' @param verbose If \code{TRUE}, print progress.
#'
#' @return A \code{FusionForest} object: a list with
#' \describe{
#'   \item{train_predictions, test_predictions}{Posterior mean of the
#'     fitted values.}
#'   \item{train_predictions_control, test_predictions_control}{Posterior
#'     mean of \eqn{m_0^{\mathrm{sh}}(X)}{m_0^sh(X)}.}
#'   \item{train_predictions_treat, test_predictions_treat}{Posterior mean
#'     of \eqn{\tau(X)}.}
#'   \item{train_predictions_deconf, test_predictions_deconf}{Posterior mean
#'     of \eqn{c(X)}.  The training values cover the RWD rows only.}
#'   \item{train_predictions_deviation, test_predictions_deviation}{Posterior
#'     mean of \eqn{d(X)}.  The training values cover the RWD rows only.}
#'   \item{sigma}{Posterior draws of \eqn{\sigma} on the standardised scale.
#'     For the \code{"_scale"} error models this is a pooled scale.}
#'   \item{acceptance_ratio_control, acceptance_ratio_treat,
#'     acceptance_ratio_deconf, acceptance_ratio_deviation}{Acceptance
#'     rates of the tree proposals.}
#'   \item{train_predictions_sample_control, ...}{All posterior draws.  Only
#'     with \code{store_posterior_sample = TRUE}.}
#'   \item{dp_mix_prop, dp_locations, ...}{Posterior draws of the error
#'     mixture.  Only for the mixture error models.}
#'   \item{meta}{Settings used by the summary functions.}
#' }
#'
#' @examples
#' # Small simulated fusion example combining an RCT and RWD
#' set.seed(1)
#' n <- 100
#' X <- matrix(rnorm(n * 3), n, 3)
#' s <- rbinom(n, 1, 0.5)  # 1 = RCT, 0 = RWD
#' a <- rbinom(n, 1, 0.5)  # treatment
#' y <- X[, 1] + 0.5 * a + rnorm(n)
#'
#' fit <- FusionForest(
#'   y = y,
#'   X_train_control = X,
#'   X_train_treat = X,
#'   treatment_indicator_train = a,
#'   source_indicator_train = s,
#'   N_post = 50, N_burn = 25,
#'   verbose = FALSE
#' )
#' print(fit)
#' summary(fit)
#'
#' @importFrom Rcpp evalCpp
#' @useDynLib FusionForests, .registration = TRUE
#' @importFrom stats sd qchisq qnorm runif
#' @export
FusionForest <- function(
  y,
  status                      = NULL,
  observed_left_time          = NULL,
  observed_right_time         = NULL,
  interval_censoring_indicator = NULL,
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
  number_of_trees_control     = 200,
  number_of_trees_treat       = 100,
  number_of_trees_deconf      = 50,
  number_of_trees_deviation   = 50,
  k_control                   = 1,
  k_treat                     = 0.5,
  k_deconf                    = 1,
  k_deviation                 = 1,
  power_control               = 2.0,
  base_control                = 0.95,
  power_deviation             = 2.0,
  base_deviation              = 0.95,
  power_treat                 = 3.0,
  base_treat                  = 0.95,
  power_deconf                = 3.0,
  base_deconf                 = 0.25,
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
  error_dist                  = c("source_hdp_scale", "gaussian",
                                  "shared_dp", "source_dp",
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

  # Per-forest tree-topology priors (power_*, base_*) are passed straight to
  # the backend; there is no shared global power/base.

  treatment_coding <- match.arg(treatment_coding,
                                c("centered", "binary", "adaptive"))

  error_dist <- match.arg(error_dist,
                          c("source_hdp_scale", "gaussian",
                            "shared_dp", "source_dp",
                            "source_dp_scale", "source_hdp"))
  mixture_mode <- switch(error_dist,
                         "gaussian"         = 0L,
                         "shared_dp"        = 1L,
                         "source_dp"        = 2L,
                         "source_dp_scale"  = 3L,
                         "source_hdp"       = 4L,
                         "source_hdp_scale" = 5L)
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

  covariate_matrices <- list(X_train_control = X_train_control,
                             X_train_treat   = X_train_treat,
                             X_test_control  = X_test_control,
                             X_test_treat    = X_test_treat,
                             X_test_deconf   = X_test_deconf)
  for (nm in names(covariate_matrices)) {
    if (!is.null(covariate_matrices[[nm]]) && anyNA(covariate_matrices[[nm]]))
      stop(nm, " contains missing values (NA/NaN); ",
           "missing covariates are not supported.")
  }

  treatment_indicator_train <- as.integer(treatment_indicator_train)
  if (!all(source_indicator_train %in% c(0L, 1L)))
    stop("source_indicator_train must be 0 (RWD) or 1 (RCT).")
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

  # RWD subset for the deconfounding forest
  n_deconf <- sum(source_indicator_train == 0L)
  if (n_deconf <= 0L) stop("At least one real-world-data (source = 0) row is required.")
  X_train_deconf <- X_train_control[source_indicator_train == 0L, , drop = FALSE]
  p_deconf       <- ncol(X_train_deconf)

  # RWD subset for the deviation (d) forest (same rows as deconf)
  n_deviation     <- n_deconf
  X_train_deviation <- X_train_deconf  # same subset, same covariates
  p_deviation       <- p_deconf

  # Leaf-prior standard deviations.  All four forests use the paper's
  # parametrisation omega_X = k_X / (2 sqrt(number_of_trees_X)); see the
  # roxygen for the meaning of k.  Setting k_deviation = 0 is a special case
  # that effectively disables the deviation forest d by collapsing its prior
  # to a point mass at zero.
  validate_k <- function(name, val) {
    if (!is.numeric(val) || length(val) != 1L || !is.finite(val) || val < 0)
      stop(sprintf("`%s` must be a single non-negative finite number.", name))
  }
  validate_k("k_control", k_control)
  validate_k("k_treat",   k_treat)
  validate_k("k_deconf",  k_deconf)
  validate_k("k_deviation", k_deviation)

  omega_control <- k_control / (2 * sqrt(number_of_trees_control))
  omega_treat   <- k_treat   / (2 * sqrt(number_of_trees_treat))
  omega_deconf  <- k_deconf  / (2 * sqrt(number_of_trees_deconf))
  omega_deviation <- if (k_deviation <= 0) 1e-10
                     else k_deviation / (2 * sqrt(number_of_trees_deviation))

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
      if (!all(s %in% c(0L, 1L))) stop("source_indicator_test must be 0 (RWD) or 1 (RCT).")
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
  power_control   <- as.numeric(power_control)[1L]
  base_control    <- as.numeric(base_control)[1L]
  power_deviation <- as.numeric(power_deviation)[1L]
  base_deviation  <- as.numeric(base_deviation)[1L]
  power_treat     <- as.numeric(power_treat)[1L]
  base_treat      <- as.numeric(base_treat)[1L]
  power_deconf    <- as.numeric(power_deconf)[1L]
  base_deconf     <- as.numeric(base_deconf)[1L]
  p_grow  <- as.numeric(p_grow)[1L]
  p_prune <- as.numeric(p_prune)[1L]

  ## ------------------------------------------------------------------
  ## Outcome-specific setup and C++ call
  ## ------------------------------------------------------------------

  if (outcome_type == "right-censored") {

    y <- as.numeric(y)
    if (timescale == "time") y <- log(y)

    ## Interval-censoring bounds.  Defaults reproduce right-censoring.
    icArgsSupplied <- !(is.null(observed_left_time) &&
                       is.null(observed_right_time) &&
                       is.null(interval_censoring_indicator))
    if (icArgsSupplied) {
      if (is.null(observed_left_time) || is.null(observed_right_time) ||
          is.null(interval_censoring_indicator))
        stop("If any of observed_left_time / observed_right_time / ",
             "interval_censoring_indicator is supplied, all three must be.")
      observed_left_time           <- as.numeric(observed_left_time)
      observed_right_time          <- as.numeric(observed_right_time)
      interval_censoring_indicator <- as.numeric(interval_censoring_indicator)
      if (length(observed_left_time) != n_train ||
          length(observed_right_time) != n_train ||
          length(interval_censoring_indicator) != n_train)
        stop("observed_left_time / observed_right_time / ",
             "interval_censoring_indicator must each have length ", n_train, ".")
      if (any(interval_censoring_indicator == 1 &
              observed_left_time >= observed_right_time))
        stop("For interval-censored rows we require ",
             "observed_left_time < observed_right_time.")
      if (timescale == "time") {
        observed_left_time  <- log(observed_left_time)
        observed_right_time <- log(observed_right_time)
      }
    } else {
      observed_left_time           <- y
      observed_right_time          <- y
      interval_censoring_indicator <- rep(0, n_train)
    }

    cens_inf  <- censored_info(y, status)
    y_mean    <- cens_inf$mu
    y         <- y - y_mean
    observed_left_time  <- observed_left_time  - y_mean
    observed_right_time <- observed_right_time - y_mean

    if (is.null(sigma)) {
      sigma_hat  <- cens_inf$sd
      sigma_known <- FALSE
    } else {
      sigma_hat  <- sigma
      sigma_known <- TRUE
    }

    y                   <- y                   / sigma_hat
    observed_left_time  <- observed_left_time  / sigma_hat
    observed_right_time <- observed_right_time / sigma_hat
    survival <- TRUE
    qchi    <- qchisq(1.0 - q, nu)
    lambda  <- (sigma_hat^2 * qchi) / nu

    fit <- .call_cpp_backend(
      TRUE, n_train, p_treat, p_control, X_train_treat,
      X_train_control, y, status,
      observed_left_time, observed_right_time, interval_censoring_indicator,
      survival, treatment_indicator_train,
      source_indicator_train, n_test, X_test_control, X_test_treat,
      X_test_deconf, X_test_deviation, treatment_indicator_test,
      source_indicator_test, n_deconf, p_deconf, X_train_deconf,
      number_of_trees_deconf, n_deviation, p_deviation,
      X_train_deviation, number_of_trees_deviation, omega_deviation,
      number_of_trees_treat, number_of_trees_control,
      omega_treat, omega_control, omega_deconf,
      power_control, base_control, power_deviation, base_deviation,
      power_treat, base_treat, power_deconf, base_deconf,
      p_grow, p_prune, sigma_known, sigma_hat, lambda,
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
      fit$train_predictions_deviation <- exp(fit$train_predictions_deviation * sigma_hat)
      fit$test_predictions_deviation  <- exp(fit$test_predictions_deviation  * sigma_hat)
      if (store_posterior_sample) {
        fit$train_predictions_sample_control <- exp(fit$train_predictions_sample_control * sigma_hat + y_mean)
        fit$test_predictions_sample_control  <- exp(fit$test_predictions_sample_control  * sigma_hat + y_mean)
        fit$train_predictions_sample_treat   <- exp(fit$train_predictions_sample_treat   * sigma_hat)
        fit$test_predictions_sample_treat    <- exp(fit$test_predictions_sample_treat    * sigma_hat)
        fit$train_predictions_sample_deconf  <- exp(fit$train_predictions_sample_deconf  * sigma_hat)
        fit$test_predictions_sample_deconf   <- exp(fit$test_predictions_sample_deconf   * sigma_hat)
        fit$train_predictions_sample_deviation <- exp(fit$train_predictions_sample_deviation * sigma_hat)
        fit$test_predictions_sample_deviation  <- exp(fit$test_predictions_sample_deviation  * sigma_hat)
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
      fit$train_predictions_deviation <- fit$train_predictions_deviation * sigma_hat
      fit$test_predictions_deviation  <- fit$test_predictions_deviation  * sigma_hat
      if (store_posterior_sample) {
        fit$train_predictions_sample_control <- fit$train_predictions_sample_control * sigma_hat + y_mean
        fit$test_predictions_sample_control  <- fit$test_predictions_sample_control  * sigma_hat + y_mean
        fit$train_predictions_sample_treat   <- fit$train_predictions_sample_treat   * sigma_hat
        fit$test_predictions_sample_treat    <- fit$test_predictions_sample_treat    * sigma_hat
        fit$train_predictions_sample_deconf  <- fit$train_predictions_sample_deconf  * sigma_hat
        fit$test_predictions_sample_deconf   <- fit$test_predictions_sample_deconf   * sigma_hat
        fit$train_predictions_sample_deviation <- fit$train_predictions_sample_deviation * sigma_hat
        fit$test_predictions_sample_deviation  <- fit$test_predictions_sample_deviation  * sigma_hat
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

    ## Dummy interval-censoring vectors so the C++ entry point's signature
    ## is satisfied.  Ignored since is_survival = FALSE; no augmentation runs.
    observed_left_time           <- y
    observed_right_time          <- y
    interval_censoring_indicator <- rep(0, n_train)

    fit <- .call_cpp_backend(
      TRUE, n_train, p_treat, p_control, X_train_treat,
      X_train_control, y, status,
      observed_left_time, observed_right_time, interval_censoring_indicator,
      survival, treatment_indicator_train,
      source_indicator_train, n_test, X_test_control, X_test_treat,
      X_test_deconf, X_test_deviation, treatment_indicator_test,
      source_indicator_test, n_deconf, p_deconf, X_train_deconf,
      number_of_trees_deconf, n_deviation, p_deviation,
      X_train_deviation, number_of_trees_deviation, omega_deviation,
      number_of_trees_treat, number_of_trees_control,
      omega_treat, omega_control, omega_deconf,
      power_control, base_control, power_deviation, base_deviation,
      power_treat, base_treat, power_deconf, base_deconf,
      p_grow, p_prune, sigma_known, sigma_hat, lambda,
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
    fit$train_predictions_deviation <- fit$train_predictions_deviation * sigma_hat
    fit$test_predictions_deviation  <- fit$test_predictions_deviation  * sigma_hat
    if (store_posterior_sample) {
      fit$train_predictions_sample_control <- fit$train_predictions_sample_control * sigma_hat + y_mean
      fit$test_predictions_sample_control  <- fit$test_predictions_sample_control  * sigma_hat + y_mean
      fit$train_predictions_sample_treat   <- fit$train_predictions_sample_treat   * sigma_hat
      fit$test_predictions_sample_treat    <- fit$test_predictions_sample_treat    * sigma_hat
      fit$train_predictions_sample_deconf  <- fit$train_predictions_sample_deconf  * sigma_hat
      fit$test_predictions_sample_deconf   <- fit$test_predictions_sample_deconf   * sigma_hat
      fit$train_predictions_sample_deviation <- fit$train_predictions_sample_deviation * sigma_hat
      fit$test_predictions_sample_deviation  <- fit$test_predictions_sample_deviation  * sigma_hat
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
    error_dist    = error_dist,
    # Treatment coding (and test-row propensities for "adaptive"), needed by
    # fusion_estimand() to rebuild the linear predictor of each arm.
    treatment_coding = treatment_coding,
    propensity_test  = if (treatment_coding == "adaptive") propensity_test,
    n_train       = n_train,
    n_rct         = sum(source_indicator_train == 1),
    n_rwd         = sum(source_indicator_train == 0),
    p_control     = p_control,
    p_treat       = p_treat,
    N_post        = N_post,
    N_burn        = N_burn
  )
  class(fit) <- c("FusionForest", class(fit))

  return(fit)
}


# Internal helper: call the four-forest C++ backend.  The leading
# use_four_forest argument is kept so that the frozen FusionForest copies in
# the paper's simulation scripts, which call this helper positionally, keep
# working.
.call_cpp_backend <- function(
  use_four_forest, n_train, p_treat, p_control, X_train_treat,
  X_train_control, y, status,
  observed_left_time, observed_right_time, interval_censoring_indicator,
  is_survival, treatment_indicator_train,
  source_indicator_train, n_test, X_test_control, X_test_treat,
  X_test_deconf, X_test_deviation, treatment_indicator_test,
  source_indicator_test, n_deconf, p_deconf, X_train_deconf,
  number_of_trees_deconf, n_deviation, p_deviation,
  X_train_deviation, number_of_trees_deviation, omega_deviation,
  number_of_trees_treat, number_of_trees_control,
  omega_treat, omega_control, omega_deconf,
  power_control, base_control, power_deviation, base_deviation,
  power_treat, base_treat, power_deconf, base_deconf,
  p_grow, p_prune, sigma_known, sigma_hat, lambda,
  nu, N_post, N_burn, store_posterior_sample, verbose,
  treatment_coding, propensity_train, propensity_test,
  mixture_mode, mixture_K, mixture_prior_atom_variance, mixture_mass_init
) {

  shared <- list(
    nSEXP                        = n_train,
    p_treatSEXP                  = p_treat,
    p_controlSEXP                = p_control,
    X_train_treatSEXP            = X_train_treat,
    X_train_controlSEXP          = X_train_control,
    ySEXP                        = y,
    status_indicatorSEXP         = status,
    is_survivalSEXP              = is_survival,
    observed_left_timeSEXP       = observed_left_time,
    observed_right_timeSEXP      = observed_right_time,
    interval_censoring_indicatorSEXP = interval_censoring_indicator,
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
    power_deconfSEXP             = power_deconf,
    base_deconfSEXP              = base_deconf,
    p_grow_deconfSEXP            = p_grow,
    p_prune_deconfSEXP           = p_prune,
    omega_deconfSEXP             = omega_deconf,
    no_trees_treatSEXP           = number_of_trees_treat,
    power_treatSEXP              = power_treat,
    base_treatSEXP               = base_treat,
    p_grow_treatSEXP             = p_grow,
    p_prune_treatSEXP            = p_prune,
    omega_treatSEXP              = omega_treat,
    no_trees_controlSEXP         = number_of_trees_control,
    power_controlSEXP            = power_control,
    base_controlSEXP             = base_control,
    p_grow_controlSEXP           = p_grow,
    p_prune_controlSEXP          = p_prune,
    omega_controlSEXP            = omega_control,
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

  if (!isTRUE(use_four_forest))
    stop("The three-forest backend has been removed; ",
         "only the four-forest model is available.")

  extra <- list(
    X_test_deviationSEXP     = X_test_deviation,
    n_deviationSEXP          = n_deviation,
    p_deviationSEXP          = p_deviation,
    X_train_deviationSEXP    = X_train_deviation,
    no_trees_deviationSEXP   = number_of_trees_deviation,
    power_deviationSEXP      = power_deviation,
    base_deviationSEXP       = base_deviation,
    p_grow_deviationSEXP     = p_grow,
    p_prune_deviationSEXP    = p_prune,
    omega_deviationSEXP      = omega_deviation
  )
  do.call(FusionForest_cpp, c(shared, extra))
}
