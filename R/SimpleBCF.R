#' SimpleBCF
#'
#' Two-forest Bayesian Causal Forest:
#' \eqn{Y = \mu(X, e) + \tau(X) \cdot A + \sigma\varepsilon},
#' where \eqn{\mu} is a prognostic forest (optionally including the
#' propensity score \eqn{e}) and \eqn{\tau} is a treatment effect
#' forest.  Both forests use a standard BART prior.
#'
#' @param y Numeric vector of outcomes (length n).
#' @param X_train Numeric matrix of covariates (n x p).
#' @param treatment_indicator Integer vector of treatment
#'   assignments (0/1, length n).
#' @param propensity_score Optional numeric vector of estimated
#'   propensity scores (length n).  Appended as an extra covariate
#'   to the prognostic forest.
#' @param X_test Optional test covariate matrix (n_test x p).
#' @param treatment_indicator_test Integer vector (0/1) for test.
#'   Defaults to all-ones if \code{NULL}.
#' @param propensity_score_test Optional propensity scores for
#'   the test set (length n_test).
#' @param number_of_trees_prog,number_of_trees_treat Number of
#'   trees in the prognostic / treatment forest.
#' @param power,base Tree topology prior parameters.
#' @param p_grow,p_prune Probabilities of grow / prune proposals.
#' @param nu Degrees of freedom for the inverse-chi-squared prior
#'   on sigma^2.
#' @param q Quantile for setting the scale parameter lambda.
#' @param sigma Optional fixed sigma.  If \code{NULL}, sigma is
#'   estimated.
#' @param N_post,N_burn Number of posterior / burn-in MCMC
#'   iterations.
#' @param verbose Logical; print progress bar.
#' @param irs Integer IRS mode (applied to both forests):
#'   0 = off, 1 = skip-then-draw, 2 = draw-then-decide,
#'   3 = uniform random routing.
#' @param store_posterior_sample Logical; if \code{TRUE}, return
#'   the full posterior sample matrices for test predictions.
#'
#' @return A named list with:
#' \describe{
#'   \item{train_predictions}{Posterior mean of
<<<<<<< HEAD
#'     mu(x,e) + b*tau(x) (length n).}
#'   \item{test_predictions}{Posterior mean of
#'     mu(x,e) + b*tau(x) (length n_test).}
=======
#'     mu(x,e) + b * tau(x) (length n).}
#'   \item{test_predictions}{Posterior mean of
#'     mu(x,e) + b * tau(x) (length n_test).}
>>>>>>> borrow
#'   \item{train_predictions_prog}{Posterior mean of mu(x,e).}
#'   \item{test_predictions_prog}{Posterior mean of mu(x,e).}
#'   \item{train_predictions_treat}{Posterior mean of tau(x)
#'     (CATE estimates on training set).}
#'   \item{test_predictions_treat}{Posterior mean of tau(x)
#'     (CATE estimates on test set).}
#'   \item{sigma}{Posterior samples of sigma.}
#'   \item{acceptance_ratio_prog}{Acceptance rate (prognostic).}
#'   \item{acceptance_ratio_treat}{Acceptance rate (treatment).}
#'   \item{test_predictions_treat_sample}{(If
#'     \code{store_posterior_sample}) N_post x n_test matrix of
#'     per-iteration tau(x) predictions.}
#'   \item{test_predictions_sample}{(If
#'     \code{store_posterior_sample}) N_post x n_test matrix of
#'     per-iteration total predictions.}
#' }
#'
<<<<<<< HEAD
=======
#' @examples
#' set.seed(1)
#' n <- 100
#' X <- matrix(rnorm(n * 3), n, 3)
#' a <- rbinom(n, 1, 0.5)
#' y <- X[, 1] + 0.5 * a + rnorm(n)
#' fit <- SimpleBCF(y = y, X_train = X, treatment_indicator = a,
#'                  N_post = 50, N_burn = 25, verbose = FALSE)
#' mean(fit$train_predictions_treat)  # average estimated CATE
#'
>>>>>>> borrow
#' @importFrom Rcpp evalCpp
#' @useDynLib FusionForests, .registration = TRUE
#' @importFrom stats sd qchisq
#' @export
SimpleBCF <- function(
  y,
  X_train,
  treatment_indicator,
  propensity_score      = NULL,
  X_test                = NULL,
  treatment_indicator_test = NULL,
  propensity_score_test = NULL,
  number_of_trees_prog  = 200,
  number_of_trees_treat = 50,
  power                 = 2.0,
  base                  = 0.95,
  p_grow                = 0.4,
  p_prune               = 0.4,
  nu                    = 3,
  q                     = 0.90,
  sigma                 = NULL,
  N_post                = 1000,
  N_burn                = 1000,
  verbose               = TRUE,
  irs                   = 0L,
  store_posterior_sample = FALSE
) {

  ## Input validation
  if (!is.matrix(X_train)) X_train <- as.matrix(X_train)
  n <- nrow(X_train)
  p <- ncol(X_train)
  if (length(y) != n)
    stop("length(y) must equal nrow(X_train).")
  if (length(treatment_indicator) != n)
    stop("length(treatment_indicator) must equal n.")

  ## Build prognostic design matrix (append propensity)
  if (!is.null(propensity_score)) {
    X_prog <- cbind(X_train, e = propensity_score)
  } else {
    X_prog <- X_train
  }
  X_treat <- X_train
  p_prog  <- ncol(X_prog)
  p_treat <- ncol(X_treat)

  ## Test data
  if (!is.null(X_test)) {
    if (!is.matrix(X_test)) X_test <- as.matrix(X_test)
    if (ncol(X_test) != p)
      stop("X_test must have same columns as X_train.")
    n_test <- nrow(X_test)

    if (!is.null(propensity_score_test)) {
      X_test_prog <- cbind(X_test,
                           e = propensity_score_test)
    } else if (!is.null(propensity_score)) {
      X_test_prog <- cbind(X_test,
                           e = rep(0.5, n_test))
    } else {
      X_test_prog <- X_test
    }
    X_test_treat <- X_test

    if (is.null(treatment_indicator_test))
      treatment_indicator_test <- rep(1L, n_test)
  } else {
    n_test <- 1L
    X_test_prog  <- matrix(colMeans(X_prog), nrow = 1)
    X_test_treat <- matrix(colMeans(X_treat), nrow = 1)
    treatment_indicator_test <- 1L
  }

  ## Centre y
  y       <- as.numeric(y)
  y_mean  <- mean(y)
  y_range <- max(y) - min(y)
  y       <- y - y_mean

  ## Sigma prior
  if (is.null(sigma)) {
    sigma_hat   <- sd(y)
    sigma_known <- FALSE
  } else {
    sigma_hat   <- sigma
    sigma_known <- TRUE
  }
  qchi   <- qchisq(1.0 - q, nu)
  lambda <- (sigma_hat^2 * qchi) / nu

  ## Leaf prior scales
  omega_prog  <- y_range /
    (4 * sqrt(number_of_trees_prog))
  omega_treat <- 0.5 * sd(y) /
    sqrt(number_of_trees_treat)

  ## Flatten matrices (row-major for C++)
  X_prog       <- as.numeric(t(X_prog))
  X_treat      <- as.numeric(t(X_treat))
  X_test_prog  <- as.numeric(t(X_test_prog))
  X_test_treat <- as.numeric(t(X_test_treat))

  ## Call C++
  fit <- SimpleBCF_cpp(
    nSEXP       = n,
    p_progSEXP  = p_prog,
    p_treatSEXP = p_treat,
    X_train_progSEXP  = X_prog,
    X_train_treatSEXP = X_treat,
    ySEXP = y,
    treatment_indicatorSEXP =
      as.integer(treatment_indicator),
    n_testSEXP = n_test,
    X_test_progSEXP  = X_test_prog,
    X_test_treatSEXP = X_test_treat,
    treatment_indicator_testSEXP =
      as.integer(treatment_indicator_test),
    no_trees_progSEXP  = number_of_trees_prog,
    no_trees_treatSEXP = number_of_trees_treat,
    powerSEXP   = power,
    baseSEXP    = base,
    p_growSEXP  = p_grow,
    p_pruneSEXP = p_prune,
    omega_progSEXP  = omega_prog,
    omega_treatSEXP = omega_treat,
    sigma_knownSEXP = sigma_known,
    sigmaSEXP   = sigma_hat,
    lambdaSEXP  = lambda,
    nuSEXP      = nu,
    N_postSEXP  = as.integer(N_post),
    N_burnSEXP  = as.integer(N_burn),
    verboseSEXP = verbose,
    irsSEXP     = as.integer(irs),
    store_posterior_sampleSEXP = store_posterior_sample
  )

  ## Back-transform: undo mean shift on prognostic + total
  fit$train_predictions      <- fit$train_predictions +
    y_mean
  fit$test_predictions       <- fit$test_predictions +
    y_mean
  fit$train_predictions_prog <- fit$train_predictions_prog +
    y_mean
  fit$test_predictions_prog  <- fit$test_predictions_prog +
    y_mean
  # treatment effect predictions: NO mean shift
  if (store_posterior_sample) {
    if (!is.null(fit$test_predictions_sample))
      fit$test_predictions_sample <-
        fit$test_predictions_sample + y_mean
  }

  ## Discard burn-in sigma draws
  if (!sigma_known) fit$sigma <- fit$sigma[-(1:N_burn)]

  fit
}
