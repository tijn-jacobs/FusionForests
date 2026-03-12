#' SimpleBART
#'
#' Standard single-forest BART model: \eqn{Y = f(X) + \sigma\varepsilon}.
#' A clean entry point for benchmarking and development (e.g., testing IRS).
#'
#' @param y Numeric vector of outcomes (length n).
#' @param X_train Numeric matrix of training covariates (n x p).
#' @param X_test  Optional numeric matrix of test covariates (n_test x p).
#'   If \code{NULL}, test predictions are returned at the column means.
#' @param number_of_trees Number of trees in the BART ensemble (default 200).
#' @param power,base Tree topology prior parameters.
#' @param p_grow,p_prune Probabilities of grow / prune proposals.
#' @param nu Degrees of freedom for the inverse-chi-squared prior on sigma^2.
#' @param q  Quantile for setting the scale parameter lambda.
#' @param sigma Optional fixed sigma.  If \code{NULL}, sigma is estimated.
#' @param N_post,N_burn Number of posterior / burn-in MCMC iterations.
#' @param verbose Logical; print progress bar.
#' @param irs Integer IRS mode: 0 = off (default), 1 = skip-then-draw
#'   (NaN obs excluded from MH ratio, routed after acceptance),
#'   2 = draw-then-decide (routing drawn before MH, all obs in ratio).
#'
#' @return A named list with:
#' \describe{
#'   \item{train_predictions}{Posterior mean fitted values (length n).}
#'   \item{test_predictions}{Posterior mean test predictions (length n_test).}
#'   \item{sigma}{Posterior samples of sigma (after burn-in).}
#'   \item{acceptance_ratio}{Mean tree-proposal acceptance rate.}
#' }
#'
#' @importFrom Rcpp evalCpp
#' @useDynLib FusionForests, .registration = TRUE
#' @importFrom stats sd qchisq
#' @export
SimpleBART <- function(
  y,
  X_train,
  X_test          = NULL,
  number_of_trees = 200,
  power           = 2.0,
  base            = 0.95,
  p_grow          = 0.4,
  p_prune         = 0.4,
  nu              = 3,
  q               = 0.90,
  sigma           = NULL,
  N_post          = 1000,
  N_burn          = 1000,
  verbose         = TRUE,
  irs             = 0L
) {

  ## Input validation
  if (!is.matrix(X_train)) X_train <- as.matrix(X_train)
  n <- nrow(X_train)
  p <- ncol(X_train)
  if (length(y) != n) stop("length(y) must equal nrow(X_train).")

  ## Test data
  if (!is.null(X_test)) {
    if (!is.matrix(X_test)) X_test <- as.matrix(X_test)
    if (ncol(X_test) != p) stop("X_test must have the same number of columns as X_train.")
    n_test <- nrow(X_test)
    X_test <- as.numeric(t(X_test))
  } else {
    n_test <- 1L
    X_test <- as.numeric(colMeans(X_train))
  }

  ## Centre y at its mean; scale omega to span the observed range
  y       <- as.numeric(y)
  y_min   <- min(y)
  y_max   <- max(y)
  y_mean  <- mean(y)
  y_range <- y_max - y_min
  y       <- y - y_mean

  ## Sigma prior: lambda calibrated on the centred scale
  if (is.null(sigma)) {
    sigma_hat   <- sd(y)
    sigma_known <- FALSE
  } else {
    sigma_hat   <- sigma
    sigma_known <- TRUE
  }

  qchi   <- qchisq(1.0 - q, nu)
  lambda <- (sigma_hat^2 * qchi) / nu

  ## Flatten training matrix (row-major for C++)
  X_train <- as.numeric(t(X_train))

  ## Leaf prior scale: 2-sigma interval covers half the observed range, k = 2
  omega <- y_range / (4 * sqrt(number_of_trees))

  ## Call C++
  fit <- SimpleBART_cpp(
    nSEXP         = n,
    pSEXP         = p,
    X_trainSEXP   = X_train,
    ySEXP         = y,
    n_testSEXP    = n_test,
    X_testSEXP    = X_test,
    no_treesSEXP  = number_of_trees,
    powerSEXP     = power,
    baseSEXP      = base,
    p_growSEXP    = p_grow,
    p_pruneSEXP   = p_prune,
    omegaSEXP     = omega,
    sigma_knownSEXP = sigma_known,
    sigmaSEXP     = sigma_hat,
    lambdaSEXP    = lambda,
    nuSEXP        = nu,
    N_postSEXP    = as.integer(N_post),
    N_burnSEXP    = as.integer(N_burn),
    verboseSEXP   = verbose,
    irsSEXP       = as.integer(irs)
  )

  ## Back-transform: undo the mean shift
  fit$train_predictions <- fit$train_predictions + y_mean
  fit$test_predictions  <- fit$test_predictions  + y_mean

  ## Discard burn-in sigma draws
  if (!sigma_known) fit$sigma <- fit$sigma[-(1:N_burn)]

  fit
}
