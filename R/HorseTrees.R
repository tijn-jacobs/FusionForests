#' Horseshoe Regression Trees (HorseTrees)
#'
#' Fits a Bayesian Horseshoe Trees model with a single learner.
#' Implements regularization on the step heights using a global-local Horseshoe
#' prior, controlled via the parameter \code{k}. Supports continuous, binary,
#' right-censored, and interval-censored (survival) outcomes.
#'
#' For continuous outcomes, the model centers and optionally standardizes the
#' outcome using a prior guess of the standard deviation.
#' For binary outcomes, the function uses a probit link formulation.
#' For right-censored outcomes (survival data), the function can handle
#' follow-up times either on the original time scale or log-transformed.
#' For interval-censored outcomes, provide \code{left_time} and
#' \code{right_time} instead of \code{y} and \code{status}; the event
#' indicators are derived internally following the
#' \code{survival::Surv(type = "interval2")} convention.
#' Generalized implementation with multiple prior possibilities is given by
#' \code{\link{ShrinkageTrees}}.
#' 
#' @examples
#' # Minimal example: continuous outcome
#' n <- 25
#' p <- 5
#' X <- matrix(rnorm(n * p), ncol = p)
#' y <- X[, 1] + rnorm(n)
#' fit1 <- HorseTrees(y = y, X_train = X, outcome_type = "continuous", 
#'                    number_of_trees = 5, N_post = 75, N_burn = 25, 
#'                    verbose = FALSE)
#'
#' # Minimal example: binary outcome
#' X <- matrix(rnorm(n * p), ncol = p)
#' y <- ifelse(X[, 1] + rnorm(n) > 0, 1, 0)
#' fit2 <- HorseTrees(y = y, X_train = X, outcome_type = "binary", 
#'                    number_of_trees = 5, N_post = 75, N_burn = 25, 
#'                    verbose = FALSE)
#'
#' # Minimal example: right-censored outcome
#' X <- matrix(rnorm(n * p), ncol = p)
#' time <- rexp(n, rate = 0.1)
#' status <- rbinom(n, 1, 0.7)
#' fit3 <- HorseTrees(y = time, status = status, X_train = X,
#'                    outcome_type = "right-censored", number_of_trees = 5,
#'                    N_post = 75, N_burn = 25, verbose = FALSE)
#'
#' # Minimal example: interval-censored outcome
#' X <- matrix(rnorm(n * p), ncol = p)
#' true_t <- rexp(n, rate = 0.1)
#' left_t  <- true_t * runif(n, 0.5, 1)
#' right_t <- true_t * runif(n, 1, 1.5)
#' # Mark some as exact, some as right-censored
#' exact <- sample(n, 8); left_t[exact] <- true_t[exact]; right_t[exact] <- true_t[exact]
#' rc <- sample(setdiff(seq_len(n), exact), 5); right_t[rc] <- Inf
#' fit4 <- HorseTrees(left_time = left_t, right_time = right_t, X_train = X,
#'                    outcome_type = "interval-censored", number_of_trees = 5,
#'                    N_post = 75, N_burn = 25, verbose = FALSE)
#'
#' # Larger continuous example (not run automatically)
#' \donttest{
#' n <- 100
#' p <- 100
#' X <- matrix(rnorm(100 * p), ncol = p)
#' X_test <- matrix(rnorm(50 * p), ncol = p)
#' y <- X[, 1] + X[, 2] - X[, 3] + rnorm(100, sd = 0.5)
#'
#' fit5 <- HorseTrees(y = y,
#'                    X_train = X,
#'                    X_test = X_test,
#'                    outcome_type = "continuous",
#'                    number_of_trees = 200,
#'                    N_post = 2500,
#'                    N_burn = 2500,
#'                    store_posterior_sample = TRUE,
#'                    verbose = TRUE)
#'
#' plot(fit4$sigma, type = "l", ylab = expression(sigma),
#'      xlab = "Iteration", main = "Sigma traceplot")
#'
#' hist(fit4$train_predictions_sample[, 1],
#'      main = "Posterior distribution of prediction outcome individual 1",
#'      xlab = "Prediction", breaks = 20)
#' }
#'                        
#' @param y Outcome vector. Numeric. Can represent continuous outcomes, binary
#' outcomes (0/1), or follow-up times for survival data. Set to \code{NULL}
#' (default) when using \code{outcome_type = "interval-censored"}, as values
#' are derived from \code{left_time} and \code{right_time}.
#' @param status Optional censoring indicator vector (1 = event occurred,
#' 0 = censored). Required if \code{outcome_type = "right-censored"}.
#' For interval-censored outcomes, this is derived automatically from
#' \code{left_time} and \code{right_time}.
#' @param X_train Covariate matrix for training. Each row corresponds to an
#' observation, and each column to a covariate.
#' @param X_test Optional covariate matrix for test data. If NULL, defaults to
#' the mean of the training covariates.
#' @param left_time Optional numeric vector of left (lower) time boundaries.
#' Required when \code{outcome_type = "interval-censored"}. Exact events
#' have \code{left_time == right_time}; right-censored observations have
#' \code{right_time = Inf}; interval-censored observations have finite
#' \code{left_time < right_time}.
#' @param right_time Optional numeric vector of right (upper) time boundaries.
#' Required when \code{outcome_type = "interval-censored"}. Use \code{Inf}
#' for right-censored observations.
#' @param outcome_type Type of outcome. One of \code{"continuous"},
#' \code{"binary"}, \code{"right-censored"}, or \code{"interval-censored"}.
#' @param timescale Indicates the scale of follow-up times. Options are 
#' `"time"` (nonnegative follow-up times, will be log-transformed internally) 
#'   or `"log"` (already log-transformed). Only used when 
#'   `outcome_type = "right-censored"` or `"interval-censored"`.
#' @param number_of_trees Number of trees in the ensemble. Default is 200.
#' @param k Horseshoe scale hyperparameter (default 0.1). This parameter 
#' controls the overall level of shrinkage by setting the scale for both 
#' global and local shrinkage components. The local and global hyperparameters 
#' are parameterized as 
#' \eqn{\alpha = \frac{k}{\sqrt{\mathrm{number\_of\_trees}}}} 
#' to ensure adaptive regularization across trees.
#' @param power Power parameter for tree structure prior. Default is 2.0.
#' @param base Base parameter for tree structure prior. Default is 0.95.
#' @param p_grow Probability of proposing a grow move. Default is 0.4.
#' @param p_prune Probability of proposing a prune move. Default is 0.4.
#' @param nu Degrees of freedom for the error distribution prior. Default is 3.
#' @param q Quantile hyperparameter for the error variance prior. 
#' Default is 0.90.
#' @param sigma Optional known value for error standard deviation. If NULL, 
#' estimated from data.
#' @param N_post Number of posterior samples to store. Default is 1000.
#' @param N_burn Number of burn-in iterations. Default is 1000.
#' @param delayed_proposal Number of delayed iterations before proposal. Only 
#' for reversible updates. Default is 5.
#' @param store_posterior_sample Logical; whether to store posterior samples for
#'  each iteration. Default is TRUE.
#' @param n_chains Number of independent MCMC chains to run. Default is
#'   \code{1} (standard single-chain behaviour). When \code{n_chains > 1} the
#'   chains are run in parallel via \code{parallel::mclapply} and their
#'   posterior samples are pooled into a single \code{ShrinkageTrees} object,
#'   so all existing \code{print}, \code{summary}, and \code{predict} methods
#'   work without modification. On Windows, \code{mclapply} falls back to
#'   sequential execution.
#' @param verbose Logical; whether to print verbose output. Default is TRUE.
#'
#' @return An S3 object of class \code{"ShrinkageTrees"} with the following elements:
#' \describe{
#'   \item{train_predictions}{Vector of posterior mean predictions on the 
#'   training data.}
#'   \item{test_predictions}{Vector of posterior mean predictions on the test 
#'   data (or on mean covariate vector if `X_test` not provided).}
#'   \item{sigma}{Vector of posterior samples of the error variance.}
#'   \item{acceptance_ratio}{Average acceptance ratio across trees during 
#'   sampling.}
#'   \item{train_predictions_sample}{Matrix of posterior samples of training 
#'   predictions (iterations in rows, observations in columns). Present only 
#'   if `store_posterior_sample = TRUE`.}
#'   \item{test_predictions_sample}{Matrix of posterior samples of test 
#'   predictions. Present only if `store_posterior_sample = TRUE`.}
#'   \item{train_probabilities}{Vector of posterior mean probabilities on the 
#'   training data (only for `outcome_type = "binary"`).}
#'   \item{test_probabilities}{Vector of posterior mean probabilities on the 
#'   test data (only for `outcome_type = "binary"`).}
#'   \item{train_probabilities_sample}{Matrix of posterior samples of training 
#'   probabilities (only for `outcome_type = "binary"` and if 
#'   `store_posterior_sample = TRUE`).}
#'   \item{test_probabilities_sample}{Matrix of posterior samples of test 
#'   probabilities (only for `outcome_type = "binary"` and if 
#'   `store_posterior_sample = TRUE`).}
#' }
#' 
#' @seealso
#' Model family: \code{\link{ShrinkageTrees}} (flexible prior choice),
#' \code{\link{CausalHorseForest}} (causal inference),
#' \code{\link{CausalShrinkageForest}} (causal, flexible prior).
#'
#' Survival wrappers: \code{\link{SurvivalBART}}, \code{\link{SurvivalDART}}.
#'
#' S3 methods: \code{\link{print.ShrinkageTrees}},
#' \code{\link{summary.ShrinkageTrees}},
#' \code{\link{predict.ShrinkageTrees}},
#' \code{\link{plot.ShrinkageTrees}}.
#' 
#' @importFrom Rcpp evalCpp
#' @useDynLib FusionForests, .registration = TRUE
#' @importFrom stats sd qchisq qnorm pnorm
#' @importFrom parallel mclapply detectCores
#' @export

HorseTrees <- function(y = NULL,
                       status = NULL,
                       X_train,
                       X_test = NULL,
                       left_time = NULL,
                       right_time = NULL,
                       outcome_type = "continuous",
                       timescale = "time",
                       number_of_trees = 200,
                       k = 0.1,
                       power = 2.0,
                       base = 0.95,
                       p_grow = 0.4,
                       p_prune = 0.4,
                       nu = 3,
                       q = 0.90,
                       sigma = NULL,
                       N_post = 1000,
                       N_burn = 1000,
                       delayed_proposal = 5,
                       store_posterior_sample = TRUE,
                       n_chains = 1,
                       verbose = TRUE) {

  # ── Multi-chain dispatch ──────────────────────────────────────────────────
  if (n_chains > 1) {
    mc <- match.call()

    chain_args <- list(
      y                      = y,
      status                 = status,
      X_train                = X_train,
      X_test                 = X_test,
      left_time              = left_time,
      right_time             = right_time,
      outcome_type           = outcome_type,
      timescale              = timescale,
      number_of_trees        = number_of_trees,
      k                      = k,
      power                  = power,
      base                   = base,
      p_grow                 = p_grow,
      p_prune                = p_prune,
      nu                     = nu,
      q                      = q,
      sigma                  = sigma,
      N_post                 = N_post,
      N_burn                 = N_burn,
      delayed_proposal       = delayed_proposal,
      store_posterior_sample = store_posterior_sample,
      n_chains               = 1,
      verbose                = FALSE
    )

    n_cores <- if (.Platform$OS.type == "windows") 1L
                else min(n_chains, parallel::detectCores(logical = FALSE))
    if (verbose)
      message("Running ", n_chains, " chains (", n_cores, " cores) ...")

    chains <- parallel::mclapply(
      seq_len(n_chains),
      function(i) do.call(HorseTrees, chain_args),
      mc.cores = n_cores
    )

    combined      <- .combine_chains(chains)
    combined$call <- mc
    return(combined)
  }

  # Check outcome_type value
  allowed_types <- c("continuous", "binary", "right-censored", "interval-censored")
  if (!outcome_type %in% allowed_types) {
    stop("Invalid outcome_type. Please choose 'continuous', 'binary',
         'right-censored', or 'interval-censored'.")
  }

  # Check that y is provided for non-interval-censored types
  if (is.null(y) && outcome_type != "interval-censored") {
    stop("'y' is required when outcome_type is not 'interval-censored'.")
  }

  # Check interval-censored arguments
  if (outcome_type == "interval-censored") {
    if (is.null(left_time) || is.null(right_time))
      stop("outcome_type = 'interval-censored' requires 'left_time' and 'right_time'.")
    if (length(left_time) != length(right_time))
      stop("'left_time' and 'right_time' must have the same length.")
    if (any(left_time > right_time))
      stop("All 'left_time' values must be <= corresponding 'right_time' values.")
    if (timescale == "time" && any(left_time <= 0))
      stop("left_time contains non-positive values, but timescale = 'time' requires strictly positive times.")
  }

  # Check consistency with status argument
  if (outcome_type == "right-censored" && is.null(status)) {
    stop("You specified outcome_type = 'right-censored', but did not provide a
         'status' vector.")
  }
  if (!is.null(status) && !is.null(y) && length(status) != length(y)) {
    stop("The length of 'status' must match the length of 'y'.")
  }

  if (!outcome_type %in% c("right-censored", "interval-censored") && !is.null(status)) {
    warning("You provided a 'status' vector, but outcome_type is not
            'right-censored' or 'interval-censored'. The 'status' vector will be ignored.")
  }

  # Check binary data consistency
  if (outcome_type == "binary" && !all(y %in% c(0, 1))) {
    stop("For outcome_type = 'binary', y must contain only 0 and 1.")
  }
  if (outcome_type != "binary" && !is.null(y) && all(y %in% c(0, 1))) {
    warning("The outcome y contains only 0 and 1, but outcome_type is not set to
            'binary'. Consider setting outcome_type = 'binary'.")
  }

  # Check survival data and timescale
  if (outcome_type == "right-censored" && timescale == "time" && any(y < 0)) {
    stop("Outcome contains negative values, but timescale = 'time' for survival
         data requires non-negative times.")
  }

  # Retrieve dimensions of training data
  n_train <- nrow(X_train)
  p_features <- ncol(X_train)

  # Check if dimensions covariates match with outcome
  if (outcome_type == "interval-censored") {
    if (length(left_time) != n_train)
      stop("The length of 'left_time' must match the number of rows in X_train.")
  } else {
    if (length(y) != n_train)
      stop("The length of outcome vector y must match the number of rows in X_train.")
  }
  
  # Check if dimensions match with test data
  if (!is.null(X_test)) {
    n_test <- nrow(X_test)
    if (ncol(X_test) != p_features) {
      stop("The number of covariates in X_test must match X_train.")
    }
    X_test <- as.numeric(t(X_test))
  } else {
    n_test <- 1
    X_test <- as.numeric(colMeans(X_train))
  }
    
  # Force data types to be numeric and plain arrays
  N_post <- as.integer(N_post)[1]
  N_burn <- as.integer(N_burn)[1]
  power <- as.numeric(power)[1]
  base <- as.numeric(base)[1]
  p_grow <- as.numeric(p_grow)[1]
  p_prune <- as.numeric(p_prune)[1]
  X_train_mat <- X_train                 # preserve matrix for predict() / vi plots
  X_train <- as.numeric(t(X_train))
  

  if (outcome_type == "right-censored") {

    # Convert y to numeric for C++ compatibility
    y <- as.numeric(y)

    # Log-transform survival times if timescale = "time"
    if (timescale == "time") {
      y <- log(y)
    }

    # Save y before centering/standardizing (for predict())
    y_train_raw <- y

    # Obtain estimated mean and standard deviation using censored_info()
    cens_inf <- censored_info(y, status)

    # Center the data using estimated mean
    y_mean <- cens_inf$mu
    y <- y - y_mean

    # Determine sigma (timescale parameter) and whether it is known
    if (is.null(sigma)) {
      sigma_hat <- cens_inf$sd
      sigma_known <- FALSE
    } else {
      sigma_hat <- sigma
      sigma_known <- TRUE
    }

    # Standardize the centered data
    y <- y / sigma_hat

    # Set survival flag
    survival <- TRUE

    # Compute lambda parameter for error distribution prior
    qchi <- qchisq(1.0 - q, nu)
    lambda <- (sigma_hat^2 * qchi) / nu

    # Fit a HorseTrees model
    fit <- HorseTrees_cpp(
      nSEXP = n_train,
      pSEXP = p_features,
      n_testSEXP = n_test,
      X_trainSEXP = X_train,
      ySEXP = y,
      status_indicatorSEXP = status,
      is_survivalSEXP = survival,
      observed_left_timeSEXP = numeric(n_train),
      observed_right_timeSEXP = y + 0,
      interval_censoring_indicatorSEXP = numeric(n_train),
      X_testSEXP = X_test,
      number_of_treesSEXP = number_of_trees,
      N_postSEXP = N_post,
      N_burnSEXP = N_burn,
      delayed_proposalSEXP = delayed_proposal,
      powerSEXP = power,
      baseSEXP = base,
      p_growSEXP = p_grow,
      p_pruneSEXP = p_prune,
      nuSEXP = nu,
      lambdaSEXP = lambda,
      dirichlet_boolSEXP = FALSE,
      a_dirichletSEXP = 1,
      b_dirichletSEXP = 1,
      rho_dirichletSEXP = 1,
      sigmaSEXP = sigma_hat,
      sigma_knownSEXP = sigma_known,
      omegaSEXP = 1,
      param1SEXP = k / sqrt(number_of_trees),
      param2SEXP = k / sqrt(number_of_trees),
      prior_typeSEXP = "horseshoe",
      reversibleSEXP = TRUE,
      store_parametersSEXP = FALSE,
      store_posterior_sampleSEXP = store_posterior_sample,
      verboseSEXP = verbose
    )
    
    if (timescale == "time") {
      fit$train_predictions <- exp(fit$train_predictions * sigma_hat + y_mean)
      fit$test_predictions <- exp(fit$test_predictions * sigma_hat + y_mean)
      if (store_posterior_sample) {
        fit$train_predictions_sample <- exp(fit$train_predictions_sample * sigma_hat + y_mean)
        fit$test_predictions_sample <- exp(fit$test_predictions_sample * sigma_hat + y_mean)
      }
    } else { # If timescale is "log"
      fit$train_predictions <- fit$train_predictions * sigma_hat + y_mean
      fit$test_predictions <- fit$test_predictions * sigma_hat + y_mean
      if (store_posterior_sample) { 
        fit$train_predictions_sample <- fit$train_predictions_sample * sigma_hat + y_mean
        fit$test_predictions_sample <- fit$test_predictions_sample * sigma_hat + y_mean
      }
    }

  # If interval-censored
  } else if (outcome_type == "interval-censored") {

    left_time <- as.numeric(left_time)
    right_time <- as.numeric(right_time)

    # Derive status: 1 if exact (left == right), 0 otherwise
    status <- as.integer(left_time == right_time)

    # Derive interval-censoring indicator: 1 if left < right and right is finite
    ic_indicator <- as.integer(left_time < right_time & is.finite(right_time))

    # Derive initial y values for MCMC
    y <- ifelse(status == 1, left_time,
                ifelse(ic_indicator == 1, (left_time + right_time) / 2,
                       left_time))

    # For right-censored obs (right_time == Inf): C++ expects finite boundary
    right_time[!is.finite(right_time)] <- left_time[!is.finite(right_time)]

    # Log-transform if timescale = "time"
    if (timescale == "time") {
      y <- log(y)
      left_time <- log(left_time)
      right_time <- log(right_time)
    }

    # Save before centering/standardizing (for predict())
    y_train_raw <- y
    left_time_raw <- left_time
    right_time_raw <- right_time

    # Estimate mu and sd using censored_info with interval-censored extension
    cens_inf <- censored_info(y, status, left_time = left_time,
                              right_time = right_time, ic_indicator = ic_indicator)

    # Center using estimated mean
    y_mean <- cens_inf$mu
    y <- y - y_mean
    left_time <- left_time - y_mean
    right_time <- right_time - y_mean

    # Determine sigma
    if (is.null(sigma)) {
      sigma_hat <- cens_inf$sd
      sigma_known <- FALSE
    } else {
      sigma_hat <- sigma
      sigma_known <- TRUE
    }

    # Standardize
    y <- y / sigma_hat
    left_time <- left_time / sigma_hat
    right_time <- right_time / sigma_hat

    survival <- TRUE

    qchi <- qchisq(1.0 - q, nu)
    lambda <- (sigma_hat^2 * qchi) / nu

    fit <- HorseTrees_cpp(
      nSEXP = n_train,
      pSEXP = p_features,
      n_testSEXP = n_test,
      X_trainSEXP = X_train,
      ySEXP = y,
      status_indicatorSEXP = status,
      is_survivalSEXP = survival,
      observed_left_timeSEXP = left_time,
      observed_right_timeSEXP = right_time,
      interval_censoring_indicatorSEXP = ic_indicator,
      X_testSEXP = X_test,
      number_of_treesSEXP = number_of_trees,
      N_postSEXP = N_post,
      N_burnSEXP = N_burn,
      delayed_proposalSEXP = delayed_proposal,
      powerSEXP = power,
      baseSEXP = base,
      p_growSEXP = p_grow,
      p_pruneSEXP = p_prune,
      nuSEXP = nu,
      lambdaSEXP = lambda,
      dirichlet_boolSEXP = FALSE,
      a_dirichletSEXP = 1,
      b_dirichletSEXP = 1,
      rho_dirichletSEXP = 1,
      sigmaSEXP = sigma_hat,
      sigma_knownSEXP = sigma_known,
      omegaSEXP = 1,
      param1SEXP = k / sqrt(number_of_trees),
      param2SEXP = k / sqrt(number_of_trees),
      prior_typeSEXP = "horseshoe",
      reversibleSEXP = TRUE,
      store_parametersSEXP = FALSE,
      store_posterior_sampleSEXP = store_posterior_sample,
      verboseSEXP = verbose
    )

    # Back-transform predictions (identical to right-censored)
    if (timescale == "time") {
      fit$train_predictions <- exp(fit$train_predictions * sigma_hat + y_mean)
      fit$test_predictions <- exp(fit$test_predictions * sigma_hat + y_mean)
      if (store_posterior_sample) {
        fit$train_predictions_sample <- exp(fit$train_predictions_sample * sigma_hat + y_mean)
        fit$test_predictions_sample <- exp(fit$test_predictions_sample * sigma_hat + y_mean)
      }
    } else {
      fit$train_predictions <- fit$train_predictions * sigma_hat + y_mean
      fit$test_predictions <- fit$test_predictions * sigma_hat + y_mean
      if (store_posterior_sample) {
        fit$train_predictions_sample <- fit$train_predictions_sample * sigma_hat + y_mean
        fit$test_predictions_sample <- fit$test_predictions_sample * sigma_hat + y_mean
      }
    }

  # If binary
  } else if (outcome_type == "binary") {
    y <- as.numeric(y)
    y_train_raw      <- y
    latent_threshold <- qnorm(mean(y))
    sigma_hat        <- 1
    sigma_known      <- TRUE
    y_mean           <- 0
    lambda           <- 0   # unused for probit
    fit <- probitHorseTrees_cpp(
      nSEXP = n_train,
      pSEXP = p_features,
      n_testSEXP = n_test,
      X_trainSEXP = X_train,
      ySEXP = y,
      X_testSEXP = X_test,
      number_of_treesSEXP = number_of_trees,
      N_postSEXP = N_post,
      N_burnSEXP = N_burn,
      delayed_proposalSEXP = delayed_proposal,
      powerSEXP = power,
      baseSEXP = base,
      p_growSEXP = p_grow,
      p_pruneSEXP = p_prune,
      omegaSEXP = 1,
      latent_thresholdSEXP = latent_threshold,
      param1SEXP = k / sqrt(number_of_trees),
      param2SEXP = k / sqrt(number_of_trees),
      prior_typeSEXP = "horseshoe",
      reversibleSEXP = TRUE,
      store_posterior_sampleSEXP = store_posterior_sample,
      verboseSEXP = verbose
    )

    # Add the pnorm transform
    fit$train_probabilities <- pnorm(fit$train_predictions)
    fit$test_probabilities<- pnorm(fit$test_predictions)
    if (store_posterior_sample) {
      fit$train_probabilities_sample <- pnorm(fit$train_predictions_sample)
      fit$test_probabilities_sample <- pnorm(fit$test_predictions_sample)
    }

  # Otherwise, continuous
  } else {

    # Force outcome to plain numeric vector
    y <- as.numeric(y)

    # Save y before centering/standardizing (for predict())
    y_train_raw <- y

    # Create dummy status vector (not used for continuous)
    status <- rep(1, n_train)
    survival <- FALSE

    # Determine prior guess of sigma 
    if (is.null(sigma)) {
      sigma_hat <- sd(y)      # Estimate sigma from data
      sigma_known <- FALSE
    } else {
      sigma_hat <- sigma      # Use provided sigma
      sigma_known <- TRUE
    }

    # Compute hyperparameters of error variance prior
    qchi <- qchisq(1.0 - q, nu)
    lambda <- (sigma_hat^2 * qchi) / nu

    # Compute mean and standardize  the outcome
    y_mean <- mean(y)
    y <- y - y_mean
    y <- y / sigma_hat  # Standardize y

    fit <- HorseTrees_cpp(
      nSEXP = n_train,
      pSEXP = p_features,
      n_testSEXP = n_test,
      X_trainSEXP = X_train,
      ySEXP = y,
      status_indicatorSEXP = status,
      is_survivalSEXP = survival,
      observed_left_timeSEXP = numeric(n_train),
      observed_right_timeSEXP = y + 0,
      interval_censoring_indicatorSEXP = numeric(n_train),
      X_testSEXP = X_test,
      number_of_treesSEXP = number_of_trees,
      N_postSEXP = N_post,
      N_burnSEXP = N_burn,
      delayed_proposalSEXP = delayed_proposal,
      powerSEXP = power,
      baseSEXP = base,
      p_growSEXP = p_grow,
      p_pruneSEXP = p_prune,
      nuSEXP = nu,
      lambdaSEXP = lambda,
      dirichlet_boolSEXP = FALSE,
      a_dirichletSEXP = 1,
      b_dirichletSEXP = 1,
      rho_dirichletSEXP = 1,
      sigmaSEXP = sigma_hat,
      sigma_knownSEXP = sigma_known,
      omegaSEXP = 1,
      param1SEXP = k / sqrt(number_of_trees),
      param2SEXP = k / sqrt(number_of_trees),
      prior_typeSEXP = "horseshoe",
      reversibleSEXP = TRUE,
      store_parametersSEXP = FALSE,
      store_posterior_sampleSEXP = store_posterior_sample,
      verboseSEXP = verbose
    )
    
    fit$train_predictions <- fit$train_predictions * sigma_hat + y_mean
    fit$test_predictions <- fit$test_predictions * sigma_hat + y_mean
    if (store_posterior_sample) {
      fit$train_predictions_sample <- fit$train_predictions_sample * sigma_hat + y_mean
      fit$test_predictions_sample <- fit$test_predictions_sample * sigma_hat + y_mean
    }
  }
  
  # remove burn-in of sigma
  if (!sigma_known) {
      fit$sigma <- fit$sigma[-(1:N_burn)]
    } 

    obj <- NewShrinkageTrees(
    fit              = fit,
    call             = match.call(),
    outcome_type     = outcome_type,
    timescale        = timescale,
    prior_type_user  = "horseshoe",
    prior_type_cpp   = "horseshoe",
    n_train          = n_train,
    p_features       = p_features,
    n_test           = n_test,
    test_provided    = !is.null(X_test),
    number_of_trees  = number_of_trees,
    N_post           = N_post,
    N_burn           = N_burn,
    store_posterior_sample = store_posterior_sample,
    sigma_hat        = sigma_hat,
    sigma_known      = sigma_known,
    y_mean           = y_mean,
    y_train          = y_train_raw,
    X_train          = X_train_mat,
    status_train     = if (outcome_type %in% c("right-censored", "interval-censored")) status else NULL,
    left_time_train  = if (outcome_type == "interval-censored") left_time_raw else NULL,
    right_time_train = if (outcome_type == "interval-censored") right_time_raw else NULL,
    ic_indicator_train = if (outcome_type == "interval-censored") ic_indicator else NULL,
    power            = power,
    base             = base,
    p_grow           = p_grow,
    p_prune          = p_prune,
    delayed_proposal = delayed_proposal,
    reversible       = TRUE,
    param1           = k / sqrt(number_of_trees),
    param2           = k / sqrt(number_of_trees),
    omega            = 1,
    a_dirichlet      = 1,
    b_dirichlet      = 1,
    rho_dirichlet    = 1,
    nu               = nu,
    lambda           = lambda,
    latent_threshold = if (outcome_type == "binary") latent_threshold else NULL
  )

  return(obj)
}

