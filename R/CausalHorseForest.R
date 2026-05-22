#' Causal Horseshoe Forests
#'
#' This function fits a (Bayesian) Causal Horseshoe Forest.
#' It can be used for estimation of conditional average treatments effects of
#' survival data given high-dimensional covariates. The outcome is decomposed in
#' a prognostic part (control) and a treatment effect part. For both of these,
#' we specify a Horseshoe Trees regression function.
#' Supports continuous, right-censored, and interval-censored outcomes.
#'
#' @param y Outcome vector. For survival, represents follow-up times (can be on
#' original or log scale depending on \code{timescale}). Set to \code{NULL}
#' when using \code{outcome_type = "interval-censored"}, as values are derived
#' from \code{left_time} and \code{right_time}.
#' @param status Optional event indicator vector (1 = event occurred,
#' 0 = censored). Required when \code{outcome_type = "right-censored"}.
#' For interval-censored outcomes, this is derived automatically from
#' \code{left_time} and \code{right_time}.
#' @param X_train_control Covariate matrix for the control forest. Rows 
#' correspond to samples, columns to covariates.
#' @param X_train_treat Covariate matrix for the treatment forest. Rows 
#' correspond to samples, columns to covariates.
#' @param treatment_indicator_train Vector indicating treatment assignment for 
#' training samples (1 = treated, 0 = control).
#' @param X_test_control Optional test covariate matrix for control forest. If 
#' \code{NULL}, defaults to column means of \code{X_train_control}.
#' @param X_test_treat Optional test covariate matrix for treatment forest. If 
#' \code{NULL}, defaults to column means of \code{X_train_treat}.
#' @param treatment_indicator_test Optional vector indicating treatment 
#' assignment for test samples.
#' @param left_time Optional numeric vector of left (lower) time boundaries.
#' Required when \code{outcome_type = "interval-censored"}. Exact events
#' have \code{left_time == right_time}; right-censored observations have
#' \code{right_time = Inf}; interval-censored observations have finite
#' \code{left_time < right_time}.
#' @param right_time Optional numeric vector of right (upper) time boundaries.
#' Required when \code{outcome_type = "interval-censored"}. Use \code{Inf}
#' for right-censored observations.
#' @param outcome_type Type of outcome: one of \code{"continuous"},
#' \code{"right-censored"}, or \code{"interval-censored"}.
#' Default is \code{"continuous"}.
#' @param timescale For survival outcomes: either \code{"time"} (original time
#' scale, log-transformed internally) or \code{"log"} (already log-transformed).
#' Used when \code{outcome_type} is \code{"right-censored"} or
#' \code{"interval-censored"}.
#' @param number_of_trees Number of trees in each forest. Default is 200.
#' @param k Horseshoe prior scale hyperparameter. Default is 0.1. Controls 
#' global-local shrinkage on step heights.
#' @param power Power parameter for tree structure prior. Default is 2.0.
#' @param base Base parameter for tree structure prior. Default is 0.95.
#' @param p_grow Probability of proposing a grow move. Default is 0.4.
#' @param p_prune Probability of proposing a prune move. Default is 0.4.
#' @param nu Degrees of freedom for the error variance prior. Default is 3.
#' @param q Quantile parameter for error variance prior. Default is 0.90.
#' @param sigma Optional known standard deviation of the outcome. If 
#' \code{NULL}, estimated from data.
#' @param N_post Number of posterior samples to store. Default is 5000.
#' @param N_burn Number of burn-in iterations. Default is 5000.
#' @param delayed_proposal Number of delayed iterations before proposal updates.
#'  Default is 5.
#' @param store_posterior_sample Logical; whether to store posterior samples of
#' predictions. Default is \code{FALSE}.
#' @param treatment_coding Treatment coding scheme for the two-forest model.
#'   One of \code{"centered"} (default), \code{"binary"}, \code{"adaptive"},
#'   or \code{"invariant"}.
#'   \code{"centered"} uses \eqn{b_i \in \{-1/2, 1/2\}};
#'   \code{"binary"} uses \eqn{b_i \in \{0, 1\}};
#'   \code{"adaptive"} uses \eqn{b_i = A_i - \hat{e}(x_i)} where
#'   \eqn{\hat{e}(x_i)} is the estimated propensity score;
#'   \code{"invariant"} treats \eqn{b_0, b_1} as parameters estimated within
#'   the Gibbs sampler with \eqn{b_j \sim N(0, 1/2)} priors, yielding a
#'   parameterisation-invariant model (Hahn et al., 2020, Section 5.2).
#' @param propensity Optional numeric vector of propensity scores
#'   \eqn{\hat{e}(x_i)} for training observations. Required when
#'   \code{treatment_coding = "adaptive"}.
#' @param propensity_test Optional numeric vector of propensity scores for
#'   test observations. Only used when \code{treatment_coding = "adaptive"}.
#'   Defaults to \code{0.5} for all test observations if not provided.
#' @param n_chains Number of independent MCMC chains to run. Default is
#'   \code{1} (standard single-chain behaviour). When \code{n_chains > 1} the
#'   chains are run in parallel via \code{parallel::mclapply} and their
#'   posterior samples are pooled into a single \code{CausalShrinkageForest}
#'   object, so all existing \code{print} and \code{summary} methods work
#'   without modification. On Windows, \code{mclapply} falls back to
#'   sequential execution.
#' @param verbose Logical; whether to print verbose output during sampling.
#' Default is \code{TRUE}.
#'
#' @return An S3 object of class \code{"CausalShrinkageForest"} containing:
#' \describe{
#'   \item{train_predictions}{Posterior mean predictions on training data 
#'   (combined forest).}
#'   \item{test_predictions}{Posterior mean predictions on test data 
#'   (combined forest).}
#'   \item{train_predictions_control}{Estimated control outcomes on training 
#'   data.}
#'   \item{test_predictions_control}{Estimated control outcomes on test data.}
#'   \item{train_predictions_treat}{Estimated treatment effects on training 
#'   data.}
#'   \item{test_predictions_treat}{Estimated treatment effects on test data.}
#'   \item{sigma}{Vector of posterior samples for the error standard deviation.}
#'   \item{acceptance_ratio_control}{Average acceptance ratio in control 
#'   forest.}
#'   \item{acceptance_ratio_treat}{Average acceptance ratio in treatment 
#'   forest.}
#'   \item{train_predictions_sample_control}{Matrix of posterior samples for 
#'   control predictions (if \code{store_posterior_sample = TRUE}).}
#'   \item{test_predictions_sample_control}{Matrix of posterior samples for 
#'   control predictions (if \code{store_posterior_sample = TRUE}).}
#'   \item{train_predictions_sample_treat}{Matrix of posterior samples for 
#'   treatment effects (if \code{store_posterior_sample = TRUE}).}
#'   \item{test_predictions_sample_treat}{Matrix of posterior samples for 
#'   treatment effects (if \code{store_posterior_sample = TRUE}).}
#' }
#'
#' @details
#' The model separately regularizes the control and treatment trees using
#' Horseshoe priors with global-local shrinkage on the step heights.
#' This approach is designed for robust estimation of heterogeneous treatment
#' effects in high-dimensional settings.
#' It supports continuous, right-censored, and interval-censored survival
#' outcomes. For interval-censored data, provide \code{left_time} and
#' \code{right_time} instead of \code{y} and \code{status}; the event
#' indicators are derived internally following the
#' \code{survival::Surv(type = "interval2")} convention.
#' 
#' @examples
#' # Example: Continuous outcome and homogeneous treatment effect
#' n <- 50
#' p <- 3
#' X_control <- matrix(runif(n * p), ncol = p)
#' X_treat <- matrix(runif(n * p), ncol = p)
#' treatment <- rbinom(n, 1, 0.5)
#' tau <- 2
#' y <- X_control[, 1] + (0.5 - treatment) * tau + rnorm(n)
#'
#' fit <- CausalHorseForest(
#'   y = y,
#'   X_train_control = X_control,
#'   X_train_treat = X_treat,
#'   treatment_indicator_train = treatment,
#'   outcome_type = "continuous",
#'   number_of_trees = 5,
#'   N_post = 10,
#'   N_burn = 5,
#'   store_posterior_sample = TRUE,
#'   verbose = FALSE
#' )
#'
#' \donttest{
#' ## Example: Right-censored survival outcome
#' # Set data dimensions
#' n <- 100
#' p <- 1000
#' 
#' # Generate covariates
#' X <- matrix(runif(n * p), ncol = p)
#' X_treat <- X
#' treatment <- rbinom(n, 1, pnorm(X[, 1] - 1/2))
#' 
#' # Generate true survival times depending on X and treatment
#' linpred <- X[, 1] - X[, 2] + (treatment - 0.5) * (1 + X[, 2] / 2 + X[, 3] / 3 
#'                                                   + X[, 4] / 4)
#' true_time <- linpred + rnorm(n, 0, 0.5)
#' 
#' # Generate censoring times
#' censor_time <- log(rexp(n, rate = 1 / 5))
#' 
#' # Observed times and event indicator
#' time_obs <- pmin(true_time, censor_time)
#' status <- as.numeric(true_time == time_obs)
#' 
#' # Estimate propensity score using HorseTrees
#' fit_prop <- HorseTrees(
#'   y = treatment,
#'   X_train = X,
#'   outcome_type = "binary",
#'   number_of_trees = 200,
#'   N_post = 1000,
#'   N_burn = 1000
#' )
#' 
#' # Retrieve estimated probability of treatment (propensity score)
#' propensity <- fit_prop$train_probabilities
#' 
#' # Combine propensity score with covariates for control forest
#' X_control <- cbind(propensity, X)
#' 
#' # Fit the Causal Horseshoe Forest for survival outcome
#' fit_surv <- CausalHorseForest(
#'   y = time_obs,
#'   status = status,
#'   X_train_control = X_control,
#'   X_train_treat = X_treat,
#'   treatment_indicator_train = treatment,
#'   outcome_type = "right-censored",
#'   timescale = "log",
#'   number_of_trees = 200,
#'   k = 0.1,
#'   N_post = 1000,
#'   N_burn = 1000,
#'   store_posterior_sample = TRUE
#' )
#' 
#' ## Evaluate and summarize results
#' 
#' # Evaluate C-index if survival package is available
#' if (requireNamespace("survival", quietly = TRUE)) {
#'   predicted_survtime <- fit_surv$train_predictions
#'   cindex_result <- survival::concordance(survival::Surv(time_obs, status) ~ predicted_survtime)
#'   c_index <- cindex_result$concordance
#'   cat("C-index:", round(c_index, 3), "\n")
#' } else {
#'   cat("Package 'survival' not available. Skipping C-index computation.\n")
#' }
#' 
#' # Compute posterior ATE samples
#' ate_samples <- rowMeans(fit_surv$train_predictions_sample_treat)
#' mean_ate <- mean(ate_samples)
#' ci_95 <- quantile(ate_samples, probs = c(0.025, 0.975))
#' 
#' cat("Posterior mean ATE:", round(mean_ate, 3), "\n")
#' cat("95% credible interval: [", round(ci_95[1], 3), ", ", round(ci_95[2], 3), "]\n", sep = "")
#' 
#' # Plot histogram of ATE samples
#' hist(
#'   ate_samples,
#'   breaks = 30,
#'   col = "steelblue",
#'   freq = FALSE,
#'   border = "white",
#'   xlab = "Average Treatment Effect (ATE)",
#'   main = "Posterior distribution of ATE"
#' )
#' abline(v = mean_ate, col = "orange3", lwd = 2)
#' abline(v = ci_95, col = "orange3", lty = 2, lwd = 2)
#' abline(v = 1.541667, col = "darkred", lwd = 2)
#' legend(
#'   "topright",
#'   legend = c("Mean", "95% CI", "Truth"),
#'   col = c("orange3", "orange3", "red"),
#'   lty = c(1, 2, 1),
#'   lwd = 2
#' )
#' 
#' ## Plot individual CATE estimates
#' 
#' # Summarize posterior distribution per patient
#' posterior_matrix <- fit_surv$train_predictions_sample_treat
#' posterior_mean <- colMeans(posterior_matrix)
#' posterior_ci <- apply(posterior_matrix, 2, quantile, probs = c(0.025, 0.975))
#' 
#' df_cate <- data.frame(
#'   mean = posterior_mean,
#'   lower = posterior_ci[1, ],
#'   upper = posterior_ci[2, ]
#' )
#' 
#' # Sort patients by posterior mean CATE
#' df_cate_sorted <- df_cate[order(df_cate$mean), ]
#' n_patients <- nrow(df_cate_sorted)
#' 
#' # Create the plot
#' plot(
#'   x = df_cate_sorted$mean,
#'   y = 1:n_patients,
#'   type = "n",
#'   xlab = "CATE per patient (95% credible interval)",
#'   ylab = "Patient index (sorted)",
#'   main = "Posterior CATE estimates",
#'   xlim = range(df_cate_sorted$lower, df_cate_sorted$upper)
#' )
#' 
#' # Add CATE intervals
#' segments(
#'   x0 = df_cate_sorted$lower,
#'   x1 = df_cate_sorted$upper,
#'   y0 = 1:n_patients,
#'   y1 = 1:n_patients,
#'   col = "steelblue"
#' )
#' 
#' # Add mean points
#' points(df_cate_sorted$mean, 1:n_patients, pch = 16, col = "orange3", lwd = 0.1)
#' 
#' # Add reference line at 0
#' abline(v = 0, col = "black", lwd = 2)
#' 
#' }
#' 
#' @seealso
#' Model family: \code{\link{HorseTrees}} (non-causal, horseshoe prior),
#' \code{\link{ShrinkageTrees}} (non-causal, flexible prior),
#' \code{\link{CausalShrinkageForest}} (causal, flexible prior).
#'
#' Survival wrappers: \code{\link{SurvivalBCF}}, \code{\link{SurvivalShrinkageBCF}}.
#'
#' S3 methods: \code{\link{print.CausalShrinkageForest}},
#' \code{\link{summary.CausalShrinkageForest}},
#' \code{\link{predict.CausalShrinkageForest}},
#' \code{\link{plot.CausalShrinkageForest}}.
#'
#' @importFrom Rcpp evalCpp
#' @useDynLib FusionForests, .registration = TRUE
#' @importFrom stats sd qchisq runif
#' @importFrom parallel mclapply detectCores
#' @export
CausalHorseForest <- function(y = NULL,
                              status = NULL,
                              X_train_control,
                              X_train_treat,
                              treatment_indicator_train,
                              X_test_control = NULL,
                              X_test_treat = NULL,
                              treatment_indicator_test = NULL,
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
                              N_post = 5000,
                              N_burn = 5000,
                              delayed_proposal = 5,
                              store_posterior_sample = FALSE,
                              treatment_coding = "centered",
                              propensity = NULL,
                              propensity_test = NULL,
                              n_chains = 1,
                              verbose = TRUE) {

  # ── Multi-chain dispatch ──────────────────────────────────────────────────
  if (n_chains > 1) {
    mc <- match.call()

    chain_args <- list(
      y                         = y,
      status                    = status,
      X_train_control           = X_train_control,
      X_train_treat             = X_train_treat,
      treatment_indicator_train = treatment_indicator_train,
      X_test_control            = X_test_control,
      X_test_treat              = X_test_treat,
      treatment_indicator_test  = treatment_indicator_test,
      left_time                 = left_time,
      right_time                = right_time,
      outcome_type              = outcome_type,
      timescale                 = timescale,
      number_of_trees           = number_of_trees,
      k                         = k,
      power                     = power,
      base                      = base,
      p_grow                    = p_grow,
      p_prune                   = p_prune,
      nu                        = nu,
      q                         = q,
      sigma                     = sigma,
      N_post                    = N_post,
      N_burn                    = N_burn,
      delayed_proposal          = delayed_proposal,
      store_posterior_sample    = store_posterior_sample,
      treatment_coding          = treatment_coding,
      propensity                = propensity,
      propensity_test           = propensity_test,
      n_chains                  = 1,
      verbose                   = FALSE
    )

    n_cores <- if (.Platform$OS.type == "windows") 1L
                else min(n_chains, parallel::detectCores(logical = FALSE))
    if (verbose)
      message("Running ", n_chains, " chains (", n_cores, " cores) ...")

    chains <- parallel::mclapply(
      seq_len(n_chains),
      function(i) do.call(CausalHorseForest, chain_args),
      mc.cores = n_cores
    )

    combined      <- .combine_causal_chains(chains)
    combined$call <- mc
    return(combined)
  }

  # Check outcome_type value
  allowed_types <- c("continuous", "right-censored", "interval-censored")
  if (!outcome_type %in% allowed_types) {
    stop("Invalid outcome_type. Please choose 'continuous', 'right-censored',
         or 'interval-censored'.")
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

  # Check survival data and timescale
  if (outcome_type == "right-censored" && timescale == "time" && any(y < 0)) {
    stop("Outcome contains negative values, but timescale = 'time' for survival
         data requires non-negative times.")
  }
  
  # Coerce to matrix so nrow/ncol are always defined
  if (!is.matrix(X_train_control)) X_train_control <- as.matrix(X_train_control)
  if (!is.matrix(X_train_treat))   X_train_treat   <- as.matrix(X_train_treat)

  # Retrieve dimensions of training data
  n_train   <- nrow(X_train_control)
  p_control <- ncol(X_train_control)
  p_treat   <- ncol(X_train_treat)

  # Check matching row numbers
  n_obs <- if (outcome_type == "interval-censored") length(left_time) else length(y)
  if (nrow(X_train_control) != n_obs) {
    stop("X_train_control rows must match length of outcome data.")
  }
  if (nrow(X_train_treat) != n_obs) {
    stop("X_train_treat rows must match length of outcome data.")
  }
  if (length(treatment_indicator_train) != n_obs) {
    stop("treatment_indicator_train must match length of outcome data.")
  }
  if (!all(treatment_indicator_train %in% c(0, 1))) {
    stop("treatment_indicator_train must contain only 0 and 1.")
  }

  # Validate treatment_coding
  treatment_coding <- match.arg(treatment_coding, c("centered", "binary", "adaptive", "invariant"))
  if (treatment_coding == "adaptive" && is.null(propensity)) {
    stop("propensity scores are required when treatment_coding = 'adaptive'.")
  }
  if (!is.null(propensity) && length(propensity) != n_obs) {
    stop("Length of 'propensity' must match the number of training observations.")
  }

  # Prepare propensity vectors for C++
  propensity_train_cpp <- if (!is.null(propensity)) as.numeric(propensity) else numeric(n_obs)

  # If test provided, check
  test_provided_flag <- !is.null(X_test_control) && !is.null(X_test_treat)
  if (!is.null(X_test_control) && !is.null(X_test_treat)) {
    n_test <- nrow(X_test_control)
    if (ncol(X_test_control) != p_control || ncol(X_test_treat) != p_treat) {
      stop("Number of columns in X_test_control or X_test_treat does not match 
           training data.")
    }
    if (!is.null(treatment_indicator_test) && 
        length(treatment_indicator_test) != n_test) {
      stop("treatment_indicator_test length must match number of test rows.")
    }
    X_test_control <- as.numeric(t(X_test_control))
    X_test_treat <- as.numeric(t(X_test_treat))
  } else {
    n_test <- 1
    X_test_control <- as.numeric(colMeans(X_train_control))
    X_test_treat <- as.numeric(colMeans(X_train_treat))
    treatment_indicator_test <- as.integer(rep(1, n_test))
  }
  
  # Treatment indicators as integer
  treatment_indicator_train <- as.integer(treatment_indicator_train)
  if (is.null(treatment_indicator_test)) {
    treatment_indicator_test <- as.integer(rep(1, n_test))
  } else {
    treatment_indicator_test <- as.integer(treatment_indicator_test)
  }

  # Prepare propensity for test data
  propensity_test_cpp <- if (!is.null(propensity_test)) {
    as.numeric(propensity_test)
  } else {
    rep(0.5, n_test)
  }

  prior_type_control_user <- "horseshoe"
  prior_type_treat_user <- "horseshoe"
  
  # Force data types to be numeric and plain arrays
  N_post <- as.integer(N_post)[1]
  N_burn <- as.integer(N_burn)[1]
  power <- as.numeric(power)[1]
  base <- as.numeric(base)[1]
  p_grow <- as.numeric(p_grow)[1]
  p_prune <- as.numeric(p_prune)[1]
  X_control_mat <- X_train_control        # preserve matrix for predict() / vi plots
  X_treat_mat   <- X_train_treat
  X_train_treat <- as.numeric(t(X_train_treat))
  X_train_control <- as.numeric(t(X_train_control))

  # For interval-censored: derive y/status/ic_indicator from left_time/right_time
  # before saving originals
  if (outcome_type == "interval-censored") {
    left_time <- as.numeric(left_time)
    right_time <- as.numeric(right_time)

    status <- as.integer(left_time == right_time)
    ic_indicator <- as.integer(left_time < right_time & is.finite(right_time))

    y <- ifelse(status == 1, left_time,
                ifelse(ic_indicator == 1, (left_time + right_time) / 2,
                       left_time))

    # C++ expects finite boundary for right-censored obs
    right_time[!is.finite(right_time)] <- left_time[!is.finite(right_time)]
  }

  # Save originals for data slot (used by predict())
  y_horse_raw      <- y
  status_horse_raw <- status
  left_time_raw    <- if (outcome_type == "interval-censored") left_time else NULL
  right_time_raw   <- if (outcome_type == "interval-censored") right_time else NULL

  if (outcome_type == "right-censored") {

    # Convert y to numeric for C++ compatibility
    y <- as.numeric(y)

    # Log-transform survival times if timescale = "time"
    if (timescale == "time") {
      y <- log(y)
    }

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
    
    # Fit a Causal Horseshoe Forest
    fit <- CausalHorseForest_cpp(
      nSEXP = n_train,
      p_treatSEXP = p_treat,
      p_controlSEXP = p_control,
      X_train_treatSEXP = X_train_treat,
      X_train_controlSEXP = X_train_control,
      ySEXP = y,
      status_indicatorSEXP = status,
      is_survivalSEXP = survival,
      observed_left_timeSEXP = numeric(n_train),
      observed_right_timeSEXP = y + 0,
      interval_censoring_indicatorSEXP = numeric(n_train),
      treatment_indicatorSEXP = treatment_indicator_train,
      n_testSEXP = n_test,
      X_test_controlSEXP = X_test_control,
      X_test_treatSEXP = X_test_treat,
      treatment_indicator_testSEXP = treatment_indicator_test,
      no_trees_treatSEXP = number_of_trees,
      power_treatSEXP = power,
      base_treatSEXP = base,
      p_grow_treatSEXP = p_grow,
      p_prune_treatSEXP = p_prune,
      omega_treatSEXP = 1/2,
      prior_type_treatSEXP = "horseshoe",
      param1_treatSEXP = k / sqrt(number_of_trees),
      param2_treatSEXP = k / sqrt(number_of_trees),
      reversible_treatSEXP = TRUE,
      dirichlet_bool_treatSEXP = FALSE,
      a_dirichlet_treatSEXP = 1.0,
      b_dirichlet_treatSEXP = 1.0, 
      rho_dirichlet_treatSEXP = 1.0,  
      no_trees_controlSEXP = number_of_trees,
      power_controlSEXP = power,
      base_controlSEXP = base,
      p_grow_controlSEXP = p_grow,
      p_prune_controlSEXP = p_prune,
      omega_controlSEXP = 1/2,
      prior_type_controlSEXP = "horseshoe",
      param1_controlSEXP = k / sqrt(number_of_trees),
      param2_controlSEXP = k / sqrt(number_of_trees),
      reversible_controlSEXP = TRUE,
      dirichlet_bool_controlSEXP = FALSE,
      a_dirichlet_controlSEXP = 1.0,
      b_dirichlet_controlSEXP = 1.0, 
      rho_dirichlet_controlSEXP = 1.0,  
      sigma_knownSEXP = sigma_known,
      sigmaSEXP = sigma_hat,
      lambdaSEXP = lambda,
      nuSEXP = nu,
      N_postSEXP = N_post,
      N_burnSEXP = N_burn,
      delayed_proposalSEXP = delayed_proposal,
      store_posterior_sample_controlSEXP = store_posterior_sample,
      store_posterior_sample_treatSEXP = store_posterior_sample,
      verboseSEXP = verbose,
      treatment_codingSEXP = treatment_coding,
      propensity_trainSEXP = propensity_train_cpp,
      propensity_testSEXP = propensity_test_cpp
    )
   
    if (timescale == "time") {
      
      # Total
      fit$train_predictions <- exp(fit$train_predictions * sigma_hat + y_mean)
      fit$test_predictions <- exp(fit$test_predictions * sigma_hat + y_mean)
      
      # Control
      fit$train_predictions_control <- exp(fit$train_predictions_control * 
                                             sigma_hat + y_mean)
      fit$test_predictions_control <- exp(fit$test_predictions_control * 
                                            sigma_hat + y_mean)
      
      # Treatment
      fit$train_predictions_treat <- exp(fit$train_predictions_treat * 
                                           sigma_hat)
      fit$test_predictions_treat <- exp(fit$test_predictions_treat * sigma_hat)
      
      # Posterior samples
      if (store_posterior_sample) { #
        fit$train_predictions_sample_control <- 
          exp(fit$train_predictions_sample_control * sigma_hat + y_mean)
        fit$test_predictions_sample_control <- 
          exp(fit$test_predictions_sample_control * sigma_hat + y_mean)
        fit$train_predictions_sample_treat <- 
          exp(fit$train_predictions_sample_treat * sigma_hat)
        fit$test_predictions_sample_treat <- 
          exp(fit$test_predictions_sample_treat * sigma_hat)
      }
    } else {
      # Total
      fit$train_predictions <- fit$train_predictions * sigma_hat + y_mean
      fit$test_predictions <- fit$test_predictions * sigma_hat + y_mean
      
      # Control
      fit$train_predictions_control <- 
        fit$train_predictions_control * sigma_hat + y_mean
      fit$test_predictions_control <- 
        fit$test_predictions_control * sigma_hat + y_mean
      
      # Treatment
      fit$train_predictions_treat <- fit$train_predictions_treat * sigma_hat
      fit$test_predictions_treat <- fit$test_predictions_treat * sigma_hat
      
      # Posterior samples
      if (store_posterior_sample) {
        fit$train_predictions_sample_control <- 
          fit$train_predictions_sample_control * sigma_hat + y_mean
        fit$test_predictions_sample_control <- 
          fit$test_predictions_sample_control * sigma_hat + y_mean
        fit$train_predictions_sample_treat <- 
          fit$train_predictions_sample_treat * sigma_hat
        fit$test_predictions_sample_treat <- 
          fit$test_predictions_sample_treat * sigma_hat
      }
    }
    
  # If interval-censored
  } else if (outcome_type == "interval-censored") {

    y <- as.numeric(y)

    # Log-transform if timescale = "time"
    if (timescale == "time") {
      y <- log(y)
      left_time <- log(left_time)
      right_time <- log(right_time)
    }

    # Estimate mu and sd
    cens_inf <- censored_info(y, status, left_time = left_time,
                              right_time = right_time, ic_indicator = ic_indicator)

    y_mean <- cens_inf$mu
    y <- y - y_mean
    left_time <- left_time - y_mean
    right_time <- right_time - y_mean

    if (is.null(sigma)) {
      sigma_hat <- cens_inf$sd
      sigma_known <- FALSE
    } else {
      sigma_hat <- sigma
      sigma_known <- TRUE
    }

    y <- y / sigma_hat
    left_time <- left_time / sigma_hat
    right_time <- right_time / sigma_hat

    survival <- TRUE

    qchi <- qchisq(1.0 - q, nu)
    lambda <- (sigma_hat^2 * qchi) / nu

    fit <- CausalHorseForest_cpp(
      nSEXP = n_train,
      p_treatSEXP = p_treat,
      p_controlSEXP = p_control,
      X_train_treatSEXP = X_train_treat,
      X_train_controlSEXP = X_train_control,
      ySEXP = y,
      status_indicatorSEXP = status,
      is_survivalSEXP = survival,
      observed_left_timeSEXP = left_time,
      observed_right_timeSEXP = right_time,
      interval_censoring_indicatorSEXP = ic_indicator,
      treatment_indicatorSEXP = treatment_indicator_train,
      n_testSEXP = n_test,
      X_test_controlSEXP = X_test_control,
      X_test_treatSEXP = X_test_treat,
      treatment_indicator_testSEXP = treatment_indicator_test,
      no_trees_treatSEXP = number_of_trees,
      power_treatSEXP = power,
      base_treatSEXP = base,
      p_grow_treatSEXP = p_grow,
      p_prune_treatSEXP = p_prune,
      omega_treatSEXP = 1/2,
      prior_type_treatSEXP = "horseshoe",
      param1_treatSEXP = k / sqrt(number_of_trees),
      param2_treatSEXP = k / sqrt(number_of_trees),
      reversible_treatSEXP = TRUE,
      dirichlet_bool_treatSEXP = FALSE,
      a_dirichlet_treatSEXP = 1.0,
      b_dirichlet_treatSEXP = 1.0,
      rho_dirichlet_treatSEXP = 1.0,
      no_trees_controlSEXP = number_of_trees,
      power_controlSEXP = power,
      base_controlSEXP = base,
      p_grow_controlSEXP = p_grow,
      p_prune_controlSEXP = p_prune,
      omega_controlSEXP = 1/2,
      prior_type_controlSEXP = "horseshoe",
      param1_controlSEXP = k / sqrt(number_of_trees),
      param2_controlSEXP = k / sqrt(number_of_trees),
      reversible_controlSEXP = TRUE,
      dirichlet_bool_controlSEXP = FALSE,
      a_dirichlet_controlSEXP = 1.0,
      b_dirichlet_controlSEXP = 1.0,
      rho_dirichlet_controlSEXP = 1.0,
      sigma_knownSEXP = sigma_known,
      sigmaSEXP = sigma_hat,
      lambdaSEXP = lambda,
      nuSEXP = nu,
      N_postSEXP = N_post,
      N_burnSEXP = N_burn,
      delayed_proposalSEXP = delayed_proposal,
      store_posterior_sample_controlSEXP = store_posterior_sample,
      store_posterior_sample_treatSEXP = store_posterior_sample,
      verboseSEXP = verbose,
      treatment_codingSEXP = treatment_coding,
      propensity_trainSEXP = propensity_train_cpp,
      propensity_testSEXP = propensity_test_cpp
    )

    # Back-transform (identical to right-censored)
    if (timescale == "time") {
      fit$train_predictions <- exp(fit$train_predictions * sigma_hat + y_mean)
      fit$test_predictions <- exp(fit$test_predictions * sigma_hat + y_mean)
      fit$train_predictions_control <- exp(fit$train_predictions_control *
                                             sigma_hat + y_mean)
      fit$test_predictions_control <- exp(fit$test_predictions_control *
                                            sigma_hat + y_mean)
      fit$train_predictions_treat <- exp(fit$train_predictions_treat * sigma_hat)
      fit$test_predictions_treat <- exp(fit$test_predictions_treat * sigma_hat)
      if (store_posterior_sample) {
        fit$train_predictions_sample_control <-
          exp(fit$train_predictions_sample_control * sigma_hat + y_mean)
        fit$test_predictions_sample_control <-
          exp(fit$test_predictions_sample_control * sigma_hat + y_mean)
        fit$train_predictions_sample_treat <-
          exp(fit$train_predictions_sample_treat * sigma_hat)
        fit$test_predictions_sample_treat <-
          exp(fit$test_predictions_sample_treat * sigma_hat)
      }
    } else {
      fit$train_predictions <- fit$train_predictions * sigma_hat + y_mean
      fit$test_predictions <- fit$test_predictions * sigma_hat + y_mean
      fit$train_predictions_control <-
        fit$train_predictions_control * sigma_hat + y_mean
      fit$test_predictions_control <-
        fit$test_predictions_control * sigma_hat + y_mean
      fit$train_predictions_treat <- fit$train_predictions_treat * sigma_hat
      fit$test_predictions_treat <- fit$test_predictions_treat * sigma_hat
      if (store_posterior_sample) {
        fit$train_predictions_sample_control <-
          fit$train_predictions_sample_control * sigma_hat + y_mean
        fit$test_predictions_sample_control <-
          fit$test_predictions_sample_control * sigma_hat + y_mean
        fit$train_predictions_sample_treat <-
          fit$train_predictions_sample_treat * sigma_hat
        fit$test_predictions_sample_treat <-
          fit$test_predictions_sample_treat * sigma_hat
      }
    }

     # Otherwise, continuous
  } else {

    # Force outcome to plain numeric vector
    y <- as.numeric(y)

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
    
    # Fit a Causal Horseshoe Forest
    fit <- CausalHorseForest_cpp(
      nSEXP = n_train,
      p_treatSEXP = p_treat,
      p_controlSEXP = p_control,
      X_train_treatSEXP = X_train_treat,
      X_train_controlSEXP = X_train_control,
      ySEXP = y,
      status_indicatorSEXP = status,
      is_survivalSEXP = survival,
      observed_left_timeSEXP = numeric(n_train),
      observed_right_timeSEXP = y + 0,
      interval_censoring_indicatorSEXP = numeric(n_train),
      treatment_indicatorSEXP = treatment_indicator_train,
      n_testSEXP = n_test,
      X_test_controlSEXP = X_test_control,
      X_test_treatSEXP = X_test_treat,
      treatment_indicator_testSEXP = treatment_indicator_test,
      no_trees_treatSEXP = number_of_trees,
      power_treatSEXP = power,
      base_treatSEXP = base,
      p_grow_treatSEXP = p_grow,
      p_prune_treatSEXP = p_prune,
      omega_treatSEXP = 1/2,
      prior_type_treatSEXP = "horseshoe",
      param1_treatSEXP = k / sqrt(number_of_trees),
      param2_treatSEXP = k / sqrt(number_of_trees),
      reversible_treatSEXP = TRUE,
      dirichlet_bool_treatSEXP = FALSE,
      a_dirichlet_treatSEXP = 1.0,
      b_dirichlet_treatSEXP = 1.0,
      rho_dirichlet_treatSEXP = 1.0,
      no_trees_controlSEXP = number_of_trees,
      power_controlSEXP = power,
      base_controlSEXP = base,
      p_grow_controlSEXP = p_grow,
      p_prune_controlSEXP = p_prune,
      omega_controlSEXP = 1/2,
      prior_type_controlSEXP = "horseshoe",
      param1_controlSEXP = k / sqrt(number_of_trees),
      param2_controlSEXP = k / sqrt(number_of_trees),
      reversible_controlSEXP = TRUE,
      dirichlet_bool_controlSEXP = FALSE,
      a_dirichlet_controlSEXP = 1.0,
      b_dirichlet_controlSEXP = 1.0,
      rho_dirichlet_controlSEXP = 1.0,
      sigma_knownSEXP = sigma_known,
      sigmaSEXP = sigma_hat,
      lambdaSEXP = lambda,
      nuSEXP = nu,
      N_postSEXP = N_post,
      N_burnSEXP = N_burn,
      delayed_proposalSEXP = delayed_proposal,
      store_posterior_sample_controlSEXP = store_posterior_sample,
      store_posterior_sample_treatSEXP = store_posterior_sample,
      verboseSEXP = verbose,
      treatment_codingSEXP = treatment_coding,
      propensity_trainSEXP = propensity_train_cpp,
      propensity_testSEXP = propensity_test_cpp
    )
    
    # Total
    fit$train_predictions <- fit$train_predictions * sigma_hat + y_mean
    fit$test_predictions <- fit$test_predictions * sigma_hat + y_mean
    
    # Control
    fit$train_predictions_control <- fit$train_predictions_control * 
      sigma_hat + y_mean
    fit$test_predictions_control <- fit$test_predictions_control * 
      sigma_hat + y_mean
    
    # Treatment
    fit$train_predictions_treat <- fit$train_predictions_treat * sigma_hat 
    fit$test_predictions_treat <- fit$test_predictions_treat * sigma_hat 
    
    # Posterior samples
    if (store_posterior_sample) {
      fit$train_predictions_sample <- 
        fit$train_predictions_sample *  sigma_hat + y_mean
      fit$test_predictions_sample <- 
        fit$test_predictions_sample * sigma_hat +  y_mean
      fit$train_predictions_sample_control <- 
        fit$train_predictions_sample_control * sigma_hat + y_mean
      fit$test_predictions_sample_control <- 
        fit$test_predictions_sample_control * sigma_hat + y_mean
      fit$train_predictions_sample_treat <- 
        fit$train_predictions_sample_treat * sigma_hat 
      fit$test_predictions_sample_treat <- 
        fit$test_predictions_sample_treat * sigma_hat 
    }
    
  }
  
  # remove burn-in of sigma
  if (!sigma_known) {
    fit$sigma <- fit$sigma[-(1:N_burn)]
  } 

  prior_type_control_user <- "horseshoe"
  prior_type_treat_user <- "horseshoe"
  dirichlet_bool_control <- FALSE
  dirichlet_bool_treat <- FALSE
  prior_type_control_cpp <- "horseshoe"
  prior_type_treat_cpp <- "horseshoe"

  obj <- NewCausalShrinkageForest(
    fit = fit,
    call = match.call(),
    outcome_type = outcome_type,
    timescale = timescale,
    n_train = n_train,
    p_control = p_control,
    p_treat = p_treat,
    n_test = n_test,
    test_provided = test_provided_flag,
    number_of_trees_control = number_of_trees,
    number_of_trees_treat = number_of_trees,
    N_post = N_post,
    N_burn = N_burn,
    store_posterior_sample = store_posterior_sample,
    sigma_hat = sigma_hat,
    sigma_known = sigma_known,
    y_mean = y_mean,
    prior_type_control_user = prior_type_control_user,
    prior_type_control_cpp = prior_type_control_cpp,
    prior_type_treat_user = prior_type_treat_user,
    prior_type_treat_cpp = prior_type_treat_cpp,
    dirichlet_bool_control = dirichlet_bool_control,
    dirichlet_bool_treat = dirichlet_bool_treat,
    # Data stored for predict()
    y_train                   = y_horse_raw,
    X_train_control           = X_control_mat,
    X_train_treat             = X_treat_mat,
    treatment_indicator_train = treatment_indicator_train,
    status_train              = if (outcome_type %in% c("right-censored", "interval-censored")) status_horse_raw else NULL,
    left_time_train           = left_time_raw,
    right_time_train          = right_time_raw,
    ic_indicator_train        = if (outcome_type == "interval-censored") ic_indicator else NULL,
    # Hyperparameters stored for predict()
    p_grow           = p_grow,
    p_prune          = p_prune,
    delayed_proposal = delayed_proposal,
    nu               = nu,
    lambda           = lambda,
    power_control    = power,
    base_control     = base,
    param1_control   = k / sqrt(number_of_trees),
    param2_control   = k / sqrt(number_of_trees),
    omega_control    = 1/2,
    reversible_control   = TRUE,
    a_dirichlet_control  = 0.5,
    b_dirichlet_control  = 1.0,
    rho_dirichlet_control = p_control,
    power_treat      = power,
    base_treat       = base,
    param1_treat     = k / sqrt(number_of_trees),
    param2_treat     = k / sqrt(number_of_trees),
    omega_treat      = 1/2,
    reversible_treat     = TRUE,
    a_dirichlet_treat    = 0.5,
    b_dirichlet_treat    = 1.0,
    rho_dirichlet_treat  = p_treat,
    treatment_coding     = treatment_coding,
    propensity_train     = propensity
  )

  return(obj)
}

