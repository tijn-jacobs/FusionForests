#' SurvivalBART
#'
#' Fits an Accelerated Failure Time (AFT) model using the classical
#' Bayesian Additive Regression Trees (BART) prior:
#' \eqn{\log(Y) = f(x) + \varepsilon}.
#' Supports both right-censored and interval-censored survival outcomes.
#'
#' @param time Outcome vector of (non-negative) survival times. Required for
#'   right-censored outcomes; set to \code{NULL} when using interval censoring.
#' @param status Event indicator (1 = event, 0 = censored). Required for
#'   right-censored outcomes; derived automatically for interval censoring.
#' @param X_train Design matrix for training data.
#' @param X_test Optional test matrix. If NULL, predictions are computed at
#'   the column means of \code{X_train}.
#' @param timescale Either \code{"time"} (log-transform internally) or
#'   \code{"log"} (already log-transformed).
#' @param number_of_trees Number of trees in the ensemble. Default is 200.
#' @param k Scaling constant used to calibrate the prior variance of the
#'   step heights.
#' @param N_post Number of posterior samples to store.
#' @param N_burn Number of burn-in iterations.
#' @param store_posterior_sample Logical; if \code{TRUE} (default), store
#'   the full \eqn{N_\text{post} \times n} matrix of posterior predictions.
#'   Required for \code{predict()}, \code{plot(type = "survival")} with full
#'   posterior credible bands, and custom posterior analyses. Set to
#'   \code{FALSE} to save memory when only posterior means are needed.
#' @param verbose Logical; print sampling progress.
#' @param left_time Optional numeric vector of left (lower) time boundaries
#'   for interval-censored data. Exact events have
#'   \code{left_time == right_time}; right-censored observations have
#'   \code{right_time = Inf}; interval-censored observations have finite
#'   \code{left_time < right_time}. When provided together with
#'   \code{right_time}, the model is fitted with
#'   \code{outcome_type = "interval-censored"} and \code{time}/\code{status}
#'   are ignored.
#' @param right_time Optional numeric vector of right (upper) time boundaries.
#'   Use \code{Inf} for right-censored observations.
#' @param ... Additional arguments passed to \code{\link{ShrinkageTrees}}
#'   to override default hyperparameters.
#'
#' @details
#' This function provides a survival-specific interface for classical BART
#' under an AFT formulation for right-censored or interval-censored outcomes.
#'
#' For right-censored data, supply \code{time} and \code{status}.
#' For interval-censored data, supply \code{left_time} and \code{right_time}
#' instead; event indicators are derived internally following the
#' \code{survival::Surv(type = "interval2")} convention.
#'
#' Structural regularisation is induced through the standard Gaussian
#' leaf prior and tree depth prior of Chipman, George & McCulloch (2010).
#'
#' Users requiring alternative shrinkage priors (e.g., Horseshoe or
#' Dirichlet splitting priors) should use \code{\link{ShrinkageTrees}}
#' directly.
#'
#' @references
#' Chipman, H. A., George, E. I., & McCulloch, R. E. (2010).
#' Bayesian Additive Regression Trees.
#' Annals of Applied Statistics.
#' 
#' @seealso
#' Related models: \code{\link{SurvivalDART}} (Dirichlet sparsity),
#' \code{\link{HorseTrees}} (horseshoe prior),
#' \code{\link{ShrinkageTrees}} (general shrinkage priors).
#'
#' S3 methods: \code{\link{print.ShrinkageTrees}},
#' \code{\link{summary.ShrinkageTrees}},
#' \code{\link{predict.ShrinkageTrees}},
#' \code{\link{plot.ShrinkageTrees}}.
#'
#' @return
#' An object of class \code{"ShrinkageTrees"} fitted under a classical
#' BART prior within an AFT formulation.
#'
#' See \code{\link{ShrinkageTrees}} for a full description of returned components
#'
#' @examples
#' set.seed(1)
#' n <- 30; p <- 5
#' X <- matrix(rnorm(n * p), ncol = p)
#' time <- rexp(n, rate = exp(0.5 * X[, 1]))
#' status <- rbinom(n, 1, 0.7)
#'
#' fit <- SurvivalBART(time = time, status = status, X_train = X,
#'                     number_of_trees = 5, N_post = 50, N_burn = 25,
#'                     verbose = FALSE)
#'
#' # S3 methods
#' print(fit)
#' smry <- summary(fit)
#'
#' # Posterior predictions on new data
#' X_new <- matrix(rnorm(10 * p), ncol = p)
#' pred <- predict(fit, newdata = X_new)
#' print(pred)
#'
#' # Diagnostic plot (requires ggplot2)
#' if (requireNamespace("ggplot2", quietly = TRUE)) {
#'   plot(fit, type = "trace")
#'
#'   # Posterior survival curves for training data
#'   plot(fit, type = "survival")
#'
#'   # Posterior predictive survival curves for new data
#'   plot(pred, type = "survival")
#'   plot(pred, type = "survival", obs = c(1, 5))
#' }
#'
#' # Interval-censored example
#' set.seed(11)
#' n <- 30; p <- 5
#' X <- matrix(rnorm(n * p), ncol = p)
#' true_t <- rexp(n, rate = exp(0.5 * X[, 1]))
#' left_t  <- true_t * runif(n, 0.5, 1)
#' right_t <- true_t * runif(n, 1, 1.5)
#' exact <- sample(n, 10); left_t[exact] <- true_t[exact]; right_t[exact] <- true_t[exact]
#' rc <- sample(setdiff(seq_len(n), exact), 5); right_t[rc] <- Inf
#'
#' fit_ic <- SurvivalBART(left_time = left_t, right_time = right_t,
#'                         X_train = X, number_of_trees = 5,
#'                         N_post = 50, N_burn = 25, verbose = FALSE)
#'
#' @export
SurvivalBART <- function(
  time = NULL,
  status = NULL,
  X_train,
  X_test = NULL,
  timescale = "time",
  number_of_trees = 200,
  k = 2.0,
  N_post = 1000,
  N_burn = 1000,
  store_posterior_sample = TRUE,
  verbose = TRUE,
  left_time = NULL,
  right_time = NULL,
  ...
) {

  interval_censored <- !is.null(left_time) && !is.null(right_time)

  if (interval_censored) {
    if (any(left_time <= 0) && timescale == "time")
      stop("Survival times must be strictly positive when timescale = 'time'.")

    # Derive representative values for local_hp calibration
    y_proxy <- ifelse(left_time == right_time, left_time,
                      ifelse(is.finite(right_time),
                             (left_time + right_time) / 2, left_time))

    if (timescale == "time") {
      local_hp <- (max(log(y_proxy)) - min(log(y_proxy))) /
        (k * 2 * sqrt(number_of_trees))
    } else {
      local_hp <- (max(y_proxy) - min(y_proxy)) /
        (k * 2 * sqrt(number_of_trees))
    }

    fit <- ShrinkageTrees(
            left_time = left_time,
            right_time = right_time,
            X_train = X_train,
            X_test = X_test,
            outcome_type = "interval-censored",
            timescale = timescale,
            number_of_trees = number_of_trees,
            prior_type = "standard",
            local_hp = local_hp,
            N_post = N_post,
            N_burn = N_burn,
            store_posterior_sample = store_posterior_sample,
            verbose = verbose,
            ...
    )
  } else {
    if (is.null(time) || is.null(status))
      stop("'time' and 'status' are required for right-censored outcomes.")

    if (any(time <= 0) && timescale == "time")
      stop("Survival times must be strictly positive when timescale = 'time'.")

    if (timescale == "time") {
      local_hp <- (max(log(time)) - min(log(time))) /
        (k * 2 * sqrt(number_of_trees))
    } else {
      local_hp <- (max(time) - min(time)) /
        (k * 2 * sqrt(number_of_trees))
    }

    fit <- ShrinkageTrees(
            y = time,
            status = status,
            X_train = X_train,
            X_test = X_test,
            outcome_type = "right-censored",
            timescale = timescale,
            number_of_trees = number_of_trees,
            prior_type = "standard",
            local_hp = local_hp,
            N_post = N_post,
            N_burn = N_burn,
            store_posterior_sample = store_posterior_sample,
            verbose = verbose,
            ...
    )
  }

  fit$wrapper <- "SurvivalBART"
  return(fit)
}


#' SurvivalDART
#'
#' Fits an Accelerated Failure Time (AFT) model using the Dirichlet splitting
#' prior (DART), which induces structural sparsity through a Beta-Dirichlet
#' hierarchy on splitting probabilities.
#' Supports both right-censored and interval-censored survival outcomes.
#'
#' @inheritParams SurvivalBART
#' @param number_of_trees Number of trees in the ensemble. Default is 200.
#' @param a_dirichlet,b_dirichlet Beta hyperparameters controlling sparsity
#'   in the Dirichlet splitting rule.
#' @param rho_dirichlet Expected number of active predictors. If NULL,
#'   defaults to the number of covariates in \code{X_train}.
#' @param k Scaling constant used to calibrate the prior variance of the
#'   step heights.
#' @param ... Additional arguments passed to \code{\link{ShrinkageTrees}}
#'   to override default hyperparameters.
#'
#' @details
#' This function provides a survival-specific wrapper for DART under an
#' AFT formulation for right-censored or interval-censored outcomes.
#'
#' For right-censored data, supply \code{time} and \code{status}.
#' For interval-censored data, supply \code{left_time} and \code{right_time}
#' instead; event indicators are derived internally following the
#' \code{survival::Surv(type = "interval2")} convention.
#'
#' Structural regularisation is induced through a Dirichlet prior on
#' splitting probabilities, encouraging sparse feature usage in
#' high-dimensional settings.
#'
#' Users requiring alternative shrinkage priors on the leaf parameters
#' (e.g., Horseshoe or half-Cauchy priors) should use
#' \code{\link{ShrinkageTrees}} directly.
#'
#' @seealso
#' Related models: \code{\link{SurvivalBART}} (standard BART prior),
#' \code{\link{ShrinkageTrees}} (general shrinkage priors).
#'
#' S3 methods: \code{\link{print.ShrinkageTrees}},
#' \code{\link{summary.ShrinkageTrees}},
#' \code{\link{predict.ShrinkageTrees}},
#' \code{\link{plot.ShrinkageTrees}}.
#'
#' @return
#' An object of class \code{"ShrinkageTrees"} fitted under a Dirichlet
#' splitting prior (DART) within an AFT formulation.
#'
#' See \code{\link{ShrinkageTrees}} for a full description of returned components.
#'
#' @examples
#' set.seed(2)
#' n <- 30; p <- 5
#' X <- matrix(rnorm(n * p), ncol = p)
#' time <- rexp(n, rate = exp(0.5 * X[, 1]))
#' status <- rbinom(n, 1, 0.7)
#'
#' fit <- SurvivalDART(time = time, status = status, X_train = X,
#'                     number_of_trees = 5, N_post = 50, N_burn = 25,
#'                     verbose = FALSE)
#'
#' # S3 methods
#' print(fit)
#' smry <- summary(fit)
#'
#' # Posterior predictions on new data
#' X_new <- matrix(rnorm(10 * p), ncol = p)
#' pred <- predict(fit, newdata = X_new)
#' print(pred)
#'
#' # Variable importance and survival plots (requires ggplot2)
#' if (requireNamespace("ggplot2", quietly = TRUE)) {
#'   plot(fit, type = "vi", n_vi = 5)
#'
#'   # Posterior survival curves for training data
#'   plot(fit, type = "survival")
#'
#'   # Posterior predictive survival curves for new data
#'   plot(pred, type = "survival")
#'   plot(pred, type = "survival", obs = c(1, 5))
#' }
#'
#' # Interval-censored example
#' set.seed(12)
#' n <- 30; p <- 5
#' X <- matrix(rnorm(n * p), ncol = p)
#' true_t <- rexp(n, rate = exp(0.5 * X[, 1]))
#' left_t  <- true_t * runif(n, 0.5, 1)
#' right_t <- true_t * runif(n, 1, 1.5)
#' exact <- sample(n, 10); left_t[exact] <- true_t[exact]; right_t[exact] <- true_t[exact]
#' rc <- sample(setdiff(seq_len(n), exact), 5); right_t[rc] <- Inf
#'
#' fit_ic <- SurvivalDART(left_time = left_t, right_time = right_t,
#'                         X_train = X, number_of_trees = 5,
#'                         N_post = 50, N_burn = 25, verbose = FALSE)
#'
#' @export
SurvivalDART <- function(
  time = NULL,
  status = NULL,
  X_train,
  X_test = NULL,
  timescale = "time",
  number_of_trees = 200,
  a_dirichlet = 0.5,
  b_dirichlet = 1.0,
  rho_dirichlet = NULL,
  k = 2.0,
  N_post = 1000,
  N_burn = 1000,
  store_posterior_sample = TRUE,
  verbose = TRUE,
  left_time = NULL,
  right_time = NULL,
  ...
) {

  interval_censored <- !is.null(left_time) && !is.null(right_time)

  if (is.null(rho_dirichlet)) {
    rho_dirichlet <- ncol(X_train)
  }

  if (interval_censored) {
    if (any(left_time <= 0) && timescale == "time")
      stop("Survival times must be strictly positive when timescale = 'time'.")

    # Derive representative values for local_hp calibration
    y_proxy <- ifelse(left_time == right_time, left_time,
                      ifelse(is.finite(right_time),
                             (left_time + right_time) / 2, left_time))

    if (timescale == "time") {
      local_hp <- (max(log(y_proxy)) - min(log(y_proxy))) /
        (k * 2 * sqrt(number_of_trees))
    } else {
      local_hp <- (max(y_proxy) - min(y_proxy)) /
        (k * 2 * sqrt(number_of_trees))
    }

    fit <- ShrinkageTrees(
            left_time = left_time,
            right_time = right_time,
            X_train = X_train,
            X_test = X_test,
            outcome_type = "interval-censored",
            timescale = timescale,
            number_of_trees = number_of_trees,
            prior_type = "dirichlet",
            local_hp = local_hp,
            a_dirichlet = a_dirichlet,
            b_dirichlet = b_dirichlet,
            rho_dirichlet = rho_dirichlet,
            N_post = N_post,
            N_burn = N_burn,
            store_posterior_sample = store_posterior_sample,
            verbose = verbose,
            ...
          )
  } else {
    if (is.null(time) || is.null(status))
      stop("'time' and 'status' are required for right-censored outcomes.")

    if (any(time <= 0) && timescale == "time")
      stop("Survival times must be strictly positive when timescale = 'time'.")

    if (timescale == "time") {
      local_hp <- (max(log(time)) - min(log(time))) /
        (k * 2 * sqrt(number_of_trees))
    } else {
      local_hp <- (max(time) - min(time)) /
        (k * 2 * sqrt(number_of_trees))
    }

    fit <- ShrinkageTrees(
            y = time,
            status = status,
            X_train = X_train,
            X_test = X_test,
            outcome_type = "right-censored",
            timescale = timescale,
            number_of_trees = number_of_trees,
            prior_type = "dirichlet",
            local_hp = local_hp,
            a_dirichlet = a_dirichlet,
            b_dirichlet = b_dirichlet,
            rho_dirichlet = rho_dirichlet,
            N_post = N_post,
            N_burn = N_burn,
            store_posterior_sample = store_posterior_sample,
            verbose = verbose,
            ...
          )
  }

  fit$wrapper <- "SurvivalDART"
  return(fit)
}


#' SurvivalBCF (Bayesian Causal Forest for survival data)
#'
#' Fits an Accelerated Failure Time (AFT) version of Bayesian Causal Forest (BCF):
#' \eqn{Y = \mu(x) + W \tau(x) + \varepsilon}, where separate forests are used
#' for the prognostic (control) function \eqn{\mu(x)} and the treatment effect
#' function \eqn{\tau(x)}.
#'
#' This wrapper provides a survival-specific implementation using classical
#' BART-style priors for both forests. Supports both right-censored and
#' interval-censored survival outcomes.
#'
#' @inheritParams SurvivalBART
#' @param treatment Treatment indicator (0/1) for training data.
#' @param number_of_trees_control Number of trees in the control forest.
#'   Default is 200.
#' @param number_of_trees_treat Number of trees in the treatment forest.
#'   Default is 50.
#' @param power_control,base_control Tree-structure prior parameters for the
#'   control forest.
#' @param power_treat,base_treat Tree-structure prior parameters for the
#'   treatment forest.
#' @param ... Additional arguments passed to \code{\link{CausalShrinkageForest}}
#'   to override default hyperparameters.
#' @param propensity Optional vector of propensity scores. If provided,
#' it is appended to the control forest design matrix. Required when
#' \code{treatment_coding = "adaptive"}.
#' @param treatment_coding Character string specifying how the treatment
#' indicator enters the model. One of \code{"centered"} (default, maps to
#' \eqn{-1/2} and \eqn{1/2}), \code{"binary"} (maps to \eqn{0} and \eqn{1}),
#' \code{"adaptive"} (maps to \eqn{z_i - \hat{e}(x_i)}, where
#' \eqn{\hat{e}(x_i)} is the propensity score), or \code{"invariant"}
#' (parameter-expanded coding with \eqn{b_0, b_1 \sim N(0, 1/2)} estimated
#' within the Gibbs sampler; Hahn et al., 2020, Section 5.2).
#'
#' @details
#' This function implements a simplified AFT-BCF model for right-censored or
#' interval-censored survival outcomes. Structural regularisation is induced
#' through classical BART priors on the tree structure and leaf parameters.
#'
#' For right-censored data, supply \code{time} and \code{status}.
#' For interval-censored data, supply \code{left_time} and \code{right_time}
#' instead; event indicators are derived internally following the
#' \code{survival::Surv(type = "interval2")} convention.
#'
#' Users requiring alternative shrinkage priors (e.g., Horseshoe or Dirichlet
#' splitting priors) should use \code{\link{SurvivalShrinkageBCF}} or call
#' \code{\link{CausalShrinkageForest}} directly.
#'
#' @references
#' Hahn, P. R., Murray, J. S., & Carvalho, C. M. (2020).
#' Bayesian regression tree models for causal inference:
#' Regularization, confounding, and heterogeneous effects.
#' Bayesian Analysis.
#'
#' @seealso
#' Related models: \code{\link{SurvivalShrinkageBCF}} (Dirichlet sparsity),
#' \code{\link{CausalHorseForest}} (horseshoe prior),
#' \code{\link{CausalShrinkageForest}} (general shrinkage priors).
#'
#' S3 methods: \code{\link{print.CausalShrinkageForest}},
#' \code{\link{summary.CausalShrinkageForest}},
#' \code{\link{predict.CausalShrinkageForest}},
#' \code{\link{plot.CausalShrinkageForest}}.
#'
#' @return
#' An object of class \code{"CausalShrinkageForest"} corresponding to a
#' survival BCF model under classical BART priors.
#'
#' See \code{\link{CausalShrinkageForest}} for returned components.
#'
#' @examples
#' set.seed(3)
#' n <- 30; p <- 5
#' X <- matrix(rnorm(n * p), ncol = p)
#' treatment <- rbinom(n, 1, 0.5)
#' log_T <- X[, 1] + treatment * (-0.5) + rnorm(n)
#' time <- exp(log_T)
#' status <- rbinom(n, 1, 0.7)
#'
#' fit <- SurvivalBCF(time = time, status = status, X_train = X,
#'                    treatment = treatment,
#'                    number_of_trees_control = 5,
#'                    number_of_trees_treat = 5,
#'                    N_post = 50, N_burn = 25,
#'                    verbose = FALSE)
#'
#' # S3 methods
#' print(fit)
#' smry <- summary(fit)
#'
#' # Posterior ATE
#' cat("ATE:", round(smry$treatment_effect$ate, 3), "\n")
#'
#' # Diagnostic and treatment-effect plots (requires ggplot2)
#' if (requireNamespace("ggplot2", quietly = TRUE)) {
#'   plot(fit, type = "trace")
#'   plot(fit, type = "ate")
#'   plot(fit, type = "cate")
#' }
#'
#' # Interval-censored causal example
#' set.seed(13)
#' n <- 30; p <- 5
#' X <- matrix(rnorm(n * p), ncol = p)
#' treatment <- rbinom(n, 1, 0.5)
#' true_t <- exp(X[, 1] + treatment * (-0.5) + rnorm(n))
#' left_t  <- true_t * runif(n, 0.5, 1)
#' right_t <- true_t * runif(n, 1, 1.5)
#' exact <- sample(n, 10); left_t[exact] <- true_t[exact]; right_t[exact] <- true_t[exact]
#' rc <- sample(setdiff(seq_len(n), exact), 5); right_t[rc] <- Inf
#'
#' fit_ic <- SurvivalBCF(left_time = left_t, right_time = right_t,
#'                        X_train = X, treatment = treatment,
#'                        number_of_trees_control = 5,
#'                        number_of_trees_treat = 5,
#'                        N_post = 50, N_burn = 25, verbose = FALSE)
#'
#' @export
SurvivalBCF <- function(
  time = NULL,
  status = NULL,
  X_train,
  treatment,
  timescale = "time",
  propensity = NULL,
  treatment_coding = "centered",
  number_of_trees_control = 200,
  number_of_trees_treat = 50,
  power_control = 2,
  base_control = 0.95,
  power_treat = 3,
  base_treat = 0.25,
  N_post = 1000,
  N_burn = 1000,
  store_posterior_sample = TRUE,
  verbose = TRUE,
  left_time = NULL,
  right_time = NULL,
  ...
) {

  interval_censored <- !is.null(left_time) && !is.null(right_time)

  if (!interval_censored && (is.null(time) || is.null(status)))
    stop("'time' and 'status' are required for right-censored outcomes.")

  if (interval_censored && any(left_time <= 0) && timescale == "time")
    stop("Survival times must be strictly positive when timescale = 'time'.")

  if (!interval_censored && any(time <= 0) && timescale == "time")
    stop("Survival times must be strictly positive when timescale = 'time'.")

  if (length(treatment) != nrow(X_train))
    stop("Length of treatment must match number of training observations.")

  if (!is.null(propensity)) {
    X_train_control <- cbind(propensity, X_train)
  } else {
    X_train_control <- X_train
  }

  if (interval_censored) {
    # Derive status/ic_indicator/y from boundaries for censored_info
    ic_status    <- as.integer(left_time == right_time)
    ic_indicator <- as.integer(left_time < right_time & is.finite(right_time))
    y_derived    <- ifelse(ic_status == 1, left_time,
                           ifelse(ic_indicator == 1,
                                  (left_time + right_time) / 2, left_time))

    info <- censored_info(y_derived, ic_status,
                          left_time = left_time, right_time = right_time,
                          ic_indicator = ic_indicator)
    sd_time <- info$sd

    fit <- CausalShrinkageForest(
      left_time = left_time,
      right_time = right_time,
      X_train_control = X_train_control,
      X_train_treat = X_train,
      treatment_indicator_train = treatment,
      outcome_type = "interval-censored",
      timescale = timescale,

      prior_type_control = "standard-halfcauchy",
      prior_type_treat = "standard",

      local_hp_treat = 1 / (sd_time * sqrt(number_of_trees_treat) * 0.674),

      number_of_trees_control = number_of_trees_control,
      number_of_trees_treat = number_of_trees_treat,

      power_control = power_control,
      base_control = base_control,
      power_treat = power_treat,
      base_treat = base_treat,

      treatment_coding = treatment_coding,
      propensity = propensity,

      N_post = N_post,
      N_burn = N_burn,
      store_posterior_sample = store_posterior_sample,
      verbose = verbose,
      ...
    )
  } else {
    info <- censored_info(time, status)
    sd_time <- info$sd

    fit <- CausalShrinkageForest(
      y = time,
      status = status,
      X_train_control = X_train_control,
      X_train_treat = X_train,
      treatment_indicator_train = treatment,
      outcome_type = "right-censored",
      timescale = timescale,

      prior_type_control = "standard-halfcauchy",
      prior_type_treat = "standard",

      local_hp_treat = 1 / (sd_time * sqrt(number_of_trees_treat) * 0.674),

      number_of_trees_control = number_of_trees_control,
      number_of_trees_treat = number_of_trees_treat,

      power_control = power_control,
      base_control = base_control,
      power_treat = power_treat,
      base_treat = base_treat,

      treatment_coding = treatment_coding,
      propensity = propensity,

      N_post = N_post,
      N_burn = N_burn,
      store_posterior_sample = store_posterior_sample,
      verbose = verbose,
      ...
    )
  }

  fit$wrapper <- "SurvivalBCF"
  return(fit)
}

#' SurvivalShrinkageBCF (Shrinkage Bayesian Causal Forest for survival data)
#'
#' Fits a survival version of a Bayesian Causal Forest (BCF) under an
#' accelerated failure time (AFT) model, combining Dirichlet splitting
#' priors with global-local shrinkage. Supports both right-censored and
#' interval-censored survival outcomes.
#'
#' This wrapper extends \code{\link{SurvivalBCF}} by incorporating
#' Dirichlet sparsity in both the prognostic (control) and treatment
#' forests, while applying additional shrinkage to the control forest
#' via a half-Cauchy prior.
#'
#' @inheritParams SurvivalBCF
#' @param a_dir First shape parameter of the Beta prior controlling the 
#' sparsity level in the Dirichlet splitting rule.
#' @param b_dir Second shape parameter of the Beta prior controlling the 
#' sparsity level in the Dirichlet splitting rule.
#'
#' @details
#' The SurvivalShrinkageBCF model decomposes the outcome as
#' \deqn{
#'   \log T = \mu(x) + a \cdot \tau(x) + \varepsilon,
#' }
#' where \eqn{\mu(x)} represents the prognostic (control) component and 
#' \eqn{\tau(x)} the heterogeneous treatment effect.
#'
#' In contrast to \code{SurvivalBCF}, this function:
#' \itemize{
#'   \item Applies a Dirichlet splitting prior to both forests, inducing 
#'   structural sparsity in variable selection.
#'   \item Combines Dirichlet sparsity with additional half-Cauchy shrinkage 
#'   in the control forest.
#' }
#'
#' The Dirichlet prior follows the sparse splitting framework of Linero (2018),
#' where splitting probabilities are governed by a Beta-Dirichlet hierarchy.
#' The sparsity level is controlled by \code{a_dir} and \code{b_dir}.
#'
#' Survival outcomes are modeled using an AFT formulation with censoring
#' handled via data augmentation. Both right-censored and interval-censored
#' data are supported. For interval-censored data, supply \code{left_time}
#' and \code{right_time} instead of \code{time} and \code{status}.
#'
#' @references
#' Caron, A., Baio, G., & Manolopoulou, I. (2022).
#' Shrinkage Bayesian Causal Forests for Heterogeneous Treatment Effects Estimation.
#' Journal of Computational and Graphical Statistics, 31(4), 1202--1214.
#' https://doi.org/10.1080/10618600.2022.2067549
#'
#' @seealso
#' Related models: \code{\link{SurvivalBCF}} (standard BART priors),
#' \code{\link{CausalShrinkageForest}} (general shrinkage priors),
#' \code{\link{CausalHorseForest}} (horseshoe prior).
#'
#' S3 methods: \code{\link{print.CausalShrinkageForest}},
#' \code{\link{summary.CausalShrinkageForest}},
#' \code{\link{predict.CausalShrinkageForest}},
#' \code{\link{plot.CausalShrinkageForest}}.
#'
#' @return
#' An object of class \code{"CausalShrinkageForest"} fitted with
#' Dirichlet splitting priors and additional shrinkage.
#'
#' @examples
#' set.seed(4)
#' n <- 30; p <- 5
#' X <- matrix(rnorm(n * p), ncol = p)
#' treatment <- rbinom(n, 1, 0.5)
#' log_T <- X[, 1] + treatment * (-0.5) + rnorm(n)
#' time <- exp(log_T)
#' status <- rbinom(n, 1, 0.7)
#'
#' fit <- SurvivalShrinkageBCF(time = time, status = status, X_train = X,
#'                              treatment = treatment,
#'                              number_of_trees_control = 5,
#'                              number_of_trees_treat = 5,
#'                              N_post = 50, N_burn = 25,
#'                              verbose = FALSE)
#'
#' # S3 methods
#' print(fit)
#' smry <- summary(fit)
#'
#' # Posterior ATE with 95% credible interval
#' cat("ATE:", round(smry$treatment_effect$ate, 3), "\n")
#'
#' # Diagnostic and treatment-effect plots (requires ggplot2)
#' if (requireNamespace("ggplot2", quietly = TRUE)) {
#'   plot(fit, type = "trace")
#'   plot(fit, type = "cate")
#' }
#'
#' # Interval-censored causal example
#' set.seed(14)
#' n <- 30; p <- 5
#' X <- matrix(rnorm(n * p), ncol = p)
#' treatment <- rbinom(n, 1, 0.5)
#' true_t <- exp(X[, 1] + treatment * (-0.5) + rnorm(n))
#' left_t  <- true_t * runif(n, 0.5, 1)
#' right_t <- true_t * runif(n, 1, 1.5)
#' exact <- sample(n, 10); left_t[exact] <- true_t[exact]; right_t[exact] <- true_t[exact]
#' rc <- sample(setdiff(seq_len(n), exact), 5); right_t[rc] <- Inf
#'
#' fit_ic <- SurvivalShrinkageBCF(left_time = left_t, right_time = right_t,
#'                                 X_train = X, treatment = treatment,
#'                                 number_of_trees_control = 5,
#'                                 number_of_trees_treat = 5,
#'                                 N_post = 50, N_burn = 25, verbose = FALSE)
#'
#' @export
SurvivalShrinkageBCF <- function(
  time = NULL,
  status = NULL,
  X_train,
  treatment,
  timescale = "time",
  propensity = NULL,
  treatment_coding = "centered",
  a_dir = 0.5,
  b_dir = 1.0,
  number_of_trees_control = 200,
  number_of_trees_treat = 50,
  power_control = 2,
  base_control = 0.95,
  power_treat = 3,
  base_treat = 0.25,
  N_post = 1000,
  N_burn = 1000,
  store_posterior_sample = TRUE,
  verbose = TRUE,
  left_time = NULL,
  right_time = NULL,
  ...
) {

  interval_censored <- !is.null(left_time) && !is.null(right_time)

  if (!interval_censored && (is.null(time) || is.null(status)))
    stop("'time' and 'status' are required for right-censored outcomes.")

  if (interval_censored && any(left_time <= 0) && timescale == "time")
    stop("Survival times must be strictly positive when timescale = 'time'.")

  if (!interval_censored && any(time <= 0) && timescale == "time")
    stop("Survival times must be strictly positive when timescale = 'time'.")

  if (length(treatment) != nrow(X_train))
    stop("Length of treatment must match number of training observations.")

  if (!is.null(propensity)) {
    X_control <- cbind(propensity, X_train)
    rho_dir_control <- ncol(X_control)
    rho_dir_treat <- ncol(X_train)
  } else {
    X_control <- X_train
    rho_dir_control <- ncol(X_train)
    rho_dir_treat <- ncol(X_train)
  }

  if (interval_censored) {
    # Derive status/ic_indicator/y from boundaries for censored_info
    ic_status    <- as.integer(left_time == right_time)
    ic_indicator <- as.integer(left_time < right_time & is.finite(right_time))
    y_derived    <- ifelse(ic_status == 1, left_time,
                           ifelse(ic_indicator == 1,
                                  (left_time + right_time) / 2, left_time))

    info <- censored_info(y_derived, ic_status,
                          left_time = left_time, right_time = right_time,
                          ic_indicator = ic_indicator)
    sd_time <- info$sd

    fit <- CausalShrinkageForest(
      left_time = left_time,
      right_time = right_time,
      X_train_control = X_control,
      X_train_treat = X_train,
      treatment_indicator_train = treatment,
      outcome_type = "interval-censored",
      timescale = timescale,

      prior_type_control = "dirichlet-halfcauchy",
      prior_type_treat = "dirichlet",

      local_hp_treat = 1 / (sd_time * sqrt(number_of_trees_treat) * 0.674),

      number_of_trees_control = number_of_trees_control,
      number_of_trees_treat = number_of_trees_treat,

      power_control = power_control,
      base_control = base_control,
      power_treat = power_treat,
      base_treat = base_treat,

      a_dirichlet_control = a_dir,
      b_dirichlet_control = b_dir,
      rho_dirichlet_control = rho_dir_control,
      a_dirichlet_treat = a_dir,
      b_dirichlet_treat = b_dir,
      rho_dirichlet_treat = rho_dir_treat,

      treatment_coding = treatment_coding,
      propensity = propensity,

      N_post = N_post,
      N_burn = N_burn,
      store_posterior_sample = store_posterior_sample,
      verbose = verbose,
      ...
    )
  } else {
    info <- censored_info(time, status)
    sd_time <- info$sd

    fit <- CausalShrinkageForest(
      y = time,
      status = status,
      X_train_control = X_control,
      X_train_treat = X_train,
      treatment_indicator_train = treatment,
      outcome_type = "right-censored",
      timescale = timescale,

      prior_type_control = "dirichlet-halfcauchy",
      prior_type_treat = "dirichlet",

      local_hp_treat = 1 / (sd_time * sqrt(number_of_trees_treat) * 0.674),

      number_of_trees_control = number_of_trees_control,
      number_of_trees_treat = number_of_trees_treat,

      power_control = power_control,
      base_control = base_control,
      power_treat = power_treat,
      base_treat = base_treat,

      a_dirichlet_control = a_dir,
      b_dirichlet_control = b_dir,
      rho_dirichlet_control = rho_dir_control,
      a_dirichlet_treat = a_dir,
      b_dirichlet_treat = b_dir,
      rho_dirichlet_treat = rho_dir_treat,

      treatment_coding = treatment_coding,
      propensity = propensity,

      N_post = N_post,
      N_burn = N_burn,
      store_posterior_sample = store_posterior_sample,
      verbose = verbose,
      ...
    )
  }

  fit$wrapper <- "SurvivalShrinkageBCF"
  return(fit)
}
