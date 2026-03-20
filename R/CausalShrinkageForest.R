#' General Causal Shrinkage Forests
#'
#' Fits a (Bayesian) Causal Shrinkage Forest model for estimating heterogeneous treatment effects.
#' This function generalizes \code{\link{CausalHorseForest}} by allowing flexible
#' global-local shrinkage priors on the step heights in both the control and treatment forests.
#' It supports continuous, right-censored, and interval-censored survival outcomes.
#'
#' @param y Outcome vector. Numeric. Represents continuous outcomes or follow-up times.
#' Set to \code{NULL} when using \code{outcome_type = "interval-censored"}, as
#' values are derived from \code{left_time} and \code{right_time}.
#' @param status Optional event indicator vector (1 = event occurred, 0 = censored).
#' Required when \code{outcome_type = "right-censored"}. For interval-censored
#' outcomes, this is derived automatically from \code{left_time} and
#' \code{right_time}.
#' @param X_train_control Covariate matrix for the control forest. Rows correspond to samples,
#' columns to covariates.
#' @param X_train_treat Covariate matrix for the treatment forest.
#' @param treatment_indicator_train Vector indicating treatment assignment for training samples
#' (1 = treated, 0 = control).
#' @param X_test_control Optional covariate matrix for control forest test data. Defaults to
#' column means of \code{X_train_control} if NULL.
#' @param X_test_treat Optional covariate matrix for treatment forest test data. Defaults to
#' column means of \code{X_train_treat} if NULL.
#' @param treatment_indicator_test Optional vector indicating treatment assignment for test data.
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
#' @param timescale For survival outcomes: either \code{"time"} (original scale, log-transformed
#' internally) or \code{"log"} (already log-transformed). Default is \code{"time"}.
#' Used when \code{outcome_type} is \code{"right-censored"} or
#' \code{"interval-censored"}.
#' @param number_of_trees_control Number of trees in the control forest. Default is 200.
#' @param number_of_trees_treat Number of trees in the treatment forest. Default is 200.
#' @param prior_type_control Type of prior on control forest step heights. One of 
#' \code{"horseshoe"}, \code{"horseshoe_fw"}, or \code{"half-cauchy"}.
#' Default is \code{"horseshoe"}.
#' @param prior_type_treat Type of prior on treatment forest step heights. Same options as 
#' \code{prior_type_control}.
#' @param local_hp_control Local hyperparameter controlling shrinkage on individual steps 
#' (control forest). Required for all prior types.
#' @param local_hp_treat Local hyperparameter for treatment forest.
#' @param global_hp_control Global hyperparameter for control forest. Required for horseshoe-type
#' priors; ignored for \code{"half-cauchy"}.
#' @param global_hp_treat Global hyperparameter for treatment forest.
#' @param a_dirichlet_control First shape parameter of the Beta prior used in the
#' Dirichlet–Sparse splitting rule for the control forest. Together with 
#' `b_dirichlet_control`, it controls the expected sparsity level.
#' @param b_dirichlet_control Second shape parameter of the Beta prior for the 
#' sparsity level in the control forest. Larger values shrink splitting 
#' probabilities more strongly toward uniform sparsity.
#' @param rho_dirichlet_control Sparsity hyperparameter for the control forest.
#' Represents the *expected number of active predictors*. If left NULL, it 
#' defaults to the number of covariates in the control forest.
#' @param a_dirichlet_treat First shape parameter of the Beta prior used in the
#' Dirichlet–Sparse splitting rule for the treatment forest.
#' @param b_dirichlet_treat Second shape parameter of the Beta prior governing 
#' sparsity in the treatment forest.
#' @param rho_dirichlet_treat Sparsity hyperparameter for the treatment forest, 
#' interpreted as the expected number of active predictors. Defaults to the 
#' number of covariates in the treatment forest if not specified.
#' @param power_control Power parameter for the control forest tree structure prior splitting probability.
#' @param power_treat   Power parameter for the treatment forest tree structure prior splitting probability.
#' @param base_control  Base parameter for the control forest tree structure prior splitting probability.
#' @param base_treat    Base parameter for the treatment forest tree structure prior splitting probability.
#' @param p_grow Probability of proposing a grow move. Default is 0.5. These are fixed at 0.5 for prior_type
#' \code{"standard"} and \code{"dirichlet"}.
#' @param p_prune Probability of proposing a prune move. Default is 0.5. These are fixed at 0.5 for prior_type 
#' \code{"standard"} and \code{"dirichlet"}.
#' @param nu Degrees of freedom for the error variance prior. Default is 3.
#' @param q Quantile parameter for error variance prior. Default is 0.90.
#' @param sigma Optional known standard deviation of the outcome. If NULL, estimated from data.
#' @param N_post Number of posterior samples to store. Default is 5000.
#' @param N_burn Number of burn-in iterations. Default is 5000.
#' @param delayed_proposal Number of delayed iterations before proposal updates. Default is 5.
#' @param store_posterior_sample Logical; whether to store posterior samples of predictions.
#' Default is \code{FALSE}.
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
#' @param verbose Logical; whether to print verbose output. Default is \code{TRUE}.
#'
#' @return An S3 object of class \code{"CausalShrinkageForest"} containing:
#' \describe{
#'   \item{train_predictions}{Posterior mean predictions on training data (combined forest).}
#'   \item{test_predictions}{Posterior mean predictions on test data (combined forest).}
#'   \item{train_predictions_control}{Estimated control outcomes on training data.}
#'   \item{test_predictions_control}{Estimated control outcomes on test data.}
#'   \item{train_predictions_treat}{Estimated treatment effects on training data.}
#'   \item{test_predictions_treat}{Estimated treatment effects on test data.}
#'   \item{sigma}{Vector of posterior samples for the error standard deviation.}
#'   \item{acceptance_ratio_control}{Average acceptance ratio in control forest.}
#'   \item{acceptance_ratio_treat}{Average acceptance ratio in treatment forest.}
#'   \item{train_predictions_sample_control}{Matrix of posterior samples for control predictions 
#'   (if \code{store_posterior_sample = TRUE}).}
#'   \item{test_predictions_sample_control}{Matrix of posterior samples for control predictions 
#'   (if \code{store_posterior_sample = TRUE}).}
#'   \item{train_predictions_sample_treat}{Matrix of posterior samples for treatment effects 
#'   (if \code{store_posterior_sample = TRUE}).}
#'   \item{test_predictions_sample_treat}{Matrix of posterior samples for treatment effects 
#'   (if \code{store_posterior_sample = TRUE}).}
#' }
#'
#' @details
#' This function is a flexible generalization of \code{CausalHorseForest}.
#' The Causal Shrinkage Forest model decomposes the outcome into a prognostic 
#' (control) and a treatment effect part. Each part is modeled by its own 
#' shrinkage tree ensemble, with separate flexible global-local shrinkage 
#' priors. It is particularly useful for estimating heterogeneous treatment 
#' effects in high-dimensional settings. Further
#' methodological details on the Horseshoe Forest framework can be found in 
#' Jacobs, van Wieringen & van der Pas (2025).
#'
#' The \code{horseshoe} prior is the fully Bayesian global-local shrinkage
#' prior, where both the global and local shrinkage parameters are assigned
#' half-Cauchy distributions with scale hyperparameters \code{global_hp} and
#' \code{local_hp}, respectively. The global shrinkage parameter is defined
#' separately for each tree, allowing adaptive regularization per tree.
#'
#' The \code{horseshoe_fw} prior (forest-wide horseshoe) is similar to
#' \code{horseshoe}, except that the global shrinkage parameter is shared
#' across all trees in the forest simultaneously. 
#'
#' The \code{half-cauchy} prior considers only local shrinkage and does not
#' include a global shrinkage component. It places a half-Cauchy prior on each
#' local shrinkage parameter with scale hyperparameter \code{local_hp}.
#' 
#' The \code{dirichlet} prior implements the Dirichlet–Sparse splitting rule of 
#' Linero (2018), in which splitting probabilities follow a Dirichlet prior 
#' whose concentration is controlled by a Beta sparsity parameter 
#' (\code{a_dirichlet}, \code{b_dirichlet}) and an expected sparsity level 
#' \code{rho_dirichlet}.
#'
#' @references
#' Jacobs, T., van Wieringen, W. N., & van der Pas, S. L. (2025).  
#' *Horseshoe Forests for High-Dimensional Causal Survival Analysis.*  
#' arXiv:2507.22004. https://doi.org/10.48550/arXiv.2507.22004
#' 
#' Chipman, H. A., George, E. I., & McCulloch, R. E. (2010). 
#' *BART: Bayesian additive regression trees.* Annals of Applied Statistics.
#'
#' Linero, A. R. (2018). *Bayesian regression trees for high-dimensional 
#' prediction and variable selection.* Journal of the American Statistical 
#' Association.
#' 
#' @examples
#' # Example: Continuous outcome, homogeneous treatment effect, two priors
#' n <- 50
#' p <- 3
#' X <- matrix(runif(n * p), ncol = p)
#' X_treat <- X_control <- X
#' treat <- rbinom(n, 1, X[,1])
#' tau <- 2
#' y <- X[, 1] + (0.5 - treat) * tau + rnorm(n)
#' 
#' # Fit a standard Causal Horseshoe Forest
#' fit_horseshoe <- CausalShrinkageForest(y = y,
#'                                        X_train_control = X_control,
#'                                        X_train_treat = X_treat,
#'                                        treatment_indicator_train = treat,
#'                                        outcome_type = "continuous",
#'                                        number_of_trees_treat = 5,
#'                                        number_of_trees_control = 5,
#'                                        prior_type_control = "horseshoe",
#'                                        prior_type_treat = "horseshoe",
#'                                        local_hp_control = 0.1/sqrt(5),
#'                                        local_hp_treat = 0.1/sqrt(5),
#'                                        global_hp_control = 0.1/sqrt(5),
#'                                        global_hp_treat = 0.1/sqrt(5),
#'                                        N_post = 10,
#'                                        N_burn = 5,
#'                                        store_posterior_sample = TRUE,
#'                                        verbose = FALSE
#' )
#' 
#' # Fit a Causal Shrinkage Forest with half-cauchy prior
#' fit_halfcauchy <- CausalShrinkageForest(y = y,
#'                                         X_train_control = X_control,
#'                                         X_train_treat = X_treat,
#'                                         treatment_indicator_train = treat,
#'                                         outcome_type = "continuous",
#'                                         number_of_trees_treat = 5,
#'                                         number_of_trees_control = 5,
#'                                         prior_type_control = "half-cauchy",
#'                                         prior_type_treat = "half-cauchy",
#'                                         local_hp_control = 1/sqrt(5),
#'                                         local_hp_treat = 1/sqrt(5),
#'                                         N_post = 10,
#'                                         N_burn = 5,
#'                                         store_posterior_sample = TRUE,
#'                                         verbose = FALSE
#' )
#' 
#' # Posterior mean CATEs
#' CATE_horseshoe <- colMeans(fit_horseshoe$train_predictions_sample_treat)
#' CATE_halfcauchy <- colMeans(fit_halfcauchy$train_predictions_sample_treat)
#' 
#' # Posteriors of the ATE
#' post_ATE_horseshoe <- rowMeans(fit_horseshoe$train_predictions_sample_treat)
#' post_ATE_halfcauchy <- rowMeans(fit_halfcauchy$train_predictions_sample_treat)
#' 
#' # Posterior mean ATE
#' ATE_horseshoe <- mean(post_ATE_horseshoe)
#' ATE_halfcauchy <- mean(post_ATE_halfcauchy)
#'
#' # Example: Interval-censored causal survival outcome
#' n <- 50; p <- 3
#' X_ic <- matrix(rnorm(n * p), ncol = p)
#' treat_ic <- rbinom(n, 1, 0.5)
#' true_t <- rexp(n, rate = exp(-X_ic[, 1] - 0.5 * treat_ic))
#' left_t  <- true_t * runif(n, 0.5, 1)
#' right_t <- true_t * runif(n, 1, 1.5)
#' exact <- sample(n, 15)
#' left_t[exact] <- true_t[exact]; right_t[exact] <- true_t[exact]
#' rc <- sample(setdiff(seq_len(n), exact), 10); right_t[rc] <- Inf
#'
#' fit_ic <- CausalShrinkageForest(
#'   left_time = left_t, right_time = right_t,
#'   X_train_control = X_ic, X_train_treat = X_ic,
#'   treatment_indicator_train = treat_ic,
#'   outcome_type = "interval-censored",
#'   number_of_trees_control = 5, number_of_trees_treat = 5,
#'   prior_type_control = "horseshoe", prior_type_treat = "horseshoe",
#'   local_hp_control = 0.1/sqrt(5), local_hp_treat = 0.1/sqrt(5),
#'   global_hp_control = 0.1/sqrt(5), global_hp_treat = 0.1/sqrt(5),
#'   N_post = 10, N_burn = 5,
#'   store_posterior_sample = TRUE, verbose = FALSE)
#'
#' @seealso
#' Model family: \code{\link{CausalHorseForest}} (causal, horseshoe prior),
#' \code{\link{ShrinkageTrees}} (non-causal, flexible prior),
#' \code{\link{HorseTrees}} (non-causal, horseshoe prior).
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
#' @importFrom stats sd qchisq qnorm runif
#' @importFrom parallel mclapply detectCores
#' @export
CausalShrinkageForest <- function(y = NULL,
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
                                  number_of_trees_control = 200,
                                  number_of_trees_treat = 200,
                                  prior_type_control = "horseshoe",
                                  prior_type_treat = "horseshoe",
                                  local_hp_control = NULL,
                                  local_hp_treat = NULL,
                                  global_hp_control = NULL,
                                  global_hp_treat = NULL,
                                  a_dirichlet_control = 0.5,
                                  a_dirichlet_treat = 0.5,
                                  b_dirichlet_control = 1.0,
                                  b_dirichlet_treat = 1.0,
                                  rho_dirichlet_control = NULL,
                                  rho_dirichlet_treat = NULL,
                                  power_control = 2.0,
                                  power_treat = 2.0,
                                  base_control = 0.95,
                                  base_treat = 0.95,
                                  p_grow = 0.5,
                                  p_prune = 0.5,
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
      number_of_trees_control   = number_of_trees_control,
      number_of_trees_treat     = number_of_trees_treat,
      prior_type_control        = prior_type_control,
      prior_type_treat          = prior_type_treat,
      local_hp_control          = local_hp_control,
      local_hp_treat            = local_hp_treat,
      global_hp_control         = global_hp_control,
      global_hp_treat           = global_hp_treat,
      a_dirichlet_control       = a_dirichlet_control,
      a_dirichlet_treat         = a_dirichlet_treat,
      b_dirichlet_control       = b_dirichlet_control,
      b_dirichlet_treat         = b_dirichlet_treat,
      rho_dirichlet_control     = rho_dirichlet_control,
      rho_dirichlet_treat       = rho_dirichlet_treat,
      power_control             = power_control,
      power_treat               = power_treat,
      base_control              = base_control,
      base_treat                = base_treat,
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
      function(i) do.call(CausalShrinkageForest, chain_args),
      mc.cores = n_cores
    )

    combined      <- .combine_causal_chains(chains)
    combined$call <- mc
    return(combined)
  }

  # Check prior_type value
  allowed_prior <- c("horseshoe", "horseshoe_fw", "half-cauchy",
                     "standard", "dirichlet", "standard-halfcauchy", "dirichlet-halfcauchy")
  if (!prior_type_control %in% allowed_prior) {
    stop("Invalid prior_type_control Choose 'horseshoe', 'horseshoe_fw',
         'half-cauchy', 'standard', 'standard-halfnormal', 'standard-halfcauchy', 'dirichlet', or 'dirichlet-halfcauchy'.")
  }
  
  if (!prior_type_treat %in% allowed_prior) {
    stop("Invalid prior_type_treat. Choose 'horseshoe', 'horseshoe_fw',
         'half-cauchy', 'standard', 'standard-halfnormal', 'standard-halfcauchy', 'dirichlet', or 'dirichlet-halfcauchy'.")
  }
  
  # Prior-specific checks
  if (prior_type_control %in% c("horseshoe", "horseshoe_fw")) {
    if (is.null(local_hp_control) || is.null(global_hp_control)) {
      stop("For prior_type_control = 'horseshoe' or 'horseshoe_fw', you must provide both local_hp and global_hp.")
    }
  }
  
  if (prior_type_treat %in% c("horseshoe", "horseshoe_fw")) {
    if (is.null(local_hp_treat) || is.null(global_hp_treat)) {
      stop("For prior_type_treat = 'horseshoe' or 'horseshoe_fw', you must provide both local_hp and global_hp.")
    }
  }

  prior_type_control_user <- prior_type_control
  prior_type_treat_user <- prior_type_treat
  
  if (prior_type_control == "half-cauchy") {
    if (is.null(local_hp_control)) {
      stop("For prior_type = 'half-cauchy', you must provide local_hp_control.")
    }
    if (!is.null(global_hp_control)) {
      warning("global_hp_control is ignored for 'half-cauchy'.")
    }
    global_hp_control <- 1
    prior_type_control <- "halfcauchy"
  }
  
  if (prior_type_treat == "half-cauchy") {
    if (is.null(local_hp_treat)) {
      stop("For prior_type_treat = 'half-cauchy', you must provide 
           local_hp_treat.")
    }
    if (!is.null(global_hp_treat)) {
      warning("global_hp_treat is ignored for 'half-cauchy'.")
    }
    global_hp_treat <- 1
    prior_type_treat <- "halfcauchy"
  }
  
  reversible_flag_control <- TRUE
  reversible_flag_treat <- TRUE

  if (prior_type_control %in% c("standard", "dirichlet")) {
    if (is.null(local_hp_control)) {
      stop("For prior_type_control = 'standard' or 'dirichlet', you must provide local_hp.")
    }
    if (!is.null(global_hp_control)) {
      warning("global_hp_control is ignored for 'standard' or 'dirichlet' prior.")
    }

    global_hp_control <- 1          # placeholder (ignored by C++)
    reversible_flag_control <- FALSE

    if (delayed_proposal > 0) {
      delayed_proposal <- 0
    }
  }

  if (prior_type_control %in% c("dirichlet", "dirichlet-halfcauchy")) {
    dirichlet_bool_control <- TRUE
    if (prior_type_control == "dirichlet-halfcauchy") prior_type_control <- "standard-halfcauchy"
  } else {
    dirichlet_bool_control <- FALSE
  }

  if (prior_type_control %in% c("standard-halfcauchy", "dirichlet-halfcauchy")) {

    if (!is.null(global_hp_control)) {
      warning("global_hp_control is ignored for 'standard' or 'dirichlet' prior.")
    }

    global_hp_control <- 1          # placeholder (ignored by C++)
    reversible_flag_control <- FALSE

    if (delayed_proposal > 0) {
      delayed_proposal <- 0
    }
  }

  if (prior_type_treat %in% c("standard", "dirichlet", "standard-halfcauchy")) {
    if (is.null(local_hp_treat)) {
      stop("For prior_type_treat = 'standard' or 'dirichlet', you must provide local_hp_treat.")
    }
    if (!is.null(global_hp_treat)) {
      warning("global_hp_treat is ignored for 'standard' or 'dirichlet' prior.")
    }

    global_hp_treat <- 1          # placeholder (ignored by C++)
    reversible_flag_treat <- FALSE

    if (delayed_proposal > 0) {
      delayed_proposal <- 0
    }
  } 

  if (prior_type_treat %in% c("dirichlet", "dirichlet-halfcauchy")) {
    dirichlet_bool_treat <- TRUE
    if (prior_type_treat == "dirichlet-halfcauchy") prior_type_treat <- "standard-halfcauchy"
  } else {
    dirichlet_bool_treat <- FALSE
  }


  if (prior_type_treat %in% c("standard-halfcauchy", "dirichlet-halfcauchy")) {
    if (!is.null(global_hp_treat)) {
      warning("global_hp_treat is ignored for 'standard' or 'dirichlet' prior.")
    }

    global_hp_treat <- 1          # placeholder (ignored by C++)
    reversible_flag_treat <- FALSE

    if (delayed_proposal > 0) {
      delayed_proposal <- 0
    }
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
  if (p_control < 1L) stop("X_train_control must have at least one column.")
  if (p_treat   < 1L) stop("X_train_treat must have at least one column.")

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

  # Force data types to be numeric and plain arrays
  N_post <- as.integer(N_post)[1]
  N_burn <- as.integer(N_burn)[1]
  power_control <- as.numeric(power_control)[1]
  power_treat <- as.numeric(power_treat)[1]
  base_control <- as.numeric(base_control)[1]
  base_treat <- as.numeric(base_treat)[1]
  p_grow <- as.numeric(p_grow)[1]
  p_prune <- as.numeric(p_prune)[1]
  X_control_mat <- X_train_control        # preserve matrix for predict() / vi plots
  X_treat_mat   <- X_train_treat
  X_train_treat <- as.numeric(t(X_train_treat))
  X_train_control <- as.numeric(t(X_train_control))
  
  # Default rho to number of features
  if (is.null(rho_dirichlet_control)) {
    rho_dirichlet_control <- p_control
  }
  if (is.null(rho_dirichlet_treat)) {
    rho_dirichlet_treat <- p_treat
  }
  
  # For interval-censored: derive y/status/ic_indicator from left_time/right_time
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
  y_causal_raw      <- y
  status_causal_raw <- status
  left_time_raw     <- if (outcome_type == "interval-censored") left_time else NULL
  right_time_raw    <- if (outcome_type == "interval-censored") right_time else NULL

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

    # REFACTOR THIS CODE BELOW; make shorter
    if (prior_type_control %in% c("standard-halfcauchy")) {
      if (is.null(local_hp_control)) {
        local_hp_control <- 2.0
      } else {
        local_hp_control <- local_hp_control / sigma_hat
      }
    }

    if (prior_type_treat %in% c("standard-halfcauchy")) {
      if (is.null(local_hp_treat)) {
        local_hp_treat <- 2.0
      } else {
        local_hp_treat <- local_hp_treat / sigma_hat
      }
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
      no_trees_treatSEXP = number_of_trees_treat,
      power_treatSEXP = power_treat,
      base_treatSEXP = base_treat,
      p_grow_treatSEXP = p_grow,
      p_prune_treatSEXP = p_prune,
      omega_treatSEXP = 1/2,
      prior_type_treatSEXP = prior_type_treat,
      param1_treatSEXP = local_hp_treat,
      param2_treatSEXP = global_hp_treat,
      reversible_treatSEXP = reversible_flag_treat, 
      dirichlet_bool_treatSEXP = dirichlet_bool_treat,
      a_dirichlet_treatSEXP = a_dirichlet_treat,
      b_dirichlet_treatSEXP = b_dirichlet_treat, 
      rho_dirichlet_treatSEXP = rho_dirichlet_treat,  
      no_trees_controlSEXP = number_of_trees_control,
      power_controlSEXP = power_control,
      base_controlSEXP = base_control,
      p_grow_controlSEXP = p_grow,
      p_prune_controlSEXP = p_prune,
      omega_controlSEXP = 1/2,
      prior_type_controlSEXP = prior_type_control,
      param1_controlSEXP = local_hp_control,
      param2_controlSEXP = global_hp_control,
      reversible_controlSEXP = reversible_flag_control,
      dirichlet_bool_controlSEXP = dirichlet_bool_control,
      a_dirichlet_controlSEXP = a_dirichlet_control,
      b_dirichlet_controlSEXP = b_dirichlet_control, 
      rho_dirichlet_controlSEXP = rho_dirichlet_control,  
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

    if (prior_type_control %in% c("standard-halfcauchy")) {
      if (is.null(local_hp_control)) {
        local_hp_control <- 2.0
      } else {
        local_hp_control <- local_hp_control / sigma_hat
      }
    }

    if (prior_type_treat %in% c("standard-halfcauchy")) {
      if (is.null(local_hp_treat)) {
        local_hp_treat <- 2.0
      } else {
        local_hp_treat <- local_hp_treat / sigma_hat
      }
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
      no_trees_treatSEXP = number_of_trees_treat,
      power_treatSEXP = power_treat,
      base_treatSEXP = base_treat,
      p_grow_treatSEXP = p_grow,
      p_prune_treatSEXP = p_prune,
      omega_treatSEXP = 1/2,
      prior_type_treatSEXP = prior_type_treat,
      param1_treatSEXP = local_hp_treat,
      param2_treatSEXP = global_hp_treat,
      reversible_treatSEXP = reversible_flag_treat,
      dirichlet_bool_treatSEXP = dirichlet_bool_treat,
      a_dirichlet_treatSEXP = a_dirichlet_treat,
      b_dirichlet_treatSEXP = b_dirichlet_treat,
      rho_dirichlet_treatSEXP = rho_dirichlet_treat,
      no_trees_controlSEXP = number_of_trees_control,
      power_controlSEXP = power_control,
      base_controlSEXP = base_control,
      p_grow_controlSEXP = p_grow,
      p_prune_controlSEXP = p_prune,
      omega_controlSEXP = 1/2,
      prior_type_controlSEXP = prior_type_control,
      param1_controlSEXP = local_hp_control,
      param2_controlSEXP = global_hp_control,
      reversible_controlSEXP = reversible_flag_control,
      dirichlet_bool_controlSEXP = dirichlet_bool_control,
      a_dirichlet_controlSEXP = a_dirichlet_control,
      b_dirichlet_controlSEXP = b_dirichlet_control,
      rho_dirichlet_controlSEXP = rho_dirichlet_control,
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

    # REFACTOR THIS CODE BELOW; make shorter
    if (prior_type_control %in% c("standard-halfcauchy")) {
      if (is.null(local_hp_control)) {
        local_hp_control <- 2.0
      } else {
        local_hp_control <- local_hp_control / sigma_hat
      }
    }

    if (prior_type_treat %in% c("standard-halfcauchy")) {
      if (is.null(local_hp_treat)) {
        local_hp_treat <- 2.0
      } else {
        local_hp_treat <- local_hp_treat / sigma_hat
      }
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
      no_trees_treatSEXP = number_of_trees_treat,
      power_treatSEXP = power_treat,
      base_treatSEXP = base_treat,
      p_grow_treatSEXP = p_grow,
      p_prune_treatSEXP = p_prune,
      omega_treatSEXP = 1/2,
      prior_type_treatSEXP = prior_type_treat,
      param1_treatSEXP = local_hp_treat,
      param2_treatSEXP = global_hp_treat,
      reversible_treatSEXP = reversible_flag_treat, 
      dirichlet_bool_treatSEXP = dirichlet_bool_treat,
      a_dirichlet_treatSEXP = a_dirichlet_treat,
      b_dirichlet_treatSEXP = b_dirichlet_treat, 
      rho_dirichlet_treatSEXP = rho_dirichlet_treat,  
      no_trees_controlSEXP = number_of_trees_control,
      power_controlSEXP = power_control,
      base_controlSEXP = base_control,
      p_grow_controlSEXP = p_grow,
      p_prune_controlSEXP = p_prune,
      omega_controlSEXP = 1/2,
      prior_type_controlSEXP = prior_type_control,
      param1_controlSEXP = local_hp_control,
      param2_controlSEXP = global_hp_control,
      reversible_controlSEXP = reversible_flag_control,
      dirichlet_bool_controlSEXP = dirichlet_bool_control,
      a_dirichlet_controlSEXP = a_dirichlet_control,
      b_dirichlet_controlSEXP = b_dirichlet_control, 
      rho_dirichlet_controlSEXP = rho_dirichlet_control,
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

  prior_type_control_cpp <- prior_type_control
  prior_type_treat_cpp <- prior_type_treat
  dirichlet_bool_control <- FALSE
  dirichlet_bool_treat <- FALSE

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
    number_of_trees_control = number_of_trees_control,
    number_of_trees_treat = number_of_trees_treat,
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
    y_train                   = y_causal_raw,
    X_train_control           = X_control_mat,
    X_train_treat             = X_treat_mat,
    treatment_indicator_train = treatment_indicator_train,
    status_train              = if (outcome_type %in% c("right-censored", "interval-censored")) status_causal_raw else NULL,
    left_time_train           = left_time_raw,
    right_time_train          = right_time_raw,
    ic_indicator_train        = if (outcome_type == "interval-censored") ic_indicator else NULL,
    # Hyperparameters stored for predict()
    p_grow           = p_grow,
    p_prune          = p_prune,
    delayed_proposal = delayed_proposal,
    nu               = nu,
    lambda           = lambda,
    power_control    = power_control,
    base_control     = base_control,
    param1_control   = local_hp_control,
    param2_control   = global_hp_control,
    omega_control    = 1/2,
    reversible_control   = reversible_flag_control,
    a_dirichlet_control  = a_dirichlet_control,
    b_dirichlet_control  = b_dirichlet_control,
    rho_dirichlet_control = rho_dirichlet_control,
    power_treat      = power_treat,
    base_treat       = base_treat,
    param1_treat     = local_hp_treat,
    param2_treat     = global_hp_treat,
    omega_treat      = 1/2,
    reversible_treat     = reversible_flag_treat,
    a_dirichlet_treat    = a_dirichlet_treat,
    b_dirichlet_treat    = b_dirichlet_treat,
    rho_dirichlet_treat  = rho_dirichlet_treat,
    treatment_coding     = treatment_coding,
    propensity_train     = propensity
  )

  return(obj)
}

