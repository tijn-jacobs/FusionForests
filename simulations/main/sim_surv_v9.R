library(FusionForests)
library(ShrinkageTrees)
library(doParallel)
library(foreach)
library(evd)
library(MASS)

# Local copies of FusionForest / CausalShrinkageForest with the internal outcome
# scaling removed.  The DGP below pre-standardises the latent log-survival to
# mean 0, unit variance, so the fits no longer estimate a scale from the
# (censored) data -- which let us draw a residual sigma for the RWD as well.
CausalShrinkageForest_temp <- function (y = NULL, status = NULL, X_train_control, X_train_treat, 
                                        treatment_indicator_train, X_test_control = NULL, X_test_treat = NULL, 
                                        treatment_indicator_test = NULL, left_time = NULL, right_time = NULL, 
                                        outcome_type = "continuous", timescale = "time", number_of_trees_control = 200, 
                                        number_of_trees_treat = 200, prior_type_control = "horseshoe", 
                                        prior_type_treat = "horseshoe", local_hp_control = NULL, 
                                        local_hp_treat = NULL, global_hp_control = NULL, global_hp_treat = NULL, 
                                        a_dirichlet_control = 0.5, a_dirichlet_treat = 0.5, b_dirichlet_control = 1, 
                                        b_dirichlet_treat = 1, rho_dirichlet_control = NULL, rho_dirichlet_treat = NULL, 
                                        power_control = 2, power_treat = 2, base_control = 0.95, 
                                        base_treat = 0.95, p_grow = 0.5, p_prune = 0.5, nu = 3, q = 0.9, 
                                        sigma = NULL, N_post = 5000, N_burn = 5000, delayed_proposal = 5, 
                                        store_posterior_sample = FALSE, treatment_coding = "centered", 
                                        propensity = NULL, propensity_test = NULL, n_chains = 1, 
                                        verbose = TRUE) 
{
  if (n_chains > 1) {
    mc <- match.call()
    chain_args <- list(y = y, status = status, X_train_control = X_train_control, 
                       X_train_treat = X_train_treat, treatment_indicator_train = treatment_indicator_train, 
                       X_test_control = X_test_control, X_test_treat = X_test_treat, 
                       treatment_indicator_test = treatment_indicator_test, 
                       left_time = left_time, right_time = right_time, outcome_type = outcome_type, 
                       timescale = timescale, number_of_trees_control = number_of_trees_control, 
                       number_of_trees_treat = number_of_trees_treat, prior_type_control = prior_type_control, 
                       prior_type_treat = prior_type_treat, local_hp_control = local_hp_control, 
                       local_hp_treat = local_hp_treat, global_hp_control = global_hp_control, 
                       global_hp_treat = global_hp_treat, a_dirichlet_control = a_dirichlet_control, 
                       a_dirichlet_treat = a_dirichlet_treat, b_dirichlet_control = b_dirichlet_control, 
                       b_dirichlet_treat = b_dirichlet_treat, rho_dirichlet_control = rho_dirichlet_control, 
                       rho_dirichlet_treat = rho_dirichlet_treat, power_control = power_control, 
                       power_treat = power_treat, base_control = base_control, 
                       base_treat = base_treat, p_grow = p_grow, p_prune = p_prune, 
                       nu = nu, q = q, sigma = sigma, N_post = N_post, N_burn = N_burn, 
                       delayed_proposal = delayed_proposal, store_posterior_sample = store_posterior_sample, 
                       treatment_coding = treatment_coding, propensity = propensity, 
                       propensity_test = propensity_test, n_chains = 1, 
                       verbose = FALSE)
    n_cores <- .resolve_cores(n_chains)
    if (verbose) 
      message("Running ", n_chains, " chains (", n_cores, 
              " cores) ...")
    chains <- parallel::mclapply(seq_len(n_chains), function(i) do.call(CausalShrinkageForest, 
                                                                        chain_args), mc.cores = n_cores)
    combined <- .combine_causal_chains(chains)
    combined$call <- mc
    return(combined)
  }
  allowed_prior <- c("horseshoe", "horseshoe_fw", "half-cauchy", 
                     "standard", "dirichlet", "standard-halfcauchy", "dirichlet-halfcauchy")
  if (!prior_type_control %in% allowed_prior) {
    stop("Invalid prior_type_control Choose 'horseshoe', 'horseshoe_fw',\n         'half-cauchy', 'standard', 'standard-halfnormal', 'standard-halfcauchy', 'dirichlet', or 'dirichlet-halfcauchy'.")
  }
  if (!prior_type_treat %in% allowed_prior) {
    stop("Invalid prior_type_treat. Choose 'horseshoe', 'horseshoe_fw',\n         'half-cauchy', 'standard', 'standard-halfnormal', 'standard-halfcauchy', 'dirichlet', or 'dirichlet-halfcauchy'.")
  }
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
      stop("For prior_type_treat = 'half-cauchy', you must provide \n           local_hp_treat.")
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
    global_hp_control <- 1
    reversible_flag_control <- FALSE
    if (delayed_proposal > 0) {
      delayed_proposal <- 0
    }
  }
  if (prior_type_control %in% c("dirichlet", "dirichlet-halfcauchy")) {
    dirichlet_bool_control <- TRUE
    if (prior_type_control == "dirichlet-halfcauchy") 
      prior_type_control <- "standard-halfcauchy"
  }
  else {
    dirichlet_bool_control <- FALSE
  }
  if (prior_type_control %in% c("standard-halfcauchy", "dirichlet-halfcauchy")) {
    if (!is.null(global_hp_control)) {
      warning("global_hp_control is ignored for 'standard' or 'dirichlet' prior.")
    }
    global_hp_control <- 1
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
    global_hp_treat <- 1
    reversible_flag_treat <- FALSE
    if (delayed_proposal > 0) {
      delayed_proposal <- 0
    }
  }
  if (prior_type_treat %in% c("dirichlet", "dirichlet-halfcauchy")) {
    dirichlet_bool_treat <- TRUE
    if (prior_type_treat == "dirichlet-halfcauchy") 
      prior_type_treat <- "standard-halfcauchy"
  }
  else {
    dirichlet_bool_treat <- FALSE
  }
  if (prior_type_treat %in% c("standard-halfcauchy", "dirichlet-halfcauchy")) {
    if (!is.null(global_hp_treat)) {
      warning("global_hp_treat is ignored for 'standard' or 'dirichlet' prior.")
    }
    global_hp_treat <- 1
    reversible_flag_treat <- FALSE
    if (delayed_proposal > 0) {
      delayed_proposal <- 0
    }
  }
  allowed_types <- c("continuous", "right-censored", "interval-censored")
  if (!outcome_type %in% allowed_types) {
    stop("Invalid outcome_type. Please choose 'continuous', 'right-censored',\n         or 'interval-censored'.")
  }
  if (is.null(y) && outcome_type != "interval-censored") {
    stop("'y' is required when outcome_type is not 'interval-censored'.")
  }
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
  if (outcome_type == "right-censored" && is.null(status)) {
    stop("You specified outcome_type = 'right-censored', but did not provide a\n         'status' vector.")
  }
  if (!is.null(status) && !is.null(y) && length(status) != 
      length(y)) {
    stop("The length of 'status' must match the length of 'y'.")
  }
  if (!outcome_type %in% c("right-censored", "interval-censored") && 
      !is.null(status)) {
    warning("You provided a 'status' vector, but outcome_type is not\n            'right-censored' or 'interval-censored'. The 'status' vector will be ignored.")
  }
  if (outcome_type == "right-censored" && timescale == "time" && 
      any(y < 0)) {
    stop("Outcome contains negative values, but timescale = 'time' for survival\n         data requires non-negative times.")
  }
  if (!is.matrix(X_train_control)) 
    X_train_control <- as.matrix(X_train_control)
  if (!is.matrix(X_train_treat)) 
    X_train_treat <- as.matrix(X_train_treat)
  n_train <- nrow(X_train_control)
  p_control <- ncol(X_train_control)
  p_treat <- ncol(X_train_treat)
  n_obs <- if (outcome_type == "interval-censored") 
    length(left_time)
  else length(y)
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
  if (p_control < 1L) 
    stop("X_train_control must have at least one column.")
  if (p_treat < 1L) 
    stop("X_train_treat must have at least one column.")
  treatment_coding <- match.arg(treatment_coding, c("centered", 
                                                    "binary", "adaptive", "invariant"))
  if (treatment_coding == "adaptive" && is.null(propensity)) {
    stop("propensity scores are required when treatment_coding = 'adaptive'.")
  }
  if (!is.null(propensity) && length(propensity) != n_obs) {
    stop("Length of 'propensity' must match the number of training observations.")
  }
  propensity_train_cpp <- if (!is.null(propensity)) 
    as.numeric(propensity)
  else numeric(n_obs)
  test_provided_flag <- !is.null(X_test_control) && !is.null(X_test_treat)
  if (!is.null(X_test_control) && !is.null(X_test_treat)) {
    n_test <- nrow(X_test_control)
    if (ncol(X_test_control) != p_control || ncol(X_test_treat) != 
        p_treat) {
      stop("Number of columns in X_test_control or X_test_treat does not match \n           training data.")
    }
    if (!is.null(treatment_indicator_test) && length(treatment_indicator_test) != 
        n_test) {
      stop("treatment_indicator_test length must match number of test rows.")
    }
    X_test_control <- as.numeric(t(X_test_control))
    X_test_treat <- as.numeric(t(X_test_treat))
  }
  else {
    n_test <- 1
    X_test_control <- as.numeric(colMeans(X_train_control))
    X_test_treat <- as.numeric(colMeans(X_train_treat))
    treatment_indicator_test <- as.integer(rep(1, n_test))
  }
  treatment_indicator_train <- as.integer(treatment_indicator_train)
  if (is.null(treatment_indicator_test)) {
    treatment_indicator_test <- as.integer(rep(1, n_test))
  }
  else {
    treatment_indicator_test <- as.integer(treatment_indicator_test)
  }
  propensity_test_cpp <- if (!is.null(propensity_test)) {
    as.numeric(propensity_test)
  }
  else {
    rep(0.5, n_test)
  }
  N_post <- as.integer(N_post)[1]
  N_burn <- as.integer(N_burn)[1]
  power_control <- as.numeric(power_control)[1]
  power_treat <- as.numeric(power_treat)[1]
  base_control <- as.numeric(base_control)[1]
  base_treat <- as.numeric(base_treat)[1]
  p_grow <- as.numeric(p_grow)[1]
  p_prune <- as.numeric(p_prune)[1]
  X_control_mat <- X_train_control
  X_treat_mat <- X_train_treat
  X_train_treat <- as.numeric(t(X_train_treat))
  X_train_control <- as.numeric(t(X_train_control))
  if (is.null(rho_dirichlet_control)) {
    rho_dirichlet_control <- p_control
  }
  if (is.null(rho_dirichlet_treat)) {
    rho_dirichlet_treat <- p_treat
  }
  if (outcome_type == "interval-censored") {
    left_time <- as.numeric(left_time)
    right_time <- as.numeric(right_time)
    status <- as.integer(left_time == right_time)
    ic_indicator <- as.integer(left_time < right_time & is.finite(right_time))
    y <- ifelse(status == 1, left_time, ifelse(ic_indicator == 
                                                 1, (left_time + right_time)/2, left_time))
    right_time[!is.finite(right_time)] <- left_time[!is.finite(right_time)]
  }
  y_causal_raw <- y
  status_causal_raw <- status
  left_time_raw <- if (outcome_type == "interval-censored") 
    left_time
  else NULL
  right_time_raw <- if (outcome_type == "interval-censored") 
    right_time
  else NULL
  if (outcome_type == "right-censored") {
    y <- as.numeric(y)
    if (timescale == "time") {
      y <- log(y)
    }
    # No internal scaling: the caller pre-standardises the latent log-survival
    # to mean 0, unit variance.  We anchor the sigma prior at 1 and let the
    # sampler draw sigma (sigma_known = FALSE) unless sigma is supplied.
    y_mean <- 0
    y <- y - y_mean
    if (is.null(sigma)) {
      sigma_hat <- 1
      sigma_known <- FALSE
    }
    else {
      sigma_hat <- sigma
      sigma_known <- TRUE
    }
    if (prior_type_control %in% c("standard-halfcauchy")) {
      if (is.null(local_hp_control)) {
        local_hp_control <- 2
      }
      else {
        local_hp_control <- local_hp_control/sigma_hat
      }
    }
    if (prior_type_treat %in% c("standard-halfcauchy")) {
      if (is.null(local_hp_treat)) {
        local_hp_treat <- 2
      }
      else {
        local_hp_treat <- local_hp_treat/sigma_hat
      }
    }
    y <- y/sigma_hat
    survival <- TRUE
    qchi <- qchisq(1 - q, nu)
    lambda <- (sigma_hat^2 * qchi)/nu
    fit <- CausalHorseForest_cpp(nSEXP = n_train, p_treatSEXP = p_treat, 
                                 p_controlSEXP = p_control, X_train_treatSEXP = X_train_treat, 
                                 X_train_controlSEXP = X_train_control, ySEXP = y, 
                                 status_indicatorSEXP = status, is_survivalSEXP = survival, 
                                 observed_left_timeSEXP = numeric(n_train), observed_right_timeSEXP = y + 
                                   0, interval_censoring_indicatorSEXP = numeric(n_train), 
                                 treatment_indicatorSEXP = treatment_indicator_train, 
                                 n_testSEXP = n_test, X_test_controlSEXP = X_test_control, 
                                 X_test_treatSEXP = X_test_treat, treatment_indicator_testSEXP = treatment_indicator_test, 
                                 no_trees_treatSEXP = number_of_trees_treat, power_treatSEXP = power_treat, 
                                 base_treatSEXP = base_treat, p_grow_treatSEXP = p_grow, 
                                 p_prune_treatSEXP = p_prune, omega_treatSEXP = 1/2, 
                                 prior_type_treatSEXP = prior_type_treat, param1_treatSEXP = local_hp_treat, 
                                 param2_treatSEXP = global_hp_treat, reversible_treatSEXP = reversible_flag_treat, 
                                 dirichlet_bool_treatSEXP = dirichlet_bool_treat, 
                                 a_dirichlet_treatSEXP = a_dirichlet_treat, b_dirichlet_treatSEXP = b_dirichlet_treat, 
                                 rho_dirichlet_treatSEXP = rho_dirichlet_treat, no_trees_controlSEXP = number_of_trees_control, 
                                 power_controlSEXP = power_control, base_controlSEXP = base_control, 
                                 p_grow_controlSEXP = p_grow, p_prune_controlSEXP = p_prune, 
                                 omega_controlSEXP = 1/2, prior_type_controlSEXP = prior_type_control, 
                                 param1_controlSEXP = local_hp_control, param2_controlSEXP = global_hp_control, 
                                 reversible_controlSEXP = reversible_flag_control, 
                                 dirichlet_bool_controlSEXP = dirichlet_bool_control, 
                                 a_dirichlet_controlSEXP = a_dirichlet_control, b_dirichlet_controlSEXP = b_dirichlet_control, 
                                 rho_dirichlet_controlSEXP = rho_dirichlet_control, 
                                 sigma_knownSEXP = sigma_known, sigmaSEXP = sigma_hat, 
                                 lambdaSEXP = lambda, nuSEXP = nu, N_postSEXP = N_post, 
                                 N_burnSEXP = N_burn, delayed_proposalSEXP = delayed_proposal, 
                                 store_posterior_sample_controlSEXP = store_posterior_sample, 
                                 store_posterior_sample_treatSEXP = store_posterior_sample, 
                                 verboseSEXP = verbose, treatment_codingSEXP = treatment_coding, 
                                 propensity_trainSEXP = propensity_train_cpp, propensity_testSEXP = propensity_test_cpp)
    if (timescale == "time") {
      fit$train_predictions <- exp(fit$train_predictions * 
                                     sigma_hat + y_mean)
      fit$test_predictions <- exp(fit$test_predictions * 
                                    sigma_hat + y_mean)
      fit$train_predictions_control <- exp(fit$train_predictions_control * 
                                             sigma_hat + y_mean)
      fit$test_predictions_control <- exp(fit$test_predictions_control * 
                                            sigma_hat + y_mean)
      fit$train_predictions_treat <- exp(fit$train_predictions_treat * 
                                           sigma_hat)
      fit$test_predictions_treat <- exp(fit$test_predictions_treat * 
                                          sigma_hat)
      if (store_posterior_sample) {
        fit$train_predictions_sample_control <- exp(fit$train_predictions_sample_control * 
                                                      sigma_hat + y_mean)
        fit$test_predictions_sample_control <- exp(fit$test_predictions_sample_control * 
                                                     sigma_hat + y_mean)
        fit$train_predictions_sample_treat <- exp(fit$train_predictions_sample_treat * 
                                                    sigma_hat)
        fit$test_predictions_sample_treat <- exp(fit$test_predictions_sample_treat * 
                                                   sigma_hat)
      }
    }
    else {
      fit$train_predictions <- fit$train_predictions * 
        sigma_hat + y_mean
      fit$test_predictions <- fit$test_predictions * sigma_hat + 
        y_mean
      fit$train_predictions_control <- fit$train_predictions_control * 
        sigma_hat + y_mean
      fit$test_predictions_control <- fit$test_predictions_control * 
        sigma_hat + y_mean
      fit$train_predictions_treat <- fit$train_predictions_treat * 
        sigma_hat
      fit$test_predictions_treat <- fit$test_predictions_treat * 
        sigma_hat
      if (store_posterior_sample) {
        fit$train_predictions_sample_control <- fit$train_predictions_sample_control * 
          sigma_hat + y_mean
        fit$test_predictions_sample_control <- fit$test_predictions_sample_control * 
          sigma_hat + y_mean
        fit$train_predictions_sample_treat <- fit$train_predictions_sample_treat * 
          sigma_hat
        fit$test_predictions_sample_treat <- fit$test_predictions_sample_treat * 
          sigma_hat
      }
    }
  }
  else if (outcome_type == "interval-censored") {
    y <- as.numeric(y)
    if (timescale == "time") {
      y <- log(y)
      left_time <- log(left_time)
      right_time <- log(right_time)
    }
    # No internal scaling: the caller pre-standardises the latent log-survival
    # to mean 0, unit variance.  This sidesteps the survreg-based scale estimate
    # (which returns NaN when there are no exact event times) and lets the
    # sampler draw sigma from a unit-anchored prior for the RWD as well.
    y_mean <- 0
    y <- y - y_mean
    left_time <- left_time - y_mean
    right_time <- right_time - y_mean
    if (is.null(sigma)) {
      sigma_hat <- 1
      sigma_known <- FALSE
    }
    else {
      sigma_hat <- sigma
      sigma_known <- TRUE
    }
    if (prior_type_control %in% c("standard-halfcauchy")) {
      if (is.null(local_hp_control)) {
        local_hp_control <- 2
      }
      else {
        local_hp_control <- local_hp_control/sigma_hat
      }
    }
    if (prior_type_treat %in% c("standard-halfcauchy")) {
      if (is.null(local_hp_treat)) {
        local_hp_treat <- 2
      }
      else {
        local_hp_treat <- local_hp_treat/sigma_hat
      }
    }
    y <- y/sigma_hat
    left_time <- left_time/sigma_hat
    right_time <- right_time/sigma_hat
    survival <- TRUE
    qchi <- qchisq(1 - q, nu)
    lambda <- (sigma_hat^2 * qchi)/nu
    fit <- CausalHorseForest_cpp(nSEXP = n_train, p_treatSEXP = p_treat, 
                                 p_controlSEXP = p_control, X_train_treatSEXP = X_train_treat, 
                                 X_train_controlSEXP = X_train_control, ySEXP = y, 
                                 status_indicatorSEXP = status, is_survivalSEXP = survival, 
                                 observed_left_timeSEXP = left_time, observed_right_timeSEXP = right_time, 
                                 interval_censoring_indicatorSEXP = ic_indicator, 
                                 treatment_indicatorSEXP = treatment_indicator_train, 
                                 n_testSEXP = n_test, X_test_controlSEXP = X_test_control, 
                                 X_test_treatSEXP = X_test_treat, treatment_indicator_testSEXP = treatment_indicator_test, 
                                 no_trees_treatSEXP = number_of_trees_treat, power_treatSEXP = power_treat, 
                                 base_treatSEXP = base_treat, p_grow_treatSEXP = p_grow, 
                                 p_prune_treatSEXP = p_prune, omega_treatSEXP = 1/2, 
                                 prior_type_treatSEXP = prior_type_treat, param1_treatSEXP = local_hp_treat, 
                                 param2_treatSEXP = global_hp_treat, reversible_treatSEXP = reversible_flag_treat, 
                                 dirichlet_bool_treatSEXP = dirichlet_bool_treat, 
                                 a_dirichlet_treatSEXP = a_dirichlet_treat, b_dirichlet_treatSEXP = b_dirichlet_treat, 
                                 rho_dirichlet_treatSEXP = rho_dirichlet_treat, no_trees_controlSEXP = number_of_trees_control, 
                                 power_controlSEXP = power_control, base_controlSEXP = base_control, 
                                 p_grow_controlSEXP = p_grow, p_prune_controlSEXP = p_prune, 
                                 omega_controlSEXP = 1/2, prior_type_controlSEXP = prior_type_control, 
                                 param1_controlSEXP = local_hp_control, param2_controlSEXP = global_hp_control, 
                                 reversible_controlSEXP = reversible_flag_control, 
                                 dirichlet_bool_controlSEXP = dirichlet_bool_control, 
                                 a_dirichlet_controlSEXP = a_dirichlet_control, b_dirichlet_controlSEXP = b_dirichlet_control, 
                                 rho_dirichlet_controlSEXP = rho_dirichlet_control, 
                                 sigma_knownSEXP = sigma_known, sigmaSEXP = sigma_hat, 
                                 lambdaSEXP = lambda, nuSEXP = nu, N_postSEXP = N_post, 
                                 N_burnSEXP = N_burn, delayed_proposalSEXP = delayed_proposal, 
                                 store_posterior_sample_controlSEXP = store_posterior_sample, 
                                 store_posterior_sample_treatSEXP = store_posterior_sample, 
                                 verboseSEXP = verbose, treatment_codingSEXP = treatment_coding, 
                                 propensity_trainSEXP = propensity_train_cpp, propensity_testSEXP = propensity_test_cpp)
    if (timescale == "time") {
      fit$train_predictions <- exp(fit$train_predictions * 
                                     sigma_hat + y_mean)
      fit$test_predictions <- exp(fit$test_predictions * 
                                    sigma_hat + y_mean)
      fit$train_predictions_control <- exp(fit$train_predictions_control * 
                                             sigma_hat + y_mean)
      fit$test_predictions_control <- exp(fit$test_predictions_control * 
                                            sigma_hat + y_mean)
      fit$train_predictions_treat <- exp(fit$train_predictions_treat * 
                                           sigma_hat)
      fit$test_predictions_treat <- exp(fit$test_predictions_treat * 
                                          sigma_hat)
      if (store_posterior_sample) {
        fit$train_predictions_sample_control <- exp(fit$train_predictions_sample_control * 
                                                      sigma_hat + y_mean)
        fit$test_predictions_sample_control <- exp(fit$test_predictions_sample_control * 
                                                     sigma_hat + y_mean)
        fit$train_predictions_sample_treat <- exp(fit$train_predictions_sample_treat * 
                                                    sigma_hat)
        fit$test_predictions_sample_treat <- exp(fit$test_predictions_sample_treat * 
                                                   sigma_hat)
      }
    }
    else {
      fit$train_predictions <- fit$train_predictions * 
        sigma_hat + y_mean
      fit$test_predictions <- fit$test_predictions * sigma_hat + 
        y_mean
      fit$train_predictions_control <- fit$train_predictions_control * 
        sigma_hat + y_mean
      fit$test_predictions_control <- fit$test_predictions_control * 
        sigma_hat + y_mean
      fit$train_predictions_treat <- fit$train_predictions_treat * 
        sigma_hat
      fit$test_predictions_treat <- fit$test_predictions_treat * 
        sigma_hat
      if (store_posterior_sample) {
        fit$train_predictions_sample_control <- fit$train_predictions_sample_control * 
          sigma_hat + y_mean
        fit$test_predictions_sample_control <- fit$test_predictions_sample_control * 
          sigma_hat + y_mean
        fit$train_predictions_sample_treat <- fit$train_predictions_sample_treat * 
          sigma_hat
        fit$test_predictions_sample_treat <- fit$test_predictions_sample_treat * 
          sigma_hat
      }
    }
  }
  else {
    y <- as.numeric(y)
    status <- rep(1, n_train)
    survival <- FALSE
    # No internal scaling: caller pre-standardises y.
    if (is.null(sigma)) {
      sigma_hat <- 1
      sigma_known <- FALSE
    }
    else {
      sigma_hat <- sigma
      sigma_known <- TRUE
    }
    if (prior_type_control %in% c("standard-halfcauchy")) {
      if (is.null(local_hp_control)) {
        local_hp_control <- 2
      }
      else {
        local_hp_control <- local_hp_control/sigma_hat
      }
    }
    if (prior_type_treat %in% c("standard-halfcauchy")) {
      if (is.null(local_hp_treat)) {
        local_hp_treat <- 2
      }
      else {
        local_hp_treat <- local_hp_treat/sigma_hat
      }
    }
    qchi <- qchisq(1 - q, nu)
    lambda <- (sigma_hat^2 * qchi)/nu
    y_mean <- 0
    y <- y - y_mean
    y <- y/sigma_hat
    fit <- CausalHorseForest_cpp(nSEXP = n_train, p_treatSEXP = p_treat, 
                                 p_controlSEXP = p_control, X_train_treatSEXP = X_train_treat, 
                                 X_train_controlSEXP = X_train_control, ySEXP = y, 
                                 status_indicatorSEXP = status, is_survivalSEXP = survival, 
                                 observed_left_timeSEXP = numeric(n_train), observed_right_timeSEXP = y + 
                                   0, interval_censoring_indicatorSEXP = numeric(n_train), 
                                 treatment_indicatorSEXP = treatment_indicator_train, 
                                 n_testSEXP = n_test, X_test_controlSEXP = X_test_control, 
                                 X_test_treatSEXP = X_test_treat, treatment_indicator_testSEXP = treatment_indicator_test, 
                                 no_trees_treatSEXP = number_of_trees_treat, power_treatSEXP = power_treat, 
                                 base_treatSEXP = base_treat, p_grow_treatSEXP = p_grow, 
                                 p_prune_treatSEXP = p_prune, omega_treatSEXP = 1/2, 
                                 prior_type_treatSEXP = prior_type_treat, param1_treatSEXP = local_hp_treat, 
                                 param2_treatSEXP = global_hp_treat, reversible_treatSEXP = reversible_flag_treat, 
                                 dirichlet_bool_treatSEXP = dirichlet_bool_treat, 
                                 a_dirichlet_treatSEXP = a_dirichlet_treat, b_dirichlet_treatSEXP = b_dirichlet_treat, 
                                 rho_dirichlet_treatSEXP = rho_dirichlet_treat, no_trees_controlSEXP = number_of_trees_control, 
                                 power_controlSEXP = power_control, base_controlSEXP = base_control, 
                                 p_grow_controlSEXP = p_grow, p_prune_controlSEXP = p_prune, 
                                 omega_controlSEXP = 1/2, prior_type_controlSEXP = prior_type_control, 
                                 param1_controlSEXP = local_hp_control, param2_controlSEXP = global_hp_control, 
                                 reversible_controlSEXP = reversible_flag_control, 
                                 dirichlet_bool_controlSEXP = dirichlet_bool_control, 
                                 a_dirichlet_controlSEXP = a_dirichlet_control, b_dirichlet_controlSEXP = b_dirichlet_control, 
                                 rho_dirichlet_controlSEXP = rho_dirichlet_control, 
                                 sigma_knownSEXP = sigma_known, sigmaSEXP = sigma_hat, 
                                 lambdaSEXP = lambda, nuSEXP = nu, N_postSEXP = N_post, 
                                 N_burnSEXP = N_burn, delayed_proposalSEXP = delayed_proposal, 
                                 store_posterior_sample_controlSEXP = store_posterior_sample, 
                                 store_posterior_sample_treatSEXP = store_posterior_sample, 
                                 verboseSEXP = verbose, treatment_codingSEXP = treatment_coding, 
                                 propensity_trainSEXP = propensity_train_cpp, propensity_testSEXP = propensity_test_cpp)
    fit$train_predictions <- fit$train_predictions * sigma_hat + 
      y_mean
    fit$test_predictions <- fit$test_predictions * sigma_hat + 
      y_mean
    fit$train_predictions_control <- fit$train_predictions_control * 
      sigma_hat + y_mean
    fit$test_predictions_control <- fit$test_predictions_control * 
      sigma_hat + y_mean
    fit$train_predictions_treat <- fit$train_predictions_treat * 
      sigma_hat
    fit$test_predictions_treat <- fit$test_predictions_treat * 
      sigma_hat
    if (store_posterior_sample) {
      fit$train_predictions_sample <- fit$train_predictions_sample * 
        sigma_hat + y_mean
      fit$test_predictions_sample <- fit$test_predictions_sample * 
        sigma_hat + y_mean
      fit$train_predictions_sample_control <- fit$train_predictions_sample_control * 
        sigma_hat + y_mean
      fit$test_predictions_sample_control <- fit$test_predictions_sample_control * 
        sigma_hat + y_mean
      fit$train_predictions_sample_treat <- fit$train_predictions_sample_treat * 
        sigma_hat
      fit$test_predictions_sample_treat <- fit$test_predictions_sample_treat * 
        sigma_hat
    }
  }
  if (!sigma_known) {
    fit$sigma <- fit$sigma[-seq_len(N_burn)]
  }
  prior_type_control_cpp <- prior_type_control
  prior_type_treat_cpp <- prior_type_treat
  dirichlet_bool_control <- FALSE
  dirichlet_bool_treat <- FALSE
  obj <- NewCausalShrinkageForest(fit = fit, call = match.call(), 
                                  outcome_type = outcome_type, timescale = timescale, n_train = n_train, 
                                  p_control = p_control, p_treat = p_treat, n_test = n_test, 
                                  test_provided = test_provided_flag, number_of_trees_control = number_of_trees_control, 
                                  number_of_trees_treat = number_of_trees_treat, N_post = N_post, 
                                  N_burn = N_burn, store_posterior_sample = store_posterior_sample, 
                                  sigma_hat = sigma_hat, sigma_known = sigma_known, y_mean = y_mean, 
                                  prior_type_control_user = prior_type_control_user, prior_type_control_cpp = prior_type_control_cpp, 
                                  prior_type_treat_user = prior_type_treat_user, prior_type_treat_cpp = prior_type_treat_cpp, 
                                  dirichlet_bool_control = dirichlet_bool_control, dirichlet_bool_treat = dirichlet_bool_treat, 
                                  y_train = y_causal_raw, X_train_control = X_control_mat, 
                                  X_train_treat = X_treat_mat, treatment_indicator_train = treatment_indicator_train, 
                                  status_train = if (outcome_type %in% c("right-censored", 
                                                                         "interval-censored")) 
                                    status_causal_raw
                                  else NULL, left_time_train = left_time_raw, right_time_train = right_time_raw, 
                                  ic_indicator_train = if (outcome_type == "interval-censored") 
                                    ic_indicator
                                  else NULL, p_grow = p_grow, p_prune = p_prune, delayed_proposal = delayed_proposal, 
                                  nu = nu, lambda = lambda, power_control = power_control, 
                                  base_control = base_control, param1_control = local_hp_control, 
                                  param2_control = global_hp_control, omega_control = 1/2, 
                                  reversible_control = reversible_flag_control, a_dirichlet_control = a_dirichlet_control, 
                                  b_dirichlet_control = b_dirichlet_control, rho_dirichlet_control = rho_dirichlet_control, 
                                  power_treat = power_treat, base_treat = base_treat, param1_treat = local_hp_treat, 
                                  param2_treat = global_hp_treat, omega_treat = 1/2, reversible_treat = reversible_flag_treat, 
                                  a_dirichlet_treat = a_dirichlet_treat, b_dirichlet_treat = b_dirichlet_treat, 
                                  rho_dirichlet_treat = rho_dirichlet_treat, treatment_coding = treatment_coding, 
                                  propensity_train = propensity)
  return(obj)
}



FusionForest_temp <- function (y, status = NULL, observed_left_time = NULL, observed_right_time = NULL, 
                               interval_censoring_indicator = NULL, X_train_control, X_train_treat, 
                               treatment_indicator_train, source_indicator_train, X_test_control = NULL, 
                               X_test_treat = NULL, X_test_deconf = NULL, treatment_indicator_test = NULL, 
                               source_indicator_test = NULL, outcome_type = "continuous", 
                               timescale = "time", decomposition = "four-forest", number_of_trees_control = 200, 
                               number_of_trees_treat = 100, number_of_trees_deconf = 50, 
                               number_of_trees_deviation = 50, k_control = 0.5, k_treat = 0.5, 
                               k_deconf = 0.5, k_deviation = 0.5, power_control = 2, base_control = 0.95,
                               power_deviation = 2, base_deviation = 0.95, power_treat = 3, 
                               base_treat = 0.95, power_deconf = 3, base_deconf = 0.25, 
                               p_grow = 0.4, p_prune = 0.4, nu = 3, q = 0.9, sigma = NULL, 
                               N_post = 5000, N_burn = 5000, treatment_coding = c("centered", 
                                                                                  "binary", "adaptive"), propensity_train = NULL, propensity_test = NULL, 
                               error_dist = c("gaussian", "shared_dp", "source_dp", "source_dp_scale", 
                                              "source_hdp", "source_hdp_scale"), error_truncation_K = 50L, 
                               error_atom_scale = 0.5, error_mass_init = 1, store_posterior_sample = FALSE, 
                               verbose = TRUE) 
{
  allowed_types <- c("continuous", "right-censored")
  if (!outcome_type %in% allowed_types) 
    stop("Invalid outcome_type. Choose 'continuous' or 'right-censored'.")
  if (!identical(decomposition, "four-forest")) 
    stop("decomposition = ", deparse(decomposition), " is no longer ", 
         "supported. Only 'four-forest' is available in the R interface; ", 
         "the three-forest backend is retained in FusionForest_cpp.")
  treatment_coding <- match.arg(treatment_coding, c("centered", 
                                                    "binary", "adaptive"))
  error_dist <- match.arg(error_dist, c("gaussian", "shared_dp", 
                                        "source_dp", "source_dp_scale", "source_hdp", "source_hdp_scale"))
  mixture_mode <- switch(error_dist, gaussian = 0L, shared_dp = 1L, 
                         source_dp = 2L, source_dp_scale = 3L, source_hdp = 4L, 
                         source_hdp_scale = 5L)
  error_truncation_K <- as.integer(error_truncation_K)[1L]
  if (error_truncation_K < 2L) 
    stop("error_truncation_K must be at least 2.")
  if (!is.numeric(error_atom_scale) || length(error_atom_scale) != 
      1L || !is.finite(error_atom_scale) || error_atom_scale <= 
      0) 
    stop("error_atom_scale must be a single positive number.")
  if (!is.numeric(error_mass_init) || length(error_mass_init) != 
      1L || !is.finite(error_mass_init) || error_mass_init <= 
      0) 
    stop("error_mass_init must be a single positive number.")
  mixture_prior_atom_variance <- as.numeric(error_atom_scale)^2
  mixture_mass_init <- as.numeric(error_mass_init)
  if (outcome_type == "right-censored" && is.null(status)) 
    stop("outcome_type = 'right-censored' requires a 'status' vector.")
  if (outcome_type != "right-censored" && !is.null(status)) 
    warning("'status' is ignored for outcome_type = '", outcome_type, 
            "'.")
  if (outcome_type == "right-censored" && timescale == "time" && 
      any(y < 0)) 
    stop("Negative values in y with timescale = 'time': survival times must be non-negative.")
  n_train <- nrow(X_train_control)
  p_control <- ncol(X_train_control)
  p_treat <- ncol(X_train_treat)
  if (nrow(X_train_control) != length(y)) 
    stop("X_train_control rows must match length(y).")
  if (nrow(X_train_treat) != length(y)) 
    stop("X_train_treat rows must match length(y).")
  if (length(treatment_indicator_train) != length(y)) 
    stop("treatment_indicator_train must match length(y).")
  if (length(source_indicator_train) != length(y)) 
    stop("source_indicator_train must match length(y).")
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
  }
  else {
    propensity_train <- numeric(0)
  }
  n_deconf <- sum(source_indicator_train == 0L)
  if (n_deconf <= 0L) 
    stop("At least one real-world-data (source = 0) row is required.")
  X_train_deconf <- X_train_control[source_indicator_train == 
                                      0L, , drop = FALSE]
  p_deconf <- ncol(X_train_deconf)
  use_four_forest <- TRUE
  n_deviation <- n_deconf
  X_train_deviation <- X_train_deconf
  p_deviation <- p_deconf
  validate_k <- function(name, val) {
    if (!is.numeric(val) || length(val) != 1L || !is.finite(val) || 
        val < 0) 
      stop(sprintf("`%s` must be a single non-negative finite number.", 
                   name))
  }
  validate_k("k_control", k_control)
  validate_k("k_treat", k_treat)
  validate_k("k_deconf", k_deconf)
  validate_k("k_deviation", k_deviation)
  omega_control <- k_control/sqrt(number_of_trees_control)
  omega_treat <- k_treat/sqrt(number_of_trees_treat)
  omega_deconf <- k_deconf/sqrt(number_of_trees_deconf)
  if (use_four_forest) {
    omega_deviation <- if (k_deviation <= 0)
      1e-10
    else k_deviation/sqrt(number_of_trees_deviation)
  }
  else {
    omega_deviation <- NULL
  }
  if (!is.null(X_test_control) && !is.null(X_test_treat)) {
    if (!is.matrix(X_test_control) || !is.matrix(X_test_treat)) 
      stop("X_test_control and X_test_treat must be matrices.")
    n_test <- nrow(X_test_control)
    if (nrow(X_test_treat) != n_test) 
      stop("X_test_control and X_test_treat must have the same number of rows.")
    if (ncol(X_test_control) != p_control || ncol(X_test_treat) != 
        p_treat) 
      stop("Test matrix column counts must match training: p_control and p_treat.")
    source_indicator_test <- if (is.null(source_indicator_test)) {
      rep.int(1L, n_test)
    }
    else {
      s <- as.integer(source_indicator_test)
      if (length(s) != n_test) 
        stop("source_indicator_test length must match number of test rows.")
      if (!all(s %in% c(0L, 1L))) 
        stop("source_indicator_test must be 0 (RWD) or 1 (RCT).")
      s
    }
    treatment_indicator_test <- if (is.null(treatment_indicator_test)) {
      rep.int(1L, n_test)
    }
    else {
      t <- as.integer(treatment_indicator_test)
      if (length(t) != n_test) 
        stop("treatment_indicator_test length must match number of test rows.")
      t
    }
    if (treatment_coding == "adaptive") {
      if (is.null(propensity_test)) 
        stop("treatment_coding = 'adaptive' requires propensity_test.")
      if (length(propensity_test) != n_test) 
        stop("propensity_test must match the number of test rows.")
      if (any(propensity_test <= 0 | propensity_test >= 
              1)) 
        stop("propensity_test must lie strictly in (0, 1).")
      propensity_test <- as.numeric(propensity_test)
    }
    else {
      propensity_test <- numeric(0)
    }
    X_test_deconf <- if (is.null(X_test_deconf)) {
      X_test_control
    }
    else {
      if (!is.matrix(X_test_deconf)) 
        stop("X_test_deconf must be a matrix.")
      if (nrow(X_test_deconf) != n_test) 
        stop("X_test_deconf rows must match X_test_control.")
      if (ncol(X_test_deconf) != p_deconf) 
        stop("X_test_deconf must have ", p_deconf, " columns.")
      X_test_deconf
    }
    X_test_deviation <- X_test_deconf
    X_test_control <- as.numeric(t(X_test_control))
    X_test_treat <- as.numeric(t(X_test_treat))
    X_test_deconf <- as.numeric(t(X_test_deconf))
    X_test_deviation <- as.numeric(t(X_test_deviation))
  }
  else {
    n_test <- 1L
    X_test_control <- as.numeric(colMeans(X_train_control))
    X_test_treat <- as.numeric(colMeans(X_train_treat))
    X_test_deconf <- X_test_control
    X_test_deviation <- X_test_control
    treatment_indicator_test <- 1L
    source_indicator_test <- 1L
    if (treatment_coding == "adaptive") {
      if (is.null(propensity_test)) 
        stop("treatment_coding = 'adaptive' requires propensity_test.")
      propensity_test <- as.numeric(propensity_test)[1L]
      if (length(propensity_test) != 1L) 
        stop("propensity_test must have one entry when no test data is given.")
    }
    else {
      propensity_test <- numeric(0)
    }
  }
  X_train_control <- as.numeric(t(X_train_control))
  X_train_treat <- as.numeric(t(X_train_treat))
  X_train_deconf <- as.numeric(t(X_train_deconf))
  X_train_deviation <- as.numeric(t(X_train_deviation))
  N_post <- as.integer(N_post)[1L]
  N_burn <- as.integer(N_burn)[1L]
  power_control <- as.numeric(power_control)[1L]
  base_control <- as.numeric(base_control)[1L]
  power_deviation <- as.numeric(power_deviation)[1L]
  base_deviation <- as.numeric(base_deviation)[1L]
  power_treat <- as.numeric(power_treat)[1L]
  base_treat <- as.numeric(base_treat)[1L]
  power_deconf <- as.numeric(power_deconf)[1L]
  base_deconf <- as.numeric(base_deconf)[1L]
  p_grow <- as.numeric(p_grow)[1L]
  p_prune <- as.numeric(p_prune)[1L]
  if (outcome_type == "right-censored") {
    y <- as.numeric(y)
    if (timescale == "time") 
      y <- log(y)
    icArgsSupplied <- !(is.null(observed_left_time) && is.null(observed_right_time) && 
                          is.null(interval_censoring_indicator))
    if (icArgsSupplied) {
      if (is.null(observed_left_time) || is.null(observed_right_time) || 
          is.null(interval_censoring_indicator)) 
        stop("If any of observed_left_time / observed_right_time / ", 
             "interval_censoring_indicator is supplied, all three must be.")
      observed_left_time <- as.numeric(observed_left_time)
      observed_right_time <- as.numeric(observed_right_time)
      interval_censoring_indicator <- as.numeric(interval_censoring_indicator)
      if (length(observed_left_time) != n_train || length(observed_right_time) != 
          n_train || length(interval_censoring_indicator) != 
          n_train) 
        stop("observed_left_time / observed_right_time / ", 
             "interval_censoring_indicator must each have length ", 
             n_train, ".")
      if (any(interval_censoring_indicator == 1 & observed_left_time >= 
              observed_right_time)) 
        stop("For interval-censored rows we require ", 
             "observed_left_time < observed_right_time.")
      if (timescale == "time") {
        observed_left_time <- log(observed_left_time)
        observed_right_time <- log(observed_right_time)
      }
    }
    else {
      observed_left_time <- y
      observed_right_time <- y
      interval_censoring_indicator <- rep(0, n_train)
    }
    # No internal scaling: the caller pre-standardises the latent log-survival
    # to mean 0, unit variance.  We anchor the sigma prior at 1 and let the
    # sampler draw sigma (sigma_known = FALSE) unless sigma is supplied.
    y_mean <- 0
    y <- y - y_mean
    observed_left_time <- observed_left_time - y_mean
    observed_right_time <- observed_right_time - y_mean
    if (is.null(sigma)) {
      sigma_hat <- 1
      sigma_known <- FALSE
    }
    else {
      sigma_hat <- sigma
      sigma_known <- TRUE
    }
    y <- y/sigma_hat
    observed_left_time <- observed_left_time/sigma_hat
    observed_right_time <- observed_right_time/sigma_hat
    survival <- TRUE
    qchi <- qchisq(1 - q, nu)
    lambda <- (sigma_hat^2 * qchi)/nu
    fit <- .call_cpp_backend(use_four_forest, n_train, p_treat, 
                             p_control, X_train_treat, X_train_control, y, status, 
                             observed_left_time, observed_right_time, interval_censoring_indicator, 
                             survival, treatment_indicator_train, source_indicator_train, 
                             n_test, X_test_control, X_test_treat, X_test_deconf, 
                             X_test_deviation, treatment_indicator_test, source_indicator_test, 
                             n_deconf, p_deconf, X_train_deconf, number_of_trees_deconf, 
                             n_deviation, p_deviation, X_train_deviation, number_of_trees_deviation, 
                             omega_deviation, number_of_trees_treat, number_of_trees_control, 
                             omega_treat, omega_control, omega_deconf, power_control, 
                             base_control, power_deviation, base_deviation, power_treat, 
                             base_treat, power_deconf, base_deconf, p_grow, p_prune, 
                             sigma_known, sigma_hat, lambda, nu, N_post, N_burn, 
                             store_posterior_sample, verbose, treatment_coding, 
                             propensity_train, propensity_test, mixture_mode, 
                             error_truncation_K, mixture_prior_atom_variance, 
                             mixture_mass_init)
    if (timescale == "time") {
      fit$train_predictions <- exp(fit$train_predictions * 
                                     sigma_hat + y_mean)
      fit$test_predictions <- exp(fit$test_predictions * 
                                    sigma_hat + y_mean)
      fit$train_predictions_control <- exp(fit$train_predictions_control * 
                                             sigma_hat + y_mean)
      fit$test_predictions_control <- exp(fit$test_predictions_control * 
                                            sigma_hat + y_mean)
      fit$train_predictions_treat <- exp(fit$train_predictions_treat * 
                                           sigma_hat)
      fit$test_predictions_treat <- exp(fit$test_predictions_treat * 
                                          sigma_hat)
      fit$train_predictions_deconf <- exp(fit$train_predictions_deconf * 
                                            sigma_hat)
      fit$test_predictions_deconf <- exp(fit$test_predictions_deconf * 
                                           sigma_hat)
      if (use_four_forest) {
        fit$train_predictions_deviation <- exp(fit$train_predictions_deviation * 
                                                 sigma_hat)
        fit$test_predictions_deviation <- exp(fit$test_predictions_deviation * 
                                                sigma_hat)
      }
      if (store_posterior_sample) {
        fit$train_predictions_sample_control <- exp(fit$train_predictions_sample_control * 
                                                      sigma_hat + y_mean)
        fit$test_predictions_sample_control <- exp(fit$test_predictions_sample_control * 
                                                     sigma_hat + y_mean)
        fit$train_predictions_sample_treat <- exp(fit$train_predictions_sample_treat * 
                                                    sigma_hat)
        fit$test_predictions_sample_treat <- exp(fit$test_predictions_sample_treat * 
                                                   sigma_hat)
        fit$train_predictions_sample_deconf <- exp(fit$train_predictions_sample_deconf * 
                                                     sigma_hat)
        fit$test_predictions_sample_deconf <- exp(fit$test_predictions_sample_deconf * 
                                                    sigma_hat)
        if (use_four_forest) {
          fit$train_predictions_sample_deviation <- exp(fit$train_predictions_sample_deviation * 
                                                          sigma_hat)
          fit$test_predictions_sample_deviation <- exp(fit$test_predictions_sample_deviation * 
                                                         sigma_hat)
        }
      }
    }
    else {
      fit$train_predictions <- fit$train_predictions * 
        sigma_hat + y_mean
      fit$test_predictions <- fit$test_predictions * sigma_hat + 
        y_mean
      fit$train_predictions_control <- fit$train_predictions_control * 
        sigma_hat + y_mean
      fit$test_predictions_control <- fit$test_predictions_control * 
        sigma_hat + y_mean
      fit$train_predictions_treat <- fit$train_predictions_treat * 
        sigma_hat
      fit$test_predictions_treat <- fit$test_predictions_treat * 
        sigma_hat
      fit$train_predictions_deconf <- fit$train_predictions_deconf * 
        sigma_hat
      fit$test_predictions_deconf <- fit$test_predictions_deconf * 
        sigma_hat
      if (use_four_forest) {
        fit$train_predictions_deviation <- fit$train_predictions_deviation * 
          sigma_hat
        fit$test_predictions_deviation <- fit$test_predictions_deviation * 
          sigma_hat
      }
      if (store_posterior_sample) {
        fit$train_predictions_sample_control <- fit$train_predictions_sample_control * 
          sigma_hat + y_mean
        fit$test_predictions_sample_control <- fit$test_predictions_sample_control * 
          sigma_hat + y_mean
        fit$train_predictions_sample_treat <- fit$train_predictions_sample_treat * 
          sigma_hat
        fit$test_predictions_sample_treat <- fit$test_predictions_sample_treat * 
          sigma_hat
        fit$train_predictions_sample_deconf <- fit$train_predictions_sample_deconf * 
          sigma_hat
        fit$test_predictions_sample_deconf <- fit$test_predictions_sample_deconf * 
          sigma_hat
        if (use_four_forest) {
          fit$train_predictions_sample_deviation <- fit$train_predictions_sample_deviation * 
            sigma_hat
          fit$test_predictions_sample_deviation <- fit$test_predictions_sample_deviation * 
            sigma_hat
        }
      }
    }
  }
  else {
    y <- as.numeric(y)
    status <- rep(1L, n_train)
    survival <- FALSE
    # No internal scaling: caller pre-standardises y.
    if (is.null(sigma)) {
      sigma_hat <- 1
      sigma_known <- FALSE
    }
    else {
      sigma_hat <- sigma
      sigma_known <- TRUE
    }
    qchi <- qchisq(1 - q, nu)
    lambda <- (sigma_hat^2 * qchi)/nu
    y_mean <- 0
    y <- (y - y_mean)/sigma_hat
    observed_left_time <- y
    observed_right_time <- y
    interval_censoring_indicator <- rep(0, n_train)
    fit <- .call_cpp_backend(use_four_forest, n_train, p_treat, 
                             p_control, X_train_treat, X_train_control, y, status, 
                             observed_left_time, observed_right_time, interval_censoring_indicator, 
                             survival, treatment_indicator_train, source_indicator_train, 
                             n_test, X_test_control, X_test_treat, X_test_deconf, 
                             X_test_deviation, treatment_indicator_test, source_indicator_test, 
                             n_deconf, p_deconf, X_train_deconf, number_of_trees_deconf, 
                             n_deviation, p_deviation, X_train_deviation, number_of_trees_deviation, 
                             omega_deviation, number_of_trees_treat, number_of_trees_control, 
                             omega_treat, omega_control, omega_deconf, power_control, 
                             base_control, power_deviation, base_deviation, power_treat, 
                             base_treat, power_deconf, base_deconf, p_grow, p_prune, 
                             sigma_known, sigma_hat, lambda, nu, N_post, N_burn, 
                             store_posterior_sample, verbose, treatment_coding, 
                             propensity_train, propensity_test, mixture_mode, 
                             error_truncation_K, mixture_prior_atom_variance, 
                             mixture_mass_init)
    fit$train_predictions <- fit$train_predictions * sigma_hat + 
      y_mean
    fit$test_predictions <- fit$test_predictions * sigma_hat + 
      y_mean
    fit$train_predictions_control <- fit$train_predictions_control * 
      sigma_hat + y_mean
    fit$test_predictions_control <- fit$test_predictions_control * 
      sigma_hat + y_mean
    fit$train_predictions_treat <- fit$train_predictions_treat * 
      sigma_hat
    fit$test_predictions_treat <- fit$test_predictions_treat * 
      sigma_hat
    fit$train_predictions_deconf <- fit$train_predictions_deconf * 
      sigma_hat
    fit$test_predictions_deconf <- fit$test_predictions_deconf * 
      sigma_hat
    if (use_four_forest) {
      fit$train_predictions_deviation <- fit$train_predictions_deviation * 
        sigma_hat
      fit$test_predictions_deviation <- fit$test_predictions_deviation * 
        sigma_hat
    }
    if (store_posterior_sample) {
      fit$train_predictions_sample_control <- fit$train_predictions_sample_control * 
        sigma_hat + y_mean
      fit$test_predictions_sample_control <- fit$test_predictions_sample_control * 
        sigma_hat + y_mean
      fit$train_predictions_sample_treat <- fit$train_predictions_sample_treat * 
        sigma_hat
      fit$test_predictions_sample_treat <- fit$test_predictions_sample_treat * 
        sigma_hat
      fit$train_predictions_sample_deconf <- fit$train_predictions_sample_deconf * 
        sigma_hat
      fit$test_predictions_sample_deconf <- fit$test_predictions_sample_deconf * 
        sigma_hat
      if (use_four_forest) {
        fit$train_predictions_sample_deviation <- fit$train_predictions_sample_deviation * 
          sigma_hat
        fit$test_predictions_sample_deviation <- fit$test_predictions_sample_deviation * 
          sigma_hat
      }
    }
  }
  if (!sigma_known) 
    fit$sigma <- fit$sigma[-(1:N_burn)]
  if (!is.null(fit$dp_locations)) {
    fit$dp_locations <- lapply(fit$dp_locations, function(m) m * 
                                 sigma_hat)
    fit$error_dist <- error_dist
  }
  if (!is.null(fit$dp_sigma_g)) {
    fit$dp_sigma_g <- lapply(fit$dp_sigma_g, function(v) v * 
                               sigma_hat)
  }
  if (!is.null(fit$dp_locations_shared)) 
    fit$dp_locations_shared <- fit$dp_locations_shared * 
    sigma_hat
  if (!is.null(fit$dp_mu_g)) 
    fit$dp_mu_g <- lapply(fit$dp_mu_g, function(v) v * sigma_hat)
  fit$meta <- list(sigma_hat = sigma_hat, y_mean = y_mean, 
                   timescale = timescale, outcome_type = outcome_type, decomposition = decomposition, 
                   error_dist = error_dist)
  class(fit) <- c("FusionForest", class(fit))
  return(fit)
}

# These temp copies call unexported package internals (CausalHorseForest_cpp,
# NewCausalShrinkageForest, .call_cpp_backend, FusionForest_cpp, ...).  Bind each
# to its package namespace so those free variables resolve.
environment(CausalShrinkageForest_temp) <- asNamespace("ShrinkageTrees")
environment(FusionForest_temp)           <- asNamespace("FusionForests")

# ── DGP ───────────────────────────────────────────────────────────────────────
p_total     <- 10L               # observed covariates

# Cross design: sweep one lambda over `lambda_vals` while holding the other at 1.
lambda_vals    <- c(0.0, 0.5, 1.0, 1.5, 2.0)  # swept values for each axis
lambda_d_fixed <- 1.0                          # held when sweeping lambda_u
lambda_u_fixed <- 1.0                          # held when sweeping lambda_d
sigma_rct   <- 0.75              # RCT residual SD
target_cens <- 0.35              # target right-censoring fraction per source
visit_probs <- seq(0.1, 0.8, 0.1) # RWD inspection visits: deciles q10..q80


# Covariates: multivariate normal with AR(1) dependence, Sigma_ij = rho^|i-j|.
rho_X    <- 0.3
Sigma_X  <- rho_X ^ abs(outer(seq_len(p_total), seq_len(p_total), "-"))
draw_X   <- function(n) MASS::mvrnorm(n, mu = rep(0, p_total), Sigma = Sigma_X)

m0   <- function(X)  (2 * X[, 1] - X[, 2] * X[, 3] + 0.5 * X[, 4]^2)
dev  <- function(X, ld) ld * (X[, 4] - 0.5 * X[, 5])  # RWD baseline deviation
tau  <- function(X)  (1/2 + X[, 1] - 0.5 * X[, 2]^2)         # true CATE
conf <- function(U, lu) lu * (U)               # confounding via unobs. U

# Latent log-survival generators (no censoring), one per source.
latent_rct <- function(n) {
  X <- draw_X(n)
  A <- rbinom(n, 1L, 0.5)                              # randomised
  list(X = X, A = A,
       logT = m0(X) + A * tau(X) + rnorm(n, 0, sigma_rct))
}
latent_rwd <- function(n, ld, lu) {
  gamma_em   <- 0.5772156649
  beta_scale <- sqrt(6) / pi
  mu_loc     <- -beta_scale * gamma_em
  X <- draw_X(n)
  U <- runif(n)                                        # RWD-only, unobserved
  e <- plogis(X[, 1] + U)                              # true propensity score
  A <- rbinom(n, 1L, e)                                # selection on X1, U
  eps <- -rgumbel(n, loc = mu_loc, scale = beta_scale) # mean-0 residual (SD 1)
  list(X = X, A = A, e = e, eps = eps,
       logT = m0(X) + dev(X, ld) + A * tau(X) + A * conf(U, lu) + eps)
}

# Calibrate the RCT exponential censoring rate (time scale) to ~target.
# P(cens) = E[1 - exp(-rate * T)], monotone increasing in rate.
solve_cens_rate <- function(logT, target = target_cens) {
  Tt <- exp(logT)
  uniroot(function(r) mean(1 - exp(-r * Tt)) - target, c(1e-10, 1e6))$root
}

# Per-cell calibration, fixed once on a large pre-simulation:
#   rate_rct -- RCT exponential censoring rate.
#   visits   -- RWD inspection times (time scale) at the deciles q10..q80.
calibrate <- function(ld, lu, n_cal = 2e5) {
  rate_rct <- solve_cens_rate(latent_rct(n_cal)$logT)
  visits   <- as.numeric(quantile(exp(latent_rwd(n_cal, ld, lu)$logT),
                                  probs = visit_probs))
  list(rate_rct = rate_rct, visits = visits)
}

# Simulate one full dataset (RCT right-censored, RWD interval-censored).
# All time bounds are returned on the LOG scale (timescale = "log").
make_data <- function(n_rct, n_rwd, ld, lu, rate_rct, visits) {
  r <- latent_rct(n_rct)
  w <- latent_rwd(n_rwd, ld, lu)
  # Standardise the true (latent) log-survival of the pooled sources to mean 0,
  # unit variance.  The same affine map `std` is applied to every log-time bound
  # constructed below, so all fits operate on a common scale.  Predicted CATEs
  # are on this scale and are rescaled by `scale_sd` before evaluation.
  logT_true <- c(r$logT, w$logT)
  scale_mu  <- mean(logT_true)
  scale_sd  <- sd(logT_true)
  std       <- function(z) (z - scale_mu) / scale_sd
  logfloor <- min(c(r$logT, w$logT)) - 5    # finite stand-in for log(0)
  
  # --- RCT: exponential right censoring ---
  logC   <- log(rexp(n_rct, rate = rate_rct))
  y_rct  <- pmin(r$logT, logC)
  st_rct <- as.integer(r$logT <= logC)
  icc_rct   <- rep(0L, n_rct)               # 0 = not interval-censored
  left_rct  <- y_rct                        # ignored for status==1 / right-cens
  right_rct <- y_rct
  
  # --- RWD: interval censoring on the decile visits ---
  T0         <- exp(w$logT)
  last_visit <- visits[length(visits)]      # q80
  breaks     <- c(0, visits)                # bin edges: 0, q10, ..., q80
  j          <- findInterval(T0, breaks)    # 1..length(visits) below q80
  beyond     <- T0 >= last_visit            # event after the last visit
  j[!beyond] <- pmax(j[!beyond], 1L)
  
  left_rwd  <- ifelse(j == 1L, logfloor, log(breaks[pmin(j, length(breaks))]))
  right_rwd <- log(breaks[pmin(j + 1L, length(breaks))])
  st_rwd    <- rep(0L, n_rwd)               # no exact event times in the RWD
  icc_rwd   <- rep(1L, n_rwd)               # interval-censored by default
  # events beyond the last visit: right-censored at q80 (icc = 0)
  left_rwd[beyond]  <- log(last_visit)
  right_rwd[beyond] <- log(last_visit)
  icc_rwd[beyond]   <- 0L
  y_rwd <- right_rwd                        # placeholder; bounds carry the info

  # --- standardise all log-time bounds to the common scale ---
  y_rct     <- std(y_rct)
  left_rct  <- std(left_rct)
  right_rct <- std(right_rct)
  left_rwd  <- std(left_rwd)
  right_rwd <- std(right_rwd)
  y_rwd     <- std(y_rwd)

  list(
    X_rct = r$X, X_rwd = w$X,
    A_rct = r$A, A_rwd = w$A,
    y_rct = y_rct, y_rwd = y_rwd,
    st_rct = st_rct, st_rwd = st_rwd,
    left_rct = left_rct, right_rct = right_rct, icc_rct = icc_rct,
    left_rwd = left_rwd, right_rwd = right_rwd, icc_rwd = icc_rwd,
    X      = rbind(r$X, w$X),
    A      = c(r$A, w$A),
    S      = c(rep(1L, n_rct), rep(0L, n_rwd)),
    y      = c(y_rct, y_rwd),
    status = c(st_rct, st_rwd),
    left   = c(left_rct, left_rwd),
    right  = c(right_rct, right_rwd),
    icc    = c(icc_rct, icc_rwd),
    # True RWD propensity score (uses unobserved U).  e_rwd: training-row
    # propensity for the RWD-only fit.  e_all: propensity on the combined
    # evaluation set, with 0.5 for the randomised RCT rows.
    e_rwd  = w$e,
    e_all  = c(rep(0.5, n_rct), w$e),
    # SD used to standardise the pooled latent log-survival.  Predicted CATEs
    # come back on the standardised scale and are multiplied by this to recover
    # the log-time scale before evaluation against tau(X).
    scale_sd = scale_sd,
    n_rct = n_rct, n_rwd = n_rwd
  )
}

# ── Fits (all via FusionForest, log-time, with interval-censoring args) ───────
fit_fusion <- function(d, N_post, N_burn) {
  FusionForest_temp(
    y                            = d$y,
    status                       = d$status,
    observed_left_time           = d$left,
    observed_right_time          = d$right,
    interval_censoring_indicator = d$icc,
    X_train_control           = d$X,
    X_train_treat             = d$X,
    treatment_indicator_train = d$A,
    source_indicator_train    = d$S,
    X_test_control            = d$X,
    X_test_treat              = d$X,
    X_test_deconf             = d$X,
    treatment_indicator_test  = rep(1L, nrow(d$X)),
    source_indicator_test     = rep(1L, nrow(d$X)),
    outcome_type              = "right-censored",
    timescale                 = "log",
    error_dist                = "source_hdp_scale",
    number_of_trees_control = 200,
    number_of_trees_treat = 100,
    number_of_trees_deconf = 50,
    number_of_trees_deviation = 50,
    k_control = 0.5,
    k_treat = 0.25,
    k_deconf = 0.25,
    k_deviation = 0.5,
    N_post = N_post, N_burn = N_burn,
    treatment_coding = "binary",
    store_posterior_sample = TRUE, verbose = FALSE)
}

# ── Single-source fits (CausalShrinkageForest) ───────────────────────────────
# FusionForest is reserved for the multi-source fusion fit; a single-source
# FusionForest is ill-defined.  The RCT is right-censored; the RWD is interval-
# censored, so each baseline uses the matching CausalShrinkageForest outcome
# type.  Both predict the CATE on the combined covariate set Xev so that metrics
# can be split by population.
n_trees_csf <- 200L

# RCT-only: right-censored on the log-time scale.
fit_single_rc <- function(y, status, X, A, Xev, N_post, N_burn) {
  CausalShrinkageForest_temp(
    y                         = y,
    status                    = status,
    X_train_control           = X,
    X_train_treat             = X,
    treatment_indicator_train = A,
    X_test_control            = Xev,
    X_test_treat              = Xev,
    treatment_indicator_test  = rep(1L, nrow(Xev)),
    outcome_type              = "right-censored",
    timescale                 = "log",
    prior_type_control        = "standard",
    prior_type_treat          = "standard",
    local_hp_control          = 0.25 / sqrt(n_trees_csf),
    local_hp_treat            = 0.25 / sqrt(100),
    number_of_trees_control   = n_trees_csf,
    number_of_trees_treat     = 100,
    power_control = 2,
    power_treat = 3,
    base_control = 0.95,
    base_treat = 0.25,
    N_post = N_post, N_burn = N_burn,
    treatment_coding = "binary",
    store_posterior_sample = TRUE, verbose = FALSE)
}

# RWD-only: interval-censored on the log-time scale.  Bounds use the interval2
# convention: interval-censored rows have finite left < right; rows censored
# beyond the last visit have right = Inf (right-censored).
fit_single_ic <- function(left_time, right_time, X, A, Xev,
                          prop_train, prop_test, N_post, N_burn) {
  Xc    <- cbind(X,   propensity = prop_train)   # control forest gets e
  Xev_c <- cbind(Xev, propensity = prop_test)
  CausalShrinkageForest_temp(
    left_time                 = left_time,
    right_time                = right_time,
    X_train_control           = Xc,
    X_train_treat             = X,
    treatment_indicator_train = A,
    X_test_control            = Xev_c,
    X_test_treat              = Xev,
    treatment_indicator_test  = rep(1L, nrow(Xev)),
    outcome_type              = "interval-censored",
    timescale                 = "log",
    prior_type_control        = "standard",
    prior_type_treat          = "standard",
    local_hp_control          = 0.25 / sqrt(n_trees_csf),
    local_hp_treat            = 0.25 / sqrt(100),
    number_of_trees_control   = n_trees_csf,
    number_of_trees_treat     = 100,
    power_control = 2,
    power_treat = 3,
    base_control = 0.95,
    base_treat = 0.25,
    N_post = N_post, N_burn = N_burn,
    treatment_coding = "binary",
    store_posterior_sample = TRUE, verbose = FALSE)
}

# ── Metrics ───────────────────────────────────────────────────────────────────
# point : length-n posterior-mean CATE on the combined set.
# samp  : posterior CATE draws (FusionForest returns N_post x n_obs).  We coerce
#         to n_obs x N_post by matching the axis whose length equals length(point).
# truth : length-n true CATE.  idx : rows of the target population.
as_obs_by_draw <- function(samp, n_obs) {
  if (nrow(samp) == n_obs) samp else t(samp)
}
eval_cate <- function(point, samp, truth, idx) {
  p  <- point[idx]; tr <- truth[idx]
  M  <- as_obs_by_draw(samp, length(point))[idx, , drop = FALSE]  # n_idx x draws
  ci <- apply(M, 1, quantile, probs = c(0.025, 0.975))            # 2 x n_idx
  c(rmse     = sqrt(mean((p - tr)^2)),
    bias     = mean(p - tr),
    coverage = mean(tr >= ci[1, ] & tr <= ci[2, ]),
    width    = mean(ci[2, ] - ci[1, ]),
    # mean posterior variance of the CATE over evaluation points: the model's
    # own uncertainty per estimate (within-fit), distinct from the across-
    # replication estimator variance computed in post-processing.
    postvar  = mean(apply(M, 1, var)))
}

metric_cols <- c("rmse", "bias", "coverage", "width", "postvar")

make_rows <- function(method, point, samp, truth, n_rct, n_all) {
  pops <- list(RCT = seq_len(n_rct),
               RWD = (n_rct + 1L):n_all,
               All = seq_len(n_all))
  do.call(rbind, lapply(names(pops), function(pop) {
    vals <- if (is.null(point)) setNames(rep(NA_real_, length(metric_cols)), metric_cols)
    else eval_cate(point, samp, truth, pops[[pop]])
    data.frame(method = method, population = pop, as.list(vals),
               stringsAsFactors = FALSE)
  }))
}

# ── One replication: all three estimators on one dataset ─────────────────────
run_one_sim <- function(seed, n_rct, n_rwd, N_post, N_burn, ld, lu,
                        rate_rct, visits) {
  set.seed(seed)
  d     <- make_data(n_rct, n_rwd, ld, lu, rate_rct, visits)
  truth <- tau(d$X)
  n_all <- n_rct + n_rwd
  safe  <- function(expr) tryCatch(expr, error = function(e) NULL)
  
  # RWD interval2 bounds: keep finite (left, right] for interval-censored rows;
  # set right = Inf for rows censored beyond the last visit (right-censored).
  lt_rwd <- d$left_rwd
  rt_rwd <- d$right_rwd
  rt_rwd[d$icc_rwd == 0] <- Inf
  
  ff  <- safe(fit_fusion(d, N_post, N_burn))
  rct <- safe(fit_single_rc(d$y_rct, d$st_rct, d$X_rct, d$A_rct, d$X,
                            N_post, N_burn))
  rwd <- safe(fit_single_ic(lt_rwd, rt_rwd, d$X_rwd, d$A_rwd, d$X,
                            d$e_rwd, d$e_all, N_post, N_burn))

  # The fits run on the standardised log scale, so their treatment-effect
  # predictions are tau(X) / scale_sd.  Multiply back by scale_sd to recover the
  # log-time CATE before comparing with the true tau(X).
  s <- d$scale_sd
  res <- rbind(
    make_rows("Fusion",   if (!is.null(ff))  ff$test_predictions_treat * s,
              if (!is.null(ff))  ff$test_predictions_sample_treat * s,  truth, n_rct, n_all),
    make_rows("RCT-only", if (!is.null(rct)) rct$test_predictions_treat * s,
              if (!is.null(rct)) rct$test_predictions_sample_treat * s, truth, n_rct, n_all),
    make_rows("RWD-only", if (!is.null(rwd)) rwd$test_predictions_treat * s,
              if (!is.null(rwd)) rwd$test_predictions_sample_treat * s, truth, n_rct, n_all)
  )
  # realised censoring sanity-checks
  res$cens_rct    <- mean(d$st_rct == 0)            # RCT right-censored
  res$ic_rwd      <- mean(d$icc_rwd == 1)           # RWD interval-censored
  res$rc_rwd_tail <- mean(d$icc_rwd == 0)           # RWD right-censored past q80
  res
}

# ── Parallel runner over replications (one grid cell) ────────────────────────
run_simulation_tidy <- function(n_rep, n_rct, n_rwd, N_post, N_burn,
                                ld, lu, seed_offset = 1000L) {
  cal <- calibrate(ld, lu)        # per-cell RCT rate + RWD decile visits (fixed)
  cat(sprintf("  RCT cens rate = %.4g | RWD visits (q10..q80) = %s\n",
              cal$rate_rct, paste(round(cal$visits, 2), collapse = ", ")))
  foreach(
    i = seq_len(n_rep),
    .combine = "rbind",
    .packages = c("FusionForests", "ShrinkageTrees", "evd", "MASS")
  ) %dopar% {
    res <- run_one_sim(seed_offset + i, n_rct, n_rwd, N_post, N_burn, ld, lu,
                       cal$rate_rct, cal$visits)
    res$lambda_d <- ld
    res$lambda_u <- lu
    res$Iter     <- i
    res
  }
}

# ── Main (HPC) ────────────────────────────────────────────────────────────────
args <- commandArgs(trailingOnly = TRUE)
if (length(args) > 0) {
  num_cores <- as.integer(args[1]) - 1L
} else {
  num_cores <- parallel::detectCores() - 1L
}
registerDoParallel(cores = num_cores)
cat("Number of cores being used (1 free):", num_cores, "\n")
cat("SIMULATION: FusionForest vs RCT-only vs RWD-only (SURVIVAL v7) -- HPC\n")
cat("RWD interval-censored at deciles q10..q80; RCT right-censored ~35%.\n")
cat("Covariates: MVN AR(1), rho =", rho_X, "\n")

# Simulation settings.
M       <- num_cores * 1L        # replications per grid cell
n_rct   <- 200
n_rwd   <- 600L
N_post  <- 1000L
N_burn  <- 1000L
cat(sprintf("M = %d reps/cell, n_rct = %d, n_rwd = %d, N_post = %d, N_burn = %d\n",
            M, n_rct, n_rwd, N_post, N_burn))

# Cross grid: vary lambda_u (at lambda_d = 1) and vary lambda_d (at lambda_u = 1).
# The shared cell (lambda_d = 1, lambda_u = 1) appears in both slices; run once.
grid <- unique(rbind(
  data.frame(lambda_d = lambda_d_fixed, lambda_u = lambda_vals),   # vary u
  data.frame(lambda_d = lambda_vals,    lambda_u = lambda_u_fixed) # vary d
))
rownames(grid) <- NULL
cat(sprintf("Grid: %d (lambda_d, lambda_u) cells\n", nrow(grid)))

all_cells <- vector("list", nrow(grid))
for (g in seq_len(nrow(grid))) {
  ld <- grid$lambda_d[g]; lu <- grid$lambda_u[g]
  cat(sprintf("\nCell %d/%d: lambda_d = %.1f, lambda_u = %.1f\n",
              g, nrow(grid), ld, lu))
  all_cells[[g]] <- run_simulation_tidy(
    n_rep = M, n_rct = n_rct, n_rwd = n_rwd,
    N_post = N_post, N_burn = N_burn,
    ld = ld, lu = lu, seed_offset = 1000L * g)
}
final_flat_df <- do.call(rbind, all_cells)

# Output file path (! NAME MUST BE FILENAME_output.rds !).
output_file <- file.path(Sys.getenv("TMPDIR"), "sim_surv_v9_output.rds")
cat("\nSaving all results to:", output_file, "\n")
saveRDS(final_flat_df, file = output_file)
cat("All results successfully saved in one file.\n")


# ──────────────────────────────────────────────────────────────────────────────
# LOCAL POST-PROCESSING (uncomment after downloading the .rds from the HPC).
# Place this script and sim_surv_v9_output.rds in the same folder and run from the repo
# root.  Prints, per (lambda_d, lambda_u, method, population):
#   rmse     -- mean pointwise CATE RMSE over replications
#   bias     -- mean integrated (averaged-over-points) signed bias
#   variance -- across-replication variance of the integrated CATE, var(bias_r)
#   sd       -- sqrt(variance)
#   coverage, width, postvar -- as recorded
# ──────────────────────────────────────────────────────────────────────────────
# res <- readRDS("simulations/main/sim_surv_v9_output.rds")
# 
# grp <- cbind(rmse, bias, coverage, width, postvar) ~
#   lambda_d + lambda_u + method + population
# mean_tbl <- aggregate(grp, data = res,
#                       FUN = function(x) mean(x, na.rm = TRUE), na.action = na.pass)
# var_tbl  <- aggregate(bias ~ lambda_d + lambda_u + method + population,
#                       data = res, FUN = function(x) var(x, na.rm = TRUE), na.action = na.pass)
# names(var_tbl)[ncol(var_tbl)] <- "variance"
# summary_tbl <- merge(mean_tbl, var_tbl)
# summary_tbl$sd <- sqrt(summary_tbl$variance)
# 
# summary_tbl$method     <- factor(summary_tbl$method,
#                                  levels = c("Fusion", "RCT-only", "RWD-only"))
# summary_tbl$population  <- factor(summary_tbl$population,
#                                   levels = c("RCT", "RWD", "All"))
# ord <- order(summary_tbl$lambda_d, summary_tbl$lambda_u,
#              summary_tbl$population, summary_tbl$method)
# 
# cat("\n=== CATE metrics (mean over replications) ===\n")
# print(summary_tbl[ord, c("lambda_d", "lambda_u", "population", "method",
#                          "rmse", "bias", "variance", "sd", "coverage", "width",
#                          "postvar")],
#       row.names = FALSE, digits = 3)
# 
# # Realised censoring fractions per cell:
# # cens_tbl <- aggregate(cbind(cens_rct, ic_rwd, rc_rwd_tail) ~ lambda_d + lambda_u,
# #   data = res, FUN = function(x) mean(x, na.rm = TRUE))
# # print(cens_tbl, row.names = FALSE, digits = 3)
# 
# # Boxplots over replications (combined "All" population).  Each call draws a
# # 2x2 grid of metrics; within a panel the boxes are grouped by the swept lambda
# # from small to high, with the three estimators side by side per group, and one
# # shared legend beneath.  We make two such plots: one sweeping lambda_u (at
# # lambda_d = 1) and one sweeping lambda_d (at lambda_u = 1).
# a <- res[res$population == "All", ]
# meth <- c("Fusion", "RCT-only", "RWD-only")
# # Estimator colours: darker green (Fusion), orange (RCT-only), teal (RWD-only).
# cols <- c("goldenrod1", "olivedrab", "firebrick3")
# a$method <- factor(a$method, levels = meth)
# metrics  <- c("rmse", "bias", "coverage", "postvar")
# # Pretty y-axis labels for each metric.
# metric_labs <- c(rmse = "RMSE", bias = "bias", coverage = "coverage",
#                  width = "CI width", postvar = "posterior variance")
# 
# fig_dir <- "notes/general/figures"
# dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)
# # Draws the 2x2 metric grid to the active device (so it shows in the RStudio
# # plot pane) and, when `file` is given, also writes the same figure to PDF.
# plot_panel_grid <- function(dat, group, group_lab, file = NULL) {
#   dat[[group]] <- factor(dat[[group]], levels = sort(unique(dat[[group]])))
#   render <- function() {
#     op <- par(mfrow = c(2, 2),
#               family = "serif",         # serif font (Times / Computer Modern)
#               oma = c(3, 0, 0, 0),       # outer bottom margin for shared legend
#               mar = c(4, 4.5, 2, 1.5),   # per-panel margins (space between panels)
#               mgp = c(2.6, 0.8, 0),      # axis-title / label / line positions
#               cex.lab = 1.3, cex.axis = 1.1)  # larger fonts
#     nb  <- length(meth)               # estimators per group
#     ng  <- nlevels(dat[[group]])      # number of swept-lambda levels
#     gap <- 1.5                       # extra horizontal space between groups
#     at      <- as.vector(outer(seq_len(nb), (seq_len(ng) - 1) * (nb + gap), "+"))
#     centers <- (seq_len(ng) - 1) * (nb + gap) + (nb + 1) / 2
#     seps    <- centers[-ng] + diff(centers) / 2
#     for (metric in metrics) {
#       boxplot(dat[[metric]] ~ dat$method + dat[[group]], col = cols, xaxt = "n",
#               at = at, xlim = range(at) + c(-0.5, 0.5), outline = FALSE,
#               main = "", ylab = metric_labs[metric], xlab = group_lab, sep = " ")
#       axis(1, at = centers, labels = levels(dat[[group]]))
#       abline(v = seps, lty = 3, col = "grey")
#       if (metric == "coverage") abline(h = 0.95, lty = 2)
#       if (metric == "bias")     abline(h = 0,    lty = 2)
#     }
#     par(fig = c(0, 1, 0, 1), oma = c(0, 0, 0, 0), mar = c(0, 0, 0, 0),
#         family = "serif", new = TRUE)
#     plot.new()
#     legend("bottom", legend = meth, fill = cols, horiz = TRUE, bty = "n",
#            xpd = TRUE, cex = 1.2)
#     par(op)
#   }
#   render()                                   # 1) draw to the active device
#   if (!is.null(file)) {                       # 2) also write the PDF
#     pdf(file, width = 10, height = 8, family = "Times")
#     render()
#     dev.off()
#   }
# }
# 
# # Plot 1: vary lambda_u, holding lambda_d = 1.  Shown in-session and saved.
# plot_panel_grid(a[a$lambda_d == 1, ], "lambda_u", "lambda_u",
#                 file = file.path(fig_dir, "sim_vary_lambda_u.pdf"))
# # Plot 2: vary lambda_d, holding lambda_u = 1.  Shown in-session and saved.
# plot_panel_grid(a[a$lambda_u == 1, ], "lambda_d", "lambda_d",
#                 file = file.path(fig_dir, "sim_vary_lambda_d.pdf"))
# 
