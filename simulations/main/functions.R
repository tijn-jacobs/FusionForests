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