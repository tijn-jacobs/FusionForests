#' @noRd
NewShrinkageTrees <- function(
  fit,
  call,
  outcome_type,
  timescale,
  prior_type_user,
  prior_type_cpp,
  n_train,
  p_features,
  n_test,
  test_provided,
  number_of_trees,
  N_post,
  N_burn,
  store_posterior_sample,
  sigma_hat,
  sigma_known,
  y_mean,
  # Data stored for predict()
  y_train,
  X_train,
  status_train,
  left_time_train = NULL,
  right_time_train = NULL,
  ic_indicator_train = NULL,
  # Hyperparameters stored for predict()
  power,
  base,
  p_grow,
  p_prune,
  delayed_proposal,
  reversible,
  param1,
  param2,
  omega,
  a_dirichlet,
  b_dirichlet,
  rho_dirichlet,
  nu,
  lambda,
  latent_threshold
) {

  meta <- list(
    call = call,
    outcome_type = outcome_type,
    timescale = timescale,
    prior = list(
      prior_type_user = prior_type_user,
      prior_type_cpp = prior_type_cpp
    ),
    data_info = list(
      n_train = n_train,
      p_features = p_features,
      n_test = n_test,
      test_provided = test_provided
    ),
    mcmc = list(
      number_of_trees = number_of_trees,
      N_post = N_post,
      N_burn = N_burn,
      store_posterior_sample = store_posterior_sample
    ),
    preprocess = list(
      sigma_hat = sigma_hat,
      sigma_known = sigma_known,
      y_mean = y_mean,
      latent_threshold = latent_threshold
    ),
    data = list(
      y_train            = y_train,
      X_train            = X_train,
      status_train       = status_train,
      left_time_train    = left_time_train,
      right_time_train   = right_time_train,
      ic_indicator_train = ic_indicator_train
    ),
    args = list(
      power            = power,
      base             = base,
      p_grow           = p_grow,
      p_prune          = p_prune,
      delayed_proposal = delayed_proposal,
      reversible       = reversible,
      param1           = param1,
      param2           = param2,
      omega            = omega,
      a_dirichlet      = a_dirichlet,
      b_dirichlet      = b_dirichlet,
      rho_dirichlet    = rho_dirichlet,
      nu               = nu,
      lambda           = lambda
    )
  )

  obj <- c(fit, meta)
  class(obj) <- "ShrinkageTrees"
  obj
}

#' @noRd
NewCausalShrinkageForest <- function(
  fit,
  call,
  outcome_type,
  timescale,
  n_train,
  p_control,
  p_treat,
  n_test,
  test_provided,
  number_of_trees_control,
  number_of_trees_treat,
  N_post,
  N_burn,
  store_posterior_sample,
  sigma_hat,
  sigma_known,
  y_mean,
  prior_type_control_user,
  prior_type_control_cpp,
  prior_type_treat_user,
  prior_type_treat_cpp,
  dirichlet_bool_control = FALSE,
  dirichlet_bool_treat = FALSE,
  # Data stored for predict()
  y_train,
  X_train_control,
  X_train_treat,
  treatment_indicator_train,
  status_train,
  left_time_train = NULL,
  right_time_train = NULL,
  ic_indicator_train = NULL,
  # Hyperparameters stored for predict()
  p_grow,
  p_prune,
  delayed_proposal,
  nu,
  lambda,
  power_control,
  base_control,
  param1_control,
  param2_control,
  omega_control,
  reversible_control,
  a_dirichlet_control,
  b_dirichlet_control,
  rho_dirichlet_control,
  power_treat,
  base_treat,
  param1_treat,
  param2_treat,
  omega_treat,
  reversible_treat,
  a_dirichlet_treat,
  b_dirichlet_treat,
  rho_dirichlet_treat,
  treatment_coding = "centered",
  propensity_train = NULL
) {

  meta <- list(
    call = call,
    outcome_type = outcome_type,
    timescale = timescale,

    data_info = list(
      n_train = n_train,
      n_test = n_test,
      test_provided = test_provided,
      p_control = p_control,
      p_treat = p_treat
    ),

    mcmc = list(
      number_of_trees_control = number_of_trees_control,
      number_of_trees_treat = number_of_trees_treat,
      N_post = N_post,
      N_burn = N_burn,
      store_posterior_sample = store_posterior_sample
    ),

    preprocess = list(
      sigma_hat = sigma_hat,
      sigma_known = sigma_known,
      y_mean = y_mean
    ),

    prior = list(
      control = list(
        prior_type_user = prior_type_control_user,
        prior_type_cpp = prior_type_control_cpp,
        dirichlet = dirichlet_bool_control
      ),
      treat = list(
        prior_type_user = prior_type_treat_user,
        prior_type_cpp = prior_type_treat_cpp,
        dirichlet = dirichlet_bool_treat
      )
    ),

    data = list(
      y_train                  = y_train,
      X_train_control          = X_train_control,
      X_train_treat            = X_train_treat,
      treatment_indicator_train = treatment_indicator_train,
      status_train             = status_train,
      left_time_train          = left_time_train,
      right_time_train         = right_time_train,
      ic_indicator_train       = ic_indicator_train,
      propensity_train         = propensity_train
    ),

    treatment_coding = treatment_coding,

    args = list(
      p_grow           = p_grow,
      p_prune          = p_prune,
      delayed_proposal = delayed_proposal,
      nu               = nu,
      lambda           = lambda,
      control = list(
        power       = power_control,
        base        = base_control,
        param1      = param1_control,
        param2      = param2_control,
        omega       = omega_control,
        reversible  = reversible_control,
        a_dirichlet = a_dirichlet_control,
        b_dirichlet = b_dirichlet_control,
        rho_dirichlet = rho_dirichlet_control
      ),
      treat = list(
        power       = power_treat,
        base        = base_treat,
        param1      = param1_treat,
        param2      = param2_treat,
        omega       = omega_treat,
        reversible  = reversible_treat,
        a_dirichlet = a_dirichlet_treat,
        b_dirichlet = b_dirichlet_treat,
        rho_dirichlet = rho_dirichlet_treat
      )
    )
  )

  obj <- c(fit, meta)
  class(obj) <- "CausalShrinkageForest"
  obj
}

