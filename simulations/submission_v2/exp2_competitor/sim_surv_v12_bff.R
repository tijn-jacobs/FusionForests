# ──────────────────────────────────────────────────────────────────────────────
# Bayesian Fusion Forest (BFF) on the combined RCT + RWD data -- HPC.
#
# Same model and code as simulations/exp1_confounding_heterogeneity/sim_surv_v12.R, but the censoring
# mechanism is changed: BOTH sources are right-censored at ~35% (no interval
# censoring), matching the deepAFT / RSF competitor sims.  The FusionForest is
# fit to the pooled data and the CATE is reported per population (RCT/RWD/All).
#
# FusionForest is Bayesian, so the posterior CATE sample gives coverage / width /
# postvar directly (no bootstrap).  Output columns match the competitor sims
# (method, population, lambda_d, lambda_u, Iter, rmse, bias, coverage, width,
# postvar) so the rows can be row-bound for a side-by-side comparison.
# ──────────────────────────────────────────────────────────────────────────────

library(FusionForests)
library(ShrinkageTrees)
library(doParallel)
library(foreach)
library(evd)
library(MASS)

# ── Local FusionForest copy with the internal outcome scaling removed ─────────
# Verbatim from sim_surv_v12.R: the DGP pre-standardises the latent log-survival
# to mean 0, unit variance, so the fit must not re-estimate a scale from the
# (censored) data.  Bound to the FusionForests namespace so its calls to
# unexported internals (.call_cpp_backend, FusionForest_cpp, ...) resolve.
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
environment(FusionForest_temp) <- asNamespace("FusionForests")

# ── DGP ───────────────────────────────────────────── (verbatim from v12) ──────
p_total     <- 10L               # observed covariates

# Cross design: sweep one lambda over `lambda_vals` while holding the other at 1.
lambda_vals    <- c(0.0, 0.5, 1.0, 1.5, 2.0)  # swept values for each axis
lambda_d_fixed <- 1.0                          # held when sweeping lambda_u
lambda_u_fixed <- 1.0                          # held when sweeping lambda_d
sigma_rct   <- 0.75              # RCT residual SD
target_cens <- 0.35              # target right-censoring fraction (both sources)

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
solve_cens_rate <- function(logT, target = target_cens) {
  Tt <- exp(logT)
  uniroot(function(r) mean(1 - exp(-r * Tt)) - target, c(1e-10, 1e6))$root
}

# Per-cell calibration, fixed once on a large pre-simulation: the RCT and the RWD
# each get an independent exponential right-censoring rate tuned to ~target_cens.
calibrate <- function(ld, lu, n_cal = 1e5) {
  rate_rct <- solve_cens_rate(latent_rct(n_cal)$logT)
  rate_rwd <- solve_cens_rate(latent_rwd(n_cal, ld, lu)$logT)
  list(rate_rct = rate_rct, rate_rwd = rate_rwd)
}

# Simulate one full dataset.  Both sources are right-censored: the RCT and the
# RWD each get independent exponential censoring on the time scale, tuned to
# ~target_cens.  Outcomes are returned on the standardised LOG-time scale.
make_data <- function(n_rct, n_rwd, ld, lu, rate_rct, rate_rwd) {
  r <- latent_rct(n_rct)
  w <- latent_rwd(n_rwd, ld, lu)
  # Standardise the pooled latent log-survival to mean 0, unit variance; the same
  # affine map is applied to both sources.  Predicted CATEs come back on this
  # scale and are rescaled by `scale_sd` before evaluation.
  logT_true <- c(r$logT, w$logT)
  scale_mu  <- mean(logT_true)
  scale_sd  <- sd(logT_true)
  std       <- function(z) (z - scale_mu) / scale_sd

  # --- RCT: exponential right censoring ---
  logC_rct <- log(rexp(n_rct, rate = rate_rct))
  y_rct    <- pmin(r$logT, logC_rct)
  st_rct   <- as.integer(r$logT <= logC_rct)

  # --- RWD: exponential right censoring (same mechanism as the RCT) ---
  logC_rwd <- log(rexp(n_rwd, rate = rate_rwd))
  y_rwd    <- pmin(w$logT, logC_rwd)
  st_rwd   <- as.integer(w$logT <= logC_rwd)

  # --- standardise the log-time outcomes to the common scale ---
  y_rct <- std(y_rct)
  y_rwd <- std(y_rwd)

  list(
    X_rct = r$X, X_rwd = w$X,
    A_rct = r$A, A_rwd = w$A,
    y_rct = y_rct, y_rwd = y_rwd,
    st_rct = st_rct, st_rwd = st_rwd,
    X      = rbind(r$X, w$X),
    A      = c(r$A, w$A),
    S      = c(rep(1L, n_rct), rep(0L, n_rwd)),
    y      = c(y_rct, y_rwd),
    status = c(st_rct, st_rwd),
    e_rwd  = w$e,
    e_all  = c(rep(0.5, n_rct), w$e),
    scale_sd = scale_sd,
    n_rct = n_rct, n_rwd = n_rwd
  )
}

# ── Bayesian Fusion Forest fit (combined RCT + RWD) ───────────────────────────
# sim_surv_v12.R's fit_fusion, minus the interval-censoring arguments: both
# sources are right-censored, so we pass only (y, status).
fit_fusion <- function(d, N_post, N_burn) {
  FusionForest_temp(
    y                         = d$y,
    status                    = d$status,
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
    k_treat = 0.5,
    k_deconf = 0.5,
    k_deviation = 0.5,
    N_post = N_post, N_burn = N_burn,
    treatment_coding = "binary",
    store_posterior_sample = TRUE, verbose = FALSE)
}

# ── Metrics (posterior CATE sample) ───────────────────────────────────────────
# FusionForest returns posterior CATE draws (N_post x n_obs); coverage / width /
# postvar come from the 95% credible interval and the mean posterior variance.
as_obs_by_draw <- function(samp, n_obs) {
  if (nrow(samp) == n_obs) samp else t(samp)
}
eval_cate <- function(point, samp, truth, idx) {
  p  <- point[idx]; tr <- truth[idx]
  M  <- as_obs_by_draw(samp, length(point))[idx, , drop = FALSE]
  ci <- apply(M, 1, quantile, probs = c(0.025, 0.975))
  c(rmse     = sqrt(mean((p - tr)^2)),
    bias     = mean(p - tr),
    coverage = mean(tr >= ci[1, ] & tr <= ci[2, ]),
    width    = mean(ci[2, ] - ci[1, ]),
    postvar  = mean(apply(M, 1, var)))
}

metric_cols <- c("rmse", "bias", "coverage", "width", "postvar")

make_rows <- function(method, point, samp, truth, n_rct, n_all) {
  pops <- list(RCT = seq_len(n_rct),
               RWD = (n_rct + 1L):n_all,
               All = seq_len(n_all))
  do.call(rbind, lapply(names(pops), function(pop) {
    vals <- if (is.null(point))
      setNames(rep(NA_real_, length(metric_cols)), metric_cols)
    else eval_cate(point, samp, truth, pops[[pop]])
    data.frame(method = method, population = pop, as.list(vals),
               stringsAsFactors = FALSE)
  }))
}

# ── One replication: BFF on the combined dataset ──────────────────────────────
run_one_sim <- function(seed, n_rct, n_rwd, N_post, N_burn, ld, lu,
                        rate_rct, rate_rwd) {
  set.seed(seed)
  d     <- make_data(n_rct, n_rwd, ld, lu, rate_rct, rate_rwd)
  truth <- tau(d$X)
  n_all <- n_rct + n_rwd
  safe  <- function(expr) tryCatch(expr, error = function(e) {
    message("    BFF fit failed: ", conditionMessage(e)); NULL })

  s  <- d$scale_sd          # standardised-log CATE -> log-time scale
  ff <- safe(fit_fusion(d, N_post, N_burn))

  res <- make_rows("Fusion",
                   if (!is.null(ff)) ff$test_predictions_treat * s,
                   if (!is.null(ff)) ff$test_predictions_sample_treat * s,
                   truth, n_rct, n_all)
  res$cens_rct <- mean(d$st_rct == 0)
  res$cens_rwd <- mean(d$st_rwd == 0)
  res
}

# ── Parallel runner over replications (one grid cell) ─────────────────────────
run_simulation_tidy <- function(n_rep, n_rct, n_rwd, N_post, N_burn,
                                ld, lu, seed_offset = 1000L) {
  cal <- calibrate(ld, lu)        # per-cell RCT + RWD exponential censoring rates
  cat(sprintf("  RCT cens rate = %.4g | RWD cens rate = %.4g\n",
              cal$rate_rct, cal$rate_rwd))
  foreach(
    i = seq_len(n_rep),
    .combine = "rbind",
    .packages = c("FusionForests", "ShrinkageTrees", "evd", "MASS")
  ) %dopar% {
    res <- run_one_sim(seed_offset + i, n_rct, n_rwd, N_post, N_burn, ld, lu,
                       cal$rate_rct, cal$rate_rwd)
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
cat("SIMULATION: Bayesian Fusion Forest on combined RCT+RWD (right-censored) -- HPC\n")
cat("RCT and RWD both right-censored ~35%.\n")
cat("Covariates: MVN AR(1), rho =", rho_X, "\n")

# Simulation settings.
M       <- 1000L                 # replications per grid cell
n_rct   <- 150L                  # v12 settings
n_rwd   <- 350L                  # v12 settings
N_post  <- 5000L                 # v12 settings
N_burn  <- 5000L                 # v12 settings
cat(sprintf("M = %d reps/cell, n_rct = %d, n_rwd = %d, N_post = %d, N_burn = %d\n",
            M, n_rct, n_rwd, N_post, N_burn))

# Full 3x3 grid: lambda_d and lambda_u each in {0, 1, 2} (nine cells).
grid <- expand.grid(lambda_d = c(0, 1, 2), lambda_u = c(0, 1, 2))
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
output_file <- file.path(Sys.getenv("TMPDIR"), "sim_surv_v12_bff_output.rds")
cat("\nSaving all results to:", output_file, "\n")
saveRDS(final_flat_df, file = output_file)
cat("All results successfully saved in one file.\n")
