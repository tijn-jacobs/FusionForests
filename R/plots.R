# -- Internal helpers ----------------------------------------------------------

.check_ggplot2 <- function() {
  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    stop(
      "Package 'ggplot2' is needed for plotting. ",
      "Install it with: install.packages('ggplot2')",
      call. = FALSE
    )
  }
}

# Reshape a concatenated posterior vector into a long data.frame
# with columns: iteration, chain, value.
.to_draws_df <- function(vec, par_name, N_post, n_chains) {
  mat <- matrix(vec, nrow = N_post, ncol = n_chains)
  data.frame(
    iteration = rep(seq_len(N_post), n_chains),
    chain     = factor(rep(seq_len(n_chains), each = N_post)),
    value     = as.vector(mat)
  )
}

# Extract feature names from a matrix, falling back to X1, X2, ...
.feature_names <- function(mat) {
  nms <- colnames(mat)
  if (is.null(nms)) nms <- paste0("X", seq_len(ncol(mat)))
  nms
}

# Build a top-n variable importance plot from a split_probs matrix.
.vi_intervals <- function(sp, X_mat, n_vi, title, ...) {
  colnames(sp) <- .feature_names(X_mat)
  means <- colMeans(sp)
  top <- order(means, decreasing = TRUE)[seq_len(min(n_vi, ncol(sp)))]
  sp_top <- sp[, top, drop = FALSE]

  # Compute summary statistics for each variable
  vi_df <- data.frame(
    variable = factor(colnames(sp_top), levels = rev(colnames(sp_top))),
    mean     = colMeans(sp_top),
    lo50     = apply(sp_top, 2, quantile, 0.25),
    hi50     = apply(sp_top, 2, quantile, 0.75),
    lo95     = apply(sp_top, 2, quantile, 0.025),
    hi95     = apply(sp_top, 2, quantile, 0.975)
  )

  ggplot2::ggplot(vi_df, ggplot2::aes(x = .data$mean, y = .data$variable)) +
    ggplot2::geom_linerange(
      ggplot2::aes(xmin = .data$lo95, xmax = .data$hi95),
      colour = "grey70", linewidth = 0.5
    ) +
    ggplot2::geom_linerange(
      ggplot2::aes(xmin = .data$lo50, xmax = .data$hi50),
      colour = "steelblue", linewidth = 1.5
    ) +
    ggplot2::geom_point(size = 2, colour = "steelblue") +
    ggplot2::labs(x = "Inclusion probability", y = NULL, title = title) +
    ggplot2::theme_bw()
}

# Traceplot: line plot of MCMC draws over iterations, coloured by chain.
.trace_plot <- function(vec, par_name, N_post, n_chains, ...) {
  df <- .to_draws_df(vec, par_name, N_post, n_chains)
  ggplot2::ggplot(df, ggplot2::aes(
    x = .data$iteration, y = .data$value, colour = .data$chain
  )) +
    ggplot2::geom_line(alpha = 0.7, linewidth = 0.3) +
    ggplot2::labs(x = "Iteration", y = par_name, colour = "Chain") +
    ggplot2::theme_bw()
}

# Overlaid density plot of MCMC draws, one curve per chain.
.density_plot <- function(vec, par_name, N_post, n_chains, ...) {
  df <- .to_draws_df(vec, par_name, N_post, n_chains)
  ggplot2::ggplot(df, ggplot2::aes(
    x = .data$value, colour = .data$chain, fill = .data$chain
  )) +
    ggplot2::geom_density(alpha = 0.15) +
    ggplot2::labs(x = par_name, y = "Density", colour = "Chain",
                  fill = "Chain") +
    ggplot2::theme_bw()
}

# Posterior density with shaded credible region.
.area_plot <- function(draws, par_name, prob = 0.95, ...) {
  ci <- quantile(draws, probs = c((1 - prob) / 2, 1 - (1 - prob) / 2))
  df <- data.frame(value = draws)
  ci_pct <- paste0(round(prob * 100), "%")

  ggplot2::ggplot(df, ggplot2::aes(x = .data$value)) +
    ggplot2::geom_density(fill = "steelblue", alpha = 0.3, colour = "steelblue") +
    ggplot2::geom_vline(xintercept = median(draws), colour = "steelblue",
                        linewidth = 0.8) +
    ggplot2::geom_vline(xintercept = ci, colour = "steelblue",
                        linetype = "dashed", linewidth = 0.5) +
    ggplot2::labs(x = par_name, y = "Density",
                  subtitle = paste0(ci_pct, " credible interval")) +
    ggplot2::theme_bw()
}


# -- Survival curve helper -----------------------------------------------------

# Compute posterior survival curves for specified observations or population avg.
# Returns a data.frame with columns: time, surv_mean, surv_lower, surv_upper, obs_id
.survival_curves <- function(x, obs, t_grid, level) {

  if (!x$outcome_type %in% c("right-censored", "interval-censored"))
    stop("Survival curve plot requires a survival outcome type.", call. = FALSE)

  alpha     <- (1 - level) / 2
  sigma_hat <- x$preprocess$sigma_hat
  sigma_log <- x$sigma * sigma_hat   # posterior sigma on log-time scale
  N_post    <- length(sigma_log)
  has_samples <- isTRUE(x$mcmc$store_posterior_sample) &&
                 !is.null(x$train_predictions_sample)

  # Reconstruct log-scale mu: y_train stores log-times for both timescale values
  if (has_samples) {
    mu_matrix <- if (x$timescale == "time")
      log(x$train_predictions_sample)  # N_post x n_train
    else
      x$train_predictions_sample
  }
  mu_mean <- if (x$timescale == "time")
    log(x$train_predictions)           # n_train vector
  else
    x$train_predictions

  # Auto-generate time grid on original time scale
  if (is.null(t_grid)) {
    orig_times <- exp(x$data$y_train)  # original-scale times
    t_lo  <- max(min(orig_times) * 0.5, 1e-6)
    t_hi  <- max(orig_times) * 1.2
    t_grid <- seq(t_lo, t_hi, length.out = 200)
  }
  log_t <- log(t_grid)  # length n_t
  n_t   <- length(t_grid)

  # -- Population average (obs = NULL) -----------------------------------------
  if (is.null(obs)) {
    n_train <- length(mu_mean)
    S_sum   <- matrix(0, N_post, n_t)

    if (has_samples) {
      for (i in seq_len(n_train)) {
        z <- (rep(1, N_post) %o% log_t - mu_matrix[, i]) / sigma_log
        S_sum <- S_sum + (1 - pnorm(z))
      }
    } else {
      for (i in seq_len(n_train)) {
        z <- (rep(1, N_post) %o% log_t - mu_mean[i]) / sigma_log
        S_sum <- S_sum + (1 - pnorm(z))
      }
    }
    S_avg <- S_sum / n_train

    df <- data.frame(
      time       = t_grid,
      surv_mean  = colMeans(S_avg),
      surv_lower = cummin(apply(S_avg, 2, quantile, alpha)),
      surv_upper = cummin(apply(S_avg, 2, quantile, 1 - alpha)),
      obs_id     = factor("Average")
    )
    return(df)
  }

  # -- Individual observations -------------------------------------------------
  if (any(obs < 1) || any(obs > length(mu_mean)))
    stop("obs indices must be between 1 and ", length(mu_mean), ".", call. = FALSE)

  dfs <- vector("list", length(obs))
  for (k in seq_along(obs)) {
    i <- obs[k]
    if (has_samples) {
      mu_i <- mu_matrix[, i]  # N_post vector
    } else {
      mu_i <- rep(mu_mean[i], N_post)
    }
    z <- (rep(1, N_post) %o% log_t - mu_i) / sigma_log  # N_post x n_t
    S <- 1 - pnorm(z)

    dfs[[k]] <- data.frame(
      time       = t_grid,
      surv_mean  = colMeans(S),
      surv_lower = cummin(apply(S, 2, quantile, alpha)),
      surv_upper = cummin(apply(S, 2, quantile, 1 - alpha)),
      obs_id     = factor(paste0("Obs ", i))
    )
  }
  do.call(rbind, dfs)
}


# Compute posterior predictive survival curves for a ShrinkageTreesPrediction.
# predictions_sample: N_post x n matrix, sigma: N_post vector, both on log-time
# scale if timescale == "time".
.survival_curves_pred <- function(x, obs, t_grid, level) {

  if (!x$outcome_type %in% c("right-censored", "interval-censored"))
    stop("Survival curve plot requires a survival outcome type.", call. = FALSE)
  if (is.null(x$predictions_sample) || is.null(x$sigma))
    stop("Survival curve plot requires posterior samples. ",
         "These are stored automatically when predict() is called on a ",
         "survival model.", call. = FALSE)

  alpha     <- (1 - level) / 2
  sigma_log <- x$sigma          # already on log-time scale
  N_post    <- length(sigma_log)

  # Log-scale mu matrix (N_post x n)
  mu_matrix <- if (x$timescale == "time")
    log(x$predictions_sample)
  else
    x$predictions_sample

  n_obs <- ncol(mu_matrix)

  # Auto-generate time grid on original time scale
  if (is.null(t_grid)) {
    orig_preds <- if (x$timescale == "time") x$mean else exp(x$mean)
    t_lo  <- max(min(orig_preds) * 0.2, 1e-6)
    t_hi  <- max(orig_preds) * 3
    t_grid <- seq(t_lo, t_hi, length.out = 200)
  }
  log_t <- log(t_grid)
  n_t   <- length(t_grid)

  # -- Population average (obs = NULL) -----------------------------------------
  if (is.null(obs)) {
    S_sum <- matrix(0, N_post, n_t)
    for (i in seq_len(n_obs)) {
      z <- (rep(1, N_post) %o% log_t - mu_matrix[, i]) / sigma_log
      S_sum <- S_sum + (1 - pnorm(z))
    }
    S_avg <- S_sum / n_obs

    df <- data.frame(
      time       = t_grid,
      surv_mean  = colMeans(S_avg),
      surv_lower = cummin(apply(S_avg, 2, quantile, alpha)),
      surv_upper = cummin(apply(S_avg, 2, quantile, 1 - alpha)),
      obs_id     = factor("Average")
    )
    return(df)
  }

  # -- Individual observations -------------------------------------------------
  if (any(obs < 1) || any(obs > n_obs))
    stop("obs indices must be between 1 and ", n_obs, ".", call. = FALSE)

  dfs <- vector("list", length(obs))
  for (k in seq_along(obs)) {
    i <- obs[k]
    mu_i <- mu_matrix[, i]
    z <- (rep(1, N_post) %o% log_t - mu_i) / sigma_log
    S <- 1 - pnorm(z)

    dfs[[k]] <- data.frame(
      time       = t_grid,
      surv_mean  = colMeans(S),
      surv_lower = cummin(apply(S, 2, quantile, alpha)),
      surv_upper = cummin(apply(S, 2, quantile, 1 - alpha)),
      obs_id     = factor(paste0("Obs ", i))
    )
  }
  do.call(rbind, dfs)
}


# Shared ggplot2 builder for survival curve data.frames.
# df: data.frame from .survival_curves() or .survival_curves_pred()
# km_data: optional data.frame(time, surv) for KM overlay
.plot_survival_curves <- function(df, obs, level, km_data = NULL,
                                  title_prefix = "Posterior") {
  ci_pct   <- paste0(round(level * 100), "%")
  n_curves <- nlevels(df$obs_id)

  if (n_curves == 1L) {
    p <- ggplot2::ggplot(df, ggplot2::aes(x = .data$time, y = .data$surv_mean)) +
      ggplot2::geom_ribbon(
        ggplot2::aes(ymin = .data$surv_lower, ymax = .data$surv_upper),
        fill = "steelblue", alpha = 0.3
      ) +
      ggplot2::geom_line(colour = "steelblue", linewidth = 0.8)
  } else {
    p <- ggplot2::ggplot(df, ggplot2::aes(
      x = .data$time, y = .data$surv_mean, colour = .data$obs_id,
      fill = .data$obs_id
    )) +
      ggplot2::geom_ribbon(
        ggplot2::aes(ymin = .data$surv_lower, ymax = .data$surv_upper),
        alpha = 0.15, colour = NA
      ) +
      ggplot2::geom_line(linewidth = 0.7)
  }

  title <- if (is.null(obs)) {
    paste0(title_prefix, " population-averaged survival curve")
  } else {
    paste0(title_prefix, " survival curve", if (n_curves > 1) "s" else "")
  }

  if (!is.null(km_data)) {
    p <- p +
      ggplot2::geom_step(
        data = km_data,
        ggplot2::aes(x = .data$time, y = .data$surv),
        colour = "black", linetype = "dashed", linewidth = 0.5,
        inherit.aes = FALSE
      )
  }

  p +
    ggplot2::labs(
      x        = "Time",
      y        = "Survival probability",
      title    = title,
      subtitle = paste0(ci_pct, " credible band"),
      colour   = NULL,
      fill     = NULL
    ) +
    ggplot2::ylim(0, 1) +
    ggplot2::theme_minimal()
}


# -- plot.ShrinkageTrees -------------------------------------------------------

#' Plot diagnostics for a ShrinkageTrees model
#'
#' Visualises posterior draws using \pkg{ggplot2}.
#' Requires the suggested package \pkg{ggplot2}.
#'
#' @param x A \code{ShrinkageTrees} object.
#' @param type Character; one of:
#'   \describe{
#'     \item{\code{"trace"}}{Sigma traceplot across MCMC iterations (one line
#'       per chain). Useful for assessing chain mixing.}
#'     \item{\code{"density"}}{Overlaid posterior density of sigma, one curve
#'       per chain.}
#'     \item{\code{"vi"}}{Posterior credible intervals for variable inclusion
#'       probabilities (top \code{n_vi} predictors). Only available for
#'       Dirichlet prior models.}
#'     \item{\code{"survival"}}{Posterior survival curves
#'     \eqn{S(t | x_i) = 1 - \Phi((\log t - \mu_i) / \sigma)} with pointwise
#'     credible bands, derived from the AFT log-normal model. Only available
#'     for survival outcome types (\code{"right-censored"} or
#'     \code{"interval-censored"}).
#'     \strong{Population-averaged curve} (default, \code{obs = NULL}):
#'     computes \eqn{\bar{S}(t) = n^{-1} \sum_i S(t | x_i)} at each MCMC
#'     iteration. The credible band reflects posterior uncertainty in both
#'     \eqn{\mu_i} and \eqn{\sigma} when \code{store_posterior_sample = TRUE},
#'     or sigma-only uncertainty otherwise.
#'     \strong{Individual curves} (\code{obs = c(1, 5, ...)}):
#'     one curve per selected training observation with its own credible band.
#'     Set \code{km = TRUE} to overlay the Kaplan--Meier estimate as a
#'     non-parametric reference (population-averaged plot only).}
#'   }
#' @param n_vi Integer; number of top variables to display when
#'   \code{type = "vi"}. Default \code{10}.
#' @param obs Integer vector of training-set observation indices for
#'   individual survival curves, or \code{NULL} (default) for the
#'   population-averaged curve. Indices must be between 1 and the number
#'   of training observations. Used only when \code{type = "survival"}.
#' @param t_grid Optional numeric vector of time points (on the original
#'   time scale) at which to evaluate the survival function. If \code{NULL}
#'   (default), a grid of 200 equally spaced points is generated from the
#'   range of observed training times. Used only when
#'   \code{type = "survival"}.
#' @param level Width of the pointwise credible band for
#'   \code{type = "survival"}. Default \code{0.95} (i.e.\ 95\% credible
#'   interval at each time point).
#' @param km Logical; if \code{TRUE} and \code{type = "survival"}
#'   with \code{obs = NULL}, overlay the Kaplan--Meier curve (dashed black
#'   step function) as a non-parametric reference. Default \code{FALSE}.
#'   Requires the \pkg{survival} package. Ignored with a message when
#'   \code{obs} is not \code{NULL}.
#' @param ... Additional arguments (currently unused).
#' @return A \pkg{ggplot2} object.
#'
#' @examples
#' \dontrun{
#' set.seed(1)
#' n <- 50; p <- 5
#' X <- matrix(rnorm(n * p), ncol = p)
#' y <- X[, 1] + rnorm(n)
#'
#' # Fit a continuous model
#' fit <- ShrinkageTrees(
#'   y = y, X_train = X,
#'   prior_type = "horseshoe",
#'   local_hp = 0.1, global_hp = 0.1,
#'   N_post = 200, N_burn = 100
#' )
#'
#' # Sigma traceplot -- check chain mixing
#' plot(fit, type = "trace")
#'
#' # Overlaid posterior densities of sigma per chain
#' plot(fit, type = "density")
#'
#' # --- Survival curves (requires survival outcome) ---
#' time <- rexp(n, rate = exp(0.5 * X[, 1]))
#' status <- rbinom(n, 1, 0.7)
#'
#' fit_surv <- SurvivalBART(
#'   time = time, status = status, X_train = X,
#'   number_of_trees = 10, N_post = 200, N_burn = 100,
#'   verbose = FALSE
#' )
#'
#' # Population-averaged survival curve with 95% credible band
#' plot(fit_surv, type = "survival")
#'
#' # With Kaplan-Meier overlay for comparison
#' plot(fit_surv, type = "survival", km = TRUE)
#'
#' # Individual curves for selected observations
#' plot(fit_surv, type = "survival", obs = c(1, 5, 10))
#'
#' # Single individual with 90% credible band
#' plot(fit_surv, type = "survival", obs = 1, level = 0.90)
#'
#' # Custom time grid
#' plot(fit_surv, type = "survival", obs = c(1, 10),
#'      t_grid = seq(0.1, 20, length.out = 100))
#' }
#' @export
plot.ShrinkageTrees <- function(x,
                                type = c("trace", "density", "vi", "survival"),
                                n_vi = 10,
                                obs = NULL,
                                t_grid = NULL,
                                level = 0.95,
                                km = FALSE,
                                ...) {
  type <- match.arg(type)
  .check_ggplot2()

  if (type == "survival") {
    df <- .survival_curves(x, obs = obs, t_grid = t_grid, level = level)

    # Build KM data if requested
    km_data <- NULL
    if (isTRUE(km) && !is.null(obs)) {
      message("km = TRUE is ignored for individual survival curves ",
              "(only available when obs = NULL).")
    }
    if (isTRUE(km) && is.null(obs)) {
      if (!requireNamespace("survival", quietly = TRUE)) {
        warning("Package 'survival' needed for KM overlay. Skipping.",
                call. = FALSE)
      } else {
        orig_times <- exp(x$data$y_train)
        km_fit <- survival::survfit(
          survival::Surv(orig_times, x$data$status_train) ~ 1
        )
        km_data <- data.frame(time = km_fit$time, surv = km_fit$surv)
      }
    }

    return(.plot_survival_curves(df, obs = obs, level = level,
                                 km_data = km_data))
  }

  n_ch   <- if (!is.null(x$mcmc$n_chains)) x$mcmc$n_chains else 1L
  N_post <- x$mcmc$N_post

  if (type == "trace") {
    .trace_plot(x$sigma, "sigma", N_post, n_ch, ...)

  } else if (type == "density") {
    .density_plot(x$sigma, "sigma", N_post, n_ch, ...)

  } else {
    sp <- x$split_probs
    if (is.null(sp))
      stop("Variable importance not available (split_probs is NULL).",
           call. = FALSE)
    .vi_intervals(sp, x$data$X_train, n_vi,
                  title = "Variable importance", ...)
  }
}


# -- plot.CausalShrinkageForest ------------------------------------------------

#' Plot diagnostics for a CausalShrinkageForest model
#'
#' Visualises posterior draws using \pkg{ggplot2}.
#' Requires the suggested package \pkg{ggplot2}.
#'
#' @param x A \code{CausalShrinkageForest} object.
#' @param type Character; one of:
#'   \describe{
#'     \item{\code{"trace"}}{Sigma traceplot (chain mixing).}
#'     \item{\code{"density"}}{Overlaid posterior density of sigma per chain.}
#'     \item{\code{"ate"}}{Posterior density of the average treatment effect
#'       (ATE) with 95 \% credible region. Requires
#'       \code{store_posterior_sample = TRUE}.}
#'     \item{\code{"cate"}}{Point estimates and 95 \% credible intervals for
#'       the CATE of each training observation, sorted by posterior mean.
#'       Requires \code{store_posterior_sample = TRUE}.}
#'     \item{\code{"vi"}}{Posterior credible intervals for variable inclusion
#'       probabilities (Dirichlet prior only). Controlled by \code{forest}.}
#'   }
#' @param forest For \code{type = "vi"}: which forest to display.
#'   One of \code{"both"} (default), \code{"control"}, or \code{"treat"}.
#'   When \code{"both"}, a named list of two \pkg{ggplot2} objects is returned.
#' @param n_vi Integer; number of top variables for \code{type = "vi"}.
#'   Default \code{10}.
#' @param ... Additional arguments (currently unused).
#' @return A \pkg{ggplot2} object, or (for \code{type = "vi"} with
#'   \code{forest = "both"}) a named list with elements \code{control} and
#'   \code{treat}.
#'
#' @examples
#' \dontrun{
#' set.seed(1)
#' n <- 60; p <- 5
#' X <- matrix(rnorm(n * p), ncol = p)
#' w <- rbinom(n, 1, 0.5)
#' y <- X[, 1] + w * 1.5 * (X[, 2] > 0) + rnorm(n, sd = 0.5)
#'
#' fit <- CausalShrinkageForest(
#'   y = y,
#'   X_train_control = X, X_train_treat = X,
#'   treatment_indicator_train = w,
#'   prior_type_control = "horseshoe", prior_type_treat = "horseshoe",
#'   local_hp_control = 0.1, global_hp_control = 0.1,
#'   local_hp_treat  = 0.1, global_hp_treat  = 0.1,
#'   N_post = 200, N_burn = 100,
#'   store_posterior_sample = TRUE
#' )
#'
#' # Sigma traceplot -- check chain mixing
#' plot(fit, type = "trace")
#'
#' # Overlaid posterior densities of sigma
#' plot(fit, type = "density")
#'
#' # Posterior distribution of the average treatment effect
#' plot(fit, type = "ate")
#'
#' # Conditional treatment effects for each observation
#' plot(fit, type = "cate")
#' }
#' @export
plot.CausalShrinkageForest <- function(x,
                                       type   = c("trace", "density",
                                                  "ate", "cate", "vi"),
                                       forest = c("both", "control", "treat"),
                                       n_vi   = 10,
                                       ...) {
  type   <- match.arg(type)
  forest <- match.arg(forest)
  .check_ggplot2()

  n_ch   <- if (!is.null(x$mcmc$n_chains)) x$mcmc$n_chains else 1L
  N_post <- x$mcmc$N_post

  # -- Sigma trace / density --------------------------------------------------
  if (type == "trace") {
    return(.trace_plot(x$sigma, "sigma", N_post, n_ch, ...))
  }

  if (type == "density") {
    return(.density_plot(x$sigma, "sigma", N_post, n_ch, ...))
  }

  # -- ATE posterior ----------------------------------------------------------
  if (type == "ate") {
    sc <- x$train_predictions_sample_control
    st <- x$train_predictions_sample_treat
    if (is.null(sc) || is.null(st))
      stop("ATE plot requires store_posterior_sample = TRUE.", call. = FALSE)

    ate_draws <- rowMeans(st) - rowMeans(sc)
    return(.area_plot(ate_draws, "ATE", prob = 0.95, ...))
  }

  # -- CATE distribution ------------------------------------------------------
  if (type == "cate") {
    sc <- x$train_predictions_sample_control
    st <- x$train_predictions_sample_treat
    if (is.null(sc) || is.null(st))
      stop("CATE plot requires store_posterior_sample = TRUE.", call. = FALSE)

    cate_draws <- st - sc
    cate_mean  <- colMeans(cate_draws)
    cate_lo    <- apply(cate_draws, 2, quantile, 0.025)
    cate_hi    <- apply(cate_draws, 2, quantile, 0.975)

    ord <- order(cate_mean)
    df  <- data.frame(
      obs  = seq_along(cate_mean),
      cate = cate_mean[ord],
      lo   = cate_lo[ord],
      hi   = cate_hi[ord]
    )

    return(
      ggplot2::ggplot(df, ggplot2::aes(x = .data$cate, y = .data$obs)) +
        ggplot2::geom_vline(xintercept = 0, linetype = "dashed",
                            colour = "grey50") +
        ggplot2::geom_linerange(
          ggplot2::aes(xmin = .data$lo, xmax = .data$hi),
          colour = "grey70"
        ) +
        ggplot2::geom_point(size = 0.8) +
        ggplot2::labs(
          x     = "CATE",
          y     = "Observation (sorted by posterior mean CATE)",
          title = "Conditional Average Treatment Effects"
        ) +
        ggplot2::theme_bw()
    )
  }

  # -- Variable importance ----------------------------------------------------
  if (forest %in% c("control", "both")) {
    sp_c <- x$split_probs_control
    if (is.null(sp_c))
      stop("Control variable importance not available.", call. = FALSE)
    p_control <- .vi_intervals(sp_c, x$data$X_train_control, n_vi,
                               title = "Variable importance - control forest",
                               ...)
  }
  if (forest %in% c("treat", "both")) {
    sp_t <- x$split_probs_treat
    if (is.null(sp_t))
      stop("Treatment variable importance not available.", call. = FALSE)
    p_treat <- .vi_intervals(sp_t, x$data$X_train_treat, n_vi,
                             title = "Variable importance - treatment forest",
                             ...)
  }

  if (forest == "control") return(p_control)
  if (forest == "treat")   return(p_treat)
  list(control = p_control, treat = p_treat)
}


# -- plot.ShrinkageTreesPrediction ---------------------------------------------

#' Plot posterior predictive survival curves
#'
#' Plots posterior predictive survival curves for new observations from
#' a \code{ShrinkageTreesPrediction} object. Only available for survival
#' outcome types (\code{"right-censored"} or \code{"interval-censored"}).
#'
#' @param x A \code{ShrinkageTreesPrediction} object returned by
#'   \code{\link{predict.ShrinkageTrees}} for a survival model.
#' @param type Character; currently only \code{"survival"} is supported.
#' @param obs Integer vector of predicted-observation indices for individual
#'   survival curves, or \code{NULL} (default) for the population-averaged
#'   curve across all predicted observations.
#' @param t_grid Optional numeric vector of time points (on the original
#'   time scale) at which to evaluate the survival function. If \code{NULL}
#'   (default), a grid of 200 equally spaced points is generated
#'   automatically.
#' @param level Width of the pointwise credible band. Default \code{0.95}.
#' @param ... Additional arguments (currently unused).
#' @return A \pkg{ggplot2} object.
#'
#' @examples
#' \dontrun{
#' # Fit a survival model
#' fit_surv <- SurvivalBART(
#'   time = time, status = status, X_train = X,
#'   number_of_trees = 20, N_post = 200, N_burn = 100,
#'   store_posterior_sample = TRUE, verbose = FALSE
#' )
#'
#' # Predict on new data
#' pred <- predict(fit_surv, newdata = X_test)
#'
#' # Population-averaged posterior predictive survival curve
#' plot(pred, type = "survival")
#'
#' # Individual posterior predictive curves
#' plot(pred, type = "survival", obs = c(1, 5, 10))
#' }
#' @seealso \code{\link{predict.ShrinkageTrees}},
#'   \code{\link{plot.ShrinkageTrees}}
#' @export
plot.ShrinkageTreesPrediction <- function(x,
                                          type = "survival",
                                          obs = NULL,
                                          t_grid = NULL,
                                          level = 0.95,
                                          ...) {
  type <- match.arg(type, choices = "survival")
  .check_ggplot2()

  df <- .survival_curves_pred(x, obs = obs, t_grid = t_grid, level = level)
  .plot_survival_curves(df, obs = obs, level = level,
                        title_prefix = "Posterior predictive")
}

