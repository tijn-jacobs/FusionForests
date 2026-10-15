#' Posterior linear projection of the treatment effect
#'
#' Summarises the heterogeneous treatment effect by a linear model in a few
#' covariates.  Each posterior draw of the effect is projected onto a
#' linear basis by least squares (Woody, Carvalho and Murray, 2020).  The
#' result is a posterior sample of the projection coefficients.  No
#' refitting is needed.
#'
#' @details
#' At iteration \eqn{r}, the coefficients solve
#' \deqn{\gamma^{(r)} = \arg\min_{\gamma} \sum_i w_i^{(r)}
#'   \{\theta^{(r)}(x_i) - \phi(x_i)^{\top}\gamma\}^2.}{gamma^(r) =
#'   argmin_gamma sum_i w_i^(r) (theta^(r)(x_i) - phi(x_i)' gamma)^2.}
#' Here \eqn{\theta^{(r)}} is \eqn{\tau^{(r)}} or
#' \eqn{\exp\tau^{(r)}}{exp(tau^(r))}, depending on \code{target} and
#' \code{af_scale}, and \eqn{\phi} is the basis given by \code{basis}.
#'
#' The weights are equal by default.  With
#' \code{weights = "bayesian_bootstrap"}, they are drawn from a
#' Dirichlet(1, ..., 1) distribution at each iteration (Rubin, 1981).  This
#' adds the uncertainty about the covariate distribution.
#'
#' The basis columns other than the intercept are centred on \code{X_eval},
#' and optionally scaled.  The centring and scaling constants are returned
#' as attributes.
#'
#' @param fit A \code{FusionForest} fit with
#'   \code{store_posterior_sample = TRUE}.
#' @param basis One-sided formula for the basis, such as
#'   \code{~ x1 + x2 + x1:x2}.
#' @param X_eval Data frame with the covariates of the evaluation rows: the
#'   test or training rows of the fit (see \code{which}).  It must contain
#'   the variables in \code{basis}.
#' @param target \code{"cate"} projects \eqn{\tau(x)} on the log-time
#'   scale.  \code{"af"} projects the acceleration factor; see
#'   \code{af_scale}.
#' @param af_scale For \code{target = "af"} only.  \code{"multiplicative"}
#'   (default) projects \eqn{\exp\tau}{exp(tau)}, so a coefficient is the
#'   change in the acceleration factor per unit of the covariate.
#'   \code{"log"} projects \eqn{\tau}, as \code{target = "cate"} does.
#' @param which Evaluation rows: \code{"test"} (default) or \code{"train"}.
#'   These rows define the target population.
#' @param weights \code{"uniform"} (default) or \code{"bayesian_bootstrap"}.
#' @param center,scale Centre or scale the basis columns other than the
#'   intercept.  Defaults: \code{TRUE} and \code{FALSE}.
#' @param seed Optional seed for the Bayesian bootstrap weights.
#'
#' @return A matrix of posterior draws of the coefficients, with one row
#'   per iteration and one column per basis term.  The attributes record
#'   \code{target}, \code{af_scale}, \code{which}, \code{weights}, the
#'   centring and scaling constants, and the \code{formula}.
#'
#' @references
#' Woody, S., Carvalho, C.~M., and Murray, J.~S. (2020).  Bayesian inference
#' with posterior projection.  \emph{Journal of Computational and Graphical
#' Statistics}, 29(4), 798--808.
#'
#' Rubin, D.~B. (1981).  The Bayesian bootstrap.  \emph{The Annals of
#' Statistics}, 9, 130--134.
#'
#' @examples
#' # Continuous fusion fit that stores the posterior draws
#' set.seed(1)
#' n <- 100
#' X <- matrix(rnorm(n * 3), n, 3)
#' colnames(X) <- paste0("x", 1:3)
#' s <- rbinom(n, 1, 0.5)
#' a <- rbinom(n, 1, 0.5)
#' y <- X[, 1] + (0.5 + 0.3 * X[, 2]) * a + rnorm(n)
#'
#' fit <- FusionForest(
#'   y = y,
#'   X_train_control = X, X_train_treat = X,
#'   treatment_indicator_train = a, source_indicator_train = s,
#'   X_test_control = X, X_test_treat = X,
#'   treatment_indicator_test = a, source_indicator_test = s,
#'   N_post = 50, N_burn = 25,
#'   store_posterior_sample = TRUE, verbose = FALSE
#' )
#'
#' # Summarise the treatment effect by a linear model in x1, x2 and x3
#' proj <- fusion_projection(fit, basis = ~ x1 + x2 + x3,
#'                           X_eval = as.data.frame(X))
#' colMeans(proj)
#'
#' @importFrom stats model.matrix rgamma lm.wfit
#' @export
fusion_projection <- function(
  fit,
  basis,
  X_eval,
  target   = c("cate", "af"),
  af_scale = c("multiplicative", "log"),
  which    = c("test", "train"),
  weights  = c("uniform", "bayesian_bootstrap"),
  center   = TRUE,
  scale    = FALSE,
  seed     = NULL
) {

  target   <- match.arg(target)
  af_scale <- match.arg(af_scale)
  which    <- match.arg(which)
  weights  <- match.arg(weights)

  if (!inherits(basis, "formula") || length(basis) != 2L)
    stop("'basis' must be a one-sided formula, e.g. ~ x1 + x2.")
  if (!is.data.frame(X_eval))
    stop("'X_eval' must be a data.frame.")

  meta <- fit$meta
  if (is.null(meta))
    stop("fit$meta is missing.  Refit with the current package version.")

  ## -----------------------------------------------------------------
  ## Posterior draws of tau on the user log-time scale
  ## -----------------------------------------------------------------

  comp <- .log_scale_components(fit, meta, which = which)
  tau  <- comp$tau                                # R x n_ev matrix
  R    <- nrow(tau)
  n_ev <- ncol(tau)

  if (nrow(X_eval) != n_ev)
    stop("nrow(X_eval) = ", nrow(X_eval), " does not match the number of ",
         which, " evaluation points in the fit (", n_ev, ").")

  ## -----------------------------------------------------------------
  ## Target functional theta^(r)
  ## -----------------------------------------------------------------

  theta <- if (target == "cate" || af_scale == "log") tau else exp(tau)

  ## -----------------------------------------------------------------
  ## Design matrix Phi (centre / scale non-intercept columns)
  ## -----------------------------------------------------------------

  Phi <- stats::model.matrix(basis, data = X_eval)
  has_intercept <- "(Intercept)" %in% colnames(Phi)
  non_int <- setdiff(colnames(Phi), "(Intercept)")

  center_vec <- NULL
  scale_vec  <- NULL
  if (length(non_int) > 0L && (isTRUE(center) || isTRUE(scale))) {
    sc <- scale(Phi[, non_int, drop = FALSE],
                center = isTRUE(center), scale = isTRUE(scale))
    Phi[, non_int] <- sc
    if (isTRUE(center)) center_vec <- attr(sc, "scaled:center")
    if (isTRUE(scale))  scale_vec  <- attr(sc, "scaled:scale")
  }

  k        <- ncol(Phi)
  col_names <- colnames(Phi)

  ## -----------------------------------------------------------------
  ## Solve
  ## -----------------------------------------------------------------

  if (weights == "uniform") {
    qr_obj <- qr(Phi)
    if (qr_obj$rank < k)
      stop("Design matrix Phi is rank-deficient (rank ", qr_obj$rank,
           " < ", k, ").  Drop collinear basis terms.")
    # qr.coef on the n x R matrix t(theta) returns a k x R matrix in one call.
    gamma <- t(qr.coef(qr_obj, t(theta)))
  } else {
    if (!is.null(seed)) set.seed(seed)
    gamma <- matrix(NA_real_, R, k)
    g_draws <- matrix(rgamma(R * n_ev, shape = 1, rate = 1), R, n_ev)
    w_draws <- g_draws / rowSums(g_draws)
    for (r in seq_len(R)) {
      fit_r <- lm.wfit(Phi, theta[r, ], w = w_draws[r, ])
      gamma[r, ] <- fit_r$coefficients
    }
  }

  colnames(gamma) <- col_names

  attr(gamma, "target")   <- target
  attr(gamma, "af_scale") <- if (target == "af") af_scale else NA_character_
  attr(gamma, "which")    <- which
  attr(gamma, "weights")  <- weights
  attr(gamma, "center")   <- center_vec
  attr(gamma, "scale")    <- scale_vec
  attr(gamma, "formula")  <- basis
  gamma
}
