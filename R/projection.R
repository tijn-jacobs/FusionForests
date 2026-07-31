#' Posterior linear projection of the CATE or acceleration factor
#'
#' Projects each posterior draw of the heterogeneous treatment effect onto a
#' user-supplied linear basis via (weighted) least squares.  The resulting
#' \eqn{R \times k} collection of coefficient vectors is a sample from the
#' posterior of the projection, i.e.\ the pushforward of \eqn{\tau(\cdot)}
#' (or \eqn{\exp\tau(\cdot)}) under the projection map of Woody, Carvalho
#' and Murray (2020).  Uncertainty is inherited exactly from the original
#' posterior without refitting; see \code{notes/linear_projection.tex}.
#'
#' @details
#' At iteration \eqn{r} the coefficient vector solves
#' \deqn{\gamma^{(r)} = \arg\min_{\gamma}
#'   \sum_i w_i^{(r)} \bigl(\theta^{(r)}(x_i) - \phi(x_i)^{\top}\gamma\bigr)^2,}
#' where \eqn{\theta^{(r)}} is one of \eqn{\tau^{(r)}} (\code{target = "cate"}),
#' \eqn{\exp\tau^{(r)}} (\code{target = "af"}, \code{af_scale = "multiplicative"}),
#' or \eqn{\tau^{(r)}} again (\code{target = "af"}, \code{af_scale = "log"};
#' kept as an alias for clarity).
#'
#' With \code{weights = "uniform"} the QR factorisation of \eqn{\Phi} is
#' computed once and reused.  With \code{weights = "bayesian_bootstrap"}
#' fresh Dirichlet weights \eqn{w^{(r)} \sim \mathrm{Dirichlet}(1,\ldots,1)}
#' are drawn at each iteration (Rubin, 1981), injecting
#' covariate-distribution uncertainty into the projection.
#'
#' Columns of \eqn{\Phi} other than the intercept are centred (and optionally
#' scaled) once, on \code{X_eval}, before the projection.  The centring and
#' scaling vectors are returned as attributes so coefficients can be mapped
#' back to original units.
#'
#' @param fit A fitted \code{FusionForest} object with
#'   \code{store_posterior_sample = TRUE}.
#' @param basis One-sided formula defining the projection basis, e.g.\
#'   \code{~ x1 + x2 + x1:x2}.  Built into a design matrix via
#'   \code{stats::model.matrix}.
#' @param X_eval Data frame of evaluation covariates.  Must have one row
#'   per posterior-sample column of the chosen \code{which} set
#'   (i.e.\ the same rows passed as \code{X_test_*} or \code{X_train_*}
#'   when fitting).  Column names must match the variables referenced in
#'   \code{basis}.
#' @param target Character; \code{"cate"} projects \eqn{\tau(x)} on the
#'   AFT log-time scale, \code{"af"} projects the acceleration factor
#'   \eqn{\exp\tau(x)} or its log (see \code{af_scale}).
#' @param af_scale Only used when \code{target = "af"}.
#'   \code{"multiplicative"} (default) projects \eqn{\exp\tau} directly,
#'   so \eqn{\gamma_k} is the change in the AF per unit of covariate
#'   \eqn{k}.  \code{"log"} projects \eqn{\tau} (equivalent to
#'   \code{target = "cate"}; kept as a labelled alias).
#' @param which Which evaluation rows to use, \code{"test"} (default) or
#'   \code{"train"}.  Determines the implicit target population of the
#'   summary.
#' @param weights \code{"uniform"} (default) or \code{"bayesian_bootstrap"}.
#'   The latter draws fresh Dirichlet weights at each MCMC iteration.
#' @param center,scale Logical; centre and/or scale the non-intercept
#'   columns of \eqn{\Phi} before the projection.  Defaults: \code{TRUE},
#'   \code{FALSE}.
#' @param seed Integer or \code{NULL}.  Sets the RNG seed used for the
#'   Bayesian-bootstrap weights, for reproducibility.
#'
#' @return An \eqn{R \times k} numeric matrix of posterior draws of the
#'   projection coefficients, one row per MCMC iteration, with
#'   \code{colnames} from \code{model.matrix}.  Attributes:
#'   \code{target}, \code{af_scale}, \code{which}, \code{weights},
#'   \code{center} and \code{scale} vectors (\code{NULL} if not applied),
#'   and the original \code{formula}.
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
#' # Continuous fusion fit with stored posterior samples
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
#' # Project the posterior CATE surface onto a linear basis
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
