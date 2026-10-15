#' FusionForests: Bayesian Tree Ensembles for Data Fusion and Causal
#' Inference
#'
#' Bayesian tree ensembles for data fusion and causal inference.  The main
#' model, [FusionForest()], combines a randomised controlled trial with an
#' observational study.  The observational data may be confounded.
#' [fusion_estimand()] computes survival differences and acceleration
#' factors.  [fusion_projection()] summarises the treatment effect by a
#' linear model.
#'
#' Single-study models are re-exported from the \pkg{ShrinkageTrees}
#' package; see [reexports].
#'
#' @keywords internal
#' @importFrom stats median quantile
#' @importFrom utils head
"_PACKAGE"
