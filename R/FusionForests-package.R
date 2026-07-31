#' FusionForests: Bayesian Tree Ensembles for Data Fusion and Causal
#' Inference
#'
#' Bayesian tree ensemble models for data fusion and causal inference.
#' The flagship model [FusionForest()] combines data from a randomised
#' controlled trial and an observational study using separate tree
#' forests, without assuming the observational data are unconfounded.
#' Posterior summaries of treatment effect estimands are available via
#' [fusion_estimand()] and interpretable linear projections via
#' [fusion_projection()].
#'
#' Single-study causal and survival models are re-exported from the
#' \pkg{ShrinkageTrees} package; see [reexports].
#'
#' @keywords internal
#' @importFrom stats median quantile
#' @importFrom utils head
"_PACKAGE"
