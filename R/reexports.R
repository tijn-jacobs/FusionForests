# Single-study causal and survival models are developed and maintained
# in the ShrinkageTrees package on CRAN.  They are re-exported here so
# that library(FusionForests) provides every model from the accompanying
# paper in one namespace.  roxygen2 generates man/reexports.Rd from the
# @export tags below.

#' @importFrom ShrinkageTrees ShrinkageTrees
#' @export
ShrinkageTrees::ShrinkageTrees

#' @importFrom ShrinkageTrees HorseTrees
#' @export
ShrinkageTrees::HorseTrees

#' @importFrom ShrinkageTrees CausalShrinkageForest
#' @export
ShrinkageTrees::CausalShrinkageForest

#' @importFrom ShrinkageTrees CausalHorseForest
#' @export
ShrinkageTrees::CausalHorseForest

#' @importFrom ShrinkageTrees SurvivalBART
#' @export
ShrinkageTrees::SurvivalBART

#' @importFrom ShrinkageTrees SurvivalDART
#' @export
ShrinkageTrees::SurvivalDART

#' @importFrom ShrinkageTrees SurvivalBCF
#' @export
ShrinkageTrees::SurvivalBCF

#' @importFrom ShrinkageTrees SurvivalShrinkageBCF
#' @export
ShrinkageTrees::SurvivalShrinkageBCF
