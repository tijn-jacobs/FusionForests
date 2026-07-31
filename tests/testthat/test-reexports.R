test_that("ShrinkageTrees models are re-exported", {
  reexported <- c(
    "ShrinkageTrees", "HorseTrees",
    "CausalShrinkageForest", "CausalHorseForest",
    "SurvivalBART", "SurvivalDART",
    "SurvivalBCF", "SurvivalShrinkageBCF"
  )
  exports <- getNamespaceExports("FusionForests")
  for (fn in reexported) {
    expect_true(fn %in% exports, label = paste0(fn, " exported"))
    expect_identical(
      get(fn, envir = asNamespace("FusionForests")),
      get(fn, envir = asNamespace("ShrinkageTrees")),
      label = paste0(fn, " identical to ShrinkageTrees::", fn)
    )
  }
})
