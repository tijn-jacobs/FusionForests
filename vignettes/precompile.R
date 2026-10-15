# Precompute the vignette.  The model fits take too long for CRAN, so we
# knit FusionForests.Rmd.orig locally.  The result, FusionForests.Rmd,
# holds the code, output and figures as static text; CRAN only renders it.
#
# Steps:
#   1. Install the current package version with devtools::install().
#   2. Run this script from the package root.
#   3. Rerun this script after every edit of FusionForests.Rmd.orig.
#
# The heavy chunks are cached in vignettes/cache/.
# To force a full rerun, delete that folder.

old_wd <- setwd("vignettes")
knitr::knit("FusionForests.Rmd.orig", output = "FusionForests.Rmd")
setwd(old_wd)
