
setwd("/Users/tijnjacobs/Library/CloudStorage/OneDrive-VrijeUniversiteitAmsterdam/Documents/GitHub/FusionForests")
devtools::load_all()
devtools::test()
devtools::document()
devtools::check()


source("examples/cate_rmse_comparison.R")


devtools::document()                                      
devtools::install()