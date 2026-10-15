# Vignette workflow

The vignette is precomputed. Its model fits take too long for R CMD check,
so we knit it locally and ship the static result.

## Files

| File | Role | Shipped |
|:---|:---|:---|
| `FusionForests.Rmd.orig` | Source. Prose and R code that runs. Edit this file. | No |
| `precompile.R` | Knits the `.orig` file into `FusionForests.Rmd`. | No |
| `FusionForests.Rmd` | Generated. Static code, output and figure links. Do not edit. | Yes |
| `figures/` | Generated figures. | Yes |
| `cache/` | knitr cache of the model fits (large). | No |
| `FEATURE_NOTES.md` | Temporary notes on missing features. | No |

## Update the vignette

1. Edit `FusionForests.Rmd.orig`.
2. Install the current package version: `devtools::install()`.
3. From the package root, run `source("vignettes/precompile.R")`.
4. Preview the result:

   ```r
   html <- rmarkdown::render("vignettes/FusionForests.Rmd",
                             output_dir = tempdir(), quiet = TRUE)
   browseURL(html)
   ```

5. Run `devtools::check()`.

## Notes

- Cached chunks rerun only when their code or options change.
  To force a full rerun, delete `cache/`.
- A global chunk option change in the setup chunk can invalidate the cache.
- Commit `FusionForests.Rmd` and `figures/` together with the `.orig` file.
