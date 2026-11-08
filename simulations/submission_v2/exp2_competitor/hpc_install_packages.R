# ──────────────────────────────────────────────────────────────────────────────
# Install every R package the competitor simulations need, into a user library.
#
#   Rscript simulations/exp2_competitor/hpc_install_packages.R
#
# Installs into the first entry of .libPaths() that is writable, or creates
# ~/R/library and uses that. Packages already present are skipped, so the script
# is safe to re-run after a partial failure.
#
# It ends with a load test: every package is attached, so a green run means the
# simulations will actually start. That check is the point -- installation
# succeeding and loading succeeding are different things on a cluster, where a
# package can build against the wrong system library and only fail at
# dyn.load() time.
# ──────────────────────────────────────────────────────────────────────────────

CRAN <- "https://cloud.r-project.org"

# ── Package groups, by which simulation needs them ────────────────────────────
pkgs <- list(
  common  = c("doParallel", "foreach", "MASS", "evd"),
  xgbaft  = c("xgboost"),
  # gbm comes in as a bujar dependency; it is listed so the load test covers it,
  # since the bjtree script attaches it directly to silence its startup banner.
  bjtree  = c("bujar", "gbm"),
  # Only the BJ-ELM fitting path is inlined in the simulation script, so the
  # dependencies are just ginv, map_dfc and WKM. emplik must be attached rather
  # than merely installed: cumsumsurv reaches its C routine through .C() with no
  # PACKAGE argument. The authors' CV_BJ_ELM() would also need rms, caret,
  # rootSolve, scorecard and dplyr; it is not used.
  bjelm   = c("purrr", "emplik"),
  deepaft = c("dnn", "survival")
)
all_pkgs <- unique(unlist(pkgs, use.names = FALSE))

# ── Library path ──────────────────────────────────────────────────────────────
# Resolved in this order, so the target is explicit rather than guessed:
#   1. the first command-line argument
#   2. $R_LIBS, the variable the Slurm job sets
#   3. $HOME/rpackages, if it exists -- the convention used for the tarball
#      installs (R CMD INSTALL --library=$HOME/rpackages ...)
#   4. $R_LIBS_USER, or the first writable entry of .libPaths()
#
# Getting this wrong is silent: the packages install somewhere, and the job then
# fails to find them. Check the "Installing into:" line below against the
# R_LIBS in your Slurm script.
args <- commandArgs(trailingOnly = TRUE)
lib <- ""
if (length(args) > 0 && nzchar(args[1])) {
  lib <- args[1]
} else if (nzchar(Sys.getenv("R_LIBS"))) {
  lib <- strsplit(Sys.getenv("R_LIBS"), .Platform$path.sep)[[1]][1]
} else if (dir.exists(path.expand("~/rpackages"))) {
  lib <- path.expand("~/rpackages")
} else {
  lib <- Sys.getenv("R_LIBS_USER")
  if (!nzchar(lib) || !dir.exists(lib)) {
    writable <- .libPaths()[file.access(.libPaths(), 2) == 0]
    lib <- if (length(writable)) writable[1] else path.expand("~/R/library")
  }
}
dir.create(lib, recursive = TRUE, showWarnings = FALSE)
.libPaths(c(lib, .libPaths()))
cat("Installing into:", lib, "\n")
cat("R version      :", R.version.string, "\n")
cat("R_LIBS         :", if (nzchar(Sys.getenv("R_LIBS"))) Sys.getenv("R_LIBS")
                        else "(not set)", "\n\n")

# ── Install what is missing ───────────────────────────────────────────────────
missing <- all_pkgs[!vapply(all_pkgs, requireNamespace, logical(1),
                            quietly = TRUE)]
if (length(missing)) {
  cat("Missing:", paste(missing, collapse = ", "), "\n\n")
  # Ncpus speeds up compilation of the heavier dependencies (xgboost and bujar
  # both pull in trees of compiled packages).
  install.packages(missing, lib = lib, repos = CRAN,
                   Ncpus = max(1L, parallel::detectCores() - 1L))
} else {
  cat("All packages already present.\n")
}

# ── Load test ─────────────────────────────────────────────────────────────────
cat("\n── Load test ──────────────────────────────────────────────────\n")
status <- vapply(all_pkgs, function(p) {
  ok <- tryCatch({ suppressPackageStartupMessages(
    library(p, character.only = TRUE, lib.loc = .libPaths())); TRUE },
    error = function(e) { message("  ", p, ": ", conditionMessage(e)); FALSE })
  ver <- if (ok) as.character(utils::packageVersion(p)) else "--"
  cat(sprintf("  %-12s %-10s %s\n", p, ver, if (ok) "ok" else "FAILED"))
  ok
}, logical(1))

# ── Verdict ───────────────────────────────────────────────────────────────────
cat("\n───────────────────────────────────────────────────────────────\n")
if (all(status)) {
  cat("All packages load; the simulations are ready to run.\n")
} else {
  failed <- all_pkgs[!status]
  if (length(failed)) cat("Packages that failed to load:",
                          paste(failed, collapse = ", "), "\n")
  cat("\nCommon causes on a cluster:\n")
  cat("  - a package built against a module stack that has since changed.\n")
  cat("    The giveaway is a missing versioned .so, e.g.\n")
  cat("      libicui18n.so.73: cannot open shared object file\n")
  cat("    from stringi. That one is worth fixing permanently by making\n")
  cat("    stringi compile its own bundled ICU instead of linking the\n")
  cat("    system one:\n")
  cat("      R CMD REMOVE --library=", lib, " stringi\n", sep = "")
  cat("      Rscript -e 'install.packages(\"stringi\", lib=\"", lib, "\",\n",
      sep = "")
  cat("          repos=\"", CRAN, "\",\n", sep = "")
  cat("          configure.args=\"--disable-pkg-config\")'\n")
  cat("    then re-run this script.\n")
  cat("  - a system library is missing; load the relevant module first\n")
  cat("      xgboost      needs a C++14 compiler and OpenMP\n")
  cat("      bujar        pulls in mboost and gbm\n")
  cat("  - the library path is read-only, or differs from the R_LIBS your\n")
  cat("    Slurm job sets; compare with the 'Installing into' line above\n")
  quit(save = "no", status = 1L)
}
