## Regression check for FusionForests.
##
## From the root of the repository:
##
##   source("dev/regression/check.R")
##   res <- regression_check()
##
## The check installs two versions of the package into private libraries,
## runs dev/regression/cases.R on each in a separate R process and compares
## the results bit for bit:
##
##   reference : the commit `reference` (default: the last commit before the
##               simplification work).  Read with `git archive`, which does
##               not touch the working tree or the history.
##   candidate : the working tree as it is now.
##
## The results of the reference are cached outside the repository (see
## regression_cache_dir()), so it is installed and run once per commit, per
## version of cases.R and per R version.  Nothing is installed into your own
## library and the package loaded in this session is not touched.
##
## A change that is meant to preserve behaviour passes when all cases are
## identical.  For a change that is meant to alter results (a bug fix), the
## report shows which cases and which outputs changed, so that you can
## confirm that nothing else did.  Afterwards, pass the new commit as
## `reference`.

regression_reference <- "bf3e2632f6c5ddfc1e8e52110efab76d9fead9be"

regression_check <- function(reference = regression_reference,
                             repo = ".",
                             run_tests = TRUE,
                             refresh_reference = FALSE) {

  repo <- normalizePath(repo)
  desc <- file.path(repo, "DESCRIPTION")
  if (!file.exists(desc) ||
      read.dcf(desc, "Package")[1, 1] != "FusionForests")
    stop("Run this from the root of the FusionForests repository.")
  cases <- file.path(repo, "dev", "regression", "cases.R")

  work <- tempfile("ff_regression_")
  dir.create(work)
  on.exit(unlink(work, recursive = TRUE), add = TRUE)

  ## Child processes must find the same packages as this session.
  old_libs <- Sys.getenv("R_LIBS", unset = NA)
  Sys.setenv(R_LIBS = paste(.libPaths(), collapse = .Platform$path.sep))
  on.exit(if (is.na(old_libs)) Sys.unsetenv("R_LIBS")
          else Sys.setenv(R_LIBS = old_libs), add = TRUE)

  ## ---- Reference ----------------------------------------------------
  ref_id   <- .reference_id(reference, repo)
  ref_file <- file.path(
    regression_cache_dir(),
    sprintf("reference_%s_%s_R%s.rds", ref_id,
            substr(unname(tools::md5sum(cases)), 1, 8), getRversion()))
  if (refresh_reference || !file.exists(ref_file)) {
    message("Reference ", ref_id, ": installing and running the cases ...")
    src <- file.path(work, "reference_src")
    .export_reference(reference, repo, src)
    lib <- file.path(work, "reference_lib")
    .install_package(src, lib)
    dir.create(dirname(ref_file), recursive = TRUE, showWarnings = FALSE)
    .run_cases(cases, lib, ref_file)
  } else {
    message("Reference ", ref_id, ": using cached results.")
  }

  ## ---- Candidate ----------------------------------------------------
  message("Candidate (working tree): installing and running the cases ...")
  src <- file.path(work, "candidate_src")
  .export_working_tree(repo, src)
  .check_rcpp_exports(src)
  lib <- file.path(work, "candidate_lib")
  .install_package(src, lib)
  cand_file <- file.path(work, "candidate.rds")
  .run_cases(cases, lib, cand_file)

  ## ---- Compare ------------------------------------------------------
  res <- .compare_results(readRDS(ref_file), readRDS(cand_file))
  res$reference <- ref_id
  class(res) <- "regression_check"
  print(res)

  ## ---- Unit tests on the candidate ----------------------------------
  if (run_tests) res$tests_passed <- .run_unit_tests(repo, lib)

  invisible(res)
}

## Where the reference results are kept.  Delete the folder to start afresh.
regression_cache_dir <- function() {
  file.path(tools::R_user_dir("FusionForests", "cache"), "regression")
}

## ---------------------------------------------------------------------
## Sources of the two versions
## ---------------------------------------------------------------------

## What R CMD INSTALL needs.  inst/, man/, vignettes/ and tests/ are left
## out: they do not affect the results.
.package_paths <- c("DESCRIPTION", "NAMESPACE", "LICENSE", "R", "src",
                    "data")

.git <- function(repo, args, ...) {
  system2("git", c("-C", shQuote(repo), args), ...)
}

## Commit hash of `reference`, or a label when it is a directory.
.reference_id <- function(reference, repo) {
  if (dir.exists(reference))
    return(paste0("dir_", substr(unname(tools::md5sum(
      file.path(reference, "DESCRIPTION"))), 1, 10)))
  sha <- suppressWarnings(
    .git(repo, c("rev-parse", "--verify", "--quiet",
                 shQuote(paste0(reference, "^{commit}"))),
         stdout = TRUE, stderr = TRUE))
  if (!is.null(attr(sha, "status")) || length(sha) != 1L)
    stop("'", reference, "' is neither a commit of this repository nor a ",
         "directory.", if (length(sha)) "\ngit says: ",
         paste(sha, collapse = "\n"), call. = FALSE)
  substr(sha, 1, 10)
}

## `reference` is a commit (read with git archive) or a directory that holds
## a copy of the package.
.export_reference <- function(reference, repo, dest) {
  dir.create(dest)
  if (dir.exists(reference)) {
    .copy_package(reference, dest)
    return(invisible(dest))
  }
  tar_file <- tempfile(fileext = ".tar")
  on.exit(unlink(tar_file))
  status <- .git(repo, c("archive", "--format=tar",
                         paste0("--output=", shQuote(tar_file)),
                         shQuote(reference), .package_paths))
  if (status != 0L) stop("git archive failed for '", reference, "'.")
  utils::untar(tar_file, exdir = dest)
  invisible(dest)
}

.export_working_tree <- function(repo, dest) {
  dir.create(dest)
  .copy_package(repo, dest)
}

## Copy the package files, without compiled objects.
.copy_package <- function(from, to) {
  for (p in .package_paths) {
    src <- file.path(from, p)
    if (!file.exists(src)) next
    if (!dir.exists(src)) {
      file.copy(src, file.path(to, p))
      next
    }
    files <- list.files(src, recursive = TRUE, all.files = TRUE,
                        no.. = TRUE)
    files <- files[!grepl("\\.(o|so|dll|dylib|tmp)$", files)]
    for (f in files) {
      target <- file.path(to, p, f)
      dir.create(dirname(target), recursive = TRUE, showWarnings = FALSE)
      file.copy(file.path(src, f), target)
    }
  }
  invisible(to)
}

## The Rcpp glue must match the C++ sources, or the installed candidate is
## not the code in the working tree.
.check_rcpp_exports <- function(src) {
  glue   <- file.path(src, c("R/RcppExports.R", "src/RcppExports.cpp"))
  before <- lapply(glue, readLines)
  Rcpp::compileAttributes(src)
  after  <- lapply(glue, readLines)
  if (!identical(before, after))
    warning("R/RcppExports.R or src/RcppExports.cpp is out of date.  The ",
            "check uses regenerated glue; run Rcpp::compileAttributes() ",
            "in the repository.", call. = FALSE, immediate. = TRUE)
  invisible(NULL)
}

## ---------------------------------------------------------------------
## Install and run
## ---------------------------------------------------------------------

.r_bin <- function(name) file.path(R.home("bin"), name)

.install_package <- function(src, lib) {
  dir.create(lib)
  log <- tempfile(fileext = ".log")
  status <- system2(
    .r_bin("R"),
    c("CMD", "INSTALL", "--no-docs", "--no-multiarch",
      paste0("--library=", shQuote(lib)), shQuote(src)),
    stdout = log, stderr = log)
  if (status != 0L) {
    writeLines(utils::tail(readLines(log), 40))
    stop("R CMD INSTALL failed (output above).")
  }
  invisible(lib)
}

.run_cases <- function(cases, lib, out) {
  status <- system2(.r_bin("Rscript"),
                    c(shQuote(cases), shQuote(lib), shQuote(out)))
  if (status != 0L || !file.exists(out))
    stop("The cases did not run to completion (output above).")
  invisible(out)
}

## The testthat suite, against the installed candidate, in its own process.
.run_unit_tests <- function(repo, lib) {
  if (!requireNamespace("testthat", quietly = TRUE)) {
    message("testthat is not installed: unit tests skipped.")
    return(NA)
  }
  message("\nUnit tests (tests/testthat) on the candidate ...")
  code <- sprintf(paste0(
    ".libPaths(c(%s, .libPaths())); ",
    "testthat::test_dir(%s, package = 'FusionForests', ",
    "load_package = 'installed', reporter = 'summary', ",
    "stop_on_failure = TRUE)"),
    deparse(lib), deparse(file.path(repo, "tests", "testthat")))
  status <- system2(.r_bin("Rscript"), c("-e", shQuote(code)))
  passed <- status == 0L
  message(if (passed) "Unit tests: all passed."
          else "Unit tests: FAILURES (output above).")
  passed
}

## ---------------------------------------------------------------------
## Comparison
## ---------------------------------------------------------------------

## Paths at which two objects differ, with a description of the difference.
.differences <- function(a, b, path = "") {
  if (identical(a, b)) return(NULL)
  row <- function(what) data.frame(path = path, difference = what)
  child <- function(key) {
    label <- if (is.character(key)) key else paste0("[[", key, "]]")
    .differences(a[[key]], b[[key]],
                 if (nzchar(path)) paste0(path, "$", label) else label)
  }
  both_lists <- is.list(a) && is.list(b) &&
    identical(class(a), class(b))
  if (both_lists && !is.null(names(a)) && !is.null(names(b)) &&
      !anyDuplicated(names(a)) && !anyDuplicated(names(b))) {
    keys <- union(names(a), names(b))
    out  <- lapply(keys, function(k) {
      if (!k %in% names(b)) return(.sub(row("only in reference"), path, k))
      if (!k %in% names(a)) return(.sub(row("only in candidate"), path, k))
      child(k)
    })
    out <- do.call(rbind, out)
    if (is.null(out) && !identical(names(a), names(b)))
      out <- row("same elements in a different order")
    if (is.null(out)) out <- row("attributes differ")
    return(out)
  }
  if (both_lists && is.null(names(a)) && is.null(names(b)) &&
      length(a) == length(b)) {
    out <- do.call(rbind, lapply(seq_along(a), child))
    if (is.null(out)) out <- row("attributes differ")
    return(out)
  }
  row(.describe_difference(a, b))
}

.sub <- function(row, path, key) {
  row$path <- if (nzchar(path)) paste0(path, "$", key) else key
  row
}

.shape <- function(x) {
  d <- if (is.null(dim(x))) length(x) else paste(dim(x), collapse = "x")
  sprintf("%s[%s]", paste(class(x), collapse = "/"), d)
}

.describe_difference <- function(a, b) {
  if (inherits(a, "recorded_error") || inherits(b, "recorded_error")) {
    outcome <- function(x) if (inherits(x, "recorded_error"))
      sprintf("error \"%s\"", x$message) else "no error"
    return(sprintf("%s in reference, %s in candidate", outcome(a),
                   outcome(b)))
  }
  if (!identical(.shape(a), .shape(b)))
    return(sprintf("%s in reference, %s in candidate", .shape(a), .shape(b)))
  if (is.numeric(a) && is.numeric(b)) {
    if (!identical(is.na(a), is.na(b))) return("NA pattern differs")
    d <- abs(as.numeric(a) - as.numeric(b))
    d <- d[is.finite(d)]
    if (length(d) && max(d) > 0)
      return(sprintf("max abs difference %.3g (%d of %d values differ)",
                     max(d), sum(d > 0), length(a)))
    return("attributes or non-finite values differ")
  }
  if (is.character(a) && is.character(b)) {
    i <- which(a != b)[1]
    if (!is.na(i))
      return(sprintf("text differs: \"%s\" in reference, \"%s\" in candidate",
                     a[i], b[i]))
  }
  "values differ"
}

.compare_results <- function(ref, cand) {
  ids      <- union(names(ref$cases), names(cand$cases))
  factors  <- do.call(rbind, lapply(ids, function(id) {
    x <- if (!is.null(ref$cases[[id]])) ref$cases[[id]] else cand$cases[[id]]
    as.data.frame(as.list(x$factors))
  }))
  details  <- lapply(ids, function(id)
    .differences(ref$cases[[id]], cand$cases[[id]]))
  names(details) <- ids
  factors$identical <- vapply(details, is.null, logical(1))
  rownames(factors) <- NULL

  list(
    info      = list(reference = ref$info, candidate = cand$info),
    cases     = cbind(id = ids, factors),
    details   = details[!factors$identical],
    helpers   = .differences(ref$helpers, cand$helpers)
  )
}

print.regression_check <- function(x, max_rows = 15, ...) {
  cs    <- x$cases
  n_bad <- sum(!cs$identical)
  rule  <- strrep("-", 70)
  cat("\n", rule, "\nFusionForests regression check\n", rule, "\n", sep = "")
  cat(sprintf("Reference : %s (version %s)\n", x$reference,
              x$info$reference$version))
  cat(sprintf("Candidate : working tree (version %s)\n",
              x$info$candidate$version))
  cat(sprintf("Cases     : %d, of which %d fits and %d invalid calls\n",
              nrow(cs), sum(cs$group != "invalid"),
              sum(cs$group == "invalid")))
  cat(sprintf("Run time  : reference %.0f s, candidate %.0f s\n",
              x$info$reference$seconds, x$info$candidate$seconds))
  if (!identical(x$info$reference$survival, x$info$candidate$survival))
    cat("NOTE: the 'survival' package was available in only one of the ",
        "two runs;\n      censored fits may differ for that reason.  ",
        "Use refresh_reference = TRUE.\n", sep = "")

  if (n_bad == 0 && is.null(x$helpers)) {
    cat("\nRESULT: IDENTICAL.  Every returned object, message, printed ",
        "line,\nsummary and random number state matches the reference.\n",
        sep = "")
    return(invisible(x))
  }

  cat(sprintf("\nRESULT: %d of %d cases DIFFER.\n", n_bad, nrow(cs)))
  show <- function(d) {
    d <- utils::head(d, max_rows)
    cat(sprintf("  %s\n      %s\n", d$path, d$difference), sep = "")
  }
  if (!is.null(x$helpers)) {
    cat("\nInternal helpers that differ:\n")
    show(x$helpers)
  }
  if (n_bad > 0) {
    cat("\nDiffering cases by factor (differing / total):\n")
    for (f in c("group", "data", "outcome", "error_dist", "coding",
                "variant")) {
      tab <- table(cs[[f]], factor(cs$identical, c(FALSE, TRUE)))
      hit <- tab[, 1] > 0
      cat(sprintf("  %-10s %s\n", f, paste(sprintf(
        "%s %d/%d", rownames(tab)[hit], tab[hit, 1], rowSums(tab)[hit]),
        collapse = ", ")))
    }
    all_diff <- do.call(rbind, x$details)
    counts   <- sort(table(all_diff$path), decreasing = TRUE)
    cat("\nOutputs that differ (number of cases):\n")
    shown <- utils::head(counts, max_rows)
    for (i in seq_along(shown))
      cat(sprintf("  %4d  %s\n", shown[[i]], names(shown)[i]))
    if (length(counts) > max_rows)
      cat(sprintf("  ... and %d more outputs\n", length(counts) - max_rows))
    first <- names(x$details)[1]
    cat("\nFirst differing case: ", first, "\n", sep = "")
    show(x$details[[1]])
    cat("\nAll differing cases: res$cases[!res$cases$identical, ]\n",
        "Detail of one case : res$details[[\"<id>\"]]\n", sep = "")
  }
  invisible(x)
}
