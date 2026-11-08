# ══════════════════════════════════════════════════════════════════════════════
# smoke_test.R -- run the revision scripts at tiny settings, to catch errors
# before the HPC runs.  The scripts themselves are not edited: each is copied to
# a temporary folder with its settings replaced (M = 2, 50/50 draws, 2 bootstrap
# refits, small calibration samples), and runs there.  The analysis scripts then
# run on that output.  Nothing in results/ is touched.
#
# Run from the repository root:
#   Rscript simulations/revision_v3/smoke_test.R          # fusion scripts
#   Rscript simulations/revision_v3/smoke_test.R --all    # also the competitors
#   Rscript simulations/revision_v3/smoke_test.R --dry-run  # only show the edits
#   Rscript simulations/revision_v3/smoke_test.R --only=exp7  # one experiment
# A number sets the workers per script (default up to 4), e.g. ... smoke_test.R 8
#
# The fit functions catch errors and return NA, so a script can finish with
# every fit failed.  The check therefore also looks at the metrics per method.
# ══════════════════════════════════════════════════════════════════════════════

base_dir <- "simulations/revision_v3"
args     <- commandArgs(trailingOnly = TRUE)
all_runs <- "--all" %in% args
dry_run  <- "--dry-run" %in% args
num_arg  <- suppressWarnings(as.integer(args))
workers  <- if (any(!is.na(num_arg))) num_arg[!is.na(num_arg)][1] else
  min(4L, max(1L, parallel::detectCores() - 1L))

if (!dir.exists(base_dir)) {
  stop("Run this from the repository root.")
}

# ── The scripts, per experiment ───────────────────────────────────────────────
exps <- list(
  list(dir = "exp1_confounding_heterogeneity", sims = "sim_surv_v13",
       analysis = "analyse_surv_v13"),
  list(dir = "exp2_competitor", sims = "sim_surv_v13_bff", analysis = NULL),
  list(dir = "exp3_covariate_dimension",
       sims = paste0("sim_hd_v2", c("a", "b", "c", "d")),
       analysis = "analyse_hd_v2"),
  list(dir = "exp4_error_distribution", sims = "sim_err_v3",
       analysis = "analyse_err_v3"),
  list(dir = "exp5_prior_sensitivity", sims = "sim_prior_v1",
       analysis = "analyse_prior_v1"),
  list(dir = "exp6_assumption_violations", sims = "sim_viol_v1",
       analysis = "analyse_viol_v1"),
  list(dir = "exp7_interval_handling", sims = "sim_ic_v1",
       analysis = "analyse_ic_v1"),
  list(dir = "pilot_ktau", sims = "sim_ktau_v1", analysis = "analyse_ktau_v1")
)
if (all_runs) {
  exps[[2]] <- list(
    dir = "exp2_competitor",
    sims = paste0("sim_surv_v13_",
                  c("bff", "ye", "deepaft", "xgbaft", "bjelm", "bjtree")),
    analysis = "analyse_competitors_v13")
}

# --only=<text> runs only the experiments whose folder name contains <text>.
only <- sub("^--only=", "", grep("^--only=", args, value = TRUE))
if (length(only)) {
  exps <- exps[vapply(exps, function(e) grepl(only[1], e$dir, fixed = TRUE), TRUE)]
}

# ── The small settings ────────────────────────────────────────────────────────
small <- list(
  c("^M +<-.*",      "M <- 2L"),
  c("^N_post +<-.*", "N_post <- 50L"),
  c("^N_burn +<-.*", "N_burn <- 50L"),
  c("^B_boot +<-.*", "B_boot <- 2L"),
  c("^TUNE <- TRUE", "TUNE <- FALSE"),
  c("n_cal = [0-9e]+", "n_cal = 5000")
)
make_small <- function(lines) {
  n_hit <- integer(length(small))
  for (k in seq_along(small)) {
    n_hit[k] <- sum(grepl(small[[k]][1], lines))
    lines    <- sub(small[[k]][1], small[[k]][2], lines)
  }
  if (n_hit[1] != 1L) stop("expected one line setting M, found ", n_hit[1])
  list(lines = lines, n_hit = n_hit)
}

if (dry_run) {
  labs <- c("M", "N_post", "N_burn", "B_boot", "TUNE", "n_cal")
  for (e in exps) for (s in e$sims) {
    path <- file.path(base_dir, e$dir, paste0(s, ".R"))
    out  <- make_small(readLines(path, warn = FALSE))
    invisible(parse(text = out$lines))
    cat(sprintf("%-22s %s\n", s, paste(labs[out$n_hit > 0], collapse = ", ")))
  }
  cat("\nAll scripts parse after the edits.  Nothing was run.\n")
  quit(save = "no")
}

# ── Package check ─────────────────────────────────────────────────────────────
need <- c("FusionForests", "ShrinkageTrees", "doParallel", "foreach", "evd", "MASS")
if (all_runs) need <- c(need, "glmnet", "dnn", "survival", "xgboost", "bujar", "gbm",
                        "emplik")
miss <- need[!vapply(need, requireNamespace, logical(1), quietly = TRUE)]
if (length(miss)) stop("Missing packages: ", paste(miss, collapse = ", "))
ff_version <- packageVersion("FusionForests")
cat("FusionForests", as.character(ff_version), "| ShrinkageTrees",
    as.character(packageVersion("ShrinkageTrees")), "| survival installed:",
    requireNamespace("survival", quietly = TRUE), "\n")
if (ff_version < "1.1.2") stop("FusionForests must be at least 1.1.2.")

# ── Run ───────────────────────────────────────────────────────────────────────
# Next to R's session folder, which is deleted on exit: the logs must stay.
tmp <- file.path(dirname(tempdir()),
                 paste0("smoke_", format(Sys.time(), "%Y%m%d_%H%M%S")))
dir.create(tmp)
rscript <- file.path(R.home("bin"), "Rscript")
thread1 <- c("OMP_NUM_THREADS=1", "OPENBLAS_NUM_THREADS=1", "MKL_NUM_THREADS=1")

# One row of the report for a finished simulation script.
check_output <- function(path) {
  if (!file.exists(path)) return("no output file")
  res    <- readRDS(path)
  metric <- intersect(c("rmse", "cate_all_rmse"), names(res))[1]
  if (is.na(metric)) return(sprintf("ok, %d rows", nrow(res)))
  groups <- if ("method" %in% names(res)) res$method else rep("all", nrow(res))
  na_by  <- tapply(is.na(res[[metric]]), groups, mean)
  if (any(na_by == 1)) {
    return(paste("ALL FITS FAILED for:",
                 paste(names(na_by)[na_by == 1], collapse = ", ")))
  }
  if (any(na_by > 0)) {
    return(sprintf("some fits failed (up to %.0f%%), %d rows", 100 * max(na_by),
                   nrow(res)))
  }
  sprintf("ok, %d rows", nrow(res))
}

report <- NULL
add    <- function(script, status, secs) {
  report <<- rbind(report, data.frame(script = script, status = status,
                                      seconds = round(secs),
                                      stringsAsFactors = FALSE))
}

for (e in exps) {
  res_dir <- file.path(tmp, e$dir, "results")
  dir.create(res_dir, recursive = TRUE)
  for (s in e$sims) {
    cat(sprintf("Running %s ...\n", s))
    script <- file.path(res_dir, paste0(s, ".R"))
    writeLines(make_small(readLines(file.path(base_dir, e$dir, paste0(s, ".R")),
                                    warn = FALSE))$lines, script)
    log  <- file.path(res_dir, paste0(s, ".log"))
    secs <- system.time(
      code <- system2(rscript, c(shQuote(script), workers + 1L), stdout = log,
                      stderr = log, env = c(paste0("TMPDIR=", res_dir), thread1))
    )[["elapsed"]]
    status <- if (code != 0) sprintf("ERROR (exit %d), see the log", code) else
      check_output(file.path(res_dir, paste0(s, "_output.rds")))
    add(s, status, secs)
  }
  if (!is.null(e$analysis)) {
    cat(sprintf("Running %s ...\n", e$analysis))
    lines  <- readLines(file.path(base_dir, e$dir, paste0(e$analysis, ".R")),
                        warn = FALSE)
    script <- file.path(res_dir, paste0(e$analysis, ".R"))
    writeLines(gsub(base_dir, tmp, lines, fixed = TRUE), script)
    log  <- file.path(res_dir, paste0(e$analysis, ".log"))
    secs <- system.time(
      code <- system2(rscript, shQuote(script), stdout = log, stderr = log)
    )[["elapsed"]]
    add(e$analysis, if (code == 0) "ok" else
      sprintf("ERROR (exit %d), see the log", code), secs)
  }
}

# ── Report ────────────────────────────────────────────────────────────────────
cat("\n")
print(report, row.names = FALSE, right = FALSE)
cat("\nLogs and output are in", tmp, "\n")
bad <- grepl("ERROR|ALL FITS FAILED|no output", report$status)
if (any(bad)) {
  cat("\nFAILED:", paste(report$script[bad], collapse = ", "), "\n")
  quit(save = "no", status = 1)
}
cat("\nAll scripts ran.\n")
