## Regression cases for FusionForests.
##
## Run by dev/regression/check.R in a separate R process, once for the
## reference version and once for the working tree:
##
##   Rscript cases.R <library> <output.rds> [package]
##
## Every case fixes the seed, calls the package and records everything that a
## user can observe: the returned object, warnings, the error message, the
## printed output, the state of the random number generator afterwards and
## the results of the summary functions.  Two versions agree when all of
## these are identical.

args <- commandArgs(trailingOnly = TRUE)
lib  <- args[1]
out  <- args[2]
pkg  <- if (length(args) >= 3) args[3] else "FusionForests"

ns <- loadNamespace(pkg, lib.loc = lib)
stopifnot(dirname(normalizePath(getNamespaceInfo(ns, "path"))) ==
            normalizePath(lib))
RNGkind("Mersenne-Twister", "Inversion", "Rejection")

FusionForest      <- getExportedValue(pkg, "FusionForest")
fusion_estimand   <- getExportedValue(pkg, "fusion_estimand")
fusion_projection <- getExportedValue(pkg, "fusion_projection")

## ---------------------------------------------------------------------
## Data
## ---------------------------------------------------------------------

## "cont4": four continuous covariates, balanced sources.
## "mixed": binary, tied and constant covariates, 30% RWD, confounded
##          treatment in the RWD, a tau forest on two covariates only and
##          separate test covariates for the d and c forests.
make_data <- function(type) {
  set.seed(if (type == "cont4") 101 else 202)
  if (type == "cont4") {
    n  <- 80; nt <- 25
    X  <- matrix(rnorm(n * 4), n, 4)
    Xt <- matrix(rnorm(nt * 4), nt, 4)
    s  <- rbinom(n, 1, 0.5)
    treat_cols <- 1:4
  } else {
    n  <- 100; nt <- 30
    gen <- function(m) cbind(rnorm(m), rbinom(m, 1, 0.4), rbinom(m, 1, 0.7),
                             sample(0:3, m, replace = TRUE), rnorm(m), 1)
    X  <- gen(n); Xt <- gen(nt)
    s  <- rbinom(n, 1, 0.7)
    treat_cols <- 1:2
  }
  colnames(X) <- colnames(Xt) <- paste0("x", seq_len(ncol(X)))
  ps <- ifelse(s == 1, 0.5, plogis(0.8 * X[, 1]))
  a  <- rbinom(n, 1, ps)
  logT <- 1 + X[, 1] + 0.5 * X[, 2] + (0.5 + 0.3 * X[, 2]) * a +
    (1 - s) * (0.3 * X[, 3] + 0.4 * a) +
    ifelse(s == 1, 0.4, 0.7) * rnorm(n)
  Tt <- exp(logT)
  C  <- rexp(n, 1 / (2 * mean(Tt)))
  tm <- pmin(Tt, C)
  st <- as.integer(Tt <= C)
  ## Interval censoring: half of the events become intervals.
  ic    <- as.numeric(st == 1 & runif(n) < 0.5)
  L     <- ifelse(ic == 1, tm * 0.7, tm)
  R     <- ifelse(ic == 1, tm * 1.4, tm)
  st_ic <- ifelse(ic == 1, 0L, st)
  list(
    n = n, nt = nt, X = X, Xtreat = X[, treat_cols, drop = FALSE],
    s = s, a = a, ps = pmin(pmax(ps, 0.05), 0.95),
    logT = logT, tm = tm, st = st, L = L, R = R, ic = ic, st_ic = st_ic,
    Xt = Xt, Xt_treat = Xt[, treat_cols, drop = FALSE],
    Xt_deconf = Xt[rev(seq_len(nt)), , drop = FALSE],
    s_t = rbinom(nt, 1, 0.5), a_t = rbinom(nt, 1, 0.5),
    ps_t = plogis(0.8 * Xt[, 1]) * 0.9 + 0.05
  )
}
data_sets <- list(cont4 = make_data("cont4"), mixed = make_data("mixed"))

outcome_args <- function(d, outcome) {
  switch(outcome,
    continuous = list(y = d$logT),
    rc_time = list(y = d$tm, status = d$st,
                   outcome_type = "right-censored"),
    rc_log  = list(y = log(d$tm), status = d$st,
                   outcome_type = "right-censored", timescale = "log"),
    ic_time = list(y = d$tm, status = d$st_ic,
                   observed_left_time = d$L, observed_right_time = d$R,
                   interval_censoring_indicator = d$ic,
                   outcome_type = "right-censored"),
    ic_log  = list(y = log(d$tm), status = d$st_ic,
                   observed_left_time = log(d$L),
                   observed_right_time = log(d$R),
                   interval_censoring_indicator = d$ic,
                   outcome_type = "right-censored", timescale = "log"))
}

## Arguments of one fit.  `test`: "full" (test rows with indicators),
## "none" (prediction at the column means) or "deconf" (also X_test_deconf).
base_args <- function(data, outcome, error_dist, coding, test = "full") {
  d <- data_sets[[data]]
  a <- c(outcome_args(d, outcome), list(
    X_train_control = d$X, X_train_treat = d$Xtreat,
    treatment_indicator_train = d$a, source_indicator_train = d$s,
    error_dist = error_dist, treatment_coding = coding,
    error_truncation_K = 10L, N_post = 30, N_burn = 20,
    store_posterior_sample = TRUE, verbose = FALSE))
  if (test != "none") {
    a <- c(a, list(X_test_control = d$Xt, X_test_treat = d$Xt_treat,
                   treatment_indicator_test = d$a_t,
                   source_indicator_test = d$s_t))
    if (test == "deconf") a$X_test_deconf <- d$Xt_deconf
  }
  if (coding == "adaptive") {
    a$propensity_train <- d$ps
    a$propensity_test  <- if (test == "none") 0.5 else d$ps_t
  }
  a
}

## ---------------------------------------------------------------------
## Cases
## ---------------------------------------------------------------------

outcomes <- c("continuous", "rc_time", "rc_log", "ic_time", "ic_log")
errors   <- c("source_hdp_scale", "gaussian", "source_gaussian", "shared_dp",
              "source_dp", "source_dp_scale", "source_hdp")
codings  <- c("centered", "binary", "adaptive")

cases <- list()
add_case <- function(group, data, outcome, error_dist, coding, variant,
                     args) {
  f  <- c(group = group, data = data, outcome = outcome,
          error_dist = error_dist, coding = coding, variant = variant)
  id <- paste(f, collapse = "|")
  stopifnot(is.null(cases[[id]]))
  cases[[id]] <<- list(id = id, factors = f, args = args,
                       seed = 1000L + length(cases))
}

## A. Full factorial on the continuous covariates.
for (o in outcomes) for (e in errors) for (cd in codings)
  for (tst in c("full", "none"))
    add_case("factorial", "cont4", o, e, cd, paste0("test_", tst),
             base_args("cont4", o, e, cd, tst))

## B. Mixed covariates: unsplittable variables, small leaves, p_treat <
##    p_control, separate X_test_deconf.
for (o in outcomes) for (e in errors) for (cd in codings)
  add_case("factorial", "mixed", o, e, cd, "test_deconf",
           base_args("mixed", o, e, cd, "deconf"))

## C. One argument (group) changed at a time.
d1 <- data_sets$cont4
few_rwd <- c(which(d1$s == 1), which(d1$s == 0)[1:6])
variants <- list(
  no_samples   = list(store_posterior_sample = FALSE),
  k_dev_zero   = list(k_deviation = 0),
  k_values     = list(k_control = 0.5, k_treat = 0.25, k_deconf = 2,
                      k_deviation = 0.3),
  sigma_fixed  = list(sigma = 0.8),
  burn_zero    = list(N_burn = 0),
  burn_one     = list(N_burn = 1, N_post = 1),
  trees        = list(number_of_trees_control = 30,
                      number_of_trees_treat = 20,
                      number_of_trees_deconf = 10,
                      number_of_trees_deviation = 15),
  tree_prior   = list(power_control = 1.5, base_control = 0.8,
                      power_treat = 2, base_treat = 0.5,
                      power_deconf = 1, base_deconf = 0.9,
                      power_deviation = 3, base_deviation = 0.3),
  nu_q         = list(nu = 5, q = 0.75),
  grow_prune   = list(p_grow = 0.1, p_prune = 0.7),
  K_two        = list(error_truncation_K = 2L),
  K_default    = list(error_truncation_K = 50L),
  atom_mass    = list(error_atom_scale = 2, error_mass_init = 5),
  verbose      = list(verbose = TRUE),
  one_test_row = list(X_test_control = d1$Xt[1, , drop = FALSE],
                      X_test_treat = d1$Xt_treat[1, , drop = FALSE],
                      treatment_indicator_test = 0L,
                      source_indicator_test = 0L),
  test_default = list(treatment_indicator_test = NULL,
                      source_indicator_test = NULL),
  few_rwd      = "few_rwd",
  all_rwd      = "all_rwd"
)
subset_rows <- function(a, rows) {
  per_row <- c("y", "status", "observed_left_time", "observed_right_time",
               "interval_censoring_indicator", "treatment_indicator_train",
               "source_indicator_train", "propensity_train")
  for (nm in intersect(per_row, names(a))) a[[nm]] <- a[[nm]][rows]
  for (nm in c("X_train_control", "X_train_treat"))
    a[[nm]] <- a[[nm]][rows, , drop = FALSE]
  a
}
for (o in c("continuous", "rc_time", "ic_time"))
  for (e in c("source_hdp_scale", "gaussian", "shared_dp", "source_dp_scale"))
    for (v in names(variants)) {
      a <- base_args("cont4", o, e, "centered")
      if (identical(variants[[v]], "few_rwd")) {
        a <- subset_rows(a, few_rwd)
      } else if (identical(variants[[v]], "all_rwd")) {
        a$source_indicator_train <- rep(0L, d1$n)
      } else {
        a[names(variants[[v]])] <- variants[[v]]
      }
      add_case("variant", "cont4", o, e, "centered", v, a)
    }

## D. Longer chains with the default truncation level.
for (o in c("continuous", "rc_time", "ic_time")) for (e in errors) {
  a <- base_args("cont4", o, e, "centered")
  a[c("N_post", "N_burn", "error_truncation_K", "store_posterior_sample")] <-
    list(250, 250, 50L, FALSE)
  add_case("long", "cont4", o, e, "centered", "250+250", a)
}

## E. Invalid input: the error or warning message must stay the same.
bad <- function(name, outcome, change, coding = "centered") {
  a <- base_args("cont4", outcome, "gaussian", coding)
  a <- change(a)
  add_case("invalid", "cont4", outcome, "gaussian", coding, name, a)
}
set_arg <- function(...) { v <- list(...); function(a) { a[names(v)] <- v; a } }
drop_arg <- function(nm) function(a) { a[nm] <- NULL; a }
bad("outcome_type",   "continuous", set_arg(outcome_type = "binary"))
bad("no_status",      "rc_time",    drop_arg("status"))
bad("status_ignored", "continuous", set_arg(status = d1$st))
bad("negative_time",  "rc_time",    set_arg(y = -d1$tm))
bad("na_covariate",   "continuous",
    function(a) { a$X_train_control[3, 2] <- NA; a })
bad("rows_treat",     "continuous",
    set_arg(X_train_treat = d1$Xtreat[-1, , drop = FALSE]))
bad("len_treatment",  "continuous",
    set_arg(treatment_indicator_train = d1$a[-1]))
bad("len_source",     "continuous",
    set_arg(source_indicator_train = d1$s[-1]))
bad("source_values",  "continuous",
    set_arg(source_indicator_train = d1$s + 1L))
bad("no_rwd",         "continuous",
    set_arg(source_indicator_train = rep(1L, d1$n)))
bad("no_propensity",  "continuous", drop_arg("propensity_train"), "adaptive")
bad("propensity_01",  "continuous",
    set_arg(propensity_train = replace(d1$ps, 1, 1)), "adaptive")
bad("no_prop_test",   "continuous", drop_arg("propensity_test"), "adaptive")
bad("len_prop_test",  "continuous",
    set_arg(propensity_test = d1$ps_t[-1]), "adaptive")
bad("K_one",          "continuous", set_arg(error_truncation_K = 1L))
bad("atom_scale",     "continuous", set_arg(error_atom_scale = -1))
bad("mass_init",      "continuous", set_arg(error_mass_init = 0))
bad("k_negative",     "continuous", set_arg(k_treat = -1))
bad("k_vector",       "continuous", set_arg(k_control = c(1, 2)))
bad("coding",         "continuous", set_arg(treatment_coding = "foo"))
bad("error_dist",     "continuous", set_arg(error_dist = "foo"))
bad("ic_partial",     "ic_time",    drop_arg("observed_left_time"))
bad("ic_order",       "ic_time", function(a) {
  i <- which(a$interval_censoring_indicator == 1)[1]
  a$observed_left_time[i] <- a$observed_right_time[i]; a })
bad("ic_length",      "ic_time",
    set_arg(observed_right_time = d1$R[-1]))
bad("test_ncol",      "continuous",
    set_arg(X_test_treat = d1$Xt_treat[, -1, drop = FALSE]))
bad("test_nrow",      "continuous",
    set_arg(X_test_treat = d1$Xt_treat[-1, , drop = FALSE]))
bad("test_not_matrix", "continuous",
    set_arg(X_test_control = as.data.frame(d1$Xt)))
bad("len_source_test", "continuous",
    set_arg(source_indicator_test = d1$s_t[-1]))
bad("source_test_values", "continuous",
    set_arg(source_indicator_test = d1$s_t + 1L))
bad("len_treat_test", "continuous",
    set_arg(treatment_indicator_test = d1$a_t[-1]))
bad("deconf_ncol",    "continuous",
    set_arg(X_test_deconf = d1$Xt[, -1, drop = FALSE]))
bad("na_test",        "continuous",
    function(a) { a$X_test_control[1, 1] <- NaN; a })

## ---------------------------------------------------------------------
## Recording
## ---------------------------------------------------------------------

## Evaluate expr.  Returns its value, or its error message, and the
## warnings it raised.
evaluate <- function(expr) {
  warn  <- character()
  value <- withCallingHandlers(
    tryCatch(expr, error = function(e)
      structure(list(message = conditionMessage(e)),
                class = "recorded_error")),
    warning = function(w) {
      warn <<- c(warn, conditionMessage(w))
      invokeRestart("muffleWarning")
    })
  list(value = value, warnings = unique(warn))
}

## The same for the summary functions: the warnings travel with the value.
safe <- function(expr) {
  res <- evaluate(expr)
  if (length(res$warnings))
    attr(res$value, "recorded_warnings") <- res$warnings
  res$value
}

## A formula carries its environment, which never compares equal between
## two R processes.  Keep its text.
plain_formula <- function(x) {
  if (!is.null(attr(x, "formula")))
    attr(x, "formula") <- deparse(attr(x, "formula"))
  x
}

## Everything the summary functions return for one fit.
derived <- function(fit, case) {
  d  <- data_sets[[case$factors[["data"]]]]
  t0 <- if (case$factors[["outcome"]] == "continuous") exp(median(d$logT))
        else median(d$tm)
  X_eval <- as.data.frame(case$args$X_train_treat)
  ## Test rows of this fit (one row when it predicts at the column means).
  n_ev   <- NCOL(fit$test_predictions_sample_treat)
  X_test <- as.data.frame(d$Xt)[seq_len(n_ev), , drop = FALSE]
  est <- function(...) safe(fusion_estimand(fit, ...))
  prj <- function(..., X = X_eval, which = "train")
    safe(plain_formula(fusion_projection(fit, basis = ~ x1 + x2, X_eval = X,
                                         which = which, ...)))
  list(
    print         = safe(capture.output(print(fit))),
    summary       = safe(summary(fit)),
    print_summary = safe(capture.output(print(summary(fit)))),
    af            = est(estimand = "AF"),
    af_avg_bb     = est(estimand = "AF", population_average = TRUE,
                        seed = 11),
    sd_rwd        = est(estimand = "SD", time = t0, target_source = "rwd"),
    sd_rct        = est(estimand = "SD", time = t0, target_source = "rct"),
    sd_avg_bb     = est(estimand = "SD", time = t0,
                        population_average = TRUE, seed = 12),
    sd_avg_equal  = est(estimand = "SD", time = t0, target_source = "rct",
                        population_average = TRUE,
                        bayesian_bootstrap = FALSE),
    sd_no_time    = est(estimand = "SD"),
    proj_cate     = prj(),
    proj_af       = prj(target = "af"),
    proj_af_log   = prj(target = "af", af_scale = "log", scale = TRUE),
    proj_bb       = prj(weights = "bayesian_bootstrap", seed = 13),
    proj_test     = prj(X = X_test, which = "test")
  )
}

record <- function(case) {
  set.seed(case$seed)
  printed <- capture.output(
    fitted <- evaluate(do.call(FusionForest, case$args)))
  value <- fitted$value
  res <- list(
    factors  = case$factors,
    value    = value,
    warnings = fitted$warnings,
    ## The run time is the only part of the output that may differ.
    printed  = sub("Done in [0-9]+ seconds", "Done in <t> seconds",
                   gsub("\r", "\n", printed)),
    rng      = .Random.seed
  )
  if (!inherits(value, "recorded_error")) res$derived <- derived(value, case)
  res
}

## The preliminary censored-normal fit, on its own.
helper_checks <- function() {
  ci <- get0("censored_info", envir = ns)
  d  <- data_sets$mixed
  lapply(list(
    right    = quote(ci(log(d$tm), d$st)),
    interval = quote(ci(log(d$tm), d$st_ic, log(d$L), log(d$R), d$ic)),
    no_event = quote(ci(log(d$tm), rep(0, d$n))),
    all_event = quote(ci(log(d$tm), rep(1, d$n)))
  ), function(call) safe(eval(call)))
}

t_start <- proc.time()[["elapsed"]]
results <- vector("list", length(cases))
names(results) <- names(cases)
for (i in seq_along(cases)) {
  results[[i]] <- record(cases[[i]])
  if (i %% 50 == 0 || i == length(cases))
    message(sprintf("  %d / %d cases", i, length(cases)))
}

saveRDS(list(
  info = list(package = pkg,
              version = as.character(utils::packageVersion(pkg, lib)),
              r_version = R.version.string,
              survival = requireNamespace("survival", quietly = TRUE),
              seconds = proc.time()[["elapsed"]] - t_start),
  cases   = results,
  helpers = helper_checks()
), out)
