# ──────────────────────────────────────────────────────────────────────────────
# Prototype: Buckley-James boosting with extreme learning machines (BJ-ELM).
#
# Implementation from the supplementary material of
#   Kong & Zhang, "Buckley-James Boosting Model based on Extreme Learning
#   Machine and Random Survival Forests", Biometrical Journal.
# It is a standalone implementation, NOT part of the bujar package -- bujar
# 0.2-11 has no "elm" learner.
#
# Semiparametric AFT: iterative Buckley-James imputation of the censored
# responses combined with boosting over ELM base learners.  No error
# distribution is specified, which makes it the frequentist counterpart to the
# fusion forest's nonparametric error model.
#
# Right censoring only, so BOTH sources are right-censored here, as in the
# deepAFT comparison.
#
# Four estimators, matching the deepAFT design:
#   RCT-S, RCT-T, Pooled-S, Pooled-T
# where the pooled fits carry the source indicator S as an extra covariate and
# the CATE is evaluated at S = 1 (the trial estimand).
#
# One replication, point estimates only.  Bootstrap intervals come later.
#
# Run from the repository root:
#   Rscript simulations/exp2_competitor/prototypes/proto_bjelm.R
# ──────────────────────────────────────────────────────────────────────────────

source("simulations/exp2_competitor/prototypes/dgp_prototype.R")

# ── Load the BJ-ELM implementation ────────────────────────────────────────────
# BJ_ELM.R opens with source('BJ_ELM_Code/R/Main_Functions/ELM_Boosting.R'), a
# path that does not exist in this repository.  We load ELM_Boosting.R directly
# and then evaluate BJ_ELM.R with that one line dropped.  ELM_Boosting.R pulls
# in MASS, purrr, emplik, rms, caret, rootSolve, scorecard and dplyr; emplik
# also supplies the WKM() and cumsumsurv C routine that BJ_ELM.R calls.
BJ_DIR <- "simulations/exp2_competitor/bimj2456-sup-0001-suppmat/R/Main_Functions"
source(file.path(BJ_DIR, "ELM_Boosting.R"))
eval(parse(text = grep("^\\s*source\\(", readLines(file.path(BJ_DIR, "BJ_ELM.R")),
                       value = TRUE, invert = TRUE)))

# ── Config ────────────────────────────────────────────────────────────────────
# Kong & Zhang suggest nhid in [5, 40] and nbase in [10, 40]; their defaults are
# 20 and 25.  CV_BJ_ELM() in BJ_ELM.R selects both by 5-fold CV on the C-index
# and needs a data frame with `time` and `status` columns plus the compareC
# package for estC(); wire that in once the point estimates look sane.
NHID  <- 20L                # hidden neurons per ELM base learner
NBASE <- 25L                # number of base learners (boosting Mstop)

N_RCT <- 150L
N_RWD <- 350L
LD    <- 1.0                # between-source heterogeneity
LU    <- 1.0                # unmeasured confounding strength

# No seed: rerun the script a few times to see how much of the spread is the
# estimator and how much is the random hidden layer.

# ── Data ──────────────────────────────────────────────────────────────────────
cal <- calibrate(LD, LU)
d   <- make_data(N_RCT, N_RWD, LD, LU, cal, censoring = "right")
describe(d)

# ── Fit and predict, kept separate ────────────────────────────────────────────
# The separation matters for the S-learner: the hidden-layer weights are random,
# so fitting twice and differencing the two predictions would contaminate the
# CATE with initialisation noise.  Fit once, predict at A = 1 and at A = 0.
#
# d$y is ALREADY the standardised LOG survival time, which is what BJ_ELM wants
# for `y`.  Do not log it again.  `status` is 1 = event, 0 = right-censored,
# matching the package convention (0 = alive, 1 = dead).
bjelm_fit <- function(rows, Z) {
  BJ_ELM(x      = Z[rows, , drop = FALSE],
         y      = d$y[rows],
         status = d$status[rows],
         nhid   = NHID,
         nbase  = NBASE)$model
}

bjelm_pred <- function(model, newZ) {
  as.numeric(pre_BJ_ELM(model = model, newx = as.matrix(newZ)))
}

# ── Meta-learners ─────────────────────────────────────────────────────────────
# S-learner: one model with A as a covariate; CATE = mu(x, A=1) - mu(x, A=0).
cate_S <- function(rows, extra, eval_extra) {
  Z <- cbind(d$X, extra, A = d$A)
  m <- bjelm_fit(rows, Z)
  bjelm_pred(m, cbind(d$X, eval_extra, A = 1)) -
    bjelm_pred(m, cbind(d$X, eval_extra, A = 0))
}

# T-learner: a separate model per arm; CATE = mu_1(x) - mu_0(x).
cate_T <- function(rows, extra, eval_extra) {
  Z  <- cbind(d$X, extra)
  ev <- cbind(d$X, eval_extra)
  m1 <- bjelm_fit(rows[d$A[rows] == 1L], Z)
  m0 <- bjelm_fit(rows[d$A[rows] == 0L], Z)
  bjelm_pred(m1, ev) - bjelm_pred(m0, ev)
}

# Designs: the pooled fits carry the source indicator S and are evaluated at
# S = 1 (the trial estimand), as in the deepAFT comparison.
rows_rct  <- which(d$S == 1L)
rows_pool <- seq_along(d$S)
none      <- d$X[, 0, drop = FALSE]             # zero-column placeholder
Smat      <- cbind(S = d$S)
S1mat     <- cbind(S = rep(1L, nrow(d$X)))

designs <- list(
  list(name = "RCT-S",    rows = rows_rct,  fun = cate_S, extra = none, ev = none),
  list(name = "RCT-T",    rows = rows_rct,  fun = cate_T, extra = none, ev = none),
  list(name = "Pooled-S", rows = rows_pool, fun = cate_S, extra = Smat, ev = S1mat),
  list(name = "Pooled-T", rows = rows_pool, fun = cate_T, extra = Smat, ev = S1mat)
)

# ── Run ───────────────────────────────────────────────────────────────────────
cat(sprintf("\n=== BJ-ELM, nhid = %d, nbase = %d ===\n", NHID, NBASE))
out <- list()
for (des in designs) {
  lab <- paste0("bjelm-", des$name)
  res <- tryCatch({
    cate <- des$fun(des$rows, des$extra, des$ev)
    report(d, cate, lab)
  }, error = function(e) {
    message("  ", lab, " failed: ", conditionMessage(e))
    NULL
  })
  if (!is.null(res)) out[[lab]] <- res
}

if (length(out)) {
  cat("\n=== CATE metrics, one replication ===\n")
  print(do.call(rbind, out), row.names = FALSE, digits = 3)
} else {
  cat("\nNo estimator completed.\n")
}
