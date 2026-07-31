# ──────────────────────────────────────────────────────────────────────────────
# Prototype: Buckley-James boosting (bujar) as a CATE comparator.
#
#   install.packages("bujar"); library(bujar); help("bujar")
#
# Semiparametric AFT: iterative Buckley-James imputation of the censored
# responses combined with L2 boosting.  No error distribution is specified,
# which makes it the frequentist counterpart to the fusion forest's
# nonparametric error model.
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
#   Rscript simulations/exp2_competitor/prototypes/proto_bujar.R
# ──────────────────────────────────────────────────────────────────────────────

setwd("~/Library/CloudStorage/OneDrive-VrijeUniversiteitAmsterdam/Documents/GitHub/FusionForests")
source("simulations/exp2_competitor/prototypes/dgp_prototype.R")
library(bujar)

# ── Config ────────────────────────────────────────────────────────────────────
# LEARNERS: the base learners to try; everything below loops over this vector.
# bujar 0.2-11 implements "linear.regression" (default), "pspline", "tree",
# "enet", "enet2", "mars", and MCP/SCAD; ACOSSO is not supported and there is
# no "elm".  Wang & Wang (2010) benchmark the first three.
#
# "tree" is the default here because it makes the cleanest comparison: boosted
# trees against a Bayesian tree ensemble puts both methods in the same function
# class, so what the experiment isolates is the fusion and the uncertainty
# quantification rather than the base learner.
LEARNERS <- c("tree")

N_RCT  <- 150L
N_RWD  <- 350L
LD     <- 1.0               # between-source heterogeneity
LU     <- 1.0               # unmeasured confounding strength
DEGREE <- 2L                # 1 = additive, 2 = second-order interactions.
                            # The DGP baseline contains x2*x3, so an additive
                            # fit cannot represent it; keep this at 2.
MSTOP  <- 50L               # boosting iterations (bujar default)
NU     <- 0.1               # boosting step length (bujar default)
TUNING <- FALSE             # TRUE + cv = TRUE selects mstop by CV; slow

# No seed: rerun the script a few times to see the run-to-run spread.

# ── Data ──────────────────────────────────────────────────────────────────────
cal <- calibrate(LD, LU)
d   <- make_data(N_RCT, N_RWD, LD, LU, cal, censoring = "right")
describe(d)

Xdf <- as.data.frame(d$X)                       # bujar wants named columns
names(Xdf) <- paste0("x", seq_len(ncol(Xdf)))

# ── Fit and predict, kept separate ────────────────────────────────────────────
# The separation matters for the S-learner: one fit, predicted at A = 1 and at
# A = 0.  Fitting twice would double the cost, and for any learner with a random
# component it would also contaminate the CATE with initialisation noise.
#
# d$y is ALREADY the standardised LOG survival time.  Do not log it again.
# `cens` is the event indicator (1 = event, 0 = right-censored).
bj_fit <- function(rows, extra, learner) {
  bujar(y       = d$y[rows],
        cens    = d$status[rows],
        x       = cbind(Xdf[rows, , drop = FALSE], extra[rows, , drop = FALSE]),
        learner = learner,
        degree  = DEGREE,
        mstop   = MSTOP,
        nu      = NU,
        tuning  = TUNING,
        cv      = TUNING,
        nfold   = 5L)
}

bj_pred <- function(fit, newdata) as.numeric(predict(fit, newx = newdata))

# ── Meta-learners ─────────────────────────────────────────────────────────────
# S-learner: one model with A as a covariate; CATE = mu(x, A=1) - mu(x, A=0).
cate_S <- function(rows, extra_cols, eval_extra, learner) {
  extra <- cbind(extra_cols, A = d$A)
  fit   <- bj_fit(rows, extra, learner)
  bj_pred(fit, cbind(Xdf, eval_extra, A = 1L)) -
    bj_pred(fit, cbind(Xdf, eval_extra, A = 0L))
}

# T-learner: a separate model per arm; CATE = mu_1(x) - mu_0(x).
cate_T <- function(rows, extra_cols, eval_extra, learner) {
  ev  <- cbind(Xdf, eval_extra)
  r1  <- rows[d$A[rows] == 1L]
  r0  <- rows[d$A[rows] == 0L]
  bj_pred(bj_fit(r1, extra_cols, learner), ev) -
    bj_pred(bj_fit(r0, extra_cols, learner), ev)
}

# Designs: which rows train the model, and what extra columns they carry.
# The pooled design adds S; the CATE is evaluated at S = 1 throughout.
rows_rct  <- which(d$S == 1L)
rows_pool <- seq_along(d$S)
none      <- d$X[, 0, drop = FALSE]             # zero-column placeholder
Sdf       <- data.frame(S = d$S)
S1df      <- data.frame(S = rep(1L, nrow(Xdf)))

designs <- list(
  list(name = "RCT-S",    rows = rows_rct,  fun = cate_S, extra = none, ev = none),
  list(name = "RCT-T",    rows = rows_rct,  fun = cate_T, extra = none, ev = none),
  list(name = "Pooled-S", rows = rows_pool, fun = cate_S, extra = Sdf,  ev = S1df),
  list(name = "Pooled-T", rows = rows_pool, fun = cate_T, extra = Sdf,  ev = S1df)
)

# ── Run ───────────────────────────────────────────────────────────────────────
out <- list()
for (learner in LEARNERS) {
  cat("\n=== bujar, learner =", learner, "===\n")
  for (des in designs) {
    lab <- paste0("bujar-", learner, "-", des$name)
    res <- tryCatch({
      cate <- des$fun(des$rows, des$extra, des$ev, learner)
      report(d, cate, lab)
    }, error = function(e) {
      message("  ", lab, " failed: ", conditionMessage(e))
      NULL
    })
    if (!is.null(res)) out[[lab]] <- res
  }
}

if (length(out)) {
  cat("\n=== CATE metrics, one replication ===\n")
  print(do.call(rbind, out), row.names = FALSE, digits = 3)
} else {
  cat("\nNo estimator completed. Check the learner name against ?bujar.\n")
}
