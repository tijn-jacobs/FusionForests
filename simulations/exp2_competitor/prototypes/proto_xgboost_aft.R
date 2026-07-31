# ──────────────────────────────────────────────────────────────────────────────
# Prototype: XGBoost with the AFT likelihood as a CATE comparator.
#
#   install.packages("xgboost"); library(xgboost)
#   Barnwal, Cho & Hocking (JCGS), "Survival regression with accelerated
#   failure time model in XGBoost".
#
# The only comparator considered that accepts INTERVAL-censored labels, so it
# is fitted on the paper's actual design (RCT right-censored, RWD interval-
# censored) rather than on a degraded right-censored-only version.  That is the
# point of including it, so `censoring = "mixed"` below is deliberate.
#
# Four estimators, matching the deepAFT design:
#   RCT-S, RCT-T, Pooled-S, Pooled-T
#
# One replication, point estimates only.  Bootstrap intervals come later.
#
# Run from the repository root:
#   Rscript simulations/exp2_competitor/prototypes/proto_xgboost_aft.R
# ──────────────────────────────────────────────────────────────────────────────

setwd("~/Library/CloudStorage/OneDrive-VrijeUniversiteitAmsterdam/Documents/GitHub/FusionForests")
source("simulations/exp2_competitor/prototypes/dgp_prototype.R")
library(xgboost)

# ── Config ────────────────────────────────────────────────────────────────────
# aft_loss_distribution is the analogue of the error model the fusion forest
# treats nonparametrically, so it is a substantive comparison point rather than
# a nuisance setting.  Tune it and report the winner.
#   candidates: "normal", "logistic", "extreme"
DISTS   <- c("normal")
SCALE   <- 1.0              # aft_loss_distribution_scale
NROUNDS <- 200L
PARAMS  <- list(objective     = "survival:aft",
                eval_metric   = "aft-nloglik",
                tree_method   = "hist",
                learning_rate = 0.05,
                max_depth     = 2L)

N_RCT <- 150L
N_RWD <- 350L
LD    <- 1.0                # between-source heterogeneity
LU    <- 1.0                # unmeasured confounding strength

# No seed: rerun the script a few times to see the run-to-run spread.

# ── Data ──────────────────────────────────────────────────────────────────────
cal <- calibrate(LD, LU)
d   <- make_data(N_RCT, N_RWD, LD, LU, cal, censoring = "mixed")
describe(d)

# ── One XGBoost-AFT fit -> AFT linear predictor mu on newX ────────────────────
# Labels go in on the TIME scale; the AFT loss takes the log internally.  Our
# bounds are standardised log-times, so exp() puts them where xgboost wants
# them and the fit works on the standardised log scale throughout.
#
#   exact           lower = upper = t
#   right-censored  lower = t, upper = +Inf
#   interval        lower = a, upper = b
#
# IMPORTANT: for survival:aft, predict() returns exp(margin), i.e. a predicted
# survival TIME.  The CATE is a difference on the log scale, so we need the
# raw margin -- hence outputmargin = TRUE.  Dropping it silently gives a
# difference of times instead of a log-ratio.
xgb_mu <- function(rows, Z, newZ, dist) {
  dtrain <- xgb.DMatrix(as.matrix(Z[rows, , drop = FALSE]))
  setinfo(dtrain, "label_lower_bound", exp(d$lower[rows]))
  setinfo(dtrain, "label_upper_bound", exp(d$upper[rows]))
  params <- c(PARAMS,
              list(aft_loss_distribution       = dist,
                   aft_loss_distribution_scale = SCALE))
  bst <- xgb.train(params, dtrain, nrounds = NROUNDS)
  predict(bst, xgb.DMatrix(as.matrix(newZ)), outputmargin = TRUE)
}

# ── Meta-learners ─────────────────────────────────────────────────────────────
# S-learner: one model with A as a covariate; CATE = mu(x, A=1) - mu(x, A=0).
cate_S <- function(rows, extra, eval_extra, dist) {
  Z   <- cbind(d$X, extra, A = d$A)
  ev1 <- cbind(d$X, eval_extra, A = 1L)
  ev0 <- cbind(d$X, eval_extra, A = 0L)
  xgb_mu(rows, Z, ev1, dist) - xgb_mu(rows, Z, ev0, dist)
}

# T-learner: a separate model per arm; CATE = mu_1(x) - mu_0(x).
cate_T <- function(rows, extra, eval_extra, dist) {
  Z  <- cbind(d$X, extra)
  ev <- cbind(d$X, eval_extra)
  r1 <- rows[d$A[rows] == 1L]
  r0 <- rows[d$A[rows] == 0L]
  xgb_mu(r1, Z, ev, dist) - xgb_mu(r0, Z, ev, dist)
}

# Designs: the pooled fits carry the source indicator S and are evaluated at
# S = 1 (the trial estimand), as in the deepAFT comparison.
rows_rct  <- which(d$S == 1L)
rows_pool <- seq_along(d$S)
none      <- d$X[, 0, drop = FALSE]             # zero-column placeholder
Smat      <- cbind(S = d$S)
S1mat     <- cbind(S = rep(1L, nrow(d$X)))

designs <- list(
  list(name = "RCT-S",    rows = rows_rct,  fun = cate_S, extra = none,  ev = none),
  list(name = "RCT-T",    rows = rows_rct,  fun = cate_T, extra = none,  ev = none),
  list(name = "Pooled-S", rows = rows_pool, fun = cate_S, extra = Smat,  ev = S1mat),
  list(name = "Pooled-T", rows = rows_pool, fun = cate_T, extra = Smat,  ev = S1mat)
)

# ── Run ───────────────────────────────────────────────────────────────────────
out <- list()
for (dist in DISTS) {
  cat("\n=== XGBoost-AFT, aft_loss_distribution =", dist, "===\n")
  for (des in designs) {
    lab <- paste0("xgbAFT-", dist, "-", des$name)
    res <- tryCatch({
      cate <- des$fun(des$rows, des$extra, des$ev, dist)
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
  cat("\nNo estimator completed.\n")
}
