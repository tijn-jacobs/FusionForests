################################################################################
## data/analysis 1/make_competitor_caterpillars.R
##
## The three additional machine-learning comparators on the ACTG175 + MACS
## cohort, in exactly the shape of the deepAFT section of analysis.R:
##
##   XGBoost-AFT   gradient-boosted trees under an accelerated failure time loss
##   BJ-ELM        Buckley-James boosting over extreme learning machines
##   BJ-trees      Buckley-James boosting over regression trees
##
## Each is fitted as an S- and a T-learner, on the trial alone and on a naive
## pool of both sources with a source indicator, and the CATE is evaluated at
## S = 1 (the transported / trial estimand) over every subject of the combined
## cohort, in the rct-then-macs order of M1_combined_samples / M2_combined_samples.
##
## Intervals come from a stratified nonparametric bootstrap -- none of the three
## has a posterior -- the frequentist analogue of the BART credible interval.
##
## The two Buckley-James methods consume right-censored data only, so the
## interval-censored MACS deaths reach them through the right-censored
## last-lab-year encoding, the same approximation the deepAFT competitor makes.
## XGBoost-AFT is the exception: its loss takes interval labels, so it is fitted
## on the actual bounds. That is the whole reason for including the method, and
## the application is where the capability can be exercised -- the MACS deaths
## really are interval-censored. It also means XGBoost-AFT sees strictly more
## information here than the other three, which is stated in ANALYSIS_SM.tex.
##
## Writes, per method:
##   data/analysis 1/analysis_<tag>.rds    summary, fits (point + draws), settings
##   data/analysis 1/analysis_<tag>.txt    the summary as text
##   notes/general/figures/cate_caterpillar_<tag>_{rct,pooled}_{s,t}.pdf
##
## Run from the project root:
##   Rscript "data/analysis 1/make_competitor_caterpillars.R"
################################################################################

suppressPackageStartupMessages({
  library(ggplot2)
  library(survival)
})

proj    <- "."
out_dir <- file.path(proj, "data", "analysis 1")
fig_dir <- file.path(proj, "notes", "general", "figures")
dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)

## --- Settings ---------------------------------------------------------------
B      <- 100L   # bootstrap refits per estimator, matching the deepAFT section
NCORES <- max(1L, parallel::detectCores() - 1L)   # 1 = sequential

## BJ-ELM must run sequentially. Its ELM step solves a least-squares problem
## with MASS::ginv(), which calls LAPACK through the platform BLAS. On macOS
## that is Apple's Accelerate framework, which is not fork-safe: a forked
## mclapply child segfaults inside La.svd ("invalid permissions"). The point
## estimate survives because it runs in the parent. XGBoost and BJ-trees are
## unaffected -- boosting and trees never touch LAPACK.
##
## Sequential is the certain fix. A PSOCK cluster would also work, since those
## are fresh processes rather than forks, but it needs every object exported by
## hand and buys little here.
NCORES_BY_METHOD <- c(xgbaft = NCORES, bjtree = NCORES, bjelm = 1L)
SEED   <- 1L

## XGBoost tuning, following Barnwal, Cho & Hocking: random search over their
## six hyperparameters, roughly 100 trials, 5-fold CV. aft_loss_distribution is
## fixed to the normal rather than searched, as in the simulation.
XGB_TUNE     <- TRUE
XGB_TRIALS   <- 100L
XGB_NFOLD    <- 5L
XGB_MAXROUND <- 500L
XGB_EARLY    <- 20L

## Kong & Zhang's defaults for BJ-ELM; Wang & Wang's for BJ-trees. degree = 2
## lets the tree learner form second-order interactions.
NHID <- 20L; NBASE <- 25L
DEGREE <- 2L; MSTOP <- 50L; NU <- 0.1

## Which methods to run; drop entries to run a subset.
METHODS <- c("xgbaft", "bjelm", "bjtree")

## --- Cohort (data prep only, no model fitting) ------------------------------
source(file.path(out_dir, "data_prep.R"))
rct    <- build_rct(male_only = TRUE)
macs   <- build_macs(cd4_band = c(200, 500))
d_fuse <- rbind(rct, macs)
d_fuse <- d_fuse[is.finite(d_fuse$log_time), ]
src_M2 <- as.integer(d_fuse$source == "RCT")
n_rct  <- nrow(rct); n_macs <- nrow(macs)

cat(sprintf("Cohort: %d trial + %d cohort = %d subjects\n",
            n_rct, n_macs, nrow(d_fuse)))

## --- Scaling ----------------------------------------------------------------
## The covariates and the log-time response are z-scored on the combined cohort,
## exactly as in the deepAFT section. Trees are indifferent to covariate scale,
## but the ELM's sigmoid activations are not, and standardising the response
## puts every method on the same footing as the simulation, where the CATE is
## estimated on a standardised scale and rescaled afterwards. The centring
## cancels in the treated-minus-control contrast; only the SD is undone.
cov_mu <- colMeans(as.matrix(d_fuse[, harmCovars]))
cov_sd <- apply(as.matrix(d_fuse[, harmCovars]), 2, sd)
cov_sd[cov_sd == 0] <- 1
y_mu   <- mean(d_fuse$log_time)
y_sd   <- sd(d_fuse$log_time)

Z    <- function(df) scale(as.matrix(df[, harmCovars]),
                           center = cov_mu, scale = cov_sd)
ylog <- function(df) (df$log_time - y_mu) / y_sd     # standardised log time
Xev  <- Z(d_fuse)                                     # fixed evaluation set

## Interval labels for XGBoost-AFT, on the same standardised log-time scale.
## data_prep.R already encodes the three cases; status_ic separates an exact
## trial event from a right-censored one, and ic_indicator flags the MACS
## deaths whose event time is known only to lie in a bracket.
##   exact           lower = upper = log t
##   right-censored  lower = log c,  upper = Inf
##   interval        lower = log a,  upper = log b
ybounds <- function(df) {
  lo <- (df$obs_left_log_t - y_mu) / y_sd
  hi <- rep(Inf, nrow(df))
  ex <- df$ic_indicator == 0L & df$status_ic == 1L
  ic <- df$ic_indicator == 1L
  hi[ex] <- lo[ex]
  hi[ic] <- (df$obs_right_log_t[ic] - y_mu) / y_sd
  list(lo = lo, hi = hi)
}

## --- Designs ----------------------------------------------------------------
## A design carries the training response, covariates and treatment, plus the
## fixed evaluation covariates. The pooled design adds the source indicator and
## evaluates at S = 1.
## `y`/`status` drive the two Buckley-James methods; `lo`/`hi` drive
## XGBoost-AFT, which is the only one that can use the interval bounds.
des_rct <- function(rdf) {
  b <- ybounds(rdf)
  list(y = ylog(rdf), status = rdf$status, lo = b$lo, hi = b$hi,
       X = Z(rdf), A = rdf$treat, Xev = Xev)
}
des_pool <- function(fdf, src) {
  b <- ybounds(fdf)
  list(y = ylog(fdf), status = fdf$status, lo = b$lo, hi = b$hi,
       X = cbind(Z(fdf), S = src), A = fdf$treat,
       Xev = cbind(Xev, S = 1L))
}

## Bootstrap resamples the trial and cohort rows separately, sizes fixed; the
## evaluation set is never resampled.
make_des <- function(type, boot = FALSE) {
  if (type == "rct") {
    rdf <- if (boot) rct[sample.int(n_rct, n_rct, TRUE), ] else rct
    return(des_rct(rdf))
  }
  if (!boot) return(des_pool(d_fuse, src_M2))
  i_r <- sample.int(n_rct,  n_rct,  TRUE)
  i_m <- sample.int(n_macs, n_macs, TRUE)
  des_pool(rbind(rct[i_r, ], macs[i_m, ]),
           c(rep(1L, n_rct), rep(0L, n_macs)))
}

## --- Method: XGBoost-AFT ----------------------------------------------------
## Labels go in on the TIME scale; the AFT loss takes the log internally. Our
## response is a standardised log time, so exp() puts it where xgboost wants it
## and the fit works on the standardised log scale throughout. Right-censored
## rows get an infinite upper bound.
##
## predict() returns exp(margin) for survival:aft, i.e. a predicted time. The
## CATE is a difference on the log scale, so outputmargin = TRUE is required.
xgb_base <- list(objective = "survival:aft", eval_metric = "aft-nloglik",
                 aft_loss_distribution = "normal", tree_method = "hist",
                 nthread = 1L)
xgb_grid <- list(
  learning_rate               = c(0.001, 0.01, 0.1, 1.0),
  max_depth                   = 2:10,
  min_child_weight            = c(0.001, 0.1, 1, 10, 100),
  reg_alpha                   = c(0.001, 0.01, 0.1, 1, 10, 100),
  reg_lambda                  = c(0.001, 0.01, 0.1, 1, 10, 100),
  aft_loss_distribution_scale = c(0.5, 0.8, 1.1, 1.4, 1.7, 2.0))

xgb_dm <- function(lo, hi, Zm) {
  d <- xgboost::xgb.DMatrix(as.matrix(Zm))
  xgboost::setinfo(d, "label_lower_bound", exp(lo))
  xgboost::setinfo(d, "label_upper_bound", exp(hi))   # exp(Inf) = Inf is fine
  d
}

xgb_tune <- function(lo, hi, Zm) {
  draws <- lapply(seq_len(XGB_TRIALS), function(i)
    c(xgb_base, lapply(xgb_grid, function(v) v[sample.int(length(v), 1L)])))
  dtr <- xgb_dm(lo, hi, Zm)
  sc  <- lapply(draws, function(p) {
    cv <- tryCatch(xgboost::xgb.cv(params = p, data = dtr, nrounds = XGB_MAXROUND,
                                   nfold = XGB_NFOLD, early_stopping_rounds = XGB_EARLY,
                                   verbose = 0, showsd = FALSE),
                   error = function(e) NULL)
    if (is.null(cv)) return(NULL)
    el <- as.data.frame(cv$evaluation_log)
    mc <- grep("^test_.*_mean$", names(el), value = TRUE)[1]
    it <- if (!is.null(cv$best_iteration)) cv$best_iteration else which.min(el[[mc]])
    if (!is.finite(el[[mc]][it])) return(NULL)
    list(score = el[[mc]][it], params = p, nrounds = it)
  })
  sc <- Filter(Negate(is.null), sc)
  if (!length(sc)) stop("every XGBoost tuning trial failed")
  sc[[which.min(vapply(sc, `[[`, numeric(1), "score"))]]
}

## Signature matches the other methods so cate_T can call them alike; for
## XGBoost the first two arguments are the interval bounds, not y and status.
xgb_fit_pred <- function(lo, hi, Zm, newZ, tuned) {
  bst <- xgboost::xgb.train(tuned$params, xgb_dm(lo, hi, Zm),
                            nrounds = tuned$nrounds, verbose = 0)
  predict(bst, xgboost::xgb.DMatrix(as.matrix(newZ)), outputmargin = TRUE)
}

## --- Method: BJ-ELM ---------------------------------------------------------
## Fitting code from the supplementary material of Kong & Zhang, inlined so the
## script has no companion files. Only the fitting path is reproduced; their
## CV_BJ_ELM() is not used. Needs MASS (ginv) and emplik (WKM, and the
## cumsumsurv C routine reached through .C() without a PACKAGE argument, so
## emplik must be attached rather than merely installed).
## NOTE: these are defined at top level, not inside a loader function. emplik
## exports its own cumsumsurv(), so a `<<-` assignment walks up the search path
## and hits that locked package binding. Defining at top level puts ours in the
## global environment, which masks the package version -- which is exactly what
## the authors' own script does.
actfun <- function(x) 1 / (1 + exp(-1 * x))
randomMatrix <- function(nCols, nRows)
  matrix(runif(nCols * nRows, min = -1, max = 1), ncol = nCols)
elmtrain.default <- function(x, y, nhid) {
  if (nhid < 1) stop("number of hidden neurons must be >= 1")
  T <- t(y); P <- t(x)
  inpweight <- randomMatrix(nrow(P), nhid)
  tempH <- inpweight %*% P
  biashid <- runif(nhid, min = -1, max = 1)
  biasMatrix <- matrix(rep(biashid, ncol(P)), nrow = nhid, ncol = ncol(P), byrow = FALSE)
  tempH <- tempH + biasMatrix
  H <- 1 / (1 + exp(-1 * tempH))
  outweight <- ginv(t(H), tol = sqrt(.Machine$double.eps)) %*% t(T)
  fitted.values <- t(t(t(H) %*% outweight))
  list(inpweight = inpweight, biashid = biashid, outweight = outweight,
       nhid = nhid, fitted.values = fitted.values)
}
ELM_boosting <- function(x, y, nhid = 20, nbase = 25, step = 0.1) {
  bias <- matrix(NA, ncol = nbase, nrow = nhid)
  hid_weights <- vector("list", length = nbase)
  output_weights <- matrix(NA, ncol = nbase, nrow = nhid)
  target <- matrix(NA, ncol = nbase + 1, nrow = nrow(x)); target[, 1] <- y
  fit_value <- matrix(NA, ncol = nbase, nrow = nrow(x))
  for (k in 1:nbase) {
    bl <- elmtrain.default(x = x, y = target[, k], nhid = nhid)
    bias[, k] <- c(bl$biashid); hid_weights[[k]] <- t(bl$inpweight)
    output_weights[, k] <- c(bl$outweight)
    target[, k + 1] <- c(target[, k]) - c(step * bl$fitted.values)
    fit_value[, k] <- c(bl$fitted.values)
  }
  list(y_hat = step * apply(fit_value, 1, sum), hid_weights = hid_weights,
       bias = bias, output_weights = output_weights, residuals = target,
       step = step)
}
cumsumsurv <- function(x) {
  if (any(is.na(x))) stop("NaNs")
  .C("cumsumsurv", x = as.numeric(x), s = as.numeric(x), LLL = length(x))$s
}
bj_iter <- function(x, y, status, y_hat, nhid, nbase) {
  N <- length(status); u <- y_hat; res <- y - u
  o <- order(res, -status)
  resorder <- res[o]; dorder <- status[o]; dorder[N] <- 1
  uorder <- u[o]; ystar <- y[o]; xorder <- as.matrix(x[o, ])
  temp <- WKM(x = resorder, d = dorder, zc = 1:N)
  jifen <- cumsumsurv(resorder * temp$jump); S <- temp$surv
  for (i in 1:N) if (dorder[i] == 0) ystar[i] <- uorder[i] + jifen[i] / S[i]
  ELM_boosting(x = xorder, y = ystar, nhid = nhid, nbase = nbase, step = 0.1)
}
BJ_ELM <- function(x, y, status, nhid = 20, nbase = 25) {
  maxiter <- 30; eps <- 1e-05; x <- as.matrix(x)
  m <- vector("list", 3)
  m[[1]] <- ELM_boosting(x = x, y = y, nhid = nhid, nbase = nbase, step = 0.1)
  f <- matrix(NA, ncol = nrow(x), nrow = 3); f[1, ] <- m[[1]]$y_hat
  for (i in 2:3) {
    m[[i]] <- bj_iter(x, y, status, f[i - 1, ], nhid, nbase)
    f[i, ] <- m[[i]]$y_hat
  }
  k <- 2
  while (k <= maxiter && eps <= sum(abs(f[2, ] - f[3, ]))) {
    f[2, ] <- f[3, ]; m[[2]] <- m[[3]]
    m[[3]] <- bj_iter(x, y, status, f[2, ], nhid, nbase)
    f[3, ] <- m[[3]]$y_hat; k <- k + 1
  }
  list(model = m[[3]])
}
pre_BJ_ELM <- function(model, newx) {
  hw <- model$hid_weights; bias <- model$bias
  beta <- model$output_weights; step <- model$step
  cols <- lapply(seq_len(ncol(beta)), function(x)
    actfun(newx %*% as.matrix(hw[[x]]) +
             matrix(rep(bias[, x], nrow(newx)), nrow = nrow(newx),
                    byrow = TRUE)) %*% matrix(beta[, x], ncol = 1))
  step * apply(do.call(cbind, cols), 1, sum)
}
bjelm_fit_pred <- function(y, status, Zm, newZ, tuned)
  as.numeric(pre_BJ_ELM(BJ_ELM(x = as.matrix(Zm), y = y, status = status,
                               nhid = NHID, nbase = NBASE)$model,
                        as.matrix(newZ)))

## --- Method: BJ-trees -------------------------------------------------------
## bujar wants a data frame with named columns, and predict() takes `newx`.
as_frame <- function(M) {
  df <- as.data.frame(M)
  names(df) <- make.names(colnames(M), unique = TRUE)
  df
}
bjtree_fit_pred <- function(y, status, Zm, newZ, tuned) {
  fit <- bujar::bujar(y = y, cens = status, x = as_frame(Zm),
                      learner = "tree", degree = DEGREE,
                      mstop = MSTOP, nu = NU,
                      tuning = FALSE, cv = FALSE, n.cores = 1L)
  as.numeric(predict(fit, newx = as_frame(newZ)))
}

## --- Meta-learners ----------------------------------------------------------
## S-learner: ONE fit, predicted at A = 1 and A = 0. Fitting twice would double
## the cost and, for the ELM, leak its random hidden layer into the contrast.
cate_S <- function(des, fitp, tuned) {
  Zm  <- cbind(des$X, A = des$A)
  ev1 <- cbind(des$Xev, A = 1); ev0 <- cbind(des$Xev, A = 0)
  if (identical(fitp, xgb_fit_pred)) {
    bst <- xgboost::xgb.train(tuned$params, xgb_dm(des$lo, des$hi, Zm),
                              nrounds = tuned$nrounds, verbose = 0)
    p <- function(nz) predict(bst, xgboost::xgb.DMatrix(as.matrix(nz)),
                              outputmargin = TRUE)
    return(p(ev1) - p(ev0))
  }
  if (identical(fitp, bjelm_fit_pred)) {
    m <- BJ_ELM(x = as.matrix(Zm), y = des$y, status = des$status,
                nhid = NHID, nbase = NBASE)$model
    return(as.numeric(pre_BJ_ELM(m, as.matrix(ev1))) -
             as.numeric(pre_BJ_ELM(m, as.matrix(ev0))))
  }
  fit <- bujar::bujar(y = des$y, cens = des$status, x = as_frame(Zm),
                      learner = "tree", degree = DEGREE, mstop = MSTOP,
                      nu = NU, tuning = FALSE, cv = FALSE, n.cores = 1L)
  as.numeric(predict(fit, newx = as_frame(ev1))) -
    as.numeric(predict(fit, newx = as_frame(ev0)))
}

## T-learner: a separate fit per arm, no treatment input.
cate_T <- function(des, fitp, tuned) {
  i1 <- des$A == 1; i0 <- des$A == 0
  ## XGBoost takes (lower, upper); the Buckley-James methods take (y, status).
  a <- if (identical(fitp, xgb_fit_pred)) des$lo else des$y
  b <- if (identical(fitp, xgb_fit_pred)) des$hi else des$status
  fitp(a[i1], b[i1], des$X[i1, , drop = FALSE], des$Xev, tuned) -
    fitp(a[i0], b[i0], des$X[i0, , drop = FALSE], des$Xev, tuned)
}

## --- Caterpillar (mirrors make_caterpillar() in analysis.R) -----------------
src_cols <- c(RCT = "#0072B2", RWD = "#D55E00")   # Okabe-Ito, as in analysis.R
## `compact = TRUE` emits a panel sized for a small-multiples grid: no legend,
## no axis titles, and a canvas whose base_size survives being placed at a
## quarter of the text width. The full-size variant keeps everything and is what
## the main-text caterpillar uses. Generating at 3.2in with base_size 14 and
## placing at 1.6in leaves the text at an effective 7pt; generating at 14in and
## shrinking to 1.6in would leave it at 3pt.
make_caterpillar <- function(samples, src, out_path, compact = FALSE,
                             width = if (compact) 3.2 else 14,
                             height = if (compact) 2.4 else 8,
                             base_size = if (compact) 14 else 24,
                             xlim = NULL) {
  af <- exp(samples)
  mean_af <- colMeans(af)
  lo_af <- apply(af, 2, quantile, 0.025); hi_af <- apply(af, 2, quantile, 0.975)
  ord <- order(mean_af)
  src_lab <- factor(ifelse(src == 1L, "RCT", "RWD"), levels = c("RCT", "RWD"))
  df <- data.frame(rank = seq_along(ord), mean = mean_af[ord],
                   lo = lo_af[ord], hi = hi_af[ord], source = src_lab[ord])
  p <- ggplot(df, aes(y = rank, xmin = lo, xmax = hi, colour = source)) +
    geom_linerange(alpha = 0.45, linewidth = 0.6) +
    geom_point(aes(x = mean), colour = "black", size = 0.6) +
    geom_vline(xintercept = 1, linetype = "dashed", colour = "grey40",
               linewidth = 0.6) +
    scale_colour_manual(values = src_cols) +
    coord_cartesian(xlim = xlim) +
    scale_y_continuous(breaks = NULL) +
    labs(y = if (compact) NULL else "Patients (ordered by bootstrap mean)",
         x = if (compact) NULL else expression("Acceleration factor"),
         colour = NULL) +
    guides(colour = guide_legend(
      override.aes = list(linewidth = 4, alpha = 1, size = 0))) +
    theme_minimal(base_size = base_size) +
    theme(text = element_text(family = "Times"),
          legend.position = if (compact) "none" else "top",
          legend.text = element_text(size = base_size, family = "Times"),
          legend.key.width = unit(40, "pt"),
          legend.key.height = unit(16, "pt"),
          legend.spacing.x = unit(8, "pt"),
          plot.margin = if (compact) margin(2, 4, 2, 2) else margin(6, 6, 6, 6),
          panel.grid.major.y = element_blank(),
          panel.grid.minor.y = element_blank(),
          panel.background = element_rect(fill = "white", colour = NA))
  cairo_pdf(out_path, width = width, height = height)
  print(p); invisible(dev.off())
  cat("    wrote ", basename(out_path), "\n", sep = "")
}

## --- Summaries (identical definitions to the deepAFT section) ---------------
af_width <- function(draws) {
  if (is.null(draws) || !is.matrix(draws)) return(NA_real_)
  q <- apply(exp(draws), 1, quantile, c(0.025, 0.975), na.rm = TRUE)
  mean(q[2, ] - q[1, ])
}
pct_benefit <- function(draws) {
  if (is.null(draws) || !is.matrix(draws)) return(NA_real_)
  100 * mean(apply(draws, 1, function(z) mean(z > 0, na.rm = TRUE)) >= 0.95)
}

## --- Run --------------------------------------------------------------------
spec <- list(
  xgbaft = list(label = "XGBoost-AFT", fitp = xgb_fit_pred, prep = function() library(xgboost)),
  bjelm  = list(label = "BJ-ELM",      fitp = bjelm_fit_pred,
                prep = function() { library(MASS); library(emplik) }),
  bjtree = list(label = "BJ-trees",    fitp = bjtree_fit_pred, prep = function() library(bujar)))

xlims <- list(rct_s = c(0.5, 2.5), rct_t = c(0, 10),
              pooled_s = c(0.5, 2.5), pooled_t = c(0, 10))

## Errors raised inside mclapply happen in a forked child, where message()
## output is usually lost. The failure is therefore returned as a value and
## reported after the fact, so a bootstrap that fails on every refit says why
## instead of silently producing NULL draws.
attempt <- function(expr) tryCatch(list(ok = TRUE, v = expr),
  error = function(e) list(ok = FALSE, msg = conditionMessage(e)))
safe <- function(expr, what) {
  r <- attempt(expr)
  if (r$ok) return(r$v)
  message("      ", what, " failed: ", r$msg)
  NULL
}

for (tag in METHODS) {
  sp <- spec[[tag]]
  cat("\n", strrep("=", 70), "\n  ", sp$label,
      "  (bootstrap on ", NCORES_BY_METHOD[[tag]], " core",
      if (NCORES_BY_METHOD[[tag]] > 1L) "s" else "", ")\n",
      strrep("=", 70), "\n", sep = "")
  sp$prep()

  ## Tuning, once per design, reused for the fit and every bootstrap refit.
  tuned <- list(rct = NULL, pool = NULL)
  if (tag == "xgbaft" && XGB_TUNE) {
    set.seed(SEED)
    cat("  tuning (random search,", XGB_TRIALS, "trials,", XGB_NFOLD, "-fold CV)\n")
    dr <- des_rct(rct)
    tuned$rct <- xgb_tune(dr$lo, dr$hi, cbind(dr$X, A = dr$A))
    dp <- des_pool(d_fuse, src_M2)
    tuned$pool <- xgb_tune(dp$lo, dp$hi, cbind(dp$X, A = dp$A))
    for (k in c("rct", "pool")) {
      p <- tuned[[k]]$params
      cat(sprintf("    %-5s nloglik %.4f | nrounds %d | lr %g, depth %d, scale %g\n",
                  k, tuned[[k]]$score, tuned[[k]]$nrounds,
                  p$learning_rate, p$max_depth, p$aft_loss_distribution_scale))
    }
  }

  ests <- list(
    list(name = paste0(tag, "-RCT-S"),    type = "rct",  fun = cate_S, tk = "rct"),
    list(name = paste0(tag, "-RCT-T"),    type = "rct",  fun = cate_T, tk = "rct"),
    list(name = paste0(tag, "-Pooled-S"), type = "pool", fun = cate_S, tk = "pool"),
    list(name = paste0(tag, "-Pooled-T"), type = "pool", fun = cate_T, tk = "pool"))

  set.seed(SEED)
  fits <- lapply(ests, function(est) {
    cat("  ", est$name, " (point + ", B, " bootstrap)\n", sep = "")
    tu    <- tuned[[est$tk]]
    point <- safe(est$fun(make_des(est$type), sp$fitp, tu), est$name)
    nc   <- NCORES_BY_METHOD[[tag]]
    boot <- parallel::mclapply(seq_len(B), function(b)
      attempt(est$fun(make_des(est$type, TRUE), sp$fitp, tu)),
      mc.cores = nc)
    ## mclapply itself returns a try-error if a worker dies outright.
    boot <- lapply(boot, function(r)
      if (inherits(r, "try-error")) list(ok = FALSE, msg = as.character(r)) else r)
    bad  <- Filter(function(r) !isTRUE(r$ok), boot)
    if (length(bad))
      message("      ", length(bad), "/", B, " bootstrap refits failed; first: ",
              bad[[1]]$msg)
    draws <- do.call(cbind, lapply(Filter(function(r) isTRUE(r$ok), boot), `[[`, "v"))
    list(name  = est$name,
         point = if (is.null(point)) NULL else point * y_sd,
         draws = if (is.matrix(draws)) draws * y_sd else NULL)
  })
  names(fits) <- vapply(ests, `[[`, "", "name")

  summ <- data.frame(
    method      = vapply(fits, `[[`, "", "name"),
    af_ci_width = vapply(fits, function(f) af_width(f$draws), 0),
    pct_benefit = vapply(fits, function(f) pct_benefit(f$draws), 0),
    stringsAsFactors = FALSE)
  rownames(summ) <- NULL
  cat("\n  === ", sp$label, " on the combined cohort (AF scale) ===\n", sep = "")
  print(summ, row.names = FALSE, digits = 3)

  saveRDS(list(summary = summ, fits = fits, tuned = tuned,
               settings = list(B = B, nhid = NHID, nbase = NBASE,
                               degree = DEGREE, mstop = MSTOP)),
          file.path(out_dir, paste0("analysis_", tag, ".rds")))
  writeLines(c(paste(sp$label, "competitor (S- and T-learner) vs the fusion forest"),
               paste0("Combined RCT + MACS cohort, B = ", B, " stratified bootstrap."),
               "MACS interval-censored deaths approximated by the right-censored",
               "last-lab-year encoding (see ANALYSIS_SM.tex).", "",
               capture.output(print(summ, row.names = FALSE, digits = 3))),
             file.path(out_dir, paste0("analysis_", tag, ".txt")))

  for (f in fits) {
    if (is.null(f$draws) || !is.matrix(f$draws)) next
    if (nrow(f$draws) != length(src_M2))
      stop("draws rows (", nrow(f$draws), ") != cohort size (",
           length(src_M2), "): the cohort build does not match the fit.")
    suffix <- tolower(sub(paste0("^", tag, "-"), "", f$name))   # rct-s, pooled-t
    suffix <- gsub("-", "_", suffix)
    make_caterpillar(
      t(f$draws), src_M2,
      file.path(fig_dir, paste0("cate_caterpillar_", tag, "_", suffix, ".pdf")),
      xlim = xlims[[suffix]])
    make_caterpillar(
      t(f$draws), src_M2,
      file.path(fig_dir, paste0("cate_caterpillar_", tag, "_", suffix, "_small.pdf")),
      compact = TRUE, xlim = xlims[[suffix]])
  }
}

cat("\nDone. Figures in ", fig_dir, "\n", sep = "")
