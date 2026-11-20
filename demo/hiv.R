# Data fusion of the ACTG 175 trial and synthetic observational HIV data:
# the main analysis of the paper.  hiv_rwd is synthetic, so the results
# approximate, but do not reproduce, those of the paper.
#
# Treatment: an alternative regimen (1) against zidovudine monotherapy (0).
# Target population: the patients of both sources.

if (!requireNamespace("speff2trial", quietly = TRUE))
  stop("Install the speff2trial package for the ACTG 175 data.")
library(FusionForests)

# MCMC length: 5000 burn-in and 5000 posterior draws as in the paper (the
# default); lower both for a quick run.

ask <- function(what, default = 5000L) {
  if (!interactive()) return(default)
  x <- suppressWarnings(as.integer(
    readline(sprintf("%s [Enter for %d]: ", what, default))))
  if (is.na(x) || x < 1L) default else x
}
N_burn <- ask("Number of burn-in draws")
N_post <- ask("Number of posterior draws")
covariates <- c("age", "cd4", "cd8", "race", "prior_art_years")

# ---- Data -----------------------------------------------------------------
# Trial: the men of ACTG 175, follow-up in years.
data("ACTG175", package = "speff2trial")
t175 <- ACTG175[ACTG175$gender == 1, ]
rct <- data.frame(trt = t175$treat, age = t175$age, cd4 = t175$cd40,
                  cd8 = t175$cd80, race = t175$race,
                  prior_art_years = t175$preanti / 365.25,
                  time = t175$days / 365.25, status = t175$cens)

# Observational data: an event lies in (time_lower, time_upper]; the other
# patients are right-censored at time_lower.
rwd <- hiv_rwd

X   <- as.matrix(rbind(rct[, covariates], rwd[, covariates]))
trt <- c(rct$trt, rwd$trt)
src <- rep(1:0, c(nrow(rct), nrow(rwd)))      # 1 = trial, 0 = observational
y   <- log(c(rct$time, rwd$time_lower))
ic  <- c(rep(0, nrow(rct)), rwd$status)       # interval-censored rows
right <- log(c(rct$time,
               ifelse(rwd$status == 1, rwd$time_upper, rwd$time_lower)))

# ---- Fusion forest ----------------------------------------------------------
# The default priors and error distribution are those of the paper.
set.seed(2026)
fit <- FusionForest(
  y = y, status = c(rct$status, rep(0, nrow(rwd))),
  observed_left_time = y, observed_right_time = right,
  interval_censoring_indicator = ic,
  X_train_control = X, X_train_treat = X,
  treatment_indicator_train = trt, source_indicator_train = src,
  outcome_type = "right-censored", timescale = "log",
  treatment_coding = "binary",
  N_post = N_post, N_burn = N_burn, store_posterior_sample = TRUE)

# ---- Trial only, predicted for all patients ---------------------------------
# Needs ShrinkageTrees >= 2.1.1 (earlier versions ignore the weights of the
# treatment forest).
fit_trial <- CausalShrinkageForest(
  y = y[src == 1], status = rct$status,
  X_train_control = X[src == 1, ], X_train_treat = X[src == 1, ],
  treatment_indicator_train = trt[src == 1],
  X_test_control = X[src == 0, ], X_test_treat = X[src == 0, ],
  treatment_indicator_test = trt[src == 0],
  outcome_type = "right-censored", timescale = "log",
  prior_type_control = "standard", prior_type_treat = "standard",
  local_hp_control = 1 / sqrt(200), local_hp_treat = 1 / sqrt(100),
  number_of_trees_control = 200, number_of_trees_treat = 100,
  power_control = 2, base_control = 0.95,
  power_treat = 2, base_treat = 0.95,
  treatment_coding = "binary",
  N_post = N_post, N_burn = N_burn, store_posterior_sample = TRUE)

# ---- Results ----------------------------------------------------------------
# Posterior draws of the log-time CATE (draws x patients).
tau <- list(
  fusion = fit$train_predictions_sample_treat,
  trial  = cbind(fit_trial$train_predictions_sample_treat,
                 fit_trial$test_predictions_sample_treat))

# Mean width of the 95% interval of the acceleration factor, and the share
# of patients with P(tau > 0 | data) > 0.95.
summarise <- function(s) {
  af <- exp(s)
  c(width = mean(apply(af, 2, quantile, 0.975) -
                 apply(af, 2, quantile, 0.025)),
    pct_P_above_0.95 = 100 * mean(colMeans(s > 0) > 0.95))
}
groups <- list(trial = src == 1, observational = src == 0, all = src >= 0)
for (m in names(tau)) {
  cat("\n", m, "\n", sep = "")
  print(round(sapply(groups, function(g) summarise(tau[[m]][, g])), 2))
}

# Caterpillar plot of the acceleration factor per patient.
caterpillar <- function(s, main) {
  af  <- exp(s)
  ord <- order(colMeans(af))
  lo  <- apply(af, 2, quantile, 0.025)[ord]
  hi  <- apply(af, 2, quantile, 0.975)[ord]
  col <- ifelse(src[ord] == 1, "#0072B2", "#D55E00")
  plot(NA, xlim = range(lo, hi), ylim = c(1, length(ord)), yaxt = "n",
       xlab = "Acceleration factor", ylab = "Patients", main = main)
  segments(lo, seq_along(ord), hi, seq_along(ord), col = col)
  points(colMeans(af)[ord], seq_along(ord), pch = 16, cex = 0.2)
  abline(v = 1, lty = 2)
  legend("topleft", c("Trial", "Observational"),
         col = c("#0072B2", "#D55E00"), lwd = 2, bty = "n")
}
op <- par(mfrow = c(1, 2))
caterpillar(tau$fusion, "Fusion forest")
caterpillar(tau$trial, "Trial only")
par(op)

# Summary of the fusion CATE by a single decision tree: a shallow tree on
# the posterior-mean CATE; each leaf reports the posterior mean and 95%
# interval of its acceleration factor.
if (requireNamespace("rpart", quietly = TRUE)) {
  tau_mean <- colMeans(tau$fusion)
  tree <- rpart::rpart(tau_mean ~ ., data = data.frame(tau_mean, X),
                       control = rpart::rpart.control(maxdepth = 3, cp = 0.01,
                                                      minbucket = 30))
  node <- rownames(tree$frame)[tree$where]
  leaves <- t(sapply(split(seq_along(node), node), function(i) {
    af <- rowMeans(exp(tau$fusion[, i, drop = FALSE]))
    c(n = length(i), af = mean(af), quantile(af, c(0.025, 0.975)))
  }))
  cat("\nLeaves of the summary tree (node, n, acceleration factor)\n")
  print(round(leaves, 2))
  plot(tree, uniform = TRUE, margin = 0.1)
  text(tree, use.n = TRUE)
}

# Linear projection of the deviation forest d(x) of the observational
# patients onto the covariates; log-timescale, CD4 and CD8 per 100 cells/mm^3, 
# covariates centred.
Z <- rwd[, covariates]
Z[, c("cd4", "cd8")] <- Z[, c("cd4", "cd8")] / 100
Phi <- model.matrix(~ ., Z)
Phi[, -1] <- scale(Phi[, -1], scale = FALSE)
G <- fit$train_predictions_sample_deviation   # observational rows only
gam <- t(apply(G, 1, function(g)
  lm.wfit(Phi, g, w = rexp(length(g)))$coefficients))
cat("\nProjection of the deviation forest\n")
print(round(cbind(mean = colMeans(gam),
                  t(apply(gam, 2, quantile, c(0.025, 0.975)))), 3))


