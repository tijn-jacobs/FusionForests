## ACTG175 — Survival BCF on time-to-event outcome (log-time)
##
## Outcome   : log(days), event indicator from `cens`
## Treatment : ZDV monotherapy (0) vs combination therapy (1)
## Model     : 2-forest BCF (ShrinkageTrees::CausalShrinkageForest), AFT-style.
##             We pass log(days) with timescale = "log" so every returned
##             object (mu, tau, samples) stays on the log-time scale and no
##             back-transform gymnastics are needed downstream.

setwd("~/Library/CloudStorage/OneDrive-VrijeUniversiteitAmsterdam/Documents/GitHub/FusionForests/")

library(speff2trial)
library(ShrinkageTrees)
library(ggplot2)
library(survival)

data(ACTG175)

# -------------------------------------------------------------------------
# 1. CLEAN
# -------------------------------------------------------------------------
covars <- c("age", "wtkg", "karnof", "cd40", "cd80",
            "symptom", "str2", "gender", "race", "hemo")

X        <- as.matrix(ACTG175[, covars])
time     <- ACTG175$days
log_time <- log(time)
stat     <- ACTG175$cens
z        <- as.integer(ACTG175$treat)        # 0 = ZDV only, 1 = combination
n        <- nrow(X)

cat(sprintf("n = %d  |  events = %d  |  censored = %d\n",
            n, sum(stat == 1), sum(stat == 0)))

# -------------------------------------------------------------------------
# 2. FIT — CausalShrinkageForest, log-time scale, both priors "standard".
# -------------------------------------------------------------------------
n_trees_treat   <- 50
n_trees_control <- 200

# Range of (imputed) log-times, used to set leaf-prior scales.
log_info         <- ShrinkageTrees:::censored_info(log_time, stat)
local_hp_treat   <- (log_info$max - log_info$min) / (2 * 2 * sqrt(n_trees_treat))
local_hp_control <- (log_info$max - log_info$min) / (2 * 2 * sqrt(n_trees_control))

fit <- ShrinkageTrees::CausalShrinkageForest(
  y                         = log_time,
  status                    = stat,
  X_train_control           = X,
  X_train_treat             = X,
  treatment_indicator_train = z,
  outcome_type              = "right-censored",
  timescale                 = "log",            # keep everything on log-time
  prior_type_control        = "standard",       # was "standard-halfcauchy"
  prior_type_treat          = "standard",
  local_hp_treat            = local_hp_treat,
  local_hp_control          = local_hp_control,
  number_of_trees_control   = n_trees_control,
  number_of_trees_treat     = n_trees_treat,
  power_control             = 2,
  base_control              = 0.95,
  power_treat               = 3,
  base_treat                = 0.25,
  treatment_coding          = "centered",
  N_post                    = 2000,
  N_burn                    = 1000,
  store_posterior_sample    = TRUE,
  verbose                   = TRUE
)

# `fit$sigma` is on the standardised internal scale; recover log-time sigma.
sigma_hat <- log_info$sd
sig_post  <- fit$sigma * sigma_hat

# -------------------------------------------------------------------------
# 3. CATERPILLAR PLOT — patient-level CATE (log-time scale)
# -------------------------------------------------------------------------
tau_samp  <- fit$train_predictions_sample_treat   # N_post x n, log-time
cate_mean <- colMeans(tau_samp)
cate_lo   <- apply(tau_samp, 2, quantile, probs = 0.025)
cate_hi   <- apply(tau_samp, 2, quantile, probs = 0.975)
ord       <- order(cate_mean)

cat("CATE summary (log-time):\n")
cat(sprintf("  est:  min=%.3f  med=%.3f  max=%.3f\n",
            min(cate_mean), median(cate_mean), max(cate_mean)))
cat(sprintf("  CI half-width (median): %.3f\n",
            median((cate_hi - cate_lo) / 2)))

cat_df <- data.frame(
  rank = seq_along(cate_mean),
  est  = cate_mean[ord],
  lo   = cate_lo[ord],
  hi   = cate_hi[ord]
)

p_cat <- ggplot(cat_df, aes(x = rank)) +
  geom_linerange(aes(ymin = lo, ymax = hi),
                 color = "steelblue", alpha = 0.25) +
  geom_point(aes(y = est), size = 0.4, color = "navy") +
  geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
  labs(x = "Patient (ordered by posterior mean CATE)",
       y = "CATE on log-time scale  (log time ratio)",
       title = "Patient-level CATE — caterpillar plot") +
  theme_minimal()
print(p_cat)

# -------------------------------------------------------------------------
# 4. SIGMA TRACE (log-time scale)
# -------------------------------------------------------------------------
plot(sig_post, type = "l",
     xlab = "MCMC iteration",
     ylab = expression(sigma),
     main = "Posterior trace of sigma (log-time)")

# -------------------------------------------------------------------------
# 5. ATE POSTERIOR
# -------------------------------------------------------------------------
ate_samples <- rowMeans(tau_samp)            # log-time scale
ate_mean    <- mean(ate_samples)
ate_ci      <- quantile(ate_samples, probs = c(0.025, 0.975))

p_ate <- ggplot(data.frame(ate = ate_samples), aes(x = ate)) +
  geom_density(fill = "steelblue", alpha = 0.4) +
  geom_vline(xintercept = ate_mean) +
  geom_vline(xintercept = ate_ci, linetype = "dashed") +
  labs(x = "ATE on log-time scale  (log time ratio)",
       y = "Posterior density",
       title = sprintf(
         "Posterior of ATE: %.3f  (95%% CrI: %.3f, %.3f)   |   exp(ATE) = %.2f",
         ate_mean, ate_ci[1], ate_ci[2], exp(ate_mean))) +
  theme_minimal()
print(p_ate)

# -------------------------------------------------------------------------
# 6. C-INDEX (in-sample, AFT-style — predicted log-time as the score)
# -------------------------------------------------------------------------
# pred = mu(X) + b * tau(X),  with b = z - 0.5 under centered coding.
pred_logT <- fit$train_predictions_control +
             (z - 0.5) * fit$train_predictions_treat

# reverse = TRUE because higher pred_logT means LONGER survival (not risk).
ci    <- concordance(Surv(time, stat) ~ pred_logT, reverse = TRUE)
c_est <- ci$concordance
c_se  <- sqrt(ci$var)

cat(sprintf("C-index (in-sample): %.3f  (SE %.3f, 95%% CI %.3f-%.3f)\n",
            c_est, c_se,
            c_est - 1.96 * c_se, c_est + 1.96 * c_se))

# -------------------------------------------------------------------------
# 7. SURVIVAL CURVES — model-implied marginal survival vs Kaplan-Meier
# -------------------------------------------------------------------------
# AFT-Gaussian: log(T) = mu(X) + b * tau(X) + sigma * eps,  eps ~ N(0,1).
# S_z(t) = E_X[ 1 - Phi((log t - pred_z)/sigma_log) ].
mu_samp  <- fit$train_predictions_sample_control   # N_post x n, log-time
tau_post <- fit$train_predictions_sample_treat     # N_post x n, log-time
N_post   <- nrow(mu_samp)

thin   <- as.integer(seq(1, N_post, length.out = min(N_post, 200)))
M      <- length(thin)
t_grid <- seq(min(time), max(time), length.out = 100)
log_t  <- log(t_grid)

S1_mat <- matrix(NA_real_, M, length(t_grid))
S0_mat <- matrix(NA_real_, M, length(t_grid))
for (k in seq_len(M)) {
  d     <- thin[k]
  pred1 <- mu_samp[d, ] + 0.5 * tau_post[d, ]
  pred0 <- mu_samp[d, ] - 0.5 * tau_post[d, ]
  sig   <- sig_post[d]
  S1_mat[k, ] <- rowMeans(1 - pnorm(outer(log_t, pred1, "-") / sig))
  S0_mat[k, ] <- rowMeans(1 - pnorm(outer(log_t, pred0, "-") / sig))
}

mod_df <- rbind(
  data.frame(time = t_grid,
             surv = apply(S0_mat, 2, median),
             lo   = apply(S0_mat, 2, quantile, 0.025),
             hi   = apply(S0_mat, 2, quantile, 0.975),
             arm  = "ZDV only"),
  data.frame(time = t_grid,
             surv = apply(S1_mat, 2, median),
             lo   = apply(S1_mat, 2, quantile, 0.025),
             hi   = apply(S1_mat, 2, quantile, 0.975),
             arm  = "Combination")
)

km    <- survfit(Surv(time, stat) ~ z)
km_df <- data.frame(
  time = km$time,
  surv = km$surv,
  arm  = rep(c("ZDV only", "Combination"), km$strata)
)

arm_cols <- c("ZDV only" = "tomato", "Combination" = "steelblue")

p_surv <- ggplot() +
  geom_ribbon(data = mod_df,
              aes(x = time, ymin = lo, ymax = hi, fill = arm),
              alpha = 0.15, color = NA) +
  geom_step(data = km_df,
            aes(x = time, y = surv, color = arm),
            linewidth = 1) +
  geom_line(data = mod_df,
            aes(x = time, y = surv, color = arm),
            linewidth = 0.8, linetype = "dashed") +
  scale_color_manual(values = arm_cols) +
  scale_fill_manual(values = arm_cols) +
  labs(x = "Days", y = "Survival probability",
       color = "Arm", fill = "Arm",
       title = "Model-implied marginal survival (dashed + 95% band) vs KM (solid)") +
  theme_minimal()
print(p_surv)

# -------------------------------------------------------------------------
# 8. MEAN log(T) DIFFERENCE AMONG EVENT-EXPERIENCERS
# -------------------------------------------------------------------------
# Empirical (biased toward 0 under unequal censoring): simple sample mean.
ev        <- stat == 1
diff_emp  <- mean(log_time[ev & z == 1]) - mean(log_time[ev & z == 0])
tt        <- t.test(log_time[ev] ~ z[ev])
cat(sprintf(
  "Events: n_treated = %d, n_control = %d\n",
  sum(ev & z == 1), sum(ev & z == 0)))
cat(sprintf(
  "Empirical mean log(T) diff (combo - ZDV) | events: %.3f  (95%% CI %.3f, %.3f)\n",
  diff_emp, tt$conf.int[1], tt$conf.int[2]))

# Model-based: posterior mean of CATE averaged over event-experiencers.
tau_ev_samp <- rowMeans(tau_post[, ev, drop = FALSE])
mean_diff   <- mean(tau_ev_samp)
mean_ci     <- quantile(tau_ev_samp, c(0.025, 0.975))
cat(sprintf(
  "Model-based mean tau on events:        %.3f  (95%% CrI %.3f, %.3f)\n",
  mean_diff, mean_ci[1], mean_ci[2]))
