set.seed(1)

# Right-censored data with known, unequal error scales per source.  The
# censoring scales with the source, so both sources lose about 60% of the
# events.
n_s   <- 300L
s_rct <- 0.5
s_rwd <- 1.5
src_s <- rep(c(1L, 0L), each = n_s / 2)
sd_s  <- ifelse(src_s == 1L, s_rct, s_rwd)
X_s   <- matrix(rnorm(n_s * 2), n_s, 2)
a_s   <- rbinom(n_s, 1, 0.5)
m_s   <- 1 + 0.5 * X_s[, 1]
log_t <- m_s + sd_s * rnorm(n_s)
log_c <- m_s + sd_s * rnorm(n_s, -0.25, 0.5)

test_that("censored times are augmented with the scale of their own source", {
  # With the pooled scale for every observation, the fitted RCT scale ends up
  # near 0.9 and the ratio of the two scales near 1.3 under this design.
  set.seed(1)
  fit <- FusionForest(
    y                         = pmin(log_t, log_c),
    status                    = as.integer(log_t <= log_c),
    X_train_control           = X_s,
    X_train_treat             = X_s,
    treatment_indicator_train = a_s,
    source_indicator_train    = src_s,
    outcome_type              = "right-censored",
    timescale                 = "log",
    error_dist                = "source_gaussian",
    N_post = 200L, N_burn = 200L, verbose = FALSE
  )
  sg_rwd <- mean(fit$dp_sigma_g[[1]])
  sg_rct <- mean(fit$dp_sigma_g[[2]])
  expect_lt(abs(sg_rct - s_rct), 0.2)
  expect_lt(abs(sg_rwd - s_rwd), 0.4)
  expect_gt(sg_rwd / sg_rct, 2)
})
