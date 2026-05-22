set.seed(1)

n_train <- 60L
n_test  <- 30L
p       <- 3L
X_tr    <- matrix(rnorm(n_train * p), n_train, p)
X_te    <- matrix(rnorm(n_test  * p), n_test,  p)
trt_tr  <- as.integer(rbinom(n_train, 1, 0.5))
trt_te  <- as.integer(rbinom(n_test,  1, 0.5))
src_tr  <- as.integer(c(rep(1L, 30), rep(0L, 30)))
src_te  <- as.integer(rep(c(0L, 1L), length.out = n_test))

y_cnt <- 1 + X_tr[, 1] + trt_tr * (X_tr[, 1] + 0.5 * X_tr[, 2]) +
         rnorm(n_train, sd = 0.5)
stat  <- as.integer(rbinom(n_train, 1, 0.8))

N_post <- 50L
N_burn <- 20L

X_te_df <- data.frame(x1 = X_te[, 1], x2 = X_te[, 2], x3 = X_te[, 3])
X_tr_df <- data.frame(x1 = X_tr[, 1], x2 = X_tr[, 2], x3 = X_tr[, 3])

fit_for_projection <- function() {
  FusionForest(
    y                         = y_cnt,
    X_train_control           = X_tr,
    X_train_treat             = X_tr,
    treatment_indicator_train = trt_tr,
    source_indicator_train    = src_tr,
    X_test_control            = X_te,
    X_test_treat              = X_te,
    treatment_indicator_test  = trt_te,
    source_indicator_test     = src_te,
    timescale                 = "log",
    outcome_type              = "right-censored",
    status                    = stat,
    store_posterior_sample    = TRUE,
    N_post = N_post, N_burn = N_burn, verbose = FALSE
  )
}

# ---------------------------------------------------------------------------
test_that("fusion_projection returns an R x k matrix with right attributes", {
  fit <- fit_for_projection()
  g <- fusion_projection(fit, ~ x1 + x2, X_eval = X_te_df, target = "cate")
  expect_true(is.matrix(g))
  expect_equal(dim(g), c(N_post, 3L))                    # intercept + x1 + x2
  expect_equal(colnames(g), c("(Intercept)", "x1", "x2"))
  expect_equal(attr(g, "target"),  "cate")
  expect_equal(attr(g, "weights"), "uniform")
  expect_equal(attr(g, "which"),   "test")
  expect_true(is.na(attr(g, "af_scale")))
  expect_true(!is.null(attr(g, "center")))               # centring on by default
})

test_that("af_scale = 'log' is identical to target = 'cate'", {
  fit <- fit_for_projection()
  g_cate <- fusion_projection(fit, ~ x1 + x2, X_eval = X_te_df,
                              target = "cate")
  g_af_log <- fusion_projection(fit, ~ x1 + x2, X_eval = X_te_df,
                                target = "af", af_scale = "log")
  attributes(g_cate) <- NULL
  attributes(g_af_log) <- NULL
  expect_equal(g_af_log, g_cate)
})

test_that("af_scale = 'multiplicative' differs from cate", {
  fit <- fit_for_projection()
  g_cate <- fusion_projection(fit, ~ x1 + x2, X_eval = X_te_df,
                              target = "cate")
  g_af <- fusion_projection(fit, ~ x1 + x2, X_eval = X_te_df,
                            target = "af", af_scale = "multiplicative")
  expect_equal(attr(g_af, "af_scale"), "multiplicative")
  # Different functional => different projection coefficients.
  expect_false(isTRUE(all.equal(unname(g_af), unname(g_cate),
                                check.attributes = FALSE)))
})

test_that("uniform projection is invariant to row reordering", {
  fit <- fit_for_projection()
  ord <- sample(seq_len(n_test))

  # Reorder both X_eval and the corresponding tau columns identically.
  fit_perm <- fit
  fit_perm$test_predictions_sample_treat <-
    fit$test_predictions_sample_treat[, ord, drop = FALSE]
  fit_perm$test_predictions_sample_control <-
    fit$test_predictions_sample_control[, ord, drop = FALSE]
  fit_perm$test_predictions_sample_deconf <-
    fit$test_predictions_sample_deconf[, ord, drop = FALSE]
  if (!is.null(fit$test_predictions_sample_deviation))
    fit_perm$test_predictions_sample_deviation <-
      fit$test_predictions_sample_deviation[, ord, drop = FALSE]

  g1 <- fusion_projection(fit,      ~ x1 + x2, X_eval = X_te_df)
  g2 <- fusion_projection(fit_perm, ~ x1 + x2, X_eval = X_te_df[ord, ])
  attributes(g1) <- attributes(g2) <- NULL
  expect_equal(g1, g2, tolerance = 1e-10)
})

test_that("Bayesian bootstrap is reproducible under fixed seed", {
  fit <- fit_for_projection()
  g_a <- fusion_projection(fit, ~ x1 + x2, X_eval = X_te_df,
                           weights = "bayesian_bootstrap", seed = 1L)
  g_b <- fusion_projection(fit, ~ x1 + x2, X_eval = X_te_df,
                           weights = "bayesian_bootstrap", seed = 1L)
  g_c <- fusion_projection(fit, ~ x1 + x2, X_eval = X_te_df,
                           weights = "bayesian_bootstrap", seed = 2L)
  attributes(g_a) <- attributes(g_b) <- attributes(g_c) <- NULL
  expect_equal(g_a, g_b)
  expect_false(isTRUE(all.equal(g_a, g_c)))
})

test_that("which = 'train' works and returns correct shape", {
  fit <- fit_for_projection()
  g <- fusion_projection(fit, ~ x1 + x2, X_eval = X_tr_df, which = "train")
  expect_equal(dim(g), c(N_post, 3L))
  expect_equal(attr(g, "which"), "train")
})

test_that("Bad X_eval row count throws an informative error", {
  fit <- fit_for_projection()
  expect_error(
    fusion_projection(fit, ~ x1 + x2, X_eval = X_te_df[seq_len(5), ]),
    regexp = "nrow\\(X_eval\\)"
  )
})

test_that("Recovery: projection mean is close to OLS-on-truth", {
  # Sanity check: under no MCMC noise the projection collapses to OLS of tau
  # on Phi.  With short chains we expect the posterior mean of gamma to be
  # close to the OLS fit of the posterior mean of tau on Phi.
  fit <- fit_for_projection()
  g <- fusion_projection(fit, ~ x1 + x2, X_eval = X_te_df, center = FALSE)
  tau_mean <- colMeans(fit$test_predictions_sample_treat)
  ols_ref  <- coef(lm(tau_mean ~ X_te_df$x1 + X_te_df$x2))
  expect_equal(unname(colMeans(g)), unname(ols_ref), tolerance = 1e-8)
})
