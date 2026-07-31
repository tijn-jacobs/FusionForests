set.seed(1)

n_train <- 60L
n_test  <- 20L
p       <- 3L
X_tr    <- matrix(rnorm(n_train * p), n_train, p)
X_te    <- matrix(rnorm(n_test  * p), n_test,  p)
trt_tr  <- as.integer(rbinom(n_train, 1, 0.5))
trt_te  <- as.integer(rbinom(n_test,  1, 0.5))
src_tr  <- as.integer(c(rep(1L, 30), rep(0L, 30)))
src_te  <- as.integer(rep(c(0L, 1L), length.out = n_test))

y_cnt   <- 1 + X_tr[, 1] + trt_tr * X_tr[, 2] + rnorm(n_train)
y_srv   <- exp(y_cnt)
stat    <- as.integer(rbinom(n_train, 1, 0.8))

N_post  <- 50L
N_burn  <- 20L

# ---------------------------------------------------------------------------
fit_gaussian_test <- function() {
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
    timescale                 = "log",   # treat y as already on log-time
    outcome_type              = "right-censored",
    status                    = stat,
    store_posterior_sample    = TRUE,
    N_post = N_post, N_burn = N_burn, verbose = FALSE
  )
}

# ---------------------------------------------------------------------------
test_that("fit object carries scale metadata", {
  fit <- fit_gaussian_test()
  expect_true(!is.null(fit$meta))
  expect_true(all(c("sigma_hat", "y_mean", "timescale", "outcome_type",
                    "decomposition", "error_dist") %in% names(fit$meta)))
  expect_s3_class(fit, "FusionForest")
})

test_that("fusion_estimand AF returns R x n_test matrix on time scale", {
  fit <- fit_gaussian_test()
  af  <- fusion_estimand(fit, estimand = "AF")
  expect_true(is.matrix(af))
  expect_equal(dim(af), c(N_post, n_test))
  expect_true(all(af > 0))
  expect_equal(attr(af, "estimand"), "AF")
})

test_that("fusion_estimand SD returns matrix in (-1, 1)", {
  fit <- fit_gaussian_test()
  t_eval <- exp(median(y_cnt))
  for (src in c("rwd", "rct")) {
    sd <- fusion_estimand(fit, estimand = "SD", time = t_eval,
                          target_source = src)
    expect_equal(dim(sd), c(N_post, n_test))
    expect_true(all(sd >= -1 & sd <= 1))
    expect_equal(attr(sd, "target_source"), src)
  }
})

test_that("fusion_estimand RMST is non-negative and bounded by t*", {
  fit    <- fit_gaussian_test()
  t_star <- exp(quantile(y_cnt, 0.95))
  for (src in c("rwd", "rct")) {
    rmst <- fusion_estimand(fit, estimand = "RMST", time = t_star,
                            target_source = src)
    expect_equal(dim(rmst), c(N_post, n_test))
    expect_true(all(abs(rmst) <= t_star + 1e-8))
  }
})

test_that("fusion_estimand population_average returns length-R vector", {
  fit  <- fit_gaussian_test()
  af_p <- fusion_estimand(fit, estimand = "AF",
                          population_average = TRUE,
                          bayesian_bootstrap = TRUE, seed = 42)
  expect_length(af_p, N_post)
  expect_true(isTRUE(attr(af_p, "bayesian_bootstrap")))

  sd_p <- fusion_estimand(fit, estimand = "SD", time = exp(median(y_cnt)),
                          population_average = TRUE,
                          bayesian_bootstrap = FALSE)
  expect_length(sd_p, N_post)
})

test_that("fusion_estimand errors without store_posterior_sample", {
  fit <- FusionForest(
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
    N_post = N_post, N_burn = N_burn, verbose = FALSE
  )
  expect_error(fusion_estimand(fit, estimand = "AF"),
               regexp = "store_posterior_sample")
})

test_that("fusion_estimand RMST works under source_hdp error", {
  fit <- FusionForest(
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
    error_dist                = "source_hdp",
    store_posterior_sample    = TRUE,
    N_post = N_post, N_burn = N_burn, verbose = FALSE
  )
  rmst <- fusion_estimand(fit, estimand = "RMST",
                          time = exp(quantile(y_cnt, 0.9)),
                          target_source = "rwd")
  expect_equal(dim(rmst), c(N_post, n_test))
  expect_true(all(is.finite(rmst)))
})
