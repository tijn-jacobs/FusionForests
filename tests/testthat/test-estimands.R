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
    error_dist                = "gaussian",
    store_posterior_sample    = TRUE,
    N_post = N_post, N_burn = N_burn, verbose = FALSE
  )
}

# ---------------------------------------------------------------------------
test_that("fit object carries scale metadata", {
  fit <- fit_gaussian_test()
  expect_true(!is.null(fit$meta))
  expect_true(all(c("sigma_hat", "y_mean", "timescale", "outcome_type",
                    "treatment_coding", "error_dist") %in% names(fit$meta)))
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

test_that("fusion_estimand SD works under source_hdp_scale error", {
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
    error_dist                = "source_hdp_scale",
    store_posterior_sample    = TRUE,
    N_post = N_post, N_burn = N_burn, verbose = FALSE
  )
  for (src in c("rwd", "rct")) {
    sd <- fusion_estimand(fit, estimand = "SD",
                          time = exp(quantile(y_cnt, 0.5)),
                          target_source = src)
    expect_equal(dim(sd), c(N_post, n_test))
    expect_true(all(is.finite(sd) & sd >= -1 & sd <= 1))
  }
})

test_that("arm linear predictor matches the fitted values for every coding", {
  # All test rows are treated RWD rows, so the backend's own fitted mean
  # equals the treated-arm linear predictor in target source s_t = 0.
  pi_tr <- rep(0.5, n_train)
  pi_te <- rep(0.4, n_test)
  for (coding in c("centered", "binary", "adaptive")) {
    fit <- FusionForest(
      y                         = y_cnt,
      X_train_control           = X_tr,
      X_train_treat             = X_tr,
      treatment_indicator_train = trt_tr,
      source_indicator_train    = src_tr,
      X_test_control            = X_te,
      X_test_treat              = X_te,
      treatment_indicator_test  = rep(1L, n_test),
      source_indicator_test     = rep(0L, n_test),
      treatment_coding          = coding,
      propensity_train          = pi_tr,
      propensity_test           = pi_te,
      store_posterior_sample    = TRUE,
      N_post = N_post, N_burn = N_burn, verbose = FALSE
    )
    comp <- FusionForests:::.log_scale_components(fit, fit$meta, "test")
    eta1 <- FusionForests:::.linear_predictor(comp, fit$meta, a = 1, s_t = 0L)
    expect_equal(colMeans(eta1), fit$test_predictions, tolerance = 1e-8,
                 info = coding)
  }
})

test_that("fusion_estimand errors when the treatment coding is unknown", {
  fit <- fit_gaussian_test()
  fit$meta$treatment_coding <- NULL
  expect_error(fusion_estimand(fit, estimand = "SD",
                               time = exp(median(y_cnt))),
               regexp = "treatment_coding")
})

test_that("fusion_estimand SD matches the closed form for a Gaussian fit", {
  fit    <- fit_gaussian_test()
  t_eval <- exp(median(y_cnt))
  comp   <- FusionForests:::.log_scale_components(fit, fit$meta, "test")
  sigma  <- fit$sigma * fit$meta$sigma_hat
  for (src in c("rwd", "rct")) {
    s_t  <- if (src == "rct") 1L else 0L
    eta0 <- FusionForests:::.causal_linear_predictor(comp, fit$meta, 0, s_t)
    eta1 <- FusionForests:::.causal_linear_predictor(comp, fit$meta, 1, s_t)
    surv0 <- 1 - pnorm((log(t_eval) - eta0) / sigma)
    surv1 <- 1 - pnorm((log(t_eval) - eta1) / sigma)
    sd <- fusion_estimand(fit, estimand = "SD", time = t_eval,
                          target_source = src)
    expect_equal(unclass(sd)[, ], surv1 - surv0, tolerance = 1e-10,
                 ignore_attr = TRUE)
  }
})

test_that("fusion_estimand SD uses the per-source scale of the target", {
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
    error_dist                = "source_hdp_scale",
    store_posterior_sample    = TRUE,
    N_post = N_post, N_burn = N_burn, verbose = FALSE
  )
  t_eval <- exp(median(y_cnt))
  comp   <- FusionForests:::.log_scale_components(fit, fit$meta, "test")
  for (s_t in 0:1) {
    g     <- s_t + 1L                      # group 1 = RWD, group 2 = RCT
    eta0  <- FusionForests:::.causal_linear_predictor(comp, fit$meta, 0, s_t)
    eta1  <- FusionForests:::.causal_linear_predictor(comp, fit$meta, 1, s_t)
    pi_m  <- fit$dp_mix_prop[[g]]
    theta <- fit$dp_locations[[g]]
    sig   <- fit$dp_sigma_g[[g]]
    manual <- matrix(0, N_post, n_test)
    for (k in seq_len(ncol(pi_m))) {
      z0 <- (log(t_eval) - eta0 - theta[, k]) / sig
      z1 <- (log(t_eval) - eta1 - theta[, k]) / sig
      manual <- manual + pi_m[, k] * (pnorm(z0) - pnorm(z1))
    }
    sd <- fusion_estimand(fit, estimand = "SD", time = t_eval,
                          target_source = if (s_t == 1L) "rct" else "rwd")
    expect_equal(unclass(sd)[, ], manual, tolerance = 1e-10,
                 ignore_attr = TRUE)
  }
})

test_that("fusion_estimand AF does not depend on the target source", {
  fit <- fit_gaussian_test()
  af_rwd <- fusion_estimand(fit, estimand = "AF", target_source = "rwd")
  af_rct <- fusion_estimand(fit, estimand = "AF", target_source = "rct")
  expect_equal(unclass(af_rwd)[, ], unclass(af_rct)[, ],
               ignore_attr = TRUE)
})

test_that("equal-weight population average equals the row means", {
  fit <- fit_gaussian_test()
  af  <- fusion_estimand(fit, estimand = "AF")
  af_p <- fusion_estimand(fit, estimand = "AF", population_average = TRUE,
                          bayesian_bootstrap = FALSE)
  expect_equal(as.numeric(af_p), rowMeans(af))
})

test_that("fusion_estimand works under adaptive treatment coding", {
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
    treatment_coding          = "adaptive",
    propensity_train          = rep(0.5, n_train),
    propensity_test           = rep(0.5, n_test),
    store_posterior_sample    = TRUE,
    N_post = N_post, N_burn = N_burn, verbose = FALSE
  )
  expect_equal(fit$meta$propensity_test, rep(0.5, n_test))
  sd <- fusion_estimand(fit, estimand = "SD", time = exp(median(y_cnt)))
  expect_equal(dim(sd), c(N_post, n_test))
  expect_true(all(is.finite(sd) & sd >= -1 & sd <= 1))
})

test_that("fusion_estimand validates time and estimand", {
  fit <- fit_gaussian_test()
  expect_error(fusion_estimand(fit, estimand = "SD"), "'time' is required")
  expect_error(fusion_estimand(fit, estimand = "SD", time = -1),
               "single positive")
  expect_error(fusion_estimand(fit, estimand = "RMST", time = 1))
})

test_that("fusion_estimand SD uses the source scale under source_gaussian", {
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
    error_dist                = "source_gaussian",
    store_posterior_sample    = TRUE,
    N_post = N_post, N_burn = N_burn, verbose = FALSE
  )
  t_eval <- exp(median(y_cnt))
  comp   <- FusionForests:::.log_scale_components(fit, fit$meta, "test")
  for (s_t in 0:1) {
    sig   <- fit$dp_sigma_g[[s_t + 1L]]
    eta0  <- FusionForests:::.causal_linear_predictor(comp, fit$meta, 0, s_t)
    eta1  <- FusionForests:::.causal_linear_predictor(comp, fit$meta, 1, s_t)
    manual <- pnorm((log(t_eval) - eta0) / sig) -
      pnorm((log(t_eval) - eta1) / sig)
    sd <- fusion_estimand(fit, estimand = "SD", time = t_eval,
                          target_source = if (s_t == 1L) "rct" else "rwd")
    expect_equal(unclass(sd)[, ], manual, tolerance = 1e-10,
                 ignore_attr = TRUE)
  }
})

test_that("the causal predictor drops c(x) and keeps the untreated mean", {
  for (coding in c("centered", "binary")) {
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
      treatment_coding          = coding,
      store_posterior_sample    = TRUE,
      N_post = N_post, N_burn = N_burn, verbose = FALSE
    )
    comp <- FusionForests:::.log_scale_components(fit, fit$meta, "test")
    lp   <- FusionForests:::.linear_predictor
    clp  <- FusionForests:::.causal_linear_predictor
    for (s_t in 0:1) {
      # The untreated arm is the observed untreated mean of the source.
      expect_equal(clp(comp, fit$meta, 0, s_t), lp(comp, fit$meta, 0, s_t),
                   info = coding)
      # The treated arm adds tau only, without c.
      expect_equal(clp(comp, fit$meta, 1, s_t) - clp(comp, fit$meta, 0, s_t),
                   comp$tau, info = coding)
    }
    # In the RCT the causal and observed treated means coincide.
    expect_equal(clp(comp, fit$meta, 1, 1L), lp(comp, fit$meta, 1, 1L),
                 info = coding)
  }
})
