set.seed(1)

n   <- 60
X   <- matrix(rnorm(n * 3), n, 3)
trt <- as.integer(rbinom(n, 1, 0.5))
src <- as.integer(rep(c(1L, 0L), each = n / 2))
y   <- 1 + X[, 1] + trt * X[, 2] + rnorm(n)

fit <- FusionForest(
  y                         = y,
  X_train_control           = X,
  X_train_treat             = X,
  treatment_indicator_train = trt,
  source_indicator_train    = src,
  N_post = 50L, N_burn = 20L, verbose = FALSE
)

# ---------------------------------------------------------------------------
test_that("print.FusionForest shows the model settings and returns invisibly", {
  out <- capture.output(res <- print(fit))
  expect_identical(res, fit)
  expect_true(any(grepl("FusionForest fit", out)))
  expect_true(any(grepl("Treatment coding: centered", out)))
  expect_true(any(grepl("Error model:\\s+source_hdp_scale", out)))
  expect_true(any(grepl("RCT: 30, RWD: 30", out)))
  expect_true(any(grepl("Posterior draws: 50 \\(burn-in: 20\\)", out)))
})

test_that("summary.FusionForest returns the documented components", {
  s <- summary(fit)
  expect_s3_class(s, "summary.FusionForest")
  expect_named(s, c("meta", "sigma", "treatment_effects",
                    "acceptance_ratios"))
  expect_length(s$treatment_effects, n)
  expect_named(s$acceptance_ratios,
               c("control", "treat", "deconf", "deviation"))
  expect_true(all(s$acceptance_ratios >= 0 & s$acceptance_ratios <= 1))
})

test_that("print.summary.FusionForest prints every section", {
  out <- capture.output(res <- print(summary(fit)))
  expect_s3_class(res, "summary.FusionForest")
  expect_true(any(grepl("Summary of FusionForest fit", out)))
  expect_true(any(grepl("Residual standard deviation", out)))
  expect_true(any(grepl("individual treatment effects", out)))
  expect_true(any(grepl("acceptance ratios", out)))
})
