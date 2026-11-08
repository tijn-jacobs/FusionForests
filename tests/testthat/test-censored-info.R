test_that("the base R fit reduces to the closed form when every row is an event", {
  set.seed(1)
  y    <- rnorm(500, 2, 1.5)
  info <- censored_normal_fit(y, y)
  expect_equal(info$mu, mean(y), tolerance = 1e-4)
  expect_equal(info$sd, sqrt(mean((y - mean(y))^2)), tolerance = 1e-4)
})

test_that("interval-censored rows enter the preliminary fit through their bounds", {
  # Half exact events, half interval-censored at unit-width visits.
  set.seed(1)
  n      <- 3000L
  log_t  <- rnorm(n, 2, 1.5)
  exact  <- seq_len(n) <= n / 2
  left   <- floor(log_t)
  right  <- left + 1
  status <- as.integer(exact)
  ic     <- as.integer(!exact)
  y      <- ifelse(exact, log_t, right)

  info <- censored_info(y, status, left, right, ic)
  expect_equal(info$mu, 2, tolerance = 0.05)
  expect_equal(info$sd, 1.5, tolerance = 0.05)

  # Reading the same rows as right-censored at y overstates the mean.
  naive <- censored_info(y, status)
  expect_gt(naive$mu, info$mu + 0.2)
})

test_that("right-censored rows use right_time and match the plain fit", {
  set.seed(1)
  n      <- 1000L
  log_t  <- rnorm(n, 1, 1)
  log_c  <- rnorm(n, 1.5, 1)
  obs    <- pmin(log_t, log_c)
  status <- as.integer(log_t <= log_c)
  plain  <- censored_info(obs, status)

  # y is only a placeholder for the right-censored rows.
  y    <- ifelse(status == 1, obs, 0)
  info <- censored_info(y, status, obs, obs, rep(0L, n))
  expect_equal(info$mu, plain$mu, tolerance = 1e-3)
  expect_equal(info$sd, plain$sd, tolerance = 1e-3)
})

test_that("infinite bounds are read as open ends", {
  set.seed(1)
  n     <- 1000L
  log_t <- rnorm(n)
  left  <- c(-Inf, -1, 1)[findInterval(log_t, c(-1, 1)) + 1L]
  right <- c(-1, 1, Inf)[findInterval(log_t, c(-1, 1)) + 1L]
  info  <- censored_info(rep(0, n), rep(0L, n), left, right, rep(1L, n))
  expect_equal(info$mu, 0, tolerance = 0.2)
  expect_equal(info$sd, 1, tolerance = 0.2)
})

test_that("data without events or finite intervals give finite values", {
  info <- censored_info(c(1, 2, 3), rep(0L, 3))
  expect_true(is.finite(info$mu) && is.finite(info$sd) && info$sd > 0)
})

test_that("the base R fit agrees with survreg", {
  skip_if_not_installed("survival")
  set.seed(1)
  n     <- 800L
  log_t <- rnorm(n, 1, 1.2)
  # Events, right-censored rows and interval-censored rows.
  type  <- sample(c("event", "right", "interval"), n, replace = TRUE)
  log_c <- log_t - rexp(n)
  lo    <- ifelse(type == "interval", floor(log_t), log_t)
  lo    <- ifelse(type == "right", log_c, lo)
  hi    <- ifelse(type == "interval", floor(log_t) + 1, log_t)
  hi    <- ifelse(type == "right", Inf, hi)

  info <- censored_normal_fit(lo, hi)
  ref  <- survival::survreg(
    survival::Surv(lo, ifelse(is.finite(hi), hi, NA), type = "interval2") ~ 1,
    dist = "gaussian")
  expect_equal(info$mu, as.numeric(ref$coefficients[1]), tolerance = 1e-3)
  expect_equal(info$sd, as.numeric(ref$scale), tolerance = 1e-3)

  # The plain right-censored path.
  y      <- pmin(log_t, log_t + rnorm(n))
  st     <- as.integer(y == log_t)
  info_r <- censored_normal_fit(y, ifelse(st == 1, y, Inf))
  ref_r  <- survival::survreg(survival::Surv(y, st) ~ 1, dist = "gaussian")
  expect_equal(info_r$mu, as.numeric(ref_r$coefficients[1]), tolerance = 1e-3)
  expect_equal(info_r$sd, as.numeric(ref_r$scale), tolerance = 1e-3)
})
