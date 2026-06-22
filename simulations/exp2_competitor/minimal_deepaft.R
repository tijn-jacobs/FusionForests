# Minimal deepAFT example -- flat, no functions.  Run line by line.

library(dnn)
library(survival)

set.seed(1)

# ── very easy DGP ──────────────────────────────────────────────────────────────
n <- 200
x1 <- rnorm(n)
x2 <- rnorm(n)
A  <- rbinom(n, 1, 0.5)

tau  <- 1 + x1                       # true CATE
logT <- x1 + A * tau + rnorm(n)      # log survival time
T    <- exp(logT)                    # event time

C      <- rexp(n, rate = 1 / quantile(T, 0.7))  # censoring time
obs    <- pmin(T, C)                            # observed time
status <- as.integer(T <= C)                    # 1 = event, 0 = censored

# covariate matrix: x1, x2, treatment
Z <- cbind(x1, x2, A)

# ── fit deepAFT ────────────────────────────────────────────────────────────────
# NOTE: identity output activation is "idu" (NOT "linear").
model <- dNNmodel(units       = c(8, 8, 1),
                  activation  = c("elu", "elu", "idu"),
                  input_shape = ncol(Z))

fit <- deepAFT(Surv(obs, status) ~ Z, model = model)

# ── predict counterfactuals -> CATE ────────────────────────────────────────────
Z1 <- cbind(x1, x2, A = 1)
Z0 <- cbind(x1, x2, A = 0)

pred1 <- predict(fit, newdata = Z1)
pred0 <- predict(fit, newdata = Z0)

str(pred1)                           # predict() returns a list; we want $predictors

cate <- pred1$predictors - pred0$predictors

# ── check against truth ───────────────────────────────────────────────────────
plot(tau, cate); abline(0, 1, col = "red")
cat("RMSE:", sqrt(mean((cate - tau)^2)), " bias:", mean(cate - tau), "\n")
