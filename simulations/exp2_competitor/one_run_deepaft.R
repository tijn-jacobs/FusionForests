# One single deepAFT run -- generate data, fit, predict.  Run line by line.

library(dnn)
library(survival)
library(MASS)
library(evd)

set.seed(1)

# ── one dataset (same DGP style as the main sim, RCT-like, right-censored) ─────
n       <- 300
p       <- 10
rho_X   <- 0.3
Sigma_X <- rho_X ^ abs(outer(seq_len(p), seq_len(p), "-"))

X <- mvrnorm(n, mu = rep(0, p), Sigma = Sigma_X)
A <- rbinom(n, 1L, 0.5)

m0  <- 2 * X[, 1] - X[, 2] * X[, 3] + 0.5 * X[, 4]^2
tau <- 1/2 + X[, 1] - 0.5 * X[, 2]^2          # true CATE
logT <- m0 + A * tau + rnorm(n, 0, 0.75)

# standardise log-survival, then exp() back to a positive time for Surv()
logT_std <- (logT - mean(logT)) / sd(logT)
scale_sd <- sd(logT)

time   <- exp(logT_std)                            # positive event time
ctime  <- rexp(n, rate = 1 / quantile(time, 0.7))  # random censoring time
obs    <- pmin(time, ctime)                        # observed time
status <- as.integer(time <= ctime)                # 1 = event, 0 = censored

# ── fit deepAFT (treatment as a covariate) ────────────────────────────────────
Z <- cbind(X, A = A)

model <- dNNmodel(units       = c(16, 16, 1),
                  activation  = c("elu", "elu", "linear"),
                  input_shape = ncol(Z))

fit <- deepAFT(Surv(obs, status) ~ Z, model = model)

# ── predict counterfactuals and form the CATE ─────────────────────────────────
Z1 <- cbind(X, A = 1)
Z0 <- cbind(X, A = 0)

pred1 <- predict(fit, newdata = Z1)
pred0 <- predict(fit, newdata = Z0)

str(pred1)        # inspect what predict() returns

# adjust the next two lines once str() shows the right component
lp1 <- pred1$lp
lp0 <- pred0$lp

cate <- (lp1 - lp0) * scale_sd

# ── check against truth ───────────────────────────────────────────────────────
plot(tau, cate); abline(0, 1, col = "red")
cat("RMSE:", sqrt(mean((cate - tau)^2)), " bias:", mean(cate - tau), "\n")
