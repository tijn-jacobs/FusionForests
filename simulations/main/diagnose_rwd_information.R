################################################################################
# DIAGNOSTIC: why is RCT-only beating fusion in the main sim?
# Hypothesis: equal-width interval censoring 0..max(T) on heavy-tailed event
# times collapses almost all RWD events into bin 1, so the RWD is nearly
# uninformative. We compare fusion under three RWD observation schemes against
# RCT-only, on the mild-confounding cell S1 (where fusion should help).
################################################################################
.libPaths(c("~/Rlib", .libPaths()))
suppressMessages(library(FusionForests))

p <- 10L; n1 <- 150L; n0 <- 350L; n_bins <- 10L
NP <- 1500L; NB <- 1500L
args <- commandArgs(trailingOnly = TRUE)
scen_nm <- if (length(args) >= 1) args[1] else "S1"
regime  <- if (length(args) >= 2) args[2] else "heterogeneous"
scenarios <- list(S0=c(alpha_U=0,beta_U=1,gamma_U=0), S1=c(alpha_U=.5,beta_U=1,gamma_U=0),
                  S2=c(alpha_U=1.5,beta_U=1.5,gamma_U=0), S3=c(alpha_U=1,beta_U=1,gamma_U=.5))
scn <- scenarios[[scen_nm]]
lambda_d <- if (regime == "heterogeneous") 0.5 else 0.0; gamma_E <- 0.5772156649
cat(sprintf("CELL: %s / %s\n", scen_nm, regime))

true_tau <- function(X) 1 + 0.5 * X[, 1] + (1/3) * (X[, 2] > 0)
m0_sh    <- function(X) sin(pi * X[, 1]) + X[, 2] * X[, 3] + (X[, 4]^2 - 1/12)
g_dev    <- function(X) X[, 1] - 0.5 * X[, 5]
slogT <- function(X, A, S, U) m0_sh(X) + (1-S)*lambda_d*g_dev(X) +
  scn["beta_U"]*U + A*(true_tau(X) + scn["gamma_U"]*U)
eps_rct <- function(n) if (regime=="heterogeneous") rnorm(n,0,0.6) else rnorm(n,0,1)
eps_rwd <- function(n) if (regime=="heterogeneous") (1.0*sqrt(6)/pi)*(log(rexp(n))+gamma_E) else rnorm(n,0,1)

X1 <- matrix(runif(n1*p,-.5,.5),n1,p); U1 <- rnorm(n1); A1 <- rbinom(n1,1,.5)
lT1 <- slogT(X1,A1,1,U1) + eps_rct(n1)
a0 <- 0
X0 <- matrix(runif(n0*p,-.5,.5),n0,p); U0 <- rnorm(n0)
A0 <- rbinom(n0,1,plogis(a0 + X0[,1] + X0[,2] + scn["alpha_U"]*U0))
lT0 <- slogT(X0,A0,0,U0) + eps_rwd(n0)
T0 <- exp(lT0); T1 <- exp(lT1)
vmax <- as.numeric(quantile(c(T1,T0), 0.999)); tRCT <- vmax/3

# RCT right+admin censoring
C1 <- rexp(n1, rate = 0.14); obs1 <- pmin(T1,C1,tRCT); st1 <- as.integer(T1<=pmin(C1,tRCT))

X <- rbind(X1,X0); A <- c(A1,A0); S <- c(rep(1L,n1),rep(0L,n0)); truth <- true_tau(X)

# --- bin occupancy: equal-width vs quantile grid ---
eqbrks <- seq(0, vmax, length.out = n_bins+1L)
qbrks  <- unique(c(0, quantile(T0, probs = seq(1,n_bins-1)/n_bins), vmax))
cat(sprintf("vmax(99.9%%)=%.1f  median T_rwd=%.2f  tRCT=%.1f\n", vmax, median(T0), tRCT))
cat("Equal-width bin counts (RWD):\n"); print(as.integer(table(cut(T0, eqbrks, include.lowest=TRUE))))
cat(sprintf("  -> %.0f%% of RWD events in bin 1\n", 100*mean(T0 <= eqbrks[2])))
cat("Quantile bin counts (RWD):\n"); print(as.integer(table(cut(T0, qbrks, include.lowest=TRUE))))

logfloor <- min(c(lT1,lT0)) - 5
make_ic <- function(brks){
  j <- findInterval(T0, brks, rightmost.closed=TRUE)
  j <- pmin(pmax(j, 1L), length(brks) - 1L)
  list(left=ifelse(j==1, logfloor, log(brks[j])), right=log(pmin(brks[j+1], vmax)),
       icc=rep(1L,n0), st=rep(0L,n0), y=log(pmin(brks[j+1], vmax)))
}

fit_fusion <- function(y,st,icc,left,right){
  f <- FusionForest(y=y,status=st,observed_left_time=left,observed_right_time=right,
    interval_censoring_indicator=icc, X_train_control=X,X_train_treat=X,
    treatment_indicator_train=A, source_indicator_train=S,
    X_test_control=X,X_test_treat=X,X_test_deconf=X,
    treatment_indicator_test=rep(1L,nrow(X)), source_indicator_test=rep(1L,nrow(X)),
    outcome_type="right-censored", timescale="log", decomposition="four-forest",
    number_of_trees_control=200,number_of_trees_treat=100,
    number_of_trees_deconf=50,number_of_trees_deviation=50,
    k_control=.5,k_treat=.5,k_deconf=.5,k_g=.5,
    N_post=NP,N_burn=NB,treatment_coding="centered",
    error_dist="source_hdp_scale",store_posterior_sample=TRUE,verbose=FALSE)
  sqrt(mean((f$test_predictions_treat - truth)^2))
}
fit_rct <- function(){
  src <- rep(1L,n1); src[n1] <- 0L
  f <- FusionForest(y=log(obs1),status=st1, X_train_control=X1,X_train_treat=X1,
    treatment_indicator_train=A1, source_indicator_train=src,
    X_test_control=X,X_test_treat=X, treatment_indicator_test=rep(1L,nrow(X)),
    source_indicator_test=rep(1L,nrow(X)), outcome_type="right-censored",
    timescale="log", decomposition="four-forest",
    number_of_trees_control=200,number_of_trees_treat=100,
    number_of_trees_deconf=1,number_of_trees_deviation=1,
    k_control=.5,k_treat=.5, N_post=NP,N_burn=NB,treatment_coding="centered",
    error_dist="gaussian",store_posterior_sample=TRUE,verbose=FALSE)
  sqrt(mean((f$test_predictions_treat - truth)^2))
}

# common RCT block for fusion fits
y_rct <- log(obs1); st_rct <- st1
ic_eq <- make_ic(eqbrks); ic_q <- make_ic(qbrks)

cat("\nFitting...\n")
r_rct  <- fit_rct()
r_eq   <- fit_fusion(c(y_rct, ic_eq$y), c(st_rct, ic_eq$st), c(rep(0L,n1), ic_eq$icc),
                     c(y_rct, ic_eq$left), c(y_rct, ic_eq$right))
r_q    <- fit_fusion(c(y_rct, ic_q$y), c(st_rct, ic_q$st), c(rep(0L,n1), ic_q$icc),
                     c(y_rct, ic_q$left), c(y_rct, ic_q$right))
# continuous RWD (fully observed)
r_cont <- fit_fusion(c(y_rct, lT0), c(st_rct, rep(1L,n0)), c(rep(0L,n1), rep(0L,n0)),
                     c(y_rct, lT0), c(y_rct, lT0))

cat("\n===== CATE RMSE (S1 heterogeneous, mild confounding) =====\n")
cat(sprintf("  RCT-only                         : %.4f\n", r_rct))
cat(sprintf("  Fusion, RWD interval (equal-width): %.4f\n", r_eq))
cat(sprintf("  Fusion, RWD interval (quantile)   : %.4f\n", r_q))
cat(sprintf("  Fusion, RWD fully observed        : %.4f\n", r_cont))
