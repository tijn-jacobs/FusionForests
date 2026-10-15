# =============================================================================
# Build the semi-synthetic data sets breast_fusion and breast_fusion_truth.
#
# Real ingredients (from the survival package):
#   - covariates          : GBSG2 trial (gbsg, RCT) and Rotterdam tumour bank
#                           (rotterdam, RWD), harmonised to a common set
#   - RWD treatment       : the real (non-random) hormon assignment in
#                           Rotterdam, relabelled as a generic treatment A
#   - hidden confounders  : year of surgery and chemotherapy (Rotterdam only),
#                           which drive the real RWD treatment assignment and
#                           are left out of breast_fusion
#   - baseline + error    : coefficients and scale of a Weibull AFT fitted
#                           to the GBSG2 relapse-free survival data
#   - censoring           : drawn from each source's real censoring
#                           distribution (reverse Kaplan-Meier)
#
# Made-up ingredients:
#   - RCT treatment       : re-randomised, A ~ Bernoulli(1/2)
#   - effect modification : tau(x) = real average effect + made-up
#                           modification by pgr and grade
#   - hidden-confounder effects on the outcome (gamma_chemo, gamma_year)
#   - one non-linear baseline term (age < 40)
#
# The treatment is deliberately generic ("treated" / "control") so that the
# example implies no clinical claim about hormonal therapy.
#
# Anyone can source this file to rebuild the data.  The final line writes
# data/ and only makes sense from the package source root; it is commented
# out so that sourcing this file never overwrites the shipped data by
# accident.  The example analysis is in breast_fusion_analysis.R.
# =============================================================================

library(survival)

# ---- Configuration ---------------------------------------------------------
seed        <- 1L
gamma_chemo <- 0.30        # made up: log-time gain from chemotherapy
gamma_year  <- 0.04        # made up: log-time gain per calendar year
year_ref    <- 1986.5      # centre of the calendar-year effect
rct_years   <- 1984:1989   # GBSG2 recruitment period


# ---- 1. Harmonise ----------------------------------------------------------
# Common covariates: age, meno (0/1), size (0: <=20, 1: 20-50, 2: >50 mm),
# grade (0: grade 1-2, 1: grade 3), nodes, pgr, er.  Rotterdam has no
# grade 1, so grades 1 and 2 are merged.  Times are in years.
# Rotterdam relapse-free survival follows ?rotterdam: relapse time if
# relapsed, otherwise death/last follow-up time; event = relapse or death.

harmonise_rct <- function(d) {
  data.frame(
    age    = d$age,
    meno   = as.integer(d$meno),
    size   = as.integer(cut(d$size, c(-Inf, 20, 50, Inf))) - 1L,
    grade  = as.integer(d$grade == 3),
    nodes  = d$nodes,
    pgr    = d$pgr,
    er     = d$er,
    A_real = as.integer(d$hormon),
    chemo  = 1L,                      # every GBSG2 patient had chemotherapy
    year   = NA_real_,                # not recorded; drawn below
    time   = d$rfstime / 365.25,
    event  = as.integer(d$status)
  )
}

harmonise_os <- function(d) {
  data.frame(
    age    = d$age,
    meno   = as.integer(d$meno),
    size   = as.integer(d$size) - 1L, # factor levels <=20, 20-50, >50
    grade  = as.integer(d$grade == 3),
    nodes  = d$nodes,
    pgr    = d$pgr,
    er     = d$er,
    A_real = as.integer(d$hormon),
    chemo  = as.integer(d$chemo),
    year   = d$year,
    time   = ifelse(d$recur == 1, d$rtime, d$dtime) / 365.25,
    event  = as.integer(pmax(d$recur, d$death))
  )
}

rct <- harmonise_rct(survival::gbsg)
os  <- harmonise_os(survival::rotterdam)

set.seed(seed)
rct$year <- sample(rct_years, nrow(rct), replace = TRUE)

x_names <- c("age", "meno", "size", "grade", "nodes", "pgr", "er")


# ---- 2. Reference model and data-generating process ------------------------
# Weibull AFT on the real GBSG2 outcomes.  Its covariate coefficients give
# the baseline m0(x), its treatment coefficient the average effect, and its
# scale the error scale.  All are on the log-time scale.
form_x   <- ~ age + meno + size + grade + log1p(nodes) + log1p(pgr) +
  log1p(er)
ref_fit  <- survreg(
  update(form_x, Surv(time, event) ~ . + A_real),
  data = rct, dist = "weibull"
)
beta_x   <- coef(ref_fit)[colnames(model.matrix(form_x, rct))]
beta_A   <- unname(coef(ref_fit)["A_real"])
sigma_W  <- ref_fit$scale

# Standardisation constants for the effect modifier, from the pooled data.
lpgr_all <- log1p(c(rct$pgr, os$pgr))
pgr_mu   <- mean(lpgr_all)
pgr_sd   <- sd(lpgr_all)

m0_fn <- function(d) {
  drop(model.matrix(form_x, d) %*% beta_x) -
    0.25 * (d$age < 40)                       # made-up non-linearity
}

tau_fn <- function(d) {
  z_pgr <- (log1p(d$pgr) - pgr_mu) / pgr_sd
  beta_A + 0.25 * z_pgr - 0.20 * d$grade      # made-up modification
}

hidden_fn <- function(d) {
  gamma_chemo * (d$chemo - 1) + gamma_year * (d$year - year_ref)
}

# Draw censoring times from the reverse Kaplan-Meier of the real data.
# Mass beyond the last censoring time is placed at that time.
draw_censoring <- function(time, event, n) {
  km   <- survfit(Surv(time, 1 - event) ~ 1)
  F_c  <- 1 - km$surv
  idx  <- findInterval(runif(n), F_c, left.open = TRUE) + 1L
  idx  <- pmin(idx, length(km$time))
  km$time[idx]
}

simulate_source <- function(d, A) {
  n      <- nrow(d)
  tau    <- tau_fn(d)
  logT   <- m0_fn(d) + hidden_fn(d) + A * tau +
    sigma_W * log(rexp(n))                    # Weibull (extreme-value) error
  C      <- draw_censoring(d$time, d$event, n)
  T_true <- exp(logT)
  list(
    data  = data.frame(d[, x_names],
                       A      = A,
                       time   = pmin(T_true, C),
                       status = as.integer(T_true <= C)),
    truth = data.frame(tau = tau, logT = logT,
                       chemo = d$chemo, year = d$year)
  )
}


# ---- 3. Simulate -----------------------------------------------------------
set.seed(seed)
A_rct   <- rbinom(nrow(rct), 1L, 0.5)         # re-randomised
sim_rct <- simulate_source(rct, A_rct)
sim_os  <- simulate_source(os, os$A_real)     # real, confounded assignment

cat("Reference AFT (real GBSG2): beta_A = ", round(beta_A, 3),
    ", sigma = ", round(sigma_W, 3), "\n", sep = "")


# ---- 4. Assemble the shipped data sets -------------------------------------
# Rows are aligned: breast_fusion_truth[i, ] belongs to breast_fusion[i, ],
# and both carry the same id.
source_lab <- factor(rep(c("RCT", "RWD"),
                         c(nrow(sim_rct$data), nrow(sim_os$data))),
                     levels = c("RCT", "RWD"))
sim_data   <- rbind(sim_rct$data, sim_os$data)
sim_truth  <- rbind(sim_rct$truth, sim_os$truth)
n_total    <- nrow(sim_data)

breast_fusion <- data.frame(
  id     = seq_len(n_total),
  source = source_lab,
  S      = as.integer(source_lab == "RCT"),
  A      = as.integer(sim_data$A),
  time   = sim_data$time,
  status = as.integer(sim_data$status),
  age    = as.integer(sim_data$age),
  meno   = as.integer(sim_data$meno),
  size   = as.integer(sim_data$size),
  grade  = as.integer(sim_data$grade),
  nodes  = as.integer(sim_data$nodes),
  pgr    = as.integer(sim_data$pgr),
  er     = as.integer(sim_data$er)
)

breast_fusion_truth <- data.frame(
  id     = seq_len(n_total),
  source = source_lab,
  tau    = sim_truth$tau,
  logT   = sim_truth$logT,
  chemo  = as.integer(sim_truth$chemo),
  year   = as.integer(sim_truth$year)
)

# Sanity checks before anything is written.
stopifnot(
  nrow(breast_fusion) == nrow(survival::gbsg) + nrow(survival::rotterdam),
  identical(breast_fusion$id, breast_fusion_truth$id),
  identical(breast_fusion$source, breast_fusion_truth$source),
  !anyNA(breast_fusion), !anyNA(breast_fusion_truth),
  all(breast_fusion$time > 0),
  all(breast_fusion$status %in% 0:1),
  all(breast_fusion$A %in% 0:1)
)
str(breast_fusion)
str(breast_fusion_truth)


# ---- 5. Store in data/ (uncomment deliberately) ----------------------------
# Writes data/breast_fusion.rda and data/breast_fusion_truth.rda.  Afterwards
# run devtools::document() and check that data(breast_fusion) works.
#
# usethis::use_data(breast_fusion, breast_fusion_truth,
#                   overwrite = TRUE, compress = "xz")
