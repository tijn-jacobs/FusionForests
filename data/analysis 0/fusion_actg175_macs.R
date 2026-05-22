## fusion_actg175_macs.R
##
## Side-by-side comparison of two FusionForest survival fits:
##   M1 -- ACTG175 (RCT) only
##   M2 -- ACTG175 (RCT) + MACS (RWD) fusion
##
## Outputs a single multi-panel PDF
## (data/analysis 0/fusion_actg175_macs.pdf) with five rows of paired
## diagnostics:
##   row 1  KM by treatment, ACTG175 vs MACS (data only)
##   row 2  sigma traceplots                  (M1 vs M2)
##   row 3  subject-level CATE caterpillars  (M1 vs M2)
##   row 4  ATE posterior densities          (M1 vs M2)
##   row 5  empirical KM with model-implied per-arm survival overlay
##
## Endpoint: event-free survival (EFS-cd4-fix; variant v3 in
## data/analysis 0/endpoint_comparison.R). ACTG175 contributes its
## published composite `cens` = >=50% CD4 decline from baseline OR
## AIDS-defining event OR all-cause death. MACS is reconstructed to
## match this composite:
##   (i)   AIDS:        AIDSCASE in {2,3}, time = DATE1yy
##   (ii)  CD4 decline: first lab in lab_rslt where LEU3N drops to
##                      <=50% of the subject's earliest 1991-95 LEU3N
##                      (relative rule, matching ACTG175 -- NOT the
##                      absolute <200/<14% rule used previously)
##   (iii) Death:       all-cause, DEATH in {1,2,3,4}. The death DATE
##                      is redacted in the MACS public release, so
##                      death-event times fall back to the subject's
##                      last lab-visit year (known bias).
## See endpoint_comparison.R for the rule comparison and concordance
## with the previous coding.
##
## Treatment contrast (matches ACTG175$treat):
##   0 = ZDV monotherapy
##   1 = ZDV+ddI  OR  ZDV+ddC  OR  ddI monotherapy
##   excluded: ddC mono, ddI+ddC (186), AZT+ddI+ddC (187)

setwd("~/Library/CloudStorage/OneDrive-VrijeUniversiteitAmsterdam/Documents/GitHub/FusionForests/")

suppressPackageStartupMessages({
  library(speff2trial)
  library(survival)
  library(FusionForests)
  library(ShrinkageTrees)
  library(ggplot2)
  library(patchwork)
})

proj        <- "."
macsDir    <- file.path(proj, "data", "MACS PDS")
actgEra    <- 1991:1995
# Tier 1 harmonised covariates (age, wtkg, cd4, cd8) are augmented with:
#   anchor_year      -- calendar year of anchor visit (constant 1992 in RCT)
#   race             -- binary, 0 = White non-Hispanic, 1 = other
#   prior_art_years  -- years of prior antiretroviral exposure before anchor
# Symptomatic HIV at baseline is also clinically relevant but its MACS
# analogue requires reading section1.dat (symptom form) whose layout is
# not visible here; left as a follow-up.
harmCovars <- c("age", "wtkg", "cd4", "cd8",
                "anchor_year", "race", "prior_art_years")
artCodes  <- c(92, 94, 147, 180, 185, 186, 187)
lbToKg    <- 0.453592

# ---------------------------------------------------------------------
# 0. Helpers (SAS .inp parser, MACS reader, treatment classifier)
# ---------------------------------------------------------------------

parse_sas_inp <- function(path) {
  ln <- readLines(path, warn = FALSE)
  ln <- gsub("/\\*.*?\\*/", "", ln)
  ln <- trimws(ln)
  rows <- list()
  for (s in ln) {
    if (s == "" || s == ";" || toupper(s) == "INPUT") next
    m <- regmatches(s, regexec(
      "^@\\s*([0-9]+)\\s+([A-Za-z_][A-Za-z0-9_]*)\\s+([0-9]+)\\.([0-9]+)\\s*$",
      s))[[1]]
    if (length(m) == 5) {
      st <- as.integer(m[2]); wd <- as.integer(m[4])
      rows[[length(rows) + 1L]] <- data.frame(
        name = m[3], start = st, end = st + wd - 1L,
        decimals = as.integer(m[5]), stringsAsFactors = FALSE)
      next
    }
    m <- regmatches(s, regexec(
      "^\\$?([A-Za-z_][A-Za-z0-9_]*)\\s+\\$?([0-9]+)\\s*-\\s*([0-9]+)\\s*$",
      s))[[1]]
    if (length(m) == 4) {
      rows[[length(rows) + 1L]] <- data.frame(
        name = m[2], start = as.integer(m[3]),
        end = as.integer(m[4]),
        decimals = NA_integer_, stringsAsFactors = FALSE)
      next
    }
    m <- regmatches(s, regexec(
      "^\\$?([A-Za-z_][A-Za-z0-9_]*)\\s+\\$?([0-9]+)\\s*$", s))[[1]]
    if (length(m) == 3) {
      pos <- as.integer(m[3])
      rows[[length(rows) + 1L]] <- data.frame(
        name = m[2], start = pos, end = pos,
        decimals = NA_integer_, stringsAsFactors = FALSE)
      next
    }
  }
  do.call(rbind, rows)
}

read_macs_subset <- function(dat_path, layout, vars) {
  miss <- setdiff(vars, layout$name)
  if (length(miss))
    stop("Variables not in layout: ", paste(miss, collapse = ", "))
  lay <- layout[match(vars, layout$name), , drop = FALSE]
  lines <- readLines(dat_path, warn = FALSE)
  out <- as.data.frame(setNames(
    lapply(seq_len(nrow(lay)), function(i) {
      v <- trimws(substr(lines, lay$start[i], lay$end[i]))
      v[v == ""] <- NA_character_
      x <- suppressWarnings(as.numeric(v))
      if (!is.na(lay$decimals[i])) x <- x / 10^lay$decimals[i]
      x
    }), lay$name),
    stringsAsFactors = FALSE)
  out
}

classify_treat <- function(drg_set) {
  art <- intersect(drg_set, artCodes)
  if (length(art) == 0)             return(NA_integer_)
  if (any(art %in% c(186, 187)))    return(NA_integer_)
  if (identical(sort(art), 94))     return(NA_integer_)   # ddC mono
  if (identical(sort(art), 92))     return(0L)            # ZDV mono
  return(1L)
}

# ---------------------------------------------------------------------
# 1. Frame builders
# ---------------------------------------------------------------------

build_rct <- function() {
  data(ACTG175)
  d <- ACTG175
  # anchor_year: ACTG175 enrolled Dec 1991 - Oct 1992; the public release
  # carries no per-subject date, so we set a constant 1992.
  # race: ACTG175 codes 0 = White, 1 = non-White; pass through unchanged.
  # prior_art_years: convert preanti (days of previously received ART)
  # to years so the scale matches the MACS derivation below.
  data.frame(
    id              = as.character(d$pidnum),
    source          = "RCT",
    treat           = as.integer(d$treat),
    log_time        = log(d$days / 365.25),
    status          = as.integer(d$cens),
    age             = d$age,
    wtkg            = d$wtkg,
    cd4             = d$cd40,
    cd8             = d$cd80,
    anchor_year     = 1992L,
    race            = as.integer(d$race),
    prior_art_years = d$preanti / 365.25,
    stringsAsFactors = FALSE
  )
}

build_macs <- function() {
  cat("Building MACS-RWD frame ...\n")
  inp <- function(f) file.path(macsDir, "sasinp", f)
  dat <- function(f) file.path(macsDir, "data",   f)

  lay_macsid   <- parse_sas_inp(inp("macsid.inp"))
  lay_section2 <- parse_sas_inp(inp("section2.inp"))
  lay_drugf1   <- parse_sas_inp(inp("drugf1.inp"))
  lay_lab      <- parse_sas_inp(inp("lab_rslt.inp"))
  lay_outcome  <- parse_sas_inp(inp("outcome.inp"))
  lay_phy      <- parse_sas_inp(inp("phy_exam.inp"))
  lay_outcome$start[lay_outcome$name == "DEATH"] <- 989
  lay_outcome$end  [lay_outcome$name == "DEATH"] <- 989

  macsid   <- read_macs_subset(dat("macsid.dat"), lay_macsid,
    c("CASEID", "STATUS02", "STATUS10"))
  section2 <- read_macs_subset(dat("section2.dat"), lay_section2,
    c("CASEID", "VISIT", "BORNY", "RACE"))
  drugf1   <- read_macs_subset(dat("drugf1.dat"), lay_drugf1,
    c("CASEID", "VISIT", "AVQY", "DRGAV", "AVNW"))
  lab      <- read_macs_subset(dat("lab_rslt.dat"), lay_lab,
    c("CASEID", "LDATY", "LEU3N", "LEU2N"))
  outcome  <- read_macs_subset(dat("outcome.dat"), lay_outcome,
    c("CASEID", "AIDSCASE", "DATE1yy", "SELFCD4Dyy", "DEATH"))
  phy      <- read_macs_subset(dat("phy_exam.dat"), lay_phy,
    c("CASEID", "VISIT", "LDWGT"))

  hiv <- macsid$CASEID[!(macsid$STATUS02 %in% 1 | macsid$STATUS10 %in% 1)]
  drugf1   <- drugf1  [drugf1$CASEID   %in% hiv, ]
  lab      <- lab     [lab$CASEID      %in% hiv, ]
  outcome  <- outcome [outcome$CASEID  %in% hiv, ]
  phy      <- phy     [phy$CASEID      %in% hiv, ]
  section2 <- section2[section2$CASEID %in% hiv, ]

  curr <- drugf1[drugf1$AVQY %in% actgEra & drugf1$AVNW %in% 2, ]
  drugs_by_subj <- split(curr$DRGAV, curr$CASEID)
  anchor_yr     <- vapply(split(curr$AVQY, curr$CASEID),
                          function(x) min(x, na.rm = TRUE), numeric(1))
  treat_int     <- vapply(drugs_by_subj, classify_treat, integer(1))

  base <- data.frame(id = names(drugs_by_subj),
                     treat = treat_int,
                     anchor_year = anchor_yr,
                     stringsAsFactors = FALSE)
  base <- base[!is.na(base$treat), ]

  # ----- MACS endpoint = EFS-cd4-fix (v3 in endpoint_comparison.R) ----
  # Composite matched to ACTG175$cens:
  #   (i)   AIDS dx:         AIDSCASE in {2, 3}, time = DATE1yy
  #   (ii)  CD4 >=50% decl:  reconstructed from lab_rslt -- baseline =
  #                          earliest 1991-95 LEU3N, event = first lab
  #                          where LEU3N <= 0.5 * baseline
  #   (iii) all-cause death: DEATH in {1, 2, 3, 4}; death DATE is
  #                          redacted, so death-event time falls back
  #                          to the subject's last lab-visit year
  # Previous versions used SELFCD4Dyy (absolute <200/<14% rule) and
  # DEATH %in% c(1, 2) (AIDS-related only). Both diverged from
  # ACTG175's composite; see endpoint_comparison.R for the analysis.

  lab_cd4 <- lab[!is.na(lab$LEU3N) & lab$LEU3N > 0 &
                 !is.na(lab$LDATY) & lab$LDATY > 0, ]
  lab_cd4 <- lab_cd4[order(lab_cd4$CASEID, lab_cd4$LDATY), ]
  base_cd4 <- aggregate(LEU3N ~ CASEID,
                        data = lab_cd4[lab_cd4$LDATY %in% actgEra, ],
                        FUN = function(x) x[1])
  names(base_cd4)[2] <- "cd4_base"
  lab_cd4 <- merge(lab_cd4, base_cd4, by = "CASEID")
  decl_rows <- lab_cd4[lab_cd4$LEU3N <= 0.5 * lab_cd4$cd4_base, ]
  cd4_rel <- aggregate(LDATY ~ CASEID, data = decl_rows, FUN = min)
  names(cd4_rel)[2] <- "cd4_rel_yr"

  oc <- outcome
  oc$aids_yr <- ifelse(oc$AIDSCASE %in% c(2, 3) & oc$DATE1yy > 0,
                       oc$DATE1yy, NA_real_)
  oc$is_dead <- !is.na(oc$DEATH) & oc$DEATH %in% c(1, 2, 3, 4)

  last_lab <- aggregate(LDATY ~ CASEID, data = lab,
                        FUN = function(x) max(x, na.rm = TRUE))
  names(last_lab)[2] <- "last_year"

  base <- merge(base, oc[, c("CASEID", "aids_yr", "is_dead")],
                by.x = "id", by.y = "CASEID", all.x = TRUE)
  base <- merge(base, cd4_rel,
                by.x = "id", by.y = "CASEID", all.x = TRUE)
  base <- merge(base, last_lab,
                by.x = "id", by.y = "CASEID", all.x = TRUE)
  base$is_dead[is.na(base$is_dead)] <- FALSE

  prior_aids <- !is.na(base$aids_yr) & base$aids_yr < base$anchor_year
  base <- base[!prior_aids, ]

  # Composite event time: earliest of {aids, cd4_rel, death-imputed}.
  # NA stays NA in pmin if we go through Inf instead of na.rm = TRUE
  # (which behaves inconsistently across R versions for all-NA rows).
  death_yr <- ifelse(base$is_dead, base$last_year, NA_real_)
  comp <- cbind(base$aids_yr, base$cd4_rel_yr, death_yr)
  comp[is.na(comp)] <- Inf
  ev_yr <- pmin(comp[, 1], comp[, 2], comp[, 3])
  ev_yr[is.infinite(ev_yr)] <- NA_real_
  base$has_event <- !is.na(ev_yr)
  base$end_year  <- ifelse(base$has_event, ev_yr, base$last_year)
  base$log_time  <- log(pmax(base$end_year - base$anchor_year, 0.5))
  base$status    <- as.integer(base$has_event)

  byr <- aggregate(BORNY ~ CASEID, data = section2,
                   FUN = function(x) max(x, na.rm = TRUE))
  base$age <- base$anchor_year - byr$BORNY[match(base$id, byr$CASEID)]

  phy_w <- phy[!is.na(phy$LDWGT) & phy$LDWGT > 0, ]
  wt_med <- aggregate(LDWGT ~ CASEID, data = phy_w,
                      FUN = function(x) median(x, na.rm = TRUE))
  base$wtkg <- wt_med$LDWGT[match(base$id, wt_med$CASEID)] * lbToKg

  lab_e <- lab[lab$LDATY %in% actgEra, ]
  cd4_med <- aggregate(LEU3N ~ CASEID,
                       data = lab_e[!is.na(lab_e$LEU3N) & lab_e$LEU3N > 0, ],
                       FUN = function(x) median(x, na.rm = TRUE))
  cd8_med <- aggregate(LEU2N ~ CASEID,
                       data = lab_e[!is.na(lab_e$LEU2N) & lab_e$LEU2N > 0, ],
                       FUN = function(x) median(x, na.rm = TRUE))
  base$cd4 <- cd4_med$LEU3N[match(base$id, cd4_med$CASEID)]
  base$cd8 <- cd8_med$LEU2N[match(base$id, cd8_med$CASEID)]

  # Race: MACS RACE 1 = White non-Hispanic, everything else (incl. White
  # Hispanic) collapses to 1 so the binary matches ACTG175's 0/1 coding.
  race_d <- aggregate(RACE ~ CASEID,
                      data = section2[!is.na(section2$RACE), ],
                      FUN = function(x) x[1])
  base$race <- as.integer(
    race_d$RACE[match(base$id, race_d$CASEID)] != 1)

  # Prior ART exposure (years before anchor_year). For each subject we
  # take the earliest drugf1 record carrying an ART code (artCodes); if
  # that year precedes anchor_year, prior_art_years = anchor - first;
  # otherwise 0. Approximates ACTG175's preanti (days of prior ART).
  prior_art <- drugf1[drugf1$DRGAV %in% artCodes, ]
  first_art <- aggregate(AVQY ~ CASEID, data = prior_art, FUN = min)
  names(first_art)[2] <- "first_art_yr"
  base <- merge(base, first_art, by.x = "id", by.y = "CASEID",
                all.x = TRUE)
  base$prior_art_years <- ifelse(
    !is.na(base$first_art_yr) & base$first_art_yr < base$anchor_year,
    base$anchor_year - base$first_art_yr, 0)

  out <- data.frame(
    id = base$id, source = "MACS", treat = base$treat,
    log_time = base$log_time, status = base$status,
    age = base$age, wtkg = base$wtkg, cd4 = base$cd4, cd8 = base$cd8,
    anchor_year = base$anchor_year,
    race = base$race,
    prior_art_years = base$prior_art_years,
    stringsAsFactors = FALSE)
  out <- out[complete.cases(out[, harmCovars]), ]
  cat("MACS frame:", nrow(out), "subjects\n")
  out
}

# ---------------------------------------------------------------------
# 2. Assemble
# ---------------------------------------------------------------------

cat("\n", strrep("=", 70), "\n  ASSEMBLE\n",
    strrep("=", 70), "\n", sep = "")
rct  <- build_rct()
macs <- build_macs()

cat(sprintf("RCT  n=%d  treat0=%d  treat1=%d  events=%d\n",
            nrow(rct),  sum(rct$treat == 0),
            sum(rct$treat == 1), sum(rct$status == 1)))
cat(sprintf("MACS n=%d  treat0=%d  treat1=%d  events=%d\n",
            nrow(macs), sum(macs$treat == 0),
            sum(macs$treat == 1), sum(macs$status == 1)))

# ---------------------------------------------------------------------
# 3. Fit M1 (RCT only) and M2 (RCT + MACS)
# ---------------------------------------------------------------------

N_POST <- 5000
N_BURN <- 5000

# --- Non-fusion fits (M1, M3): CausalShrinkageForest, mirroring ACTG175.R
#     Same model class, same hyperparameter calibration:
#       n_trees_treat = 50, n_trees_control = 200
#       prior_type_{control,treat} = "standard"
#       local_hp_{treat,control} from censored_info range
#       power/base: control (2, 0.95), treat (3, 0.25)
#       treatment_coding = "centered"
run_fit_csf <- function(d) {
  X <- as.matrix(d[, harmCovars])
  n_trees_treat   <- 50
  n_trees_control <- 200
  log_info <- ShrinkageTrees:::censored_info(d$log_time, d$status)
  local_hp_treat   <- (log_info$max - log_info$min) /
                      (2 * 2 * sqrt(n_trees_treat))
  local_hp_control <- (log_info$max - log_info$min) /
                      (2 * 2 * sqrt(n_trees_control))
  fit <- ShrinkageTrees::CausalShrinkageForest(
    y                         = d$log_time,
    status                    = d$status,
    X_train_control           = X,
    X_train_treat             = X,
    treatment_indicator_train = d$treat,
    outcome_type              = "right-censored",
    timescale                 = "log",
    prior_type_control        = "standard",
    prior_type_treat          = "standard",
    local_hp_treat            = local_hp_treat,
    local_hp_control          = local_hp_control,
    number_of_trees_control   = n_trees_control,
    number_of_trees_treat     = n_trees_treat,
    power_control             = 2,
    base_control              = 0.95,
    power_treat               = 3,
    base_treat                = 0.25,
    treatment_coding          = "centered",
    N_post                    = N_POST,
    N_burn                    = N_BURN,
    store_posterior_sample    = TRUE,
    verbose                   = FALSE
  )
  # CausalShrinkageForest stores sigma on the standardised internal scale;
  # recover the log-time-scale posterior so downstream survival overlays
  # use the correct sigma.
  fit$sigma_scaled <- fit$sigma * log_info$sd
  fit
}

# --- Fusion fit (M2): FusionForest, three-forest decomposition.
run_fit_ff <- function(d, src, n_trees_deconf = 200) {
  X <- as.matrix(d[, harmCovars])
  fit <- FusionForest(
    y                         = d$log_time,
    status                    = d$status,
    X_train_control           = X,
    X_train_treat             = X,
    treatment_indicator_train = d$treat,
    source_indicator_train    = src,
    outcome_type              = "right-censored",
    timescale                 = "log",
    decomposition             = "three-forest",
    treatment_coding          = "centered",
    number_of_trees_control   = 200,
    number_of_trees_treat     = 100,
    number_of_trees_deconf    = n_trees_deconf,
    N_post                    = N_POST,
    N_burn                    = N_BURN,
    store_posterior_sample    = TRUE,
    verbose                   = FALSE
  )
  fit$sigma_scaled <- fit$sigma
  fit
}

cat("\n", strrep("=", 70), "\n  FIT M1: RCT only (CausalShrinkageForest)\n",
    strrep("=", 70), "\n", sep = "")
fit_M1 <- run_fit_csf(rct)

cat("\n", strrep("=", 70), "\n  FIT M2: RCT + MACS fusion (FusionForest)\n",
    strrep("=", 70), "\n", sep = "")
d_fuse  <- rbind(rct, macs)
d_fuse  <- d_fuse[is.finite(d_fuse$log_time), ]
src_M2  <- as.integer(d_fuse$source == "RCT")
fit_M2  <- run_fit_ff(d_fuse, src_M2, n_trees_deconf = 200)

cat("\n", strrep("=", 70),
    "\n  FIT M3: MACS only, naive RWD (CausalShrinkageForest)\n",
    strrep("=", 70), "\n", sep = "")
fit_M3 <- run_fit_csf(macs)

# ---------------------------------------------------------------------
# 4. Plot helpers
# ---------------------------------------------------------------------

# Evaluation indices per model: the population each estimand averages
# over.  M1 = RCT population, M2 = RCT population (so the fusion ATE is
# directly comparable to M1), M3 = MACS population (the only available
# population in a pure RWD analysis).
eval_idx_M1 <- seq_len(nrow(rct))
eval_idx_M2 <- which(d_fuse$source == "RCT")
eval_idx_M3 <- seq_len(nrow(macs))

# With timescale = "log" FusionForest keeps every prediction on the log
# scale (no exp() back-transform), so the stored objects are already
# tau(X) and mu(X). No further transformation needed.
log_cate_samples <- function(fit) fit$train_predictions_sample_treat

# Per-subject log-time prediction (expected log T) per arm:
#   linpred_z(X) = mu(X) + tau(X) * (z - 0.5)
linpred_per_arm <- function(fit) {
  mu0 <- fit$train_predictions_control
  tau <- fit$train_predictions_treat
  list(z0 = mu0 - tau / 2,
       z1 = mu0 + tau / 2)
}

theme_panel <- function() {
  theme_minimal(base_size = 10) +
    theme(legend.position = "bottom",
          plot.title = element_text(size = 11, face = "bold"))
}

# ---------------------------------------------------------------------
# 5. Plots row 1: data KM by treatment for ACTG175 and for MACS
# ---------------------------------------------------------------------

km_data <- function(d, label) {
  d$treat_lab <- factor(d$treat, levels = 0:1,
    labels = c("ZDV mono (Z=0)", "ZDV+other / ddI (Z=1)"))
  fit <- survfit(Surv(log_time, status) ~ treat_lab, data = d)
  s <- summary(fit)
  data.frame(time = s$time, surv = s$surv,
             lower = s$lower, upper = s$upper,
             strata = s$strata, label = label)
}
km_df <- rbind(
  km_data(rct,                   "ACTG175 (M1 data)"),
  km_data(rbind(rct, macs),      "Pooled (M2 data)"),
  km_data(macs,                  "MACS (M3 data)"))
km_df$strata <- factor(km_df$strata)

plot_km_dataset <- function(lbl) {
  ggplot(km_df[km_df$label == lbl, ],
         aes(x = exp(time), y = surv, colour = strata, fill = strata)) +
    geom_step(linewidth = 0.7) +
    geom_ribbon(aes(ymin = lower, ymax = upper),
                alpha = 0.15, colour = NA) +
    labs(x = "Years from baseline", y = "Survival probability",
         colour = NULL, fill = NULL,
         title = paste0("KM - ", lbl)) +
    theme_panel()
}
p11 <- plot_km_dataset("ACTG175 (M1 data)")
p12 <- plot_km_dataset("Pooled (M2 data)")
p13 <- plot_km_dataset("MACS (M3 data)")

# ---------------------------------------------------------------------
# 6. Plots row 2: sigma traces
# ---------------------------------------------------------------------

plot_sigma <- function(fit, label) {
  df <- data.frame(iter = seq_along(fit$sigma_scaled),
                   sigma = fit$sigma_scaled)
  ggplot(df, aes(x = iter, y = sigma)) +
    geom_line(colour = "steelblue", linewidth = 0.4) +
    labs(x = "MCMC iteration (post burn-in)",
         y = expression(sigma),
         title = paste0("Sigma trace - ", label)) +
    theme_panel()
}
p21 <- plot_sigma(fit_M1, "M1 (RCT only)")
p22 <- plot_sigma(fit_M2, "M2 (RCT + MACS)")
p23 <- plot_sigma(fit_M3, "M3 (MACS only)")

# ---------------------------------------------------------------------
# 7. Plots row 3: subject-level CATE caterpillars (RCT subjects only)
# ---------------------------------------------------------------------

plot_caterpillar <- function(fit, idx, label, pop) {
  log_samp <- log_cate_samples(fit)[, idx, drop = FALSE]
  est <- colMeans(log_samp)
  lo  <- apply(log_samp, 2, quantile, probs = 0.025)
  hi  <- apply(log_samp, 2, quantile, probs = 0.975)
  ord <- order(est)
  df  <- data.frame(rank = seq_along(est),
                    est = est[ord], lo = lo[ord], hi = hi[ord])
  ggplot(df, aes(x = rank, y = est)) +
    geom_ribbon(aes(ymin = lo, ymax = hi),
                fill = "steelblue", alpha = 0.3) +
    geom_line(colour = "steelblue") +
    geom_hline(yintercept = 0, linetype = "dashed") +
    labs(x = paste0(pop, " subject (ordered by posterior mean CATE)"),
         y = "CATE on log-survival scale",
         title = paste0("CATE caterpillar - ", label)) +
    theme_panel()
}
p31 <- plot_caterpillar(fit_M1, eval_idx_M1, "M1 (RCT only)",       "RCT")
p32 <- plot_caterpillar(fit_M2, eval_idx_M2, "M2 (RCT + MACS)",     "RCT")
p33 <- plot_caterpillar(fit_M3, eval_idx_M3, "M3 (MACS only)",      "MACS")

# ---------------------------------------------------------------------
# 8. Plots row 4: ATE posterior densities
# ---------------------------------------------------------------------

plot_ate <- function(fit, idx, label, pop) {
  log_samp <- log_cate_samples(fit)[, idx, drop = FALSE]
  ate_iter <- rowMeans(log_samp)
  qs       <- quantile(ate_iter, c(0.025, 0.5, 0.975))
  ggplot(data.frame(ate = ate_iter), aes(x = ate)) +
    geom_density(fill = "steelblue", alpha = 0.4) +
    geom_vline(xintercept = mean(ate_iter)) +
    geom_vline(xintercept = qs[c(1, 3)], linetype = "dashed") +
    labs(x = sprintf("ATE on log-survival scale (%s population)", pop),
         y = "Posterior density",
         title = sprintf("ATE - %s: %.3f (%.3f, %.3f)",
                         label, mean(ate_iter), qs[1], qs[3])) +
    theme_panel()
}
p41 <- plot_ate(fit_M1, eval_idx_M1, "M1", "RCT")
p42 <- plot_ate(fit_M2, eval_idx_M2, "M2", "RCT")
p43 <- plot_ate(fit_M3, eval_idx_M3, "M3", "MACS")

# ---------------------------------------------------------------------
# 9. Plots row 5: empirical KM (RCT) overlaid with model-implied
#    per-arm survival curves averaged over the RCT covariate distribution
# ---------------------------------------------------------------------

# Empirical KM on a chosen evaluation cohort, x-axis in years.
build_emp_df <- function(d) {
  fit <- survfit(Surv(log_time, status) ~ treat, data = d)
  s   <- summary(fit)
  out <- data.frame(time = exp(s$time), surv = s$surv,
                    strata = s$strata)
  out$arm <- ifelse(grepl("=0", out$strata), "Z=0", "Z=1")
  out
}
emp_df_M1 <- build_emp_df(rct)
emp_df_M2 <- build_emp_df(rct)
emp_df_M3 <- build_emp_df(macs)

# Posterior survival curve per arm with 95% credible band, integrated
# over the RCT covariate distribution.
#
# At iteration i: linpred_z,i(X) = mu_i(X) + tau_i(X) * (z - 1/2),
# S_z,i(t) = mean_X[ 1 - Phi( (t - linpred_z,i(X)) / sigma_i ) ].
# Posterior summaries are taken across i.
model_implied_surv <- function(fit, idx, grid_log_t) {
  muS  <- fit$train_predictions_sample_control[, idx, drop = FALSE]
  tauS <- fit$train_predictions_sample_treat  [, idx, drop = FALSE]
  sigS <- fit$sigma_scaled
  nIter <- nrow(muS)
  s0 <- matrix(NA_real_, nIter, length(grid_log_t))
  s1 <- matrix(NA_real_, nIter, length(grid_log_t))
  for (i in seq_len(nIter)) {
    lp0 <- muS[i, ] - tauS[i, ] / 2
    lp1 <- muS[i, ] + tauS[i, ] / 2
    z0  <- outer(grid_log_t, lp0, "-") / sigS[i]
    z1  <- outer(grid_log_t, lp1, "-") / sigS[i]
    s0[i, ] <- rowMeans(1 - pnorm(z0))
    s1[i, ] <- rowMeans(1 - pnorm(z1))
  }
  summarise <- function(s, arm) data.frame(
    time  = exp(grid_log_t),
    surv  = colMeans(s),
    lower = apply(s, 2, quantile, probs = 0.025),
    upper = apply(s, 2, quantile, probs = 0.975),
    arm   = arm, kind = "model")
  rbind(summarise(s0, "Z=0"), summarise(s1, "Z=1"))
}
plot_km_overlay <- function(fit, idx, emp_df, label) {
  grid_log_t <- seq(log(min(emp_df$time)), log(max(emp_df$time)),
                    length.out = 200)
  mod <- model_implied_surv(fit, idx, grid_log_t)
  ggplot() +
    geom_ribbon(data = mod,
                aes(x = time, ymin = lower, ymax = upper, fill = arm),
                alpha = 0.18) +
    geom_step(data = emp_df,
              aes(x = time, y = surv, colour = arm),
              linewidth = 0.6) +
    geom_line(data = mod,
              aes(x = time, y = surv, colour = arm),
              linewidth = 0.8, linetype = "dashed") +
    labs(x = "Years from baseline", y = "Survival probability",
         colour = NULL, fill = NULL,
         title = paste0("Empirical KM + model-implied (95% CrI) - ", label)) +
    theme_panel()
}
p51 <- plot_km_overlay(fit_M1, eval_idx_M1, emp_df_M1, "M1 (RCT only)")
p52 <- plot_km_overlay(fit_M2, eval_idx_M2, emp_df_M2, "M2 (RCT + MACS)")
p53 <- plot_km_overlay(fit_M3, eval_idx_M3, emp_df_M3, "M3 (MACS only)")

# ---------------------------------------------------------------------
# 9b. Plots rows 6-7: per-arm KM overlays ACTG175 vs MACS
#
# MACS follow-up extends well past the RCT horizon; that is a strength
# (more events, longer-term signal) but it also means the "full" view
# and the "RCT-horizon" view tell different stories. Row 6 shows the
# full follow-up; row 7 restricts the x-axis to max(rct$time) so the
# two datasets are compared on the same horizon.
# ---------------------------------------------------------------------

rct_xmax <- max(exp(rct$log_time), na.rm = TRUE)

km_by_dataset_one_arm <- function(treat_val) {
  d <- rbind(
    cbind(rct [rct $treat == treat_val, ], dataset = "ACTG175 (RCT)"),
    cbind(macs[macs$treat == treat_val, ], dataset = "MACS (RWD)"))
  fit <- survfit(Surv(log_time, status) ~ dataset, data = d)
  s   <- summary(fit)
  data.frame(time = exp(s$time), surv = s$surv,
             lower = s$lower, upper = s$upper,
             strata = s$strata)
}
km_by_dataset_both <- function() {
  d <- rbind(
    cbind(rct,  dataset = "ACTG175 (RCT)"),
    cbind(macs, dataset = "MACS (RWD)"))
  d$grp <- paste0(d$dataset, " | ",
                  ifelse(d$treat == 0, "Z=0", "Z=1"))
  fit <- survfit(Surv(log_time, status) ~ grp, data = d)
  s   <- summary(fit)
  data.frame(time = exp(s$time), surv = s$surv,
             lower = s$lower, upper = s$upper,
             strata = s$strata)
}

plot_overlay <- function(df, title, x_max = NULL, ribbon_alpha = 0.15) {
  p <- ggplot(df,
              aes(x = time, y = surv, colour = strata, fill = strata)) +
    geom_step(linewidth = 0.7) +
    geom_ribbon(aes(ymin = lower, ymax = upper),
                alpha = ribbon_alpha, colour = NA) +
    labs(x = "Years from baseline", y = "Survival probability",
         colour = NULL, fill = NULL, title = title) +
    theme_panel()
  if (!is.null(x_max))
    p <- p + coord_cartesian(xlim = c(0, x_max))
  p
}

df_g0 <- km_by_dataset_one_arm(0)
df_g1 <- km_by_dataset_one_arm(1)
df_g  <- km_by_dataset_both()

p61 <- plot_overlay(df_g0, "KM Z=0 (ZDV mono) - ACTG175 vs MACS")
p62 <- plot_overlay(df_g1, "KM Z=1 (combo)    - ACTG175 vs MACS")
p63 <- plot_overlay(df_g,  "KM both arms      - ACTG175 vs MACS",
                    ribbon_alpha = 0.08)

p71 <- plot_overlay(df_g0,
                    sprintf("Z=0, x <= %.1f yr (RCT horizon)", rct_xmax),
                    x_max = rct_xmax)
p72 <- plot_overlay(df_g1,
                    sprintf("Z=1, x <= %.1f yr (RCT horizon)", rct_xmax),
                    x_max = rct_xmax)
p73 <- plot_overlay(df_g,
                    sprintf("Both arms, x <= %.1f yr (RCT horizon)",
                            rct_xmax),
                    x_max = rct_xmax, ribbon_alpha = 0.08)

# ---------------------------------------------------------------------
# 9c. Linear projection of CATE: M1 (RCT only) vs M2 (Fusion)
#
# Mirrors examples/projection_demo.R. Each posterior draw of tau(x) is
# projected onto a linear basis in the harmonised covariates via a
# Bayesian-bootstrap weighted least squares (Woody, Carvalho & Murray
# 2020). The coefficients inherit posterior uncertainty without
# refitting; the shift M1 -> M2 is exactly the effect of borrowing the
# MACS RWD arm under the SAME basis evaluated at the SAME rows (RCT
# subjects -- the target population).
#
# fusion_projection() doesn't expose row subsetting, so we use the
# inline projector from the demo for both M1 (CSF, has no fit$meta)
# and M2 (FF), restricted to the RCT rows of d_fuse. Non-intercept
# columns are centred AND scaled so the coefficients are "per SD"
# (necessary because age, cd4 etc. live on very different numeric
# scales). anchor_year is dropped from the basis because it is
# constant (1992) across all RCT subjects and would make the design
# matrix rank-deficient on the RCT evaluation set.
# ---------------------------------------------------------------------

project_tau_matrix <- function(tau_draws, X_eval, basis,
                               scale_cols = TRUE) {
  R    <- nrow(tau_draws)
  n_ev <- ncol(tau_draws)
  stopifnot(nrow(X_eval) == n_ev)
  Phi <- model.matrix(basis, data = X_eval)
  non_int <- setdiff(colnames(Phi), "(Intercept)")
  if (length(non_int))
    Phi[, non_int] <- scale(Phi[, non_int],
                            center = TRUE, scale = scale_cols)
  g_w <- matrix(rgamma(R * n_ev, 1, 1), R, n_ev)
  w   <- g_w / rowSums(g_w)
  gamma <- matrix(NA_real_, R, ncol(Phi))
  for (r in seq_len(R))
    gamma[r, ] <- lm.wfit(Phi, tau_draws[r, ], w = w[r, ])$coefficients
  colnames(gamma) <- colnames(Phi)
  gamma
}

proj_basis <- ~ age + wtkg + cd4 + cd8 + race + prior_art_years
X_rct_eval <- as.data.frame(rct[, all.vars(proj_basis)])

rct_in_fuse <- which(d_fuse$source == "RCT")
stopifnot(length(rct_in_fuse) == nrow(rct))

cat("\n", strrep("=", 70),
    "\n  PROJECT: M1 (RCT only) and M2 (Fusion) onto linear basis\n",
    strrep("=", 70), "\n", sep = "")

set.seed(1)
g_proj_M1 <- project_tau_matrix(
  fit_M1$train_predictions_sample_treat,
  X_rct_eval, proj_basis)
g_proj_M2 <- project_tau_matrix(
  fit_M2$train_predictions_sample_treat[, rct_in_fuse, drop = FALSE],
  X_rct_eval, proj_basis)

psum <- function(v) c(mean = mean(v),
                      lo   = quantile(v, 0.025, names = FALSE),
                      hi   = quantile(v, 0.975, names = FALSE))

proj_rows <- lapply(colnames(g_proj_M1), function(nm) {
  s1 <- psum(g_proj_M1[, nm]); s2 <- psum(g_proj_M2[, nm])
  rbind(
    data.frame(coef = nm, model = "M1 (RCT only)",
               mean = s1["mean"], lo = s1["lo"], hi = s1["hi"],
               stringsAsFactors = FALSE),
    data.frame(coef = nm, model = "M2 (Fusion)",
               mean = s2["mean"], lo = s2["lo"], hi = s2["hi"],
               stringsAsFactors = FALSE))
})
proj_df <- do.call(rbind, proj_rows)
proj_df$coef  <- factor(proj_df$coef,
                        levels = rev(colnames(g_proj_M1)))
proj_df$model <- factor(proj_df$model,
                        levels = c("M1 (RCT only)", "M2 (Fusion)"))

# Console table
cat("\nProjection coefficients on log-survival scale ",
    "(non-intercept slopes are per SD of covariate):\n", sep = "")
proj_tbl <- do.call(rbind, lapply(colnames(g_proj_M1), function(nm) {
  s1 <- psum(g_proj_M1[, nm]); s2 <- psum(g_proj_M2[, nm])
  data.frame(
    coef  = nm,
    M1    = sprintf("%+0.3f [%+0.3f, %+0.3f]",
                    s1["mean"], s1["lo"], s1["hi"]),
    M2    = sprintf("%+0.3f [%+0.3f, %+0.3f]",
                    s2["mean"], s2["lo"], s2["hi"]),
    shift = sprintf("%+0.3f", s2["mean"] - s1["mean"]),
    stringsAsFactors = FALSE)
}))
print(proj_tbl, row.names = FALSE)

# Forest-style ggplot: one row per coefficient, two intervals each
# (M1 vs M2), dashed line at zero. The arrow between posterior means
# is rendered with a thin segment so the shift M1 -> M2 reads at a glance.
shift_df <- do.call(rbind, lapply(colnames(g_proj_M1), function(nm) {
  data.frame(coef = nm,
             x0 = mean(g_proj_M1[, nm]),
             x1 = mean(g_proj_M2[, nm]),
             stringsAsFactors = FALSE)
}))
shift_df$coef <- factor(shift_df$coef,
                        levels = rev(colnames(g_proj_M1)))

p_proj <- ggplot(proj_df,
                 aes(x = mean, y = coef, colour = model)) +
  geom_vline(xintercept = 0, linetype = "dashed", colour = "grey50") +
  geom_segment(data = shift_df,
               aes(x = x0, xend = x1, y = coef, yend = coef),
               inherit.aes = FALSE,
               colour = "grey30", linewidth = 0.4,
               arrow = grid::arrow(length = grid::unit(0.10, "inches"),
                                   type = "closed")) +
  geom_errorbarh(aes(xmin = lo, xmax = hi),
                 position = position_dodge(width = 0.55),
                 height = 0.22, linewidth = 0.6) +
  geom_point(position = position_dodge(width = 0.55), size = 2.2) +
  scale_colour_manual(values = c("M1 (RCT only)" = "firebrick",
                                 "M2 (Fusion)"   = "steelblue")) +
  labs(x = "Projection coefficient (CATE, log time-ratio)",
       y = NULL, colour = NULL,
       title = "Linear projection of CATE: M1 (RCT only) -> M2 (Fusion)",
       subtitle = paste("Bayesian-bootstrap weighted LS;",
                        "non-intercept slopes per SD;",
                        "95% credible intervals;",
                        "evaluated on RCT rows")) +
  theme_panel()

# ---------------------------------------------------------------------
# 10. Stitch and export
# ---------------------------------------------------------------------

panel <- (p11 | p12 | p13) /
         (p21 | p22 | p23) /
         (p31 | p32 | p33) /
         (p41 | p42 | p43) /
         (p51 | p52 | p53) /
         (p61 | p62 | p63) /
         (p71 | p72 | p73) /
         p_proj +
  plot_annotation(
    title    = "ACTG175 vs ACTG175+MACS fusion vs MACS only",
    subtitle = "M1 = RCT only, M2 = RCT + MACS fusion, M3 = MACS only (naive RWD)")

out_pdf <- "data/analysis 0/fusion_actg175_macs.pdf"
ggsave(out_pdf, panel, width = 18, height = 30)
cat("\nSaved combined figure: ", out_pdf, "\n", sep = "")

# Console ATE summary
ate_M1 <- rowMeans(log_cate_samples(fit_M1)[, eval_idx_M1, drop = FALSE])
ate_M2 <- rowMeans(log_cate_samples(fit_M2)[, eval_idx_M2, drop = FALSE])
ate_M3 <- rowMeans(log_cate_samples(fit_M3)[, eval_idx_M3, drop = FALSE])
cat("\nATE on log-survival scale:\n")
cat(sprintf("  M1 (RCT only,    eval RCT) : %.4f  95%% CrI (%.4f, %.4f)\n",
            mean(ate_M1), quantile(ate_M1, .025), quantile(ate_M1, .975)))
cat(sprintf("  M2 (RCT + MACS,  eval RCT) : %.4f  95%% CrI (%.4f, %.4f)\n",
            mean(ate_M2), quantile(ate_M2, .025), quantile(ate_M2, .975)))
cat(sprintf("  M3 (MACS only,   eval MACS): %.4f  95%% CrI (%.4f, %.4f)\n",
            mean(ate_M3), quantile(ate_M3, .025), quantile(ate_M3, .975)))

invisible(list(fit_M1 = fit_M1, fit_M2 = fit_M2, fit_M3 = fit_M3,
               rct = rct, macs = macs, panel = panel))
