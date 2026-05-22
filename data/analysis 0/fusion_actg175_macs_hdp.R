## fusion_actg175_macs_hdp.R
##
## Twin of fusion_actg175_macs.R with the FusionForest residual model
## switched from a single Gaussian to a source-HDP centred-DP mixture
## (error_dist = "source_hdp"). M1 (RCT only) and M3 (MACS only) are
## CausalShrinkageForest fits and stay Gaussian -- only M2 (fusion)
## changes. Output PDF is data/analysis 0/fusion_actg175_macs_hdp.pdf so
## the two versions can be compared side by side.
##
## In addition to the rows produced by the Gaussian script, an extra
## diagnostics row is added for the nonparametric error model:
##   - trace of the top-level concentration gamma
##   - traces of per-source concentrations M_RWD, M_RCT
##   - per-source posterior residual density (implied by the mixture)
##   - occupied-cluster count per source
##
## Outputs a single multi-panel PDF with the original five paired-
## diagnostic rows plus the KM-overlay and projection rows:
##   row 1  KM by treatment, ACTG175 vs MACS (data only)
##   row 2  sigma traceplots                  (M1 vs M2)
##   row 3  subject-level CATE caterpillars  (M1 vs M2)
##   row 4  ATE posterior densities          (M1 vs M2)
##   row 5  empirical KM with model-implied per-arm survival overlay
##
## Endpoint (matches survival_curves.R, page-4 setting):
##   RCT (ACTG175) -- published EFS composite `cens` (>=50% CD4
##                    decline from baseline, AIDS-defining event, or
##                    all-cause death; days = follow-up).
##   RWD (MACS)    -- all-cause death (DEATH in {1, 2, 3, 4}) with
##                    last-lab-year as the event-time fallback (the
##                    public release redacts the DEATH date, so last
##                    observed contact is used for those who died and
##                    as the censoring time for those who didn't).
##                    Variant v5 in endpoint_comparison.R. Labelled
##                    EFS to match the RCT side, even though it is
##                    structurally OS on the RWD side; trade-off is a
##                    clean death signal at the cost of dropping the
##                    CD4 and AIDS arms of the v3 composite. Prior-
##                    AIDS exclusion is retained for cohort
##                    comparability with the v3 variants.
## See endpoint_comparison.R for the rule comparison and concordance
## with the previous v3 coding.
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
# Tier 1 harmonised covariates (age, cd4, cd8) are augmented with:
#   anchor_year      -- calendar year of anchor visit (constant 1992 in RCT)
#   race             -- binary, 0 = White non-Hispanic, 1 = other
#   prior_art_years  -- years of prior antiretroviral exposure before anchor
# Symptomatic HIV at baseline is also clinically relevant but its MACS
# analogue requires reading section1.dat (symptom form) whose layout is
# not visible here; left as a follow-up.
# wtkg is dropped: phy_exam$LDWGT is missing for ~67% of MACS subjects and
# was the sole driver of the complete.cases filter, which was cutting the
# MACS cohort by two thirds.
harmCovars <- c("age", "cd4", "cd8",
                "anchor_year", "race", "prior_art_years")
artCodes  <- c(92, 94, 147, 180, 185, 186, 187)

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

build_rct <- function(male_only = FALSE) {
  data(ACTG175)
  d <- ACTG175
  # Optional male-only filter (trial-eligibility-aligned cohort: matches
  # the MACS RWD arm, which is male-only by study design). Drops the
  # ~15% of ACTG175 subjects with gender != 1.
  if (male_only) {
    n_before <- nrow(d)
    d <- d[!is.na(d$gender) & d$gender == 1, ]
    cat(sprintf(
      "  RCT male-only filter: dropped %d of %d non-male subjects\n",
      n_before - nrow(d), n_before))
  }
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
    cd4             = d$cd40,
    cd8             = d$cd80,
    anchor_year     = 1992L,
    race            = as.integer(d$race),
    prior_art_years = d$preanti / 365.25,
    stringsAsFactors = FALSE
  )
}

build_macs <- function(cd4_band = NULL) {
  cat("Building MACS-RWD frame ...\n")
  inp <- function(f) file.path(macsDir, "sasinp", f)
  dat <- function(f) file.path(macsDir, "data",   f)

  lay_macsid   <- parse_sas_inp(inp("macsid.inp"))
  lay_section2 <- parse_sas_inp(inp("section2.inp"))
  lay_drugf1   <- parse_sas_inp(inp("drugf1.inp"))
  lay_lab      <- parse_sas_inp(inp("lab_rslt.inp"))
  lay_outcome  <- parse_sas_inp(inp("outcome.inp"))
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

  hiv <- macsid$CASEID[!(macsid$STATUS02 %in% 1 | macsid$STATUS10 %in% 1)]
  drugf1   <- drugf1  [drugf1$CASEID   %in% hiv, ]
  lab      <- lab     [lab$CASEID      %in% hiv, ]
  outcome  <- outcome [outcome$CASEID  %in% hiv, ]
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

  # ----- MACS endpoint = EFS / OS-as-EFS (v5 in endpoint_comparison.R)
  # Event       = all-cause death (DEATH in {1, 2, 3, 4}).
  # Event time  = last lab-visit year. DEATH date is redacted in the
  #               public release, so last observed contact is the
  #               event-time fallback for those who died.
  # Censoring   = last lab-visit year for those who did not die.
  # Prior-AIDS exclusion is retained for cohort comparability with the
  # v3 composite and the other variants in endpoint_comparison.R.
  # Labelled EFS to match the RCT side even though structurally it is
  # OS-only on the RWD side. Death is the most reliably captured event
  # in MACS, so this gives a clean event signal at the cost of dropping
  # the CD4 and AIDS arms of the v3 composite. Mirrors
  # survival_curves.R::build_rwd (page-4 RWD endpoint).

  # ----- Optional baseline-CD4 band filter (trial-eligibility) --------
  # Mirrors ACTG175's CD4 200-500 entry criterion on the MACS side
  # (matches survival_curves.R::build_rwd). Baseline CD4 is the
  # earliest 1991-95 LEU3N reading per subject. Subjects with no
  # 1991-95 reading are dropped because trial eligibility cannot be
  # verified. base_cd4 is computed only when the filter is active --
  # the v5 endpoint no longer needs CD4-decline events, so building
  # base_cd4 unconditionally would be wasted work.
  if (!is.null(cd4_band)) {
    lab_era <- lab[lab$LDATY %in% actgEra &
                   !is.na(lab$LEU3N) & lab$LEU3N > 0, ]
    lab_era <- lab_era[order(lab_era$CASEID, lab_era$LDATY), ]
    base_cd4 <- aggregate(LEU3N ~ CASEID, data = lab_era,
                          FUN = function(x) x[1])
    names(base_cd4)[2] <- "cd4_base"
    keep_ids <- base_cd4$CASEID[!is.na(base_cd4$cd4_base) &
                                base_cd4$cd4_base >= cd4_band[1] &
                                base_cd4$cd4_base <= cd4_band[2]]
    n_before <- nrow(base)
    base <- base[base$id %in% keep_ids, ]
    cat(sprintf(
      "  MACS CD4-band filter [%d, %d]: kept %d of %d subjects\n",
      cd4_band[1], cd4_band[2], nrow(base), n_before))
  }

  oc <- outcome
  oc$aids_yr <- ifelse(oc$AIDSCASE %in% c(2, 3) & oc$DATE1yy > 0,
                       oc$DATE1yy, NA_real_)
  oc$is_dead <- !is.na(oc$DEATH) & oc$DEATH %in% c(1, 2, 3, 4)

  last_lab <- aggregate(LDATY ~ CASEID, data = lab,
                        FUN = function(x) max(x, na.rm = TRUE))
  names(last_lab)[2] <- "last_year"

  base <- merge(base, oc[, c("CASEID", "aids_yr", "is_dead")],
                by.x = "id", by.y = "CASEID", all.x = TRUE)
  base <- merge(base, last_lab,
                by.x = "id", by.y = "CASEID", all.x = TRUE)
  base$is_dead[is.na(base$is_dead)] <- FALSE

  prior_aids <- !is.na(base$aids_yr) & base$aids_yr < base$anchor_year
  base <- base[!prior_aids, ]

  # v5 event time: last lab-visit year for everyone -- the event time
  # for those who died (DEATH date is redacted; last contact is the
  # fallback) and the censoring time for those who didn't. Clipped to
  # >= 0.5 y so log_time stays finite for subjects whose anchor year
  # equals their last lab year.
  base$has_event <- base$is_dead
  base$end_year  <- base$last_year
  base$log_time  <- log(pmax(base$end_year - base$anchor_year, 0.5))
  base$status    <- as.integer(base$has_event)

  byr <- aggregate(BORNY ~ CASEID, data = section2,
                   FUN = function(x) max(x, na.rm = TRUE))
  base$age <- base$anchor_year - byr$BORNY[match(base$id, byr$CASEID)]

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
    age = base$age, cd4 = base$cd4, cd8 = base$cd8,
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

cat("\n", strrep("=", 70),
    "\n  ASSEMBLE (default + trial-aligned cohorts)\n",
    strrep("=", 70), "\n", sep = "")

cat("\n-- Default cohort --\n")
rct_default  <- build_rct()
macs_default <- build_macs()
cat(sprintf("RCT  n=%d  treat0=%d  treat1=%d  events=%d\n",
            nrow(rct_default),  sum(rct_default$treat == 0),
            sum(rct_default$treat == 1), sum(rct_default$status == 1)))
cat(sprintf("MACS n=%d  treat0=%d  treat1=%d  events=%d\n",
            nrow(macs_default), sum(macs_default$treat == 0),
            sum(macs_default$treat == 1), sum(macs_default$status == 1)))

# Trial-aligned cohort: RCT male only (gender == 1) AND MACS baseline
# CD4 in [200, 500] cells/uL -- mirrors the page-4 cohort in
# survival_curves.R. MACS is already men-only by design and ACTG175 is
# already CD4-banded by protocol, so each filter only bites on one side.
cat("\n-- Trial-aligned cohort ",
    "(RCT male; MACS baseline CD4 in [200, 500]) --\n", sep = "")
rct_aligned  <- build_rct(male_only = TRUE)
macs_aligned <- build_macs(cd4_band = c(200, 500))
cat(sprintf("RCT  n=%d  treat0=%d  treat1=%d  events=%d\n",
            nrow(rct_aligned),  sum(rct_aligned$treat == 0),
            sum(rct_aligned$treat == 1), sum(rct_aligned$status == 1)))
cat(sprintf("MACS n=%d  treat0=%d  treat1=%d  events=%d\n",
            nrow(macs_aligned), sum(macs_aligned$treat == 0),
            sum(macs_aligned$treat == 1), sum(macs_aligned$status == 1)))

# ---------------------------------------------------------------------
# 3. Analysis function -- runs the full M1/M2/M3 fit + figure pipeline
#    for a given (rct, macs) frame pair. Called twice below (default
#    cohort and trial-aligned cohort) so the two versions can be
#    compared side by side. The `tag` argument is embedded in the
#    output PDF filename and the plot title.
# ---------------------------------------------------------------------

run_analysis <- function(rct, macs, tag) {

cat("\n", strrep("#", 70),
    "\n  RUN ANALYSIS: cohort = ", tag,
    "  (RCT n=", nrow(rct), ", MACS n=", nrow(macs), ")\n",
    strrep("#", 70), "\n", sep = "")

N_POST <- 7500
N_BURN <- 2500

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
    # ---- Nonparametric residual ---------------------------------------
    # source-HDP: shared atoms theta_k* via a top-level DP(gamma, H),
    # per-source weights pi_{sk} and centring mu_s. Truncation K = 30 is
    # plenty for the modest n in this analysis. Atoms are N(0, 0.5^2) on
    # the standardised residual scale; per-source mass M_s starts at 1
    # (Escobar-West updates take it from there).
    error_dist                = "source_hdp",
    error_truncation_K        = 30L,
    error_atom_scale          = 0.5,
    error_mass_init           = 1.0,
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

proj_basis <- ~ age + cd4 + cd8 + race + prior_art_years
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
# 9d. HDP error-distribution diagnostics (M2 only)
#
# MixtureDP group convention is group 0 = RWD, group 1 = RCT. fit_M2
# carries (when error_dist = "source_hdp"):
#   fit$dp_K                  truncation K (scalar)
#   fit$dp_mix_prop[[g]]      R x K  posterior stick weights pi_{g,k}
#   fit$dp_locations[[g]]     R x K  atom locations theta_{g,k}
#                             (theta_k* + mu_g on response scale)
#   fit$dp_locations_shared   R x K  top-level atoms theta_k*
#   fit$dp_beta               R x K  top-level stick weights
#   fit$dp_mass[[g]]          R-vec  per-source concentration M_g
#   fit$dp_gamma              R-vec  top-level concentration gamma
#   fit$dp_mu_g[[g]]          R-vec  per-source centring mu_g
#   fit$sigma                 R-vec  residual SD on standardised scale
# ---------------------------------------------------------------------

stopifnot(!is.null(fit_M2$dp_mass), !is.null(fit_M2$dp_gamma))
hdp_K <- fit_M2$dp_K
# FusionForest stores sigma on the standardised internal scale (per
# fit$meta$sigma_hat) but rescales dp_locations to the response scale.
# Multiply sigma by sigma_hat to put both on the log-time scale for the
# implied-density plot below; mu_g, atoms, and predictions are already
# on the response scale.
sigma_resp <- fit_M2$sigma * fit_M2$meta$sigma_hat
hdp   <- list(
  mass_rwd    = fit_M2$dp_mass[[1]],
  mass_rct   = fit_M2$dp_mass[[2]],
  gamma_top  = fit_M2$dp_gamma,
  mu_rwd      = fit_M2$dp_mu_g[[1]],
  mu_rct     = fit_M2$dp_mu_g[[2]],
  pi_rwd      = fit_M2$dp_mix_prop[[1]],
  pi_rct     = fit_M2$dp_mix_prop[[2]],
  loc_rwd     = fit_M2$dp_locations[[1]],
  loc_rct    = fit_M2$dp_locations[[2]],
  loc_shared = fit_M2$dp_locations_shared,
  sigma      = sigma_resp)

# Occupied component counts -- a component is "active" at iteration r if
# its posterior weight exceeds 1 / (2K) (matches the rule in
# examples/NP_error_test.R). Trace shows how many residual atoms the
# model is actually using.
occ_threshold <- 1 / (2 * hdp_K)
occ_rwd  <- rowSums(hdp$pi_rwd  > occ_threshold)
occ_rct <- rowSums(hdp$pi_rct > occ_threshold)

cat("\n", strrep("=", 70),
    "\n  HDP diagnostics (M2)\n",
    strrep("=", 70), "\n", sep = "")
cat(sprintf("  truncation K              : %d\n", hdp_K))
cat(sprintf("  top-level gamma           : %.3f  (95%% CrI %.3f, %.3f)\n",
            mean(hdp$gamma_top),
            quantile(hdp$gamma_top, .025),
            quantile(hdp$gamma_top, .975)))
cat(sprintf("  per-source mass M_OS      : %.3f  (95%% CrI %.3f, %.3f)\n",
            mean(hdp$mass_rwd),
            quantile(hdp$mass_rwd, .025),
            quantile(hdp$mass_rwd, .975)))
cat(sprintf("  per-source mass M_RCT     : %.3f  (95%% CrI %.3f, %.3f)\n",
            mean(hdp$mass_rct),
            quantile(hdp$mass_rct, .025),
            quantile(hdp$mass_rct, .975)))
cat(sprintf("  per-source mu_OS          : %+0.3f (95%% CrI %+0.3f, %+0.3f)\n",
            mean(hdp$mu_rwd),
            quantile(hdp$mu_rwd, .025),
            quantile(hdp$mu_rwd, .975)))
cat(sprintf("  per-source mu_RCT         : %+0.3f (95%% CrI %+0.3f, %+0.3f)\n",
            mean(hdp$mu_rct),
            quantile(hdp$mu_rct, .025),
            quantile(hdp$mu_rct, .975)))
cat(sprintf("  occupied K (mean)         : RWD = %.2f,  RCT = %.2f\n",
            mean(occ_rwd), mean(occ_rct)))
cat(sprintf("  sigma (log-time scale)    : %.3f  (95%% CrI %.3f, %.3f)\n",
            mean(sigma_resp),
            quantile(sigma_resp, .025),
            quantile(sigma_resp, .975)))

# Trace plot helper: param vector vs iteration with a 100-iter moving
# average to make the level easier to read.
plot_hdp_trace <- function(v, title, ylab = "value") {
  df <- data.frame(iter = seq_along(v), value = v)
  ggplot(df, aes(x = iter, y = value)) +
    geom_line(colour = "grey55", linewidth = 0.3) +
    geom_smooth(method = "loess", span = 0.2,
                se = FALSE, colour = "steelblue", linewidth = 0.6) +
    labs(x = "MCMC iteration (post burn-in)", y = ylab, title = title) +
    theme_panel()
}

p_hdp_gamma <- plot_hdp_trace(
  hdp$gamma_top, "HDP top-level concentration gamma",
  ylab = expression(gamma))
p_hdp_mass  <- {
  df <- rbind(
    data.frame(iter = seq_along(hdp$mass_rwd),
               value = hdp$mass_rwd, source = "RWD"),
    data.frame(iter = seq_along(hdp$mass_rct),
               value = hdp$mass_rct, source = "RCT"))
  ggplot(df, aes(x = iter, y = value, colour = source)) +
    geom_line(linewidth = 0.4, alpha = 0.6) +
    scale_colour_manual(values = c(RWD = "firebrick", RCT = "steelblue")) +
    labs(x = "MCMC iteration", y = expression(M[s]),
         colour = NULL, title = "Per-source concentration M_s") +
    theme_panel()
}
p_hdp_occ <- {
  df <- rbind(
    data.frame(iter = seq_along(occ_rwd),  value = occ_rwd,  source = "RWD"),
    data.frame(iter = seq_along(occ_rct), value = occ_rct, source = "RCT"))
  ggplot(df, aes(x = iter, y = value, colour = source)) +
    geom_line(linewidth = 0.4, alpha = 0.6) +
    scale_colour_manual(values = c(RWD = "firebrick", RCT = "steelblue")) +
    labs(x = "MCMC iteration",
         y = sprintf("occupied K (threshold = 1/(2K) = %.3f)",
                     occ_threshold),
         colour = NULL,
         title = sprintf("Occupied clusters per source (truncation K=%d)",
                         hdp_K)) +
    theme_panel()
}

# Posterior residual density per source on the response (log-time)
# scale. At MCMC iteration r the implied density of the residual is
#   f_s,r(e) = sum_k pi_{s,k,r} * N(e; theta_{s,k,r}, sigma_r).
# theta_{s,k,r} is the wrapper-rescaled atom location (response scale);
# sigma_r is fit$sigma * fit$meta$sigma_hat, also on the response
# scale. We thin the chain to ~200 draws, evaluate on a fixed grid,
# and summarise pointwise (posterior mean + 95% pointwise CrI).
hdp_resid_density <- function(loc_mat, pi_mat, sigma_v, grid) {
  R <- nrow(loc_mat)
  thin_idx <- as.integer(seq(1L, R, length.out = min(R, 200L)))
  M <- length(thin_idx)
  out <- matrix(0, M, length(grid))
  for (m in seq_len(M)) {
    r   <- thin_idx[m]
    pi  <- pi_mat[r, ]
    th  <- loc_mat[r, ]
    sg  <- sigma_v[r]
    # dnorm vectorised over grid; sum across active components.
    keep <- pi > 1e-8
    if (!any(keep)) next
    contribs <- outer(grid, th[keep], function(g, t) dnorm(g, t, sg))
    out[m, ] <- as.numeric(contribs %*% pi[keep])
  }
  data.frame(
    grid  = grid,
    mean  = colMeans(out),
    lo    = apply(out, 2, quantile, probs = 0.025),
    hi    = apply(out, 2, quantile, probs = 0.975)
  )
}

grid_eps <- seq(-3, 3, length.out = 200) * mean(hdp$sigma)
dens_rwd  <- hdp_resid_density(hdp$loc_rwd,  hdp$pi_rwd,  hdp$sigma, grid_eps)
dens_rct <- hdp_resid_density(hdp$loc_rct, hdp$pi_rct, hdp$sigma, grid_eps)
dens_df  <- rbind(cbind(dens_rwd,  source = "RWD"),
                  cbind(dens_rct, source = "RCT"))

# Gaussian-baseline reference: N(0, mean(sigma)^2).
ref_df <- data.frame(
  grid  = grid_eps,
  value = dnorm(grid_eps, 0, mean(hdp$sigma)))

p_hdp_dens <- ggplot(dens_df,
                     aes(x = grid, y = mean, colour = source, fill = source)) +
  geom_ribbon(aes(ymin = lo, ymax = hi), alpha = 0.18, colour = NA) +
  geom_line(linewidth = 0.7) +
  geom_line(data = ref_df, aes(x = grid, y = value),
            inherit.aes = FALSE,
            colour = "grey30", linetype = "dashed", linewidth = 0.5) +
  scale_colour_manual(values = c(RWD = "firebrick", RCT = "steelblue")) +
  scale_fill_manual(values = c(RWD = "firebrick", RCT = "steelblue")) +
  labs(x = "Residual (log-survival scale)",
       y = "Posterior density",
       colour = NULL, fill = NULL,
       title = "Implied per-source residual density (HDP)",
       subtitle = "Dashed grey = Gaussian baseline N(0, mean(sigma)^2)") +
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
         p_proj /
         (p_hdp_gamma | p_hdp_mass | p_hdp_occ) /
         p_hdp_dens +
  plot_annotation(
    title    = sprintf(
      "ACTG175 vs ACTG175+MACS fusion (HDP) vs MACS only -- %s cohort",
      tag),
    subtitle = paste("M1 = RCT only (Gaussian),",
                     "M2 = RCT + MACS fusion (source_hdp residual),",
                     "M3 = MACS only (Gaussian)"))

out_pdf <- sprintf("data/analysis 0/fusion_actg175_macs_hdp_%s.pdf", tag)
ggsave(out_pdf, panel, width = 18, height = 36)
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

list(fit_M1 = fit_M1, fit_M2 = fit_M2, fit_M3 = fit_M3,
     rct = rct, macs = macs, panel = panel, tag = tag)

}  # end run_analysis()

# ---------------------------------------------------------------------
# 11. Run both analyses (default cohort + trial-aligned cohort)
#     so the two versions can be compared side by side.
# ---------------------------------------------------------------------

res_default <- run_analysis(rct_default, macs_default, "default")
res_aligned <- run_analysis(rct_aligned, macs_aligned, "aligned")

invisible(list(default = res_default, aligned = res_aligned))
