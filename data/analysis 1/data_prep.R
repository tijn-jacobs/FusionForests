################################################################################
## data/analysis 1/data_prep.R
##
## Self-contained data-construction helpers for the ACTG175 + MACS fusion
## analysis. Builds two frames:
##   - rct  : ACTG175 trial subjects (male-only, trial-aligned cohort)
##   - macs : MACS observational cohort (HIV+, ART in 1991-95, ZDV vs
##            combo/ddI contrast, baseline CD4 in [200, 500], AIDS-free
##            at anchor)
## Endpoint:
##   RCT  : published EFS composite `cens` (>=50% CD4 decline OR AIDS
##          OR all-cause death; days = follow-up).
##   MACS : v5 (all-cause death only) with last-lab-year as event-time
##          fallback. See data/analysis 0/endpoint_comparison.R for the
##          full variant comparison.
## Treatment contrast (matches ACTG175$treat):
##   0 = ZDV monotherapy
##   1 = ZDV+ddI OR ZDV+ddC OR ddI monotherapy
##   excluded: ddC mono, ddI+ddC (186), AZT+ddI+ddC (187)
## Harmonised covariates (six): age, cd4, cd8, anchor_year, race,
##                              prior_art_years.
##
## Source this file from analysis.R; it defines (but does not execute)
## build_rct() and build_macs() plus helpers.  Set `macsDir` before
## sourcing if the package layout differs from the default.
################################################################################

setwd("~/Library/CloudStorage/OneDrive-VrijeUniversiteitAmsterdam/Documents/GitHub/FusionForests")

suppressPackageStartupMessages({
  library(speff2trial)
})

## ---- Project paths and constants -------------------------------------------
## `proj` collides with stats::proj() in the search path, so we restrict
## the existence check to a character variable.
if (!exists("proj",    mode = "character")) proj    <- "."
if (!exists("macsDir", mode = "character")) macsDir <- file.path(proj, "data", "MACS PDS")
actgEra    <- 1991:1995
artCodes   <- c(92, 94, 147, 180, 185, 186, 187)
harmCovars <- c("age", "cd4", "cd8",
                "anchor_year", "race", "prior_art_years")

## ---- SAS .inp layout parser ------------------------------------------------
## Reads a SAS fixed-width input layout and returns a data.frame with one
## row per variable (name, start, end, decimals).  Handles numeric @POS
## directives and character $POS-POS / $POS ranges.
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

## ---- MACS treatment classifier ---------------------------------------------
## Per-subject classification given the set of antiretroviral drug codes
## observed in their 1991-95 records.  Mirrors the ACTG175 binary contrast.
classify_treat <- function(drg_set) {
  art <- intersect(drg_set, artCodes)
  if (length(art) == 0)             return(NA_integer_)
  if (any(art %in% c(186, 187)))    return(NA_integer_)
  if (identical(sort(art), 94))     return(NA_integer_)   # ddC mono
  if (identical(sort(art), 92))     return(0L)            # ZDV mono
  return(1L)
}

## ---- Build ACTG175 (RCT) frame --------------------------------------------
## With male_only = TRUE this returns the page-2 trial-aligned cohort:
## ACTG175 subjects with gender == 1 (drops ~15%).  Six harmonised
## covariates are populated:
##   anchor_year     constant 1992L (ACTG175 enrolled Dec 1991 - Oct 1992;
##                   the public release has no per-subject enrolment date).
##   race            ACTG175 codes 0 = White, 1 = non-White; pass through.
##   prior_art_years preanti (days of prior ART) / 365.25.
build_rct <- function(male_only = TRUE) {
  data(ACTG175)
  d <- ACTG175
  if (male_only) {
    n_before <- nrow(d)
    d <- d[!is.na(d$gender) & d$gender == 1, ]
    cat(sprintf("  RCT male-only filter: kept %d of %d subjects\n",
                nrow(d), n_before))
  }
  ## Trivial IC columns so RCT and MACS frames share the same schema and
  ## can be rbind()'d.  RCT events / right-censoring are at day resolution
  ## and already point-identified, so under the IC encoding RCT subjects
  ## inherit their published status with a zero-width interval centred at
  ## log_time (ic_indicator = 0; observed_right_time = log_time serves as
  ## the right-censoring lower bound when status_ic = 0).
  log_time <- log(d$days / 365.25)
  data.frame(
    id              = as.character(d$pidnum),
    source          = "RCT",
    treat           = as.integer(d$treat),
    log_time        = log_time,
    status          = as.integer(d$cens),
    obs_left_log_t  = log_time,
    obs_right_log_t = log_time,
    ic_indicator    = 0L,
    status_ic       = as.integer(d$cens),
    age             = d$age,
    cd4             = d$cd40,
    cd8             = d$cd80,
    anchor_year     = 1992L,
    race            = as.integer(d$race),
    prior_art_years = d$preanti / 365.25,
    stringsAsFactors = FALSE
  )
}

## ---- Build MACS (RWD) frame -----------------------------------------------
## With cd4_band = c(200, 500) this returns the page-2 trial-aligned RWD
## cohort: HIV+ MACS subjects with an ART record in 1991-95, treatment
## classifiable into the ACTG175 binary contrast, baseline CD4 (earliest
## 1991-95 LEU3N) within the trial entry band, and no AIDS diagnosis prior
## to the anchor year (= earliest 1991-95 AVQY on a relevant regimen).
##
## Endpoint = v5: all-cause death (DEATH in {1, 2, 3, 4}).  Event time
## falls back to the subject's last lab-visit year because the MACS public
## release redacts DEATH dates; same value serves as censoring time for
## those who did not die.  Clipped at 0.5 y so log_time stays finite for
## subjects whose anchor == last lab.
build_macs <- function(cd4_band = c(200, 500)) {
  cat("Building MACS-RWD frame ...\n")
  inp <- function(f) file.path(macsDir, "sasinp", f)
  dat <- function(f) file.path(macsDir, "data",   f)

  lay_macsid   <- parse_sas_inp(inp("macsid.inp"))
  lay_section2 <- parse_sas_inp(inp("section2.inp"))
  lay_drugf1   <- parse_sas_inp(inp("drugf1.inp"))
  lay_lab      <- parse_sas_inp(inp("lab_rslt.inp"))
  lay_outcome  <- parse_sas_inp(inp("outcome.inp"))
  # The outcome layout file places DEATH at the wrong column in the
  # public release; this is the documented fix used in the preliminary
  # analysis (data/analysis 0/fusion_actg175_macs_hdp.R:196-197).
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

  ## --- HIV+ restriction --------------------------------------------------
  hiv <- macsid$CASEID[!(macsid$STATUS02 %in% 1 | macsid$STATUS10 %in% 1)]
  drugf1   <- drugf1  [drugf1$CASEID   %in% hiv, ]
  lab      <- lab     [lab$CASEID      %in% hiv, ]
  outcome  <- outcome [outcome$CASEID  %in% hiv, ]
  section2 <- section2[section2$CASEID %in% hiv, ]

  ## --- Anchor + treatment classification --------------------------------
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

  ## --- Trial-eligibility CD4 band ---------------------------------------
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
    cat(sprintf("  MACS CD4-band filter [%d, %d]: kept %d of %d subjects\n",
                cd4_band[1], cd4_band[2], nrow(base), n_before))
  }

  ## --- Outcome construction (v5: death-only) ----------------------------
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

  base$has_event <- base$is_dead
  base$end_year  <- base$last_year
  base$log_time  <- log(pmax(base$end_year - base$anchor_year, 0.5))
  base$status    <- as.integer(base$has_event)

  ## --- Interval-censoring encoding (v5, MACS-side) ----------------------
  ## Dead MACS subjects have their DEATH date redacted; we only know that
  ## the event happened AFTER the last lab visit and BEFORE the cohort
  ## data cutoff.  The right-censored encoding above puts the event at
  ## last_year (biased downward).  The interval-censored encoding instead
  ## sets status_ic = 0 and ic_indicator = 1, with the event known to lie
  ## in (last_year, t_max_macs] where t_max_macs is the latest lab year in
  ## the cohort plus a 1-year safety margin (so subjects whose last lab is
  ## at t_max don't produce a degenerate interval).
  ## Alive subjects remain right-censored at last_year (status_ic = 0,
  ## ic_indicator = 0); under the C++ contract observed_right_time is
  ## then the lower bound on the unknown event time.
  t_max_macs <- max(base$last_year, na.rm = TRUE) + 1
  base$obs_left_log_t  <- base$log_time
  base$obs_right_log_t <- ifelse(
    base$is_dead,
    log(pmax(t_max_macs - base$anchor_year, 0.5)),
    base$log_time)
  base$ic_indicator    <- as.integer(base$is_dead)
  base$status_ic       <- 0L

  ## --- Harmonised covariates --------------------------------------------
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

  race_d <- aggregate(RACE ~ CASEID,
                      data = section2[!is.na(section2$RACE), ],
                      FUN = function(x) x[1])
  base$race <- as.integer(
    race_d$RACE[match(base$id, race_d$CASEID)] != 1)

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
    obs_left_log_t  = base$obs_left_log_t,
    obs_right_log_t = base$obs_right_log_t,
    ic_indicator    = base$ic_indicator,
    status_ic       = base$status_ic,
    age = base$age, cd4 = base$cd4, cd8 = base$cd8,
    anchor_year = base$anchor_year,
    race = base$race,
    prior_art_years = base$prior_art_years,
    stringsAsFactors = FALSE)
  out <- out[complete.cases(out[, harmCovars]), ]
  cat(sprintf("MACS frame: %d subjects (Z=0: %d, Z=1: %d)\n",
              nrow(out), sum(out$treat == 0), sum(out$treat == 1)))
  out
}

