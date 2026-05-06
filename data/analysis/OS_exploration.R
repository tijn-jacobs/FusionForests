## OS_exploration.R
##
## First-pass screen of three candidate observational HIV cohorts as
## potential OS arms for fusing with the ACTG175 RCT (FusionForest).
##
##   MACS  PDS  -- men-only, baseline visits 1991-1995  (primary)
##   WIHS  PDS  -- women-only, baseline 1994-1995       (sensitivity)
##   MWCCS PDS  -- 2019-2023, no 1990s ART data         (skipped)
##
## Goal: tabulate treatment, outcome and baseline covariate availability
## in each cohort and check whether a "ZDV-monotherapy vs combo" contrast
## with non-trivial cell counts is reachable in 1991-1995. No fusion
## model is fit here -- design choices are listed at the bottom for the
## user to confirm before proceeding.
##
## Run from project root:
##   Rscript data/analysis/OS_exploration.R

setwd("~/Library/CloudStorage/OneDrive-VrijeUniversiteitAmsterdam/Documents/GitHub/FusionForests/")

suppressPackageStartupMessages({
  library(speff2trial)
  library(survival)
})

proj      <- "."
macsDir   <- file.path(proj, "data", "MACS PDS")
wihsDir   <- file.path(proj, "data", "WIHS PDS")
actgEra   <- 1991:1995

banner <- function(x) {
  cat("\n", strrep("=", 70), "\n", x, "\n",
      strrep("=", 70), "\n", sep = "")
}

# ---------------------------------------------------------------------
# 0. SAS .inp parser + fast fixed-width reader for MACS .dat files
# ---------------------------------------------------------------------

parse_sas_inp <- function(path) {
  ln <- readLines(path, warn = FALSE)
  ln <- gsub("/\\*.*?\\*/", "", ln)
  ln <- trimws(ln)
  rows <- list()
  for (s in ln) {
    if (s == "" || s == ";" || toupper(s) == "INPUT") next
    # Form A: @start NAME W.D     (implied decimal)
    m <- regmatches(s, regexec(
      "^@\\s*([0-9]+)\\s+([A-Za-z_][A-Za-z0-9_]*)\\s+([0-9]+)\\.([0-9]+)\\s*$",
      s))[[1]]
    if (length(m) == 5) {
      st <- as.integer(m[2]); wd <- as.integer(m[4])
      rows[[length(rows) + 1L]] <- data.frame(
        name = m[3], start = st, end = st + wd - 1L,
        decimals = as.integer(m[5]),
        stringsAsFactors = FALSE)
      next
    }
    # Form B: NAME start-end       (range; optional $ prefix)
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
    # Form C: NAME start            (single column; optional $ prefix)
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

# ---------------------------------------------------------------------
# 1. ACTG175 reference (RCT)
# ---------------------------------------------------------------------

banner("1. ACTG175 RCT reference")

data(ACTG175)
cat("n =", nrow(ACTG175), "; vars =", ncol(ACTG175), "\n\n")
cat("treat (binary) ----------------------------------\n")
print(table(treat = ACTG175$treat, useNA = "ifany"))
cat("\narms (4-arm) ------------------------------------\n")
print(table(arms = ACTG175$arms, useNA = "ifany"))
cat("\ncens (event indicator) --------------------------\n")
print(table(cens = ACTG175$cens, useNA = "ifany"))
cat("\ndays (follow-up) summary ------------------------\n")
print(summary(ACTG175$days))
cat("\nKey baseline covariates -------------------------\n")
print(summary(ACTG175[, c("age", "wtkg", "karnof", "cd40", "cd80",
                          "race", "gender", "homo", "drugs",
                          "hemo", "symptom", "z30", "zprior")]))

# ---------------------------------------------------------------------
# 2. MACS ingestion
# ---------------------------------------------------------------------

banner("2. MACS PDS  (primary observational candidate)")

inp <- function(f) file.path(macsDir, "sasinp", f)
dat <- function(f) file.path(macsDir, "data",   f)

cat("Parsing SAS layouts ...\n")
lay_macsid   <- parse_sas_inp(inp("macsid.inp"))
lay_section2 <- parse_sas_inp(inp("section2.inp"))
lay_drugf1   <- parse_sas_inp(inp("drugf1.inp"))
lay_lab      <- parse_sas_inp(inp("lab_rslt.inp"))
lay_outcome  <- parse_sas_inp(inp("outcome.inp"))

cat("Reading macsid (subject-level) ... ")
macsid <- read_macs_subset(
  dat("macsid.dat"), lay_macsid,
  c("CASEID", "MACSCODE", "STATUS02", "STATUS10"))
cat(nrow(macsid), "rows\n")

# HIV+ subjects: drop seronegative-only recruits.
# STATUS02 == 1 or STATUS10 == 1 flags seronegative recruits in the
# 2001-03 / 2010 expansions; original-cohort & 1987-90 cohort serostatus
# is encoded in MACSCODE. The macsid codebook uses MACSCODE values where
# 20s/30s indicate cohort+positivity. To stay defensive, simply drop
# rows where any seronegative flag is set; users can refine later.
hiv_pos <- macsid[!(macsid$STATUS02 %in% 1 | macsid$STATUS10 %in% 1), ]
cat("After dropping known seronegative recruits: ", nrow(hiv_pos), "\n")

cat("Reading section2 (baseline + demographics) ... ")
section2 <- read_macs_subset(
  dat("section2.dat"), lay_section2,
  c("CASEID", "VISIT", "DAT2Y", "BORNY", "RACE"))
cat(nrow(section2), "rows\n")

cat("Reading drugf1 (antivirals) ... ")
drugf1 <- read_macs_subset(
  dat("drugf1.dat"), lay_drugf1,
  c("CASEID", "VISIT", "AVQY", "DRGAV", "AVNW"))
cat(nrow(drugf1), "rows\n")

cat("Reading lab_rslt (CD4/CD8) ... ")
lab <- read_macs_subset(
  dat("lab_rslt.dat"), lay_lab,
  c("CASEID", "VISIT", "LDATY", "LEU3N", "LEU3P", "LEU2N", "LEU2P"))
cat(nrow(lab), "rows\n")

cat("Reading outcome (AIDS / death / CD4 decline) ... ")
# NB: outcome.inp places DEATH at col 937, but the actual data file is
# shifted +52 cols and DEATH lives at col 989 (verified empirically:
# col 989 holds codes 1/2/3/4 matching the codebook
# 1=AIDS prior dx, 2=AIDS no prior dx, 3=Not AIDS, 4=Unknown).
# DTHDATEmm/DTHDATEyy appear to be redacted entirely from the public
# release -- no 4-digit year is present anywhere in cols 985-1080.
# We therefore read a death INDICATOR but not a death DATE.
lay_outcome_fixed <- lay_outcome
lay_outcome_fixed$start[lay_outcome_fixed$name == "DEATH"] <- 989
lay_outcome_fixed$end  [lay_outcome_fixed$name == "DEATH"] <- 989
outcome <- read_macs_subset(
  dat("outcome.dat"), lay_outcome_fixed,
  c("CASEID", "AIDSCASE",
    "DATE1mm", "DATE1yy",
    "AIDSREPDmm", "AIDSREPDyy",
    "SELFCD4Dmm", "SELFCD4Dyy",
    "DEATH"))
cat(nrow(outcome), "rows\n")

# Restrict everything to HIV+ subjects we kept above
keep_id <- hiv_pos$CASEID
section2 <- section2[section2$CASEID %in% keep_id, ]
drugf1   <- drugf1  [drugf1$CASEID   %in% keep_id, ]
lab      <- lab     [lab$CASEID      %in% keep_id, ]
outcome  <- outcome [outcome$CASEID  %in% keep_id, ]

cat("\n2a. Drug-form coverage by visit-year (HIV+) ----\n")
print(table(year = drugf1$AVQY, useNA = "ifany"))

cat("\n2b. DRGAV (drug code) frequencies in 1991-1995 --\n")
drug_era <- drugf1[drugf1$AVQY %in% actgEra, ]
drg_tbl <- sort(table(drug_era$DRGAV, useNA = "ifany"), decreasing = TRUE)
print(head(drg_tbl, 25))

cat("\nKey ART codes:\n")
art_codes <- c(
  "092 = ZDV (AZT)", "094 = ddC", "147 = ddI",
  "180 = AZT+ddI blinded", "185 = AZT+ddC blinded",
  "186 = ddI+ddC blinded", "187 = AZT+ddI+ddC blinded")
cat(paste(" ", art_codes, collapse = "\n"), "\n")
target <- c(92, 94, 147, 180, 185, 186, 187)
cat("\nCount of records 1991-1995 with target codes:\n")
print(table(DRGAV = drug_era$DRGAV[drug_era$DRGAV %in% target]))

cat("\n2c. Subject-level treatment contrast in 1991-1995\n")
# For each subject, classify based on any 1991-1995 record currently on a
# specific therapy (AVNW == 2 means "currently taking" per the codebook).
on_now <- drug_era[!is.na(drug_era$AVNW) & drug_era$AVNW == 2, ]
zdv_subj <- unique(on_now$CASEID[on_now$DRGAV == 92])
ddi_subj <- unique(on_now$CASEID[on_now$DRGAV == 147])
ddc_subj <- unique(on_now$CASEID[on_now$DRGAV == 94])
combo_subj <- unique(on_now$CASEID[on_now$DRGAV %in% c(180, 185, 186, 187)])
any_other_art <- unique(c(ddi_subj, ddc_subj, combo_subj))

zdv_only <- setdiff(zdv_subj, any_other_art)
not_zdv_only_art <- union(any_other_art,
                          intersect(zdv_subj, any_other_art))
cat("On any ZDV record (current):       ", length(zdv_subj), "\n")
cat("On ddI / ddC / dual / triple combo:", length(any_other_art), "\n")
cat("ZDV-monotherapy-only candidates:   ", length(zdv_only), "\n")
cat("Combo / non-ZDV-mono candidates:   ", length(not_zdv_only_art),
    "\n")

cat("\n2d. Baseline CD4/CD8 availability in 1991-1995 -\n")
lab_era <- lab[lab$LDATY %in% actgEra, ]
cat("Lab records with LDATY in 1991-1995:", nrow(lab_era), "\n")
cat("Subjects with >=1 CD4 in this window:",
    length(unique(lab_era$CASEID[!is.na(lab_era$LEU3N)])), "\n")
cat("Subjects with >=1 CD8 in this window:",
    length(unique(lab_era$CASEID[!is.na(lab_era$LEU2N)])), "\n")

cat("\n2e. Outcomes available --------------------------\n")
n_aids  <- sum(outcome$AIDSCASE %in% c(2, 3), na.rm = TRUE)
n_death <- sum(outcome$DEATH    %in% c(1, 2), na.rm = TRUE)
n_cd4dec <- sum(!is.na(outcome$SELFCD4Dyy) & outcome$SELFCD4Dyy > 0)
cat("Subjects with AIDS diagnosis  (AIDSCASE in 2,3):", n_aids,  "\n")
cat("Subjects with death recorded  (DEATH in 1,2):   ", n_death, "\n")
cat("Subjects with CD4-decline date (<200 / <14%):   ", n_cd4dec, "\n")
cat("AIDS dx year range: ",
    min(outcome$DATE1yy[outcome$DATE1yy > 0], na.rm = TRUE), "-",
    max(outcome$DATE1yy[outcome$DATE1yy > 0], na.rm = TRUE), "\n")
cat("Death year: redacted from MACS public release\n")
cat("Breakdown of DEATH codes:\n")
print(table(DEATH = outcome$DEATH, useNA = "ifany"))

cat("\n2f. MACS fusion-readiness summary --------------\n")
macs_summary <- data.frame(
  metric = c("HIV+ subjects (after exclusions)",
             "ZDV-mono candidates 1991-1995",
             "Non-ZDV-mono ART candidates 1991-1995",
             "Baseline CD4 in 1991-1995",
             "AIDS events recorded (any year)",
             "Death events recorded (any year)"),
  n = c(nrow(hiv_pos), length(zdv_only),
        length(not_zdv_only_art),
        length(unique(lab_era$CASEID[!is.na(lab_era$LEU3N)])),
        n_aids, n_death))
print(macs_summary, row.names = FALSE)

# ---------------------------------------------------------------------
# 3. WIHS ingestion (lighter pass)
# ---------------------------------------------------------------------

banner("3. WIHS PDS  (sensitivity / female-subgroup candidate)")

wcsv <- function(f) file.path(wihsDir, "CSV", f)

cat("Reading WIHS CSVs ...\n")
w_f01     <- read.csv(wcsv("f01.csv"),     stringsAsFactors = FALSE)
w_f02     <- read.csv(wcsv("f02.csv"),     stringsAsFactors = FALSE)
w_f07     <- read.csv(wcsv("f07.csv"),     stringsAsFactors = FALSE)
w_drug1   <- read.csv(wcsv("drug1.csv"),   stringsAsFactors = FALSE)
w_labsum  <- read.csv(wcsv("labsum.csv"),  stringsAsFactors = FALSE)
w_outcome <- read.csv(wcsv("outcome.csv"), stringsAsFactors = FALSE)

cat("f01 (sociodemog) :", nrow(w_f01), "rows,", length(unique(w_f01$CASEID)),
    "subjects\n")
cat("f02 (medical hx) :", nrow(w_f02), "rows\n")
cat("f07 (phys exam)  :", nrow(w_f07), "rows\n")
cat("drug1 (antivir)  :", nrow(w_drug1), "rows\n")
cat("labsum (labs)    :", nrow(w_labsum), "rows\n")
cat("outcome          :", nrow(w_outcome), "rows\n")

cat("\n3a. WIHS visit-1 calendar coverage --------------\n")
# IDTESD in the public release is the interview YEAR (4-digit);
# the full date is redacted. VERSSD holds the form-version date.
v1 <- w_f01[w_f01$VISIT == 1, ]
v1_year <- suppressWarnings(as.numeric(v1$IDTESD))
print(table(v1_year, useNA = "ifany"))

cat("\n3b. ART codes 1991-1995 in drug1 ---------------\n")
art_era_w <- w_drug1[!is.na(w_drug1$BGNYD1) &
                     w_drug1$BGNYD1 %in% actgEra, ]
cat("Records with start-year in 1991-1995:", nrow(art_era_w), "\n")
art_codes_w <- sort(table(art_era_w$DRUGD1), decreasing = TRUE)
print(head(art_codes_w, 15))

cat("\n3c. Baseline lab availability (visit 1) --------\n")
lab_v1 <- w_labsum[w_labsum$VISIT == 1, ]
cat("Visit 1 labsum rows:", nrow(lab_v1), "\n")
cat("Subjects w/ baseline CD4N:",
    length(unique(lab_v1$CASEID[!is.na(lab_v1$CD4N) & lab_v1$CD4N > 0])),
    "\n")
cat("Subjects w/ baseline CD8N:",
    length(unique(lab_v1$CASEID[!is.na(lab_v1$CD8N) & lab_v1$CD8N > 0])),
    "\n")
cat("Subjects w/ baseline VLOAD:",
    length(unique(lab_v1$CASEID[!is.na(lab_v1$VLOAD) & lab_v1$VLOAD > 0])),
    "\n")

cat("\n3d. Outcome events in WIHS (year granularity) --\n")
cat("Total outcome rows :", nrow(w_outcome), "\n")
cat("AIDS events        :", sum(w_outcome$OUTCAIDS == 1, na.rm = TRUE),
    "\n")
cat("Deaths             :", sum(w_outcome$OUTCDTH  == 1, na.rm = TRUE),
    "\n")
cat("Date-report year range:",
    min(w_outcome$DATEREPT[w_outcome$DATEREPT > 0], na.rm = TRUE), "-",
    max(w_outcome$DATEREPT[w_outcome$DATEREPT > 0], na.rm = TRUE), "\n")

# ---------------------------------------------------------------------
# 4. Covariate-alignment table
# ---------------------------------------------------------------------

banner("4. Covariate alignment ACTG175 vs MACS vs WIHS")

align <- data.frame(
  variable = c("Age", "Weight (wtkg)", "Race", "Gender",
               "Homosexual contact", "IV-drug use",
               "Hemophilia", "Symptomatic HIV",
               "Karnofsky score", "Prior ZDV (zprior/z30)",
               "Baseline CD4", "Baseline CD8",
               "HIV viral load",
               "Treatment contrast (ZDV-mono vs combo)",
               "Outcome event (AIDS/death/CD4 decline)",
               "Time-to-event resolution"),
  ACTG175 = c("yes", "yes", "yes", "yes (mostly M)",
              "yes (homo)", "yes (drugs)", "yes (hemo)",
              "yes", "yes", "yes",
              "yes (cd40)", "yes (cd80)", "no",
              "randomised binary",
              "yes (cens)", "days"),
  MACS = c("yes (BORNY)", "yes (phy_exam:WTIN2)", "yes (RACE)",
           "men only", "men only -- proxy", "needs section3 lookup",
           "absent (typical MACS)", "needs PWA/symptom form",
           "absent (use proxy)",
           "yes (drugf1: prior ZDV via AVSY/AVRSY)",
           "yes (LEU3N)", "yes (LEU2N)", "yes (VLOAD post-1996)",
           "constructed from drugf1 codes",
           "yes (AIDS+death+CD4decline)", "month"),
  WIHS = c("yes (DOB_PE)", "yes (PCWTPE)", "yes (RACE in f01 derivatives)",
           "women only -- mismatch",
           "n/a (women only)", "yes (f04: drug behaviour)",
           "absent", "yes (f02 SY* fields)", "yes (f27)",
           "yes (drug1 BGNYD1/CURRD1)", "yes (CD4N)", "yes (CD8N)",
           "yes (VLOAD)",
           "constructed from drug1 codes",
           "yes (OUTCAIDS, OUTCDTH)", "year")
)
print(align, row.names = FALSE)

# ---------------------------------------------------------------------
# 5. Descriptive KM in MACS subset (purely diagnostic)
# ---------------------------------------------------------------------

banner("5. Descriptive KM: ZDV-mono vs combo in MACS 1991-1995")

# Anchor each candidate subject to the earliest 1991-1995 drugf1 record
# of their assigned arm; use year of AIDS/death (whichever earlier) as
# event date and BIRTHDTY-anchored last-known visit year for censoring.
zdv_df  <- data.frame(CASEID = zdv_only,         arm = 0L)
combo_df <- data.frame(CASEID = setdiff(not_zdv_only_art, zdv_only),
                       arm = 1L)
cohort  <- rbind(zdv_df, combo_df)
cat("Candidate cohort: n =", nrow(cohort),
    "(", sum(cohort$arm == 0), "ZDV-mono /",
    sum(cohort$arm == 1), "combo)\n")

if (nrow(cohort) >= 50) {
  # Anchor year = first AVQY in 1991-1995 for that subject
  anchor <- aggregate(AVQY ~ CASEID, data = drug_era, FUN = min)
  names(anchor)[2] <- "anchor_year"
  cohort <- merge(cohort, anchor, by = "CASEID", all.x = TRUE)

  # Event year: earliest of AIDS dx or CD4-decline (year resolution).
  # Death dates are redacted in MACS public release, so death only
  # contributes as an event flag without a date -- treated as occurring
  # at the last lab year if no AIDS/CD4-decline date is available.
  oc <- outcome[outcome$CASEID %in% cohort$CASEID, ]
  oc$aids_year <- ifelse(oc$AIDSCASE %in% c(2, 3) & oc$DATE1yy > 0,
                         oc$DATE1yy, NA)
  oc$cd4dec_year <- ifelse(oc$SELFCD4Dyy > 0, oc$SELFCD4Dyy, NA)
  oc$event_year <- pmin(oc$aids_year, oc$cd4dec_year, na.rm = TRUE)
  oc$is_dead <- oc$DEATH %in% c(1, 2)
  oc$has_event <- !is.na(oc$event_year) | oc$is_dead
  cohort <- merge(cohort,
                  oc[, c("CASEID", "event_year", "has_event", "is_dead")],
                  by = "CASEID", all.x = TRUE)

  # Censor year: last lab visit year if no event
  last_lab <- aggregate(LDATY ~ CASEID, data = lab,
                        FUN = function(x) max(x, na.rm = TRUE))
  names(last_lab)[2] <- "last_year"
  cohort <- merge(cohort, last_lab, by = "CASEID", all.x = TRUE)

  # If subject is dead but has no AIDS/CD4-decline year, use last lab
  # year as the (rough) event time.
  cohort$end_year <- ifelse(!is.na(cohort$event_year),
                            cohort$event_year, cohort$last_year)
  cohort$status <- as.integer(!is.na(cohort$has_event) & cohort$has_event)
  cohort$time_yr <- pmax(cohort$end_year - cohort$anchor_year, 0,
                         na.rm = TRUE)

  use <- !is.na(cohort$time_yr) & cohort$time_yr > 0
  cat("Usable rows for KM diagnostic:", sum(use), "\n")
  if (sum(use) >= 30) {
    fit <- survfit(Surv(time_yr, status) ~ arm, data = cohort[use, ])
    print(fit)
    sd  <- survdiff(Surv(time_yr, status) ~ arm, data = cohort[use, ])
    cat("\nLog-rank chi-sq =", round(sd$chisq, 2),
        " (df =", length(sd$n) - 1, ")\n")
  } else {
    cat("Too few usable rows; KM skipped.\n")
  }
} else {
  cat("Cohort too small for descriptive KM; skipped.\n")
}

# ---------------------------------------------------------------------
# 6. Open design questions (no model fit yet)
# ---------------------------------------------------------------------

banner("6. Open design questions for the user")

cat("
The following choices need confirmation before building the OS frame
and calling FusionForest:

  (a) OUTCOME definition.
      ACTG175 cens = composite (death OR AIDS OR >=50% CD4 decline).
      In MACS we have all three components at month resolution
      (DTHDATEmm/yy, DATE1mm/yy, SELFCD4Dmm/yy). In WIHS only AIDS
      and death are reliably present, at year resolution. Match the
      composite, or simplify to death-only?

  (b) BASELINE-VISIT ANCHOR for MACS.
      Single year (e.g. 1991) vs first 1991-1995 record per subject
      vs first record after seroconversion. Affects covariate values
      and follow-up duration.

  (c) TREATMENT DICHOTOMY.
      ACTG175 'treat' contrasts ZDV-only vs (ZDV+ddI / ZDV+ddC / ddI).
      In MACS we currently classify by AVNW=='currently taking' with
      DRGAV codes 092 (ZDV), 094 (ddC), 147 (ddI), 180/185/186/187
      (combos). Subjects on multiple regimens within 1991-1995 need
      a rule (last regimen? first regimen? majority time?).

  (d) WHETHER TO INCLUDE WIHS.
      Gender mismatch (women vs ACTG175's mostly-men) makes effect
      transfer fragile. Could be (i) excluded, (ii) used as a
      gender-restricted sensitivity arm, (iii) included as OS with
      gender added as a covariate and source-effect absorbing the gap.

  (e) HANDLING TREATMENT SWITCHERS.
      Subjects who switch from ZDV-mono to a combo within follow-up
      can be (i) classified by intention-to-treat at anchor, (ii) by
      time-varying exposure (out of scope for FusionForest), or
      (iii) excluded.

  (f) MWCCS role.
      Currently dropped. If the user wants long-term (post-2019)
      survival follow-up of historic MACS/WIHS subjects, MWCCS can be
      linked via pubid (drop first digit). Not relevant for a 1991-
      1995 fusion analysis but flagged for completeness.
")

invisible(NULL)
