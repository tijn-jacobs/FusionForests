## fusion_actg175.R
##
## Survival fusion of the ACTG175 RCT with MACS and/or WIHS observational
## cohorts via FusionForest.
##
## Pipeline:
##   1. Build harmonized frames {RCT, MACS, WIHS}
##   2. KM plot per dataset (one panel each), treatment 0 vs 1
##   3. Fit FusionForest for each fusion combination
##         F0: RCT only (baseline)
##         F1: RCT + MACS
##         F2: RCT + WIHS
##         F3: RCT + MACS + WIHS
##   4. Compare ATE posteriors across the four fits
##
## Treatment contrast (matches ACTG175$treat):
##   0 = ZDV monotherapy
##   1 = ZDV+ddI  OR  ZDV+ddC  OR  ddI monotherapy
##   excluded: ddC mono, ddI+ddC (186), AZT+ddI+ddC (187)
##
## Run from the project root:
##   source("data/analysis/fusion_actg175.R")

setwd("~/Library/CloudStorage/OneDrive-VrijeUniversiteitAmsterdam/Documents/GitHub/FusionForests/")

suppressPackageStartupMessages({
  library(speff2trial)
  library(survival)
  library(survminer)
  library(FusionForests)
  library(ggplot2)
})

proj       <- "."
macsDir    <- file.path(proj, "data", "MACS PDS")
wihsDir    <- file.path(proj, "data", "WIHS PDS")
actgEra    <- 1991:1995
harmCovars <- c("age", "wtkg", "cd4", "cd8")

# ---------------------------------------------------------------------
# 0. SAS .inp parser + fast fixed-width reader (from OS_exploration.R)
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

lbToKg <- 0.453592

# ---------------------------------------------------------------------
# 1. Build RCT frame from ACTG175
# ---------------------------------------------------------------------

build_rct <- function() {
  data(ACTG175)
  d <- ACTG175
  data.frame(
    id      = as.character(d$pidnum),
    source  = "RCT",
    treat   = as.integer(d$treat),
    log_time = log(d$days / 365.25),
    status  = as.integer(d$cens),
    age     = d$age,
    wtkg    = d$wtkg,
    cd4     = d$cd40,
    cd8     = d$cd80,
    stringsAsFactors = FALSE
  )
}

# ---------------------------------------------------------------------
# 2. Build MACS-OS frame
# ---------------------------------------------------------------------

# Treatment classification helper (used by both MACS and WIHS):
# given the set of distinct ART codes a subject was currently on within
# the era, return 0 (ZDV-mono), 1 (ZDV+other / ddI-mono), or NA
# (excluded: ddC-mono, triple, no relevant ART).
classify_treat <- function(drg_set) {
  art <- intersect(drg_set, c(92, 94, 147, 180, 185, 186, 187))
  if (length(art) == 0)             return(NA_integer_)
  if (any(art %in% c(186, 187)))    return(NA_integer_)
  if (identical(sort(art), 94))     return(NA_integer_)   # ddC mono
  if (identical(sort(art), 92))     return(0L)            # ZDV mono
  return(1L)                                              # combo / ddI mono
}

build_macs <- function() {
  cat("Building MACS-OS frame ...\n")
  inp <- function(f) file.path(macsDir, "sasinp", f)
  dat <- function(f) file.path(macsDir, "data",   f)

  lay_macsid   <- parse_sas_inp(inp("macsid.inp"))
  lay_section2 <- parse_sas_inp(inp("section2.inp"))
  lay_drugf1   <- parse_sas_inp(inp("drugf1.inp"))
  lay_lab      <- parse_sas_inp(inp("lab_rslt.inp"))
  lay_outcome  <- parse_sas_inp(inp("outcome.inp"))
  lay_phy      <- parse_sas_inp(inp("phy_exam.inp"))

  # DEATH lives at col 989 in the data file, not 937 as the .inp claims
  lay_outcome$start[lay_outcome$name == "DEATH"] <- 989
  lay_outcome$end  [lay_outcome$name == "DEATH"] <- 989

  cat("  reading macsid ... ");   macsid <- read_macs_subset(
    dat("macsid.dat"), lay_macsid,
    c("CASEID", "STATUS02", "STATUS10")); cat(nrow(macsid), "\n")
  cat("  reading section2 ... "); section2 <- read_macs_subset(
    dat("section2.dat"), lay_section2,
    c("CASEID", "VISIT", "DAT2Y", "BORNY")); cat(nrow(section2), "\n")
  cat("  reading drugf1 ... ");   drugf1 <- read_macs_subset(
    dat("drugf1.dat"), lay_drugf1,
    c("CASEID", "VISIT", "AVQY", "DRGAV", "AVNW")); cat(nrow(drugf1), "\n")
  cat("  reading lab_rslt ... "); lab <- read_macs_subset(
    dat("lab_rslt.dat"), lay_lab,
    c("CASEID", "LDATY", "LEU3N", "LEU2N")); cat(nrow(lab), "\n")
  cat("  reading outcome ... ");  outcome <- read_macs_subset(
    dat("outcome.dat"), lay_outcome,
    c("CASEID", "AIDSCASE", "DATE1yy", "SELFCD4Dyy", "DEATH"))
  cat(nrow(outcome), "\n")
  cat("  reading phy_exam ... "); phy <- read_macs_subset(
    dat("phy_exam.dat"), lay_phy,
    c("CASEID", "VISIT", "PEDTY", "LDWGT")); cat(nrow(phy), "\n")

  # Filter to HIV-positive subjects
  hiv <- macsid$CASEID[!(macsid$STATUS02 %in% 1 | macsid$STATUS10 %in% 1)]
  drugf1   <- drugf1  [drugf1$CASEID   %in% hiv, ]
  lab      <- lab     [lab$CASEID      %in% hiv, ]
  outcome  <- outcome [outcome$CASEID  %in% hiv, ]
  phy      <- phy     [phy$CASEID      %in% hiv, ]
  section2 <- section2[section2$CASEID %in% hiv, ]

  # Currently-taking ART records in the ACTG175 era
  curr <- drugf1[drugf1$AVQY %in% actgEra &
                 drugf1$AVNW %in% 2, ]

  drugs_by_subj <- split(curr$DRGAV, curr$CASEID)
  anchor_yr     <- vapply(split(curr$AVQY, curr$CASEID),
                          function(x) min(x, na.rm = TRUE),
                          numeric(1))
  treat_int     <- vapply(drugs_by_subj, classify_treat, integer(1))

  base <- data.frame(
    id          = names(drugs_by_subj),
    treat       = treat_int,
    anchor_year = anchor_yr,
    stringsAsFactors = FALSE)
  base <- base[!is.na(base$treat), ]

  # Outcome: time-to-(AIDS or CD4-decline or death-flag) on year scale
  oc <- outcome
  oc$aids_yr   <- ifelse(oc$AIDSCASE %in% c(2, 3) & oc$DATE1yy > 0,
                         oc$DATE1yy, NA)
  oc$cd4dec_yr <- ifelse(oc$SELFCD4Dyy > 0, oc$SELFCD4Dyy, NA)
  oc$is_dead   <- oc$DEATH %in% c(1, 2)
  oc$ev_yr     <- pmin(oc$aids_yr, oc$cd4dec_yr, na.rm = TRUE)
  oc$has_event <- !is.na(oc$ev_yr) | oc$is_dead

  # Last-seen year: latest LDATY across labs
  last_lab <- aggregate(LDATY ~ CASEID, data = lab,
                        FUN = function(x) max(x, na.rm = TRUE))
  names(last_lab)[2] <- "last_year"

  base <- merge(base, oc[, c("CASEID", "ev_yr", "has_event")],
                by.x = "id", by.y = "CASEID", all.x = TRUE)
  base <- merge(base, last_lab,
                by.x = "id", by.y = "CASEID", all.x = TRUE)
  base$has_event[is.na(base$has_event)] <- FALSE
  base$end_year <- ifelse(!is.na(base$ev_yr), base$ev_yr, base$last_year)

  # Drop subjects with prior AIDS event before anchor (entry criterion)
  prior_aids <- !is.na(base$ev_yr) & base$ev_yr < base$anchor_year
  base <- base[!prior_aids, ]

  base$log_time <- log(pmax(base$end_year - base$anchor_year, 0.5))
  base$status  <- as.integer(base$has_event)

  # Covariates: age, wtkg, cd4, cd8
  base$age <- base$anchor_year - aggregate(BORNY ~ CASEID, data = section2,
    FUN = function(x) max(x, na.rm = TRUE))$BORNY[
      match(base$id, aggregate(BORNY ~ CASEID, data = section2,
        FUN = function(x) max(x, na.rm = TRUE))$CASEID)]
  # Build LDWGT lookup: nearest phy_exam visit's weight (in lbs -> kg)
  phy_w <- phy[!is.na(phy$LDWGT) & phy$LDWGT > 0, ]
  wt_med <- aggregate(LDWGT ~ CASEID, data = phy_w,
                      FUN = function(x) median(x, na.rm = TRUE))
  base$wtkg <- wt_med$LDWGT[match(base$id, wt_med$CASEID)] * lbToKg

  # Baseline CD4/CD8: median LEU3N/LEU2N within ±1y of anchor
  lab_e <- lab[!is.na(lab$LEU3N) & lab$LEU3N > 0 &
               !is.na(lab$LDATY) & lab$LDATY > 0, ]
  lab_e$key <- lab_e$CASEID
  # Use median across the era (1991-1995) per subject for robustness
  cd4_med <- aggregate(LEU3N ~ CASEID,
                       data = lab_e[lab_e$LDATY %in% actgEra, ],
                       FUN = function(x) median(x, na.rm = TRUE))
  cd8_med <- aggregate(LEU2N ~ CASEID,
                       data = lab[!is.na(lab$LEU2N) & lab$LEU2N > 0 &
                                  lab$LDATY %in% actgEra, ],
                       FUN = function(x) median(x, na.rm = TRUE))
  base$cd4 <- cd4_med$LEU3N[match(base$id, cd4_med$CASEID)]
  base$cd8 <- cd8_med$LEU2N[match(base$id, cd8_med$CASEID)]

  # Final subject-level frame
  out <- data.frame(
    id      = base$id,
    source  = "MACS",
    treat   = base$treat,
    log_time = base$log_time,
    status  = base$status,
    age     = base$age,
    wtkg    = base$wtkg,
    cd4     = base$cd4,
    cd8     = base$cd8,
    stringsAsFactors = FALSE
  )
  out <- out[complete.cases(out[, harmCovars]), ]
  cat("MACS frame:", nrow(out), "subjects\n")
  out
}

# ---------------------------------------------------------------------
# 3. Build WIHS-OS frame
# ---------------------------------------------------------------------

build_wihs <- function() {
  cat("Building WIHS-OS frame ...\n")
  wcsv <- function(f) file.path(wihsDir, "CSV", f)

  f01     <- read.csv(wcsv("f01.csv"),     stringsAsFactors = FALSE)
  f07     <- read.csv(wcsv("f07.csv"),     stringsAsFactors = FALSE)
  drug1   <- read.csv(wcsv("drug1.csv"),   stringsAsFactors = FALSE)
  labsum  <- read.csv(wcsv("labsum.csv"),  stringsAsFactors = FALSE)
  outcome <- read.csv(wcsv("outcome.csv"), stringsAsFactors = FALSE)

  # Visit 1 baseline (1994-95). IDTESD in public release is the year.
  v1 <- f01[f01$VISIT == 1, c("CASEID", "IDTESD")]
  v1$anchor_year <- suppressWarnings(as.numeric(v1$IDTESD))
  v1 <- v1[!is.na(v1$anchor_year), ]

  # WIHS DRUG1 form only starts being collected ~visit 9 (1998-99), so
  # we cannot read treatment "at visit 1" directly. Instead:
  #   - keep visit 1 as the time-zero anchor (1994-95 entry)
  #   - classify each subject by the ART regimen at their EARLIEST
  #     DRUG1 record on a relevant code (CURRD1 == 1)
  #   - subjects who are never on a relevant ART are excluded
  # This is "first ART exposure" rather than "treatment at baseline";
  # the temporal gap relative to ACTG175 is a known limitation.
  art_codes_w <- c(92, 94, 147, 180, 185, 186, 187)
  art_curr <- drug1[drug1$CURRD1 %in% 1 &
                    drug1$DRUGD1 %in% art_codes_w, ]
  first_v <- aggregate(VISIT ~ CASEID, data = art_curr, FUN = min)
  names(first_v)[2] <- "first_art_visit"
  art_first <- merge(art_curr, first_v,
                     by.x = c("CASEID", "VISIT"),
                     by.y = c("CASEID", "first_art_visit"))
  drugs_by_subj <- split(art_first$DRUGD1, art_first$CASEID)
  treat_int <- vapply(drugs_by_subj, classify_treat, integer(1))

  trt <- data.frame(
    id    = names(drugs_by_subj),
    treat = treat_int,
    stringsAsFactors = FALSE)
  trt <- trt[!is.na(trt$treat), ]
  base <- merge(trt, v1[, c("CASEID", "anchor_year")],
                by.x = "id", by.y = "CASEID")

  # Outcome: AIDS or death event year (year-resolution from outcome.csv)
  oc <- outcome
  oc$is_aids   <- !is.na(oc$OUTCAIDS) & oc$OUTCAIDS == 1
  oc$is_death  <- !is.na(oc$OUTCDTH)  & oc$OUTCDTH  == 1
  oc$ev_yr     <- ifelse((oc$is_aids | oc$is_death) & oc$DATEREPT > 0,
                         oc$DATEREPT, NA)
  ev <- aggregate(ev_yr ~ CASEID, data = oc[!is.na(oc$ev_yr), ],
                  FUN = function(x) min(x, na.rm = TRUE))
  base$event_year <- ev$ev_yr[match(base$id, ev$CASEID)]
  base$has_event  <- !is.na(base$event_year)

  # Last-seen: max DATEREPT in outcome.csv (every subject has one row/visit)
  last_seen <- aggregate(DATEREPT ~ CASEID, data = oc[oc$DATEREPT > 0, ],
                         FUN = function(x) max(x, na.rm = TRUE))
  base$last_year <- last_seen$DATEREPT[match(base$id, last_seen$CASEID)]

  base$end_year <- ifelse(base$has_event, base$event_year, base$last_year)
  base$log_time <- log(pmax(base$end_year - base$anchor_year, 0.5))
  base$status   <- as.integer(base$has_event)

  # Drop subjects whose AIDS event predates baseline
  prior <- base$has_event & base$event_year < base$anchor_year
  base <- base[!prior, ]

  # Covariates
  # Age: DOB_PE in the public release is the 4-digit birth year (the
  # codebook documents mm/dd/yy but the dd/mm components are redacted).
  dob <- f07[f07$VISIT == 1, c("CASEID", "DOB_PE")]
  dob$dob_year <- suppressWarnings(as.numeric(dob$DOB_PE))
  dob <- dob[!is.na(dob$dob_year) & dob$dob_year >= 1900 &
             dob$dob_year <= 2010, ]
  base$age <- base$anchor_year -
              dob$dob_year[match(base$id, dob$CASEID)]

  # Weight: PCWTPE in pounds at visit 1 -> kg
  wt <- f07[f07$VISIT == 1 & !is.na(f07$PCWTPE) & f07$PCWTPE > 0,
            c("CASEID", "PCWTPE")]
  base$wtkg <- wt$PCWTPE[match(base$id, wt$CASEID)] * lbToKg

  # CD4/CD8 at baseline visit
  lb <- labsum[labsum$VISIT == 1, c("CASEID", "CD4N", "CD8N")]
  base$cd4 <- lb$CD4N[match(base$id, lb$CASEID)]
  base$cd8 <- lb$CD8N[match(base$id, lb$CASEID)]
  base$cd4[!is.na(base$cd4) & base$cd4 <= 0] <- NA
  base$cd8[!is.na(base$cd8) & base$cd8 <= 0] <- NA

  out <- data.frame(
    id      = base$id,
    source  = "WIHS",
    treat   = base$treat,
    log_time = base$log_time,
    status  = base$status,
    age     = base$age,
    wtkg    = base$wtkg,
    cd4     = base$cd4,
    cd8     = base$cd8,
    stringsAsFactors = FALSE
  )
  out <- out[complete.cases(out[, harmCovars]), ]
  cat("WIHS frame:", nrow(out), "subjects\n")
  out
}

# ---------------------------------------------------------------------
# 4. Assemble frames
# ---------------------------------------------------------------------

cat("\n", strrep("=", 70), "\n  ASSEMBLE\n", strrep("=", 70), "\n",
    sep = "")
rct  <- build_rct()
macs <- build_macs()
wihs <- build_wihs()

cat("\nFrame counts and treatment splits ----------------\n")
for (nm in c("rct", "macs", "wihs")) {
  d <- get(nm)
  cat(sprintf("  %-5s  n=%4d  treat0=%4d  treat1=%4d  events=%4d\n",
              nm, nrow(d),
              sum(d$treat == 0), sum(d$treat == 1),
              sum(d$status == 1)))
}

# ---------------------------------------------------------------------
# 5. KM plot per dataset
# ---------------------------------------------------------------------

cat("\n", strrep("=", 70), "\n  KM PLOTS\n", strrep("=", 70), "\n",
    sep = "")
all_dat <- rbind(rct, macs, wihs)
all_dat$source <- factor(all_dat$source, levels = c("RCT", "MACS", "WIHS"))
all_dat$treat_lab <- factor(all_dat$treat,
  levels = 0:1,
  labels = c("ZDV mono (0)", "ZDV+other / ddI mono (1)"))

km_fits <- list()
for (s in levels(all_dat$source)) {
  d_s <- all_dat[all_dat$source == s, ]
  km_fits[[s]] <- survfit(Surv(log_time, status) ~ treat_lab, data = d_s)
}

# Combined faceted plot built directly from survfit summaries
km_long <- do.call(rbind, lapply(levels(all_dat$source), function(s) {
  fit <- km_fits[[s]]
  ds <- summary(fit)
  data.frame(time = ds$time, surv = ds$surv,
             lower = ds$lower, upper = ds$upper,
             strata = ds$strata, source = s)
}))
km_gg <- ggplot(km_long, aes(x = time, y = surv, colour = strata,
                             fill = strata)) +
  geom_step(linewidth = 0.7) +
  geom_ribbon(aes(ymin = lower, ymax = upper),
              alpha = 0.15, colour = NA) +
  facet_wrap(~ source, scales = "free_x") +
  labs(x = "log(years from baseline)", y = "Survival probability",
       colour = NULL, fill = NULL,
       title = "Kaplan-Meier survival by treatment, per dataset") +
  theme_minimal() +
  theme(legend.position = "bottom")
print(km_gg)
ggsave("data/analysis/km_by_dataset.pdf", km_gg, width = 10, height = 4)

# Log-rank tests
cat("\nLog-rank tests (treat 0 vs 1):\n")
for (s in levels(all_dat$source)) {
  d_s <- all_dat[all_dat$source == s, ]
  sd  <- survdiff(Surv(log_time, status) ~ treat, data = d_s)
  p   <- 1 - pchisq(sd$chisq, df = length(sd$n) - 1)
  cat(sprintf("  %-5s  chi-sq = %6.2f  df = %d  p = %.4g\n",
              s, sd$chisq, length(sd$n) - 1, p))
}

# ---------------------------------------------------------------------
# 6. FusionForest fits
# ---------------------------------------------------------------------

cat("\n", strrep("=", 70), "\n  FUSION FORESTS\n",
    strrep("=", 70), "\n", sep = "")

fit_fusion <- function(label, sets) {
  d <- do.call(rbind, sets)
  d <- d[is.finite(d$log_time), ]
  X <- as.matrix(d[, harmCovars])
  src <- as.integer(d$source == "RCT")
  cat(sprintf("\n[%s]  n_total = %d  (RCT %d / OS %d)\n",
              label, nrow(d), sum(src == 1), sum(src == 0)))
  set.seed(1)
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
    number_of_trees_deconf    = 200,
    N_post                    = 2000,
    N_burn                    = 1000,
    store_posterior_sample    = TRUE,
    verbose                   = FALSE
  )
  list(label = label, fit = fit, frame = d, src = src)
}

# F0: RCT only — baseline. Force last RCT row to act as a dummy OS row
# (FusionForest requires at least one source = 0).
rct_only <- rct
rct_only_sets <- list(rct_only)
# Trick: pretend the last RCT row is OS so we still get a fit
rct_only_setsB <- list(rct_only)

# Actually for "RCT only" baseline, mark a single arbitrary row as OS so
# the constraint is met but it has minimal influence.
mark_one_os <- function(d) { d$source[nrow(d)] <- "OS_dummy"; d }

F0 <- fit_fusion("RCT only",   list(mark_one_os(rct)))
F1 <- fit_fusion("RCT + MACS", list(rct, macs))
F2 <- fit_fusion("RCT + WIHS", list(rct, wihs))
F3 <- fit_fusion("RCT + MACS + WIHS", list(rct, macs, wihs))

# ---------------------------------------------------------------------
# 7. ATE comparison
# ---------------------------------------------------------------------

cat("\n", strrep("=", 70), "\n  ATE COMPARISON (RCT subjects)\n",
    strrep("=", 70), "\n", sep = "")

ate_summary <- function(F) {
  cate_samples <- F$fit$train_predictions_sample_treat
  rct_idx <- which(F$src == 1L)
  if (length(rct_idx) == 0) rct_idx <- seq_len(ncol(cate_samples))
  ate_iter <- rowMeans(cate_samples[, rct_idx, drop = FALSE])
  qs <- quantile(ate_iter, c(0.025, 0.5, 0.975))
  data.frame(
    fit  = F$label,
    n    = nrow(F$frame),
    mean = mean(ate_iter),
    q025 = qs[1], q500 = qs[2], q975 = qs[3])
}

ate_tab <- do.call(rbind, lapply(list(F0, F1, F2, F3), ate_summary))
print(ate_tab, row.names = FALSE)

# Posterior densities of the ATE on the RCT population
ate_long <- do.call(rbind, lapply(list(F0, F1, F2, F3), function(F) {
  rct_idx <- which(F$src == 1L)
  if (length(rct_idx) == 0)
    rct_idx <- seq_len(ncol(F$fit$train_predictions_sample_treat))
  ate_iter <- rowMeans(
    F$fit$train_predictions_sample_treat[, rct_idx, drop = FALSE])
  data.frame(fit = F$label, ate = ate_iter)
}))
ate_long$fit <- factor(ate_long$fit,
  levels = c("RCT only", "RCT + MACS", "RCT + WIHS", "RCT + MACS + WIHS"))

p_ate <- ggplot(ate_long, aes(x = ate, fill = fit, colour = fit)) +
  geom_density(alpha = 0.25) +
  geom_vline(xintercept = 0, linetype = "dashed") +
  labs(x = "ATE on log-survival scale (RCT-population)",
       y = "Posterior density", fill = NULL, colour = NULL,
       title = "ATE posteriors across fusion designs") +
  theme_minimal() +
  theme(legend.position = "bottom")
print(p_ate)
ggsave("data/analysis/ate_posteriors.pdf", p_ate, width = 8, height = 4.5)

cat("\nDone.  Saved: data/analysis/km_by_dataset.pdf, ",
    "data/analysis/ate_posteriors.pdf\n", sep = "")

invisible(list(F0 = F0, F1 = F1, F2 = F2, F3 = F3, ate_tab = ate_tab))
