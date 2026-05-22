## endpoint_comparison.R
##
## Quantify how the candidate event-endpoint definitions differ
## between ACTG175 (RCT) and MACS (RWD), so we can pick one for the
## fusion pipeline.
##
## Motivation
##   The current pipeline (fusion_actg175_macs.R) calls its endpoint
##   "EFS" and treats ACTG175 and MACS as carrying the same event.
##   On closer reading the two composites are NOT identical:
##     - CD4 component is RELATIVE (>=50% decline from baseline) in
##       ACTG175 but ABSOLUTE (<200 cells/uL OR <14%) in MACS.
##     - Death component is ALL-CAUSE in ACTG175 but coded as
##       DEATH %in% c(1,2) (AIDS-related only) in MACS, which
##       silently drops codes 3 (Not AIDS) and 4 (Unknown).
##     - MACS death DATES are redacted in the public release
##       (verified empirically), so death-event times are imputed
##       by the subject's last lab-visit year.
##
##   ACTG175's public release exposes only the composite event
##   indicator `cens` and its time `days`; the three components are
##   NOT individually released. So on the RCT side we can compare
##   variants only descriptively, with one diagnostic reconstruction
##   of >=50% CD4 decline from the three CD4 snapshots
##   (cd40 baseline, cd420 ~20 wk, cd496 ~96 wk).
##
## Variants (M = MACS-side rule; ACTG175 stays at `cens` throughout)
##   v1 EFS-current   M: cd4_abs OR aids OR death_aids   <- current pipeline
##   v2 EFS-death-fix M: cd4_abs OR aids OR death_all_cause
##   v3 EFS-cd4-fix   M: cd4_rel(>=50%) OR aids OR death_all_cause
##   v4 PFS           M: aids OR death_all_cause         (drops CD4 entirely)
##   v5 OS            M: death_all_cause only
##
## Outputs
##   data/analysis 0/endpoint_comparison.pdf  -- event-count table panel
##                                             plus KM curves per variant
##   Console: counts, concordance vs v1, log-rank p-values.

setwd("~/Library/CloudStorage/OneDrive-VrijeUniversiteitAmsterdam/Documents/GitHub/FusionForests/")

suppressPackageStartupMessages({
  library(speff2trial)
  library(survival)
  library(ggplot2)
  library(patchwork)
})

proj     <- "."
macsDir  <- file.path(proj, "data", "MACS PDS")
actgEra  <- 1991:1995
artCodes <- c(92, 94, 147, 180, 185, 186, 187)

# ---------------------------------------------------------------------
# 0. Helpers (same parser, reader, classifier as fusion_actg175_macs.R)
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
  if (length(art) == 0)          return(NA_integer_)
  if (any(art %in% c(186, 187))) return(NA_integer_)
  if (identical(sort(art), 94))  return(NA_integer_)
  if (identical(sort(art), 92))  return(0L)
  1L
}

# ---------------------------------------------------------------------
# 1. ACTG175 base frame + reconstructed CD4-50% diagnostic
# ---------------------------------------------------------------------
#
# ACTG175 reports CD4 at baseline (cd40), week 20+/-5 (cd420), and
# week 96+/-5 (cd496); r=1 flags missing 96-week measurement. We flag
# a reconstructed >=50% decline if cd420 OR cd496 falls below half of
# cd40. The decline TIME is approximated by the nominal window
# midpoint (140 d for 20 wk, 672 d for 96 wk); a real ACTG175 event
# time would be the confirmation date which is not in the release.
# This reconstruction is purely DIAGNOSTIC: it ignores AIDS and death,
# so it is not a proper endpoint -- it is reported so we can compare
# the CD4-component event rate to MACS's CD4-rel variant.

build_actg <- function() {
  data(ACTG175)
  d <- ACTG175

  cd_d20 <- !is.na(d$cd420) & d$cd420 <= 0.5 * d$cd40
  cd_d96 <- !is.na(d$cd496) & d$cd496 <= 0.5 * d$cd40
  cd_decl_days <- ifelse(cd_d20, 140,
                  ifelse(cd_d96, 672, NA_real_))

  data.frame(
    id           = as.character(d$pidnum),
    treat        = as.integer(d$treat),
    days         = d$days,                # time to cens or censoring
    cens         = as.integer(d$cens),    # composite event indicator
    cd_decl_50   = as.integer(cd_d20 | cd_d96),
    cd_decl_days = cd_decl_days,
    stringsAsFactors = FALSE)
}

# ---------------------------------------------------------------------
# 2. MACS base frame with all per-component event tuples
# ---------------------------------------------------------------------
#
# For each subject we derive:
#   anchor_year   -- min AVQY of currently-on-ART records in 1991-95
#   treat         -- 0/1 by classify_treat()
#   aids_yr       -- DATE1yy if AIDSCASE in {2, 3}, else NA
#   cd4_abs_yr    -- SELFCD4Dyy (year CD4 first <200/<14%), else NA
#   cd4_rel_yr    -- year of FIRST lab in lab_rslt where LEU3N drops
#                    to <=50% of the subject's earliest 1991-95 CD4
#   death_any     -- DEATH in {1, 2, 3, 4} (all-cause; date redacted)
#   death_aids    -- DEATH in {1, 2}   (AIDS-related, current rule)
#   last_year     -- max LDATY in lab_rslt (used as death-time fallback)
# Subjects with an AIDS dx PRIOR to anchor are excluded.

build_macs <- function() {
  cat("Building MACS base frame ...\n")
  inp <- function(f) file.path(macsDir, "sasinp", f)
  dat <- function(f) file.path(macsDir, "data",   f)

  lay_macsid  <- parse_sas_inp(inp("macsid.inp"))
  lay_drugf1  <- parse_sas_inp(inp("drugf1.inp"))
  lay_lab     <- parse_sas_inp(inp("lab_rslt.inp"))
  lay_outcome <- parse_sas_inp(inp("outcome.inp"))
  lay_outcome$start[lay_outcome$name == "DEATH"] <- 989
  lay_outcome$end  [lay_outcome$name == "DEATH"] <- 989

  macsid  <- read_macs_subset(dat("macsid.dat"), lay_macsid,
                              c("CASEID","STATUS02","STATUS10"))
  drugf1  <- read_macs_subset(dat("drugf1.dat"), lay_drugf1,
                              c("CASEID","VISIT","AVQY","DRGAV","AVNW"))
  lab     <- read_macs_subset(dat("lab_rslt.dat"), lay_lab,
                              c("CASEID","LDATY","LEU3N","LEU2N"))
  outcome <- read_macs_subset(dat("outcome.dat"), lay_outcome,
                              c("CASEID","AIDSCASE","DATE1yy",
                                "SELFCD4Dyy","DEATH"))

  hiv <- macsid$CASEID[!(macsid$STATUS02 %in% 1 | macsid$STATUS10 %in% 1)]
  drugf1  <- drugf1 [drugf1 $CASEID %in% hiv, ]
  lab     <- lab    [lab    $CASEID %in% hiv, ]
  outcome <- outcome[outcome$CASEID %in% hiv, ]

  # Anchor + treatment
  curr <- drugf1[drugf1$AVQY %in% actgEra & drugf1$AVNW %in% 2, ]
  drugs_by_subj <- split(curr$DRGAV, curr$CASEID)
  anchor_yr <- vapply(split(curr$AVQY, curr$CASEID),
                      function(x) min(x, na.rm = TRUE), numeric(1))
  treat_int <- vapply(drugs_by_subj, classify_treat, integer(1))
  base <- data.frame(id = names(drugs_by_subj), treat = treat_int,
                     anchor_year = unname(anchor_yr),
                     stringsAsFactors = FALSE)
  base <- base[!is.na(base$treat), ]

  # Per-subject baseline CD4 (earliest 1991-95 LEU3N) and first lab
  # YEAR at which LEU3N <= 0.5 * baseline. We work per subject because
  # the >=50% rule is path-dependent across labs.
  lab_cd4 <- lab[!is.na(lab$LEU3N) & lab$LEU3N > 0 &
                 !is.na(lab$LDATY) & lab$LDATY > 0, ]
  lab_cd4 <- lab_cd4[order(lab_cd4$CASEID, lab_cd4$LDATY), ]
  base_cd4 <- aggregate(LEU3N ~ CASEID,
                        data = lab_cd4[lab_cd4$LDATY %in% actgEra, ],
                        FUN = function(x) x[1])
  names(base_cd4)[2] <- "cd4_base"
  lab_cd4 <- merge(lab_cd4, base_cd4, by = "CASEID")
  lab_cd4 <- lab_cd4[lab_cd4$LDATY >= ave(lab_cd4$LDATY, lab_cd4$CASEID,
                                          FUN = min), ]
  decl_rows <- lab_cd4[lab_cd4$LEU3N <= 0.5 * lab_cd4$cd4_base, ]
  cd4_rel_first <- aggregate(LDATY ~ CASEID, data = decl_rows,
                             FUN = min)
  names(cd4_rel_first)[2] <- "cd4_rel_yr"

  # Last lab year (death-time fallback)
  last_lab <- aggregate(LDATY ~ CASEID, data = lab,
                        FUN = function(x) max(x, na.rm = TRUE))
  names(last_lab)[2] <- "last_year"

  # Outcomes
  oc <- outcome
  oc$aids_yr    <- ifelse(oc$AIDSCASE %in% c(2,3) & oc$DATE1yy > 0,
                          oc$DATE1yy, NA_real_)
  oc$cd4_abs_yr <- ifelse(oc$SELFCD4Dyy > 0, oc$SELFCD4Dyy, NA_real_)
  oc$death_any  <- !is.na(oc$DEATH) & oc$DEATH %in% c(1, 2, 3, 4)
  oc$death_aids <- oc$DEATH %in% c(1, 2)

  base <- merge(base, oc[, c("CASEID","aids_yr","cd4_abs_yr",
                             "death_any","death_aids")],
                by.x = "id", by.y = "CASEID", all.x = TRUE)
  base <- merge(base, last_lab,      by.x = "id", by.y = "CASEID",
                all.x = TRUE)
  base <- merge(base, cd4_rel_first, by.x = "id", by.y = "CASEID",
                all.x = TRUE)
  base$death_any [is.na(base$death_any)]  <- FALSE
  base$death_aids[is.na(base$death_aids)] <- FALSE

  # Drop subjects with AIDS predating anchor (RCT entry analogue)
  prior_aids <- !is.na(base$aids_yr) & base$aids_yr < base$anchor_year
  base <- base[!prior_aids, ]

  cat(sprintf("MACS base: n=%d   (Z=0: %d, Z=1: %d)\n",
              nrow(base),
              sum(base$treat == 0), sum(base$treat == 1)))
  base
}

# ---------------------------------------------------------------------
# 3. Per-variant event constructor
# ---------------------------------------------------------------------
#
# Given the MACS base frame and a logical spec naming which component
# branches feed the event, returns (status, t_years). Death (no
# observable date in the public release) contributes its last-lab-year
# fallback time only when the chosen death flag is set; otherwise it
# does not contribute.

combine_macs <- function(b, cd4_abs = FALSE, cd4_rel = FALSE,
                         aids = FALSE, death_any = FALSE,
                         death_aids = FALSE) {
  n <- nrow(b)
  yr_mat <- matrix(Inf, nrow = n, ncol = 0)
  add <- function(mat, v) cbind(mat, ifelse(is.na(v), Inf, v))
  if (cd4_abs) yr_mat <- add(yr_mat, b$cd4_abs_yr)
  if (cd4_rel) yr_mat <- add(yr_mat, b$cd4_rel_yr)
  if (aids)    yr_mat <- add(yr_mat, b$aids_yr)
  d_flag <- (death_any  & b$death_any) | (death_aids & b$death_aids)
  yr_mat <- add(yr_mat, ifelse(d_flag, b$last_year, NA_real_))

  ev_yr <- if (ncol(yr_mat) == 0) rep(Inf, n) else
           apply(yr_mat, 1, min)
  ev_yr[is.infinite(ev_yr)] <- NA_real_
  has_event <- !is.na(ev_yr)
  end_year  <- ifelse(has_event, ev_yr, b$last_year)
  t_years   <- pmax(end_year - b$anchor_year, 0.5)
  data.frame(id = b$id, treat = b$treat,
             t_years = t_years,
             status  = as.integer(has_event),
             stringsAsFactors = FALSE)
}

variants <- list(
  v1 = list(name = "EFS-current",
            spec = list(cd4_abs = TRUE,  cd4_rel = FALSE,
                        aids = TRUE,
                        death_any = FALSE, death_aids = TRUE)),
  v2 = list(name = "EFS-death-fix",
            spec = list(cd4_abs = TRUE,  cd4_rel = FALSE,
                        aids = TRUE,
                        death_any = TRUE,  death_aids = FALSE)),
  v3 = list(name = "EFS-cd4-fix",
            spec = list(cd4_abs = FALSE, cd4_rel = TRUE,
                        aids = TRUE,
                        death_any = TRUE,  death_aids = FALSE)),
  v4 = list(name = "PFS",
            spec = list(cd4_abs = FALSE, cd4_rel = FALSE,
                        aids = TRUE,
                        death_any = TRUE,  death_aids = FALSE)),
  v5 = list(name = "OS",
            spec = list(cd4_abs = FALSE, cd4_rel = FALSE,
                        aids = FALSE,
                        death_any = TRUE,  death_aids = FALSE))
)

# ---------------------------------------------------------------------
# 4. Build, count, tabulate
# ---------------------------------------------------------------------

cat("\n", strrep("=", 70), "\n  ENDPOINT COMPARISON\n",
    strrep("=", 70), "\n", sep = "")
actg <- build_actg()
macs <- build_macs()
cat(sprintf("ACTG175 base: n=%d   (Z=0: %d, Z=1: %d)\n",
            nrow(actg),
            sum(actg$treat == 0), sum(actg$treat == 1)))

macs_v <- lapply(variants, function(v)
  do.call(combine_macs, c(list(b = macs), v$spec)))
names(macs_v) <- vapply(variants, `[[`, character(1), "name")

# Summary row builder
summ <- function(d, label, study) {
  ev1 <- sum(d$treat == 1 & d$status == 1)
  ev0 <- sum(d$treat == 0 & d$status == 1)
  n1  <- sum(d$treat == 1); n0 <- sum(d$treat == 0)
  py1 <- sum(d$t_years[d$treat == 1])
  py0 <- sum(d$t_years[d$treat == 0])
  # Log-rank by treat
  sd <- tryCatch(
    survdiff(Surv(t_years, status) ~ treat, data = d),
    error = function(e) NULL)
  chi <- if (is.null(sd)) NA_real_ else sd$chisq
  p   <- if (is.null(sd)) NA_real_ else
         1 - pchisq(sd$chisq, df = length(sd$n) - 1)
  data.frame(
    study  = study,
    variant = label,
    n0 = n0, n1 = n1,
    ev0 = ev0, ev1 = ev1,
    rate0_per100py = round(100 * ev0 / max(py0, 1e-9), 2),
    rate1_per100py = round(100 * ev1 / max(py1, 1e-9), 2),
    logrank_chi = round(chi, 2),
    logrank_p   = signif(p, 3),
    stringsAsFactors = FALSE)
}

# ACTG175 rows
actg_efs <- data.frame(treat = actg$treat,
                       t_years = actg$days / 365.25,
                       status  = actg$cens,
                       stringsAsFactors = FALSE)
actg_cd4 <- data.frame(treat   = actg$treat,
                       t_years = ifelse(is.na(actg$cd_decl_days),
                                        actg$days,
                                        actg$cd_decl_days) / 365.25,
                       status  = actg$cd_decl_50,
                       stringsAsFactors = FALSE)

cnt_rows <- list(
  summ(actg_efs, "EFS-published (cens)",       "ACTG175"),
  summ(actg_cd4, "CD4-50%-only (diagnostic)",  "ACTG175"))
for (k in names(macs_v))
  cnt_rows[[length(cnt_rows) + 1]] <- summ(macs_v[[k]], k, "MACS")

cnt_tab <- do.call(rbind, cnt_rows)
cat("\nEvent counts and crude rates per arm:\n")
print(cnt_tab, row.names = FALSE)

# Wide event-count comparison (MACS variants only)
mcomp <- do.call(rbind, lapply(names(macs_v), function(k) {
  d <- macs_v[[k]]
  data.frame(variant = k,
             events_total = sum(d$status == 1),
             events_Z0    = sum(d$status == 1 & d$treat == 0),
             events_Z1    = sum(d$status == 1 & d$treat == 1),
             event_rate   = round(mean(d$status), 3),
             med_yr_Z0    = round(median(d$t_years[d$treat == 0]), 2),
             med_yr_Z1    = round(median(d$t_years[d$treat == 1]), 2),
             stringsAsFactors = FALSE)
}))
cat("\nMACS event counts across variants (same n in each row):\n")
print(mcomp, row.names = FALSE)

# Concordance: per-subject status under v1 vs each alternative
cat("\n", strrep("=", 70),
    "\n  Status concordance (MACS): v1 EFS-current vs alternatives\n",
    strrep("=", 70), "\n", sep = "")
ref <- macs_v[["EFS-current"]]$status
for (k in setdiff(names(macs_v), "EFS-current")) {
  alt <- macs_v[[k]]$status
  tbl <- table(EFS_current = ref, alt = alt,
               dnn = c("EFS-current", k))
  cat("\n", k, " vs EFS-current:\n", sep = "")
  print(tbl)
  cat(sprintf("  agreement = %.1f%%   v1=1 only = %d   %s=1 only = %d\n",
              100 * mean(ref == alt),
              sum(ref == 1 & alt == 0),
              k, sum(ref == 0 & alt == 1)))
}

# ---------------------------------------------------------------------
# 5. KM panels
# ---------------------------------------------------------------------

theme_panel <- function() {
  theme_minimal(base_size = 10) +
    theme(legend.position = "bottom",
          plot.title = element_text(size = 10, face = "bold"))
}

build_km_df <- function(d) {
  d2 <- data.frame(
    t = d$t_years, s = d$status,
    arm = factor(d$treat, levels = 0:1,
                 labels = c("Z=0 (ZDV mono)", "Z=1 (combo/ddI)")))
  fit <- survfit(Surv(t, s) ~ arm, data = d2)
  ds  <- summary(fit)
  data.frame(time = ds$time, surv = ds$surv,
             lower = ds$lower, upper = ds$upper,
             strata = ds$strata)
}

km_panel <- function(df, title) {
  ggplot(df, aes(x = time, y = surv, colour = strata, fill = strata)) +
    geom_step(linewidth = 0.7) +
    geom_ribbon(aes(ymin = lower, ymax = upper),
                alpha = 0.15, colour = NA) +
    ylim(0, 1) +
    labs(x = "Years from anchor", y = "Survival probability",
         colour = NULL, fill = NULL, title = title) +
    theme_panel()
}

# Row 1: ACTG175 (only the published composite is comparable; the CD4
# diagnostic uses approximated event times so its KM has only a few
# steps -- shown for context).
p_a_efs <- km_panel(build_km_df(actg_efs),
                    "ACTG175 -- EFS-published (cens)")
p_a_cd4 <- km_panel(build_km_df(actg_cd4),
                    "ACTG175 -- CD4-50%-only (diagnostic)")

# Row 2-3: MACS under each variant
p_macs <- lapply(names(macs_v), function(k) {
  km_panel(build_km_df(macs_v[[k]]), sprintf("MACS -- %s", k))
})
names(p_macs) <- names(macs_v)

# Event-count table rendered as a ggplot text panel so it can sit in
# the same multi-page PDF as the KM curves.
tab_text <- capture.output(print(cnt_tab, row.names = FALSE))
p_tab <- ggplot() +
  annotate("text", x = 0, y = 1,
           label = paste(tab_text, collapse = "\n"),
           hjust = 0, vjust = 1, family = "mono", size = 3) +
  xlim(0, 1) + ylim(0, 1) +
  theme_void() +
  labs(title = "Event counts and crude rates per arm") +
  theme(plot.title = element_text(size = 11, face = "bold"))

# Layout: 1) ACTG175 row (2 plots), 2) MACS row 1 (v1-v3),
#         3) MACS row 2 (v4-v5 + blank), 4) counts table
blank <- ggplot() + theme_void()
panel <- (p_a_efs | p_a_cd4 | blank) /
         (p_macs[["EFS-current"]]   | p_macs[["EFS-death-fix"]] |
          p_macs[["EFS-cd4-fix"]]) /
         (p_macs[["PFS"]]            | p_macs[["OS"]]            |
          blank) /
         p_tab +
  plot_annotation(
    title    = "Endpoint comparison: ACTG175 vs MACS variants",
    subtitle = paste(
      "Row 1: ACTG175 (only `cens` is a proper endpoint; CD4-50%-only",
      "is a diagnostic reconstruction from 3 CD4 snapshots).",
      "Row 2-3: MACS under each variant. v1 EFS-current is the current",
      "pipeline; v3 EFS-cd4-fix is the closest match to ACTG175."))

out_pdf <- "data/analysis 0/endpoint_comparison.pdf"
ggsave(out_pdf, panel, width = 16, height = 18)
cat("\nSaved: ", out_pdf, "\n", sep = "")

invisible(list(
  actg     = actg,
  macs     = macs,
  variants = macs_v,
  counts   = cnt_tab,
  panel    = panel))
