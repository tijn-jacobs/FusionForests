## survival_curves.R
##
## Kaplan-Meier survival curves with 95% confidence bands for the
## ACTG175 RCT and the MACS RWD arm under a common EFS endpoint.
##
## Treatment arms (same dichotomy on both sides):
##   Z = 0  --  ZDV monotherapy.
##              RCT: ACTG175$treat == 0 (zidovudine alone).
##              RWD: MACS subjects whose 1991-95 ART drug set is
##                   exactly {92} (ZDV) -- see classify_treat().
##   Z = 1  --  Combination / ddI-containing therapy.
##              RCT: ACTG175$treat == 1 (ZDV+ddI, ZDV+zalcitabine,
##                   or ddI alone -- everything non-ZDV-mono in the
##                   trial).
##              RWD: MACS subjects with any other ART combination
##                   from the allowed code set
##                   {92, 94, 147, 180, 185}; subjects on no ART,
##                   on protease inhibitors (codes 186/187, outside
##                   the ACTG175 regimen set), or on ddI-only
##                   (code 94 alone) are dropped to keep the two
##                   arms aligned with the trial's contrast.
##
## Endpoint (called EFS throughout):
##   RCT (ACTG175) -- published EFS composite `cens` (CD4 decline,
##                    AIDS, or all-cause death; days = follow-up).
##   RWD (MACS)    -- all-cause death (DEATH in {1, 2, 3, 4}) with
##                    last-lab-year as the event-time fallback (the
##                    public release redacts death dates). This is
##                    variant v5 in endpoint_comparison.R; we label
##                    it EFS here because we treat it as the
##                    RWD-side EFS for this comparison, even though
##                    it is a strict subset of the RCT composite.
##                    Death is the most reliably captured event in
##                    MACS, so this gives a clean event signal at
##                    the cost of dropping the CD4 and AIDS arms of
##                    the composite.
##
## Panels (per page):
##   1. RCT alone        -- ACTG175 by treatment (Z=0, Z=1)
##   2. RWD alone        -- MACS    by treatment (Z=0, Z=1)
##   3. Combined, x-axis 0 to max(RCT)  -- all four strata
##   4. Combined, x-axis 0 to max(RWD)  -- all four strata, longer x
##
## Pages:
##   1. Default anchor   -- earliest 1991-95 record where the subject
##                          is currently taking a relevant ART
##                          regimen (prevalent-user design).
##   2. New-user anchor  -- same as page 1, but with subjects who had
##                          ANY pre-1991 "currently taking" record on
##                          a contrast regimen (artCodes) excluded.
##                          Implements option 1 from the design
##                          discussion: incident users within the
##                          trial calendar window, eliminating the
##                          pre-anchor exposure that page 1's anchor
##                          silently rolls into g(X).
##   3. Male-only        -- only the sex filter is applied: RCT
##      (sex filter)         restricted to male subjects (gender == 1),
##                          MACS kept at its default cohort. Isolates
##                          the effect of the sex restriction from the
##                          CD4-band restriction, because the latter
##                          drops ~57% of MACS subjects on its own.
##   4. Trial-aligned    -- same as page 1 default anchor, with the
##      cohort               ACTG175 entry criteria mirrored on both
##                          sides: RCT restricted to male subjects
##                          (gender == 1), MACS restricted to baseline
##                          CD4 in [200, 500] cells/uL. MACS is
##                          already men-only by design and ACTG175 is
##                          already CD4-banded by protocol, so each
##                          filter only bites on one side.
##
## All panels use coord_cartesian(ylim = c(0, 1)) so the survival
## axis always spans the full 0-1 range, regardless of empirical
## maximum/minimum.
##
## The MACS (RWD) endpoint construction mirrors the v5 logic from
## endpoint_comparison.R::combine_macs(.., death_any = TRUE). It is
## reimplemented here so the script is self-contained.
##
## Output:
##   data/analysis 1/survival_curves.pdf                       -- 3-page diagnostic
##   notes/general/figures/survival_curves_manuscript.pdf -- single-page
##       manuscript version (trial-aligned cohort) of the combined plot at
##       the full 25-y RWD horizon, with a ggmagnify inset zooming into the
##       upper-left region where RCT and RWD follow-up overlap.
##         page 1: default cohort        (rct, rwd)
##         page 2: trial-aligned cohort  (rct_m, rwd_cd4)

setwd("~/Library/CloudStorage/OneDrive-VrijeUniversiteitAmsterdam/Documents/GitHub/FusionForests/")

suppressPackageStartupMessages({
  library(speff2trial)
  library(survival)
  library(ggplot2)
  library(patchwork)
  library(ggmagnify)   # for the manuscript inset figure (section 5)
})

proj     <- "."
macsDir  <- file.path(proj, "data", "MACS PDS")
actgEra  <- 1991:1995
artCodes <- c(92, 94, 147, 180, 185, 186, 187)

# ---------------------------------------------------------------------
# 0. Helpers (parser, reader, classifier) -- same as the rest of the
#    pipeline; duplicated to keep this script self-contained.
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
# 1. Build RCT (ACTG175) and RWD (MACS, v3 endpoint)
# ---------------------------------------------------------------------

build_rct <- function(male_only = FALSE) {
  data(ACTG175)
  d <- ACTG175
  if (male_only) {
    n_before <- nrow(d)
    d <- d[!is.na(d$gender) & d$gender == 1, ]
    cat(sprintf(
      "  male-only filter: dropped %d non-male subjects\n",
      n_before - nrow(d)))
  }
  data.frame(
    t_years = d$days / 365.25,
    status  = as.integer(d$cens),
    treat   = as.integer(d$treat),
    stringsAsFactors = FALSE)
}

build_rwd <- function(new_user = FALSE, cd4_band = NULL) {
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
                              c("CASEID","LDATY","LEU3N"))
  outcome <- read_macs_subset(dat("outcome.dat"), lay_outcome,
                              c("CASEID","AIDSCASE","DATE1yy","DEATH"))

  hiv <- macsid$CASEID[!(macsid$STATUS02 %in% 1 |
                         macsid$STATUS10 %in% 1)]
  drugf1  <- drugf1 [drugf1 $CASEID %in% hiv, ]
  lab     <- lab    [lab    $CASEID %in% hiv, ]
  outcome <- outcome[outcome$CASEID %in% hiv, ]

  # Anchor + treatment classification (1991-95 era, currently on ART)
  curr <- drugf1[drugf1$AVQY %in% actgEra & drugf1$AVNW %in% 2, ]
  drugs_by_subj <- split(curr$DRGAV, curr$CASEID)
  anchor_yr <- vapply(split(curr$AVQY, curr$CASEID),
                      function(x) min(x, na.rm = TRUE), numeric(1))
  treat_int <- vapply(drugs_by_subj, classify_treat, integer(1))
  base <- data.frame(id = names(drugs_by_subj), treat = treat_int,
                     anchor_year = unname(anchor_yr),
                     stringsAsFactors = FALSE)
  base <- base[!is.na(base$treat), ]

  # ----- Optional new-user filter (option 1 from the design note) -----
  # Default behaviour keeps subjects whose first "currently taking"
  # record on a contrast regimen falls in 1991-95 -- which silently
  # includes subjects on ZDV (or another artCode) since 1987-90, i.e.
  # prevalent users with informative pre-anchor exposure. Setting
  # new_user = TRUE drops anyone with any pre-1991 AVNW == 2 record on
  # a regimen in artCodes, so the cohort is restricted to incident
  # users of the contrast regimens within the trial calendar window.
  if (new_user) {
    prior <- drugf1[!is.na(drugf1$AVQY) &
                    drugf1$AVQY < min(actgEra) &
                    drugf1$AVNW %in% 2 &
                    drugf1$DRGAV %in% artCodes, ]
    prior_ids <- unique(prior$CASEID)
    n_before <- nrow(base)
    base <- base[!(base$id %in% prior_ids), ]
    cat(sprintf(
      "  new-user filter: dropped %d prevalent users (pre-1991 ART)\n",
      n_before - nrow(base)))
  }

  # ----- Optional baseline-CD4 band filter (trial-eligibility) --------
  # Mirrors ACTG175's CD4 200-500 entry criterion on the MACS side.
  # Baseline CD4 is the earliest 1991-95 LEU3N reading per subject,
  # consistent with the same anchor logic used for the relative-decline
  # rule in build_macs (endpoint_comparison.R). Subjects with no
  # 1991-95 LEU3N reading at all are dropped: we cannot verify trial
  # eligibility for them.
  if (!is.null(cd4_band)) {
    lab_era <- lab[lab$LDATY %in% actgEra &
                   !is.na(lab$LEU3N) & lab$LEU3N > 0, ]
    lab_era <- lab_era[order(lab_era$CASEID, lab_era$LDATY), ]
    cd4_base <- aggregate(LEU3N ~ CASEID, data = lab_era,
                          FUN = function(x) x[1])
    names(cd4_base)[2] <- "cd4_base"
    base <- merge(base, cd4_base, by.x = "id", by.y = "CASEID",
                  all.x = TRUE)
    n_before <- nrow(base)
    keep <- !is.na(base$cd4_base) &
            base$cd4_base >= cd4_band[1] &
            base$cd4_base <= cd4_band[2]
    base <- base[keep, ]
    cat(sprintf(
      "  CD4-band filter [%d, %d]: dropped %d subjects (NA or out of range)\n",
      cd4_band[1], cd4_band[2], n_before - nrow(base)))
  }

  # ----- MACS endpoint = EFS (death-only construction, v5) -----------
  # Event       = all-cause death (DEATH in {1, 2, 3, 4}).
  # Event time  = last lab-visit year (DEATH date redacted in the
  #               public release, so we use last observed contact as
  #               the event-time fallback for those who died).
  # Censoring   = last lab-visit year for those who did not die.
  # The prior-AIDS exclusion is retained for cohort comparability with
  # the other variants in endpoint_comparison.R. We label this EFS
  # (matching the RCT side) although structurally it is OS only --
  # see the file header for the rationale.
  oc <- outcome
  oc$aids_yr <- ifelse(oc$AIDSCASE %in% c(2, 3) & oc$DATE1yy > 0,
                       oc$DATE1yy, NA_real_)
  oc$is_dead <- !is.na(oc$DEATH) & oc$DEATH %in% c(1, 2, 3, 4)

  last_lab <- aggregate(LDATY ~ CASEID, data = lab,
                        FUN = function(x) max(x, na.rm = TRUE))
  names(last_lab)[2] <- "last_year"

  base <- merge(base, oc[, c("CASEID","aids_yr","is_dead")],
                by.x = "id", by.y = "CASEID", all.x = TRUE)
  base <- merge(base, last_lab,
                by.x = "id", by.y = "CASEID", all.x = TRUE)
  base$is_dead[is.na(base$is_dead)] <- FALSE

  prior_aids <- !is.na(base$aids_yr) & base$aids_yr < base$anchor_year
  base <- base[!prior_aids, ]

  has_event <- base$is_dead
  t_years   <- pmax(base$last_year - base$anchor_year, 0.5)

  data.frame(
    t_years = t_years,
    status  = as.integer(has_event),
    treat   = base$treat,
    stringsAsFactors = FALSE)
}

log_n <- function(d, tag) {
  cat(sprintf("  %s: n=%d (Z=0: %d, Z=1: %d)  events=%d\n",
              tag, nrow(d), sum(d$treat == 0), sum(d$treat == 1),
              sum(d$status == 1)))
}

cat("Building RCT (ACTG175, full) ...\n")
rct   <- build_rct(male_only = FALSE)
log_n(rct, "RCT")

cat("Building RCT (ACTG175, male only) ...\n")
rct_m <- build_rct(male_only = TRUE)
log_n(rct_m, "RCT-male")

cat("Building RWD (MACS, default anchor: first 1991-95 ART) ...\n")
rwd <- build_rwd(new_user = FALSE)
log_n(rwd, "RWD-default")

cat("Building RWD (MACS, new-user anchor: no pre-1991 ART) ...\n")
rwd_nu <- build_rwd(new_user = TRUE)
log_n(rwd_nu, "RWD-new-user")

cat("Building RWD (MACS, baseline CD4 in [200, 500]) ...\n")
rwd_cd4 <- build_rwd(new_user = FALSE, cd4_band = c(200, 500))
log_n(rwd_cd4, "RWD-CD4-band")

# ---------------------------------------------------------------------
# 2. Fit Kaplan-Meier per panel (survfit + factor labels)
# ---------------------------------------------------------------------

attach_arm <- function(d) {
  d$arm <- factor(d$treat, levels = 0:1,
                  labels = c("ZDV mono", "Combination"))
  d
}

# ---------------------------------------------------------------------
# 3. Panels: ggplot2 with step-shaped survival lines and a step-shaped
#    filled 95% CI ribbon (no censoring tick marks)
# ---------------------------------------------------------------------
#
# survminer::ggsurvplot only renders the CI as either a smoothly
# interpolated ribbon ("ribbon") or a pair of step-shaped lines
# without fill ("step"); neither gives a step-shaped FILLED ribbon.
# We therefore step-expand the survfit output (each segment becomes a
# horizontal plateau, with a same-x vertical jump at each event) and
# plot the ribbon and line directly with ggplot2. Both layers share
# the expanded data so they land on the same plateaus.

# Okabe-Ito colourblind-safe palette, shared with the rest of the paper. Source
# is the hue (RCT warm, RWD cool) and treatment the lightness within each pair.
#   orange #E69F00 / vermillion  #D55E00   (RCT: ZDV mono, Combination)
#   sky blue #56B4E9 / blue #0072B2        (RWD: ZDV mono, Combination)
pal_rct <- c("#E69F00", "#D55E00")                # ZDV mono, Combination
pal_rwd <- c("#56B4E9", "#0072B2")
pal_all <- c("#E69F00", "#D55E00",
             "#56B4E9", "#0072B2")

# Survfit -> tidy long frame (time, surv, lower, upper, arm) with a
# (time = 0, surv = 1, lower = 1, upper = 1) start row per stratum.
tidy_km <- function(fit) {
  if (is.null(fit$strata)) {
    strat_lvl <- "(all)"
    strat_vec <- rep(strat_lvl, length(fit$time))
  } else {
    strat_lvl <- sub("^[^=]+=", "", names(fit$strata))
    strat_vec <- rep(strat_lvl, fit$strata)
  }
  body <- data.frame(
    time  = fit$time,
    surv  = fit$surv,
    lower = pmax(0, ifelse(is.na(fit$lower), 0, fit$lower)),
    upper = pmin(1, ifelse(is.na(fit$upper), 1, fit$upper)),
    arm   = strat_vec, stringsAsFactors = FALSE)
  starts <- data.frame(
    time = 0, surv = 1, lower = 1, upper = 1,
    arm = strat_lvl, stringsAsFactors = FALSE)
  d <- rbind(starts, body)
  d$arm <- factor(d$arm, levels = strat_lvl)
  d[order(d$arm, d$time), ]
}

# Duplicate each row at the next event time so a linear-interpolation
# geom (geom_ribbon, geom_line) draws horizontal plateaus with vertical
# jumps -- i.e. a true step shape.
step_expand <- function(d) {
  do.call(rbind, lapply(split(d, d$arm), function(g) {
    g <- g[order(g$time), ]
    n <- nrow(g)
    if (n < 2) return(g)
    hold <- data.frame(
      time  = g$time[-1],
      surv  = g$surv[-n],
      lower = g$lower[-n],
      upper = g$upper[-n],
      arm   = g$arm[-n])
    out <- rbind(g, hold)
    # At each shared time, "hold" must come before "original" so the
    # ribbon ramps up to the old level and then jumps to the new one.
    out$is_hold <- c(rep(0L, n), rep(1L, n - 1))
    out <- out[order(out$time, -out$is_hold), ]
    out$is_hold <- NULL
    out
  }))
}

surv_panel <- function(fit, palette, xmax, title, show_legend = TRUE,
                       base_size = 11, line_size = 0.7) {
  d <- step_expand(tidy_km(fit))
  pal_named <- setNames(palette, levels(d$arm))
  # Two subtleties for the legend:
  #   (a) Map fill only on the ribbon layer, not globally on ggplot(),
  #       and suppress the fill guide -- otherwise colour and fill
  #       each produce a 4-entry legend (8 total per panel).
  #   (b) On panels with show_legend = FALSE, remove the colour guide
  #       AT THE SCALE level (guide = "none") and via show.legend on
  #       the line layer. theme(legend.position = "none") only hides
  #       the legend locally, so patchwork::plot_layout(guides =
  #       "collect") would still pick up the unprefixed "Z=0 / Z=1"
  #       entries from the single (RCT, RWD) panels and merge them
  #       with the prefixed "RCT - / RWD - " entries from the
  #       combined panels, giving 8 items in the shared legend.
  ggplot(d, aes(x = time, y = surv, colour = arm)) +
    geom_ribbon(aes(ymin = lower, ymax = upper, fill = arm),
                alpha = 0.18, colour = NA, show.legend = FALSE) +
    geom_line(linewidth = line_size, show.legend = show_legend) +
    scale_colour_manual(values = pal_named,
                        guide = if (show_legend) "legend" else "none") +
    scale_fill_manual  (values = pal_named, guide = "none") +
    coord_cartesian(xlim = c(0, xmax), ylim = c(0, 1)) +
    labs(x = "Time (years)", y = "Survival probability",
         colour = NULL, title = title) +
    theme_minimal(base_size = base_size) +
    theme(legend.position = if (show_legend) "bottom" else "none",
          plot.title      = element_text(face = "bold",
                                         size = base_size * 1.0))
}

# Single panels: suppress their own legend so the figure carries
# exactly one shared 4-strata legend (collected from the combined
# panels by patchwork::plot_layout(guides = "collect") below).
build_page <- function(rct_df, rwd_df, anchor_label) {
  rct_df <- attach_arm(rct_df)
  rwd_df <- attach_arm(rwd_df)

  all_df <- rbind(
    data.frame(t_years = rct_df$t_years, status = rct_df$status,
               stratum = paste0("RCT - ", as.character(rct_df$arm)),
               stringsAsFactors = FALSE),
    data.frame(t_years = rwd_df$t_years, status = rwd_df$status,
               stratum = paste0("RWD - ", as.character(rwd_df$arm)),
               stringsAsFactors = FALSE))
  all_df$stratum <- factor(all_df$stratum, levels = c(
    "RCT - ZDV mono", "RCT - Combination",
    "RWD - ZDV mono", "RWD - Combination"))

  fit_rct <- survfit(Surv(t_years, status) ~ arm,     data = rct_df)
  fit_rwd <- survfit(Surv(t_years, status) ~ arm,     data = rwd_df)
  fit_all <- survfit(Surv(t_years, status) ~ stratum, data = all_df)

  rct_xmax <- max(rct_df$t_years, na.rm = TRUE)
  rwd_xmax <- max(rwd_df$t_years, na.rm = TRUE)
  cat(sprintf(
    "[%s] RCT n=%d ev=%d xmax=%.2f y   RWD n=%d ev=%d xmax=%.2f y\n",
    anchor_label,
    nrow(rct_df), sum(rct_df$status == 1), rct_xmax,
    nrow(rwd_df), sum(rwd_df$status == 1), rwd_xmax))

  p_rct <- surv_panel(
    fit_rct, pal_rct, rct_xmax,
    sprintf("RCT (ACTG175) -- EFS, x in [0, %.1f y]   n=%d, ev=%d",
            rct_xmax, nrow(rct_df), sum(rct_df$status == 1)),
    show_legend = FALSE)
  p_rwd <- surv_panel(
    fit_rwd, pal_rwd, rwd_xmax,
    sprintf("RWD (MACS) -- EFS, x in [0, %.1f y]   n=%d, ev=%d",
            rwd_xmax, nrow(rwd_df), sum(rwd_df$status == 1)),
    show_legend = FALSE)
  p_cmb_rct <- surv_panel(
    fit_all, pal_all, rct_xmax,
    sprintf("Combined, x in [0, %.1f y] (RCT horizon)", rct_xmax))
  p_cmb_rwd <- surv_panel(
    fit_all, pal_all, rwd_xmax,
    sprintf("Combined, x in [0, %.1f y] (RWD horizon)", rwd_xmax))

  panel <- (p_rct | p_rwd) /
           (p_cmb_rct | p_cmb_rwd) +
    plot_layout(guides = "collect") &
    theme(legend.position = "bottom")
  panel +
    plot_annotation(
      title    = sprintf(
        "Kaplan-Meier survival: ACTG175 (RCT) vs MACS (RWD) -- %s",
        anchor_label),
      subtitle = paste(
        "EFS endpoint: RCT composite `cens`;",
        "RWD all-cause death (v5).",
        "Bands are 95% step-shaped filled CIs; y-axis spans 0-1."))
}

page_default <- build_page(
  rct,   rwd,
  "default anchor (first 1991-95 ART record)")
page_newuser <- build_page(
  rct,   rwd_nu,
  "new-user anchor (no pre-1991 ART)")
page_maleonly <- build_page(
  rct_m, rwd,
  "male-only (RCT gender == 1; MACS default cohort)")
page_aligned <- build_page(
  rct_m, rwd_cd4,
  "trial-eligibility-aligned (RCT male; MACS baseline CD4 in [200, 500])")

# ---------------------------------------------------------------------
# 4. Stitch and save (four-page PDF:
#       1 default | 2 new-user | 3 male-only | 4 trial-aligned)
# ---------------------------------------------------------------------

out_pdf <- "data/analysis 1/survival_curves.pdf"
pdf(out_pdf, width = 14, height = 10)
print(page_default)
print(page_newuser)
print(page_maleonly)
print(page_aligned)
invisible(dev.off())
cat("\nSaved: ", out_pdf, " (4 pages)\n", sep = "")

# ---------------------------------------------------------------------
# 5. Manuscript figure: combined full-horizon plot + ggmagnify inset
# ---------------------------------------------------------------------
#
# Layout (single panel):
#   - Main plot:  combined Kaplan-Meier at the full RWD horizon
#                 (~25 years), all four strata.
#   - Inset:      zoom into the overlapping follow-up window
#                 x in [0, rct_xmax] and y in [zoom_y_lo, 1.0]. By
#                 default it is placed OUTSIDE the main panel, in the
#                 right margin -- coord_cartesian(clip = "off") plus a
#                 fat right plot.margin make room on the canvas. The
#                 inset can therefore be as large as the main panel
#                 (or larger) without overlapping any curves.
#                 Connector lines tie the source rectangle to the
#                 inset.
#
# Tunable:
#   inset_to        c(xmin, xmax, ymin, ymax) in DATA coords of the
#                   main plot. Default puts the inset to the right of
#                   the panel; set values inside [0, rwd_xmax] x [0, 1]
#                   to embed the inset inside the panel instead.
#   right_margin_pt right plot margin (in pt); needs to be at least
#                   the inset's data-space width times (pixels per data
#                   unit). The default (380 pt) leaves room for the
#                   default outside-the-panel inset on a 16-inch
#                   canvas.
#
# Built from the default-anchor data so the manuscript figure matches
# the "main" cohort definition; rerun against `rct_m` / `rwd_cd4` if a
# trial-aligned manuscript figure is wanted instead.

build_manuscript_fig <- function(rct_df, rwd_df,
                                 zoom_y_lo       = 0.5,
                                 inset_to        = NULL,
                                 right_margin_pt = 185,
                                 base_size       = 30,
                                 font_family     = "Times") {
  rct_df <- attach_arm(rct_df)
  rwd_df <- attach_arm(rwd_df)

  all_df <- rbind(
    data.frame(t_years = rct_df$t_years, status = rct_df$status,
               stratum = paste0("RCT - ", as.character(rct_df$arm)),
               stringsAsFactors = FALSE),
    data.frame(t_years = rwd_df$t_years, status = rwd_df$status,
               stratum = paste0("RWD - ", as.character(rwd_df$arm)),
               stringsAsFactors = FALSE))
  all_df$stratum <- factor(all_df$stratum, levels = c(
    "RCT - ZDV mono", "RCT - Combination",
    "RWD - ZDV mono", "RWD - Combination"))

  fit_all  <- survfit(Surv(t_years, status) ~ stratum, data = all_df)
  rct_xmax <- max(rct_df$t_years, na.rm = TRUE)
  rwd_xmax <- max(rwd_df$t_years, na.rm = TRUE)

  # Default: inset to the right of the panel, roughly the same width
  # as the main panel and full y-range (looks like a side-by-side).
  if (is.null(inset_to)) {
    inset_to <- c(rwd_xmax * 0.55, rwd_xmax * 1.2, 0.5, 1.)
  }

  p_full <- surv_panel(
    fit_all, pal_all, rwd_xmax,
    "Kaplan-Meier survival: ACTG175 (RCT) vs MACS (RWD)",
    base_size = base_size, line_size = 1.25)

  p_full +
    # Manuscript version: drop the surv_panel title -- the figure
    # caption in the manuscript carries that information.
    labs(title = NULL) +
    # Override the surv_panel coord so drawing isn't clipped at the
    # panel edge -- the inset and its connector lines need to render
    # outside [0, rwd_xmax].
    
    coord_cartesian(xlim = c(0, rwd_xmax), ylim = c(0, 1),
                    clip = "off") +
    # Thicken and lengthen the coloured key glyphs so the four strata
    # read cleanly at manuscript scale; the in-panel line linewidth
    # (0.7) is unaffected.
    guides(colour = guide_legend(
      override.aes = list(linewidth = 3))) +
    theme(
      # Times throughout the manuscript figure -- propagates to every
      # text element (axis labels, tick labels, legend) via the global
      # `text` slot; individual element_text() overrides below inherit
      # it. Requires cairo_pdf (used by `out_manuscript`) so the system
      # Times face is embedded.
      text            = element_text(family = font_family),
      plot.margin     = margin(t = 10, r = right_margin_pt,
                               b = 10, l = 10),
      legend.text     = element_text(size = base_size,
                                     family = font_family),
      legend.key.width  = unit(40, "pt"),
      legend.key.height = unit(16, "pt"),
      legend.spacing.x  = unit(8, "pt"),
      # Opaque white panel background -- this is what ggmagnify
      # captures from the source area and replays at the destination,
      # so it acts as the inset's own opaque background and masks the
      # underlying main-panel curves in the overlap region. With the
      # theme_minimal default (transparent panel.background), the
      # capture has no background and the underlying curves show
      # through the inset.
      panel.background = element_rect(fill = "white", colour = NA)
      # Note: no panel.border -- the right edge at x = rwd_xmax and
      # the top edge at y = 1 would cut through the inset's footprint
      # (which spans x in [0.8*rwd_xmax, 1.4*rwd_xmax] and y up to
      # 1.095) and look like stray lines behind the inset.
    ) +
    # Belt-and-suspenders: also paint an opaque white rectangle at the
    # destination, so even if ggmagnify's capture/replay doesn't carry
    # the panel background, the inset still sits on a clean surface.
    annotate("rect",
             xmin = inset_to[1], xmax = inset_to[2],
             ymin = inset_to[3], ymax = inset_to[4],
             fill = "white", colour = NA) +
    geom_magnify(
      from     = c(0, rct_xmax, zoom_y_lo, 1.0),
      to       = inset_to,
      # recompute = TRUE re-renders the inset's geoms fresh at the
      # inset's coordinate system instead of capturing the source area
      # as a grid grob and replaying it via a viewport transform. The
      # capture/replay path can lose vector fidelity (especially for
      # alpha-blended ribbons) and is the usual cause of a blurry
      # inset next to a sharp main panel. With recompute = TRUE the
      # inset is a fresh ggplot rendering, fully vector.
      recompute = TRUE,
      # target.linewidth thickens the inset's lines slightly so they
      # read crisply at the smaller inset scale; matched to the
      # bumped main-panel linewidth (1.4).
      target.linewidth = 1.7,
      shadow   = TRUE,
      axes     = "xy",
      colour   = "grey30",
      proj = "single",
      proj.linetype = "dotted",
      linetype = 1)
}

manuscript_fig         <- build_manuscript_fig(rct,   rwd)
manuscript_fig_aligned <- build_manuscript_fig(rct_m, rwd_cd4)

# cairo_pdf renders alpha-transparent regions (our CI ribbons) as
# native vector instead of rasterizing them as the default pdf device
# does -- this is the usual cause of the inset looking "blurry" when
# the inset is small. Wide canvas gives the outside-panel inset room
# to live without being squeezed.  Manuscript PDF is the trial-aligned
# cohort only (page 1 in earlier drafts -- the default cohort -- has
# been removed).
out_manuscript <- "notes/general/figures/survival_curves_manuscript.pdf"
cairo_pdf(out_manuscript, width = 18, height = 10)
print(manuscript_fig_aligned)
invisible(dev.off())
cat("Saved: ", out_manuscript, "\n", sep = "")

invisible(list(rct = rct, rct_m = rct_m,
               rwd = rwd, rwd_nu = rwd_nu, rwd_cd4 = rwd_cd4,
               page_default   = page_default,
               page_newuser   = page_newuser,
               page_maleonly  = page_maleonly,
               page_aligned   = page_aligned,
               manuscript_fig         = manuscript_fig,
               manuscript_fig_aligned = manuscript_fig_aligned))


