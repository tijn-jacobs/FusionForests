# Example: Fusing the ACTG175 trial with the MACS observational cohort

We illustrate the proposed FusionForest methodology on a survival
outcome by combining a randomized clinical trial with a contemporaneous
observational HIV cohort. The trial and the cohort enrolled overlapping
patient populations over the same calendar window, but under
fundamentally different design regimes: in the trial, treatment is
assigned by randomization within a tightly controlled eligibility
criterion, while in the cohort treatment is chosen jointly by patient
and clinician and may be confounded with prognosis. The aim of the
example is to demonstrate how, under the FusionForest model, the
observational arm can sharpen inference about the conditional treatment
effect on the trial population without distorting it through
unrecognized confounding.

## Data sources

The randomized trial component is the AIDS Clinical Trials Group
Protocol 175 (ACTG175), a four-arm double-blind trial of nucleoside
reverse-transcriptase inhibitor regimens in HIV-1 infected adults with
CD4 counts of 200-500 cells/mm$^3$. We use the public-release version
of the data, comprising $n = 2139$ subjects randomized between December
1991 and October 1992 across four arms (ZDV, ZDV+ddI, ZDV+ddC, ddI),
followed for a maximum of 1231 days.

The real-world-data (RWD) arm is drawn from the Multicenter AIDS
Cohort Study (MACS), a longitudinal natural-history cohort of men who
have sex with men, enrolled at four U.S. centres beginning in 1984
with semi-annual visits. We extract HIV-positive subjects (excluding
subjects flagged as seronegative recruits in the 2001--03 and 2010
expansion cohorts) who have at least one antiretroviral-form record
in 1991--1995, the calendar window over which ACTG175 subjects were
on protocol.

## Treatment definition

Both data sources are restricted to a binary contrast that mirrors the
ACTG175 primary comparison: $Z = 0$ for ZDV monotherapy, and $Z = 1$
for the pooled combination/ddI-monotherapy group (ZDV+ddI, ZDV+ddC,
ddI alone). For ACTG175 this corresponds to the trial's binary
`treat` variable. For MACS, we use the `drugf1` antiretroviral form,
restrict to records currently in use (`AVNW = 2`) within 1991--1995,
and classify each subject by the union of drug codes observed in that
window: subjects on ZDV alone (DRGAV = 092) are assigned to $Z = 0$;
subjects with any record of ddI (147), ZDV+ddI blinded combination
(180), or ZDV+ddC blinded combination (185) are assigned to $Z = 1$.
Subjects whose only exposure is ddC monotherapy (094), ddI+ddC (186),
or triple AZT+ddI+ddC (187) are excluded, as these regimens were not
part of the ACTG175 randomization.

## Outcome

The primary outcome is *event-free survival* (EFS), specifically the
ACTG175 primary endpoint of right-censored time to a composite event
of (i) AIDS-defining diagnosis, (ii) $\geq$50% confirmed decline in
CD4 count from baseline, or (iii) all-cause death. ACTG175 records
this directly as $(T, \delta) = (\texttt{days}, \texttt{cens})$; the
three components are not separately released, so on the RCT side the
composite indicator and its time are taken as published.

For MACS the composite is reconstructed component-wise to match the
ACTG175 definition (this is variant `EFS-cd4-fix`, `v3`, in the
companion script
[`endpoint_comparison.R`](endpoint_comparison.R)):

- **AIDS** is read from the `outcome` form as `AIDSCASE` $\in \{2, 3\}$
  ("CDC AIDS diagnosis" or "AIDS by death only") with event year
  `DATE1yy`.
- **CD4 decline** is reconstructed from the `lab_rslt` time series,
  *not* from the `SELFCD4D` field. The `SELFCD4D` field codes the
  date of the first CD4 measurement falling below an *absolute*
  threshold (<200 cells/$\mu$L or <14%), which does not match
  ACTG175's *relative* $\geq$50%-decline rule. For each MACS subject
  we therefore take their earliest 1991--1995 `LEU3N` measurement as
  the baseline CD4, and define the event year as the year of the
  first subsequent lab with `LEU3N` $\leq 0.5 \times$ baseline.
- **Death** is taken as all-cause: `DEATH` $\in \{1, 2, 3, 4\}$ in
  the `outcome` form (AIDS-prior-dx, AIDS-no-prior-dx, Not-AIDS, or
  Unknown cause). An earlier version of this analysis kept only
  codes 1 and 2 (AIDS-related deaths), which silently dropped
  approximately 25% of all recorded deaths in the cohort.

The MACS public release redacts death dates entirely (verified
empirically: no four-digit year appears in the columns where
`DTHDATEyy` should sit). For subjects flagged as deceased without an
AIDS or CD4-decline date preceding death, the event time therefore
falls back to the subject's last laboratory-visit year. This is a
known downward bias on death-event times and is treated as part of
the MACS-side noise rather than corrected.

All event and censoring times are computed relative to a
subject-specific anchor: for ACTG175, the date of randomization
(day resolution); for MACS, the year of the earliest 1991--1995
antiretroviral record on a relevant regimen (year resolution).
Subjects with an AIDS event preceding the anchor are excluded so that
both populations are AIDS-free at baseline, in line with the ACTG175
entry criterion.

We work throughout on the natural log scale: define $Y = \log T$, with
times expressed in years, clamped from below at 0.5 years to handle
year-resolution outcomes. The FusionForest fit is therefore specified
with `outcome_type = "right-censored"` and `timescale = "log"`.

## Covariates

We harmonize a four-dimensional covariate vector $X = (\text{age},
\text{wtkg}, \text{cd4}, \text{cd8})$ that is well populated in both
sources. Age is taken at anchor (anchor year minus year of birth);
weight in kilograms is derived from `LDWGT` (pounds) in the MACS
physical-exam form via the conversion factor 0.453592, mirroring the
ACTG175 `wtkg` field. Baseline CD4 and CD8 counts are the
subject-level median of `LEU3N` and `LEU2N` measurements within the
1991--1995 window, anchored to the same year as the treatment record.
Sex, race, Karnofsky score, and prior ART history are not retained,
either because they are constant within MACS (sex), inconsistently
encoded across sources, or unavailable in the public-release MACS
forms.

## Fusion estimator

Let $S \in \{0, 1\}$ index the data source, with $S = 1$ for ACTG175
(trial) and $S = 0$ for MACS (RWD). We fit the FusionForest model
with three forests: a prognostic forest $\mu(X)$, a treatment forest
$\tau(X)$, and a deconfounding forest $g(X)$ that absorbs
source-specific bias entering through unmeasured confounders. The
log-survival outcome is decomposed as
$$
Y \,=\, \mu(X) \,+\, \tau(X)\,(Z - \tfrac{1}{2}) \,+\, (1 - S)\,g(X)
\,+\, \varepsilon,
$$
with $\varepsilon$ following the right-censored Gaussian likelihood
implemented in FusionForest. Centred treatment coding is used so that
both arms inform $\tau$ symmetrically, and the deconfounding forest is
active only on observational rows. Each forest is given 200/100/200
trees and a Bayesian additive regression tree prior; the MCMC sampler
is run for 1000 burn-in iterations followed by 2000 posterior draws.
The conditional average treatment effect is reported on the
log-survival scale on the ACTG175 covariate distribution; the
corresponding posterior of the average treatment effect is obtained
by averaging the per-subject CATE samples within source $S = 1$.

## Caveats

Five features of this example warrant explicit acknowledgment.

1. **Death-time imputation.** MACS death dates are redacted from the
   public release, so for subjects flagged as deceased without a
   preceding AIDS or CD4-decline date, the event time falls back to
   the subject's last laboratory-visit year. This is a downward
   bias on death-event times and propagates into the upper tail of
   the survival distribution.
2. **Time resolution.** MACS outcomes are at year resolution, ACTG175
   at day resolution. We accommodate this by working on the log-year
   scale with a half-year floor.
3. **CD4 component is reconstructed on the MACS side.** ACTG175's
   $\geq$50%-confirmed-decline rule cannot be read directly from the
   MACS `outcome` form (whose `SELFCD4D` codes an absolute <200/<14%
   crossing, a different event). We therefore reconstruct the
   relative rule from each subject's `LEU3N` time series in
   `lab_rslt`, using the earliest 1991--1995 reading as baseline.
   The MACS reconstruction is at year resolution and is subject to
   non-uniform CD4-measurement frequency between subjects, whereas
   the ACTG175 indicator was confirmed by a follow-up measurement.
   Variants of this rule (absolute threshold, dropping the CD4
   component, all-cause-death only) are tabulated in
   [`endpoint_comparison.R`](endpoint_comparison.R).
4. **Cohort size asymmetry.** Even after exclusions, the MACS RWD
   arm is small relative to the trial $(n_{\text{MACS}} \approx 270$
   versus $n_{\text{RCT}} = 2139)$, limiting the marginal
   information that fusion can borrow but also making the example a
   useful stress test for the FusionForest prior.
5. **The RCT components are not separately released.** ACTG175's
   public `cens` is the composite of all three components, so the
   RCT side cannot be re-specified as overall survival or as
   AIDS-and-death only without an external data source. The MACS
   side could be re-specified as either; the script
   `endpoint_comparison.R` shows how the MACS event counts shift
   under those alternative rules.
