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

The observational arm is drawn from the Multicenter AIDS Cohort Study
(MACS), a longitudinal natural-history cohort of men who have sex with
men, enrolled at four U.S. centres beginning in 1984 with semi-annual
visits. We extract HIV-positive subjects (excluding subjects flagged as
seronegative recruits in the 2001--03 and 2010 expansion cohorts) who
have at least one antiretroviral-form record in 1991--1995, the
calendar window over which ACTG175 subjects were on protocol.

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

The outcome is a right-censored time to a composite event of (i) AIDS
diagnosis, (ii) $\geq$50% confirmed decline in CD4 count, or (iii)
all-cause death, matching the ACTG175 primary endpoint. ACTG175
records this directly as $(T, \delta) = (\texttt{days},
\texttt{cens})$. For MACS the same composite is reconstructed from the
`outcome` form: AIDS dates (`DATE1mm/yy`) and CD4-decline dates
(`SELFCD4Dmm/yy`) are available at month resolution, but death dates
are redacted from the MACS public release, so death-only events are
right-anchored at the subject's last laboratory-visit year. All event
and censoring times are computed relative to a subject-specific anchor:
for ACTG175, the date of randomization; for MACS, the year of the
earliest 1991--1995 antiretroviral record on a relevant regimen.
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
(trial) and $S = 0$ for MACS (observational). We fit the FusionForest
model with three forests: a prognostic forest $\mu(X)$, a treatment
forest $\tau(X)$, and a deconfounding forest $g(X)$ that absorbs
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

Three features of this example warrant explicit acknowledgment. First,
the MACS death dates are redacted, so the composite-event time is
imputed by the last lab-visit year for death-only events; this
introduces noise in the upper tail of the survival distribution.
Second, the calendar resolution of MACS outcomes is coarser (year)
than that of ACTG175 (day), and we accommodate this by working on the
log-year scale with a half-year floor. Third, even after the
exclusions, the MACS observational arm is small relative to the trial
$(n_{\text{MACS}} \approx 270$ versus $n_{\text{RCT}} = 2139)$,
limiting the marginal information that fusion can borrow but also
making the example a useful stress test for the FusionForest prior.
