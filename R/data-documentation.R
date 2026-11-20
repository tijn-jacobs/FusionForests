#' Semi-synthetic RCT and real-world data for data fusion
#'
#' A combined data set of a randomised controlled trial (RCT) and a
#' real-world data (RWD) source with a right-censored time-to-event outcome,
#' built to illustrate \code{\link{FusionForest}}.  The covariates are real;
#' the outcomes are simulated.  The data say nothing about any real
#' treatment.
#'
#' @details
#' The covariates come from the German Breast Cancer Study Group 2 trial
#' (\code{\link[survival]{gbsg}}; the RCT rows) and the Rotterdam tumour bank
#' (\code{\link[survival]{rotterdam}}; the RWD rows), harmonised to a common
#' set.  The treatment \code{A} is a generic binary treatment.  In the RCT it
#' is randomised with probability 1/2.  In the RWD it is the real, non-random
#' treatment assignment of the Rotterdam cohort, so it is confounded.  Part
#' of that confounding acts through variables that are not included here
#' (year of surgery and chemotherapy; see \code{\link{breast_fusion_truth}}).
#'
#' Event times follow a Weibull accelerated failure time model whose
#' baseline coefficients and scale were estimated from the real GBSG2
#' outcomes.  The heterogeneous treatment effect and the effects of the
#' unobserved variables are made up.  Censoring times are drawn from each
#' source's real censoring distribution, so follow-up in the RCT is shorter
#' than in the RWD.  The script that builds the data and an example analysis
#' are installed with the package, in
#' \code{system.file("scripts", package = "FusionForests")}.
#'
#' @format A data frame with 3668 rows (686 RCT, 2982 RWD) and 13 columns:
#' \describe{
#'   \item{id}{Row identifier, matching \code{breast_fusion_truth$id}.}
#'   \item{source}{Factor, \code{"RCT"} or \code{"RWD"}.}
#'   \item{S}{Source indicator (1 = RCT, 0 = RWD), as used by
#'     \code{FusionForest}.}
#'   \item{A}{Treatment indicator (1 = treated, 0 = control).}
#'   \item{time}{Observed relapse-free time in years.}
#'   \item{status}{Event indicator (1 = event, 0 = censored).}
#'   \item{age}{Age in years.}
#'   \item{meno}{Menopausal status (0 = pre, 1 = post).}
#'   \item{size}{Tumour size (0 = at most 20 mm, 1 = 20--50 mm,
#'     2 = more than 50 mm).}
#'   \item{grade}{Tumour grade (0 = grade 1 or 2, 1 = grade 3).}
#'   \item{nodes}{Number of positive lymph nodes.}
#'   \item{pgr}{Progesterone receptor (fmol/l).}
#'   \item{er}{Oestrogen receptor (fmol/l).}
#' }
#'
#' @source Covariates from the \pkg{survival} package data sets
#'   \code{gbsg} and \code{rotterdam}; outcomes simulated.
#'
#' @references
#' Royston, P. and Altman, D. G. (2013).  External validation of a Cox
#' prognostic model: principles and methods.  \emph{BMC Medical Research
#' Methodology}, 13, 33.
#'
#' Schumacher, M., Bastert, G., Bojar, H., et al. (1994).  Randomized 2 x 2
#' trial evaluating hormonal treatment and the duration of chemotherapy in
#' node-positive breast cancer patients.  \emph{Journal of Clinical
#' Oncology}, 12, 2086--2093.
#'
#' @seealso \code{\link{breast_fusion_truth}} for the simulation truth.
#'
#' @examples
#' table(breast_fusion$source, breast_fusion$A)
"breast_fusion"

#' Simulation truth for breast_fusion
#'
#' The true treatment effects, uncensored event times and unobserved
#' confounders behind \code{\link{breast_fusion}}.  Rows are aligned with
#' \code{breast_fusion} and share its \code{id}.  Use these only to check
#' estimates; a real analysis would not have them.
#'
#' @format A data frame with 3668 rows and 6 columns:
#' \describe{
#'   \item{id}{Row identifier, matching \code{breast_fusion$id}.}
#'   \item{source}{Factor, \code{"RCT"} or \code{"RWD"}.}
#'   \item{tau}{True conditional average treatment effect on the log-time
#'     scale; \code{exp(tau)} is the acceleration factor.}
#'   \item{logT}{Uncensored log event time (log years).}
#'   \item{chemo}{Chemotherapy (1 = yes, 0 = no); unobserved confounder.
#'     All RCT rows are 1.}
#'   \item{year}{Year of surgery; unobserved confounder.  Real for the RWD
#'     rows, drawn uniformly from 1984--1989 for the RCT rows.}
#' }
#'
#' @source Simulated; see \code{\link{breast_fusion}}.
#'
#' @examples
#' summary(breast_fusion_truth$tau)
"breast_fusion_truth"

#' Synthetic observational data for the HIV data-fusion analysis
#'
#' A synthetic version of the observational data in the HIV analysis of the
#' paper: HIV-positive men on antiretroviral therapy in 1991--95, with a CD4
#' count of 200 to 500 cells/mm\eqn{^3} at baseline.  Combined with the men
#' of the ACTG 175 trial (\code{ACTG175} in the \pkg{speff2trial} package),
#' it runs that analysis end to end:
#' \code{demo("hiv", package = "FusionForests")}.
#'
#' The data are synthetic.  Analyses of them approximate, but do not
#' reproduce, the results of the paper.  Do not use them to draw
#' conclusions about HIV treatment.
#'
#' @details
#' Baseline is the first 1991--95 visit with current antiretroviral
#' therapy, and the treatment is the regimen at that visit.  An event lies
#' in the interval (\code{time_lower}, \code{time_upper}]; other patients
#' are right-censored at \code{time_lower}.  With
#' \code{\link{FusionForest}}, use \code{y = log(time_lower)},
#' \code{status = 0}, \code{observed_left_time = log(time_lower)},
#' \code{observed_right_time = log(ifelse(status == 1, time_upper,
#' time_lower))} and \code{interval_censoring_indicator = status}.
#'
#' @format A data frame with 401 rows and 10 columns:
#' \describe{
#'   \item{trt}{Treatment (1 = an alternative regimen, 0 = zidovudine
#'     monotherapy).}
#'   \item{regimen}{Factor: \code{"ZDV"} (zidovudine), \code{"ddI"}
#'     (didanosine), \code{"ZDV+ddI"} or \code{"ZDV+ddC"} (zidovudine with
#'     didanosine or zalcitabine).}
#'   \item{age}{Age in years.}
#'   \item{cd4, cd8}{CD4 and CD8 count (cells/mm\eqn{^3}).}
#'   \item{race}{Race (0 = white, 1 = non-white).}
#'   \item{prior_art_years}{Years of antiretroviral therapy before
#'     baseline.}
#'   \item{time_lower}{Lower bound of the event time in years: the event
#'     lies after \code{time_lower}.  For right-censored patients, the
#'     censoring time.}
#'   \item{time_upper}{Upper bound of the event time in years: the event
#'     lies at or before \code{time_upper}.  \code{NA} for right-censored
#'     patients.}
#'   \item{status}{Censoring type: 1 = interval-censored, with the event
#'     in (\code{time_lower}, \code{time_upper}]; 0 = right-censored at
#'     \code{time_lower}.}
#' }
#'
#' @source Synthesised from observational data on HIV-positive men.
#'
#' @examples
#' table(hiv_rwd$regimen, hiv_rwd$status)
#'
#' \dontrun{
#' # The full analysis (needs speff2trial; takes long)
#' demo("hiv", package = "FusionForests")
#' }
"hiv_rwd"
