# Supervisor feedback — manuscript v3 (23 June 2026)

Transcription of the handwritten comments, organised by section. Each entry quotes the
manuscript text the comment points at, followed by the comment as written.

_Status: complete. Main manuscript pp. 1–21 and Supplementary Materials pp. 1, 2, 3, 5.
Pages not listed carry no comments._

_Tracking: a **✅ Resolved:** note marks a point already addressed in the working draft,
with a one-line description of how. Unmarked points are open. Additional general remarks
received 28 July 2026 are included below, merged into existing entries where they
duplicate them._

_Last full audit against the draft: 29 July 2026. Every entry below was checked against
the current .tex sources: 59 actionable points, of which 51 are resolved and 8 open. Four
further comments are purely complimentary and need no action; they carry no ✅._

---

## General remarks

Written across the top of page 1: _"Op het manuscript heb ik vooral kleine punten
aangegeven. Na het lezen van het hele document is dit mijn belangrijkste feedback:"_

- _"laat ons meer aanwezig zijn in het paper. Welke keuzes hebben we gemaakt en waarom?
  Dus niet: 'Er wordt BART gebruikt', maar 'We hebben voor BART gekozen omdat…'"_
  **✅ Resolved:** ownership added at the main choice points: the estimand (§2.1 "We choose
  this estimand for its interpretation and its causal properties"), BART (§2.4 "We build
  each regression function… and adopt the standard formulation" plus three reasons), the
  ladder (§2.5 two-considerations paragraph), and the intro vehicle sentence ("We build our
  framework on Bayesian additive regression trees…").
- _"hoe ben je tot de keuze voor methoden om mee te vergelijken gekomen? dit is me niet
  duidelijk geworden en het komt nu wat mager over op dat vlak"_
  **✅ Resolved (29–30 July, pink):** §3 now names four method families with references
  and the selection logic (scarcity of comparable methods; three structural limitations;
  the adaptations made); the full four-method table sits in the main text (Table 1) with
  the figure showing the strongest competitor; SM sm:sim-competitors keeps the method
  descriptions.
- _"hamer er meer op welke problemen dit oplost, i.h.b. random censoring"_
  **✅ Resolved:** censoring now hammered across the paper: §2.1 mixed-types sentence, the
  intro novelty paragraph ("in a single sampler") and its interval-censoring gap sentence,
  §3 setup ("The two censoring types mirror the application"), §4 competitor sentence, and
  a Discussion recap sentence ("The model handled the right-censored trial and the
  interval-censored cohort jointly").
- _"door het te hebben over 'a flexible black-box competitor' zwak je het argument
  eigenlijk af. Het is er maar één, en er zit een soort impliciete kritiek ('black box')
  in. Waarom verwijs je niet naar die methode met de naam danwel de auteurs? En een
  subtiel verschil: heb het er liever over dat het Bayesian fusion forest wél
  interpreteerbaar is dan dat een ander algoritme dat niet is."_
  **✅ Resolved:** "black box" removed throughout; the competitor is named at first use in
  §3 and §4 (deepAFT, norman2024deepaft); framing flipped positive — the intro now says
  "Machine-learning predictors … offer comparable flexibility, and we benchmark our
  framework against them", and §4 closes on "The fusion delivers this certainty with
  calibrated uncertainty" without the negative tail.

### Additional general remarks (received 28 July 2026)

_Two further remarks duplicate existing entries and are merged there: the introduction
structure / novelty point (see §1 → Overall) and the regularisation-ladder question
(see §2.5). The rest:_

- _"De onderbouwing van de keuze voor methoden om mee te vergelijken. De vergelijking komt
  nu erg beperkt over, maar wellicht is er een goede reden om het te houden bij de methodes
  die je gekozen hebt? Aangeven dat je methode beter werkt dan één alternatief dat niet eens
  een naam krijgt, maar wordt benoemd als 'een flexibele black-box methode', komt over als
  het winnen van een wel erg makkelijke wedstrijd zolang de mogelijkheid open wordt gelaten
  dat je gewoon een zwakke methode hebt gekozen om mee te vergelijken. Mocht het
  bijvoorbeeld zo zijn dat dit de enige andere methode is die momenteel bestaat voor
  gecensureerde uitkomsten, dan wordt het verhaal ineens een stuk sterker."_ — deepens the
  second and fourth bullets above.
  _Progress (28 July):_ the §3 comparator paragraph now closes with the explicit claim:
  "No existing alternative combines causal effect estimation with data fusion and support
  for interval censoring. The fusion forest offers all three."
- _"Wellicht kan je duidelijker aanzetten dat onze methode right en interval censoring kan
  combineren. En dan duidelijk maken dat de blackbox machine learner dat niet kan."_ —
  extends the censoring bullet above.
  **✅ Resolved:** stated in four places: §2.1 ("The two sources may carry different
  censoring types: in our application the trial is right-censored while the cohort is
  interval-censored"), the §3 comparator paragraph (none of the ML methods supports
  interval censoring), §4 main text ("The network cannot use the interval-censored MACS
  outcomes and receives a right-censored approximation instead"), and the Discussion
  recap.
- _"Het punt van het vergelijk met de blackbox machine learner is mij onduidelijk. Denk dat
  dat aan helderheid en stroomlijning kan winnen."_
  _Progress (28 July):_ intro ¶5 now states the purpose up front ("…offer comparable
  flexibility, and we benchmark our framework against them in the simulation study and the
  application").
  **✅ Resolved (30 July):** the §3 comparison now has a clear arc — scarcity, named
  methods, three limitations, adaptations — and the results paragraph states why the
  figure shows the deep network (strongest alternative) with the full comparison in the
  SM.

**Notation:** a star (⭐) in the text means _"referentie(s) toevoegen"_. The mark is used
throughout the paper.

**Recurring style point:** strings of consecutive sentences all starting with "We …" read
poorly and should be combined. Raised in the abstract and at §1, paragraph 1.

---

## Title and abstract (p. 1)

- At the title _"Bayesian tree ensembles for heterogeneous treatment effects on survival
  from randomised and real-world data"_ — "tree ensembles" underlined, with above it:
  _"'fusion forests'?"_
  **✅ Resolved:** title now reads "Bayesian fusion forests …" (in revision colour).
- In the left margin, with a bracket spanning _"We relax the unconfoundedness assumption on
  the real-world data."_ through _"We allow for both right- and interval-censored data."_:
  _"dit is 5x 'We [verb]' in 7 zinnen, meer afwisseling in zinsstructuur zou de flow
  verbeteren"_
  **✅ Resolved:** abstract rewritten; "We…" openings reduced from 7 to 3 with varied
  subjects (The framework…, A shared baseline prognosis…, The model…).
- With an arrow to _"A simulation study shows efficiency gains over a trial-only analysis
  across varying levels of confounding and between-source heterogeneity."_: _"met een
  trade-off? Of waarom dan dit? Een tipje van de sluier oplichten maakt deze claim
  geloofwaardiger"_
  **✅ Resolved:** added after the relaxation claim: "In return, we assume the treatment
  effect transports across the two sources."
- At _"The Bayesian fusion forest also improves on a flexible black-box competitor in both
  accuracy and calibration."_ — "a flexible black-box competitor" underlined, with:
  _"omdat het er maar één is, komt dit niet zo sterk over. Niet sterk genoeg om in de
  abstract te vermelden"_
  **✅ Resolved:** the sentence was removed from the abstract entirely. The comparison is
  still made in §3 and §4, where the method is named.
- At _"The fusion identifies a benefit for nearly every patient where the trial alone is
  inconclusive."_ — written above "where": _"whereas?"_
  **✅ Resolved:** "where" → "whereas".
- At _"Alternative titles:"_: _"ik ben voor je huidige keuze, maar dan met 'fusion forests'
  erin"_
  **✅ Resolved:** current title kept with "fusion forests" in it; the alternatives list has
  been deleted from main.tex.

---

## 1. Introduction

_The circled numbers ①–⑥ are the supervisor's paragraph numbering._

### Overall (p. 3, top margin)

⭐ _"De introductie voelt nog wat hak op de tak. Het lijkt te beginnen met een duidelijke
structuur (de 3 uitdagingen), maar nadat punt 1 daarvan wordt besproken nemen we een afslag
naar Bayes, i.h.b. bomen, en keren we niet meer terug. Idee: beschrijf tussen ② en ③ al
vast wat je oplossing gaat zijn, zgn. een overgang à la: 'Some of these ideas have appeared
before: [lit review]'. En eindig dan met: 'What's novel about our approach is …'"_

_Reiterated in the additional general remarks (28 July):_ _"De introductie, die een mooie
structuur belooft met drie uitdagingen, maar vervolgens na het behandelen van één uitdaging
een andere afslag neemt. In de introductie mag je tevens veel explicieter benoemen wat er
nou nieuw is."_
**✅ Resolved:** the restructured version has been adopted and the old introduction
deleted; INTRODUCTION.tex now holds one section. The three-challenges thread runs through
the whole introduction: a solution paragraph labelling each challenge ("The Bayesian
fusion forest addresses the three challenges in a single nonparametric model. For the
first / second / third challenge, … Existing methods meet some of these challenges but
none meets all three"), challenge signposts opening both literature paragraphs, the
interval-censoring gap sentence closing the frequentist paragraph ("Neither accommodates
interval censoring, the third challenge"), an owned BART choice ("We impose a Bayesian
additive regression tree (BART) prior on each component of our model"), and the novelty
claims distributed to the paragraphs where each challenge is discussed rather than
collected in a closing paragraph (round-2 revision).

### Paragraph 1 (p. 2)

- At _"Researchers estimate the HTE from either a randomised controlled trial (RCT) or
  observational real-world data (RWD)."_ — sentence struck through, with the suggested
  replacement: _"of iets als: 'HTE, the key ingredient for precision medicine, are
  typically estimated using either data from an RCT or RWD.'"_
  **✅ Resolved:** sentence deleted (the margin comment below preferred deletion over the
  replacement); the precision-medicine link moved into the new opener.
- With an arrow pointing to _"heterogeneous treatment effects"_: _"zet dit vooraan, want
  daar gaat je paper over, niet over precision medicine"_
  **✅ Resolved:** ¶1 now opens "Heterogeneous treatment effects (HTE) are central to
  precision medicine."
- In the left margin, on the same struck-through sentence _"Researchers estimate the HTE
  from either a randomised controlled trial (RCT) or observational real-world data
  (RWD)."_: _"volgens mij kunnen we prima zonder deze zin, en het stelt het erg binair
  tegenover elkaar terwijl je niet weet of dat terecht is"_
  **✅ Resolved:** deleted; the RCT and RWD acronym definitions moved into the two
  sentences that follow.
- At _"Randomised trials are the gold standard for causal effect estimation, but they are
  underpowered for subgroup effects and often have limited follow-up."_ — ⭐ at the end of
  the sentence.
  **✅ Resolved:** cites rothwell2005subgroup (already in the bib).
- Spanning _"Real-world data from registries and cohorts are larger and often follow
  patients for longer. Regulators increasingly accept them as evidence."_: _"één zin van
  maken"_
  **✅ Resolved:** merged into one sentence ("…, and regulators increasingly accept them
  as evidence").
- At _"Regulators increasingly accept them as evidence."_ — ⭐ at the end of the sentence.
  **✅ Resolved:** cites sherman2016realworld and fda2018framework (already in the bib).
- Spanning _"We combine the two data sources. We identify the effect from the trial and
  borrow power and longer follow-up from the real-world data."_: _"al dat 'We …' achter
  elkaar leest niet fijn, hier zou ik ook één zin van maken"_
  **✅ Resolved:** merged with a colon: "We combine the two data sources: the trial
  identifies the treatment effect and the real-world data contribute power and longer
  follow-up."

### Paragraph 2 (p. 2)

- At _"The third is the survival outcome."_ — "the survival outcome" struck through and
  replaced by "censoring", giving _"The third is censoring."_
  **✅ Resolved:** applied as written.

### Paragraph 3 (p. 2)

- No comments.

### Paragraph 4 (p. 2)

- At _"Bayesian methods combine data sources by borrowing information through the prior."_
  — "the prior" circled, with: _"Is dit de enige manier? Zo lijkt je het nu te stellen,
  maar in de volgende zin gaat het al over de likelihood."_
  **✅ Resolved:** topic sentence now reads "through the prior or the likelihood"; the
  power-prior sentence then instantiates the likelihood route and the MAP prior the prior
  route.
- At _"None of these Bayesian methods targets heterogeneous treatment effects"_ — an
  insertion caret after "targets", with no legible text. FIXED.

### Paragraph 5 (p. 3)

- In the right margin, with a bracket spanning _"We will show that a structured approach
  tailored to the problem at hand and its statistical constraints outperforms a generic
  black-box competitor in both accuracy and calibration."_: _"wees explicieter over je
  bijdrage, want nu leest het alsof dit het is, terwijl ik denk dat het dit is"_ — with an
  arrow pointing down to paragraph 6.
  **✅ Resolved:** the "We will show that a structured approach…" sentence is deleted; the
  result-preview role is now carried by the §3 comparator paragraph. ¶5 ends on the
  ML-limitations sentence, setting up ¶6 as the contribution.

### Paragraph 6 (p. 3)

- At the paragraph _"We develop a Bayesian nonparametric framework to estimate
  heterogeneous treatment effects on survival outcomes. …"_, continuing the comment above:
  _"Maar dat komt nu over als een 'organization of the paper' paragraaf"_
  **✅ Resolved:** ¶6 rebuilt as an explicit novelty paragraph: "Three elements of the
  framework are new" (decomposition + identification with Proposition refs; per-component
  priors with borrowing and confounding shrinkage; source-specific nonparametric error),
  closing with "To our knowledge no existing method offers this combination."

---

## 2. Methodology

### 2.1 Causal framework (pp. 3–4)

- At _"We use the convention that R = ∞ stands for T ∈ [L, ∞). These correspond to exact
  observation (δ = 1: L = R = T), …"_ — "These" circled, with: _"Hier volg ik het even
  niet, waar verwijs je naar met 'these'? Kan het explicieter?"_
  **✅ Resolved:** rewritten — δ introduced as distinguishing three observation types, one
  explicit sentence per type; the R = ∞ convention folded into the right-censoring line.
- At _"The corresponding estimand on the multiplicative time scale is the acceleration
  factor exp(τ(x))."_ — ⭐ at the end of the sentence.
  **✅ Resolved:** cited `pang2021flexible` (Pang, Platt, Schuster & Abrahamowicz, 2021,
  _Stat Methods Med Res_ 30(11):2526–2542) at the end of the sentence, in revision colour.
- At Assumption 1, _"Consistency. T = A T(1) + (1 − A) T(0)."_, with an arrow drawn from
  _"We impose the following assumptions on the causal structure and the censoring
  mechanism."_: _"Ik ben deze aanname niet in deze vorm gewend, is er literatuur waarin het
  zo staat?"_
  **✅ Resolved:** restated in the standard conditional form: "For each a ∈ {0,1}, if
  A = a then T = T(a) and C = C(a)" — now also covering the censoring process, and
  matching the form the SM proof invokes. The implied equality of the observed triple
  (L, R, δ) with its potential counterpart is a corollary, to be stated as a remark in the
  assumption-context pass, along with a citation.
- In the right margin, with a brace spanning Assumptions 1–5 (from _"Consistency"_ through
  _"Conditionally non-informative censoring"_): _"geef (meer) context bij elke aanname,
  i.h.b. in hoeverre dit standaard aannames zijn"_
  **✅ Resolved:** a context paragraph now follows the assumption block: Assumptions 1, 3
  and 5 labelled standard with one or two sentences each (consistency content + the
  observed-triple corollary; positivity as by-design in the trial and overlap in the RWD;
  censoring as the conditional non-informative version, citing sun2006statistical), then a
  bridge sentence handing Assumptions 2 and 4 to the existing identification-strategy
  paragraph. Both citations reuse existing bib entries (dahabreh2019generalizing for
  positivity/overlap; sun2006statistical for censoring) — zero new references.

### 2.2 Accelerated failure time decomposition (pp. 4–5)

- At the section heading _"2.2 Accelerated failure time decomposition"_: ⭐ _"ref naar
  degene die de confounding function heeft geïntroduceerd"_
  **✅ Resolved:** the definition of c(x) now cites `robins2000sensitivity`, in revision
  colour.
- At _"Treatment assignment depends on confounders outside X, which bias the observed
  contrast."_ — "depends" struck through, replaced by "may depend".
  **✅ Resolved:** now reads "Treatment assignment may depend on confounders outside $X$".
- At Proposition 1, _"Accelerated failure time decomposition"_, in the left margin: _"dit
  en prop 2 zijn jouw resultaten, wees dan minder bescheiden."_
- Continuing below, on the same proposition: _"Wees er duidelijk in dat jij dit hebt
  gevonden, en benoem waarom het nuttig is."_
  **✅ Resolved (inverted after a literature check):** Prop 1 and Prop 2(i)–(ii) turn out
  to be close analogues of results in Yang et al. (2025, Bernoulli) and Ye et al. (2025),
  so asserting ownership would be wrong. Instead: one short attribution sentence after
  Prop 2 ("Similar identification results appear in …"), and the immodesty redirected to
  what is defensibly ours — part (iii) (the structural non-identification paragraph) and
  the component-wise prior construction. The intro First-claim was recast accordingly
  ("we formalise … and show that the trial is necessary to separate the treatment effect
  from the confounding bias").
- And: _"Noem dit ook in de introductie."_
  **✅ Resolved:** the intro novelty paragraph names the propositions; after the
  literature check its claim was recast to what is ours ("we formalise … and show that
  the trial is necessary to separate the treatment effect from the confounding bias
  (Propositions 1 and 2)").
- At equation (5), _"log T = m₀(X, S) + A τ(X) + (1 − S) A c(X) + ε"_ — the term "A τ(X)"
  circled, with: _"voor consistentie met (3)"_ (i.e. write it as τ(X) A, as in equation 3).
  **✅ Resolved:** equation (5) now writes the term as τ(X) A, matching (3), in revision
  colour.

### 2.3 Nonparametric error distribution (p. 5)

- In the right margin at the section heading and opening paragraph _"We model the error ε
  in (5) with a hierarchical Dirichlet process mixture of normals (HDPM)."_: _"je zou hier
  eerst globaal vertellen waarom / wat is wenselijkst van het modelleren van de verdeling
  van ε? Het wordt op zich al snel duidelijk, maar ik zou anders beginnen"_
  **✅ Resolved:** subsection now opens with "Four requirements guide our model for the
  error ε…" (flexible shape, source-specific, linked, mean zero), closing with the HDPM
  meeting all four; the centring paragraph refers back to the fourth requirement.

### 2.4 Overview of BART (p. 6)

- In the left margin, with a bracket spanning _"Bayesian additive regression trees
  (Chipman et al., 2010) provide a flexible nonparametric prior over an unknown function
  f : ℝᵖ → ℝ. The prior represents f as a sum of J regression trees:"_ through _"The
  function g(x; Tⱼ, Hⱼ) routes x through the splits of Tⱼ to a terminal node r and returns
  the step height h\_{j,r} at that node."_: _"past iets als: 'We choose the standard
  formulation of BART'? ik mis in deze tekst een beetje onze aanwezigheid: hebben wij
  keuzes gemaakt, of is deze prior ons overkomen?"_
  **✅ Resolved:** §2.4 now opens with "We build each regression function in our model on
  BART and adopt the standard formulation of Chipman et al." plus three reasons for the
  choice (flexibility, interpretable per-component regularisation, conjugate Gibbs updates
  extending to censoring).

### 2.5 Prior specification (pp. 7–8)

- At the section heading _"2.5 Prior specification"_ — "Prior specification" struck
  through, replaced by _"The Bayesian fusion forest"_.
  **✅ Resolved:** subsection heading now reads "The Bayesian fusion forest", in revision
  colour.
- In the right margin at the opening paragraph _"We place a separate Bayesian additive
  regression tree prior on each component of the decomposition (3): the baseline prognosis
  m₀, the treatment effect τ, and the confounding function c. …"_: _"mooie toelichting"_
- With an arrow to the end of _"The stronger regularisation reflects the expectation that
  the treatment-effect surface is simpler than the baseline prognosis."_: _"kun je voordat
  je de ladder bespreekt kort toelichten waarom je τ en c sterker regulariseert?"_
  _Reiterated in the additional general remarks (28 July):_ _"Je maakt duidelijke keuzes
  met je ladder aan regularisatie, en het klinkt goed. Ik zou graag beter begrijpen: waarom
  kies je voor het ene element voor sterkere regularisatie dan het andere?"_
  **✅ Resolved:** a "Two considerations order the ladder" paragraph now precedes the
  per-component discussion — information and expected complexity, delivered in one pass
  per component (prognosis: every observation, complex → most flexible prior; treatment
  effect: identified from the trial contrast alone, few modifiers → shrunk to
  homogeneity, guarding against spurious heterogeneity; confounding function: only the
  contrast difference, bias from unobserved covariates → shrunk to a constant shift).
  Iterated 28 July from a longer two-pass version to this six-sentence form. The
  now-duplicated sentence the arrow pointed at was deleted.
- At _"The confounding function c is the third component, and we regularise it most strongly
  of all."_ — "regularise it most strongly of all." circled.
  **✅ Resolved:** the circle asked why, with no text. The new ladder paragraph answers it
  before the sentence is reached: the confounding bias arises from covariates we do not
  observe, so there is little reason to expect it to vary with the observed ones, hence the
  shrinkage towards a constant shift. The sentence itself is unchanged.
- With an arrow to _"We refer to this model as the Bayesian fusion forest."_: _"heel mooi!
  Ik zou het wel al eerder aankondigen, in elk geval in de headers van deze subsectie"_
  **✅ Resolved:** the name now appears in the §2.5 heading, and earlier in the title and in
  the opening sentence of introduction ¶6 ("We develop the _Bayesian fusion forest_, a
  nonparametric framework for …"). The §2.5 sentence "We refer to this model as the Bayesian
  fusion forest" is kept as the formal definition.
  _Update (28 July):_ the abstract now opens "We develop the _Bayesian fusion forest_, a
  nonparametric framework…", so the name appears in title, abstract, introduction ¶6, and
  the §2.5 heading.

---

## 3. Simulation study

### Opening paragraph (p. 9)

- At _"We compare the Bayesian fusion forest on the combined data sources with a Bayesian
  causal forest on the RCT and RWD alone."_ — "Bayesian causal forest on the RCT and RWD
  alone" underlined, with: _"Waarom Bayesian causal forest?"_
  **✅ Resolved:** added: "The causal forest is the single-source counterpart of the fusion
  forest, so the comparison isolates the value of combining the sources rather than
  differences in model class."
- At _"We assess its robustness to unmeasured confounding and to between-source
  heterogeneity, benchmark it against a flexible deep-learning competitor, and study its
  behaviour as the covariate dimension grows."_: _"dit voelt wat mager, maar wellicht is er
  een goede reden voor — deel die dan gelijk. Zie ook mijn opmerking op de volgende
  pagina"_
  _Progress (28 July):_ sentence rewritten to "benchmark it against flexible
  machine-learning methods" to make room for an extended comparator set.
  **✅ Resolved (30 July):** the extended set is run and written up — four families in §3
  with the selection rationale, full table and references in SM sm:sim-competitors.

### 3.1 Simulation setup (p. 10)

- In the right margin, with an insertion mark before _"The single-source baseline is an
  accelerated failure time Bayesian causal forest (Jacobs, 2026), fitted to the trial and
  to the real-world data separately."_ and a long arrow down to item _"2) Comparison with a
  black-box competitor."_: _"het verdient een eigen paragraaf, aan het begin van sectie 3,
  om toe te lichten waar je mee vergelijkt en waarom je die selectie hebt gemaakt"_
  **✅ Resolved (structure):** comparator paragraph added as the second paragraph of §3:
  single-source causal forests as the two reference points (unbiased-but-imprecise,
  precise-but-confounded), then the machine-learning class with its three structural
  limitations (no causal target → S/T-learners; no fusion → trial-only and naive pool; no
  interval censoring → right-censored versions). The individual method names carry a TODO
  comment until the extended comparison is final. The duplicate baseline sentence in §3.1
  was removed.

### Figure 1 (p. 11)

- At the y-axis label _"RMSE"_ of the top-left panel: _"zet deze even rechtop met de
  'las=…' optie"_
  **✅ Resolved:** `las = 1` added in all three figure scripts (exp1, exp2, exp3); axis-title
  spacing widened; scripts now write straight to notes/general/figures/; figures
  regenerated.
- At the figure legend _"Fusion / RCT−only / RWD−only"_: _"ik vind je figuren mooi, met
  rood en groen vraag ik me wel af hoe leesbaar het is voor kleurenblinde mensen?"_
  _Closed (28 July): current palette kept by decision; no change._

### 3.2 Simulation results (p. 11)

- At the section heading _"3.2 Simulation results"_: _"mooi gepresenteerd."_

---

## 4. Application: survival after HIV (pp. 14–15)

- At the legend of Figure 4, _"RCT - ZDV mono / RCT - Combination / RWD - ZDV mono / RWD -
  Combination"_, with a bracket under the legend: _"dit verschil is haast niet te
  onderscheiden, misschien biedt een RColorBrewer palet uitkomst"_
  **✅ Resolved:** palette in `data/analysis 1/survival_curves.R` widened: "RCT - ZDV mono"
  tomato → darkorange, so the RCT pair no longer reads as two near-identical reds against
  firebrick; "RWD - Combination" steelblue → steelblue4 to separate the RWD pair too.
  Figure regenerated.
- At _"The Bayesian fusion forest estimates each patient's effect far more precisely than
  the trial alone."_ — "more precisely" underlined, with: _"dat kunnen we toch niet zeggen,
  of heb ik gemist dat we de echte causale effecten weten?"_
  **✅ Resolved:** now reads "estimates each patient's effect with lower posterior variance
  than the trial alone", in revision colour. The claim is about the posterior, not about
  closeness to the unknown truth. Matches the phrasing already used in the Discussion.
- In the top margin of page 15: _"waar vinden we de code voor je simulaties en
  data-analyse?"_
  **✅ Resolved:** Data availability now states the GitHub repository also holds the
  simulation and analysis code.

---

## Funding (p. 17)

- At _"This work was supported by the European Research Council [grant number 101074802];
  and the SURF Cooperative [grant number EINF-18803]."_: _"dit moet wat uitgebreider, zie
  mijn eerdere mails voor de volledige tekst"_
  **✅ Resolved:** full ERC Horizon Europe disclaimer plus the SURF e-infrastructure
  sentence added to DISCUSSION.tex, in revision colour.
  _Update (28 July):_ correct grant number is 101074802; the new block is corrected to
  match. _Update (29 July):_ the short one-line statement is now commented out in the
  source (not deleted); only the full disclaimer renders.

---

# Supplementary Materials

## S.1.1 Proof of Proposition 1 (SM p. 1)

- At _"Case A = 0, S ∈ {0,1}. The two treated-arm terms in (S1) vanish."_ — "treated-arm
  terms" underlined, with an arrow from equation (S2): _"deze terminologie heb ik in het
  paper niet eerder gezien. Het kwartje valt snel, maar ik zou hier nog toevoegen: ',
  τ(X)A and (1−S)A c(X),'"_
  **✅ Resolved:** the sentence now names the two terms: "The two treated-arm terms in
  (S1), τ(X) A and (1 − S) A c(X), vanish." The appositive is in revision colour.

## S.1.1 continued (SM p. 2)

- At _"The second term equals τ(X)."_ — sentence struck through, with a bracket in the left
  margin spanning _"The first term equals m₀(X, 1) by (S2) at a = 0:"_ through equation
  (S5): _"dit beargumenteer je pas in de volgende zin, maar leest alsof het al bewezen is.
  Weglaten kan hier prima"_
  **✅ Resolved:** sentence deleted. The transportability argument that follows now states
  the result once, at the point where it is established.
- At _"The RCT observes this distribution."_ — sentence underlined, with: _"deze verwoording
  is wat ongebruikelijk, maar kan hier prima weggelaten worden"_
  **✅ Resolved:** sentence deleted; the functionals sentence runs straight into "This
  identifies τ from the RCT alone."

## S.1.2 Proof of Proposition 2 (SM p. 3)

- At equation (S14), _"c(X) = E[log T | A = 1, X, S = 0] − E[log T | A = 0, X, S = 0] −
  τ(X)"_, with a brace in the right margin: _"hebben we de stappen (S12) en (S13) dan nog
  nodig?"_
  **✅ Resolved:** no, they were a round trip — (S12) restates the A = 1, S = 0 case of
  Prop. 1, which was itself obtained by rearranging the definition of c, so subtracting
  (S13) returned to that same definition. The same two equations also reappear verbatim in
  Part (iii), where they do real work (two equations, three unknown functions). Part (ii)
  now opens straight from the definition (S8). Neither deleted equation carried a label,
  so nothing referenced them; later SM equation numbers shift down by two.
- At _"Part (i) supplies such a τ from the RCT."_: _"Is dit argument eerder gebruikt in de
  context van confounding functions? Benoem dat dan"_
  **✅ Resolved:** sentence added after it, in revision colour: "This is the identification
  strategy underlying the confounding-function approach to combining randomised and
  observational data \citep{Kallus2018, YangLiuZengWang2025}."
  YangLiuZengWang2025 (Bernoulli 31(4):2987–3012) verified as apt: joint estimation of τ
  and c from both sources, semiparametrically efficient, plus the test for unmeasured
  confounding.
  _Kallus2018 accepted as apt (28 July); no further verification needed._
- At _"Larger RWD samples cannot resolve it."_ — underlined, with: _"neem je hier alvast een
  voorschot op vragen van reviewers?"_
- And below: _"(ik vind het een leuke toevoeging en misschien zelfs wel op zijn plek in de
  main text na prop. 2)"_
  **✅ Resolved:** a four-sentence paragraph now follows Proposition 2 in the main text, in
  revision colour: the non-identification in part (iii) is structural rather than
  statistical, larger real-world samples cannot resolve it, separating τ from c needs
  external information, and the trial supplies it through part (i). That last sentence is
  new relative to the SM, turning the caveat into the motivation for the design. The SM
  keeps its own version so the proof stays self-contained; the overlap is intentional.

## S.2 (SM p. 5)

- In the right margin, with a bracket spanning _"The construction justifies the two-forest
  decomposition of the baseline prognosis. A single BART on m₀(X, S) with S as a covariate
  cannot encode the preference for borrowing, and two independent baselines would imply a
  needlessly diffuse prior on their difference. The shared-plus-deviation form instead
  places the prior directly on the between-source difference and tunes the borrowing
  through the deviation scale."_: _"dit taalgebruik zou ik laten terugkomen in de intro van
  Sectie 2.5 (die al mooi is, maar ik vind dit nog helderder)"_
  **✅ Resolved:** the SM core (single BART cannot encode the preference; independent
  baselines imply a needlessly diffuse prior on the difference; shared-plus-deviation
  places the prior on the difference and tunes borrowing via k_d) now replaces the wordier
  passage in the §2.5 baseline paragraph. Net one sentence shorter; MAP-prior sentence and
  the small/large-k_d dial kept.
