# Supervisor feedback — round 2, manuscript of 29 July 2026

_Convention: round-2 changes are marked in the draft in purple (\RevisionColourTwo);
orange (\RevisionColour) remains the round-1 marking; bright pink (\RevisionColourThree)
marks the extended machine-learning comparison. ✅ notes track resolutions as in the
round-1 file._

## General remarks

- The introduction is marked up with two labels, **(x)** for the paragraph _"Our solution
  addresses the three challenges in a single Bayesian model. …"_ (p. 2) and **(xx)** for
  the paragraph _"We develop the Bayesian fusion forest, a nonparametric framework for
  heterogeneous treatment effects on survival outcomes. …"_ (p. 3), with the comment:
  _"(x) en (xx) zijn wat dubbelop. Ik zou ze grotendeels combineren op de plek van (x) en
  dan wat specifieke details (zoals de AFT decompositie) verplaatsen naar de paragrafen
  waarin je meer uitweidt over de uitdagingen"_
  **✅ Resolved (adopted 30 July):** the restructure is now the introduction; the old
  version is deleted. (x) rewritten with explicit for-the-first/second/third challenge
  labels and the dictated transition; (xx) deleted, its content distributed:
  decomposition + estimand → the challenge-1 paragraph (with the dictated "Here, we
  extend…" sentence), priors/borrowing → the challenge-2 paragraph, error model and BART
  properties → the BART paragraph.
- The three challenges are tracked through the introduction with the markers **chall. 1**,
  **chall. 2** and **chall. 3**, and with **impliciet** / **expliciet** noted in the
  margins of p. 2.

---

## Abstract and title page (p. 1)

- At _"The framework relaxes the unconfoundedness assumption on the real-world data. In
  return, we assume that the treatment effect transports across the two sources."_ — "In
  return" circled, with: _"Ik twijfel of dit goed Engels is, misschien wel hoor! Maar
  anders werkt: 'By assuming the tx effect …, we can relax …'"_
  **✅ Resolved:** merged into one sentence keeping benefit-first order and avoiding the
  gerund opening of the suggested variant: "The framework relaxes the unconfoundedness
  assumption on the real-world data by assuming instead that the treatment effect
  transports across the two sources."
- At _"The model accommodates right- and interval-censored data."_ — bracketed, with:
  _"deze zin eerder, en ik vind dat dit juist nieuw en bijzonder is: 'Our method opens up
  interval-censored & right-censored data to data fusion' (oid)"_ — preceded by a
  struck-through _"we propose"_.
  **✅ Resolved:** sentence moved up to third position (after the transportability
  sentence) and recast per the suggestion: "Our method opens up right- and
  interval-censored outcomes to data fusion." ("outcomes" avoids "data … to data
  fusion".) The old feature-style sentence is deleted.
- At _"Unmeasured confounding can bias treatment effect estimates from observational data.
  We combine the two data sources: the trial identifies the treatment effect and the
  real-world data contribute power and longer follow-up."_ — vertical lines in the right
  margin and a caret, with a short note reading _"aim vs. strengths of the …"_ (partly
  illegible).
  **✅ Resolved:** read as "state the aim, not just the strengths". Now: "But unmeasured
  confounding can bias treatment effect estimates from observational data. We aim to
  combine the strengths of the two data sources: the trial identifies the treatment
  effect and the real-world data contribute power and longer follow-up." ("But" added to
  mark the pivot from RWD praise to its weakness.) Applied in both intro versions.
- Verbal (29 July): the abstract's decomposition names three components but a "deviation
  term" appears three sentences later — should the deviation be named in the
  decomposition?
  **✅ Resolved:** the decomposition sentence now names all four components ("a shared
  baseline prognosis, a source-specific deviation, a treatment effect, and a confounding
  function"), which also makes "Each component receives a Bayesian tree ensemble prior"
  exactly right (four forests). The later sentence now refers back ("The shared baseline
  prognosis … the deviation …") instead of introducing the deviation anew.

---

## 1. Introduction

### Paragraph "Our solution addresses the three challenges" — label (x) (p. 2)

- At _"Our solution addresses the three challenges in a single Bayesian model."_ —
  "Our solution" replaced by _"The Bayesian fusion forest"_.
  **✅ Resolved (in PROPOSED):** opens "The Bayesian fusion forest addresses the three
  challenges in a single nonparametric model."
- After the same sentence, with a caret before _"A confounding function absorbs the
  unmeasured confounding in the real-world data."_: _"for the first challenge,"_
  **✅ Resolved (in PROPOSED):** all three solution sentences carry explicit labels ("For
  the first/second/third challenge, …"), with censoring made the explicit third.
- Check marks and numbers placed on the three sentences: (1) _"A confounding function
  absorbs the unmeasured confounding in the real-world data."_, (2, marked "2nd") _"A
  shared baseline prognosis with a source-specific deviation captures the heterogeneity
  between the sources."_, (3) _"A source-specific error distribution captures heterogeneity
  beyond the mean."_
  **✅ Resolved (in PROPOSED):** sentences (2) and (3) sit together under the
  second-challenge label; the error sentence covers heterogeneity beyond the mean.
- After _"Existing methods meet some of these challenges but none meets all three."_:
  _"We will now provide more context on the challenges and solutions,"_
  **✅ Resolved (in PROPOSED):** added as the closing sentence of the paragraph.

### Paragraph "A confounding function addresses the first challenge" (p. 2)

- After _"Mao et al. (2025) target the conditional restricted mean survival time under right
  censoring. Neither accommodates interval censoring, the third challenge."_: _"Here, we
  extend the confounding function framework to an AFT model with interval censoring so that
  challenge one and three are jointly addressed."_
  **✅ Resolved (in PROPOSED):** dictated sentence added (Proposition refs later removed at
  TJ's request), followed by the estimand sentence moved from (xx).
- In the margins alongside this paragraph, the markers _"impliciet"_ and _"expliciet"_ with
  the challenge numbers 1, 2 and 3.
  **✅ Resolved (in PROPOSED):** challenge references made explicit — the paragraph opens
  "For the first challenge, …" and closes on "…the third challenge" and "challenges one
  and three".

### Bottom of p. 2

- In the bottom-left margin, with an arrow: _"te vroeg hier: HDPM, onzekerheid?"_
  (reading uncertain)
  **✅ Resolved (in PROPOSED, read as "too early for HDPM/uncertainty here"):** the HDPM
  and uncertainty material appears only in the BART paragraph at the end of the
  introduction, not in the earlier solution or challenge paragraphs.

### Paragraph "We develop the Bayesian fusion forest" — label (xx) (p. 3)

- In the top margin, as a suggested opening: _"The method we develop to address the three
  challenges, is a NP framework …"_, with _"(Bayesian ff, the)"_ inserted.
  **✅ Moot (in PROPOSED):** the (xx) paragraph is deleted per the general remark; its
  opening role is taken by the (x) paragraph's "The Bayesian fusion forest addresses the
  three challenges in a single nonparametric model."
- At _"Machine-learning predictors such as (deep) neural networks offer comparable
  flexibility, and we benchmark our framework against them in the simulation study and the
  application."_: _"dit mag meer aandacht krijgen in de paper"_
  **✅ Resolved (pink):** the comparison is extended to four method families (DNN,
  XGBoost-AFT, BJ-ELM, BJ-trees). The main text names them with references, reports the
  full four-method table (Table 1) with the coverage/width analysis, and states the deep
  network is the strongest alternative (lowest error in three of four configurations,
  shown in the figure); SM sm:sim-competitors keeps the method descriptions and details.
- At _"First, we formalise an accelerated failure time decomposition whose components carry
  independent causal meanings and show that the trial is necessary to separate the treatment
  effect from the confounding bias (Propositions 1 and 2)."_ — marked _"chall. 3"_.
- At _"A confounding function absorbs the bias from unmeasured confounding."_ — marked
  _"chall. 1"_.
- At _"Second, we design a separate tree-ensemble prior for each component of the
  decomposition. The priors encode borrowing between the sources and shrink the confounding
  bias towards a constant shift."_ — marked _"chall. 2"_.
  **✅ Resolved (in PROPOSED):** the three (xx) claims are redistributed along the
  challenge markers — AFT/censoring and the confounding function into the challenge-1/3
  paragraph, the borrowing priors into the challenge-2 paragraph.
- At _"Third, we model the error distribution nonparametrically and let it differ between
  the sources."_: _"hier mag je ook meer over zeggen, puttend uit je introductie op 2.3"_
  **✅ Resolved (in PROPOSED):** the BART paragraph gains a §2.3-derived why-sentence ("The
  error distribution deserves the same flexibility: it plausibly differs between the
  sources yet not entirely.") plus the HDPM sentence (differ in shape and scale, share
  mixture components).

---

## 2. Methodology

### 2.1 Causal framework (p. 4)

- After _"We therefore allow for confounding in the real-world data at the explicit cost of
  retaining Assumption 4."_ — a struck-through phrase, followed by: _"and we will make the
  interplay explicit in props 1 & 2 in the next section"_
  **✅ Resolved:** added as its own sentence: "We make this interplay explicit in
  Propositions 1 and 2 in the next section."

### 2.2 Accelerated failure time decomposition (p. 5)

- At _"A baseline prognosis and the causal effect enter in both sources, while the
  confounding bias enters only in the real-world data."_ — "Intuitively," inserted at the
  start of the sentence, and appended at the end, replacing a struck-through phrase: _"In
  proposition 1, we will make this intuition precise, and in prop. 2 we show how the
  decomposition translates to identification results."_
  **✅ Resolved:** "Intuitively," prefixed, and the dictated text appended as two short
  sentences: "Proposition 1 makes this intuition precise. Proposition 2 shows how the
  decomposition translates to identification results."
- In the right margin at _"The censoring assumption enters the likelihood and the data
  augmentation in the sampler, not the causal identification argument."_: _"This will enable
  us to handle interval censoring in addition to right-censoring in the sequel!"_
  **✅ Resolved:** appended: "This separation enables us to handle interval censoring in
  addition to right censoring in the sequel." ("This separation" names the antecedent;
  exclamation dropped.)

### 2.3 Nonparametric error distribution (p. 6)

- In the right margin at _"A model that treats them as fully independent forfeits useful
  pooling. The loss is greatest when one source is small."_: _"mooi!"_

### 2.4 Overview of BART (p. 7)

- In the top-right margin at _"Three properties motivate this choice. BART captures
  nonlinear and interaction effects without requiring us to specify them."_: _"ook dit mag
  al in de introductie aan bod komen"_
  **✅ Resolved (in the PROPOSED intro):** its BART paragraph now carries properties 1 and
  2 (nonlinearities/interactions without specification; interpretable per-component
  regularisation). Property 3 (conjugate Gibbs updates) stays in §2.4 as too technical for
  the intro.

---

## 3. Simulation study (p. 10)

- In the top margin, as a suggested rewrite of the opening: _"There are currently very few
  methods to reasonably compare the BFF to. Existing methods share three structural
  limitations. The first limitation is …. The second [limitation] is …. The third
  [limitation] is …"_
  **✅ Resolved:** the machine-learning paragraph now opens with the dictated skeleton:
  scarcity sentence, "The first/second/third limitation is…" enumeration (prediction not
  causal; no fusion mechanism; no interval censoring).
- Continuing in the same margin: _"and we needed to make several adaptations to make them
  suitable for comparison with the BFF."_
  **✅ Resolved:** "We make several adaptations to render the methods suitable for
  comparison with the fusion forest." precedes the adaptation sentences (S/T-learners;
  trial-only and naive pool). The right-censoring workaround stays in the setup
  (experiment 2 item); the "Each adaptation answers one limitation" closer was cut on
  29 July.
- At _"We assess its robustness to unmeasured confounding and to between-source
  heterogeneity, benchmark it against flexible machine-learning methods, and study its
  behaviour as the covariate dimension grows."_ — "flexible machine-learning methods"
  underlined, with a short illegible note and a downward arrow.
  **✅ Resolved (read as "connect to the paragraph below"):** the sentence now sits at the
  end of ¶1 directly above the machine-learning paragraph that unpacks it.
- In the left margin at _"The second class consists of flexible machine-learning methods for
  survival prediction."_: _"nieuwe paragraaf"_
  **✅ Resolved:** the two-classes framing is dissolved — the single-source baseline moved
  into ¶1 and the machine-learning methods now have their own paragraph.
- In the right margin alongside the same paragraph: _"Er ook search for data fusion methods
  for right censored and interval censored data,"_
  **✅ Resolved:** added: "In particular, we found no data fusion method for right- and
  interval-censored outcomes."
