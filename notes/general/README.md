# Notes

Working notes for a manuscript to be submitted to *Biostatistics*, special issue
**Statistical Foundations of AI and Real-World Evidence Generation**
(submission deadline: 30 June 2026; guest editors: Lorin Crawford, Jean Feng,
Harald Binder).

The issue invites work on statistical methods for AI/ML developed with
real-world data, including causal inference, transportability across
populations, and the integration of experimental and real-world evidence — the
remit this project sits in.

## Folder layout

Each subfolder corresponds to one manuscript section:

- `introduction/` — motivation and positioning
- `causal/` — causal framework and estimands
- `methodology/` — model specification and error distribution
- `posterior_inference/` — inference, projection, computation
- `data_analysis/` — clinical background and applied analysis
- `general/` — shared assets: `references.bib`, `latex_template.tex`, this README

Within each section folder, the **ALL-CAPS `.tex` file** (e.g.
`INTRODUCTION.tex`, `ANALYSIS.tex`) is the drafting file that will feed the
final manuscript. The other `.tex` files in the same folder are working notes
that the draft pulls from. Section folders without an all-caps file yet have
not started drafting — notes only.

## Conventions

- All `.tex` files share the preamble in `general/latex_template.tex`.
- Bibliography is centralized in `general/references.bib`; cite via
  `\bibliography{../general/references}` with `apalike` style.
- *Biostatistics* allows format-neutral initial submission; the OUP
  "Traditional Medium – 1 Column" Overleaf template will be applied at
  manuscript-assembly time. Author guidelines:
  <https://academic.oup.com/biostatistics/pages/author-guidelines>.
