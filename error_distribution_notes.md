# Error-distribution notes: AFTrees DP walkthrough + design sketch for FusionForests

## 1. Purpose

These notes document how the reference package `inst/AFTrees-master/` (Henderson, Louis, Rosner, Varadhan, _Biostatistics_ 2020) implements its single-source **centered Dirichlet process** (CDP) mixture on the residual distribution of an AFT-BART model, and sketch a forward-looking three-way switch for FusionForests:

1. `error_dist = "gaussian"` — current behaviour (single normal residual).
2. `error_dist = "shared_dp"` — one DP pooled across data sources, AFTrees-style.
3. `error_dist = "source_dp"` — per-source DPs, eventually linked by an HDP as described in [`notes/error_distribution.tex`](../notes/error_distribution.tex).

These are notes for _future_ implementation; nothing in the FusionForests sampler changes yet.

---

## 2. AFTrees DP walkthrough

### 2.1 State ownership

| State                                  | Owner    | Field           |
| -------------------------------------- | -------- | --------------- |
| Cluster labels $Z_i$                   | `Mixdev` | `labels[]`      |
| Stick weights $\pi_h$                  | `Mixdev` | `mix_prop[]`    |
| Atoms $\theta_h$                       | `Mixdev` | `locations[]`   |
| Concentration $\alpha$                 | `Mixdev` | `mass[0]`       |
| Error scale $\sigma$ (DP branch)       | `Mixdev` | `sig[0]`        |
| Error scale $\sigma$ (Gaussian branch) | `Sdev`   | `s`             |
| BART leaf-mean updates                 | `MuS`    | unchanged by DP |

`Mixdev` is defined entirely in [`inst/AFTrees-master/src/Mixdev.h`](AFTrees-master/src/Mixdev.h) — it is a small header-only class, 150 lines.

### 2.2 Outer Gibbs sweep

[`inst/AFTrees-master/src/mbart.cpp:329-428`](AFTrees-master/src/mbart.cpp) is the MCMC loop. One sweep:

1. **Tree backfitting** (lines 334–362). For each tree $i$, the working response is
   $$\texttt{YDat1} = Y - \texttt{indiv\_locations} - \sum_j \texttt{fit}_j + \texttt{fit}_i.$$
   The DP shift is subtracted only when DP is active (the `if(non_par)` guard at line 341). The tree MH step is `Metrop()`; leaf draws are inside `Node::currentFits`.
2. **Post-backfit residuals** (lines 366–369): `eps = Y − mtotalfit`. These are the inputs to all DP updates.
3. **DP block** (lines 376–393):
   ```cpp
   mix.updateLabels   (labs, mixvals, locations, eps, sig, NumObs, nclust);
   mix.tabCounts      (cluster_counts, labs, NumObs, nclust);
   mix.updateMix      (mixvals, mass, labs, cluster_counts, NumObs, nclust);
   mix.updateLocations(locations, mixvals, eps, labs, cluster_counts, sig,
                       sigtau_sq, NumObs, nclust);
   mix.getIndivLocations(indiv_locations, locations, labs, NumObs, nclust);
   mix.updateSigma    (labs, locations, eps, sig, kappa, sigdf, NumObs, nclust);
   if (num_censored > 0)
       truncNormImpute(Y, Yobserved, delta, mtotalfit, indiv_locations,
                       sig, NumObs);
   ```
4. **Parametric branch** (lines 394–403): `Sdev::drawPost()` + the no-DP variant `truncNormImpute_SP` (omits the cluster shift). This is the `npind=0` path.

### 2.3 Individual updates

- **`Mixdev::updateLabels`** ([`Mixdev.h:15-39`](AFTrees-master/src/Mixdev.h)): categorical, weights $\propto \pi_h\,\varphi((\varepsilon_i - \theta_h)/\sigma)$.
- **`Mixdev::updateMix`** ([`Mixdev.h:48-82`](AFTrees-master/src/Mixdev.h)): finite truncated stick-breaking (Ishwaran–James), `V_h ~ Beta(n_h + 1, α + Σ_{j>h} n_j)`. Concentration updated by Gamma conjugacy with fixed hyperprior `Gamma(psi1=2.0, psi2=0.1)` set in the `Mixdev` constructor (line 12).
- **`Mixdev::updateLocations`** ([`Mixdev.h:83-112`](AFTrees-master/src/Mixdev.h)): conjugate Gaussian per atom (prior $N(0, \kappa^2)$, likelihood from cluster members), **followed by a deterministic post-hoc recentre**
  $$\theta_h \leftarrow \theta_h - \sum_k \pi_k \theta_k.$$
  This is the Yang et al. (2010) centred-DP enforcement — implemented as global renormalisation, not parameter expansion. Compare with the per-source recentre $\mu_s = \sum_k \pi_{sk}\theta_k^*$ proposed in [`notes/error_distribution.tex`](../notes/error_distribution.tex) §3.3.
- **`Mixdev::updateSigma`** ([`Mixdev.h:128-145`](AFTrees-master/src/Mixdev.h)): inverse-Gamma conjugate, sum-of-squares about the current per-observation atom.
- **`truncNormImpute`** ([`AFTrees-master/src/truncNormImpute.cpp:9-27`](AFTrees-master/src/truncNormImpute.cpp)): truncated-normal draw with mean `mtotalfit[i] + indiv_locations[i]` — the cluster shift is part of the imputation mean. The `_SP` variant simply drops that term.

### 2.4 R-level glue

- [`inst/AFTrees-master/R/aftrees.R`](AFTrees-master/R/aftrees.R) is the main entry. The DP toggle is the `npind` argument passed into `.C("mbart", ...)` — `1` enables DP, `0` reverts to a single Gaussian.
- [`inst/AFTrees-master/R/FindKappa.R:29-51`](AFTrees-master/R/FindKappa.R) calibrates $\kappa$ (the atom-prior scale) from a preliminary log-normal `survreg` fit, using a lookup table for common quantiles plus numerical integration as a fallback.
- Posterior returns include `mix.prop`, `locations`, `mass` — each a `K × ndpost` (or length-`ndpost`) draw matrix.

### 2.5 Variants — three patterns already in AFTrees

- **`IndivAFT`**: one pooled BART+DP fit; treatment enters as a covariate column.
- **`IndivAFTNew`**: two `.C("mbart")` calls, one per treatment arm, each with its own `FindKappa`. This is the practical blueprint for option (3) in its independent-DPs flavour.
- **`IndivAFTSeparate`**: thin wrapper that calls `AFTrees()` twice on the two arms — syntactic sugar for the same idea.

The "Separate" label refers to data stratification, not DP structure. None of the three variants implement a hierarchical link between the two arms' residual distributions; that is novel work for us.

### 2.6 Single-source assumption

A search for `source`, `group`, `strata`, `S` across `inst/AFTrees-master/{src,R}/` returns nothing. The residual mixture is genuinely single-source: one `labels[]`, one `locations[]`, one `mix_prop[]`, one `sig[0]`.

---

## 3. Mapping to FusionForests

Where our existing code lines up with the AFTrees hooks (so the eventual extension knows what it touches):

| AFTrees hook                                       | FusionForests analogue                                                                                                           |
| -------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| Outer Gibbs loop, `mbart.cpp:329-428`              | [`src/OuterGibbsFunctions.cpp`](../src/OuterGibbsFunctions.cpp)                                                                  |
| Backfit residual construction, `mbart.cpp:339-346` | Per-forest residual loops inside `OuterGibbsFunctions.cpp` (the analogous BLAS-style subtractions feeding `StanForestFunctions`) |
| `Sdev::drawPost` / `Mixdev::updateSigma`           | `UpdateSigma` in `OuterGibbsFunctions.cpp:123`                                                                                   |
| `truncNormImpute` / `_SP`                          | `AugmentCensoredObservations` overloads in `OuterGibbsFunctions.cpp:3,41`                                                        |
| R entry `aftrees.R` + `FindKappa.R`                | [`R/FusionForest.R`](../R/FusionForest.R), helpers in `R/helpers.R`                                                              |

There is currently no `Mixdev`-equivalent class in `src/` and no DP-related state anywhere in the sampler.

---

## 4. Design sketch — three options

Selected at the R level via `error_dist = c("gaussian", "shared_dp", "source_dp")`; dispatched in C++ via a `mixture_mode` enum, mirroring AFTrees's `npind` flag pattern.

### 4.1 `gaussian` (default, current behaviour)

Equivalent to AFTrees's `npind = 0`. Backfit residual omits any per-observation cluster shift; `UpdateSigma` runs as today; censored augmentation uses the existing mean (no cluster offset).

### 4.2 `shared_dp` (pooled DP across sources)

Direct port of AFTrees:

- One `Mixdev`-equivalent state across all observations regardless of `S` — `labels[]`, `mix_prop[]`, `locations[]`, `mass[0]`, `sig[0]`.
- Atom recentre uses the global mixture, as in `Mixdev::updateLocations`.
- R side: one `FindKappa` call on pooled residuals from a preliminary parametric fit; one `kappa` passed in.
- Returns: one set of posterior matrices `mix.prop / locations / mass`.

This is the cheapest extension — fits naturally into our outer Gibbs and unblocks DP-style flexibility without introducing any source-aware state.

### 4.3 `source_dp` (per-source DPs)

Two stages, in increasing complexity:

**Stage A — independent per-source DPs (`IndivAFTSeparate` blueprint).**
Maintain `Mixdev` state per source: `labels_s[]`, `mix_prop_s[]`, `locations_s[]`, `mass_s[0]` for $s\in\{0,1\}$. No shared atoms. Independent `FindKappa` per source. Cheap to implement and gives an empirical floor on how much per-source flexibility actually matters before committing to the full HDP machinery.

**Stage B — full HDP-CDP** (the [`notes/error_distribution.tex`](../notes/error_distribution.tex) model).
Shared atoms $\theta_k^*$ across sources, per-source weights $\pi_{sk}$. Adds:

- Top-level stick-breaking weights $\beta$ updated via the table-count augmentation $\{t_{sk}\}$ (LaTeX note §Step 5a–5b).
- Per-source recentring: replace the global `muG` recentre in `Mixdev::updateLocations` with a per-source loop $\mu_s = \sum_k \pi_{sk}\theta_k^*$, so each source's centred atoms are $\theta_k^* - \mu_s$.
- A two-level concentration prior on $(\gamma, M_0, M_1)$, each updated via Escobar–West auxiliary variables (LaTeX note §Step 7).

Stage B reuses Stage A's per-source state — Stage A is a working prefix, not a throwaway.

---

## 5. Open design questions (deferred)

- **Truncation level K.** AFTrees uses $K = 200$ in `aftrees.R` but $K = 50$ in `IndivAFTNew.R`. Should we expose it as an argument or fix it? Either is fine; the LaTeX note defaults to 50. Yes, expose as argument. Then we can fine-tune later.
- **Source-specific $\sigma_s$ vs shared $\sigma$.** [`notes/error_distribution.tex`](../notes/error_distribution.tex) §Extensions notes that splitting $\sigma$ by source is a one-line change. Cheap; doubles state. Defer to empirical evidence. Leave this as one for now. We can extend it later.
- **DP-aware censoring in `shared_dp`.** AFTrees splits `truncNormImpute` from `truncNormImpute_SP` — the cluster shift is part of the imputation mean. Our `AugmentCensoredObservations` would need an analogous "with cluster shift" overload as soon as any DP option is active.
- **Posterior return shape.** Where do `mix.prop / locations / mass` (or per-source versions) live in the `FusionForest()` return list? Need to decide before exposing them at the R level. Easiest: a nested `error_dist` sublist alongside `train_predictions_*`. What does AFTrees do?

---

## 6. References

- LaTeX research note (the model we're aiming toward): [`notes/error_distribution.tex`](../notes/error_distribution.tex)
- AFTrees C++ DP code:
  - [`inst/AFTrees-master/src/mbart.cpp`](AFTrees-master/src/mbart.cpp) — outer loop, lines 329–428
  - [`inst/AFTrees-master/src/Mixdev.h`](AFTrees-master/src/Mixdev.h) — full DP state + updates
  - [`inst/AFTrees-master/src/truncNormImpute.cpp`](AFTrees-master/src/truncNormImpute.cpp) — DP-aware censoring
- AFTrees R-level glue:
  - [`inst/AFTrees-master/R/aftrees.R`](AFTrees-master/R/aftrees.R)
  - [`inst/AFTrees-master/R/FindKappa.R`](AFTrees-master/R/FindKappa.R)
  - [`inst/AFTrees-master/R/IndivAFTNew.R`](AFTrees-master/R/IndivAFTNew.R)
- FusionForests integration points:
  - [`src/OuterGibbsFunctions.cpp`](../src/OuterGibbsFunctions.cpp)
  - [`src/FusionForest.cpp`](../src/FusionForest.cpp)
  - [`src/Prerequisites.h`](../src/Prerequisites.h)
  - [`R/FusionForest.R`](../R/FusionForest.R)
