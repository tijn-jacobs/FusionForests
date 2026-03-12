# Informed Random Splitting (IRS) — Implementation Guide

This document describes the complete implementation of **Informed Random Splitting (IRS)** for handling missing covariates in the FusionForests BART framework.

---

## 1. Problem Statement

Standard BART crashes when observations have missing covariates (`NaN`). The original `StanTree::FindLeaf()` returns `nullptr` when it encounters a `NaN` at a split variable, causing segfaults in `FitTree()` and corrupted sufficient statistics in `GetAllLeafStatistics()`.

IRS solves this by making **stochastic, informed left/right routing decisions** at tree nodes where the split variable is missing for a given observation.

---

## 2. Architectural Decisions

### 2.1 Routing Indicator Storage

Routing decisions are stored in a **routing map** — one per tree in the forest:

```cpp
using RoutingMap = std::unordered_map<StanTree*, std::vector<int8_t>>;
```

Each entry maps an internal tree node to a vector of length `n` (number of training observations):

- `0` = observation is **not missing** at this node's split variable (deterministic routing)
- `1` = go **left**
- `-1` = go **right**

This lives on `StanForest` as `std::vector<RoutingMap> routing_maps` (one per tree).

### 2.2 Routing Lifecycle — Nog vs. Deeper Internal Nodes

This is the most critical design decision. The mutability of routing indicators depends on whether a node is a **nog** (both children are leaves, checked via `IsNog()`):

| Node type                                                | Routing behaviour                                                                                                                                                                                                             |
| -------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Nog node**                                             | Routing is **redrawn every iteration** as a Gibbs step, using `ComputeIRSProbability` with current residuals and sigma. This is valid because only leaf step heights depend on the routing — no further tree structure below. |
| **Deeper internal node** (has internal-node descendants) | Routing is **frozen** — stored indicators persist because downstream splits were conditional on these routing decisions.                                                                                                      |

**Transitions:**

- When a nog node gains a grandchild (birth on one of its children), it stops being a nog and its routing freezes.
- When all descendants are pruned and it becomes a nog again, its routing becomes live and gets redrawn.

**No explicit frozen/live tracking is needed** — we simply check `node->IsNog()` at routing time.

### 2.3 Test-Time Prediction

At test time (new data), there is no routing map. Missing observations are routed with **uniform random P(left) = 0.5**. This is averaged across posterior samples.

### 2.4 Informed Probability Computation

For a missing observation `i` at internal node with children `left` and `right`, the routing probability is:

```
P(go left) = p_L / (p_L + p_R)
```

where `p_L` and `p_R` are posterior predictive densities of the residual under each child's leaf distribution:

```
p_d = N(r_i | hat_h_d, hat_v_d + sigma^2)
```

with posterior mean `hat_h_d = (sum_d / sigma^2) / (n_d / sigma^2 + 1/tau_h^2)` and posterior variance `hat_v_d = 1 / (n_d / sigma^2 + 1/tau_h^2)`.

The computation uses **log-sum-exp** for numerical stability and clamps the result to `[0.01, 0.99]`.

---

## 3. File-by-File Changes

### 3.1 `src/StanTree.h`

- Added `RoutingMap` typedef
- Declared two new `FindLeaf` overloads:
  - **Uniform random** (`Random&`): for test-time prediction, flips fair coin at NaN
  - **Routing-map** (`size_t obs_index, RoutingMap&`): for training MCMC, looks up stored indicator

### 3.2 `src/StanTree.cpp`

Implemented both `FindLeaf` overloads after the original (which returns `nullptr` on NaN):

- **Uniform random variant:** On NaN, calls `random.uniform() < 0.5` to decide left/right, then recurses.
- **Routing-map variant:** On NaN, looks up `routing_map[this][obs_index]`; if `1` goes left, otherwise right.

### 3.3 `src/StanTreeFunctions.h`

Declared two new `FitTree` overloads:

- `FitTree(..., RoutingMap&)` — uses routing-map `FindLeaf` for training
- `FitTree(..., Random&)` — uses uniform-random `FindLeaf` for test-time

### 3.4 `src/StanTreeFunctions.cpp`

Implemented both `FitTree` overloads, each calling the corresponding `FindLeaf` variant for every observation.

### 3.5 `src/StanForestFunctions.h`

Added declarations for all IRS functions:

- `ComputeIRSProbability` — informed routing probability
- `DrawRoutingIndicators` — called after birth acceptance
- `RedrawNogRouting` — Gibbs step each iteration per tree
- `RemoveRoutingIndicators` — called after death acceptance
- Routing-map overloads of `GetSufficientStatistics` (birth and death variants)
- Routing-map overload of `GetAllLeafStatistics`
- Routing-map overload of `DrawAllLeafMeans`

### 3.6 `src/StanForestFunctions.cpp`

**MakeCutpoints NaN guard (Step 1):**

- Uniform-cutpoints branch: skips NaN values in min/max loop; if all values are NaN for a variable, sets its cutpoints to empty.
- Empirical-cutpoints branch: filters NaN values before sorting.

**ComputeIRSProbability (Step 3):**

- Computes posterior predictive densities for left and right children
- Uses log-sum-exp for numerical stability
- Clamps output to `[0.01, 0.99]`
- Edge case: if `n_d == 0` for a child, uses prior (`hat_h = 0`, `hat_v = tau_h^2`)

**DrawRoutingIndicators (Step 4):**
Two-pass algorithm called after a birth proposal is accepted:

1. **Pass 1:** Iterate observations in the leaf being split. Non-NaN at `split_var`: store `0`, accumulate left/right sufficient stats. NaN: mark for Pass 2.
2. **Pass 2:** For NaN observations: call `ComputeIRSProbability` with the Pass 1 stats, draw routing (store `1` or `-1`).

**RedrawNogRouting (Step 5):**
Iterates all entries in the routing map. For each node that `IsNog()`, redraws all non-zero indicators using `ComputeIRSProbability` with current residuals. This is the per-iteration Gibbs update.

**RemoveRoutingIndicators (Step 6):**
Simple `routing_map.erase(dying_node)`.

**Routing-map GetSufficientStatistics — birth variant (Step 8):**
Uses routing-map `FindLeaf` to determine if each observation falls in the target leaf. For the proposed `split_var`, NaN observations are **skipped** (they contribute to neither left nor right count). Their routing will be drawn by `DrawRoutingIndicators` after acceptance.

**Routing-map GetSufficientStatistics — death variant (Step 8):**
Uses routing-map `FindLeaf` to determine if each observation is in `left_leaf` or `right_leaf`. Stored routing indicators already exist.

**Routing-map GetAllLeafStatistics (Step 9):**
Uses routing-map `FindLeaf` to route each observation to its leaf.

**Routing-map DrawAllLeafMeans (Step 9):**
Calls routing-map `GetAllLeafStatistics`, then draws leaf means as usual.

### 3.7 `src/StanForest.h`

- Added `bool use_irs` member (default `false`)
- Added `std::vector<RoutingMap> routing_maps` member (one per tree)
- Added `void SetIRS(bool v)` setter
- Added `void Predict(size_t, size_t, double*, double*, Random&)` overload for test-time
- Added friend declaration for IRS-aware `BirthDeathStep` overload

### 3.8 `src/StanForest.cpp`

- Constructor initialises `use_irs(false)` and `routing_maps(num_trees_init)`
- `SetNumTrees` resizes `routing_maps` alongside `trees`
- **`Draw()` method:** When `use_irs` is true, for each tree j:
  1. `RedrawNogRouting(trees[j], ...)` — Gibbs-redraw routing at nog nodes
  2. `FitTree(..., routing_maps[j])` — remove tree j's contribution using routing map
  3. `BirthDeathStep(..., routing_maps[j])` — IRS-aware birth/death
  4. `DrawAllLeafMeans(..., routing_maps[j])` — IRS-aware leaf draws
  5. `FitTree(..., routing_maps[j])` — add back tree j's contribution

  When `use_irs` is false, the original code path is preserved unchanged.

- Added `Predict(p, n, x, fp, Random&)` — test-time prediction with uniform random routing

### 3.9 `src/StanBirthDeath.h`

Added declaration for IRS-aware `BirthDeathStep` overload that takes `RoutingMap&`.

### 3.10 `src/StanBirthDeath.cpp`

Added IRS-aware `BirthDeathStep` overload:

- **Birth path:** Calls routing-map `GetSufficientStatistics` (which skips NaN at proposed split_var). After acceptance, calls `DrawRoutingIndicators` to populate the routing map for the new internal node.
- **Death path:** Calls routing-map `GetSufficientStatistics`. After acceptance, calls `RemoveRoutingIndicators` to erase the dying node's entry from the routing map, then prunes.

### 3.11 `src/ForestEngine.h`

- Added `void SetIRS(bool v)` — delegates to `StanForest::SetIRS`
- Added `void Predict(size_t, size_t, double*, double*, Random&)` — delegates to `StanForest::Predict` with `Random&`
- Added `SetUpForest` overload taking `size_t num_cuts` (uniform cutpoints) instead of `int* nc`

### 3.12 `src/SimpleBART.h` / `src/SimpleBART.cpp`

- Added `irsSEXP` parameter to the function signature
- Converts to `bool irs` and calls `forest.SetIRS(true)` when enabled
- Test-time prediction uses `forest.Predict(p, n_test, X_test, testpred, random)` when IRS is on

### 3.13 `R/SimpleBART.R`

- Added `irs = FALSE` argument to the R wrapper
- Passes `irsSEXP = irs` to the C++ function

### 3.14 `R/RcppExports.R` / `src/RcppExports.cpp`

- Updated `SimpleBART_cpp` to accept 20 parameters (added `irsSEXP`)
- Updated `CallEntries` registration from 19 to 20 args

### 3.15 `src/Prerequisites.h`

- `#include <unordered_map>` was already present (needed for `RoutingMap`)

---

## 4. MCMC Lifecycle Summary

Each MCMC iteration, for each tree `j`:

```
1. RedrawNogRouting(tree_j, routing_maps[j])
   └── For each node in routing_map: if IsNog(), redraw non-zero indicators
       using ComputeIRSProbability with current residuals

2. FitTree(tree_j, routing_maps[j])  →  subtract from all_fit  →  compute residuals

3. BirthDeathStep(tree_j, routing_maps[j])
   ├── Birth path:
   │   ├── GetSufficientStatistics(routing_map) — skips NaN at split_var
   │   ├── MH accept/reject
   │   └── If accepted: DrawRoutingIndicators for new internal node
   └── Death path:
       ├── GetSufficientStatistics(routing_map)
       ├── MH accept/reject
       └── If accepted: RemoveRoutingIndicators for dying node

4. DrawAllLeafMeans(tree_j, routing_maps[j])
   └── GetAllLeafStatistics(routing_map) → draw leaf means

5. FitTree(tree_j, routing_maps[j])  →  add back to all_fit
```

---

## 5. Key Design Rationale

### Why store routing indicators instead of flipping coins each traversal?

If we flipped a fresh coin every time we traversed the tree, the same observation could end up in different leaves across different calls within the same iteration. This would make sufficient statistics inconsistent — the birth/death MH ratio would be computed with one routing, but the leaf means would be drawn with another.

### Why redraw at nog nodes but freeze at deeper internals?

When a node is a nog, only the leaf step heights below it depend on the routing. These are redrawn every iteration anyway (as conjugate Gibbs updates). So the routing indicators can be updated as part of the same Gibbs sweep.

But when a node has internal-node descendants, the tree splits below were proposed and accepted conditional on the routing at this node. Changing the routing would invalidate the basis for those downstream splits. So routing must be frozen.

### Why skip NaN observations in the birth MH ratio?

During a birth proposal, we don't yet know how to route NaN observations at the proposed split variable. Rather than making a temporary routing decision that biases the MH ratio, we simply exclude these observations from the sufficient statistics. The NaN observations get their routing indicators drawn by `DrawRoutingIndicators` after the birth is accepted.

This is interesting. NOTE HERE. Look into this myself.

### Why uniform random (P=0.5) at test time?

At test time we don't have the tree-specific residuals needed for informed routing. Uniform random routing, averaged across many posterior samples, gives an unbiased estimate. This is analogous to how other missing-data BART methods handle prediction.

---

## 6. Verification Plan

1. **Build**: `R CMD INSTALL .` — no compilation errors
2. **No-missingness regression**: Run with `irs=TRUE` on complete data — should give identical results (no routing map entries created)
3. **Crash fix**: Data with NaN values, `irs=TRUE` — no segfault
4. **Simple DGP**: `Y = X1 + eps`, `X2` noise with missingness — model recovers `f(X1)`
5. **Informed vs. uniform**: `Y = X1 + X2 + eps`, mask `X2` for half — informed IRS should outperform uniform
