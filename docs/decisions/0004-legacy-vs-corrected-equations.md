# 0004 — Legacy-compatible vs corrected model equations

**Status:** Accepted (2026-07-18) · **Relates to:** plan §7 Phase 3; baseline §3.3.1, §3.4,
§3.7.2, §4.4, §2.6, App A; inventory `b26-corr-*` rows.

## Context

The 2026 baseline documents several mathematical issues in the 2016 formulation. Some are outright
corrections; one (the inference update) is genuinely ambiguous because the original code is lost.
We must decide, per issue, whether to implement a legacy-compatible variant, a corrected variant,
or both.

## Decision

- **Group inference update (ambiguous):** implement **both** named variants and compare —
  `legacy_adaptive_threshold` (historical-compatible, reproduces the 2016 signed/soft-threshold
  behavior as closely as the spec allows) and `proximal_group` / a clearly derived smooth-gradient
  method (the modern model). Neither is presented as "the" original without the label. (§3.3.1)
- **Corrected outright (modern edition uses corrected form; note the correction):**
  - Group activity uses `z_k = sqrt(ε + Σ Λ_ki s_i²)` with documented ε.
  - Group-penalty derivative is treated as a **signed gradient**, not a nonnegative adaptive soft
    threshold.
  - ML group-weight gradient keeps the **1/2 factor** from differentiating the square root (App A);
    absorbable only into a proportional learning rate.
  - **Gabor envelope** uses the negative quadratic exponent (`eq:corrected-gabor`).
  - Recurrent update is first-order in **Δt/τ**, not absolute `t`; noise-scaling convention with
    Δt/τ defined explicitly in Phase 3.
  - **No "energy landscape"** language for the antisymmetric-containing dynamics; use fixed-point /
    basin geometry; robustness measured, never inferred from weight magnitude.
- **Λ parameterization:** nonnegative (e.g. softplus); resolve scale non-identifiability by row/
  column normalization or explicit regularizer; record the chosen convention.
- All analytic gradients verified against finite differences before training (plan Gate P3).

## Alternatives considered

- Only implement the corrected model — rejected: loses the ability to check compatibility with the
  historical results and to attribute differences to the correction vs the data change.
- Only reproduce the legacy behavior — rejected: propagates known math errors into the new edition.

## Consequences

"Legacy vs corrected inference" becomes a required ablation (plan §6E / §7 Phase 6E). Evidence
status in the inventory distinguishes faithful-reimplementation (legacy settings) from modernized
(corrected) runs.
