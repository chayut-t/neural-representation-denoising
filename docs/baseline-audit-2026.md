# 2026 Baseline Audit (Regression Oracle)

**Purpose (Plan §7 Phase 1, task 3).** A machine-checkable extract of the frozen 2026
baseline PDF — title, structure, figure/table list, citations, and visible numerical claims.
This is the *regression oracle* for migration into the new `dissertation/` edition: after any
migration or rebuild, the working edition must still contain this structure and these claims
(unless a decision record and the claim-evidence matrix explicitly change them). Source
compilation alone does not prove equivalence to the reviewed PDF.

- **Baseline PDF:** `legacy/rewrite-2026/source/thesis.pdf`
- **PDF SHA-256:** `c89423fa6470997760c274295444711532cb67e7b7eef824861e8c6eb16fed8a`
- **Pages:** 58 · **Content source commit:** `0c7617fcc28e6a3778fc44f5a7a59b3d4aa571e5`
- **Archival tag:** `archive/rewrite-2026-baseline`
- Page numbers below are **physical PDF pages** (1-based), extracted with `pdftotext -layout`.

## Title and front matter

- **Title:** *Denoising Objectives for Learning and Stabilizing Neural Representations*
- **Subtitle:** *Group Structure in Sparse Codes and Noise-Robust Spatial Attractors*
- **Author:** Chayut Thanapirom · **Degree:** A dissertation in Physics · **Date:** July 2026
- **Committee:** Michael R. DeWeese (Chair), Bruno A. Olshausen, Hernan G. Garcia
- **Copyright:** © 2026 Chayut Thanapirom
- Front matter: abstract, dedication, TOC, list of figures, list of tables, Acknowledgments,
  and a **"Scope and reproducibility"** chapter (states training code/checkpoints/seeds
  unavailable; numerical comparisons are descriptive, not replications).

## Chapter and section outline (with PDF page)

| Page | Number | Title |
|---|---|---|
| 12 | Chapter 1 | Questions, Thesis, and Contributions |
| 12 | 1.1 | The problem: learning structure from restoration |
| 12 | 1.2 | Representations and the level of explanation |
| 13 | 1.3 | Study I: dependencies in sparse image codes |
| 13 | 1.4 | Study II: restoration and transport in a spatial attractor |
| 14 | 1.5 | Contributions |
| 15 | 1.6 | Evidence and scope |
| 15 | 1.7 | Roadmap |
| 16 | Chapter 2 | Primer: Denoising, Structured Codes, and Recurrent Dynamics |
| 16 | 2.1 | Purpose and notation |
| 16 | 2.2 | Denoising as conditional estimation |
| 17 | 2.3 | Sparse coding primer |
| 18 | 2.3.1 | Iterative inference |
| 18 | 2.4 | Structured and group sparsity |
| 19 | 2.5 | Finite inference as a differentiable computation |
| 19 | 2.6 | Recurrent dynamics, attractors, and transport |
| 20 | 2.7 | The shared framework and its limits |
| 21 | Chapter 3 | Learning Group Structure in Sparse Image Codes |
| 21 | 3.1 | Question and evidence status |
| 21 | 3.2 | Related structured sparse models |
| 22 | 3.3 | Model |
| 22 | 3.3.1 | Mathematical clarification of inference |
| 23 | 3.4 | Why maximum likelihood is expensive |
| 23 | 3.5 | Learning group weights with denoising |
| 24 | 3.6 | Toy geometry and recovery |
| 25 | 3.7 | Natural-image experiment |
| 25 | 3.7.1 | Denoising comparison |
| 26 | 3.7.2 | Dictionaries and learned groups |
| 29 | 3.8 | Minimum-wiring embedding |
| 29 | 3.9 | What the study supports |
| 32 | Chapter 4 | Noise-Robust Location Codes in Recurrent Networks |
| 32 | 4.1 | Question and evidence status |
| 32 | 4.2 | Biological observations and computational hypotheses |
| 33 | 4.3 | The coding tradeoff: precision and ambiguity |
| 33 | 4.4 | Recurrent model |
| 34 | 4.4.1 | Write, hold, read, and move |
| 34 | 4.5 | Training by restoration through time |
| 35 | 4.6 | Prescribed location codes |
| 35 | 4.6.1 | Localized place code |
| 35 | 4.6.2 | Ring code and mixed place--periodic code |
| 36 | 4.7 | Results (4.7.1 learned structure, 4.7.2 write/hold/move, 4.7.3 random noise) |
| 41 | 4.7.4 | Fixed connection perturbation |
| 41 | 4.7.5 | Long-run fixed-point map |
| 42 | 4.8 | How to interpret periodic error correction |
| 42 | 4.9 | Multiple modules and two dimensions |
| 43 | 4.10 | What the study supports |
| 44 | Chapter 5 | Synthesis, Limitations, and a Testable Research Program |
| 44 | 5.1 | What does restoration learn? |
| 44 | 5.2 | Restoration toward a set and transport along it |
| 45 | 5.3 | What is common and what is not |
| 46 | 5.4 | Related work and interpretation |
| 46 | 5.5 | Limitations that change the claims (5.5.1--5.5.4) |
| 47 | 5.6 | A staged research program (Stages 1--5; 5.6.1--5.6.5) |
| 48 | 5.7 | Conclusion |

Appendices (after Chapter 5): **A** Maximum-Likelihood Gradient for Group Weights;
**B** Image Data and Training Details (B.1 identifiers/preprocessing, B.2 group-model training
record); **C** Attractor Training Details; **D** Symbols and Evidence Scope (glossary, evidence
labels, evidence boundary). Backmatter: bibliography (`plainnat`).

## Figures (label, PDF page, LaTeX label)

| # | Page | LaTeX label | Short title |
|---|---|---|---|
| 3.1 | 24 | `fig:group-isocost` | Group-penalty iso-cost surfaces (ℓ2 / plane+axis / factorial) |
| 3.2 | 25 | `fig:group-toy` | Toy data and learned contour (plane vs axis) |
| 3.3 | 26 | `fig:group-denoising-examples` | Denoising examples at 3 dB input SNR |
| 3.4 | 27 | `fig:group-bases` | Learned dictionaries (diag / 3x3 / learned / DAE) |
| 3.5 | 27 | `fig:group-examples` | 32 selected learned groups |
| 3.6 | 28 | `fig:group-all` | All-group visualization (Gabor-fit marks) |
| 3.7 | 30 | `fig:group-wiring-map` | 2D minimum-wiring embedding |
| 3.8 | 30 | `fig:group-wiring-dimension` | Wiring cost vs embedding dimension |
| 4.1 | 36 | `fig:grid-schematic` | Grid-cell firing-field schematic (source-authored) |
| 4.2 | 37 | `fig:attractor-weights` | Learned recurrent weights (S, A; place vs mixed) |
| 4.3 | 38 | `fig:attractor-write` | Write operation |
| 4.4 | 39 | `fig:attractor-hold` | Hold operation |
| 4.5 | 40 | `fig:attractor-move` | Move operation |
| 4.6 | 40 | `fig:attractor-random-noise` | Error under random state/weight noise |
| 4.7 | 41 | `fig:attractor-fixed-perturbation` | Error under a fixed weight perturbation |
| 4.8 | 42 | `fig:attractor-fixed-points` | Long-run input--output staircase map |

**16 figures.** Image files under `source/images/` and `source/images/kz/`. Fig 4.1
(`kz/example_grid.png`) is a source-authored/illustrative schematic, not an experiment output.

## Tables (label, PDF page, LaTeX label)

| # | Page | LaTeX label | Title | Kind |
|---|---|---|---|---|
| 2.1 | 16 | `tab:notation-core` | Core notation | source-authored (notation) |
| 2.2 | 20 | `tab:framework-comparison` | Corruption--restoration template | source-authored (conceptual) |
| 3.1 | 26 | `tab:denoising-results` | Patch denoising SNR (dB) | empirical (historical values) |
| 3.2 | 29 | `tab:group-statistics` | Group graph statistics | empirical (historical values) |
| 5.1 | 45 | `tab:synthesis-comparison` | Evidence-calibrated study comparison | source-authored (conceptual) |
| C (47) | — | `tab:attractor-training-config` | Attractor-training configuration | source-authored (config record) |

**6 tables.** Only 3.1 and 3.2 carry empirical numbers; the rest are notation/conceptual/config.

## Visible numerical claims (headline oracle values)

**Table 3.1 — mean output SNR (dB), whitened test patches** (columns: Input, DAE, Diag-1,
Diag, 3x3-1, 3x3, Fully learned). Fully learned is the max in every row:

| Input | DAE | Diag-1 | Diag | 3x3-1 | 3x3 | Fully learned |
|---|---|---|---|---|---|---|
| 0.000 | 4.039 | 3.755 | 3.771 | 4.200 | 4.236 | **4.309** |
| 3.000 | 5.776 | 5.541 | 5.593 | 5.996 | 6.059 | **6.121** |
| 6.000 | 7.838 | 7.667 | 7.747 | 8.059 | 8.139 | **8.146** |

**Table 3.2 — thresholded group graph statistics** (mean ± sd *within one matrix*, not across
seeds; threshold = 0.1 × row max):

| Statistic | Diagonal | 3x3 | Learned 0 dB | Learned 3 dB | Learned 6 dB |
|---|---|---|---|---|---|
| Group size | 1 | 9 | 9.0±1.5 | 9.4±1.4 | 8.8±2.4 |
| Membership size | 1 | 9 | 9.0±1.7 | 9.4±2.6 | 8.8±3.1 |
| Group neighbor size | 0 | 24 | 30.2±5.8 | 33.9±6.5 | 32.3±10.1 |
| Unit neighbor size | 0 | 24 | 30.5±6.1 | 31.4±6.5 | 29.5±9.9 |

**Other embedded numeric parameters (prose/appendix, not experiment outputs):**
- Sparse model: `N = M = K = 256`; patches `16×16`; input SNR `0/3/6 dB`; van Hateren
  35 train / 35 test images; whitening `f_c = 0.2375` cycles/px, center crop `1024×1024`,
  freq crop `512×512`, patch stride 8.
- Group-model training (App B.2): inference step `α = 0.05`, **150 inference steps**,
  50,000 iterations, dict LR `0.001`, group LR `0.01/0.005/0.003` for `0/3/6 dB`, LR schedule
  `η_i = η_0/(1+i/1000)`, membership threshold `0.1 × row max`, Gabor envelope corrected sign.
- Attractor (App C): `N = 64`, state/weight noise sd `0.1`, velocity gain `α = 100`, `τ = 5`,
  50 unrolled steps, 40,000 iterations, batch 256, LR `0.002` (a,b,c) / `0.05` (S,A),
  symmetric-weight max abs entry 1, antisymmetric unconstrained. Best mixed code = 32:32 split,
  grid frequency 7.

## Citations (30 unique bibkeys)

`alain14, ba09, banino18, by09, cueva18, gardner22, gregor10, ha00, ha01ica, ha01sc, hj98,
ht05, jenatton11, ky09, lehtinen18, mairal11, me14, ob96, ob97, oj71, se01, sh12, song19,
ss02, ss11, stemmler15, wei15, yoon13, zc99, zhang17`

## Equation labels per source file

chap2: 13 · chap3: 13 · chap4: 10 · appendix: 2 · chap1/chap5: 0 (prose only).
Key corrected equations vs 2016: group activity `eq:group-activity` (√ with ε), signed group
gradient `eq:group-coefficient-gradient` (not a soft threshold), corrected Gabor
`eq:corrected-gabor` (negative envelope exponent), ML gradient 1/2 factor
(`eq:lambda-likelihood-gradient`, App A), discrete recurrence `eq:attractor-discrete`
(Δt/τ step), S/A decomposition `eq:attractor-effective-weight`.

## Regression checks the working edition must pass

1. Title, subtitle, author, committee, degree, date match the table above.
2. All 5 chapters + 4 appendices present, in order, with the listed section titles.
3. All 16 figure labels and 6 table labels resolve (no missing/duplicate labels).
4. Empirical values in Tables 3.1/3.2 match this oracle **or** are superseded by a generated
   artifact whose change is recorded in `rewrite-2026-inventory.csv` + the claim-evidence matrix.
5. All 30 citations resolve; no undefined `\cite`/`\ref`.
6. The five mathematical corrections above are preserved (not regressed to 2016 forms).
