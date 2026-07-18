# Claim–Evidence Matrix

**Purpose (Plan §7 Phase 1, tasks 5–7).** Turn the scientific hypotheses (plan §4) and the
bounded claims stated in the 2026 baseline PDF (§1.5 Contributions, §3.9, §4.10, §5.7) into an
auditable matrix. Each claim carries a null hypothesis, intervention, controls, primary and
secondary metrics, failure interpretation, its location in the 2026 baseline, and the target
section of the working dissertation edition. Every claim is marked **confirmatory** or
**exploratory** *before* full runs (plan §9.1 stage 3–6), and confirmatory metrics/splits are
frozen here in prose before any confirmatory experiment.

Cross-references: hypotheses map to `docs/rewrite-2026-inventory.csv` (`target_claim_id`) and
the historical→modern generators in plan §8. Evidence-language discipline follows PDF §1.6 and
App D: "shows" = derivation or directly visible property; "supports/consistent with" =
controlled experiment; "suggests" = interpretation needing further controls; no result is
called statistically significant without a multi-seed sampling analysis.

---

## Part 1 — Study I: Group sparse coding (baseline Chapter 3)

### C3-H1 — Denoising-through-inference recovers known group structure (toy)
- **Type:** Confirmatory (synthetic; cheap → ≥20 seeds).
- **Null hypothesis:** Backpropagating a denoising loss through finite sparse inference does
  not recover the planted plane-vs-axis group structure better than chance/factorial baseline.
- **Intervention:** Corrupt samples on a known plane∪axis distribution; train dictionary +
  group matrix Λ by differentiating the denoising loss through the unrolled solver.
- **Controls:** factorial (diagonal) penalty; random/shuffled group init; alignment up to group
  and coefficient permutation; sample-size and noise ablations.
- **Primary metric:** permutation-invariant group-recovery score (adjusted Rand / assignment
  precision-recall) vs planted structure.
- **Secondary:** reconstruction MSE; finite-diff gradient check; convergence traces.
- **Failure interpretation:** if recovery is at chance, the denoising surrogate does not
  identify dependency structure even in the ideal case → weaken C1/C2, document.
- **Baseline location:** §3.6 (`fig:group-toy`), Contribution C2 (§1.5), App A.
- **Target section:** dissertation Ch3 "synthetic validation".

### C3-H2 — Learned groups capture reproducible feature dependencies (natural color images)
- **Type:** Confirmatory (representation track, `color_whitened`).
- **Null:** within-group feature similarity (position/orientation/frequency/phase/scale/color
  direction) is indistinguishable from degree- and weight-matched null graphs.
- **Intervention:** learn group structure on public color images (DIV2K train split), opponent-
  color whitened space; characterize atoms (Gabor fits, corrected envelope) and group graphs.
- **Controls:** weight-preserving / degree-preserving null graphs; permutation tests; threshold-
  sensitivity curves (not just 0.1×row-max); diagonal and fixed-topographic models.
- **Primary metric:** within-group vs null feature distance (permutation p, effect size).
- **Secondary:** group/membership/neighbor counts with CIs; Gabor goodness-of-fit; color-direction
  concentration.
- **Failure interpretation:** no null-beating structure → grouping is a display artifact; retire
  the "captures dependencies" claim, keep only method description.
- **Baseline location:** §3.7.2 (`fig:group-examples`, `fig:group-all`, `tab:group-statistics`),
  C2 (§1.5), §3.9.
- **Target section:** dissertation Ch3 "learned dependencies".

### C3-H3 — Learned groups improve held-out denoising under matched conditions
- **Type:** Confirmatory (paired, multi-seed).
- **Null:** under matched inference/parameter/compute, learned group structure gives no
  held-out denoising improvement over factorial sparse coding.
- **Intervention:** train the model family at fixed σ; evaluate on held-out patches/images
  (whitened SNR track) and full-image RGB (PSNR/SSIM track), paired across models.
- **Controls:** identical patches+corruptions per paired model; equal dictionary size/coeff dim;
  documented parameter counts; equal optimization budget; validation-only hyperparameter
  selection; CNN denoiser as *context only*, never as evidence for/against sparse models.
- **Primary metric:** paired ΔSNR (whitened) and ΔPSNR/ΔSSIM (RGB) with bootstrap CIs over images.
- **Secondary:** matched-compute vs best-validation comparison; per-image distribution.
- **Failure interpretation:** no paired advantage → report as null; the contribution becomes the
  method + structure analysis, not a denoising win.
- **Baseline location:** §3.7.1 (`tab:denoising-results`, `fig:group-denoising-examples`), C2 (§1.5).
- **Target section:** dissertation Ch3 "natural-image denoising".

### C3-H4 — Any advantage over a fixed topographic model survives seeds + paired stats
- **Type:** Confirmatory.
- **Null:** learned groups do not beat the strongest fixed-topographic (3×3) model once seed
  variation and paired tests are included (baseline showed only a *small* gap).
- **Intervention:** same as C3-H3 but the specific comparison is learned vs best fixed-topographic.
- **Controls:** ≥5 seeds; paired bootstrap; multiple-comparison correction within the family.
- **Primary metric:** paired Δ(learned − topographic) with CI; sign consistency across seeds.
- **Secondary:** effect size; sensitivity to group count / overcompleteness.
- **Failure interpretation:** if CI includes 0, state that learned groups are *not* reliably
  better than a well-chosen topographic prior (this is the baseline's own caveat, §3.9).
- **Baseline location:** §3.7.1, §3.9 ("small advantage over the most flexible fixed variant").
- **Target section:** dissertation Ch3 "results / limitations".

### C3-H5 — A 2D minimum-wiring embedding usefully describes the dependency graph
- **Type:** Exploratory (baseline evidence is a single layout + single cost curve).
- **Null:** the learned graph is not better described by a 2D embedding than by null graphs or
  higher-dimensional embeddings (no genuine dimensionality plateau).
- **Intervention:** optimize the wiring objective across dimensions and many random starts;
  compare against degree/weight-matched null graphs.
- **Controls:** repeated starts; dimension-normalized cost; null-graph curves; neighborhood
  stability.
- **Primary metric:** held-out/stability criterion for the plateau dimension; cost vs null.
- **Secondary:** local neighborhood preservation across starts.
- **Failure interpretation:** no null-beating plateau → present as exploratory visualization only;
  do **not** claim cortical topography (explicit baseline caveat §3.8/§3.9).
- **Baseline location:** §3.8 (`fig:group-wiring-map`, `fig:group-wiring-dimension`).
- **Target section:** dissertation Ch3 "topographic embedding".

## Part 2 — Study II: Attractor / grid codes (baseline Chapter 4)

### C4-H1 — Noise-trained recurrent dynamics restore prescribed 1D location codes
- **Type:** Confirmatory (reproduction of the original operations).
- **Null:** training with state/weight noise does not yield dynamics that write, hold, and move
  the prescribed codes (place and mixed) with bounded error.
- **Intervention:** train S (hold) and A (move) via BPTT under injected state/weight noise for
  prescribed 1D place and mixed place–periodic codes.
- **Controls:** symmetry/antisymmetry enforced by parameterization (not post-hoc); finite-diff
  gradient checks; time-step convergence; noise-statistics tests.
- **Primary metric:** decoded-vs-target location error for write/hold/move over many trajectories.
- **Secondary:** write time, hold drift rate, velocity gain accuracy.
- **Failure interpretation:** if operations don't reproduce, trace to a documented ambiguity
  (decoder, noise scaling) rather than declaring a discrepancy with 2016 (plan Gate P7).
- **Baseline location:** §4.7.1–4.7.2 (`fig:attractor-weights/write/hold/move`), C3 (§1.5), App C.
- **Target section:** dissertation Ch4 "fair 1D comparison / operations".

### C4-H2 — A mixed place–grid code retains location better than place-only under matched budget
- **Type:** Confirmatory — **this is the central, most contested claim.**
- **Null:** under equal neuron count, mean activity, weight norm, decoder capacity, and training
  compute, the mixed code does not reduce location error vs place-only.
- **Intervention:** budget-matched comparison across place:grid ratio, grid frequency, noise
  regime, and horizon; ratios/frequency chosen on a validation set then frozen.
- **Controls (all required):** equal total neurons; matched mean population activity and L1/L2
  energy; matched recurrent parameter count and spectral norm; common decoder family; matched
  training budget; validation-only selection; controls place-only, replicated place, place+1
  module, grid-only (ambiguity handled), random periodic features matched in freq/power,
  shuffled coupling, multi-module.
- **Primary metric:** location MSE / absolute error under matched budget, with paired bootstrap CIs.
- **Secondary:** circular error; drift diffusion coefficient vs time; recovery probability vs
  perturbation magnitude.
- **Failure interpretation:** if the advantage vanishes under any preregistered budget match →
  revise the central claim (plan Gate P8); report the confound.
- **Baseline location:** §4.7.3–4.7.4 (`fig:attractor-random-noise`, `fig:attractor-fixed-
  perturbation`), C4 (§1.5), §4.10. **Baseline evidence is exploratory** (same figure searches
  and displays; only neuron count matched).
- **Target section:** dissertation Ch4 "budget-matched comparison".

### C4-H3 — Robustness is explained by measured basin geometry, not weight magnitude
- **Type:** Confirmatory.
- **Null:** measured basin geometry / fixed-point density does not differ between codes in a way
  that explains any error difference (i.e., the staircase story is not mechanistic).
- **Intervention:** locate fixed points; measure basin radius/volume, contraction spectra, drift
  resistance under controlled perturbations around each fixed point.
- **Controls:** direct perturbation (not inference from A magnitude); matched budgets from C4-H2.
- **Primary metric:** fixed-point count/spacing; basin radius; contraction rate; escape probability.
- **Secondary:** local Jacobian/Fisher information; code-manifold separation for distant locations.
- **Failure interpretation:** if basin measures don't track robustness → drop the mechanistic
  claim; the baseline explicitly forbids inferring basin depth from antisymmetric weight
  magnitude (§4.7.1, §2.6).
- **Baseline location:** §4.7.5 (`fig:attractor-fixed-points`), §4.8, C4 (§1.5).
- **Target section:** dissertation Ch4 "basin / error-correction analyses".

### C4-H4 — Effect generalizes to multiple modules and 2D
- **Type:** Confirmatory but **new-in-2026** (proposed future work, not in baseline).
- **Null:** multi-module and 2D toroidal codes provide no robustness benefit beyond a single
  optimized module under equal neuron count.
- **Intervention:** implement multi-module 1D and 2D toroidal grid + path integration with two
  antisymmetric generators (A_x, A_y); preregistered bounded search; frozen test trajectories.
- **Controls:** equal neuron count; matched budgets; held-out velocity profiles; correct
  triangular/hexagonal lattice (not coordinate-wise mod 1).
- **Primary metric:** location error / catastrophic-error probability vs horizon in 2D and
  multi-module, budget-matched.
- **Secondary:** loop-closure error; generator commutation/path-dependence.
- **Failure interpretation:** if no benefit, report the negative result; do not present grid
  coding as uniformly superior.
- **Baseline location:** §4.9, §5.6.5 (stated as proposed extension).
- **Target section:** dissertation Ch4 "multi-module and 2D extensions".

### C4-H5 — Grid periodicity causes aliasing/catastrophic errors in identifiable regimes
- **Type:** Confirmatory (tradeoff characterization, not a superiority claim).
- **Null:** there is no regime where periodicity produces catastrophic (aliasing) errors that
  the mean MSE hides.
- **Intervention:** sweep perturbation magnitude, frequency, and horizon; report full error
  distribution and catastrophic-error probability, distinguishing local drift from phase jumps.
- **Controls:** perturb symmetric and antisymmetric components separately; many perturbation
  draws per scale.
- **Primary metric:** catastrophic/alias error probability vs regime; tail of error distribution.
- **Secondary:** drift vs jump decomposition.
- **Failure interpretation:** characterizing the tradeoff is itself the deliverable; the claim
  fails only if grid coding is uniformly safe (which would contradict the coding theory).
- **Baseline location:** §4.3, §4.8 (aliasing failure modes described qualitatively).
- **Target section:** dissertation Ch4 "aliasing / failure regimes".

---

## Confirmatory vs exploratory (frozen before full runs)

| Claim | Status | Note |
|---|---|---|
| C3-H1 | Confirmatory | synthetic, cheap, ≥20 seeds |
| C3-H2 | Confirmatory | needs null-graph + permutation tests |
| C3-H3 | Confirmatory | paired multi-seed, matched compute |
| C3-H4 | Confirmatory | learned vs best topographic; may be null |
| C3-H5 | **Exploratory** | wiring dimensionality; no strong claim in v1 |
| C4-H1 | Confirmatory | reproduce operations |
| C4-H2 | Confirmatory | **central claim**; baseline evidence was exploratory |
| C4-H3 | Confirmatory | direct basin measurement |
| C4-H4 | Confirmatory | new-in-2026 extension (multi-module/2D) |
| C4-H5 | Confirmatory | tradeoff characterization |

## Minimum scientific success (independent of positive results)

Success is defined by *process*, not by any claim confirming (plan §7 Phase 1, task 7):
1. Every reported experiment is reproducible from committed configs + frozen manifests.
2. Every comparison is fair (matched budgets where the mechanism requires) and statistically
   reported (multi-seed, paired CIs, effect sizes).
3. Revised claims match the evidence even if the original claims weaken or reverse — a null
   result changes the dissertation claim (plan §2.3), it is not hidden.

## Non-goals (frozen for v1.0.0)

- Not attempting to beat state-of-the-art color image denoisers (CNN is contextual only).
- No fits to biological neural recordings in the first release.
- No GUI; no mandatory external experiment-tracking service.
- No rewriting of historical text in `legacy/`; no in-place editing of the 2026 rewrite.

## Notes on the two never-mixed color tracks (feeds C3-H2/H3)

- `color_whitened`: opponent-color, DC-removed, training-only whitening — used for basis
  structure, group organization, topography, and thesis-comparable SNR (C3-H2, and the SNR
  side of C3-H3).
- `color_rgb`: normalized RGB, AWGN at σ = 15/255, 25/255, 50/255, overlapping-patch reassembly
  — used for PSNR/SSIM (the RGB side of C3-H3), incl. external BSDS500 generalization.
- Metrics from the two tracks are never combined in one table without an explicit block/label.
