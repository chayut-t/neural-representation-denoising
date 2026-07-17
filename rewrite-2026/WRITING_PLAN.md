# Dissertation writing improvement plan

## 1. Revision objective

The dissertation will connect two studies through one evidence-calibrated argument:

> Corruption-based training supplies a local restoration signal for learning structure. In sparse coding, that signal can tune dependencies among latent coefficients; in a recurrent location code, it can tune dynamics that return perturbed activity toward a desired set of states. Transport along that set is a separate operation and must not be conflated with restoration toward it.

This is a unifying computational lens, not a claim that the two models share one biological mechanism.

## 2. Main weaknesses addressed

### Writing and organization

- The introduction begins with broad claims about intelligence before stating the two concrete scientific problems.
- Key terms--representation, manifold, denoising, group structure, attractor, robustness, and error correction--are used before their operational meanings are fixed.
- Background, model assumptions, results, and interpretation often appear in the same paragraph, making the evidential status of a sentence difficult to track.
- The sparse-coding and grid-cell studies are connected mainly by the phrase “denoising autoencoder,” while their distinct state spaces, objectives, and claims are not compared explicitly.
- Captions are long procedural paragraphs, yet some essential model and evaluation details remain only in appendices.
- The conclusion overstates several findings (“proves,” “explains why grid cells exist”) that are supported only by selected simulations.

### Mathematical clarity

- The group-penalty derivative can be mistaken for an adaptive soft threshold even though the displayed quantity contains the signed coefficient. The intended gradient and any proximal threshold must be distinguished.
- The displayed Gabor envelope is missing the negative sign in its Gaussian exponent.
- The derivative of a group norm with respect to a group weight omits a factor of one half; this factor can be absorbed in a proportional update but not in an equality.
- Several normalization indices and noise definitions are ambiguous.
- The attractor chapter uses “energy landscape” for dynamics containing an antisymmetric component. An antisymmetric transport term does not, in general, define or deepen a scalar energy well.
- A broken appendix cross-reference must be replaced by a valid internal reference.

### Evidence and reproducibility

- Figures and numerical tables exist, but the experiment code, checkpoints, per-trial outputs, seeds, and uncertainty estimates do not.
- Most quantitative comparisons appear to be single-run summaries. The text must say “reported” and must not imply replication or statistical significance.
- The place-only/place-grid comparison matches total neuron count and stated noise amplitudes but does not document matching of activity, weight norms, selection budget, decoder quality, or training compute.
- The 2D grid-code construction is a proposal, not an implemented result.
- The claim that a wiring-cost plateau establishes a two-dimensional cortical organization lacks repeated initializations and null-graph controls.

## 3. Revised chapter architecture

### Front matter

- Use a new title that names the two operations: restoration and transport.
- Present the document as a standalone dissertation.
- Write the abstract last and distinguish reported findings from interpretation.
- Add a self-contained reproducibility statement without filesystem or repository references.

### Chapter 1: Questions, thesis, and contributions

- Open with the concrete problem: what does a denoising loss identify about a representation or its dynamics?
- Define the representational level of analysis and the scope of biological interpretation.
- Introduce both studies in parallel: object of learning, corruption, restoration target, learned parameters, and empirical evidence.
- State four bounded contributions and a roadmap.

### Chapter 2: Primer and common framework

- Introduce notation for observations, latent codes, reconstruction maps, and desired state sets.
- Derive the conditional-mean solution to squared-error denoising.
- Explain the small-noise connection between reconstruction displacement and the score of a smoothed density, with its assumptions and limits.
- Give concise primers on sparse coding, structured sparsity, iterative inference, algorithm unrolling, recurrent dynamics, attractors, and symmetric versus antisymmetric operators.
- Explain the evidence labels used throughout the revised edition.

### Chapter 3: Learning group structure in sparse coding

- State the scientific question and related work before the model.
- Define the group penalty and its derivative carefully; separate the well-defined objective from implementation details that cannot be recovered.
- Explain why maximum-likelihood learning requires model and posterior expectations.
- Present denoising-through-inference as the main methodological idea.
- Organize results as toy validation, natural-image denoising, learned dependencies, and topographic embedding.
- For every result, use the sequence: observation -> supported conclusion -> limitation.
- Correct the Gabor equation and weaken the two-dimensional topography claim.

### Chapter 4: Noise-robust location codes in recurrent networks

- Separate biological observations, coding hypotheses, and model assumptions.
- Define the discrete recurrent dynamics directly and state ambiguity in the noise convention.
- Explain write, hold, read, and transport operations before training.
- Derive place and periodic codes, and explain local precision versus global ambiguity.
- Present the comparisons without treating a selected one-module 1D configuration as an explanation of biological grid cells.
- Interpret the staircase map as evidence for discrete fixed points, but remove the unsupported inference from antisymmetric weight magnitude to basin depth.
- Treat multiple modules, fair budget matching, and two-dimensional navigation as future tests.

### Chapter 5: Synthesis and limits

- Compare the two studies in one explicit table.
- Distinguish restoration normal to a desired set from transport tangent to it.
- State what the reported results support, what remains suggestive, and what cannot be assessed.
- Connect the ideas to work on score-based modeling, learned iterative inference, trained navigation networks, and toroidal grid-cell population manifolds.
- End with falsifiable next experiments rather than broad claims about artificial intelligence.

### Appendices

- Preserve and correct the maximum-likelihood derivation.
- State preprocessing and training details directly while fixing notation and the broken cross-reference.
- Add a symbol glossary and an explicit reproducibility boundary.

## 4. Related-work additions

The revision will add primary sources in four focused groups:

1. Denoising, score estimation, and the modern interpretation of denoising vector fields.
2. Learned/unrolled sparse inference and optimization for overlapping structured sparsity.
3. Modern learned image denoisers, used only to contextualize the patch evaluation rather than as unmatched baselines.
4. Grid-code efficiency, trained recurrent navigation systems, direct evidence for continuous-attractor dynamics, and toroidal population geometry.

The literature cutoff for this revision is 2026-07-17. `LITERATURE_MATRIX.csv` records the intended use of each added source.

## 5. Paragraph-level writing rules

- Begin each section with its purpose and end it with the conclusion that the next section needs.
- Use claim -> evidence or derivation -> interpretation -> limitation as the default paragraph sequence.
- Define one term for each concept and avoid switching among “unit,” “neuron,” and “coefficient” without saying which level is intended.
- Replace “better,” “robust,” and “efficient” with the metric, perturbation, resource constraint, and comparison regime.
- Use present tense for mathematical facts, past tense for completed procedures, and calibrated verbs for results.
- Explain why an equation is introduced and what changes when a parameter changes; do not merely restate symbols.
- Keep captions self-contained but short enough that the main argument remains in the body.
- Use “reported result” for a figure or number that lacks an independent replication.

## 6. Acceptance checks

- Every chapter states its question, assumptions, evidence, conclusion, and limitations.
- Every figure has a self-contained caption and no filesystem provenance path.
- No prose claim relies on an unstated numerical value.
- The abstract and conclusion do not exceed the evidence in the result chapters.
- All references and cross-references resolve.
- No duplicate labels, missing citations, or overfull boxes remain in the final build log.
- Every final PDF page is rendered and visually inspected for clipping, overlap, caption legibility, and consistent headers/footers.
