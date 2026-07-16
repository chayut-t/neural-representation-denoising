# Dissertation Modernization and Reproducibility Plan

**Project:** Modern reproduction and revision of *Neural Representation Learning with Denoising Autoencoder Framework*  
**Plan date:** 2026-07-15  
**Plan status:** Execution specification; no implementation has begun  
**Intended outcome:** A public GitHub repository that can recreate every quantitative result, table, and figure in a substantially revised dissertation from documented data and versioned code.

---

## 1. Outcome and definition of done

The project is complete only when a new user can clone the repository on a clean machine, follow the documented data-license steps, run a short verification profile on CPU, run the full experiment profile on suitable GPU hardware, and build the revised dissertation PDF without manually editing any number, table, or figure.

The final repository must contain:

1. An immutable archival copy of the 2016 dissertation sources and PDF.
2. A provenance statement explaining that the current workspace contains the dissertation and generated figures but not the original experiment code.
3. A modern Python package implementing all models, training procedures, evaluation routines, and analyses used by the revised dissertation.
4. Public-data download and validation tools. Dataset files themselves must not be committed or redistributed.
5. A color-image version of the group sparse-coding study using a documented public RGB dataset.
6. A faithful one-dimensional reproduction and a modernized extension of the place/grid attractor-network study.
7. Version-controlled experiment configurations for every reported result.
8. Machine-readable result manifests containing configuration, seed, dataset hashes, environment, Git commit, hardware, metrics, and artifact hashes.
9. Scripts that generate every dissertation table and figure directly from validated result records.
10. Rewritten LaTeX sources with corrected mathematics, clearer logic, updated literature, improved prose, and no unresolved references.
11. A one-command dissertation build and a full end-to-end reproduction command.
12. Automated tests and GitHub Actions checks for the code, data fixtures, result schemas, and LaTeX build.
13. A tagged `v1.0.0` release, `CITATION.cff`, licenses, release checksums, and a permanent research-software archive such as Zenodo.

### Required completion commands

These commands do not exist yet; creating and documenting them is part of the plan. Their final meanings must be stable:

```bash
make bootstrap             # Install locked CPU/dev environment and verify system tools
make data-check            # Validate locally available datasets and licenses
make test                  # Unit, integration, numerical, and schema tests
make reproduce-quick       # Small deterministic CPU reproduction using public/synthetic fixtures
make reproduce-ch2         # Full color sparse-coding experiment suite
make reproduce-ch3         # Full attractor/grid-code experiment suite
make reproduce-all         # Both full suites plus statistical aggregation and figures
make dissertation          # Build the revised PDF from generated artifacts
make verify-release        # Re-run all non-training validation and release checks
```

`make reproduce-all` may require a GPU and substantial time. `make reproduce-quick` must finish on a normal CPU and exercise the same code paths with reduced data and iterations.

---

## 2. Project rules that must not be relaxed

### 2.1 Preserve history before modernization

- Do not overwrite or silently “clean up” historical files.
- Before moving files, create a byte-level manifest of the current workspace, including SHA-256 hashes, sizes, Git status, and the current commit.
- Preserve the submitted eScholarship PDF separately from the locally compiled PDF.
- Put historical source and generated figures under an immutable archival path and tag the preservation commit.
- Record whether each historical artifact is original, reconstructed, or regenerated.

### 2.2 Separate three kinds of evidence

Every result, caption, table, and prose claim must carry one of these statuses in the internal result registry:

- **Historical:** copied from the 2016 dissertation and not rerun.
- **Faithful reimplementation:** newly implemented from the dissertation equations and appendices, with deliberate compatibility settings.
- **Modernized:** corrected, extended, or run with new data/methods and therefore not numerically comparable without qualification.

The revised dissertation must never call a modernized run an exact reproduction of a historical result.

### 2.3 Data and results are immutable inputs to writing

- Dissertation prose may interpret results but must not contain manually transcribed numerical values.
- Tables must be generated as `.tex` fragments from result files.
- Figures must be generated from result files and included by stable paths.
- Once confirmatory experiment configurations are frozen, do not tune on the test set.
- Unexpected or negative results must change the dissertation claim, not be hidden by changing the analysis after the fact.

### 2.4 Reproducibility has a defined boundary

PyTorch documents that exact results are not guaranteed across releases, platforms, or CPU/GPU combinations. The project must therefore promise:

- exact configuration and software reconstruction from the lockfile/container;
- deterministic execution where supported on a specified reference platform;
- tolerance-based numerical reproduction across other supported platforms;
- statistical reproduction of conclusions across multiple seeds.

The repository must state this boundary prominently.

---

## 3. Default technical and scientific decisions

These defaults make the plan executable. Change one only through a short architecture/science decision record in `docs/decisions/`.

### 3.1 Software stack

| Area | Default | Reason |
|---|---|---|
| Language | Python 3.11 | Broad support across the stable scientific and Hydra ecosystem |
| Tensor/autodiff framework | PyTorch, exact version locked at scaffold time | Replaces Theano; supports GPU, unrolled inference, BPTT, and deterministic controls |
| Environment | `uv`, `pyproject.toml`, committed `uv.lock` | Fast, cross-platform locked environments |
| Configuration | Hydra 1.3 stable with structured dataclass configs | Composable datasets/models/experiments and saved resolved configs |
| Core scientific libraries | NumPy, SciPy, pandas, scikit-image, Pillow | Numerical work, image metrics, image I/O, and tabulation |
| Plotting | Matplotlib and seaborn; no manual graphics editing | Scripted PDF/SVG/PNG output |
| CLI | Typer entry points around library functions | Discoverable commands while preserving importable APIs |
| Tests | pytest, Hypothesis where useful, coverage | Numerical and property-based tests |
| Quality | Ruff, mypy, codespell, pre-commit | Formatting, linting, types, and spelling |
| LaTeX | LuaLaTeX, `latexmk`, `biber`, `biblatex` | Modern Unicode typography and reliable builds |
| Experiment logs | Local JSON/JSONL/CSV/Parquet plus TensorBoard event files | No required proprietary service or account |
| Large artifacts | Checksummed release/Zenodo archives | Public, immutable, and independent of a mutable tracking service |

Do not require Weights & Biases, an institutional cluster, or cloud credentials. Optional adapters are acceptable, but the canonical workflow must run locally.

### 3.2 Image datasets

**Primary default: DIV2K high-resolution RGB images.** The official ETH Zurich page describes 800 training, 100 validation, and 100 test images and limits the data to academic research. Use only the 800 public training HR images and 100 public validation HR images; do not depend on unavailable test HR targets.

Proposed fixed split:

- DIV2K `0001`-`0720`: training.
- DIV2K `0721`-`0800`: hyperparameter validation.
- DIV2K `0801`-`0900`: locked in-domain test set.

**External generalization set: BSDS500 color images.** Use only the official fixed test split, subject to its non-commercial research/education terms. This set tests whether the learned representation transfers beyond DIV2K.

**CI fixture:** create a tiny repository-owned synthetic RGB dataset with geometric edges, oriented gratings, smooth color fields, and correlated color features. License it with the code and use it for tests only, never as scientific evidence.

Dataset constraints:

- Do not commit, mirror, or place either third-party dataset in a GitHub release.
- Require explicit local acceptance of dataset terms before download.
- Download only from official sources by default.
- Pin archive hashes and then generate per-file hashes after extraction.
- If the official archive changes, fail with an explanatory message rather than silently accepting new bytes.
- Include a decision gate before confirmatory runs: if the academic/non-commercial terms are incompatible with the intended repository use, replace the primary dataset with a permissively licensed RGB collection and rerun all affected pilot experiments. Code may be open even when data download is restricted.

### 3.3 Two color-image tracks

One preprocessing pipeline cannot cleanly serve both representation analysis and standard visible-image denoising. Use two explicitly named tracks:

1. **Color representation track (`color_whitened`)**
   - Convert decoded RGB into an invertible orthonormal opponent-color basis.
   - Remove patch DC components.
   - Fit spatial/color whitening using training images only.
   - Learn sparse dictionaries and group structure in whitened space.
   - Use this track for basis structure, group organization, topography, and thesis-comparable SNR experiments.

2. **Color denoising track (`color_rgb`)**
   - Operate on normalized RGB patches without spatial whitening.
   - Add AWGN at fixed `sigma = 15/255, 25/255, 50/255` after precisely documenting whether values are gamma-encoded sRGB or linear light.
   - Reassemble full test images using overlapping patches and a fixed blending window.
   - Report RGB PSNR and SSIM on unclipped and clipped outputs, clearly distinguishing them.

Never combine metrics from the two tracks in one table without an explicit block or label.

### 3.4 Attractor-network data

Chapter 3 is a theoretical simulation and does not need an external image-like dataset. Its canonical data will be deterministic synthetic locations and trajectories generated by code. Public neural recordings may be discussed in the literature review, but adding a recording-data fit is out of scope unless a later decision record expands the scientific question.

### 3.5 Licensing defaults

- Code: BSD-3-Clause or MIT, chosen before the first public release.
- New dissertation text and repository-authored figures: CC BY 4.0, subject to confirming rights in the historical material.
- Historical dissertation: preserve its existing copyright notice.
- Third-party data: governed only by the original dataset terms; never covered by the repository license.
- Dependencies: generate a software bill of materials and check license compatibility before `v1.0.0`.

---

## 4. Scientific questions and claim hierarchy

The code and rewritten dissertation should be organized around claims, not around historical file names.

### 4.1 Unifying thesis

**Proposed thesis:** Corruption-based learning can reveal dependencies that define a representation manifold and can train recurrent dynamics that return noisy neural states toward that manifold; complementary transport dynamics can then move states along the manifold.

This is a proposed framing. Final wording must wait until the confirmatory results are complete.

### 4.2 Chapter 2 hypotheses

- **C2-H1:** Backpropagating a denoising objective through sparse inference can recover known group structure in controlled synthetic data.
- **C2-H2:** On public color natural images, learned group structure captures reproducible dependencies in position, orientation, spatial frequency, and color direction.
- **C2-H3:** Under matched inference, parameter, and compute conditions, learned group structure improves held-out denoising over factorial sparse coding.
- **C2-H4:** Any benefit over a fixed topographic group model remains after seed variation and paired statistical analysis.
- **C2-H5:** A two-dimensional minimum-wiring embedding is a useful description of the learned dependency graph, assessed against null graphs and higher-dimensional embeddings.

### 4.3 Chapter 3 hypotheses

- **C3-H1:** Training with state and weight noise produces recurrent dynamics that restore prescribed location-code states after perturbation.
- **C3-H2:** Under equal neuron count, activity budget, parameter constraints, training budget, and decoder quality, a place-grid code can retain location more accurately than a place-only code in specified noise regimes.
- **C3-H3:** Improved robustness is explained by directly measured basin geometry, fixed-point density, and drift resistance, not merely by visual inspection or weight magnitude.
- **C3-H4:** The effect generalizes from the original one-dimensional, one-grid-module setup to preregistered multi-module and two-dimensional settings.
- **C3-H5:** There are regimes where grid periodicity causes aliasing or catastrophic errors; the dissertation must characterize this tradeoff rather than presenting grid coding as uniformly superior.

### 4.4 Language strength

Use the following evidence language consistently:

- Mathematical derivation: “shows,” “implies under assumptions,” or “establishes.”
- Controlled numerical experiment: “supports,” “is consistent with,” or “demonstrates within this model.”
- Biological interpretation: “suggests” or “provides a possible functional explanation.”
- Do not write that simulations “prove why grid cells exist.”

---

## 5. Target repository structure

```text
neural-representation-denoising/
├── README.md
├── LICENSE
├── LICENSE-TEXT
├── CITATION.cff
├── CONTRIBUTING.md
├── CODE_OF_CONDUCT.md
├── CHANGELOG.md
├── Makefile
├── pyproject.toml
├── uv.lock
├── .python-version
├── .gitignore
├── .pre-commit-config.yaml
├── configs/
│   ├── config.yaml
│   ├── dataset/
│   ├── preprocessing/
│   ├── model/
│   ├── trainer/
│   ├── evaluation/
│   ├── experiment/ch2/
│   └── experiment/ch3/
├── src/neural_repr/
│   ├── cli/
│   ├── common/
│   ├── data/
│   ├── sparse_coding/
│   ├── attractor/
│   ├── evaluation/
│   ├── statistics/
│   ├── plotting/
│   └── provenance/
├── scripts/
│   ├── data/
│   ├── experiments/
│   ├── aggregate/
│   ├── figures/
│   └── release/
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── numerical/
│   ├── regression/
│   └── fixtures/
├── data/
│   ├── README.md
│   ├── licenses/
│   ├── manifests/
│   ├── raw/                 # gitignored
│   ├── interim/             # gitignored
│   └── processed/           # gitignored
├── runs/                    # gitignored working experiment outputs
├── results/
│   ├── frozen-configs/
│   ├── manifests/
│   ├── metrics/
│   ├── statistics/
│   └── checksums/
├── artifacts/
│   ├── figures/             # final generated PDF/SVG/PNG
│   ├── tables/              # CSV plus generated TeX
│   └── models/README.md     # links/checksums; large weights not in Git
├── dissertation/
│   ├── main.tex
│   ├── latexmkrc
│   ├── preamble/
│   ├── frontmatter/
│   ├── chapters/
│   ├── appendices/
│   ├── generated/           # links or copies of generated TeX tables/macros
│   ├── bibliography/
│   └── styles/
├── legacy/
│   └── dissertation-2016/
│       ├── README.md
│       ├── source/
│       ├── figures/
│       ├── pdf/
│       └── MANIFEST.sha256
├── docs/
│   ├── architecture.md
│   ├── reproducibility.md
│   ├── data-cards/
│   ├── experiment-cards/
│   ├── result-inventory.csv
│   ├── claim-evidence-matrix.md
│   ├── symbol-table.md
│   ├── literature-matrix.csv
│   ├── decisions/
│   └── release-runbook.md
├── containers/
│   ├── Dockerfile.cpu
│   ├── Dockerfile.cuda
│   └── README.md
└── .github/
    ├── workflows/
    ├── ISSUE_TEMPLATE/
    └── pull_request_template.md
```

Keep notebooks out of the canonical pipeline. Exploratory notebooks may live in `notebooks/`, but every result used in the dissertation must be generated by importable, tested code and a committed configuration.

---

## 6. Reproducibility contract and artifact schema

### 6.1 Every run must record

- experiment name and semantic version;
- fully resolved Hydra configuration;
- Git commit and dirty-worktree flag;
- Python, PyTorch, CUDA, cuDNN, driver, operating system, CPU, GPU, and RAM information;
- `uv.lock` hash and container digest when applicable;
- dataset name, version, license identifier/text hash, archive hash, per-file manifest hash, and split-manifest hash;
- preprocessing fit-manifest hash;
- random seeds for Python, NumPy, PyTorch CPU, PyTorch CUDA, data sampling, corruption, and bootstrap procedures;
- determinism settings and any nondeterministic operation warnings;
- start/end timestamps and resource usage;
- training history, final metrics, checkpoint hash, and failure status;
- hashes of every generated table and figure.

### 6.2 Result lifecycle

1. A run writes to `runs/<date>/<experiment>/<run-id>/`.
2. Validation checks config completeness, metric schema, finite values, expected sample counts, and file hashes.
3. Aggregation reads only validated runs.
4. Confirmatory run IDs are copied into a committed manifest in `results/manifests/`.
5. Figures and tables record the exact aggregate-manifest hash in metadata or a sidecar JSON.
6. The dissertation build checks that every included generated artifact is present in the final manifest.

### 6.3 Determinism modes

- `debug`: deterministic algorithms required; fail on nondeterministic operations; CPU or reference CUDA platform.
- `research`: deterministic algorithms requested; warnings recorded; used for normal development.
- `benchmark`: performance settings permitted, but at least five independent seeds required.

For the reference release, rerun one full seed twice on the same reference machine and require byte-identical metrics or a documented tolerance. Across other platforms, use numerical tolerances and require the same qualitative conclusions.

---

## 7. Phase-by-phase execution plan

Each phase ends with a gate. Do not begin the next dependent phase until the gate is satisfied.

### Phase 0 - Preserve the 2016 work and establish provenance

**Purpose:** Ensure modernization cannot destroy history and document what is and is not reproducible from the available materials.

Tasks:

1. Record current Git branch, commit, remotes, tags, status, and all modified/untracked files.
2. Determine why most current files appear modified. Compare content hashes and file-mode changes against `HEAD`; do not discard anything.
3. Create `legacy/dissertation-2016/MANIFEST.sha256` for all source, figure, bibliography, script, and PDF files.
4. Preserve both `thesis.pdf` and `qt0hm6p6s5.pdf`; document that the eScholarship copy adds one repository page and otherwise has matching extracted text.
5. Move or copy the historical LaTeX, images, and build script into the legacy tree using history-preserving Git operations.
6. Create `legacy/dissertation-2016/README.md` containing:
   - original title, author, institution, year, and permanent eScholarship link;
   - build environment if discoverable;
   - known unresolved reference (`chapter ??` in Appendix B);
   - statement that original experiment source code is not present in this workspace.
7. Search all branches, Git history, ignored files, neighboring archive directories explicitly authorized for the project, old backups provided by the author, and publication supplements for the original Theano/Python/MATLAB code.
8. If code is recovered, preserve it unchanged under `legacy/code-original/` with environment notes. Do not make the modern package depend on it.
9. Create `docs/result-inventory.csv`, with one row for every historical figure, table, and reported numerical claim. Required columns:
   - chapter;
   - original label;
   - source file/page;
   - scientific claim;
   - known inputs;
   - known hyperparameters;
   - historical output path;
   - original-code status;
   - planned replacement experiment;
   - planned generator command;
   - verification status.
10. Tag the preservation commit `archive/dissertation-2016` or create an equivalent signed archival tag.

**Outputs:** immutable legacy tree, manifests, provenance README, complete result inventory.

**Gate P0:** A clean checkout of the archival tag builds or at least contains the historical PDF; every historical result has an inventory row; no preexisting user file has been lost.

### Phase 1 - Freeze scope, claims, and decision records

**Purpose:** Prevent uncontrolled expansion and hindsight-driven claims.

Tasks:

1. Create decision records for:
   - primary/secondary datasets and license implications;
   - Python/PyTorch/Hydra versions;
   - color representation and noise domain;
   - legacy-compatible versus corrected model equations;
   - confirmatory seeds and statistical tests;
   - LaTeX engine/template;
   - repository/text licenses.
2. Convert Section 4 hypotheses into `docs/claim-evidence-matrix.md` with columns for null hypothesis, intervention, controls, primary metric, secondary metrics, failure interpretation, and dissertation section.
3. Mark analyses as **confirmatory** or **exploratory** before full runs.
4. Define minimum scientific success independently of positive results:
   - all experiments reproducible;
   - comparisons fair and statistically reported;
   - revised claims match evidence even if original claims weaken.
5. Define non-goals:
   - no attempt to beat state-of-the-art color denoisers;
   - no fit to biological neural recordings in the first release;
   - no GUI;
   - no mandatory external experiment-tracking service;
   - no rewriting of historical text in the archive.

**Gate P1:** All decisions that alter data, equations, baselines, or claims have an accepted record; confirmatory metrics and splits are frozen in prose before full experiments.

### Phase 2 - Scaffold the modern repository and development environment

**Purpose:** Build a stable foundation before scientific code.

Tasks:

1. Create the target directory structure from Section 5.
2. Initialize the package with typed modules and console entry points:
   - `neural-repr-data`;
   - `neural-repr-train`;
   - `neural-repr-eval`;
   - `neural-repr-aggregate`;
   - `neural-repr-figure`;
   - `neural-repr-verify`.
3. Create `pyproject.toml`, `.python-version`, and a committed `uv.lock`.
4. Separate dependency groups: `dev`, `image`, `latex-tools`, `cuda`, and `release` where practical.
5. Export `pylock.toml` or a CycloneDX SBOM during releases; do not maintain an independently edited `requirements.txt`.
6. Add Ruff, mypy, pytest, coverage, codespell, and pre-commit configuration.
7. Create CPU and CUDA containers pinned by digest. Record the supported reference GPU/CUDA stack.
8. Implement a `system-info` command that writes the environment fields in Section 6.
9. Add GitHub Actions jobs for:
   - lockfile consistency;
   - lint/type/spelling checks;
   - CPU unit and integration tests;
   - numerical smoke tests;
   - LaTeX build;
   - citation/schema validation;
   - container builds on release or scheduled runs.
10. Write a minimal README quick start and contributor setup.

**Gate P2:** A fresh Linux runner can install with `uv sync --locked`, import the package, run tests, run a CLI help command, and build a placeholder dissertation PDF.

### Phase 3 - Formalize and test the mathematics before training

**Purpose:** Correct ambiguous or inaccurate equations and ensure the implementation matches the revised notation.

#### 3A. Group sparse coding specification

1. Write a standalone mathematical specification for:
   - data vector and color/channel ordering;
   - dictionary dimensions and column normalization;
   - coefficient inference objective;
   - group activity and group penalty;
   - nonnegativity/normalization of the group matrix;
   - denoising objective;
   - dictionary and group-structure training objectives;
   - the distinction between the probabilistic maximum-likelihood objective and the denoising surrogate.
2. Use a numerically safe group activity such as

   \[
   z_k(\mathbf{s}) = \sqrt{\varepsilon + \sum_i \Lambda_{ki}s_i^2},
   \]

   and treat `epsilon` as a documented numerical parameter.
3. Parameterize `Lambda` to be nonnegative, for example with softplus, and resolve scale non-identifiability with a row/column normalization or explicit regularizer. Record the chosen convention.
4. Derive the actual inference update. The historical description mixes ISTA-style soft thresholding with an adaptive gradient of the group penalty. Implement and name separate variants if needed:
   - `legacy_adaptive_threshold` for historical compatibility;
   - `proximal_group` or a clearly derived smooth-gradient method for the modern model.
5. Numerically verify all analytic gradients against finite differences on tiny problems.
6. Compare unrolled gradients against implicit/converged gradients on toy cases.
7. Correct the Gabor model used for analysis: the Gaussian envelope needs a negative quadratic exponent and a documented factor convention.
8. Add tests for invariances, degenerate groups, zero coefficients, normalization, convergence, and reconstruction consistency.

#### 3B. Attractor-network specification

1. Make the time step explicit:

   \[
   \mathbf{u}_{t+1}=\mathbf{u}_t+\frac{\Delta t}{\tau}
   \left[-\mathbf{u}_t+(\mathbf{S}+\sum_d \alpha_d v_{d,t}\mathbf{A}_d+\Xi_{S,t})
   \mathbf{f}^{\mathrm{eff}}_t+\boldsymbol{\xi}_{u,t}\right].
   \]

2. Enforce structure by parameterization:

   \[
   \mathbf{S}=\tfrac12(\mathbf{W}+\mathbf{W}^{\mathsf T}),\qquad
   \mathbf{A}_d=\tfrac12(\mathbf{V}_d-\mathbf{V}_d^{\mathsf T}).
   \]

3. Define write, hold, and move operations with consistent indexing and units.
4. Specify whether “weight noise” is independently resampled each step, each trajectory, or fixed as a perturbation. These are different experiments.
5. Correct the epilogue derivation:
   - the first-order update uses `delta t`, not absolute `t`;
   - continuous-time amplification depends on the sign of an eigenvalue, not whether it is greater than one;
   - an antisymmetric generator produces norm-preserving motion only under the stated continuous-time assumptions.
6. Avoid calling general nonsymmetric recurrent dynamics an energy landscape. Use “fixed-point and basin structure” unless a valid Lyapunov function is derived.
7. Add numerical tests for exact symmetry/antisymmetry, zero-velocity holding, velocity reversal, time-step convergence, noise statistics, circular boundary behavior, and gradient correctness through time.

**Gate P3:** The math specification, code docstrings, LaTeX notation, and tests agree; finite-difference gradient tests pass; all known historical mathematical issues have explicit resolutions.

### Phase 4 - Implement the data, color, corruption, and patch pipelines

**Purpose:** Make data preparation deterministic, legal, inspectable, and reversible.

Tasks:

1. Implement a dataset registry containing official URLs, citations, terms URLs, expected archives, and hashes.
2. Require commands such as:

   ```bash
   neural-repr-data download div2k --accept-academic-research-terms
   neural-repr-data download bsds500 --accept-noncommercial-research-terms
   ```

3. Store acceptance timestamp and terms hash locally; do not commit personal acceptance records.
4. Validate image counts, IDs, formats, shapes, color modes, corrupt files, duplicate hashes, and split leakage.
5. Save fixed split manifests and per-image SHA-256 hashes in Git.
6. Implement deterministic patch sampling based on `(dataset hash, image ID, seed, patch index)` rather than mutable iteration order.
7. Make patch size, stride, sample count, border policy, and augmentation configurable.
8. Use only flips and 90-degree rotations in the confirmatory natural-image pipeline unless an ablation justifies other augmentation. Do not use color jitter when studying color dependencies.
9. Implement and test:
   - sRGB decode/encode;
   - optional linear-light conversion;
   - orthonormal opponent-color transform and inverse;
   - DC removal and restoration;
   - training-only whitening fit, transform, and inverse;
   - global versus per-patch normalization;
   - AWGN by target SNR;
   - AWGN by fixed sigma;
   - overlapping patch extraction/reassembly with normalized blending.
10. Store fitted preprocessing statistics as small versioned artifacts with input-manifest hashes.
11. Generate a data audit report with channel histograms, patch power, whitening covariance checks, train/test similarity checks, and sample images before/after transforms.
12. Create tiny synthetic and tiny sampled integration fixtures that do not contain restricted third-party images.

**Gate P4:** Re-running preprocessing yields identical manifests and arrays on the reference platform; inverse transforms reconstruct inputs within tolerance; no test image influences fitted preprocessing; license handling is documented.

### Phase 5 - Reimplement and validate Chapter 2 on controlled data

**Purpose:** Establish correctness before expensive color experiments.

Tasks:

1. Implement modular components:
   - dictionary object with column projection;
   - standard sparse-coding inference;
   - fixed topographic group structure;
   - learned group structure;
   - unrolled solver with optional activation checkpointing;
   - dictionary update;
   - denoising-loss group update;
   - matched DAE baselines.
2. Recreate the three-dimensional toy distribution: points on an `xy` plane or `z` axis with deterministic generation.
3. Recreate the iso-cost surfaces for factorial, grouped, and non-sparse penalties.
4. Quantify toy recovery rather than relying only on a picture:
   - align learned bases to ground truth;
   - compute group assignment precision/recall or adjusted Rand score;
   - report recovery across at least 20 seeds;
   - include failure cases and sample-size/noise ablations.
5. Run both historical-compatible and corrected inference variants on the same toy data.
6. Establish numerical regression tests using tiny fixed seeds and tolerances.
7. Benchmark inference convergence, memory, and runtime to choose the modern unroll length. Retain 150 steps as a historical compatibility configuration, not automatically as the modern default.

**Outputs:** replacement iso-cost figure, toy-data figure, quantitative recovery table, solver benchmark.

**Gate P5:** The method recovers the known group structure above the preregistered threshold or the failure is understood and documented; gradients and objective traces behave as predicted; no NaNs or unbounded scaling occur.

### Phase 6 - Run the modern color group sparse-coding study

**Purpose:** Replace the grayscale van Hateren experiment with a stronger, public, color experiment.

#### 6A. Baselines and fairness

Implement these thesis-comparable models:

1. `noisy_input` identity baseline.
2. Standard sparse coding with a single global penalty (`diag_1`).
3. Standard sparse coding with learned per-unit penalties (`diag`).
4. Fixed topographic group sparse coding with shared weights (`topographic_1`).
5. Fixed topographic group sparse coding with per-group weights (`topographic`).
6. Fully learned group sparse coding (`learned_group`).
7. Matched shallow DAE (`dae_mlp_matched`).

Add one clearly separated modern-context baseline:

8. A small convolutional denoising autoencoder (`dae_conv_reference`).

Do not use the convolutional model as evidence that group sparse coding is or is not useful; it answers a different capacity question.

Fairness constraints:

- identical train/validation/test patches and corruptions for paired models;
- equal dictionary size and coefficient dimension among sparse models;
- documented parameter counts;
- equal optimization-update budget where meaningful;
- validation-only hyperparameter selection;
- no test-based choice of noise level, group size, or basis count;
- report both best-validation and matched-compute comparisons.

#### 6B. Pilot stage

1. Start with `16 x 16 x 3` patches for comparability and manageable cost.
2. Pilot dictionary sizes of 256, 512, and 768; do not assume the original 256 units remains appropriate for three channels.
3. Pilot group counts equal to dictionary size plus smaller/larger alternatives.
4. Run one seed on a small training subset to eliminate unstable configurations.
5. Tune only the declared search space using DIV2K validation images.
6. Choose the final configurations using primary validation metrics plus stability and compute constraints.
7. Freeze resolved YAML configs and their hashes before confirmatory seeds.

#### 6C. Confirmatory representation track

For each model, noise condition, and five or more fixed seeds:

1. Train on the fixed DIV2K training manifest.
2. Evaluate thesis-comparable denoising at input SNR `0`, `3`, and `6` dB in whitened opponent space.
3. Report output SNR, reconstruction MSE, inference objective, sparsity, active-group count, and runtime.
4. Characterize every basis with:
   - spatial center and envelope;
   - orientation;
   - spatial frequency;
   - phase;
   - luminance/chromatic energy ratio;
   - normalized color direction in opponent space;
   - goodness of Gabor fit.
5. Characterize groups with weighted and thresholded statistics. The historical `10% of row maximum` membership threshold may be shown for comparability, but include threshold-sensitivity curves.
6. Measure within-group versus null-group similarity for position, orientation, frequency, phase, scale, and color direction.
7. Use permutation tests against degree-preserving or weight-shuffled null structures.
8. Compute group size, membership count, group-neighbor count, and unit-neighbor count with confidence intervals.
9. Repeat minimum-wiring embeddings across dimensions, random starts, and null graphs. Report held-out/stability criteria rather than declaring “dimension two” from a single plateau by eye.

#### 6D. Confirmatory RGB denoising track

1. Train/evaluate at sigma `15/255`, `25/255`, and `50/255`.
2. Reconstruct full DIV2K test images from overlapping patches.
3. Evaluate the untouched BSDS500 test split as an external generalization set.
4. Report per-image and aggregate PSNR/SSIM, with paired confidence intervals.
5. Include color-error metrics such as per-channel RMSE or Delta E only if the exact color-space assumptions are documented and validated.
6. Produce color examples using fixed image IDs and crop coordinates selected before final results.
7. Show original, noisy, and model outputs with identical display scaling and no manual retouching.

#### 6E. Required ablations

- grayscale versus RGB/opponent color;
- whitened versus non-whitened representation;
- learned versus fixed dictionary during group learning;
- legacy versus corrected inference;
- unroll length and convergence tolerance;
- group normalization/regularization;
- group count and dictionary overcompleteness;
- patch size `16` versus a larger preregistered size;
- single-noise training versus noise-conditional training;
- test-time solver iterations beyond the training unroll length.

**Outputs replacing Chapter 2:**

- iso-cost surfaces;
- toy recovery visualization and table;
- color denoising comparison table;
- color denoising examples;
- color basis atlas;
- representative learned groups;
- full group-structure visualization;
- group-statistics table;
- feature-similarity/null comparison figure;
- two-dimensional minimum-wiring embedding;
- wiring cost/stability versus dimension;
- ablation and cross-dataset generalization tables.

**Gate P6:** All confirmatory run IDs are frozen; no missing or failed seed is silently omitted; paired statistics are complete; the revised Chapter 2 claim language is determined from these results.

### Phase 7 - Reimplement the original one-dimensional attractor study

**Purpose:** Reproduce the original conceptual experiment before extending it.

Tasks:

1. Implement location encoders:
   - line-topology Gaussian place code;
   - ring-topology periodic grid code;
   - concatenated place-grid code.
2. Implement decoders:
   - historical weighted-average decoder;
   - common nearest-codebook or maximum-likelihood decoder applied identically to all codes;
   - optional learned decoder trained only on a training split.
3. Implement write, hold, and move trajectories.
4. Implement historical hyperparameters as a named immutable config:
   - 64 units;
   - unit and weight-noise standard deviations 0.1;
   - velocity gain 100;
   - time constant 5;
   - 50 training steps;
   - 40,000 iterations;
   - batch size 256;
   - documented learning rates and momentum schedule.
5. Parameterize symmetry/antisymmetry exactly rather than correcting weights after updates.
6. Recreate:
   - symmetric and antisymmetric weight matrices;
   - write activity traces;
   - hold traces;
   - move traces;
   - error versus hold duration;
   - error versus constant perturbation;
   - input-output fixed-point staircase.
7. Compare the new traces with historical figures qualitatively and, when source values can be digitized reliably, quantitatively. Label digitized historical values as approximate.
8. Run at least five seeds and show variability absent from the original figures.

**Gate P7:** The implementation writes, holds, and moves both codes; historical qualitative patterns are either reproduced or discrepancies are traced to an explicit ambiguity/implementation difference.

### Phase 8 - Make the place/grid comparison scientifically fair

**Purpose:** Remove confounds before making an error-correction claim.

Tasks:

1. Equalize or explicitly vary these budgets:
   - total neuron count;
   - mean population activity;
   - population L1/L2 energy;
   - tuning-width coverage;
   - recurrent parameter count;
   - spectral norm or effective weight scale;
   - training steps and examples;
   - decoder capacity.
2. Constrain both symmetric and antisymmetric weights comparably. The historical unconstrained antisymmetric matrix can make “greater traction” a weight-scale confound.
3. Choose place:grid ratio and grid frequency on a validation set, then lock them before test evaluation.
4. Include controls:
   - place-only;
   - replicated place code;
   - place plus one grid module;
   - grid-only where decoding ambiguity is handled explicitly;
   - random periodic features matched in frequency/power;
   - shuffled place-grid coupling;
   - multi-module grid code.
5. Test separately:
   - temporally resampled state noise;
   - temporally resampled weight noise;
   - fixed weight perturbation;
   - neuron dropout;
   - synapse dropout;
   - velocity noise/bias;
   - cue-reset errors;
   - long-horizon accumulation.
6. Measure:
   - location MSE and absolute error;
   - circular error where relevant;
   - catastrophic/alias error probability;
   - drift diffusion coefficient versus time;
   - recovery probability versus perturbation magnitude;
   - basin radius and basin volume;
   - fixed-point count and spacing;
   - local code Jacobian/Fisher information;
   - code-manifold separation for distant locations;
   - energy/activity/weight budgets;
   - runtime and training stability.
7. Directly test basin depth using controlled perturbations. Do not infer depth solely from the magnitude of antisymmetric weights.
8. Report performance surfaces over ratio, frequency, noise, and horizon rather than only the best curve.

**Gate P8:** Any place-grid advantage persists under at least one preregistered budget-matched comparison and its failure regimes are reported. If not, revise the central claim accordingly.

### Phase 9 - Extend the attractor study to multiple modules and two dimensions

**Purpose:** Complete the most important future work identified by the 2016 dissertation.

#### 9A. Multiple modules in one dimension

1. Define modules with explicit periods/frequencies, phases, neuron allocations, and activity normalization.
2. Compare geometric scale ratios, learned scale ratios, and matched random ratios.
3. Avoid combinatorial tuning by declaring a bounded search and selecting on validation trajectories.
4. Measure local precision versus global ambiguity/catastrophic decoding.
5. Test whether multiple modules improve robustness beyond a single optimized module under equal neuron count.

#### 9B. Two-dimensional extension

1. Define a bounded 2D arena and a toroidal alternative as separate experiments.
2. Generate deterministic training, validation, and test trajectories with held-out velocity profiles.
3. Use two antisymmetric transport generators, `A_x` and `A_y`.
4. Implement place fields on a triangular packing where appropriate.
5. Implement grid modules using a mathematically correct triangular/hexagonal lattice rather than an informal coordinate transformation.
6. Specify boundary behavior: reflecting, periodic, or cue reset.
7. Use the same decoder family and budget matching across codes.
8. Test path integration under turns, varying speeds, pauses, velocity noise, and long loops.
9. Analyze commutation/path-dependence of transport generators and quantify loop-closure error.
10. Visualize code manifolds, trajectory decoding, error maps, and basin recovery in 2D.

#### 9C. Modern training improvements

- Use multi-horizon rollout losses rather than only final-step error.
- Train on a distribution of noise levels and horizons, with a held-out evaluation grid.
- Compare curriculum and direct long-horizon training.
- Use gradient clipping and log exploding/vanishing gradient diagnostics.
- Benchmark Euler step size and, if necessary, a higher-order reference integrator for validation.
- Keep the simplest model that passes the preregistered tests; do not add architectural complexity merely to improve plots.

**Outputs:** multi-module performance surfaces, 2D weight/dynamics figures, 2D path integration examples, failure-mode maps, and expanded robustness tables.

**Gate P9:** The 2D and multi-module experiments pass numerical validation, have frozen confirmatory runs, and support only claims that survive equal-budget controls.

### Phase 10 - Statistical analysis and robustness audit

**Purpose:** Replace single-run visual arguments with reproducible inference.

Tasks:

1. Use at least five independent training seeds for expensive confirmatory models and more seeds for cheap toy/simulation experiments.
2. Use the same corruption/trajectory seeds across paired model comparisons.
3. Treat images or independently generated trajectories—not individual overlapping patches/time steps—as the primary resampling unit.
4. Report mean, standard deviation, median where useful, and 95% confidence intervals.
5. Use paired bootstrap confidence intervals for denoising and trajectory errors.
6. Use permutation tests for group-feature organization and graph null comparisons.
7. Correct for multiple comparisons within declared families or clearly label exploratory p-values.
8. Report effect sizes and raw per-image/per-seed data, not only p-values.
9. Run sensitivity analyses for:
   - seed set;
   - preprocessing choices;
   - membership threshold;
   - clipping policy;
   - decoder choice;
   - outlier handling;
   - integration step size;
   - hardware precision (`float32` versus selected `float64` checks).
10. Create an automated robustness report that marks every confirmatory claim as supported, unsupported, mixed, or not testable.

**Gate P10:** Statistical scripts reproduce all reported values from frozen manifests; robustness conclusions are machine-readable and match the proposed prose.

### Phase 11 - Build the figure/table pipeline and result-to-prose interface

**Purpose:** Eliminate manual transcription and graphics editing.

Tasks:

1. Give every revised figure and table a stable artifact ID such as `ch2-fig-basis-atlas`.
2. Map every historical result inventory row to a revised artifact or an explicit retirement reason.
3. Generate:
   - vector PDF for LaTeX;
   - SVG for GitHub/web viewing where practical;
   - color-managed PNG previews;
   - sidecar JSON with generator version, run manifest, data hash, and plotting config.
4. Use colorblind-safe palettes, readable line styles, panel labels, units, confidence bands, and consistent fonts.
5. Avoid rasterizing vector plots and avoid Photoshop/Illustrator edits.
6. For image panels, save crop IDs, coordinates, scaling, and clipping policy in config.
7. Generate all table values first as CSV/Parquet, then render `.tex` from the same data.
8. Generate a small LaTeX macro file for repeated headline values, for example `\ChTwoLearnedGroupDeltaPSNR`, so abstract/conclusion values cannot drift.
9. Add artifact tests:
   - expected dimensions/pages;
   - finite values;
   - no empty panels;
   - font embedding;
   - captions reference existing labels;
   - generated files match frozen manifests.
10. Render all dissertation pages to images and visually inspect every figure/table at final size.

**Gate P11:** Every non-historical figure/table in the revised dissertation has a single command, a committed configuration, a validated input manifest, and no manual post-processing.

### Phase 12 - Update the literature and redesign the dissertation argument

**Purpose:** Rewrite from evidence and current scholarship instead of sentence-editing the 2016 document in place.

#### 12A. Literature review

1. Build `docs/literature-matrix.csv` with topic, citation, year, primary contribution, evidence type, relation to this work, and intended section.
2. Search through the final cutoff date for:
   - denoising/score matching and manifold interpretation;
   - algorithm unrolling and differentiable optimization;
   - structured and group sparse coding;
   - learned pooling/topography and natural-image color statistics;
   - modern image denoising evaluation;
   - continuous attractor networks;
   - grid-cell coding, modules, scale ratios, error correction, path integration, and neural-manifold work;
   - biological evidence that supports or challenges the proposed model.
3. Prefer peer-reviewed papers and primary sources. Clearly label preprints.
4. Verify every factual or historical statement against its cited source.
5. Deduplicate and normalize bibliography metadata, including DOI/URL and consistent capitalization.

#### 12B. Proposed revised structure

1. **Front matter**
   - revised title selected after results;
   - abstract written last;
   - reproducibility statement and repository/DOI;
   - acknowledgments and historical note.
2. **Chapter 1: Introduction**
   - precise problem statement;
   - levels of explanation and scope;
   - representation manifolds, corruption, restoration, and transport;
   - research questions and contributions;
   - roadmap.
3. **Chapter 2: Common mathematical and computational framework**
   - denoising as conditional estimation;
   - relation to score/manifold ideas with carefully bounded claims;
   - unrolled inference and BPTT;
   - reproducibility and experimental design;
   - notation shared by later chapters.
4. **Chapter 3: Learning group structure in color sparse coding**
   - model and corrected inference;
   - synthetic validation;
   - public color data and preprocessing;
   - baselines and statistical protocol;
   - results, ablations, topology, limitations.
5. **Chapter 4: Grid codes in noise-robust attractor networks**
   - biological background separated from model assumptions;
   - recurrent dynamics and training;
   - fair 1D comparison;
   - basin/error-correction analyses;
   - multi-module and 2D extensions;
   - biological interpretation and limits.
6. **Chapter 5: Synthesis and discussion**
   - what denoising learns in feedforward/inference versus recurrent settings;
   - stabilization versus transport;
   - commonalities and differences between the studies;
   - what is computational, algorithmic, or biological;
   - negative results, limitations, and future work.
7. **Appendices**
   - complete derivations;
   - hyperparameter/config tables generated from frozen configs;
   - dataset cards and terms summary;
   - additional robustness figures;
   - reproducibility instructions;
   - symbol/acronym glossary.

#### 12C. Writing workflow for every section

1. Write a one-sentence purpose and one-paragraph claim outline.
2. Link every proposed claim to the claim-evidence matrix.
3. Draft paragraphs as claim -> evidence/derivation -> interpretation -> limitation.
4. Define terms before using them and use one term per concept.
5. Distinguish assumptions, definitions, hypotheses, results, and interpretations typographically and verbally.
6. Put model dimensions, noise definitions, and train/test protocol near the first use, not only in an appendix.
7. Explain equations in prose without restating every symbol mechanically.
8. Remove broad AI claims that are not needed for the thesis argument.
9. Replace vague words such as “better,” “efficient,” and “robust” with a metric and comparison regime.
10. Do a separate pass for:
    - logical continuity;
    - mathematical correctness;
    - statistical reporting;
    - biological claim calibration;
    - grammar and fluency;
    - concision;
    - consistent authorial voice.
11. Have a domain expert review Chapter 3 biological claims and a numerical-optimization expert review Chapter 2 inference.
12. Revise the abstract and conclusion only after all result chapters are frozen.

#### 12D. LaTeX architecture

1. Use a self-contained modern class or a documented revised-edition template; keep the historical Berkeley class only in the archive.
2. Use LuaLaTeX with an open TeX Live font such as Libertinus and matching math fonts.
3. Centralize packages, typography, theorem styles, units, colors, and custom commands.
4. Define vector/matrix/random-variable notation macros and enforce them throughout.
5. Use `cleveref`/`hyperref` consistently and eliminate hard-coded section numbers.
6. Keep one label namespace: `chap:`, `sec:`, `eq:`, `fig:`, `tab:`, `app:`.
7. Generate lists of symbols/acronyms if they improve navigation.
8. Build with `latexmk`; treat undefined references, missing citations, overfull boxes beyond a threshold, and duplicate labels as CI failures.
9. Use `\input{}` for generated tables/macros and stable paths for figures.
10. Include repository URL, version, commit, and archive DOI in the final PDF.

**Gate P12:** Every chapter has passed scientific, logic, and prose review; every claim maps to evidence; citations and cross-references resolve; the PDF builds without warnings designated as fatal.

### Phase 13 - End-to-end reproduction and independent verification

**Purpose:** Test the repository as a product, not only as a developer workspace.

Tasks:

1. Provision a clean reference machine/container from documentation only.
2. Clone the repository at the release candidate tag.
3. Run `make bootstrap` and accept/download data through the documented flow.
4. Run `make reproduce-quick`; compare to committed tolerance-based smoke references.
5. Run full Chapter 2 and Chapter 3 pipelines from frozen configurations.
6. Aggregate results and regenerate every figure/table.
7. Build the PDF.
8. Verify artifact hashes against the candidate release manifest.
9. Repeat one full seed on the same reference environment.
10. Run the quick profile on macOS and Linux CPU; run the full profile on at least the reference CUDA environment.
11. Ask an independent reader who did not implement the code to follow the README and record every ambiguity.
12. Fix documentation or automation; do not solve verification problems with undocumented manual steps.

**Gate P13:** A clean independent run produces a valid dissertation and all expected artifacts; deviations are within documented tolerances; the verifier signs the release checklist.

### Phase 14 - Publish and archive the GitHub repository

**Purpose:** Make the work citable, maintainable, and durable.

Tasks:

1. Finalize README sections:
   - scientific summary;
   - repository map;
   - quick and full reproduction;
   - hardware/time expectations;
   - data licenses and download steps;
   - artifact provenance;
   - known limitations;
   - how to cite.
2. Add `CITATION.cff` with the software release and preferred dissertation citation.
3. Add code/text licenses and a third-party notice.
4. Validate contributor and conduct documents.
5. Generate the SBOM, dependency-license report, and release checksums.
6. Create a GitHub release containing:
   - source archive;
   - revised dissertation PDF;
   - small result tables/figures;
   - checksums;
   - links to large model/result archive;
   - release notes describing differences from 2016.
7. Archive the release with Zenodo or an equivalent service and obtain a DOI.
8. Add DOI badge and release DOI to the PDF and `CITATION.cff`; rebuild once without changing scientific content.
9. Tag `v1.0.0` and protect the release tag.
10. Open issues for deferred work rather than expanding the release at the last moment.

**Gate P14:** Public URLs work without authentication except dataset downloads governed by their own terms; citation metadata validates; archive checksums match; `v1.0.0` is reproducible from its own documentation.

---

## 8. Historical-to-modern result map

This table is the minimum replacement set. The result inventory created in Phase 0 must be more granular.

| Historical result | Modern replacement | Canonical generator to create |
|---|---|---|
| Fig. 2.1 group iso-cost surfaces | Corrected vector surfaces plus parameter definitions | `make figure ID=ch3-iso-cost` |
| Fig. 2.2 toy data/recovered contour | Multi-seed toy recovery plus quantitative score | `make figure ID=ch3-toy-recovery` |
| Table 2.1 denoising SNR | Paired multi-seed color SNR plus RGB PSNR/SSIM tables | `make table ID=ch3-denoising` |
| Fig. 2.3 denoising examples | Fixed DIV2K/BSDS color crops with reproducible corruption | `make figure ID=ch3-color-denoising-examples` |
| Fig. 2.4 basis functions | Color/opponent basis atlas for all controlled models | `make figure ID=ch3-color-bases` |
| Fig. 2.5 group examples | Representative groups chosen by fixed rule, not hand selection | `make figure ID=ch3-group-examples` |
| Fig. 2.6 all-group visualization | Full weighted group map plus color-feature annotations | `make figure ID=ch3-group-map` |
| Table 2.2 group statistics | Multi-seed weighted/threshold sensitivity statistics | `make table ID=ch3-group-statistics` |
| Fig. 2.7 minimum-wiring map | Stable 2D embeddings with null comparison | `make figure ID=ch3-wiring-embedding` |
| Fig. 2.8 wiring cost vs dimension | Repeated-start/null dimensionality analysis | `make figure ID=ch3-wiring-dimension` |
| Fig. 3.1 learned weights | Multi-seed place/place-grid/multi-module weight panels | `make figure ID=ch4-weights` |
| Figs. 3.2-3.4 write/hold/move | Reproducible activity traces with decoded location | `make figure ID=ch4-operations` |
| Fig. 3.5 random-noise robustness | Budget-matched error/drift vs horizon with CIs | `make figure ID=ch4-random-noise` |
| Fig. 3.6 constant perturbation | Recovery/absolute error vs fixed perturbation with CIs | `make figure ID=ch4-fixed-perturbation` |
| Fig. 3.7 staircase/fixed points | Fixed-point density, basin radii, and decoder map | `make figure ID=ch4-basin-structure` |
| Proposed 2D generalization | Implemented 2D multi-module path integration | `make figure ID=ch4-2d-path-integration` |

Final IDs may change when chapters are renumbered, but the mapping must remain machine-readable.

---

## 9. Experiment scheduling and compute control

### 9.1 Staged execution

Do not launch the full matrix immediately.

1. **Unit/tiny stage:** seconds to minutes; synthetic fixtures; every commit.
2. **Smoke stage:** reduced data and iterations; CPU; every pull request.
3. **Pilot stage:** one seed and small subsets; choose stable search ranges.
4. **Validation sweep:** bounded declared search; may run in parallel; test data inaccessible to selection code.
5. **Configuration freeze:** commit resolved configs and hashes.
6. **Confirmatory stage:** all fixed seeds; no tuning.
7. **Robustness stage:** preregistered ablations and failure regimes.
8. **Aggregation/final figures:** read-only with respect to trained models.

### 9.2 Resource estimates to refine after benchmarks

Initial planning budget:

- 10-30 GB for downloaded images, preprocessing caches, and patch indices.
- 10-50 GB for checkpoints and raw logs before archiving/pruning.
- One CUDA GPU with at least 16 GB VRAM for the default color sparse-coding runs; activation checkpointing may be required for long unrolls.
- Several hundred GPU-hours are possible if all model/noise/seed/ablation combinations are run naively. The pilot must measure time and reduce redundant sweeps before approval.
- Attractor experiments should be cheaper but may become large with two-dimensional sweeps; vectorize trajectories and freeze a bounded confirmatory matrix.

Every full run must log wall time, peak CPU/GPU memory, energy estimate if available, and output size. The release README must replace these planning ranges with measured values.

### 9.3 Failure/restart behavior

- Checkpoint training state, optimizer, scheduler, RNG states, sampler position, and config.
- Resume must produce the same result as uninterrupted execution within the reproducibility contract.
- Failed runs remain in raw logs with a failure reason; aggregation excludes them only through explicit validation status.
- Never replace a failed seed with another seed without preserving and reporting the failure.

---

## 10. Tests required before scientific interpretation

### 10.1 Unit tests

- color transforms and exact inverses;
- SNR/sigma corruption calibration;
- patch extraction/reassembly;
- dictionary normalization;
- group-matrix constraints;
- sparse objective components;
- symmetric/antisymmetric projections;
- place/grid encoders and decoders;
- noise sampling moments;
- metric definitions.

### 10.2 Numerical tests

- finite-difference gradients for dictionary, group structure, recurrent weights, and scalar nonlinearity parameters;
- convergence against trusted small convex cases;
- solver objective monotonicity where theoretically expected;
- continuous/discrete dynamics agreement as `delta t` decreases;
- path-integration direction and velocity scaling;
- deterministic replay from saved RNG state.

### 10.3 Statistical tests of the testing code

- bootstrap interval coverage on simulated distributions;
- permutation-test calibration under a known null;
- paired versus unpaired resampling behavior;
- aggregation rejection of duplicated/missing seeds;
- prevention of patch-level pseudoreplication.

### 10.4 Regression tests

- tiny frozen output metrics with tolerances;
- artifact schema snapshots;
- figure/table inventory completeness;
- no unresolved LaTeX references/citations;
- build PDF page count and text-presence checks.

---

## 11. Risks and mitigations

| Risk | Consequence | Mitigation |
|---|---|---|
| Original code cannot be recovered | Exact historical reproduction impossible | Preserve this fact; implement from equations; label as faithful reimplementation, not original rerun |
| DIV2K/BSDS terms restrict redistribution | GitHub users cannot receive bundled data | Download from official sources after explicit terms acceptance; commit manifests only; provide synthetic CI data |
| Color dimensionality makes unrolled training expensive | Excessive VRAM/time | Pilot dictionary size/unroll length; activation checkpointing; mixed precision only after accuracy validation; separate quick/full profiles |
| Learned `Lambda` has scaling/identifiability pathologies | Unstable or uninterpretable groups | Positive parameterization, normalization/regularization, synthetic recovery tests, sensitivity analysis |
| Historical inference equation is mathematically ambiguous | Results differ from 2016 | Maintain named legacy and corrected variants; compare both; document the resolution |
| Modern denoisers dominate sparse models | Misleading “performance” framing | Treat CNN as contextual only; focus group model claims on structured prior and controlled comparisons |
| Place-grid benefit is due to power/weight/decoder confounds | Invalid error-correction conclusion | Equal-budget controls, constrained weights, common decoders, direct basin measurements |
| Grid code improves local precision but causes alias failures | Average MSE hides catastrophic errors | Report full error distribution and catastrophic-error probability across regimes |
| Test-set leakage through configuration selection | Inflated claims | Separate immutable manifests and make selection code unable to read test metrics before freeze |
| Full results vary by hardware | Reproduction disputes | Reference container/hardware, deterministic mode, repeated seed, tolerance and statistical contracts |
| Revised writing outruns results | Unsupported polished narrative | Claim-evidence matrix; write results before abstract/conclusion; generated numerical macros |
| Repository becomes overengineered | Harder reproduction | Prefer local files and Make targets; optional external services; architecture review at every phase |

---

## 12. Milestones and release sequence

1. **M0 - Archive preserved:** Phase 0, tagged archive.
2. **M1 - Reproducible skeleton (`v0.1.0`):** Phases 1-4; CI, math specs, data pipeline.
3. **M2 - Color sparse coding (`v0.2.0`):** Phases 5-6 and relevant statistics/artifacts.
4. **M3 - Attractor/grid code (`v0.3.0`):** Phases 7-9 and relevant statistics/artifacts.
5. **M4 - Complete scientific results (`v0.9.0`):** Phases 10-11; all frozen manifests and artifacts.
6. **M5 - Revised dissertation release candidate (`v1.0.0-rc1`):** Phase 12.
7. **M6 - Independently reproduced public release (`v1.0.0`):** Phases 13-14.

Each milestone must have release notes stating which results are historical, faithfully reimplemented, or modernized.

---

## 13. Final start-to-finish operator runbook

The following is the experience the final repository must support. It is also the final acceptance test.

1. Install Git, `uv`, GNU Make, a supported TeX Live distribution, and optionally Docker/CUDA.
2. Clone the `v1.0.0` tag.
3. Run `make bootstrap`.
4. Run `make test`.
5. Read `data/README.md` and the two dataset terms.
6. Download DIV2K and BSDS500 with explicit acceptance flags.
7. Run `make data-check`; confirm all hashes and split manifests.
8. Run `make reproduce-quick`; confirm smoke metrics and a small PDF build.
9. Select the documented full compute profile and device.
10. Run `make reproduce-ch2`.
11. Run `make reproduce-ch3`.
12. Run `make reproduce-all` to validate, aggregate, and generate final artifacts.
13. Run `make dissertation`.
14. Run `make verify-release`.
15. Compare the produced release manifest and PDF hash with the published reference or use documented numerical tolerances where a cross-platform rebuild cannot be byte-identical.

There must be no step that says “open this figure and adjust it,” “copy this number into TeX,” “download data from an unofficial mirror,” or “ask the author for a missing config.”

---

## 14. Final acceptance checklist

### Preservation

- [ ] Historical source/PDF/figures are immutable and checksummed.
- [ ] Original-code absence or recovery is documented accurately.
- [ ] Historical and revised results cannot be confused.

### Code

- [ ] All scientific code is importable, typed, tested, and configured outside source.
- [ ] CPU quick profile and GPU full profile use the same core implementation.
- [ ] Resume and deterministic replay are tested.
- [ ] No dissertation result depends on a notebook or manual edit.

### Data

- [ ] Official public sources and terms are documented.
- [ ] No restricted data is committed or redistributed.
- [ ] Dataset and split hashes are frozen.
- [ ] Preprocessing fits training data only and is reversible where claimed.

### Chapter 2

- [ ] Toy group recovery is quantitatively validated.
- [ ] Public RGB data replaces the grayscale natural-image experiment.
- [ ] Sparse/group baselines are matched and multi-seed.
- [ ] Color structure is analyzed, not merely displayed.
- [ ] In-domain and external denoising results include uncertainty and failure cases.
- [ ] Topographic/wiring claims use null and stability analyses.

### Chapter 3

- [ ] Original 1D operations are reproduced or discrepancies explained.
- [ ] Place/grid comparisons match neuron, activity, weight, training, and decoder budgets.
- [ ] Noise types and timescales are tested separately.
- [ ] Basin/fixed-point claims are directly measured.
- [ ] Multi-module and 2D experiments are implemented.
- [ ] Aliasing/catastrophic failures are reported.

### Dissertation

- [ ] Mathematics and implementation use the same notation and equations.
- [ ] Literature is current through the declared cutoff date.
- [ ] Every claim maps to evidence.
- [ ] Numerical values, tables, and figures are generated.
- [ ] Abstract and conclusion match the final results.
- [ ] Cross-references/citations resolve and the compiled PDF passes visual QA.
- [ ] Repository version, commit, and DOI appear in the PDF.

### Release

- [ ] Clean-machine independent reproduction is complete.
- [ ] CI is green at the release tag.
- [ ] Licenses, third-party notices, CFF, SBOM, and checksums validate.
- [ ] GitHub release and permanent archive are public.
- [ ] `v1.0.0` documentation contains measured compute/storage requirements.

---

## 15. Primary references for implementation decisions

- [Official DIV2K dataset page](https://data.vision.ee.ethz.ch/cvl/DIV2K/) - RGB data, official split description, and academic-research terms.
- [Official Berkeley Segmentation Dataset and Benchmark](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/segbench/) - external color test data and non-commercial research/education terms.
- [PyTorch reproducibility guidance](https://docs.pytorch.org/docs/stable/notes/randomness.html) - determinism limits, seed controls, and deterministic algorithms.
- [`torch.use_deterministic_algorithms` documentation](https://docs.pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html) - deterministic-operation behavior.
- [uv project locking and syncing](https://docs.astral.sh/uv/concepts/projects/sync/) - committed lockfile and `--locked` workflow.
- [Hydra official documentation](https://hydra.cc/docs/intro/) - hierarchical experiment configuration and command-line overrides.
- [GitHub citation-file documentation](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-citation-files) - repository citation metadata.
- [Zenodo CITATION.cff guidance](https://help.zenodo.org/docs/github/describe-software/citation-file/) - software archive citation metadata.

These links guide implementation; exact software versions, dataset hashes, and terms hashes must be captured in the repository at the time their respective phase is executed.

---

## 16. First action when execution begins

Begin with **Phase 0 only**. Do not scaffold, move historical files, download data, or edit dissertation prose until the current dirty worktree has been explained, preserved, and checksummed. After Gate P0, execute phases in order and stop at every gate if the documented acceptance criteria are not met.
