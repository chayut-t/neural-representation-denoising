# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

A modern reproduction and revision of the 2016 PhD dissertation *Neural Representation
Learning with Denoising Autoencoder Framework* (Chayut Thanapirom, UC Berkeley Physics).
The goal is a public repo that can recreate every quantitative result, table, and figure
in a substantially revised dissertation from documented public data and versioned code.

**Current state (important):** No implementation has begun. The repository today contains
only two things:
1. `MODERNIZATION_AND_REPRODUCIBILITY_PLAN.md` — the governing execution spec.
2. `legacy/dissertation-2016/source/` — the original LaTeX sources, PDFs (`thesis.pdf`
   locally compiled; `qt0hm6p6s5.pdf` the eScholarship submission), figures, and
   references, plus the original build script `doit.sh`.

None of the modern Python package, `Makefile`, configs, tests, or CI described in the plan
exist yet. Do not assume any command, module, or directory from the plan is present —
verify first. The plan describes the *target* state and the *order* in which to build it.

## The plan is authoritative — read it before acting

`MODERNIZATION_AND_REPRODUCIBILITY_PLAN.md` is a strict, phase-gated specification, not a
loose roadmap. Before doing substantive work, read the relevant phase. Key structural facts:

- **Work proceeds in gated phases (0–14).** Each phase ends with a gate (P0–P14); do not
  start a dependent phase until its predecessor's acceptance criteria are met.
- **Phase 0 comes first, always.** Until the current worktree is explained, preserved, and
  checksummed (a `legacy/.../MANIFEST.sha256`, a legacy README, a `docs/result-inventory.csv`,
  and an archival tag), do NOT scaffold the package, move historical files, download data,
  or edit dissertation prose. See plan §16.
- **Default technical decisions are fixed** (plan §3): Python 3.11, PyTorch (Theano is
  replaced), `uv` + `pyproject.toml` + committed `uv.lock`, Hydra 1.3 structured configs,
  Typer CLIs, pytest, Ruff/mypy, LuaLaTeX + `latexmk` + `biber`/`biblatex`. Change any of
  these only via a decision record in `docs/decisions/`.

## Hard rules that must not be relaxed (plan §2)

These override convenience and are the most common way to go wrong here:

- **Preserve history.** Never overwrite or "clean up" historical files. Use history-preserving
  git moves. Keep the eScholarship PDF separate from the locally compiled PDF.
- **Separate three kinds of evidence** for every result/table/claim: *historical* (copied
  from 2016, not rerun), *faithful reimplementation* (re-derived from the equations with
  deliberate compatibility settings), and *modernized* (corrected/extended/new data). Never
  call a modernized run an exact reproduction of a historical result.
- **No manual transcription into the dissertation.** Prose may interpret results but must not
  contain hand-typed numbers. Tables are generated `.tex` fragments; figures are generated
  from result files; headline numbers come from a generated LaTeX macro file. No manual
  graphics editing (no Photoshop/Illustrator retouching of plots).
- **Don't tune on the test set.** Config selection code must not read test metrics before the
  confirmatory freeze. Split manifests are immutable.
- **Data is not committed.** DIV2K and BSDS500 are downloaded from official sources after
  explicit terms acceptance; only manifests/hashes are committed. A tiny synthetic RGB
  fixture (code-licensed) is the only in-repo image data, and is for tests only.

## The two scientific studies (architecture at the science level)

The code and revised dissertation are organized around *claims* (plan §4), not historical
file names. There are two largely independent studies:

- **Group sparse coding** (legacy `chap2.tex`; revised Chapter 3). Learn the group structure
  of group sparse coding by backpropagating a denoising objective through unrolled sparse
  inference. Modernized to public *color* natural images (DIV2K/BSDS500) with two explicit,
  never-mixed tracks: `color_whitened` (opponent-color, whitened representation analysis)
  and `color_rgb` (RGB denoising, PSNR/SSIM). Planned code under `src/neural_repr/sparse_coding/`.
- **Attractor networks / grid codes** (legacy `chap3.tex`; revised Chapter 4). Train recurrent
  dynamics with state/weight noise so noisy location-code states are restored (denoising in
  time), with antisymmetric generators transporting states along the manifold (path
  integration). Reproduce the original 1D place/grid comparison fairly (budget-matched), then
  extend to multiple modules and 2D. Planned code under `src/neural_repr/attractor/`.

Both share the target package layout in plan §5 (`data/`, `evaluation/`, `statistics/`,
`plotting/`, `provenance/`, `cli/`, `common/`) and the reproducibility/artifact contract in §6
(every run records config, seeds, git commit, env, dataset hashes, determinism mode, and
artifact hashes). The historical→modern result map is in plan §8.

## Commands

### Legacy dissertation build (works today)

The only runnable build currently in the repo. From `legacy/dissertation-2016/source/`:

```sh
sh doit.sh          # pdflatex thesis && biber thesis && pdflatex thesis
```

This is the *historical* build (pdflatex + biber, `ucbthesis` class). Treat `legacy/` as
immutable archive — build it to inspect, but do not modernize files in place there. The
revised dissertation will use a separate LuaLaTeX + `latexmk` build under `dissertation/`.

### Planned reproduction commands (do NOT exist yet — create per the plan)

The plan defines these `make` targets as the eventual stable interface (plan §1). They are
the definition of done, not currently available:

```sh
make bootstrap        # install locked CPU/dev env, verify tools
make data-check       # validate local datasets and licenses
make test             # unit, integration, numerical, schema tests
make reproduce-quick  # small deterministic CPU reproduction on fixtures
make reproduce-ch2 / reproduce-ch3 / reproduce-all
make dissertation     # build revised PDF from generated artifacts
make verify-release
```

`reproduce-quick` must run on a normal CPU and exercise the same code paths as the full
GPU profile with reduced data/iterations.
