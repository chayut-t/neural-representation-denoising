# neural-representation-denoising

Modern reproduction and revision of the 2016 PhD dissertation *Neural Representation
Learning with Denoising Autoencoder Framework* (Chayut Thanapirom, UC Berkeley Physics),
organized around two studies:

1. **Group sparse coding** — learning group structure by backpropagating a denoising
   objective through unrolled sparse inference (revised Chapter 3).
2. **Attractor / grid codes** — noise-robust recurrent location codes with restoration and
   velocity transport (revised Chapter 4).

The goal is a public repository that regenerates every quantitative result, table, and figure
in a revised dissertation from documented public data and versioned code. See
`MODERNIZATION_AND_REPRODUCIBILITY_PLAN.md` for the governing, phase-gated spec.

> **Status:** Phases 0–2 complete (repository scaffolding, locked environment, public
> containers, working dissertation build, and CI are in place). Phase 3 (formalize + test the
> mathematics) is next. No experiments have been run yet; the study subcommands are
> placeholders that exit non-zero until their phase implements them.

## Quick start (developer)

Requires [`uv`](https://docs.astral.sh/uv/). Python 3.11 is provisioned by `uv`.

```sh
uv sync --locked --extra image   # install the locked CPU/dev environment
uv run neural-repr-verify system-info   # print the sanitized environment record
uv run pytest                    # run the test suite
```

Reproduction runs on a **public reference environment** — the committed `uv.lock`, a publicly
pullable CUDA container, and generic CUDA hardware (≥16 GB VRAM). No private infrastructure is
required; see the reproducibility boundary in the plan (§2.4).

### Build a container

```sh
docker build -f containers/Dockerfile.cpu  -t neural-repr:cpu  .   # tests / quick profile
docker build -f containers/Dockerfile.cuda -t neural-repr:cuda .   # GPU profile (linux/amd64)
```

### Build the dissertation

Requires TeX Live (LuaLaTeX, latexmk, biber). From `dissertation/`:

```sh
latexmk -r latexmkrc main.tex
```

## Contributing

- Python 3.11 (managed by `uv`); dependencies are pinned in the committed `uv.lock` — edit
  `pyproject.toml` and run `uv lock`, never hand-edit the lockfile.
- Before pushing: `uv run ruff check src tests scripts`, `uv run ruff format src tests`,
  `uv run mypy`, `uv run pytest`. CI runs these plus baseline-hash, lineage, and leak checks.
- **Never edit anything under `legacy/`** (immutable archives) and never commit dataset files
  or infrastructure identifiers (cluster/registry/host/private-path strings); CI enforces both.
- Decisions that change data, equations, baselines, the writing engine, or claims need a record
  under `docs/decisions/`.

## Layout

- `src/neural_repr/` — the Python package (studies, data, evaluation, statistics, plotting,
  provenance, CLI).
- `configs/` — Hydra configuration groups.
- `dissertation/` — the working LaTeX edition (derived from the frozen 2026 baseline).
- `legacy/` — immutable archives: the 2016 dissertation and the frozen 2026 rewrite.
- `docs/` — plan artifacts: inventory, baseline audit, claim-evidence matrix, decision records.

## License

To be finalized before first release; see `docs/decisions/0007-repository-and-text-licenses.md`.
