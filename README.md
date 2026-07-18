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

> **Status:** early scaffolding (Phase 2). No experiments have been run yet. Commands below
> are the intended stable interface; most subcommands are placeholders until later phases.

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

## Layout

- `src/neural_repr/` — the Python package (studies, data, evaluation, statistics, plotting,
  provenance, CLI).
- `configs/` — Hydra configuration groups.
- `dissertation/` — the working LaTeX edition (derived from the frozen 2026 baseline).
- `legacy/` — immutable archives: the 2016 dissertation and the frozen 2026 rewrite.
- `docs/` — plan artifacts: inventory, baseline audit, claim-evidence matrix, decision records.

## License

To be finalized before first release; see `docs/decisions/0007-repository-and-text-licenses.md`.
