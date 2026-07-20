# CLAUDE.md

Guidance for Claude Code working in this repository.

## What this is

A modern reproduction and revision of the 2016 PhD dissertation *Neural Representation
Learning with Denoising Autoencoder Framework* (Chayut Thanapirom, UC Berkeley Physics).
The goal: a public repo that can regenerate every quantitative result, table, and figure
in a revised dissertation from documented public data and versioned code.

## Current state

**Phases 0–3 complete (Gates P0–P2 independently verified; P3 passed). Phase 4 code is complete
and green but Gate P4 is OPEN**: its two dataset-materialization artifacts (real DIV2K/BSDS500
manifests + a provenance-bound audit) require authorized download and are deferred to Phase 5
entry by decision 0010; Phase 5 is blocked on Gate P4. Gate P2 covers the CUDA image's
*build-time* checks; the *runtime* GPU self-check is a blocking Gate P13 criterion (decision
0009). Phase 4 added the data pipeline (`src/neural_repr/data/`): registry, registry-aware on-disk
validation, deterministic synthetic fixture, color/whitening/corruption/patch transforms, and a
config-driven train-only fit/apply pipeline — all CPU, reproducible; dataset bytes are never
committed (only manifests/hashes; decision 0001).

- `MODERNIZATION_AND_REPRODUCIBILITY_PLAN.md` — the detailed execution spec (14 phases).
- `legacy/dissertation-2016/` (tag `archive/dissertation-2016`) and `legacy/rewrite-2026/`
  (tag `archive/rewrite-2026-baseline`) — immutable archives with `MANIFEST.sha256` each.
  The 2026 rewrite is the narrative/structure baseline the modern edition builds from.
- `docs/` — plan artifacts: `result-inventory.csv`, `rewrite-2026-inventory.csv`,
  `baseline-audit-2026.md`, `claim-evidence-matrix.md`, `tolerances.md`, `decisions/`.
- `src/neural_repr/` — the Python package (Python 3.11, committed `uv.lock`) with six Typer
  CLIs (`neural-repr-{data,train,eval,aggregate,figure,verify}`). Only `verify` does real
  work so far; the other commands are placeholders that **exit non-zero** until their phase.
- `dissertation/` — the working LuaLaTeX edition, derived from the frozen 2026 source
  (`FILE_MAP.csv` records source-hash lineage); builds via `scripts/release/build_dissertation.py`.
- `containers/` — public digest-pinned CPU/CUDA reference images. `.github/workflows/` — CI
  (lint/type/test, baseline-hash, lineage, layered leak prevention, LaTeX build).

No experiment/science code or reproduced results exist yet. Don't assume a module or command
beyond the above is present — verify first.

## Agent-to-agent communication

`workspace.local/` is the local workspace directory for this project. Use it for temporary
working material, notes, and coordination artifacts that should remain local. The directory is
git-ignored; do not commit its contents or treat anything in it as a reproducible project output.

Use `workspace.local/a2a/` specifically for local Markdown messages between agents.

- Codex messages must be named
  `workspace.local/a2a/codex-<date>-<descriptive-name>.md`.
- Claude Code messages must be named
  `workspace.local/a2a/claude-<date>-<descriptive-name>.md`.

Use ISO dates (`YYYY-MM-DD`) and a concise descriptive filename component.

Session-handoff files live in two more git-ignored `workspace.local/` subdirectories, written by
the `/create-context-and-worklog` skill:

- `workspace.local/worklogs/worklog-<YYYYMMDD>.md` — backward-looking record of a session's work.
- `workspace.local/contexts/claude-context-<YYYYMMDD>.md` — forward-looking resume brief for the
  next session; read the most recent one first when resuming.

## The two scientific studies

1. **Group sparse coding** (revised Chapter 3): learn group structure by backpropagating a
   denoising objective through unrolled sparse inference; modernize to public color images
   (DIV2K/BSDS500).
2. **Attractor networks / grid codes** (revised Chapter 4): train recurrent dynamics with
   noise so location-code states are restored (denoising in time); reproduce the 1D
   place/grid comparison, then extend to multiple modules and 2D.

## Things worth keeping in mind

- **Preserve history.** `legacy/` is an immutable archive — inspect it, don't edit in place.
  The revised dissertation goes in a new `dissertation/` tree.
- **Distinguish evidence.** Keep *historical* (copied from 2016), *faithful reimplementation*,
  and *modernized* results clearly separate; don't call a modernized run an exact reproduction.
- **Generate, don't transcribe.** Dissertation numbers/tables/figures should come from result
  files, not be hand-typed.
- **Data isn't committed.** DIV2K/BSDS500 are downloaded from official sources; only
  manifests/hashes are committed.
- The plan (`§3`) sets defaults (Python 3.11, PyTorch, uv, Hydra, Typer, LuaLaTeX). Treat
  them as sensible starting points; the plan has the full detail when you need it.

## Running experiments

GPU experiments run on remote compute, not this laptop. Before launching or driving any GPU
run, read `docs/infrastructure.local.md` — it holds the target cluster/queue, launch and
`kubectl exec` commands, and gotchas. That file is git-ignored (infra stays out of the public
repo), so keep cluster/queue/pod/bucket details there, never in committed files or prose.
CPU-only work (quick reproduction, tests, preprocessing, figures) runs locally.

## Commands

Developer workflow (requires `uv`; Python 3.11 is provisioned by uv):

```sh
uv sync --locked --extra image           # install the locked CPU/dev environment
uv run pytest -q                         # tests
uv run ruff check src tests scripts && uv run mypy   # lint + types
uv run neural-repr-verify system-info    # sanitized environment record (§6.1)
uv run python scripts/release/verify_baselines.py    # both legacy manifests must match
uv run python scripts/release/scan_public_leaks.py   # infrastructure-leak scan
uv run python scripts/release/check_lineage.py       # 2026 -> dissertation lineage
uv run python scripts/release/build_dissertation.py  # non-overwriting PDF build -> builds/dissertation/<id>/
```

The `make` targets (`make bootstrap`, `make test`, `make reproduce-*`, `make dissertation`) are
the eventual stable interface and are not built yet. The legacy 2016 build still works from
`legacy/dissertation-2016/source/` via `sh doit.sh` (pdflatex + biber; historical only).
