# CLAUDE.md

Guidance for Claude Code working in this repository.

## What this is

A modern reproduction and revision of the 2016 PhD dissertation *Neural Representation
Learning with Denoising Autoencoder Framework* (Chayut Thanapirom, UC Berkeley Physics).
The goal: a public repo that can regenerate every quantitative result, table, and figure
in a revised dissertation from documented public data and versioned code.

## Current state

- `MODERNIZATION_AND_REPRODUCIBILITY_PLAN.md` — the detailed execution spec (14 phases).
- `legacy/dissertation-2016/` — preserved original LaTeX sources, PDFs, figures (archived,
  tagged `archive/dissertation-2016`).
- `legacy/rewrite-2026/` — a self-contained 2026 narrative rewrite (58-page PDF + sources).
  This is the narrative/structure baseline the modern edition builds from.
- `docs/result-inventory.csv` — inventory of historical results.

No modern Python package, `Makefile`, configs, tests, or CI exist yet. Phase 0 (preserve
2016 work) is done; Phase 1 (freeze scope + inventory the 2026 rewrite) is next. Don't
assume any command or module from the plan is present — verify first.

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

Legacy 2016 build (the only runnable build today), from `legacy/dissertation-2016/source/`:

```sh
sh doit.sh          # pdflatex thesis && biber thesis && pdflatex thesis
```

Everything else (`make bootstrap`, `make test`, `make reproduce-*`, `make dissertation`) is
planned, not yet built. See the plan for the target interface.
