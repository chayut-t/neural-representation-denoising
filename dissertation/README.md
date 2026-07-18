# Working dissertation edition

This is the **working** dissertation, derived from the frozen 2026 baseline
(`legacy/rewrite-2026/`, tag `archive/rewrite-2026-baseline`). The baseline is immutable; all
editing happens here. See `docs/decisions/0006-latex-engine.md`.

## Provenance and lineage

- `FILE_MAP.csv` maps every file here to its 2026 source and records both SHA-256 hashes.
  - Chapters, abstract, appendix, and bibliography are **byte-identical copies** of the frozen
    baseline (same hashes) — content is not yet edited.
  - `main.tex` and `preamble/*` are **translated** from `thesis.tex`: the build engine moved
    from pdfLaTeX + `natbib` to **LuaLaTeX + latexmk + biblatex/biber** (decision 0006).
- The regression oracle for migration fidelity is `docs/baseline-audit-2026.md`.

## Build

Requires a TeX Live distribution with LuaLaTeX, latexmk, and biber. From this directory:

```sh
latexmk -r latexmkrc main.tex     # builds main.pdf in place (local iteration)
latexmk -r latexmkrc -c           # clean auxiliary files
```

The reproducible pipeline writes PDFs to `builds/dissertation/<build-id>/` (never overwriting a
prior build); `main.pdf` built in place here is for local iteration and is git-ignored.

## Migration fidelity note (baseline vs working edition)

First LuaLaTeX build (2026-07-18): builds cleanly with **no undefined references or citations**
and **no unresolved `[?]` citation marks**; title, committee, all five chapters, appendices, and
the full bibliography are present. Total length is **57 pages vs the baseline's 58** — a benign
1-page reflow from the pdfLaTeX→LuaLaTeX font metrics and the `natbib`→`biblatex` bibliography,
not dropped content (verified against the audit oracle). No scientific content has changed; the
empirical figures/tables are still the historical baseline assets pending Phase 6/11 regeneration
into `generated/`.

## Layout

- `main.tex` — driver (translated).
- `preamble/` — `packages.tex`, `style.tex`, `macros.tex` (translated; visual intent preserved).
- `frontmatter/`, `chapters/`, `appendices/` — content (byte-identical copies for now).
- `bibliography/references.bib` — bibliography (byte-identical copy).
- `generated/` — machine-generated figures/tables/macros land here in later phases.
- `styles/` — reserved for custom class/style files.
