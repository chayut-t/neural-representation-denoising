# 0006 — LaTeX engine for the working edition

**Status:** Accepted (2026-07-18) · **Relates to:** plan §3.1, §7 Phase 1 (task 4), §12D;
baseline builds with Tectonic + XeTeX.

## Context

The frozen 2026 baseline builds with **Tectonic** (XeTeX engine) via `source/doit.sh` and uses
`natbib`/`plainnat` for bibliography. The plan's default writing engine (§3.1) is **LuaLaTeX +
`latexmk` + `biber`/`biblatex`**. Phase 1 requires deciding whether the new `dissertation/` edition
retains the baseline route or migrates, including a baseline visual-fidelity test.

## Decision

**Migrate the working `dissertation/` edition to LuaLaTeX + `latexmk` + `biber`/`biblatex`** (the
plan §3.1 default), subject to a **baseline migration fidelity gate**:

- Before any scientific edit, reproduce the baseline's title, chapter order, labels, evidence
  boundary, and historical-result disclaimers under the new engine, and pass an automated
  text/structure check plus a representative rendered-page review (plan Gate P2).
- Document font/layout changes introduced by the engine switch (Tectonic/XeTeX → LuaLaTeX) in this
  record's follow-up or the migration notes.
- Bibliography migrates `natbib`/`plainnat` → `biblatex`/`biber`; citation keys are preserved (the
  30 keys in `docs/baseline-audit-2026.md`).
- The **frozen baseline** under `legacy/rewrite-2026/` keeps building with Tectonic and is never
  edited; migration happens only in the new `dissertation/` tree.

## Alternatives considered

- **Retain Tectonic/XeTeX** to minimize migration risk — reasonable, and lower-effort, but diverges
  from the plan's standardized reproducible-build default (`latexmk` treats undefined refs, missing
  citations, overfull boxes, duplicate labels as CI failures more ergonomically). Chose migration
  for the long-term CI/reproducibility contract; the fidelity gate bounds the risk.

## Consequences

Adds a one-time migration + visual-fidelity task in Phase 2/12. If the fidelity gate reveals
unacceptable layout drift, this record is revised to retain Tectonic. The baseline audit
(`docs/baseline-audit-2026.md`) is the regression oracle for the migration.
