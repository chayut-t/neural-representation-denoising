# 0007 — Repository and text licenses

**Status:** Provisional (2026-07-18) — final choice due before the first public release
(plan §3.5). · **Relates to:** plan §3.5, §7 Phase 14.

## Context

The plan requires license decisions before v1.0.0: code license (BSD-3-Clause or MIT), new
dissertation text and repository-authored figures (CC BY 4.0, subject to confirming rights in the
historical material), preservation of the historical dissertation's existing copyright, and
third-party data governed only by original dataset terms.

## Decision (provisional)

- **Code:** MIT (leaning) or BSD-3-Clause — decide before first public release; both permissive and
  compatible with the intended open reproduction.
- **New dissertation text + repository-authored figures:** CC BY 4.0, **subject to confirming
  rights** in any retained historical material.
- **Historical dissertation (`legacy/dissertation-2016/`) and the 2026 rewrite
  (`legacy/rewrite-2026/`):** preserve existing copyright notices; do not relicense archived trees.
- **Third-party data (DIV2K, BSDS500):** governed solely by original dataset terms; never covered by
  the repository license; never redistributed.
- **Dependencies:** generate an SBOM and check license compatibility before v1.0.0.

## Why provisional

The code license (MIT vs BSD-3-Clause) and the rights status of figures carried over from the 2016
material need a deliberate final confirmation. This record freezes the *framework* (permissive code,
CC BY 4.0 new text, preserved historical copyright, data under its own terms) so Phase 1 can close;
the specific code license string is chosen at Phase 14.

## Consequences

`LICENSE`, `LICENSE-TEXT`, and a third-party notice are added in Phase 2/14. No archived file is
relicensed. Revisit and promote to Accepted with the concrete code-license choice before v1.0.0.
