# 0005 — Confirmatory seeds and statistical tests

**Status:** Accepted (2026-07-18) · **Relates to:** plan §6.3, §10; claims C3-H3/H4, C4-H2/H3.
Freezes the confirmatory statistical protocol *before* full runs (plan §9.1).

## Context

The baseline reports single-run summaries with no seed-level uncertainty and calls nothing
statistically significant. The modern edition must replace visual/single-run arguments with
reproducible inference.

## Decision

- **Seeds:** ≥5 independent training seeds for expensive confirmatory models; more (≥20) for cheap
  toy/simulation experiments (e.g. C3-H1 recovery). The `benchmark` determinism mode requires ≥5.
- **Paired design:** identical corruption/trajectory seeds across paired model comparisons.
- **Resampling unit:** images or independently generated trajectories are the primary unit — never
  individual overlapping patches or time steps (avoids pseudoreplication).
- **Reporting:** mean, sd, median (where useful), and 95% CIs. **Paired bootstrap** CIs for
  denoising and trajectory errors. **Permutation tests** for group-feature organization and graph
  null comparisons. Report effect sizes and raw per-image/per-seed data, not only p-values.
- **Multiple comparisons:** correct within declared families, or clearly label exploratory p-values.
- **Determinism (plan §6.3):** `debug` (deterministic required, fail on nondeterministic ops,
  CPU/reference CUDA), `research` (deterministic requested, warnings recorded), `benchmark`
  (performance allowed, ≥5 seeds). For the reference release: rerun one full seed twice on the
  reference machine and require byte-identical metrics or a documented tolerance; across other
  platforms, tolerance + same qualitative conclusions.
- **No tuning on test:** config-selection code cannot read test metrics before the confirmatory
  freeze; split manifests immutable.

## Tolerance registry (owners + deadlines)

"Within the stated tolerances" is a normative dependency across plan §2.4, §3.1, Phase 2, and
Phase 13, so tolerances cannot stay indefinitely as "a documented tolerance." We do **not** invent
numbers before implementations/benchmarks exist; instead a committed tolerance registry
(`docs/tolerances.md` or `results/tolerances/`, created in Phase 2) fixes each tolerance by a
deadline, and **no confirmatory configuration freezes until its relevant cross-environment
tolerance is committed**:

| When due | Tolerances to fix |
|---|---|
| Phase 2 | environment/bootstrap and tiny deterministic regression tolerances |
| Phases 3–5 | numerical-gradient, solver, and faithful-reimplementation tolerances |
| Before each Phase 6–9 confirmatory freeze | primary-metric, artifact, and **cross-environment** (public-vs-private, §2.4) tolerances for that experiment family |
| Phase 10 | robustness / statistical-conclusion criteria, frozen before final aggregation |

Each registry entry has an owner and a phase deadline. Phase 10 is too late for primary
confirmatory tolerances; those are frozen at each family's confirmatory freeze.

## Alternatives considered

- 3 seeds to save compute — rejected: too few for a paired CI on the central C4-H2 claim.
- p-values without effect sizes — rejected: the plan requires effect sizes and raw data.

## Consequences

Statistical scripts must reproduce every reported value from frozen manifests (plan Gate P10). The
central grid claim (C4-H2) is only asserted if it survives a preregistered budget-matched, paired,
multi-seed comparison (plan Gate P8).
