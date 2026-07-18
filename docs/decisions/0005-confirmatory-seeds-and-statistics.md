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

## Alternatives considered

- 3 seeds to save compute — rejected: too few for a paired CI on the central C4-H2 claim.
- p-values without effect sizes — rejected: the plan requires effect sizes and raw data.

## Consequences

Statistical scripts must reproduce every reported value from frozen manifests (plan Gate P10). The
central grid claim (C4-H2) is only asserted if it survives a preregistered budget-matched, paired,
multi-seed comparison (plan Gate P8).
