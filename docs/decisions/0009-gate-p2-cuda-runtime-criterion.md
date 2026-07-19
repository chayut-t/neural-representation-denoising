# 0009 — Gate P2 acceptance criterion: split CUDA build-time vs runtime checks

**Status:** Accepted (2026-07-19) · **Relates to:** plan §7 Phase 2 Gate P2, §2.4, Phase 13;
containers/README.md; `.github/workflows/containers.yml`. Supersedes the original single
"public CUDA image passes import, numerical-smoke, provenance-schema, and canary checks"
criterion by splitting it explicitly.

## Context

The original Gate P2 criterion required that "the public CUDA image passes import,
numerical-smoke, provenance-schema, and synthetic-canary leak-scan tests." The
numerical-smoke check has two forms: a **build-time** form that runs on any (CPU) runner,
and a **runtime** form (`check-image --require-cuda`) that executes a real GPU op and
therefore needs a GPU-capable runner. No public GPU CI runner is wired up yet, so the
runtime form cannot run. Prior wording declared Gate P2 "passed" while calling the missing
runtime evidence "non-blocking" in adjacent prose — an internal contradiction the review
(codex A6) flagged: the original blocking criterion and the non-blocking note coexisted.

## Decision

Split the CUDA image criterion into two named checks with distinct gates:

- **Gate P2 (build-time, blocking now):** the public CUDA image builds without private
  credentials from digest-pinned public bases, and `check-image` (without `--require-cuda`)
  passes: import, closed provenance-schema validation, deterministic CPU numerical-smoke,
  synthetic-canary leak-scan, and the exact python/torch/CUDA **version assertions**
  (`--expect-python 3.11.15 --expect-torch 2.13 --expect-cuda 13.0`). The CPU image passes
  its full `check-image`. These run on the standard amd64 CI runner today.

- **Gate P13 (runtime, blocking there):** the CUDA **runtime** self-check
  (`check-image --require-cuda`, a real GPU op with the same version assertions) must pass on
  the public reference GPU environment. This becomes a **blocking** acceptance criterion of
  Phase 13 (public reference environment / reproduction), and the `cuda-runtime-check` CI job
  (gated on `vars.GPU_RUNNER_LABEL`) is its mechanized form. Until a GPU runner exists it does
  not run; that is acceptable because no GPU experiment is executed before Phase 13.

The runtime GPU check is therefore **not** a silent omission from P2: it is reassigned to the
gate where a GPU is first actually required, with an explicit blocking deadline.

## Alternatives considered

- **Keep Gate P2 open until a GPU runner exists** — rejected: it would block all CPU-only
  Phase 3–5 work (math formalization, data pipeline, faithful reimplementation) on GPU CI
  that is not needed until training, inverting the plan's phase ordering.
- **Leave the wording as-is** — rejected: it retained the original blocking criterion while
  declaring the same missing evidence non-blocking, which is contradictory (the defect A6
  identified).

## Consequences

- Plan Gate P2 and `CLAUDE.md` state that P2's CUDA criterion is the **build-time** check, and
  that the **runtime** GPU self-check is a **blocking Gate P13** criterion (not a dangling
  "pending" note under P2).
- `containers/README.md` and `.github/workflows/containers.yml` already separate the two
  checks; this record makes the gate assignment authoritative.
- No code change is required beyond the wording; the mechanized checks already exist.
