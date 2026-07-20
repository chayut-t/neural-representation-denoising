# 0010 — Defer real-dataset manifests and provenance-bound audit to Phase 5 entry

**Status:** Accepted (2026-07-19) · **Relates to:** plan §7 Phase 4 tasks 5 & 11, Phase 5;
decision 0001. Amends the Phase 4 completion boundary; recorded after the round-2 review
correctly found Gate P4 was declared passed before its dataset-materialization artifacts existed.

## Context

Two Phase 4 tasks cannot be completed in this environment without **authorized access** to the
official DIV2K / BSDS500 sources:

- **Task 5** — committed split manifests `data/manifests/div2k.csv` and `data/manifests/bsds500.csv`
  with per-file SHA-256. These require downloading the real image bytes (to hash them). Decision
  0001 forbids committing/mirroring the bytes; only the manifests would be committed, but the
  manifests can only be produced *from* an authorized download.
- **Task 11** — a data audit over those real manifests, including before/after **sample images**.
  A meaningful audit (channel statistics, whitening covariance with real rank/sample handling,
  train/test similarity) must run on the actual scientific data, not the synthetic fixture.

Everything in Phase 4 that does not require the real bytes is implemented and verified on the
committed synthetic fixture: the registry, immutable terms/archive records, registry-aware on-disk
validation, the color/whitening/corruption/patch pipeline with inverse round-trips, the
config-driven train-only fit/apply pipeline with provenance-bound stats, no-overwrite writers, and
the audit *machinery* (assembler + CLI) exercised on synthetic data.

## Decision

- **Gate P4 stays OPEN** until the two deferred artifacts exist. The earlier "passed" note is
  withdrawn. Phase 4's non-data-dependent work is complete and independently verifiable now.
- **The deferred artifacts are produced at the start of Phase 5**, immediately after the operator
  performs the authorized DIV2K/BSDS500 download (the `accept-terms` → download → `verify-archive`
  → build-manifest flow already implemented). Specifically, before any Phase 5 scientific run:
  1. build and commit `data/manifests/{div2k,bsds500}.csv` from the verified local bytes;
  2. run the provenance-bound audit over those manifests (carrying each manifest's SHA-256),
     emitting the required before/after sample images and covariance/statistics with committed
     acceptance thresholds;
  3. then, and only then, mark Gate P4 passed via an independent clean-checkout verification.
- **Phase 5 is blocked on Gate P4.** Phase 5 (reimplement + validate the sparse-coding study on
  controlled data) may not begin its confirmatory work until the above completes. Method-level
  correctness work that uses only the synthetic fixture (unit tests of the inference/training loop)
  may proceed in parallel, but no result that feeds a claim may use un-audited data.

## Alternatives considered

- **Fabricate placeholder DIV2K/BSDS manifests** — rejected: a manifest with invented hashes is
  worse than none (it would falsely assert provenance). Decision 0001's "fail loudly on changed
  bytes" rule depends on hashes being real.
- **Mark Gate P4 passed on the synthetic fixture alone** — rejected: that is the exact overclaim
  the round-2 review flagged. The gate names the scientific datasets, not a fixture.
- **Download DIV2K/BSDS here** — not possible: authorized official-source access is required and
  bytes must not be mirrored/committed (decision 0001).

## Consequences

The plan's Gate P4 note is rewritten to OPEN with this deferral. A Phase 5 entry checklist gains
the three materialization steps above. The audit command and manifest builder already exist and are
tested on synthetic data, so materialization at Phase 5 entry is mechanical once the bytes are
present. This record is the "formal decision that defers those exact tasks and states which later
gate blocks on them" that the review required.
