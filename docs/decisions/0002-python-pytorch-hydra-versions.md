# 0002 — Python / PyTorch / Hydra versions

**Status:** Accepted (2026-07-18) · **Relates to:** plan §3.1; ratifies the default with one
compute-environment caveat.

## Context

The plan (§3.1) fixes Python 3.11, PyTorch (exact version locked at scaffold time), `uv` +
`pyproject.toml` + committed `uv.lock`, Hydra 1.3 structured configs, Typer CLIs, pytest,
Ruff/mypy. Theano (the 2016 framework) is replaced by PyTorch.

## Decision

- Adopt the plan §3.1 stack **as the canonical/authoring environment**: Python 3.11, PyTorch
  (pin exact version at scaffold in Phase 2), Hydra 1.3, `uv` + committed `uv.lock`, Typer,
  pytest, Ruff, mypy.
- The **reference GPU execution image** may ship a different, pinned Python/PyTorch (e.g. the
  compute base image is Python 3.10 + a vendor PyTorch build). This is allowed provided: (a) the
  numerical contract (determinism modes, tolerances, multi-seed conclusions in
  [0005](0005-confirmatory-seeds-and-statistics.md)) holds, and (b) the exact runtime versions
  are recorded per run (plan §6.1). The mismatch is a portability boundary, not a divergence
  from the authored lock — see [0008](0008-offline-package-install.md).
- Exact PyTorch version, CUDA/cuDNN, and the reference container digest are pinned during Phase 2
  scaffolding and recorded in the run artifact schema.

## Alternatives considered

- Force Python 3.11 everywhere including the GPU pod — rejected for v1 planning: the available
  compute base image is 3.10 and we cannot push a custom image; forcing 3.11 would mean building
  a full offline toolchain now. Revisit if a 3.11 base becomes available.
- Keep Theano for fidelity — rejected: unmaintained; PyTorch gives GPU, unrolled inference, BPTT,
  and deterministic controls.

## Consequences

CPU `reproduce-quick` and the GPU full profile must exercise the same core implementation
(plan Gate P2 / §1). Any Python-version-sensitive behavior is caught by the numerical/regression
tests (plan §10).
