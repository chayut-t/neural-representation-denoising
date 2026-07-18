# 0008 — Offline package install on the compute environment

**Status:** Accepted (2026-07-18) · **Relates to:** plan §3.1 (uv default), §6.1, §9.
Records a divergence from the `uv`/PyPI default forced by the GPU compute environment.

## Context

The plan's default (§3.1) is `uv` + committed `uv.lock`, which normally resolves against PyPI. The
GPU compute environment used for the experiments has **no public internet** (PyPI does not resolve)
and **no ability to push a custom container image**. Packages must be installed onto an existing
base image at pod-creation time. Details of the environment itself are kept out of the public repo
(see the git-ignored `docs/infrastructure.local.md`).

## Decision

- **Authoring / local:** keep the plan default — dependencies are declared in `pyproject.toml` and
  locked in a committed `uv.lock`. The committed repo remains a standard `uv` project; local dev and
  CPU `reproduce-quick` use it directly.
- **Compute environment:** materialize the same locked set without PyPI, in this order of
  preference:
  1. the environment's sanctioned internal package index (an authenticated mirror), when available;
  2. otherwise an **offline wheelhouse** built on a networked machine for the target
     platform/Python, transferred **directly** to the environment's persistent storage (no external
     object-store staging), and installed with `pip --no-index`.
- **No external object storage** is used to stage packages, code, or data (explicit constraint).
- Every run records the actual resolved versions, base image identity, and install method
  (plan §6.1) so the offline install is auditable and reproducible.

## Alternatives considered

- Force `uv sync` against PyPI on the pod — impossible (no internet).
- Build and push a custom prebuilt image — not permitted (no image-push access).
- Rely on whatever the base image ships and don't pin — rejected: breaks the reproducibility
  contract; we still pin via `uv.lock` and reconcile on the pod.

## Consequences

A Phase 2 task adds the wheelhouse/mirror install path and documents it (in the private infra note,
not the public docs). The public reproducibility story still describes only a generic CUDA GPU and a
standard `uv`-based environment; the offline mechanism is an internal portability detail. Concrete
operational commands live in `docs/infrastructure.local.md` (git-ignored).
