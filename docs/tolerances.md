# Tolerance Registry

Numerical tolerances with **owners** and **per-phase deadlines** (decision 0005; plan §2.4,
§3.1, §6.3). "Within the stated tolerances" is a normative dependency throughout the plan, so
each tolerance must be committed here before the work that relies on it. Values are **not**
invented ahead of the implementations/benchmarks that justify them; the row is created now with
its deadline, and the value is filled in by the deadline.

**Rule:** no confirmatory configuration freezes (plan Phases 6–9) until its primary-metric,
artifact, and cross-environment tolerances are committed here.

| id | scope | metric / comparison | value | status | owner | due (phase) |
|---|---|---|---|---|---|---|
| tol-env-bootstrap | environment | `uv sync --locked` resolves to the locked set; `system-info` schema validates | exact match | committed | maintainer | Phase 2 |
| tol-regression-tiny | regression | tiny fixed-seed CPU smoke output vs committed reference (`tests/regression/numerical_smoke_reference.json`) | value 4066.733107; **capture-platform match → exact fingerprint** (`2473e4da4006f299`); other platforms → value within rel 1e-3 (interim guard; captured on dev macOS arm64, re-anchored to the Linux CPU reference container at Phase 13) | committed (interim) | maintainer | Phase 2 (re-anchor Phase 13) |
| tol-gradient-fd | numerical | analytic vs central finite-difference gradients (sparse + recurrent), float64 CPU | rel 1e-6, abs 1e-9 (BPTT/unrolled: rel 1e-5, abs 1e-7) | committed | maintainer | Phase 3 |
| tol-solver-convergence | numerical | `smooth_gradient` sparse inference vs independent scipy L-BFGS-B solve of the smoothed objective | rel 1e-3, abs 1e-4 | committed | maintainer | Phase 3 |
| tol-faithful-reimpl | faithful | legacy-compatible reimplementation vs documented historical behavior | TBD | pending | maintainer | Phases 3–5 |
| tol-ch3-denoising | confirmatory | ΔSNR / ΔPSNR / ΔSSIM primary metrics, per-family | TBD | pending | maintainer | before Phase 6 freeze |
| tol-ch3-xenv | cross-env | Ch3 metrics: public-reference vs private source run | TBD | pending | maintainer | before Phase 6 freeze |
| tol-ch4-error | confirmatory | location error / basin measures primary metrics | TBD | pending | maintainer | before Phases 7–9 freeze |
| tol-ch4-xenv | cross-env | Ch4 metrics: public-reference vs private source run | TBD | pending | maintainer | before Phases 7–9 freeze |
| tol-artifact-hash | artifact | generated figure/table byte or structured tolerance | TBD | pending | maintainer | Phase 11 |
| tol-robustness | statistical | robustness / conclusion criteria (frozen before final aggregation) | TBD | pending | maintainer | Phase 10 |

Statuses: `committed` (value fixed here), `pending` (row reserved, value due by the deadline).
Cross-environment (`*-xenv`) tolerances implement the public/private equivalence rule (§2.4):
the public reference must regenerate each reported value within these bounds.
