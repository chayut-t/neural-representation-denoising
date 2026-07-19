"""Numerical-smoke tests (plan §6.3 debug-mode spirit; Gate P2 numerical-smoke clause).

Deterministic within a process on a fixed platform: same seed -> same fingerprint.
Cross-platform values may differ (documented reproducibility boundary, §2.4), so we
assert repeatability and finiteness rather than a hard-coded value.
"""

from __future__ import annotations

import math

from neural_repr.common import numerical_smoke, seed_everything


def test_numerical_smoke_runs_and_is_finite() -> None:
    result = numerical_smoke(seed=0)
    assert set(result) >= {"seed", "value", "fingerprint", "torch_version"}
    assert math.isfinite(result["value"])
    assert len(result["fingerprint"]) == 16


def test_numerical_smoke_repeatable_same_seed() -> None:
    a = numerical_smoke(seed=0)
    b = numerical_smoke(seed=0)
    assert a["fingerprint"] == b["fingerprint"]
    assert a["value"] == b["value"]


def test_numerical_smoke_matches_committed_reference() -> None:
    """Compare the CPU result to a committed reference value within tolerance.

    This is the frozen tiny regression oracle (tol-regression-tiny; decision 0005):
    unlike the self-comparison above, it detects drift *between commits/platforms*.
    Same platform reproduces the fingerprint exactly; other platforms must match
    the value within the documented relative tolerance (plan §2.4 boundary).
    """
    import json
    from pathlib import Path

    ref = json.loads(
        (
            Path(__file__).resolve().parents[1] / "regression" / "numerical_smoke_reference.json"
        ).read_text()
    )
    result = numerical_smoke(seed=ref["seed"], device=ref["device"])
    rel = abs(result["value"] - ref["value"]) / abs(ref["value"])
    assert rel <= ref["relative_tolerance"], (
        f"numerical smoke {result['value']} drifted from reference {ref['value']} "
        f"(relative {rel:.2e} > {ref['relative_tolerance']:.0e})"
    )


def test_numerical_smoke_varies_with_seed() -> None:
    a = numerical_smoke(seed=0)
    b = numerical_smoke(seed=1)
    assert a["value"] != b["value"]


def test_seed_everything_makes_numpy_deterministic() -> None:
    import numpy as np

    seed_everything(123)
    first = np.random.rand(5).tolist()
    seed_everything(123)
    second = np.random.rand(5).tolist()
    assert first == second
