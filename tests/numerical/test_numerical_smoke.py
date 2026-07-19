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
    """Compare the CPU result to the committed reference oracle (tol-regression-tiny).

    Frozen tiny regression oracle (decision 0005): detects drift between commits and
    platforms. Policy (plan §2.4 boundary), enforced honestly per the recorded
    capture platform:

    * If the running platform matches the oracle's ``capture_platform`` exactly, the
      **fingerprint must match bit-for-bit** (exact reproduction).
    * Otherwise only ``value`` is compared, within the documented cross-platform
      relative tolerance; the fingerprint is not required to match.
    """
    import json
    import platform
    from pathlib import Path

    ref = json.loads(
        (
            Path(__file__).resolve().parents[1] / "regression" / "numerical_smoke_reference.json"
        ).read_text()
    )
    result = numerical_smoke(seed=ref["seed"], device=ref["device"])

    cap = ref["capture_platform"]
    import torch

    on_capture_platform = (
        platform.system() == cap["os_system"]
        and platform.machine() == cap["machine"]
        and platform.python_version() == cap["python_version"]
        and torch.__version__ == cap["torch_version"]
    )
    if on_capture_platform:
        assert result["fingerprint"] == ref["fingerprint"], (
            f"on the capture platform the fingerprint must reproduce exactly: "
            f"got {result['fingerprint']}, expected {ref['fingerprint']}"
        )
        assert result["value"] == ref["value"]
    else:
        rel = abs(result["value"] - ref["value"]) / abs(ref["value"])
        tol = ref["cross_platform_relative_tolerance"]
        assert rel <= tol, (
            f"numerical smoke {result['value']} drifted from reference {ref['value']} "
            f"(relative {rel:.2e} > {tol:.0e}) on a non-capture platform"
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
