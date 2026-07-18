"""Deterministic seeding and a numerical smoke check.

``numerical_smoke`` runs a tiny, fully deterministic torch computation and returns
a stable fingerprint of the result. It exercises the same seeding/determinism path
the studies will rely on, so the public reference container can prove numerics work
and reproduce bit-for-bit under a fixed seed (plan §6.3 debug-mode spirit; the
"numerical-smoke" Gate P2 clause).
"""

from __future__ import annotations

import hashlib
import random

import numpy as np


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and (if available) PyTorch CPU/CUDA RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def numerical_smoke(seed: int = 0) -> dict[str, object]:
    """Run a deterministic tiny computation; return a stable result fingerprint.

    Deterministic on a fixed platform: same seed -> same fingerprint. Across
    platforms the fingerprint may differ (documented reproducibility boundary,
    plan §2.4), but the routine must always run without error and be repeatable
    within a process.
    """
    import torch

    seed_everything(seed)
    with torch.no_grad():
        a = torch.randn(64, 64, dtype=torch.float64)
        # A stable, non-trivial reduction: symmetric matrix eigenvalue sum ~ trace.
        sym = a @ a.T
        value = float(torch.linalg.eigvalsh(sym).sum().item())
    # Round to absorb last-bit noise, then hash for a compact stable token.
    token = hashlib.sha256(f"{value:.6f}".encode()).hexdigest()[:16]
    return {
        "seed": seed,
        "value": round(value, 6),
        "fingerprint": token,
        "torch_version": torch.__version__,
    }
