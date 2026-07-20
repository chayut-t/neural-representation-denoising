"""Serialization of fitted preprocessing statistics as versioned artifacts.

Phase 4 (plan §7 task 10). A fitted transform (whitening, global scale) is stored as
a small JSON artifact tagged with the SHA-256 of the **input manifest** it was fit
on, so the artifact is traceable to exactly the data that produced it and cannot be
silently reused for a different input. Arrays are stored as nested lists (JSON) so
the artifact is diff-inspectable and platform-neutral; a schema version guards the
format.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np

from neural_repr.data.io_safe import atomic_write_text
from neural_repr.data.whitening import WhiteningStats

STATS_SCHEMA_VERSION = "1"


def whitening_to_dict(stats: WhiteningStats) -> dict[str, object]:
    """Serialize :class:`WhiteningStats` to a JSON-safe dict."""
    return {
        "schema_version": STATS_SCHEMA_VERSION,
        "kind": "zca_whitening",
        "eps": stats.eps,
        "n_samples": stats.n_samples,
        "n_features": stats.n_features,
        "input_manifest_sha256": stats.input_manifest_sha256,
        "mean": stats.mean.tolist(),
        "whiten": stats.whiten.tolist(),
        "unwhiten": stats.unwhiten.tolist(),
    }


_STATS_ALLOWED_KEYS = frozenset(
    {
        "schema_version",
        "kind",
        "eps",
        "n_samples",
        "n_features",
        "input_manifest_sha256",
        "mean",
        "whiten",
        "unwhiten",
    }
)
_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


def whitening_from_dict(payload: dict[str, object]) -> WhiteningStats:
    """Deserialize + **validate** a whitening-stats artifact (closed schema; R2 finding 9).

    Rejects: unknown keys; wrong schema version/kind; a malformed (non-``sha256:``+64hex)
    manifest digest; non-positive/non-finite ``eps``; non-positive ``n_samples``; an
    ``n_features`` that disagrees with the arrays; wrong array ranks/shapes; NaN/Inf
    values; a non-symmetric ``whiten``; and matrices that are not inverses
    (``whiten @ unwhiten ~= I``). A committed fit artifact must carry a real manifest
    digest.
    """
    unknown = set(payload) - _STATS_ALLOWED_KEYS
    if unknown:
        raise ValueError(f"unknown stats field(s): {sorted(unknown)}")
    if payload.get("schema_version") != STATS_SCHEMA_VERSION:
        raise ValueError(f"unsupported stats schema_version {payload.get('schema_version')!r}")
    if payload.get("kind") != "zca_whitening":
        raise ValueError(f"expected kind 'zca_whitening', got {payload.get('kind')!r}")

    manifest = payload.get("input_manifest_sha256")
    if manifest is not None and (not isinstance(manifest, str) or not _SHA256_RE.match(manifest)):
        raise ValueError("input_manifest_sha256 must be 'sha256:'+64 lowercase hex or null")

    eps = payload["eps"]
    n_samples = payload["n_samples"]
    n_features = payload["n_features"]
    if (
        isinstance(eps, bool)
        or not isinstance(eps, int | float)
        or not math.isfinite(eps)
        or eps <= 0.0
    ):
        raise ValueError(f"eps must be a positive finite number, got {eps!r}")
    if isinstance(n_samples, bool) or not isinstance(n_samples, int) or n_samples < 2:
        raise ValueError(f"n_samples must be an int >= 2, got {n_samples!r}")
    if isinstance(n_features, bool) or not isinstance(n_features, int) or n_features < 1:
        raise ValueError(f"n_features must be a positive int, got {n_features!r}")

    mean = np.asarray(payload["mean"], dtype=np.float64)
    whiten = np.asarray(payload["whiten"], dtype=np.float64)
    unwhiten = np.asarray(payload["unwhiten"], dtype=np.float64)
    if mean.shape != (n_features,):
        raise ValueError(f"mean shape {mean.shape} != ({n_features},)")
    if whiten.shape != (n_features, n_features) or unwhiten.shape != (n_features, n_features):
        raise ValueError("whiten/unwhiten must be (n_features, n_features)")
    for name, arr in (("mean", mean), ("whiten", whiten), ("unwhiten", unwhiten)):
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} contains non-finite values")
    if not np.allclose(whiten, whiten.T, atol=1e-8):
        raise ValueError("ZCA whiten matrix must be symmetric")
    if not np.allclose(whiten @ unwhiten, np.eye(n_features), atol=1e-6):
        raise ValueError("whiten @ unwhiten is not the identity (not inverses)")

    return WhiteningStats(
        mean=mean,
        whiten=whiten,
        unwhiten=unwhiten,
        eps=float(eps),
        n_samples=int(n_samples),
        input_manifest_sha256=manifest,
    )


def save_whitening(stats: WhiteningStats, path: Path, *, overwrite: bool = False) -> None:
    """Save fitted whitening stats, collision-failing by default (plan §0.3 no-overwrite)."""
    atomic_write_text(
        path,
        json.dumps(whitening_to_dict(stats), indent=2, sort_keys=True) + "\n",
        overwrite=overwrite,
    )


def load_whitening(path: Path) -> WhiteningStats:
    return whitening_from_dict(json.loads(path.read_text()))
