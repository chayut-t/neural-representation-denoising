"""Data audit report: channel histograms, patch power, whitening + leakage checks.

Phase 4 (plan §7 task 11). Produces a JSON-serializable audit summary from arrays
already in memory (no plotting here — rendered figures are the plotting phase). The
audit is a *quantitative* report a reviewer can diff: per-channel histograms, patch
power statistics, a check that whitened features have near-identity covariance, a
train/test similarity check, and a before/after transform summary.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from neural_repr.data.whitening import WhiteningStats, apply_whitening

Array = NDArray[np.float64]

AUDIT_SCHEMA_VERSION = "1"


def channel_histograms(
    images: Array, *, bins: int = 16, value_range: tuple[float, float] = (0.0, 1.0)
) -> dict[str, object]:
    """Per-channel value histograms over a stack of channel-last images/patches."""
    x = np.asarray(images, dtype=np.float64)
    c = x.shape[-1]
    flat = x.reshape(-1, c)
    edges = np.linspace(value_range[0], value_range[1], bins + 1)
    hists = [np.histogram(flat[:, ch], bins=edges)[0].tolist() for ch in range(c)]
    return {"bin_edges": edges.tolist(), "counts_per_channel": hists}


def patch_power(patches: Array) -> dict[str, float]:
    """Mean/median/percentiles of per-patch power (mean squared value)."""
    x = np.asarray(patches, dtype=np.float64)
    power = np.mean(x**2, axis=tuple(range(1, x.ndim)))
    return {
        "mean": float(power.mean()),
        "median": float(np.median(power)),
        "p05": float(np.percentile(power, 5)),
        "p95": float(np.percentile(power, 95)),
    }


def whitening_covariance_check(features: Array, stats: WhiteningStats) -> dict[str, float]:
    """Max abs deviation of whitened-feature covariance from the identity.

    A correctly fit whitening transform makes the covariance of the *training*
    features it was fit on close to the identity; the residual on held-out features
    is reported too (it is generally larger, which is expected and informative).
    """
    whitened = apply_whitening(features, stats)
    n = whitened.shape[0]
    centered = whitened - whitened.mean(axis=0)
    cov = (centered.T @ centered) / max(n - 1, 1)
    identity = np.eye(cov.shape[0])
    off = cov - identity
    return {
        "max_abs_off_identity": float(np.max(np.abs(off))),
        "mean_abs_off_identity": float(np.mean(np.abs(off))),
    }


def train_test_similarity(
    train_features: Array, test_features: Array, *, max_pairs: int = 512
) -> dict[str, float]:
    """Nearest-neighbor cosine similarity of test features to train features.

    High values would indicate near-duplicate content across the split (a leakage
    smell). Subsamples to ``max_pairs`` rows per side for a bounded, deterministic
    (first-N) computation suitable for an audit.
    """
    train_rows = np.asarray(train_features, dtype=np.float64)[:max_pairs]
    test_rows = np.asarray(test_features, dtype=np.float64)[:max_pairs]

    def _unit(a: Array) -> Array:
        norm = np.linalg.norm(a, axis=1, keepdims=True)
        unit: Array = a / np.maximum(norm, 1e-12)
        return unit

    sims = _unit(test_rows) @ _unit(train_rows).T
    nearest = sims.max(axis=1)
    return {
        "max_nearest_cosine": float(nearest.max()),
        "mean_nearest_cosine": float(nearest.mean()),
    }


def before_after_summary(before: Array, after: Array) -> dict[str, object]:
    """Basic moments before/after a transform, for the audit's before/after panel."""

    def _moments(a: Array) -> dict[str, float]:
        a = np.asarray(a, dtype=np.float64)
        return {
            "mean": float(a.mean()),
            "std": float(a.std()),
            "min": float(a.min()),
            "max": float(a.max()),
        }

    return {"before": _moments(before), "after": _moments(after)}


def build_audit_report(
    *,
    rgb_patches: Array,
    train_features: Array,
    test_features: Array,
    whitening: WhiteningStats,
    dataset: str,
    manifest_sha256: str | None = None,
) -> dict[str, object]:
    """Assemble the full quantitative data audit report (plan task 11).

    Combines every audit component into one JSON-serializable object: per-channel
    histograms of the RGB patches, patch-power stats, the whitening covariance check
    on the fitted (train) features, a train/test similarity (leakage) check, and a
    before/after-whitening summary. Tagged with the dataset and the input-manifest
    hash so a written report is traceable to what it audited.
    """
    whitened_train = apply_whitening(train_features, whitening)
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "dataset": dataset,
        "input_manifest_sha256": manifest_sha256,
        "n_train_features": int(np.asarray(train_features).shape[0]),
        "n_test_features": int(np.asarray(test_features).shape[0]),
        "channel_histograms": channel_histograms(rgb_patches),
        "patch_power": patch_power(rgb_patches),
        "whitening_covariance": whitening_covariance_check(train_features, whitening),
        "train_test_similarity": train_test_similarity(train_features, test_features),
        "whitening_before_after": before_after_summary(train_features, whitened_train),
    }


def write_audit_report(report: dict[str, object], path: Path, *, overwrite: bool = False) -> None:
    """Write an assembled audit report to JSON, collision-failing by default (§0.3)."""
    from neural_repr.data.io_safe import atomic_write_text

    atomic_write_text(
        path, json.dumps(report, indent=2, sort_keys=True) + "\n", overwrite=overwrite
    )
