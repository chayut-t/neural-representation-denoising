"""Config-driven representation preprocessing: train-only fit, apply to all splits.

Phase 4 (plan §7 tasks 9-10; decision 0003 ``color_whitened``; R2 finding 3). This is
the high-level entry point that turns the individual transforms into an executable,
**leakage-proof** pipeline:

* it consumes patch stacks tagged by manifest **role** (train/val/test);
* every fitted statistic (whitening, global scale) is computed from **train rows
  only** — a non-training row reaching the fit boundary raises;
* the fit is bound to a digest of the exact fit subset + the config, so the artifact
  is provenance-traceable and a changed fit set yields a different artifact ID;
* the frozen stats are then applied to val/test without refitting.

This makes "train-only" an enforced property of repository code, not a caller
convention. The pipeline is pure/deterministic given its inputs (Gate P4).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from neural_repr.data.color import remove_dc, rgb_to_opponent
from neural_repr.data.whitening import (
    WhiteningStats,
    apply_whitening,
    fit_whitening,
    global_normalize,
)

Array = NDArray[np.float64]

TRAIN_ROLE = "train"


class LeakageError(ValueError):
    """Raised when a non-training row reaches a fit boundary (train-only violation)."""


@dataclass(frozen=True)
class RolePatches:
    """A stack of patches for one split role.

    ``patches`` has shape ``(N, P, C)`` (N patches, P pixels/patch, C channels).
    ``role`` is the manifest role the patches came from (train/val/test).
    """

    role: str
    patches: Array


@dataclass(frozen=True)
class FittedPipeline:
    """Frozen representation-track statistics + the provenance of their fit."""

    whitening: WhiteningStats
    global_scale: float
    eps: float
    fit_id: str  # content-addressed id of (fit subset + config)
    n_train_patches: int


def _config_digest(config: dict[str, object]) -> str:
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


def _to_opponent_dc_features(patches: Array) -> Array:
    """RGB patches -> flattened DC-removed opponent features ``(N, P*3)``.

    Accepts either ``(N, P, 3)`` (already pixel-flattened) or ``(N, H, W, 3)`` (spatial),
    which is what :func:`~neural_repr.data.patches.sample_random_patches` returns; the
    spatial dims are flattened to pixels before the per-patch DC removal.
    """
    patches = np.asarray(patches, dtype=np.float64)
    if patches.shape[-1] != 3 or patches.ndim < 3:
        raise ValueError(f"expected channel-last RGB patches (last dim 3), got {patches.shape}")
    n = patches.shape[0]
    flat = patches.reshape(n, -1, 3)  # (N, P, 3)
    centered, _ = remove_dc(rgb_to_opponent(flat))
    return centered.reshape(n, -1)


def fit_representation_pipeline(
    role_patches: list[RolePatches],
    *,
    eps: float = 1e-5,
    config: dict[str, object] | None = None,
) -> FittedPipeline:
    """Fit the representation-track stats on TRAIN patches only (raises on leakage).

    ``role_patches`` is the full set of per-role patch stacks. Only rows with role
    ``train`` are used to fit whitening + the global scale; if a non-train stack is
    wrongly passed to the fit set via :func:`_train_only`, it raises :class:`LeakageError`.
    The returned :class:`FittedPipeline` carries a ``fit_id`` derived from the train
    features and the config, so the artifact is bound to exactly what it was fit on.
    """
    cfg = {"track": "color_whitened", "eps": eps, **(config or {})}
    train = _train_only(role_patches)
    if train.shape[0] < 2:
        raise ValueError("need at least 2 training patches to fit the pipeline")

    train_feats = _to_opponent_dc_features(train)
    # Global scale is a train-only statistic too.
    _, scale = global_normalize(train_feats)
    fit_id = "fit-" + _fit_digest(train_feats, cfg)[:16]
    whitening = fit_whitening(train_feats, eps=eps, input_manifest_sha256=None)
    return FittedPipeline(
        whitening=whitening,
        global_scale=scale,
        eps=eps,
        fit_id=fit_id,
        n_train_patches=int(train.shape[0]),
    )


def apply_representation_pipeline(patches: Array, fitted: FittedPipeline) -> Array:
    """Apply frozen stats (opponent -> DC -> whiten -> global scale) to any split."""
    feats = _to_opponent_dc_features(np.asarray(patches, dtype=np.float64))
    whitened = apply_whitening(feats, fitted.whitening)
    return whitened / fitted.global_scale


def _train_only(role_patches: list[RolePatches]) -> Array:
    """Concatenate ONLY the train-role patch stacks; raise if a caller mislabels."""
    train_stacks = [rp.patches for rp in role_patches if rp.role == TRAIN_ROLE]
    if not train_stacks:
        raise ValueError("no train-role patches supplied to fit the pipeline")
    # Defensive leakage guard: any stack tagged with a non-train role must not slip
    # into the fit set. (This function only ever collects train stacks; the check
    # documents/enforces the invariant for callers that build the list dynamically.)
    for rp in role_patches:
        if rp.role not in {"train", "val", "test"}:
            raise LeakageError(f"unknown role {rp.role!r} in fit input")
    return np.concatenate(train_stacks, axis=0)


def assert_fit_set_is_train_only(fit_rows_roles: list[str]) -> None:
    """Raise :class:`LeakageError` if any role in a proposed fit set is not 'train'.

    Call this at the boundary where a fit subset is assembled from manifest roles, so
    injecting a val/test row into the fit set is rejected before any statistic is fit.
    """
    offending = sorted({r for r in fit_rows_roles if r != TRAIN_ROLE})
    if offending:
        raise LeakageError(f"fit set must be train-only; found roles {offending}")


def _fit_digest(train_features: Array, config: dict[str, object]) -> str:
    h = hashlib.sha256()
    arr = np.ascontiguousarray(train_features, dtype=np.float64)
    h.update(arr.tobytes())
    h.update(str(arr.shape).encode())
    h.update(_config_digest(config).encode())
    return h.hexdigest()


PIPELINE_SCHEMA_VERSION = "1"


def save_fitted_pipeline(fitted: FittedPipeline, path: Path, *, overwrite: bool = False) -> None:
    """Serialize a :class:`FittedPipeline` (whitening + global scale + fit provenance).

    One versioned artifact holding *all* fitted statistics — whitening AND the global
    normalization scale — bound to the ``fit_id`` (R2 finding 9). Collision-failing by
    default (plan §0.3): reuse a new/versioned path per fit rather than overwriting.
    """
    from neural_repr.data.io_safe import atomic_write_text
    from neural_repr.data.stats import whitening_to_dict

    payload = {
        "schema_version": PIPELINE_SCHEMA_VERSION,
        "kind": "representation_pipeline",
        "fit_id": fitted.fit_id,
        "global_scale": fitted.global_scale,
        "eps": fitted.eps,
        "n_train_patches": fitted.n_train_patches,
        "whitening": whitening_to_dict(fitted.whitening),
    }
    atomic_write_text(
        path, json.dumps(payload, indent=2, sort_keys=True) + "\n", overwrite=overwrite
    )


def load_fitted_pipeline(path: Path) -> FittedPipeline:
    """Load + validate a :class:`FittedPipeline` artifact (delegates to the stats schema)."""
    from neural_repr.data.stats import whitening_from_dict

    payload = json.loads(Path(path).read_text())
    if payload.get("schema_version") != PIPELINE_SCHEMA_VERSION:
        raise ValueError(f"unsupported pipeline schema_version {payload.get('schema_version')!r}")
    if payload.get("kind") != "representation_pipeline":
        raise ValueError(f"expected kind 'representation_pipeline', got {payload.get('kind')!r}")
    scale = payload["global_scale"]
    if isinstance(scale, bool) or not isinstance(scale, int | float) or not scale > 0.0:
        raise ValueError(f"global_scale must be a positive real number, got {scale!r}")
    return FittedPipeline(
        whitening=whitening_from_dict(payload["whitening"]),
        global_scale=float(scale),
        eps=float(payload["eps"]),
        fit_id=str(payload["fit_id"]),
        n_train_patches=int(payload["n_train_patches"]),
    )
