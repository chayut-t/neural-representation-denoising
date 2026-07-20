"""Whitening (fit on training data only) and normalization.

Phase 4 (plan §7 tasks 9-10; decision 0003 ``color_whitened``). The whitening
transform is **fit on training patches only** — the fitted statistics never see
validation or test data (a Gate P4 requirement and the leakage rule of §3.2). The
fit is captured in a small, serializable :class:`WhiteningStats` artifact tagged
with the hash of the input manifest it was fit on (plan task 10), so a stored
transform is traceable to exactly the data that produced it.

We use **ZCA** whitening: ``W = V diag(1/sqrt(lambda + eps)) V^T`` from the
eigendecomposition of the covariance, which is symmetric and stays closest to the
identity (patches look like whitened versions of themselves rather than rotated
into an arbitrary basis). It is invertible: ``W_inv = V diag(sqrt(lambda + eps)) V^T``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]

DEFAULT_WHITEN_EPS: float = 1e-5


@dataclass(frozen=True)
class WhiteningStats:
    """Fitted ZCA whitening statistics (serializable, provenance-tagged).

    ``mean`` is the per-feature mean removed before whitening; ``whiten`` and
    ``unwhiten`` are the ZCA matrix and its inverse. ``eps`` is the regularizer
    added to eigenvalues. ``n_samples`` / ``input_manifest_sha256`` record what the
    fit was computed on (plan task 10) — the latter is the hash of the manifest of
    the *training* images used, so the artifact cannot be silently reused for a
    different input.
    """

    mean: Array
    whiten: Array
    unwhiten: Array
    eps: float
    n_samples: int
    input_manifest_sha256: str | None = None

    @property
    def n_features(self) -> int:
        return int(self.mean.shape[0])


def fit_whitening(
    train_features: Array,
    *,
    eps: float = DEFAULT_WHITEN_EPS,
    input_manifest_sha256: str | None = None,
) -> WhiteningStats:
    """Fit ZCA whitening on **training** feature vectors of shape ``(N, F)``.

    ``train_features`` must be a 2-D array whose rows are samples (e.g. flattened,
    DC-removed patches). Raises if it is empty or not 2-D so a bad fit fails loudly
    rather than producing a silent degenerate transform.
    """
    x = np.asarray(train_features, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"train_features must be 2-D (N, F), got shape {x.shape}")
    n, _ = x.shape
    if n < 2:
        raise ValueError(f"need at least 2 samples to fit whitening, got {n}")
    if not eps > 0.0:  # rejects non-positive AND NaN
        raise ValueError(f"eps must be a positive real number, got {eps!r}")

    mean = x.mean(axis=0)
    centered = x - mean
    cov = (centered.T @ centered) / (n - 1)
    # Symmetric eigendecomposition (cov is symmetric PSD).
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.clip(eigvals, 0.0, None)
    inv_sqrt = 1.0 / np.sqrt(eigvals + eps)
    sqrt = np.sqrt(eigvals + eps)
    whiten = (eigvecs * inv_sqrt) @ eigvecs.T
    unwhiten = (eigvecs * sqrt) @ eigvecs.T
    return WhiteningStats(
        mean=mean,
        whiten=whiten,
        unwhiten=unwhiten,
        eps=eps,
        n_samples=n,
        input_manifest_sha256=input_manifest_sha256,
    )


def apply_whitening(features: Array, stats: WhiteningStats) -> Array:
    """Whiten feature vectors ``(N, F)`` with a fitted :class:`WhiteningStats`."""
    x = np.asarray(features, dtype=np.float64)
    if x.shape[-1] != stats.n_features:
        raise ValueError(f"feature dim {x.shape[-1]} != fitted {stats.n_features}")
    return (x - stats.mean) @ stats.whiten.T


def invert_whitening(whitened: Array, stats: WhiteningStats) -> Array:
    """Inverse of :func:`apply_whitening` (reconstructs the pre-whitening features)."""
    x = np.asarray(whitened, dtype=np.float64)
    if x.shape[-1] != stats.n_features:
        raise ValueError(f"feature dim {x.shape[-1]} != fitted {stats.n_features}")
    return x @ stats.unwhiten.T + stats.mean


def global_normalize(features: Array, *, scale: float | None = None) -> tuple[Array, float]:
    """Divide every value by one global scalar: the **RMS over all entries** by default.

    Exact statistic (frozen convention): ``scale = sqrt(mean(x**2))`` over *all* array
    entries — the uncentered root-mean-square (equivalently ``std`` with ``ddof=0``
    **only when the data is already zero-mean**). In the ``color_whitened`` pipeline
    this runs on DC-removed opponent features, which are zero-mean *per patch*; the
    global RMS is therefore the intended "training-set pixel standard deviation" of the
    baseline (App B) under that centering, and we compute it as RMS rather than a
    separately-centered std so the reverse (``x * scale``) is exact. Returns
    ``(normalized, scale)``; a dataset-wide scale, not per-patch.
    """
    x = np.asarray(features, dtype=np.float64)
    if scale is None:
        scale = float(np.sqrt(np.mean(x**2)))
    if not scale > 0.0:  # rejects non-positive AND NaN
        raise ValueError(
            f"global normalization scale must be a positive real number, got {scale!r}"
        )
    return x / scale, scale


def per_patch_normalize(patches: Array, *, eps: float = 1e-8) -> tuple[Array, Array]:
    """Normalize each patch by its own RMS over the (pixel, channel) axes.

    ``patches`` has shape ``(N, P, C)``. Returns ``(normalized, scales)`` with
    ``scales`` shape ``(N, 1, 1)`` so the operation is per-patch reversible. ``eps``
    floors the scale so a flat patch does not divide by zero.
    """
    x = np.asarray(patches, dtype=np.float64)
    if x.ndim != 3:
        raise ValueError(f"per_patch_normalize expects (N, P, C), got shape {x.shape}")
    scales = np.sqrt(np.mean(x**2, axis=(-2, -1), keepdims=True))
    scales = np.maximum(scales, eps)
    return x / scales, scales
