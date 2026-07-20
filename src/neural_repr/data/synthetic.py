"""Deterministic synthetic RGB fixture (CI-only, licensed with the code).

Phase 4 (plan §7 tasks 7, 12; decision 0001). Generates a tiny repository-owned
color dataset — geometric edges, oriented gratings, smooth color fields, and
correlated color features — used **only** for tests and pipeline exercises, never as
scientific evidence. It contains no restricted third-party imagery, so it can be
generated in CI and its manifest committed.

Every image is a deterministic function of ``(seed, index, size)`` — no global RNG
state — so the generated **pixels** are identical on any platform (a Gate P4
reproducibility property). That pixel identity, not PNG-encoded-byte identity, is the
fixture's contract: PNG (zlib) encoding is not byte-identical across platforms or
library builds even for identical pixels, so the committed manifest pins each image's
``content_sha256`` (a digest over decoded pixels) and leaves the file-byte ``sha256``
empty for this regenerable fixture (see ``neural_repr.data.manifests``). Images are
returned as ``uint8`` channel-last RGB in ``[0, 255]`` so they save as ordinary PNGs.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

RGBImage = NDArray[np.uint8]

# Kinds cycle so a small fixture spans the intended structure (plan task 12).
_KINDS = ("edges", "grating", "smooth_color", "correlated_color")


def _coords(size: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    axis = np.linspace(0.0, 1.0, size, dtype=np.float64)
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    return yy, xx


def _edges(rng: np.random.Generator, size: int) -> NDArray[np.float64]:
    """A hard oriented edge between two random flat colors."""
    yy, xx = _coords(size)
    theta = rng.uniform(0.0, np.pi)
    offset = rng.uniform(0.3, 0.7)
    proj = np.cos(theta) * xx + np.sin(theta) * yy
    mask = (proj > offset)[:, :, None]
    c0 = rng.uniform(0.0, 1.0, size=3)
    c1 = rng.uniform(0.0, 1.0, size=3)
    return np.where(mask, c1, c0)


def _grating(rng: np.random.Generator, size: int) -> NDArray[np.float64]:
    """An oriented sinusoidal grating with a random per-channel phase."""
    yy, xx = _coords(size)
    theta = rng.uniform(0.0, np.pi)
    freq = rng.uniform(3.0, 8.0)
    proj = np.cos(theta) * xx + np.sin(theta) * yy
    phases = rng.uniform(0.0, 2.0 * np.pi, size=3)
    out = np.empty((size, size, 3), dtype=np.float64)
    for c in range(3):
        out[:, :, c] = 0.5 + 0.5 * np.sin(2.0 * np.pi * freq * proj + phases[c])
    return out


def _smooth_color(rng: np.random.Generator, size: int) -> NDArray[np.float64]:
    """A smooth low-frequency color field (independent gentle gradients per channel)."""
    yy, xx = _coords(size)
    out = np.empty((size, size, 3), dtype=np.float64)
    for c in range(3):
        a, b, d = rng.uniform(-1.0, 1.0, size=3)
        field = a * xx + b * yy + d * xx * yy
        field = (field - field.min()) / (np.ptp(field) + 1e-8)
        out[:, :, c] = field
    return out


def _correlated_color(rng: np.random.Generator, size: int) -> NDArray[np.float64]:
    """A grating whose channels are strongly correlated (shared luminance + tint).

    Exercises the color-dependence structure the group study targets: the three
    channels are a common spatial pattern scaled by a fixed per-channel tint plus a
    small independent perturbation, so channel covariance is high but not degenerate.
    """
    yy, xx = _coords(size)
    theta = rng.uniform(0.0, np.pi)
    freq = rng.uniform(2.0, 6.0)
    proj = np.cos(theta) * xx + np.sin(theta) * yy
    base = 0.5 + 0.4 * np.sin(2.0 * np.pi * freq * proj)
    tint = rng.uniform(0.5, 1.0, size=3)
    out = np.empty((size, size, 3), dtype=np.float64)
    for c in range(3):
        perturb = 0.05 * rng.standard_normal((size, size))
        out[:, :, c] = np.clip(base * tint[c] + perturb, 0.0, 1.0)
    return out


_GENERATORS = {
    "edges": _edges,
    "grating": _grating,
    "smooth_color": _smooth_color,
    "correlated_color": _correlated_color,
}


def synthetic_image(index: int, *, size: int = 32, seed: int = 0) -> RGBImage:
    """Generate one deterministic synthetic RGB image as ``uint8`` ``(size, size, 3)``.

    The image kind cycles with ``index`` so a fixture of a few images spans all four
    structure types. Fully determined by ``(seed, index, size)``.
    """
    if size < 4:
        raise ValueError("synthetic image size must be >= 4")
    kind = _KINDS[index % len(_KINDS)]
    rng = np.random.default_rng((seed, index))
    field = _GENERATORS[kind](rng, size)
    return np.clip(field * 255.0, 0.0, 255.0).round().astype(np.uint8)


def synthetic_dataset(n: int, *, size: int = 32, seed: int = 0) -> list[RGBImage]:
    """Generate ``n`` deterministic synthetic RGB images."""
    if n < 1:
        raise ValueError("n must be >= 1")
    return [synthetic_image(i, size=size, seed=seed) for i in range(n)]
