"""Additive white Gaussian noise (AWGN) by fixed sigma or by target SNR.

Phase 4 (plan §7 task 9; decision 0003). Two corruption modes, kept distinct
because the two color tracks use different ones:

* **fixed sigma** (``color_rgb`` denoising track): ``x_tilde = x + N(0, sigma^2)``
  with ``sigma in {15, 25, 50}/255`` — the standard visible-image benchmark. The
  document must state whether values are gamma-encoded sRGB or linear light; that is
  the caller's responsibility (recorded in the run config), not hidden here.
* **target SNR** (``color_whitened`` representation track): pick ``sigma`` so the
  signal-to-noise ratio equals a requested dB level — the thesis-comparable
  ``0/3/6 dB`` setting. SNR is defined on the signal's variance about its mean:
  ``SNR_dB = 10 log10(var(signal) / sigma^2)``.

All draws take an explicit ``numpy.random.Generator`` so corruption is reproducible
(``(dataset, image, seed, ...)`` seeding lives in :mod:`neural_repr.data.patches`).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def add_awgn_sigma(x: Array, sigma: float, rng: np.random.Generator) -> Array:
    """Add AWGN with a fixed standard deviation ``sigma`` (in the data's units)."""
    if not sigma >= 0.0:  # rejects negatives AND NaN (NaN >= 0 is False)
        raise ValueError(f"sigma must be a non-negative real number, got {sigma!r}")
    x = np.asarray(x, dtype=np.float64)
    return x + sigma * rng.standard_normal(x.shape)


def sigma_for_snr(signal: Array, snr_db: float) -> float:
    """Noise sigma that yields ``snr_db`` for ``signal`` (variance about its mean).

    ``SNR_dB = 10 log10(var(signal) / sigma^2)`` ->
    ``sigma = sqrt(var(signal) / 10^(SNR_dB/10))``.

    **Degenerate-signal policy (explicit):** a target SNR is *undefined* for a constant
    signal (variance 0 — no sigma yields a finite input SNR), so this raises
    ``ValueError`` rather than silently returning 0 (which would then measure as +inf,
    not the requested SNR). Callers that want "no noise" should pass ``sigma=0`` to
    :func:`add_awgn_sigma` directly.
    """
    signal = np.asarray(signal, dtype=np.float64)
    var = float(np.var(signal))
    if var == 0.0:
        raise ValueError(
            "target SNR is undefined for a constant signal (variance 0); "
            "use add_awgn_sigma(sigma=0) if no noise is intended"
        )
    return float(np.sqrt(var / (10.0 ** (snr_db / 10.0))))


def add_awgn_snr(x: Array, snr_db: float, rng: np.random.Generator) -> tuple[Array, float]:
    """Add AWGN calibrated to ``snr_db``; return ``(corrupted, sigma_used)``.

    ``sigma`` is computed from ``x``'s own variance (:func:`sigma_for_snr`), so the
    realized input SNR matches the request in expectation.
    """
    sigma = sigma_for_snr(x, snr_db)
    return add_awgn_sigma(x, sigma, rng), sigma


def measured_snr_db(clean: Array, noisy: Array) -> float:
    """Empirical input SNR (dB) of a corrupted signal vs its clean source.

    ``10 log10(var(clean) / mean((noisy - clean)^2))``. Used by the audit report and
    tests to confirm :func:`add_awgn_snr` hits its target within tolerance.
    """
    clean = np.asarray(clean, dtype=np.float64)
    noisy = np.asarray(noisy, dtype=np.float64)
    noise_power = float(np.mean((noisy - clean) ** 2))
    var = float(np.var(clean))
    if noise_power == 0.0:
        return float("inf")
    if var == 0.0:
        return float("-inf")
    return 10.0 * float(np.log10(var / noise_power))
