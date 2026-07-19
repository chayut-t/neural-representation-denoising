"""Tests for the corrected Gabor model (3A task 7; b26-corr-gabor).

The essential correction: the Gaussian *envelope* must decay away from its center
(the 2016 form had a positive exponent and grew). We assert envelope decay and
envelope peak-at-center, and separately that the *full* Gabor need not peak at the
center (quadrature phase puts a zero there) while staying bounded by |amplitude|.
"""

from __future__ import annotations

import pytest
import torch

from neural_repr.sparse_coding import GaborParams, gabor


def _grid(n: int = 41, extent: float = 10.0) -> tuple[torch.Tensor, torch.Tensor]:
    axis = torch.linspace(-extent, extent, n, dtype=torch.float64)
    return torch.meshgrid(axis, axis, indexing="xy")


def test_envelope_peaks_at_center() -> None:
    # No carrier oscillation (k=0, phi=0) isolates the Gaussian envelope.
    params = GaborParams(
        amplitude=2.0, t_x=1.0, t_y=-2.0, theta=0.5, sigma_x=1.5, sigma_y=2.5, k=0.0, phi=0.0
    )
    x, y = _grid()
    g = gabor(x, y, params)
    peak = g.abs().max()
    assert float(peak) == pytest.approx(params.amplitude, rel=1e-3)
    # The peak sits at the center (t_x, t_y).
    center = gabor(torch.tensor(params.t_x), torch.tensor(params.t_y), params)
    assert float(center.abs()) == pytest.approx(params.amplitude, rel=1e-6)


def test_envelope_decays_away_from_center() -> None:
    params = GaborParams(
        amplitude=1.0, t_x=0.0, t_y=0.0, theta=0.0, sigma_x=1.0, sigma_y=1.0, k=0.0, phi=0.0
    )
    near = gabor(torch.tensor(0.5), torch.tensor(0.0), params)
    far = gabor(torch.tensor(5.0), torch.tensor(0.0), params)
    farther = gabor(torch.tensor(9.0), torch.tensor(0.0), params)
    assert float(near.abs()) > float(far.abs()) > float(farther.abs())
    assert float(farther.abs()) < 1e-6  # decayed, not grown


def test_envelope_finite_far_away() -> None:
    params = GaborParams(
        amplitude=1.0, t_x=0.0, t_y=0.0, theta=0.3, sigma_x=2.0, sigma_y=1.0, k=1.0, phi=0.2
    )
    x, y = _grid(n=51, extent=100.0)
    g = gabor(x, y, params)
    assert torch.isfinite(g).all()
    assert float(g.abs().max()) <= params.amplitude + 1e-9


def test_rotation_orients_envelope() -> None:
    """Anisotropic envelope rotates with theta: the long axis follows the rotation."""
    params = GaborParams(
        amplitude=1.0, t_x=0.0, t_y=0.0, theta=0.0, sigma_x=4.0, sigma_y=1.0, k=0.0, phi=0.0
    )
    # Along x (long axis) the value stays higher than along y at the same radius.
    along_x = gabor(torch.tensor(2.0), torch.tensor(0.0), params)
    along_y = gabor(torch.tensor(0.0), torch.tensor(2.0), params)
    assert float(along_x.abs()) > float(along_y.abs())


def test_quadrature_phase_zero_at_center_larger_nearby() -> None:
    """With cos(phi)=0 the full Gabor is 0 at the center and larger off-center (B10).

    Demonstrates the corrected docstring claim: only the envelope peaks at the
    center; the full function need not. Here phi=pi/2 so cos(phi)=0.
    """
    params = GaborParams(
        amplitude=1.0,
        t_x=0.0,
        t_y=0.0,
        theta=0.0,
        sigma_x=2.0,
        sigma_y=2.0,
        k=2.0,
        phi=torch.pi / 2,
    )
    center = float(
        gabor(
            torch.tensor(0.0, dtype=torch.float64), torch.tensor(0.0, dtype=torch.float64), params
        ).abs()
    )
    off = float(
        gabor(
            torch.tensor(0.4, dtype=torch.float64), torch.tensor(0.0, dtype=torch.float64), params
        ).abs()
    )
    assert center == pytest.approx(0.0, abs=1e-12)
    assert off > center


def test_magnitude_bounded_by_amplitude_with_carrier() -> None:
    """Even with a nonzero carrier, |g| <= |amplitude| everywhere."""
    params = GaborParams(
        amplitude=1.5, t_x=0.0, t_y=0.0, theta=0.4, sigma_x=2.0, sigma_y=1.0, k=3.0, phi=0.7
    )
    x, y = _grid(n=61, extent=12.0)
    assert float(gabor(x, y, params).abs().max()) <= abs(params.amplitude) + 1e-9


def test_rejects_nonpositive_sigma() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        GaborParams(
            amplitude=1.0, t_x=0.0, t_y=0.0, theta=0.0, sigma_x=0.0, sigma_y=1.0, k=1.0, phi=0.0
        )
