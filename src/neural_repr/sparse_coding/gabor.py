"""Corrected 2D Gabor model used to summarize learned dictionary atoms.

Adopts the corrected Gabor of the 2026 baseline (``eq:corrected-gabor``,
correction ``b26-corr-gabor``, decision 0004). The Gaussian envelope carries a
**negative** quadratic exponent, so it decays away from its center; the 2016 form
had the opposite sign and grew away from the center::

    g(x, y) = a * exp[ -1/2 ( x'^2 / sigma_x^2 + y'^2 / sigma_y^2 ) ] * cos(k x' + phi)
    x' =  (x - t_x) cos(theta) - (y - t_y) sin(theta)
    y' =  (x - t_x) sin(theta) + (y - t_y) cos(theta)

Factor convention (documented, plan 3A.7): the envelope uses the ``1/2`` factor
shown in ``eq:corrected-gabor``, ``sigma_x``/``sigma_y`` are standard deviations
(not variances) in the rotated frame, ``k`` is the carrier angular frequency in
radians per pixel along ``x'``, ``theta`` rotates the ``(x, y)`` frame, and
``(t_x, t_y)`` is the center. This is a descriptive/analysis model for fitting
atoms, not part of the trained objective.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class GaborParams:
    """Parameters of the corrected 2D Gabor (``eq:corrected-gabor``).

    ``sigma_x``/``sigma_y`` must be positive (they are standard deviations).
    """

    amplitude: float
    t_x: float
    t_y: float
    theta: float
    sigma_x: float
    sigma_y: float
    k: float
    phi: float

    def __post_init__(self) -> None:
        if self.sigma_x <= 0.0 or self.sigma_y <= 0.0:
            raise ValueError("sigma_x and sigma_y must be positive (they are Gaussian std-devs)")


def gabor(x: torch.Tensor, y: torch.Tensor, params: GaborParams) -> torch.Tensor:
    """Evaluate the corrected Gabor ``g(x, y)`` on coordinate tensors.

    ``x`` and ``y`` are broadcastable coordinate grids (any matching shape); the
    result has their broadcast shape. The **Gaussian envelope** is guaranteed to
    decay to 0 away from the center ``(t_x, t_y)`` for any finite parameters. The
    full Gabor is the envelope times the carrier ``cos(k x' + phi)``, so its value
    need not peak at the center: e.g. when ``cos(phi) = 0`` the Gabor is exactly 0
    at the center and larger just off it. The magnitude is still bounded by
    ``|amplitude|`` everywhere (the envelope ``<= 1`` and ``|cos| <= 1``).
    """
    cos_t = torch.cos(torch.as_tensor(params.theta, dtype=x.dtype, device=x.device))
    sin_t = torch.sin(torch.as_tensor(params.theta, dtype=x.dtype, device=x.device))
    dx = x - params.t_x
    dy = y - params.t_y
    x_rot = dx * cos_t - dy * sin_t
    y_rot = dx * sin_t + dy * cos_t
    envelope = torch.exp(
        -0.5 * ((x_rot**2) / (params.sigma_x**2) + (y_rot**2) / (params.sigma_y**2))
    )
    carrier = torch.cos(params.k * x_rot + params.phi)
    return params.amplitude * envelope * carrier
