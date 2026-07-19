"""Prescribed location codes and the population-vector decoder.

Formalizes §4.6 of revised Chapter 4 (``chap4.tex``). Three code families:

Localized line/place code (``eq:place-line-code``), centers ``mu_i = i/(N-1)``::

    (P_line[x])_i = exp[ -(x - mu_i)^2 / (2 w^2) ],   i = 0..N-1.

Ring/periodic code (``eq:ring-code``) on phase ``q in [0, 1)``, centers
``mu_i = i/N``, with circular distance ``d_i(q) = min(|q-mu_i|, 1-|q-mu_i|)``::

    (P_ring[q])_i = exp[ -d_i(q)^2 / (2 w^2) ].

Mixed place--periodic code (``eq:place-grid-code``): ``N_0`` localized units
stacked over ``N_1`` periodic units at phase ``k x mod 1``.

Decoder (``eq:place-readout``): a population-vector average over the *place*
block, ``x_hat = sum_i f_i mu_i / sum_i f_i``. The baseline states the mixed-code
readout for the periodic block is under-specified; we therefore decode global
location from the place block only and document that choice (plan 3B, baseline
§4.6). The place decoder is biased near the boundaries because the Gaussian bump
is truncated — a known bias flagged for fair comparison, not hidden.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

DEFAULT_WIDTH: float = 0.05


def _check_width(width: float) -> None:
    if not width > 0.0:
        raise ValueError(f"tuning width must be positive, got {width}")


def _as_input_tensor(x: torch.Tensor | float) -> torch.Tensor:
    """Coerce a scalar/tensor location to a floating tensor, preserving device/dtype.

    A Python scalar becomes a float64 CPU tensor (the package default); an existing
    tensor keeps its device and dtype (integer tensors are promoted to float).
    """
    if isinstance(x, torch.Tensor):
        return x if x.is_floating_point() else x.to(torch.float64)
    return torch.as_tensor(x, dtype=torch.float64)


def line_centers(
    n: int, *, dtype: torch.dtype = torch.float64, device: torch.device | str | None = None
) -> torch.Tensor:
    """Place-code centers ``mu_i = i/(N-1)`` on ``[0, 1]`` (``eq:place-line-code``)."""
    if n < 2:
        raise ValueError("line code needs at least 2 units")
    return torch.arange(n, dtype=dtype, device=device) / (n - 1)


def ring_centers(
    n: int, *, dtype: torch.dtype = torch.float64, device: torch.device | str | None = None
) -> torch.Tensor:
    """Ring-code centers ``mu_i = i/N`` on the circle ``[0, 1)`` (``eq:ring-code``)."""
    if n < 1:
        raise ValueError("ring code needs at least 1 unit")
    return torch.arange(n, dtype=dtype, device=device) / n


def place_code(x: torch.Tensor | float, n: int, *, width: float = DEFAULT_WIDTH) -> torch.Tensor:
    """Localized line code ``P_line[x]`` (``eq:place-line-code``).

    ``x`` may be a scalar or a batch (shape ``(...,)``); returns ``(..., N)``. The
    result follows ``x``'s dtype/device when ``x`` is a tensor.
    """
    _check_width(width)
    x_t = _as_input_tensor(x)
    mu = line_centers(n, dtype=x_t.dtype, device=x_t.device)
    d = x_t.unsqueeze(-1) - mu
    return torch.exp(-(d**2) / (2.0 * width**2))


def circular_distance(q: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
    """Circular distance ``d(q, mu) = min(|q-mu|, 1-|q-mu|)`` on the unit circle."""
    raw = (q.unsqueeze(-1) - mu).abs()
    return torch.minimum(raw, 1.0 - raw)


def ring_code(q: torch.Tensor | float, n: int, *, width: float = DEFAULT_WIDTH) -> torch.Tensor:
    """Periodic ring code ``P_ring[q]`` on phase ``q in [0, 1)`` (``eq:ring-code``).

    ``q`` is taken modulo 1 first. ``q`` may be scalar or batched; returns
    ``(..., N)`` following ``q``'s dtype/device when ``q`` is a tensor.
    """
    _check_width(width)
    q_t = _as_input_tensor(q) % 1.0
    mu = ring_centers(n, dtype=q_t.dtype, device=q_t.device)
    d = circular_distance(q_t, mu)
    return torch.exp(-(d**2) / (2.0 * width**2))


@dataclass(frozen=True)
class MixedCodeSpec:
    """Layout of a mixed place--periodic code (``eq:place-grid-code``).

    ``n_place`` localized units followed by ``n_periodic`` ring units at frequency
    ``k`` (phase ``k x mod 1``). Total units ``N = n_place + n_periodic``.
    """

    n_place: int
    n_periodic: int
    frequency: int
    width: float = DEFAULT_WIDTH

    def __post_init__(self) -> None:
        if self.n_place < 2:
            raise ValueError("mixed code needs at least 2 place units (for the readout)")
        if self.n_periodic < 1:
            raise ValueError("mixed code needs at least 1 periodic unit")
        if self.frequency < 1:
            raise ValueError("mixed-code frequency k must be a positive integer")
        _check_width(self.width)

    @property
    def n_total(self) -> int:
        return self.n_place + self.n_periodic


def mixed_code(x: torch.Tensor | float, spec: MixedCodeSpec) -> torch.Tensor:
    """Mixed place--periodic code (``eq:place-grid-code``); returns ``(..., N)``."""
    x_t = _as_input_tensor(x)
    place = place_code(x_t, spec.n_place, width=spec.width)
    periodic = ring_code((spec.frequency * x_t) % 1.0, spec.n_periodic, width=spec.width)
    return torch.cat([place, periodic], dim=-1)


def decode_place(f: torch.Tensor, n_place: int, *, eps: float = 1e-12) -> torch.Tensor:
    """Population-vector readout over the place block (``eq:place-readout``).

    Uses the first ``n_place`` units (the localized block) so the same decoder
    applies to place-only and mixed codes. Returns the decoded location(s), shape
    ``f.shape[:-1]``. ``eps`` guards a fully silent bump. Centers follow ``f``'s
    dtype and device.
    """
    if n_place < 2:
        raise ValueError("place readout needs at least 2 place units")
    place_activity = f[..., :n_place]
    mu = line_centers(n_place, dtype=place_activity.dtype, device=place_activity.device)
    total = place_activity.sum(dim=-1)
    weighted = (place_activity * mu).sum(dim=-1)
    return weighted / total.clamp_min(eps)
