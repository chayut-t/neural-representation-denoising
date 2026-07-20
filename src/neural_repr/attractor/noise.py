"""State and weight perturbations with an explicit resampling protocol.

The 2016 text leaves it ambiguous whether "weight noise" is resampled each step,
each trajectory, or fixed as a single perturbation. These are *different
experiments* (baseline App C; correction context ``b26-corr-recurrence``), so
Phase 3 makes the protocol explicit (plan 3B.4). Three named modes:

* ``per_step`` — a fresh sample every recurrent step (the per-step random noise of
  the robustness sweep, ``fig:attractor-random-noise``);
* ``per_trajectory`` — one sample drawn once and reused for every step of a
  trajectory; call :meth:`WeightNoiseSampler.reset` at a trajectory boundary to
  draw an independent perturbation for the next trajectory;
* ``fixed`` — one perturbation supplied externally and held constant across all
  trajectories and steps (the single fixed random matrix added to the weights in
  ``fig:attractor-fixed-perturbation``).

State noise ``xi_t`` is ``N(0, sigma^2)`` per unit. Weight noise here follows the
Gaussian convention for the per-step/per-trajectory sweeps; the fixed-perturbation
experiment instead draws entries uniformly from ``(-sigma, sigma)`` via
:func:`uniform_weight_perturbation`, matching the baseline's stated protocol. All
draws take an explicit :class:`torch.Generator` so trajectories are reproducible,
and all accept ``dtype``/``device`` so they can match a CUDA/float32 state.

Batched semantics: a weight perturbation is a single ``(N, N)`` matrix. A
``per_trajectory`` sampler therefore models *one* trajectory; independent draws
across a batch require one sampler per item (or a ``reset`` between trajectories).
This is stated so "per trajectory" is never silently applied across a whole batch.
"""

from __future__ import annotations

from typing import Literal, get_args

import torch

ResampleMode = Literal["per_step", "per_trajectory", "fixed"]
_RESAMPLE_MODES: tuple[str, ...] = get_args(ResampleMode)


def _check_sigma(sigma: float) -> None:
    if not sigma >= 0.0:  # rejects negatives AND NaN (NaN >= 0 is False)
        raise ValueError(f"sigma must be a non-negative real number, got {sigma!r}")


def state_noise(
    shape: tuple[int, ...],
    sigma: float,
    *,
    generator: torch.Generator | None = None,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Gaussian state perturbation ``xi ~ N(0, sigma^2)`` of the given shape."""
    _check_sigma(sigma)
    return sigma * torch.randn(shape, dtype=dtype, device=device, generator=generator)


def gaussian_weight_perturbation(
    n: int,
    sigma: float,
    *,
    generator: torch.Generator | None = None,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Gaussian weight perturbation ``Xi ~ N(0, sigma^2)`` of shape ``(N, N)``.

    Used by the per-step / per-trajectory robustness sweeps.
    """
    _check_sigma(sigma)
    return sigma * torch.randn((n, n), dtype=dtype, device=device, generator=generator)


def uniform_weight_perturbation(
    n: int,
    sigma: float,
    *,
    generator: torch.Generator | None = None,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Fixed weight perturbation with entries ``~ U(-sigma, sigma)`` (``(N, N)``).

    Matches the fixed-connection-perturbation experiment
    (``fig:attractor-fixed-perturbation``). Note (baseline §4.7): an unconstrained
    random matrix has both symmetric and antisymmetric parts, so it can deform
    restoration *and* introduce drift — callers that want to perturb only ``S`` or
    only ``A`` must split it explicitly.
    """
    _check_sigma(sigma)
    u = torch.rand((n, n), dtype=dtype, device=device, generator=generator)
    return (2.0 * u - 1.0) * sigma


class WeightNoiseSampler:
    """Yields weight perturbations according to a fixed :data:`ResampleMode`.

    ``per_step`` draws every :meth:`sample` call; ``per_trajectory`` draws once and
    caches until :meth:`reset` (a trajectory boundary); ``fixed`` always returns the
    supplied perturbation. This makes the resampling protocol an explicit, testable
    object rather than an implicit code convention.
    """

    def __init__(
        self,
        mode: ResampleMode,
        n: int,
        sigma: float,
        *,
        generator: torch.Generator | None = None,
        fixed: torch.Tensor | None = None,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str | None = None,
    ) -> None:
        if mode not in _RESAMPLE_MODES:
            raise ValueError(f"unknown resample mode {mode!r}; expected one of {_RESAMPLE_MODES}")
        if n < 1:
            raise ValueError(f"n must be a positive matrix dimension, got {n}")
        _check_sigma(sigma)
        if mode == "fixed":
            if fixed is None:
                raise ValueError("mode='fixed' requires a `fixed` perturbation matrix")
            if fixed.shape != (n, n):
                raise ValueError(f"fixed perturbation must be ({n}, {n}), got {tuple(fixed.shape)}")
        elif fixed is not None:
            raise ValueError(f"`fixed` is only valid for mode='fixed', not mode={mode!r}")
        self.mode = mode
        self.n = n
        self.sigma = sigma
        self._generator = generator
        self._dtype = dtype
        self._device = device
        self._fixed = fixed
        self._cached: torch.Tensor | None = None

    def reset(self) -> None:
        """Mark a trajectory boundary: a ``per_trajectory`` sampler redraws next call."""
        self._cached = None

    def sample(self) -> torch.Tensor:
        if self.mode == "fixed":
            assert self._fixed is not None
            return self._fixed
        if self.mode == "per_trajectory":
            if self._cached is None:
                self._cached = self._draw()
            return self._cached
        return self._draw()  # per_step

    def _draw(self) -> torch.Tensor:
        return gaussian_weight_perturbation(
            self.n,
            self.sigma,
            generator=self._generator,
            dtype=self._dtype,
            device=self._device,
        )
