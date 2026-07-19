"""Multi-step trajectory helpers: write, hold, and move (§4.4 operations).

Thin wrappers around :func:`neural_repr.attractor.dynamics.step` that run the four
operations of revised Chapter 4 over several time steps. Kept separate from the
single-step recurrence so tests and later training code can compose them without
duplicating the loop. All routines are differentiable end-to-end (gradients flow
through every step), which is what makes backpropagation-through-time training
possible (§4.5, ``eq:attractor-loss``).

Each :func:`run` call is treated as **one trajectory**: a supplied
``weight_sampler`` is :meth:`~neural_repr.attractor.noise.WeightNoiseSampler.reset`
at entry, so a ``per_trajectory`` sampler draws a fresh perturbation per ``run``
(reused across that run's steps) rather than silently reusing the first draw
forever across trajectories.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from neural_repr.attractor.dynamics import Nonlinearity, Velocity, step
from neural_repr.attractor.noise import WeightNoiseSampler, state_noise


def run(
    u0: torch.Tensor,
    *,
    s_mat: torch.Tensor,
    a_mats: torch.Tensor | Sequence[torch.Tensor],
    velocity: Velocity,
    alpha: float,
    tau: float,
    dt: float,
    phi: Nonlinearity,
    n_steps: int,
    code: torch.Tensor | None = None,
    beta: float = 0.0,
    state_sigma: float = 0.0,
    weight_sampler: WeightNoiseSampler | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Run ``n_steps`` of the recurrence and return the final state ``u_T``.

    Parameters mirror :func:`~neural_repr.attractor.dynamics.step`. ``state_sigma``
    turns on fresh per-step Gaussian state noise (matching ``u0``'s dtype/device);
    ``weight_sampler`` supplies weight perturbations under its own resampling
    protocol. Both are off by default so the noiseless dynamics are the base case.
    """
    if n_steps < 0:
        raise ValueError("n_steps must be non-negative")
    if state_sigma < 0.0:
        raise ValueError(f"state_sigma must be non-negative, got {state_sigma}")
    if weight_sampler is not None:
        # Trajectory boundary: a per_trajectory sampler draws fresh for this run.
        weight_sampler.reset()
    u = u0
    for _ in range(n_steps):
        xi = (
            state_noise(
                tuple(u.shape),
                state_sigma,
                generator=generator,
                dtype=u.dtype,
                device=u.device,
            )
            if state_sigma > 0.0
            else None
        )
        big_xi = weight_sampler.sample() if weight_sampler is not None else None
        u = step(
            u,
            s_mat=s_mat,
            a_mats=a_mats,
            velocity=velocity,
            alpha=alpha,
            tau=tau,
            dt=dt,
            phi=phi,
            code=code,
            beta=beta,
            state_noise=xi,
            weight_noise=big_xi,
        )
    return u


def hold(
    u0: torch.Tensor,
    *,
    s_mat: torch.Tensor,
    a_mats: torch.Tensor | Sequence[torch.Tensor],
    alpha: float,
    tau: float,
    dt: float,
    phi: Nonlinearity,
    n_steps: int,
    state_sigma: float = 0.0,
    weight_sampler: WeightNoiseSampler | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Autonomous hold: ``v = 0``, no write input (§4.4). Returns ``u_T``."""
    return run(
        u0,
        s_mat=s_mat,
        a_mats=a_mats,
        velocity=0.0,
        alpha=alpha,
        tau=tau,
        dt=dt,
        phi=phi,
        n_steps=n_steps,
        code=None,
        beta=0.0,
        state_sigma=state_sigma,
        weight_sampler=weight_sampler,
        generator=generator,
    )
