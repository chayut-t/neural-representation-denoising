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

Operation helpers: :func:`write` (drive toward a codeword, ``beta=1``), :func:`hold`
(zero velocity, no input), and :func:`move` (constant nonzero velocity). All three,
plus the general :func:`run`, are multidirectional — ``a_mats``/``velocity`` accept a
single matrix or matching sequences. **Velocity is constant per call** in this Phase 3
formalization; the notation ``v_{d,t}`` (velocity varying per time step) is a
documented future extension, realized by calling :func:`step` in a caller-owned loop.
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
    """Autonomous hold: zero velocity, no write input (§4.4). Returns ``u_T``.

    Supports multiple transport directions: the zero velocity is expanded to match
    the number of ``a_mats`` so ``hold(..., a_mats=[A_x, A_y])`` works (round-2 review
    finding 8), not just the single-matrix case.
    """
    return run(
        u0,
        s_mat=s_mat,
        a_mats=a_mats,
        velocity=_zero_velocity(a_mats),
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


def _zero_velocity(a_mats: torch.Tensor | Sequence[torch.Tensor]) -> Velocity:
    """A zero velocity matching the number of transport directions in ``a_mats``."""
    n_directions = 1 if isinstance(a_mats, torch.Tensor) else len(a_mats)
    return 0.0 if n_directions == 1 else [0.0] * n_directions


def write(
    u0: torch.Tensor,
    code: torch.Tensor,
    *,
    s_mat: torch.Tensor,
    a_mats: torch.Tensor | Sequence[torch.Tensor],
    alpha: float,
    tau: float,
    dt: float,
    phi: Nonlinearity,
    n_steps: int,
    beta: float = 1.0,
) -> torch.Tensor:
    """Write: drive the state toward ``code`` (``beta`` in ``(0, 1]``) at zero velocity.

    Returns ``u_T`` after ``n_steps``. ``beta=1`` clamps activity to the codeword each
    step; smaller values mix it in. Multidirectional (zero velocity per direction).
    """
    if not 0.0 < beta <= 1.0:
        raise ValueError("write requires beta in (0, 1]")
    return run(
        u0,
        s_mat=s_mat,
        a_mats=a_mats,
        velocity=_zero_velocity(a_mats),
        alpha=alpha,
        tau=tau,
        dt=dt,
        phi=phi,
        n_steps=n_steps,
        code=code,
        beta=beta,
    )


def move(
    u0: torch.Tensor,
    velocity: Velocity,
    *,
    s_mat: torch.Tensor,
    a_mats: torch.Tensor | Sequence[torch.Tensor],
    alpha: float,
    tau: float,
    dt: float,
    phi: Nonlinearity,
    n_steps: int,
) -> torch.Tensor:
    """Move: run autonomous dynamics under a constant (per call) ``velocity``.

    ``velocity`` may be a scalar (1D) or one value per transport direction. Returns
    ``u_T``. Per-step-varying velocity is out of scope here (see the module docstring).
    """
    return run(
        u0,
        s_mat=s_mat,
        a_mats=a_mats,
        velocity=velocity,
        alpha=alpha,
        tau=tau,
        dt=dt,
        phi=phi,
        n_steps=n_steps,
        code=None,
        beta=0.0,
    )
