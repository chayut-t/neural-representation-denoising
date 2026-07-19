"""Structured recurrent weights: symmetric restoration + antisymmetric transport.

Formalizes §4.4 of revised Chapter 4 (``chap4.tex``) and the primer §recurrent
(``chap2.tex``). The connection matrix is decomposed
(``eq:attractor-effective-weight``, ``eq:symmetric-antisymmetric``)::

    W_eff(v) = S + alpha * v * A,   S^T = S,   A^T = -A,

and, for the 2D extension (``eq:attractor-2d-transport``)::

    W_eff(v_x, v_y) = S + alpha (v_x A_x + v_y A_y).

Structure is enforced **by parameterization**, not post-hoc (plan 3B.2): from
unconstrained square matrices ``W`` and ``V_d`` we build
``S = (W + W^T)/2`` and ``A_d = (V_d - V_d^T)/2``. This guarantees exact symmetry
and antisymmetry for every parameter value and every gradient step, so no
projection is needed after updates (the baseline flags post-hoc enforcement as an
unresolved 2016 detail; we resolve it here).

Roles are operational (baseline §4.4): ``S`` is assigned restoration, ``A`` is
assigned velocity-dependent transport. ``A`` does **not** define a scalar energy;
robustness is measured from trajectories/basins, never inferred from the magnitude
of ``A`` (correction ``b26-corr-energy``, decision 0004).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

Velocity = float | Sequence[float] | torch.Tensor


def symmetric_part(w: torch.Tensor) -> torch.Tensor:
    """Symmetric part ``S = (W + W^T)/2`` — satisfies ``S^T = S`` exactly."""
    return 0.5 * (w + w.transpose(-1, -2))


def antisymmetric_part(v: torch.Tensor) -> torch.Tensor:
    """Antisymmetric part ``A = (V - V^T)/2`` — satisfies ``A^T = -A`` exactly."""
    return 0.5 * (v - v.transpose(-1, -2))


def effective_weight(
    s_mat: torch.Tensor,
    a_mats: torch.Tensor | Sequence[torch.Tensor],
    velocities: Velocity,
    *,
    alpha: float,
) -> torch.Tensor:
    """Effective recurrent matrix ``W_eff = S + alpha * sum_d v_d A_d``.

    Parameters
    ----------
    s_mat:
        Symmetric matrix ``S`` (``(N, N)``); assumed already symmetric (build it
        with :func:`symmetric_part`).
    a_mats:
        A single antisymmetric matrix ``A`` (``(N, N)``) or a sequence of them
        (one per transport direction ``d``) for the 2D/multi-direction case.
    velocities:
        Signed velocity control(s): a scalar for one direction, or a sequence /
        1-D tensor of one value per direction in ``a_mats``. Tensor velocities are
        **preserved** (not cast to Python floats), so a differentiable/on-device
        velocity stays differentiable and avoids a device sync.
    alpha:
        Velocity gain (baseline ``alpha``).

    Returns
    -------
    torch.Tensor
        ``W_eff`` with shape ``(N, N)``.
    """
    a_list = [a_mats] if isinstance(a_mats, torch.Tensor) else list(a_mats)
    v_list = _velocity_list(velocities)
    if len(a_list) != len(v_list):
        raise ValueError(
            f"got {len(a_list)} transport matrices but {len(v_list)} velocities; they must match"
        )
    w_eff = s_mat
    for a_mat, v in zip(a_list, v_list, strict=True):
        w_eff = w_eff + alpha * v * a_mat
    return w_eff


def _velocity_list(velocities: Velocity) -> list[float | torch.Tensor]:
    """Normalize a velocity argument to a per-direction list, preserving tensors.

    A 0-D tensor or Python scalar becomes a length-1 list; a 1-D tensor is split
    into its scalar sub-tensors (each still a tensor, so autograd/device are kept);
    a Python sequence is taken elementwise.
    """
    if isinstance(velocities, torch.Tensor):
        if velocities.ndim == 0:
            return [velocities]
        if velocities.ndim == 1:
            return list(velocities.unbind(0))
        raise ValueError(f"velocity tensor must be 0-D or 1-D, got shape {tuple(velocities.shape)}")
    if isinstance(velocities, int | float):
        return [float(velocities)]
    return [v if isinstance(v, torch.Tensor) else float(v) for v in velocities]
