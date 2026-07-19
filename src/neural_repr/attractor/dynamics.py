"""Discrete recurrent dynamics with an explicit time step, plus write/hold/move.

Formalizes §4.4-§4.5 of revised Chapter 4. The model step is first-order in
``Delta t / tau`` (``eq:attractor-discrete``), **not** absolute ``t``
(correction ``b26-corr-recurrence``, decision 0004)::

    u_{t+1} = u_t + (dt/tau) [ -u_t + (S + sum_d alpha v_{d,t} A_d + Xi_t) f_eff_t + xi_t ]
    f_t     = phi(u_t).

The activity nonlinearity (``eq:attractor-nonlinearity``) is::

    phi(u) = a * log(1 + c * [u + b]_+),   [q]_+ = max(q, 0),

with **trainable** scalars ``a``, ``b``, ``c`` (see :class:`Nonlinearity`, an
``nn.Module``). ``a`` and ``c`` are kept strictly positive by a softplus
parameterization: ``c > 0`` keeps the ``log1p`` argument valid, and ``a > 0`` keeps
activity nonnegative for the decoder.

Write mixing (``eq:attractor-write``)::

    f_eff_t = (1 - beta_t) phi(u_t) + beta_t c(x_write),   beta_t in {0, 1},

so ``beta = 1`` writes the codeword and ``beta = 0`` runs autonomous recurrence.
Hold is ``v = 0``; move is ``v != 0`` (§4.4 operations). Transport is
multi-directional: ``a_mats``/``velocity`` accept a single ``(A, v)`` pair (1D) or a
sequence (2D and beyond, ``eq:attractor-2d-transport``).

Noise-scaling convention (the baseline flags this as unresolved; we fix it here,
plan 3B.4): both the state perturbation ``xi_t`` and the weight perturbation
``Xi_t`` are sampled as standard deviations *inside the bracket* — i.e. they enter
on the same footing as ``-u + W f``, and the ``dt/tau`` factor multiplies the whole
bracket. So a reported noise std of ``sigma`` means ``xi_t ~ N(0, sigma^2)`` added
to the bracket; its net contribution to ``u_{t+1}`` is scaled by ``dt/tau``. This
is one explicit, documented choice among the conventions the 2016 text left
ambiguous; :mod:`neural_repr.attractor.noise` builds these perturbations.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn

from neural_repr.attractor.weights import effective_weight

Velocity = float | Sequence[float] | torch.Tensor


def log_rectifier(
    u: torch.Tensor,
    a: torch.Tensor | float,
    b: torch.Tensor | float,
    c: torch.Tensor | float,
) -> torch.Tensor:
    """Functional activity nonlinearity ``a * log(1 + c * [u + b]_+)`` (baseline eq).

    Implements ``eq:attractor-nonlinearity``. Free function so the raw math is
    testable directly (finite-difference gradients
    w.r.t. ``a``, ``b``, ``c``). :class:`Nonlinearity` calls this with its positively
    parameterized ``a``, ``c``. Caller is responsible for ``c > 0`` here; the module
    guarantees it.
    """
    rectified = torch.clamp(u + b, min=0.0)
    return a * torch.log1p(c * rectified)


def _softplus_inverse(y: float) -> float:
    """Inverse of softplus, so ``softplus(_softplus_inverse(y)) == y`` for ``y > 0``."""
    return math.log(math.expm1(y))


class Nonlinearity(nn.Module):
    """Trainable activity nonlinearity ``phi(u) = a log(1 + c [u+b]_+)`` (baseline eq).

    Implements ``eq:attractor-nonlinearity``.
    An ``nn.Module`` with trainable parameters (baseline: ``a``, ``b``, ``c`` are all
    trainable). ``a`` and ``c`` are parameterized through softplus of unconstrained
    raw parameters, so they stay strictly positive under any gradient step —
    ``c > 0`` keeps the ``log1p`` argument valid and ``a > 0`` keeps activity
    nonnegative. ``b`` (a threshold shift) is unconstrained. The constructor takes
    the desired *effective* initial values (with ``a > 0``, ``c > 0``) and stores
    their softplus-inverse.

    Parameters default to ``float64`` to match the rest of the package; use ``.to()``
    (standard ``nn.Module`` behavior) or the ``dtype``/``device`` arguments for
    float32/CUDA work.
    """

    def __init__(
        self,
        a: float = 1.0,
        b: float = 0.0,
        c: float = 1.0,
        *,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        for name, value in (("a", a), ("b", b), ("c", c)):
            if not math.isfinite(value):
                raise ValueError(f"Nonlinearity parameter {name} must be finite, got {value}")
        if a <= 0.0:
            raise ValueError(f"Nonlinearity `a` must be positive (keeps activity >= 0), got {a}")
        if c <= 0.0:
            raise ValueError(f"Nonlinearity `c` must be positive (log1p argument), got {c}")
        self.raw_a = nn.Parameter(torch.tensor(_softplus_inverse(a), dtype=dtype, device=device))
        self.b = nn.Parameter(torch.tensor(float(b), dtype=dtype, device=device))
        self.raw_c = nn.Parameter(torch.tensor(_softplus_inverse(c), dtype=dtype, device=device))

    @property
    def a(self) -> torch.Tensor:
        """Effective (positive) amplitude ``a = softplus(raw_a)``."""
        return nn.functional.softplus(self.raw_a)

    @property
    def c(self) -> torch.Tensor:
        """Effective (positive) gain ``c = softplus(raw_c)``."""
        return nn.functional.softplus(self.raw_c)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        return log_rectifier(u, self.a, self.b, self.c)


def write_mix(u: torch.Tensor, phi: Nonlinearity, code: torch.Tensor, beta: float) -> torch.Tensor:
    """Effective activity ``f_eff = (1-beta) phi(u) + beta c(x_write)`` (``eq:attractor-write``).

    ``beta`` is nominally in ``{0, 1}`` (write / autonomous) but any value in
    ``[0, 1]`` is accepted as a soft mix; values outside ``[0, 1]`` raise.
    """
    if not 0.0 <= beta <= 1.0:
        raise ValueError("beta must lie in [0, 1] (1 writes the codeword, 0 runs autonomously)")
    activity: torch.Tensor = phi(u)
    return (1.0 - beta) * activity + beta * code


def step(
    u: torch.Tensor,
    *,
    s_mat: torch.Tensor,
    a_mats: torch.Tensor | Sequence[torch.Tensor],
    velocity: Velocity,
    alpha: float,
    tau: float,
    dt: float,
    phi: Nonlinearity,
    code: torch.Tensor | None = None,
    beta: float = 0.0,
    state_noise: torch.Tensor | None = None,
    weight_noise: torch.Tensor | None = None,
) -> torch.Tensor:
    """One discrete update ``u_{t+1}`` (``eq:attractor-discrete``).

    ``a_mats``/``velocity`` accept a single transport matrix + scalar (1D) or
    matching sequences (multi-direction, ``eq:attractor-2d-transport``); tensor
    velocities are preserved (not cast to float) so controls stay differentiable and
    on-device. ``state_noise`` (``xi_t``, shape like ``u``) and ``weight_noise``
    (``Xi_t``, ``(N, N)``) are optional perturbations added inside the bracket. When
    ``beta > 0`` a ``code`` must be supplied to write. Returns the next state, same
    shape as ``u``.
    """
    if tau <= 0.0 or dt <= 0.0:
        raise ValueError("tau and dt must be positive")
    if beta > 0.0 and code is None:
        raise ValueError("writing (beta > 0) requires a code to write toward")

    f_eff: torch.Tensor = (
        write_mix(u, phi, code, beta) if (code is not None and beta > 0.0) else phi(u)
    )

    w_eff = effective_weight(s_mat, a_mats, velocity, alpha=alpha)
    if weight_noise is not None:
        w_eff = w_eff + weight_noise

    bracket = -u + _mat_vec(w_eff, f_eff)
    if state_noise is not None:
        bracket = bracket + state_noise
    return u + (dt / tau) * bracket


def _mat_vec(w: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
    """Right-multiply ``W`` onto batched row vectors ``f`` (``(..., N)`` -> ``(..., N)``).

    Computes ``(W f^T)^T`` for each item, i.e. ``f @ W^T``, which broadcasts over
    any number of leading batch dimensions in ``f``.
    """
    return f @ w.transpose(-1, -2)
