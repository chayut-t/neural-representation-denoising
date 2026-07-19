"""Finite sparse-inference recurrences (unrolled, differentiable).

Two named variants resolve the documented ambiguity in the 2016 inference update
(baseline §3.3.1; correction ``b26-corr-group-grad``; decision 0004). The original
code is lost and its description mixes ISTA-style soft thresholding with an
adaptive gradient of the group penalty, so neither variant is presented as *the*
original:

* :func:`smooth_gradient` (a.k.a. ``proximal_group`` slot for the modern model) —
  the mathematically direct gradient step on the smoothed objective
  (``eq:group-gradient-step``)::

      s_{t+1} = s_t - eta [ Phi^T(Phi s_t - x) + lambda * grad_Omega(s_t) ].

  ``grad_Omega`` is the signed group gradient (``eq:group-coefficient-gradient``).
  This is a true (sub)gradient method for the smoothed penalty and is the variant
  the modern edition uses.

* :func:`legacy_adaptive_threshold` — the historical-compatible reading. It keeps
  an ISTA proximal step but uses a per-coefficient *adaptive* soft threshold built
  from the group coupling ``sum_k Lambda_ki / z_k(s_t)``::

      v_{t+1} = s_t - eta * Phi^T(Phi s_t - x)
      theta_i = eta * lambda * sum_k Lambda_ki / z_k(s_t)      (>= 0)
      s_{t+1} = soft(v_{t+1}, theta_i).

  As the baseline notes, scalar soft-thresholding with such a "threshold" is *not*
  the proximal operator of the overlapping-group penalty; this variant exists to
  reproduce the historical finite computation, not to solve ``eq:group-objective``.

Both are unrolled for a fixed number of steps and are differentiable end-to-end,
so a denoising loss can be backpropagated through them (primer §unrolling,
``eq:unrolled-state``). Neither claims to reach the exact ``argmin`` of the nominal
objective — the learned finite-depth system is the model (primer §unrolling).
"""

from __future__ import annotations

from typing import Literal

import torch

from neural_repr.sparse_coding.dictionary import reconstruct
from neural_repr.sparse_coding.groups import DEFAULT_EPS, group_activity, group_penalty_grad

InferenceVariant = Literal["smooth_gradient", "legacy_adaptive_threshold"]


def data_grad(phi: torch.Tensor, s: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Gradient of the data term ``0.5||x - Phi s||^2`` w.r.t. ``s``.

    Equals ``Phi^T(Phi s - x)`` (primer §sparse). Shapes: ``phi (N, M)``,
    ``s (..., M)``, ``x (..., N)`` -> ``(..., M)``.
    """
    residual = reconstruct(phi, s) - x  # (..., N)
    return residual @ phi  # (..., N) @ (N, M) -> (..., M)


def data_lipschitz(phi: torch.Tensor) -> float:
    """Lipschitz constant of the data-term gradient, ``L_data = ||Phi^T Phi||_2``.

    The largest eigenvalue of ``Phi^T Phi``.
    """
    gram = phi.transpose(-1, -2) @ phi
    lipschitz = float(torch.linalg.eigvalsh(gram).max().item())
    if lipschitz <= 0.0:
        raise ValueError("dictionary has non-positive Lipschitz constant; cannot pick a step size")
    return lipschitz


def penalty_lipschitz(
    lambda_: torch.Tensor, penalty_weight: float, *, eps: float = DEFAULT_EPS
) -> float:
    """Upper bound on the Lipschitz constant of ``penalty_weight * grad_Omega``.

    The Hessian of a single ``z_k = sqrt(eps + sum_i Lambda_ki s_i^2)`` satisfies
    ``Hess(z_k) = diag(Lambda_k)/z_k - (Lambda_k⊙s)(Lambda_k⊙s)^T / z_k^3 ⪯
    diag(Lambda_k)/sqrt(eps)`` (the subtracted term is PSD, and ``z_k >= sqrt(eps)``).
    Summing over groups, ``Hess(Omega) ⪯ diag(sum_k Lambda_ki)/sqrt(eps)``, so the
    largest curvature of ``Omega`` is bounded by ``max_i (sum_k Lambda_ki)/sqrt(eps)``.
    Returns ``penalty_weight`` times that bound.
    """
    if penalty_weight < 0.0:
        raise ValueError("penalty_weight must be non-negative")
    if eps <= 0.0:
        raise ValueError("eps must be positive")
    col_sums = lambda_.sum(dim=-2)  # sum over groups k, per coefficient i
    max_col_sum = float(col_sums.max().item())
    return float(penalty_weight * max_col_sum / (eps**0.5))


def lipschitz_step_size(phi: torch.Tensor, *, safety: float = 1.0) -> float:
    """A stable **data-term-only** step ``eta = safety / L_data``, ``L_data = ||Phi^T Phi||_2``.

    Safe for ISTA where the penalty enters through a proximal operator (its
    curvature is handled exactly by the prox). For :func:`smooth_gradient`, which
    takes an explicit gradient step on the *total* smoothed objective, use
    :func:`stable_step_size` instead — the smoothed group penalty adds curvature
    (large near ``s=0`` for small ``eps``) that this value ignores.
    """
    _check_safety(safety)
    return safety / data_lipschitz(phi)


def stable_step_size(
    phi: torch.Tensor,
    lambda_: torch.Tensor,
    penalty_weight: float,
    *,
    eps: float = DEFAULT_EPS,
    safety: float = 1.0,
) -> float:
    """Stable step for :func:`smooth_gradient`: ``safety / (L_data + L_penalty)``.

    Uses the total-objective Lipschitz bound ``L_data + penalty_weight *
    max_i(sum_k Lambda_ki)/sqrt(eps)`` (see :func:`data_lipschitz`,
    :func:`penalty_lipschitz`), so the explicit gradient step is safe for the whole
    smoothed objective, not only the data term. This is the step-size rule the math
    spec documents for smooth-gradient inference.
    """
    _check_safety(safety)
    total = data_lipschitz(phi) + penalty_lipschitz(lambda_, penalty_weight, eps=eps)
    return safety / total


def _check_safety(safety: float) -> None:
    if not 0.0 < safety <= 1.0:
        raise ValueError(f"safety must lie in (0, 1], got {safety}")


def soft_threshold(v: torch.Tensor, theta: torch.Tensor | float) -> torch.Tensor:
    """Componentwise soft threshold ``sign(v) * max(|v| - theta, 0)`` (primer).

    ``theta`` may be a scalar or broadcast to ``v``; negative thresholds are
    clamped to zero so a signed coupling can never *grow* a coefficient.
    """
    theta_t = torch.as_tensor(theta, dtype=v.dtype, device=v.device).clamp_min(0.0)
    return torch.sign(v) * (v.abs() - theta_t).clamp_min(0.0)


def smooth_gradient(
    x: torch.Tensor,
    phi: torch.Tensor,
    lambda_: torch.Tensor,
    *,
    n_steps: int,
    eta: float,
    penalty_weight: float,
    eps: float = DEFAULT_EPS,
    s0: torch.Tensor | None = None,
) -> torch.Tensor:
    """Unrolled smooth-gradient inference (``eq:group-gradient-step``).

    Returns the final code ``s_T`` with shape ``x.shape[:-1] + (M,)``. The
    returned tensor participates in autograd (gradients flow through every step
    and into ``phi``, ``lambda_``).
    """
    _validate(n_steps, eta, penalty_weight)
    s = _init_code(x, phi, s0)
    for _ in range(n_steps):
        grad = data_grad(phi, s, x) + penalty_weight * group_penalty_grad(s, lambda_, eps=eps)
        s = s - eta * grad
    return s


def legacy_adaptive_threshold(
    x: torch.Tensor,
    phi: torch.Tensor,
    lambda_: torch.Tensor,
    *,
    n_steps: int,
    eta: float,
    penalty_weight: float,
    eps: float = DEFAULT_EPS,
    s0: torch.Tensor | None = None,
) -> torch.Tensor:
    """Unrolled historical-compatible inference (ISTA + adaptive group threshold).

    Reproduces the 2016 finite computation as closely as the specification allows:
    an ISTA data step followed by a soft threshold whose per-coefficient magnitude
    is the group coupling ``eta * lambda * sum_k Lambda_ki / z_k(s_t)`` evaluated at
    the current code. Returns ``s_T`` shaped ``x.shape[:-1] + (M,)``; differentiable.
    """
    _validate(n_steps, eta, penalty_weight)
    s = _init_code(x, phi, s0)
    for _ in range(n_steps):
        v = s - eta * data_grad(phi, s, x)
        # Adaptive per-coefficient threshold from the (nonnegative) group coupling.
        z = group_activity(s, lambda_, eps=eps)  # (..., K)
        coupling = (1.0 / z) @ lambda_  # (..., M)
        theta = eta * penalty_weight * coupling
        s = soft_threshold(v, theta)
    return s


def infer(
    variant: InferenceVariant,
    x: torch.Tensor,
    phi: torch.Tensor,
    lambda_: torch.Tensor,
    **kwargs: object,
) -> torch.Tensor:
    """Dispatch to a named inference variant (decision 0004)."""
    if variant == "smooth_gradient":
        return smooth_gradient(x, phi, lambda_, **kwargs)  # type: ignore[arg-type]
    if variant == "legacy_adaptive_threshold":
        return legacy_adaptive_threshold(x, phi, lambda_, **kwargs)  # type: ignore[arg-type]
    raise ValueError(f"unknown inference variant: {variant!r}")


def _init_code(x: torch.Tensor, phi: torch.Tensor, s0: torch.Tensor | None) -> torch.Tensor:
    if s0 is not None:
        return s0
    m = phi.shape[-1]
    return x.new_zeros((*x.shape[:-1], m))


def _validate(n_steps: int, eta: float, penalty_weight: float) -> None:
    if n_steps < 0:
        raise ValueError("n_steps must be non-negative")
    if eta <= 0.0:
        raise ValueError("step size eta must be positive")
    if penalty_weight < 0.0:
        raise ValueError("penalty_weight (lambda) must be non-negative")
