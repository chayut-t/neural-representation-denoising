"""Linear-dynamics analysis encoding the continuous-time corrections.

Two corrections from the 2026 baseline are made explicit and testable here
(primer §recurrent; correction ``b26-corr-recurrence`` / ``b26-corr-energy``;
decision 0004):

1. **Amplification depends on the sign of a (continuous-time) eigenvalue, not on
   whether it exceeds one.** Linearizing ``du/dt = (-I + W_lin) u`` gives the
   continuous-time generator ``M = -I + W_lin``. A mode grows iff
   ``Re(eig(M)) > 0`` and decays iff ``Re(eig(M)) < 0``. Comparing a raw weight
   eigenvalue against ``1`` is the discrete-map heuristic and is *not* the right
   criterion for the leaky continuous-time system.

2. **An antisymmetric generator produces norm-preserving motion only under the
   stated continuous-time assumptions.** For ``dx/dt = A x`` with ``A^T = -A``,
   ``exp(A t)`` is orthogonal, so ``||x(t)||`` is exactly conserved. The discrete
   Euler map ``x + (dt) A x`` is *not* norm-preserving (its norm grows like
   ``1 + (dt)^2 ||A x||^2 / ...``); the conservation is a continuous-time fact, and
   the code labels it as such.

**Asymptotic stability is not Euclidean contraction (nonnormal caveat).** The model
studies symmetric-plus-antisymmetric, generally *nonnormal* generators. For a
nonnormal ``M`` a negative spectral abscissa (all eigenvalues in the left half
plane) guarantees only that trajectories decay *eventually*; the norm can grow
transiently by an arbitrarily large factor first. Monotone Euclidean contraction is
a stronger property governed by the *symmetric part* ``(M + M^T)/2`` — its largest
eigenvalue is the logarithmic norm (log-norm) ``mu_2(M)``, and ``d||x||^2/dt =
2 x^T M x = x^T (M + M^T) x``, so ``||x(t)||`` is nonincreasing for every ``x`` iff
``mu_2(M) <= 0``. The two tests are therefore separate functions:
:func:`is_asymptotically_stable` (spectral abscissa) and :func:`is_contractive`
(log-norm).

These helpers are analysis utilities (used by tests and later figures); they are
not part of the trained recurrence.
"""

from __future__ import annotations

import torch


def continuous_generator(w_lin: torch.Tensor) -> torch.Tensor:
    """Continuous-time generator ``M = -I + W_lin`` of the leaky linear dynamics.

    ``W_lin`` is the effective *linearized* operator acting on the state. At a fixed
    point ``u*`` of the nonlinear recurrence the correct linearization already
    absorbs the activation Jacobian: ``W_lin = W_eff @ diag(phi'(u*))`` (see
    :func:`jacobian_generator`). Passing a raw ``S``/``W_eff`` treats ``phi`` as
    locally the identity, which is only valid where ``phi'(u*) = 1``.
    """
    n = w_lin.shape[-1]
    identity = torch.eye(n, dtype=w_lin.dtype, device=w_lin.device)
    return -identity + w_lin


def jacobian_generator(w_eff: torch.Tensor, phi_prime: torch.Tensor) -> torch.Tensor:
    """Linearized continuous generator at a fixed point: ``M = -I + W_eff diag(phi')``.

    ``phi_prime`` is the vector ``phi'(u*)`` of activation derivatives at the fixed
    point. This is the operator whose spectrum/log-norm governs local stability of
    the *nonlinear* recurrence ``eq:attractor-discrete`` (up to the overall ``1/tau``
    scale, which does not change the sign of any eigenvalue for ``tau > 0``).
    """
    return continuous_generator(w_eff * phi_prime.unsqueeze(-2))


def growth_rates(w_lin: torch.Tensor) -> torch.Tensor:
    """Real parts of the eigenvalues of the continuous generator ``M = -I + W_lin``.

    A mode grows iff its value here is ``> 0`` and decays iff ``< 0`` (correction 1
    above). Returned sorted ascending; the largest entry is the spectral abscissa.
    """
    eigvals = torch.linalg.eigvals(continuous_generator(w_lin))
    return torch.sort(eigvals.real).values


def spectral_abscissa(w_lin: torch.Tensor) -> float:
    """Largest real part of ``eig(M)``, ``M = -I + W_lin`` (asymptotic decay rate)."""
    return float(growth_rates(w_lin).max().item())


def log_norm(w_lin: torch.Tensor) -> float:
    """Logarithmic norm (log-norm) ``mu_2(M)`` = largest eigenvalue of ``(M + M^T)/2``.

    Governs *instantaneous* Euclidean norm change: ``d||x||/dt <= mu_2(M) ||x||`` for
    all ``x``. ``mu_2 <= 0`` iff the flow is nonexpansive in the 2-norm. Always at
    least the spectral abscissa, with equality iff ``M`` is normal.
    """
    m = continuous_generator(w_lin)
    sym = 0.5 * (m + m.transpose(-1, -2))
    return float(torch.linalg.eigvalsh(sym).max().item())


def _check_tol(tol: float) -> None:
    if not (tol >= 0.0):  # also rejects NaN, since `NaN >= 0` is False
        raise ValueError(f"tol must be a non-negative real number, got {tol!r}")


def is_asymptotically_stable(w_lin: torch.Tensor, *, tol: float = 0.0) -> bool:
    """True iff every mode strictly decays (spectral abscissa ``< -tol``).

    Asymptotic linear stability, **not** monotone Euclidean contraction: a nonnormal
    ``M`` can satisfy this while its norm grows transiently (use :func:`is_contractive`
    for the stronger property). ``tol >= 0`` is a strictness margin (``tol=0`` is the
    open condition ``< 0``); it is required non-negative.
    """
    _check_tol(tol)
    return spectral_abscissa(w_lin) < -tol


def is_contractive(w_lin: torch.Tensor, *, tol: float = 0.0) -> bool:
    """True iff the 2-norm is **strictly** contracting (log-norm ``mu_2(M) < -tol``).

    Tests the largest eigenvalue of the symmetric part ``(M + M^T)/2`` (the log-norm
    ``mu_2(M)``), the correct condition for Euclidean contraction of the generally
    *nonnormal* generator ``M = -I + W_lin``. This is **strict** contraction: a
    norm-preserving flow at the boundary (``mu_2 = 0``) is not contractive and returns
    False. For the nonexpansive (nonincreasing, ``mu_2 <= 0``) boundary use
    :func:`is_nonexpansive`. ``tol >= 0`` required. Strictly stronger than
    :func:`is_asymptotically_stable`.
    """
    _check_tol(tol)
    return log_norm(w_lin) < -tol


def is_nonexpansive(w_lin: torch.Tensor, *, tol: float = 0.0) -> bool:
    """True iff the 2-norm is nonincreasing (log-norm ``mu_2(M) <= tol``).

    The boundary-inclusive counterpart of :func:`is_contractive`: a norm-preserving
    flow (``mu_2 = 0``) is nonexpansive but not strictly contractive. ``tol >= 0`` is a
    small nonnegative slack.
    """
    _check_tol(tol)
    return log_norm(w_lin) <= tol


def euler_step_matrix(a_mat: torch.Tensor, dt: float) -> torch.Tensor:
    """Discrete Euler transport map ``I + dt * A`` for a pure antisymmetric ``A``.

    Provided to demonstrate (in tests) that the *discrete* step is **not**
    norm-preserving even when ``A`` is antisymmetric — norm preservation is the
    continuous-time (matrix-exponential) property, not the Euler-map property.
    """
    n = a_mat.shape[-1]
    identity = torch.eye(n, dtype=a_mat.dtype, device=a_mat.device)
    return identity + dt * a_mat


def flow_matrix(a_mat: torch.Tensor, t: float) -> torch.Tensor:
    """Continuous-time flow ``exp(A t)`` of a pure antisymmetric generator ``A``.

    For ``A^T = -A`` this is orthogonal, so it preserves the L2 norm exactly
    (correction 2 above). Used in tests to contrast with :func:`euler_step_matrix`.
    """
    flow: torch.Tensor = torch.linalg.matrix_exp(a_mat * t)
    return flow
