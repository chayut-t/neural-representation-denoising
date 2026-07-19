"""Group activity, structured penalty, and its analytic gradients.

Formalizes the group-sparse-coding penalty of revised Chapter 3
(``legacy/rewrite-2026/source/chap3.tex`` and the primer ``chap2.tex``). Symbols
and equation labels match the LaTeX so the math spec, code, and dissertation agree
(plan Gate P3). See ``docs/math-spec-sparse-coding.md`` for the standalone spec.

Numerically safe group activity (``eq:group-activity``)::

    z_k(s) = sqrt(eps + sum_i Lambda_ki s_i^2),   Lambda_ki >= 0,

structured penalty (``eq:group-penalty``) ``Omega(s) = sum_k z_k(s)``, and its
coefficient gradient (``eq:group-coefficient-gradient``)::

    d Omega / d s_i = s_i * sum_k Lambda_ki / z_k(s).

The right-hand side is a *signed gradient component*, not a nonnegative adaptive
soft threshold (baseline §3.3.1 correction ``b26-corr-group-grad``, decision 0004);
that distinction is what separates the two inference variants in ``inference.py``.

Conventions
-----------
* ``eps > 0`` is a documented numerical smoothing constant (plan 3A.2): it keeps
  ``z_k`` and its gradient finite when a group is silent (``s = 0``).
* ``Lambda`` is nonnegative. :func:`softplus_lambda` maps an unconstrained real
  matrix to a nonnegative one; row-sum normalization resolves the scale
  non-identifiability (plan 3A.3, decision 0004), and the chosen convention is
  recorded there.

All tensors are batched in leading dimensions: ``s`` has shape ``(..., M)`` and
``Lambda`` has shape ``(K, M)``; activities have shape ``(..., K)``.
"""

from __future__ import annotations

from typing import Literal, get_args

import torch

# Default numerical smoothing constant inside the group square roots. Documented
# as a numerical parameter (plan 3A.2), not a fitted quantity.
DEFAULT_EPS: float = 1e-6

LambdaNormalization = Literal["row_sum", "none"]
_LAMBDA_NORMALIZATIONS: tuple[str, ...] = get_args(LambdaNormalization)


def group_activity(
    s: torch.Tensor, lambda_: torch.Tensor, *, eps: float = DEFAULT_EPS
) -> torch.Tensor:
    """Group activities ``z_k(s)`` (``eq:group-activity``).

    Parameters
    ----------
    s:
        Coefficient vector(s), shape ``(..., M)``.
    lambda_:
        Nonnegative group-weight matrix ``Lambda``, shape ``(K, M)``. Each row is
        a group; a coefficient may belong to several rows, so groups may overlap.
    eps:
        Numerical smoothing constant ``eps > 0``.

    Returns
    -------
    torch.Tensor
        Activities ``z`` with shape ``(..., K)``.
    """
    if eps <= 0.0:
        raise ValueError("eps must be positive to keep the group gradient finite at s=0")
    _check_nonnegative(lambda_)
    # (..., M) @ (M, K) -> (..., K)
    weighted = (s * s) @ lambda_.transpose(-1, -2)
    return torch.sqrt(eps + weighted)


def group_penalty(
    s: torch.Tensor, lambda_: torch.Tensor, *, eps: float = DEFAULT_EPS
) -> torch.Tensor:
    """Structured penalty ``Omega(s) = sum_k z_k(s)`` (``eq:group-penalty``).

    Returns a tensor with the batch shape of ``s`` (no coefficient/group axis).
    """
    return group_activity(s, lambda_, eps=eps).sum(dim=-1)


def group_penalty_grad(
    s: torch.Tensor, lambda_: torch.Tensor, *, eps: float = DEFAULT_EPS
) -> torch.Tensor:
    """Analytic coefficient gradient ``d Omega / d s`` (``eq:group-coefficient-gradient``).

    Returns a tensor shaped like ``s``. This is a signed gradient (it carries the
    sign of ``s_i``), deliberately *not* a nonnegative threshold.
    """
    z = group_activity(s, lambda_, eps=eps)  # (..., K)
    # sum_k Lambda_ki / z_k  ->  (..., M)
    inv_z = 1.0 / z
    coupling = inv_z @ lambda_  # (..., K) @ (K, M) -> (..., M)
    return s * coupling


def group_penalty_grad_lambda(
    s: torch.Tensor, lambda_: torch.Tensor, *, eps: float = DEFAULT_EPS
) -> torch.Tensor:
    """Penalty gradient w.r.t. the group weights, ``d Omega / d Lambda_ki``.

    Equals ``s_i^2 / (2 z_k(s))`` (appendix ``eq:app-group-weight-derivative``).
    The factor ``1/2`` comes from differentiating the square root; it is retained
    exactly (baseline App A correction ``b26-corr-ml-half``, decision 0004).

    Returns shape ``(..., K, M)``.
    """
    z = group_activity(s, lambda_, eps=eps)  # (..., K)
    half_inv_z = 0.5 / z  # (..., K)
    s2 = s * s  # (..., M)
    # outer product over the last two axes: (..., K, 1) * (..., 1, M)
    return half_inv_z.unsqueeze(-1) * s2.unsqueeze(-2)


def softplus_lambda(
    raw: torch.Tensor,
    *,
    normalization: LambdaNormalization = "row_sum",
) -> torch.Tensor:
    """Map an unconstrained real matrix to a nonnegative, scale-fixed ``Lambda``.

    Nonnegativity is enforced with softplus (plan 3A.3). The scale
    non-identifiability of ``(Lambda, lambda)`` — rescaling a group's row can be
    undone by the penalty weight — is resolved by normalizing each group (row) to
    unit sum. The recorded convention (decision 0004) is **softplus + row-sum
    normalization**; pass ``normalization="none"`` to keep the raw softplus values
    (e.g. to reproduce a fixed historical initialization).

    ``normalization`` is validated (an unknown string raises) so a config typo
    cannot silently drop the identifiability constraint.
    """
    if normalization not in _LAMBDA_NORMALIZATIONS:
        raise ValueError(
            f"unknown normalization {normalization!r}; expected one of {_LAMBDA_NORMALIZATIONS}"
        )
    lambda_ = torch.nn.functional.softplus(raw)
    if normalization == "row_sum":
        row_sums = lambda_.sum(dim=-1, keepdim=True)
        lambda_ = lambda_ / row_sums
    return lambda_


def _check_nonnegative(lambda_: torch.Tensor) -> None:
    if bool((lambda_ < 0).any()):
        raise ValueError("Lambda must be nonnegative (eq:group-activity requires Lambda_ki >= 0)")
