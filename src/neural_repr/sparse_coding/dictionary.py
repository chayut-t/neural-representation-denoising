"""Dictionary reconstruction, column normalization, and its gradient.

Formalizes the linear generative model of the sparse-coding primer
(``chap2.tex``, ``eq:sparse-generative``) and the dictionary update of Chapter 3
(``chap3.tex``, ``eq:dictionary-gradient``).

An observation is modeled as ``x = Phi s + epsilon`` with dictionary
``Phi`` of shape ``(N, M)`` (columns are atoms) and coefficients ``s`` of shape
``(..., M)``. The dictionary--coefficient scale ambiguity is removed by
constraining each column to unit L2 norm (``||phi_j||_2 = 1``); we normalize
after each update rather than penalizing, matching the baseline text.
"""

from __future__ import annotations

import torch


def reconstruct(phi: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """Linear reconstruction ``x_hat = Phi s``.

    ``phi`` has shape ``(N, M)``; ``s`` has shape ``(..., M)``; result ``(..., N)``.
    """
    return s @ phi.transpose(-1, -2)


def normalize_columns(
    phi: torch.Tensor, *, eps: float = 1e-12, on_zero: str = "raise"
) -> torch.Tensor:
    """Rescale each dictionary column to unit L2 norm (``||phi_j||_2 = 1``).

    Removes the dictionary--coefficient scale ambiguity (primer §sparse,
    Chapter 3 §3.3.1). A zero (or near-zero, ``<= eps``) column cannot be scaled to
    unit norm, so the spec's "every column is unit norm" would be violated silently.
    ``on_zero`` makes the policy explicit:

    * ``"raise"`` (default) — raise ``ValueError``; a zero atom is a caller bug.
    * ``"keep"`` — leave the zero column at zero (the previous lenient behavior),
      for callers that deliberately allow dead atoms. The result is then *not*
      strictly unit-norm on those columns.

    Nonzero columns are always rescaled exactly to unit norm.
    """
    if on_zero not in {"raise", "keep"}:
        raise ValueError(f"on_zero must be 'raise' or 'keep', got {on_zero!r}")
    if not eps > 0.0:  # a non-positive/NaN eps would let a zero column produce NaNs
        raise ValueError(f"eps must be a positive real number, got {eps!r}")
    norms = torch.linalg.vector_norm(phi, dim=0, keepdim=True)
    if on_zero == "raise" and bool((norms <= eps).any()):
        raise ValueError(
            "dictionary has a zero/degenerate column; cannot normalize to unit norm "
            "(pass on_zero='keep' to leave dead atoms at zero)"
        )
    normalized: torch.Tensor = phi / norms.clamp_min(eps)
    return normalized


def reconstruction_error(phi: torch.Tensor, s: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Half squared reconstruction error ``0.5 ||x - Phi s||^2`` (per batch item).

    This is the data-fidelity term of the inference objective
    (``eq:group-objective`` / ``eq:sparse-objective``). Returns the batch shape of
    ``x`` with the feature axis reduced.
    """
    residual = x - reconstruct(phi, s)
    return 0.5 * (residual * residual).sum(dim=-1)


def dictionary_grad(phi: torch.Tensor, s: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Negative gradient of the data term w.r.t. ``Phi`` for a fixed code.

    Implements ``-dE/dPhi = (x - Phi s) s^T`` (``eq:dictionary-gradient``). For a
    batch of ``(x, s)`` pairs the per-item outer products are summed, giving a
    ``(N, M)`` update direction (ascent direction on reconstruction quality).
    """
    residual = x - reconstruct(phi, s)  # (..., N)
    # sum over batch of outer(residual, s): (..., N, 1) * (..., 1, M) -> (N, M)
    outer = residual.unsqueeze(-1) * s.unsqueeze(-2)
    if outer.dim() > 2:
        outer = outer.reshape(-1, outer.shape[-2], outer.shape[-1]).sum(dim=0)
    return outer
