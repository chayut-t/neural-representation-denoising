"""Denoising objective for learning group structure, and the ML surrogate note.

Formalizes §3.4-§3.5 of the revised Chapter 3. The group matrix ``Lambda`` is
trained to reduce the reconstruction of a *clean* patch from a *corrupted* one
after finite inference (``eq:group-denoising-loss``)::

    x_tilde = x + nu,   nu ~ N(0, sigma^2 I)          (eq:group-corruption)
    s_T     = F^{(T)}(x_tilde; Phi, Lambda)
    x_hat   = Phi s_T
    L_denoise = || x - x_hat ||^2.

Automatic differentiation propagates the error through the decoder and all
inference steps, replacing the two intractable expectations of the maximum-
likelihood gradient (``eq:lambda-likelihood-gradient``) with a supervised target
manufactured from the clean observation.

The distinction matters and is preserved here (baseline §3.4-§3.5, decision 0004):
the denoising ``Lambda`` is optimized for the finite inference rule, corruption
distribution, loss, and schedule. It is **not** claimed to be the maximum-
likelihood group structure — that is a different estimand. :func:`ml_group_grad`
returns the exact positive/negative-phase ML gradient *integrand* (with the ``1/2``
factor) only to document the surrogate relationship; it is not used for training.
"""

from __future__ import annotations

import torch

from neural_repr.sparse_coding.dictionary import reconstruct
from neural_repr.sparse_coding.groups import DEFAULT_EPS, group_penalty_grad_lambda


def add_gaussian_noise(
    x: torch.Tensor, sigma: float, *, generator: torch.Generator | None = None
) -> torch.Tensor:
    """Corruption process ``x_tilde = x + nu``, ``nu ~ N(0, sigma^2 I)`` (``eq:group-corruption``).

    ``sigma`` is the noise standard deviation. Pass a seeded ``generator`` for
    reproducible corruption.
    """
    if sigma < 0.0:
        raise ValueError("sigma must be non-negative")
    noise = torch.randn(x.shape, dtype=x.dtype, device=x.device, generator=generator)
    return x + sigma * noise


def denoising_loss(x: torch.Tensor, s_t: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    """Denoising reconstruction loss ``|| x - Phi s_T ||^2`` (``eq:group-denoising-loss``).

    Summed over the feature axis and averaged over the batch, so the scalar is a
    mean per-patch squared error (a stable training target across batch sizes).
    """
    residual = x - reconstruct(phi, s_t)
    per_item = (residual * residual).sum(dim=-1)
    return per_item.mean()


def ml_group_grad(
    s: torch.Tensor, lambda_: torch.Tensor, *, eps: float = DEFAULT_EPS
) -> torch.Tensor:
    """Per-sample ML integrand ``d Omega / d Lambda_ki = s_i^2 / (2 z_k)`` (appendix).

    The exact maximum-likelihood gradient (``eq:lambda-likelihood-gradient``) is the
    difference of this integrand under the model prior and the posterior; both
    expectations are intractable. We expose only the integrand — **with the exact
    ``1/2`` factor** (``b26-corr-ml-half``) — to document what the denoising
    objective replaces. Returns shape ``(..., K, M)``. Not used for training.
    """
    return group_penalty_grad_lambda(s, lambda_, eps=eps)
