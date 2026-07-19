"""Group sparse coding: the formalized, tested mathematics of revised Chapter 3.

This package is the executable specification produced in Phase 3A (plan §7): the
group penalty and its gradients (:mod:`~neural_repr.sparse_coding.groups`), the
dictionary model (:mod:`~neural_repr.sparse_coding.dictionary`), the two named
inference variants (:mod:`~neural_repr.sparse_coding.inference`), the denoising
training objective (:mod:`~neural_repr.sparse_coding.denoising`), and the
corrected Gabor analysis model (:mod:`~neural_repr.sparse_coding.gabor`).

No training or data loading lives here — those are Phases 4+. See
``docs/math-spec-sparse-coding.md`` for the standalone math spec and equation-label
map back to ``legacy/rewrite-2026/source``.
"""

from __future__ import annotations

from neural_repr.sparse_coding.denoising import (
    add_gaussian_noise,
    denoising_loss,
    ml_group_grad,
)
from neural_repr.sparse_coding.dictionary import (
    dictionary_grad,
    normalize_columns,
    reconstruct,
    reconstruction_error,
)
from neural_repr.sparse_coding.gabor import GaborParams, gabor
from neural_repr.sparse_coding.groups import (
    DEFAULT_EPS,
    group_activity,
    group_penalty,
    group_penalty_grad,
    group_penalty_grad_lambda,
    softplus_lambda,
)
from neural_repr.sparse_coding.inference import (
    data_grad,
    data_lipschitz,
    infer,
    legacy_adaptive_threshold,
    lipschitz_step_size,
    penalty_lipschitz,
    smooth_gradient,
    soft_threshold,
    stable_step_size,
)

__all__ = [
    "DEFAULT_EPS",
    "GaborParams",
    "add_gaussian_noise",
    "data_grad",
    "data_lipschitz",
    "denoising_loss",
    "dictionary_grad",
    "gabor",
    "group_activity",
    "group_penalty",
    "group_penalty_grad",
    "group_penalty_grad_lambda",
    "infer",
    "legacy_adaptive_threshold",
    "lipschitz_step_size",
    "ml_group_grad",
    "normalize_columns",
    "penalty_lipschitz",
    "reconstruct",
    "reconstruction_error",
    "smooth_gradient",
    "soft_threshold",
    "softplus_lambda",
    "stable_step_size",
]
