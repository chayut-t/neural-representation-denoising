"""Noise-robust recurrent location codes: the formalized math of revised Chapter 4.

This package is the executable specification produced in Phase 3B (plan §7):
structured weights built by parameterization
(:mod:`~neural_repr.attractor.weights`), the discrete recurrence with an explicit
``dt/tau`` and the write/hold/move operations
(:mod:`~neural_repr.attractor.dynamics`), prescribed place/ring/mixed codes and the
population-vector decoder (:mod:`~neural_repr.attractor.codes`), explicit noise
resampling protocols (:mod:`~neural_repr.attractor.noise`), trajectory helpers
(:mod:`~neural_repr.attractor.trajectory`), and linear-stability analysis encoding
the continuous-time corrections (:mod:`~neural_repr.attractor.analysis`).

No training or data loading lives here — those are later phases. See
``docs/math-spec-attractor.md`` for the standalone math spec and equation-label map
back to ``legacy/rewrite-2026/source``.
"""

from __future__ import annotations

from neural_repr.attractor.analysis import (
    continuous_generator,
    euler_step_matrix,
    flow_matrix,
    growth_rates,
    is_asymptotically_stable,
    is_contractive,
    is_nonexpansive,
    jacobian_generator,
    log_norm,
    spectral_abscissa,
)
from neural_repr.attractor.codes import (
    DEFAULT_WIDTH,
    MixedCodeSpec,
    circular_distance,
    decode_place,
    line_centers,
    mixed_code,
    place_code,
    ring_centers,
    ring_code,
)
from neural_repr.attractor.dynamics import (
    Nonlinearity,
    Velocity,
    log_rectifier,
    step,
    write_mix,
)
from neural_repr.attractor.noise import (
    WeightNoiseSampler,
    gaussian_weight_perturbation,
    state_noise,
    uniform_weight_perturbation,
)
from neural_repr.attractor.trajectory import hold, move, run, write
from neural_repr.attractor.weights import (
    antisymmetric_part,
    effective_weight,
    symmetric_part,
)

__all__ = [
    "DEFAULT_WIDTH",
    "MixedCodeSpec",
    "Nonlinearity",
    "Velocity",
    "WeightNoiseSampler",
    "antisymmetric_part",
    "circular_distance",
    "continuous_generator",
    "decode_place",
    "effective_weight",
    "euler_step_matrix",
    "flow_matrix",
    "gaussian_weight_perturbation",
    "growth_rates",
    "hold",
    "is_asymptotically_stable",
    "is_contractive",
    "is_nonexpansive",
    "jacobian_generator",
    "line_centers",
    "log_norm",
    "log_rectifier",
    "mixed_code",
    "move",
    "place_code",
    "ring_centers",
    "ring_code",
    "run",
    "spectral_abscissa",
    "state_noise",
    "step",
    "symmetric_part",
    "uniform_weight_perturbation",
    "write",
    "write_mix",
]
