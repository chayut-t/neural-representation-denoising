"""Shared utilities: seeding, determinism, and numerical smoke checks."""

from __future__ import annotations

from neural_repr.common.numerics import (
    numerical_smoke,
    seed_everything,
)

__all__ = ["numerical_smoke", "seed_everything"]
