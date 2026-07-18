"""Provenance: sanitized execution-environment records and fingerprints.

Implements the two-layer provenance model from the plan (§2.4, §6.1):

- Layer 1 (committed, sanitized): full scientific detail — resolved Python and
  package versions, PyTorch build string, CUDA/cuDNN/driver, GPU model class and
  VRAM, OS/CPU/RAM, determinism flags, install-mode *category*, and an opaque
  ``execution_environment_fingerprint`` — with NO infrastructure identifiers.
- Layer 2 (non-public): the mapping from fingerprint to raw operational detail
  lives in an access-controlled store, never in this repository.

This module produces only Layer-1 content. See ``environment`` for the record
and fingerprint helpers.
"""

from __future__ import annotations

from neural_repr.provenance.environment import (
    ExecutionEnvironment,
    collect_execution_environment,
    environment_fingerprint,
)
from neural_repr.provenance.schema import SchemaError, validate_execution_record

__all__ = [
    "ExecutionEnvironment",
    "SchemaError",
    "collect_execution_environment",
    "environment_fingerprint",
    "validate_execution_record",
]
