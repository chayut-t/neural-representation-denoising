"""Closed schema for the sanitized execution-environment record (plan §6.1, §2.4).

This validator is the *primary* non-disclosure control (plan Phase 2 task 10,
control 1): it is a closed allow-list at EVERY level — it rejects unknown top-level
keys AND unknown nested keys inside ``torch``/``runtime``/``determinism_flags``, so
an infrastructure-shaped field (``hostname``, ``registry_uri``, ``account_id``, an
absolute path, …) cannot ride along even if some future code path added it. It also
rejects any string value that looks infrastructure-shaped (absolute path, registry
URL, ARN/account id), constrains enums and digest formats, enforces a conditional
rule that a public-reference record carries the lock digest and non-empty
environment facts, and recomputes the environment fingerprint against its canonical
payload so a record whose fingerprint does not match its sanitized content is
rejected.
"""

from __future__ import annotations

import math
import re
from typing import Any

from neural_repr.provenance.environment import (
    COMPAT_STATUSES,
    DETERMINISM_MODES,
    INSTALL_MODES,
    SCHEMA_VERSION,
    environment_fingerprint,
)
from neural_repr.provenance.leak_patterns import line_has_leak

_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")

# Exact set of permitted top-level fields and their acceptable types.
_ALLOWED_FIELDS: dict[str, type | tuple[type, ...]] = {
    "schema_version": str,
    "python_version": str,
    "python_implementation": str,
    "os_system": str,
    "os_release": str,
    "machine": str,
    "processor": str,
    "package_versions": dict,
    "torch": dict,
    "runtime": dict,
    "determinism_flags": dict,
    "uv_lock_sha256": (str, type(None)),
    "determinism_mode": str,
    "install_mode": str,
    "reference_compatibility_status": str,
    "public_reference_environment_digest": (str, type(None)),
    "execution_environment_fingerprint": str,
}

# Closed nested schemas: the EXACT permitted keys inside each nested dict. Unknown
# nested keys are rejected (so `runtime.hostname` / `torch.registry_uri` cannot ride
# along). Values are type-checked; string values are additionally leak-scanned.
# Each nested field is one of these kinds, so a value is type-checked (not merely
# key-checked). "int" rejects bool (a bool is an int subtype); "float" accepts int
# or float but rejects bool and non-finite; "str" values are leak-scanned; "digest"
# must be a sha256 token. All fields are nullable (a fact may be unavailable).
_FieldKind = str  # one of: "str", "digest", "int", "float", "bool"

_NESTED_FIELD_SCHEMA: dict[str, dict[str, _FieldKind]] = {
    "torch": {
        "torch_version": "str",
        "torch_cuda_version": "str",
        "cuda_available": "bool",
        "cudnn_version": "int",
        "driver_version": "str",
        "gpu_model_class": "str",
        "gpu_compute_capability": "str",
        "gpu_total_memory_gib": "float",
    },
    "runtime": {
        "cpu_count": "int",
        "total_ram_gib": "float",
        "blas_impl": "str",
        "cxx_version": "str",
        "openmp_enabled": "bool",
        "torch_num_threads": "int",
        "torch_num_interop_threads": "int",
        "torch_build_config_sha256": "digest",
    },
    "determinism_flags": {
        "deterministic_algorithms": "bool",
        "cudnn_deterministic": "bool",
        "cudnn_benchmark": "bool",
    },
}


class SchemaError(ValueError):
    """Raised when an execution-environment record violates the §6.1 schema."""


def _require_sha256(value: Any, label: str) -> None:
    if not isinstance(value, str) or not _SHA256_RE.match(value):
        raise SchemaError(f"{label} must be 'sha256:' + 64 hex chars, got {value!r}")


def _reject_infra_shaped_string(value: str, label: str) -> None:
    """Reject a string value that matches any generic infrastructure-leak pattern."""
    hit = line_has_leak(value)
    if hit is not None:
        raise SchemaError(f"{label} looks infrastructure-shaped ([{hit}]): {value!r}")


def _check_field(label: str, kind: _FieldKind, value: Any) -> None:
    """Type/format-check one nested field value (None is always allowed)."""
    if value is None:
        return
    if kind == "bool":
        if not isinstance(value, bool):
            raise SchemaError(f"{label} must be bool, got {type(value).__name__}")
    elif kind == "int":
        # bool is an int subtype in Python; reject it where an int is intended.
        if isinstance(value, bool) or not isinstance(value, int):
            raise SchemaError(f"{label} must be int, got {type(value).__name__}")
    elif kind == "float":
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise SchemaError(f"{label} must be a real number, got {type(value).__name__}")
        if not math.isfinite(float(value)):
            raise SchemaError(f"{label} must be finite, got {value!r}")
    elif kind == "digest":
        _require_sha256(value, label)
    elif kind == "str":
        if not isinstance(value, str):
            raise SchemaError(f"{label} must be str, got {type(value).__name__}")
        _reject_infra_shaped_string(value, label)
    else:  # pragma: no cover - guards a typo in the schema table
        raise SchemaError(f"{label}: unknown field kind {kind!r}")


def _validate_nested(parent: str, obj: dict[str, Any]) -> None:
    """Closed field-schema check of a nested dict: known keys AND correct value types."""
    schema = _NESTED_FIELD_SCHEMA[parent]
    unknown = set(obj) - set(schema)
    if unknown:
        raise SchemaError(f"unknown nested field(s) in {parent!r}: {sorted(unknown)}")
    for key, value in obj.items():
        _check_field(f"{parent}.{key}", schema[key], value)


# reference_compatibility_status values that denote a canonical scientific record.
_PUBLIC_REFERENCE_STATUSES = frozenset({"public-reference-run", "equivalence-verified"})


def _check_public_reference_completeness(record: dict[str, Any]) -> None:
    """A public-reference / equivalence record must be complete enough to reproduce.

    For such records the committed artifact is the reproducibility contract (§2.4),
    so it must carry the ``uv.lock`` digest (dependency identity) and non-empty
    ``torch`` and ``runtime`` facts. An incomplete record (empty nested objects, no
    lock digest) is rejected — it cannot stand in for a canonical reference run.
    Other statuses (``unknown``/``pending-equivalence``) are allowed to be partial.
    """
    if record["reference_compatibility_status"] not in _PUBLIC_REFERENCE_STATUSES:
        return
    if record["uv_lock_sha256"] is None:
        raise SchemaError(
            "a public-reference/equivalence record must include uv_lock_sha256 "
            "(dependency identity), got null"
        )
    for parent in ("torch", "runtime"):
        if not record[parent]:
            raise SchemaError(
                f"a public-reference/equivalence record must include non-empty {parent!r} facts"
            )


def validate_execution_record(record: dict[str, Any]) -> None:
    """Validate a record dict against the closed schema; raise on the first problem."""
    if not isinstance(record, dict):
        raise SchemaError(f"record must be a dict, got {type(record).__name__}")

    # Closed allow-list: reject any unknown top-level key (the primary control).
    unknown = set(record) - set(_ALLOWED_FIELDS)
    if unknown:
        raise SchemaError(f"unknown field(s) not permitted in record: {sorted(unknown)}")

    # Required-field presence + types.
    for key, expected in _ALLOWED_FIELDS.items():
        if key not in record:
            raise SchemaError(f"missing required field: {key!r}")
        if not isinstance(record[key], expected):
            raise SchemaError(
                f"field {key!r} has type {type(record[key]).__name__}, expected {expected}"
            )

    # Enum / value-format constraints.
    if record["schema_version"] != SCHEMA_VERSION:
        raise SchemaError(
            f"schema_version {record['schema_version']!r} != expected {SCHEMA_VERSION!r}"
        )
    if record["install_mode"] not in INSTALL_MODES:
        raise SchemaError(f"install_mode {record['install_mode']!r} not in {INSTALL_MODES}")
    if record["determinism_mode"] not in DETERMINISM_MODES:
        raise SchemaError(
            f"determinism_mode {record['determinism_mode']!r} not in {DETERMINISM_MODES}"
        )
    if record["reference_compatibility_status"] not in COMPAT_STATUSES:
        raise SchemaError(
            f"reference_compatibility_status {record['reference_compatibility_status']!r} "
            f"not in {COMPAT_STATUSES}"
        )

    _require_sha256(
        record["execution_environment_fingerprint"], "execution_environment_fingerprint"
    )
    if record["uv_lock_sha256"] is not None:
        _require_sha256(record["uv_lock_sha256"], "uv_lock_sha256")
    if record["public_reference_environment_digest"] is not None:
        _require_sha256(
            record["public_reference_environment_digest"], "public_reference_environment_digest"
        )

    # package_versions must map names -> version strings.
    for name, version in record["package_versions"].items():
        if not isinstance(name, str) or not isinstance(version, str):
            raise SchemaError(f"package_versions entry {name!r} is not str->str")

    # Closed nested field schemas: reject unknown nested keys AND wrong-typed values
    # (bool-as-int, non-finite floats, containers), leak-scan strings, digest fields.
    for parent in _NESTED_FIELD_SCHEMA:
        _validate_nested(parent, record[parent])

    # Conditional rule: a public-reference / equivalence record is a canonical
    # scientific artifact, so it must carry the lock identity and non-empty
    # environment facts (an empty torch/runtime record cannot stand in for it).
    _check_public_reference_completeness(record)

    # Fingerprint consistency: recompute from the canonical sanitized payload and
    # require it to match. This is what makes tampering / drift detectable.
    payload = {
        k: v
        for k, v in record.items()
        if k
        not in (
            "execution_environment_fingerprint",
            "public_reference_environment_digest",
            "reference_compatibility_status",
        )
    }
    expected_fp = environment_fingerprint(payload)
    if record["execution_environment_fingerprint"] != expected_fp:
        raise SchemaError(
            "execution_environment_fingerprint does not match canonical sanitized content"
        )
