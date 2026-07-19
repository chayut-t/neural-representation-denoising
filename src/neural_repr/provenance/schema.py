"""Closed schema for the sanitized execution-environment record (plan §6.1, §2.4).

This validator is the *primary* non-disclosure control (plan Phase 2 task 10,
control 1): it is a closed allow-list — it rejects unknown top-level keys, so an
infrastructure-shaped field (``hostname``, ``registry_uri``, ``account_id``, an
absolute path, …) cannot ride along even if some future code path added it. It
also constrains enums and digest formats and recomputes the environment
fingerprint against its canonical payload, so a record whose fingerprint does not
match its sanitized content is rejected.
"""

from __future__ import annotations

import re
from typing import Any

from neural_repr.provenance.environment import (
    COMPAT_STATUSES,
    DETERMINISM_MODES,
    INSTALL_MODES,
    SCHEMA_VERSION,
    environment_fingerprint,
)

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

# Optional digest-shaped fields nested inside `torch`/`runtime` that, when present,
# must be valid sha256 tokens.
_NESTED_DIGEST_FIELDS = {
    "runtime": ("torch_build_config_sha256",),
}


class SchemaError(ValueError):
    """Raised when an execution-environment record violates the §6.1 schema."""


def _require_sha256(value: Any, label: str) -> None:
    if not isinstance(value, str) or not _SHA256_RE.match(value):
        raise SchemaError(f"{label} must be 'sha256:' + 64 hex chars, got {value!r}")


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

    # Nested digest-shaped fields, when present, must be valid sha256 tokens.
    for parent, keys in _NESTED_DIGEST_FIELDS.items():
        nested = record.get(parent, {})
        for k in keys:
            if k in nested and nested[k] is not None:
                _require_sha256(nested[k], f"{parent}.{k}")

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
