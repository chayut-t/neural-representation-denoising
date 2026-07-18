"""Schema for the sanitized execution-environment record (plan §6.1).

A small dependency-free validator: it checks required keys, types, and — critically
— that the record carries the three §6.1 provenance fields and that ``install_mode``
is a known category (not a free-form identifier). This is the "provenance-schema"
check wired into CI and the container smoke.
"""

from __future__ import annotations

from typing import Any

from neural_repr.provenance.environment import INSTALL_MODES

# Required top-level fields and their acceptable Python types.
REQUIRED_FIELDS: dict[str, type | tuple[type, ...]] = {
    "schema_version": str,
    "python_version": str,
    "python_implementation": str,
    "os_system": str,
    "os_release": str,
    "machine": str,
    "processor": str,
    "package_versions": dict,
    "torch": dict,
    "determinism_mode": str,
    "install_mode": str,
    "execution_environment_fingerprint": str,
    "reference_compatibility_status": str,
    # may be None, so allow both
    "public_reference_environment_digest": (str, type(None)),
}


class SchemaError(ValueError):
    """Raised when an execution-environment record violates the §6.1 schema."""


def validate_execution_record(record: dict[str, Any]) -> None:
    """Validate a record dict; raise :class:`SchemaError` on the first problem."""
    for key, expected in REQUIRED_FIELDS.items():
        if key not in record:
            raise SchemaError(f"missing required field: {key!r}")
        if not isinstance(record[key], expected):
            raise SchemaError(
                f"field {key!r} has type {type(record[key]).__name__}, expected {expected}"
            )

    if record["install_mode"] not in INSTALL_MODES:
        raise SchemaError(
            f"install_mode {record['install_mode']!r} is not a known category {INSTALL_MODES}"
        )

    if not record["execution_environment_fingerprint"].startswith("sha256:"):
        raise SchemaError("execution_environment_fingerprint must be a 'sha256:' digest")

    # package_versions must map names -> version strings.
    for name, version in record["package_versions"].items():
        if not isinstance(name, str) or not isinstance(version, str):
            raise SchemaError(f"package_versions entry {name!r} is not str->str")
