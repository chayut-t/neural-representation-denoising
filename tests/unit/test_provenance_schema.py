"""Tests for the provenance-schema validator (plan §6.1; Gate P2 provenance-schema clause)."""

from __future__ import annotations

import copy

import pytest

from neural_repr.provenance import (
    SchemaError,
    collect_execution_environment,
    validate_execution_record,
)


def _valid_record() -> dict:
    return collect_execution_environment(install_mode="local").to_record()


def test_real_record_validates() -> None:
    validate_execution_record(_valid_record())  # must not raise


def test_missing_field_rejected() -> None:
    rec = _valid_record()
    del rec["execution_environment_fingerprint"]
    with pytest.raises(SchemaError, match="execution_environment_fingerprint"):
        validate_execution_record(rec)


def test_unknown_install_mode_rejected() -> None:
    rec = _valid_record()
    rec["install_mode"] = "some-cluster-name"  # an identifier, not a category
    with pytest.raises(SchemaError, match="install_mode"):
        validate_execution_record(rec)


def test_bad_fingerprint_prefix_rejected() -> None:
    rec = _valid_record()
    rec["execution_environment_fingerprint"] = "not-a-digest"
    with pytest.raises(SchemaError, match="sha256"):
        validate_execution_record(rec)


def test_wrong_type_rejected() -> None:
    rec = _valid_record()
    rec["package_versions"] = ["not", "a", "dict"]
    with pytest.raises(SchemaError, match="package_versions"):
        validate_execution_record(rec)


def test_null_reference_digest_allowed() -> None:
    rec = _valid_record()
    assert rec["public_reference_environment_digest"] is None
    validate_execution_record(rec)  # None is permitted


def test_deepcopy_roundtrip_still_valid() -> None:
    validate_execution_record(copy.deepcopy(_valid_record()))
