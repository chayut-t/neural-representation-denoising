"""Tests for the closed provenance schema (plan §6.1, §2.4; Gate P2 provenance-schema clause)."""

from __future__ import annotations

import copy

import pytest

from neural_repr.provenance import (
    SchemaError,
    collect_execution_environment,
    validate_execution_record,
)
from neural_repr.provenance.environment import environment_fingerprint


def _valid_record() -> dict:
    return collect_execution_environment(install_mode="local").to_record()


def _reseal(rec: dict) -> dict:
    """Recompute the fingerprint after mutating sanitized fields, so a test can
    target a specific field error rather than tripping the consistency check."""
    payload = {
        k: v
        for k, v in rec.items()
        if k
        not in (
            "execution_environment_fingerprint",
            "public_reference_environment_digest",
            "reference_compatibility_status",
        )
    }
    rec["execution_environment_fingerprint"] = environment_fingerprint(payload)
    return rec


def test_real_record_validates() -> None:
    validate_execution_record(_valid_record())  # must not raise


def test_deepcopy_roundtrip_still_valid() -> None:
    validate_execution_record(copy.deepcopy(_valid_record()))


def test_missing_field_rejected() -> None:
    rec = _valid_record()
    del rec["execution_environment_fingerprint"]
    with pytest.raises(SchemaError, match="execution_environment_fingerprint"):
        validate_execution_record(rec)


@pytest.mark.parametrize("bad_key", ["hostname", "registry_uri", "account_id", "scratch_path"])
def test_unknown_infrastructure_key_rejected(bad_key: str) -> None:
    """The closed allow-list must reject infrastructure-shaped unknown fields."""
    rec = _valid_record()
    rec[bad_key] = "some-value"
    with pytest.raises(SchemaError, match="unknown field"):
        validate_execution_record(rec)


def test_unknown_install_mode_rejected() -> None:
    rec = _valid_record()
    rec["install_mode"] = "some-cluster-name"  # an identifier, not a category
    _reseal(rec)
    with pytest.raises(SchemaError, match="install_mode"):
        validate_execution_record(rec)


def test_bad_fingerprint_prefix_rejected() -> None:
    rec = _valid_record()
    rec["execution_environment_fingerprint"] = "not-a-digest"
    with pytest.raises(SchemaError, match="sha256"):
        validate_execution_record(rec)


def test_short_sha256_rejected() -> None:
    rec = _valid_record()
    rec["execution_environment_fingerprint"] = "sha256:abc123"  # too short
    with pytest.raises(SchemaError, match="64 hex"):
        validate_execution_record(rec)


def test_fingerprint_mismatch_rejected() -> None:
    """A record whose content changed but fingerprint didn't must be rejected."""
    rec = _valid_record()
    rec["python_version"] = "9.9.9"  # change content, leave stale fingerprint
    with pytest.raises(SchemaError, match="does not match canonical"):
        validate_execution_record(rec)


def test_bad_uv_lock_digest_rejected() -> None:
    rec = _valid_record()
    rec["uv_lock_sha256"] = "sha256:not-hex"
    _reseal(rec)
    with pytest.raises(SchemaError, match="uv_lock_sha256"):
        validate_execution_record(rec)


def test_bad_reference_digest_rejected() -> None:
    rec = _valid_record()
    rec["public_reference_environment_digest"] = "not-a-digest"
    with pytest.raises(SchemaError, match="public_reference_environment_digest"):
        validate_execution_record(rec)


def test_wrong_type_rejected() -> None:
    rec = _valid_record()
    rec["package_versions"] = ["not", "a", "dict"]
    with pytest.raises(SchemaError, match="dict"):
        validate_execution_record(rec)


def test_null_reference_digest_allowed() -> None:
    rec = _valid_record()
    assert rec["public_reference_environment_digest"] is None
    validate_execution_record(rec)  # None is permitted


@pytest.mark.parametrize("parent", ["torch", "runtime", "determinism_flags"])
def test_unknown_nested_key_rejected(parent: str) -> None:
    """A1: the nested dicts are closed too — an infra-shaped nested key is rejected."""
    rec = _valid_record()
    rec[parent] = dict(rec[parent])
    rec[parent]["hostname"] = "private-host"
    _reseal(rec)
    with pytest.raises(SchemaError, match="unknown nested field"):
        validate_execution_record(rec)


def test_infra_shaped_nested_value_rejected() -> None:
    """A1: an allowed nested key holding an infrastructure-shaped value is rejected.

    The synthetic ECR-shaped value is assembled at runtime from fragments so no
    literal registry string sits in this tracked file (the leak scanner reads it).
    """
    rec = _valid_record()
    rec["torch"] = dict(rec["torch"])
    ecr = "0" * 12 + ".dkr.ecr." + "us-west-1" + ".amazonaws.com/x:y"
    rec["torch"]["gpu_model_class"] = ecr
    _reseal(rec)
    with pytest.raises(SchemaError, match="infrastructure-shaped"):
        validate_execution_record(rec)


def test_public_reference_requires_lock_digest() -> None:
    """A1: a public-reference record with null uv_lock_sha256 is rejected."""
    rec = collect_execution_environment(
        install_mode="public-reference-container",
        reference_compatibility_status="public-reference-run",
    ).to_record()
    rec["uv_lock_sha256"] = None
    _reseal(rec)
    with pytest.raises(SchemaError, match="uv_lock_sha256"):
        validate_execution_record(rec)


def test_public_reference_requires_nonempty_facts() -> None:
    """A1: a public-reference record with empty torch/runtime facts is rejected."""
    rec = collect_execution_environment(
        install_mode="public-reference-container",
        reference_compatibility_status="equivalence-verified",
    ).to_record()
    rec["torch"] = {}
    _reseal(rec)
    with pytest.raises(SchemaError, match="non-empty 'torch'"):
        validate_execution_record(rec)


def test_partial_record_allowed_for_unknown_status() -> None:
    """A partial record (empty facts) is fine when status is unknown/pending."""
    rec = _valid_record()  # install_mode='local', status 'unknown'
    rec["torch"] = {}
    rec["runtime"] = {}
    rec["uv_lock_sha256"] = None
    _reseal(rec)
    validate_execution_record(rec)  # must not raise


def test_runtime_records_real_compiler_facts() -> None:
    """A1: runtime carries actual sanitized compiler/BLAS/threads facts, not only a hash."""
    rec = _valid_record()
    runtime_keys = set(rec["runtime"])
    assert {"blas_impl", "cxx_version", "openmp_enabled", "torch_num_threads"} <= runtime_keys
    # torch is installed in the dev env, so at least the thread count is populated.
    assert rec["runtime"]["torch_num_threads"] is not None


def test_required_field_inventory_matches_plan_section6() -> None:
    """The schema's environment-level fields cover the §6.1 sanitized-record inventory."""
    rec = _valid_record()
    # §6.1 environment-level provenance must be representable and present.
    assert rec["uv_lock_sha256"] is not None
    assert {
        "torch_version",
        "torch_cuda_version",
        "cudnn_version",
        "driver_version",
        "gpu_model_class",
        "gpu_total_memory_gib",
    } <= set(rec["torch"])
    assert {"cpu_count", "total_ram_gib"} <= set(rec["runtime"])
    assert "execution_environment_fingerprint" in rec
    assert rec["reference_compatibility_status"] in {
        "public-reference-run",
        "equivalence-verified",
        "pending-equivalence",
        "unknown",
    }
