"""Tests for the sanitized execution-environment record (plan §2.4 layer 1, §6.1)."""

from __future__ import annotations

import getpass
import json
import os
import socket

import pytest

from neural_repr.provenance import (
    collect_execution_environment,
    environment_fingerprint,
)
from neural_repr.provenance.environment import INSTALL_MODES


def test_record_has_required_fields() -> None:
    rec = collect_execution_environment(install_mode="local").to_record()
    for key in (
        "python_version",
        "package_versions",
        "torch",
        "execution_environment_fingerprint",
        "public_reference_environment_digest",
        "reference_compatibility_status",
        "install_mode",
    ):
        assert key in rec, f"missing required field {key!r}"
    assert rec["execution_environment_fingerprint"].startswith("sha256:")


def test_record_contains_no_infrastructure_identifiers() -> None:
    """The committed record must never leak host/user/path/registry identifiers."""
    rec = collect_execution_environment(install_mode="local").to_record()
    blob = json.dumps(rec).lower()
    for probe in (
        socket.gethostname().lower(),
        getpass.getuser().lower(),
        os.path.expanduser("~").lower(),
    ):
        if probe:
            assert probe not in blob, f"record leaked identifier fragment: {probe!r}"


def test_fingerprint_is_deterministic_and_excludes_reference_digest() -> None:
    a = collect_execution_environment(install_mode="local")
    b = collect_execution_environment(
        install_mode="local", public_reference_environment_digest="sha256:deadbeef"
    )
    # Same sanitized fields -> same fingerprint; the reference digest is excluded.
    assert a.execution_environment_fingerprint == b.execution_environment_fingerprint


def test_fingerprint_changes_with_install_mode() -> None:
    a = collect_execution_environment(install_mode="local")
    b = collect_execution_environment(install_mode="public-reference-container")
    assert a.execution_environment_fingerprint != b.execution_environment_fingerprint


def test_invalid_install_mode_rejected() -> None:
    with pytest.raises(ValueError):
        collect_execution_environment(install_mode="not-a-real-mode")


def test_environment_fingerprint_stable_for_equal_payloads() -> None:
    payload = {"b": 2, "a": 1}
    assert environment_fingerprint(payload) == environment_fingerprint({"a": 1, "b": 2})


def test_install_modes_are_categories_not_identifiers() -> None:
    assert "public-reference-container" in INSTALL_MODES
    assert "private-mirror" in INSTALL_MODES
