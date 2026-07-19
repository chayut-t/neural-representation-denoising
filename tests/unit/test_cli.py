"""Smoke tests: every console entry point exposes a working ``--help``."""

from __future__ import annotations

import importlib

import pytest
from typer.testing import CliRunner

CLI_MODULES = [
    "neural_repr.cli.data",
    "neural_repr.cli.train",
    "neural_repr.cli.eval",
    "neural_repr.cli.aggregate",
    "neural_repr.cli.figure",
    "neural_repr.cli.verify",
]

runner = CliRunner()


@pytest.mark.parametrize("module_name", CLI_MODULES)
def test_cli_help_exits_zero(module_name: str) -> None:
    module = importlib.import_module(module_name)
    result = runner.invoke(module.app, ["--help"])
    assert result.exit_code == 0
    # Typer wraps/colorizes help; the stable invariant is that usage is shown.
    assert "Usage" in result.output
    assert callable(module.main)


def test_system_info_emits_fingerprint() -> None:
    from neural_repr.cli import verify

    result = runner.invoke(verify.app, ["system-info", "--install-mode", "local"])
    assert result.exit_code == 0
    assert "execution_environment_fingerprint" in result.output


# Invoked placeholder operations must FAIL (non-zero), so automation cannot mistake
# a not-yet-implemented command for a completed run (codex P2).
_PLACEHOLDER_INVOCATIONS = [
    ("neural_repr.cli.data", ["download", "div2k"]),
    ("neural_repr.cli.data", ["check"]),
    ("neural_repr.cli.train", ["run", "sparse/toy"]),
    ("neural_repr.cli.eval", ["run", "some-run-id"]),
    ("neural_repr.cli.aggregate", ["run", "some-manifest"]),
    ("neural_repr.cli.figure", ["make", "ch3-basis-atlas"]),
]


@pytest.mark.parametrize(("module_name", "args"), _PLACEHOLDER_INVOCATIONS)
def test_placeholder_operations_fail(module_name: str, args: list[str]) -> None:
    module = importlib.import_module(module_name)
    result = runner.invoke(module.app, args)
    assert result.exit_code != 0, f"{module_name} {args} should fail, not report false success"
