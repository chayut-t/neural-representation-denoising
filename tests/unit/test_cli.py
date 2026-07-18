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
