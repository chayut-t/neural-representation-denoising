"""Console entry points for neural_repr.

Each module here exposes a Typer ``app`` (and a ``main()`` wired to a console
script in ``pyproject.toml``). Commands are thin wrappers around importable
library functions, so the same behavior is available programmatically.
"""

from __future__ import annotations

__all__ = ["aggregate", "data", "eval", "figure", "train", "verify"]
