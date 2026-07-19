"""Shared numerical helpers for the Phase 3 math tests.

Central finite-difference gradient utility used to verify analytic gradients
(plan Gate P3; tolerance ``tol-gradient-fd`` in ``docs/tolerances.md``). Everything
runs in float64 on CPU for tight, platform-stable differences.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

# Central-difference step and tolerances committed as tol-gradient-fd
# (docs/tolerances.md). Two named tiers so tests and the registry cannot drift:
#   * FD_*   — closed-form analytic gradients of a single expression (tight).
#   * BPTT_* — gradients accumulated through an unrolled/BPTT computation, where
#     rounding compounds across steps, so a slightly looser bound is justified.
# The registry row records both tiers with the same numbers used here.
FD_STEP: float = 1e-6
FD_RTOL: float = 1e-6
FD_ATOL: float = 1e-9
BPTT_RTOL: float = 1e-5
BPTT_ATOL: float = 1e-7


def finite_difference_grad(
    f: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    *,
    step: float = FD_STEP,
) -> torch.Tensor:
    """Central finite-difference gradient of a scalar-valued ``f`` at ``x``.

    ``f`` must return a scalar tensor. Returns a tensor shaped like ``x``.
    """
    x = x.detach().clone().to(torch.float64)
    grad = torch.zeros_like(x)
    flat = x.reshape(-1)
    grad_flat = grad.reshape(-1)
    for idx in range(flat.numel()):
        orig = flat[idx].item()
        flat[idx] = orig + step
        f_plus = float(f(x).item())
        flat[idx] = orig - step
        f_minus = float(f(x).item())
        flat[idx] = orig
        grad_flat[idx] = (f_plus - f_minus) / (2.0 * step)
    return grad


def assert_grad_close(analytic: torch.Tensor, numeric: torch.Tensor) -> None:
    """Assert an analytic gradient matches a finite-difference one within tol-gradient-fd."""
    torch.testing.assert_close(analytic.to(torch.float64), numeric, rtol=FD_RTOL, atol=FD_ATOL)
