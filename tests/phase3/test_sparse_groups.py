"""Tests for the group penalty, its gradients, and Lambda parameterization (3A).

Covers plan 3A tasks 1-3, 5: numerically safe activity, signed coefficient
gradient vs finite differences, the exact 1/2 factor in the Lambda gradient,
nonnegativity/normalization conventions, and degenerate/zero-coefficient groups.
"""

from __future__ import annotations

import pytest
import torch
from fdcheck import assert_grad_close, finite_difference_grad

from neural_repr.sparse_coding import (
    DEFAULT_EPS,
    group_activity,
    group_penalty,
    group_penalty_grad,
    group_penalty_grad_lambda,
    softplus_lambda,
)

torch.manual_seed(0)


def _random_problem(k: int = 4, m: int = 6) -> tuple[torch.Tensor, torch.Tensor]:
    s = torch.randn(m, dtype=torch.float64)
    lam = softplus_lambda(torch.randn(k, m, dtype=torch.float64), normalization="none")
    return s, lam


def test_group_activity_matches_definition() -> None:
    s, lam = _random_problem()
    z = group_activity(s, lam, eps=DEFAULT_EPS)
    expected = torch.sqrt(DEFAULT_EPS + (lam * s.pow(2)).sum(dim=-1))
    torch.testing.assert_close(z, expected)


def test_group_activity_finite_at_zero() -> None:
    _, lam = _random_problem()
    z = group_activity(torch.zeros(lam.shape[-1], dtype=torch.float64), lam)
    # At s=0 every group activity equals sqrt(eps); no NaN/Inf.
    torch.testing.assert_close(z, torch.full_like(z, DEFAULT_EPS**0.5))


def test_group_penalty_grad_vs_finite_difference() -> None:
    s, lam = _random_problem()
    analytic = group_penalty_grad(s, lam)
    numeric = finite_difference_grad(lambda v: group_penalty(v, lam), s)
    assert_grad_close(analytic, numeric)


def test_group_penalty_grad_is_signed() -> None:
    """The coefficient gradient carries the sign of s_i (a signed gradient, not a threshold)."""
    _, lam = _random_problem()
    s = torch.tensor([1.0, -2.0, 0.5, -0.1, 3.0, -0.7], dtype=torch.float64)
    grad = group_penalty_grad(s, lam)
    # sign(grad_i) == sign(s_i) because the coupling sum_k Lambda_ki/z_k >= 0.
    assert torch.equal(torch.sign(grad), torch.sign(s))


def test_group_penalty_grad_lambda_vs_finite_difference() -> None:
    """d Omega / d Lambda_ki == s_i^2 / (2 z_k), 1/2 factor included (b26-corr-ml-half)."""
    s, lam = _random_problem()
    analytic = group_penalty_grad_lambda(s, lam)
    numeric = finite_difference_grad(lambda mat: group_penalty(s, mat), lam)
    assert_grad_close(analytic, numeric)


def test_lambda_gradient_has_explicit_half_factor() -> None:
    s, lam = _random_problem()
    z = group_activity(s, lam)
    grad = group_penalty_grad_lambda(s, lam)
    expected = 0.5 * s.pow(2).unsqueeze(-2) / z.unsqueeze(-1)
    torch.testing.assert_close(grad, expected)


def test_softplus_lambda_nonnegative_and_row_normalized() -> None:
    raw = torch.randn(5, 7, dtype=torch.float64)
    lam = softplus_lambda(raw, normalization="row_sum")
    assert bool((lam >= 0).all())
    torch.testing.assert_close(lam.sum(dim=-1), torch.ones(5, dtype=torch.float64))


def test_softplus_lambda_none_keeps_scale() -> None:
    raw = torch.randn(3, 4, dtype=torch.float64)
    lam = softplus_lambda(raw, normalization="none")
    torch.testing.assert_close(lam, torch.nn.functional.softplus(raw))


def test_softplus_lambda_rejects_unknown_normalization() -> None:
    raw = torch.randn(3, 4, dtype=torch.float64)
    with pytest.raises(ValueError, match="unknown normalization"):
        softplus_lambda(raw, normalization="rows")  # type: ignore[arg-type]


def test_group_activity_rejects_negative_lambda() -> None:
    s = torch.randn(4, dtype=torch.float64)
    bad = torch.tensor([[-1.0, 0.0, 1.0, 2.0]], dtype=torch.float64)
    with pytest.raises(ValueError, match="nonnegative"):
        group_activity(s, bad)


def test_group_activity_rejects_nonpositive_eps() -> None:
    s, lam = _random_problem()
    with pytest.raises(ValueError, match="eps must be positive"):
        group_activity(s, lam, eps=0.0)


def test_zero_coefficient_group_contributes_only_eps() -> None:
    """A group whose members are all zero-coefficient contributes sqrt(eps), grad 0 there."""
    # Group 0 weights only coefficient 0; set s_0 = 0.
    lam = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 1.0]], dtype=torch.float64)
    s = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
    z = group_activity(s, lam)
    assert z[0].item() == pytest.approx(DEFAULT_EPS**0.5)
    grad = group_penalty_grad(s, lam)
    # Coefficient 0 gets zero gradient (s_0 = 0), regardless of the eps-floored group.
    assert grad[0].item() == pytest.approx(0.0, abs=1e-12)


def test_diagonal_single_group_reduces_to_smoothed_abs() -> None:
    """K=M diagonal Lambda recovers a smoothed |s_i| penalty (standard sparse coding limit)."""
    m = 5
    lam = torch.eye(m, dtype=torch.float64)
    s = torch.randn(m, dtype=torch.float64)
    omega = group_penalty(s, lam)
    smoothed_l1 = torch.sqrt(DEFAULT_EPS + s.pow(2)).sum()
    torch.testing.assert_close(omega, smoothed_l1)


def test_group_penalty_permutation_invariance() -> None:
    """Permuting groups (rows of Lambda) leaves the summed penalty unchanged."""
    s, lam = _random_problem()
    perm = torch.tensor([2, 0, 3, 1])
    torch.testing.assert_close(group_penalty(s, lam), group_penalty(s, lam[perm]))


def test_group_activity_batched() -> None:
    _, lam = _random_problem(k=3, m=5)
    s = torch.randn(7, 5, dtype=torch.float64)
    z = group_activity(s, lam)
    assert z.shape == (7, 3)
    # Batched equals per-row.
    for i in range(7):
        torch.testing.assert_close(z[i], group_activity(s[i], lam))
