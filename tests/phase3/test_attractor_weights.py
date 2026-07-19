"""Tests for structured recurrent weights and the linear-dynamics corrections (3B).

Covers plan 3B tasks 2, 5, 6, 7: exact symmetry/antisymmetry by parameterization,
the effective-weight decomposition, and the two continuous-time corrections
(amplification depends on eigenvalue *sign*; antisymmetric flow is norm-preserving
only in continuous time).
"""

from __future__ import annotations

import pytest
import torch

from neural_repr.attractor import (
    antisymmetric_part,
    continuous_generator,
    effective_weight,
    euler_step_matrix,
    flow_matrix,
    growth_rates,
    is_asymptotically_stable,
    is_contractive,
    jacobian_generator,
    log_norm,
    spectral_abscissa,
    symmetric_part,
)

torch.manual_seed(0)


def test_symmetric_part_exact() -> None:
    w = torch.randn(8, 8, dtype=torch.float64)
    s = symmetric_part(w)
    assert float((s - s.transpose(-1, -2)).abs().max()) == 0.0


def test_antisymmetric_part_exact() -> None:
    v = torch.randn(8, 8, dtype=torch.float64)
    a = antisymmetric_part(v)
    assert float((a + a.transpose(-1, -2)).abs().max()) == 0.0


def test_antisymmetric_diagonal_is_zero() -> None:
    v = torch.randn(6, 6, dtype=torch.float64)
    a = antisymmetric_part(v)
    torch.testing.assert_close(torch.diagonal(a), torch.zeros(6, dtype=torch.float64))


def test_decomposition_reconstructs_original() -> None:
    w = torch.randn(5, 5, dtype=torch.float64)
    torch.testing.assert_close(symmetric_part(w) + antisymmetric_part(w), w)


def test_effective_weight_zero_velocity_is_symmetric_only() -> None:
    s = symmetric_part(torch.randn(6, 6, dtype=torch.float64))
    a = antisymmetric_part(torch.randn(6, 6, dtype=torch.float64))
    w = effective_weight(s, a, 0.0, alpha=100.0)
    torch.testing.assert_close(w, s)


def test_effective_weight_velocity_sign_flips_transport() -> None:
    s = symmetric_part(torch.randn(6, 6, dtype=torch.float64))
    a = antisymmetric_part(torch.randn(6, 6, dtype=torch.float64))
    w_plus = effective_weight(s, a, 1.0, alpha=10.0)
    w_minus = effective_weight(s, a, -1.0, alpha=10.0)
    # The antisymmetric contribution reverses; the symmetric part is unchanged.
    torch.testing.assert_close(0.5 * (w_plus + w_minus), s)
    torch.testing.assert_close(0.5 * (w_plus - w_minus), 10.0 * a)


def test_effective_weight_2d_two_directions() -> None:
    s = symmetric_part(torch.randn(4, 4, dtype=torch.float64))
    ax = antisymmetric_part(torch.randn(4, 4, dtype=torch.float64))
    ay = antisymmetric_part(torch.randn(4, 4, dtype=torch.float64))
    w = effective_weight(s, [ax, ay], [2.0, 3.0], alpha=1.0)
    torch.testing.assert_close(w, s + 2.0 * ax + 3.0 * ay)


def test_effective_weight_velocity_count_mismatch_raises() -> None:
    s = symmetric_part(torch.randn(4, 4, dtype=torch.float64))
    a = antisymmetric_part(torch.randn(4, 4, dtype=torch.float64))
    with pytest.raises(ValueError, match="must match"):
        effective_weight(s, [a, a], [1.0], alpha=1.0)


def test_effective_weight_preserves_tensor_velocity_gradient() -> None:
    """A tensor velocity flows gradients through W_eff (not detached to a float) (B4)."""
    s = symmetric_part(torch.randn(4, 4, dtype=torch.float64))
    a = antisymmetric_part(torch.randn(4, 4, dtype=torch.float64))
    v = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)
    w = effective_weight(s, a, v, alpha=3.0)
    w.sum().backward()
    assert v.grad is not None
    # dW/dv summed = alpha * sum(A); check the gradient is exactly that.
    assert float(v.grad) == pytest.approx(3.0 * float(a.sum()))


def test_effective_weight_1d_tensor_velocity_multi_direction() -> None:
    s = symmetric_part(torch.randn(4, 4, dtype=torch.float64))
    ax = antisymmetric_part(torch.randn(4, 4, dtype=torch.float64))
    ay = antisymmetric_part(torch.randn(4, 4, dtype=torch.float64))
    v = torch.tensor([2.0, -3.0], dtype=torch.float64)
    w = effective_weight(s, [ax, ay], v, alpha=1.0)
    torch.testing.assert_close(w, s + 2.0 * ax - 3.0 * ay)


# --- continuous-time corrections ---------------------------------------------


def test_growth_rate_uses_sign_not_unit_threshold() -> None:
    """Amplification is set by the sign of the *generator* eigenvalue (correction 1).

    The leaky continuous generator is ``M = -I + W``; a mode grows iff
    ``Re(eig(M)) > 0`` and decays iff ``< 0``. For symmetric ``W`` the generator
    eigenvalue is ``w_eig - 1``, so the correct test is the sign of that quantity,
    not a comparison of a raw weight multiplier against 1. Checked directly on a
    matrix with a known spectrum.
    """
    # Symmetric W with a known spectrum: eigenvalues 0.5 and 1.5.
    q, _ = torch.linalg.qr(torch.randn(2, 2, dtype=torch.float64))
    w = q @ torch.diag(torch.tensor([0.5, 1.5], dtype=torch.float64)) @ q.T
    rates = growth_rates(w)
    # Generator eigenvalues are 0.5-1 = -0.5 (decays) and 1.5-1 = +0.5 (grows).
    torch.testing.assert_close(rates, torch.tensor([-0.5, 0.5], dtype=torch.float64))
    assert not is_asymptotically_stable(w)


def test_symmetric_stable_is_also_contractive() -> None:
    """For a symmetric (normal) generator, asymptotic stability == contraction."""
    q, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))
    w = q @ torch.diag(torch.tensor([0.2, 0.5, 0.9], dtype=torch.float64)) @ q.T
    # All weight eigenvalues < 1 -> generator eigenvalues all < 0.
    assert is_asymptotically_stable(w)
    assert is_contractive(w)
    # Normal M: log-norm equals spectral abscissa.
    assert log_norm(w) == pytest.approx(spectral_abscissa(w))


def test_nonnormal_stable_but_not_contractive() -> None:
    """The reviewer's counterexample: asymptotically stable yet norm grows transiently (B1).

    W = [[0, 100], [0, 0]] gives M = -I + W with both eigenvalues at -1 (stable), but
    at x=[1,1], d||x||^2/dt = 2 x^T M x = 196 > 0, so it is NOT contractive. The
    log-norm (largest eigenvalue of the symmetric part) must be positive here.
    """
    w = torch.tensor([[0.0, 100.0], [0.0, 0.0]], dtype=torch.float64)
    assert is_asymptotically_stable(w)  # spectral abscissa = -1 < 0
    assert spectral_abscissa(w) == pytest.approx(-1.0)
    assert not is_contractive(w)  # log-norm > 0
    assert log_norm(w) > 0.0

    # Direct instantaneous check: d||x||^2/dt = x^T (M + M^T) x at x = [1, 1].
    m = continuous_generator(w)
    x = torch.tensor([1.0, 1.0], dtype=torch.float64)
    rate = float(x @ (m + m.T) @ x)
    assert rate == pytest.approx(196.0)


def test_transient_amplification_then_decay_for_nonnormal_stable() -> None:
    """A stable nonnormal system's norm can rise before it eventually decays."""
    w = torch.tensor([[0.0, 100.0], [0.0, 0.0]], dtype=torch.float64)
    m = continuous_generator(w)
    x0 = torch.tensor([1.0, 1.0], dtype=torch.float64)
    # Norm peaks around t~1 (>> initial) then decays; by t=20 it is far below initial.
    norms = [float((torch.linalg.matrix_exp(m * t) @ x0).norm()) for t in (0.0, 0.02, 1.0, 20.0)]
    assert norms[1] > norms[0]  # transient growth
    assert norms[2] > 10.0 * norms[0]  # large transient amplification
    assert norms[-1] < norms[0]  # eventual decay (asymptotic stability)


def test_jacobian_generator_absorbs_activation_derivative() -> None:
    """M = -I + W_eff diag(phi') is the correct linearization at a fixed point."""
    n = 4
    w_eff = torch.randn(n, n, dtype=torch.float64)
    phi_prime = torch.rand(n, dtype=torch.float64) + 0.5
    m = jacobian_generator(w_eff, phi_prime)
    expected = -torch.eye(n, dtype=torch.float64) + w_eff * phi_prime.unsqueeze(-2)
    torch.testing.assert_close(m, expected)


def test_pure_antisymmetric_generator_has_imaginary_spectrum() -> None:
    """A pure antisymmetric A has purely imaginary eigenvalues (zero growth rate)."""
    a = antisymmetric_part(torch.randn(6, 6, dtype=torch.float64))
    eig = torch.linalg.eigvals(a)
    torch.testing.assert_close(eig.real, torch.zeros(6, dtype=torch.float64), atol=1e-12, rtol=0)


def test_continuous_flow_preserves_norm_but_euler_step_does_not() -> None:
    """exp(A t) is orthogonal (norm-preserving); the Euler map I + dt A is not (correction 2)."""
    a = antisymmetric_part(torch.randn(5, 5, dtype=torch.float64))
    x = torch.randn(5, dtype=torch.float64)

    flow = flow_matrix(a, 0.7)
    x_flow = flow @ x
    assert float(x_flow.norm()) == pytest.approx(float(x.norm()), rel=1e-10)

    euler = euler_step_matrix(a, 0.7)
    x_euler = euler @ x
    # The discrete Euler step grows the norm: ||(I+dtA)x||^2 = ||x||^2 + dt^2||Ax||^2.
    assert float(x_euler.norm()) > float(x.norm())


def test_continuous_generator_definition() -> None:
    w = torch.randn(4, 4, dtype=torch.float64)
    m = continuous_generator(w)
    torch.testing.assert_close(m, w - torch.eye(4, dtype=torch.float64))
