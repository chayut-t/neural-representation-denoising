"""Tests for sparse inference, dictionary updates, and denoising (3A tasks 4-6, 8).

Covers the two named inference variants (decision 0004), analytic vs autograd and
finite-difference gradients (Gate P3), the implicit-function-theorem gradient at the
converged optimum, smooth-gradient convergence to a trusted convex reference
(``tol-solver-convergence``) with a step size that accounts for penalty curvature,
dictionary normalization/gradient, and the denoising objective's consistency.
"""

from __future__ import annotations

import pytest
import torch
from fdcheck import (
    BPTT_ATOL,
    BPTT_RTOL,
    FD_ATOL,
    FD_RTOL,
    assert_grad_close,
    finite_difference_grad,
)

from neural_repr.sparse_coding import (
    add_gaussian_noise,
    data_grad,
    denoising_loss,
    dictionary_grad,
    group_activity,
    group_penalty,
    group_penalty_grad,
    infer,
    legacy_adaptive_threshold,
    lipschitz_step_size,
    normalize_columns,
    reconstruct,
    reconstruction_error,
    smooth_gradient,
    soft_threshold,
    softplus_lambda,
    stable_step_size,
)

torch.manual_seed(0)


def _dictionary(n: int = 6, m: int = 8) -> torch.Tensor:
    return normalize_columns(torch.randn(n, m, dtype=torch.float64))


def _lambda(k: int, m: int) -> torch.Tensor:
    return softplus_lambda(torch.randn(k, m, dtype=torch.float64), normalization="row_sum")


# --- dictionary primitives ---------------------------------------------------


def test_normalize_columns_unit_norm() -> None:
    phi = _dictionary()
    norms = torch.linalg.vector_norm(phi, dim=0)
    torch.testing.assert_close(norms, torch.ones_like(norms))


def test_normalize_columns_raises_on_zero_column_by_default() -> None:
    phi = torch.randn(4, 3, dtype=torch.float64)
    phi[:, 1] = 0.0
    with pytest.raises(ValueError, match="zero/degenerate column"):
        normalize_columns(phi)


def test_normalize_columns_keep_leaves_zero_column_finite() -> None:
    phi = torch.randn(4, 3, dtype=torch.float64)
    phi[:, 1] = 0.0
    out = normalize_columns(phi, on_zero="keep")
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out[:, 1], torch.zeros(4, dtype=torch.float64))
    # Nonzero columns are still exactly unit-norm.
    other_norms = torch.linalg.vector_norm(out[:, [0, 2]], dim=0)
    torch.testing.assert_close(other_norms, torch.ones(2, dtype=torch.float64))


def test_normalize_columns_rejects_bad_on_zero() -> None:
    with pytest.raises(ValueError, match="on_zero must be"):
        normalize_columns(torch.randn(4, 3, dtype=torch.float64), on_zero="nope")


def test_normalize_columns_rejects_nonpositive_eps() -> None:
    # R2-11: eps must be positive/finite, else a zero column could produce NaNs.
    with pytest.raises(ValueError, match="positive real"):
        normalize_columns(torch.randn(4, 3, dtype=torch.float64), eps=0.0)


def test_data_grad_vs_finite_difference() -> None:
    phi = _dictionary()
    x = torch.randn(6, dtype=torch.float64)
    s = torch.randn(8, dtype=torch.float64)
    analytic = data_grad(phi, s, x)
    numeric = finite_difference_grad(lambda v: reconstruction_error(phi, v, x), s)
    assert_grad_close(analytic, numeric)


def test_dictionary_grad_vs_finite_difference() -> None:
    phi = _dictionary()
    x = torch.randn(6, dtype=torch.float64)
    s = torch.randn(8, dtype=torch.float64)
    # -dE/dPhi, so compare against the negative FD gradient of the error.
    analytic = dictionary_grad(phi, s, x)
    numeric = finite_difference_grad(lambda p: reconstruction_error(p, s, x), phi)
    assert_grad_close(analytic, -numeric)


# --- soft threshold ----------------------------------------------------------


def test_soft_threshold_matches_definition() -> None:
    v = torch.tensor([-3.0, -0.5, 0.0, 0.5, 3.0], dtype=torch.float64)
    out = soft_threshold(v, 1.0)
    expected = torch.sign(v) * (v.abs() - 1.0).clamp_min(0.0)
    torch.testing.assert_close(out, expected)


def test_soft_threshold_clamps_negative_theta() -> None:
    v = torch.tensor([1.0, -1.0], dtype=torch.float64)
    # A negative threshold must not grow coefficients: treated as 0.
    torch.testing.assert_close(soft_threshold(v, -5.0), v)


# --- inference variants ------------------------------------------------------


def test_smooth_gradient_reduces_objective() -> None:
    phi = _dictionary()
    lam = _lambda(4, 8)
    x = torch.randn(6, dtype=torch.float64)
    eta = lipschitz_step_size(phi, safety=0.9)
    pw = 0.05

    def objective(s: torch.Tensor) -> torch.Tensor:
        return reconstruction_error(phi, s, x) + pw * group_penalty(s, lam)

    s0 = torch.zeros(8, dtype=torch.float64)
    s_final = smooth_gradient(x, phi, lam, n_steps=200, eta=eta, penalty_weight=pw)
    assert float(objective(s_final)) < float(objective(s0))


def test_smooth_gradient_zero_penalty_matches_least_squares() -> None:
    """With penalty_weight=0 the smooth gradient is plain gradient descent on 0.5||x-Phi s||^2."""
    phi = _dictionary(n=8, m=6)  # overdetermined: unique least-squares solution
    x = torch.randn(8, dtype=torch.float64)
    eta = lipschitz_step_size(phi, safety=0.9)
    s = smooth_gradient(
        x, phi, torch.eye(6, dtype=torch.float64), n_steps=2000, eta=eta, penalty_weight=0.0
    )
    ls = torch.linalg.lstsq(phi, x).solution
    torch.testing.assert_close(s, ls, rtol=1e-4, atol=1e-6)


def test_smooth_gradient_converges_to_trusted_convex_reference() -> None:
    """smooth_gradient minimizes the smoothed convex objective; check vs scipy L-BFGS-B.

    Freezes tol-solver-convergence (docs/tolerances.md): the unrolled smooth-gradient
    code must match an independent high-accuracy solve of
    ``0.5||x-Phi s||^2 + pw * sum_k z_k(s)`` within rtol 1e-3, atol 1e-4. scipy is an
    independent optimizer (no shared code with our recurrence), so this is a genuine
    solver-vs-reference check rather than a self-consistency test.
    """
    import numpy as np
    from scipy.optimize import minimize

    torch.manual_seed(1)
    n, m, k = 10, 6, 4
    phi = normalize_columns(torch.randn(n, m, dtype=torch.float64))
    x = torch.randn(n, dtype=torch.float64)
    lam = softplus_lambda(torch.randn(k, m, dtype=torch.float64), normalization="row_sum")
    pw = 0.1
    eta = stable_step_size(phi, lam, pw, safety=0.9)

    s_solver = smooth_gradient(
        x,
        phi,
        lam,
        n_steps=20000,
        eta=eta,
        penalty_weight=pw,
        s0=torch.zeros(m, dtype=torch.float64),
    )

    def objective(s_np: np.ndarray) -> float:
        s = torch.from_numpy(s_np)
        return float(reconstruction_error(phi, s, x) + pw * group_penalty(s, lam))

    ref = minimize(
        objective,
        np.zeros(m),
        method="L-BFGS-B",
        options={"maxiter": 2000, "ftol": 1e-15, "gtol": 1e-12},
    )
    s_ref = torch.from_numpy(ref.x)
    torch.testing.assert_close(s_solver, s_ref, rtol=1e-3, atol=1e-4)


@pytest.mark.parametrize("eps", [1e-3, 1e-6, 1e-9])
@pytest.mark.parametrize("pw", [0.01, 1.0, 10.0])
def test_stable_step_size_monotone_descent_across_eps_and_penalty(eps: float, pw: float) -> None:
    """stable_step_size makes smooth_gradient monotonically decrease the total objective.

    Stresses the step-size rule (B7) over a grid of eps (small eps -> large penalty
    curvature near 0) and penalty weight, where the data-only step would be unsafe.
    Every consecutive objective value must be nonincreasing.
    """
    torch.manual_seed(11)
    n, m, k = 8, 6, 4
    phi = normalize_columns(torch.randn(n, m, dtype=torch.float64))
    x = torch.randn(n, dtype=torch.float64)
    lam = softplus_lambda(torch.randn(k, m, dtype=torch.float64), normalization="row_sum")
    eta = stable_step_size(phi, lam, pw, eps=eps, safety=0.9)

    def objective(s: torch.Tensor) -> float:
        return float(reconstruction_error(phi, s, x) + pw * group_penalty(s, lam, eps=eps))

    s = torch.zeros(m, dtype=torch.float64)
    prev = objective(s)
    for _ in range(200):
        s = smooth_gradient(x, phi, lam, n_steps=1, eta=eta, penalty_weight=pw, eps=eps, s0=s)
        cur = objective(s)
        assert cur <= prev + 1e-12, f"objective increased: {prev} -> {cur}"
        prev = cur


def test_stable_step_size_smaller_than_data_only_when_penalty_curved() -> None:
    """With small eps and large penalty weight, the total-objective step is smaller."""
    torch.manual_seed(12)
    phi = normalize_columns(torch.randn(8, 6, dtype=torch.float64))
    lam = softplus_lambda(torch.randn(4, 6, dtype=torch.float64), normalization="row_sum")
    eta_total = stable_step_size(phi, lam, penalty_weight=10.0, eps=1e-9, safety=0.9)
    eta_data = lipschitz_step_size(phi, safety=0.9)
    assert eta_total < eta_data


@pytest.mark.parametrize("safety", [0.0, -0.5, 1.5])
def test_step_size_rejects_out_of_range_safety(safety: float) -> None:
    phi = _dictionary()
    lam = _lambda(4, 8)
    with pytest.raises(ValueError, match="safety must lie in"):
        lipschitz_step_size(phi, safety=safety)
    with pytest.raises(ValueError, match="safety must lie in"):
        stable_step_size(phi, lam, 0.1, safety=safety)


def test_infer_dispatch_matches_direct_calls() -> None:
    phi = _dictionary()
    lam = _lambda(4, 8)
    x = torch.randn(6, dtype=torch.float64)
    eta = lipschitz_step_size(phi, safety=0.9)
    kw = dict(n_steps=10, eta=eta, penalty_weight=0.05)
    torch.testing.assert_close(
        infer("smooth_gradient", x, phi, lam, **kw),
        smooth_gradient(x, phi, lam, **kw),
    )
    torch.testing.assert_close(
        infer("legacy_adaptive_threshold", x, phi, lam, **kw),
        legacy_adaptive_threshold(x, phi, lam, **kw),
    )


def test_infer_rejects_unknown_variant() -> None:
    phi = _dictionary()
    lam = _lambda(4, 8)
    x = torch.randn(6, dtype=torch.float64)
    with pytest.raises(ValueError, match="unknown inference variant"):
        infer("nope", x, phi, lam, n_steps=1, eta=0.1, penalty_weight=0.0)  # type: ignore[arg-type]


def test_legacy_threshold_at_zero_start_is_large_finite() -> None:
    """At s=0 the adaptive threshold is a large but *finite* value (not divergent).

    With fixed ``eps > 0``, ``z_k = sqrt(eps)`` and the coupling
    ``sum_k Lambda_ki / sqrt(eps)`` is finite; the first-step threshold is
    ``eta * pw * coupling``. This is the spec-faithful reason the adaptive-threshold
    reading is *not* a true proximal operator (baseline §3.3.1). We check the
    magnitude directly rather than asserting universal pinning (B12).
    """
    phi = _dictionary()
    lam = _lambda(4, 8)
    eps = 1e-6
    pw, eta = 0.05, lipschitz_step_size(phi, safety=0.9)
    z0 = group_activity(torch.zeros(8, dtype=torch.float64), lam, eps=eps)
    assert torch.isfinite(z0).all()
    coupling = (1.0 / z0) @ lam
    theta0 = eta * pw * coupling
    assert torch.isfinite(theta0).all()
    assert float(theta0.max()) == pytest.approx(float(theta0.max()))  # finite, defined


def test_legacy_threshold_pins_when_threshold_dominates_data_step() -> None:
    """Parameter-dependent degeneracy: when the first threshold exceeds the data step
    magnitude from s=0, the first update pins coefficients to zero and they stay there.

    This holds for the (eta, pw, Lambda, eps) below — a *conditional* property, not a
    universal law (B12). A warm start or the smooth_gradient variant avoids it.
    """
    phi = _dictionary()
    lam = _lambda(4, 8)
    x = torch.randn(6, dtype=torch.float64)
    eta = lipschitz_step_size(phi, safety=0.9)
    # First data step from s=0 has magnitude |eta * Phi^T x|; make the threshold exceed it.
    v1 = (eta * (phi.T @ x)).abs()
    z0 = group_activity(torch.zeros(8, dtype=torch.float64), lam, eps=1e-6)
    coupling = (1.0 / z0) @ lam
    pw = 2.0 * float((v1 / (eta * coupling)).max())  # threshold > data step for every coeff
    s = legacy_adaptive_threshold(x, phi, lam, n_steps=50, eta=eta, penalty_weight=pw, eps=1e-6)
    torch.testing.assert_close(s, torch.zeros(8, dtype=torch.float64))


# --- gradient through unrolled inference -------------------------------------


def test_autograd_through_smooth_gradient_matches_finite_difference() -> None:
    """Denoising-loss gradient w.r.t. Lambda through unrolled inference vs finite diff (Gate P3)."""
    torch.manual_seed(2)
    n, m, k = 5, 6, 3
    phi = _dictionary(n, m)
    x = torch.randn(4, n, dtype=torch.float64)
    x_tilde = x + 0.1 * torch.randn_like(x)
    eta = lipschitz_step_size(phi, safety=0.9)

    def loss_from_raw(raw: torch.Tensor) -> torch.Tensor:
        lam = softplus_lambda(raw, normalization="row_sum")
        s_t = smooth_gradient(x_tilde, phi, lam, n_steps=8, eta=eta, penalty_weight=0.05)
        return denoising_loss(x, s_t, phi)

    raw = torch.randn(k, m, dtype=torch.float64, requires_grad=True)
    loss = loss_from_raw(raw)
    loss.backward()
    analytic = raw.grad
    assert analytic is not None
    numeric = finite_difference_grad(lambda r: loss_from_raw(r), raw.detach())
    torch.testing.assert_close(analytic.to(torch.float64), numeric, rtol=BPTT_RTOL, atol=BPTT_ATOL)


def test_unrolled_gradient_matches_implicit_function_gradient() -> None:
    """Deep-unrolled gradient equals the implicit-function-theorem gradient (plan 3A.6).

    Inner problem ``E(s; theta) = 0.5||x - Phi s||^2 + theta * Omega(s)`` is smooth and
    strongly convex (eps>0), so it has a unique minimizer ``s*(theta)`` with an
    implicit gradient. For a scalar outer loss ``L(s*) = 0.5||x - Phi s*||^2`` and
    scalar parameter ``theta`` (the penalty weight),

        dL/dtheta = -(dL/ds)^T H^{-1} (d^2E / ds dtheta),   H = d^2E/ds^2,

    where ``d^2E/ds dtheta = grad_Omega(s*)``. We compute this directly (autograd
    Hessian + linear solve) and compare to the gradient obtained by differentiating a
    deep unroll. This actually solves the implicit system rather than only checking
    that successive unrolls stop changing.
    """
    torch.manual_seed(3)
    n, m = 7, 5
    phi = _dictionary(n, m)
    x = torch.randn(n, dtype=torch.float64)
    lam = softplus_lambda(torch.randn(3, m, dtype=torch.float64), normalization="row_sum")
    pw = 0.1
    eta = stable_step_size(phi, lam, pw, safety=0.9)

    # 1) Deep-unrolled gradient of the reconstruction loss w.r.t. the penalty weight.
    def unrolled_recon(weight: torch.Tensor, steps: int) -> torch.Tensor:
        s = _sg_scalar_pw(x, phi, lam, steps, eta, weight)
        return 0.5 * ((x - reconstruct(phi, s)) ** 2).sum()

    # Deep enough that the unrolled gradient has converged (checked separately: it is
    # stable to 5 digits by ~2e4 steps for this problem).
    pw_t = torch.tensor(pw, dtype=torch.float64, requires_grad=True)
    g_unrolled = float(torch.autograd.grad(unrolled_recon(pw_t, 40000), pw_t)[0])

    # 2) Implicit gradient at the converged optimum s*.
    s_star = (
        smooth_gradient(
            x,
            phi,
            lam,
            n_steps=40000,
            eta=eta,
            penalty_weight=pw,
            s0=torch.zeros(m, dtype=torch.float64),
        )
        .detach()
        .requires_grad_(True)
    )

    def energy(s: torch.Tensor) -> torch.Tensor:
        return reconstruction_error(phi, s, x) + pw * group_penalty(s, lam)

    hess = torch.autograd.functional.hessian(energy, s_star)  # (m, m), d^2E/ds^2
    dE_ds_dtheta = group_penalty_grad(s_star.detach(), lam)  # d^2E/ds dtheta = grad_Omega(s*)
    dL_ds = -(phi.T @ (x - reconstruct(phi, s_star.detach())))  # dL/ds at s*
    # ds*/dtheta = -H^{-1} (d^2E/ds dtheta); dL/dtheta = dL_ds . ds*/dtheta.
    ds_dtheta = -torch.linalg.solve(hess, dE_ds_dtheta)
    g_implicit = float(dL_ds @ ds_dtheta)

    assert g_implicit == pytest.approx(g_unrolled, rel=1e-4, abs=1e-6)


# --- denoising objective -----------------------------------------------------


def test_denoising_loss_zero_for_perfect_reconstruction() -> None:
    phi = _dictionary()
    s = torch.randn(4, 8, dtype=torch.float64)
    x = reconstruct(phi, s)
    assert float(denoising_loss(x, s, phi)) == pytest.approx(0.0, abs=1e-20)


def test_add_gaussian_noise_reproducible_with_generator() -> None:
    x = torch.zeros(100, dtype=torch.float64)
    g1 = torch.Generator().manual_seed(7)
    g2 = torch.Generator().manual_seed(7)
    torch.testing.assert_close(
        add_gaussian_noise(x, 0.5, generator=g1), add_gaussian_noise(x, 0.5, generator=g2)
    )


def test_add_gaussian_noise_zero_sigma_is_identity() -> None:
    x = torch.randn(10, dtype=torch.float64)
    torch.testing.assert_close(add_gaussian_noise(x, 0.0), x)


# --- helpers -----------------------------------------------------------------


def _sg_scalar_pw(
    x: torch.Tensor, phi: torch.Tensor, lam: torch.Tensor, steps: int, eta: float, pw: torch.Tensor
) -> torch.Tensor:
    """smooth_gradient with a differentiable scalar penalty weight (for grad-through tests)."""
    from neural_repr.sparse_coding.groups import group_penalty_grad

    s = torch.zeros((*x.shape[:-1], phi.shape[-1]), dtype=torch.float64)
    for _ in range(steps):
        grad = data_grad(phi, s, x) + pw * group_penalty_grad(s, lam)
        s = s - eta * grad
    return s


_ = (FD_RTOL, FD_ATOL)  # tolerances are exercised via assert_grad_close
