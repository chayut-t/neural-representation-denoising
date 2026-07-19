"""Tests for the discrete recurrence, codes, noise, and grad-through-time (3B).

Covers plan 3B tasks 1, 3, 4, 7: the explicit dt/tau update, write/hold/move
operations, a nontrivial zero-velocity hold verified in activity/readout space,
velocity reversal, time-step convergence to the continuous ODE, noise resampling
protocols and statistics, circular-boundary behavior of the ring code, device/dtype
propagation, and gradient correctness through unrolled time (including the trainable
nonlinearity parameters).
"""

from __future__ import annotations

import pytest
import torch
from fdcheck import BPTT_ATOL, BPTT_RTOL, finite_difference_grad

from neural_repr.attractor import (
    MixedCodeSpec,
    Nonlinearity,
    WeightNoiseSampler,
    antisymmetric_part,
    circular_distance,
    decode_place,
    gaussian_weight_perturbation,
    hold,
    mixed_code,
    place_code,
    ring_code,
    run,
    state_noise,
    step,
    symmetric_part,
    uniform_weight_perturbation,
)
from neural_repr.attractor.dynamics import log_rectifier, write_mix

torch.manual_seed(0)


def _weights(n: int = 6) -> tuple[torch.Tensor, torch.Tensor]:
    s = symmetric_part(0.1 * torch.randn(n, n, dtype=torch.float64))
    a = antisymmetric_part(0.1 * torch.randn(n, n, dtype=torch.float64))
    return s, a


def _symmetric_with_fixed_point(u_star: torch.Tensor, f_star: torch.Tensor) -> torch.Tensor:
    """Build a symmetric ``S`` with ``S f_star = u_star`` exactly.

    ``S = (u f^T + f u^T)/||f||^2 - (u.f) f f^T / ||f||^4`` is symmetric and maps
    ``f_star -> u_star``, so ``(u_star, f_star)`` is an exact zero-velocity fixed
    point of the noiseless recurrence when ``f_star = phi(u_star)``.
    """
    f2 = float(f_star @ f_star)
    uf = float(u_star @ f_star)
    outer = torch.outer(u_star, f_star)
    s = (outer + outer.T) / f2 - uf * torch.outer(f_star, f_star) / (f2**2)
    return s


# --- nonlinearity ------------------------------------------------------------


def test_nonlinearity_matches_definition() -> None:
    phi = Nonlinearity(a=2.0, b=0.5, c=3.0)
    u = torch.tensor([-1.0, 0.0, 1.0, 2.0], dtype=torch.float64)
    expected = 2.0 * torch.log1p(3.0 * torch.clamp(u + 0.5, min=0.0))
    torch.testing.assert_close(phi(u), expected)


def test_nonlinearity_effective_params_positive_and_recovered() -> None:
    phi = Nonlinearity(a=2.0, b=-0.5, c=3.0)
    with torch.no_grad():
        assert float(phi.a) == pytest.approx(2.0)
        assert float(phi.c) == pytest.approx(3.0)
        assert float(phi.b) == pytest.approx(-0.5)
        # a, c stay positive after an arbitrary gradient perturbation of the raw params.
        phi.raw_a += -100.0
        phi.raw_c += -100.0
        assert float(phi.a) > 0.0
        assert float(phi.c) > 0.0


def test_nonlinearity_nonnegative_and_zero_below_threshold() -> None:
    phi = Nonlinearity(a=1.0, b=0.0, c=1.0)
    u = torch.tensor([-5.0, -1.0, 0.0], dtype=torch.float64)
    torch.testing.assert_close(phi(u), torch.zeros(3, dtype=torch.float64))


@pytest.mark.parametrize(
    ("a", "b", "c"), [(0.0, 0.0, 1.0), (-1.0, 0.0, 1.0), (1.0, 0.0, 0.0), (1.0, 0.0, -2.0)]
)
def test_nonlinearity_rejects_nonpositive_a_c(a: float, b: float, c: float) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        Nonlinearity(a=a, b=b, c=c)


def test_nonlinearity_rejects_nonfinite() -> None:
    with pytest.raises(ValueError, match="finite"):
        Nonlinearity(a=1.0, b=float("nan"), c=1.0)


def test_nonlinearity_parameters_are_trainable() -> None:
    phi = Nonlinearity(a=1.5, b=0.2, c=2.0)
    names = {name for name, _ in phi.named_parameters()}
    assert names == {"raw_a", "b", "raw_c"}
    u = torch.randn(4, dtype=torch.float64)
    phi(u).sum().backward()
    assert phi.raw_a.grad is not None and phi.b.grad is not None and phi.raw_c.grad is not None


def test_log_rectifier_gradients_vs_finite_difference() -> None:
    """FD check of grad w.r.t. (a, b, c) of the raw nonlinearity."""
    u = torch.linspace(-1.0, 3.0, 7, dtype=torch.float64)

    def loss(params: torch.Tensor) -> torch.Tensor:
        a, b, c = params
        return log_rectifier(u, a, b, c).sum()

    params = torch.tensor([1.3, 0.4, 2.1], dtype=torch.float64, requires_grad=True)
    loss(params).backward()
    assert params.grad is not None
    numeric = finite_difference_grad(lambda p: loss(p), params.detach())
    torch.testing.assert_close(params.grad.to(torch.float64), numeric, rtol=1e-6, atol=1e-9)


# --- single step -------------------------------------------------------------


def test_step_matches_discrete_update() -> None:
    n = 6
    s, a = _weights(n)
    phi = Nonlinearity()
    u = torch.randn(n, dtype=torch.float64)
    tau, dt, alpha, v = 5.0, 1.0, 100.0, 0.0
    out = step(u, s_mat=s, a_mats=a, velocity=v, alpha=alpha, tau=tau, dt=dt, phi=phi)
    w = s + alpha * v * a
    expected = u + (dt / tau) * (-u + phi(u) @ w.T)
    torch.testing.assert_close(out, expected)


def test_step_rejects_nonpositive_tau_dt() -> None:
    s, a = _weights()
    phi = Nonlinearity()
    u = torch.randn(6, dtype=torch.float64)
    with pytest.raises(ValueError, match="must be positive"):
        step(u, s_mat=s, a_mats=a, velocity=0.0, alpha=1.0, tau=0.0, dt=1.0, phi=phi)


def test_write_mix_beta_one_is_codeword() -> None:
    phi = Nonlinearity()
    u = torch.randn(8, dtype=torch.float64)
    code = torch.randn(8, dtype=torch.float64)
    torch.testing.assert_close(write_mix(u, phi, code, 1.0), code)
    torch.testing.assert_close(write_mix(u, phi, code, 0.0), phi(u))


def test_write_mix_rejects_beta_out_of_range() -> None:
    phi = Nonlinearity()
    u = torch.randn(4, dtype=torch.float64)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        write_mix(u, phi, u, 1.5)


# --- multi-direction transport (2D) ------------------------------------------


def test_step_multi_direction_matches_sum() -> None:
    n = 6
    s = symmetric_part(0.1 * torch.randn(n, n, dtype=torch.float64))
    ax = antisymmetric_part(0.1 * torch.randn(n, n, dtype=torch.float64))
    ay = antisymmetric_part(0.1 * torch.randn(n, n, dtype=torch.float64))
    phi = Nonlinearity()
    u = torch.randn(n, dtype=torch.float64)
    out = step(
        u, s_mat=s, a_mats=[ax, ay], velocity=[2.0, -3.0], alpha=1.0, tau=5.0, dt=1.0, phi=phi
    )
    w = s + 2.0 * ax - 3.0 * ay
    expected = u + (1.0 / 5.0) * (-u + phi(u) @ w.T)
    torch.testing.assert_close(out, expected)


# --- hold and move -----------------------------------------------------------


def test_zero_velocity_hold_holds_nontrivial_code() -> None:
    """A genuine nonzero fixed point (activity ~ a place bump) is held under v=0.

    Constructs S so that (u*, f*=phi(u*)) is an exact fixed point, then verifies the
    held state stays put in *activity/readout* space (decode phi(u_T)), not just at
    the trivial origin (B9).
    """
    n = 32
    phi = Nonlinearity(a=1.0, b=0.0, c=1.0)
    f_star = place_code(0.5, n, width=0.08)  # nonneg activity bump (in range of phi)
    # Invert phi: phi(u)=log(1+[u]_+) -> u* = exp(f*)-1 >= 0, so phi(u*) == f* exactly.
    u_star = torch.expm1(f_star)
    s = _symmetric_with_fixed_point(u_star, f_star)
    a = torch.zeros(n, n, dtype=torch.float64)

    assert float(f_star.max()) > 0.5  # nontrivial (not the origin)
    with torch.no_grad():
        torch.testing.assert_close(phi(u_star), f_star)  # confirm f* = phi(u*)
        u_t = hold(u_star, s_mat=s, a_mats=a, alpha=1.0, tau=5.0, dt=1.0, phi=phi, n_steps=50)
        # State is an exact fixed point, so it is held to numerical precision...
        torch.testing.assert_close(u_t, u_star, rtol=1e-6, atol=1e-8)
        # ...and the decoded location in activity space is unchanged.
        decoded = float(decode_place(phi(u_t), n))
    assert decoded == pytest.approx(0.5, abs=1e-3)


def test_velocity_reversal_is_antisymmetric_in_drive() -> None:
    """Reversing v negates only the antisymmetric transport contribution to one step."""
    n = 6
    s, a = _weights(n)
    phi = Nonlinearity()
    u = torch.randn(n, dtype=torch.float64)
    kw = dict(s_mat=s, a_mats=a, alpha=50.0, tau=5.0, dt=1.0, phi=phi)
    u_fwd = step(u, velocity=1.0, **kw)
    u_bwd = step(u, velocity=-1.0, **kw)
    u_hold = step(u, velocity=0.0, **kw)
    torch.testing.assert_close(0.5 * (u_fwd + u_bwd), u_hold)


def test_time_step_convergence_to_continuous_ode() -> None:
    """Halving dt halves the error vs a fine-dt reference: first-order (Euler) convergence."""
    n = 5
    s, a = _weights(n)
    phi = Nonlinearity(a=1.0, b=5.0, c=1.0)  # smooth, active region
    u0 = 0.3 * torch.randn(n, dtype=torch.float64)
    tau, alpha, horizon = 5.0, 0.0, 5.0

    def integrate(dt: float) -> torch.Tensor:
        n_steps = round(horizon / dt)
        return run(
            u0,
            s_mat=s,
            a_mats=a,
            velocity=0.0,
            alpha=alpha,
            tau=tau,
            dt=dt,
            phi=phi,
            n_steps=n_steps,
        )

    with torch.no_grad():
        ref = integrate(horizon / 20000)
        err_coarse = float((integrate(horizon / 250) - ref).norm())
        err_fine = float((integrate(horizon / 500) - ref).norm())
    ratio = err_coarse / err_fine
    assert 1.7 < ratio < 2.3, f"expected ~2x error reduction (first order), got {ratio:.2f}"


def test_move_transports_decoded_location() -> None:
    """A shift generator moves the decoded bump: +v advances, -v retreats (in activity space)."""
    n = 64
    # Nearest-neighbor antisymmetric shift on the line (transport generator).
    shift = torch.zeros(n, n, dtype=torch.float64)
    idx = torch.arange(n - 1)
    shift[idx + 1, idx] = 1.0
    shift[idx, idx + 1] = -1.0
    a = antisymmetric_part(shift)
    s = torch.zeros(n, n, dtype=torch.float64)
    # Monotone-increasing nonlinearity in its active region (preserves bump ordering);
    # its derivative ~1/(51+u) here is far from 1, so it is NOT "near-linear".
    phi = Nonlinearity(a=1.0, b=50.0, c=1.0)

    u0 = place_code(0.5, n, width=0.05)
    kw = dict(s_mat=s, a_mats=a, alpha=1.0, tau=5.0, dt=0.1, phi=phi, n_steps=10)
    with torch.no_grad():
        x_fwd = float(decode_place(phi(run(u0, velocity=1.0, **kw)), n))
        x_bwd = float(decode_place(phi(run(u0, velocity=-1.0, **kw)), n))
    assert x_fwd > 0.5 > x_bwd


# --- codes -------------------------------------------------------------------


def test_place_code_peaks_at_location() -> None:
    n = 32
    center_idx = 10
    x = center_idx / (n - 1)
    c = place_code(x, n, width=0.05)
    assert int(c.argmax()) == center_idx
    assert float(c.max()) == pytest.approx(1.0)


def test_place_readout_recovers_interior_location() -> None:
    n = 64
    for x in [0.3, 0.5, 0.7]:
        decoded = decode_place(place_code(x, n, width=0.03), n)
        assert float(decoded) == pytest.approx(x, abs=1e-3)


def test_place_readout_is_boundary_biased() -> None:
    """The truncated-bump decoder is biased inward near the boundaries (documented)."""
    n = 64
    decoded_low = float(decode_place(place_code(0.0, n, width=0.05), n))
    decoded_high = float(decode_place(place_code(1.0, n, width=0.05), n))
    assert decoded_low > 0.0
    assert decoded_high < 1.0


def test_circular_distance_wraps() -> None:
    mu = torch.tensor([0.0, 0.5, 0.9], dtype=torch.float64)
    q = torch.tensor(0.05, dtype=torch.float64)
    d = circular_distance(q, mu)
    torch.testing.assert_close(d, torch.tensor([0.05, 0.45, 0.15], dtype=torch.float64))


def test_ring_code_periodic() -> None:
    n = 24
    torch.testing.assert_close(ring_code(0.1, n), ring_code(1.1, n))
    torch.testing.assert_close(ring_code(0.1, n), ring_code(-0.9, n))


def test_ring_code_wraps_smoothly_across_boundary() -> None:
    """The ring code is continuous across the 0/1 seam (phases 0.999 and 0.001 are close)."""
    n = 32
    near_zero = ring_code(0.001, n)
    near_one = ring_code(0.999, n)
    circular_gap = float((near_zero - near_one).abs().max())
    no_wrap_gap = float((ring_code(0.001, n) - ring_code(0.5, n)).abs().max())
    assert circular_gap < 0.1 * no_wrap_gap


def test_code_width_and_size_validation() -> None:
    with pytest.raises(ValueError, match="width must be positive"):
        place_code(0.5, 8, width=0.0)
    with pytest.raises(ValueError, match="width must be positive"):
        ring_code(0.5, 8, width=-0.1)
    with pytest.raises(ValueError, match="at least 2 units"):
        place_code(0.5, 1)
    with pytest.raises(ValueError, match="at least 1 unit"):
        ring_code(0.5, 0)


def test_mixed_code_spec_validation() -> None:
    with pytest.raises(ValueError, match="at least 2 place"):
        MixedCodeSpec(n_place=1, n_periodic=4, frequency=3)
    with pytest.raises(ValueError, match="at least 1 periodic"):
        MixedCodeSpec(n_place=4, n_periodic=0, frequency=3)
    with pytest.raises(ValueError, match="frequency"):
        MixedCodeSpec(n_place=4, n_periodic=4, frequency=0)
    with pytest.raises(ValueError, match="width must be positive"):
        MixedCodeSpec(n_place=4, n_periodic=4, frequency=3, width=0.0)


def test_mixed_code_layout() -> None:
    spec = MixedCodeSpec(n_place=8, n_periodic=8, frequency=3, width=0.05)
    c = mixed_code(0.4, spec)
    assert c.shape == (16,)
    torch.testing.assert_close(c[:8], place_code(0.4, 8, width=0.05))
    torch.testing.assert_close(c[8:], ring_code((3 * 0.4) % 1.0, 8, width=0.05))


def test_mixed_code_decodes_via_place_block() -> None:
    spec = MixedCodeSpec(n_place=48, n_periodic=16, frequency=5, width=0.03)
    decoded = decode_place(mixed_code(0.6, spec), spec.n_place)
    assert float(decoded) == pytest.approx(0.6, abs=1e-3)


# --- device / dtype ----------------------------------------------------------


def test_codes_follow_input_dtype_float32() -> None:
    x = torch.tensor(0.5, dtype=torch.float32)
    c = place_code(x, 16)
    assert c.dtype == torch.float32
    decoded = decode_place(c, 16)
    assert decoded.dtype == torch.float32
    assert float(decoded) == pytest.approx(0.5, abs=1e-2)


def test_ring_code_follows_input_dtype_float32() -> None:
    q = torch.tensor(0.3, dtype=torch.float32)
    assert ring_code(q, 16).dtype == torch.float32


def test_state_noise_dtype_float32() -> None:
    g = torch.Generator().manual_seed(0)
    xi = state_noise((10,), 0.1, generator=g, dtype=torch.float32)
    assert xi.dtype == torch.float32


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_codes_and_noise_on_cuda() -> None:
    x = torch.tensor(0.5, device="cuda")
    c = place_code(x, 16)
    assert c.device.type == "cuda"
    assert decode_place(c, 16).device.type == "cuda"
    g = torch.Generator(device="cuda").manual_seed(0)
    xi = state_noise((8,), 0.1, generator=g, device="cuda")
    assert xi.device.type == "cuda"


# --- noise -------------------------------------------------------------------


def test_state_noise_statistics() -> None:
    g = torch.Generator().manual_seed(0)
    sigma = 0.1
    samples = state_noise((200000,), sigma, generator=g)
    assert abs(float(samples.mean())) < 2e-3
    assert abs(float(samples.std()) - sigma) < 5e-3


def test_state_noise_rejects_negative_sigma() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        state_noise((4,), -0.1)


def test_gaussian_weight_perturbation_statistics() -> None:
    g = torch.Generator().manual_seed(0)
    sigma = 0.1
    xi = gaussian_weight_perturbation(400, sigma, generator=g)
    assert xi.shape == (400, 400)
    assert abs(float(xi.std()) - sigma) < 5e-3


def test_uniform_weight_perturbation_range_and_variance() -> None:
    g = torch.Generator().manual_seed(0)
    sigma = 0.2
    xi = uniform_weight_perturbation(400, sigma, generator=g)
    assert float(xi.abs().max()) <= sigma
    assert abs(float(xi.var()) - sigma**2 / 3.0) < 1e-3


def test_weight_sampler_per_step_resamples() -> None:
    g = torch.Generator().manual_seed(0)
    sampler = WeightNoiseSampler("per_step", 5, 0.1, generator=g)
    assert not torch.equal(sampler.sample(), sampler.sample())


def test_weight_sampler_per_trajectory_caches_then_resets() -> None:
    g = torch.Generator().manual_seed(0)
    sampler = WeightNoiseSampler("per_trajectory", 5, 0.1, generator=g)
    first = sampler.sample()
    torch.testing.assert_close(sampler.sample(), first)  # reused within a trajectory
    sampler.reset()
    assert not torch.equal(sampler.sample(), first)  # fresh draw for the next trajectory


def test_weight_sampler_fixed_returns_supplied() -> None:
    fixed = torch.randn(5, 5, dtype=torch.float64)
    sampler = WeightNoiseSampler("fixed", 5, 0.1, fixed=fixed)
    torch.testing.assert_close(sampler.sample(), fixed)
    torch.testing.assert_close(sampler.sample(), fixed)


def test_weight_sampler_rejects_invalid_mode() -> None:
    with pytest.raises(ValueError, match="unknown resample mode"):
        WeightNoiseSampler("typo", 5, 0.1)  # type: ignore[arg-type]


def test_weight_sampler_fixed_requires_matrix() -> None:
    with pytest.raises(ValueError, match="requires a `fixed`"):
        WeightNoiseSampler("fixed", 5, 0.1)


def test_weight_sampler_fixed_shape_validated() -> None:
    with pytest.raises(ValueError, match=r"must be \(5, 5\)"):
        WeightNoiseSampler("fixed", 5, 0.1, fixed=torch.zeros(4, 4, dtype=torch.float64))


def test_weight_sampler_fixed_only_for_fixed_mode() -> None:
    with pytest.raises(ValueError, match="only valid for mode='fixed'"):
        WeightNoiseSampler("per_step", 5, 0.1, fixed=torch.zeros(5, 5, dtype=torch.float64))


def test_weight_sampler_rejects_negative_sigma() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        WeightNoiseSampler("per_step", 5, -0.1)


def test_run_resets_per_trajectory_sampler_between_trajectories() -> None:
    """Two run() calls with one per_trajectory sampler get independent draws."""
    n = 5
    s, a = _weights(n)
    phi = Nonlinearity()
    u0 = torch.randn(n, dtype=torch.float64)
    g = torch.Generator().manual_seed(0)
    sampler = WeightNoiseSampler("per_trajectory", n, 0.1, generator=g)
    kw = dict(s_mat=s, a_mats=a, velocity=0.0, alpha=10.0, tau=5.0, dt=1.0, phi=phi, n_steps=5)
    with torch.no_grad():
        first = run(u0, weight_sampler=sampler, **kw)
        second = run(u0, weight_sampler=sampler, **kw)
    assert not torch.allclose(first, second)


def test_run_rejects_negative_state_sigma() -> None:
    n = 5
    s, a = _weights(n)
    phi = Nonlinearity()
    with pytest.raises(ValueError, match="state_sigma must be non-negative"):
        run(
            torch.randn(n, dtype=torch.float64),
            s_mat=s,
            a_mats=a,
            velocity=0.0,
            alpha=1.0,
            tau=5.0,
            dt=1.0,
            phi=phi,
            n_steps=3,
            state_sigma=-0.1,
        )


def test_noiseless_run_is_deterministic() -> None:
    n = 6
    s, a = _weights(n)
    phi = Nonlinearity()
    u0 = torch.randn(n, dtype=torch.float64)
    kw = dict(s_mat=s, a_mats=a, velocity=0.2, alpha=10.0, tau=5.0, dt=1.0, phi=phi, n_steps=20)
    with torch.no_grad():
        torch.testing.assert_close(run(u0, **kw), run(u0, **kw))


# --- gradient through time ---------------------------------------------------


def test_gradient_through_time_vs_finite_difference() -> None:
    """BPTT gradient of a restoration loss w.r.t. S, A, and the nonlinearity params.

    Packs the raw nonlinearity parameters (raw_a, b, raw_c) alongside W, V so the
    finite-difference check also covers the trainable nonlinearity (B2).
    """
    torch.manual_seed(5)
    n = 5
    u0 = 0.2 * torch.randn(n, dtype=torch.float64)
    target = 0.2 * torch.randn(n, dtype=torch.float64)
    tau, dt, alpha, v = 5.0, 1.0, 10.0, 0.3

    def loss_from_params(packed: torch.Tensor) -> torch.Tensor:
        w = packed[: n * n].reshape(n, n)
        vv = packed[n * n : 2 * n * n].reshape(n, n)
        raw_a, b, raw_c = packed[2 * n * n :]
        s = symmetric_part(w)
        a = antisymmetric_part(vv)
        a_eff = torch.nn.functional.softplus(raw_a)
        c_eff = torch.nn.functional.softplus(raw_c)

        def phi(u: torch.Tensor) -> torch.Tensor:
            return log_rectifier(u, a_eff, b, c_eff)

        u_t = u0
        for _ in range(6):
            w_eff = s + alpha * v * a
            u_t = u_t + (dt / tau) * (-u_t + phi(u_t) @ w_eff.T)
        return ((phi(u_t) - target) ** 2).sum()

    packed = torch.randn(2 * n * n + 3, dtype=torch.float64, requires_grad=True)
    loss_from_params(packed).backward()
    analytic = packed.grad
    assert analytic is not None
    numeric = finite_difference_grad(lambda p: loss_from_params(p), packed.detach())
    torch.testing.assert_close(analytic.to(torch.float64), numeric, rtol=BPTT_RTOL, atol=BPTT_ATOL)
