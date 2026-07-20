# Math spec — noise-robust recurrent location codes (revised Chapter 4)

Standalone mathematical specification for Phase 3B (plan §7). Authoritative
statement of the recurrent model the code implements; code docstrings, LaTeX
(`legacy/rewrite-2026/source/chap4.tex`, `chap2.tex`, `appendix.tex`), and tests
(`tests/phase3/test_attractor_*`) all agree with it (Gate P3).

Bracketed labels are the LaTeX `\label{}`s in the 2026 source. Correction IDs
(`b26-corr-*`) are rows in `docs/rewrite-2026-inventory.csv`; their disposition is
decision 0004.

## 1. Objects and dimensions

| Symbol | Shape | Meaning |
|---|---|---|
| `u_t` | `(N,)` | recurrent internal state at step `t` |
| `f_t = phi(u_t)` | `(N,)` | activity |
| `S` | `(N, N)` | symmetric restoration component, `S^T = S` |
| `A_d` | `(N, N)` | antisymmetric transport component(s), `A_d^T = -A_d` |
| `v_t` | scalar (per direction `d`) | signed velocity control |
| `Xi_t` | `(N, N)` | weight perturbation |
| `xi_t` | `(N,)` | state perturbation |
| `c(x)` | `(N,)` | prescribed codeword for location `x` |

Scalars: `tau` time constant, `dt` step, `alpha` velocity gain, `beta` write gate.

## 2. Discrete recurrence (explicit time step)

Model step [`eq:attractor-discrete`], first-order in `dt/tau` — **not** absolute `t`
(`b26-corr-recurrence`):

    u_{t+1} = u_t + (dt/tau) [ -u_t + (S + sum_d alpha v_{d,t} A_d + Xi_t) f_eff_t + xi_t ]
    f_t     = phi(u_t).

Activity nonlinearity [`eq:attractor-nonlinearity`], trainable scalars `a, b, c`:

    phi(u) = a * log(1 + c * [u + b]_+),   [q]_+ = max(q, 0).

Effective recurrent matrix [`eq:attractor-effective-weight`,
`eq:attractor-2d-transport`]:

    W_eff(v) = S + alpha * sum_d v_d A_d,   S^T = S,   A_d^T = -A_d.

Verified: step matches the definition (`test_step_matches_discrete_update`);
`v = 0` gives symmetric-only drive (`test_effective_weight_zero_velocity_is_
symmetric_only`); first-order (Euler) convergence in `dt`
(`test_time_step_convergence_to_continuous_ode`, ~2× error reduction per halving).

**Transport is multi-directional throughout the API** (`sum_d v_d A_d`): `step`,
`run`, `hold`, and `effective_weight` all accept either a single `(A, v)` pair (1D)
or matching sequences of transport matrices and velocities (2D+,
`eq:attractor-2d-transport`), so the executable spec is not silently 1D-only.
Velocities may be Python scalars or tensors; **tensor velocities are preserved**
(not cast to float), keeping a differentiable/on-device control differentiable
(`test_effective_weight_preserves_tensor_velocity_gradient`,
`test_step_multi_direction_matches_sum`).

## 3. Structure by parameterization (not post-hoc) [plan 3B.2]

`S` and `A_d` are built from unconstrained matrices `W`, `V_d`:

    S = (W + W^T)/2,   A_d = (V_d - V_d^T)/2.

This guarantees exact symmetry/antisymmetry for every parameter value and every
gradient step, so **no projection is needed after updates** — resolving the 2016
detail flagged as unresolved (how symmetry was enforced after updates). Verified to
machine zero in `test_symmetric_part_exact` / `test_antisymmetric_part_exact`, and
`A` has zero diagonal (`test_antisymmetric_diagonal_is_zero`).

## 4. Operations: write, hold, read, move [§4.4]

- **Write** (`write`) [`eq:attractor-write`]: `f_eff_t = (1 - beta_t) phi(u_t) + beta_t c(x_write)`,
  `beta_t in [0, 1]`. `beta = 1` writes the codeword; `beta = 0` runs autonomously.
  `beta` is validated **unconditionally** in `step` — a value outside `[0, 1]` (e.g.
  `-1`) raises rather than silently running autonomous dynamics.
- **Hold** (`hold`): zero velocity, no write input — maintains state.
- **Read**: decode `x_hat` from activity (§6).
- **Move** (`move`): nonzero velocity — the antisymmetric term transports the pattern.

All four (plus the general `run`) are **multidirectional**: `a_mats`/`velocity` take a
single matrix/scalar (1D) or matching sequences (2D+); `hold` supplies a zero velocity
per transport direction. **Velocity is constant per call** in this Phase 3
formalization — the notation `v_{d,t}` (per-time-step velocity) is a documented future
extension realized by a caller-owned `step` loop.

Velocity reversal negates only the transport contribution:
`0.5 (step(+v) + step(-v)) == step(0)`
(`test_velocity_reversal_is_antisymmetric_in_drive`). A nearest-neighbor
antisymmetric shift generator transports the decoded bump forward for `+v` and back
for `-v` (`test_move_transports_decoded_location`).

## 5. Noise convention and resampling protocol [plan 3B.4]

The 2016 text is ambiguous about noise scaling and resampling. We fix explicit,
documented choices:

- **Scaling**: `xi_t` and `Xi_t` enter *inside the bracket*, on the same footing as
  `-u + W f`; the `dt/tau` factor multiplies the whole bracket. A reported noise std
  `sigma` means `xi_t ~ N(0, sigma^2)` added to the bracket, net contribution to
  `u_{t+1}` scaled by `dt/tau`.
- **Resampling** (`WeightNoiseSampler`, three *different experiments*):
  - `per_step` — fresh sample every step (per-step robustness sweep,
    `fig:attractor-random-noise`);
  - `per_trajectory` — one sample reused across a trajectory's steps;
  - `fixed` — one externally supplied perturbation held constant across all
    trajectories/steps (fixed-connection perturbation, `fig:attractor-fixed-perturbation`).
- **Distributions**: Gaussian `N(0, sigma^2)` for state noise and the per-step /
  per-trajectory weight sweeps; **uniform** `U(-sigma, sigma)` for the fixed
  perturbation (baseline protocol). Note: an unconstrained random matrix has both
  symmetric and antisymmetric parts, so it can deform restoration *and* add drift —
  a fair test must perturb `S` and `A` separately.

Statistics verified in `test_state_noise_statistics`,
`test_uniform_weight_perturbation_range_and_variance`; protocols in the
`test_weight_sampler_*` tests. All draws take an explicit `torch.Generator` for
reproducibility.

## 6. Prescribed codes and decoder [§4.6]

Line/place code [`eq:place-line-code`], centers `mu_i = i/(N-1)`:

    (P_line[x])_i = exp[ -(x - mu_i)^2 / (2 w^2) ].

Ring/periodic code [`eq:ring-code`], centers `mu_i = i/N`, circular distance
`d_i(q) = min(|q - mu_i|, 1 - |q - mu_i|)`:

    (P_ring[q])_i = exp[ -d_i(q)^2 / (2 w^2) ].

Mixed place–periodic code [`eq:place-grid-code`]: `N_0` place units stacked over
`N_1` ring units at phase `k x mod 1`, `N = N_0 + N_1`.

Decoder [`eq:place-readout`]: population-vector average over the **place block**
only, `x_hat = sum_i f_i mu_i / sum_i f_i` (`i` over the `N_0` place units). The
baseline states the periodic-block readout is under-specified, so we decode global
location from the place block for both place-only and mixed codes and document that
choice. The place decoder is **boundary-biased** (truncated Gaussian bump) — a known
bias flagged for fair comparison, tested in `test_place_readout_is_boundary_biased`,
not hidden.

## 7. Linear-dynamics corrections [primer §recurrent]

1. **Amplification depends on the sign of a continuous-time eigenvalue, not on
   exceeding one** (`b26-corr-recurrence`). Linearizing gives generator
   `M = -I + W_lin`; a mode grows iff `Re(eig(M)) > 0`, decays iff `< 0`. Comparing a
   raw weight eigenvalue against 1 is the discrete-map heuristic and is not the right
   criterion for the leaky system. `growth_rates` / `spectral_abscissa` /
   `is_asymptotically_stable` implement the correct test
   (`test_growth_rate_uses_sign_not_unit_threshold`). At a fixed point of the
   *nonlinear* recurrence the linearization is `M = -I + W_eff diag(phi'(u*))`
   (`jacobian_generator`), which absorbs the activation derivative; passing raw `S`
   treats `phi` as locally the identity.

   **Asymptotic stability is not Euclidean contraction (nonnormal caveat).** For the
   generally *nonnormal* `M`, a negative spectral abscissa guarantees only eventual
   decay; the 2-norm can grow transiently by an arbitrarily large factor first.
   Monotone contraction is governed by the log-norm `mu_2(M)` = largest eigenvalue of
   the symmetric part `(M + M^T)/2` (since `d||x||^2/dt = x^T(M + M^T)x`). The two are
   separate functions: `is_asymptotically_stable` (spectral abscissa) vs
   `is_contractive` (`log_norm`). Counterexample `W = [[0,100],[0,0]]` is
   asymptotically stable yet not contractive (tested in
   `test_nonnormal_stable_but_not_contractive`,
   `test_transient_amplification_then_decay_for_nonnormal_stable`).

2. **An antisymmetric generator is norm-preserving only in continuous time**
   (`b26-corr-energy` context). For `dx/dt = A x` with `A^T = -A`, `exp(A t)` is
   orthogonal, so `||x||` is conserved exactly; the discrete Euler map `I + dt A` is
   **not** norm-preserving. Contrasted in
   `test_continuous_flow_preserves_norm_but_euler_step_does_not`. A pure `A` has
   purely imaginary spectrum (`test_pure_antisymmetric_generator_has_imaginary_spectrum`).

3. **No "energy landscape" language** for the nonsymmetric dynamics
   (`b26-corr-energy`). `A` does not define a scalar energy; robustness is *measured*
   from trajectories/basins, never inferred from the magnitude of `A`. The code and
   docstrings use fixed-point/basin language only.

## 8. Training objective [§4.5]

Restoration-through-time loss [`eq:attractor-loss`]:

    L_RNN = || c(x_target) - f_T ||^2,

with gradients propagated through all steps (BPTT). Hold training uses `v = 0` and
updates `S` + nonlinearity; move training uses `v != 0` and updates `A`. The
nonlinearity is a trainable `nn.Module` (§2): `a`, `b`, `c` are learnable, with `a`
and `c` kept strictly positive by a softplus parameterization (`a > 0` keeps
activity nonnegative for the decoder; `c > 0` keeps the `log1p` argument valid). The
BPTT gradient w.r.t. `S`, `A`, **and the nonlinearity parameters** (through the
parameterization) matches finite differences
(`test_gradient_through_time_vs_finite_difference`, `test_log_rectifier_gradients_vs_
finite_difference`). Training itself is a later phase; Phase 3 only formalizes and
tests the mathematics.

## 9. Historical math issues and their resolutions (Gate P3)

| Issue (inventory row) | Resolution |
|---|---|
| Update written with absolute `t` (`b26-corr-recurrence`) | First-order in `dt/tau`; Euler convergence tested |
| Amplification vs "eigenvalue > 1" (`b26-corr-recurrence`) | Continuous generator `M = -I + W`; sign of `Re(eig(M))` |
| Asymptotic stability conflated with contraction (nonnormal) | Split `is_asymptotically_stable` (spectral abscissa) vs `is_contractive` (log-norm); nonnormal counterexample tested |
| "Energy landscape" for nonsymmetric dynamics (`b26-corr-energy`) | Fixed-point/basin language; robustness measured, not inferred |
| Symmetry/antisymmetry enforcement after updates (ambiguous) | Enforced by parameterization `S=(W+W^T)/2`, `A=(V-V^T)/2` |
| Trainable nonlinearity `a, b, c` | `nn.Module` params; `a, c` positive via softplus; grads FD-tested |
| Noise scaling / resampling ambiguous | Explicit bracket-scaling + three named resampling modes (with `reset` at trajectory boundaries) |
| Mixed-code decoder under-specified | Decode from the place block; choice documented |

## 10. Tolerances

- `tol-gradient-fd` — BPTT analytic gradient vs central finite differences (float64),
  the `BPTT_*` tier of the registry row shared with the sparse-coding checks.

Committed in `docs/tolerances.md`.
