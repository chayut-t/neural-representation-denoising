# Math spec — group sparse coding (revised Chapter 3)

Standalone mathematical specification for Phase 3A (plan §7). This is the
authoritative statement of the model the code implements; the code docstrings, the
LaTeX (`legacy/rewrite-2026/source/chap3.tex`, `chap2.tex`, `appendix.tex`), and the
tests (`tests/phase3/test_sparse_*`, `test_gabor.py`) all agree with it (Gate P3).

Equation labels in brackets are the LaTeX `\label{}`s in the 2026 source, so every
formula here is traceable to the dissertation. Correction IDs (`b26-corr-*`) are the
rows in `docs/rewrite-2026-inventory.csv`; the disposition of each is decision 0004.

## 1. Objects and dimensions

| Symbol | Shape | Meaning |
|---|---|---|
| `x` | `(N,)` | image patch (clean observation) |
| `x_tilde` | `(N,)` | corrupted observation `x + nu` |
| `Phi` | `(N, M)` | dictionary; columns are atoms, each unit L2 norm |
| `s` | `(M,)` | sparse coefficient vector |
| `Lambda` | `(K, M)` | nonnegative group-to-coefficient weights; row `k` is group `k` |
| `z` | `(K,)` | group activities |

Color/channel ordering (relevant from Phase 4, fixed now for the spec): a color
patch of `C` channels and `P` pixels is vectorized channel-major,
`x = [ch0 pixels; ch1 pixels; ...]`, so `N = C * P`. The 2016 study was grayscale
(`C = 1`); the modern edition uses public color images (decision 0001), and the
dictionary/inference math is unchanged under this flattening.

## 2. Group activity and penalty

Numerically safe group activity [`eq:group-activity`], with documented smoothing
constant `eps > 0` (default `1e-6`, `DEFAULT_EPS`):

    z_k(s) = sqrt(eps + sum_i Lambda_ki s_i^2),   Lambda_ki >= 0.

Structured penalty [`eq:group-penalty`]: `Omega(s) = sum_k z_k(s)`.

Inference objective [`eq:group-objective`]:

    E(s; x, Phi, Lambda) = 0.5 ||x - Phi s||^2 + lambda * Omega(s).

`eps` keeps `z_k` and its gradient finite when a group is silent; it is a numerical
parameter, not fitted. Standard sparse coding is recovered (up to scale/smoothing)
when `K = M` and `Lambda` is diagonal — verified in
`test_diagonal_single_group_reduces_to_smoothed_abs`.

## 3. Gradients

Coefficient gradient of the penalty [`eq:group-coefficient-gradient`]:

    dOmega/ds_i = s_i * sum_k Lambda_ki / z_k(s).

This is a **signed gradient component**, not a nonnegative adaptive soft threshold
(`b26-corr-group-grad`). It carries `sign(s_i)` because the coupling
`sum_k Lambda_ki/z_k >= 0` — verified in `test_group_penalty_grad_is_signed`, and
against finite differences in `test_group_penalty_grad_vs_finite_difference`.

Group-weight gradient of the penalty [appendix `eq:app-group-weight-derivative`]:

    dOmega/dLambda_ki = s_i^2 / (2 z_k(s)).

The factor `1/2` comes from differentiating the square root and is retained exactly
(`b26-corr-ml-half`). It can be absorbed into a proportional learning rate but
belongs in an equality — verified in `test_lambda_gradient_has_explicit_half_factor`.

Dictionary gradient for a fixed code [`eq:dictionary-gradient`]:

    -dE/dPhi = (x - Phi s) s^T.

Each column is renormalized to unit L2 norm after an update to remove the
dictionary–coefficient scale ambiguity (primer §sparse).

## 4. Lambda parameterization (nonnegativity + scale)

`Lambda` is produced from an unconstrained matrix by **softplus + row-sum
normalization** (`softplus_lambda`, plan 3A.3, decision 0004). Softplus enforces
`Lambda_ki >= 0`. The `(Lambda, lambda)` scale non-identifiability — rescaling a
group's row can be undone by the penalty weight — is resolved by normalizing each
row (group) to unit sum. `normalization="none"` keeps raw softplus values for
reproducing a fixed historical initialization.

## 5. Inference variants (the documented ambiguity)

The 2016 inference update is genuinely ambiguous (code lost); its description mixes
ISTA soft thresholding with an adaptive gradient of the group penalty. Decision 0004
resolves this by implementing **two named variants**, neither presented as *the*
original:

- **`smooth_gradient`** (modern; the `proximal_group` slot) — the mathematically
  direct gradient step on the smoothed objective [`eq:group-gradient-step`]:

      s_{t+1} = s_t - eta [ Phi^T(Phi s_t - x) + lambda * dOmega/ds(s_t) ].

  A true (sub)gradient method for the smoothed penalty. With `lambda = 0` it is
  plain least-squares gradient descent (`test_smooth_gradient_zero_penalty_matches_
  least_squares`); at convergence it matches an independent scipy L-BFGS-B solve of
  `E` (`test_smooth_gradient_converges_to_trusted_convex_reference`,
  freezing `tol-solver-convergence`) and its deep-unrolled gradient matches the
  implicit-function-theorem gradient at the optimum
  (`test_unrolled_gradient_matches_implicit_function_gradient`).

- **`legacy_adaptive_threshold`** (historical-compatible) — an ISTA data step
  followed by a soft threshold whose per-coefficient magnitude is the group coupling
  `eta * lambda * sum_k Lambda_ki / z_k(s_t)`. As the baseline states, scalar soft
  thresholding with such a "threshold" is **not** the proximal operator of the
  overlapping-group penalty. **Parameter-dependent degeneracy:** with fixed
  `eps > 0` the coupling at `s = 0` is *finite* (`z_k = sqrt(eps)`), giving a large
  but finite first threshold `eta * lambda * sum_k Lambda_ki / sqrt(eps)`. Whether
  that threshold pins all coefficients on the first step depends on `eta`, `lambda`,
  `Lambda`, `eps`, and the data-step magnitude; when the threshold exceeds the data
  step, coefficients pin to zero and stay there
  (`test_legacy_threshold_at_zero_start_is_large_finite`,
  `test_legacy_threshold_pins_when_threshold_dominates_data_step`). This variant is
  for reproducing the historical finite computation (typically with a warm start),
  not for solving `eq:group-objective`.

Both are unrolled for fixed depth and are differentiable end-to-end (primer
§unrolling, [`eq:unrolled-state`]).

**Step size.** The data term has Lipschitz constant `L_data = ||Phi^T Phi||_2`
(`data_lipschitz`). The smoothed group penalty adds curvature bounded by
`L_pen = lambda * max_i (sum_k Lambda_ki) / sqrt(eps)` (from
`Hess(Omega) ⪯ diag(sum_k Lambda_ki)/sqrt(eps)`; `penalty_lipschitz`) — large near
`s = 0` for small `eps`. `smooth_gradient` takes an explicit step on the *total*
objective, so it must use `eta = safety / (L_data + L_pen)` (`stable_step_size`,
`0 < safety <= 1`), verified to descend monotonically across a grid of `eps` and
penalty weights (`test_stable_step_size_monotone_descent_across_eps_and_penalty`).
`lipschitz_step_size` (`safety / L_data`) is the data-only value, safe only for the
ISTA/proximal variant where the penalty is handled by the prox.

## 6. Denoising objective vs maximum likelihood

Corruption [`eq:group-corruption`]: `x_tilde = x + nu`, `nu ~ N(0, sigma^2 I)`.
Finite inference produces `s_T = F^{(T)}(x_tilde; Phi, Lambda)`,
`x_hat = Phi s_T`. Denoising loss [`eq:group-denoising-loss`]:

    L_denoise = || x - x_hat ||^2,   Delta Lambda ∝ -dL_denoise/dLambda.

Autodiff propagates the error through the decoder and all inference steps
(`test_autograd_through_smooth_gradient_matches_finite_difference`). This replaces
the two intractable expectations of the exact ML gradient
[`eq:lambda-likelihood-gradient`], which is the difference of `dOmega/dLambda_ki`
under the model prior and the posterior (appendix `eq:app-positive-negative`).

**Estimand distinction (preserved):** the denoising `Lambda` is optimized for the
finite inference rule, corruption distribution, loss, and schedule. It is **not**
claimed to be the maximum-likelihood group structure — a different estimand
(baseline §3.4–§3.5). `ml_group_grad` exposes only the ML integrand (with the `1/2`
factor) to document the surrogate relationship; it is not used for training.

## 7. Corrected Gabor (analysis model)

Learned atoms are summarized by the corrected Gabor [`eq:corrected-gabor`,
`b26-corr-gabor`]:

    g(x, y) = a * exp[ -1/2 ( x'^2 / sigma_x^2 + y'^2 / sigma_y^2 ) ] cos(k x' + phi)
    x' =  (x - t_x) cos(theta) - (y - t_y) sin(theta)
    y' =  (x - t_x) sin(theta) + (y - t_y) cos(theta)

The Gaussian **envelope** carries a **negative** quadratic exponent, so it decays
away from the center (the 2016 form grew). Factor convention: the `1/2` factor as
shown, `sigma_x`/`sigma_y` are standard deviations in the rotated frame, `k` is the
carrier angular frequency (rad/pixel) along `x'`, `theta` rotates the frame,
`(t_x, t_y)` is the center. Only the **envelope** is guaranteed to peak at the
center; the full Gabor `envelope * cos(k x' + phi)` need not — with `cos(phi) = 0` it
is zero at the center and larger just off it — though `|g| <= |a|` everywhere. Both
the envelope decay and the quadrature-phase behavior are verified in `test_gabor.py`.
This is a descriptive fit, not part of the trained objective.

## 8. Historical math issues and their resolutions (Gate P3)

| Issue (inventory row) | Resolution |
|---|---|
| Group gradient read as a soft threshold (`b26-corr-group-grad`) | Signed gradient in `smooth_gradient`; legacy adaptive-threshold reading kept as a separate, labeled variant (decision 0004) |
| ML `1/2` factor dropped (`b26-corr-ml-half`) | `1/2` retained exactly in `group_penalty_grad_lambda` / `ml_group_grad` |
| Gabor envelope grew away from center (`b26-corr-gabor`) | Negative quadratic exponent; envelope decay + quadrature-phase behavior tested |
| Adaptive-threshold reading not a true prox | Parameter-dependent (not universal) degeneracy: finite large threshold at `s=0`; pins only when it dominates the data step — both tested |
| Smooth-gradient step ignored penalty curvature | `stable_step_size` uses `L_data + L_penalty`; monotone descent stress-tested over `eps`/penalty |
| `Lambda` scale non-identifiable | softplus + row-sum normalization (recorded here + decision 0004); unknown normalization rejected |
| `z_k` undefined derivative at `s = 0` | `eps > 0` smoothing, documented numerical parameter |

## 9. Tolerances

- `tol-gradient-fd` — analytic vs central finite-difference gradients. Two named
  tiers (in `tests/phase3/fdcheck.py`, matching the registry): closed-form
  `FD_RTOL=1e-6`/`FD_ATOL=1e-9`, and unrolled/BPTT `BPTT_RTOL=1e-5`/`BPTT_ATOL=1e-7`
  (rounding compounds across steps). Used by every `*_vs_finite_difference` test.
- `tol-solver-convergence` — `smooth_gradient` (with `stable_step_size`) vs
  independent scipy L-BFGS-B solve of `E`, rel `1e-3` / abs `1e-4`.

Both are committed in `docs/tolerances.md`.
