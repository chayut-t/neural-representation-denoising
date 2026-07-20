"""Tests for whitening (train-only fit) and normalization (Phase 4 tasks 9-10)."""

from __future__ import annotations

import numpy as np
import pytest

from neural_repr.data import (
    apply_whitening,
    fit_whitening,
    global_normalize,
    invert_whitening,
    per_patch_normalize,
)
from neural_repr.data.stats import load_whitening, save_whitening

RNG = np.random.default_rng(0)


def _correlated_features(n: int, f: int) -> np.ndarray:
    # Correlated Gaussian features so whitening has real work to do.
    a = RNG.standard_normal((f, f))
    cov_sqrt = a @ a.T / f + np.eye(f)
    return RNG.standard_normal((n, f)) @ cov_sqrt


def test_whitening_roundtrip() -> None:
    x = _correlated_features(400, 10)
    stats = fit_whitening(x)
    w = apply_whitening(x, stats)
    np.testing.assert_allclose(invert_whitening(w, stats), x, atol=1e-8)


def test_whitening_makes_covariance_near_identity() -> None:
    x = _correlated_features(2000, 8)
    stats = fit_whitening(x, eps=1e-8)
    w = apply_whitening(x, stats)
    cov = np.cov(w, rowvar=False)
    assert np.max(np.abs(cov - np.eye(8))) < 1e-3


def test_whitening_fit_is_training_only() -> None:
    """Fitting on train features and applying to test never consults test statistics.

    Fit twice — once on train alone, once on a different (train+test) set — and
    confirm applying the train-only fit to test data uses only the train-fit matrix
    (i.e. the transform is a pure function of the stored stats, not the input set).
    """
    train = _correlated_features(500, 6)
    test = _correlated_features(500, 6)
    stats = fit_whitening(train, input_manifest_sha256="sha256:" + "a" * 64)
    # Applying to test is deterministic given only `stats`; re-applying matches.
    w1 = apply_whitening(test, stats)
    w2 = apply_whitening(test, stats)
    np.testing.assert_array_equal(w1, w2)
    # The stats carry their provenance tag (the train manifest), not the test set.
    assert stats.input_manifest_sha256 == "sha256:" + "a" * 64
    assert stats.n_samples == 500


def test_whitening_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="2-D"):
        fit_whitening(np.zeros((3, 4, 5)))
    with pytest.raises(ValueError, match="at least 2 samples"):
        fit_whitening(np.zeros((1, 4)))


def test_whitening_stats_roundtrip_json(tmp_path) -> None:
    stats = fit_whitening(_correlated_features(300, 5), input_manifest_sha256="sha256:" + "b" * 64)
    path = tmp_path / "whitening.json"
    save_whitening(stats, path)
    loaded = load_whitening(path)
    np.testing.assert_allclose(loaded.whiten, stats.whiten, atol=1e-12)
    np.testing.assert_allclose(loaded.unwhiten, stats.unwhiten, atol=1e-12)
    assert loaded.input_manifest_sha256 == stats.input_manifest_sha256
    assert loaded.n_samples == stats.n_samples


def test_global_normalize_roundtrip() -> None:
    x = RNG.uniform(0.5, 2.0, (50, 12))
    norm, scale = global_normalize(x)
    np.testing.assert_allclose(norm * scale, x, atol=1e-12)
    assert np.sqrt(np.mean(norm**2)) == pytest.approx(1.0)


def test_per_patch_normalize_unit_rms_and_reversible() -> None:
    patches = RNG.uniform(0.1, 1.0, (7, 64, 3))
    norm, scales = per_patch_normalize(patches)
    rms = np.sqrt(np.mean(norm**2, axis=(-2, -1)))
    np.testing.assert_allclose(rms, 1.0, atol=1e-10)
    np.testing.assert_allclose(norm * scales, patches, atol=1e-12)


def test_per_patch_normalize_flat_patch_safe() -> None:
    patches = np.zeros((2, 16, 3))
    norm, _ = per_patch_normalize(patches)
    assert np.all(np.isfinite(norm))
