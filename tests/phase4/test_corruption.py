"""Tests for AWGN corruption by sigma and by target SNR (Phase 4 task 9)."""

from __future__ import annotations

import numpy as np
import pytest

from neural_repr.data import add_awgn_sigma, add_awgn_snr, measured_snr_db, sigma_for_snr


def test_awgn_sigma_reproducible_with_generator() -> None:
    x = np.zeros((100, 100, 3))
    a = add_awgn_sigma(x, 0.1, np.random.default_rng(7))
    b = add_awgn_sigma(x, 0.1, np.random.default_rng(7))
    np.testing.assert_array_equal(a, b)


def test_awgn_sigma_zero_is_identity() -> None:
    x = np.random.default_rng(0).uniform(0, 1, (10, 10, 3))
    np.testing.assert_array_equal(add_awgn_sigma(x, 0.0, np.random.default_rng(0)), x)


def test_awgn_sigma_empirical_std() -> None:
    x = np.zeros((400, 400, 3))
    noisy = add_awgn_sigma(x, 0.2, np.random.default_rng(1))
    assert abs(float(noisy.std()) - 0.2) < 5e-3


def test_awgn_sigma_rejects_negative() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        add_awgn_sigma(np.zeros(4), -0.1, np.random.default_rng(0))


def test_awgn_sigma_rejects_nan() -> None:
    # R2-11: NaN sigma must be rejected (NaN < 0 is False, so a naive check misses it).
    with pytest.raises(ValueError, match="non-negative real"):
        add_awgn_sigma(np.zeros(4), float("nan"), np.random.default_rng(0))


@pytest.mark.parametrize("snr_db", [0.0, 3.0, 6.0])
def test_awgn_snr_hits_target(snr_db: float) -> None:
    rng = np.random.default_rng(2)
    signal = np.random.default_rng(3).standard_normal((128, 128, 3))
    noisy, sigma = add_awgn_snr(signal, snr_db, rng)
    assert sigma > 0
    # Measured SNR is close to target (large sample -> tight).
    assert abs(measured_snr_db(signal, noisy) - snr_db) < 0.3


def test_sigma_for_snr_matches_formula() -> None:
    signal = np.random.default_rng(4).standard_normal((64, 64))
    var = float(np.var(signal))
    for snr_db in (-3.0, 0.0, 10.0):
        expected = np.sqrt(var / (10.0 ** (snr_db / 10.0)))
        assert sigma_for_snr(signal, snr_db) == pytest.approx(expected)


def test_sigma_for_snr_constant_signal_raises() -> None:
    # A target SNR is undefined for a constant signal (variance 0).
    with pytest.raises(ValueError, match="undefined for a constant signal"):
        sigma_for_snr(np.full((8, 8), 0.5), 3.0)


def test_measured_snr_infinite_when_no_noise() -> None:
    x = np.random.default_rng(5).standard_normal((10, 10))
    assert measured_snr_db(x, x) == float("inf")
