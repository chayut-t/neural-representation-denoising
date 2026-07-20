"""Tests for the data audit report (Phase 4 task 11)."""

from __future__ import annotations

import numpy as np

from neural_repr.data import (
    before_after_summary,
    channel_histograms,
    fit_whitening,
    patch_power,
    train_test_similarity,
    whitening_covariance_check,
)

RNG = np.random.default_rng(0)


def test_channel_histograms_shape_and_counts() -> None:
    imgs = RNG.uniform(0, 1, (4, 16, 16, 3))
    hist = channel_histograms(imgs, bins=8)
    assert len(hist["bin_edges"]) == 9
    counts = hist["counts_per_channel"]
    assert len(counts) == 3
    # Every value falls in some bin: per-channel counts sum to the pixel count.
    assert all(sum(c) == 4 * 16 * 16 for c in counts)


def test_patch_power_positive() -> None:
    patches = RNG.uniform(0, 1, (20, 64, 3))
    stats = patch_power(patches)
    assert stats["p05"] <= stats["median"] <= stats["p95"]
    assert stats["mean"] > 0


def test_whitening_covariance_check_near_identity_on_fit_data() -> None:
    x = RNG.standard_normal((2000, 6))
    stats = fit_whitening(x, eps=1e-8)
    check = whitening_covariance_check(x, stats)
    assert check["max_abs_off_identity"] < 1e-2


def test_train_test_similarity_bounds() -> None:
    train = RNG.standard_normal((100, 20))
    test = RNG.standard_normal((100, 20))
    sim = train_test_similarity(train, test)
    assert -1.0 <= sim["mean_nearest_cosine"] <= 1.0
    assert sim["max_nearest_cosine"] <= 1.0 + 1e-9


def test_train_test_similarity_flags_duplicates() -> None:
    train = RNG.standard_normal((50, 20))
    # A test set that copies some train rows -> nearest cosine == 1.
    test = train[:10].copy()
    sim = train_test_similarity(train, test)
    assert sim["max_nearest_cosine"] > 0.999


def test_before_after_summary_keys() -> None:
    before = RNG.uniform(0, 1, (10, 10, 3))
    after = before * 2.0
    summary = before_after_summary(before, after)
    assert set(summary["before"]) == {"mean", "std", "min", "max"}
    assert summary["after"]["max"] > summary["before"]["max"]


def test_build_and_write_audit_report(tmp_path) -> None:
    from neural_repr.data.audit import build_audit_report, write_audit_report

    rgb = RNG.uniform(0, 1, (40, 256, 3))
    train = RNG.standard_normal((40, 30))
    test = RNG.standard_normal((20, 30))
    stats = fit_whitening(train, eps=1e-6, input_manifest_sha256="sha256:" + "c" * 64)
    report = build_audit_report(
        rgb_patches=rgb,
        train_features=train,
        test_features=test,
        whitening=stats,
        dataset="synthetic",
        manifest_sha256="sha256:" + "c" * 64,
    )
    # All five components present + provenance.
    assert set(report) >= {
        "channel_histograms",
        "patch_power",
        "whitening_covariance",
        "train_test_similarity",
        "whitening_before_after",
        "dataset",
        "input_manifest_sha256",
    }
    assert report["dataset"] == "synthetic"
    assert report["n_train_features"] == 40

    import json

    path = tmp_path / "audit.json"
    write_audit_report(report, path)
    loaded = json.loads(path.read_text())
    assert loaded["input_manifest_sha256"] == "sha256:" + "c" * 64
