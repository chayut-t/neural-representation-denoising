"""Tests for the config-driven train-only fit/apply pipeline (Phase 4; R2 finding 3)."""

from __future__ import annotations

import numpy as np
import pytest

from neural_repr.data import (
    LeakageError,
    RolePatches,
    apply_representation_pipeline,
    assert_fit_set_is_train_only,
    fit_representation_pipeline,
    synthetic_image,
)
from neural_repr.data.patches import PatchConfig, sample_random_patches


def _patches(image_index: int, seed: int = 0) -> np.ndarray:
    img = synthetic_image(image_index, size=48, seed=0).astype(np.float64) / 255.0
    return sample_random_patches(
        img,
        PatchConfig(size=12, n_random=40),
        dataset_sha256="synthetic",
        image_id=f"{image_index:04d}",
        seed=seed,
    )


def test_fit_uses_train_only_and_applies_to_all() -> None:
    train = RolePatches("train", _patches(0))
    val = RolePatches("val", _patches(1))
    test = RolePatches("test", _patches(2))
    fitted = fit_representation_pipeline([train, val, test])
    assert fitted.n_train_patches == 40
    # Apply to each split without error; whitened output has the expected feature dim.
    out = apply_representation_pipeline(test.patches, fitted)
    assert out.shape == (40, 12 * 12 * 3)


def test_fit_id_is_independent_of_test_content() -> None:
    """R2 finding 3: mutating test content must not change the fitted artifact id/bytes."""
    train = RolePatches("train", _patches(0))
    test_a = RolePatches("test", _patches(2))
    test_b = RolePatches("test", _patches(3))  # different test content

    fit_a = fit_representation_pipeline([train, test_a])
    fit_b = fit_representation_pipeline([train, test_b])
    assert fit_a.fit_id == fit_b.fit_id  # test content does not influence the fit
    np.testing.assert_array_equal(fit_a.whitening.whiten, fit_b.whitening.whiten)
    assert fit_a.global_scale == fit_b.global_scale


def test_fit_id_changes_with_train_content() -> None:
    fit_a = fit_representation_pipeline([RolePatches("train", _patches(0))])
    fit_b = fit_representation_pipeline([RolePatches("train", _patches(1))])
    assert fit_a.fit_id != fit_b.fit_id


def test_injecting_test_row_into_fit_set_is_rejected() -> None:
    """A val/test role in a proposed fit set is rejected at the boundary."""
    with pytest.raises(LeakageError, match="train-only"):
        assert_fit_set_is_train_only(["train", "train", "test"])
    assert_fit_set_is_train_only(["train", "train"])  # clean -> no raise


def test_fit_requires_train_rows() -> None:
    with pytest.raises(ValueError, match="no train-role patches"):
        fit_representation_pipeline([RolePatches("val", _patches(1))])


def test_pipeline_is_deterministic() -> None:
    rp = [RolePatches("train", _patches(0)), RolePatches("test", _patches(2))]
    a = fit_representation_pipeline(rp)
    b = fit_representation_pipeline(rp)
    assert a.fit_id == b.fit_id
    np.testing.assert_array_equal(
        apply_representation_pipeline(rp[1].patches, a),
        apply_representation_pipeline(rp[1].patches, b),
    )
