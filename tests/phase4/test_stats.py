"""Tests for whitening/pipeline stats (de)serialization schema (Phase 4; R2 finding 9)."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from neural_repr.data import fit_whitening
from neural_repr.data.stats import (
    load_whitening,
    save_whitening,
    whitening_from_dict,
    whitening_to_dict,
)

RNG = np.random.default_rng(0)


def _stats(f: int = 6):
    x = RNG.standard_normal((300, f))
    return fit_whitening(x, eps=1e-6, input_manifest_sha256="sha256:" + "a" * 64)


def test_roundtrip_ok(tmp_path) -> None:
    stats = _stats()
    p = tmp_path / "w.json"
    save_whitening(stats, p)
    loaded = load_whitening(p)
    np.testing.assert_allclose(loaded.whiten, stats.whiten)
    assert loaded.input_manifest_sha256 == stats.input_manifest_sha256


def test_save_is_collision_failing(tmp_path) -> None:
    stats = _stats()
    p = tmp_path / "w.json"
    save_whitening(stats, p)
    with pytest.raises(FileExistsError):
        save_whitening(stats, p)
    save_whitening(stats, p, overwrite=True)  # explicit override works


def test_rejects_unknown_key() -> None:
    d = whitening_to_dict(_stats())
    d["surprise"] = 1
    with pytest.raises(ValueError, match="unknown stats field"):
        whitening_from_dict(d)


def test_rejects_bad_manifest_digest() -> None:
    d = whitening_to_dict(_stats())
    d["input_manifest_sha256"] = "not-a-digest"
    with pytest.raises(ValueError, match="sha256"):
        whitening_from_dict(d)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("eps", -1.0, "eps"),
        ("eps", float("nan"), "eps"),
        ("n_samples", 1, "n_samples"),
        ("n_features", 0, "n_features"),
    ],
)
def test_rejects_bad_scalars(field: str, value: object, match: str) -> None:
    d = whitening_to_dict(_stats())
    d[field] = value
    with pytest.raises(ValueError, match=match):
        whitening_from_dict(d)


def test_rejects_shape_mismatch() -> None:
    d = whitening_to_dict(_stats(6))
    d["n_features"] = 5  # inconsistent with the 6x6 arrays
    with pytest.raises(ValueError):
        whitening_from_dict(d)


def test_rejects_non_finite_matrix() -> None:
    d = whitening_to_dict(_stats())
    w = np.asarray(d["whiten"], dtype=np.float64)
    w[0, 0] = np.inf
    d["whiten"] = w.tolist()
    with pytest.raises(ValueError, match="non-finite"):
        whitening_from_dict(d)


def test_rejects_non_inverse_matrices() -> None:
    d = whitening_to_dict(_stats())
    unw = np.asarray(d["unwhiten"], dtype=np.float64) * 2.0  # no longer the inverse
    d["unwhiten"] = unw.tolist()
    with pytest.raises(ValueError, match="not the identity"):
        whitening_from_dict(d)


def test_fitted_pipeline_artifact_roundtrip(tmp_path) -> None:
    from neural_repr.data import (
        RolePatches,
        fit_representation_pipeline,
        load_fitted_pipeline,
        save_fitted_pipeline,
        synthetic_image,
    )
    from neural_repr.data.patches import PatchConfig, sample_random_patches

    img = synthetic_image(0, size=48, seed=0).astype(np.float64) / 255.0
    patches = sample_random_patches(
        img, PatchConfig(size=12, n_random=40), dataset_sha256="s", image_id="0", seed=0
    )
    fitted = fit_representation_pipeline([RolePatches("train", patches)])
    p = tmp_path / "pipeline.json"
    save_fitted_pipeline(fitted, p)
    loaded = load_fitted_pipeline(p)
    assert loaded.fit_id == fitted.fit_id
    assert loaded.global_scale == fitted.global_scale
    np.testing.assert_allclose(loaded.whitening.whiten, fitted.whitening.whiten)
    # Collision-failing.
    with pytest.raises(FileExistsError):
        save_fitted_pipeline(fitted, p)


_ = copy  # reserved for future deep-copy adversarial cases
