"""Data pipeline: registry, manifests, color/whitening/corruption/patch transforms.

Phase 4 (plan §7): deterministic, legal, inspectable, reversible data preparation.
Downloading and committing dataset bytes is out of scope — only manifests/hashes are
committed (decision 0001). See ``data/README.md`` for license handling and the
terms-acceptance flow. All array transforms are pure NumPy and reproducible on the
reference platform (Gate P4).
"""

from __future__ import annotations

from neural_repr.data.audit import (
    before_after_summary,
    build_audit_report,
    channel_histograms,
    patch_power,
    train_test_similarity,
    whitening_covariance_check,
    write_audit_report,
)
from neural_repr.data.color import (
    linear_to_srgb,
    opponent_matrix,
    opponent_to_rgb,
    remove_dc,
    restore_dc,
    rgb_to_opponent,
    srgb_to_linear,
)
from neural_repr.data.corruption import (
    add_awgn_sigma,
    add_awgn_snr,
    measured_snr_db,
    sigma_for_snr,
)
from neural_repr.data.manifests import (
    ManifestRow,
    build_manifest_row,
    inspect_image,
    read_manifest,
    resolve_under_root,
    validate_manifest,
    verify_manifest_on_disk,
    write_manifest,
)
from neural_repr.data.patches import (
    AUGMENTATIONS,
    PatchConfig,
    apply_augmentation,
    extract_grid_patches,
    reassemble_patches,
    sample_random_patches,
)
from neural_repr.data.pipeline import (
    FittedPipeline,
    LeakageError,
    RolePatches,
    apply_representation_pipeline,
    assert_fit_set_is_train_only,
    fit_representation_pipeline,
    load_fitted_pipeline,
    save_fitted_pipeline,
)
from neural_repr.data.registry import (
    BSDS500,
    DIV2K,
    DatasetSpec,
    SplitSpec,
    get_dataset,
    registered_datasets,
)
from neural_repr.data.stats import load_whitening, save_whitening
from neural_repr.data.synthetic import synthetic_dataset, synthetic_image
from neural_repr.data.whitening import (
    WhiteningStats,
    apply_whitening,
    fit_whitening,
    global_normalize,
    invert_whitening,
    per_patch_normalize,
)

__all__ = [
    "AUGMENTATIONS",
    "BSDS500",
    "DIV2K",
    "DatasetSpec",
    "FittedPipeline",
    "LeakageError",
    "ManifestRow",
    "PatchConfig",
    "RolePatches",
    "SplitSpec",
    "WhiteningStats",
    "add_awgn_sigma",
    "add_awgn_snr",
    "apply_augmentation",
    "apply_representation_pipeline",
    "apply_whitening",
    "assert_fit_set_is_train_only",
    "before_after_summary",
    "build_audit_report",
    "build_manifest_row",
    "channel_histograms",
    "extract_grid_patches",
    "fit_representation_pipeline",
    "fit_whitening",
    "get_dataset",
    "global_normalize",
    "inspect_image",
    "invert_whitening",
    "linear_to_srgb",
    "load_fitted_pipeline",
    "load_whitening",
    "measured_snr_db",
    "opponent_matrix",
    "opponent_to_rgb",
    "patch_power",
    "per_patch_normalize",
    "read_manifest",
    "reassemble_patches",
    "registered_datasets",
    "remove_dc",
    "resolve_under_root",
    "restore_dc",
    "rgb_to_opponent",
    "sample_random_patches",
    "save_fitted_pipeline",
    "save_whitening",
    "sigma_for_snr",
    "srgb_to_linear",
    "synthetic_dataset",
    "synthetic_image",
    "train_test_similarity",
    "validate_manifest",
    "verify_manifest_on_disk",
    "whitening_covariance_check",
    "write_audit_report",
    "write_manifest",
]
