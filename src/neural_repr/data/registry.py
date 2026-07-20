"""Dataset registry: official sources, terms, splits, and archive-hash policy.

Phase 4 (plan §7 tasks 1-3; decision 0001). A :class:`DatasetSpec` records the
public, citable facts about each dataset — official URLs, citation, terms URL and a
required-acceptance token, expected archive filenames, and the fixed ID splits — but
**not** any fabricated content hashes. Third-party archives are multi-GB and must be
downloaded from their official source after explicit local terms acceptance; their
SHA-256 is recorded on first authenticated download (a local, git-ignored record)
and verified byte-for-byte thereafter (plan §3.2: "fail with an explanatory message
rather than silently accepting new bytes").

Nothing here downloads or commits data. Personal terms-acceptance and archive-hash
records live under ``data/raw``/``data/interim`` (git-ignored); only per-file split
manifests and hashes are committed (plan tasks 3, 5).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SplitSpec:
    """A named ID range for a dataset split (inclusive integer IDs)."""

    name: str
    first_id: int
    last_id: int
    role: str  # "train" | "val" | "test"

    def ids(self) -> list[int]:
        return list(range(self.first_id, self.last_id + 1))

    def __post_init__(self) -> None:
        if self.last_id < self.first_id:
            raise ValueError(f"split {self.name}: last_id < first_id")
        if self.role not in {"train", "val", "test"}:
            raise ValueError(f"split {self.name}: role must be train|val|test, got {self.role!r}")


@dataclass(frozen=True)
class DatasetSpec:
    """Public, citable metadata for a dataset (no fabricated content hashes)."""

    name: str
    description: str
    official_url: str
    citation: str
    terms_url: str
    terms_acceptance_token: str  # the exact --accept-... flag a user must pass
    archives: tuple[str, ...]  # expected official archive filenames
    splits: tuple[SplitSpec, ...] = field(default_factory=tuple)
    id_format: str = "{:04d}"  # how an integer ID maps to a file stem
    file_extension: str = ".png"

    def split_by_role(self, role: str) -> list[SplitSpec]:
        return [s for s in self.splits if s.role == role]

    def train_ids(self) -> list[int]:
        ids: list[int] = []
        for s in self.split_by_role("train"):
            ids.extend(s.ids())
        return ids

    def all_ids(self) -> list[int]:
        ids: list[int] = []
        for s in self.splits:
            ids.extend(s.ids())
        return ids


# DIV2K: 800 train + 100 val HR images, academic-research terms (decision 0001).
# Fixed split: 0001-0720 train, 0721-0800 val, 0801-0900 locked in-domain test.
DIV2K = DatasetSpec(
    name="div2k",
    description="DIV2K high-resolution RGB images (ETH Zurich), academic research only.",
    official_url="https://data.vision.ee.ethz.ch/cvl/DIV2K/",
    citation="Agustsson & Timofte, NTIRE 2017 (DIV2K).",
    terms_url="https://data.vision.ee.ethz.ch/cvl/DIV2K/",
    terms_acceptance_token="--accept-academic-research-terms",
    archives=("DIV2K_train_HR.zip", "DIV2K_valid_HR.zip"),
    splits=(
        SplitSpec("div2k-train", 1, 720, "train"),
        SplitSpec("div2k-val", 721, 800, "val"),
        SplitSpec("div2k-test", 801, 900, "test"),
    ),
    id_format="{:04d}",
    file_extension=".png",
)

# BSDS500: official fixed TEST split only, non-commercial research/education terms.
# IDs are non-contiguous image numbers, so its manifest enumerates files directly
# rather than an integer range; the split is recorded by role only.
BSDS500 = DatasetSpec(
    name="bsds500",
    description=(
        "BSDS500 color images (Berkeley), non-commercial research/education; test split only."
    ),
    official_url="https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html",
    citation="Arbelaez, Maire, Fowlkes & Malik, PAMI 2011 (BSDS500).",
    terms_url="https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html",
    terms_acceptance_token="--accept-noncommercial-research-terms",
    archives=("BSR_bsds500.tgz",),
    splits=(SplitSpec("bsds500-test", 0, 0, "test"),),  # enumerated from the archive, not a range
    id_format="{}",
    file_extension=".jpg",
)

_REGISTRY: dict[str, DatasetSpec] = {DIV2K.name: DIV2K, BSDS500.name: BSDS500}


def get_dataset(name: str) -> DatasetSpec:
    """Look up a registered dataset spec by name; raise for an unknown dataset."""
    key = name.lower()
    if key not in _REGISTRY:
        raise KeyError(f"unknown dataset {name!r}; registered: {sorted(_REGISTRY)}")
    return _REGISTRY[key]


def registered_datasets() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))
