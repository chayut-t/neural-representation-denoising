"""Collect a sanitized execution-environment record (plan §2.4 layer 1, §6.1).

Design rule: this module records *scientific* provenance only, with no
infrastructure-identifying values. It deliberately does NOT collect hostname,
usernames, absolute paths, registry URIs, cluster/queue/pod names, or account
IDs. The opaque ``execution_environment_fingerprint`` is a hash over the
sanitized fields; the mapping from fingerprint to raw operational detail is a
non-public layer maintained outside this repository.
"""

from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import asdict, dataclass, field
from importlib import metadata

# Packages whose exact versions materially affect numerics / results. Kept small
# and explicit so the record is stable and reviewable; extended as deps land.
_TRACKED_PACKAGES: tuple[str, ...] = (
    "numpy",
    "scipy",
    "torch",
    "torchvision",
    "scikit-image",
    "scikit-learn",
    "pandas",
    "hydra-core",
    "omegaconf",
    "matplotlib",
    "pillow",
)

# Install-mode categories (NOT identifiers): how the environment was materialized.
INSTALL_MODES = ("public-reference-container", "private-mirror", "local", "unknown")


def _tracked_versions() -> dict[str, str]:
    out: dict[str, str] = {}
    for name in _TRACKED_PACKAGES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return dict(sorted(out.items()))


def _torch_environment() -> dict[str, object]:
    """Vendor-neutral PyTorch/GPU facts. Returns {} if torch is absent."""
    try:
        import torch
    except Exception:  # torch not installed yet in a bare env
        return {}

    info: dict[str, object] = {
        "torch_version": torch.__version__,
        "torch_cuda_version": getattr(torch.version, "cuda", None),
        "cuda_available": bool(torch.cuda.is_available()),
    }
    try:
        cudnn_version: object = torch.backends.cudnn.version()  # type: ignore[no-untyped-call]
        info["cudnn_version"] = cudnn_version
    except Exception:
        info["cudnn_version"] = None
    if torch.cuda.is_available():
        try:
            # GPU *model class* only (e.g. "NVIDIA L4"), never a host/uuid.
            info["gpu_model_class"] = torch.cuda.get_device_name(0)
            major, minor = torch.cuda.get_device_capability(0)
            info["gpu_compute_capability"] = f"{major}.{minor}"
            props = torch.cuda.get_device_properties(0)
            info["gpu_total_memory_gib"] = round(props.total_memory / (1024**3), 2)
        except Exception:
            pass
    return info


@dataclass(frozen=True)
class ExecutionEnvironment:
    """Sanitized, committable execution-environment record (no infra identifiers)."""

    schema_version: str = "1"
    python_version: str = field(default_factory=platform.python_version)
    python_implementation: str = field(default_factory=platform.python_implementation)
    os_system: str = field(default_factory=platform.system)
    os_release: str = field(default_factory=platform.release)
    machine: str = field(default_factory=platform.machine)
    processor: str = field(default_factory=lambda: platform.processor() or platform.machine())
    package_versions: dict[str, str] = field(default_factory=_tracked_versions)
    torch: dict[str, object] = field(default_factory=_torch_environment)
    determinism_mode: str = "unset"
    install_mode: str = "unknown"
    public_reference_environment_digest: str | None = None

    def fingerprint_payload(self) -> dict[str, object]:
        """The subset hashed into the fingerprint (excludes the fingerprint itself)."""
        d = asdict(self)
        d.pop("public_reference_environment_digest", None)
        return d

    @property
    def execution_environment_fingerprint(self) -> str:
        return environment_fingerprint(self.fingerprint_payload())

    def to_record(self) -> dict[str, object]:
        """Full committable record: sanitized fields + the three §6.1 fields."""
        record = asdict(self)
        record["execution_environment_fingerprint"] = self.execution_environment_fingerprint
        # reference_compatibility_status is set by the run pipeline once known;
        # a bare system-info snapshot cannot assert equivalence, so mark it pending.
        record.setdefault("reference_compatibility_status", "unknown")
        return record


def environment_fingerprint(payload: dict[str, object]) -> str:
    """Stable opaque fingerprint: SHA-256 over canonical JSON of sanitized fields."""
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def collect_execution_environment(
    *,
    determinism_mode: str = "unset",
    install_mode: str = "unknown",
    public_reference_environment_digest: str | None = None,
) -> ExecutionEnvironment:
    """Collect the current environment as a sanitized record.

    ``install_mode`` must be one of :data:`INSTALL_MODES`; it is a *category*,
    not an identifier. ``public_reference_environment_digest`` is the digest of
    the publicly pullable reference container when running under it.
    """
    if install_mode not in INSTALL_MODES:
        raise ValueError(f"install_mode must be one of {INSTALL_MODES}, got {install_mode!r}")
    return ExecutionEnvironment(
        determinism_mode=determinism_mode,
        install_mode=install_mode,
        public_reference_environment_digest=public_reference_environment_digest,
    )


if __name__ == "__main__":  # pragma: no cover - manual smoke
    print(json.dumps(collect_execution_environment().to_record(), indent=2, default=str))
