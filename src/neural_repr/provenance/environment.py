"""Collect a sanitized execution-environment record (plan §2.4 layer 1, §6.1).

Design rule: this module records *scientific* provenance only, with no
infrastructure-identifying values. It deliberately does NOT collect hostname,
usernames, absolute paths, registry URIs, cluster/queue/pod names, or account
IDs. The opaque ``execution_environment_fingerprint`` is a hash over the
sanitized fields; the mapping from fingerprint to raw operational detail is a
non-public layer maintained outside this repository.

Scope: this record holds only *environment-level* fields — those that exist at
environment-inspection time. Run-level §6.1 fields (experiment name, Hydra
config, git commit, dataset/preprocessing hashes, seeds, timestamps, training
history, artifact hashes) require a run to exist and are recorded by the run
pipeline in later phases, not here.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import platform
from dataclasses import asdict, dataclass, field
from importlib import metadata
from pathlib import Path

SCHEMA_VERSION = "2"

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

# reference_compatibility_status values (plan §6.1).
COMPAT_STATUSES = ("public-reference-run", "equivalence-verified", "pending-equivalence", "unknown")

# determinism_mode values (plan §6.3), plus "unset" for a bare snapshot.
DETERMINISM_MODES = ("debug", "research", "benchmark", "unset")


def _tracked_versions() -> dict[str, str]:
    out: dict[str, str] = {}
    for name in _TRACKED_PACKAGES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return dict(sorted(out.items()))


def _find_uv_lock() -> Path | None:
    """Locate the committed uv.lock by walking up from this file, then CWD."""
    for start in (Path(__file__).resolve(), Path.cwd().resolve()):
        for parent in [start, *start.parents]:
            candidate = parent / "uv.lock"
            if candidate.is_file():
                return candidate
    return None


def _uv_lock_sha256() -> str | None:
    lock = _find_uv_lock()
    if lock is None:
        return None
    h = hashlib.sha256()
    h.update(lock.read_bytes())
    return "sha256:" + h.hexdigest()


def _total_ram_gib() -> float | None:
    """Total physical RAM in GiB via POSIX sysconf; None where unavailable."""
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return round(pages * page_size / (1024**3), 2)
    except (ValueError, OSError, AttributeError):
        return None


def _determinism_flags() -> dict[str, object]:
    """Vendor-neutral determinism-relevant runtime flags. {} if torch is absent."""
    try:
        import torch
    except Exception:
        return {}
    flags: dict[str, object] = {}
    with contextlib.suppress(Exception):
        flags["deterministic_algorithms"] = bool(torch.are_deterministic_algorithms_enabled())
    with contextlib.suppress(Exception):
        flags["cudnn_deterministic"] = bool(torch.backends.cudnn.deterministic)
        flags["cudnn_benchmark"] = bool(torch.backends.cudnn.benchmark)
    return flags


def _runtime_details() -> dict[str, object]:
    """Numerics-affecting compiler/runtime details (vendor-neutral)."""
    info: dict[str, object] = {
        "cpu_count": os.cpu_count(),
        "total_ram_gib": _total_ram_gib(),
    }
    try:
        import torch

        # A stable hash of the torch build string (contains compiler/BLAS/OpenMP
        # config) rather than the raw multi-line string, which can vary in
        # whitespace and is noisy in a committed record.
        build = torch.__config__.show()
        info["torch_build_config_sha256"] = "sha256:" + hashlib.sha256(build.encode()).hexdigest()
        info["torch_parallel_info_present"] = True
    except Exception:
        info["torch_parallel_info_present"] = False
    return info


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
        "cudnn_version": None,
        "driver_version": None,
        "gpu_model_class": None,
        "gpu_compute_capability": None,
        "gpu_total_memory_gib": None,
    }
    try:
        cudnn_version: object = torch.backends.cudnn.version()  # type: ignore[no-untyped-call]
        info["cudnn_version"] = cudnn_version
    except Exception:
        pass
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
        try:
            # NVIDIA driver version (a number like "580.65.06"), vendor-neutral.
            drv = torch._C._cuda_getDriverVersion()  # type: ignore[attr-defined]
            info["driver_version"] = drv
        except Exception:
            pass
    return info


@dataclass(frozen=True)
class ExecutionEnvironment:
    """Sanitized, committable execution-environment record (no infra identifiers)."""

    schema_version: str = SCHEMA_VERSION
    python_version: str = field(default_factory=platform.python_version)
    python_implementation: str = field(default_factory=platform.python_implementation)
    os_system: str = field(default_factory=platform.system)
    os_release: str = field(default_factory=platform.release)
    machine: str = field(default_factory=platform.machine)
    processor: str = field(default_factory=lambda: platform.processor() or platform.machine())
    package_versions: dict[str, str] = field(default_factory=_tracked_versions)
    torch: dict[str, object] = field(default_factory=_torch_environment)
    runtime: dict[str, object] = field(default_factory=_runtime_details)
    determinism_flags: dict[str, object] = field(default_factory=_determinism_flags)
    uv_lock_sha256: str | None = field(default_factory=_uv_lock_sha256)
    determinism_mode: str = "unset"
    install_mode: str = "unknown"
    reference_compatibility_status: str = "unknown"
    public_reference_environment_digest: str | None = None

    def fingerprint_payload(self) -> dict[str, object]:
        """The subset hashed into the fingerprint.

        Excludes the fingerprint itself and the fields that describe *which*
        reference/run this is (digest, compatibility status), so the fingerprint
        identifies the environment content, not its labeling.
        """
        d = asdict(self)
        d.pop("public_reference_environment_digest", None)
        d.pop("reference_compatibility_status", None)
        return d

    @property
    def execution_environment_fingerprint(self) -> str:
        return environment_fingerprint(self.fingerprint_payload())

    def to_record(self) -> dict[str, object]:
        """Full committable record: sanitized fields + fingerprint."""
        record = asdict(self)
        record["execution_environment_fingerprint"] = self.execution_environment_fingerprint
        return record


def environment_fingerprint(payload: dict[str, object]) -> str:
    """Stable opaque fingerprint: SHA-256 over canonical JSON of sanitized fields."""
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def collect_execution_environment(
    *,
    determinism_mode: str = "unset",
    install_mode: str = "unknown",
    reference_compatibility_status: str = "unknown",
    public_reference_environment_digest: str | None = None,
) -> ExecutionEnvironment:
    """Collect the current environment as a sanitized record.

    ``install_mode`` / ``reference_compatibility_status`` / ``determinism_mode``
    are *categories*, not identifiers; each must be a known value.
    ``public_reference_environment_digest`` is the digest of the publicly
    pullable reference container when running under it.
    """
    if install_mode not in INSTALL_MODES:
        raise ValueError(f"install_mode must be one of {INSTALL_MODES}, got {install_mode!r}")
    if determinism_mode not in DETERMINISM_MODES:
        raise ValueError(
            f"determinism_mode must be one of {DETERMINISM_MODES}, got {determinism_mode!r}"
        )
    if reference_compatibility_status not in COMPAT_STATUSES:
        raise ValueError(
            f"reference_compatibility_status must be one of {COMPAT_STATUSES}, "
            f"got {reference_compatibility_status!r}"
        )
    return ExecutionEnvironment(
        determinism_mode=determinism_mode,
        install_mode=install_mode,
        reference_compatibility_status=reference_compatibility_status,
        public_reference_environment_digest=public_reference_environment_digest,
    )


if __name__ == "__main__":  # pragma: no cover - manual smoke
    print(json.dumps(collect_execution_environment().to_record(), indent=2, default=str))
