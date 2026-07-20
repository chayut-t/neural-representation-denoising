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
import re
from dataclasses import asdict, dataclass, field
from importlib import metadata
from pathlib import Path

SCHEMA_VERSION = "3"

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


# Vendor-neutral compiler/runtime facts a third party needs as scientific
# provenance, parsed out of torch's build config. Values are short tokens (names,
# versions, thread counts) — never paths or host identifiers.
_BUILD_FACT_PATTERNS: dict[str, re.Pattern[str]] = {
    "cxx_compiler": re.compile(r"C\+\+ Version:\s*(\d[\w.]*)"),
    "openmp": re.compile(r"(?:USE_OPENMP=ON|OpenMP\b)"),
    "blas": re.compile(r"(?:BLAS_INFO|BLAS)[=:]\s*(\w+)"),
}


def _blas_impl(build: str) -> str | None:
    m = _BUILD_FACT_PATTERNS["blas"].search(build)
    return m.group(1).lower() if m else None


def _cxx_version(build: str) -> str | None:
    m = _BUILD_FACT_PATTERNS["cxx_compiler"].search(build)
    return m.group(1) if m else None


def _runtime_details() -> dict[str, object]:
    """Numerics-affecting compiler/runtime facts (vendor-neutral, no paths/hosts).

    Records the actual facts a third party needs to reason about numeric
    reproducibility — BLAS implementation, C++ compiler version, OpenMP presence,
    torch thread counts — plus a stable hash of the full build string as a tamper
    check. The parsed facts are short tokens (names/versions/ints), never absolute
    paths or host identifiers.
    """
    info: dict[str, object] = {
        "cpu_count": os.cpu_count(),
        "total_ram_gib": _total_ram_gib(),
        "blas_impl": None,
        "cxx_version": None,
        "openmp_enabled": None,
        "torch_num_threads": None,
        "torch_num_interop_threads": None,
        "torch_build_config_sha256": None,
    }
    try:
        import torch

        build = torch.__config__.show()
        info["torch_build_config_sha256"] = "sha256:" + hashlib.sha256(build.encode()).hexdigest()
        info["blas_impl"] = _blas_impl(build)
        info["cxx_version"] = _cxx_version(build)
        info["openmp_enabled"] = bool(_BUILD_FACT_PATTERNS["openmp"].search(build))
        with contextlib.suppress(Exception):
            info["torch_num_threads"] = int(torch.get_num_threads())
        with contextlib.suppress(Exception):
            info["torch_num_interop_threads"] = int(torch.get_num_interop_threads())
    except Exception:
        pass
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
    """Stable opaque fingerprint: SHA-256 over canonical JSON of sanitized fields.

    Uses strict JSON (no ``default=`` coercion): a value that is not natively
    JSON-serializable raises rather than being silently stringified into the
    fingerprint, so an unsupported object cannot slip past the fingerprint boundary
    (round-2 review finding 6).
    """
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
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
