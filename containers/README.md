# Containers

Two **public reference containers** (plan §2.4, decision 0006). Both build from publicly
pullable, digest-pinned base images with no private credentials, so any third party can
reconstruct the environment. These define the reproducibility target; they are deliberately
independent of any private execution image the authors may use.

| File | Base (public, digest-pinned) | Purpose |
|---|---|---|
| `Dockerfile.cpu` | `python:3.11-slim-bookworm` | quick profile, tests, preprocessing, figures |
| `Dockerfile.cuda` | `nvidia/cuda:13.0.1-cudnn-runtime-ubuntu24.04` | full GPU profile on generic CUDA hardware (≥16 GB VRAM) |

## Pinning scope (what is and isn't guaranteed)

Pinned by digest end to end for reproducibility:

- the base images (`FROM ...@sha256:`);
- the `uv` builder image (`ghcr.io/astral-sh/uv@sha256:...`, not the mutable `:0.11.29` tag);
- all Python dependencies, via the committed `uv.lock` (`uv sync --locked`), including the
  torch/CUDA stack.

**Not** bit-for-bit pinned: the base images' own APT package set is whatever the pinned base
layer already contains, plus a small fixed list we `apt-get install` **without version pins**
(`git`, `build-essential`, `ca-certificates`). We accept this: those packages do not affect
numerical results, and pinning them to a full APT snapshot mirror is disproportionate at the
current reproduction level. If byte-level base reconstruction is later required, switch to a
dated Debian/Ubuntu snapshot source and pin versions; recorded here as a deliberate decision.

## Reference GPU/CUDA stack

The committed `uv.lock` resolves **torch 2.13.0 (from PyPI)** bundling **CUDA 13.0** runtime
wheels (`nvidia-cuda-runtime` 13.0.x, `nvidia-cudnn-cu13`). `Dockerfile.cuda` therefore uses a
**CUDA 13.0** base so the base runtime and the torch wheels agree. Python is **3.11**, provided
by `uv` (managed) rather than the base image's default, so the interpreter is pinned by the
lockfile. (Earlier drafts referred to an "official PyTorch CUDA index" and a CUDA 12.4 base;
both were incorrect — torch here comes from PyPI, and the base is now CUDA 13.0 to match.)

**Host requirement (generic):** a CUDA GPU with ≥16 GB VRAM and an NVIDIA driver that supports
**CUDA 13** (driver **>= 580.x**). Stated generically on purpose — not a specific machine.

## Build

```sh
# from the repository root
docker build -f containers/Dockerfile.cpu  -t neural-repr:cpu  .
docker build -f containers/Dockerfile.cuda -t neural-repr:cuda .   # linux/amd64
```

The CUDA image targets linux/amd64; on Apple Silicon build with `--platform linux/amd64`
(emulated) or on an amd64 host/CI runner.

## Self-checks

```sh
# CPU: import + provenance-schema + numerical-smoke + canary leak-scan
docker run --rm neural-repr:cpu neural-repr-verify check-image

# CUDA build-time (no GPU needed): assert the declared torch/CUDA stack
docker run --rm neural-repr:cuda neural-repr-verify check-image --expect-torch 2.13 --expect-cuda 13.0

# CUDA runtime (requires a GPU host): fails if CUDA is unavailable; runs a real GPU op
docker run --rm --gpus all neural-repr:cuda neural-repr-verify check-image --require-cuda --expect-torch 2.13 --expect-cuda 13.0
```

The CUDA **runtime** self-check needs a GPU-capable runner; until one is wired up it is
**pending**, not passed (see Gate P2 and `.github/workflows/containers.yml`).

## Refreshing pinned digests

Bases (and the `uv` builder) are pinned by digest. Resolve digests without a running daemon via
the registry API:

```sh
# CPU base
docker pull python:3.11-slim-bookworm
docker inspect --format='{{index .RepoDigests 0}}' python:3.11-slim-bookworm

# CUDA base (manifest-list digest)
TOKEN=$(curl -s "https://auth.docker.io/token?service=registry.docker.io&scope=repository:nvidia/cuda:pull" | python3 -c "import sys,json;print(json.load(sys.stdin)['token'])")
curl -sI -H "Authorization: Bearer $TOKEN" -H "Accept: application/vnd.oci.image.index.v1+json" \
  https://registry-1.docker.io/v2/nvidia/cuda/manifests/13.0.1-cudnn-runtime-ubuntu24.04 | grep -i docker-content-digest

# uv builder image
TOKEN=$(curl -s "https://ghcr.io/token?scope=repository:astral-sh/uv:pull" | python3 -c "import sys,json;print(json.load(sys.stdin)['token'])")
curl -sI -H "Authorization: Bearer $TOKEN" -H "Accept: application/vnd.oci.image.index.v1+json" \
  https://ghcr.io/v2/astral-sh/uv/manifests/0.11.29 | grep -i docker-content-digest
```

A scheduled CI job (`base-pullable`) verifies the pinned bases **and** the `uv` builder remain
pullable; the release process archives or mirrors the built OCI images where licensing permits.
