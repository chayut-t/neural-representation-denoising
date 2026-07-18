# Containers

Two **public reference containers** (plan §2.4, decision 0006). Both build from publicly
pullable, digest-pinned base images with no private credentials, so any third party can
reconstruct the environment. These define the reproducibility target; they are deliberately
independent of any private execution image the authors may use.

| File | Base (public, digest-pinned) | Purpose |
|---|---|---|
| `Dockerfile.cpu` | `python:3.11-slim-bookworm` | quick profile, tests, preprocessing, figures |
| `Dockerfile.cuda` | `nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04` (linux/amd64) | full GPU profile on generic CUDA hardware (≥16 GB VRAM) |

Both install the locked environment from the committed `uv.lock` via `uv sync --locked`, so the
image contents are pinned end to end.

## Build

```sh
# from the repository root
docker build -f containers/Dockerfile.cpu  -t neural-repr:cpu  .
docker build -f containers/Dockerfile.cuda -t neural-repr:cuda .   # linux/amd64
```

The CUDA image targets linux/amd64; on Apple Silicon build with `--platform linux/amd64`
(emulated) or on an amd64 host/CI runner.

## Refreshing pinned digests

Bases are pinned by digest, not mutable tag. To update:

```sh
# CPU base
docker pull python:3.11-slim-bookworm
docker inspect --format='{{index .RepoDigests 0}}' python:3.11-slim-bookworm

# CUDA base (take the linux/amd64 Descriptor digest)
docker manifest inspect -v nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
```

A scheduled CI job verifies the pinned bases remain pullable; the release process archives or
mirrors the built OCI images where licensing permits (Gate P2).

## Reference GPU/CUDA stack (generic)

CUDA 12.4 runtime + cuDNN on Ubuntu 22.04, Python 3.11, PyTorch installed from the official
CUDA wheels pinned in `uv.lock`. Described generically on purpose — the public reference is a
CUDA GPU with ≥16 GB VRAM, not a specific machine.
