# 0003 — Color representation and noise domain

**Status:** Accepted (2026-07-18) · **Relates to:** plan §3.3; claims C3-H2, C3-H3.

## Context

One preprocessing pipeline cannot serve both representation analysis (whitened, opponent-color,
thesis-comparable SNR) and standard visible-image RGB denoising (PSNR/SSIM). Mixing them produced
the calibration confusion the 2026 baseline flags (§3.7.1).

## Decision

Two explicitly named, never-mixed tracks:

1. **`color_whitened` (representation track):** decode RGB → invertible orthonormal opponent-color
   basis; remove patch DC; fit spatial/color whitening on **training images only**; learn sparse
   dictionaries + group structure in whitened space. Used for basis structure, group organization,
   topography, and thesis-comparable SNR (input SNR 0/3/6 dB).
2. **`color_rgb` (denoising track):** normalized RGB patches, **no** spatial whitening; add AWGN at
   fixed **σ = 15/255, 25/255, 50/255**, after documenting whether values are gamma-encoded sRGB or
   linear light; reassemble full images via overlapping patches + fixed blending window; report RGB
   PSNR and SSIM on both unclipped and clipped outputs, clearly distinguished.

**Rule:** metrics from the two tracks never appear in one table without an explicit block/label.
Color-error metrics (per-channel RMSE, ΔE) only if the exact color-space assumptions are documented
and validated.

## Alternatives considered

- Single unified pipeline — rejected: conflates whitened representation SNR with visible-image
  PSNR and misleads comparison (baseline §3.7.1).
- Stay grayscale — rejected per [0001](0001-datasets-and-licenses.md).

## Consequences

Confirmatory sparse-coding runs (C3-H2, C3-H3) split into two evaluation suites; the
`docs/claim-evidence-matrix.md` notes which track feeds which claim. Patch size starts at
`16×16×3` for comparability, with a larger preregistered size as an ablation.
