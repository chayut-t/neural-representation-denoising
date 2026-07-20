# Data

This repository **never commits, mirrors, or redistributes third-party dataset
bytes** (plan §2, §3.2; decision 0001). Only *manifests* (per-file split lists with
SHA-256 hashes) and *fitted preprocessing statistics* are committed. You download the
datasets yourself, from their official sources, after accepting their terms locally.

## What is and isn't committed

| Committed | Not committed (git-ignored) |
|---|---|
| Split manifests (`data/manifests/*.csv`) | Dataset image bytes (`data/raw/`, `data/interim/`, `data/processed/`) |
| Fitted preprocessing stats (small JSON artifacts) | Personal terms-acceptance records (`data/raw/records/*.terms-acceptance.json`) |
| The synthetic fixture manifest | Recorded archive hashes (`data/raw/records/*.archive-hashes.json`) |
| Dataset/preprocessing configs (`configs/`) | Downloaded official archives |

## Datasets (decision 0001)

- **DIV2K** (primary, ETH Zurich) — high-resolution RGB, **academic research only**.
  Fixed split: `0001–0720` train, `0721–0800` validation, `0801–0900` locked in-domain test.
- **BSDS500** (external generalization, Berkeley) — color, **non-commercial research/education**,
  official **test** split only.
- **synthetic** — a tiny repository-owned RGB fixture (edges, gratings, smooth/correlated color),
  licensed with the code, for **tests only** — never scientific evidence.

`neural-repr-data list` prints each dataset's official source, terms URL, and the exact
acceptance flag.

## License handling

Third-party datasets are governed **only by their original terms**, never by this
repository's code/text license (plan §3.5). Accepting those terms is your
responsibility and is recorded **locally, not in git**. If DIV2K/BSDS terms turn out
incompatible with the intended use, decision 0001 requires replacing the primary set
with a permissively licensed RGB collection and rerunning affected pilots — the code
stays open even when data download is restricted.

## Operator flow (accept → download → pin → verify)

```sh
# 1. Accept terms locally (writes a git-ignored record; timestamp is yours).
uv run neural-repr-data accept-terms div2k --accept --at "$(date -u +%FT%TZ)"

# 2. Download the official archives yourself from the official_url shown by:
uv run neural-repr-data list
#    (Downloads are not automated: they require authenticated access to the
#     official source and must not be mirrored or committed.)

# 3. On a TRUSTED first download, record the archive hash (git-ignored):
uv run neural-repr-data verify-archive div2k DIV2K_train_HR.zip <path.zip> --record

# 4. Thereafter, verify bytes against the recorded hash before use; a mismatch
#    fails loudly rather than silently accepting changed bytes (plan §3.2):
uv run neural-repr-data verify-archive div2k DIV2K_train_HR.zip <path.zip>
```

## Synthetic fixture (no download)

```sh
uv run neural-repr-data make-fixture            # writes data/raw/synthetic/*.png + manifest
uv run neural-repr-data check synthetic         # validates counts/shapes/modes/dupes/leakage
uv run neural-repr-data audit                    # writes a JSON data-audit report (task 11)
```

The fixture is deterministic from `(seed, index, size)`, so its **pixels** are
identical on any platform. That pixel identity — not PNG-encoded-byte identity — is
the reproducibility contract: PNG (zlib) encoding is not byte-identical across
platforms/library builds even for identical pixels. Accordingly each manifest row
carries two digests:

- `content_sha256` — a digest over the **decoded pixels** (dtype, shape, raw bytes);
  platform-stable, and the invariant used for reproducibility, split-leakage, and
  duplicate detection. Always present.
- `sha256` — the **file-byte** digest. Only meaningful for files whose exact bytes we
  received and keep (downloaded third-party archives/images), so it is **empty** for
  the regenerable synthetic fixture and populated for real datasets. When present it is
  verified on `check --root`; when empty it is skipped.

Third-party archive and per-file byte hashes remain full byte-integrity checks — this
split only relaxes the *synthetic-fixture* contract, never the downloaded-data ones.

## Two color tracks (decision 0003)

Preprocessing configs live in `configs/preprocessing/`:

- `color_whitened` — opponent-color, DC-removed, **train-only** ZCA whitening; for
  representation/basis/group analysis and thesis-comparable SNR (0/3/6 dB).
- `color_rgb` — normalized RGB, no whitening; AWGN at fixed σ = 15/25/50 over 255; for
  full-image PSNR/SSIM after overlapping-patch reassembly.

Metrics from the two tracks are never mixed in one table without an explicit label.
