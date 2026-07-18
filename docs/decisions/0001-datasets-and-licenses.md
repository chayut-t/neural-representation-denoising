# 0001 — Primary/secondary datasets and license implications

**Status:** Accepted (2026-07-18) · **Relates to:** plan §3.2, §3.5; claims C3-H2, C3-H3.

## Context

The 2016/2026 group-sparse study used grayscale van Hateren natural images (35 train / 35 test),
whose modern licensing and availability are less clean than current public sets. The plan
modernizes to public *color* images and requires that dataset files are never committed —
only manifests/hashes.

## Decision

- **Primary dataset:** DIV2K high-resolution RGB (ETH Zurich), academic-research terms. Use only
  the 800 public training HR images and 100 public validation HR images. Fixed split:
  `0001–0720` train, `0721–0800` hyperparameter validation, `0801–0900` locked in-domain test.
- **External generalization set:** BSDS500 color, official fixed **test** split only, under its
  non-commercial research/education terms.
- **CI fixture:** a tiny repo-owned synthetic RGB dataset (geometric edges, oriented gratings,
  smooth color fields, correlated color features), licensed with the code, for tests only —
  never scientific evidence.
- **Handling:** download only from official sources after explicit local terms acceptance; pin
  archive hashes then generate per-file hashes; commit only manifests/hashes; fail loudly if the
  official archive bytes change. **No data staged in any external bucket / cloud store.**
- **License-incompatibility gate:** if DIV2K/BSDS terms turn out incompatible with intended
  repository use, replace the primary set with a permissively licensed RGB collection and rerun
  affected pilots. Code stays open even if data download is restricted.

## Alternatives considered

- Keep van Hateren grayscale — rejected: not color, weaker modern licensing, and the plan's
  scientific upgrade is explicitly color.
- Commit a data subset for convenience — rejected: violates the no-redistribution rule (plan §2).

## Consequences

Two never-mixed color tracks follow in [0003](0003-color-representation-and-noise-domain.md).
Reproduction requires the user to accept dataset terms locally; documented in the future
`data/README.md` and `make data-check`.
