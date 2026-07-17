# 2026 revised dissertation edition

This directory contains a self-contained dissertation on denoising objectives for sparse representation learning and recurrent spatial codes.

All figures, tables, bibliography entries, and LaTeX inputs required to build the dissertation are contained under `source/`. The experiment code, model checkpoints, raw per-trial outputs, and random seeds are unavailable, so numerical claims are presented as descriptive results without claims of independent replication or statistical significance.

## Structure

- `WRITING_PLAN.md`: chapter-by-chapter writing plan.
- `LITERATURE_MATRIX.csv`: selected related work used in the dissertation.
- `source/`: rewritten LaTeX sources, bibliography, copied rendered assets, and build files.

## Build

From the `source/` directory, run:

```sh
./doit.sh
```

The build uses Tectonic and writes `thesis.pdf` beside `thesis.tex`.
