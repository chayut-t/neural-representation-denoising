# 2016 dissertation archive

This directory preserves the source and rendered artifacts for Chayut Thanapirom's dissertation, *Neural Representation Learning with Denoising Autoencoder Framework* (Ph.D. in Physics, University of California, Berkeley, 2016).

- Permanent record: https://escholarship.org/uc/item/0hm6p6s5
- Local historical build: `source/thesis.pdf` (64 pages)
- eScholarship submission: `source/qt0hm6p6s5.pdf` (65 pages)
- Historical source tree: `source/`

The eScholarship file contains one repository cover page followed by the same 64 dissertation pages as the local PDF. Normalized text extraction confirms that every dissertation page matches at an offset of one page.

## Provenance and reproducibility boundary

The archive contains the LaTeX source, bibliography, rendered figures, build script, and two PDF versions. It does **not** contain the original Theano, Python, MATLAB, or experiment-generation source code, raw datasets, trained model parameters, random seeds, or machine-readable result files. Searches of the current worktree, all Git refs, Git history, and ignored project files found only the LaTeX build script. Consequently, the figures, tables, and numerical claims are historical evidence and cannot currently be regenerated from this repository.

The local PDF reports `pdfTeX-1.40.16` as its producer and was created on 2016-05-10. The source build script invokes `pdflatex`, `biber`, and `pdflatex`. The dissertation text identifies Theano as the automatic-differentiation framework used in the group sparse-coding experiment. Exact package versions beyond those visible in the artifacts are not recoverable here.

## Known issue

Appendix B contains an unresolved cross-reference rendered as `chapter ??`. The source uses `\ref{chap:grid}`, but Chapter 3 is labeled `chap:grid_cells`. This is retained unchanged in the historical archive and corrected only in revised editions.

## Integrity

`MANIFEST.sha256` records SHA-256 hashes for the historical source, figures, bibliography, scripts, and PDFs. Do not edit files under `source/` when preparing a revised dissertation; copy required assets into a separate revision tree.
