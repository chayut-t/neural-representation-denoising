<!-- Keep changes scoped to one phase's work where possible. -->

## What this changes

<!-- Brief summary and the plan phase/task or claim ID this advances. -->

## Checklist

- [ ] `uv run ruff check` / `ruff format --check` / `mypy` / `pytest` pass locally
- [ ] `uv lock --check` passes (lockfile consistent; not hand-edited)
- [ ] No files under `legacy/` were modified (immutable baselines)
- [ ] No dataset files or infrastructure identifiers committed (leak scan passes)
- [ ] Decisions that alter data/equations/baselines/engine/claims have a `docs/decisions/` record
- [ ] Result/claim changes are reflected in the relevant inventory / claim-evidence matrix
