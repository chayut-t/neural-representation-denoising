# AGENTS.md

Guidance for Codex and other coding agents working in this repository.

## Agent-to-agent communication

`workspace.local/` is the local workspace directory for this project. Use it for temporary
working material, notes, and coordination artifacts that should remain local. The directory is
git-ignored; do not commit its contents or treat anything in it as a reproducible project output.

Use `workspace.local/a2a/` specifically for local Markdown messages between agents.

- Codex messages must be named
  `workspace.local/a2a/codex-<date>-<descriptive-name>.md`.
- Claude Code messages must be named
  `workspace.local/a2a/claude-<date>-<descriptive-name>.md`.

Use ISO dates (`YYYY-MM-DD`) and a concise descriptive filename component.

## Session handoff files

Two more `workspace.local/` subdirectories hold the session-handoff files (written by Claude
Code's `/create-context-and-worklog` skill; also git-ignored):

- `workspace.local/worklogs/worklog-<YYYYMMDD>.md` — backward-looking record of what a session did.
- `workspace.local/contexts/claude-context-<YYYYMMDD>.md` — forward-looking resume brief for the
  next session.

When picking up work, read the most recent context file first, and check `workspace.local/a2a/`
for pending messages.
