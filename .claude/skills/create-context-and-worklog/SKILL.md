---
name: create-context-and-worklog
description: >-
  Invoke ONLY via the explicit slash command /create-context-and-worklog. Writes a session
  worklog to git-ignored workspace.local/worklogs/ and a resume-context file to
  workspace.local/contexts/ for the neural-representation-denoising project (two dated Markdown
  files, worklog-<date>.md + claude-context-<date>.md). Do NOT trigger this skill from
  natural-language requests, keyword matches, or as a proactive offer — even phrases like
  "write a worklog", "save a handoff", or "checkpoint the session" should NOT auto-invoke it.
  It runs only when the user explicitly types the /create-context-and-worklog command.
---

# Create context + worklog

Capture the current session into two durable Markdown files under `workspace.local/` so a fresh
Claude Code session (or the user, later) can resume without re-deriving everything. This is
the project's session-handoff format; the reference templates are
`workspace.local/worklogs/worklog-20260719.md` and
`workspace.local/contexts/claude-context-20260719.md`.

Run this skill **only** when the user explicitly types the `/create-context-and-worklog` slash
command. Do not invoke it from natural-language requests, keyword matches, or as a proactive
offer — if the user merely says "write a worklog" or "checkpoint this" without the slash command,
do the task directly rather than through this skill. This keeps the handoff a deliberate,
user-initiated action.

## Why two files, and how they differ

They serve different readers, so keep them distinct — don't let one become a copy of the other:

- **`worklog-<date>.md`** — a *backward-looking* record of **what happened this session**: what
  was done, why, the commits, what was deferred. It's the audit trail. A reviewer reads it to
  understand the session's changes.
- **`claude-context-<date>.md`** — a *forward-looking* **resume brief for a fresh session**:
  where things stand right now, how the repo is wired, the non-obvious gotchas that would
  otherwise bite, key decisions, and exactly what to do next. A new Claude reads *only this
  file* and should be able to continue correctly.

## Before writing: gather real facts, don't guess

The value of these files is that they're accurate, so verify against the live repo rather than
recalling from the conversation. Run (with the repo's PATH prefix — TeX/brew tools aren't on
PATH by default):

```sh
export PATH="/Library/TeX/texbin:/opt/homebrew/bin:$PATH"
git log --oneline -8               # recent commits (grab the exact HEAD short hash)
git status --short                 # is the tree clean? what's uncommitted?
git branch --show-current          # branch (usually main)
uv run pytest -q 2>&1 | tail -1    # real current test count — never invent it
```

Pull the current phase/gate status from the single source of truth
(`MODERNIZATION_AND_REPRODUCIBILITY_PLAN.md` status header + `CLAUDE.md`), not from memory —
those get updated as gates pass and are authoritative. If a gate is *open* or a claim is
*deferred*, say so plainly; an honest "Gate P4 OPEN, blocked on X" is worth more than a rosy
summary.

Also scan **`workspace.local/`** — the project's git-ignored local workspace — for artifacts a
prior session left behind, since they're invisible to a fresh session that only reads the tracked
repo. Look everywhere *except* this skill's own output dirs (`workspace.local/contexts/` and
`workspace.local/worklogs/` are the handoffs themselves, not incoming items to re-surface):

```sh
find workspace.local -type f -not -path 'workspace.local/contexts/*' \
  -not -path 'workspace.local/worklogs/*' 2>/dev/null | head -50
```

The best-known such artifacts are agent-to-agent (a2a) messages under `workspace.local/a2a/` (per
`CLAUDE.md` / `AGENTS.md`: Markdown notes like `codex-<date>-*.md` / `claude-<date>-*.md`), but
treat anything else there as in scope too — review sweeps, analyses, generated notes. Surface the
relevant/unaddressed items in the context file's "NEXT" section (name the file and its gist) so
the resuming session knows they exist and what to do with them. Because the directory is
git-ignored, these files live nowhere else — if the handoff doesn't mention them, the next session
won't know they're there.

Use today's date (from the environment) for `<date>`, formatted `YYYYMMDD`, e.g.
`worklog-20260720.md`.

**The reference templates are for *shape*, not content.** The originals
(`workspace.local/worklogs/worklog-20260719.md`, `workspace.local/contexts/claude-context-20260719.md`)
contain concrete-but-stale values — a specific HEAD hash, "68 tests passing", specific gate
statuses — that were true only when they were written. Copy their *structure and tone*, never
their numbers: every hash, count, gate status, and next-step must come from the live commands
above. If you catch a stale-looking figure (e.g. a test count that doesn't match your `pytest`
run), that's the signal you copied instead of re-deriving.

## Where the files go

Write each file into its dedicated git-ignored directory under `workspace.local/` (create the
directory if it doesn't exist):

- **`workspace.local/worklogs/worklog-<date>.md`** — the backward-looking session record.
- **`workspace.local/contexts/claude-context-<date>.md`** — the forward-looking resume brief.

Both live under `workspace.local/`, the project's local workspace, so they stay local and never
enter the public history — that's deliberate. Do **not** put the handoff files under `docs/`,
`scratch/`, or anywhere tracked unless the user explicitly asks to commit a handoff. (Older copies
may still exist in `scratch/`; new ones go in the `workspace.local/` subdirs above.)

Never write cluster/queue/registry/account/host identifiers or absolute private paths into these
files — the same infrastructure non-disclosure rule that governs tracked files applies here too
(real infra detail lives only in the git-ignored `docs/infrastructure.local.md`).

## Handling an existing file for today

If `workspace.local/worklogs/worklog-<date>.md` (or the context file) already exists, do **not**
overwrite it and do **not** silently make a new name. Read it, then **append a new timestamped
section** so the day's notes accumulate in one file:

```markdown

---

## Update — <HH:MM local>

<the new session's content>
```

Add the same kind of appended section to the context file, but note that the context file is a
*current-state* document: the appended section should read as "here's the updated state now,"
superseding earlier sections rather than just adding history. Briefly say at the top of the new
section that it supersedes the earlier state.

## worklog-<date>.md structure

Follow the shape of `workspace.local/worklogs/worklog-20260719.md`. Fill sections that apply; drop
ones that don't (don't pad). Keep it terse and concrete — commit hashes, file paths, exact counts.

```markdown
# Worklog — <date>

Project: `neural-representation-denoising` (modern reproduction of the 2016 dissertation).
Branch `<branch>`, HEAD `<short-hash>`, <clean|N files uncommitted>, <N> tests passing.

## Summary of the session

<2-4 sentences: what this session accomplished and why.>

## <Phase / workstream> — <status, e.g. "Gate P3 PASSED" or "review close-out">

- <bulleted specifics: what was built/changed, with file paths and commit hashes>

## <Review close-out, if a review was addressed>

<per-finding resolution, grouped by severity if applicable>

## Repo hygiene / infra (this session)

<gitignore, infra note, teardown, plan/status edits — only if relevant>

## Not done / deferred

- <items intentionally left, with the reason (blocker, later phase, needs GPU, etc.)>

## Next

<what the next session should pick up; point at the context file>
```

## claude-context-<date>.md structure

Follow the shape of `workspace.local/contexts/claude-context-20260719.md`. This is the more
important file — a new session relies on it. Lead with a one-line "read this first" framing.

```markdown
# Resume context for a new Claude Code session — <date>

Read this first if you're picking up work on `neural-representation-denoising`. It tells you
where things stand, how the repo is wired, the non-obvious gotchas, and exactly what to do next.

## What this project is

<2-3 sentences: the project and its two studies.>

## Current state (authoritative)

- Branch `<branch>`, HEAD `<short-hash>`, <tree state>, <N> tests passing.
- <which phases/gates are complete vs open; cite decision records for deferrals>
- <tags, and any gate explicitly OPEN/blocked and on what>

## Repo layout that already exists

<the directories/modules that exist now, one line each, so the reader doesn't re-discover them>

## How to work in this repo (commands)

Start every shell with:
```sh
export PATH="/Library/TeX/texbin:/opt/homebrew/bin:$PATH"
```
Then the real commands (uv sync, pytest, ruff/mypy, release scripts, dissertation build) —
copy the ones currently in `CLAUDE.md` / README so they don't drift.

## Non-obvious gotchas (will bite you otherwise)

<numbered list of the traps: never edit legacy/, no infra identifiers in tracked files, the
closed provenance schema, uv.lock discipline, teardown test infra, no-overwrite rule, commit+push
+ keep the gate green, and any new one learned this session.>

## Key decisions already made

<the docs/decisions/ records that matter, one line each.>

## NEXT: <the next phase or task>

<a concrete, ordered plan for what to do next, with pointers to the exact plan section and source
files. Enough that a fresh session can start immediately.>
```

## After writing

Tell the user the two paths (`workspace.local/worklogs/…` and `workspace.local/contexts/…`), that
they're git-ignored (local-only, not committed), and one line on how they were verified (the
git/pytest facts you pulled). If you appended to an existing file rather than creating a new one,
say so.
