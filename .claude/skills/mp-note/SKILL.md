---
name: mp-note
description: Captures, sweeps or promotes project decisions; writes .claude/NOTES.md and AGENTS.md.
argument-hint: "<decision> | sweep | promote <id>"
user-invocable: true
---

# Note

## Procedure

**Capture** (`<decision>`):
1. Read AGENTS.md and .claude/NOTES.md.
2. Conflict check: if the decision contradicts an AGENTS.md rule or a note,
   show both and ask which wins; never keep two contradictory entries.
3. Append to NOTES.md: `### <id> (YYYY-MM-DD)` + decision + **Why** +
   **Applies to** (paths).

**Sweep**:
1. For each note, check its paths and claims against the repo; mark stale
   entries and delete those the operator confirms obsolete.
2. List entries stable for more than two weeks and unchallenged as promote
   candidates.

**Promote** (`promote <id>`):
1. Rewrite the note as a rule in the matching AGENTS.md section (no history,
   no dates); delete the note.

## Output

`note: captured <id>` | `note: N stale, N promote candidates` | `note: promoted <id> → AGENTS.md#<section>`
