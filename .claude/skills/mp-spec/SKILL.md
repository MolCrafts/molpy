---
name: mp-spec
description: Turns a requirement into a spec with an acceptance contract; writes .claude/specs/<slug>.md and a row in .claude/specs/INDEX.md.
argument-hint: "<requirement in plain words>"
user-invocable: true
---

# Spec

Read AGENTS.md for molpy conventions and .claude/NOTES.md for open decisions.

## Procedure

1. Restate the requirement in one sentence; pick a kebab `slug`.
2. Read the code the requirement touches; list current behaviour with
   `path:line` evidence.
3. Delegate to `mp-architect`: where should the change live (core, ffi, which
   bindings), what moves, what is deleted.
4. If the change computes physics or numbers, delegate to `mp-science` for
   the equations, units and reference values the spec must name.
5. Write `.claude/specs/<slug>.md`: frontmatter (`title`, `slug`, `status:
   draft`, `created`), Context, Decisions, Chain (one link per landable
   step), per link: goal, scope, acceptance (commands that must pass and
   what they prove), risks. Open questions at the end.
6. Add one row to `.claude/specs/INDEX.md` (spec link + one-line summary |
   state).
7. Ask the operator only questions whose answer changes the spec.

## Output

`spec: .claude/specs/<slug>.md — N links, N open questions`
