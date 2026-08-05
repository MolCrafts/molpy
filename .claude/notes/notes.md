# Notes

Evolving architectural decisions and project-level rules. Populated by `/mol:note`.

## release-molrs-first

**Promoted** → `.claude/notes/release.md` + CLAUDE.md § Release with molrs.
Monorepo merge under molpy: **retracted**. Pin-parity scripts: **forbidden**.

## pathlike-boundary-path-internal

**Public / user-facing** path parameters accept `str | Path` (path-like).
**At the boundary**, convert with `Path(...)` / `Path(...).expanduser()`.
**Internal fields and logic use only `pathlib.Path`** — never keep bare `str`
paths for filesystem locations after construction. Subprocess argv / env dicts
are the only place to re-emit `str(path)`.

Exceptions: symbolic names that are *not* paths (e.g. conda env name
`"AmberTools25"`) stay `str`. Path-like conda prefixes are stored as `Path`.
