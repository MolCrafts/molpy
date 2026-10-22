# Notes

Evolving decisions, captured with `/mp-note`. A note is promoted into
AGENTS.md once it is stable, then deleted here. Each entry: id and date,
decision, **Why**, **Applies to**.

### mrec-gate-red-until-04 (2026-10-22)

The `mrec` gate stays red until 04-downstream: the molrec repository's
`tests/molrs_adapter.py` still imports the deleted Python `molrs` package.
The fix belongs in molrec (adapter → `molpy`), not in a shim here.
**Why:** no backward-compatibility apparatus; downstream migrates in 04.
**Applies to:** `scripts/check.sh mrec`, `scripts/partners.py`.

### open-defects-found-by-manual-update (2026-10-22)

Found while rewriting the manual; not fixed yet (iron law: named, routed to
`/mp-fix`):

- `mp.io.lammps.LammpsForcefieldWriter` writes a force field with
  `units="lj"` as `units real` with no conversion and no error.
- `mp.io.write_lammps_data` creates or truncates the target file before it
  refuses a frame (e.g. a type with no mass), leaving a header-only file.
- `mp.io.mrec.MrecWriter` accepts a bare `str` path; every other reader and
  writer refuses one.

**Applies to:** `crates/molrs/src/io/lammps/`, `crates/molrs/src/io/mrec*`,
`molpy/src/io/`.
