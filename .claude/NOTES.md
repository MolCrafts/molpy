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
