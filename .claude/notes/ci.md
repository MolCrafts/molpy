# CI / Pre-commit Conventions

Project-specific parity policy for `/mol:ci-sync` and the `ci-guard` agent.
Migrated from the former local `molpy-ci-sync` skill and `molpy-ci-auditor` agent
(2026-06-10). The authoritative configs are `.pre-commit-config.yaml` and
`.github/workflows/ci.yml` — the pre-commit file header states the parity rule:
every CI check is mirrored as a hook, and any divergence is a bug fixed in the
same commit.

## Canonical check set

| Check | Tool | Pre-commit | CI |
|---|---|---|---|
| Format | `ruff format --check src/ tests/` | required (pre-commit stage) | required |
| Lint | `ruff check src/ tests/` | required (pre-commit stage) | required |
| Type | `ty check src/molpy/` | required (pre-commit stage) | required |
| Tests | `pytest tests/ -v -m "not external"` | required (**pre-push stage**) | required |
| Notebook strip | `nbstripout` | required when `.ipynb` present | intentionally absent |
| File hygiene | `pre-commit-hooks` (trailing-ws, eof, merge-conflict, …) | required | intentionally absent |
| Docs build | `zensical build` | intentionally absent (too slow) | required when `docs/` present |

Intentional exemptions: `nbstripout` and file hygiene are local-only hooks;
`zensical build` / docs jobs are CI-only. Do not "fix" these as parity gaps.
Docs deploy is Cloudflare Pages (builds from the repo), not a GitHub workflow.

## Audit rules

- A check counts as "present in CI" only if it runs on `push` to the main branch
  or on `pull_request` — not merely on `workflow_dispatch`.
- Shared tool versions (ruff, ty) must not drift: flag when one side is pinned
  and the other floats, or when major versions differ.
- Ruff settings live in `pyproject.toml` / `ruff.toml`; CI must not override them
  with inline flags (drift risk).
- Install both hook stages: `pre-commit install --hook-type pre-commit --hook-type pre-push`.

## Whole-tree gates: tox, never `language: system` pytest

Canonical test gate: `uv run --extra dev tox -e py` with `package = "wheel"`.
That throwaway env installs molpy as a wheel and resolves
`molcrafts-molrs==X` from **PyPI** — it cannot see monorepo editables
(`.venv` `molcrafts_molrs.pth` → sibling checkout).

**2026-07-29 incident:** `dev` still used `entry: pytest … language: system`.
Editable molrs hid missing PyPI APIs (`write_lammps_forcefield`); CI failed
while local pre-push “passed”. **Never reintroduce system-pytest as the CI mirror.**

`tox-lint` / `tox-py` / `molrs-pin-on-pypi` must keep `always_run: true`.

## Ownership split (do not merge into tox)

| Concern | Owner |
|---|---|
| Fetch / skip tests-data | `tests/conftest.py` (session fixture) |
| `molcrafts-molrs==X` exists on PyPI | `.pre-commit-config.yaml` hook **`molrs-pin-on-pypi`** (pre-push) |
| Non-editable wheel smoke | `tox -e py` `commands_pre` (molpy under `site-packages` only) |
| Format / lint / type | `tox -e lint` |

Do **not** put tests-data download or the PyPI pin probe into `pyproject.toml`
tox commands.

## Escape hatch

Hooks can be skipped per-commit without disabling them permanently:

```bash
SKIP=pytest git commit -m "wip: mid-feature"
SKIP=ty,pytest git commit -m "wip: draft"
```

Do **not** `SKIP=molrs-pin-on-pypi` to land features that need unpublished molrs.

## CI matrix convention

Test job runs on ubuntu + macos × Python 3.12/3.13 with
`pytest tests/ -v -m "not external" --cov=src/molpy`; coverage upload only from
ubuntu / 3.12.
