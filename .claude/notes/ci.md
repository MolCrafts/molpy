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

## Escape hatch

Hooks can be skipped per-commit without disabling them permanently:

```bash
SKIP=pytest git commit -m "wip: mid-feature"
SKIP=ty,pytest git commit -m "wip: draft"
```

## CI matrix convention

Test job runs on ubuntu + macos × Python 3.12/3.13 with
`pytest tests/ -v -m "not external" --cov=src/molpy`; coverage upload only from
ubuntu / 3.12.
