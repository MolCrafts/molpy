# Development Setup

After following this page you will have a working local environment with editable install, pre-commit hooks, and passing tests.

## Prerequisites

You need Python 3.12 or newer, Git, [uv](https://docs.astral.sh/uv/), and the
Rust toolchain ([`rustup`](https://rustup.rs/)): molpy builds
[molrs](molrs-backend.md), its Rust compute core, from source. molrs pins the
toolchain channel and components in its `rust-toolchain.toml`, so no manual
component setup is required. Everything else is installed by the commands
below.


## Quick setup

Clone molpy and run the shared gate entry point. It resolves the declared
partner commits into a cached sibling layout and compiles the Python extension.
Your existing sibling working trees do not affect verification.

```bash
git clone https://github.com/MolCrafts/molpy.git
cd molpy
uvx pre-commit install
python scripts/check.py verify
```

If all tests pass, the environment is ready.


## Pinned direct dependencies

`.github/partners.env` declares immutable commits for molrs, mollog, molcfg
and the shared resolver (`CI_REF`). `scripts/partners.py` bootstraps that
resolver at the declared commit. Branch names and sibling working trees never
override the manifest. `[tool.uv.sources]` uses the resolved sibling layout;
the wheel continues to declare its published dependency ranges.

Run `python scripts/check.py verify` before pushing. The pre-push hook runs
all hygiene/lint gates, then creates one partner layout, validates `uv.lock`,
installs the dependency graph, runs `uv pip check`, tests and strict docs. It
also runs the suite against published dependency ranges (`--no-sources`) and
checks the release distributions.
Set `MOLCRAFTS_PARTNER_CACHE` to a persistent directory to reuse builds.

For coordinated changes, publish the partner commit and update the declared
repository/SHA explicitly. If its package metadata changed, relock in that
layout and commit the lock:

```bash
python scripts/partners.py run -- sh -c 'uv lock && cp uv.lock "$PARTNERS_SOURCE/"'
```

CI covers OS differences on Python 3.12 and interpreter differences on Linux
3.14: four independent legs rather than a six-leg Cartesian product. A release
also checks the published dependencies using `--no-sources`.

## Developing across operating systems

Use the pinned Rust toolchain and Python 3.12 for the local gates. On Windows,
run the shell hooks in Git Bash with Python and uv on PATH; a WSL run proves
Linux behavior, so native Windows CI must still pass. The CI matrix separately
checks all three native OSes and the additional Python version.

Write text with `encoding="utf-8"`, use `pathlib` for Python paths, and preserve
LF endings through `.gitattributes`. Avoid filenames that differ only by case
and assumptions about `/tmp`, executable suffixes or a venv's `bin` directory.
Use `.venv/Scripts` and `.exe` on Windows. A container can align Linux tools,
but cannot validate Windows or macOS native bindings.

## Documentation preview

The doc site is built with [Zensical](https://zensical.org) (Material for MkDocs'
successor), configured by `zensical.toml` at the repo root. Install the doc
extras and start a local preview server from the repo root.

```bash
python scripts/partners.py run -- uv run --locked --extra doc zensical serve
```

The site is at `http://localhost:8000`. Changes to `.md` files are reflected immediately.

User-guide notebooks are pre-rendered to Markdown (Zensical does not run notebooks
at build time). After editing one, regenerate its page with
`python scripts/render_notebooks.py`.


## External tools

LAMMPS and AmberTools are optional for *using* MolPy offline recipes; packing uses molpack (`pip install molcrafts-molpack`).
they are **not** part of the unit suite. Wrappers and engines are tested with
mocks and script literals. Doc blocks that would shell out declare
`# docs: skip`.


## Common commands

```bash
ruff format --check src tests             # check formatting
ruff format src tests                     # auto-format
ruff check src                            # lint source tree
uv run --locked --extra dev python -m pytest tests/ -n auto   # the CI test command
pre-commit run --all-files                # all pre-commit hooks (see CONTRIBUTING.md "Hooks")
zensical build                            # build static doc site into site/
```


## Troubleshooting

If imports fail after pulling new code, run `uv sync --extra dev` (add `--reinstall-package molcrafts-molrs` when molrs changed). Docs site build is `uv sync --extra doc` + `uv run zensical build` (theme + mkdocstrings only — no notebook/matplotlib stack). If formatting checks fail in CI, run `ruff format src tests` locally before pushing.
