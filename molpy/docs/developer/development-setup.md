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

molpy lives in the molrs repository, at `molpy/`, next to the Python binding
it is built on (`molrs-python/`). Clone that repository and run its gates from
the root:

```bash
git clone https://github.com/MolCrafts/molrs.git
cd molrs
uvx pre-commit install
scripts/check.sh molpy molpy_docs
```

`molpy` builds the molrs wheel from the same checkout, installs it into
molpy's locked environment (`molpy/uv.lock`), then checks formatting, types
and the unit suite; `molpy_docs` builds the site with `--strict`. If both
pass, the environment is ready.


## Dependencies

`molpy/pyproject.toml`'s `[tool.uv.sources]` builds `molcrafts-molrs` from
`../molrs-python`, so molpy is always tested against the molrs of the same
commit. mollog and molcfg come from PyPI, within the `[project]` ranges.

`scripts/check.sh verify` is the pre-push gate. It runs `python` first, which
builds and tests the molrs wheel, and then `molpy` and `molpy_docs` on that
same wheel. CI runs `python molpy` in one call on each OS leg, and
`docs molpy_docs` on the Linux wheel.

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
successor), configured by `molpy/zensical.toml`. Install the doc extras and start a
local preview server from the repository root.

```bash
uv --directory molpy run --locked --extra doc zensical serve
```

The site is at `http://localhost:8000`. Changes to `.md` files are reflected immediately.

User-guide notebooks are pre-rendered to Markdown (Zensical does not run notebooks
at build time). After editing one, regenerate its page with
`python molpy/scripts/render_notebooks.py`.


## External tools

LAMMPS and AmberTools are optional for *using* MolPy offline recipes; packing uses molpack (`pip install molcrafts-molpack`).
they are **not** part of the unit suite. Engines and the AmberTools programs are tested with
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
