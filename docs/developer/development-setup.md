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

The repository holds the whole molpy project: the Rust core (`crates/molrs`,
`crates/molrs-ffi`) and one directory per language binding — `molpy/`
(Python), `molwasm/`, `molc/`, `molcxx/`. The Python binding is a maturin
project: `molpy/Cargo.toml` + `molpy/src/` build the native module
`molpy._core`, and `molpy/molpy/` is the package.

```bash
git clone https://github.com/MolCrafts/molrs.git
cd molrs
uvx pre-commit install
uv --directory molpy sync --extra dev   # builds molpy in place (editable)
cd molpy && uv run pytest               # the Python suite
```

Run `pytest`, not `python -m pytest`: the latter puts the checkout on
`sys.path` ahead of the built package.


## Gates

`scripts/check.sh verify` is the pre-push gate, every check once. Its Python
gates are `python` (build one molpy wheel from this checkout, install it in
molpy's locked environment, `molpy/uv.lock`, and run the suite on it),
`types` (`molpy/molpy/_core.pyi` against the compiled module) and `docs`
(this manual, `--strict`, on the same wheel). CI runs `python` on every OS
leg and `docs` on the Linux wheel. mollog and molcfg come from PyPI, within
`molpy/pyproject.toml`'s `[project]` ranges.

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
successor), configured by `zensical.toml` at the repository root. Install the doc
extras into molpy's environment and start a local preview server.

```bash
uv --directory molpy run --locked --extra doc zensical serve -f ../zensical.toml
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

If imports fail after pulling new code, rebuild molpy in place: `uv --directory molpy sync --extra dev --reinstall-package molcrafts-molpy` (the Rust side changed). The manual builds with `scripts/check.sh docs`. If formatting checks fail, run `uv --directory molpy run ruff format molpy tests` before pushing.
