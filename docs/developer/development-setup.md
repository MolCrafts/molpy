# Development Setup

After following this page you will have a working local environment with editable install, pre-commit hooks, and passing tests.

## Prerequisites

You need Python 3.14+, Git, and pip. Everything else is installed by the setup script below.


## Quick setup

Clone the repository, create a virtualenv, install in editable mode with dev dependencies, and run the test suite to confirm everything works.

```bash
git clone https://github.com/MolCrafts/molpy.git
cd molpy
python3.14 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"
pre-commit install
pytest tests/ -v
```

If all tests pass, the environment is ready.


## Building molrs from source

The quick setup above resolves [molrs](molrs-backend.md) — molpy's required
Rust compute core — from the published `molcrafts-molrs` wheel on PyPI within
the **major.minor** range in `pyproject.toml` (`>=X.Y.0,<X.(Y+1)`). That is
the right path for most molpy development. Import-time
`check_molrs_version` accepts patch drift inside that minor.

If you are changing the Rust core *and* molpy together, build molrs editable
from a local checkout instead (local rebuilds do **not** count as a release
for the pre-push pin gate — see [Release Process](release-process.md)). molrs
ships its Python bindings as a [maturin](https://www.maturin.rs/) project, so
this step needs the Rust toolchain — install it via
[`rustup`](https://rustup.rs/); molrs pins the toolchain channel and components
in its `rust-toolchain.toml`, so no manual component setup is required inside
the checkout.

```bash
# in a sibling checkout next to molpy
git clone https://github.com/MolCrafts/molrs.git
cd molrs
pip install maturin
maturin develop -m molrs-python/Cargo.toml --release   # installs `molrs` editable into the venv

# back in molpy, the editable install now resolves the local molrs
cd ../molpy
pip install -e ".[dev]"
python -c "import molpy as mp; print(mp.version, mp.Frame(), mp.Element('C').symbol)"
```

Re-run `maturin develop` after any change to the molrs Rust source to recompile
the extension. See the
[molrs build-from-source guide](https://molrs.molcrafts.org/getting-started/installation/)
for the native-crate and WASM build targets.


## Documentation preview

The doc site is built with [Zensical](https://zensical.org) (Material for MkDocs'
successor), configured by `zensical.toml` at the repo root. Install the doc
extras and start a local preview server from the repo root.

```bash
pip install -e ".[doc]"
zensical serve
```

The site is at `http://localhost:8000`. Changes to `.md` files are reflected immediately.

User-guide notebooks are pre-rendered to Markdown (Zensical does not run notebooks
at build time). After editing one, regenerate its page with
`python scripts/render_notebooks.py`.


## External tools

LAMMPS, Packmol, and AmberTools are optional for *using* MolPy offline recipes;
they are **not** part of the unit suite. Wrappers and engines are tested with
mocks and script literals. Doc blocks that would shell out declare
`# docs: skip`.


## Common commands

```bash
ruff format --check src tests             # check formatting
ruff format src tests                     # auto-format
ruff check src                            # lint source tree
pytest tests/ -v        # local test suite
pytest --cov=src/molpy tests/ -v          # with coverage
pre-commit run --all-files                # all pre-commit hooks
zensical build                            # build static doc site into site/
```


## Troubleshooting

If imports fail after pulling new code, reinstall the editable package: `pip install -e ".[dev]"`. To regenerate the user-guide notebook pages, run `python scripts/render_notebooks.py` (needs the `[doc]` extras, including RDKit and Packmol). If formatting checks fail in CI, run `ruff format src tests` locally before pushing.
