# Development Setup

This guide walks you through setting up a complete development environment for MolPy.

## Prerequisites

Before you begin, ensure you have:

- **Python 3.12 or higher** - MolPy requires Python 3.12+
- **Git** - For version control
- **pip** - Python package installer (comes with Python)

## Quick Setup

For experienced developers, here's the quick version:

```bash
git clone https://github.com/YOUR_USERNAME/molpy.git
cd molpy
python3.12 -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -U pip
pip install -e ".[dev]"
pre-commit install
pytest
```

## Detailed Setup

### 1. Fork and Clone the Repository

First, fork the repository on GitHub, then clone your fork:

```bash
git clone https://github.com/YOUR_USERNAME/molpy.git
cd molpy
```

Add the upstream repository as a remote:

```bash
git remote add upstream https://github.com/MolCrafts/molpy.git
```

### 2. Create a Virtual Environment

We strongly recommend using a virtual environment to isolate dependencies.

#### Using venv (recommended)

**On macOS/Linux:**
```bash
python3.12 -m venv venv
source venv/bin/activate
```

**On Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**On Windows (CMD):**
```cmd
python -m venv venv
venv\Scripts\activate.bat
```

#### Using conda

```bash
conda create -n molpy-dev python=3.12
conda activate molpy-dev
```

### 3. Install MolPy in Editable Mode

Install MolPy with development dependencies:

```bash
# Upgrade pip first
pip install -U pip

# Install MolPy in editable mode with dev dependencies
pip install -e ".[dev]"
```

This installs:
- MolPy in editable mode (changes to source code are immediately reflected)
- Development dependencies: pytest, pytest-cov, pytest-mock, pre-commit

### 4. Install Pre-commit Hooks

Pre-commit hooks automatically check your code before each commit:

```bash
pre-commit install
```

This will run:
- **Black** - Code formatting
- **Trailing whitespace** - Remove trailing whitespace
- **YAML/TOML/JSON checks** - Validate config files
- **nbstripout** - Strip notebook outputs

### 5. Verify Installation

Run the test suite to ensure everything is working:

```bash
pytest
```

You should see all tests passing. If any tests fail, check the [Troubleshooting](#troubleshooting) section.

## Optional Dependencies

### Documentation Dependencies

To build documentation locally:

```bash
pip install -e ".[doc]"
```

This installs:
- MkDocs and Material theme
- mkdocstrings for API docs
- mkdocs-jupyter for notebook rendering
- Additional dependencies: packmol, rdkit, matplotlib, scienceplots

### All Dependencies

To install everything:

```bash
pip install -e ".[all]"
```

## External Tools

Some features require external tools. These are optional but useful for certain workflows:

### LAMMPS

For running molecular dynamics simulations:

**macOS (Homebrew):**
```bash
brew install lammps
```

**Ubuntu/Debian:**
```bash
sudo apt-get install lammps
```

**From source:**
See [LAMMPS documentation](https://docs.lammps.org/Install.html)

### Packmol

For packing molecules into boxes:

**macOS (Homebrew):**
```bash
brew install packmol
```

**Ubuntu/Debian:**
```bash
sudo apt-get install packmol
```

**From source:**
See [Packmol website](http://m3g.iqm.unicamp.br/packmol/)

### AmberTools

For AMBER force field support:

See [AmberTools installation guide](https://ambermd.org/GetAmber.php)

## IDE Setup

### VS Code

Recommended extensions:

- **Python** - Microsoft's Python extension
- **Pylance** - Fast Python language server
- **Jupyter** - For notebook editing
- **autoDocstring** - Generate docstrings

Recommended settings (`.vscode/settings.json`):

```json
{
  "python.linting.enabled": true,
  "python.formatting.provider": "black",
  "python.testing.pytestEnabled": true,
  "python.testing.pytestArgs": ["tests"],
  "editor.formatOnSave": true,
  "editor.rulers": [88],
  "files.trimTrailingWhitespace": true
}
```

### PyCharm

1. Open the `molpy` directory as a project
2. Configure Python interpreter to use your virtual environment
3. Enable pytest as the test runner
4. Install Black plugin for formatting
5. Set line length to 88 in Code Style settings

## Running Tests

### Basic Usage

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_core/test_frame.py

# Run specific test function
pytest tests/test_core/test_frame.py::test_frame_creation

# Run tests matching a pattern
pytest -k "lammps"
```

### Coverage

```bash
# Run with coverage report
pytest --cov=molpy tests/

# Generate HTML coverage report
pytest --cov=molpy --cov-report=html tests/
# Open htmlcov/index.html in browser
```

### Markers

Some tests are marked with special markers:

```bash
# Skip tests requiring external tools
pytest -m "not external"

# Run only external tests
pytest -m "external"
```

## Building Documentation

### Serve Locally

```bash
# Make sure doc dependencies are installed
pip install -e ".[doc]"

# Serve documentation
mkdocs serve
```

Open http://127.0.0.1:8000 in your browser.

### Build Static Site

```bash
mkdocs build
```

The built site will be in the `site/` directory.

## Keeping Your Fork Updated

Regularly sync your fork with the upstream repository:

```bash
# Fetch upstream changes
git fetch upstream

# Merge upstream main into your main
git checkout main
git merge upstream/main

# Push to your fork
git push origin main
```

## Troubleshooting

### Tests Fail After Installation

**Problem:** Some tests fail with import errors.

**Solution:** Make sure you installed in editable mode:
```bash
pip install -e ".[dev]"
```

### Pre-commit Hooks Fail

**Problem:** Pre-commit hooks fail on first run.

**Solution:** Run manually to install hook environments:
```bash
pre-commit run --all-files
```

### Black Formatting Conflicts

**Problem:** Black and your editor disagree on formatting.

**Solution:** Ensure your editor uses the same Black version:
```bash
black --version
```

Check `pyproject.toml` for the Black configuration.

### Import Errors for Optional Dependencies

**Problem:** Tests fail with import errors for RDKit, freud, etc.

**Solution:** These are optional dependencies. Either:
- Install them: `pip install rdkit freud`
- Skip those tests: `pytest -m "not external"`

### Python Version Issues

**Problem:** Python 3.12 not found.

**Solution:** Install Python 3.12:
- **macOS:** `brew install python@3.12`
- **Ubuntu:** Use deadsnakes PPA
- **Windows:** Download from python.org

## Next Steps

Now that your environment is set up:

1. Read the [Coding Style Guide](coding-style.md)
2. Review the [Testing Guide](testing.md)
3. Check out the [Architecture Overview](architecture.md)
4. Start contributing! See [Contributing Guide](contributing.md)

## Getting Help

If you encounter issues:

- Check [GitHub Issues](https://github.com/MolCrafts/molpy/issues)
- Ask in [GitHub Discussions](https://github.com/MolCrafts/molpy/discussions)
- Review the [FAQ](../getting-started/faq.md)
