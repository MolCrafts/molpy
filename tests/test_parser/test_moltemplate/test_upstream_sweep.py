"""Sweep test over every upstream moltemplate example.

Runs only when a moltemplate examples tree is available (either via the
``MOLTEMPLATE_EXAMPLES`` env var pointing at the ``examples/`` directory,
or when ``/tmp/moltemplate_repo/examples`` exists). Asserts that every
``system.lt`` file builds without error and produces a non-empty
``(Atomistic, ForceField)`` tuple.

The expensive per-example cost makes this a ``@pytest.mark.slow`` test;
skip with ``pytest -m "not slow"``.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import pytest


def _examples_root() -> Path | None:
    env = os.environ.get("MOLTEMPLATE_EXAMPLES")
    if env and Path(env).is_dir():
        return Path(env)
    default = Path("/tmp/moltemplate_repo/examples")
    return default if default.is_dir() else None


@pytest.mark.slow
@pytest.mark.external
def test_every_upstream_system_lt_builds():
    root = _examples_root()
    if root is None:
        pytest.skip(
            "Set MOLTEMPLATE_EXAMPLES to the moltemplate/examples dir "
            "(or clone it to /tmp/moltemplate_repo)."
        )
    from molpy.io.forcefield.moltemplate import read_moltemplate_system

    paths = sorted(root.rglob("system.lt"))
    assert paths, f"no system.lt under {root}"

    failures: list[tuple[Path, str]] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for p in paths:
            try:
                atomistic, ff = read_moltemplate_system(p)
                assert atomistic is not None
                assert ff is not None
            except Exception as exc:  # noqa: BLE001
                failures.append((p, f"{type(exc).__name__}: {exc}"))

    if failures:
        report = "\n".join(f"  {p.relative_to(root)}: {err}" for p, err in failures)
        pytest.fail(
            f"{len(failures)} / {len(paths)} upstream examples failed:\n{report}"
        )
