"""One public path per symbol, across every public molpy module.

Walks ``dir()`` of each public module (not only ``__all__``), descending into
public submodules — a molpy module, or a molrs module molpy re-exports such
as ``mp.io.mrec``. A module path with a ``_``-prefixed segment is private and
not walked. Then:

* every class and function molpy defines has exactly one public path;
* every molrs class and function molpy re-exports has exactly one molpy path;
* the names molpy dropped for the molrs door are gone.
"""

from __future__ import annotations

import importlib
import inspect
from types import ModuleType

import molrs
import pytest

import molpy

#: molpy's own subpackages that are imported explicitly, not lazily.
_EXPLICIT = ("molpy.wrapper", "molpy.integrations", "molpy.integrations.metric_readers")


def _walk() -> dict[int, tuple[object, list[str]]]:
    """``id(obj) -> (obj, [public paths])`` for every class and function."""
    seen: dict[int, tuple[object, list[str]]] = {}
    visited: set[str] = set()
    roots: list[tuple[str, ModuleType]] = [("molpy", molpy)]
    roots += [(name, importlib.import_module(name)) for name in _EXPLICIT]
    for name in sorted(molpy._LAZY_SUBMODULES):
        roots.append((f"molpy.{name}", importlib.import_module(f"molpy.{name}")))
    queue = list(roots)
    while queue:
        path, module = queue.pop()
        if path in visited:
            continue
        visited.add(path)
        for attr in dir(module):
            if attr.startswith("_"):
                continue
            obj = getattr(module, attr)
            if isinstance(obj, ModuleType):
                if obj.__name__.split(".")[0] in ("molpy", "molrs"):
                    queue.append((f"{path}.{attr}", obj))
                continue
            if not (inspect.isclass(obj) or inspect.isroutine(obj)):
                continue
            owner = getattr(obj, "__module__", "") or ""
            if owner.split(".")[0] not in ("molpy", "molrs"):
                continue
            seen.setdefault(id(obj), (obj, []))[1].append(f"{path}.{attr}")
    return seen


PATHS = _walk()


def test_every_symbol_has_one_public_path() -> None:
    doubles = {
        f"{obj.__module__}.{obj.__qualname__}": sorted(paths)
        for obj, paths in PATHS.values()
        if len(paths) > 1
    }
    assert not doubles


def test_no_public_path_runs_through_a_private_module() -> None:
    assert not [
        path
        for _, paths in PATHS.values()
        for path in paths
        if any(part.startswith("_") for part in path.split("."))
    ]


@pytest.mark.parametrize(
    "module",
    [
        "molpy.core",
        "molpy.core.box",
        "molpy.core.trajectory",
        "molpy.engine.base",
        "molpy.engine.lammps",
        "molpy.engine.gromacs",
        "molpy.engine.openmm",
        "molpy.engine.cp2k",
        "molpy.engine.script",
        "molpy.wrapper.base",
        "molpy.wrapper.env",
        "molpy.wrapper.antechamber",
        "molpy.wrapper.prepgen",
        "molpy.wrapper.sander",
        "molpy.wrapper.tleap",
        "molpy.adapter.base",
        "molpy.adapter.rdkit",
        "molpy.io.readers",
        "molpy.potential",
        "molpy.conformer",
        "molpy.typifier",
    ],
)
def test_a_removed_module_is_gone(module: str) -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module)


def test_the_box_and_the_trajectory_are_the_native_classes() -> None:
    assert molpy.Box is molrs.spatial.Box
    assert molpy.Trajectory is molrs.store.Trajectory


@pytest.mark.parametrize("name", ["read_smiles", "read_amber"])
def test_a_removed_io_wrapper_is_gone(name: str) -> None:
    assert not hasattr(molpy.io, name)
    assert name not in molpy.io.__all__
