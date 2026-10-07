"""One public path per symbol, across every public molpy module.

Walks ``dir()`` of each public module (not only ``__all__``), descending into
public submodules — a molpy module, or a molrs module molpy re-exports such
as ``mp.io.mrec``. A module path with a ``_``-prefixed segment is private and
not walked. Then:

* every class and function molpy defines has exactly one public path;
* every molrs class and function molpy re-exports has exactly one molpy path,
  except that a core data class promoted to the root is also ``molpy.core``'s
  (the same object);
* a file factory has one of molrs's two shapes — a ``read_*`` / ``write_*``
  function at the top of ``molpy.io``, or a ``*Reader`` / ``*Writer`` class of
  a ``molpy.io.<fmt>`` submodule;
* the root holds the subsystem modules, the promoted core data classes and
  the version metadata only;
* each molpy mirror is the molrs subsystem by identity;
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
_EXPLICIT = ("molpy.wrapper",)


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
    def own(paths: list[str]) -> list[str]:
        # A promoted class's root path is its second, sanctioned spelling.
        return [p for p in paths if p.rsplit(".", 1)[0] != "molpy"]

    doubles = {
        f"{obj.__module__}.{obj.__qualname__}": sorted(paths)
        for obj, paths in PATHS.values()
        if len(own(paths)) > 1 or not own(paths)
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
        "molpy.core.box",
        "molpy._core",
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
        "molpy.typifier",
        "molpy.integrations",
        "molpy.integrations.metric_readers",
        "molpy._core.selector",
        "molpy._core.splitter",
        "molpy.store",
        "molpy.system",
        "molpy.spatial",
        "molpy.units",
    ],
)
def test_a_removed_module_is_gone(module: str) -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module)


#: The molrs subsystems molpy mirrors one to one, each by a module of its name.
_MIRRORS = (
    "builder",
    "compute",
    "conformer",
    "ff",
    "io",
    "md",
    "op",
    "optimize",
    "perceive",
    "signal",
    "stream",
)

#: molrs's core subsystems (Rust ``molrs::core``), mirrored together by ``molpy.core``.
_CORE = ("store", "system", "spatial", "units")

#: The core data classes a user handles directly, promoted to the root.
_PROMOTED = {
    "Angle",
    "Atom",
    "Atomistic",
    "Bead",
    "Block",
    "Bond",
    "Box",
    "CGBond",
    "CoarseGrain",
    "Dihedral",
    "DrudeParticle",
    "Element",
    "Frame",
    "Graph",
    "Improper",
    "MasslessSite",
    "Port",
    "Topology",
    "Trajectory",
    "VirtualSite",
}

#: The per-format submodules of ``molrs.io``; molpy mirrors each by a module.
_IO_FORMATS = ("lammps_bond_react", "log", "mrec", "smiles", "trajectory")


def test_the_root_holds_modules_promoted_classes_and_the_version_only() -> None:
    names = {name for name in dir(molpy) if not name.startswith("_")}
    metadata = {"version", "release_date", "__version__"}
    modules = {n for n in names if isinstance(getattr(molpy, n), ModuleType)}
    assert names - metadata - modules == _PROMOTED
    assert set(molpy.__all__) == set(molpy._LAZY_SUBMODULES) | _PROMOTED | {
        "version",
        "release_date",
    }


@pytest.mark.parametrize("name", sorted(_PROMOTED))
def test_a_promoted_class_is_the_core_one_and_the_molrs_one(name: str) -> None:
    obj = getattr(molpy, name)
    assert inspect.isclass(obj)
    assert obj is getattr(molpy.core, name)
    assert any(getattr(getattr(molrs, sub), name, None) is obj for sub in _CORE)


def test_core_mirrors_molrs_core_by_identity() -> None:
    for sub in _CORE:
        native = getattr(molrs, sub)
        assert set(native.__all__) <= set(molpy.core.__all__)
        for attr in native.__all__:
            assert getattr(molpy.core, attr) is getattr(native, attr), f"{sub}.{attr}"


@pytest.mark.parametrize("name", _MIRRORS)
def test_every_molrs_subsystem_has_its_molpy_module(name: str) -> None:
    assert name in molrs.__all__
    module = importlib.import_module(f"molpy.{name}")
    assert getattr(molpy, name) is module
    native = importlib.import_module(f"molrs.{name}")
    for attr in native.__all__:
        ours = getattr(module, attr)
        if isinstance(ours, ModuleType) and ours.__name__.startswith("molpy."):
            continue  # molpy's mirror of a molrs submodule (mp.io.log, mp.ff.ir, …)
        assert ours is getattr(native, attr), f"molpy.{name}.{attr}"


def test_the_molrs_root_has_no_subsystem_molpy_lacks() -> None:
    assert set(molrs.__all__) == set(_MIRRORS) | set(_CORE)


def test_a_fresh_import_binds_molpys_io_format_modules() -> None:
    """``mp.io.log`` is molpy's module even before anything imports it by name."""
    import subprocess
    import sys

    code = (
        "import molpy as mp; "
        f"print(*[getattr(mp.io, f).__name__ for f in {_IO_FORMATS!r}])"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    ).stdout.split()
    assert out == [f"molpy.io.{fmt}" for fmt in _IO_FORMATS]


@pytest.mark.parametrize("fmt", _IO_FORMATS)
def test_each_io_format_is_a_molpy_module_mirroring_molrs(fmt: str) -> None:
    module = importlib.import_module(f"molpy.io.{fmt}")
    assert getattr(molpy.io, fmt) is module
    native = importlib.import_module(f"molrs.io.{fmt}")
    assert set(native.__all__) <= set(module.__all__)
    for attr in native.__all__:
        assert getattr(module, attr) is getattr(native, attr), f"{fmt}.{attr}"


@pytest.mark.parametrize(
    ("ours", "native"),
    [
        ("io.read_gro", "io.read_gro"),
        ("io.read_smiles", "io.read_smiles"),
        ("io.read_amber_prmtop_system", "io.read_amber_prmtop_system"),
        ("io.write_gromacs_system", "io.write_gromacs_system"),
        ("io.read_lammps_log_str", "io.read_lammps_log_str"),
        ("io.read_frame_bytes", "io.read_frame_bytes"),
        ("io.read_mrec_trajectory", "io.read_mrec_trajectory"),
        ("io.smiles.SmilesIR", "io.smiles.SmilesIR"),
        ("io.mrec.MrecReader", "io.mrec.MrecReader"),
        ("io.log.LammpsLog", "io.log.LammpsLog"),
        ("io.trajectory.TrajectoryReader", "io.trajectory.TrajectoryReader"),
        (
            "io.lammps_bond_react.BondReactTemplate",
            "io.lammps_bond_react.BondReactTemplate",
        ),
        ("Frame", "store.Frame"),
        ("core.Frame", "store.Frame"),
        ("Trajectory", "store.Trajectory"),
        ("core.keys", "store.keys"),
        ("Atomistic", "system.Atomistic"),
        ("core.NodeRef", "system.NodeRef"),
        ("Box", "spatial.Box"),
        ("core.Cuboid", "spatial.Cuboid"),
        ("core.UnitRegistry", "units.UnitRegistry"),
        ("perceive.SmartsPattern", "perceive.SmartsPattern"),
        ("optimize.LBFGS", "optimize.LBFGS"),
        ("conformer.Conformer", "conformer.Conformer"),
        ("stream.ControlCommand", "stream.ControlCommand"),
        ("ff.forcefield.ForceField", "ff.forcefield.ForceField"),
    ],
)
def test_a_native_name_is_the_molrs_object(ours: str, native: str) -> None:
    def resolve(root: ModuleType, dotted: str) -> object:
        obj: object = root
        for part in dotted.split("."):
            obj = getattr(obj, part)
        return obj

    assert resolve(molpy, ours) is resolve(molrs, native)


def _is_factory_name(name: str) -> bool:
    return name.startswith(("read_", "write_"))


def test_a_file_factory_function_is_public_only_on_top_of_io() -> None:
    misplaced = sorted(
        path
        for obj, paths in PATHS.values()
        if inspect.isroutine(obj)
        for path in paths
        if _is_factory_name(path.rsplit(".", 1)[1])
        and path.rsplit(".", 1)[0] != "molpy.io"
    )
    assert not misplaced


def test_a_reader_or_writer_class_lives_in_an_io_format_module() -> None:
    misplaced = sorted(
        path
        for obj, paths in PATHS.values()
        if inspect.isclass(obj)
        for path in paths
        if path.rsplit(".", 1)[1].endswith(("Reader", "Writer"))
        and not (path.startswith("molpy.io.") and path.count(".") == 3)
    )
    assert not misplaced


def test_the_metric_readers_are_their_formats_modules() -> None:
    from molpy.io.log import LammpsLogMetricReader, MlpJsonlMetricReader
    from molpy.io.mrec import MrecMetricReader

    assert {"LammpsLogMetricReader", "MlpJsonlMetricReader"} <= set(
        molpy.io.log.__all__
    )
    assert "MrecMetricReader" in molpy.io.mrec.__all__
    # molrs's MrecReader is the store cursor; molpy's metric reader is another class.
    assert MrecMetricReader is not molrs.io.mrec.MrecReader
    assert LammpsLogMetricReader.format == "lammps_log"
    assert MlpJsonlMetricReader.format == "mlp_jsonl"


@pytest.mark.parametrize(
    ("module", "name"),
    [
        ("molpy.data", "get_forcefield_path"),
        ("molpy.data", "list_forcefields"),
        ("molpy.wrapper", "write_prepgen_control_file"),
        ("molpy.ff.forcefield", "read_amber_prmtop_system"),
        ("molpy.ff.forcefield", "write_gromacs_system"),
        ("molpy.stream", "read_frame_bytes"),
        ("molpy.io", "SmilesIR"),
        ("molpy.io", "TrajectoryReader"),
        ("molpy.io", "parse_lammps_log_text"),
        ("molpy", "keys"),
        ("molpy", "Cuboid"),
        ("molpy", "UnitRegistry"),
        ("molpy", "UnitPreset"),
        ("molpy", "NeighborList"),
        ("molpy", "LBFGS"),
        ("molpy", "Conformer"),
        ("molpy", "SmartsPattern"),
        ("molpy", "Perceive"),
        ("molpy", "ElementSelector"),
        ("molpy", "TrajectorySplitter"),
        ("molpy", "FrameCollection"),
    ],
)
def test_a_removed_name_is_gone(module: str, name: str) -> None:
    assert not hasattr(importlib.import_module(module), name)


def test_the_box_and_the_trajectory_are_the_native_classes() -> None:
    assert molpy.Box is molrs.spatial.Box
    assert molpy.Trajectory is molrs.store.Trajectory


def test_read_smiles_and_the_amber_system_reader_are_on_io() -> None:
    assert "read_smiles" in molpy.io.__all__
    assert "read_amber_prmtop_system" in molpy.io.__all__
    mol = molpy.io.read_smiles("CCO")
    assert isinstance(mol, molpy.Atomistic)
