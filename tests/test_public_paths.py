"""One public path per symbol, across every public molpy module.

Walks ``dir()`` of each public module (not only ``__all__``), descending into
public submodules — a molpy module, or a molrs module molpy re-exports such
as ``mp.core.keys``. A module path with a ``_``-prefixed segment is private
and not walked. Then:

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
* acronyms are cased as words in every class name;
* the names molpy dropped for the molrs door, and the names molrs renamed,
  are gone.
"""

from __future__ import annotations

import importlib
import inspect
import re
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


def _resolve(root: ModuleType, dotted: str) -> object:
    obj: object = root
    for part in dotted.split("."):
        obj = getattr(obj, part)
    return obj


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
        # molpy's own old layouts
        "molpy.core.box",
        "molpy._core",
        "molpy.core.trajectory",
        "molpy.engine.base",
        "molpy.engine._base",
        "molpy.engine.lammps",
        "molpy.engine.gromacs",
        "molpy.engine.openmm",
        "molpy.engine.cp2k",
        "molpy.engine.script",
        "molpy.wrapper.base",
        "molpy.wrapper._base",
        "molpy.wrapper.env",
        "molpy.wrapper._env",
        "molpy.wrapper.antechamber",
        "molpy.wrapper.prepgen",
        "molpy.wrapper.sander",
        "molpy.wrapper.tleap",
        "molpy.adapter.base",
        "molpy.adapter._base",
        "molpy.adapter.rdkit",
        "molpy.builder._polymer.ambertools.types",
        "molpy.data",
        "molpy.data.forcefield",
        "molpy.io._metric",
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
        # molrs's renamed modules
        "molpy.io.log",
        "molpy.io.trajectory",
        "molpy.io.lammps_bond_react",
        "molpy.ff.scale_lj",
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
    "core",
    "ff",
    "io",
    "md",
    "op",
    "optimize",
    "perceive",
    "signal",
    "stream",
)

#: molpy's own subsystems, beside the mirrors.
_OWN = ("adapter", "engine", "resources")

#: The core data classes a user handles directly, promoted to the root.
_PROMOTED = {
    "Angle",
    "Atom",
    "Atomistic",
    "Bead",
    "Block",
    "Bond",
    "Box",
    "CgBond",
    "CoarseGrain",
    "Dihedral",
    "DrudeParticle",
    "Element",
    "Frame",
    "Improper",
    "MasslessSite",
    "MolGraph",
    "Port",
    "Topology",
    "Trajectory",
    "VirtualSite",
}

#: The per-format submodules of ``molrs.io``; molpy mirrors each by a module.
_IO_FORMATS = (
    "cgsmiles",
    "dcd",
    "gro",
    "lammps",
    "mrec",
    "pdb",
    "smiles",
    "trr",
    "xtc",
    "xyz",
)

#: The submodules of ``molrs.ff``; molpy mirrors each by a module.
_FF_MODULES = (
    "charge",
    "clpol_scaling",
    "forcefield",
    "ir",
    "params",
    "potential",
    "typifier",
)


def test_the_root_holds_modules_promoted_classes_and_the_version_only() -> None:
    names = {name for name in dir(molpy) if not name.startswith("_")}
    metadata = {"version", "release_date", "__version__"}
    modules = {n for n in names if isinstance(getattr(molpy, n), ModuleType)}
    assert names - metadata - modules == _PROMOTED
    assert set(molpy.__all__) == set(_MIRRORS) | set(_OWN) | _PROMOTED | {
        "version",
        "release_date",
    }
    assert set(molpy._LAZY_SUBMODULES) == set(_MIRRORS) | set(_OWN)


def test_the_root_holds_no_function() -> None:
    assert not [
        name
        for name in dir(molpy)
        if not name.startswith("_") and inspect.isroutine(getattr(molpy, name))
    ]


@pytest.mark.parametrize("name", sorted(_PROMOTED))
def test_a_promoted_class_is_the_core_one_and_the_molrs_one(name: str) -> None:
    obj = getattr(molpy, name)
    assert inspect.isclass(obj)
    assert obj is getattr(molpy.core, name)
    assert obj is getattr(molrs.core, name)


def test_core_mirrors_molrs_core_by_identity() -> None:
    assert set(molrs.core.__all__) <= set(molpy.core.__all__)
    for attr in molrs.core.__all__:
        assert getattr(molpy.core, attr) is getattr(molrs.core, attr), attr
    for vocabulary in ("keys", "schema", "constants"):
        assert getattr(molpy.core, vocabulary) is getattr(molrs.core, vocabulary)


@pytest.mark.parametrize("name", _MIRRORS)
def test_every_molrs_subsystem_has_its_molpy_module(name: str) -> None:
    assert name in molrs.__all__
    module = importlib.import_module(f"molpy.{name}")
    assert getattr(molpy, name) is module
    native = importlib.import_module(f"molrs.{name}")
    assert set(native.__all__) <= set(module.__all__)
    for attr in native.__all__:
        ours = getattr(module, attr)
        if isinstance(ours, ModuleType) and ours.__name__.startswith("molpy."):
            continue  # molpy's mirror of a molrs submodule (mp.io.lammps, mp.ff.ir, …)
        assert ours is getattr(native, attr), f"molpy.{name}.{attr}"


def test_the_molrs_root_has_no_subsystem_molpy_lacks() -> None:
    assert set(molrs.__all__) == set(_MIRRORS)


def test_a_fresh_import_binds_molpys_io_format_modules() -> None:
    """``mp.io.lammps`` is molpy's module even before anything imports it by name."""
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


def test_molpy_io_has_every_molrs_io_format_module() -> None:
    native = {
        name
        for name in molrs.io.__all__
        if isinstance(getattr(molrs.io, name), ModuleType)
    }
    assert native == set(_IO_FORMATS)


@pytest.mark.parametrize("sub", _FF_MODULES)
def test_each_ff_submodule_is_a_molpy_module_mirroring_molrs(sub: str) -> None:
    module = importlib.import_module(f"molpy.ff.{sub}")
    assert getattr(molpy.ff, sub) is module
    native = importlib.import_module(f"molrs.ff.{sub}")
    for attr in native.__all__:
        assert getattr(module, attr) is getattr(native, attr), f"ff.{sub}.{attr}"
    assert set(molpy.ff.__all__) == set(molrs.ff.__all__) == set(_FF_MODULES)


@pytest.mark.parametrize(
    ("ours", "native"),
    [
        ("io.read_gro", "io.read_gro"),
        ("io.read_smiles_str", "io.read_smiles_str"),
        ("io.write_smiles_str", "io.write_smiles_str"),
        ("io.read_cgsmiles_str", "io.read_cgsmiles_str"),
        ("io.read_amber_prmtop_system", "io.read_amber_prmtop_system"),
        ("io.read_amber_ac", "io.read_amber_ac"),
        ("io.write_gromacs_top_system", "io.write_gromacs_top_system"),
        ("io.read_lammps_log_str", "io.read_lammps_log_str"),
        ("io.read_msgpack_frame_bytes", "io.read_msgpack_frame_bytes"),
        ("io.read_json_frame_str", "io.read_json_frame_str"),
        ("io.read_mrec_frame", "io.read_mrec_frame"),
        ("io.read_mrec_trajectory", "io.read_mrec_trajectory"),
        ("io.read_openmm_xml_forcefield", "io.read_openmm_xml_forcefield"),
        ("io.write_openmm_xml_forcefield", "io.write_openmm_xml_forcefield"),
        ("io.read_molrs_xml_forcefield", "io.read_molrs_xml_forcefield"),
        ("io.read_lammps_cmap_forcefield", "io.read_lammps_cmap_forcefield"),
        ("io.write_lammps_bond_react_map", "io.write_lammps_bond_react_map"),
        ("io.read_csv_block", "io.read_csv_block"),
        ("io.read_vasp_chgcar", "io.read_vasp_chgcar"),
        ("io.smiles.SmilesIr", "io.smiles.SmilesIr"),
        ("io.cgsmiles.CgSmilesIr", "io.cgsmiles.CgSmilesIr"),
        ("io.mrec.MrecReader", "io.mrec.MrecReader"),
        ("io.mrec.ForceFieldSection", "io.mrec.ForceFieldSection"),
        ("io.mrec.pack_mrec_zip", "io.mrec.pack_mrec_zip"),
        ("io.lammps.LammpsLog", "io.lammps.LammpsLog"),
        ("io.lammps.LammpsDumpReader", "io.lammps.LammpsDumpReader"),
        ("io.lammps.BondReactTemplate", "io.lammps.BondReactTemplate"),
        ("io.pdb.PdbReader", "io.pdb.PdbReader"),
        ("io.xyz.XyzReader", "io.xyz.XyzReader"),
        ("io.dcd.DcdReader", "io.dcd.DcdReader"),
        ("Frame", "core.Frame"),
        ("core.Frame", "core.Frame"),
        ("Trajectory", "core.Trajectory"),
        ("MolGraph", "core.MolGraph"),
        ("CgBond", "core.CgBond"),
        ("core.keys", "core.keys"),
        ("core.schema", "core.schema"),
        ("core.constants", "core.constants"),
        ("Atomistic", "core.Atomistic"),
        ("core.NodeRef", "core.NodeRef"),
        ("Box", "core.Box"),
        ("core.Cuboid", "core.Cuboid"),
        ("core.UnitRegistry", "core.UnitRegistry"),
        ("perceive.SmartsPattern", "perceive.SmartsPattern"),
        ("perceive.perceive_rings", "perceive.perceive_rings"),
        ("perceive.assign_aromaticity", "perceive.assign_aromaticity"),
        ("perceive.add_hydrogens", "perceive.add_hydrogens"),
        ("op.Superposition", "op.Superposition"),
        ("optimize.Lbfgs", "optimize.Lbfgs"),
        ("optimize.OptimizationReport", "optimize.OptimizationReport"),
        ("conformer.Conformer", "conformer.Conformer"),
        ("stream.ControlCommand", "stream.ControlCommand"),
        ("md.MdDriver", "md.MdDriver"),
        ("md.MdState", "md.MdState"),
        ("compute.Msd", "compute.Msd"),
        ("compute.Rdf", "compute.Rdf"),
        ("compute.Pca", "compute.Pca"),
        ("compute.Kmeans", "compute.Kmeans"),
        ("compute.DistributionFunction", "compute.DistributionFunction"),
        ("compute.BondOrientationalOrder", "compute.BondOrientationalOrder"),
        ("ff.forcefield.ForceField", "ff.forcefield.ForceField"),
        ("ff.forcefield.ForceFieldType", "ff.forcefield.ForceFieldType"),
        ("ff.potential.PairLjCut", "ff.potential.PairLjCut"),
        ("ff.potential.WeightedTerms", "ff.potential.WeightedTerms"),
        ("ff.potential.compile_explicit_terms", "ff.potential.compile_explicit_terms"),
        ("ff.ir.ParamSpec", "ff.ir.ParamSpec"),
        ("ff.ir.CategorySpec", "ff.ir.CategorySpec"),
        ("ff.ir.StyleSpec", "ff.ir.StyleSpec"),
        ("ff.ir.StyleDeclaration", "ff.ir.StyleDeclaration"),
        ("ff.ir.unregister_style", "ff.ir.unregister_style"),
        ("ff.ir.ArityError", "ff.ir.ArityError"),
        ("ff.typifier.TypeAssignment", "ff.typifier.TypeAssignment"),
        ("ff.typifier.OplsAaTypifier", "ff.typifier.OplsAaTypifier"),
        ("ff.typifier.Mmff94Typifier", "ff.typifier.Mmff94Typifier"),
        ("ff.params.clpol_fragment_scaling", "ff.params.clpol_fragment_scaling"),
        ("ff.clpol_scaling.scale_lj", "ff.clpol_scaling.scale_lj"),
    ],
)
def test_a_native_name_is_the_molrs_object(ours: str, native: str) -> None:
    assert _resolve(molpy, ours) is _resolve(molrs, native)


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


def test_the_top_of_io_holds_functions_and_format_modules_only() -> None:
    stray = sorted(
        name
        for name in dir(molpy.io)
        if not name.startswith("_")
        and not inspect.isroutine(getattr(molpy.io, name))
        and not isinstance(getattr(molpy.io, name), ModuleType)
    )
    assert not stray


#: Spellings an acronym-as-word rule keeps: numpy's ``DType`` and ``HBond``,
#: where H is the element.
_KEPT_CAPITALS = ("DType", "HBond")

#: Two capitals in a row that are not the start of a cased word: an acronym
#: written in capitals (``RDF``, ``LBFGS``, ``CP2K``, ``TLeap``).
_ACRONYM = re.compile(r"[A-Z]{2,}")


def test_no_class_spells_an_acronym_in_capitals() -> None:
    shouting = sorted(
        path
        for obj, paths in PATHS.values()
        if inspect.isclass(obj)
        for path in paths
        if _ACRONYM.search(re.sub("|".join(_KEPT_CAPITALS), "", path.rsplit(".", 1)[1]))
    )
    assert not shouting


#: molpy-owned module names that are container words, not what they hold.
_CONTAINER_WORDS = {
    "base",
    "common",
    "data",
    "env",
    "helpers",
    "misc",
    "types",
    "util",
    "utils",
}


def test_no_molpy_module_is_named_after_a_container_word() -> None:
    import pkgutil

    names = [
        info.name for info in pkgutil.walk_packages(molpy.__path__, prefix="molpy.")
    ]
    offenders = sorted(
        name
        for name in names
        if any(part.lstrip("_") in _CONTAINER_WORDS for part in name.split(".")[1:])
    )
    assert not offenders


def test_the_metric_readers_are_their_formats_modules() -> None:
    from molpy.io.lammps import LammpsLogMetricReader
    from molpy.io.mlp_jsonl import MlpJsonlMetricReader
    from molpy.io.mrec import MrecMetricReader

    assert "LammpsLogMetricReader" in molpy.io.lammps.__all__
    assert "MlpJsonlMetricReader" in molpy.io.mlp_jsonl.__all__
    assert "MrecMetricReader" in molpy.io.mrec.__all__
    # molrs's MrecReader is the store cursor; molpy's metric reader is another class.
    assert MrecMetricReader is not molrs.io.mrec.MrecReader
    assert LammpsLogMetricReader.format == "lammps_log"
    assert MlpJsonlMetricReader.format == "mlp_jsonl"


@pytest.mark.parametrize(
    ("module", "name"),
    [
        # molpy's own removals and renames
        ("molpy.resources", "get_forcefield_path"),
        ("molpy.resources", "list_forcefields"),
        ("molpy.wrapper", "write_prepgen_control_file"),
        ("molpy.wrapper", "EnvSpec"),
        ("molpy.wrapper", "TLeapWrapper"),
        ("molpy.engine", "LAMMPSEngine"),
        ("molpy.engine", "GROMACSEngine"),
        ("molpy.engine", "OpenMMEngine"),
        ("molpy.engine", "OpenMMSimulationConfig"),
        ("molpy.engine", "CP2KEngine"),
        ("molpy.adapter", "RDKitAdapter"),
        ("molpy.builder", "DPDistribution"),
        ("molpy.ff.typifier", "TLeapTypifier"),
        ("molpy", "data"),
        ("molpy", "keys"),
        ("molpy", "Cuboid"),
        ("molpy", "UnitRegistry"),
        ("molpy", "UnitPreset"),
        ("molpy", "NeighborList"),
        ("molpy", "Lbfgs"),
        ("molpy", "Conformer"),
        ("molpy", "SmartsPattern"),
        ("molpy", "ElementSelector"),
        ("molpy", "TrajectorySplitter"),
        ("molpy", "FrameCollection"),
        # S1: one core
        ("molpy", "Graph"),
        ("molpy", "CGBond"),
        ("molpy.core", "Graph"),
        ("molpy.core", "BondType"),
        ("molpy.ff.forcefield", "read_amber_prmtop_system"),
        ("molpy.ff.forcefield", "write_gromacs_top_system"),
        # S2: io per format
        ("molpy.io", "read_frame"),
        ("molpy.io", "write_frame"),
        ("molpy.io", "TrajectoryReader"),
        ("molpy.io", "read_smiles"),
        ("molpy.io", "write_smiles"),
        ("molpy.io", "read_mrec"),
        ("molpy.io", "write_mrec"),
        ("molpy.io", "read_frame_bytes"),
        ("molpy.io", "write_frame_bytes"),
        ("molpy.io", "read_forcefield_xml"),
        ("molpy.io", "write_forcefield_xml"),
        ("molpy.io", "read_opls_xml"),
        ("molpy.io", "read_lammps_cmap"),
        ("molpy.io", "write_lammps_cmap"),
        ("molpy.io", "write_bond_react_map"),
        ("molpy.io", "read_ac"),
        ("molpy.io", "read_prep"),
        ("molpy.io", "write_prep"),
        ("molpy.io", "read_chgcar"),
        ("molpy.io", "read_block_csv"),
        ("molpy.io", "write_block_csv"),
        ("molpy.io", "read_top"),
        ("molpy.io", "write_top"),
        ("molpy.io", "read_gromacs_top_ff"),
        ("molpy.io", "parse_lammps_log_text"),
        ("molpy.io", "SmilesIR"),
        ("molpy.io", "log"),
        ("molpy.io", "trajectory"),
        ("molpy.io", "lammps_bond_react"),
        ("molpy.io.smiles", "SmilesIR"),
        ("molpy.io.smiles", "CGSmilesIR"),
        ("molpy.io.smiles", "CgSmilesIr"),
        ("molpy.io.mrec", "pack"),
        ("molpy.io.mrec", "schema"),
        ("molpy.stream", "read_frame_bytes"),
        # S3: force field
        ("molpy.ff", "scale_lj"),
        ("molpy.ff.potential", "LJCut"),
        ("molpy.ff.potential", "kernel"),
        ("molpy.ff.potential", "TypedPotentials"),
        ("molpy.ff.ir", "Param"),
        ("molpy.ff.ir", "CategoryInfo"),
        ("molpy.ff.ir", "StyleInfo"),
        ("molpy.ff.ir", "unregister"),
        ("molpy.ff.ir", "Arity"),
        ("molpy.ff.forcefield", "Type"),
        ("molpy.ff.typifier", "Match"),
        ("molpy.ff.typifier", "OPLSAATypifier"),
        ("molpy.ff.typifier", "MMFF94Typifier"),
        ("molpy.ff.typifier", "MMFF94STypifier"),
        ("molpy.ff.params", "AMBER_SCEE"),
        ("molpy.ff.params", "AMBER_SCNB"),
        # S4: analysis, perception, geometry, dynamics
        ("molpy.perceive", "Perceive"),
        ("molpy", "Perceive"),
        ("molpy.op", "Fit"),
        ("molpy.optimize", "LBFGS"),
        ("molpy.optimize", "OptReport"),
        ("molpy.md", "MD"),
        ("molpy.md", "MDState"),
        ("molpy.compute", "MSD"),
        ("molpy.compute", "RDF"),
        ("molpy.compute", "VACF"),
        ("molpy.compute", "PMFTXY"),
        ("molpy.compute", "KMeans"),
        ("molpy.compute", "Pca2"),
        ("molpy.compute", "IRSpectrum"),
        ("molpy.compute", "BondOrder"),
        ("molpy.compute", "Dielectric"),
        ("molpy.compute", "Onsager"),
        ("molpy.compute", "Persist"),
        ("molpy.compute", "AngleDistribution"),
        ("molpy.compute", "DihedralDistribution"),
        ("molpy.compute", "DistanceDistribution"),
    ],
)
def test_a_removed_name_is_gone(module: str, name: str) -> None:
    assert not hasattr(importlib.import_module(module), name)


def test_the_box_and_the_trajectory_are_the_native_classes() -> None:
    assert molpy.Box is molrs.core.Box
    assert molpy.Trajectory is molrs.core.Trajectory
    assert molpy.MolGraph is molrs.core.MolGraph


def test_read_smiles_str_and_the_amber_system_reader_are_on_io() -> None:
    assert "read_smiles_str" in molpy.io.__all__
    assert "read_amber_prmtop_system" in molpy.io.__all__
    mol = molpy.io.read_smiles_str("CCO")
    assert isinstance(mol, molpy.Atomistic)
