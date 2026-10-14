"""One public path per symbol, mirroring the Rust owner.

``molrs`` holds the subsystems. The core data classes a user handles directly
are also on the root, as the same objects (``molrs.Frame is molrs.core.Frame``).
Every other class, function and constant has exactly one public path, and a
class's or function's ``__module__`` and ``__name__`` are that path, so
``repr``, pickle and the docs all name the owner module.

The walk is over ``dir()``, not only ``__all__``: a public module exports
exactly its ``__all__`` and nothing else, so a ``typing`` or stdlib name
imported for an annotation cannot leak into it as a second public spelling.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path
from types import ModuleType

import molrs
import pytest

SUBSYSTEMS = {
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
    "stream",
}

# The core data classes a user handles directly. Same set molpy promotes.
PROMOTED = {
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


def _public_names(module: ModuleType) -> set[str]:
    return {name for name in dir(module) if not name.startswith("_")}


def _modules() -> dict[str, ModuleType]:
    """Every public module, by its path."""
    out: dict[str, ModuleType] = {}

    def walk(module: ModuleType, path: str) -> None:
        out[path] = module
        for name in module.__all__:
            value = getattr(module, name)
            if inspect.ismodule(value):
                walk(value, f"{path}.{name}")

    walk(molrs, "molrs")
    return out


def _objects() -> dict[str, object]:
    """Every public class, function and constant, by its public path."""
    return {
        f"{path}.{name}": getattr(module, name)
        for path, module in _modules().items()
        for name in module.__all__
        if not inspect.ismodule(getattr(module, name))
    }


def _named(value: object) -> bool:
    """A class or a function: an object that states its own path."""
    return inspect.isclass(value) or inspect.isroutine(value)


def _canonical(paths: list[str]) -> list[str]:
    """Drop the root spelling of a promoted class. ``molrs.Frame`` is that
    class's second path; ``molrs.core.Frame`` is the owner path."""
    return [path for path in paths if path.rpartition(".")[0] != "molrs"]


def test_the_top_level_is_the_subsystems_and_the_handled_classes():
    assert set(molrs.__all__) == SUBSYSTEMS | PROMOTED
    assert _public_names(molrs) == SUBSYSTEMS | PROMOTED
    for name in PROMOTED:
        assert getattr(molrs, name) is getattr(molrs.core, name)


@pytest.mark.parametrize("path", sorted(_modules()))
def test_a_module_exports_exactly_its_all(path):
    module = _modules()[path]
    assert module.__name__ == path
    assert len(module.__all__) == len(set(module.__all__)), path
    assert _public_names(module) == set(module.__all__)


def test_every_class_and_function_has_one_public_path():
    paths: dict[int, list[str]] = {}
    for path, value in _objects().items():
        if _named(value):
            paths.setdefault(id(value), []).append(path)
    assert not [p for p in paths.values() if len(_canonical(p)) != 1]
    aliases = [
        path.rpartition(".")[2]
        for group in paths.values()
        for path in group
        if path.rpartition(".")[0] == "molrs"
    ]
    assert set(aliases) == PROMOTED


def test_a_class_or_function_names_its_public_path():
    wrong = [
        (path, f"{value.__module__}.{value.__name__}")
        for path, value in _objects().items()
        if _named(value)
        and path.rpartition(".")[0] != "molrs"
        and f"{value.__module__}.{value.__name__}" != path
    ]
    assert not wrong


def test_a_constant_is_a_value_not_a_second_door():
    """A public non-class, non-function name is plain data: it re-exports no
    class or function under a second name."""
    callables = [
        path
        for path, value in _objects().items()
        if not _named(value) and callable(value)
    ]
    assert not callables


# --- One shape per file-format factory --------------------------------------
#
# A factory that reads or writes a file format is either a function at the
# top of molrs.io (``read_<fmt>[_<what>]`` / ``write_<fmt>[_<what>]``) or a
# class of the format's own submodule (``molrs.io.<fmt>.<Fmt>Reader`` /
# ``<Fmt>Writer``). Nothing else may carry those names.


def _factory_name(name: str) -> bool:
    return name.startswith(("read_", "write_"))


def test_read_and_write_functions_live_at_the_top_of_io():
    stray = sorted(
        path
        for path, value in _objects().items()
        if callable(value)
        and _factory_name(path.rpartition(".")[2])
        and path.rpartition(".")[0] != "molrs.io"
    )
    assert not stray


def test_no_class_hides_a_read_or_write_factory():
    """A ``read_*`` / ``write_*`` static or class method is a second door
    onto a format: the door is a function of molrs.io."""
    doors = []
    for path, value in _objects().items():
        if not inspect.isclass(value):
            continue
        for name, attr in vars(value).items():
            if _factory_name(name) and isinstance(attr, (staticmethod, classmethod)):
                doors.append(f"{path}.{name}")
    assert not doors


def test_readers_and_writers_are_classes_of_a_format_submodule():
    stray = []
    for path, value in _objects().items():
        if not inspect.isclass(value) or not value.__name__.endswith(("Reader", "Writer")):
            continue
        owner = path.rpartition(".")[0]
        if not (owner.startswith("molrs.io.") and owner.count(".") == 2):
            stray.append(path)
    assert not stray


def test_the_top_of_io_is_functions_and_format_submodules():
    """A class belongs to one format, so it lives in that format's
    submodule; the top of molrs.io holds the read / write functions."""
    classes = [
        name for name in molrs.io.__all__ if inspect.isclass(getattr(molrs.io, name))
    ]
    assert not classes
    functions = [
        name for name in molrs.io.__all__ if inspect.isroutine(getattr(molrs.io, name))
    ]
    assert all(_factory_name(name) for name in functions), functions


def test_forcefield_is_the_data_model_and_stream_the_transport():
    assert not [n for n in molrs.ff.forcefield.__all__ if _factory_name(n)]
    assert set(molrs.stream.__all__) <= {"ControlCommand", "Publisher"}
    assert set(molrs.io.mrec.__all__) == {
        "ForceFieldSection",
        "MrecReader",
        "MrecWriter",
        "SequenceSchema",
        "pack_mrec_zip",
        "section_names",
        "validation",
    }


@pytest.mark.parametrize(
    "path",
    [
        "molrs.io.read_smiles_str",
        "molrs.io.write_smiles_str",
        "molrs.io.read_cgsmiles_str",
        "molrs.io.read_mrec_frame",
        "molrs.io.write_mrec_trajectory",
        "molrs.io.read_msgpack_frame_bytes",
        "molrs.io.write_msgpack_frame_bytes",
        "molrs.io.read_json_frame_str",
        "molrs.io.write_json_frame_str",
        "molrs.io.read_csv_block",
        "molrs.io.read_csv_block_str",
        "molrs.io.write_csv_block",
        "molrs.io.write_csv_block_str",
        "molrs.io.read_lammps_forcefield",
        "molrs.io.read_openmm_xml_forcefield",
        "molrs.io.write_openmm_xml_forcefield",
        "molrs.io.read_molrs_xml_forcefield",
        "molrs.io.write_molrs_xml_forcefield",
        "molrs.io.read_lammps_log_str",
        "molrs.io.write_lammps_bond_react_map",
        "molrs.io.write_lammps_bond_react_system",
        "molrs.io.read_lammps_data_coeffs",
        "molrs.io.read_lammps_data_coeffs_str",
        "molrs.io.write_lammps_data_coeffs_str",
        "molrs.io.pdb.PdbReader",
        "molrs.io.xyz.XyzReader",
        "molrs.io.gro.GroReader",
        "molrs.io.dcd.DcdReader",
        "molrs.io.trr.TrrReader",
        "molrs.io.xtc.XtcReader",
        "molrs.io.lammps.LammpsDumpReader",
        "molrs.io.mrec.pack_mrec_zip",
        "molrs.io.mrec.MrecReader",
        "molrs.io.mrec.MrecWriter",
        "molrs.io.smiles.SmilesIr",
        "molrs.io.cgsmiles.CgSmilesIr",
        "molrs.io.lammps.LammpsLog",
        "molrs.io.lammps.BondReactTemplate",
        "molrs.io.mrec.ForceFieldSection",
        "molrs.core.Frame",
        "molrs.core.Block",
        "molrs.core.Box",
        "molrs.core.MolGraph",
        "molrs.core.Atomistic",
        "molrs.core.Element",
        "molrs.core.NeighborList",
        "molrs.core.Quantity",
        "molrs.core.keys.PORTS",
        "molrs.core.keys.FRAG_ID",
        "molrs.core.keys.LAMMPS_UNITS",
        "molrs.core.schema.ColumnSpec",
        "molrs.core.constants.AMBER_COULOMB",
        "molrs.core.constants.AMBER_SCEE",
        "molrs.core.constants.BOHR_RADIUS",
    ],
)
def test_the_one_path_exists(path):
    assert path in _objects()


# --- Every door names its format --------------------------------------------

_DOOR = re.compile(r"^(read|write)_[a-z0-9]+(_[a-z0-9]+)*$")


def _io_doors() -> dict[str, object]:
    return {
        name: getattr(molrs.io, name)
        for name in molrs.io.__all__
        if inspect.isroutine(getattr(molrs.io, name))
    }


def _parameters(function: object) -> list[str]:
    try:
        return list(inspect.signature(function).parameters)
    except (TypeError, ValueError):  # a builtin without a text signature
        return []


def test_no_door_dispatches_on_a_format_argument():
    """A door that picks a format for the caller is format dispatch: every
    door names its format, so none takes a ``format`` / ``fmt`` / ``encoding
    name`` argument."""
    dispatching = sorted(
        name
        for name, door in _io_doors().items()
        if {"format", "fmt"} & set(_parameters(door))
    )
    assert not dispatching


def test_every_door_is_read_or_write_of_a_named_format():
    doors = _io_doors()
    assert doors
    assert all(_DOOR.match(name) for name in doors), sorted(doors)
    generic = {"read_frame", "write_frame", "read", "write", "read_file", "write_file"}
    assert not generic & set(doors)


def test_memory_doors_end_in_str_or_bytes():
    """Text in memory is ``_str``, bytes ``_bytes``: always the last word, and
    never spelled another way (``_text``, ``_string``, ``from_bytes``…)."""
    names = list(_io_doors())
    for word in ("str", "bytes"):
        assert all(n.endswith(f"_{word}") for n in names if f"_{word}_" in n), word
    for spelling in ("_text", "_string", "parse_", "format_", "_from_", "_to_"):
        assert not [n for n in names if spelling in n], spelling
    assert "read_smiles_str" in names and "read_msgpack_frame_bytes" in names


@pytest.mark.parametrize(
    ("door", "reader"),
    [
        ("read_pdb_trajectory", "molrs.io.pdb.PdbReader"),
        ("read_xyz_trajectory", "molrs.io.xyz.XyzReader"),
        ("read_gro_trajectory", "molrs.io.gro.GroReader"),
        ("read_lammps_dump_trajectory", "molrs.io.lammps.LammpsDumpReader"),
        ("read_dcd_trajectory", "molrs.io.dcd.DcdReader"),
        ("read_trr_trajectory", "molrs.io.trr.TrrReader"),
        ("read_xtc_trajectory", "molrs.io.xtc.XtcReader"),
    ],
)
def test_a_trajectory_door_is_its_formats_reader(door, reader):
    """``read_<fmt>_trajectory`` returns ``molrs.io.<fmt>.<Fmt>Reader``."""
    assert reader in _objects()
    assert getattr(molrs.io, door).__doc__
    module, _, name = reader.rpartition(".")
    assert name in getattr(molrs.io, module.rpartition(".")[2]).__all__


def test_a_reader_or_writer_is_named_after_its_module_with_word_cased_acronyms():
    """``io.<fmt>.<Fmt>…Reader``: the class name starts with its module's name
    cased as a word (``PdbReader``, ``LammpsDumpReader``), never an all-caps
    acronym (``PDBReader``, ``DCDTrajReader``)."""
    wrong = []
    for path, value in _objects().items():
        if not inspect.isclass(value) or not value.__name__.endswith(("Reader", "Writer")):
            continue
        module = path.rpartition(".")[0].rpartition(".")[2]
        word = "".join(part.capitalize() for part in module.split("_"))
        if not value.__name__.startswith(word) or re.search(r"[A-Z]{2}", value.__name__):
            wrong.append(path)
    assert not wrong



# Every subpackage cases acronyms as words. Kept as their owners write them:
# numpy's ``DType``, and ``HBond`` -- H is the element, not an acronym.
_CASING_KEPT_PREFIXES = ("DType", "HBond")
_ALL_CAPS_ACRONYM = re.compile(r"[A-Z]{2,}[a-z]|[A-Z]{3,}|[a-z0-9][A-Z]{2,}$|^[A-Z]{2,}$")


def _without_kept_prefix(name: str) -> str:
    for kept in _CASING_KEPT_PREFIXES:
        if name.startswith(kept):
            return name[len(kept) :]
    return name


def test_class_names_case_acronyms_as_words():
    """``PdbReader``, ``SmilesIr``, ``Mmff94Typifier``, ``Rdf``, ``MdState``,
    ``Lbfgs``: an acronym is cased as a word, never ``PDB``, ``IR``, ``MSD`` or
    ``MD``, in every subpackage."""
    wrong = sorted(
        path
        for path, value in _objects().items()
        if inspect.isclass(value)
        and _ALL_CAPS_ACRONYM.search(_without_kept_prefix(value.__name__))
    )
    assert not wrong


def test_function_names_are_snake_case():
    """A public function is ``snake_case``: no capital letter, so no acronym
    can be spelled in capitals (``rdf``, never ``RDF``)."""
    wrong = sorted(
        path
        for path, value in _objects().items()
        if inspect.isroutine(value) and value.__name__ != value.__name__.lower()
    )
    assert not wrong


def test_compute_is_classes_only():
    """``molrs.compute`` mirrors ``molrs::compute``, which has no public free
    function: every export is a class (or the ``Compute`` protocol), and the
    retired ``molrs.signal`` module is gone."""
    assert not [
        name
        for name in molrs.compute.__all__
        if not inspect.isclass(getattr(molrs.compute, name))
    ]
    assert not hasattr(molrs, "signal")


def test_smarts_is_perceptions_wholly():
    """SMARTS is generated by ``SmartsPattern.from_environment`` and written by
    ``str(pattern)``; ``molrs.io`` holds no SMARTS door."""
    assert callable(molrs.perceive.SmartsPattern.from_environment)
    assert not [n for n in molrs.io.__all__ if "smarts" in n]


def test_find_matches_has_no_mapped_shortcut():
    """``SmartsMatch.mapping`` is the one door onto the atom-map projection."""
    assert "mapped" not in molrs.perceive.SmartsPattern.find_matches.__text_signature__


def test_core_constants_mirror_rust_in_full():
    """``molrs.core.constants`` is ``molrs::core::constants``, name for name."""
    rust = Path(__file__).parents[2] / "molrs" / "src" / "core" / "constants.rs"
    source = rust.read_text(encoding="utf-8")
    names = set(re.findall(r"^pub const ([A-Z0-9_]+):", source, re.MULTILINE)) - {"ALL"}
    assert names
    assert set(molrs.core.constants.__all__) == names


def test_the_version_is_the_package_version():
    from importlib.metadata import version

    assert molrs.__version__ == version("molcrafts-molrs")


def test_unit_conversions_come_from_the_unit_registry():
    units = molrs.core.UnitRegistry()
    assert units.factor("kcal", "kJ") == 4.184
    assert units.factor("nm", "angstrom") == 10.0
    assert units.factor("angstrom", "nm") == 0.1
    assert units.factor("bohr", "angstrom") == pytest.approx(0.529177210903, rel=1e-15)
    names = set(molrs.core.constants.__all__)
    assert not {n for n in names if n.endswith(("_PER_NM", "_PER_BOHR", "_PER_KCAL", "_PER_CM3"))}


def test_scripts_convert_units_through_the_registry():
    """No engine-check script spells a conversion factor."""
    root = Path(__file__).parents[2] / "scripts"
    factor = re.compile(
        r"(?<![\w.])4\.184(?![\w])"
        # π/180, 180/π, MMFF's mdyne·Å → kcal/mol, k_B in kcal/(mol·K) by hand
        r"|(?<![\w.])(0\.0174532|57\.29577|143\.9325|0\.001987)"
        r"|(?<![\w.])1\.987[\d_]*e-3"
        # a degree ↔ radian conversion: `/ 180`, `* 180`, `180 /`, `180.0 *`
        r"|[/*]\s*180(?![\d])|(?<![\w.])180(\.\d*)?\s*[/*]"
    )
    offenders = []
    for path in sorted(root.iterdir()):
        if path.suffix not in {".py", ".sh"}:
            continue
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            code = line.split("#", 1)[0]
            if factor.search(code):
                offenders.append(f"{path.name}:{n}: {line.strip()}")
    assert not offenders, "\n".join(offenders)


def test_ff_layers_hold_the_registry_compiler_and_fragment_table():
    for name in (
        "register_category",
        "register_style",
        "register_engine_form",
        "unregister_style",
        "styles",
        "categories",
        "evaluate",
        "StyleDeclaration",
    ):
        assert hasattr(molrs.ff.style_registry, name), name
    assert molrs.ff.compile.PotentialCompiler
    assert molrs.ff.compile.ExplicitTerms
    assert "c2c1im" in molrs.ff.clpol_scaling.fragment_table()
