"""Architecture gates for the molpy.core -> molrs ownership cutover."""

from __future__ import annotations

import ast
import importlib
import json
from pathlib import Path

import molrs
import molpy
import pytest

from molpy.core import atomistic, cg, entity, forcefield

ROOT = Path(__file__).parents[2]
CORE = ROOT / "src" / "molpy" / "core"
MANIFEST = json.loads(
    (ROOT / ".claude" / "specs" / "molrs-core-cutover.manifest.json").read_text(
        encoding="utf-8"
    )
)


def test_identity_reexports_are_canonical_molrs_objects():
    modules = {
        "entity": entity,
        "atomistic": atomistic,
        "cg": cg,
        "forcefield": forcefield,
    }
    canonical_modules = {
        "entity": molrs.views,
        "atomistic": molrs.views,
        "cg": molrs.views,
        "forcefield": molrs,
    }
    for module_name, names in MANIFEST["identity_reexports"].items():
        if module_name == "ops.scale_lj":
            module = importlib.import_module("molpy.core.ops.scale_lj")
            for name in names:
                assert getattr(module, name) is getattr(molrs, name)
            continue
        for name in names:
            if name == "AtomisticForcefield":
                assert forcefield.AtomisticForcefield is molrs.ForceField
                continue
            assert getattr(modules[module_name], name) is getattr(
                canonical_modules[module_name], name
            )


def test_sugar_types_are_native_subclasses_without_inner_facades():
    from molpy.core.box import Box
    from molpy.core.region import AndRegion, BoxRegion, SphereRegion
    from molpy.core.trajectory import Trajectory
    from molpy.core.unit import UnitSystem

    pairs = [
        (atomistic.Atomistic, molrs.Atomistic),
        (cg.CoarseGrain, molrs.CoarseGrain),
        (Box, molrs.Box),
        (Trajectory, molrs.Trajectory),
        (UnitSystem, molrs.UnitRegistry),
        (BoxRegion, molrs.Cuboid),
        (SphereRegion, molrs.Sphere),
        (AndRegion, molrs.Region),
    ]
    for sugar, native in pairs:
        assert issubclass(sugar, native)
        assert "_inner" not in vars(sugar)


def test_manifest_covers_public_top_level_definitions():
    allowed = {
        (module, name)
        for category in ("identity_reexports", "python_sugar")
        for module, names in MANIFEST[category].items()
        for name in names
    }
    # Private helpers/mixins are implementation details inside an already
    # classified sugar module; every public top-level class/function is listed.
    for path in CORE.rglob("*.py"):
        relative = path.relative_to(CORE)
        module = ".".join(relative.with_suffix("").parts)
        if module in {"__init__", "ops.__init__", *MANIFEST["excluded_molpy_owned"]}:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        public = {
            node.name
            for node in tree.body
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
            and not node.name.startswith("_")
        }
        missing = sorted(name for name in public if (module, name) not in allowed)
        assert not missing, f"{module} has unclassified public definitions: {missing}"


def test_removed_duplicate_kernels_stay_removed():
    for relative in MANIFEST["removed"]:
        base = ROOT / "src" / "molpy"
        assert not (base / relative).exists(), relative

    forbidden_attributes = {"to_molrs", "from_molrs"}
    for path in CORE.rglob("*.py"):
        if path.name in {"config.py", "logger.py", "script.py"}:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                assert node.attr not in forbidden_attributes, f"{path}: {node.attr}"
                assert not (
                    node.attr == "linalg"
                    and isinstance(node.value, ast.Name)
                    and node.value.id in {"np", "numpy"}
                ), f"{path}: NumPy numerical kernel"


def test_dead_type_bucket_surface_is_not_exported():
    assert not hasattr(molpy, "TypeBucket")
    assert "TypeBucket" not in molpy.__all__
    assert "TypeBucket" not in forcefield.__all__


def test_frame_block_and_element_are_molrs_only_exports():
    for package in (molpy, importlib.import_module("molpy.core")):
        for name in ("Frame", "Block", "Element", "ElementData"):
            assert not hasattr(package, name)
            assert name not in package.__all__


def test_removed_core_storage_and_element_imports_fail():
    for statement in (
        "from molpy import Frame",
        "from molpy import Block",
        "from molpy.core import Frame",
        "from molpy.core import Block",
        "from molpy import Element",
        "from molpy.core import Element",
    ):
        with pytest.raises(ImportError):
            exec(statement, {})

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("molpy.core.frame")
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("molpy.core.element")

    assert not hasattr(molrs, "ElementData")


def test_frame_has_only_simbox_and_exact_dtype_meta_surface():
    frame = molrs.Frame()
    assert hasattr(frame, "simbox")
    assert hasattr(frame, "meta")
    assert not hasattr(frame, "box")
    assert not hasattr(frame, "metadata")


def test_production_never_reads_removed_frame_surface():
    forbidden = {"box", "metadata"}
    source_root = ROOT / "src" / "molpy"
    for path in source_root.rglob("*.py"):
        if path.name == "version.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Attribute) or node.attr not in forbidden:
                continue
            if node.attr == "metadata":
                # ForceField and result metadata remain independent public APIs.
                if isinstance(node.value, ast.Name) and node.value.id in {
                    "ff",
                    "forcefield",
                    "result",
                    "r",
                }:
                    continue
            if node.attr == "box":
                # Scientific objects other than Frame may own a box.
                if not (
                    isinstance(node.value, ast.Name)
                    and node.value.id.startswith("frame")
                ):
                    continue
            pytest.fail(f"removed Frame surface in {path}: .{node.attr}")
