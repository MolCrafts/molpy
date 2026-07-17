"""The typifier contract: ``MolGraph -> MolGraph``, and only ``match`` differs.

Three things are pinned here, and each of them is a design claim the package used
to violate:

1. **The pipeline does not assume an atomistic graph.** ``Typifier`` is generic
   over the graph; an ``Atomistic`` and a ``CoarseGrain`` are both ``molrs.Graph``
   leaves. A 20-line coarse-grained typifier must plug in without ``base.py``
   changing a character.
2. **The flow is written once.** ``typify`` is concrete; a subclass overrides
   ``match`` and nothing else.
3. **Every typifier is named after a force field or a tool.** There is no
   ``ParamTypifier``, no ``ElementTypifier``, no ``RegionTypifier``: "assign
   parameters to a graph whose types are known" is a component, not a typifier.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import pkgutil
import textwrap
from pathlib import Path

import pytest
import molrs

import molpy as mp
import molpy.typifier as typifier_pkg
from molpy.core.atomistic import Atomistic
from molpy.core.cg import CoarseGrain
from molpy.typifier import (
    AmberToolsTypifier,
    ClpTypifier,
    ForceFieldParams,
    Match,
    Typifier,
)
from molpy.typifier.affected_region import AffectedRegion
from molpy.typifier.cache import RetypeCache

TYPIFIER_DIR = Path(typifier_pkg.__file__).parent

#: Names that would mean the generic layer had learned about one particular
#: force field's decomposition of a molecule into terms.
_FORCE_FIELD_NAMES = frozenset(
    {"Bond", "Angle", "Dihedral", "Improper", "ForceField", "BondType"}
)


class _BeadTypifier(Typifier[CoarseGrain]):
    """Twenty lines of coarse-grained typifier. It writes ``match`` and stops."""

    def match(self, graph: CoarseGrain) -> Match:
        return Match(nodes=tuple({"type": "B"} for _ in graph.nodes))


def _cg(n: int = 3) -> CoarseGrain:
    cg = CoarseGrain()
    beads = [cg.def_bead(x=float(i), y=0.0, z=0.0) for i in range(n)]
    for a, b in zip(beads, beads[1:], strict=False):
        cg.def_cgbond(a, b)
    return cg


def _chain(n: int = 5) -> Atomistic:
    s = mp.Atomistic()
    atoms = [s.def_atom(element="C", x=1.5 * i, y=0.0, z=0.0) for i in range(n)]
    for a, b in zip(atoms, atoms[1:], strict=False):
        s.def_bond(a, b)
    return s


class TestOnePipeline:
    def test_match_is_the_only_abstract_step(self):
        assert Typifier.__abstractmethods__ == frozenset({"match"})

    def test_the_base_cannot_be_instantiated(self):
        with pytest.raises(TypeError, match="abstract"):
            Typifier()  # type: ignore[abstract]

    @pytest.mark.parametrize("cls", [ClpTypifier, AmberToolsTypifier, _BeadTypifier])
    def test_no_typifier_reimplements_the_flow(self, cls):
        assert "typify" not in cls.__dict__
        assert "match" in cls.__dict__

    def test_the_abstract_error_comes_from_abcmeta_not_a_hand_written_guard(self):
        source = (TYPIFIER_DIR / "base.py").read_text(encoding="utf-8")
        assert "does not know how to type" not in source


class TestGenericOverTheGraph:
    def test_the_pipeline_names_no_force_field_decomposition(self):
        """``base.py`` may say ``Link``; it may not say ``Bond``."""
        tree = ast.parse((TYPIFIER_DIR / "base.py").read_text(encoding="utf-8"))
        names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        names |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
        for alias in ast.walk(tree):
            if isinstance(alias, ast.alias):
                names.add(alias.name.rsplit(".", 1)[-1])

        assert not (names & _FORCE_FIELD_NAMES), names & _FORCE_FIELD_NAMES

    def test_the_link_to_forcefield_type_mapping_lives_in_exactly_one_place(self):
        holders = [
            path.name
            for path in TYPIFIER_DIR.glob("*.py")
            if "_FF_TYPE_OF" in path.read_text(encoding="utf-8")
        ]
        assert holders == ["forcefield.py"]

    def test_a_typifier_never_guesses_whether_its_graph_is_a_fragment(self):
        """Truncation is provenance, not a readable property (iron law 5).

        A radical is a perfectly good molecule, and a connectivity graph with no
        bond orders looks under-coordinated everywhere. So ``typify`` takes the
        graph at face value and contains no hydrogen perception at all.
        """
        tree = ast.parse(textwrap.dedent(inspect.getsource(Typifier.typify)))
        assert not [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "find_hydrogens"
        ]
        assert not [n for n in ast.walk(tree) if isinstance(n, ast.If | ast.Try)]

    def test_the_party_that_cut_the_graph_completes_it_unconditionally(self):
        """A region is a cut, so it perceives missing hydrogens unconditionally."""
        from molpy.typifier.region import RegionTypes

        tree = ast.parse(textwrap.dedent(inspect.getsource(RegionTypes.of.__func__)))
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "find_hydrogens"
        ]
        assert len(calls) == 1
        assert not [n for n in ast.walk(tree) if isinstance(n, ast.If | ast.Try)]


class TestCoarseGrainNeedsNoContractChange:
    """The proof: a coarse-grained typifier plugs into the unchanged pipeline."""

    def test_a_bead_typifier_is_a_typifier(self):
        assert isinstance(_BeadTypifier(), Typifier)

    def test_it_types_a_coarse_grained_graph(self):
        cg = _cg()

        typed = _BeadTypifier().typify(cg)

        assert isinstance(typed, CoarseGrain)
        assert all(bead["type"] == "B" for bead in typed.beads)
        assert all(bead.get("type") is None for bead in cg.beads)

    def test_a_bead_exposes_no_chemical_perception_facade(self):
        cg = _cg()
        assert not hasattr(cg, "complete_valence")

    def test_the_retype_cache_accepts_it(self):
        assert RetypeCache(_BeadTypifier()) is not None

    def test_the_assembler_accepts_it(self):
        from molpy.builder.assembly import GraphAssembler

        assembler = GraphAssembler(
            mp.Reaction("[N:1].[O:2]>>[N:1][O:2]"), typifier=_BeadTypifier(), reach=1
        )
        assert assembler is not None


class _ElementTypifier(Typifier[Atomistic]):
    def match(self, graph: Atomistic) -> Match:
        return Match(nodes=tuple({"type": f"t_{a['element']}"} for a in graph.atoms))


class TestTypifyReturnsANewGraph:
    def test_the_input_is_untouched_and_the_caps_fall_off(self):
        source = _chain()

        typed = _ElementTypifier().typify(source)

        assert typed is not source
        assert len(list(typed.atoms)) == len(list(source.atoms))
        assert all(atom.get("type") is None for atom in source.atoms)
        assert all(atom["type"] == "t_C" for atom in typed.atoms)


class TestNamingAndSurface:
    def test_every_exported_typifier_inherits_the_molrs_base(self):
        for name in typifier_pkg.__all__:
            if name.endswith("Typifier") and name != "Typifier":
                assert issubclass(getattr(typifier_pkg, name), molrs.Typifier)

    def test_every_exported_typifier_is_named_after_a_forcefield_or_a_tool(self):
        exported = {
            name
            for name in typifier_pkg.__all__
            if name.endswith("Typifier") and name != "Typifier"
        }
        assert exported == {
            "ClpTypifier",
            "AmberToolsTypifier",
            "OPLSAATypifier",
            "MMFFTypifier",
        }

    @pytest.mark.parametrize(
        "banned",
        [
            "ParamTypifier",
            "ElementTypifier",
            "AtomTypifier",
            "RegionTypifier",
            "ForceFieldTypifier",
            "TypeScope",
            "PairTypifier",
        ],
    )
    def test_the_invented_abstractions_are_gone(self, banned):
        assert not hasattr(typifier_pkg, banned)

    def test_force_field_params_is_a_component_not_a_typifier(self):
        assert not issubclass(ForceFieldParams, Typifier)
        assert "ForceFieldParams" in typifier_pkg.__all__

    @pytest.mark.parametrize(
        "dead", ["bond", "angle", "dihedral", "pair", "mmff", "atomistic", "scope"]
    )
    def test_the_dead_modules_are_gone(self, dead):
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(f"molpy.typifier.{dead}")

    def test_no_symbol_is_defined_twice_in_the_package(self):
        seen: dict[str, str] = {}
        clashes: list[str] = []
        for module in pkgutil.iter_modules([str(TYPIFIER_DIR)]):
            path = TYPIFIER_DIR / f"{module.name}.py"
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in tree.body:
                if not isinstance(node, ast.ClassDef | ast.FunctionDef):
                    continue
                if node.name in seen:
                    clashes.append(f"{node.name}: {seen[node.name]} + {module.name}")
                seen[node.name] = module.name
        assert not clashes, clashes


class TestRegionTypingIsNotAKindOfTypifier:
    def test_no_typifier_carries_region_methods(self):
        for cls in (Typifier, ClpTypifier, AmberToolsTypifier, _BeadTypifier):
            assert not hasattr(cls, "typify_region")
            assert not hasattr(cls, "retype_region")
            assert not hasattr(cls, "relaxed")
            assert not hasattr(cls, "scope")

    def test_a_region_is_just_a_graph_the_typifier_types(self):
        from molpy.typifier.region import RegionTypes

        chain = _chain(9)
        seed = list(chain.atoms)[4]
        region = AffectedRegion.around(chain, [seed], reach=2)

        snapshot = RegionTypes.of(region, _ElementTypifier())

        assert len(snapshot.atoms) == len(region.interior)
