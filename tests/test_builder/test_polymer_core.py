"""Unit tests for core polymer builder module: exceptions, types, and builder logic."""

import pytest

from molpy.builder.polymer.core import (
    AssemblyError,
    AmbiguousPortsError,
    NoCompatiblePortsError,
    PolymerBuildResult,
    PolymerBuilder,
    SequenceError,
    TypifierProtocol,
)
from molpy.core.atomistic import Atomistic


# ---- Exception Hierarchy Tests ----


class TestExceptionHierarchy:
    def test_assembly_error_is_exception(self):
        assert issubclass(AssemblyError, Exception)

    def test_sequence_error_is_assembly_error(self):
        assert issubclass(SequenceError, AssemblyError)

    def test_ambiguous_ports_is_assembly_error(self):
        assert issubclass(AmbiguousPortsError, AssemblyError)

    def test_no_compatible_ports_is_assembly_error(self):
        assert issubclass(NoCompatiblePortsError, AssemblyError)

    def test_catch_all_via_assembly_error(self):
        """All polymer exceptions should be catchable via AssemblyError."""
        exceptions = [
            SequenceError("test"),
            AmbiguousPortsError("test"),
            NoCompatiblePortsError("test"),
        ]
        for exc in exceptions:
            with pytest.raises(AssemblyError):
                raise exc


# ---- Type Definition Tests ----


class TestPolymerBuildResult:
    def test_defaults(self):
        polymer = Atomistic()
        result = PolymerBuildResult(polymer=polymer)
        assert result.polymer is polymer
        assert result.connection_history == []
        assert result.total_steps == 0


# ---- PolymerBuilder Init Tests ----


class TestPolymerBuilderInit:
    def test_requires_connector_or_reacter(self):
        with pytest.raises(TypeError, match="One of"):
            PolymerBuilder(library={"A": Atomistic()})

    def test_cannot_provide_both(self):
        from unittest.mock import MagicMock

        with pytest.raises(TypeError, match="not both"):
            PolymerBuilder(
                library={"A": Atomistic()},
                connector=MagicMock(),
                reacter=MagicMock(),
            )

    def test_reacter_shortcut(self):
        from unittest.mock import MagicMock

        builder = PolymerBuilder(library={"A": Atomistic()}, reacter=MagicMock())
        assert builder.connector is not None

    def test_connector_direct(self):
        from unittest.mock import MagicMock

        connector = MagicMock()
        builder = PolymerBuilder(library={"A": Atomistic()}, connector=connector)
        assert builder.connector is connector

    def test_placer_stored(self):
        from unittest.mock import MagicMock

        placer = MagicMock()
        builder = PolymerBuilder(
            library={"A": Atomistic()},
            reacter=MagicMock(),
            placer=placer,
        )
        assert builder.placer is placer

    def test_typifier_stored(self):
        from unittest.mock import MagicMock

        typifier = MagicMock(spec=TypifierProtocol)
        builder = PolymerBuilder(
            library={"A": Atomistic()},
            reacter=MagicMock(),
            typifier=typifier,
        )
        assert builder.typifier is typifier


# ---- Dead-Code Consolidation Regression Tests (builder-reacter-01) ----


class TestPolymerModuleConsolidation:
    """Regression tests: single exception hierarchy + live core re-exports.

    Guards against the duplicated exception hierarchies (core.py vs errors.py)
    and the stale polymer_builder.py fork shadowing core.PolymerBuilder.
    """

    def test_exported_polymer_builder_is_core_implementation(self):
        """Public PolymerBuilder export must be the live core implementation."""
        from molpy.builder.polymer import PolymerBuilder, core

        assert PolymerBuilder is core.PolymerBuilder

    def test_exported_build_result_is_core_build_result(self):
        """Public PolymerBuildResult export must be the core dataclass."""
        from molpy.builder.polymer import PolymerBuildResult, core

        assert PolymerBuildResult is core.PolymerBuildResult

    def test_connector_exception_catchable_via_public_assembly_error(self):
        """errors.AmbiguousPortsError (raised by connectors) must be caught
        via core.AssemblyError -- one hierarchy, not two."""
        from molpy.builder.polymer.core import AssemblyError
        from molpy.builder.polymer.errors import AmbiguousPortsError

        with pytest.raises(AssemblyError):
            raise AmbiguousPortsError("ambiguous")


# ---- Validation Tests ----


class TestPolymerBuilderValidation:
    def test_empty_graph_raises(self):
        from unittest.mock import MagicMock

        builder = PolymerBuilder(library={"A": Atomistic()}, reacter=MagicMock())
        with pytest.raises(ValueError, match="empty"):
            builder.build("{}")

    def test_missing_label_raises(self):
        from unittest.mock import MagicMock

        builder = PolymerBuilder(library={"A": Atomistic()}, reacter=MagicMock())
        with pytest.raises(SequenceError, match="not found in library"):
            builder.build("{[#UNKNOWN]|3}")


# ---- Build Behavior Tests (RDKit-dependent, absorbed from the removed
# ---- test_polymer_builder.py fork suite) ----


def _make_reacter():
    from molpy.reacter import Reacter, form_single_bond

    return Reacter(
        name="test",
        anchor_selector_left=lambda a, port_atom: port_atom,
        anchor_selector_right=lambda a, port_atom: port_atom,
        leaving_selector_left=lambda a, anchor: [],
        leaving_selector_right=lambda a, anchor: [],
        bond_former=form_single_bond,
    )


def _make_monomer(bigsmiles: str) -> Atomistic:
    """Build a 3D test monomer from a BigSMILES string (requires RDKit)."""
    from molpy.adapter import RDKitAdapter
    from molpy.parser.smiles import bigsmilesir_to_monomer, parse_bigsmiles

    ir = parse_bigsmiles(bigsmiles)
    monomer = bigsmilesir_to_monomer(ir)

    monomer = RDKitAdapter(internal=monomer).generate_3d(
        add_hydrogens=True, optimize=False
    )
    monomer = monomer.get_topo(gen_angle=True, gen_dihe=True)

    for idx, atom in enumerate(monomer.atoms):
        atom["id"] = idx + 1

    return monomer


@pytest.mark.external
class TestPolymerBuilderBehavior:
    """End-to-end build behavior of the live core.PolymerBuilder."""

    @pytest.fixture(autouse=True)
    def _require_rdkit(self):
        pytest.importorskip("rdkit", reason="RDKit is not installed")

    def _builder(self, library, port_map):
        from molpy.builder.polymer import Connector

        connector = Connector(reacter=_make_reacter(), port_map=port_map)
        return PolymerBuilder(library=library, connector=connector)

    def test_build_linear_chain(self):
        library = {"A": _make_monomer("{[<]CC[>]}")}
        builder = self._builder(library, {("A", "A"): (">", "<")})

        result = builder.build("{[#A][#A][#A]}")

        assert result.polymer is not None
        assert result.total_steps == 2  # Two connections for 3 monomers
        assert len(result.connection_history) == 2

    def test_build_with_repeat_operator(self):
        library = {"A": _make_monomer("{[<]CC[>]}")}
        builder = self._builder(library, {("A", "A"): (">", "<")})

        result = builder.build("{[#A]|3}")

        assert result.polymer is not None
        assert result.total_steps == 2  # Two connections for 3 monomers

    def test_build_branched_structure(self):
        # A: linear monomer with $ ports; B: 3-arm branch point with 3 $ ports
        library = {
            "A": _make_monomer("{[][$]CC[$][]}"),
            "B": _make_monomer("{[]C(C[$])(C[$])C[$][]}"),
        }
        rules = {(li, r): ("$", "$") for li in library for r in library}
        builder = self._builder(library, rules)

        result = builder.build("{[#A][#B]([#A])[#A]}")

        assert result.polymer is not None
        assert result.total_steps == 3  # Three connections for 4 monomers

    def test_build_cyclic_structure(self):
        # Ring closure consumes a second port pair per monomer, so the
        # monomers need multi-connectable $ ports.
        library = {"A": _make_monomer("{[][$]CC[$][]}")}
        builder = self._builder(library, {("A", "A"): ("$", "$")})

        result = builder.build("{[#A]1[#A][#A]1}")

        assert result.polymer is not None
        assert result.total_steps == 3  # Three connections for 3 monomers in ring

    def test_connection_history_tracking(self):
        library = {"A": _make_monomer("{[<]CC[>]}")}
        builder = self._builder(library, {("A", "A"): (">", "<")})

        result = builder.build("{[#A][#A]}")

        assert len(result.connection_history) == 1
        rxn_result = result.connection_history[0]
        assert hasattr(rxn_result, "product")
        assert hasattr(rxn_result, "reaction_name")
        assert rxn_result.reaction_name == "test"
