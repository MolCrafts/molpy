"""Unit tests for core polymer builder module: exceptions, types, and builder logic."""

import pytest

from molpy.builder.polymer.core import (
    AssemblyError,
    AmbiguousPortsError,
    BondKindConflictError,
    GeometryError,
    MissingConnectorRule,
    NoCompatiblePortsError,
    OrientationUnavailableError,
    PolymerBuildResult,
    PolymerBuilder,
    PortReuseError,
    PositionMissingError,
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

    def test_missing_connector_rule_is_assembly_error(self):
        assert issubclass(MissingConnectorRule, AssemblyError)

    def test_no_compatible_ports_is_assembly_error(self):
        assert issubclass(NoCompatiblePortsError, AssemblyError)

    def test_bond_kind_conflict_is_assembly_error(self):
        assert issubclass(BondKindConflictError, AssemblyError)

    def test_port_reuse_is_assembly_error(self):
        assert issubclass(PortReuseError, AssemblyError)

    def test_geometry_error_is_assembly_error(self):
        """GeometryError should be catchable via AssemblyError."""
        assert issubclass(GeometryError, AssemblyError)

    def test_orientation_unavailable_is_geometry_error(self):
        assert issubclass(OrientationUnavailableError, GeometryError)

    def test_position_missing_is_geometry_error(self):
        assert issubclass(PositionMissingError, GeometryError)

    def test_catch_all_via_assembly_error(self):
        """All polymer exceptions should be catchable via AssemblyError."""
        exceptions = [
            SequenceError("test"),
            AmbiguousPortsError("test"),
            MissingConnectorRule("test"),
            NoCompatiblePortsError("test"),
            BondKindConflictError("test"),
            PortReuseError("test"),
            GeometryError("test"),
            OrientationUnavailableError("test"),
            PositionMissingError("test"),
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
