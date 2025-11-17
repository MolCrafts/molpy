"""
Error types for polymer assembly.

All errors provide clear, actionable messages for debugging.
"""


class AssemblyError(Exception):
    """Base exception for polymer assembly errors."""

    pass


class SequenceError(AssemblyError):
    """Invalid sequence (e.g., too short, unknown labels)."""

    pass


class AmbiguousPortsError(AssemblyError):
    """Cannot uniquely determine which ports to connect."""

    pass


class MissingConnectorRule(AssemblyError):
    """TableConnector has no rule for a given monomer pair."""

    pass


class NoCompatiblePortsError(AssemblyError):
    """No compatible port pair found between two monomers."""

    pass


class BondKindConflictError(AssemblyError):
    """Conflicting bond kind specifications."""

    pass


class PortReuseError(AssemblyError):
    """Attempt to reuse a consumed port (multiplicity = 0)."""

    pass


class GeometryError(Exception):
    """Base exception for geometry-related errors."""

    pass


class OrientationUnavailableError(GeometryError):
    """Cannot infer orientation for a port (no neighbors, no role info)."""

    pass


class PositionMissingError(GeometryError):
    """Entity is missing required 3D position data."""

    pass
