"""
Error types for polymer assembly.

All errors provide clear, actionable messages for debugging.
"""


class AssemblyError(Exception):
    """Base exception for polymer assembly errors."""


class SequenceError(AssemblyError):
    """Invalid sequence (e.g., too short, unknown labels)."""


class AmbiguousPortsError(AssemblyError):
    """Cannot uniquely determine which ports to connect."""


class NoCompatiblePortsError(AssemblyError):
    """No compatible port pair found between two monomers."""
