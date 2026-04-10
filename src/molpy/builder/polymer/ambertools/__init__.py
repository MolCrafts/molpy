"""
AmberTools polymer builder module.

Public API:
    - AmberPolymerBuilder: Main entry point for polymer building
    - AmberBuildResult: Result dataclass containing Frame, ForceField, and paths
"""

from .amber_builder import AmberPolymerBuilder
from .types import AmberBuildResult

__all__ = [
    "AmberPolymerBuilder",
    "AmberBuildResult",
]
