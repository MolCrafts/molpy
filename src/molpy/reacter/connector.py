"""
Connector interface for integrating Reacter with the linear polymer builder.

This module provides connectors that use Reacter for polymer assembly,
allowing flexible reaction specification with default and specialized reactors.
"""

from molpy.core.atomistic import Atomistic
from molpy.core.entity import Entity

from .base import Reacter, ReactionResult


class MonomerLinker:
    """
    Connector adapter that manages Reacter instances for polymer assembly.

    This connector allows specifying a default reaction for most connections,
    with the ability to override specific monomer pair connections with
    specialized reacters.

    **Port Selection Philosophy:**
    This connector does NOT handle port selection. The caller must explicitly
    provide port_L and port_R when calling connect(). Ports must be marked
    directly on atoms using the "port" or "ports" attribute.
    Port selection should be handled by the higher-level builder logic.

    Example:
        >>> from molpy.reacter import Reacter, MonomerLinker
        >>> from molpy import Atomistic
        >>>
        >>> # Mark ports on atoms
        >>> atom_a["port"] = "1"
        >>> atom_b["port"] = "2"
        >>>
        >>> # Default reaction for most connections
        >>> default_reacter = Reacter(...)
        >>>
        >>> # Special reaction for A-B connection
        >>> special_reacter = Reacter(...)
        >>>
        >>> connector = MonomerLinker(
        ...     default_reaction=default_reacter,
        ...     specialized_reactions={('A', 'B'): special_reacter}
        ... )
        >>>
        >>> # Explicit port specification required
        >>> product = connector.connect(
        ...     left=struct_a, right=struct_b,
        ...     left_type='A', right_type='B',
        ...     port_L='1', port_R='2'  # REQUIRED
        ... )
    """

    def __init__(
        self,
        default_reaction: Reacter,
        specialized_reactions: dict[tuple[str, str], Reacter] | None = None,
    ):
        """
        Initialize connector with default and specialized reacters.

        Args:
            default_reaction: Default Reacter used for most connections
            specialized_reactions: Dict mapping (left_type, right_type) -> specialized Reacter
        """
        self.default_reaction = default_reaction
        self.specialized_reactions = specialized_reactions or {}
        self._history: list[ReactionResult] = []

    def connect(
        self,
        left: Atomistic,
        right: Atomistic,
        port_L: str,
        port_R: str,
        left_type: str | None = None,
        right_type: str | None = None,
    ) -> Atomistic:
        """
        Connect two Atomistic structures using appropriate reacter.

        **IMPORTANT: port_L and port_R must be explicitly specified.**
        Ports must be marked on atoms using the "port" or "ports" attribute.
        No automatic port selection or fallback is performed.

        This connector is responsible for locating **port atoms** by name.
        The underlying :class:`Reacter` only operates on **anchors** and
        receives explicit port atoms which it converts to anchors via its
        ``anchor_selector_left`` / ``anchor_selector_right`` callables.

        Args:
            left: Left Atomistic structure
            right: Right Atomistic structure
            port_L: Port name on left structure (REQUIRED)
            port_R: Port name on right structure (REQUIRED)
            left_type: Type label for left structure (e.g., 'A', 'B')
            right_type: Type label for right structure

        Returns:
            Connected Atomistic assembly

        Raises:
            ValueError: If ports not found on structures
        """
        # Validate ports exist by checking if any atom has the port marker
        left_has_port = any(
            atom.get("port") == port_L
            or (isinstance(atom.get("ports"), list) and port_L in atom.get("ports", []))
            for atom in left.atoms
        )
        if not left_has_port:
            raise ValueError(
                f"Port '{port_L}' not found on left structure. "
                f"Atoms must have 'port' or 'ports' attribute set to '{port_L}'"
            )

        right_has_port = any(
            atom.get("port") == port_R
            or (isinstance(atom.get("ports"), list) and port_R in atom.get("ports", []))
            for atom in right.atoms
        )
        if not right_has_port:
            raise ValueError(
                f"Port '{port_R}' not found on right structure. "
                f"Atoms must have 'port' or 'ports' attribute set to '{port_R}'"
            )

        # Select reacter based on monomer types
        reacter = self._select_reacter(left_type, right_type)

        # Locate concrete port atoms – Reacter itself only works with anchors
        from molpy.reacter.selectors import find_port_atom

        port_atom_L = find_port_atom(left, port_L)
        port_atom_R = find_port_atom(right, port_R)

        # Execute reaction on explicit port atoms
        result: ReactionResult = reacter.run(
            left,
            right,
            port_atom_L=port_atom_L,
            port_atom_R=port_atom_R,
        )

        # Store in history for retypification
        self._history.append(result)

        return result.product_info.product

    def _select_reacter(
        self,
        left_type: str | None,
        right_type: str | None,
    ) -> Reacter:
        """Select appropriate reacter based on monomer types."""
        if left_type and right_type:
            key = (left_type, right_type)
            if key in self.specialized_reactions:
                return self.specialized_reactions[key]
            # Try reverse direction
            reverse_key = (right_type, left_type)
            if reverse_key in self.specialized_reactions:
                return self.specialized_reactions[reverse_key]

        return self.default_reaction

    def get_history(self) -> list[ReactionResult]:
        """
        Get history of all reactions performed.

        Useful for batch retypification after polymer assembly.

        Returns:
            List of ReactionResult for each connection made
        """
        return self._history.copy()

    def get_all_modified_atoms(self) -> set[Entity]:
        """
        Get all atoms that have been modified across all reactions.

        Returns:
            Set of all atoms that need retypification
        """
        modified: set[Entity] = set()
        for result in self._history:
            modified.update(result.topology_changes.modified_atoms)
        return modified

    def needs_retypification(self) -> bool:
        """
        Check if any reactions require retypification.

        Returns:
            True if retypification needed
        """
        return any(r.metadata.requires_retype for r in self._history)

    def clear_history(self):
        """Clear reaction history."""
        self._history.clear()
