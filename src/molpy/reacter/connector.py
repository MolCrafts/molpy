"""
Connector interface for integrating Reacter with the linear polymer builder.

This module provides connectors that use Reacter for polymer assembly,
allowing flexible reaction specification with default and specialized reactors.
"""

from molpy import Atomistic
from molpy.core.wrappers.monomer import Monomer

from .base import ProductSet, Reacter


class ReacterConnector:
    """
    Connector adapter that manages Reacter instances for polymer assembly.

    This connector allows specifying a default reaction for most connections,
    with the ability to override specific monomer pair connections with
    specialized reacters.

    **Port Selection Philosophy:**
    This connector does NOT handle port selection. The caller must explicitly
    provide port_L and port_R when calling connect(). Port selection should
    be handled by the higher-level builder logic.

    Example:
        >>> from molpy.reacter import Reacter, ReacterConnector
        >>>
        >>> # Default reaction for most connections
        >>> default_reacter = Reacter(...)
        >>>
        >>> # Special reaction for A-B connection
        >>> special_reacter = Reacter(...)
        >>>
        >>> connector = ReacterConnector(
        ...     default=default_reacter,
        ...     overrides={('A', 'B'): special_reacter}
        ... )
        >>>
        >>> # Explicit port specification required
        >>> product = connector.connect(
        ...     left=monomer_a, right=monomer_b,
        ...     left_type='A', right_type='B',
        ...     port_L='1', port_R='2'  # REQUIRED
        ... )
    """

    def __init__(
        self,
        default: Reacter,
        overrides: dict[tuple[str, str], Reacter] | None = None,
    ):
        """
        Initialize connector with default and override reacters.

        Args:
            default: Default Reacter used for most connections
            overrides: Dict mapping (left_type, right_type) -> specialized Reacter
        """
        self.default = default
        self.overrides = overrides or {}
        self._history: list[ProductSet] = []

    def connect(
        self,
        left: Monomer,
        right: Monomer,
        port_L: str,
        port_R: str,
        left_type: str | None = None,
        right_type: str | None = None,
    ) -> Atomistic:
        """
        Connect two monomers using appropriate reacter.

        **IMPORTANT: port_L and port_R must be explicitly specified.**
        No automatic port selection or fallback is performed.

        Args:
            left: Left monomer
            right: Right monomer
            port_L: Port name on left monomer (REQUIRED)
            port_R: Port name on right monomer (REQUIRED)
            left_type: Type label for left monomer (e.g., 'A', 'B')
            right_type: Type label for right monomer

        Returns:
            Connected Atomistic assembly

        Raises:
            ValueError: If ports not found on monomers
        """
        # Validate ports exist
        if port_L not in left.ports:
            raise ValueError(
                f"Port '{port_L}' not found on left monomer. "
                f"Available ports: {list(left.ports.keys())}"
            )
        if port_R not in right.ports:
            raise ValueError(
                f"Port '{port_R}' not found on right monomer. "
                f"Available ports: {list(right.ports.keys())}"
            )

        # Select reacter based on monomer types
        reacter = self._select_reacter(left_type, right_type)

        # Execute reaction
        product = reacter.run(left, right, port_L=port_L, port_R=port_R)

        # Store in history for retypification
        self._history.append(product)

        return product.product

    def _select_reacter(
        self,
        left_type: str | None,
        right_type: str | None,
    ) -> Reacter:
        """Select appropriate reacter based on monomer types."""
        if left_type and right_type:
            key = (left_type, right_type)
            if key in self.overrides:
                return self.overrides[key]
            # Try reverse direction
            reverse_key = (right_type, left_type)
            if reverse_key in self.overrides:
                return self.overrides[reverse_key]

        return self.default

    def get_history(self) -> list[ProductSet]:
        """
        Get history of all reactions performed.

        Useful for batch retypification after polymer assembly.

        Returns:
            List of ProductSet for each connection made
        """
        return self._history.copy()

    def get_all_modified_atoms(self) -> set:
        """
        Get all atoms that have been modified across all reactions.

        Returns:
            Set of all atoms that need retypification
        """
        modified = set()
        for product in self._history:
            modified.update(product.notes.get("modified_atoms", []))
        return modified

    def needs_retypification(self) -> bool:
        """
        Check if any reactions require retypification.

        Returns:
            True if retypification needed
        """
        return any(p.notes.get("needs_retypification", False) for p in self._history)

    def clear_history(self):
        """Clear reaction history."""
        self._history.clear()
