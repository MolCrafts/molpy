"""
Connector for polymer assembly.

The Connector decides which ports to connect between adjacent monomers
and executes the chemical reaction via a Reacter.
"""

from collections.abc import Mapping
from typing import Any

from molpy.core.atomistic import Atom, Atomistic
from molpy.core.entity import Entity
from molpy.reacter.base import Reacter
from molpy.typifier.atomistic import TypifierBase

from .errors import AmbiguousPortsError
from .port_utils import port_role, ports_compatible


class ConnectorContext(dict[str, Any]):
    """Shared context passed to the connector during linear build.

    Keys:
    - step: int (current connection step index)
    - left_label: str (label of left monomer)
    - right_label: str (label of right monomer)
    - sequence: list[str] (full sequence being built)
    """

    pass


class Connector:
    """Select ports and execute reactions between adjacent monomers.

    Port selection strategy (applied in order):
    1. Explicit port_map lookup for (left_label, right_label)
    2. Compatibility: ``>`` on left pairs with ``<`` on right
    3. Single-port: each side has exactly one unconsumed port
    4. Common name: both sides share a port name (for ``$`` ports)
    5. Raise AmbiguousPortsError
    """

    def __init__(
        self,
        reacter: Reacter,
        *,
        port_map: dict[tuple[str, str], tuple[str, str]] | None = None,
        overrides: dict[tuple[str, str], Reacter] | None = None,
    ):
        self.default = reacter
        self.port_map = port_map or {}
        self.overrides = overrides or {}
        self._history: list = []

    def get_reacter(self, left_type: str, right_type: str) -> Reacter:
        """Get the appropriate Reacter for a structure pair."""
        return self.overrides.get((left_type, right_type), self.default)

    def select_ports(
        self,
        left: Atomistic,
        right: Atomistic,
        left_ports: Mapping[str, list[Atom]],
        right_ports: Mapping[str, list[Atom]],
        ctx: ConnectorContext,
    ) -> tuple[str, int, str, int, None]:
        """Select which ports to connect.

        Args:
            left: Left Atomistic structure.
            right: Right Atomistic structure.
            left_ports: Available ports on left (name -> list[Atom]).
            right_ports: Available ports on right (name -> list[Atom]).
            ctx: Context with step info and labels.

        Returns:
            (left_port_name, left_idx, right_port_name, right_idx, None)
        """
        left_label = ctx.get("left_label", "")
        right_label = ctx.get("right_label", "")

        # 1. Explicit port_map
        key = (left_label, right_label)
        if key in self.port_map:
            port_L, port_R = self.port_map[key]
            if port_L not in left_ports:
                raise AmbiguousPortsError(
                    f"Port '{port_L}' not found on left ({left_label})"
                )
            if port_R not in right_ports:
                raise AmbiguousPortsError(
                    f"Port '{port_R}' not found on right ({right_label})"
                )
            return (port_L, 0, port_R, 0, None)

        # 2. Compatibility: find pairs where left > connects to right <
        # Prefer directional match: left's "right" role + right's "left" role
        compatible = []
        for lname in left_ports:
            for rname in right_ports:
                if ports_compatible(lname, rname):
                    lr = port_role(lname)
                    rr = port_role(rname)
                    if lr == "right" and rr == "left":
                        # Exact directional match — use immediately
                        return (lname, 0, rname, 0, None)
                    compatible.append((lname, 0, rname, 0))

        if len(compatible) == 1:
            lname, li, rname, ri = compatible[0]
            return (lname, li, rname, ri, None)

        # 3. Single-port on each side
        if len(left_ports) == 1 and len(right_ports) == 1:
            ln = next(iter(left_ports))
            rn = next(iter(right_ports))
            return (ln, 0, rn, 0, None)

        # 4. Common port name (for symmetric ports like $)
        common = set(left_ports) & set(right_ports)
        if common:
            name = next(iter(common))
            return (name, 0, name, 0, None)

        # 5. Give up
        raise AmbiguousPortsError(
            f"Cannot auto-select ports between {left_label} and {right_label}: "
            f"left ports={list(left_ports.keys())}, right ports={list(right_ports.keys())}."
        )

    def connect(
        self,
        left: Atomistic,
        right: Atomistic,
        left_type: str,
        right_type: str,
        port_atom_L: Entity,
        port_atom_R: Entity,
        typifier: TypifierBase | None = None,
    ) -> "ReactionResult":
        """Execute the chemical reaction between two structures."""
        from molpy.reacter.base import ReactionResult

        reacter = self.get_reacter(left_type, right_type)

        result: ReactionResult = reacter.run(
            left,
            right,
            port_atom_L=port_atom_L,
            port_atom_R=port_atom_R,
            compute_topology=True,
            typifier=typifier,
        )

        self._history.append(result)
        return result
