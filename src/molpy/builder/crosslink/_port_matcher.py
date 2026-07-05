"""Port-based site front-end (crosslink-03).

:class:`PortMatcher` turns modelled ``atom["port"]`` markers — from BigSMILES
``<`` / ``>`` / ``$`` descriptors or ``[*:n]`` atom-class ports — into the same
occurrence shape molrs returns (``list[dict[map_number, handle]]``). "Sites
marked at modelling time" and "sites found now by SMARTS" therefore merge onto a
single molrs edit engine — the selection front-end is swappable, the
:class:`~molpy.builder.crosslink.Crosslinker` edit path is the one and only.
"""

from __future__ import annotations

from molpy.core.atomistic import Atomistic


class PortMatcher:
    """Read ``atom[port_key]`` markers as molrs-shaped occurrences.

    Each atom whose ``port_key`` field equals the requested ``port_name`` yields
    one occurrence ``{1: handle}`` (map number 1 = the port-bearing atom),
    matching ``molrs.SmartsPattern.find_matches_mapped`` in shape so a
    ``Crosslinker`` can consume either front-end.
    """

    def __init__(self, *, port_key: str = "port") -> None:
        self._port_key = port_key

    def find_matches_mapped(
        self, graph: Atomistic, port_name: str
    ) -> list[dict[int, int]]:
        """Occurrences for every atom carrying ``port_key == port_name``."""
        return [
            {1: atom.handle}
            for atom in graph.atoms
            if atom.get(self._port_key) == port_name
        ]
