"""2D depiction — ``molrs::depict``.

Drawing coordinates for a molecular graph, the 2D counterpart of
:mod:`molpy.conformer`: rings as regular polygons, chains as a zigzag,
substituents spread into the free angle, disconnected species side by side.
The copy keeps every property and relation, so a 2D viewer draws formal
charges and ports from the frame ``to_frame()`` writes::

    flat = molpy.depict.Depictor().layout(mol)
    frame = flat.to_frame()   # x, y in the drawing plane, z = 0
"""

from molpy._core import Depictor

__all__ = ["Depictor"]
