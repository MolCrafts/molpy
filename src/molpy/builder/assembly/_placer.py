"""Geometry: where the pieces sit before the reaction joins them.

Placement answers "do these two components have meaningful relative
coordinates?", which is a fact about the *input*, not about which builder you
reached for. A packed melt already has them and must not be disturbed; fresh
template copies land on top of one another and must be laid out. So a placer is a
constructor argument, not a subclass.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from molpy.core import fields
from molrs import Element

if TYPE_CHECKING:
    from molpy.core.atomistic import Atom, Atomistic

#: Extra separation (Å) beyond the summed covalent radii, so the two components
#: start apart rather than exactly touching. An initial guess, converged by the
#: geometry optimisation that follows.
_BOND_BUFFER = 0.0


class Placer(ABC):
    """Position the components a set of bindings is about to join.

    Called once, before any reaction is applied, with the bindings the selector
    chose. Implementations mutate ``world``'s coordinates in place.
    """

    @abstractmethod
    def place(self, world: Atomistic, bonds: list[tuple[int, int]]) -> None:
        """Move components so each bond's two endpoints sit at bonding range.

        ``bonds`` are the ``(handle, handle)`` endpoints of the bonds the reaction
        is about to form. Note that only one endpoint need be a site atom — in
        ``[O;%a:1][H].[C:2][O;%b][H]>>[O:1][C:2]`` the bond runs from a site
        oxygen to a plain carbon — so a placer keys on the bond, not on
        ``fields.SITE``.
        """


class ResiduePlacer(Placer):
    """Lay residues out along the bonds a selector chose.

    Walks the residue graph the bindings imply, and for each parent → child edge
    rigidly moves the whole child residue so its site atom sits one bond length
    from the parent's, pointing away from the parent. Only residues are moved,
    never individual atoms: a template's internal geometry is its own business.

    The bond length is the sum of the two atoms' covalent radii. That is an
    **initial guess**: the equilibrium length is the force field's answer and no
    force field has been applied yet. A geometry optimisation downstream converges
    it. An unknown element raises rather than defaulting to carbon — you may guess
    a value, never an identity.
    """

    def __init__(self, buffer: float = _BOND_BUFFER) -> None:
        self._buffer = buffer

    def place(self, world: Atomistic, bonds: list[tuple[int, int]]) -> None:
        residues = self._residues(world)
        if len(residues) < 2:
            return
        edges = self._residue_edges(world, bonds)
        placed: set[int] = set()

        for parent, child, parent_site, child_site in self._walk(edges, residues):
            if child in placed or child == parent:
                continue
            self._move_residue(
                residues[child], residues[parent], parent_site, child_site
            )
            placed.add(child)

    # -- residue bookkeeping -------------------------------------------------

    @staticmethod
    def _residues(world: Atomistic) -> dict[int, list[Atom]]:
        out: dict[int, list[Atom]] = {}
        for atom in world.atoms:
            residue = atom.get(fields.RES_ID)
            if residue is None:
                continue
            out.setdefault(int(residue), []).append(atom)
        return out

    @staticmethod
    def _residue_edges(
        world: Atomistic, bonds: list[tuple[int, int]]
    ) -> list[tuple[int, int, Atom, Atom]]:
        """``(residue_a, residue_b, endpoint_a, endpoint_b)`` per forming bond."""
        edges: list[tuple[int, int, Atom, Atom]] = []
        for handle_a, handle_b in bonds:
            atom_a = world._intern_node(handle_a)
            atom_b = world._intern_node(handle_b)
            res_a = atom_a.get(fields.RES_ID)
            res_b = atom_b.get(fields.RES_ID)
            if res_a is None or res_b is None or int(res_a) == int(res_b):
                continue
            edges.append((int(res_a), int(res_b), atom_a, atom_b))
        return edges

    @staticmethod
    def _walk(
        edges: list[tuple[int, int, Atom, Atom]], residues: dict[int, list[Atom]]
    ):
        """BFS the residue graph from the lowest id, yielding placement steps."""
        adjacency: dict[int, list[tuple[int, Atom, Atom]]] = {r: [] for r in residues}
        for ra, rb, sa, sb in edges:
            adjacency.setdefault(ra, []).append((rb, sa, sb))
            adjacency.setdefault(rb, []).append((ra, sb, sa))

        seen = {min(residues)}
        queue = [min(residues)]
        while queue:
            parent = queue.pop(0)
            for child, parent_site, child_site in adjacency.get(parent, ()):
                if child in seen:
                    continue
                seen.add(child)
                queue.append(child)
                yield parent, child, parent_site, child_site

    # -- geometry ------------------------------------------------------------

    def _move_residue(
        self,
        child_atoms: list[Atom],
        parent_atoms: list[Atom],
        parent_site: Atom,
        child_site: Atom,
    ) -> None:
        child_coords = self._coords(child_atoms)
        parent_coords = self._coords(parent_atoms)
        parent_pos = np.asarray(self._xyz(parent_site), dtype=float)
        child_pos = np.asarray(self._xyz(child_site), dtype=float)

        outward = self._outward(parent_pos, parent_coords)
        target = parent_pos + outward * self._bond_length(parent_site, child_site)

        # Aim the child's own outward direction back at the parent, then slide its
        # site atom onto the target. Rigid: the template's internal geometry is
        # untouched.
        child_outward = self._outward(child_pos, child_coords)
        rotation = self._align(child_outward, -outward)
        moved = (child_coords - child_pos) @ rotation.T + target

        for atom, position in zip(child_atoms, moved, strict=True):
            atom[fields.POS_X] = float(position[0])
            atom[fields.POS_Y] = float(position[1])
            atom[fields.POS_Z] = float(position[2])

    @classmethod
    def _coords(cls, atoms: list[Atom]) -> np.ndarray:
        return np.array([cls._xyz(atom) for atom in atoms], dtype=float)

    @staticmethod
    def _xyz(atom: Atom) -> tuple[float, float, float]:
        return (atom[fields.POS_X], atom[fields.POS_Y], atom[fields.POS_Z])

    @staticmethod
    def _outward(site_pos: np.ndarray, residue_coords: np.ndarray) -> np.ndarray:
        """Unit vector from the residue's centroid through its site atom.

        A site atom sits on the residue's surface, so this points out of the
        residue — the direction a partner should approach from.
        """
        direction = site_pos - residue_coords.mean(axis=0)
        norm = float(np.linalg.norm(direction))
        if norm < 1e-9:
            # a one-atom residue, or a site at the centroid: any axis will do
            return np.array([1.0, 0.0, 0.0])
        return direction / norm

    @classmethod
    def _align(cls, source: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Rotation matrix taking unit vector ``source`` onto unit vector ``target``.

        Always a proper rotation (``det == +1``). For antiparallel vectors the
        naive ``-I`` is a *reflection*: it would mirror the residue and invert its
        chirality. Turn 180° about any perpendicular axis instead.
        """
        v = np.cross(source, target)
        c = float(np.dot(source, target))
        if np.allclose(v, 0.0):
            if c > 0:
                return np.eye(3)
            axis = cls._perpendicular(source)
            outer = np.outer(axis, axis)
            return 2.0 * outer - np.eye(3)  # 180 deg about `axis`; det == +1
        kmat = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
        return np.eye(3) + kmat + kmat @ kmat * (1.0 / (1.0 + c))

    @staticmethod
    def _perpendicular(vector: np.ndarray) -> np.ndarray:
        """Any unit vector orthogonal to ``vector``."""
        axis = np.array([1.0, 0.0, 0.0])
        if abs(float(np.dot(vector, axis))) > 0.9:
            axis = np.array([0.0, 1.0, 0.0])
        perpendicular = np.cross(vector, axis)
        return perpendicular / np.linalg.norm(perpendicular)

    def _bond_length(self, a: Atom, b: Atom) -> float:
        return self._radius(a) + self._radius(b) + self._buffer

    @staticmethod
    def _radius(atom: Atom) -> float:
        """Covalent radius (Å). An unknown element raises — never assume carbon."""
        symbol = atom.get(fields.ELEMENT)
        if not symbol:
            raise KeyError(
                f"atom {atom.handle} has no {fields.ELEMENT.key}; placement needs "
                "the element to look up a covalent radius (it may not be guessed)"
            )
        return Element(str(symbol)).covalent
