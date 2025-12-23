"""
Placer components for positioning monomers during polymer assembly.

A Placer consists of:
- Separator: Determines distance between monomers
- Orienter: Determines monomer orientation

This module provides VdW-based placement with linear alignment.
"""

from typing import Protocol

import numpy as np

from molpy.core.atomistic import Atomistic
from molpy.core.element import Element

from .port_utils import PortInfo

__all__ = [
    "CovalentSeparator",
    "LinearOrienter",
    "Orienter",
    "Placer",
    "Separator",
    "VdWSeparator",
    "create_covalent_linear_placer",
    "create_vdw_linear_placer",
]


class Separator(Protocol):
    """Protocol for calculating separation distance between structures."""

    def get_separation(
        self,
        left_struct: Atomistic,
        right_struct: Atomistic,
        left_port: PortInfo,
        right_port: PortInfo,
    ) -> float:
        """
        Calculate separation distance between structures.

        Args:
            left_struct: Previous structure in sequence
            right_struct: Next structure to place
            left_port: Connection port on left structure
            right_port: Connection port on right structure

        Returns:
            Separation distance in Angstroms
        """
        ...


class Orienter(Protocol):
    """Protocol for determining structure orientation."""

    def get_orientation(
        self,
        left_struct: Atomistic,
        right_struct: Atomistic,
        left_port: PortInfo,
        right_port: PortInfo,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate target position and orientation for right structure.

        Args:
            left_struct: Previous structure in sequence
            right_struct: Next structure to place
            left_port: Connection port on left structure
            right_port: Connection port on right structure

        Returns:
            Tuple of (translation_vector, rotation_matrix)
            - translation_vector: 3D vector to move right structure
            - rotation_matrix: 3x3 rotation matrix to orient right structure
        """
        ...


class VdWSeparator:
    """
    Separator based on van der Waals radii.

    Calculates separation as sum of VdW radii of the two port anchor atoms,
    plus an optional buffer distance.

    NOTE: VdW radii are designed for non-bonded contacts (~3-4 Å).
    For bonded atoms, use CovalentSeparator instead.
    """

    def __init__(self, buffer: float = 0.0):
        """
        Initialize VdW separator.

        Args:
            buffer: Additional buffer distance in Angstroms (default: 0.0)
        """
        self.buffer = buffer

    def get_separation(
        self,
        left_struct: Atomistic,
        right_struct: Atomistic,
        left_port: PortInfo,
        right_port: PortInfo,
    ) -> float:
        """
        Calculate separation based on VdW radii.

        Args:
            left_struct: Previous structure in sequence
            right_struct: Next structure to place
            left_port: Connection port on left structure
            right_port: Connection port on right structure

        Returns:
            Separation distance = vdw_left + vdw_right + buffer
        """
        # Get port anchor atoms
        left_anchor = left_port.target
        right_anchor = right_port.target

        # Get element symbols
        left_symbol = left_anchor.get("symbol", "C")
        right_symbol = right_anchor.get("symbol", "C")

        # Get VdW radii
        try:
            left_vdw = Element(left_symbol).vdw
        except KeyError:
            left_vdw = 1.70  # Default to carbon

        try:
            right_vdw = Element(right_symbol).vdw
        except KeyError:
            right_vdw = 1.70  # Default to carbon

        return left_vdw + right_vdw + self.buffer


class CovalentSeparator:
    """
    Separator based on typical bond lengths (for bonded atoms).

    Uses realistic bond lengths based on element types.
    Typical bond lengths:
    - C-C: 1.54 Å (single), 1.34 Å (double)
    - C-O: 1.43 Å (single), 1.23 Å (double)
    - C-N: 1.47 Å (single)
    - O-H: 0.96 Å
    - N-H: 1.01 Å
    """

    def __init__(self, buffer: float = 0.0):
        """
        Initialize covalent separator.

        Args:
            buffer: Additional buffer distance in Angstroms (default: 0.0)
                   Can be negative to account for slight compression
        """
        self.buffer = buffer

        # Typical single bond lengths (in Angstroms)
        self.bond_lengths = {
            ("C", "C"): 1.54,
            ("C", "O"): 1.43,
            ("C", "N"): 1.47,
            ("C", "S"): 1.82,
            ("C", "H"): 1.09,
            ("O", "H"): 0.96,
            ("N", "H"): 1.01,
            ("O", "O"): 1.48,
            ("N", "N"): 1.45,
            ("S", "S"): 2.05,
        }

    def get_separation(
        self,
        left_struct: Atomistic,
        right_struct: Atomistic,
        left_port: PortInfo,
        right_port: PortInfo,
    ) -> float:
        """
        Calculate separation based on typical bond lengths.

        Args:
            left_struct: Previous structure in sequence
            right_struct: Next structure to place
            left_port: Connection port on left structure
            right_port: Connection port on right structure

        Returns:
            Separation distance = typical_bond_length + buffer
        """
        # Get port anchor atoms
        left_anchor = left_port.target
        right_anchor = right_port.target

        # Get element symbols
        left_symbol = left_anchor.get("symbol", "C")
        right_symbol = right_anchor.get("symbol", "C")

        # Look up bond length (try both orderings)
        bond_key = (left_symbol, right_symbol)
        reverse_key = (right_symbol, left_symbol)

        if bond_key in self.bond_lengths:
            bond_length = self.bond_lengths[bond_key] / 2
        elif reverse_key in self.bond_lengths:
            bond_length = self.bond_lengths[reverse_key] / 2
        else:
            # Default: C-C bond length
            bond_length = 1.54

        return bond_length + self.buffer


class LinearOrienter:
    """
    Orienter for linear polymer arrangement.

    Aligns the next monomer so that:
    1. The two port atoms are separated by the specified distance
    2. The port connection axis of the next monomer aligns with
       the port connection axis of the previous monomer
    3. The monomer extends in a linear fashion
    """

    def get_orientation(
        self,
        left_struct: Atomistic,
        right_struct: Atomistic,
        left_port: PortInfo,
        right_port: PortInfo,
        separation: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate linear alignment transformation.

        Strategy:
        1. Get direction vector from left port anchor (outward)
        2. Place right structure so its port anchor is at the target position
        3. Align right structure's port direction with left port direction

        Args:
            left_struct: Previous structure in sequence
            right_struct: Next structure to place
            left_port: Connection port on left structure
            right_port: Connection port on right structure
            separation: Distance between port anchors

        Returns:
            Tuple of (translation_vector, rotation_matrix)
        """
        # Get port anchor atoms
        left_anchor = left_port.target
        right_anchor = right_port.target

        # Get positions from x/y/z fields
        left_anchor_pos = np.array(
            [
                left_anchor["x"],
                left_anchor["y"],
                left_anchor["z"],
            ]
        )
        right_anchor_pos = np.array(
            [
                right_anchor["x"],
                right_anchor["y"],
                right_anchor["z"],
            ]
        )

        # Calculate port direction vectors
        # For left port: direction pointing outward (away from structure center)
        left_direction = self._get_port_direction(left_struct, left_port)

        # For right port: direction pointing inward (toward structure center)
        # We'll reverse it to get the connection direction
        right_direction = self._get_port_direction(right_struct, right_port)

        # Target position for right anchor
        target_pos = left_anchor_pos + left_direction * separation

        # Translation: move right anchor to target position
        translation = target_pos - right_anchor_pos

        # Rotation: align right port direction with -left_direction
        # (so they face each other)
        # Rotate FROM right_direction TO -left_direction
        rotation = self._rotation_matrix_from_vectors(right_direction, -left_direction)

        return translation, rotation

    def _get_port_direction(self, struct: Atomistic, port: PortInfo) -> np.ndarray:
        """
        Calculate direction vector for a port.

        Strategy:
        - If port has a neighbor atom (bonded), use that direction
        - Otherwise, use direction from structure centroid to port anchor

        Args:
            struct: Atomistic structure containing the port
            port: Port to calculate direction for

        Returns:
            Normalized 3D direction vector
        """
        anchor = port.target
        anchor_pos = np.array(
            [
                anchor["x"],
                anchor["y"],
                anchor["z"],
            ]
        )

        # Try to find a bonded neighbor
        bonds = list(struct.bonds)

        neighbor_pos = None
        for bond in bonds:
            if bond.itom == anchor:
                neighbor = bond.jtom
                neighbor_pos = np.array(
                    [
                        neighbor["x"],
                        neighbor["y"],
                        neighbor["z"],
                    ]
                )
                break
            elif bond.jtom == anchor:
                neighbor = bond.itom
                neighbor_pos = np.array(
                    [
                        neighbor["x"],
                        neighbor["y"],
                        neighbor["z"],
                    ]
                )
                break

        if neighbor_pos is not None:
            # Direction from neighbor to anchor (pointing out)
            direction = anchor_pos - neighbor_pos
        else:
            # Fallback: use direction from centroid
            atoms = list(struct.atoms)
            positions = [np.array([a["x"], a["y"], a["z"]]) for a in atoms]
            centroid = np.mean(positions, axis=0)
            direction = anchor_pos - centroid

        # Normalize
        norm = np.linalg.norm(direction)
        if norm > 1e-6:
            return direction / norm
        else:
            # Default to x-axis if position is degenerate
            return np.array([1.0, 0.0, 0.0])

    def _rotation_matrix_from_vectors(
        self, vec_from: np.ndarray, vec_to: np.ndarray
    ) -> np.ndarray:
        """
        Calculate rotation matrix to align vec_from with vec_to.

        Uses Rodrigues' rotation formula.

        Args:
            vec_from: Source direction (normalized)
            vec_to: Target direction (normalized)

        Returns:
            3x3 rotation matrix
        """
        # Normalize vectors
        a = vec_from / np.linalg.norm(vec_from)
        b = vec_to / np.linalg.norm(vec_to)

        # Calculate rotation axis and angle
        v = np.cross(a, b)
        c = np.dot(a, b)

        # Handle aligned or opposite vectors
        if np.allclose(v, 0):
            if c > 0:
                # Already aligned
                return np.eye(3)
            else:
                # Opposite directions - rotate 180° around any perpendicular axis
                # Find a perpendicular vector
                if abs(a[0]) < 0.9:
                    perp = np.cross(a, [1, 0, 0])
                else:
                    perp = np.cross(a, [0, 1, 0])
                perp = perp / np.linalg.norm(perp)
                # 180° rotation around perp
                return 2 * np.outer(perp, perp) - np.eye(3)

        # Rodrigues' formula
        s = np.linalg.norm(v)
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])

        rotation_matrix = np.eye(3) + kmat + kmat @ kmat * ((1 - c) / (s**2))
        return rotation_matrix


class Placer:
    """
    Combined placer for positioning structures during assembly.

    Uses a Separator to determine distance and an Orienter to
    determine orientation.
    """

    def __init__(self, separator: Separator, orienter: LinearOrienter):
        """
        Initialize placer.

        Args:
            separator: Separator for calculating distance
            orienter: Orienter for calculating orientation
        """
        self.separator = separator
        self.orienter = orienter

    def place_monomer(
        self,
        left_struct: Atomistic,
        right_struct: Atomistic,
        left_port: PortInfo,
        right_port: PortInfo,
    ) -> None:
        """
        Position right_struct relative to left_struct.

        Modifies right_struct's atomic coordinates in-place.

        Args:
            left_struct: Previous structure in sequence
            right_struct: Next structure to place
            left_port: Connection port on left structure
            right_port: Connection port on right structure
        """
        # Calculate separation
        separation = self.separator.get_separation(
            left_struct, right_struct, left_port, right_port
        )

        # Calculate orientation
        translation, rotation = self.orienter.get_orientation(
            left_struct, right_struct, left_port, right_port, separation
        )

        # Use right port anchor as pivot for rotation
        right_anchor = right_port.target
        right_anchor_pos = np.array(
            [
                right_anchor["x"],
                right_anchor["y"],
                right_anchor["z"],
            ]
        )

        # Apply transformation to right structure
        self._apply_transform(
            right_struct, translation, rotation, pivot=right_anchor_pos
        )

    def _apply_transform(
        self,
        struct: Atomistic,
        translation: np.ndarray,
        rotation: np.ndarray,
        pivot: np.ndarray | None = None,
    ) -> None:
        """
        Apply rotation and translation to all atoms in structure.

        Rotation is applied around a pivot point (default: centroid),
        then translation is applied.

        Args:
            struct: Atomistic structure to transform
            translation: 3D translation vector
            rotation: 3x3 rotation matrix
            pivot: Optional pivot point for rotation (default: centroid)
        """
        atoms = list(struct.atoms)

        # Calculate centroid if no pivot given
        if pivot is None:
            positions = [np.array([atom["x"], atom["y"], atom["z"]]) for atom in atoms]
            pivot = np.mean(positions, axis=0)

        # Transform each atom
        for atom in atoms:
            # Get current position from x/y/z fields
            xyz = np.array(
                [
                    atom["x"],
                    atom["y"],
                    atom["z"],
                ]
            )

            # Rotate around pivot, then translate
            # 1. Move to origin
            pos_centered = xyz - pivot
            # 2. Rotate
            pos_rotated = rotation @ pos_centered
            # 3. Move back and translate
            new_pos = pos_rotated + pivot + translation

            # Update position to x/y/z fields
            atom["x"] = float(new_pos[0])
            atom["y"] = float(new_pos[1])
            atom["z"] = float(new_pos[2])
