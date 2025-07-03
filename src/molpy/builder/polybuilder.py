"""
Flexible PolymerBuilder for molpy - Template-based polymer construction.

This module provides a modular and extensible builder system for constructing
polymers using reusable monomer templates with context-aware anchor matching.
"""
from dataclasses import dataclass, field
import numpy as np
from ..core.atomistic import Atomistic
from ..core.wrapper import Wrapper
from molpy.op.geometry import rotation_matrix_from_vectors


@dataclass
class AnchorRule:
    """
    Context-aware anchor matching rule for polymer construction.

    Defines how an anchor atom should behave based on the context
    of neighboring monomers in the polymer chain.
    """

    init: int
    end: int
    deletes: list[int] = field(default_factory=list)


class Monomer(Wrapper):
    """
    Template for a monomer unit with anchor definitions.

    Inherits from Wrapper to enable composable functionality.
    Contains the structural information and anchor rules needed
    to construct and connect monomers in polymer chains.
    """

    anchors: list[AnchorRule] = field(default_factory=list)

    def __init__(
        self,
        struct: Atomistic,
        anchors: list[AnchorRule] = [],
    ):
        """Initialize Monomer with struct, anchors, and metadata."""
        super().__init__(struct)
        self.anchors = anchors

    
class PolymerBuilder:

    def __init__(self, monomers: dict[str, Monomer]):
        """
        Initialize PolymerBuilder with a dictionary of named monomers.

        Parameters
        ----------
        monomers : dict[str, Monomer]
            Dictionary mapping monomer names to Monomer instances.
        """
        self.monomers = monomers

    def build(
        self,
        path: np.ndarray,
        seq: list[str],):
        all_positions = []
        all_symbols   = []

        for i, name in enumerate(seq):
            mon = self.monomers[name]
            rule = mon.anchors[0]
            coords = mon.xyz.copy()   # (N_i, 3)
            syms   = mon.struct.symbols.copy()     # length N_i

            # 1) compute the “native” direction in the monomer
            v_native = coords[rule.end] - coords[rule.init]

            # 2) compute target direction: next path segment
            if i < len(path) - 1:
                v_target = path[i + 1] - path[i]
            else:
                # for the last monomer, just reuse the previous direction
                v_target = path[i] - path[i - 1]

            # 3) build rotation matrix and apply
            R = rotation_matrix_from_vectors(v_native, v_target)
            coords = coords.dot(R.T)

            # 4) translate so that init anchor lands exactly on path[i]
            shift = path[i] - coords[rule.init]
            coords += shift

            # 5) drop any atoms that should be deleted (to avoid overlap)
            if i > 0 and rule.deletes:
                keep = np.ones(len(coords), dtype=bool)
                keep[rule.deletes] = False
                coords = coords[keep]
                syms   = syms[keep]

            # 6) append to the growing chain
            all_positions.append(coords)
            all_symbols.append(syms)

        # 7) concatenate into one Atomistic
        all_positions = np.vstack(all_positions)
        all_symbols   = sum([list(s) for s in all_symbols], [])

        return Atomistic(positions=all_positions, symbols=all_symbols)