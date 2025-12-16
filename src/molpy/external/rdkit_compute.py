"""RDKit-based compute operations for RDKitAdapter.

This module provides compute nodes that operate on RDKitAdapter instances
to perform RDKit-based molecular operations like 3D coordinate generation
and geometry optimization.
"""

from __future__ import annotations

from dataclasses import dataclass
import warnings

from rdkit import Chem
from rdkit.Chem import AllChem

from molpy.compute.base import Compute
from molpy.external.rdkit_adapter import MP_ID, RDKitAdapter


@dataclass
class OptimizeGeometry(Compute[RDKitAdapter, RDKitAdapter]):
    """RDKit-based geometry optimization for RDKitAdapter.

    This compute node optimizes the geometry of a molecule using RDKit's
    force field methods (UFF or MMFF94).

    Attributes:
        max_opt_iters: Maximum optimization iterations
        forcefield: Force field to use ("UFF" or "MMFF94")
        update_internal: Whether to sync internal structure after optimization
        raise_on_failure: Whether to raise exception on optimization failure (default: False)
                          If False, warnings are issued but optimization continues

    Examples:
        >>> from molpy.external import RDKitAdapter, OptimizeGeometry
        >>>
        >>> adapter = RDKitAdapter(internal=atomistic)
        >>> optimizer = OptimizeGeometry(forcefield="UFF", max_opt_iters=200)
        >>> adapter = optimizer(adapter)
    """

    max_opt_iters: int = 200
    forcefield: str = "UFF"
    update_internal: bool = True
    raise_on_failure: bool = False

    def compute(self, input: RDKitAdapter) -> RDKitAdapter:
        """Execute geometry optimization.

        Args:
            input: RDKitAdapter to process

        Returns:
            The same RDKitAdapter instance (mutated)

        Raises:
            ValueError: If no conformer found
            RuntimeError: If optimization fails and raise_on_failure=True
        """
        if not input.has_external():
            input.sync_to_external()

        original_mol = input.get_external()
        # Create a copy to avoid modifying the original
        # Chem.Mol() creates a deep copy, but we need to ensure conformer is copied
        mol = Chem.Mol(original_mol)

        # Explicitly copy conformer if it wasn't copied automatically
        if mol.GetNumConformers() == 0 and original_mol.GetNumConformers() > 0:
            original_conf = original_mol.GetConformer()
            new_conf = Chem.Conformer(mol.GetNumAtoms())
            for i in range(mol.GetNumAtoms()):
                pos = original_conf.GetAtomPosition(i)
                new_conf.SetAtomPosition(i, pos)
            mol.AddConformer(new_conf, assignId=True)

        # Sanitize molecule to ensure proper atom types and hybridization
        # This is required for UFF/MMFF optimization
        try:
            Chem.SanitizeMol(mol)
        except Exception as e:
            # If sanitization fails, try with less strict settings
            try:
                mol.UpdatePropertyCache(strict=False)
                # Try to fix common issues
                for atom in mol.GetAtoms():
                    if atom.GetHybridization() == Chem.HybridizationType.UNSPECIFIED:
                        # Try to infer hybridization from connectivity
                        num_neighbors = atom.GetDegree()
                        if num_neighbors == 0:
                            atom.SetHybridization(Chem.HybridizationType.S)
                        elif num_neighbors == 1:
                            atom.SetHybridization(Chem.HybridizationType.SP)
                        elif num_neighbors == 2:
                            atom.SetHybridization(Chem.HybridizationType.SP2)
                        elif num_neighbors == 3:
                            atom.SetHybridization(Chem.HybridizationType.SP3)
                        elif num_neighbors == 4:
                            atom.SetHybridization(Chem.HybridizationType.SP3D)
                mol.UpdatePropertyCache(strict=False)
            except Exception as e2:
                raise RuntimeError(
                    f"Failed to prepare molecule for optimization: {e}. "
                    f"Fallback also failed: {e2}. "
                    "The molecule may have structural issues."
                ) from e

        # Update property cache before optimization (required by RDKit)
        mol.UpdatePropertyCache(strict=False)

        if mol.GetNumConformers() == 0:
            raise ValueError(
                "Cannot optimize geometry: no conformer found. "
                "Enable embedding or provide a molecule with coordinates."
            )

        if self.forcefield == "UFF":
            # Store coordinates before optimization to check if they changed
            conf_before = mol.GetConformer() if mol.GetNumConformers() > 0 else None
            coords_before = None
            if conf_before is not None:
                coords_before = [
                    conf_before.GetAtomPosition(i) for i in range(mol.GetNumAtoms())
                ]

            opt_result = AllChem.UFFOptimizeMolecule(  # type: ignore[attr-defined]
                mol, maxIters=int(self.max_opt_iters)
            )

            # Check if coordinates actually changed
            coords_changed = False
            if conf_before is not None and mol.GetNumConformers() > 0:
                conf_after = mol.GetConformer()
                if coords_before is not None:
                    for i in range(mol.GetNumAtoms()):
                        pos_before = coords_before[i]
                        pos_after = conf_after.GetAtomPosition(i)
                        # Use a more lenient threshold (1e-5) to detect coordinate changes
                        # This accounts for numerical precision and ensures we catch real optimizations
                        if (
                            abs(pos_before.x - pos_after.x) > 1e-5
                            or abs(pos_before.y - pos_after.y) > 1e-5
                            or abs(pos_before.z - pos_after.z) > 1e-5
                        ):
                            coords_changed = True
                            break

            if opt_result != 0:
                msg = (
                    f"UFF optimization returned code {opt_result}. "
                    f"Code 1 typically means convergence not reached within {self.max_opt_iters} iterations. "
                    "The structure may still be improved."
                )
                if self.raise_on_failure:
                    raise RuntimeError(msg)
                else:
                    warnings.warn(msg, UserWarning)
            elif not coords_changed:
                # Optimization returned success (0) but coordinates didn't change
                # This can happen if the structure is already optimized
                # However, if there are long bonds, this might indicate an issue
                # Check if there are unusually long bonds that should have been optimized
                max_bond_length = 0.0
                if mol.GetNumConformers() > 0:
                    conf = mol.GetConformer()
                    for bond in mol.GetBonds():
                        begin_idx = bond.GetBeginAtomIdx()
                        end_idx = bond.GetEndAtomIdx()
                        pos1 = conf.GetAtomPosition(begin_idx)
                        pos2 = conf.GetAtomPosition(end_idx)
                        bond_length = (
                            (pos1.x - pos2.x) ** 2
                            + (pos1.y - pos2.y) ** 2
                            + (pos1.z - pos2.z) ** 2
                        ) ** 0.5
                        max_bond_length = max(max_bond_length, bond_length)

                if (
                    max_bond_length > 2.0
                ):  # Typical bond length is ~1.5 Å, >2.0 is suspicious
                    warnings.warn(
                        f"UFF optimization completed but coordinates did not change, "
                        f"despite long bonds detected (max bond length: {max_bond_length:.3f} Å). "
                        f"This may indicate the optimization did not work properly. "
                        f"Consider using MMFF94 force field or increasing max_opt_iters.",
                        UserWarning,
                    )
                else:
                    warnings.warn(
                        "UFF optimization completed but coordinates did not change. "
                        "The structure may already be optimized or at a local minimum. "
                        "For ring/cyclic structures, this often indicates the geometry is already optimal.",
                        UserWarning,
                    )
        elif self.forcefield == "MMFF94":
            try:
                opt_result = AllChem.MMFFOptimizeMolecule(  # type: ignore[attr-defined]
                    mol, maxIters=int(self.max_opt_iters)
                )
                if opt_result != 0:
                    msg = (
                        f"MMFF94 optimization returned code {opt_result}. "
                        f"Code 1 typically means convergence not reached within {self.max_opt_iters} iterations."
                    )
                    if self.raise_on_failure:
                        raise RuntimeError(msg)
                    else:
                        warnings.warn(msg, UserWarning)
            except Exception as e:
                msg = (
                    f"MMFF94 optimization failed: {e}. "
                    "MMFF parameters may not be available for this molecule."
                )
                if self.raise_on_failure:
                    raise RuntimeError(msg) from e
                else:
                    warnings.warn(msg, UserWarning)
        else:
            raise ValueError(
                f"Unknown force field: {self.forcefield}. Use 'UFF' or 'MMFF94'."
            )

        input.set_external(mol)

        if self.update_internal:
            # For geometry optimization, only update coordinates, preserve topology
            # This preserves bond types and other topology properties
            input.sync_to_internal(update_topology=False)

        return input


@dataclass
class Generate3D(Compute[RDKitAdapter, RDKitAdapter]):
    """RDKit-based 3D generation pipeline for RDKitAdapter.

    This compute node performs a configurable sequence of RDKit operations
    on a RDKitAdapter:
    - Add explicit hydrogens (optional)
    - Sanitize molecule (optional)
    - Generate 3D coordinates via embedding (optional)
    - Optimize geometry with force field (optional)

    The node **mutates** the passed RDKitAdapter, updating both the external
    Chem.Mol and optionally the internal Atomistic structure.

    Attributes:
        add_hydrogens: Whether to add explicit hydrogens before embedding
        sanitize: Whether to sanitize the molecule
        embed: Whether to perform 3D coordinate embedding
        optimize: Whether to optimize geometry after embedding
        max_embed_attempts: Maximum number of embedding attempts
        embed_random_seed: Random seed for embedding (None for random)
        max_opt_iters: Maximum optimization iterations
        forcefield: Force field to use ("UFF" or "MMFF94")
        update_internal: Whether to sync internal structure after modifications

    Examples:
        >>> from molpy.external import RDKitAdapter, Generate3D
        >>> from molpy import Atomistic
        >>>
        >>> # Create adapter from Atomistic
        >>> atomistic = Atomistic()
        >>> # ... add atoms and bonds ...
        >>> adapter = RDKitAdapter(internal=atomistic)
        >>>
        >>> # Generate 3D coordinates
        >>> op = Generate3D(
        ...     add_hydrogens=True,
        ...     sanitize=True,
        ...     embed=True,
        ...     optimize=True
        ... )
        >>> adapter = op(adapter)  # Mutates adapter
        >>>
        >>> # Access updated structure
        >>> updated_atomistic = adapter.get_internal()
    """

    add_hydrogens: bool = True
    sanitize: bool = True
    embed: bool = True
    optimize: bool = True

    max_embed_attempts: int = 10
    embed_random_seed: int | None = 0

    max_opt_iters: int = 200
    forcefield: str = "UFF"

    update_internal: bool = True

    def compute(self, input: RDKitAdapter) -> RDKitAdapter:
        """Execute the 3D generation pipeline.

        Args:
            input: RDKitAdapter to process

        Returns:
            The same RDKitAdapter instance (mutated)

        Raises:
            ValueError: If adapter has no external representation and sync fails
            RuntimeError: If embedding or optimization fails
        """
        if not input.has_external():
            input.sync_to_external()

        mol = input.get_external()
        mol = Chem.Mol(mol)

        if self.add_hydrogens:
            mol.UpdatePropertyCache(strict=False)
            # Store original atom count to identify newly added atoms
            original_count = mol.GetNumAtoms()
            mol = Chem.AddHs(mol, addCoords=True)
            # Assign MP_ID to newly added hydrogen atoms
            # Use negative indices to avoid conflicts with existing atoms
            for idx in range(original_count, mol.GetNumAtoms()):
                rd_atom = mol.GetAtomWithIdx(idx)
                if not rd_atom.HasProp(MP_ID):
                    # Use negative index as temporary id for new atoms
                    rd_atom.SetIntProp(MP_ID, -(idx - original_count + 1))

        if self.sanitize:
            try:
                Chem.SanitizeMol(mol)
            except Exception as e:
                raise RuntimeError(
                    f"Sanitization failed: {e}. "
                    "The molecule may have invalid valency or other issues."
                ) from e

        if self.embed:
            if mol.GetNumAtoms() == 0:
                raise ValueError("Cannot embed 3D coordinates for empty molecule")

            params = AllChem.ETKDGv3()  # type: ignore[attr-defined]
            if self.embed_random_seed is not None:
                params.randomSeed = int(self.embed_random_seed)
            params.useRandomCoords = True

            embed_result = AllChem.EmbedMolecule(mol, params)  # type: ignore[attr-defined]

            attempts = 1
            while embed_result == -1 and attempts < self.max_embed_attempts:
                params.useRandomCoords = True
                if self.embed_random_seed is not None:
                    params.randomSeed = int(self.embed_random_seed) + attempts
                embed_result = AllChem.EmbedMolecule(mol, params)  # type: ignore[attr-defined]
                attempts += 1

            if embed_result == -1:
                raise RuntimeError(
                    f"3D embedding failed after {self.max_embed_attempts} attempts. "
                    "The molecule may be too large or have structural issues."
                )

        if self.optimize:
            # Use OptimizeGeometry recipe for optimization
            optimizer = OptimizeGeometry(
                max_opt_iters=self.max_opt_iters,
                forcefield=self.forcefield,
                update_internal=False,  # We'll update internal at the end
                raise_on_failure=False,  # Don't raise, just warn
            )
            input.set_external(mol)
            input = optimizer(input)
            mol = input.get_external()

        input.set_external(mol)

        if self.update_internal:
            input.sync_to_internal()

        return input
