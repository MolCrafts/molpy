"""RDKit-based molecular operations for RDKitAdapter.

This module provides frozen-dataclass tools that operate on RDKitAdapter
instances.  ``Generate3D`` inherits ``Tool`` and is auto-registered in
``ToolRegistry``.  ``OptimizeGeometry`` is an internal helper (not a Tool).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import warnings

from rdkit import Chem
from rdkit.Chem import AllChem

from molpy.adapter.rdkit import MP_ID, RDKitAdapter
from molpy.tool.base import Tool


# ---------------------------------------------------------------------------
# Helpers (pure functions operating on Chem.Mol, returning new Mol)
# ---------------------------------------------------------------------------


def _sanitize(mol: Chem.Mol) -> Chem.Mol:
    """Sanitize *mol* for force-field readiness.

    Tries strict sanitization first; falls back to inferring hybridization
    from connectivity when that fails.

    Returns:
        A sanitized copy (the input is never mutated).

    Raises:
        RuntimeError: If both strict and fallback sanitization fail.
    """
    mol = Chem.Mol(mol)
    try:
        Chem.SanitizeMol(mol)
    except Exception as e:
        try:
            mol.UpdatePropertyCache(strict=False)
            for atom in mol.GetAtoms():
                if atom.GetHybridization() == Chem.HybridizationType.UNSPECIFIED:
                    degree = atom.GetDegree()
                    hyb = {
                        0: Chem.HybridizationType.S,
                        1: Chem.HybridizationType.SP,
                        2: Chem.HybridizationType.SP2,
                        3: Chem.HybridizationType.SP3,
                    }.get(degree, Chem.HybridizationType.SP3D)
                    atom.SetHybridization(hyb)
            mol.UpdatePropertyCache(strict=False)
        except Exception as e2:
            raise RuntimeError(
                f"Failed to prepare molecule for optimization: {e}. "
                f"Fallback also failed: {e2}. "
                "The molecule may have structural issues."
            ) from e
    return mol


def _optimize_uff(
    mol: Chem.Mol,
    max_iters: int,
    raise_on_failure: bool,
) -> Chem.Mol:
    """Run UFF optimization on a copy of *mol*.

    Returns:
        A new Mol with optimized coordinates.
    """
    mol = Chem.Mol(mol)
    mol.UpdatePropertyCache(strict=False)

    conf_before = mol.GetConformer() if mol.GetNumConformers() > 0 else None
    coords_before = None
    if conf_before is not None:
        coords_before = [
            conf_before.GetAtomPosition(i) for i in range(mol.GetNumAtoms())
        ]

    opt_result = AllChem.UFFOptimizeMolecule(  # type: ignore[attr-defined]
        mol, maxIters=int(max_iters)
    )

    coords_changed = False
    if coords_before is not None and mol.GetNumConformers() > 0:
        conf_after = mol.GetConformer()
        for i in range(mol.GetNumAtoms()):
            p0 = coords_before[i]
            p1 = conf_after.GetAtomPosition(i)
            if (
                abs(p0.x - p1.x) > 1e-5
                or abs(p0.y - p1.y) > 1e-5
                or abs(p0.z - p1.z) > 1e-5
            ):
                coords_changed = True
                break

    if opt_result != 0:
        msg = (
            f"UFF optimization returned code {opt_result}. "
            f"Code 1 typically means convergence not reached within {max_iters} iterations. "
            "The structure may still be improved."
        )
        if raise_on_failure:
            raise RuntimeError(msg)
        warnings.warn(msg, UserWarning)
    elif not coords_changed:
        _warn_unchanged_coords(mol)

    return mol


def _optimize_mmff(
    mol: Chem.Mol,
    max_iters: int,
    raise_on_failure: bool,
) -> Chem.Mol:
    """Run MMFF94 optimization on a copy of *mol*.

    Returns:
        A new Mol with optimized coordinates.
    """
    mol = Chem.Mol(mol)
    mol.UpdatePropertyCache(strict=False)

    try:
        opt_result = AllChem.MMFFOptimizeMolecule(  # type: ignore[attr-defined]
            mol, maxIters=int(max_iters)
        )
        if opt_result != 0:
            msg = (
                f"MMFF94 optimization returned code {opt_result}. "
                f"Code 1 typically means convergence not reached within {max_iters} iterations."
            )
            if raise_on_failure:
                raise RuntimeError(msg)
            warnings.warn(msg, UserWarning)
    except Exception as e:
        msg = (
            f"MMFF94 optimization failed: {e}. "
            "MMFF parameters may not be available for this molecule."
        )
        if raise_on_failure:
            raise RuntimeError(msg) from e
        warnings.warn(msg, UserWarning)

    return mol


def _warn_unchanged_coords(mol: Chem.Mol) -> None:
    """Emit a warning when optimization did not change coordinates."""
    max_bond_length = 0.0
    if mol.GetNumConformers() > 0:
        conf = mol.GetConformer()
        for bond in mol.GetBonds():
            p1 = conf.GetAtomPosition(bond.GetBeginAtomIdx())
            p2 = conf.GetAtomPosition(bond.GetEndAtomIdx())
            bl = ((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2 + (p1.z - p2.z) ** 2) ** 0.5
            max_bond_length = max(max_bond_length, bl)

    if max_bond_length > 2.0:
        warnings.warn(
            f"UFF optimization completed but coordinates did not change, "
            f"despite long bonds detected (max bond length: {max_bond_length:.3f} A). "
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


def _add_hydrogens(mol: Chem.Mol) -> Chem.Mol:
    """Add explicit hydrogens, assigning MP_ID to new atoms.

    Returns:
        A new Mol with explicit hydrogens.
    """
    mol = Chem.Mol(mol)
    mol.UpdatePropertyCache(strict=False)
    original_count = mol.GetNumAtoms()
    mol = Chem.AddHs(mol, addCoords=True)
    for idx in range(original_count, mol.GetNumAtoms()):
        rd_atom = mol.GetAtomWithIdx(idx)
        if not rd_atom.HasProp(MP_ID):
            rd_atom.SetIntProp(MP_ID, -(idx - original_count + 1))
    return mol


def _embed(
    mol: Chem.Mol,
    max_attempts: int,
    random_seed: int | None,
) -> Chem.Mol:
    """Embed 3D coordinates into a copy of *mol*.

    Returns:
        A new Mol with 3D coordinates.

    Raises:
        ValueError: If the molecule has no atoms.
        RuntimeError: If embedding fails after *max_attempts*.
    """
    mol = Chem.Mol(mol)
    if mol.GetNumAtoms() == 0:
        raise ValueError("Cannot embed 3D coordinates for empty molecule")

    params = AllChem.ETKDGv3()  # type: ignore[attr-defined]
    if random_seed is not None:
        params.randomSeed = int(random_seed)
    params.useRandomCoords = True

    embed_result = AllChem.EmbedMolecule(mol, params)  # type: ignore[attr-defined]
    attempts = 1
    while embed_result == -1 and attempts < max_attempts:
        params.useRandomCoords = True
        if random_seed is not None:
            params.randomSeed = int(random_seed) + attempts
        embed_result = AllChem.EmbedMolecule(mol, params)  # type: ignore[attr-defined]
        attempts += 1

    if embed_result == -1:
        raise RuntimeError(
            f"3D embedding failed after {max_attempts} attempts. "
            "The molecule may be too large or have structural issues."
        )
    return mol


# ---------------------------------------------------------------------------
# Compute nodes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OptimizeGeometry:  # Not a Tool — internal helper for Generate3D
    """RDKit-based geometry optimization for RDKitAdapter.

    Attributes:
        max_opt_iters: Maximum optimization iterations
        forcefield: Force field to use ("UFF" or "MMFF94")
        update_internal: Whether to sync internal structure after optimization
        raise_on_failure: Whether to raise exception on optimization failure

    Examples:
        >>> optimizer = OptimizeGeometry(forcefield="UFF", max_opt_iters=200)
        >>> result_adapter = optimizer(adapter)
    """

    max_opt_iters: int = 200
    forcefield: str = "UFF"
    update_internal: bool = True
    raise_on_failure: bool = False

    def run(self, input: RDKitAdapter) -> RDKitAdapter:
        new_adapter = input.copy()

        if not new_adapter.has_external():
            new_adapter.sync_to_external()

        original_mol = new_adapter.get_external()
        mol = Chem.Mol(original_mol)

        # Copy conformer explicitly if needed
        if mol.GetNumConformers() == 0 and original_mol.GetNumConformers() > 0:
            original_conf = original_mol.GetConformer()
            new_conf = Chem.Conformer(mol.GetNumAtoms())
            for i in range(mol.GetNumAtoms()):
                pos = original_conf.GetAtomPosition(i)
                new_conf.SetAtomPosition(i, pos)
            mol.AddConformer(new_conf, assignId=True)

        mol = _sanitize(mol)

        if mol.GetNumConformers() == 0:
            raise ValueError(
                "Cannot optimize geometry: no conformer found. "
                "Enable embedding or provide a molecule with coordinates."
            )

        if self.forcefield == "UFF":
            mol = _optimize_uff(mol, self.max_opt_iters, self.raise_on_failure)
        elif self.forcefield == "MMFF94":
            mol = _optimize_mmff(mol, self.max_opt_iters, self.raise_on_failure)
        else:
            raise ValueError(
                f"Unknown force field: {self.forcefield}. Use 'UFF' or 'MMFF94'."
            )

        new_adapter.set_external(mol)

        if self.update_internal:
            new_adapter.sync_to_internal(update_topology=False)

        return new_adapter

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.run(*args, **kwargs)


@dataclass(frozen=True)
class Generate3D(Tool):
    """RDKit-based 3D generation pipeline for RDKitAdapter.

    Pipeline stages (each optional):
    1. Add explicit hydrogens
    2. Sanitize molecule
    3. Generate 3D coordinates via embedding
    4. Optimize geometry with force field

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
        >>> op = Generate3D(add_hydrogens=True, embed=True, optimize=True)
        >>> result_adapter = op(adapter)
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

    def run(self, input: RDKitAdapter) -> RDKitAdapter:
        new_adapter = input.copy()

        if not new_adapter.has_external():
            new_adapter.sync_to_external()

        mol = new_adapter.get_external()
        mol = Chem.Mol(mol)

        if self.add_hydrogens:
            mol = _add_hydrogens(mol)

        if self.sanitize:
            try:
                Chem.SanitizeMol(mol)
            except Exception as e:
                raise RuntimeError(
                    f"Sanitization failed: {e}. "
                    "The molecule may have invalid valency or other issues."
                ) from e

        if self.embed:
            mol = _embed(mol, self.max_embed_attempts, self.embed_random_seed)

        if self.optimize:
            new_adapter.set_external(mol)
            optimizer = OptimizeGeometry(
                max_opt_iters=self.max_opt_iters,
                forcefield=self.forcefield,
                update_internal=False,
                raise_on_failure=False,
            )
            # OptimizeGeometry.run already creates a copy, but we pass new_adapter
            # which is already a copy, so we use it directly
            if not new_adapter.has_external():
                new_adapter.sync_to_external()
            original_mol = new_adapter.get_external()
            opt_mol = Chem.Mol(original_mol)
            if opt_mol.GetNumConformers() == 0 and original_mol.GetNumConformers() > 0:
                original_conf = original_mol.GetConformer()
                new_conf = Chem.Conformer(opt_mol.GetNumAtoms())
                for i in range(opt_mol.GetNumAtoms()):
                    pos = original_conf.GetAtomPosition(i)
                    new_conf.SetAtomPosition(i, pos)
                opt_mol.AddConformer(new_conf, assignId=True)
            opt_mol = _sanitize(opt_mol)
            if opt_mol.GetNumConformers() > 0:
                if self.forcefield == "UFF":
                    opt_mol = _optimize_uff(opt_mol, self.max_opt_iters, False)
                elif self.forcefield == "MMFF94":
                    opt_mol = _optimize_mmff(opt_mol, self.max_opt_iters, False)
            mol = opt_mol

        new_adapter.set_external(mol)

        if self.update_internal:
            new_adapter.sync_to_internal()

        return new_adapter
