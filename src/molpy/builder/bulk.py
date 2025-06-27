import numpy as np
from itertools import product
from typing import Sequence, Iterable, Optional

from molpy.builder.base import StructBuilder, BaseSiteProvider
from molpy.core.atomistic import AtomicStruct
from molpy.core.region import Region


class CrystalSite(BaseSiteProvider):
    """Site represented by a cell matrix and fractional basis."""

    def __init__(self, cell: Sequence[Sequence[float]], basis: Iterable[Sequence[float]]):
        self.cell = np.asarray(cell, dtype=float)
        if self.cell.shape != (3, 3):
            raise ValueError("cell must be a 3x3 matrix")
        basis = np.asarray(list(basis), dtype=float)
        if basis.ndim == 1:
            basis = basis.reshape(1, 3)
        if basis.shape[1] != 3:
            raise ValueError("basis must be of shape (n, 3)")
        self.basis = basis

    def _get_bounds(self, region: Region):
        bounds = np.asarray(region.bounds, dtype=float)
        if bounds.shape == (2, 3):
            lo, hi = bounds[0], bounds[1]
        elif bounds.shape == (3, 2):
            lo, hi = bounds[:, 0], bounds[:, 1]
        else:
            raise ValueError("Unknown bounds shape")
        return lo, hi

    def gen_site(self, region: Region) -> np.ndarray:
        """Return all lattice positions within *region*."""
        lo, hi = self._get_bounds(region)

        corners = np.array(list(product(*zip(lo, hi))))
        inv_cell = np.linalg.inv(self.cell)
        frac_corners = corners @ inv_cell
        fmin = frac_corners.min(axis=0)
        fmax = frac_corners.max(axis=0)

        n_min = []
        n_max = []
        eps = 1e-8
        for i in range(3):
            lower = np.min(np.ceil(fmin[i] - self.basis[:, i] - eps))
            upper = np.max(np.floor(fmax[i] - self.basis[:, i] - eps))
            n_min.append(int(lower))
            n_max.append(int(upper))
        n_min = np.array(n_min, dtype=int)
        n_max = np.array(n_max, dtype=int)

        positions = []
        for nx in range(n_min[0], n_max[0] + 1):
            for ny in range(n_min[1], n_max[1] + 1):
                for nz in range(n_min[2], n_max[2] + 1):
                    trans = np.array([nx, ny, nz], dtype=float)
                    for b in self.basis:
                        frac = b + trans
                        xyz = frac @ self.cell
                        if region.isin(np.array([xyz]))[0]:
                            positions.append(xyz)
        return np.array(positions)

class CrystalBuilder(StructBuilder):
    """Replicate a template structure on a crystal lattice."""

    def __init__(self, lattice: CrystalSite):
        super().__init__(lattice)
        self.lattice = lattice  # Keep for backward compatibility

    def build(self, region: Region, template: AtomicStruct, name: str = "crystal") -> AtomicStruct:
        """
        Build crystal structure by filling region with template at lattice sites.
        
        Args:
            region: Region to fill
            template: Template structure to replicate
            name: Name for resulting structure
            
        Returns:
            Crystal structure
        """
        return self.fill_basis(region, template, name)

    def fill_basis(self, region: Region, template: AtomicStruct, name: str = "crystal") -> AtomicStruct:
        """
        Fill the region by placing the template at each lattice site (including basis offsets).
        """
        positions = self.lattice.gen_site(region)
        return self._place_template_at_positions(positions, template, name)

    def fill_lattice(self, region: Region, template: AtomicStruct, name: str = "crystal_cell") -> AtomicStruct:
        """
        Fill the region by placing the entire template at each unit cell origin (without basis offsets).
        """
        lo, hi = self.lattice._get_bounds(region)
        corners = np.array(list(product(*zip(lo, hi))))
        inv_cell = np.linalg.inv(self.lattice.cell)
        frac_corners = corners @ inv_cell
        fmin = frac_corners.min(axis=0)
        fmax = frac_corners.max(axis=0)
        n_min = np.ceil(fmin - 1e-8).astype(int)
        n_max = np.floor(fmax + 1e-8).astype(int)
        positions = []
        for nx in range(n_min[0], n_max[0] + 1):
            for ny in range(n_min[1], n_max[1] + 1):
                for nz in range(n_min[2], n_max[2] + 1):
                    frac = np.array([nx, ny, nz], dtype=float)
                    xyz = frac @ self.lattice.cell
                    if region.isin(np.array([xyz]))[0]:
                        positions.append(xyz)
        return self._place_template_at_positions(np.array(positions), template, name)

class FCCBuilder(CrystalBuilder):
    """Face-centered cubic lattice builder."""

    def __init__(self, a: float):
        cell = np.diag([a, a, a]).tolist()
        basis = [
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.5],
        ]
        super().__init__(CrystalSite(cell, basis))

class BCCBuilder(CrystalBuilder):
    """Body-centered cubic lattice builder."""

    def __init__(self, a: float):
        cell = np.diag([a, a, a]).tolist()
        basis = [
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
        ]
        super().__init__(CrystalSite(cell, basis))

class HCPBuilder(CrystalBuilder):
    """Hexagonal close-packed lattice builder."""

    def __init__(self, a: float, c: float):
        cell = np.array([
            [a, 0.0, 0.0],
            [0.5 * a, np.sqrt(3) / 2 * a, 0.0],
            [0.0, 0.0, c],
        ]).tolist()
        basis = [
            [0.0, 0.0, 0.0],
            [2.0 / 3.0, 1.0 / 3.0, 0.5],
        ]
        super().__init__(CrystalSite(cell, basis))

def bulk(symbol: str, crystalstructure: str, a: Optional[float] = None, c: Optional[float] = None, region: Optional[Region] = None):
    cs = crystalstructure.lower()
    if cs == "fcc":
        if a is None:
            raise ValueError("Parameter 'a' must be provided for fcc")
        builder = FCCBuilder(a)
    elif cs == "bcc":
        if a is None:
            raise ValueError("Parameter 'a' must be provided for bcc")
        builder = BCCBuilder(a)
    elif cs == "hcp":
        if a is None or c is None:
            raise ValueError("Parameters 'a' and 'c' must be provided for hcp")
        builder = HCPBuilder(a, c)
    else:
        raise ValueError(f"Unknown crystal structure: {crystalstructure}")

    if region is None:
        return builder

    template = AtomicStruct(symbol)
    template.def_atom(name=symbol, element=symbol, xyz=[0.0, 0.0, 0.0])
    return builder.fill_basis(region, template)

class RandomWalkSite:
    """
    Self-avoiding random walk position generator.
    
    Generates a chain of positions using self-avoiding random walk algorithm,
    suitable for polymer chain generation.
    """
    
    def __init__(
        self, 
        step_length: float = 1.0,
        max_attempts: int = 1000,
        seed: Optional[int] = None
    ):
        """
        Initialize the random walk generator.
        
        Args:
            step_length: Distance between consecutive steps
            max_attempts: Maximum attempts to find valid next step
            seed: Random seed for reproducible results
        """
        self.step_length = step_length
        self.max_attempts = max_attempts
        self.rng = np.random.RandomState(seed)
        
        # Define possible step directions (6 directions + diagonal)
        self.directions = np.array([
            [1, 0, 0], [-1, 0, 0],  # ±x
            [0, 1, 0], [0, -1, 0],  # ±y
            [0, 0, 1], [0, 0, -1],  # ±z
            # Add some diagonal directions for more flexibility
            [1, 1, 0], [1, -1, 0], [-1, 1, 0], [-1, -1, 0],
            [1, 0, 1], [1, 0, -1], [-1, 0, 1], [-1, 0, -1],
            [0, 1, 1], [0, 1, -1], [0, -1, 1], [0, -1, -1],
        ], dtype=float)
        
        # Normalize to step_length
        norms = np.linalg.norm(self.directions, axis=1)
        self.directions = self.directions / norms[:, np.newaxis] * step_length
    
    def gen_site(
        self, 
        n_steps: int, 
        start_pos: Optional[np.ndarray] = None,
        exclusion_radius: Optional[float] = None
    ) -> np.ndarray:
        """
        Generate positions using self-avoiding random walk.
        
        Args:
            n_steps: Number of steps in the walk
            start_pos: Starting position (default: origin)
            exclusion_radius: Minimum distance between positions (default: step_length/2)
            
        Returns:
            Array of positions with shape (n_steps+1, 3)
        """
        if start_pos is None:
            start_pos = np.array([0.0, 0.0, 0.0])
        
        if exclusion_radius is None:
            exclusion_radius = self.step_length * 0.6
        
        positions = [start_pos.copy()]
        current_pos = start_pos.copy()
        
        for step in range(n_steps):
            valid_step_found = False
            
            for attempt in range(self.max_attempts):
                # Choose random direction
                direction_idx = self.rng.randint(len(self.directions))
                next_pos = current_pos + self.directions[direction_idx]
                
                # Check self-avoidance
                if self._is_valid_position(next_pos, positions, exclusion_radius):
                    positions.append(next_pos)
                    current_pos = next_pos
                    valid_step_found = True
                    break
            
            if not valid_step_found:
                print(f"Warning: Could not find valid step at position {step+1}, "
                      f"terminating walk early with {len(positions)} positions")
                break
        
        return np.array(positions)
    
    def _is_valid_position(
        self, 
        new_pos: np.ndarray, 
        existing_positions: list, 
        exclusion_radius: float
    ) -> bool:
        """Check if new position satisfies self-avoidance constraint."""
        for pos in existing_positions:
            distance = np.linalg.norm(new_pos - pos)
            if distance < exclusion_radius:
                return False
        return True


class RandomWalkBuilder(StructBuilder):
    """
    Builder for creating polymer chains using self-avoiding random walk.
    """
    
    def __init__(
        self, 
        step_length: float = 1.5,
        max_attempts: int = 1000,
        seed: Optional[int] = None
    ):
        """
        Initialize the random walk builder.
        
        Args:
            step_length: Distance between monomers
            max_attempts: Maximum attempts per step
            seed: Random seed for reproducibility
        """
        walk_generator = RandomWalkSite(step_length, max_attempts, seed)
        super().__init__(walk_generator)
    
    def build(
        self, 
        monomer_template: AtomicStruct,
        n_monomers: int,
        start_position: Optional[np.ndarray] = None,
        name: str = "random_walk_polymer"
    ) -> AtomicStruct:
        """
        Build a polymer chain using self-avoiding random walk.
        
        Args:
            monomer_template: Template structure for each monomer
            n_monomers: Number of monomers in the chain
            start_position: Starting position for the walk
            name: Name for the resulting structure
            
        Returns:
            Polymer structure with monomers placed along random walk
        """
        # Generate positions (n_monomers positions from n_monomers-1 steps)
        positions = self.site_provider.gen_site(
            n_steps=n_monomers - 1,
            start_pos=start_position
        )
        
        # Place monomers at generated positions
        return self._place_template_at_positions(positions, monomer_template, name)


class SAWPolymerBuilder(RandomWalkBuilder):
    """
    Specialized builder for self-avoiding walk polymers with connectivity.
    """
    
    def build_connected_polymer(
        self,
        monomer_template: AtomicStruct,
        n_monomers: int,
        bond_length: float = 1.5,
        start_position: Optional[np.ndarray] = None,
        name: str = "saw_polymer"
    ) -> AtomicStruct:
        """
        Build a polymer with explicit bonds between consecutive monomers.
        
        Args:
            monomer_template: Template for each monomer unit
            n_monomers: Number of monomers
            bond_length: Target bond length between monomers
            start_position: Starting position
            name: Structure name
            
        Returns:
            Connected polymer structure
        """
        # Build the basic structure
        polymer = self.build(monomer_template, n_monomers, start_position, name)
        
        # Add bonds between consecutive monomers
        # This is a simplified version - in practice, you'd need to identify
        # connection points on each monomer and create proper bonds
        n_atoms_per_monomer = len(monomer_template.atoms)
        
        for i in range(n_monomers - 1):
            # Connect last atom of monomer i to first atom of monomer i+1
            atom1_idx = (i + 1) * n_atoms_per_monomer - 1  # Last atom of monomer i
            atom2_idx = (i + 1) * n_atoms_per_monomer      # First atom of monomer i+1
            
            if atom1_idx < len(polymer.atoms) and atom2_idx < len(polymer.atoms):
                from molpy.core.atomistic import Bond
                bond = Bond(polymer.atoms[atom1_idx], polymer.atoms[atom2_idx])
                bond["length"] = bond_length
                polymer.bonds.add(bond)
        
        return polymer
