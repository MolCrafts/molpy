"""
Core molecular structure classes for the molpy framework.

This module provides the fundamental building blocks for molecular modeling:
- Entity: Base class with dictionary-like behavior
- Spatial operations for atoms and structures
- Topological entities: bonds, angles, dihedrals
- Hierarchical structure management
- Collections and containers
"""

from typing import Callable, Optional

import numpy as np
from numpy.typing import ArrayLike

from .protocol import Entity, Entities, Struct
from .topology import Topology

class Atom(Entity):
    """
    Class representing an atom with spatial coordinates.
    
    Combines Entity's dictionary behavior with spatial operations through wrappers.
    """

    def __init__(self, name: str = "", xyz: Optional[ArrayLike] = None, **kwargs):
        """
        Initialize an atom.
        
        Args:
            name: Atom name/symbol
            xyz: 3D coordinates
            **kwargs: Additional properties
        """
        super().__init__(name=name, **kwargs)
        if xyz is not None:
            self.xyz = xyz

    def __repr__(self) -> str:
        """Return a string representation of the atom."""
        return f"<Atom {self.get('name', 'unnamed')}>"

    @property
    def xyz(self) -> np.ndarray:
        """Get the xyz coordinates as a numpy array."""
        return np.array(self.get("xyz", [0.0, 0.0, 0.0]), dtype=float)

    @xyz.setter
    def xyz(self, value: ArrayLike) -> None:
        """Set the xyz coordinates."""
        value = np.asarray(value, dtype=float)
        if value.shape != (3,):
            raise ValueError("xyz must be a 3D vector")
        self["xyz"] = value.tolist()  # Store as list for easier comparison

    @property
    def name(self) -> str:
        """Get the name of the atom."""
        return self.get("name", "")


class ManyBody(Entity):
    """
    Base class for entities involving multiple atoms.
    
    Handles bonds, angles, dihedrals, and other multi-atom entities.
    """

    def __init__(self, *atoms, **kwargs):
        """
        Initialize a ManyBody entity.

        Args:
            *atoms: Atoms involved in the entity
            **kwargs: Additional properties
        """
        super().__init__(**kwargs)
        if not all(isinstance(atom, Atom) for atom in atoms):
            raise TypeError("All arguments must be Atom instances")
        self._atoms = tuple(atoms)

    def __hash__(self) -> int:
        """Return a hash based on constituent atoms and class."""
        return hash(self._atoms) + hash(self.__class__.__name__)

    @property
    def atoms(self):
        """Get the atoms involved in the entity."""
        return self._atoms


class Bond(ManyBody):
    """Class representing a bond between two atoms."""

    def __init__(self, atom1=None, atom2=None, itom=None, jtom=None, **kwargs):
        """
        Initialize a bond between two atoms.

        Args:
            atom1: First atom in the bond
            atom2: Second atom in the bond
            itom: Alternative name for first atom (for backward compatibility)
            jtom: Alternative name for second atom (for backward compatibility)
            **kwargs: Additional properties (e.g., bond_type, length)
        """
        # Handle backward compatibility with itom/jtom naming
        if itom is not None:
            atom1 = itom
        if jtom is not None:
            atom2 = jtom
            
        if atom1 is None or atom2 is None:
            raise ValueError("Must provide both atoms for the bond")
        if atom1 is atom2:
            raise ValueError("Cannot create bond between same atom")
        # Sort atoms for consistent ordering
        sorted_atoms = sorted([atom1, atom2], key=lambda a: id(a))
        super().__init__(*sorted_atoms, **kwargs)

    @property
    def atom1(self):
        """Get the first atom in the bond."""
        return self._atoms[0]

    @property
    def atom2(self):
        """Get the second atom in the bond."""
        return self._atoms[1]

    # Aliases for backward compatibility
    @property
    def itom(self):
        """Alias for atom1."""
        return self.atom1

    @property
    def jtom(self):
        """Alias for atom2."""
        return self.atom2

    def __repr__(self) -> str:
        """Return a string representation of the bond."""
        return f"<Bond: {self.atom1.name}-{self.atom2.name}>"

    def __eq__(self, other) -> bool:
        """Check equality based on the atoms in the bond."""
        if not isinstance(other, Bond):
            return False
        return (self.atom1 is other.atom1 and self.atom2 is other.atom2) or \
               (self.atom1 is other.atom2 and self.atom2 is other.atom1)

    @property
    def length(self) -> float:
        """Calculate the bond length."""
        return float(np.linalg.norm(self.atom1.xyz - self.atom2.xyz))

    def to_dict(self) -> dict:
        """Convert the bond to a dictionary."""
        result = super().to_dict()
        # Add atom indices if atoms have ids
        if hasattr(self.atom1, 'get') and 'id' in self.atom1:
            result['i'] = self.atom1['id']
        if hasattr(self.atom2, 'get') and 'id' in self.atom2:
            result['j'] = self.atom2['id']
        return result


class Angle(ManyBody):
    """Class representing an angle between three atoms."""

    def __init__(self, atom1: Atom, vertex: Atom, atom2: Atom, **kwargs):
        """
        Initialize an angle.

        Args:
            atom1: First atom in the angle
            vertex: Vertex atom (center of angle)
            atom2: Third atom in the angle
            **kwargs: Additional properties
        """
        if len({id(atom1), id(vertex), id(atom2)}) != 3:
            raise ValueError("All three atoms must be different")
        # Sort end atoms for consistent ordering
        end_atoms = sorted([atom1, atom2], key=lambda a: id(a))
        super().__init__(end_atoms[0], vertex, end_atoms[1], **kwargs)

    @property
    def atom1(self):
        """Get the first atom in the angle."""
        return self._atoms[0]

    @property
    def vertex(self):
        """Get the vertex atom (center of angle)."""
        return self._atoms[1]

    @property
    def atom2(self):
        """Get the third atom in the angle."""
        return self._atoms[2]

    # Aliases for backward compatibility
    @property
    def itom(self):
        """Alias for atom1."""
        return self.atom1

    @property
    def jtom(self):
        """Alias for vertex."""
        return self.vertex

    @property
    def ktom(self):
        """Alias for atom2."""
        return self.atom2

    def __repr__(self) -> str:
        """Return a string representation of the angle."""
        return f"<Angle: {self.atom1.name}-{self.vertex.name}-{self.atom2.name}>"

    @property
    def value(self) -> float:
        """Calculate the angle value in radians."""
        v1 = self.atom1.xyz - self.vertex.xyz
        v2 = self.atom2.xyz - self.vertex.xyz
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        # Clamp to handle numerical errors
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return float(np.arccos(cos_angle))

    def to_dict(self) -> dict:
        """Convert the angle to a dictionary."""
        result = super().to_dict()
        if hasattr(self.atom1, 'get') and 'id' in self.atom1:
            result.update({
                'i': self.atom1["id"],
                'j': self.vertex["id"], 
                'k': self.atom2["id"]
            })
        return result


class Dihedral(ManyBody):
    """Class representing a dihedral angle between four atoms."""

    def __init__(self, atom1: Atom, atom2: Atom, atom3: Atom, atom4: Atom, **kwargs):
        """
        Initialize a dihedral angle.

        Args:
            atom1: First atom in the dihedral
            atom2: Second atom in the dihedral
            atom3: Third atom in the dihedral  
            atom4: Fourth atom in the dihedral
            **kwargs: Additional properties
        """
        if len({id(atom1), id(atom2), id(atom3), id(atom4)}) != 4:
            raise ValueError("All four atoms must be different")
        
        # Ensure consistent ordering based on central bond
        if id(atom2) > id(atom3):
            atom1, atom2, atom3, atom4 = atom4, atom3, atom2, atom1
            
        super().__init__(atom1, atom2, atom3, atom4, **kwargs)

    @property
    def atom1(self):
        """Get the first atom in the dihedral."""
        return self._atoms[0]

    @property
    def atom2(self):
        """Get the second atom in the dihedral."""
        return self._atoms[1]

    @property
    def atom3(self):
        """Get the third atom in the dihedral."""
        return self._atoms[2]

    @property
    def atom4(self):
        """Get the fourth atom in the dihedral."""
        return self._atoms[3]

    # Aliases for backward compatibility
    @property
    def itom(self):
        """Alias for atom1."""
        return self.atom1

    @property
    def jtom(self):
        """Alias for atom2."""
        return self.atom2

    @property
    def ktom(self):
        """Alias for atom3."""
        return self.atom3

    @property
    def ltom(self):
        """Alias for atom4."""
        return self.atom4

    def __repr__(self) -> str:
        """Return a string representation of the dihedral."""
        return f"<Dihedral: {self.atom1.name}-{self.atom2.name}-{self.atom3.name}-{self.atom4.name}>"

    @property
    def value(self) -> float:
        """Calculate the dihedral angle value in radians."""
        # Vectors along the bonds
        b1 = self.atom2.xyz - self.atom1.xyz
        b2 = self.atom3.xyz - self.atom2.xyz
        b3 = self.atom4.xyz - self.atom3.xyz
        
        # Normal vectors to the planes
        n1 = np.cross(b1, b2)
        n2 = np.cross(b2, b3)
        
        # Normalize normal vectors
        n1_norm = np.linalg.norm(n1)
        n2_norm = np.linalg.norm(n2)
        
        if n1_norm < 1e-10 or n2_norm < 1e-10:
            return 0.0  # Degenerate case
            
        n1 = n1 / n1_norm
        n2 = n2 / n2_norm
        
        # Calculate dihedral angle
        cos_angle = np.dot(n1, n2)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)
        
        # Check sign using the middle bond vector
        if np.dot(np.cross(n1, n2), b2) < 0:
            angle = -angle
            
        return float(angle)

    def to_dict(self) -> dict:
        """Convert the dihedral to a dictionary."""
        result = super().to_dict()
        if all(hasattr(atom, 'get') and 'id' in atom for atom in self._atoms):
            result.update({
                'i': self.atom1["id"],
                'j': self.atom2["id"],
                'k': self.atom3["id"],
                'l': self.atom4["id"]
            })
        return result


class Improper(Dihedral):
    """
    Class representing an improper dihedral angle.
    
    An improper dihedral is used to maintain planarity in molecular structures.
    """

    def __init__(self, center: Atom, atom1: Atom, atom2: Atom, atom3: Atom, **kwargs):
        """
        Initialize an improper dihedral angle.

        Args:
            center: Central atom
            atom1: First bonded atom
            atom2: Second bonded atom 
            atom3: Third bonded atom
            **kwargs: Additional properties
        """
        # Sort the three bonded atoms for consistent ordering
        bonded_atoms = sorted([atom1, atom2, atom3], key=lambda a: id(a))
        super().__init__(center, bonded_atoms[0], bonded_atoms[1], bonded_atoms[2], **kwargs)

    def __repr__(self) -> str:
        """Return a string representation of the improper dihedral."""
        return f"<Improper: {self.atom1.name}({self.atom2.name},{self.atom3.name},{self.atom4.name})>"


class AtomicStructure(Struct):
    """
    Structure containing atoms, bonds, angles, and dihedrals.
    
    Basic structure functionality that can be enhanced with wrappers
    for spatial operations and hierarchical management.
    """

    def __init__(self, name: str = "", **props):
        """
        Initialize an atomic structure.
        
        Args:
            name: Structure name
            **props: Additional properties
        """
        super().__init__(name=name, **props)
        self["atoms"] = Entities()
        self["bonds"] = Entities()
        self["angles"] = Entities()
        self["dihedrals"] = Entities()

    def __repr__(self) -> str:
        """Return a string representation of the structure."""
        return f"<AtomicStructure: {len(self.atoms)} atoms>"

    @property
    def atoms(self):
        """Get the atoms in the structure."""
        return self["atoms"]

    @property
    def bonds(self):
        """Get the bonds in the structure."""
        return self["bonds"]

    @property
    def angles(self):
        """Get the angles in the structure."""
        return self["angles"]

    @property
    def dihedrals(self):
        """Get the dihedrals in the structure."""
        return self["dihedrals"]

    @property
    def xyz(self) -> np.ndarray:
        """Get the coordinates of all atoms in the structure."""
        if len(self.atoms) == 0:
            return np.array([]).reshape(0, 3)
        return np.array([atom.xyz for atom in self.atoms], dtype=float)

    @xyz.setter
    def xyz(self, value: ArrayLike) -> None:
        """
        Set the coordinates of all atoms in the structure.
        
        Args:
            value: Array of shape (n_atoms, 3) with new coordinates
        """
        value = np.asarray(value)
        if value.shape != (len(self.atoms), 3):
            raise ValueError(f"xyz must have shape ({len(self.atoms)}, 3), got {value.shape}")
        
        for i, atom in enumerate(self.atoms):
            atom.xyz = value[i]

    def add_atom(self, atom):
        """
        Add an atom to the structure.
        
        Args:
            atom: Atom to add
            
        Returns:
            The added atom
        """
        return self.atoms.add(atom)

    def def_atom(self, **props):
        """
        Create and add an atom with given properties.
        
        Args:
            **props: Atom properties
            
        Returns:
            The created atom
        """
        atom = Atom(**props)
        return self.add_atom(atom)

    def add_atoms(self, atoms):
        """
        Add multiple atoms to the structure.
        
        Args:
            atoms: Sequence of atoms to add
        """
        self.atoms.extend(atoms)

    def add_bond(self, bond):
        """
        Add a bond to the structure.
        
        Args:
            bond: Bond to add
            
        Returns:
            The added bond
        """
        return self.bonds.add(bond)

    def def_bond(self, atom1, atom2, **kwargs):
        """
        Create and add a bond between two atoms.
        
        Args:
            atom1: First atom or its index
            atom2: Second atom or its index
            **kwargs: Bond properties
            
        Returns:
            The created bond
        """
        if isinstance(atom1, int):
            atom1 = self.atoms[atom1]
        if isinstance(atom2, int):
            atom2 = self.atoms[atom2]
            
        if not isinstance(atom1, Atom) or not isinstance(atom2, Atom):
            raise TypeError("Arguments must be Atom instances or valid indices")
            
        bond = Bond(atom1, atom2, **kwargs)
        return self.add_bond(bond)

    def add_bonds(self, bonds):
        """
        Add multiple bonds to the structure.
        
        Args:
            bonds: Sequence of bonds to add
        """
        self.bonds.extend(bonds)

    def add_angle(self, angle):
        """
        Add an angle to the structure.
        
        Args:
            angle: Angle to add
            
        Returns:
            The added angle
        """
        return self.angles.add(angle)

    def add_angles(self, angles):
        """
        Add multiple angles to the structure.
        
        Args:
            angles: Sequence of angles to add
        """
        self.angles.extend(angles)

    def add_dihedral(self, dihedral):
        """
        Add a dihedral to the structure.
        
        Args:
            dihedral: Dihedral to add
            
        Returns:
            The added dihedral
        """
        return self.dihedrals.add(dihedral)

    def add_dihedrals(self, dihedrals):
        """
        Add multiple dihedrals to the structure.
        
        Args:
            dihedrals: Sequence of dihedrals to add
        """
        self.dihedrals.extend(dihedrals)

    def remove_atom(self, atom):
        """
        Remove an atom from the structure.
        
        Args:
            atom: Atom instance, index, or name to remove
        """
        self.atoms.remove(atom)

    def remove_bond(self, bond):
        """
        Remove a bond from the structure.
        
        Args:
            bond: Bond instance or tuple of atoms
        """
        if isinstance(bond, tuple):
            # Find and remove bond between these atoms
            atom1, atom2 = bond
            target_bond = self.bonds.get_by(
                lambda b: (b.atom1 is atom1 and b.atom2 is atom2) or
                         (b.atom1 is atom2 and b.atom2 is atom1)
            )
            if target_bond:
                self.bonds.remove(target_bond)
        else:
            self.bonds.remove(bond)

    def get_atom_by(self, condition: Callable):
        """
        Get an atom based on a condition.

        Args:
            condition: Function that takes an atom and returns a boolean

        Returns:
            The first atom that satisfies the condition, or None
        """
        return self.atoms.get_by(condition)

    def get_topology(self, attrs=None):
        """
        Get the topology of the structure.

        Args:
            attrs: List of atom attributes to include

        Returns:
            A Topology object representing the structure's topology.
        """
        # Create a simple topology-like object for testing
        from .topology import Topology

        topo = Topology()
        atoms = {atom: i for i, atom in enumerate(self.atoms)}
        atom_attrs = {}

        if attrs:
            for attr in attrs:
                atom_attrs[attr] = [atom[attr] for atom in atoms]
        topo.add_atoms(len(atoms), **atom_attrs)
        bonds = self["bonds"]
        topo.add_bonds([(atoms[bond.itom], atoms[bond.jtom]) for bond in bonds])
        return topo
    
    def gen_topo_items(self, topo: Topology | None = None, is_angle: bool = False, is_dihedral: bool = False):
        """
        Generate topology items for angles or dihedrals and add them to the structure.

        Args:
            topo: Topology object to use. If None, creates from current structure.
            is_angle: Whether to generate angle items
            is_dihedral: Whether to generate dihedral items

        Returns:
            List of generated topology items (Angle or Dihedral objects)
        """
        if topo is None:
            topo = self.get_topology()
        
        generated_items = []
        
        if is_angle:
            # Generate angles using the dedicated method
            angles = self.gen_angles(topo)
            for angle in angles:
                self.angles.add(angle)
            generated_items.extend(angles)
        
        if is_dihedral:
            # Generate dihedrals using the dedicated method
            dihedrals = self.gen_dihedrals(topo)
            for dihedral in dihedrals:
                self.dihedrals.add(dihedral)
            generated_items.extend(dihedrals)
        
        return generated_items

    def add_struct(self, struct):
        """
        Add another structure to the current structure.
        
        This merges the atoms and bonds from the other structure.

        Args:
            struct: The structure to add
            
        Returns:
            Self for method chaining
        """
        # Merge atoms and topological entities
        self.add_atoms(struct.atoms)
        self.add_bonds(struct.bonds)
        self.add_angles(struct.angles)
        self.add_dihedrals(struct.dihedrals)
        
        return self

    @classmethod
    def concat(cls, name: str, structs):
        """
        Concatenate multiple structures into a new structure.

        Args:
            name: Name for the new structure
            structs: Sequence of structures to concatenate
            
        Returns:
            New structure containing all input structures
        """
        result = cls(name)
        for struct in structs:
            result.add_struct(struct)
        return result

    def to_frame(self):
        """
        Convert the structure to a Frame object.

        Returns:
        - A Frame object containing the structure's data.
        """
        from .frame import Frame

        frame = Frame()

        # Handle empty structure case
        if "atoms" not in self or len(self["atoms"]) == 0:
            # Create empty dict for atoms
            frame["atoms"] = {}
            return frame

        # Set atom IDs and create name-to-index mapping
        atom_name_idx_mapping = {}
        atom_dicts = []
        for i, atom in enumerate(self["atoms"]):
            atom["id"] = i
            atom_name_idx_mapping[atom["name"]] = i
            atom_dicts.append(atom.to_dict())

        # Create atoms data directly as dict for xarray
        atoms_data = {}
        if atom_dicts:
            # Collect all keys from atom dictionaries
            all_keys = set()
            for atom_dict in atom_dicts:
                all_keys.update(atom_dict.keys())
            
            # Build arrays for each property
            for key in all_keys:
                values = [atom_dict.get(key) for atom_dict in atom_dicts]
                atoms_data[key] = values

        frame["atoms"] = atoms_data
        
        # Create bonds data if bonds exist
        if "bonds" in self and len(self["bonds"]) > 0:
            bdicts = []
            for bond in self["bonds"]:
                bdict = bond.to_dict()
                # Get atom indices by name
                iname = bond.itom["name"] if "name" in bond.itom else bond.itom.get("name", "")
                jname = bond.jtom["name"] if "name" in bond.jtom else bond.jtom.get("name", "")
                if iname in atom_name_idx_mapping and jname in atom_name_idx_mapping:
                    bdict["i"] = atom_name_idx_mapping[iname]
                    bdict["j"] = atom_name_idx_mapping[jname]
                bdicts.append(bdict)
            
            if bdicts:
                # Create bonds data directly as dict
                bonds_data = {}
                all_bond_keys = set()
                for bdict in bdicts:
                    all_bond_keys.update(bdict.keys())
                
                for key in all_bond_keys:
                    bonds_data[key] = [bdict.get(key) for bdict in bdicts]
                
                # Add sequential ids
                bonds_data["id"] = list(range(len(bdicts)))
                frame["bonds"] = bonds_data

        # Create angles data if angles exist
        if "angles" in self and len(self["angles"]) > 0:
            adicts = []
            for angle in self["angles"]:
                adict = angle.to_dict()
                # Get atom indices by name
                iname = angle.itom["name"] if "name" in angle.itom else angle.itom.get("name", "")
                jname = angle.jtom["name"] if "name" in angle.jtom else angle.jtom.get("name", "")
                kname = angle.ktom["name"] if "name" in angle.ktom else angle.ktom.get("name", "")
                if all(name in atom_name_idx_mapping for name in [iname, jname, kname]):
                    adict["i"] = atom_name_idx_mapping[iname]
                    adict["j"] = atom_name_idx_mapping[jname]
                    adict["k"] = atom_name_idx_mapping[kname]
                adicts.append(adict)
            
            if adicts:
                angles_data = {}
                all_angle_keys = set()
                for adict in adicts:
                    all_angle_keys.update(adict.keys())
                
                for key in all_angle_keys:
                    angles_data[key] = [adict.get(key) for adict in adicts]
                
                angles_data["id"] = list(range(len(adicts)))
                frame["angles"] = angles_data

        # Create dihedrals data if dihedrals exist
        if "dihedrals" in self and len(self["dihedrals"]) > 0:
            ddicts = []
            for dihedral in self["dihedrals"]:
                ddict = dihedral.to_dict()
                # Get atom indices by name
                iname = dihedral.itom["name"] if "name" in dihedral.itom else dihedral.itom.get("name", "")
                jname = dihedral.jtom["name"] if "name" in dihedral.jtom else dihedral.jtom.get("name", "")
                kname = dihedral.ktom["name"] if "name" in dihedral.ktom else dihedral.ktom.get("name", "")
                lname = dihedral.ltom["name"] if "name" in dihedral.ltom else dihedral.ltom.get("name", "")
                if all(name in atom_name_idx_mapping for name in [iname, jname, kname, lname]):
                    ddict["i"] = atom_name_idx_mapping[iname]
                    ddict["j"] = atom_name_idx_mapping[jname]
                    ddict["k"] = atom_name_idx_mapping[kname]
                    ddict["l"] = atom_name_idx_mapping[lname]
                ddicts.append(ddict)
            
            if ddicts:
                dihedrals_data = {}
                all_dihedral_keys = set()
                for ddict in ddicts:
                    all_dihedral_keys.update(ddict.keys())
                
                for key in all_dihedral_keys:
                    dihedrals_data[key] = [ddict.get(key) for ddict in ddicts]
                
                dihedrals_data["id"] = list(range(len(ddicts)))
                frame["dihedrals"] = dihedrals_data

        # Create impropers data if impropers exist
        if "impropers" in self and len(self["impropers"]) > 0:
            idicts = []
            for improper in self["impropers"]:
                idict = improper.to_dict()
                # Get atom indices by name (impropers use same structure as dihedrals)
                iname = improper.itom["name"] if "name" in improper.itom else improper.itom.get("name", "")
                jname = improper.jtom["name"] if "name" in improper.jtom else improper.jtom.get("name", "")
                kname = improper.ktom["name"] if "name" in improper.ktom else improper.ktom.get("name", "")
                lname = improper.ltom["name"] if "name" in improper.ltom else improper.ltom.get("name", "")
                if all(name in atom_name_idx_mapping for name in [iname, jname, kname, lname]):
                    idict["i"] = atom_name_idx_mapping[iname]
                    idict["j"] = atom_name_idx_mapping[jname]
                    idict["k"] = atom_name_idx_mapping[kname]
                    idict["l"] = atom_name_idx_mapping[lname]
                idicts.append(idict)
            
            if idicts:
                impropers_data = {}
                all_improper_keys = set()
                for idict in idicts:
                    all_improper_keys.update(idict.keys())
                
                for key in all_improper_keys:
                    impropers_data[key] = [idict.get(key) for idict in idicts]
                
                impropers_data["id"] = list(range(len(idicts)))
                frame["impropers"] = impropers_data

        # Add structure metadata
        frame._meta["structure_name"] = self.get("name", "")
        frame._meta["n_atoms"] = len(self["atoms"])
        frame._meta["n_bonds"] = len(self.get("bonds", []))
        frame._meta["n_angles"] = len(self.get("angles", []))
        frame._meta["n_dihedrals"] = len(self.get("dihedrals", []))
        frame._meta["n_impropers"] = len(self.get("impropers", []))

        return frame

    def gen_angles(self, topo=None):
        """
        Generate angle objects from bond connectivity.
        
        Args:
            topo: Optional topology object to use. If None, creates from current structure.
            
        Returns:
            List of Angle objects
        """
        if topo is None:
            topo = self.get_topology()
        
        # Check if we have enough atoms for angles
        if len(self.atoms) < 3:
            return []
        
        # The Topology class automatically finds angles using graph isomorphism
        # Convert topology angles to Angle objects
        angles = []
        try:
            angle_indices_array = topo.angles
            atom_list = list(self.atoms)  # Convert to list for proper indexing
            
            if angle_indices_array is not None and len(angle_indices_array) > 0:
                # Handle both 1D and 2D arrays
                if angle_indices_array.ndim == 1:
                    if len(angle_indices_array) >= 3:
                        angle_indices_array = [angle_indices_array]
                    else:
                        return []
                
                for angle_indices in angle_indices_array:
                    if len(angle_indices) >= 3:
                        i, j, k = angle_indices[:3]
                        if i < len(atom_list) and j < len(atom_list) and k < len(atom_list):
                            atom1 = atom_list[i]
                            vertex = atom_list[j]
                            atom3 = atom_list[k]
                            # Check that all atoms are different
                            if len({id(atom1), id(vertex), id(atom3)}) == 3:
                                angle = Angle(atom1, vertex, atom3)
                                angles.append(angle)
        except (IndexError, AttributeError) as e:
            # Handle cases where topology properties fail
            pass
        
        return angles

    def gen_dihedrals(self, topo=None):
        """
        Generate dihedral objects from bond connectivity.
        
        Args:
            topo: Optional topology object to use. If None, creates from current structure.
            
        Returns:
            List of Dihedral objects
        """
        if topo is None:
            topo = self.get_topology()
        
        # Check if we have enough atoms for dihedrals
        if len(self.atoms) < 4:
            return []
        
        # The Topology class automatically finds dihedrals using graph isomorphism
        # Convert topology dihedrals to Dihedral objects
        dihedrals = []
        try:
            dihedral_indices_array = topo.dihedrals
            atom_list = list(self.atoms)  # Convert to list for proper indexing
            
            if dihedral_indices_array is not None and len(dihedral_indices_array) > 0:
                # Handle both 1D and 2D arrays
                if dihedral_indices_array.ndim == 1:
                    if len(dihedral_indices_array) >= 4:
                        dihedral_indices_array = [dihedral_indices_array]
                    else:
                        return []
                
                for dihedral_indices in dihedral_indices_array:
                    if len(dihedral_indices) >= 4:
                        i, j, k, l = dihedral_indices[:4]
                        if all(idx < len(atom_list) for idx in [i, j, k, l]):
                            atom1 = atom_list[i]
                            atom2 = atom_list[j]
                            atom3 = atom_list[k]
                            atom4 = atom_list[l]
                            # Check that all atoms are different
                            if len({id(atom1), id(atom2), id(atom3), id(atom4)}) == 4:
                                dihedral = Dihedral(atom1, atom2, atom3, atom4)
                                dihedrals.append(dihedral)
        except (IndexError, AttributeError) as e:
            # Handle cases where topology properties fail
            pass
        
        return dihedrals
    
    def __call__(self, **kwargs):
        """
        Create a new instance of this AtomicStructure with optional modifications.
        
        This method creates a proper deep copy of the structure, avoiding
        the double-initialization problem in the base Struct class.
        
        Args:
            **kwargs: Properties to modify in the copy
            
        Returns:
            A new AtomicStructure instance with copied atoms, bonds, and angles
        """
        import copy
        import inspect
        
        # Get constructor parameters
        init_signature = inspect.signature(self.__class__.__init__)
        constructor_kwargs = {}
        modification_kwargs = {}
        
        # Separate constructor arguments from modification arguments
        for key, value in kwargs.items():
            if key in init_signature.parameters:
                constructor_kwargs[key] = value
            else:
                modification_kwargs[key] = value
        
        # Use existing name if not specified
        if 'name' not in constructor_kwargs:
            constructor_kwargs['name'] = self.get('name', '')
            
        # Create a new empty instance
        new_instance = self.__class__(**constructor_kwargs)
        
        # Clear any auto-created collections
        new_instance['atoms'].clear()
        new_instance['bonds'].clear()
        new_instance['angles'].clear()
        new_instance['dihedrals'].clear()
        
        # Deep copy atoms with modifications
        atom_mapping = {}  # Map old atoms to new atoms
        for atom in self.atoms:
            new_atom_data = copy.deepcopy(atom.to_dict())
            # Apply modifications to atom data
            for key, value in modification_kwargs.items():
                new_atom_data[key] = value
            new_atom = Atom(**new_atom_data)
            new_instance.add_atom(new_atom)
            atom_mapping[atom] = new_atom
        
        # Deep copy bonds
        for bond in self.bonds:
            new_atom1 = atom_mapping.get(bond.atom1)
            new_atom2 = atom_mapping.get(bond.atom2)
            if new_atom1 and new_atom2:
                bond_data = copy.deepcopy(bond.to_dict())
                # Remove atom references from bond data
                bond_data.pop('atoms', None)
                bond_data.pop('_atoms', None)
                new_bond = Bond(new_atom1, new_atom2, **bond_data)
                new_instance.add_bond(new_bond)
        
        # Deep copy angles
        for angle in self.angles:  
            new_atom1 = atom_mapping.get(angle.itom)
            new_atom2 = atom_mapping.get(angle.jtom)
            new_atom3 = atom_mapping.get(angle.ktom)
            if new_atom1 and new_atom2 and new_atom3:
                angle_data = copy.deepcopy(angle.to_dict())
                # Remove atom references from angle data
                angle_data.pop('atoms', None)
                angle_data.pop('_atoms', None)
                new_angle = Angle(new_atom1, new_atom2, new_atom3, **angle_data)
                new_instance.add_angle(new_angle)
        
        # Copy other non-structural properties
        for key, value in self.items():
            if key not in ['atoms', 'bonds', 'angles', 'dihedrals'] and key not in modification_kwargs:
                if hasattr(value, 'copy'):
                    new_instance[key] = value.copy()
                else:
                    new_instance[key] = value
        
        return new_instance

    # ...existing code...

