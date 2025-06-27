"""
Core molecular structure classes for the molpy framework.

This module provides the fundamental building blocks for molecular modeling:
- Entity: Base class with dictionary-like behavior
- Spatial operations for atoms and structures
- Topological entities: bonds, angles, dihedrals
- Hierarchical structure management
- Collections and containers
"""

from typing import Callable, Iterable

import numpy as np
from numpy.typing import ArrayLike

from .protocol import Entity, Entities, Struct
from .wrapper import Wrapper
from .topology import Topology
from .utils import to_dict_of_list, to_list_of_dict
import copy

class Atom(Entity):
    """
    Class representing an atom with spatial coordinates.
    
    Combines Entity's dictionary behavior with spatial operations through wrappers.
    """

    def __init__(self, name: str = "", **kwargs):
        """
        Initialize an atom.
        
        Args:
            name: Atom name/symbol
            xyz: 3D coordinates
            **kwargs: Additional properties
        """
        super().__init__(name=name, **kwargs)

    def __repr__(self) -> str:
        """Return a string representation of the atom."""
        return f"<Atom {self.name}>"

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

    @property
    def atoms(self):
        """Get the atoms involved in the entity."""
        return self._atoms


class Bond(ManyBody):
    """Class representing a bond between two atoms."""

    def __init__(self, itom=None, jtom=None, **kwargs):
        """
        Initialize a bond between two atoms.

        Args:
            itom: First atom in the bond
            jtom: Second atom in the bond
            **kwargs: Additional properties (e.g., bond_type, length)
        """
        if itom is jtom:
            raise ValueError("Cannot create bond between same atom")
        sorted_atoms = sorted([itom, jtom], key=lambda a: id(a))
        super().__init__(*sorted_atoms, **kwargs)

    @property
    def itom(self):
        """Get the first atom in the bond."""
        return self._atoms[0]

    @property
    def jtom(self):
        """Get the second atom in the bond."""
        return self._atoms[1]

    def __repr__(self) -> str:
        """Return a string representation of the bond."""
        return f"<Bond: {self.itom.name}-{self.jtom.name}>"

    def __eq__(self, other) -> bool:
        """Check equality based on the atoms in the bond."""
        if not isinstance(other, Bond):
            return False
        return (self.itom is other.itom and self.jtom is other.jtom) or \
               (self.itom is other.jtom and self.jtom is other.itom)


class Angle(ManyBody):
    """Class representing an angle between three atoms."""

    def __init__(self, itom: Atom, jtom: Atom, ktom: Atom, **kwargs):
        """
        Initialize an angle.

        Args:
            itom: First atom in the angle
            jtom: jtom atom (center of angle)
            ktom: Third atom in the angle
            **kwargs: Additional properties
        """
        if len({id(itom), id(jtom), id(ktom)}) != 3:
            raise ValueError("All three atoms must be different")
        # Sort end atoms for consistent ordering
        end_atoms = sorted([itom, ktom], key=lambda a: id(a))
        super().__init__(end_atoms[0], jtom, end_atoms[1], **kwargs)

    @property
    def itom(self):
        """Get the first atom in the angle."""
        return self._atoms[0]

    @property
    def jtom(self):
        """Get the jtom atom (center of angle)."""
        return self._atoms[1]

    @property
    def ktom(self):
        """Get the third atom in the angle."""
        return self._atoms[2]

    def __repr__(self) -> str:
        """Return a string representation of the angle."""
        return f"<Angle: {self.itom.name}-{self.jtom.name}-{self.ktom.name}>"

    @property
    def value(self) -> float:
        """Calculate the angle value in radians."""
        v1 = self.itom.xyz - self.jtom.xyz
        v2 = self.jtom.xyz - self.jtom.xyz
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        # Clamp to handle numerical errors
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return float(np.arccos(cos_angle))


class Dihedral(ManyBody):
    """Class representing a dihedral angle between four atoms."""

    def __init__(self, itom: Atom, jtom: Atom, ktom: Atom, ltom: Atom, **kwargs):
        """
        Initialize a dihedral angle.

        Args:
            itom: First atom in the dihedral
            jtom: Second atom in the dihedral
            ktom: Third atom in the dihedral  
            ltom: Fourth atom in the dihedral
            **kwargs: Additional properties
        """
        if len({id(itom), id(jtom), id(ktom), id(ltom)}) != 4:
            raise ValueError("All four atoms must be different")
        
        # Ensure consistent ordering based on central bond
        if id(jtom) > id(ktom):
            itom, jtom, ktom, ltom = ltom, ktom, jtom, itom
            
        super().__init__(itom, jtom, ktom, ltom, **kwargs)

    @property
    def itom(self):
        """Get the first atom in the dihedral."""
        return self._atoms[0]

    @property
    def jtom(self):
        """Get the second atom in the dihedral."""
        return self._atoms[1]

    @property
    def ktom(self):
        """Get the third atom in the dihedral."""
        return self._atoms[2]

    @property
    def ltom(self):
        """Alias for atom4."""
        return self._atoms[3]

    def __repr__(self) -> str:
        """Return a string representation of the dihedral."""
        return f"<Dihedral: {self.itom.name}-{self.jtom.name}-{self.ktom.name}-{self.ltom.name}>"

    @property
    def value(self) -> float:
        """Calculate the dihedral angle value in radians."""
        # Vectors along the bonds
        b1 = self.jtom.xyz - self.itom.xyz
        b2 = self.ktom.xyz - self.jtom.xyz
        b3 = self.ltom.xyz - self.ktom.xyz
        
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


class Improper(Dihedral):
    """
    Class representing an improper dihedral angle.
    
    An improper dihedral is used to maintain planarity in molecular structures.
    """

    def __init__(self, itom: Atom, jtom: Atom, ktom: Atom, ltom: Atom, **kwargs):
        """
        Initialize an improper dihedral angle.

        Args:
            itom: Central atom
            itom: First bonded atom
            jtom: Second bonded atom 
            ltom: Third bonded atom
            **kwargs: Additional properties
        """
        # Sort the three bonded atoms for consistent ordering
        bonded_atoms = sorted([jtom, ktom, ltom], key=lambda a: id(a))
        super().__init__(itom, bonded_atoms[0], bonded_atoms[1], bonded_atoms[2], **kwargs)

    def __repr__(self) -> str:
        """Return a string representation of the improper dihedral."""
        return f"<Improper: {self.itom.name}({self.jtom.name},{self.ktom.name},{self.ltom.name})>"


class AtomicStruct(Wrapper):
    """
    Structure containing atoms, bonds, angles, and dihedrals.
    
    Basic structure functionality that can be enhanced with wrappers
    for spatial operations and hierarchical management.
    """

    def __init__(self, **props):
        """
        Initialize an atomic structure.
        
        Args:
            name: Structure name
            **props: Additional properties
        """
        super().__init__(Struct(**props))
        self["atoms"] = Entities[Atom]()
        self["bonds"] = Entities[Bond]()
        self["angles"] = Entities[Angle]()
        self["dihedrals"] = Entities[Dihedral]()

    def __repr__(self) -> str:
        """Return a string representation of the structure."""
        return f"<AtomicStruct: {len(self.atoms)} atoms>"

    @property
    def atoms(self) -> Entities[Atom]:
        """Get the atoms in the structure."""
        return self._wrapped["atoms"]
    
    @atoms.setter
    def atoms(self, value: Iterable[Atom]):
        self._wrapped["atoms"] = Entities[Atom](value)

    @property
    def bonds(self) -> Entities[Bond]:
        """Get the bonds in the structure."""
        return self._wrapped["bonds"]
    
    @bonds.setter
    def bonds(self, value: Iterable[Bond]):
        self._wrapped["bonds"] = Entities[Bond](value)

    @property
    def angles(self) -> Entities[Angle]:
        """Get the angles in the structure."""
        return self._wrapped["angles"]

    @angles.setter
    def angles(self, value: Iterable[Angle]):
        self._wrapped["angles"] = Entities[Angle](value)

    @property
    def dihedrals(self) -> Entities[Dihedral]:
        """Get the dihedrals in the structure."""
        return self._wrapped["dihedrals"]
    
    @dihedrals.setter
    def dihedrals(self, value: Iterable[Dihedral]):
        self._wrapped["dihedrals"] = Entities[Dihedral](value)

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

    def def_bond(self, i: Atom | int, j: Atom | int, **kwargs):
        """
        Create and add a bond between two atoms.
        
        Args:
            i: First atom or its index
            j: Second atom or its index
            **kwargs: Bond properties
            
        Returns:
            The created bond
        """
        itom = self.atoms[i] if isinstance(i, int) else i
        jtom = self.atoms[j] if isinstance(j, int) else j

        if not isinstance(itom, Atom) or not isinstance(jtom, Atom):
            raise TypeError("Arguments must be Atom instances or valid indices")
            
        bond = Bond(itom, jtom, **kwargs)
        return self.add_bond(bond)

    def del_bond(self, i: Atom | int, j: Atom | int):
        """
        Remove a bond from the structure.

        Args:
            i: First atom or its index
            j: Second atom or its index
        """
        itom = self.atoms[i] if isinstance(i, int) else i
        jtom = self.atoms[j] if isinstance(j, int) else j

        if not isinstance(itom, Atom) or not isinstance(jtom, Atom):
            raise TypeError("Arguments must be Atom instances or valid indices")

        bond = Bond(itom, jtom)
        return self.bonds.remove(bond)

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

    def def_angle(self, i: Atom | int, j: Atom | int, k: Atom | int, **kwargs):
        """
        Create and add an angle between three atoms.
        
        Args:
            i: First atom or its index
            j: Center atom or its index
            k: Third atom or its index
            **kwargs: Angle properties
            
        Returns:
            The created angle
        """
        itom = self.atoms[i] if isinstance(i, int) else i
        jtom = self.atoms[j] if isinstance(j, int) else j
        ktom = self.atoms[k] if isinstance(k, int) else k

        if not all(isinstance(atom, Atom) for atom in [itom, jtom, ktom]):
            raise TypeError("Arguments must be Atom instances or valid indices")

        angle = Angle(itom, jtom, ktom, **kwargs)
        return self.add_angle(angle)

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
        return self

    def remove_bond(self, bond):
        """
        Remove a bond from the structure.
        
        Args:
            bond: Bond instance or tuple of atoms
        """
        self.bonds.remove(bond)
        return self

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
    def concat(cls, name: str, structs, reindex=True):
        """
        Concatenate multiple structures into a new structure.

        Args:
            name: Name for the new structure
            structs: Sequence of structures to concatenate
            reindex: Whether to reindex the atoms in the concatenated structure

        Returns:
            New structure containing all input structures
        """
        result = cls(name)
        for struct in structs:
            result.add_struct(struct)
        if reindex:
            for i, atom in enumerate(result.atoms, 1):
                atom["id"] = i
        return result
    
    def to_dict(self):

        data = {
            "atoms": {},
            "bonds": {},
            "angles": {},
            "dihedrals": {},
            "impropers": {}
        }

        if not self.atoms:
            return data

        atom_dicts = [a.to_dict() for a in self.atoms]
        atom_map = {a: i for i, a in enumerate(self.atoms)}
        data["atoms"] = to_dict_of_list(atom_dicts)

        def serialize_topo(items, ref_keys):
            return [
                dict(item.to_dict(), **{k: atom_map[getattr(item, k + "tom")] for k in ref_keys})
                for item in items
            ]

        data["bonds"] = to_dict_of_list(serialize_topo(self.bonds, ["i", "j"]))
        data["angles"] = to_dict_of_list(serialize_topo(self.angles, ["i", "j", "k"]))
        data["dihedrals"] = to_dict_of_list(serialize_topo(self.dihedrals, ["i", "j", "k", "l"]))
        data["impropers"] = to_dict_of_list(serialize_topo(self.impropers, ["i", "j", "k", "l"]))

        return data

    @classmethod
    def from_dict(cls, data):
        struct = cls()

        atom_dicts = to_list_of_dict(data["atoms"])
        struct.atoms = [Atom.from_dict(d) for d in atom_dicts]
        atoms_by_index = struct.atoms  # index to Atom list

        bonds = []
        bond_dicts = to_list_of_dict(data["bonds"])
        for bd in bond_dicts:
            itom = atoms_by_index[bd.pop("i")]
            jtom = atoms_by_index[bd.pop("j")]
            bond = Bond(itom, jtom, **{k: v for k, v in bd.items()})
            bonds.append(bond)
        struct.bonds = bonds

        angles = []
        angle_dicts = to_list_of_dict(data["angles"])
        for ad in angle_dicts:
            itom = atoms_by_index[ad.pop("i")]
            jtom = atoms_by_index[ad.pop("j")]
            ktom = atoms_by_index[ad.pop("k")]
            angle = Angle(itom, jtom, ktom, **{k: v for k, v in ad.items()})
            angles.append(angle)
        struct.angles = angles

        dihedrals = []
        dihedral_dicts = to_list_of_dict(data["dihedrals"])
        for dd in dihedral_dicts:
            itom = atoms_by_index[dd.pop("i")]
            jtom = atoms_by_index[dd.pop("j")]
            ktom = atoms_by_index[dd.pop("k")]
            ltom = atoms_by_index[dd.pop("l")]
            dihedral = Dihedral(itom, jtom, ktom, ltom, **{k: v for k, v in dd.items()})
            dihedrals.append(dihedral)
        struct.dihedrals = dihedrals

        return struct

    # def to_frame(self):
    #     """
    #     Convert the structure to a Frame object.

    #     Returns:
    #     - A Frame object containing the structure's data.
    #     """
    #     from .frame import Frame

    #     frame = Frame()

    #     # Handle empty structure case
    #     if "atoms" not in self or len(self["atoms"]) == 0:
    #         # Create empty dict for atoms
    #         frame["atoms"] = {}
    #         return frame

    #     # Set atom IDs and create name-to-index mapping
    #     atom_mapping = {}
    #     atom_dicts = []
    #     for i, atom in enumerate(self["atoms"]):
    #         atom_mapping[atom] = i
    #         atom_dicts.append(atom.to_dict())

    #     # Create atoms data directly as dict for xarray
    #     atoms_data = {}
    #     if atom_dicts:
    #         # Collect all keys from atom dictionaries
    #         all_keys = set()
    #         for atom_dict in atom_dicts:
    #             all_keys.update(atom_dict.keys())
            
    #         # Build arrays for each property
    #         for key in all_keys:
    #             values = [atom_dict.get(key) for atom_dict in atom_dicts]
    #             atoms_data[key] = values

    #     frame["atoms"] = atoms_data
        
    #     # Create bonds data if bonds exist
    #     if "bonds" in self and len(self["bonds"]) > 0:
    #         bdicts = []
    #         for bond in self["bonds"]:
    #             bdict = bond.to_dict()

    #             bdict["i"] = atom_mapping[bond.itom]
    #             bdict["j"] = atom_mapping[bond.jtom]
    #             bdicts.append(bdict)
            
    #         if bdicts:
    #             # Create bonds data directly as dict
    #             bonds_data = {}
    #             all_bond_keys = set()
    #             for bdict in bdicts:
    #                 all_bond_keys.update(bdict.keys())
                
    #             for key in all_bond_keys:
    #                 bonds_data[key] = [bdict.get(key) for bdict in bdicts]
                
    #             # Add sequential ids
    #             bonds_data["id"] = list(range(1, len(bdicts)+1))
    #             frame["bonds"] = bonds_data

    #     # Create angles data if angles exist
    #     if "angles" in self and len(self["angles"]) > 0:
    #         adicts = []
    #         for angle in self["angles"]:
    #             adict = angle.to_dict()
    #             # Get atom indices by name
    #             iname = angle.itom["name"] if "name" in angle.itom else angle.itom.get("name", "")
    #             jname = angle.jtom["name"] if "name" in angle.jtom else angle.jtom.get("name", "")
    #             kname = angle.ktom["name"] if "name" in angle.ktom else angle.ktom.get("name", "")
    #             if all(name in atom_mapping for name in [iname, jname, kname]):
    #                 adict["i"] = atom_mapping[iname]
    #                 adict["j"] = atom_mapping[jname]
    #                 adict["k"] = atom_mapping[kname]
    #             adicts.append(adict)
            
    #         if adicts:
    #             angles_data = {}
    #             all_angle_keys = set()
    #             for adict in adicts:
    #                 all_angle_keys.update(adict.keys())
                
    #             for key in all_angle_keys:
    #                 angles_data[key] = [adict.get(key) for adict in adicts]

    #             angles_data["id"] = list(range(1, len(adicts)+1))
    #             frame["angles"] = angles_data

    #     # Create dihedrals data if dihedrals exist
    #     if "dihedrals" in self and len(self["dihedrals"]) > 0:
    #         ddicts = []
    #         for dihedral in self["dihedrals"]:
    #             ddict = dihedral.to_dict()
    #             # Get atom indices by name
    #             iname = dihedral.itom["name"] if "name" in dihedral.itom else dihedral.itom.get("name", "")
    #             jname = dihedral.jtom["name"] if "name" in dihedral.jtom else dihedral.jtom.get("name", "")
    #             kname = dihedral.ktom["name"] if "name" in dihedral.ktom else dihedral.ktom.get("name", "")
    #             lname = dihedral.ltom["name"] if "name" in dihedral.ltom else dihedral.ltom.get("name", "")
    #             if all(name in atom_mapping for name in [iname, jname, kname, lname]):
    #                 ddict["i"] = atom_mapping[iname]
    #                 ddict["j"] = atom_mapping[jname]
    #                 ddict["k"] = atom_mapping[kname]
    #                 ddict["l"] = atom_mapping[lname]
    #             ddicts.append(ddict)
            
    #         if ddicts:
    #             dihedrals_data = {}
    #             all_dihedral_keys = set()
    #             for ddict in ddicts:
    #                 all_dihedral_keys.update(ddict.keys())
                
    #             for key in all_dihedral_keys:
    #                 dihedrals_data[key] = [ddict.get(key) for ddict in ddicts]

    #             dihedrals_data["id"] = list(range(1, len(ddicts)+1))
    #             frame["dihedrals"] = dihedrals_data

    #     # Create impropers data if impropers exist
    #     if "impropers" in self and len(self["impropers"]) > 0:
    #         idicts = []
    #         for improper in self["impropers"]:
    #             idict = improper.to_dict()
    #             # Get atom indices by name (impropers use same structure as dihedrals)
    #             iname = improper.itom["name"] if "name" in improper.itom else improper.itom.get("name", "")
    #             jname = improper.jtom["name"] if "name" in improper.jtom else improper.jtom.get("name", "")
    #             kname = improper.ktom["name"] if "name" in improper.ktom else improper.ktom.get("name", "")
    #             lname = improper.ltom["name"] if "name" in improper.ltom else improper.ltom.get("name", "")
    #             if all(name in atom_mapping for name in [iname, jname, kname, lname]):
    #                 idict["i"] = atom_mapping[iname]
    #                 idict["j"] = atom_mapping[jname]
    #                 idict["k"] = atom_mapping[kname]
    #                 idict["l"] = atom_mapping[lname]
    #             idicts.append(idict)
            
    #         if idicts:
    #             impropers_data = {}
    #             all_improper_keys = set()
    #             for idict in idicts:
    #                 all_improper_keys.update(idict.keys())
                
    #             for key in all_improper_keys:
    #                 impropers_data[key] = [idict.get(key) for idict in idicts]

    #             impropers_data["id"] = list(range(1, len(idicts)+1))
    #             frame["impropers"] = impropers_data

    #     # Add structure metadata
    #     frame._meta["structure_name"] = self.get("name", "")
    #     frame._meta["n_atoms"] = len(self["atoms"])
    #     frame._meta["n_bonds"] = len(self.get("bonds", []))
    #     frame._meta["n_angles"] = len(self.get("angles", []))
    #     frame._meta["n_dihedrals"] = len(self.get("dihedrals", []))
    #     frame._meta["n_impropers"] = len(self.get("impropers", []))

    #     return frame

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
                            itom = atom_list[i]
                            jtom = atom_list[j]
                            atom3 = atom_list[k]
                            # Check that all atoms are different
                            if len({id(itom), id(jtom), id(atom3)}) == 3:
                                angle = Angle(itom, jtom, atom3)
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
                            itom = atom_list[i]
                            jtom = atom_list[j]
                            atom3 = atom_list[k]
                            atom4 = atom_list[l]
                            # Check that all atoms are different
                            if len({id(itom), id(jtom), id(atom3), id(atom4)}) == 4:
                                dihedral = Dihedral(itom, jtom, atom3, atom4)
                                dihedrals.append(dihedral)
        except (IndexError, AttributeError) as e:
            # Handle cases where topology properties fail
            pass
        
        return dihedrals
    
    def get_substruct(self, indices: list[int]):
        """Extract substructure containing only specified atoms."""
        # TODO: TO BE TESTED!
        new_struct = mp.AtomicStruct(name=f"{self._wrapped.get('name', 'molecule')}_template")
        
        # Map old indices to new indices
        index_map = {}
        sorted_indices = sorted(indices)

        # Add atoms
        for new_idx, old_idx in enumerate(sorted_indices):
            atom = self.atoms[old_idx]
            new_atom = atom.copy()
            new_atom["id"] = new_idx
            new_struct.atoms.add(new_atom)
            index_map[old_idx] = new_idx
        
        # Add bonds between extracted atoms
        for bond in self.bonds:
            atom1_idx = bond.itom.get("id", bond.itom.get("index"))
            atom2_idx = bond.jtom.get("id", bond.jtom.get("index"))
            
            if atom1_idx in index_map and atom2_idx in index_map:
                new_bond = mp.Bond(
                    new_struct.atoms[index_map[atom1_idx]],
                    new_struct.atoms[index_map[atom2_idx]],
                    **{k: v for k, v in bond.items() if k not in ['itom', 'jtom']}
                )
                new_struct.bonds.add(new_bond)
        
        return new_struct
    
    def __call__(self, **new_prop) -> "AtomicStruct":
        """
        Create a new instance of this AtomicStruct with optional modifications.
        
        This method creates a proper deep copy of the structure, avoiding
        the double-initialization problem in the base Struct class.
        
        Args:
            **new_prop: Properties to modify in the copy
            
        Returns:
            A new AtomicStruct instance with copied atoms, bonds, angles, and dihedrals
        """
        old_prop = self._wrapped.to_dict()
        old_prop.update(new_prop)
        new_instance = self.__class__(**old_prop)

        atom_mapping = {}
        for old_atom in self.atoms:
            new_atom = Atom(**copy.deepcopy(old_atom.to_dict()))
            new_instance.add_atom(new_atom)
            atom_mapping[old_atom] = new_atom

        def _deepcopy_topo(items, cls, atom_keys, add_method):
            for item in items:
                atoms = [atom_mapping.get(getattr(item, k + "tom")) for k in atom_keys]
                if all(atoms):
                    data = copy.deepcopy(item.to_dict())
                    # Remove atom references from data if present
                    for k in atom_keys:
                        data.pop(f"{k}tom", None)
                    new_item = cls(*atoms, **data)
                    add_method(new_item)

        # Deep copy bonds, angles, dihedrals
        _deepcopy_topo(self.bonds, Bond, ["i", "j"], new_instance.add_bond)
        _deepcopy_topo(self.angles, Angle, ["i", "j", "k"], new_instance.add_angle)
        _deepcopy_topo(self.dihedrals, Dihedral, ["i", "j", "k", "l"], new_instance.add_dihedral)
        # _deepcopy_topo(self.dihedrals, Improper, ["i", "j", "k", "l"], new_instance.add_improper)

        return new_instance

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k == '_wrapped':
                setattr(result, k, copy.deepcopy(v, memo))
            else:
                setattr(result, k, copy.deepcopy(v, memo))
        return result

            

    @classmethod
    def from_frame(cls, frame, name: str = ""):
        """
        Create a new AtomicStruct from a Frame object.
        
        Args:
            frame: Frame object containing the structure data
            name: Name for the new structure (optional)
            
        Returns:
            New AtomicStruct instance populated from the frame
        """
        # Get structure name from frame metadata if not provided
        if not name and "structure_name" in frame:
            name = frame["structure_name"]
        elif not name and hasattr(frame, '_meta') and "structure_name" in frame._meta:
            name = frame._meta["structure_name"]
        
        # Create new structure instance
        struct = cls(name=name)
        
        # Handle empty frame case
        if "atoms" not in frame or not frame["atoms"]:
            return struct
        
        atoms_data = frame["atoms"]
        
        # Determine number of atoms from the dataset
        n_atoms = 0
        if hasattr(atoms_data, 'dims'):
            # Get the first dimension size from any data variable
            for key, data_array in atoms_data.data_vars.items():
                if data_array.ndim >= 1:
                    n_atoms = data_array.shape[0]
                    break
        
        if n_atoms == 0:
            return struct
        
        # Create atoms with all their properties
        atom_list = []
        for i in range(n_atoms):
            atom_props = {}
            
            # Extract properties for this atom from xarray DataArrays
            for key, data_array in atoms_data.data_vars.items():
                if data_array.ndim == 0:
                    # Scalar value - same for all atoms
                    atom_props[key] = data_array.values.item()
                elif data_array.ndim == 1:
                    # 1D array - one value per atom
                    if i < len(data_array.values):
                        atom_props[key] = data_array.values[i]
                elif data_array.ndim >= 2:
                    # Multi-dimensional array (like xyz coordinates)
                    if i < data_array.shape[0]:
                        atom_props[key] = data_array.values[i]
            
            # Create and add atom
            atom = Atom(**atom_props)
            struct.add_atom(atom)
            atom_list.append(atom)
        
        # Create bonds if they exist in the frame
        if "bonds" in frame and frame["bonds"]:
            bonds_data = frame["bonds"]
            
            # Determine number of bonds
            n_bonds = 0
            if hasattr(bonds_data, 'dims'):
                for key, data_array in bonds_data.data_vars.items():
                    if data_array.ndim >= 1:
                        n_bonds = data_array.shape[0]
                        break
            
            for i in range(n_bonds):
                bond_props = {}
                
                # Extract properties for this bond
                for key, data_array in bonds_data.data_vars.items():
                    if data_array.ndim == 0:
                        bond_props[key] = data_array.values.item()
                    elif data_array.ndim >= 1 and i < data_array.shape[0]:
                        bond_props[key] = data_array.values[i]
                
                # Get atom indices
                if "i" in bond_props and "j" in bond_props:
                    i_idx = bond_props.pop("i")
                    j_idx = bond_props.pop("j")
                    
                    # Create bond with atom references
                    if i_idx < len(atom_list) and j_idx < len(atom_list):
                        bond = Bond(atom_list[i_idx], atom_list[j_idx], **bond_props)
                        struct.add_bond(bond)
        
        # Create angles if they exist in the frame
        if "angles" in frame and frame["angles"]:
            angles_data = frame["angles"]
            
            # Determine number of angles
            n_angles = 0
            if hasattr(angles_data, 'dims'):
                for key, data_array in angles_data.data_vars.items():
                    if data_array.ndim >= 1:
                        n_angles = data_array.shape[0]
                        break
            
            for i in range(n_angles):
                angle_props = {}
                
                # Extract properties for this angle
                for key, data_array in angles_data.data_vars.items():
                    if data_array.ndim == 0:
                        angle_props[key] = data_array.values.item()
                    elif data_array.ndim >= 1 and i < data_array.shape[0]:
                        angle_props[key] = data_array.values[i]
                
                # Get atom indices
                if "i" in angle_props and "j" in angle_props and "k" in angle_props:
                    i_idx = angle_props.pop("i")
                    j_idx = angle_props.pop("j")
                    k_idx = angle_props.pop("k")
                    
                    # Create angle with atom references
                    if all(idx < len(atom_list) for idx in [i_idx, j_idx, k_idx]):
                        angle = Angle(atom_list[i_idx], atom_list[j_idx], atom_list[k_idx], **angle_props)
                        struct.add_angle(angle)
        
        # Create dihedrals if they exist in the frame
        if "dihedrals" in frame and frame["dihedrals"]:
            dihedrals_data = frame["dihedrals"]
            
            # Determine number of dihedrals
            n_dihedrals = 0
            if hasattr(dihedrals_data, 'dims'):
                for key, data_array in dihedrals_data.data_vars.items():
                    if data_array.ndim >= 1:
                        n_dihedrals = data_array.shape[0]
                        break
            
            for i in range(n_dihedrals):
                dihedral_props = {}
                
                # Extract properties for this dihedral
                for key, data_array in dihedrals_data.data_vars.items():
                    if data_array.ndim == 0:
                        dihedral_props[key] = data_array.values.item()
                    elif data_array.ndim >= 1 and i < data_array.shape[0]:
                        dihedral_props[key] = data_array.values[i]
                
                # Get atom indices
                if all(idx in dihedral_props for idx in ["i", "j", "k", "l"]):
                    i_idx = dihedral_props.pop("i")
                    j_idx = dihedral_props.pop("j")
                    k_idx = dihedral_props.pop("k")
                    l_idx = dihedral_props.pop("l")
                    
                    # Create dihedral with atom references
                    if all(idx < len(atom_list) for idx in [i_idx, j_idx, k_idx, l_idx]):
                        dihedral = Dihedral(atom_list[i_idx], atom_list[j_idx], 
                                          atom_list[k_idx], atom_list[l_idx], **dihedral_props)
                        struct.add_dihedral(dihedral)
        
        # Create impropers if they exist in the frame
        if "impropers" in frame and frame["impropers"]:
            impropers_data = frame["impropers"]
            
            # Determine number of impropers
            n_impropers = 0
            if hasattr(impropers_data, 'dims'):
                for key, data_array in impropers_data.data_vars.items():
                    if data_array.ndim >= 1:
                        n_impropers = data_array.shape[0]
                        break
            
            for i in range(n_impropers):
                improper_props = {}
                
                # Extract properties for this improper
                for key, data_array in impropers_data.data_vars.items():
                    if data_array.ndim == 0:
                        improper_props[key] = data_array.values.item()
                    elif data_array.ndim >= 1 and i < data_array.shape[0]:
                        improper_props[key] = data_array.values[i]
                
                # Get atom indices
                if all(idx in improper_props for idx in ["i", "j", "k", "l"]):
                    i_idx = improper_props.pop("i")
                    j_idx = improper_props.pop("j")
                    k_idx = improper_props.pop("k")
                    l_idx = improper_props.pop("l")
                    
                    # Create improper with atom references
                    if all(idx < len(atom_list) for idx in [i_idx, j_idx, k_idx, l_idx]):
                        improper = Improper(atom_list[i_idx], atom_list[j_idx], 
                                          atom_list[k_idx], atom_list[l_idx], **improper_props)
                        # Initialize impropers collection if it doesn't exist
                        if "impropers" not in struct:
                            struct["impropers"] = Entities()
                        struct["impropers"].add(improper)
        
        return struct