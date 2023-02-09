# author: Roy Kid
# contact: lijichen365@126.com
# date: 2023-01-08
# version: 0.0.1

from .struct import StructArray
from .topology import Topology
from .entity import Residue, Atom
import numpy as np
# from .box import Box

class Frame:

    def __init__(self, ):

        self.atoms = StructArray()
        self.topology = Topology()
        self.box = None
        self.residues = StructArray()

        self._natoms = 0
        self._nresidues = 0

    @classmethod
    def from_chemfile_frame(cls, chemfile_frame):

        molpy_frame = cls()
        
        # load box
        molpy_frame.box = chemfile_frame.cell.matrix

        xyz = chemfile_frame.positions
        molpy_frame.atoms['xyz'] = xyz
        natoms = len(xyz)
        molpy_frame._natoms = natoms

        an_atom = chemfile_frame.atoms[0]

        atom_list_properties = an_atom.list_properties()
        for prop in atom_list_properties:
            molpy_frame.atoms[prop] = [getattr(atom, prop) for atom in chemfile_frame.atoms]

        # load Intrinsic properties
        Intrinsic_props = ['name', 'atomic_number', 'charge', 'mass', 'type',] 
        for prop in Intrinsic_props:
            if hasattr(an_atom, prop):
                molpy_frame.atoms[prop] = [getattr(atom, prop) for atom in chemfile_frame.atoms]

        assert molpy_frame.atoms.length == molpy_frame.atoms.length

        # load topology
        ## load bond
        bonds = chemfile_frame.topology.bonds
        angles = chemfile_frame.topology.angles
        dihedrals = chemfile_frame.topology.dihedrals
        impropers = chemfile_frame.topology.impropers

        molpy_frame.topology.add_bonds(bonds)
        molpy_frame.topology.add_angles(angles)
        molpy_frame.topology.add_dihedrals(dihedrals)
        molpy_frame.topology.add_impropers(impropers)

        # load residue

        residues = chemfile_frame.topology.residues
        nresidues = len(residues)
        molpy_frame._nresidues = nresidues
        names = []
        ids = []
        index = np.empty((nresidues), dtype=object)
        props = []

        for i, residue in enumerate(residues):
            ids.append(residue.id)
            names.append(residue.name)
            props.append({k:residue[k] for k in residue.list_properties()})
            index[i] = np.array(residue.atoms, copy=True)
            
        molpy_frame.residues['id'] = np.array(ids)
        molpy_frame.residues['name'] = np.array(names)
        molpy_frame.residues['index'] = index
        keys = props[0].keys()
        for k in keys:
            molpy_frame.residues[k] = np.array([p[k] for p in props])

        return molpy_frame

    @property
    def natoms(self):
        return self.atoms.length

    @property
    def nbonds(self):
        return self.topology.nbonds

    @property
    def nangles(self):
        return self.topology.nangles

    @property
    def ndihedrals(self):
        return self.topology.ndihedrals

    @property
    def nimpropers(self):
        return self.topology.nimpropers

    @property
    def nresidues(self):
        return self._nresidues

    @property
    def positions(self):
        return self.atoms['positions']

    @property
    def properties(self):
        return self.atoms.keys()

    def __getitem__(self, key):
        return self.atoms[key]

    def get_residue(self, name):
        i = list(self.residues['name']).index(name)
        residue_dict = {}
        for k in self.residues.keys():
            residue_dict[k] = self.residues[k][i]
        
        residue = Residue(residue_dict['name'], residue_dict['id'])
        atom_idx = residue_dict['index']
        atoms = self.atoms[atom_idx]
        
        for atom in atoms:
            atom_props = {k:atom[k] for k in atom.dtype.names}
            atom_props['residue'] = residue_dict['name']
            residue.add_atom(Atom(**atom_props))

        # add bond
        bond_idx = self.topology.bonds['index']
        bond_mask = np.logical_and(np.isin(bond_idx[:, 1], atom_idx), np.isin(bond_idx[:, 0], atom_idx))
        residue_bond_idx = bond_idx[bond_mask]
        atom_idx_accu = np.cumsum(atom_idx.astype(int))
        remap_residue_bond_idx = np.zeros_like(residue_bond_idx, dtype=int)
        remap_residue_bond_idx[:, 0] = atom_idx_accu[residue_bond_idx[:,0]]
        remap_residue_bond_idx[:, 1] = atom_idx_accu[residue_bond_idx[:,1]]

        residue.add_bonds(remap_residue_bond_idx)

        return residue
