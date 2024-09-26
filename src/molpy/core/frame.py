from molpy.core.space import Box
import molpy as mp
import pyarrow as pa
from .struct import Struct

class Frame(dict):

    def __init__(self):
        super().__init__()
        self['props'] = {}

    def to_struct(self):

        struct = Struct()
        atoms = self['atoms']
        for atom in atoms.to_pylist():
            struct.add_atom(
                mp.Atom(**atom)
            )
        
        if 'bonds' in self:
            bonds = self['bonds']
            for bond in bonds.to_pylist():

                itom = struct['atoms'].get_by(lambda atom: atom['id'] == bond['i'])
                jtom = struct['atoms'].get_by(lambda atom: atom['id'] == bond['j'])
                struct.add_bond(
                    mp.Bond(itom, jtom, **bond)
                )

        if 'angles' in self:
            angles = self['angles']
            for angle in angles.to_pylist():
                itom = struct['atoms'].get_by(lambda atom: atom['id'] == angle['i'])
                jtom = struct['atoms'].get_by(lambda atom: atom['id'] == angle['j'])
                ktom = struct['atoms'].get_by(lambda atom: atom['id'] == angle['k'])
                struct.add_angle(
                    mp.Angle(itom, jtom, ktom, **angle)
                )

        if 'dihedrals' in self:
            dihedrals = self['dihedrals']
            for dihedral in dihedrals.to_pylist():
                itom = struct['atoms'].get_by(lambda atom: atom['id'] == dihedral['i'])
                jtom = struct['atoms'].get_by(lambda atom: atom['id'] == dihedral['j'])
                ktom = struct['atoms'].get_by(lambda atom: atom['id'] == dihedral['k'])
                ltom = struct['atoms'].get_by(lambda atom: atom['id'] == dihedral['l'])
                struct.add_dihedral(
                    mp.Dihedral(itom, jtom, ktom, ltom, **dihedral)
                )

        if 'impropers' in self:
            impropers = self['impropers']
            for improper in impropers.to_pylist():
                itom = struct['atoms'].get_by_id(improper['i'])
                jtom = struct['atoms'].get_by_id(improper['j'])
                ktom = struct['atoms'].get_by_id(improper['k'])
                ltom = struct['atoms'].get_by_id(improper['l'])
                struct.add_improper(
                    mp.Improper(itom, jtom, ktom, ltom, **improper)
                ) 

        return struct
    
    def split(self, key):

        frames = []
        masks = self['atoms'][key]
        unique_mask = masks.unique()

        for mask in unique_mask:
            frame = Frame()
            atom_mask = pa.compute.equal(masks, mask)
            frame['atoms'] = self['atoms'].filter(atom_mask)
            atom_id_of_this_frame = frame['atoms']['id']
            if 'bonds' in self:
                bond_i = self['bonds']['i']
                bond_j = self['bonds']['j']
                bond_mask = pa.compute.and_(pa.compute.is_in(bond_i, atom_id_of_this_frame), pa.compute.is_in(bond_j, atom_id_of_this_frame))
                frame['bonds'] = self['bonds'].filter(bond_mask)

            if 'angles' in self:
                angle_i = self['angles']['i']
                angle_j = self['angles']['j']
                angle_k = self['angles']['k']
                angle_mask = pa.compute.is_in(angle_i, atom_id_of_this_frame) and pa.compute.is_in(angle_j, atom_id_of_this_frame) and pa.compute.is_in(angle_k, atom_id_of_this_frame)
                frame['angles'] = self['angles'].filter(angle_mask)

            if 'dihedrals' in self:
                dihedral_i = self['dihedrals']['i']
                dihedral_j = self['dihedrals']['j']
                dihedral_k = self['dihedrals']['k']
                dihedral_l = self['dihedrals']['l']
                dihedral_mask = pa.compute.is_in(dihedral_i, atom_id_of_this_frame) and pa.compute.is_in(dihedral_j, atom_id_of_this_frame) and pa.compute.is_in(dihedral_k, atom_id_of_this_frame) and pa.compute.is_in(dihedral_l, atom_id_of_this_frame)
                frame['dihedrals'] = self['dihedrals'].filter(dihedral_mask)

            if 'impropers' in self:
                improper_i = self['impropers']['i']
                improper_j = self['impropers']['j']
                improper_k = self['impropers']['k']
                improper_l = self['impropers']['l']
                improper_mask = pa.compute.is_in(improper_i, atom_id_of_this_frame) and pa.compute.is_in(improper_j, atom_id_of_this_frame) and pa.compute.is_in(improper_k, atom_id_of_this_frame) and pa.compute.is_in(improper_l, atom_id_of_this_frame)
                frame['impropers'] = self['impropers'].filter(improper_mask)

            frame['props']['n_atoms'] = len(frame['atoms'])
            frame['props']['n_bonds'] = len(frame['bonds']) if 'bonds' in frame else 0
            frame['props']['n_angles'] = len(frame['angles']) if 'angles' in frame else 0
            frame['props']['n_dihedrals'] = len(frame['dihedrals']) if 'dihedrals' in frame else 0
            frame['props']['n_impropers'] = len(frame['impropers']) if 'impropers' in frame else 0
            frame['props']['n_atomtypes'] = len(frame['atoms']['type'].unique())
            frame['props']['n_bondtypes'] = len(frame['bonds']['type'].unique()) if 'bonds' in frame else 0
            frame['props']['n_angletypes'] = len(frame['angles']['type'].unique()) if 'angles' in frame else 0
            frame['props']['n_dihedraltypes'] = len(frame['dihedrals']['type'].unique()) if 'dihedrals' in frame else 0
            frame['props']['n_impropertypes'] = len(frame['impropers']['type'].unique()) if 'impropers' in frame else 0


            frames.append(frame)

        return frames