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
            id_ = atom.pop('id')
            struct.add_atom(
                mp.Atom(id_, **atom)
            )
        struct['atoms'].sort(key=lambda atom: atom.id)
        
        bonds = self['bonds']
        for bond in bonds.to_pylist():
            itom = struct['atoms'].get_by_id(bond.pop('i'))
            jtom = struct['atoms'].get_by_id(bond.pop('j'))
            struct.add_bond(
                mp.Bond(itom, jtom, **bond)
            )

        angles = self['angles']
        for angle in angles.to_pylist():
            itom = struct['atoms'].get_by_id(angle.pop('i'))
            jtom = struct['atoms'].get_by_id(angle.pop('j'))
            ktom = struct['atoms'].get_by_id(angle.pop('k'))
            struct.add_angle(
                mp.Angle(itom, jtom, ktom, **angle)
            )

        dihedrals = self['dihedrals']
        for dihedral in dihedrals.to_pylist():
            itom = struct['atoms'].get_by_id(dihedral.pop('i'))
            jtom = struct['atoms'].get_by_id(dihedral.pop('j'))
            ktom = struct['atoms'].get_by_id(dihedral.pop('k'))
            ltom = struct['atoms'].get_by_id(dihedral.pop('l'))
            struct.add_dihedral(
                mp.Dihedral(itom, jtom, ktom, ltom, **dihedral)
            )

        impropers = self['impropers']
        for improper in impropers.to_pylist():
            itom = struct['atoms'].get_by_id(improper.pop('i'))
            jtom = struct['atoms'].get_by_id(improper.pop('j'))
            ktom = struct['atoms'].get_by_id(improper.pop('k'))
            ltom = struct['atoms'].get_by_id(improper.pop('l'))
            struct.add_improper(
                mp.Improper(itom, jtom, ktom, ltom, **improper)
            ) 

        return struct
    