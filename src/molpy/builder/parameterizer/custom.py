from pathlib import Path
from .ambertool import Parameterizer

import molpy as mp
import numpy as np

class Custom(Parameterizer):

    def __init__(self, forcefield: mp.ForceField, work_dir: str | Path):
        super().__init__(forcefield.name, work_dir)
        self.forcefield = forcefield

    def parameterize(self, struct, **kwargs) -> mp.Struct:
        
        temp_dir = self.work_dir/struct.name
        if not temp_dir.exists():
            temp_dir.mkdir(parents=True)

        sub_ff = mp.ForceField(struct.name, self.forcefield.unit)

        atom_type = struct.atoms.type

        for type_ in atom_type:
            atype = self.forcefield.get_atomtype(type_)
            astyle = atype.style
            if astyle not in sub_ff.atomstyles:
                sub_ff.def_atomstyle(astyle.name, *astyle.params, **astyle.named_params)
            sub_ff.get_atomstyle(astyle.name).def_atomtype(type_, *atype.type_idx, *atype.params, **atype.named_params)

            for k, v in atype.named_params.items():
                if k not in struct.atoms:
                    n_atoms = struct.n_atoms
                    # TODO: ndim
                    struct.atoms[k] = np.zeros(n_atoms, dtype=type(v))
                struct.atoms[k][np.array(atom_type) == atype.name] = v

            for ptype in self.forcefield.pairtypes:
                if atype.type_idx[0] in ptype.type_idx:
                    pstyle = ptype.style
                    if pstyle not in sub_ff.pairstyles:
                        sub_ff.def_pairstyle(pstyle.name, *pstyle.params, **pstyle.named_params)
                    sub_ff.get_pairstyle(pstyle.name).def_pairtype(type_, *ptype.type_idx, *ptype.params, **ptype.named_params)

        mp.io.save_struct(temp_dir/f"{struct.name}.pdb", struct)

        mp.io.save_forcefield(temp_dir/f"{struct.name}.ff", sub_ff, format='lammps')

        return struct

    def load_struct(self, name, format='pdb'):

        return mp.io.load_struct(self.work_dir/name/f"{name}.{format}")
    
    def load_forcefield(self, name, format='lammps'):
        path = self.work_dir/name/f"{name}.ff"
        return mp.io.load_forcefield(path, format=format)