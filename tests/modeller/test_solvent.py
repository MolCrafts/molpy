# author: Roy Kid
# contact: lijichen365@126.com
# date: 2023-02-05
# version: 0.0.1

import numpy as np
import molpy as mp

class TestSolvent:

    def test_basic_single_particle(self):

        system = mp.System('single')
        system.box.reset(10, 10, 10)
        system.forcefield.def_atomType('A', mass=18, charge=11.8)

        template = mp.Molecule('LJ')
        template.add_atom(
            mp.Atom(name='A', type='A', xyz=[0, 0, 0])
        )

        solModeller = mp.modeller.Solvent(system.box)
        molecule_list = solModeller.add_solvent(100, template, seed=0)
        for m in molecule_list:
            system.add_molecule(m)

        system.render()
        system.write('test.data', 'LAMMPS Data')

