# author: Roy Kid
# contact: lijichen365@126.com
# date: 2023-02-04
# version: 0.0.1

import molpy as mp

class TestSystem:

    def test_write(self):

        h2o = mp.presets.molecules.tip3p()
        ff = mp.presets.forcefields.tip3p()
        system = mp.System('test_write_hip3p')
        system.forcefield = ff

        system.add_molecule(h2o)
        system.add_molecule(h2o.translate([0, 0, 3]))

        assert len(system.molecules) == 2

        system.box.reset(10, 10, 10, 0, 0, 0, 0, 0, 7.07)

        system.write('tip3p.data', 'LAMMPS Data')