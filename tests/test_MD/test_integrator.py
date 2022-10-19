# author: Roy Kid
# contact: lijichen365@126.com
# date: 2022-10-05
# version: 0.0.1

import pytest
import molpy as mp
import numpy as np
import numpy.testing as npt
from molpy.core.nblist import NeighborList

class TestIntegrator:

    @pytest.mark.skip(reason="API frequently change")
    def test_verlet(self):

        system = mp.System()
        box = system.set_box(10, 10, 10)

        assert box.Lx == box.Ly == box.Lz == 10

        system.add_atoms(xyz=np.array([[0., 0., 0.], [1.5, 0., 0.]]),
            type=['c', 'c'], mass=np.array([1, 1]))
        
        assert system.frame.n_atoms == 2

        atomType_c = system.forcefield.def_atom('c', 'C', )

        assert atomType_c.name == 'c'
        assert atomType_c.typeClass == 'C'

        pairType_cc = system.forcefield.def_pair(system.forcefield.PairStyle.lj_cut, atomType_c, atomType_c, epsilon=1.0, sigma=1.0)

        assert pairType_cc.type1 == atomType_c
        assert pairType_cc.type2 == atomType_c
        assert pairType_cc.param['epsilon'] == 1.0
        assert pairType_cc.param['sigma'] == 1.0

        staticFrame = system.frame.to_static()
        nblist = NeighborList(system.frame.box, staticFrame['xyz'], dict(r_max=2.0, exclude_ii=True))

        pairs = nblist.get_pairs()

        assert pairs.shape[-1] == 2

        types = staticFrame['type']

        assert len(types) == 2

        atomTypes_id1 = types[pairs[:, 0]]
        atomTypes_id2 = types[pairs[:, 1]]

        assert atomTypes_id1 == np.array(['c'])
        assert atomTypes_id2 == np.array(['c'])

        atomType_pairs = []

        for p1, p2 in zip(atomTypes_id1, atomTypes_id2):
            at1 = system.forcefield.get_atom(p1)
            at2 = system.forcefield.get_atom(p2)
            atomType_pairs.append((at1, at2))

        assert len(atomType_pairs) == 1

        params = system.forcefield.get_pairs(atomType_pairs)

        assert len(params['epsilon']) == 1
        assert len(params.sigma) == 1

        # run first MD loop
        integrator = mp.MD.integrator.Verlet(dt=0.001)

        r = staticFrame['xyz']
        v = np.zeros_like(r)
        f = np.zeros_like(r)
        m = staticFrame['mass']
        
        # first-half step
        r, v = integrator.initial_integrate(r, v, f, m)

        # calc force
        ljcut = mp.potential.LJCut(r_cutoff=4.0, is_pbc=True)
        energy = ljcut.energy(r, pairs, box.to_matrix(), params)
        force = f = ljcut.force(r, pairs, box.to_matrix(), params)

        npt.assert_allclose(energy, -0.3203366)
        npt.assert_allclose(force, np.array([[-1.158029, 0.0, 0.0], [1.158029, 0.0, 0.0]]), rtol=1e6)

        # second-half step
        v = integrator.final_integrate(v, f, m)
        
        # run next MD loop
        r, v = integrator.initial_integrate(r, v, f, m)
        f = ljcut.force(r, pairs, box.to_matrix(), params)
        v = integrator.final_integrate(v, f, m)