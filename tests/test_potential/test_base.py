import pytest
import numpy as np
import molpy as mp

class TestBasePotential:

    def test_bond_potential(self):

        bond_pot = mp.potential.BondPotential('harmonic', np.array([1.0, 2.0]), np.array([1.0, 2.0]))
        assert bond_pot.name == 'harmonic'
