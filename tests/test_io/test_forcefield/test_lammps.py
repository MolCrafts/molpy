import pytest
import molpy as mp

class TestLammpsForceField:

    @pytest.fixture(scope='class')
    def forcefield(self, test_data_path):
        path = test_data_path / 'forcefield/lammps/'
        return mp.ForceField.from_file([path / 'in.peptide', path / 'data.peptide'])
    
    def test_styles(self, forcefield):
        
        assert forcefield.n_atom_styles == 1
        assert forcefield.atom_styles[0].name == 'full'

        assert forcefield.n_bond_styles == 1
        assert forcefield.bond_styles[0].name == 'harmonic'

        assert forcefield.n_angle_styles == 1
        assert forcefield.angle_styles[0].name == 'charmm'

        assert forcefield.n_dihedral_styles == 1
        assert forcefield.dihedral_styles[0].name == 'charmm'

        assert forcefield.n_improper_styles == 1
        assert forcefield.improper_styles[0].name == 'harmonic'

        assert forcefield.n_pair_styles == 1
        assert forcefield.pair_styles[0].name == 'lj/cut/coul/long'

    def test_types(self, forcefield):

        assert forcefield.atom_styles[0].n_types == 14
        assert forcefield.bond_styles[0].n_types == 18
        assert forcefield.angle_styles[0].n_types == 31
        assert forcefield.dihedral_styles[0].n_types == 21
        assert forcefield.improper_styles[0].n_types == 2
        assert forcefield.pair_styles[0].n_types == 14 * 14

    def test_atom_types(self, forcefield):

        atom_types = forcefield.atom_styles[0].types
        assert atom_types[0]['mass'] == 12.011
        assert atom_types[1]['mass'] == 12.011

        assert atom_types.mass.shape == (14,)

    def test_bond_types(self, forcefield):

        bond_types = forcefield.bond_styles[0].types
        assert bond_types[0]['k'] == 249.999999
        assert bond_types[0]['r0'] == 1.490000

    def test_angle_types(self, forcefield):

        angle_types = forcefield.angle_styles[0].types
        assert angle_types[0]['k'] == 33.000000

    def test_dihedral_types(self, forcefield):

        dihedral_types = forcefield.dihedral_styles[0].types
        assert dihedral_types[0]['k1'] == 0.000000

    def test_improper_types(self, forcefield):

        improper_types = forcefield.improper_styles[0].types
        assert improper_types[0]['k'] == 10.500000

    def test_pair_types(self, forcefield):

        pair_types = forcefield.pair_styles[0].types