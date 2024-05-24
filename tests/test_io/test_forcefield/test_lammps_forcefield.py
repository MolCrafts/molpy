import pytest
import molpy as mp

class TestLammpsForceField:

    @pytest.fixture(scope='class')
    def forcefield(self, test_data_path):
        path = test_data_path / 'forcefields/lammps/'
        return mp.load_forcefield(path / 'peptide.in', 'lammps', [path / 'peptide.data']).load()
    
    def test_styles(self, forcefield):
        
        assert forcefield.n_atomstyles == 1
        assert forcefield.atomstyles[0].name == 'full'

        assert forcefield.n_bondstyles == 1
        assert forcefield.bondstyles[0].name == 'harmonic'

        assert forcefield.n_anglestyles == 1
        assert forcefield.anglestyles[0].name == 'charmm'

        assert forcefield.n_dihedralstyles == 1
        assert forcefield.dihedralstyles[0].name == 'charmm'

        assert forcefield.n_improperstyles == 1
        assert forcefield.improperstyles[0].name == 'harmonic'

        assert forcefield.n_pairstyles == 1
        assert forcefield.pairstyles[0].name == 'lj/charmm/coul/long'

    def test_types(self, forcefield):

        assert forcefield.n_atomtypes == 14
        assert forcefield.n_bondtypes == 18
        assert forcefield.n_angletypes == 31
        assert forcefield.n_dihedraltypes == 21
        assert forcefield.n_impropertypes == 2
        assert forcefield.n_pairtypes == 14

    def test_atom_types(self, forcefield):

        atom_types = forcefield.atomstyles[0].types
        assert atom_types[0]['mass'] == 12.011
        assert atom_types[1]['mass'] == 12.011

        assert atom_types.mass.shape == (14,)

    def test_bond_types(self, forcefield):

        bond_types = forcefield.bondstyles[0].types
        assert bond_types[0]['k'] == 249.999999
        assert bond_types[0]['r0'] == 1.490000

    def test_angle_types(self, forcefield):

        angle_types = forcefield.anglestyles[0].types
        assert angle_types[0]['k'] == 33.000000

    def test_dihedral_types(self, forcefield):

        dihedral_types = forcefield.dihedralstyles[0].types
        assert dihedral_types[0]['k1'] == 0.000000

    def test_improper_types(self, forcefield):

        improper_types = forcefield.improperstyles[0].types
        assert improper_types[0]['k'] == 10.500000

    def test_pair_types(self, forcefield):

        pair_types = forcefield.pairstyles[0].types