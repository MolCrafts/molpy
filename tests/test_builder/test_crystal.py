import molpy as mp


class TestCrystal:

    def test_fcc(self):

        region = mp.Cube([0, 0, 0], 10)
        fcc = mp.builder.FCC(region, 1.0)
        lattice = fcc.create_lattice()

        assert lattice.shape == (4 * 10 ** 3, 3)
        assert lattice[0, 0] == 0.0
