import molpy as mp
import numpy.testing as npt

class TestUnit:

    def test_convert(self):
        npt.assert_allclose(mp.Unit.convert(1, "angstrom", "nm"), 0.1)
        assert mp.Unit.convert(1, "fs", "ps") == 0.001
        assert mp.Unit.convert(1, "kcal/mol", "kJ/mol") == 4.184

    def test_reduce(self):
        mp.Unit.set_fundamental("1 g/mol", "1 nm", "1 kcal/mol")
        npt.assert_allclose(mp.Unit.reduce(1, "angstrom"), 0.1)
        npt.assert_allclose(mp.Unit.reduce(1, "fs"), 0.0020454828)

    def test_constants(self):

        assert mp.Unit.constants.boltzmann == 1.380649e-23