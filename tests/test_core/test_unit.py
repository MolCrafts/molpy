from pint import UnitRegistry
import numpy as np
import molpy as mp

class TestUnit:

    def test_unit(self):
        unit = mp.Unit()
        assert "angstrom" in unit
        unit.define("@alias angstrom = length")
        assert "length" in unit
        np.testing.assert_allclose((1 * unit.length).to("nanometer").magnitude , 0.1)

    def test_lj_unit(self):
        unit = mp.Unit(style="lj")
        assert "length" in unit

        np.testing.assert_allclose((1 * unit.nanometer).to(unit.length), 4.0)
        
