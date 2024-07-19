import molpy as mp

class TestElement:

    def test_get_by_number(self):

        assert mp.Element[1].symbol == "H"
        assert mp.Element[2].symbol == "He"

    def test_get_by_name(self):

        assert mp.Element["hydrogen"].symbol == "H"
        assert mp.Element["helium"].symbol == "He"

    def test_get_by_symbol(self):

        assert mp.Element["H"].name == "hydrogen"
        assert mp.Element["D"].name == "deuterium"
