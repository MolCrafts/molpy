# author: Roy Kid
# contact: lijichen365@126.com
# date: 2022-07-18
# version: 0.0.1

from molpy.forcefield import ForceField

class TestForcefield:

    def test_load_xml(self):
        
        ff = ForceField()
        ff.load_xml('tests/test_io/data/forcefield.xml')
        assert 1