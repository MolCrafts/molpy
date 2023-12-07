# author: Roy Kid
# contact: lijichen365@126.com
# date: 2023-10-04
# version: 0.0.1

from molpy import Alias

class TestAliases:

    def test_default(self):

        alias = Alias()
        assert alias.timestep.keyword == "_ts"
        assert alias.timestep.unit == "fs"

    def test_pickle(self):

        import pickle

        alias = Alias()
        data = pickle.dumps(alias)
        alias = pickle.loads(data)
        assert alias.timestep.keyword == "_ts"
        assert alias.timestep.unit == "fs"

    def test_scope(self):

        alias = Alias("test1")
        alias.test1.set("mass", "_mass", float, "amu", "atomic mass")
        assert alias.test1.mass.keyword == "_mass"
        assert alias.test1.mass.unit == "amu"

        assert alias.energy.keyword == "_energy"
        assert alias.energy.unit is "meV"

        