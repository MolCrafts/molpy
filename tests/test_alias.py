# author: Roy Kid
# contact: lijichen365@126.com
# date: 2023-10-04
# version: 0.0.1

from molpy import Aliases

class TestAliases:

    def test_default(self):

        alias = Aliases()
        assert alias.timestep.keyword == "_ts"
        assert alias.timestep.unit == "fs"

    def test_hash(self):

        alias = Aliases()
        data = {}
        data[alias.timestep] = 1
        assert data[alias.timestep] == 1

    def test_pickle(self):

        import pickle

        alias = Aliases()
        data = pickle.dumps(alias)
        alias = pickle.loads(data)
        assert alias.timestep.keyword == "_ts"
        assert alias.timestep.unit == "fs"

    def test_scope(self):

        alias = Aliases("test1")
        alias.set("mass", "_mass", "amu", "atomic mass")

        alias = Aliases("test2")
        alias.set("type", "_type", None, "atomic type")

        assert alias.type.keyword == "_type"
        assert alias.type.unit is None

        alias["test1"]
        assert alias.mass.keyword == "_mass"
        assert alias.mass.unit == "amu"
        