# author: Roy Kid
# contact: lijichen365@126.com
# date: 2023-10-04
# version: 0.0.1

from molpy import Keywords

class TestKeywords:

    def test_define(self):

        kw = Keywords("test")
        kw.set("timestep", "_mp_ts_", "fs", "time step")

        assert kw.timestep == "_mp_ts_"
        assert kw.get_alias("_mp_ts_") == "timestep"
        assert kw.get_keyword("timestep") == "_mp_ts_"
        assert kw.get_unit("timestep") == "fs"

    def test_iter(self):

        kw = Keywords("test")
        kw.set("timestep", "_mp_ts_", "fs", "time step")
        kw.set("name", "_mp_name_", None, "atomic name")

        for _kw in kw:
            assert _kw
