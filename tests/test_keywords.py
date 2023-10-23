# author: Roy Kid
# contact: lijichen365@126.com
# date: 2023-10-04
# version: 0.0.1

import pickle
from molpy import Alias

class TestKeywords:

    def test_define(self):

        ts = Alias("timesteps", "_mp_ts_", "fs", "time step")

        assert ts.keyword == "_mp_ts_"

    def test_get_alias(self):

        ts = Alias.timestep
        assert ts.keyword == "_mp_ts_"

    def test_pickle(self):

        timesteps = Alias("timesteps", "_mp_ts_", "fs", "time step")
        obj = pickle.dumps(timesteps)
        alias = pickle.loads(obj)

        assert isinstance(alias, Alias)
        assert isinstance(alias.keyword, str)
        assert alias.keyword == "_mp_ts_"