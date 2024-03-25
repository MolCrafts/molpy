import molpy as mp

class TestAlias:

    def test_get_str_alias(self):

        assert mp.Alias.timestep == "_ts"

    def test_get_alias(self):

        assert mp.Alias["timestep"].alias == "timestep"

    def test_new_scope(self):

        mp.Alias("test")
        mp.Alias.test.set("timestep", "timestep", int, "fs", "simulation timestep")
        assert "test" in mp.Alias._scopes
        assert mp.Alias.test.timestep == "timestep"

    def test_hash_alias(self):

        assert hash(mp.Alias.timestep)

    def test_override(self):

        mp.Alias('rmd17')
        mp.Alias.rmd17.set("energy", "_rmd_energy", float, "kcal/mol", "RMD energy")
        assert mp.Alias.energy == '_energy'