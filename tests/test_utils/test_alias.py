import molpy as mp

class TestAlias:

    xyz = mp.alias.xyz
    assert xyz == "xyz"
