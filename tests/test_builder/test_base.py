import pytest
from abc import ABCMeta

from molpy.builder.base import LatticeBuilder, StructBuilder, BuildManager

def test_base_abstract():
    # klasy bazowe muszą być abstrakcyjne (nie można ich instancjonować)
    with pytest.raises(TypeError):
        LatticeBuilder()
    with pytest.raises(TypeError):
        StructBuilder()

def test_build_manager_calls_correct_methods(monkeypatch):
    class DummyLattice(LatticeBuilder):
        def create_sites(self, **params):
            assert params == {'foo': 1}
            return ['SITES']

    class DummyStruct(StructBuilder):
        def populate(self, sites, **params):
            assert sites == ['SITES']
            assert params == {'bar': 2}
            return 'STRUCT'

    manager = BuildManager(DummyLattice(), DummyStruct())
    out = manager.build(foo=1, bar=2)
    assert out == 'STRUCT'
