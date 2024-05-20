import pytest
import molpy as mp

Item = mp.builder.base.Item
ItemList = mp.builder.base.ItemList


class TestDynamicStruct:

    @pytest.fixture(scope="class", name="item")
    def test_init(self):
        return Item(**{mp.Alias.name: "O", mp.Alias.atomtype: "o1"})

    def test_getitem(self, item):

        assert item[mp.Alias.name] == "O"
        assert item[mp.Alias.atomtype] == "o1"

        with pytest.raises(KeyError):
            item["not_exist_key"]

    def test_getattr(self, item):

        assert item.name == "O"
        assert item.atomtype == "o1"

        assert item.get(mp.Alias.name) == "O"

