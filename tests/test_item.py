# author: Roy Kid
# contact: lijichen365@126.com
# date: 2021-10-22
# version: 0.0.1

import pytest
from molpy.item import Item

class TestBase:
    
    @pytest.fixture(scope='class')
    def itemA(self):
        a = Item('a')
        a.prop1 = 1.234
        yield a
        
    def test_serialize(self, itemA):
        pickle = itemA.serialize()
        assert pickle['prop1'] == 1.234
        assert pickle['_itemType'] == 'Item'
        
    def test_deserialize(self, itemA):
        pickle = itemA.serialize()
        item = Item().deserialize(pickle)
        assert item.prop1 == 1.234
        assert item.itemType == 'Item'
        