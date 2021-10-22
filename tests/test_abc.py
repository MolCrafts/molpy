from molpy.utils import check_properties
from molpy.base import Item

import pytest

def test_check_properties():
    item = Item('test')
    
    @check_properties(item, position='required')
    def use_prop1():
        return True
    
    item.use_prop1 = use_prop1
    
    @check_properties(item, name='required')
    def use_prop2():
        return True
    
    item.use_prop2 = use_prop2
    
    with pytest.raises(AttributeError):
        assert item.use_prop1()
        assert item.moveTo()
        
    assert item.use_prop2()