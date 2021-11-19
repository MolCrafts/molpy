# author: Roy Kid
# contact: lijichen365@126.com
# date: 2021-10-22
# version: 0.0.1

import pytest
from molpy.base import Item
from molpy.base import *

class TestItem:
    
    @pytest.fixture(scope='class')
    def item(self):
        a = Item('a')
        a.prop1 = 1.234
        yield a
        
    def test_name(self, item):
        assert item.name == 'a'
        
    def test_hash(self, item):
        assert hash(item)
        
    def test_chech(self, item):
        assert item.check_properties(name=str)
        
    def test_set(self, item):
        setattr(item, 'prop2', 'test')
        
    def test_get(self, item):
        assert getattr(item, 'prop2') == 'test'
        
class TestNode:
    
    @pytest.fixture(scope='class')
    def A(self):
        A = Node('A')
        yield A
        
class TestEdge:
    
    @pytest.fixture(scope='class')
    def AB(self):
        A = Node('A')
        B = Node('B')
        yield Edge('AB')
        
class TestGraph:
    
    @pytest.fixture(scope='class')
    def G(self):
        yield Graph('G')