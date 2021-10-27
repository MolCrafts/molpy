# author: Roy Kid
# contact: lijichen365@126.com
# date: 2021-10-22
# version: 0.0.1

import pytest
from molpy.base import *

class TestBase:
    
    @pytest.fixture(scope='class')
    def itemA(self):
        a = Node('a')
        a.prop1 = 1.234
        yield a
        
