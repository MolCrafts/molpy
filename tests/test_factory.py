# author: Roy Kid
# contact: lijichen365@126.com
# date: 2021-10-22
# version: 0.0.1

import pytest
from molpy.atom import Atom

from molpy.factory import fromDict

class TestFactory:
    
    # @pytest.fixture(scope='class')
    def testAtomJson(self):
        d = {'_itemType': 'Atom',
             'prop1': 'test1',
             '_format': {}}
        
        atomD = fromDict(d)
        assert atomD.prop1 == 'test1'
        assert isinstance(atomD, Atom)