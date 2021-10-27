# author: Roy Kid
# contact: lijichen365@126.com
# date: 2021-10-22
# version: 0.0.1

import pytest
from molpy.atom import Atom

class TestBond:
    
    @pytest.fixture(scope='class')
    def AB(self, ):
        A = Atom('A')
        B = Atom('B')
        yield A, B
    
    @pytest.fixture(scope='class')
    def bondAB(self, AB):
        A, B = AB
        yield A.bondto(B)
        
    @pytest.fixture(scope='class')
    def bondBA(self, AB):
        A, B = AB
        yield B.bondto(A)
        
    def test_sync(self, bondAB, bondBA):
        bondAB.prop = 1
        assert bondBA.prop == 1
        
    @pytest.fixture(scope='class')
    def CD(self, ):
        C = Atom('C')
        C.type = 'left'
        D = Atom('D')
        D.type = 'right'
        yield C, D
        
    def test_signature(self, AB, CD):
        A, B = AB
        C, D = CD
        assert A.bondto(B) != C.bondto(D)
        assert A.bondto(B) == B.bondto(A)
        
    def test_attribute(self, AB):
        A, B = AB
        bond = A.bondto(B)
        bond.update({'attr1': 1})
        assert bond.attr1 == 1
        
    def test_unpack(self, AB):
        A, B = AB
        bond = A.bondto(B)
        a, b = bond
        assert a == A
        assert b == B