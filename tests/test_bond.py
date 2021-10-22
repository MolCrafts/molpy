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
        A.type = 'left'
        B = Atom('B')
        B.type = 'right'
        yield A, B
    
    @pytest.fixture(scope='class')
    def bondAB(self, AB):
        A, B = AB
        yield A.bondto(B)
        
    @pytest.fixture(scope='class')
    def bondBA(self, AB):
        A, B = AB
        yield B.bondto(A)
    
    def test_identity(self, bondAB, bondBA):
        assert bondAB.atomType1 == 'left'
        assert bondAB == bondBA
        
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
        assert A.bondto(B) == C.bondto(D)