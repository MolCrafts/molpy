# author: Roy Kid
# contact: lijichen365@126.com
# date: 2021-11-18
# version: 0.0.1

import pytest
import molpy as mp

class TestAngle:
    
    @pytest.fixture(scope='class')
    def angle(self):
        itom = mp.Atom('i')
        jtom = mp.Atom('j')
        ktom = mp.Atom('k')
        angle = mp.Angle(itom, jtom, ktom)
        yield angle

    @pytest.fixture(scope='class')
    def angleType(self):
        ff = mp.ForceField('test')
        at = ff.defAngleType('ijk', itomName='i', jtomName='j', ktomName='k', style='harmonic', k=30.0, theta0=114) 
        assert at is not None
        yield at
        
    def test_equal(self, angle, angleType):
        assert angle.atomNameEqualTo(angleType)
        angleType.render(angle)
        assert angle.style == 'harmonic'
        assert angle.k == 30.0
        assert angle.theta0 == 114
        