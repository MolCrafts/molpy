# author: Roy Kid
# contact: lijichen365@126.com
# date: 2021-10-22
# version: 0.0.2

import pytest
from molpy.atom import Atom
from molpy.forcefield import ForceField
from molpy.group import Group
from molpy import fromPDB
from pathlib import Path

from molpy.io.xml import read_xml_forcefield

class TestForceField:
    
    @pytest.fixture(scope='class')
    def H2Os(self):
        yield fromPDB(Path(__file__).parent/'samples/waterbox_31ang.pdb')
    
    @pytest.fixture(scope='class')
    def mpidff(self):
        yield read_xml_forcefield(Path(__file__).parent/'samples/mpidwater.xml')
    
    @pytest.fixture(scope='class')
    def H2Ogt(self, H2Os, mpidff):
        oneH2O = H2Os[1]
        yield oneH2O, mpidff.matchTemplate(oneH2O, criterion='medium')
    
    def testReadH2OandFF(self, H2Os, mpidff):
        assert len(H2Os) == 996
        assert mpidff.natomTypes == 2
        assert mpidff.nbondTypes == 1
        
    def testMatchTemplate(self, H2Os, mpidff):
        
        oneH2O = H2Os[1]
        assert oneH2O.natoms == 3
        assert mpidff.matchTemplate(oneH2O, criterion='medium')
        
    def testPatch(self, H2Ogt, mpidff):
        oneH2O, H2OT = H2Ogt
        mpidff.patch(H2OT, oneH2O)
        
        assert oneH2O.nbonds == H2OT.nbonds
        
    # def testRenderAtom(self, H2Ogt, mpidff):
    #     oneH2O, H2OT = H2Ogt
    #     O = oneH2O.atoms[0]
    #     mpidff.renderAtom(O)
    #     assert O.name == 'O'
    #     assert O.type.get('class') == 'OW'
    #     assert O.type.mass == '15.999'
        
    # def testRenderBond(self, H2Ogt, mpidff):
    #     oneH2O, H2OT = H2Ogt
    #     HO = oneH2O.bonds[0]
    #     mpidff.renderBond(HO)
    #     assert HO.class1 == 'OW'   
        
    
#     @pytest.fixture(scope='class')
#     def AB(self, ):
#         A = Atom('A')
#         A.type = 'left'
#         B = Atom('B')
#         B.type = 'right'
#         yield A, B
    
#     @pytest.fixture(scope='class')
#     def bondAB(self, AB):
#         A, B = AB
#         yield A.bondto(B, type='A-B')
        
#     @pytest.fixture(scope='class')
#     def groupAB(self, AB):
#         group = Group('AB')
#         group.type = 'ab'
#         group.addAtoms(AB)
#         yield group
        
#     @pytest.fixture(scope='class')
#     def forcefield(self):
#         ff = ForceField('ABff')
#         template = Group('AB')
#         template.type = 'ab'
#         AT = Atom('A')
#         AT.type = 'left'
#         BT = Atom('B')
#         BT.type = 'right'
#         template.addAtoms([AT, BT])
#         template.addBond(AT, BT, type='A-B', k=1.234)
#         ff.registerTemplate(template)
#         yield ff
        
#     def test_forcefield(self, forcefield):
#         ff = forcefield
#         template = ff.getTemplateByName('AB')
#         assert template.nbonds == 1
#         assert template.getBonds()[0].k == 1.234
#         assert template.getAtomByName('A').type == 'left'
        
#     def test_matchTemplate(self, groupAB, forcefield):
#         """it doesn't support auto match yet,
#             so if group.type == forcefield.type, match

#         Args:
#             group (group): a unknown group
#             forcefield (ForceField): forcefield
#         """
#         template = forcefield.matchTemplate(groupAB)
#         assert template.type == groupAB.type
        
#     def test_matchBond(self, bondAB, forcefield):
#         forcefield.matchBond(bondAB, forcefield.getTemplateByName('AB'))
#         assert bondAB.k == 1.234