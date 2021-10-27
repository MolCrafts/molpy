# # author: Roy Kid
# # contact: lijichen365@126.com
# # date: 2021-10-22
# # version: 0.0.1

# import pytest
# from molpy.atom import Atom
# from molpy.forcefield import ForceField
# from molpy.group import Group

# class TestForceField:
    
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