# # author: Roy Kid
# # contact: lijichen365@126.com
# # date: 2021-10-22
# # version: 0.0.1

# from molpy.group import Group
# from molpy.bond import Bond
# from molpy.atomType import AtomType

# class ForceField:
    
#     def __init__(self, name) -> None:
#         self.name = name
#         self._templates = {}
#         self._atomType = {}
        
#     def setAtomType(self, name, **attr):
#         atomType = self._atomType.get(name, None)
#         if atomType is not None:
#             raise KeyError(f'atomType {name} has been defined')
#         self._atomType[name] = AtomType(name, **attr)
    
#     def getAtomType(self, name):
#         atomType = self._atomType.get(name, None)
#         if atomType is None:
#             raise KeyError(f'atomType {name} is not defined yet')
#         return atomType
        
#     def registerTemplate(self, group: Group):
#         # self._templates[group] = group.properties
#         pass
        
#     def getTemplateByName(self, groupName: str):
#         for t in self._templates:
#             if t.name == groupName:
#                 return t
            
#     def matchTemplate(self, group: Group):
#         """TODO: True template matching

#         Args:
#             group (Group): group to be matched

#         Returns:
#             Group: matched template
#         """
#         for template in self._templates:
#             if template.type == group.type:
#                 return template
        
#     def matchGroupOfBonds(self, group: Group, template: Group):
#         groupBonds = sorted(group.getBonds())
#         tempBonds = sorted(template.getBonds())
        
#         for g, t in zip(groupBonds, tempBonds):
#             g.update(**t.properties)
            
#     def matchBond(self, bond: Bond, template: Group):
#         for tarbond in template.getBonds():
#             if tarbond.type == bond.type:
#                 bond.update(**tarbond.properties)
#         return bond