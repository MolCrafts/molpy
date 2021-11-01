# author: Roy Kid
# contact: lijichen365@126.com
# date: 2021-10-22
# version: 0.0.1

from typing import Literal
from molpy.atom import Atom
from molpy.base import Item
from molpy.group import Group
from molpy.bond import Bond
from molpy.angle import Angle
from molpy.dihedral import Dihedral

class Template(Group):
    
    def __init__(self, name, group=None, **attr):
        super().__init__(name, group=group, **attr)
        self.patches = {}  # {name: Template}
        
class AtomType(Item):
    
    def __init__(self, name, **attr) -> None:
        super().__init__(name)
        self.update(attr)
        
class BondType(Item):
    
    def __init__(self, name, **attr) -> None:
        super().__init__(name)
        self.update(attr)

class ForceField:
    
    def __init__(self, name) -> None:
        self.name = name
        self._templates = {}
        self._atomType = {}
        self._bondType = {}
        
    def defAtomType(self, name, **attr):
        atomType = self._atomType.get(name, None)
        if atomType is not None:
            raise KeyError(f'atomType {name} has been defined')
        self._atomType[name] = AtomType(name, **attr)
        
    def defBondType(self, name, **attr):
        BondType = self._atomType.get(name, None)
        if AtomType is not None:
            raise KeyError(f'bondType {name} has been defined')
        self._bondType[name] = BondType(name, **attr)
    
    def getAtomType(self, name):
        atomType = self._atomType.get(name, None)
        if atomType is None:
            raise KeyError(f'atomType {name} is not defined yet')
        return atomType
        
    def defTemplate(self, template: Template):
        
        self._templates[template.name] = template
        
    def getTemplateByName(self, groupName: str):
        for t in self._templates:
            if t.name == groupName:
                return t
            
    def matchTemplate(self, group: Group, criterion:Literal['low', 'medium', 'high']='low')->Template:
        """find the matching template in the forcefield

        Args:
            group (Group): group to be match
            
        Return:
            template (Group): template matched
        """
        
        def validate(group, altTemp: dict)->Template:
            
            if len(altTemp) == 2:
                raise KeyError(f'{len(altTemp)} template matched: {list(altTemp.keys())}')
            
            return altTemp[group.name]
        
        altTemp = {}
        
        # LOW standard, query whether there is a template with the same name
            
        if criterion == 'low':
            if group.name in self._templates:
                altTemp[group.name] = self._templates[group.name]
            
            return validate(group, altTemp)
            
        
        # MEDIUM, find which latent templates which have same atoms and bonds. Can not tell allotrope
        if criterion == 'medium':
            for templateName, template in self._templates.items():
                # for atom in group.atoms:
                #     if atom not in template.atoms:
                #         continue
                if sorted(group.atoms) == sorted(template.atoms):
                    altTemp[templateName] = template
            
            tmp = {}
            for templateName, template in altTemp.items():
                # for bond in group.bonds:
                #     if bond not in template.bonds:
                #          continue
                if sorted(group.bonds) == sorted(template.bonds):
                    tmp[templateName] = template
            altTemp = tmp
                
            return validate(group, altTemp)
            
        # HIGH, may check chiral or something
        
    def patchTemplate(self, group: Group, criterion:Literal['low', 'medium', 'high']='low'):
        template = self.matchTemplate(group, criterion)
        
        # It seems we only need to retrieve topology
        template.patch(group)

        for atom in group.atoms:
            self.patchAtom(atom)
        for bond in group.bonds:
            self.patchBond(bond)
            
    def patchAtom(self, atom):
        
        if atom.type.name in self._atomType:
            atom.type = self._atomType[atom.type.name]
            
    def patchBond(self, bond):
        
        if bond.type.name in self._bondType:
            bond.type = self._bondType[bond.type.name]
        
    def matchGroupOfBonds(self, group: Group, template: Group):
        groupBonds = sorted(group.getBonds())
        tempBonds = sorted(template.getBonds())
        
        for g, t in zip(groupBonds, tempBonds):
            g.update(**t.properties)
            
    def matchBond(self, bond: Bond, template: Group):
        for tarbond in template.getBonds():
            if tarbond.type == bond.type:
                bond.update(**tarbond.properties)
        return bond
    
