# author: Roy Kid
# contact: lijichen365@126.com
# date: 2021-10-22
# version: 0.0.1

from typing import Literal, TypeVar
from molpy.atom import Atom
from molpy.base import Item
from molpy.group import Group
from molpy.bond import Bond
from molpy.angle import Angle
from molpy.dihedral import Dihedral
import molpy as mp

class Template(Group):
    
    def __init__(self, name, group=None, **attr):
        super().__init__(name, group=group, **attr)
        self.patches = {}  # {name: Template}
 
class AtomType(Item):
    
    atomTypeID = 1
    atomTypes = {}
    
    def __init__(self, name, **attr) -> None:
        if name not in AtomType.atomTypes:
            super().__init__(name)
            self.update(attr)
            self.atomTypeID = AtomType.atomTypeID
            AtomType.atomTypeID += 1
            AtomType.atomTypes['name'] = self
        else:
            return AtomType.atomTypes['name']
        
class BondType(Item):
    
    def __init__(self, name, **attr) -> None:
        super().__init__(name)
        self.update(attr)

class AngleType(Item):
    
    def __init__(self, name, **attr) -> None:
        super().__init__(name)
        self.update(attr)
        
class DihedralType(Item):
    
    def __init__(self, name, **attr) -> None:
        super().__init__(name)
        self.update(attr)

class ForceField:
    
    def __init__(self, name, unit='SI') -> None:
        self.name = name
        self._templates = {}
        self._atomTypes = {}
        self._bondTypes = {}
        self._angleTypes = {}
        self._dihedralTypes = {}
        
    @property
    def natomTypes(self):
        return len(self._atomTypes)
    
    @property
    def nbondTypes(self):
        return len(self._bondTypes)
    
    @property
    def ntemplates(self):
        return len(self._templates)
        
    def defAtomType(self, atomName, **attr):
        atomType = self._atomTypes.get(atomName, None)
        if atomType is not None:
            raise KeyError(f'atomType {atomName} has been defined')
        self._atomTypes[atomName] = AtomType(atomName, **attr)
        
    def defBondType(self, bondName, **attr):
        bondType = self._bondTypes.get(bondName, None)
        if bondType is not None:
            raise KeyError(f'bondType {bondName} has been defined')
        self._bondTypes[bondName] = BondType(bondName, **attr)
        
    def defAngleType(self, angleName, **attr):
        angleType = self._angleTypes.get(angleName, None)
        if angleType is not None:
            raise KeyError(f'angleType {angleName} has been defined')
        self._angleTypes[angleName] = AngleType(angleName, **attr)
        
    def defDihedralType(self, dihedralName, **attr):
        dihedralType = self._dihedralTypes.get(dihedralName, None)
        if dihedralType is not None:
            raise KeyError(f'dihedralType {dihedralName} has been defined')
        self._dihedralTypes[dihedralName] = DihedralType(dihedralName, **attr)
    
    def getAtomType(self, name):
        atomType = self._atomTypes.get(name, None)
        if atomType is None:
            raise KeyError(f'atomType {name} is not defined yet')
        return atomType
        
    def defTemplate(self, template: Template):
        
        self._templates[template.name] = template
        
    def getTemplateByName(self, groupName: str):
        for t in self._templates:
            if t.name == groupName:
                return t
            
    def matchTemplate(self, group: Group, criterion:Literal['low', 'medium', 'high']='medium')->Template:
        """find the matching template in the forcefield

        Args:
            group (Group): group to be match
            
        Return:
            template (Group): template matched
        """
        
        def validate(altTemp: dict)->Template:
            
            if len(altTemp) != 1:
                raise KeyError(f'{len(altTemp)} template matched: {list(altTemp.keys())}')
            
            return list(altTemp.values())[0]
        
        altTemp = {}
        
        # LOW standard, query whether there is a template with the same name
            
        if criterion == 'low':
            if group.name in self._templates:
                altTemp[group.name] = self._templates[group.name]
            
            return validate(altTemp)
            
        
        # MEDIUM, find which latent templates which have same atoms and bonds. Can not tell allotrope
        if criterion == 'medium' or criterion == 'high':
            
            for templateName, template in self._templates.items():
                if group.natoms == template.natoms:
                    for i, gatom in enumerate(group.atoms):
                        if not template.hasAtom(gatom, 'name'):
                            break
                    if i == group.natoms-1:
                        altTemp[templateName] = template
                    
            if criterion == 'high':
                
                tmp = {}
                sortedBond = sorted(group.bonds)
                for templateName, template in altTemp.items():
                    # for bond in group.bonds:
                    #     if bond not in template.bonds:
                    #          continue
                    if sortedBond == sorted(template.bonds):
                        tmp[templateName] = template
                altTemp = tmp
                
            return validate(altTemp)
        
    def patch(self, template: Template, group: Group):
        """patch topology info from template

        Args:
            template (Template): [description]
            group (Group): [description]
        """
        
        bonds = template.getBonds()
        for bond in bonds:
            atom, btom = bond
            group.addBondByName(atom.name, btom.name, **bond.properties)
            
        angleTs = template.getAngles()
        for T in angleTs:
            itom, jtom, ktom = T
            group.addAngleByName(itom.name, jtom.name, ktom.name, **T.properties)
            
    def render(self, group:TypeVar('Group-like', Group, Template)):
        """Add information from the forcefield to the group

        Args:
            group (Group): [description]
        """
        # render atom from forcefield.atomType
        for atom in group.atoms:
            self.renderAtom(atom)
        
        # render bond from forcefield.bondType
        for bond in group.bonds:
            self.renderBond(bond)
            
        # render angle from forcefield.angleType 
        for angle in group.angles:
            self.renderAngle(angle)

        # render dihedral from forcefield.dihedralType 
        for dihedral in group.dihedrals:
            self.renderDihedral(dihedral)
            
    def renderAtom(self, atom):
        
        pass
            
    def renderBond(self, bond):
        pass


    def renderAngle(self, angle):
        
        if angle.name in self._angleTypes:
            angle.type = self._angleTypes[angle.type.name]
            
    def renderDihedral(self, dihedral):
        
        if dihedral.name in self._dihedralTypes:
            dihedral.type = self._dihedralTypes[dihedral.type.name]           
        
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
    
    def loadXML(self, file):
        with open(file, 'r') as f:
            mp.read_xml_forcefield(f, create_using=self)
        return self

    @property
    def atomTypes(self):
        return list(self._atomTypes.keys())
    
    @property
    def bondTypes(self):
        return list(self._bondTypes.keys())
    
    @property
    def angleTypes(self):
        return list(self._angleTypes.keys())
    
    @property
    def dihedralTypes(self):
        return list(self._dihedralTypes.keys())
    