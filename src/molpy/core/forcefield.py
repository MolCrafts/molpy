# author: Roy Kid
# contact: lijichen365@126.com
# date: 2023-09-29
# version: 0.0.1

from typing import Any
import numpy as np

class ItemType:

    def __init__(self, **props):
        self.props = props
        self._id = None
        
    @property
    def id(self):
        return self._id
    
    @id.setter
    def id(self, value:int):
        self._id = value

    def __eq__(self, other):
        return self.id == other.id

class AtomType(ItemType):

    def __init__(self, name:str, **props):
        super().__init__(**props)
        self.name = name

class BondType(ItemType):

    def __init__(self, atomType1:AtomType, atomType2:AtomType, **props):
        super().__init__(**props)
        self.atomType1 = atomType1
        self.atomType2 = atomType2

class ForceField:
    """
    """
    def __init__(self):
        self._atomTypes = []
        self._bondTypes = []

        self._bondPots = []
        self._bondParams = ParamSet()

    def def_atomtype(self, name:str, **props):
        _at = AtomType(name, **props)
        for i, at in enumerate(self._atomTypes):
            if at is None:
                _at.id = i
                self._atomTypes[i] = _at
        else:
            _at.id = len(self._atomTypes)
            self._atomTypes.append(_at)
        return _at
    
    def def_bondtype(self, atomType1:AtomType, atomType2:AtomType, **props):
        _bt = BondType(atomType1, atomType2, **props)
        for i, bt in enumerate(self._bondTypes):
            if bt is None:
                bt._id = i
                self._bondTypes[i] = bt
        else:
            _bt.id = len(self._bondTypes)
            self._bondTypes.append(_bt)
        return _bt
    
    # def get_bondpot(self, style, **args):

    #     bondpot = create_bondPot(style, **args)
    #     self._bondPots.append(bondpot)

    #     return 
    
    # def set_bondpot_param(self, style, bond_ids, value):

    #     self._bondParams.set_param(style, bond_ids, value)

    # def get_bond_param(self, style):

    #     return self._bondParams.get_param(style)

    # @classmethod
    # def from_xml(cls, path):
    #     import xml.etree.ElementTree as ET
    #     ff = cls()
    #     root = ET.parse(path).getroot()  # <ForceField>

    #     # get atomTypes
    #     atomTypes = root.find("AtomTypes")
    #     if atomTypes is None:
    #         raise ValueError("ForceField XML file must have AtomTypes tag")

    #     for at in atomTypes:
    #         at_dict = at.attrib
    #         ff.def_atomType(at_dict.pop('name'), className=at_dict.pop('class', None), **at_dict)

    #     # get residues
    #     residues = root.find("Residues")

    #     if residues:
    #         for residue in residues:
                
    #             re = ff.def_residue(residue.attrib['name'])

    #             atomTypes = residue.findall("Atom")
    #             if atomTypes:
    #                 for at in atomTypes:
    #                     at_dict = deepcopy(at.attrib)
    #                     re.def_atomType(at_dict.pop('name'), **at_dict)

    #             bondTypes = residue.findall("Bond")
    #             if bondTypes:
    #                 for bt in bondTypes:
    #                     bt_dict = deepcopy(bt.attrib)
    #                     re.def_bondType(
    #                         re.get_atomType(bt_dict.pop('atomName1')),
    #                         re.get_atomType(bt_dict.pop('atomName2')),
    #                         **bt_dict
    #                     )

    #     # get force
    #     for force in root:
    #         if force.tag.endswith("Force"):
    #             force_name = force.tag[:-5]
    #             for item in force:
    #                 if item.tag == "Bond":
    #                     bt_dict = deepcopy(item.attrib)
    #                     atom1 = bt_dict.pop('class1', 'atomName1')
    #                     atom2 = bt_dict.pop('class2', 'atomName2')
    #                     atomClass1 = ff._parameters.atomClasses[atom1]
    #                     atomClass2 = ff._parameters.atomClasses[atom2]
    #                     ff.def_bondType(atomClass1, atomClass2, style=force_name[:-4], **bt_dict)

    #     return ff

class Potential:

    def __init__(self):
        pass

    @property
    def styly(self):
        return self.__class__.__name__
    

class Param:

    def __init__(self, name):

        self.name = name
        self._params:dict[tuple, Any] = {}

    def set_param(self, idx:tuple, value:Any):

        self._params[tuple(idx)] = value

    def get_param(self, idx:tuple):

        return self._params.get(tuple(idx), None)
    
class PairParam(Param):

    def __init__(self, name):

        super().__init__(name)

    def set_param(self, idx:tuple, value:Any):
        assert len(idx) == 2, "pair param must have two index"
        super().set_param(idx, value)

    def get_param(self, idx:tuple, mix:str='geometry'):
        assert len(idx) == 2, "pair param must have two index"
        a = super().get_param(idx)
        if a:
            return a
        else:
            b = super().get_param(idx[::-1])
            if b:
                return b
            else:
                if mix:
                    a = super().get_param((idx[0], idx[0]))
                    b = super().get_param((idx[1], idx[1]))
                    if mix == 'geometry':
                        return np.sqrt(a*b)
                    elif mix == 'arithmetic':
                        return 0.5*(a+b)
                else:
                    return None
                
class AngleParam(Param):

    def __init__(self, name):
            
        super().__init__(name)

    def set_param(self, idx:tuple, value:Any):
        assert len(idx) == 3, "angle param must have three index"
        super().set_param(idx, value)

    def get_param(self, idx:tuple):
        assert len(idx) == 3, "angle param must have three index"
        a = super().get_param(idx)
        if a:
            return a
        else:
            return super().get_param(idx[::-1])

  
class ParamSet:
    """ A sparse matrix for parameters
    """
    def __init__(self):

        self._params:dict[str, Param] = {}

    def set_param(self, name, value):

        self._params[name] = value

    def get_param(self, name):

        return self._params.get(name, None)
              