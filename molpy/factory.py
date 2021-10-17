# author: Roy Kid
# contact: lijichen365@126.com
# date: 2021-10-17
# version: 0.0.1

from molpy.atom import Atom
from molpy.group import Group


def full(groupName, atomNames, **properties):
    """ build up a group with atoms

    Args:
        names (List[atomName]): list of atomName to init Atoms
        properties (Dict[str:List]): the len of lists should match with names list
    """
    _atoms = []
    for i, name in enumerate(atomNames):
        atom = Atom(name)
        for k, v in properties.items():
            setattr(atom, k, v[i])
            
    group = Group(groupName)
    for atom in _atoms:
        group.add(atom)
        
    return group

def fromPDB():
    pass

def fromLAMMPS():
    pass

