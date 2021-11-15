# author: Roy Kid
# contact: lijichen365@126.com
# date: 2021-10-28
# version: 0.0.1

from molpy.group import Group
from molpy.atom import Atom

def from_networkx_graph(name, G):
    
    adj = dict(G.adj)
    group = Group(name)
    
    atoms = [Atom(f'{i}') for i in adj]
    group.addAtoms(atoms)
        
    for uname, nbs in adj.items():
        for nb, dd in nbs.items():
            group.addBond(atoms[uname], atoms[nb], **dd)
            
    return group

def toJAXMD(group: Group):
    d = {}
    R = group.getPositions()
    d['positions'] = R
    atomTypes = group.getAtomTypes()
    d['atomTypes'] = atomTypes
    elements = group.getElements()
    d['elements'] = elements
    bonds = group.getAdjacencyList()
    d['bonds'] = bonds
    natoms = group.natoms
    d['natoms'] = natoms
    
    return d
    
def fromJAXMD():
    pass

def toJraph(group, node_features, edge_features, ):
    pass