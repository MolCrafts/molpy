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