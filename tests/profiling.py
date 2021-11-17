# author: Roy Kid
# contact: lijichen365@126.com
# date: 2021-11-16
# version: 0.0.1
import sys
sys.path.append('/home/roy/work/molpy')
import numpy as np
import molpy as mp
from memory_profiler import profile

import cProfile

@profile
def test_atoms_memory(n):
    tmp = []
    for _ in range(n):
        tmp.append(mp.Atom(str(_)))
    
def test_atoms_time(n):
    tmp = []
    for _ in range(n):
        tmp.append(mp.Atom(str(_)))

@profile
def test_group_memory(n):
    g = mp.Group('g')
    for _ in range(n):
        g.addAtom(mp.Atom(str(_)))
        if _ != 0:
            g.addBondByIndex(_-1, _)
            
def test_group_time(n):
    g = mp.Group('g')
    for _ in range(n):
        g.addAtom(mp.Atom(str(_)))
        if _ != 0:
            g.addBondByIndex(_-1, _)
            
def test_copy_group(n):
    g = mp.Group('g')
    for _ in range(n):
        g.addAtom(mp.Atom(str(_)))
        if _ != 0:
            g.addBondByIndex(_-1, _)    
            
    gg = g.copy()
    gg.natoms
    
if __name__ == '__main__':
    # profiling
    # cProfile.run('test_atoms_time(100)')
    # test_atoms_memory(100)
    
    # cProfile.run('test_group_time(100)')
    # test_group_memory(100)
    
    # cProfile.run('test_copy_group(10000)')
    pass
    
    
    