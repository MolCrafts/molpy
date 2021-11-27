# author: Roy Kid
# contact: lijichen365@126.com
# date: 2021-10-22
# version: 0.0.1

import pytest
import networkx as nx
from molpy import fromNetworkXGraph
from molpy import fromASE
import numpy as np
import molpy as mp

class TestNetworkX:
    def test_from_path_graph(self):
        
        G = fromNetworkXGraph('path_graph', nx.path_graph(10))
        assert G.natoms == 10
        assert G.nbonds == 9
        
    def test_from_cubical_graph(self):
        G = fromNetworkXGraph('cubical_graph', nx.cubical_graph())
        assert G.natoms == 8
        assert G.nbonds == 12
        
    def test_from_complete_graph(self):
        G = fromNetworkXGraph('complete_graph', nx.complete_graph(5))
        assert G.natoms == 5
        assert G.nbonds == 10

   
class TestAuxiliaryMethod:
    
    def test_trace(self, particle):
        
        rng = np.random.default_rng()
        sites = rng.random((5, 3))
        
        particleList = mp.trace(particle, sites)
        
        tmp = []
        for atom in particleList:
            tmp.append(atom.position)
            
        assert np.array_equal(sites, np.array(tmp))

    def test_from_ase(self):
        mol = ase.build.molecule("bicyclobutane")
        g   = fromASE(mol)
        symbols = mol.get_chemical_symbols()
        g_symbols = g.getSymbols()
        assert len(symbols) == len(g_symbols)
        assert all([a == b for a, b in zip(symbols, g_symbols)])    
        dpos = g.positions - mol.get_positions()
        assert not np.any(dpos)