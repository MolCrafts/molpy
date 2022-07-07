# author: Roy Kid
# contact: lijichen365@126.com
# date: 2022-07-03
# version: 0.0.1

from itertools import combinations
from typing import Dict, List, Optional, Tuple
import networkx as nx
import numpy as np
from itertools import combinations

class PyGraph:

    def __init__(self):

        self._adj = {}

    def add_edge(self, i, j, **attr):

        if i not in self._adj:
            self._adj[i] = {}
            if j not in self._adj:
                self._adj[j] = attr
        if j not in self._adj:
            self._adj[j] = {}
            if i not in self._adj:
                self._adj[i] = attr

        self._adj[i][j].update(attr)
        self._adj[j][i].update(attr)

    def calc_edges(self):
        tmp = []
        for i in self._adj:
            for j in self._adj[i]:
                tmp.append([i, j])
        bonds = np.array(tmp)
        bonds = np.where(bonds[:0]>bonds[1:], bonds[:, ::-1], bonds)
        bonds = np.unique(bonds, axis=0)
        return bonds

class Graph(nx.Graph):

    def calc_edges(self):
        adj = self._adj
        tmp = []
        for i, js in adj.items():
            for j in js:
                tmp.append([i, j])
        bonds = np.array(tmp)
        bonds = np.where((bonds[:, 0]>bonds[:, 1]).reshape((-1, 1)), bonds[:, ::-1], bonds)
        bonds = np.unique(bonds, axis=0)
        return bonds

    def calc_angles(self):
        adj = self._adj
        tmp = []
        for c, ps in adj.items():
            if len(ps) < 2:
                continue
            for (i, j) in combinations(ps, 2):
                tmp.append([i, c, j])
            
        angles = np.array(tmp)
        angles = np.where((angles[:,0]>angles[:,2]).reshape((-1, 1)), angles[:, ::-1], angles)
        angles = np.unique(angles, axis=0)
        return angles        

    def calc_dihedrals(self):

        topo = self._adj
        rawDihes = []
        for jtom, ps in topo.items():
            if len(ps) < 2:
                continue
            for (itom, ktom) in combinations(ps, 2):
                
                for ltom in topo[itom]:
                    if ltom != jtom:
                        rawDihes.append([ltom, itom, jtom, ktom])
                for ltom in topo[ktom]:
                    if ltom != jtom:
                        rawDihes.append([itom, jtom, ktom, ltom])
        
        # remove duplicates
        dihedrals = np.array(rawDihes)
        dihedrals = np.where((dihedrals[:,1]>dihedrals[:,2]).reshape((-1, 1)), dihedrals[:, ::-1], dihedrals)
        dihedrals = np.unique(dihedrals, axis=0)
        return dihedrals


class Topo:

    def __init__(self):

        self._g = Graph()
        self._edges:Dict[int, Dict[int, int]] = dict()
        self._angles:Dict[int, Dict[int, Dict[int, int]]] = dict()
        self._dihedrals:Dict[int, Dict[int, Dict[int, Dict[int, int]]]] = dict()

    # ---= data load interface =---
    def add_nodes(self, ids:List[int]):

        self._g.add_nodes_from(ids)

    def add_edges(self, connects:List[List[int]], ids:Optional[List[int]]=None):

        if ids is None:
            for i in range(len(connects)):
                self.add_edge(*connects[i])
        else:
            for i in range(len(connects)):
                self.add_edge(*connects[i], ids[i])

    def add_angles(self, angles:List[List[int]], ids:Optional[List[int]]=None):

        for i in range(len(angles)):
            self.add_angle(*angles[i], ids[i])

    def add_edge(self, i, j, index:Optional[int]=None):

        self._g.add_edge(i, j)
        
        if i not in self._edges:
            self._edges[i] = dict()
        if j not in self._edges:
            self._edges[j] = dict()

        self._edges[i][j] = index
        self._edges[j][i] = index

    def add_angle(self, i, j, k, index):

        self._g.add_edges_from([(i, j), (j, k)])

        if i not in self._angles:
            self._angles[i] = dict()
        if j not in self._angles[i]:
            self._angles[i][j] = dict()
        if k not in self._angles:
            self._angles[k] = dict()
        if j not in self._angles[k]:
            self._angles[k][j] = dict()
        self._angles[i][j][k] = index
        self._angles[k][j][i] = index

    def add_dihedral(self, i, j, k, l, index):

        self._g.add_edges_from([(i, j), (j, k), (k, l)])

        if i not in self._dihedrals:
            self._dihedrals[i] = dict()
        if j not in self._dihedrals[i]:
            self._dihedrals[i][j] = dict()
        if k not in self._dihedrals[i][j]:
            self._dihedrals[i][j][k] = dict()

        if l not in self._dihedrals:
            self._dihedrals[l] = dict()
        if j not in self._dihedrals[l]:
            self._dihedrals[l][j] = dict()
        if k not in self._dihedrals[l][j]:
            self._dihedrals[l][j][k] = dict()

        self._dihedrals[i][j][k][l] = index
        self._dihedrals[l][j][k][i] = index

    def get_edge(self, i, j):

        return self._edges[i][j]

    def get_angle(self, i, j, k):

        return self._angles[i][j][k]

    def get_dihedral(self, i, j, k, l):

        return self._dihedrals[i][j][k][l]

    def calc_edges(self):

        return self._g.calc_edges()

    def calc_angles(self):

        return self._g.calc_angles()

    def calc_dihedrals(self):

        return self._g.calc_dihedrals()
