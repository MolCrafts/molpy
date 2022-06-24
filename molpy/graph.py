# author: Roy Kid
# contact: lijichen365@126.com
# date: 2022-06-22
# version: 0.0.1

from typing import Dict
import numpy as np
from numpy.lib import recfunctions as rfn
import networkx as nx

class Topo(nx.Graph):

    pass

class AttribHolder:

    def __init__(self):

        self._nodes = {}
        self._edges = {}

        self._n_nodes = 0
        self._n_edges = 0

    def append_nodes(self, **attr):

        assert AttribHolder.is_align(attr.values()), ValueError()
        offset = self._n_nodes

        for field, value in attr.items():
            if field in self._nodes:
                
                now = self._nodes[field]

                self._nodes[field] = np.concatenate([now, value])

            else:

                self._nodes[field] = np.array(value)

        ids = AttribHolder.get_continuous_id(len(value), offset)
        self._n_nodes += len(value)

        return ids


    def append_edges(self, **attr):

        assert AttribHolder.is_align(attr.values()), ValueError()
        offset = self._n_edges

        for field, value in attr.items():
            if field in self._edges:
                self._edges[field] = np.concatenate([self._edges[field], value])

            else:
                self._edges[field] = np.array(value)

        ids = AttribHolder.get_continuous_id(len(value), offset)
        self._n_edges += len(value)
        return ids

    @staticmethod
    def is_align(values)->int:
        """ check if input attributes are aligned.

        Args:
            values (_type_): _description_

        Returns:
            int: _description_
        """
        lengths = list(map(len, values))
        maxLen = max(lengths)
        minLen = min(lengths)
        return maxLen == minLen

    @staticmethod
    def get_continuous_id(length, offset):

        return np.arange(length).astype(int) + offset

    def get_nodes(self):

        return self._nodes

    def get_edges(self):
        
        return self._edges

    def get_node_slice(self, slice):

        s = {field: value[slice] for field, value in self._nodes.items() }

        return s

    def get_edge_slice(self, slice):

        s = {field: value[slice] for field, value in self._edges.items() }

        return s

    def get_node_attr(self, field):
        return self._nodes[field]

    def get_edge_attr(self, field):
        return self._edges[field]

    @property
    def nodes(self):

        return self.get_nodes()

    @property
    def edges(self):

        return self.get_edges()

class Graph:

    def __init__(self):

        self.topo = Topo()
        self.attribs = AttribHolder()

        self._n_nodes = 0
        self._n_edges = 0

    def add_nodes(self, **attr):
        """ add nodes to the graph by providing aligned attributes.

        """

        node_ids = self.attribs.append_nodes(**attr)

        self.topo.add_nodes_from(node_ids)


    def add_edges(self, pairs, **attr):

        edge_ids = self.attribs.append_edges(**attr)

        pairs = [(*pair,  edgeId) for pair, edgeId in zip(pairs, edge_ids)]

        self.topo.add_weighted_edges_from(pairs)

    def get_edges_attr(self, key):
        """ get an attribute of all edges

        Args:
            key (str): a field of edge attribute

        Returns:
            np.NDArray: $(N_{n_edges}, `attrib.shape`)$
        """
        return self.attribs.get_edge_attr(key)


    def get_nodes_attr(self, key):
        """ get an attribute from all nodes

        Args:
            key (str): a field of edge attribute
        
        Returns:
            np.NDArray: $(N_{n_nodes}, `attrib.shape`)$
        """
        return self.attribs.get_node_attr(key)

    def get_node_attr(self, i):
        """ get an attribute of the node by the index of the atom

        Args:
            i (int): index of atom i
        
        Returns:
            np.NDArray: Dict[str, np.NDArray]
        
        """
        return self.attribs.nodes[i]


    def __getitem__(self, key):
        if isinstance(key, str):
            return self.attribs.get_node_attr(key)

    def get_edge_attr(self, i, j):
        """ get an attribute of the edge by the index of two atoms

        Args:
            i (int): index of atom i
            j (int): index of atom j
        
        Returns:
            np.NDArray: Dict[str, np.NDArray]
        
        """
        edgeId = self.topo.edges[i, j]['weight']
        return self.attribs.get_edge_slice(edgeId)

