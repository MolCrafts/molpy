# author: Roy Kid
# contact: lijichen365@126.com
# date: 2022-06-22
# version: 0.0.1

import pytest
from molpy.graph import AttribHolder, Graph, Topo
import numpy as np
import numpy.testing as npt

class TestAttr:

    @pytest.fixture(scope='class', name='attr')
    def test_append_nodes(self):

        A = AttribHolder()
        ids = A.append_nodes(a=np.arange(5), b=np.arange(15).reshape(5, 3))
        npt.assert_equal(ids, np.arange(5))
        ids = A.append_nodes(a=np.arange(3), b=np.arange(9).reshape(3, 3))
        npt.assert_equal(ids, np.arange(3)+5)
        yield A

    def test_append_edges(self, attr):
        pass




        

class TestTopo:

    @pytest.fixture(scope='class', name='linear')
    def test_linear(self):

        G = Topo()


class TestGraph:

    @pytest.fixture(scope='class', name='linear')
    def test_create(self):

        G = Graph()
        G.add_nodes(charge=np.arange(5), position=np.ones((5,3)), type=[1,2,3,4,5])
        G.add_edges([(0, 1), (1, 2), (2, 3), (3, 4)], r0=np.array([1,2,3,4]))
        yield G

    def test_add_fields(self, linear:Graph):

        linear.add_nodes(id=np.arange(5))
        # linear.add_edges([(4, 5), (5, 6), (6, 7)], id=np.arange(4))

    def test_add_nodes_and_edges(self, linear:Graph):

        # linear.add_nodes(charge=np.arange(3)+5, position=np.ones((3,3)), type=[6,7,8])
        linear.add_edges([(4, 5), (5, 6), (6, 7)], r0=np.array([5,6,7]))

    def test_get_sub_graph(self, linear):

        pass

    def test_get_node_attr(self, linear):

        npt.assert_equal(linear['charge'], np.arange(5))
        npt.assert_equal(linear['type'], np.arange(5)+1)

    def test_get_an_edge_attr(self, linear):

        edge = linear.get_edge_attr(0, 1)
        assert edge['r0'] == 1

    def test_get_all_edges_attr(self, linear):

        r0 = linear.get_edges_attr('r0')
        npt.assert_equal(r0, np.arange(7)+1)

    
