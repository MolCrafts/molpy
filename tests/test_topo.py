# # author: Roy Kid
# # contact: lijichen365@126.com
# # date: 2022-07-03
# # version: 0.0.1

# import pytest
# from molpy.topo import Topo

# class TestTopo:

#     def test_set_get(self):

#         topo = Topo()
#         topo.add_bond(1, 2)
#         topo.add_bond(2, 3, index=10)
#         topo.add_angle(1, 2, 3, index=9)
#         topo.add_dihedral(1, 2, 3, 4, index=8)

#         topo.add_bond(1, 3)
#         topo.add_angle(1, 2, 4, index=7)
#         topo.add_angle(1, 3, 4, index=7) 
#         topo.add_dihedral(1, 2, 3, 5, index=7)
#         topo.add_dihedral(1, 2, 4, 5, index=7)
#         topo.add_dihedral(1, 3, 4, 5, index=7)

#         assert topo.get_edge(1, 2) is None
#         assert topo.get_edge(2, 3) == 10
#         assert topo.get_angle(1, 2, 3) == 9
#         assert topo.get_angle(1, 2, 4) == 7
#         assert topo.get_angle(1, 3, 4) == 7
#         assert topo.get_dihedral(1, 2, 3, 4) == 8
#         assert topo.get_dihedral(1, 2, 3, 5) == 7
#         assert topo.get_dihedral(1, 2, 4, 5) == 7
#         assert topo.get_dihedral(1, 3, 4, 5) == 7

#     def test_reason_topo_info(self):    

#         cycle = Topo()
#         cycle.add_bonds([
#             [1, 2],
#             [2, 3],
#             [3, 4],
#             [4, 5],
#             [5, 1]
#         ])
#         assert len(cycle.calc_bonds()) == 5
#         assert len(cycle.calc_angles()) == 5
#         assert len(cycle.calc_dihedrals()) == 5


