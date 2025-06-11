from igraph import Graph
import pytest
from molpy.reacter.lammps import find_path_with_valid_branches


def make_adipic_acid_graph():
    g = Graph()
    g.add_vertices(20)
    g.add_edges([(0, 1), (1, 2), (2, 5), (5, 8), (8, 11)])
    g.add_edges([(1, 3), (1, 4), (2, 6), (2, 7), (5, 9), (5, 10), (8, 12), (8, 13)])
    g.add_edges([(0, 18), (0, 14), (18, 19)])
    g.add_edges([(11, 15), (11, 16), (16, 17)])
    g.vs["element"] = list("CCCHHCHHCHHCHHOOOHOH")
    return g


def make_hexamethylenediamine_graph():

    g = Graph()
    g.add_vertices(24)
    g.add_edges([(0, 2), (2, 5), (5, 8), (8, 11), (11, 14), (14, 17), (17, 1)])
    g.add_edges(
        [
            (0, 3),
            (0, 4),
            (2, 6),
            (2, 7),
            (5, 9),
            (5, 10),
            (8, 12),
            (8, 13),
            (11, 15),
            (11, 16),
            (14, 18),
            (14, 19),
            (17, 20),
            (17, 21),
            (1, 22),
            (1, 23),
        ]
    )
    g.vs["element"] = list("NNCHHCHHCHHCHHCHHCHHHHHH")
    return g

@pytest.mark.parametrize("graph, start, end, expected_atoms", [
    (make_adipic_acid_graph(), 5, 0, [5, 2, 7, 6, 1, 4, 3, 0, 14, 18, 19]),
    (make_hexamethylenediamine_graph(), 5, 0, [5, 2, 7, 6, 0, 3, 4]),
])
def test_collect_atoms_index_between_to_site(graph, start, end, expected_atoms):

    result = find_path_with_valid_branches(graph, start, end)
    assert sorted(result) == sorted(expected_atoms), f"Expected {expected_atoms}, got {result}"
