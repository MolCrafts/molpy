import molpy as mp
import pytest
from molpy.builder.reacter_lammps import get_main_chain_and_branches


test_cases = [
    (
        [(0, 1), (1, 2), (2, 3), (3, 4), (2, 5), (5, 6), (5, 7)],
        0, 4,
        [0, 1, 2, 3, 4],
        [5, 6, 7],
        "Main chain with branches"
    ),
    (
        [(0, 1), (1, 2), (2, 3)],
        0, 3,
        [0, 1, 2, 3],
        [],
        "Linear chain with no branches"
    ),
    (
        [(0, 1), (1, 2), (2, 3), (2, 4)],
        0, 3,
        [0, 1, 2, 3],
        [4],
        "Single branch off main chain"
    ),
    (
        [(0, 1), (1, 2), (2, 3), (0, 4)],
        0, 3,
        [0, 1, 2, 3],
        [],
        "Branch on start atom (ignored)"
    ),
    (
        [(0, 1), (1, 2), (2, 3), (2, 4), (3, 5), (5, 6)],
        0, 3,
        [0, 1, 2, 3],
        [4, 5, 6],
        "Multiple recursive branches"
    ),
    (
        [(0, 1), (1, 2), (2, 3), (3, 0), (2, 4)],
        0, 2,
        [0, 1, 2],
        [3, 4],
        "Ring with branch"
    ),
    (
        [(0, 1), (1, 2), (2, 3), (5, 6)],
        0, 3,
        [0, 1, 2, 3],
        [],
        "Disconnected component should be ignored"
    ),
    (
        [(0, 1), (1, 2), (2, 3), (3, 4), (2, 5), (3, 5)],
        0, 4,
        [0, 1, 2, 3, 4],
        [5],
        "Shared branch between two main chain atoms"
    )
]


@pytest.mark.parametrize("edges,start,end,expected_path,expected_branches,desc", test_cases)
def test_main_chain_and_branches(edges, start, end, expected_path, expected_branches, desc):
    g = mp.Topology(edges=edges)
    path, branches = get_main_chain_and_branches(g, start, end)

    assert path == expected_path, f"[{desc}] Path mismatch"
    assert set(branches) == set(expected_branches), f"[{desc}] Branch mismatch"