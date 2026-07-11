"""Valence completion (``complete_valence``): counts, order, immutability, and
the sp3 cap geometry that lets a sliced region parameterise in one AM1-BCC pass.
"""

import numpy as np
import pytest

from molpy.core.atomistic import Atomistic

TET = np.deg2rad(109.47)  # ideal tetrahedral angle
#: standard C–H cap length (Å) used by molrs.add_hydrogens placement.
_CAP_LEN_C = 1.09
#: the four tetrahedral vertex directions (unit, mutually 109.47° apart).
VERTS = [
    np.array(v, float) / np.sqrt(3.0)
    for v in ([1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1])
]


def _carbon_with(existing_dirs: list[np.ndarray], bond: float = 1.1) -> Atomistic:
    """A carbon at the origin with one *valence-satisfied* H per direction in
    ``dirs`` — so completion only caps the carbon, not the placeholders."""
    mol = Atomistic()
    c = mol.def_atom({"element": "C", "x": 0.0, "y": 0.0, "z": 0.0})
    for d in existing_dirs:
        p = bond * d
        nb = mol.def_atom(
            {"element": "H", "x": float(p[0]), "y": float(p[1]), "z": float(p[2])}
        )
        mol.def_bond(c, nb)
    return mol


def _bond_dirs(mol: Atomistic, atom) -> list[np.ndarray]:
    """Unit vectors from ``atom`` to each of its bond neighbours."""
    p = np.array([atom["x"], atom["y"], atom["z"]], float)
    out = []
    for nb in mol.get_neighbors(atom):
        v = np.array([nb["x"], nb["y"], nb["z"]], float) - p
        out.append(v / np.linalg.norm(v))
    return out


def _pairwise_angles(dirs: list[np.ndarray]) -> list[float]:
    return [
        float(np.arccos(np.clip(np.dot(dirs[i], dirs[j]), -1.0, 1.0)))
        for i in range(len(dirs))
        for j in range(i + 1, len(dirs))
    ]


@pytest.mark.parametrize("degree", [0, 1, 2, 3])
def test_completes_carbon_to_four_bonds(degree):
    """A carbon of any degree ends up 4-coordinate; caps are appended H."""
    mol = _carbon_with(VERTS[:degree])
    capped = mol.complete_valence()

    carbon = list(capped.atoms)[0]
    assert len(capped.get_neighbors(carbon)) == 4
    # caps are appended after the (1 + degree) originals; each is a new H.
    caps = list(capped.atoms)[mol.n_atoms :]
    assert len(caps) == 4 - degree
    assert all(a["element"] == "H" for a in caps)


@pytest.mark.parametrize("degree", [0, 1, 2, 3])
def test_caps_sit_at_tetrahedral_angles(degree):
    """Every C–X pair around the completed carbon is ~109.47° — the sp3 geometry
    a charge (sqm) calculation expects, so no pre-minimisation is needed."""
    mol = _carbon_with(VERTS[:degree])
    capped = mol.complete_valence()
    carbon = list(capped.atoms)[0]

    angles = _pairwise_angles(_bond_dirs(capped, carbon))
    assert angles, "carbon should have >=2 bonds after completion"
    assert np.allclose(angles, TET, atol=np.deg2rad(1.0))


def test_cap_bond_length_matches_element():
    """A cap H sits one standard C–H length from its carbon."""
    mol = _carbon_with(VERTS[:3])
    capped = mol.complete_valence()
    carbon = list(capped.atoms)[0]
    h = list(capped.atoms)[-1]  # the single appended cap
    d = np.linalg.norm(
        np.array([h["x"], h["y"], h["z"]])
        - np.array([carbon["x"], carbon["y"], carbon["z"]])
    )
    assert d == pytest.approx(_CAP_LEN_C, abs=1e-2)


def test_oxygen_capped_to_valence_two():
    """An ether-like O (one bond) gets exactly one cap (valence 2), not four."""
    mol = Atomistic()
    o = mol.def_atom({"element": "O", "x": 0.0, "y": 0.0, "z": 0.0})
    h = mol.def_atom({"element": "H", "x": 0.96, "y": 0.0, "z": 0.0})
    mol.def_bond(o, h)
    capped = mol.complete_valence()
    oxygen = list(capped.atoms)[0]
    assert len(capped.get_neighbors(oxygen)) == 2
    assert capped.n_atoms == mol.n_atoms + 1  # exactly one cap added


def test_double_bonded_oxygen_is_not_overcapped():
    """A C=O oxygen is valence-satisfied by bond order 2 and gets no H cap."""
    mol = Atomistic()
    c = mol.def_atom({"element": "C", "x": 0.0, "y": 0.0, "z": 0.0})
    o = mol.def_atom({"element": "O", "x": 1.2, "y": 0.0, "z": 0.0})
    mol.def_bond(c, o, order=2.0)

    capped = mol.complete_valence()
    oxygen = list(capped.atoms)[1]
    assert len(capped.get_neighbors(oxygen)) == 1
    assert capped.n_atoms == 4  # formaldehyde-like CH2=O, not C(OH)H3


def test_input_is_untouched():
    """``complete_valence`` returns a new graph; the caller's struct is unchanged."""
    mol = _carbon_with(VERTS[:2])
    before = mol.n_atoms
    capped = mol.complete_valence()
    assert capped is not mol
    assert mol.n_atoms == before  # no caps leaked back onto the input
    assert capped.n_atoms == before + 2


# ---------------------------------------------------------------------------
# Completion must not destroy the graph it completes.
#
# ``complete_valence`` used to copy only atoms and *bonds*, so a fragment that
# carried angles and dihedrals came back with none. That silently deleted the
# very terms a force-field typifier was about to label — a "return the completed
# molecule" function throwing topology away.
# ---------------------------------------------------------------------------


def _chain(n: int) -> Atomistic:
    """A straight carbon chain with its angles and dihedrals perceived."""
    mol = Atomistic()
    carbons = [
        mol.def_atom({"element": "C", "x": 1.5 * i, "y": 0.0, "z": 0.0})
        for i in range(n)
    ]
    for a, b in zip(carbons, carbons[1:], strict=False):
        mol.def_bond(a, b)
    return mol.get_topo(gen_angle=True, gen_dihe=True)


class TestCompletionPreservesTopology:
    def test_angles_and_dihedrals_survive(self):
        chain = _chain(4)
        assert len(list(chain.angles)) == 2
        assert len(list(chain.dihedrals)) == 1

        capped = chain.complete_valence()

        assert len(list(capped.angles)) == 2
        assert len(list(capped.dihedrals)) == 1

    def test_caps_add_bonds_only(self):
        """A cap is context, not a term: it never invents an angle or dihedral."""
        chain = _chain(4)
        n_bonds, n_angles = len(list(chain.bonds)), len(list(chain.angles))

        capped = chain.complete_valence()
        n_caps = len(list(capped.atoms)) - len(list(chain.atoms))

        assert n_caps == 10  # 3 + 2 + 2 + 3 hydrogens
        assert len(list(capped.bonds)) == n_bonds + n_caps
        assert len(list(capped.angles)) == n_angles

    def test_link_data_is_carried_over(self):
        chain = _chain(4)
        next(iter(chain.angles))["type"] = "c3-c3-c3"

        capped = chain.complete_valence()

        assert next(iter(capped.angles))["type"] == "c3-c3-c3"


class TestCompletionIsIdentityOnCompleteMolecules:
    """The pipeline caps unconditionally, so capping a whole molecule must be a
    no-op — including aromatics, hypervalent sulfur, and bare ions."""

    @pytest.mark.parametrize(
        "smiles",
        ["O", "CCO", "c1ccccc1", "OCCOCCO", "CC(=O)OC", "c1ccncc1", "CS(=O)(=O)C"],
    )
    def test_no_atom_is_added(self, smiles):
        from molpy.conformer import Conformer
        from molpy.parser import parse_molecule

        mol, _ = Conformer(add_hydrogens=True, seed=1).generate(parse_molecule(smiles))
        before = len(list(mol.atoms))

        assert len(list(mol.complete_valence().atoms)) == before
