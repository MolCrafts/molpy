import pytest

from molpy.parser.smiles import (
    AtomIR,
    BigSmilesChainIR,
    BigSmilesIR,
    BondDescriptorIR,
    BondIR,
    RepeatSegmentIR,
    SmilesIR,
    SmilesParser,
    StochasticDistributionIR,
    StochasticObjectIR,
)


def mk_smiles_ir(atom_specs, bond_tuples):
    atoms = [
        AtomIR(symbol=a) if isinstance(a, str) else AtomIR(**a) for a in atom_specs
    ]
    bonds = [BondIR(atoms[i], atoms[j], b) for (i, j, b) in bond_tuples]
    return SmilesIR(atoms=atoms, bonds=bonds)


plain_smiles = [
    ("CCC", mk_smiles_ir(["C", "C", "C"], [(0, 1, "-"), (1, 2, "-")])),
    ("C=C", mk_smiles_ir(["C", "C"], [(0, 1, "=")])),
    ("C#C", mk_smiles_ir(["C", "C"], [(0, 1, "#")])),
    ("CN", mk_smiles_ir(["C", "N"], [(0, 1, "-")])),
]

branch_smiles = [
    ("C(C)C", mk_smiles_ir(["C", "C", "C"], [(0, 1, "-"), (0, 2, "-")])),
    (
        "CC(C)O",
        mk_smiles_ir(["C", "C", "C", "O"], [(0, 1, "-"), (1, 2, "-"), (1, 3, "-")]),
    ),
    (
        "CC(O)(C)C",
        mk_smiles_ir(
            ["C", "C", "O", "C", "C"],
            [(0, 1, "-"), (1, 2, "-"), (1, 3, "-"), (1, 4, "-")],
        ),
    ),
    (
        "C(C(C(C)C)O)N",
        mk_smiles_ir(
            ["C", "C", "C", "C", "C", "O", "N"],
            [
                (0, 1, "-"),
                (1, 2, "-"),
                (2, 3, "-"),
                (2, 4, "-"),
                (1, 5, "-"),
                (0, 6, "-"),
            ],
        ),
    ),
    (
        "CC(=O)O",
        mk_smiles_ir(["C", "C", "O", "O"], [(0, 1, "-"), (1, 2, "="), (1, 3, "-")]),
    ),
    (
        "CC(C(=O)O)N",
        mk_smiles_ir(
            ["C", "C", "C", "O", "O", "N"],
            [(0, 1, "-"), (1, 2, "-"), (2, 3, "="), (2, 4, "-"), (1, 5, "-")],
        ),
    ),
]

ring_smiles = [
    (
        "C1CCCCC1",
        mk_smiles_ir(
            ["C", "C", "C", "C", "C", "C"],
            [
                (0, 1, "-"),
                (1, 2, "-"),
                (2, 3, "-"),
                (3, 4, "-"),
                (4, 5, "-"),
                (5, 0, "-"),
            ],
        ),
    ),
    (
        "C1=CC=CC=C1",
        mk_smiles_ir(
            ["C", "C", "C", "C", "C", "C"],
            [
                (0, 1, "="),
                (1, 2, "-"),
                (2, 3, "="),
                (3, 4, "-"),
                (4, 5, "="),
                (5, 0, "-"),
            ],
        ),
    ),
    (
        "C1CC2CCC1C2",
        mk_smiles_ir(
            ["C", "C", "C", "C", "C", "C", "C"],
            [
                (0, 1, "-"),
                (1, 2, "-"),
                (2, 3, "-"),
                (3, 4, "-"),
                (4, 0, "-"),
                (2, 5, "-"),
                (5, 6, "-"),
                (6, 3, "-"),
            ],
        ),
    ),
    (
        "C1CCC2C(C1)CCC2",
        mk_smiles_ir(
            ["C", "C", "C", "C", "C", "C", "C", "C", "C"],
            [
                (5, 0, "-"),
                (0, 1, "-"),
                (1, 2, "-"),
                (2, 3, "-"),
                (3, 4, "-"),
                (4, 5, "-"),
                (4, 6, "-"),
                (6, 7, "-"),
                (7, 8, "-"),
                (3, 8, "-"),
            ],
        ),
    ),
    (
        "C1CC2CC3CC1CC2C3",
        mk_smiles_ir(
            ["C", "C", "C", "C", "C", "C", "C", "C", "C", "C"],
            [
                (0, 1, "-"),
                (1, 2, "-"),
                (2, 3, "-"),
                (3, 4, "-"),
                (4, 0, "-"),
                (2, 5, "-"),
                (5, 6, "-"),
                (6, 7, "-"),
                (7, 3, "-"),
                (6, 8, "-"),
                (8, 9, "-"),
                (9, 4, "-"),
            ],
        ),
    ),
]

aromatic_smiles = [
    (
        "c1ccccc1",
        mk_smiles_ir(
            ["c", "c", "c", "c", "c", "c"],
            [
                (0, 1, "-"),
                (1, 2, "-"),
                (2, 3, "-"),
                (3, 4, "-"),
                (4, 5, "-"),
                (5, 0, "-"),
            ],
        ),
    ),
    (
        "c1ccncc1",
        mk_smiles_ir(
            ["c", "c", "c", "n", "c", "c"],
            [
                (0, 1, "-"),
                (1, 2, "-"),
                (2, 3, "-"),
                (3, 4, "-"),
                (4, 5, "-"),
                (5, 0, "-"),
            ],
        ),
    ),
    (
        "c1ccccc1O",
        mk_smiles_ir(
            ["c", "c", "c", "c", "c", "c", "O"],
            [
                (0, 1, "-"),
                (1, 2, "-"),
                (2, 3, "-"),
                (3, 4, "-"),
                (4, 5, "-"),
                (5, 0, "-"),
                (5, 6, "-"),
            ],
        ),
    ),
    (
        "c1cc(Cl)ccc1",
        mk_smiles_ir(
            ["c", "c", "c", "Cl", "c", "c", "c"],
            [
                (0, 1, "-"),
                (1, 2, "-"),
                (2, 3, "-"),
                (2, 4, "-"),
                (4, 5, "-"),
                (5, 6, "-"),
                (0, 6, "-"),
            ],
        ),
    ),
    (
        "c1coccc1",
        mk_smiles_ir(
            ["c", "c", "o", "c", "c", "c"],
            [
                (0, 1, "-"),
                (1, 2, "-"),
                (2, 3, "-"),
                (3, 4, "-"),
                (4, 5, "-"),
                (5, 0, "-"),
            ],
        ),
    ),
    (
        "n1ccccc1",
        mk_smiles_ir(
            ["n", "c", "c", "c", "c", "c"],
            [
                (0, 1, "-"),
                (1, 2, "-"),
                (2, 3, "-"),
                (3, 4, "-"),
                (4, 5, "-"),
                (0, 5, "-"),
            ],
        ),
    ),
]

charged_smiles = [
    ("[Na+]", mk_smiles_ir([{"symbol": "Na", "charge": 1}], [])),
    ("[Cl-]", mk_smiles_ir([{"symbol": "Cl", "charge": -1}], [])),
    (
        "[O-]C(=O)C",
        mk_smiles_ir(
            [{"symbol": "O", "charge": -1}, "C", "O", "C"],
            [(0, 1, "-"), (1, 2, "="), (1, 3, "-")],
        ),
    ),
    ("[NH4+]", mk_smiles_ir([{"symbol": "N", "h_count": 4, "charge": 1}], [])),
    (
        "[C-]#[O+]",
        mk_smiles_ir(
            [{"symbol": "C", "charge": -1}, {"symbol": "O", "charge": 1}], [(0, 1, "#")]
        ),
    ),
    ("[Fe+2]", mk_smiles_ir([{"symbol": "Fe", "charge": 2}], [])),
    (
        "[C@@H](O)C(=O)O",
        mk_smiles_ir(
            [{"symbol": "C", "chiral": "@@", "h_count": 1}, "O", "C", "O", "O"],
            [(0, 1, "-"), (0, 2, "-"), (2, 3, "="), (2, 4, "-")],
        ),
    ),
]

isotope_smiles = [
    ("[13CH4]", mk_smiles_ir([{"symbol": "C", "isotope": 13, "h_count": 4}], [])),
    ("[2H]O", mk_smiles_ir([{"symbol": "H", "isotope": 2}, "O"], [(0, 1, "-")])),
    (
        "[18O]=C=O",
        mk_smiles_ir(
            [{"symbol": "O", "isotope": 18}, "C", "O"], [(0, 1, "="), (1, 2, "=")]
        ),
    ),
    ("[15NH3]", mk_smiles_ir([{"symbol": "N", "isotope": 15, "h_count": 3}], [])),
    ("[36Cl-]", mk_smiles_ir([{"symbol": "Cl", "isotope": 36, "charge": -1}], [])),
]

chirality_smiles = [
    (
        "C[C@H](O)C(=O)O",
        mk_smiles_ir(
            ["C", {"symbol": "C", "chiral": "@", "h_count": 1}, "O", "C", "O", "O"],
            [(0, 1, "-"), (1, 2, "-"), (1, 3, "-"), (3, 4, "="), (3, 5, "-")],
        ),
    ),
    (
        "C[C@@H](O)N",
        mk_smiles_ir(
            ["C", {"symbol": "C", "chiral": "@@", "h_count": 1}, "O", "N"],
            [(0, 1, "-"), (1, 2, "-"), (1, 3, "-")],
        ),
    ),
    (
        "N[C@](C)(O)C(=O)O",
        mk_smiles_ir(
            ["N", {"symbol": "C", "chiral": "@"}, "C", "O", "C", "O", "O"],
            [
                (0, 1, "-"),
                (1, 2, "-"),
                (1, 3, "-"),
                (1, 4, "-"),
                (4, 5, "="),
                (4, 6, "-"),
            ],
        ),
    ),
    (
        "F[C@H](Br)Cl",
        mk_smiles_ir(
            ["F", {"symbol": "C", "chiral": "@", "h_count": 1}, "Br", "Cl"],
            [(0, 1, "-"), (1, 2, "-"), (1, 3, "-")],
        ),
    ),
]

dot_smiles = [
    ("C.C", [mk_smiles_ir(["C"], []), mk_smiles_ir(["C"], [])]),
    ("CC.O", [mk_smiles_ir(["C", "C"], [(0, 1, "-")]), mk_smiles_ir(["O"], [])]),
    ("Na.Cl", [mk_smiles_ir(["Na"], []), mk_smiles_ir(["Cl"], [])]),
    (
        "C(=O)O.[Na+]",
        [
            mk_smiles_ir(["C", "O", "O"], [(0, 1, "="), (0, 2, "-")]),
            mk_smiles_ir([{"symbol": "Na", "charge": 1}], []),
        ],
    ),
    (
        "C1CCCCC1.CC",
        [
            mk_smiles_ir(
                ["C", "C", "C", "C", "C", "C"],
                [
                    (0, 1, "-"),
                    (1, 2, "-"),
                    (2, 3, "-"),
                    (3, 4, "-"),
                    (4, 5, "-"),
                    (5, 0, "-"),
                ],
            ),
            mk_smiles_ir(["C", "C"], [(0, 1, "-")]),
        ],
    ),
]

nested_smiles = [
    (
        "CC(C(C(C)O)O)O",
        mk_smiles_ir(
            ["C", "C", "C", "C", "C", "O", "O", "O"],
            [
                (0, 1, "-"),
                (1, 2, "-"),
                (2, 3, "-"),
                (3, 4, "-"),
                (3, 5, "-"),
                (2, 6, "-"),
                (1, 7, "-"),
            ],
        ),
    ),
    (
        "CC(=O)OC1=CC=CC=C1C(=O)O",
        mk_smiles_ir(
            ["C", "C", "O", "O", "C", "C", "C", "C", "C", "C", "C", "O", "O"],
            [
                (0, 1, "-"),
                (1, 2, "="),
                (1, 3, "-"),
                (3, 4, "-"),
                (4, 5, "="),
                (5, 6, "-"),
                (6, 7, "="),
                (7, 8, "-"),
                (8, 9, "="),
                (4, 9, "-"),
                (9, 10, "-"),
                (10, 11, "="),
                (10, 12, "-"),
            ],
        ),
    ),
    (
        "O=C(NCC1=CC=CC=C1)C2=CC=CC=C2",
        mk_smiles_ir(
            [
                "O",
                "C",
                "N",
                "C",
                "C",
                "C",
                "C",
                "C",
                "C",
                "C",
                "C",
                "C",
                "C",
                "C",
                "C",
                "C",
            ],
            [
                (0, 1, "="),
                (1, 2, "-"),
                (2, 3, "-"),
                (3, 4, "-"),
                (4, 5, "="),
                (5, 6, "-"),
                (6, 7, "="),
                (7, 8, "-"),
                (8, 9, "="),
                (4, 9, "-"),
                (1, 10, "-"),
                (10, 11, "="),
                (11, 12, "-"),
                (12, 13, "="),
                (13, 14, "-"),
                (14, 15, "="),
                (10, 15, "-"),
            ],
        ),
    ),
    (
        "CC(=O)Nc1ccc(O)cc1",
        mk_smiles_ir(
            ["C", "C", "O", "N", "c", "c", "c", "c", "O", "c", "c"],
            [
                (0, 1, "-"),
                (1, 2, "="),
                (1, 3, "-"),
                (3, 4, "-"),
                (4, 5, "-"),
                (5, 6, "-"),
                (6, 7, "-"),
                (7, 8, "-"),
                (7, 9, "-"),
                (9, 10, "-"),
                (4, 10, "-"),
            ],
        ),
    ),
    (
        "C1=CC=C2C=CC=CC2=C1",
        mk_smiles_ir(
            ["C", "C", "C", "C", "C", "C", "C", "C", "C", "C"],
            [
                (0, 1, "="),
                (1, 2, "-"),
                (2, 3, "="),
                (3, 4, "-"),
                (4, 5, "="),
                (5, 6, "-"),
                (6, 7, "="),
                (7, 8, "-"),
                (3, 8, "-"),
                (8, 9, "="),
                (0, 9, "-"),
            ],
        ),
    ),
]

func_smiles = [
    ("CCO", mk_smiles_ir(["C", "C", "O"], [(0, 1, "-"), (1, 2, "-")])),
    (
        "CC(=O)O",
        mk_smiles_ir(["C", "C", "O", "O"], [(0, 1, "-"), (1, 2, "="), (1, 3, "-")]),
    ),
    ("CCN", mk_smiles_ir(["C", "C", "N"], [(0, 1, "-"), (1, 2, "-")])),
    ("CCS", mk_smiles_ir(["C", "C", "S"], [(0, 1, "-"), (1, 2, "-")])),
    (
        "CNC=O",
        mk_smiles_ir(["C", "N", "C", "O"], [(0, 1, "-"), (1, 2, "-"), (2, 3, "=")]),
    ),
    ("COC", mk_smiles_ir(["C", "O", "C"], [(0, 1, "-"), (1, 2, "-")])),
    (
        "CCOC(=O)C",
        mk_smiles_ir(
            ["C", "C", "O", "C", "O", "C"],
            [(0, 1, "-"), (1, 2, "-"), (2, 3, "-"), (3, 4, "="), (3, 5, "-")],
        ),
    ),
    (
        "CC(=O)OC",
        mk_smiles_ir(
            ["C", "C", "O", "O", "C"],
            [(0, 1, "-"), (1, 2, "="), (1, 3, "-"), (3, 4, "-")],
        ),
    ),
    (
        "CC(=O)N",
        mk_smiles_ir(["C", "C", "O", "N"], [(0, 1, "-"), (1, 2, "="), (1, 3, "-")]),
    ),
]

complex_smiles = [
    (
        "C1=CC2=C(C=C1)C=CC=C2",
        mk_smiles_ir(
            ["C", "C", "C", "C", "C", "C", "C", "C", "C", "C"],
            [
                (5, 0, "-"),
                (0, 1, "="),
                (1, 2, "-"),
                (2, 3, "="),
                (3, 4, "-"),
                (4, 5, "="),
                (3, 6, "-"),
                (6, 7, "="),
                (7, 8, "-"),
                (8, 9, "="),
                (2, 9, "-"),
            ],
        ),
    ),
    (
        "C1CC=CC=C1",
        mk_smiles_ir(
            ["C", "C", "C", "C", "C", "C"],
            [
                (0, 1, "-"),
                (1, 2, "-"),
                (2, 3, "="),
                (3, 4, "-"),
                (4, 5, "="),
                (0, 5, "-"),
            ],
        ),
    ),
    (
        "C1C=CC=CC1",
        mk_smiles_ir(
            ["C", "C", "C", "C", "C", "C"],
            [
                (0, 1, "-"),
                (1, 2, "="),
                (2, 3, "-"),
                (3, 4, "="),
                (4, 5, "-"),
                (0, 5, "-"),
            ],
        ),
    ),
    (
        "C1CC=CCC1",
        mk_smiles_ir(
            ["C", "C", "C", "C", "C", "C"],
            [
                (0, 1, "-"),
                (1, 2, "-"),
                (2, 3, "="),
                (3, 4, "-"),
                (4, 5, "-"),
                (0, 5, "-"),
            ],
        ),
    ),
    (
        "C1C#CCC1",
        mk_smiles_ir(
            ["C", "C", "C", "C", "C"],
            [(0, 1, "-"), (1, 2, "#"), (2, 3, "-"), (3, 4, "-"), (0, 4, "-")],
        ),
    ),
]

error_smiles = [
    "C1CC",  # Unclosed ring
    "C(=O",  # Unclosed parenthesis
]

parser = SmilesParser()


class TestSmilesParser:
    @pytest.mark.parametrize("smiles, expected", plain_smiles)
    def test_plain_smiles(self, smiles, expected):
        result = parser.parse_smiles(smiles)
        assert result == expected

    @pytest.mark.parametrize("smiles, expected", branch_smiles)
    def test_branch_smiles(self, smiles, expected):
        result = parser.parse_smiles(smiles)
        assert result == expected

    @pytest.mark.parametrize("smiles, expected", ring_smiles)
    def test_ring_smiles(self, smiles, expected):
        result = parser.parse_smiles(smiles)
        assert result == expected

    @pytest.mark.parametrize("smiles, expected", aromatic_smiles)
    def test_aromatic_smiles(self, smiles, expected):
        result = parser.parse_smiles(smiles)
        assert result == expected

    @pytest.mark.parametrize("smiles, expected", charged_smiles)
    def test_charged_smiles(self, smiles, expected):
        result = parser.parse_smiles(smiles)
        assert result == expected

    @pytest.mark.parametrize("smiles, expected", isotope_smiles)
    def test_isotope_smiles(self, smiles, expected):
        result = parser.parse_smiles(smiles)
        assert result == expected

    @pytest.mark.parametrize("smiles, expected", chirality_smiles)
    def test_chirality_smiles(self, smiles, expected):
        result = parser.parse_smiles(smiles)
        assert result == expected

    @pytest.mark.parametrize("smiles, expected", dot_smiles)
    def test_dot_smiles(self, smiles, expected):
        result = parser.parse_dot_smiles(smiles)
        assert result == expected

    @pytest.mark.parametrize("smiles, expected", nested_smiles)
    def test_nested_smiles(self, smiles, expected):
        result = parser.parse_smiles(smiles)
        assert result == expected

    @pytest.mark.parametrize("smiles, expected", func_smiles)
    def test_functional_groups(self, smiles, expected):
        result = parser.parse_smiles(smiles)
        assert result == expected

    @pytest.mark.parametrize("smiles, expected", complex_smiles)
    def test_complex_smiles(self, smiles, expected):
        result = parser.parse_smiles(smiles)
        assert result == expected

    @pytest.mark.parametrize("smiles", error_smiles)
    def test_error_smiles(self, smiles):
        """Test that invalid SMILES raise exceptions"""
        with pytest.raises(Exception):
            parser.parse_smiles(smiles)


# ======================== BigSMILES IR tests ========================


def mk_bigsmiles_ir(start_specs, segments_specs):
    """
    Helper to build expected BigSmilesIR.

    start_specs: (atom_specs, bond_tuples) for start_smiles
    segments_specs: list of segment specs, each segment is:
        {
            "objects": [{"left": desc_spec, "right": desc_spec, "units": [unit_specs],
                         "end_groups": [eg_specs] or None, "distribution": dist_spec or None}],
            "implicit": (atom_specs, bond_tuples) or None
        }

    Returns BigSmilesIR with properly collected atoms and bonds.
    """

    start_smiles = mk_smiles_ir(*start_specs) if start_specs else mk_smiles_ir([], [])

    # Collect all atoms and bonds
    all_atoms = list(start_smiles.atoms)
    all_bonds = list(start_smiles.bonds)

    segments = []
    for seg_spec in segments_specs:
        objs = []
        for obj_spec in seg_spec["objects"]:
            left_desc = (
                BondDescriptorIR(**obj_spec["left"])
                if obj_spec["left"]
                else BondDescriptorIR()
            )
            right_desc = (
                BondDescriptorIR(**obj_spec["right"])
                if obj_spec["right"]
                else BondDescriptorIR()
            )
            units = [mk_smiles_ir(*u) for u in obj_spec["units"]]

            # Collect atoms and bonds from units
            for unit in units:
                all_atoms.extend(unit.atoms)
                all_bonds.extend(unit.bonds)

            end_groups = (
                [mk_smiles_ir(*eg) for eg in obj_spec["end_groups"]]
                if obj_spec.get("end_groups")
                else None
            )

            # Collect atoms and bonds from end groups
            if end_groups:
                for eg in end_groups:
                    all_atoms.extend(eg.atoms)
                    all_bonds.extend(eg.bonds)

            dist = (
                StochasticDistributionIR(**obj_spec["distribution"])
                if obj_spec.get("distribution")
                else None
            )
            objs.append(
                StochasticObjectIR(
                    left_descriptor=left_desc,
                    right_descriptor=right_desc,
                    repeat_units=units,
                    end_groups=end_groups,
                    distribution=dist,
                )
            )
        implicit = (
            mk_smiles_ir(*seg_spec["implicit"]) if seg_spec.get("implicit") else None
        )

        # Collect atoms and bonds from implicit smiles
        if implicit:
            all_atoms.extend(implicit.atoms)
            all_bonds.extend(implicit.bonds)

        segments.append(
            RepeatSegmentIR(stochastic_objects=objs, implicit_smiles=implicit)
        )

    chain = BigSmilesChainIR(start_smiles=start_smiles, repeat_segments=segments)
    return BigSmilesIR(atoms=all_atoms, bonds=all_bonds, chain=chain)


# Bond descriptor test cases (field variants)
bond_descriptor_cases = [
    # (name, kwargs, expected_checks)
    ("empty", {}, {"symbol": None, "index": None, "generation": None}),
    ("symbol_only", {"symbol": "<"}, {"symbol": "<", "index": None}),
    ("symbol_index", {"symbol": ">", "index": 1}, {"symbol": ">", "index": 1}),
    (
        "symbol_index_gen",
        {"symbol": "$", "index": 2, "generation": [0, 1]},
        {"symbol": "$", "index": 2, "generation": [0, 1]},
    ),
    ("generation_only", {"generation": [1, 2, 3]}, {"generation": [1, 2, 3]}),
]

# Stochastic distribution test cases
distribution_cases = [
    (
        "flory_schulz",
        {"name": "flory_schulz", "params": [0.8]},
        {"name": "flory_schulz", "params": [0.8]},
    ),
    (
        "schulz_zimm",
        {"name": "schulz_zimm", "params": [2.0, 100.0]},
        {"name": "schulz_zimm", "params": [2.0, 100.0]},
    ),
    (
        "poisson",
        {"name": "poisson", "params": [50.0]},
        {"name": "poisson", "params": [50.0]},
    ),
]

# Stochastic object test cases (minimal/full field coverage)
stochastic_object_cases = [
    # (name, left_desc, right_desc, units, end_groups, distribution)
    (
        "minimal_one_unit",
        {"symbol": "<", "index": 1},
        {"symbol": ">", "index": 1},
        [(["C", "C"], [(0, 1, "-")])],
        None,
        None,
    ),
    (
        "two_units",
        {"symbol": "<", "index": 1},
        {"symbol": ">", "index": 1},
        [(["C", "C"], [(0, 1, "-")]), (["O"], [])],
        None,
        None,
    ),
    (
        "with_end_groups",
        {"symbol": "<", "index": 2},
        {"symbol": ">", "index": 2},
        [(["C", "C", "O"], [(0, 1, "-"), (1, 2, "-")])],
        [(["H"], []), (["OH"], [])],
        None,
    ),
    (
        "with_distribution",
        {"symbol": "<", "index": 3},
        {"symbol": ">", "index": 3},
        [(["C", "C"], [(0, 1, "-")])],
        None,
        {"name": "flory_schulz", "params": [0.9]},
    ),
    (
        "full_fields",
        {"symbol": "$", "index": 10, "generation": [0]},
        {"symbol": "$", "index": 10, "generation": [0]},
        [(["C", "C"], [(0, 1, "-")]), (["C", "O"], [(0, 1, "-")])],
        [(["Br"], [])],
        {"name": "poisson", "params": [25.0]},
    ),
]

# Chain and segment test cases (compositional)
chain_segment_cases = [
    # (name, start_specs, segments_specs)
    (
        "empty_start_one_segment",
        ([], []),
        [
            {
                "objects": [
                    {
                        "left": {"symbol": "<", "index": 1},
                        "right": {"symbol": ">", "index": 1},
                        "units": [(["C", "C"], [(0, 1, "-")])],
                    }
                ],
                "implicit": None,
            }
        ],
    ),
    (
        "with_start_one_segment",
        (["Br"], []),
        [
            {
                "objects": [
                    {
                        "left": {"symbol": "<", "index": 1},
                        "right": {"symbol": ">", "index": 1},
                        "units": [(["C", "C"], [(0, 1, "-")])],
                    }
                ],
                "implicit": None,
            }
        ],
    ),
    (
        "segment_with_implicit",
        (["C", "C"], [(0, 1, "-")]),
        [
            {
                "objects": [
                    {
                        "left": {"symbol": "<", "index": 1},
                        "right": {"symbol": ">", "index": 1},
                        "units": [(["O"], [])],
                    }
                ],
                "implicit": (["Cl"], []),
            }
        ],
    ),
    (
        "multiple_objects_in_segment",
        ([], []),
        [
            {
                "objects": [
                    {
                        "left": {"symbol": "<", "index": 1},
                        "right": {"symbol": ">", "index": 1},
                        "units": [(["C"], [])],
                    },
                    {
                        "left": {"symbol": "<", "index": 2},
                        "right": {"symbol": ">", "index": 2},
                        "units": [(["O"], [])],
                    },
                ],
                "implicit": None,
            }
        ],
    ),
    (
        "two_segments",
        (["Br"], []),
        [
            {
                "objects": [
                    {
                        "left": {"symbol": "<", "index": 1},
                        "right": {"symbol": ">", "index": 1},
                        "units": [(["C", "C"], [(0, 1, "-")])],
                    }
                ],
                "implicit": None,
            },
            {
                "objects": [
                    {
                        "left": {"symbol": "<", "index": 2},
                        "right": {"symbol": ">", "index": 2},
                        "units": [(["O"], [])],
                    }
                ],
                "implicit": (["Cl"], []),
            },
        ],
    ),
]


class TestBigSmilesIR:
    @pytest.mark.parametrize("name,kwargs,expected", bond_descriptor_cases)
    def test_bond_descriptor_fields(self, name, kwargs, expected):
        desc = BondDescriptorIR(**kwargs)
        for key, val in expected.items():
            assert getattr(desc, key) == val

    @pytest.mark.parametrize("name,kwargs,expected", distribution_cases)
    def test_stochastic_distribution_fields(self, name, kwargs, expected):
        dist = StochasticDistributionIR(**kwargs)
        for key, val in expected.items():
            assert getattr(dist, key) == val

    @pytest.mark.parametrize(
        "name,left_desc,right_desc,units,end_groups,distribution",
        stochastic_object_cases,
    )
    def test_stochastic_object_fields(
        self, name, left_desc, right_desc, units, end_groups, distribution
    ):
        left = BondDescriptorIR(**left_desc)
        right = BondDescriptorIR(**right_desc)
        unit_irs = [mk_smiles_ir(*u) for u in units]
        eg_irs = [mk_smiles_ir(*eg) for eg in end_groups] if end_groups else None
        dist_ir = StochasticDistributionIR(**distribution) if distribution else None

        sobj = StochasticObjectIR(
            left_descriptor=left,
            right_descriptor=right,
            repeat_units=unit_irs,
            end_groups=eg_irs,
            distribution=dist_ir,
        )

        assert sobj.left_descriptor.symbol == left_desc["symbol"]
        assert sobj.right_descriptor.symbol == right_desc["symbol"]
        assert len(sobj.repeat_units) == len(units)
        if end_groups:
            assert sobj.end_groups is not None and len(sobj.end_groups) == len(
                end_groups
            )
        else:
            assert sobj.end_groups is None
        if distribution:
            assert (
                sobj.distribution is not None
                and sobj.distribution.name == distribution["name"]
            )
        else:
            assert sobj.distribution is None

    @pytest.mark.parametrize("name,start_specs,segments_specs", chain_segment_cases)
    def test_chain_and_segments(self, name, start_specs, segments_specs):
        expected_ir = mk_bigsmiles_ir(start_specs, segments_specs)

        # Verify chain structure
        chain = expected_ir.chain
        assert chain.start_smiles is not None
        assert len(chain.repeat_segments) == len(segments_specs)

        # Verify first segment has expected object count
        if segments_specs:
            assert len(chain.repeat_segments[0].stochastic_objects) == len(
                segments_specs[0]["objects"]
            )

    def test_bigsmiles_ir_inherits_smilesir(self):
        assert issubclass(BigSmilesIR, SmilesIR)

    def test_bigsmiles_ir_has_chain_field(self):
        start = mk_smiles_ir([], [])
        chain = BigSmilesChainIR(start_smiles=start, repeat_segments=[])
        big = BigSmilesIR(atoms=[], bonds=[], chain=chain)

        assert hasattr(big, "chain")
        assert isinstance(big.chain, BigSmilesChainIR)
        assert isinstance(big, SmilesIR)

    def test_parse_bigsmiles_method_exists(self):
        # Verify SmilesParser has parse_bigsmiles entrypoint
        parse_fn = getattr(parser, "parse_bigsmiles", None)
        if parse_fn is None:
            pytest.xfail("SmilesParser.parse_bigsmiles not implemented yet")


# ======================== BigSMILES Parser tests ========================


# 1️⃣ Basic SMILES compatibility (baseline)
basic_bigsmiles_compat = [
    ("CCO", mk_bigsmiles_ir((["C", "C", "O"], [(0, 1, "-"), (1, 2, "-")]), [])),
    (
        "C1CCCCC1",
        mk_bigsmiles_ir(
            (
                ["C", "C", "C", "C", "C", "C"],
                [
                    (0, 1, "-"),
                    (1, 2, "-"),
                    (2, 3, "-"),
                    (3, 4, "-"),
                    (4, 5, "-"),
                    (5, 0, "-"),
                ],
            ),
            [],
        ),
    ),
    (
        "CC(=O)O",
        mk_bigsmiles_ir(
            (["C", "C", "O", "O"], [(0, 1, "-"), (1, 2, "="), (1, 3, "-")]), []
        ),
    ),
]


# 2️⃣ Simple repeat units {} (basic stochastic polymerization)
simple_repeat_bigsmiles = [
    (
        "{[<]CC[>]}",
        mk_bigsmiles_ir(
            ([], []),  # empty start_smiles
            [
                {
                    "objects": [
                        {
                            "left": {"symbol": "<"},
                            "right": {"symbol": ">"},
                            "units": [(["C", "C"], [(0, 1, "-")])],
                        }
                    ],
                    "implicit": None,
                }
            ],
        ),
    ),
    (
        "{[<]CCO[>]}",
        mk_bigsmiles_ir(
            ([], []),
            [
                {
                    "objects": [
                        {
                            "left": {"symbol": "<"},
                            "right": {"symbol": ">"},
                            "units": [(["C", "C", "O"], [(0, 1, "-"), (1, 2, "-")])],
                        }
                    ],
                    "implicit": None,
                }
            ],
        ),
    ),
    (
        "{[<]C(=O)O[>]}",
        mk_bigsmiles_ir(
            ([], []),
            [
                {
                    "objects": [
                        {
                            "left": {"symbol": "<"},
                            "right": {"symbol": ">"},
                            "units": [(["C", "O", "O"], [(0, 1, "="), (0, 2, "-")])],
                        }
                    ],
                    "implicit": None,
                }
            ],
        ),
    ),
]


# 3️⃣ Repeat units with comma-separated monomers (random copolymer)
stochastic_copolymer_bigsmiles = [
    (
        "{[<]CC[>],[<]OCC[>]}",
        mk_bigsmiles_ir(
            ([], []),
            [
                {
                    "objects": [
                        {
                            "left": {"symbol": "<"},
                            "right": {"symbol": ">"},
                            "units": [
                                (["C", "C"], [(0, 1, "-")]),
                                (["O", "C", "C"], [(0, 1, "-"), (1, 2, "-")]),
                            ],
                        }
                    ],
                    "implicit": None,
                }
            ],
        ),
    ),
]


# 4️⃣ Block copolymers (multiple {} segments)
block_copolymer_bigsmiles = [
    (
        "{[<]CC[>]}{[<]O[>]}",
        mk_bigsmiles_ir(
            ([], []),
            [
                {
                    "objects": [
                        {
                            "left": {"symbol": "<"},
                            "right": {"symbol": ">"},
                            "units": [(["C", "C"], [(0, 1, "-")])],
                        }
                    ],
                    "implicit": None,
                },
                {
                    "objects": [
                        {
                            "left": {"symbol": "<"},
                            "right": {"symbol": ">"},
                            "units": [(["O"], [])],
                        }
                    ],
                    "implicit": None,
                },
            ],
        ),
    ),
    (
        "{[<]CC[>]}{[<]OCCO[>]}",
        mk_bigsmiles_ir(
            ([], []),
            [
                {
                    "objects": [
                        {
                            "left": {"symbol": "<"},
                            "right": {"symbol": ">"},
                            "units": [(["C", "C"], [(0, 1, "-")])],
                        }
                    ],
                    "implicit": None,
                },
                {
                    "objects": [
                        {
                            "left": {"symbol": "<"},
                            "right": {"symbol": ">"},
                            "units": [
                                (
                                    ["O", "C", "C", "O"],
                                    [(0, 1, "-"), (1, 2, "-"), (2, 3, "-")],
                                )
                            ],
                        }
                    ],
                    "implicit": None,
                },
            ],
        ),
    ),
]


# 5️⃣ Mixed with plain SMILES
mixed_bigsmiles = [
    (
        "CC{[<]O[>]}",
        mk_bigsmiles_ir(
            (["C", "C"], [(0, 1, "-")]),  # start_smiles
            [
                {
                    "objects": [
                        {
                            "left": {"symbol": "<"},
                            "right": {"symbol": ">"},
                            "units": [(["O"], [])],
                        }
                    ],
                    "implicit": None,
                }
            ],
        ),
    ),
    (
        "{[<]CC[>]}O",
        mk_bigsmiles_ir(
            ([], []),
            [
                {
                    "objects": [
                        {
                            "left": {"symbol": "<"},
                            "right": {"symbol": ">"},
                            "units": [(["C", "C"], [(0, 1, "-")])],
                        }
                    ],
                    "implicit": (["O"], []),  # implicit smiles after stochastic object
                }
            ],
        ),
    ),
]


# 6️⃣ Bond descriptors with indices
indexed_descriptor_bigsmiles = [
    (
        "{[<1]CC[>1]}",
        mk_bigsmiles_ir(
            ([], []),
            [
                {
                    "objects": [
                        {
                            "left": {"symbol": "<", "index": 1},
                            "right": {"symbol": ">", "index": 1},
                            "units": [(["C", "C"], [(0, 1, "-")])],
                        }
                    ],
                    "implicit": None,
                }
            ],
        ),
    ),
]


class TestBigSmilesParser:
    """Test BigSMILES string parsing into IR structures."""

    @pytest.mark.parametrize("smiles,expected", basic_bigsmiles_compat)
    def test_basic_smiles_compatibility(self, smiles, expected):
        """BigSMILES is SMILES superset - all plain SMILES must parse."""
        ir = parser.parse_bigsmiles(smiles)
        assert ir == expected

    @pytest.mark.parametrize("bigsmiles,expected", simple_repeat_bigsmiles)
    def test_simple_repeat_units(self, bigsmiles, expected):
        """Test basic {} repeat unit parsing."""
        ir = parser.parse_bigsmiles(bigsmiles)
        assert ir == expected

    @pytest.mark.parametrize("bigsmiles,expected", stochastic_copolymer_bigsmiles)
    def test_stochastic_copolymer(self, bigsmiles, expected):
        """Test random copolymer with comma-separated monomers."""
        ir = parser.parse_bigsmiles(bigsmiles)
        assert ir == expected

    @pytest.mark.parametrize("bigsmiles,expected", block_copolymer_bigsmiles)
    def test_block_copolymers(self, bigsmiles, expected):
        """Test block copolymer with multiple {} segments."""
        ir = parser.parse_bigsmiles(bigsmiles)
        assert ir == expected

    @pytest.mark.parametrize("bigsmiles,expected", mixed_bigsmiles)
    def test_mixed_with_smiles(self, bigsmiles, expected):
        """Test BigSMILES mixed with plain SMILES fragments."""
        ir = parser.parse_bigsmiles(bigsmiles)
        assert ir == expected

    @pytest.mark.parametrize("bigsmiles,expected", indexed_descriptor_bigsmiles)
    def test_indexed_descriptors(self, bigsmiles, expected):
        """Test bond descriptors with indices."""
        ir = parser.parse_bigsmiles(bigsmiles)
        assert ir == expected

    def test_empty_bigsmiles(self):
        """Test empty string returns valid IR."""
        ir = parser.parse_bigsmiles("")
        expected = mk_bigsmiles_ir(([], []), [])
        assert ir == expected


RDKIT_FIND = True
try:
    pass
except:
    RDKIT_FIND = False


@pytest.mark.skipif(not RDKIT_FIND, reason="rdkit not find")
class TestRDKitConverter:
    """Test SmilesIR → RDKit Mol conversion."""

    # Test cases: (smiles_string, expected_num_atoms, expected_num_bonds)
    converter_test_cases = [
        ("C", 1, 0),  # Methane
        ("CC", 2, 1),  # Ethane
        ("C=C", 2, 1),  # Ethene
        ("C#C", 2, 1),  # Ethyne
        ("c1ccccc1", 6, 6),  # Benzene
        ("CCO", 3, 2),  # Ethanol
        ("CC(C)C", 4, 3),  # Isobutane
    ]

    @pytest.mark.parametrize("smiles,n_atoms,n_bonds", converter_test_cases)
    def test_smilesir_to_mol_basic(self, smiles, n_atoms, n_bonds):
        """Test basic IR → Mol conversion."""

        from molpy.parser.smiles import smilesir_to_mol

        ir = parser.parse_smiles(smiles)
        mol = smilesir_to_mol(ir)

        assert mol is not None
        assert mol.GetNumAtoms() == n_atoms
        assert mol.GetNumBonds() == n_bonds

    def test_smilesir_to_mol_charged_atoms(self):
        """Test conversion with charged atoms."""

        from molpy.parser.smiles import smilesir_to_mol

        # [NH4+]
        ir = parser.parse_smiles("[NH4+]")
        mol = smilesir_to_mol(ir)

        assert mol.GetNumAtoms() == 1
        n_atom = mol.GetAtomWithIdx(0)
        assert n_atom.GetSymbol() == "N"
        assert n_atom.GetFormalCharge() == 1
        assert n_atom.GetTotalNumHs() == 4

    def test_smilesir_to_mol_isotopes(self):
        """Test conversion with isotopes."""

        from molpy.parser.smiles import smilesir_to_mol

        # [13C]
        ir = parser.parse_smiles("[13C]")
        mol = smilesir_to_mol(ir)

        assert mol.GetNumAtoms() == 1
        c_atom = mol.GetAtomWithIdx(0)
        assert c_atom.GetSymbol() == "C"
        assert c_atom.GetIsotope() == 13

    def test_smilesir_to_mol_aromatic(self):
        """Test conversion with aromatic atoms."""

        from molpy.parser.smiles import smilesir_to_mol

        # Benzene
        ir = parser.parse_smiles("c1ccccc1")
        mol = smilesir_to_mol(ir)

        assert mol.GetNumAtoms() == 6
        # All atoms should be aromatic after sanitization
        for atom in mol.GetAtoms():
            assert atom.GetIsAromatic()

    def test_smilesir_to_mol_chirality(self):
        """Test conversion with chiral centers."""

        from rdkit import Chem

        from molpy.parser.smiles import smilesir_to_mol

        # L-alanine: N[C@@H](C)C(=O)O
        ir = parser.parse_smiles("N[C@@H](C)C(=O)O")
        mol = smilesir_to_mol(ir)

        # Check chiral center exists
        chiral_atoms = [
            a
            for a in mol.GetAtoms()
            if a.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED
        ]
        assert len(chiral_atoms) > 0
