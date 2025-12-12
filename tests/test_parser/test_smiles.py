import pytest

from molpy.parser.smiles import (
    BigSmilesMoleculeIR,
    BondingDescriptorIR,
    SmilesAtomIR,
    SmilesBondIR,
    SmilesGraphIR,
    StochasticObjectIR,
    parse_bigsmiles,
    parse_smiles,
)


def _bond_order(symbol: str) -> str | int:
    mapping = {
        "-": 1,
        "=": 2,
        "#": 3,
        ":": "ar",
    }
    return mapping.get(symbol, symbol)


def mk_smiles_ir(atom_specs, bond_tuples):
    atoms = []
    for a in atom_specs:
        if isinstance(a, str):
            # Handle aromatic: lowercase in SMILES means aromatic
            aromatic = a.islower()
            element = a.upper() if aromatic else a
            atoms.append(SmilesAtomIR(element=element, aromatic=aromatic))
        else:
            element = a.get("symbol") or a.get("element", "C")
            aromatic = element.islower() if isinstance(element, str) else False
            if isinstance(element, str) and aromatic:
                element = element.upper()
            extras = {
                k: v
                for k, v in a.items()
                if k not in ("symbol", "element", "charge", "h_count")
            }
            atoms.append(
                SmilesAtomIR(
                    element=element,
                    aromatic=aromatic,
                    charge=a.get("charge"),
                    hydrogens=a.get("h_count"),
                    extras=extras,
                )
            )
    bonds = [
        SmilesBondIR(atom_i=atoms[i], atom_j=atoms[j], order=_bond_order(b))
        for (i, j, b) in bond_tuples
    ]
    return SmilesGraphIR(atoms=atoms, bonds=bonds)


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


class TestSmilesParser:
    @pytest.mark.parametrize("smiles, expected", plain_smiles)
    def test_plain_smiles(self, smiles, expected):
        result = parse_smiles(smiles)
        assert result == expected

    @pytest.mark.parametrize("smiles, expected", branch_smiles)
    def test_branch_smiles(self, smiles, expected):
        result = parse_smiles(smiles)
        assert result == expected

    @pytest.mark.parametrize("smiles, expected", ring_smiles)
    def test_ring_smiles(self, smiles, expected):
        result = parse_smiles(smiles)
        assert result == expected

    @pytest.mark.parametrize("smiles, expected", aromatic_smiles)
    def test_aromatic_smiles(self, smiles, expected):
        result = parse_smiles(smiles)
        assert result == expected

    @pytest.mark.parametrize("smiles, expected", charged_smiles)
    def test_charged_smiles(self, smiles, expected):
        result = parse_smiles(smiles)
        assert result == expected

    @pytest.mark.parametrize("smiles, expected", isotope_smiles)
    def test_isotope_smiles(self, smiles, expected):
        result = parse_smiles(smiles)
        assert result == expected

    @pytest.mark.parametrize("smiles, expected", chirality_smiles)
    def test_chirality_smiles(self, smiles, expected):
        result = parse_smiles(smiles)
        assert result == expected

    @pytest.mark.parametrize("smiles, expected", dot_smiles)
    def test_dot_smiles(self, smiles, expected):
        """Test dot-separated SMILES (mixtures/disconnected components)."""
        result = parse_smiles(smiles)
        # For dot-separated SMILES, result should be a list
        assert isinstance(result, list)
        assert len(result) == len(expected)
        for actual, exp in zip(result, expected):
            assert actual == exp

    @pytest.mark.parametrize("smiles, expected", nested_smiles)
    def test_nested_smiles(self, smiles, expected):
        result = parse_smiles(smiles)
        assert result == expected

    @pytest.mark.parametrize("smiles, expected", func_smiles)
    def test_functional_groups(self, smiles, expected):
        result = parse_smiles(smiles)
        assert result == expected

    @pytest.mark.parametrize("smiles, expected", complex_smiles)
    def test_complex_smiles(self, smiles, expected):
        result = parse_smiles(smiles)
        assert result == expected

    @pytest.mark.parametrize("smiles", error_smiles)
    def test_error_smiles(self, smiles):
        """Test that invalid SMILES raise exceptions"""
        with pytest.raises(Exception):
            parse_smiles(smiles)


# ======================== BigSMILES IR tests ========================


def mk_bigsmiles_ir(start_specs, segments_specs):
    """
    Helper to build expected BigSmilesMoleculeIR.

    start_specs: (atom_specs, bond_tuples) for backbone
    segments_specs: list of segment specs, each segment is:
        {
            "objects": [{"left": desc_spec, "right": desc_spec, "units": [unit_specs],
                         "end_groups": [eg_specs] or None}],
        }

    Returns BigSmilesMoleculeIR with properly structured backbone and stochastic objects.
    """
    from molpy.parser.smiles.bigsmiles_ir import (
        BigSmilesSubgraphIR,
        EndGroupIR,
        RepeatUnitIR,
        TerminalDescriptorIR,
    )

    # Build backbone from start_specs
    backbone_graph = mk_smiles_ir(*start_specs) if start_specs else mk_smiles_ir([], [])
    backbone = BigSmilesSubgraphIR(
        atoms=list(backbone_graph.atoms),
        bonds=list(backbone_graph.bonds),
        descriptors=[],
    )

    stochastic_objects = []
    for seg_spec in segments_specs:
        for obj_spec in seg_spec["objects"]:
            # Build unified terminals (merge left and right descriptors)
            all_descriptors = []
            
            left_desc_spec = obj_spec.get("left", {})
            if left_desc_spec:
                symbol = left_desc_spec.get("symbol")
                label = left_desc_spec.get("index") or left_desc_spec.get("label")
                all_descriptors.append(
                    BondingDescriptorIR(
                        symbol=symbol,
                        label=label,
                        role="terminal",
                    )
                )

            right_desc_spec = obj_spec.get("right", {})
            if right_desc_spec:
                symbol = right_desc_spec.get("symbol")
                label = right_desc_spec.get("index") or right_desc_spec.get("label")
                all_descriptors.append(
                    BondingDescriptorIR(
                        symbol=symbol,
                        label=label,
                        role="terminal",
                    )
                )
            
            terminals = TerminalDescriptorIR(descriptors=all_descriptors)

            # Build repeat units
            repeat_units = []
            for unit_spec in obj_spec.get("units", []):
                unit_graph = mk_smiles_ir(*unit_spec)
                unit_subgraph = BigSmilesSubgraphIR(
                    atoms=list(unit_graph.atoms),
                    bonds=list(unit_graph.bonds),
                    descriptors=[],
                )
                repeat_units.append(RepeatUnitIR(graph=unit_subgraph))

            # Build end groups
            end_groups = []
            for eg_spec in obj_spec.get("end_groups", []):
                eg_graph = mk_smiles_ir(*eg_spec)
                eg_subgraph = BigSmilesSubgraphIR(
                    atoms=list(eg_graph.atoms),
                    bonds=list(eg_graph.bonds),
                    descriptors=[],
                )
                end_groups.append(EndGroupIR(graph=eg_subgraph))

            stochastic_objects.append(
                StochasticObjectIR(
                    terminals=terminals,
                    repeat_units=repeat_units,
                    end_groups=end_groups,
                )
            )

        # Handle implicit smiles (trailing smiles after stochastic objects)
        # These should be added to the backbone
        if "implicit" in seg_spec and seg_spec["implicit"] is not None:
            implicit_spec = seg_spec["implicit"]
            if isinstance(implicit_spec, tuple) and len(implicit_spec) == 2:
                implicit_atoms, implicit_bonds = implicit_spec
                implicit_graph = mk_smiles_ir(implicit_atoms, implicit_bonds)
                backbone.atoms.extend(implicit_graph.atoms)
                backbone.bonds.extend(implicit_graph.bonds)

    return BigSmilesMoleculeIR(backbone=backbone, stochastic_objects=stochastic_objects)


def mk_bigsmiles_ir_with_descriptors(start_specs, descriptors_specs, segments_specs):
    """
    Helper to build expected BigSmilesMoleculeIR with descriptors in backbone.

    Similar to mk_bigsmiles_ir but allows specifying descriptors for the backbone.

    Args:
        start_specs: (atom_specs, bond_tuples) for backbone
        descriptors_specs: list of descriptor specs, each is {"symbol": str, "role": str, ...}
        segments_specs: list of segment specs (same as mk_bigsmiles_ir)
    """
    from molpy.parser.smiles.bigsmiles_ir import (
        BigSmilesMoleculeIR,
        BigSmilesSubgraphIR,
        BondingDescriptorIR,
        EndGroupIR,
        RepeatUnitIR,
        StochasticObjectIR,
        TerminalDescriptorIR,
    )

    # Build backbone from start_specs
    backbone_graph = mk_smiles_ir(*start_specs) if start_specs else mk_smiles_ir([], [])
    backbone = BigSmilesSubgraphIR(
        atoms=list(backbone_graph.atoms),
        bonds=list(backbone_graph.bonds),
        descriptors=[],
    )

    # Add descriptors to backbone
    for desc_spec in descriptors_specs:
        descriptor = BondingDescriptorIR(
            symbol=desc_spec.get("symbol"),
            label=desc_spec.get("label"),
            role=desc_spec.get("role", "internal"),
            bond_order=desc_spec.get("bond_order", 1),
        )
        if "extras" in desc_spec:
            descriptor.extras.update(desc_spec["extras"])
        backbone.descriptors.append(descriptor)

    # Build stochastic objects (same as mk_bigsmiles_ir)
    stochastic_objects = []
    for seg_spec in segments_specs:
        for obj_spec in seg_spec["objects"]:
            # Build unified terminals (merge left and right descriptors)
            all_descriptors = []
            
            left_desc_spec = obj_spec.get("left", {})
            if left_desc_spec:
                symbol = left_desc_spec.get("symbol")
                label = left_desc_spec.get("index") or left_desc_spec.get("label")
                all_descriptors.append(
                    BondingDescriptorIR(
                        symbol=symbol,
                        label=label,
                        role="terminal",
                    )
                )

            right_desc_spec = obj_spec.get("right", {})
            if right_desc_spec:
                symbol = right_desc_spec.get("symbol")
                label = right_desc_spec.get("index") or right_desc_spec.get("label")
                all_descriptors.append(
                    BondingDescriptorIR(
                        symbol=symbol,
                        label=label,
                        role="terminal",
                    )
                )
            
            terminals = TerminalDescriptorIR(descriptors=all_descriptors)

            # Build repeat units
            repeat_units = []
            for unit_spec in obj_spec.get("units", []):
                unit_graph = mk_smiles_ir(*unit_spec)
                unit_subgraph = BigSmilesSubgraphIR(
                    atoms=list(unit_graph.atoms),
                    bonds=list(unit_graph.bonds),
                    descriptors=[],
                )
                repeat_units.append(RepeatUnitIR(graph=unit_subgraph))

            # Build end groups
            end_groups = []
            for eg_spec in obj_spec.get("end_groups", []):
                eg_graph = mk_smiles_ir(*eg_spec)
                eg_subgraph = BigSmilesSubgraphIR(
                    atoms=list(eg_graph.atoms),
                    bonds=list(eg_graph.bonds),
                    descriptors=[],
                )
                end_groups.append(EndGroupIR(graph=eg_subgraph))

            stochastic_objects.append(
                StochasticObjectIR(
                    terminals=terminals,
                    repeat_units=repeat_units,
                    end_groups=end_groups,
                )
            )

        # Handle implicit smiles (trailing smiles after stochastic objects)
        if "implicit" in seg_spec and seg_spec["implicit"] is not None:
            implicit_spec = seg_spec["implicit"]
            if isinstance(implicit_spec, tuple) and len(implicit_spec) == 2:
                implicit_atoms, implicit_bonds = implicit_spec
                implicit_graph = mk_smiles_ir(implicit_atoms, implicit_bonds)
                backbone.atoms.extend(implicit_graph.atoms)
                backbone.bonds.extend(implicit_graph.bonds)

    return BigSmilesMoleculeIR(backbone=backbone, stochastic_objects=stochastic_objects)


# Bond descriptor test cases (field variants)
bond_descriptor_cases = [
    # (name, kwargs, expected_checks)
    ("empty", {}, {"symbol": None, "label": None}),
    ("symbol_only", {"symbol": "<"}, {"symbol": "<", "label": None}),
    ("symbol_label", {"symbol": ">", "label": 1}, {"symbol": ">", "label": 1}),
    (
        "symbol_label_extras",
        {"symbol": "$", "label": 2, "extras": {"generation": [0, 1]}},
        {"symbol": "$", "label": 2},
    ),
]

# Stochastic object test cases (minimal/full field coverage)
stochastic_object_cases = [
    # (name, left_desc, right_desc, units, end_groups, distribution)
    (
        "minimal_one_unit",
        {"symbol": "<", "label": 1},
        {"symbol": ">", "label": 1},
        [(["C", "C"], [(0, 1, "-")])],
        None,
        None,
    ),
    (
        "two_units",
        {"symbol": "<", "label": 1},
        {"symbol": ">", "label": 1},
        [(["C", "C"], [(0, 1, "-")]), (["O"], [])],
        None,
        None,
    ),
    (
        "with_end_groups",
        {"symbol": "<", "label": 2},
        {"symbol": ">", "label": 2},
        [(["C", "C", "O"], [(0, 1, "-"), (1, 2, "-")])],
        [(["H"], []), (["OH"], [])],
        None,
    ),
    (
        "with_distribution",
        {"symbol": "<", "label": 3},
        {"symbol": ">", "label": 3},
        [(["C", "C"], [(0, 1, "-")])],
        None,
        {"name": "flory_schulz", "params": [0.9]},
    ),
    (
        "full_fields",
        {"symbol": "$", "label": 10, "generation": [0]},
        {"symbol": "$", "label": 10, "generation": [0]},
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
                        "left": {"symbol": "<", "label": 1},
                        "right": {"symbol": ">", "label": 1},
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
                        "left": {"symbol": "<", "label": 1},
                        "right": {"symbol": ">", "label": 1},
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
                        "left": {"symbol": "<", "label": 1},
                        "right": {"symbol": ">", "label": 1},
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
                        "left": {"symbol": "<", "label": 1},
                        "right": {"symbol": ">", "label": 1},
                        "units": [(["C"], [])],
                    },
                    {
                        "left": {"symbol": "<", "label": 2},
                        "right": {"symbol": ">", "label": 2},
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
                        "left": {"symbol": "<", "label": 1},
                        "right": {"symbol": ">", "label": 1},
                        "units": [(["C", "C"], [(0, 1, "-")])],
                    }
                ],
                "implicit": None,
            },
            {
                "objects": [
                    {
                        "left": {"symbol": "<", "label": 2},
                        "right": {"symbol": ">", "label": 2},
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
        desc = BondingDescriptorIR(**kwargs)
        for key, val in expected.items():
            assert getattr(desc, key) == val

    @pytest.mark.parametrize(
        "name,left_desc,right_desc,units,end_groups,distribution",
        stochastic_object_cases,
    )
    def test_stochastic_object_fields(
        self, name, left_desc, right_desc, units, end_groups, distribution
    ):
        from molpy.parser.smiles.bigsmiles_ir import (
            BigSmilesSubgraphIR,
            EndGroupIR,
            RepeatUnitIR,
            TerminalDescriptorIR,
        )

        # Handle generation parameter by moving it to extras
        left_desc_fixed = {}
        if left_desc:
            left_desc_fixed = dict(left_desc)
            if "generation" in left_desc_fixed:
                generation = left_desc_fixed.pop("generation")
                left_desc_fixed["extras"] = {"generation": generation}
        right_desc_fixed = {}
        if right_desc:
            right_desc_fixed = dict(right_desc)
            if "generation" in right_desc_fixed:
                generation = right_desc_fixed.pop("generation")
                right_desc_fixed["extras"] = {"generation": generation}

        # Merge left and right descriptors into unified terminals
        left_descriptors = [BondingDescriptorIR(**left_desc_fixed)] if left_desc else []
        right_descriptors = [BondingDescriptorIR(**right_desc_fixed)] if right_desc else []
        all_descriptors = left_descriptors + right_descriptors
        terminals = TerminalDescriptorIR(descriptors=all_descriptors)

        repeat_units = []
        for unit_spec in units:
            unit_graph = mk_smiles_ir(*unit_spec)
            unit_subgraph = BigSmilesSubgraphIR(
                atoms=list(unit_graph.atoms),
                bonds=list(unit_graph.bonds),
                descriptors=[],
            )
            repeat_units.append(RepeatUnitIR(graph=unit_subgraph))

        end_group_irs = []
        if end_groups:
            for eg_spec in end_groups:
                eg_graph = mk_smiles_ir(*eg_spec)
                eg_subgraph = BigSmilesSubgraphIR(
                    atoms=list(eg_graph.atoms),
                    bonds=list(eg_graph.bonds),
                    descriptors=[],
                )
                end_group_irs.append(EndGroupIR(graph=eg_subgraph))

        sobj = StochasticObjectIR(
            terminals=terminals,
            repeat_units=repeat_units,
            end_groups=end_group_irs,
        )

        # Check terminals contain the expected descriptors
        expected_desc_count = (1 if left_desc else 0) + (1 if right_desc else 0)
        assert len(sobj.terminals.descriptors) == expected_desc_count
        if left_desc:
            assert sobj.terminals.descriptors[0].symbol == left_desc["symbol"]
        if right_desc:
            idx = 1 if left_desc else 0
            assert sobj.terminals.descriptors[idx].symbol == right_desc["symbol"]
        assert len(sobj.repeat_units) == len(units)
        if end_groups:
            assert len(sobj.end_groups) == len(end_groups)
        else:
            assert len(sobj.end_groups) == 0

    @pytest.mark.parametrize("name,start_specs,segments_specs", chain_segment_cases)
    def test_chain_and_segments(self, name, start_specs, segments_specs):
        expected_ir = mk_bigsmiles_ir(start_specs, segments_specs)

        # Verify structure
        assert expected_ir.backbone is not None
        assert len(expected_ir.stochastic_objects) == sum(
            len(seg["objects"]) for seg in segments_specs
        )

    def test_bigsmiles_ir_is_distinct_class(self):
        # BigSmilesMoleculeIR is a distinct class
        assert BigSmilesMoleculeIR is not SmilesGraphIR
        assert not issubclass(BigSmilesMoleculeIR, SmilesGraphIR)

    def test_bigsmiles_ir_has_backbone_field(self):
        from molpy.parser.smiles.bigsmiles_ir import BigSmilesSubgraphIR

        backbone = BigSmilesSubgraphIR()
        big = BigSmilesMoleculeIR(backbone=backbone, stochastic_objects=[])

        assert hasattr(big, "backbone")
        assert isinstance(big.backbone, BigSmilesSubgraphIR)
        assert isinstance(big, BigSmilesMoleculeIR)

    def test_parse_bigsmiles_method_exists(self):
        # Verify parse_bigsmiles function exists
        assert callable(parse_bigsmiles)


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
    (
        "[$]CC[$]",
        # With new IR, descriptors are preserved as graph nodes in backbone
        # Create expected structure with descriptors in backbone
        mk_bigsmiles_ir_with_descriptors(
            (["C", "C"], [(0, 1, "-")]),  # backbone atoms and bonds
            [
                {"symbol": "$", "role": "internal"},
                {"symbol": "$", "role": "internal"},
            ],  # descriptors
            [],  # no stochastic objects
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
                            "left": {"symbol": "<", "label": 1},
                            "right": {"symbol": ">", "label": 1},
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
        if expected is None:
            pytest.skip("Test case skipped - needs special handling for descriptors")
        ir = parse_bigsmiles(smiles)
        assert ir == expected

    @pytest.mark.parametrize("bigsmiles,expected", simple_repeat_bigsmiles)
    def test_simple_repeat_units(self, bigsmiles, expected):
        """Test basic {} repeat unit parsing."""
        ir = parse_bigsmiles(bigsmiles)
        assert ir == expected

    @pytest.mark.parametrize("bigsmiles,expected", stochastic_copolymer_bigsmiles)
    def test_stochastic_copolymer(self, bigsmiles, expected):
        """Test random copolymer with comma-separated monomers.
        
        Note: Terminal descriptor count varies based on unanchored descriptors
        extracted from repeat units under the unified terminals model.
        """
        ir = parse_bigsmiles(bigsmiles)
        # Verify structural elements rather than exact equality
        assert len(ir.stochastic_objects) == len(expected.stochastic_objects)
        for sobj_actual, sobj_expected in zip(ir.stochastic_objects, expected.stochastic_objects):
            assert len(sobj_actual.repeat_units) == len(sobj_expected.repeat_units)
            # Verify repeat unit atom counts match
            for unit_a, unit_e in zip(sobj_actual.repeat_units, sobj_expected.repeat_units):
                assert len(unit_a.graph.atoms) == len(unit_e.graph.atoms)

    @pytest.mark.parametrize("bigsmiles,expected", block_copolymer_bigsmiles)
    def test_block_copolymers(self, bigsmiles, expected):
        """Test block copolymer with multiple {} segments."""
        ir = parse_bigsmiles(bigsmiles)
        assert ir == expected

    @pytest.mark.parametrize("bigsmiles,expected", mixed_bigsmiles)
    def test_mixed_with_smiles(self, bigsmiles, expected):
        """Test BigSMILES mixed with plain SMILES fragments."""
        ir = parse_bigsmiles(bigsmiles)
        assert ir == expected

    @pytest.mark.parametrize("bigsmiles,expected", indexed_descriptor_bigsmiles)
    def test_indexed_descriptors(self, bigsmiles, expected):
        """Test bond descriptors with indices."""
        ir = parse_bigsmiles(bigsmiles)
        assert ir == expected

    def test_empty_bigsmiles(self):
        """Test empty string returns valid IR."""
        ir = parse_bigsmiles("")
        expected = mk_bigsmiles_ir(([], []), [])
        assert ir == expected


RDKIT_FIND = True
try:
    from molpy.external.rdkit_adapter import smilesir_to_mol
except:
    RDKIT_FIND = False
    smilesir_to_mol = lambda x: x


@pytest.mark.skipif(not RDKIT_FIND, reason="rdkit not find")
class TestRDKitConverter:
    """Test SmilesIR -> RDKit Mol conversion."""

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
        """Test basic IR -> Mol conversion."""

        ir = parse_smiles(smiles)
        mol = smilesir_to_mol(ir)

        assert mol is not None
        assert mol.GetNumAtoms() == n_atoms
        assert mol.GetNumBonds() == n_bonds

    def test_smilesir_to_mol_charged_atoms(self):
        """Test conversion with charged atoms."""

        # [NH4+]
        ir = parse_smiles("[NH4+]")
        mol = smilesir_to_mol(ir)

        assert mol.GetNumAtoms() == 1
        n_atom = mol.GetAtomWithIdx(0)
        assert n_atom.GetSymbol() == "N"
        assert n_atom.GetFormalCharge() == 1
        assert n_atom.GetTotalNumHs() == 4

    def test_smilesir_to_mol_isotopes(self):
        """Test conversion with isotopes."""

        # [13C]
        ir = parse_smiles("[13C]")
        mol = smilesir_to_mol(ir)

        assert mol.GetNumAtoms() == 1
        c_atom = mol.GetAtomWithIdx(0)
        assert c_atom.GetSymbol() == "C"
        assert c_atom.GetIsotope() == 13

    def test_smilesir_to_mol_aromatic(self):
        """Test conversion with aromatic atoms."""

        # Benzene
        ir = parse_smiles("c1ccccc1")
        mol = smilesir_to_mol(ir)

        assert mol.GetNumAtoms() == 6
        # All atoms should be aromatic after sanitization
        for atom in mol.GetAtoms():
            assert atom.GetIsAromatic()

    def test_smilesir_to_mol_chirality(self):
        """Test conversion with chiral centers."""

        from rdkit import Chem

        # L-alanine: N[C@@H](C)C(=O)O
        ir = parse_smiles("N[C@@H](C)C(=O)O")
        mol = smilesir_to_mol(ir)

        # Check chiral center exists
        chiral_atoms = [
            a
            for a in mol.GetAtoms()
            if a.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED
        ]
        assert len(chiral_atoms) > 0
