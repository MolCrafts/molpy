# author: Roy Kid
# contact: lijichen365@126.com
# date: 2021-10-26
# version: 0.0.1

import pytest

from molpy.io.smiles import read_smiles
from molpy.factory import full

testcases = (
            (
                "CCCC",
                [
                    (0, {"element": "C", "charge": 0, "hcount": 3, "aromatic": False}),
                    (1, {"element": "C", "charge": 0, "hcount": 2, "aromatic": False}),
                    (2, {"element": "C", "charge": 0, "hcount": 2, "aromatic": False}),
                    (3, {"element": "C", "charge": 0, "hcount": 3, "aromatic": False}),
                ],
                [(0, 1, {"order": 1}), (1, 2, {"order": 1}), (2, 3, {"order": 1})],
                False,
            ),
            (
                "CCC(CC)CC",
                [
                    (0, {"element": "C", "charge": 0, "hcount": 3, "aromatic": False}),
                    (1, {"element": "C", "charge": 0, "hcount": 2, "aromatic": False}),
                    (2, {"element": "C", "charge": 0, "hcount": 1, "aromatic": False}),
                    (3, {"element": "C", "charge": 0, "hcount": 2, "aromatic": False}),
                    (4, {"element": "C", "charge": 0, "hcount": 3, "aromatic": False}),
                    (5, {"element": "C", "charge": 0, "hcount": 2, "aromatic": False}),
                    (6, {"element": "C", "charge": 0, "hcount": 3, "aromatic": False}),
                ],
                [
                    (0, 1, {"order": 1}),
                    (1, 2, {"order": 1}),
                    (2, 3, {"order": 1}),
                    (2, 5, {"order": 1}),
                    (3, 4, {"order": 1}),
                    (5, 6, {"order": 1}),
                ],
                False,
            ),
            (
                "C=C",
                [
                    (0, {"element": "C", "charge": 0, "hcount": 2, "aromatic": False}),
                    (1, {"element": "C", "charge": 0, "hcount": 2, "aromatic": False}),
                ],
                [(0, 1, {"order": 2})],
                False,
            ),
            (
                "C1CC1",
                [
                    (0, {"charge": 0, "element": "C", "hcount": 2, "aromatic": False}),
                    (1, {"charge": 0, "element": "C", "hcount": 2, "aromatic": False}),
                    (2, {"charge": 0, "element": "C", "hcount": 2, "aromatic": False}),
                ],
                [(0, 1, {"order": 1}), (0, 2, {"order": 1}), (1, 2, {"order": 1})],
                False,
            ),
            (
                "C%11CC%11",
                [
                    (0, {"charge": 0, "element": "C", "hcount": 2, "aromatic": False}),
                    (1, {"charge": 0, "element": "C", "hcount": 2, "aromatic": False}),
                    (2, {"charge": 0, "element": "C", "hcount": 2, "aromatic": False}),
                ],
                [(0, 1, {"order": 1}), (0, 2, {"order": 1}), (1, 2, {"order": 1})],
                False,
            ),
            (
                "c1ccccc1",
                [
                    (0, {"charge": 0, "element": "C", "hcount": 1, "aromatic": True}),
                    (1, {"charge": 0, "element": "C", "hcount": 1, "aromatic": True}),
                    (2, {"charge": 0, "element": "C", "hcount": 1, "aromatic": True}),
                    (3, {"charge": 0, "element": "C", "hcount": 1, "aromatic": True}),
                    (4, {"charge": 0, "element": "C", "hcount": 1, "aromatic": True}),
                    (5, {"charge": 0, "element": "C", "hcount": 1, "aromatic": True}),
                ],
                [
                    (0, 1, {"order": 1.5}),
                    (0, 5, {"order": 1.5}),
                    (1, 2, {"order": 1.5}),
                    (2, 3, {"order": 1.5}),
                    (3, 4, {"order": 1.5}),
                    (4, 5, {"order": 1.5}),
                ],
                False,
            ),
            (
                "C1CC=1",
                [
                    (0, {"charge": 0, "element": "C", "hcount": 1, "aromatic": False}),
                    (1, {"charge": 0, "element": "C", "hcount": 2, "aromatic": False}),
                    (2, {"charge": 0, "element": "C", "hcount": 1, "aromatic": False}),
                ],
                [(0, 1, {"order": 1}), (0, 2, {"order": 2}), (1, 2, {"order": 1})],
                False,
            ),
            (
                "C=1CC=1",
                [
                    (0, {"charge": 0, "element": "C", "hcount": 1, "aromatic": False}),
                    (1, {"charge": 0, "element": "C", "hcount": 2, "aromatic": False}),
                    (2, {"charge": 0, "element": "C", "hcount": 1, "aromatic": False}),
                ],
                [(0, 1, {"order": 1}), (0, 2, {"order": 2}), (1, 2, {"order": 1})],
                False,
            ),
            (
                "C=1CC1",
                [
                    (0, {"charge": 0, "element": "C", "hcount": 1, "aromatic": False}),
                    (1, {"charge": 0, "element": "C", "hcount": 2, "aromatic": False}),
                    (2, {"charge": 0, "element": "C", "hcount": 1, "aromatic": False}),
                ],
                [(0, 1, {"order": 1}), (0, 2, {"order": 2}), (1, 2, {"order": 1})],
                False,
            ),
            (
                "C%11CC=%11",
                [
                    (0, {"charge": 0, "element": "C", "hcount": 1, "aromatic": False}),
                    (1, {"charge": 0, "element": "C", "hcount": 2, "aromatic": False}),
                    (2, {"charge": 0, "element": "C", "hcount": 1, "aromatic": False}),
                ],
                [(0, 1, {"order": 1}), (0, 2, {"order": 2}), (1, 2, {"order": 1})],
                False,
            ),
            (
                "OS(=O)(=S)O",
                [
                    (0, {"charge": 0, "element": "O", "hcount": 1, "aromatic": False}),
                    (1, {"charge": 0, "element": "S", "hcount": 0, "aromatic": False}),
                    (2, {"charge": 0, "element": "O", "hcount": 0, "aromatic": False}),
                    (3, {"charge": 0, "element": "S", "hcount": 0, "aromatic": False}),
                    (4, {"charge": 0, "element": "O", "hcount": 1, "aromatic": False}),
                ],
                [
                    (0, 1, {"order": 1}),
                    (1, 2, {"order": 2}),
                    (1, 3, {"order": 2}),
                    (1, 4, {"order": 1}),
                ],
                False,
            ),
            (
                "C(C(C(C(C(C(C(C(C(C(C(C(C(C(C(C(C(C(C(C(C))))))))))))))))))))C",
                [
                    (0, {"charge": 0, "element": "C", "hcount": 2, "aromatic": False}),
                    (1, {"charge": 0, "element": "C", "hcount": 2, "aromatic": False}),
                    (2, {"charge": 0, "element": "C", "hcount": 2, "aromatic": False}),
                    (3, {"charge": 0, "element": "C", "hcount": 2, "aromatic": False}),
                    (4, {"charge": 0, "element": "C", "hcount": 2, "aromatic": False}),
                    (5, {"charge": 0, "element": "C", "hcount": 2, "aromatic": False}),
                    (6, {"charge": 0, "element": "C", "hcount": 2, "aromatic": False}),
                    (7, {"charge": 0, "element": "C", "hcount": 2, "aromatic": False}),
                    (8, {"charge": 0, "element": "C", "hcount": 2, "aromatic": False}),
                    (9, {"charge": 0, "element": "C", "hcount": 2, "aromatic": False}),
                    (10, {"charge": 0, "element": "C", "hcount": 2, "aromatic": False}),
                    (11, {"charge": 0, "element": "C", "hcount": 2, "aromatic": False}),
                    (12, {"charge": 0, "element": "C", "hcount": 2, "aromatic": False}),
                    (13, {"charge": 0, "element": "C", "hcount": 2, "aromatic": False}),
                    (14, {"charge": 0, "element": "C", "hcount": 2, "aromatic": False}),
                    (15, {"charge": 0, "element": "C", "hcount": 2, "aromatic": False}),
                    (16, {"charge": 0, "element": "C", "hcount": 2, "aromatic": False}),
                    (17, {"charge": 0, "element": "C", "hcount": 2, "aromatic": False}),
                    (18, {"charge": 0, "element": "C", "hcount": 2, "aromatic": False}),
                    (19, {"charge": 0, "element": "C", "hcount": 2, "aromatic": False}),
                    (20, {"charge": 0, "element": "C", "hcount": 3, "aromatic": False}),
                    (21, {"charge": 0, "element": "C", "hcount": 3, "aromatic": False}),
                ],
                [
                    (0, 1, {"order": 1}),
                    (0, 21, {"order": 1}),
                    (1, 2, {"order": 1}),
                    (2, 3, {"order": 1}),
                    (3, 4, {"order": 1}),
                    (4, 5, {"order": 1}),
                    (5, 6, {"order": 1}),
                    (6, 7, {"order": 1}),
                    (7, 8, {"order": 1}),
                    (8, 9, {"order": 1}),
                    (9, 10, {"order": 1}),
                    (10, 11, {"order": 1}),
                    (11, 12, {"order": 1}),
                    (12, 13, {"order": 1}),
                    (13, 14, {"order": 1}),
                    (14, 15, {"order": 1}),
                    (15, 16, {"order": 1}),
                    (16, 17, {"order": 1}),
                    (17, 18, {"order": 1}),
                    (18, 19, {"order": 1}),
                    (19, 20, {"order": 1}),
                ],
                False,
            ),
            (
                "N1CC2CCCC2CC1",
                [
                    (0, {"charge": 0, "element": "N", "hcount": 1, "aromatic": False}),
                    (1, {"charge": 0, "element": "C", "hcount": 2, "aromatic": False}),
                    (2, {"charge": 0, "element": "C", "hcount": 1, "aromatic": False}),
                    (3, {"charge": 0, "element": "C", "hcount": 2, "aromatic": False}),
                    (4, {"charge": 0, "element": "C", "hcount": 2, "aromatic": False}),
                    (5, {"charge": 0, "element": "C", "hcount": 2, "aromatic": False}),
                    (6, {"charge": 0, "element": "C", "hcount": 1, "aromatic": False}),
                    (7, {"charge": 0, "element": "C", "hcount": 2, "aromatic": False}),
                    (8, {"charge": 0, "element": "C", "hcount": 2, "aromatic": False}),
                ],
                [
                    (0, 1, {"order": 1}),
                    (0, 8, {"order": 1}),
                    (1, 2, {"order": 1}),
                    (2, 3, {"order": 1}),
                    (2, 6, {"order": 1}),
                    (3, 4, {"order": 1}),
                    (4, 5, {"order": 1}),
                    (5, 6, {"order": 1}),
                    (6, 7, {"order": 1}),
                    (7, 8, {"order": 1}),
                ],
                False,
            ),
            (
                "c1ccc2ccccc2c1",
                [
                    (0, {"charge": 0, "element": "C", "aromatic": True}),
                    (1, {"charge": 0, "element": "C", "aromatic": True}),
                    (2, {"charge": 0, "element": "C", "aromatic": True}),
                    (3, {"charge": 0, "element": "C", "aromatic": True}),
                    (4, {"charge": 0, "element": "C", "aromatic": True}),
                    (5, {"charge": 0, "element": "C", "aromatic": True}),
                    (6, {"charge": 0, "element": "C", "aromatic": True}),
                    (7, {"charge": 0, "element": "C", "aromatic": True}),
                    (8, {"charge": 0, "element": "C", "aromatic": True}),
                    (9, {"charge": 0, "element": "C", "aromatic": True}),
                    (10, {"charge": 0, "element": "H", "aromatic": False}),
                    (11, {"charge": 0, "element": "H", "aromatic": False}),
                    (12, {"charge": 0, "element": "H", "aromatic": False}),
                    (13, {"charge": 0, "element": "H", "aromatic": False}),
                    (14, {"charge": 0, "element": "H", "aromatic": False}),
                    (15, {"charge": 0, "element": "H", "aromatic": False}),
                    (16, {"charge": 0, "element": "H", "aromatic": False}),
                    (17, {"charge": 0, "element": "H", "aromatic": False}),
                ],
                [
                    (0, 1, {"order": 1.5}),
                    (0, 9, {"order": 1.5}),
                    (0, 10, {"order": 1}),
                    (1, 2, {"order": 1.5}),
                    (1, 11, {"order": 1}),
                    (2, 3, {"order": 1.5}),
                    (2, 12, {"order": 1}),
                    (3, 4, {"order": 1.5}),
                    (3, 8, {"order": 1.5}),
                    (4, 5, {"order": 1.5}),
                    (4, 13, {"order": 1}),
                    (5, 6, {"order": 1.5}),
                    (5, 14, {"order": 1}),
                    (6, 7, {"order": 1.5}),
                    (6, 15, {"order": 1}),
                    (7, 8, {"order": 1.5}),
                    (7, 16, {"order": 1}),
                    (8, 9, {"order": 1.5}),
                    (9, 17, {"order": 1}),
                ],
                True,
            ),
            (
                "C12(CCCCC1)CCCCC2",
                [
                    (0, {"charge": 0, "element": "C", "hcount": 0, "aromatic": False}),
                    (1, {"charge": 0, "element": "C", "hcount": 2, "aromatic": False}),
                    (2, {"charge": 0, "element": "C", "hcount": 2, "aromatic": False}),
                    (3, {"charge": 0, "element": "C", "hcount": 2, "aromatic": False}),
                    (4, {"charge": 0, "element": "C", "hcount": 2, "aromatic": False}),
                    (5, {"charge": 0, "element": "C", "hcount": 2, "aromatic": False}),
                    (6, {"charge": 0, "element": "C", "hcount": 2, "aromatic": False}),
                    (7, {"charge": 0, "element": "C", "hcount": 2, "aromatic": False}),
                    (8, {"charge": 0, "element": "C", "hcount": 2, "aromatic": False}),
                    (9, {"charge": 0, "element": "C", "hcount": 2, "aromatic": False}),
                    (10, {"charge": 0, "element": "C", "hcount": 2, "aromatic": False}),
                ],
                [
                    (0, 1, {"order": 1}),
                    (0, 5, {"order": 1}),
                    (0, 6, {"order": 1}),
                    (0, 10, {"order": 1}),
                    (1, 2, {"order": 1}),
                    (2, 3, {"order": 1}),
                    (3, 4, {"order": 1}),
                    (4, 5, {"order": 1}),
                    (6, 7, {"order": 1}),
                    (7, 8, {"order": 1}),
                    (8, 9, {"order": 1}),
                    (9, 10, {"order": 1}),
                ],
                False,
            ),
            (
                "[H]C([H])([H])[H]",
                [(0, {"charge": 0, "element": "C", "hcount": 4, "aromatic": False})],
                [],
                False,
            ),
            (
                "[H]O([H])[H]O([H])[H]",
                [
                    (0, {"charge": 0, "element": "H", "hcount": 0, "aromatic": False}),
                    (1, {"charge": 0, "element": "O", "hcount": 2, "aromatic": False}),
                    (2, {"charge": 0, "element": "H", "hcount": 0, "aromatic": False}),
                    (3, {"charge": 0, "element": "H", "hcount": 0, "aromatic": False}),
                    (4, {"charge": 0, "element": "O", "hcount": 2, "aromatic": False}),
                    (5, {"charge": 0, "element": "H", "hcount": 0, "aromatic": False}),
                    (6, {"charge": 0, "element": "H", "hcount": 0, "aromatic": False})
                ],
                [(1, 3, {"order": 1}), (3, 4, {"order": 1})],
                False,
            ),
            (
                "[H]=C",
                [
                    (0, {"charge": 0, "element": "H", "hcount": 0, "aromatic": False}),
                    (1, {"charge": 0, "element": "C", "hcount": 2, "aromatic": False}),
                ],
                [(0, 1, {"order": 2})],
                False,
            ),
            (
                "[H][H]",
                [
                    (0, {"charge": 0, "element": "H", "hcount": 0, "aromatic": False}),
                    (1, {"charge": 0, "element": "H", "hcount": 0, "aromatic": False}),
                ],
                [(0, 1, {"order": 1})],
                False,
            ),
            (
                "[13CH3-1:1]",
                [
                    (
                        0,
                        {
                            "charge": -1,
                            "class": 1,
                            "element": "C",
                            "hcount": 3,
                            "isotope": 13,
                            "aromatic": False,
                        },
                    )
                ],
                [],
                False,
            ),
            (
                "[013CH3-:1]",
                [
                    (
                        0,
                        {
                            "charge": -1,
                            "class": 1,
                            "element": "C",
                            "hcount": 3,
                            "isotope": 13,
                            "aromatic": False,
                        },
                    )
                ],
                [],
                False,
            ),
            (
                "[Cu++]",
                [(0, {"charge": 2, "element": "Cu", "hcount": 0, "aromatic": False})],
                [],
                False,
            ),
            (
                "[Cu+2]",
                [(0, {"charge": 2, "element": "Cu", "hcount": 0, "aromatic": False})],
                [],
                False,
            ),
            (
                "[2H][CH2]",
                [
                    (
                        0,
                        {
                            "charge": 0,
                            "element": "H",
                            "hcount": 0,
                            "isotope": 2,
                            "aromatic": False,
                        },
                    ),
                    (1, {"charge": 0, "element": "C", "hcount": 2, "isotope": 0, "aromatic": False}),
                ],
                [(0, 1, {"order": 1})],
                False,
            ),
            ("*", [(0, {"charge": 0, "aromatic": False, "hcount": 0})], [], False),
            ("[*--]", [(0, {"charge": -2, "aromatic": False, "hcount": 0})], [], False),
            ("[*-]", [(0, {"charge": -1, "aromatic": False, "hcount": 0})], [], False),
            (
                "[H+]",
                [(0, {"element": "H", "charge": 1, "hcount": 0, "aromatic": False})],
                [],
                False,
            ),
            (
                "[cH-1]1[cH-1][cH-1]1",
                [
                    (0, {"element": "C", "aromatic": False, "charge": -1, "hcount": 1}),
                    (1, {"element": "C", "aromatic": False, "charge": -1, "hcount": 1}),
                    (2, {"element": "C", "aromatic": False, "charge": -1, "hcount": 1}),
                ],
                [(0, 1, {"order": 1}), (1, 2, {"order": 1}), (2, 0, {"order": 1})],
                False,
            ),
            (
                "[Rh-](Cl)(Cl)(Cl)(Cl)$[Rh-](Cl)(Cl)(Cl)Cl",
                [
                    (
                        0,
                        {"charge": -1, "element": "Rh", "hcount": 0, "aromatic": False},
                    ),
                    (1, {"charge": 0, "element": "Cl", "hcount": 0, "aromatic": False}),
                    (2, {"charge": 0, "element": "Cl", "hcount": 0, "aromatic": False}),
                    (3, {"charge": 0, "element": "Cl", "hcount": 0, "aromatic": False}),
                    (4, {"charge": 0, "element": "Cl", "hcount": 0, "aromatic": False}),
                    (
                        5,
                        {"charge": -1, "element": "Rh", "hcount": 0, "aromatic": False},
                    ),
                    (6, {"charge": 0, "element": "Cl", "hcount": 0, "aromatic": False}),
                    (7, {"charge": 0, "element": "Cl", "hcount": 0, "aromatic": False}),
                    (8, {"charge": 0, "element": "Cl", "hcount": 0, "aromatic": False}),
                    (9, {"charge": 0, "element": "Cl", "hcount": 0, "aromatic": False}),
                ],
                [
                    (0, 1, {"order": 1}),
                    (0, 2, {"order": 1}),
                    (0, 3, {"order": 1}),
                    (0, 4, {"order": 1}),
                    (0, 5, {"order": 4}),
                    (5, 6, {"order": 1}),
                    (5, 7, {"order": 1}),
                    (5, 8, {"order": 1}),
                    (5, 9, {"order": 1}),
                ],
                False,
            ),
            (
                "c1occc1",
                [
                    (0, {"charge": 0, "element": "C", "aromatic": True}),
                    (1, {"charge": 0, "element": "O", "aromatic": True}),
                    (2, {"charge": 0, "element": "C", "aromatic": True}),
                    (3, {"charge": 0, "element": "C", "aromatic": True}),
                    (4, {"charge": 0, "element": "C", "aromatic": True}),
                    (5, {"charge": 0, "element": "H", "aromatic": False}),
                    (6, {"charge": 0, "element": "H", "aromatic": False}),
                    (7, {"charge": 0, "element": "H", "aromatic": False}),
                    (8, {"charge": 0, "element": "H", "aromatic": False}),
                ],
                [
                    (0, 1, {"order": 1.5}),
                    (0, 4, {"order": 1.5}),
                    (0, 5, {"order": 1}),
                    (1, 2, {"order": 1.5}),
                    (2, 3, {"order": 1.5}),
                    (2, 6, {"order": 1}),
                    (3, 4, {"order": 1.5}),
                    (3, 7, {"order": 1}),
                    (4, 8, {"order": 1}),
                ],
                True,
            ),
            (
                "[O-]C(O)CN",
                [
                    (0, {"charge": -1, "element": "O", "aromatic": False}),
                    (1, {"charge": 0, "element": "C", "aromatic": False}),
                    (2, {"charge": 0, "element": "O", "aromatic": False}),
                    (3, {"charge": 0, "element": "C", "aromatic": False}),
                    (4, {"charge": 0, "element": "N", "aromatic": False}),
                    (5, {"charge": 0, "element": "H", "aromatic": False}),
                    (6, {"charge": 0, "element": "H", "aromatic": False}),
                    (7, {"charge": 0, "element": "H", "aromatic": False}),
                    (8, {"charge": 0, "element": "H", "aromatic": False}),
                    (9, {"charge": 0, "element": "H", "aromatic": False}),
                    (10, {"charge": 0, "element": "H", "aromatic": False}),
                ],
                [
                    (0, 1, {"order": 1}),
                    (1, 2, {"order": 1}),
                    (1, 3, {"order": 1}),
                    (1, 5, {"order": 1}),
                    (2, 6, {"order": 1}),
                    (3, 4, {"order": 1}),
                    (3, 7, {"order": 1}),
                    (3, 8, {"order": 1}),
                    (4, 9, {"order": 1}),
                    (4, 10, {"order": 1}),
                ],
                True,
            ),
            (
                "[*+]1[*][*]1",
                [
                    (0, {"charge": 1, "aromatic": True, "hcount": 0}),
                    (1, {"charge": 0, "aromatic": True, "hcount": 0}),
                    (2, {"charge": 0, "aromatic": True, "hcount": 0}),
                ],
                [
                    (0, 1, {"order": 1.5}),
                    (1, 2, {"order": 1.5}),
                    (2, 0, {"order": 1.5}),
                ],
                False,
            ),
            (
                "N1[*][*]1",
                [
                    (0, {"element": "N", "charge": 0, "aromatic": True, "hcount": 1}),
                    (1, {"element": "", "charge": 0, "aromatic": True, "hcount": 0}),
                    (2, {"element": "", "charge": 0, "aromatic": True, "hcount": 0}),
                ],
                [
                    (0, 1, {"order": 1.5}),
                    (1, 2, {"order": 1.5}),
                    (2, 0, {"order": 1.5}),
                ],
                False,
            ),
        )

def make_mol(node_data, edge_data):
    atomName = [n[0] for n in node_data]
    atomProperty = [n[1] for n in node_data]
    atomProperty_nested = {}
    for k in atomProperty[0]:
        atomProperty_nested[k] = []
    for d in atomProperty:
        for k, v in d.items():
            atomProperty_nested[k].append(v)
    return full("smiles", atomName, **atomProperty_nested)    

class TestSmiles:
    @pytest.mark.parametrize(
        "smiles, node_data, edge_data, explicit_h", testcases
    )
    def test_read(self, smiles, node_data, edge_data, explicit_h):
        found = read_smiles(smiles, explicit_hydrogen=explicit_h)
        expected = make_mol(node_data, edge_data)
        for bondidx in edge_data:
            expected.addBondByIndex(bondidx[0], bondidx[1], **bondidx[2])
            
    # @pytest.mark.parametrize('node_data, edge_data, expl_h', testcases)
    # def test_write(node_data, edge_data, expl_h):
    #     mol = make_mol(node_data, edge_data)
    #     smiles = write_smiles(mol)
    #     found = read_smiles(smiles, explicit_hydrogen=expl_h, reinterpret_aromatic=False)
    #     assertEqualGraphs(mol, found)

    @pytest.mark.parametrize('smiles, error_type', (
        ('[CL-]', ValueError),
        ('[HH]', ValueError),
        ('c1ccccc2', KeyError),
        ('C%1CC1', ValueError),
        ('c1c1CC', ValueError),
        ('CC11C', ValueError),
        ('1CCC1', ValueError),
        ('cccccc', ValueError),
        ('C=1CC-1', ValueError),
    ))
    def test_invalid_smiles(self, smiles, error_type):
        with pytest.raises(error_type):
            read_smiles(smiles)


    @pytest.mark.parametrize('smiles, n_records',[
        (r'F/C=C/F', 2),
        (r'C(\F)=C/F', 2),
        (r'F\C=C/F', 2),
        (r'C(/F)=C/F', 2),
        ('NC(Br)=[C@]=C(O)C', 1),
        ('c1ccccc1', 0)
    ])
    def test_stereo_logging(self, smiles, n_records):
        with pytest.warns(None) as records:
            read_smiles(smiles, explicit_hydrogen=False)
        
        assert len(records) == n_records