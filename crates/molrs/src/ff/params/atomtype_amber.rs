//! Antechamber atom-type definition rules — `ATOMTYPE_AMBER.DEF`.
//!
//! DO NOT HAND-EDIT — regenerate with `scripts/gen_param_tables.py`, which emits
//! this table from AmberTools' own `.DAT` / `.DEF` files. That is where the table
//! came FROM; it is not what the table IS — this is ordinary source, not a build
//! artefact.
//!
//! Source: `$AMBERHOME/dat/antechamber/ATOMTYPE_AMBER.DEF` (AmberTools).

use crate::ff::params::{
    AtdRule, AtdTable, AtomPattern, AtomProp, PatternAtom, PropConstraint, PropExpr, PropUnit,
    WildAtom, WildAtomSpec,
};

/// `WILDATOM XX C N O S P`
#[rustfmt::skip]
const WILD_XX: &[WildAtomSpec] = &[
    WildAtomSpec { z: 6, degree: None },
    WildAtomSpec { z: 7, degree: None },
    WildAtomSpec { z: 8, degree: None },
    WildAtomSpec { z: 16, degree: None },
    WildAtomSpec { z: 15, degree: None },
];

/// `WILDATOM XA O S`
#[rustfmt::skip]
const WILD_XA: &[WildAtomSpec] = &[
    WildAtomSpec { z: 8, degree: None },
    WildAtomSpec { z: 16, degree: None },
];

/// `WILDATOM XB N P`
#[rustfmt::skip]
const WILD_XB: &[WildAtomSpec] = &[
    WildAtomSpec { z: 7, degree: None },
    WildAtomSpec { z: 15, degree: None },
];

/// `WILDATOM XC F Cl Br I`
#[rustfmt::skip]
const WILD_XC: &[WildAtomSpec] = &[
    WildAtomSpec { z: 9, degree: None },
    WildAtomSpec { z: 17, degree: None },
    WildAtomSpec { z: 35, degree: None },
    WildAtomSpec { z: 53, degree: None },
];

/// The 4 `WILDATOM` aliases declared by `ATOMTYPE_AMBER.DEF`.
#[rustfmt::skip]
pub const WILDATOMS: &[WildAtom] = &[
    WildAtom { name: "XX", specs: WILD_XX },
    WildAtom { name: "XA", specs: WILD_XA },
    WildAtom { name: "XB", specs: WILD_XB },
    WildAtom { name: "XC", specs: WILD_XC },
];

/// The 65 `ATD` rules of `ATOMTYPE_AMBER.DEF`, in file order.
///
/// Order is significant: the FIRST rule that matches wins — which is why the
/// table's own last row is the constraint-free fall-through (`DU`, or `ANY` in
/// `ATOMTYPE_SYBYL.DEF`). It matches anything nothing above it matched, and
/// antechamber does reach it: `-at amber` types nitromethane's nitro oxygens
/// `DU`. It is a rule of the table, not a fallback the engine invents.
///
/// No rule here carries an `alternate`: `PARMCHK.DAT`'s `equivalent_flag`
/// column describes the GAFF atom-type namespace, and this file is not written
/// in it. Nothing in this table is ever renamed by the 2-colouring pass.
pub const RULES: &[AtdRule] = &[
    AtdRule {
        atom_type: "CT",
        alternate: None,
        residue: "*",
        atomic_number: Some(6),
        degree: Some(4),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "C",
        alternate: None,
        residue: "*",
        atomic_number: Some(6),
        degree: Some(3),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: Some(&[AtomPattern {
            atom: PatternAtom::Wild(WILD_XA),
            degree: Some(1),
            property: None,
            label: None,
            children: &[],
        }]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "CN",
        alternate: None,
        residue: "*",
        atomic_number: Some(6),
        degree: Some(3),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: Some(PropExpr {
            constraints: &[
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::Rg5,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::Rg6,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[
                        PropUnit {
                            count: None,
                            prop: AtomProp::Ar1,
                            relation: None,
                        },
                        PropUnit {
                            count: None,
                            prop: AtomProp::Ar2,
                            relation: None,
                        },
                        PropUnit {
                            count: None,
                            prop: AtomProp::Ar3,
                            relation: None,
                        },
                    ],
                },
            ],
        }),
        environment: Some(&[
            AtomPattern {
                atom: PatternAtom::Element(6),
                degree: Some(3),
                property: None,
                label: None,
                children: &[],
            },
            AtomPattern {
                atom: PatternAtom::Element(6),
                degree: Some(3),
                property: None,
                label: None,
                children: &[],
            },
            AtomPattern {
                atom: PatternAtom::Element(7),
                degree: Some(3),
                property: None,
                label: None,
                children: &[AtomPattern {
                    atom: PatternAtom::Element(1),
                    degree: None,
                    property: None,
                    label: None,
                    children: &[],
                }],
            },
        ]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "CB",
        alternate: None,
        residue: "*",
        atomic_number: Some(6),
        degree: Some(3),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: Some(PropExpr {
            constraints: &[
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::Rg5,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::Rg6,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[
                        PropUnit {
                            count: None,
                            prop: AtomProp::Ar1,
                            relation: None,
                        },
                        PropUnit {
                            count: None,
                            prop: AtomProp::Ar2,
                            relation: None,
                        },
                        PropUnit {
                            count: None,
                            prop: AtomProp::Ar3,
                            relation: None,
                        },
                    ],
                },
            ],
        }),
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "CR",
        alternate: None,
        residue: "*",
        atomic_number: Some(6),
        degree: Some(3),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: Some(PropExpr {
            constraints: &[
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::Rg5,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[
                        PropUnit {
                            count: None,
                            prop: AtomProp::Ar1,
                            relation: None,
                        },
                        PropUnit {
                            count: None,
                            prop: AtomProp::Ar2,
                            relation: None,
                        },
                        PropUnit {
                            count: None,
                            prop: AtomProp::Ar3,
                            relation: None,
                        },
                    ],
                },
            ],
        }),
        environment: Some(&[
            AtomPattern {
                atom: PatternAtom::Element(7),
                degree: Some(3),
                property: None,
                label: None,
                children: &[],
            },
            AtomPattern {
                atom: PatternAtom::Element(7),
                degree: Some(3),
                property: None,
                label: None,
                children: &[],
            },
        ]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "CR",
        alternate: None,
        residue: "*",
        atomic_number: Some(6),
        degree: Some(3),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: Some(PropExpr {
            constraints: &[
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::Rg5,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[
                        PropUnit {
                            count: None,
                            prop: AtomProp::Ar1,
                            relation: None,
                        },
                        PropUnit {
                            count: None,
                            prop: AtomProp::Ar2,
                            relation: None,
                        },
                        PropUnit {
                            count: None,
                            prop: AtomProp::Ar3,
                            relation: None,
                        },
                    ],
                },
            ],
        }),
        environment: Some(&[
            AtomPattern {
                atom: PatternAtom::Element(7),
                degree: Some(2),
                property: None,
                label: None,
                children: &[],
            },
            AtomPattern {
                atom: PatternAtom::Element(7),
                degree: Some(3),
                property: None,
                label: None,
                children: &[AtomPattern {
                    atom: PatternAtom::Element(1),
                    degree: None,
                    property: None,
                    label: None,
                    children: &[],
                }],
            },
        ]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "CK",
        alternate: None,
        residue: "*",
        atomic_number: Some(6),
        degree: Some(3),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: Some(PropExpr {
            constraints: &[
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::Rg5,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[
                        PropUnit {
                            count: None,
                            prop: AtomProp::Ar1,
                            relation: None,
                        },
                        PropUnit {
                            count: None,
                            prop: AtomProp::Ar2,
                            relation: None,
                        },
                        PropUnit {
                            count: None,
                            prop: AtomProp::Ar3,
                            relation: None,
                        },
                    ],
                },
            ],
        }),
        environment: Some(&[
            AtomPattern {
                atom: PatternAtom::Element(7),
                degree: Some(2),
                property: None,
                label: None,
                children: &[],
            },
            AtomPattern {
                atom: PatternAtom::Element(7),
                degree: Some(3),
                property: None,
                label: None,
                children: &[],
            },
        ]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "CC",
        alternate: None,
        residue: "*",
        atomic_number: Some(6),
        degree: Some(3),
        hydrogen_count: Some(0),
        ewd_count: None,
        atom_property: Some(PropExpr {
            constraints: &[
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::Rg5,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[
                        PropUnit {
                            count: None,
                            prop: AtomProp::Ar1,
                            relation: None,
                        },
                        PropUnit {
                            count: None,
                            prop: AtomProp::Ar2,
                            relation: None,
                        },
                        PropUnit {
                            count: None,
                            prop: AtomProp::Ar3,
                            relation: None,
                        },
                    ],
                },
            ],
        }),
        environment: Some(&[
            AtomPattern {
                atom: PatternAtom::Element(6),
                degree: Some(3),
                property: None,
                label: None,
                children: &[],
            },
            AtomPattern {
                atom: PatternAtom::Element(7),
                degree: Some(3),
                property: None,
                label: None,
                children: &[
                    AtomPattern {
                        atom: PatternAtom::Element(6),
                        degree: Some(3),
                        property: None,
                        label: None,
                        children: &[],
                    },
                    AtomPattern {
                        atom: PatternAtom::Element(1),
                        degree: None,
                        property: None,
                        label: None,
                        children: &[],
                    },
                ],
            },
        ]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "CC",
        alternate: None,
        residue: "*",
        atomic_number: Some(6),
        degree: Some(3),
        hydrogen_count: Some(0),
        ewd_count: None,
        atom_property: Some(PropExpr {
            constraints: &[
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::Rg5,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[
                        PropUnit {
                            count: None,
                            prop: AtomProp::Ar1,
                            relation: None,
                        },
                        PropUnit {
                            count: None,
                            prop: AtomProp::Ar2,
                            relation: None,
                        },
                        PropUnit {
                            count: None,
                            prop: AtomProp::Ar3,
                            relation: None,
                        },
                    ],
                },
            ],
        }),
        environment: Some(&[
            AtomPattern {
                atom: PatternAtom::Element(6),
                degree: Some(3),
                property: None,
                label: None,
                children: &[],
            },
            AtomPattern {
                atom: PatternAtom::Element(7),
                degree: Some(2),
                property: None,
                label: None,
                children: &[AtomPattern {
                    atom: PatternAtom::Element(6),
                    degree: Some(3),
                    property: None,
                    label: None,
                    children: &[],
                }],
            },
        ]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "CW",
        alternate: None,
        residue: "*",
        atomic_number: Some(6),
        degree: Some(3),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: Some(PropExpr {
            constraints: &[
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::Rg5,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[
                        PropUnit {
                            count: None,
                            prop: AtomProp::Ar1,
                            relation: None,
                        },
                        PropUnit {
                            count: None,
                            prop: AtomProp::Ar2,
                            relation: None,
                        },
                        PropUnit {
                            count: None,
                            prop: AtomProp::Ar3,
                            relation: None,
                        },
                    ],
                },
            ],
        }),
        environment: Some(&[
            AtomPattern {
                atom: PatternAtom::Element(6),
                degree: Some(3),
                property: None,
                label: None,
                children: &[],
            },
            AtomPattern {
                atom: PatternAtom::Element(7),
                degree: Some(3),
                property: None,
                label: None,
                children: &[AtomPattern {
                    atom: PatternAtom::Element(1),
                    degree: None,
                    property: None,
                    label: None,
                    children: &[],
                }],
            },
        ]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "CV",
        alternate: None,
        residue: "*",
        atomic_number: Some(6),
        degree: Some(3),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: Some(PropExpr {
            constraints: &[
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::Rg5,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[
                        PropUnit {
                            count: None,
                            prop: AtomProp::Ar1,
                            relation: None,
                        },
                        PropUnit {
                            count: None,
                            prop: AtomProp::Ar2,
                            relation: None,
                        },
                        PropUnit {
                            count: None,
                            prop: AtomProp::Ar3,
                            relation: None,
                        },
                    ],
                },
            ],
        }),
        environment: Some(&[
            AtomPattern {
                atom: PatternAtom::Element(6),
                degree: Some(3),
                property: None,
                label: None,
                children: &[],
            },
            AtomPattern {
                atom: PatternAtom::Element(7),
                degree: Some(2),
                property: None,
                label: None,
                children: &[],
            },
        ]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "C*",
        alternate: None,
        residue: "*",
        atomic_number: Some(6),
        degree: Some(3),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: Some(PropExpr {
            constraints: &[
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::Rg5,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[
                        PropUnit {
                            count: None,
                            prop: AtomProp::Ar1,
                            relation: None,
                        },
                        PropUnit {
                            count: None,
                            prop: AtomProp::Ar2,
                            relation: None,
                        },
                        PropUnit {
                            count: None,
                            prop: AtomProp::Ar3,
                            relation: None,
                        },
                    ],
                },
            ],
        }),
        environment: Some(&[
            AtomPattern {
                atom: PatternAtom::Element(6),
                degree: Some(3),
                property: None,
                label: None,
                children: &[],
            },
            AtomPattern {
                atom: PatternAtom::Element(6),
                degree: Some(3),
                property: None,
                label: None,
                children: &[],
            },
        ]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "CQ",
        alternate: None,
        residue: "*",
        atomic_number: Some(6),
        degree: Some(3),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: Some(PropExpr {
            constraints: &[
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::Rg6,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[
                        PropUnit {
                            count: None,
                            prop: AtomProp::Ar1,
                            relation: None,
                        },
                        PropUnit {
                            count: None,
                            prop: AtomProp::Ar2,
                            relation: None,
                        },
                        PropUnit {
                            count: None,
                            prop: AtomProp::Ar3,
                            relation: None,
                        },
                    ],
                },
            ],
        }),
        environment: Some(&[
            AtomPattern {
                atom: PatternAtom::Element(7),
                degree: Some(2),
                property: None,
                label: None,
                children: &[],
            },
            AtomPattern {
                atom: PatternAtom::Element(7),
                degree: Some(2),
                property: None,
                label: None,
                children: &[],
            },
        ]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "CM",
        alternate: None,
        residue: "*",
        atomic_number: Some(6),
        degree: Some(3),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: Some(PropExpr {
            constraints: &[
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::Rg6,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[
                        PropUnit {
                            count: None,
                            prop: AtomProp::Ar1,
                            relation: None,
                        },
                        PropUnit {
                            count: None,
                            prop: AtomProp::Ar2,
                            relation: None,
                        },
                        PropUnit {
                            count: None,
                            prop: AtomProp::Ar3,
                            relation: None,
                        },
                    ],
                },
            ],
        }),
        environment: Some(&[
            AtomPattern {
                atom: PatternAtom::Element(6),
                degree: Some(3),
                property: None,
                label: None,
                children: &[],
            },
            AtomPattern {
                atom: PatternAtom::Element(6),
                degree: Some(3),
                property: None,
                label: None,
                children: &[AtomPattern {
                    atom: PatternAtom::Element(7),
                    degree: Some(2),
                    property: None,
                    label: None,
                    children: &[AtomPattern {
                        atom: PatternAtom::Element(6),
                        degree: Some(3),
                        property: None,
                        label: None,
                        children: &[AtomPattern {
                            atom: PatternAtom::Element(7),
                            degree: Some(3),
                            property: None,
                            label: None,
                            children: &[AtomPattern {
                                atom: PatternAtom::Element(6),
                                degree: Some(3),
                                property: None,
                                label: None,
                                children: &[],
                            }],
                        }],
                    }],
                }],
            },
        ]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "CM",
        alternate: None,
        residue: "*",
        atomic_number: Some(6),
        degree: Some(3),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: Some(PropExpr {
            constraints: &[
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::Rg6,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[
                        PropUnit {
                            count: None,
                            prop: AtomProp::Ar1,
                            relation: None,
                        },
                        PropUnit {
                            count: None,
                            prop: AtomProp::Ar2,
                            relation: None,
                        },
                        PropUnit {
                            count: None,
                            prop: AtomProp::Ar3,
                            relation: None,
                        },
                    ],
                },
            ],
        }),
        environment: Some(&[
            AtomPattern {
                atom: PatternAtom::Element(6),
                degree: Some(3),
                property: None,
                label: None,
                children: &[],
            },
            AtomPattern {
                atom: PatternAtom::Element(6),
                degree: Some(3),
                property: None,
                label: None,
                children: &[AtomPattern {
                    atom: PatternAtom::Element(7),
                    degree: Some(3),
                    property: None,
                    label: None,
                    children: &[AtomPattern {
                        atom: PatternAtom::Element(6),
                        degree: Some(3),
                        property: None,
                        label: None,
                        children: &[AtomPattern {
                            atom: PatternAtom::Element(7),
                            degree: Some(3),
                            property: None,
                            label: None,
                            children: &[AtomPattern {
                                atom: PatternAtom::Element(6),
                                degree: Some(3),
                                property: None,
                                label: None,
                                children: &[],
                            }],
                        }],
                    }],
                }],
            },
        ]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "CM",
        alternate: None,
        residue: "*",
        atomic_number: Some(6),
        degree: Some(3),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: Some(PropExpr {
            constraints: &[
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::Rg6,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[
                        PropUnit {
                            count: None,
                            prop: AtomProp::Ar1,
                            relation: None,
                        },
                        PropUnit {
                            count: None,
                            prop: AtomProp::Ar2,
                            relation: None,
                        },
                        PropUnit {
                            count: None,
                            prop: AtomProp::Ar3,
                            relation: None,
                        },
                    ],
                },
            ],
        }),
        environment: Some(&[
            AtomPattern {
                atom: PatternAtom::Element(6),
                degree: Some(3),
                property: None,
                label: None,
                children: &[],
            },
            AtomPattern {
                atom: PatternAtom::Element(7),
                degree: Some(3),
                property: None,
                label: None,
                children: &[AtomPattern {
                    atom: PatternAtom::Element(6),
                    degree: Some(3),
                    property: None,
                    label: None,
                    children: &[AtomPattern {
                        atom: PatternAtom::Element(7),
                        degree: Some(2),
                        property: None,
                        label: None,
                        children: &[AtomPattern {
                            atom: PatternAtom::Element(6),
                            degree: Some(3),
                            property: None,
                            label: None,
                            children: &[AtomPattern {
                                atom: PatternAtom::Element(6),
                                degree: Some(3),
                                property: None,
                                label: None,
                                children: &[],
                            }],
                        }],
                    }],
                }],
            },
        ]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "CM",
        alternate: None,
        residue: "*",
        atomic_number: Some(6),
        degree: Some(3),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: Some(PropExpr {
            constraints: &[
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::Rg6,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[
                        PropUnit {
                            count: None,
                            prop: AtomProp::Ar1,
                            relation: None,
                        },
                        PropUnit {
                            count: None,
                            prop: AtomProp::Ar2,
                            relation: None,
                        },
                        PropUnit {
                            count: None,
                            prop: AtomProp::Ar3,
                            relation: None,
                        },
                    ],
                },
            ],
        }),
        environment: Some(&[
            AtomPattern {
                atom: PatternAtom::Element(7),
                degree: Some(3),
                property: None,
                label: None,
                children: &[],
            },
            AtomPattern {
                atom: PatternAtom::Element(6),
                degree: Some(3),
                property: None,
                label: None,
                children: &[AtomPattern {
                    atom: PatternAtom::Element(6),
                    degree: Some(3),
                    property: None,
                    label: None,
                    children: &[AtomPattern {
                        atom: PatternAtom::Element(7),
                        degree: Some(3),
                        property: None,
                        label: None,
                        children: &[AtomPattern {
                            atom: PatternAtom::Element(6),
                            degree: Some(3),
                            property: None,
                            label: None,
                            children: &[AtomPattern {
                                atom: PatternAtom::Element(7),
                                degree: Some(3),
                                property: None,
                                label: None,
                                children: &[],
                            }],
                        }],
                    }],
                }],
            },
        ]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "CA",
        alternate: None,
        residue: "*",
        atomic_number: Some(6),
        degree: Some(3),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: Some(PropExpr {
            constraints: &[PropConstraint {
                units: &[
                    PropUnit {
                        count: None,
                        prop: AtomProp::Ar1,
                        relation: None,
                    },
                    PropUnit {
                        count: None,
                        prop: AtomProp::Ar2,
                        relation: None,
                    },
                    PropUnit {
                        count: None,
                        prop: AtomProp::Ar3,
                        relation: None,
                    },
                ],
            }],
        }),
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "CA",
        alternate: None,
        residue: "*",
        atomic_number: Some(6),
        degree: Some(3),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: Some(&[
            AtomPattern {
                atom: PatternAtom::Element(7),
                degree: Some(3),
                property: None,
                label: None,
                children: &[],
            },
            AtomPattern {
                atom: PatternAtom::Element(7),
                degree: Some(3),
                property: None,
                label: None,
                children: &[],
            },
            AtomPattern {
                atom: PatternAtom::Element(7),
                degree: Some(3),
                property: None,
                label: None,
                children: &[],
            },
        ]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "CD",
        alternate: None,
        residue: "*",
        atomic_number: Some(6),
        degree: Some(3),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: Some(&[
            AtomPattern {
                atom: PatternAtom::Element(6),
                degree: Some(3),
                property: None,
                label: None,
                children: &[],
            },
            AtomPattern {
                atom: PatternAtom::Element(6),
                degree: Some(3),
                property: None,
                label: None,
                children: &[],
            },
        ]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "CM",
        alternate: None,
        residue: "*",
        atomic_number: Some(6),
        degree: Some(3),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "CZ",
        alternate: None,
        residue: "*",
        atomic_number: Some(6),
        degree: Some(2),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "H",
        alternate: None,
        residue: "*",
        atomic_number: Some(1),
        degree: Some(1),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: Some(&[AtomPattern {
            atom: PatternAtom::Element(7),
            degree: None,
            property: None,
            label: None,
            children: &[],
        }]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "HO",
        alternate: None,
        residue: "*",
        atomic_number: Some(1),
        degree: Some(1),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: Some(&[AtomPattern {
            atom: PatternAtom::Element(8),
            degree: None,
            property: None,
            label: None,
            children: &[],
        }]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "HS",
        alternate: None,
        residue: "*",
        atomic_number: Some(1),
        degree: Some(1),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: Some(&[AtomPattern {
            atom: PatternAtom::Element(16),
            degree: None,
            property: None,
            label: None,
            children: &[],
        }]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "HP",
        alternate: None,
        residue: "*",
        atomic_number: Some(1),
        degree: Some(1),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: Some(&[AtomPattern {
            atom: PatternAtom::Element(6),
            degree: None,
            property: None,
            label: None,
            children: &[AtomPattern {
                atom: PatternAtom::Element(7),
                degree: Some(4),
                property: None,
                label: None,
                children: &[],
            }],
        }]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "HW",
        alternate: None,
        residue: "*",
        atomic_number: Some(1),
        degree: Some(1),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: Some(&[AtomPattern {
            atom: PatternAtom::Element(8),
            degree: None,
            property: None,
            label: None,
            children: &[AtomPattern {
                atom: PatternAtom::Element(1),
                degree: Some(1),
                property: None,
                label: None,
                children: &[],
            }],
        }]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "H3",
        alternate: None,
        residue: "*",
        atomic_number: Some(1),
        degree: Some(1),
        hydrogen_count: None,
        ewd_count: Some(3),
        atom_property: None,
        environment: Some(&[AtomPattern {
            atom: PatternAtom::Element(6),
            degree: Some(4),
            property: None,
            label: None,
            children: &[],
        }]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "H2",
        alternate: None,
        residue: "*",
        atomic_number: Some(1),
        degree: Some(1),
        hydrogen_count: None,
        ewd_count: Some(2),
        atom_property: None,
        environment: Some(&[AtomPattern {
            atom: PatternAtom::Element(6),
            degree: Some(4),
            property: None,
            label: None,
            children: &[],
        }]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "H1",
        alternate: None,
        residue: "*",
        atomic_number: Some(1),
        degree: Some(1),
        hydrogen_count: None,
        ewd_count: Some(1),
        atom_property: None,
        environment: Some(&[AtomPattern {
            atom: PatternAtom::Element(6),
            degree: Some(4),
            property: None,
            label: None,
            children: &[],
        }]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "HC",
        alternate: None,
        residue: "*",
        atomic_number: Some(1),
        degree: Some(1),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: Some(&[AtomPattern {
            atom: PatternAtom::Element(6),
            degree: Some(4),
            property: None,
            label: None,
            children: &[],
        }]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "H5",
        alternate: None,
        residue: "*",
        atomic_number: Some(1),
        degree: Some(1),
        hydrogen_count: None,
        ewd_count: Some(2),
        atom_property: None,
        environment: Some(&[AtomPattern {
            atom: PatternAtom::Wild(WILD_XX),
            degree: None,
            property: Some(PropExpr {
                constraints: &[PropConstraint {
                    units: &[
                        PropUnit {
                            count: None,
                            prop: AtomProp::Ar1,
                            relation: None,
                        },
                        PropUnit {
                            count: None,
                            prop: AtomProp::Ar2,
                            relation: None,
                        },
                        PropUnit {
                            count: None,
                            prop: AtomProp::Ar3,
                            relation: None,
                        },
                    ],
                }],
            }),
            label: None,
            children: &[],
        }]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "H4",
        alternate: None,
        residue: "*",
        atomic_number: Some(1),
        degree: Some(1),
        hydrogen_count: None,
        ewd_count: Some(1),
        atom_property: None,
        environment: Some(&[AtomPattern {
            atom: PatternAtom::Wild(WILD_XX),
            degree: None,
            property: Some(PropExpr {
                constraints: &[PropConstraint {
                    units: &[
                        PropUnit {
                            count: None,
                            prop: AtomProp::Ar1,
                            relation: None,
                        },
                        PropUnit {
                            count: None,
                            prop: AtomProp::Ar2,
                            relation: None,
                        },
                        PropUnit {
                            count: None,
                            prop: AtomProp::Ar3,
                            relation: None,
                        },
                    ],
                }],
            }),
            label: None,
            children: &[],
        }]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "HA",
        alternate: None,
        residue: "*",
        atomic_number: Some(1),
        degree: Some(1),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: Some(&[AtomPattern {
            atom: PatternAtom::Wild(WILD_XX),
            degree: None,
            property: Some(PropExpr {
                constraints: &[PropConstraint {
                    units: &[
                        PropUnit {
                            count: None,
                            prop: AtomProp::Ar1,
                            relation: None,
                        },
                        PropUnit {
                            count: None,
                            prop: AtomProp::Ar2,
                            relation: None,
                        },
                        PropUnit {
                            count: None,
                            prop: AtomProp::Ar3,
                            relation: None,
                        },
                    ],
                }],
            }),
            label: None,
            children: &[],
        }]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "HA",
        alternate: None,
        residue: "*",
        atomic_number: Some(1),
        degree: Some(1),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "F",
        alternate: None,
        residue: "*",
        atomic_number: Some(9),
        degree: Some(1),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "Cl",
        alternate: None,
        residue: "*",
        atomic_number: Some(17),
        degree: Some(1),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "Br",
        alternate: None,
        residue: "*",
        atomic_number: Some(35),
        degree: Some(1),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "I",
        alternate: None,
        residue: "*",
        atomic_number: Some(53),
        degree: Some(1),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "P",
        alternate: None,
        residue: "*",
        atomic_number: Some(15),
        degree: None,
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "N1",
        alternate: None,
        residue: "*",
        atomic_number: Some(7),
        degree: Some(1),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "NB",
        alternate: None,
        residue: "*",
        atomic_number: Some(7),
        degree: Some(2),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: Some(PropExpr {
            constraints: &[
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::Rg5,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[
                        PropUnit {
                            count: None,
                            prop: AtomProp::Ar1,
                            relation: None,
                        },
                        PropUnit {
                            count: None,
                            prop: AtomProp::Ar2,
                            relation: None,
                        },
                        PropUnit {
                            count: None,
                            prop: AtomProp::Ar3,
                            relation: None,
                        },
                    ],
                },
            ],
        }),
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "NC",
        alternate: None,
        residue: "*",
        atomic_number: Some(7),
        degree: Some(2),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: Some(PropExpr {
            constraints: &[
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::Rg6,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[
                        PropUnit {
                            count: None,
                            prop: AtomProp::Ar1,
                            relation: None,
                        },
                        PropUnit {
                            count: None,
                            prop: AtomProp::Ar2,
                            relation: None,
                        },
                        PropUnit {
                            count: None,
                            prop: AtomProp::Ar3,
                            relation: None,
                        },
                    ],
                },
            ],
        }),
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "N2",
        alternate: None,
        residue: "*",
        atomic_number: Some(7),
        degree: None,
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: Some(&[AtomPattern {
            atom: PatternAtom::Element(6),
            degree: Some(3),
            property: None,
            label: None,
            children: &[
                AtomPattern {
                    atom: PatternAtom::Element(7),
                    degree: Some(3),
                    property: None,
                    label: None,
                    children: &[],
                },
                AtomPattern {
                    atom: PatternAtom::Element(7),
                    degree: Some(3),
                    property: None,
                    label: None,
                    children: &[],
                },
            ],
        }]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "N2",
        alternate: None,
        residue: "*",
        atomic_number: Some(7),
        degree: None,
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: Some(&[AtomPattern {
            atom: PatternAtom::Element(6),
            degree: Some(3),
            property: None,
            label: None,
            children: &[
                AtomPattern {
                    atom: PatternAtom::Element(7),
                    degree: Some(3),
                    property: None,
                    label: None,
                    children: &[],
                },
                AtomPattern {
                    atom: PatternAtom::Element(7),
                    degree: Some(3),
                    property: None,
                    label: None,
                    children: &[],
                },
            ],
        }]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "NO",
        alternate: None,
        residue: "*",
        atomic_number: Some(7),
        degree: Some(3),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: Some(&[
            AtomPattern {
                atom: PatternAtom::Element(8),
                degree: Some(1),
                property: None,
                label: None,
                children: &[],
            },
            AtomPattern {
                atom: PatternAtom::Element(8),
                degree: Some(1),
                property: None,
                label: None,
                children: &[],
            },
        ]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "NA",
        alternate: None,
        residue: "*",
        atomic_number: Some(7),
        degree: Some(3),
        hydrogen_count: Some(1),
        ewd_count: None,
        atom_property: Some(PropExpr {
            constraints: &[
                PropConstraint {
                    units: &[
                        PropUnit {
                            count: None,
                            prop: AtomProp::Rg5,
                            relation: None,
                        },
                        PropUnit {
                            count: None,
                            prop: AtomProp::Rg6,
                            relation: None,
                        },
                    ],
                },
                PropConstraint {
                    units: &[
                        PropUnit {
                            count: None,
                            prop: AtomProp::Ar1,
                            relation: None,
                        },
                        PropUnit {
                            count: None,
                            prop: AtomProp::Ar2,
                            relation: None,
                        },
                        PropUnit {
                            count: None,
                            prop: AtomProp::Ar3,
                            relation: None,
                        },
                    ],
                },
            ],
        }),
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "N2",
        alternate: None,
        residue: "*",
        atomic_number: Some(7),
        degree: Some(3),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: Some(PropExpr {
            constraints: &[PropConstraint {
                units: &[PropUnit {
                    count: None,
                    prop: AtomProp::Nr,
                    relation: None,
                }],
            }],
        }),
        environment: Some(&[AtomPattern {
            atom: PatternAtom::Wild(WILD_XX),
            degree: None,
            property: Some(PropExpr {
                constraints: &[PropConstraint {
                    units: &[
                        PropUnit {
                            count: None,
                            prop: AtomProp::Ar1,
                            relation: None,
                        },
                        PropUnit {
                            count: None,
                            prop: AtomProp::Ar2,
                            relation: None,
                        },
                        PropUnit {
                            count: None,
                            prop: AtomProp::Ar3,
                            relation: None,
                        },
                    ],
                }],
            }),
            label: None,
            children: &[],
        }]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "N*",
        alternate: None,
        residue: "*",
        atomic_number: Some(7),
        degree: Some(3),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: Some(PropExpr {
            constraints: &[PropConstraint {
                units: &[
                    PropUnit {
                        count: None,
                        prop: AtomProp::Ar1,
                        relation: None,
                    },
                    PropUnit {
                        count: None,
                        prop: AtomProp::Ar2,
                        relation: None,
                    },
                    PropUnit {
                        count: None,
                        prop: AtomProp::Ar3,
                        relation: None,
                    },
                ],
            }],
        }),
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "N",
        alternate: None,
        residue: "*",
        atomic_number: Some(7),
        degree: Some(3),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: Some(&[AtomPattern {
            atom: PatternAtom::Element(6),
            degree: Some(3),
            property: None,
            label: None,
            children: &[AtomPattern {
                atom: PatternAtom::Wild(WILD_XA),
                degree: Some(1),
                property: None,
                label: None,
                children: &[],
            }],
        }]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "NT",
        alternate: None,
        residue: "*",
        atomic_number: Some(7),
        degree: Some(3),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "N3",
        alternate: None,
        residue: "*",
        atomic_number: Some(7),
        degree: Some(4),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "O2",
        alternate: None,
        residue: "*",
        atomic_number: Some(8),
        degree: Some(1),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: Some(&[AtomPattern {
            atom: PatternAtom::Element(6),
            degree: None,
            property: None,
            label: None,
            children: &[AtomPattern {
                atom: PatternAtom::Element(8),
                degree: Some(1),
                property: None,
                label: None,
                children: &[],
            }],
        }]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "O2",
        alternate: None,
        residue: "*",
        atomic_number: Some(8),
        degree: Some(1),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: Some(&[AtomPattern {
            atom: PatternAtom::Element(15),
            degree: None,
            property: None,
            label: None,
            children: &[],
        }]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "O",
        alternate: None,
        residue: "*",
        atomic_number: Some(8),
        degree: Some(1),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: Some(&[AtomPattern {
            atom: PatternAtom::Element(6),
            degree: None,
            property: None,
            label: None,
            children: &[],
        }]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "O",
        alternate: None,
        residue: "*",
        atomic_number: Some(8),
        degree: Some(1),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: Some(&[AtomPattern {
            atom: PatternAtom::Element(16),
            degree: None,
            property: None,
            label: None,
            children: &[],
        }]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "OH",
        alternate: None,
        residue: "*",
        atomic_number: Some(8),
        degree: Some(2),
        hydrogen_count: Some(1),
        ewd_count: None,
        atom_property: None,
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "OW",
        alternate: None,
        residue: "*",
        atomic_number: Some(8),
        degree: Some(2),
        hydrogen_count: Some(2),
        ewd_count: None,
        atom_property: None,
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "OS",
        alternate: None,
        residue: "*",
        atomic_number: Some(8),
        degree: Some(2),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "SH",
        alternate: None,
        residue: "*",
        atomic_number: Some(16),
        degree: Some(2),
        hydrogen_count: Some(1),
        ewd_count: None,
        atom_property: None,
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "SH",
        alternate: None,
        residue: "*",
        atomic_number: Some(16),
        degree: Some(2),
        hydrogen_count: Some(2),
        ewd_count: None,
        atom_property: None,
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "S",
        alternate: None,
        residue: "*",
        atomic_number: Some(16),
        degree: Some(2),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "SO",
        alternate: None,
        residue: "*",
        atomic_number: Some(16),
        degree: Some(4),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "LP",
        alternate: None,
        residue: "*",
        atomic_number: Some(0),
        degree: None,
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "DU",
        alternate: None,
        residue: "*",
        atomic_number: None,
        degree: None,
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: None,
        environment_bonds: None,
    },
];

/// `ATOMTYPE_AMBER.DEF` as one typed table.
pub const ATOMTYPE_AMBER: AtdTable = AtdTable {
    name: "ATOMTYPE_AMBER.DEF",
    wildatoms: WILDATOMS,
    rules: RULES,
};
