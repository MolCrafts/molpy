//! Antechamber atom-type definition rules — `ATOMTYPE_GFF.DEF`.
//!
//! DO NOT HAND-EDIT — regenerate with `scripts/gen_param_tables.py`, which emits
//! this table from AmberTools' own `.DAT` / `.DEF` files. That is where the table
//! came FROM; it is not what the table IS — this is ordinary source, not a build
//! artefact.
//!
//! Source: `$AMBERHOME/dat/antechamber/ATOMTYPE_GFF.DEF` (AmberTools).

use crate::ff::params::{
    Alternate, AlternatePass, AtdRule, AtdTable, AtomPattern, AtomProp, PatternAtom,
    PropConstraint, PropExpr, PropRelation, PropUnit, WildAtom, WildAtomSpec,
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

/// `WILDATOM XD S P`
#[rustfmt::skip]
const WILD_XD: &[WildAtomSpec] = &[
    WildAtomSpec { z: 16, degree: None },
    WildAtomSpec { z: 15, degree: None },
];

/// The 5 `WILDATOM` aliases declared by `ATOMTYPE_GFF.DEF`.
#[rustfmt::skip]
pub const WILDATOMS: &[WildAtom] = &[
    WildAtom { name: "XX", specs: WILD_XX },
    WildAtom { name: "XA", specs: WILD_XA },
    WildAtom { name: "XB", specs: WILD_XB },
    WildAtom { name: "XC", specs: WILD_XC },
    WildAtom { name: "XD", specs: WILD_XD },
];

/// The 296 `ATD` rules of `ATOMTYPE_GFF.DEF`, in file order.
///
/// Order is significant: the FIRST rule that matches wins — which is why the
/// table's own last row is the constraint-free fall-through (`DU`, or `ANY` in
/// `ATOMTYPE_SYBYL.DEF`). It matches anything nothing above it matched, and
/// antechamber does reach it: `-at amber` types nitromethane's nitro oxygens
/// `DU`. It is a rule of the table, not a fallback the engine invents.
///
/// 88 of these rules carry an `alternate`: the phase-2 name
/// `PARMCHK.DAT` pairs their atom type with. The rule emits the phase-1 name;
/// the typifier's 2-colouring passes rename one colour of each conjugated
/// (or biphenyl-bridge) system to the alternate, which is the only way a type
/// no ATD row declares (`cd`, `cq`) is ever assigned.
pub const RULES: &[AtdRule] = &[
    AtdRule {
        atom_type: "cx",
        alternate: None,
        residue: "*",
        atomic_number: Some(6),
        degree: Some(4),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: Some(PropExpr {
            constraints: &[PropConstraint {
                units: &[PropUnit {
                    count: None,
                    prop: AtomProp::Rg3,
                    relation: None,
                }],
            }],
        }),
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "cy",
        alternate: None,
        residue: "*",
        atomic_number: Some(6),
        degree: Some(4),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: Some(PropExpr {
            constraints: &[PropConstraint {
                units: &[PropUnit {
                    count: None,
                    prop: AtomProp::Rg4,
                    relation: None,
                }],
            }],
        }),
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "c3",
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
        atom_type: "c",
        alternate: None,
        residue: "*",
        atomic_number: Some(6),
        degree: Some(3),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: Some(PropExpr {
            constraints: &[PropConstraint {
                units: &[PropUnit {
                    count: Some(2),
                    prop: AtomProp::Dl,
                    relation: None,
                }],
            }],
        }),
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
        atom_type: "c",
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
                        count: Some(1),
                        prop: AtomProp::DbStrict,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[PropUnit {
                        count: Some(0),
                        prop: AtomProp::Dl,
                        relation: None,
                    }],
                },
            ],
        }),
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
        atom_type: "c",
        alternate: None,
        residue: "*",
        atomic_number: Some(6),
        degree: Some(3),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: Some(PropExpr {
            constraints: &[PropConstraint {
                units: &[PropUnit {
                    count: Some(3),
                    prop: AtomProp::SbAny,
                    relation: None,
                }],
            }],
        }),
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
        atom_type: "cz",
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
        atom_type: "cp",
        alternate: Some(Alternate {
            atom_type: "cq",
            pass: AlternatePass::Bridge,
        }),
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
                        prop: AtomProp::Ar1,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[PropUnit {
                        count: Some(1),
                        prop: AtomProp::Rg6,
                        relation: None,
                    }],
                },
            ],
        }),
        environment: Some(&[
            AtomPattern {
                atom: PatternAtom::Wild(WILD_XX),
                degree: None,
                property: Some(PropExpr {
                    constraints: &[PropConstraint {
                        units: &[PropUnit {
                            count: None,
                            prop: AtomProp::Ar1,
                            relation: None,
                        }],
                    }],
                }),
                label: None,
                children: &[],
            },
            AtomPattern {
                atom: PatternAtom::Wild(WILD_XX),
                degree: None,
                property: Some(PropExpr {
                    constraints: &[PropConstraint {
                        units: &[PropUnit {
                            count: None,
                            prop: AtomProp::Ar1,
                            relation: None,
                        }],
                    }],
                }),
                label: None,
                children: &[],
            },
            AtomPattern {
                atom: PatternAtom::Wild(WILD_XX),
                degree: None,
                property: Some(PropExpr {
                    constraints: &[PropConstraint {
                        units: &[PropUnit {
                            count: None,
                            prop: AtomProp::Ar1,
                            relation: None,
                        }],
                    }],
                }),
                label: None,
                children: &[],
            },
        ]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "ca",
        alternate: None,
        residue: "*",
        atomic_number: Some(6),
        degree: Some(3),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: Some(PropExpr {
            constraints: &[PropConstraint {
                units: &[PropUnit {
                    count: None,
                    prop: AtomProp::Ar1,
                    relation: None,
                }],
            }],
        }),
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "cc",
        alternate: Some(Alternate {
            atom_type: "cd",
            pass: AlternatePass::Conjugated,
        }),
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
                        prop: AtomProp::SbAny,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::DbAny,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::Ar2,
                        relation: None,
                    }],
                },
            ],
        }),
        environment: Some(&[AtomPattern {
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
        }]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "cc",
        alternate: Some(Alternate {
            atom_type: "cd",
            pass: AlternatePass::Conjugated,
        }),
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
                        prop: AtomProp::SbAny,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::DbAny,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::Ar2,
                        relation: None,
                    }],
                },
            ],
        }),
        environment: Some(&[AtomPattern {
            atom: PatternAtom::Element(6),
            degree: Some(3),
            property: None,
            label: None,
            children: &[AtomPattern {
                atom: PatternAtom::Element(6),
                degree: Some(2),
                property: None,
                label: None,
                children: &[],
            }],
        }]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "cc",
        alternate: Some(Alternate {
            atom_type: "cd",
            pass: AlternatePass::Conjugated,
        }),
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
                        prop: AtomProp::SbAny,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::DbAny,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::Ar2,
                        relation: None,
                    }],
                },
            ],
        }),
        environment: Some(&[AtomPattern {
            atom: PatternAtom::Element(6),
            degree: Some(3),
            property: None,
            label: None,
            children: &[AtomPattern {
                atom: PatternAtom::Wild(WILD_XB),
                degree: Some(2),
                property: None,
                label: None,
                children: &[],
            }],
        }]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "cc",
        alternate: Some(Alternate {
            atom_type: "cd",
            pass: AlternatePass::Conjugated,
        }),
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
                        prop: AtomProp::SbAny,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::DbAny,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::Ar2,
                        relation: None,
                    }],
                },
            ],
        }),
        environment: Some(&[AtomPattern {
            atom: PatternAtom::Wild(WILD_XB),
            degree: Some(2),
            property: None,
            label: None,
            children: &[AtomPattern {
                atom: PatternAtom::Wild(WILD_XB),
                degree: Some(2),
                property: None,
                label: None,
                children: &[],
            }],
        }]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "cc",
        alternate: Some(Alternate {
            atom_type: "cd",
            pass: AlternatePass::Conjugated,
        }),
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
                        prop: AtomProp::SbAny,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::DbAny,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::Ar2,
                        relation: None,
                    }],
                },
            ],
        }),
        environment: Some(&[AtomPattern {
            atom: PatternAtom::Wild(WILD_XB),
            degree: Some(2),
            property: None,
            label: None,
            children: &[AtomPattern {
                atom: PatternAtom::Element(6),
                degree: Some(2),
                property: None,
                label: None,
                children: &[],
            }],
        }]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "cc",
        alternate: Some(Alternate {
            atom_type: "cd",
            pass: AlternatePass::Conjugated,
        }),
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
                        prop: AtomProp::SbAny,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::DbAny,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::Ar2,
                        relation: None,
                    }],
                },
            ],
        }),
        environment: Some(&[AtomPattern {
            atom: PatternAtom::Wild(WILD_XB),
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
        }]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "cc",
        alternate: Some(Alternate {
            atom_type: "cd",
            pass: AlternatePass::Conjugated,
        }),
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
                        prop: AtomProp::SbAny,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::DbAny,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::Ar2,
                        relation: None,
                    }],
                },
            ],
        }),
        environment: Some(&[AtomPattern {
            atom: PatternAtom::Element(6),
            degree: Some(3),
            property: Some(PropExpr {
                constraints: &[PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::SbAny,
                        relation: Some(PropRelation::BondedToPrev),
                    }],
                }],
            }),
            label: None,
            children: &[],
        }]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "cc",
        alternate: Some(Alternate {
            atom_type: "cd",
            pass: AlternatePass::Conjugated,
        }),
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
                        prop: AtomProp::SbAny,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::DbAny,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::Ar2,
                        relation: None,
                    }],
                },
            ],
        }),
        environment: Some(&[AtomPattern {
            atom: PatternAtom::Wild(WILD_XB),
            degree: Some(2),
            property: Some(PropExpr {
                constraints: &[PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::SbAny,
                        relation: Some(PropRelation::BondedToPrev),
                    }],
                }],
            }),
            label: None,
            children: &[],
        }]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "cc",
        alternate: Some(Alternate {
            atom_type: "cd",
            pass: AlternatePass::Conjugated,
        }),
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
                        prop: AtomProp::SbAny,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::DbAny,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::Ar2,
                        relation: None,
                    }],
                },
            ],
        }),
        environment: Some(&[AtomPattern {
            atom: PatternAtom::Wild(WILD_XD),
            degree: Some(3),
            property: Some(PropExpr {
                constraints: &[
                    PropConstraint {
                        units: &[PropUnit {
                            count: None,
                            prop: AtomProp::SbAny,
                            relation: Some(PropRelation::BondedToPrev),
                        }],
                    },
                    PropConstraint {
                        units: &[PropUnit {
                            count: None,
                            prop: AtomProp::DbAny,
                            relation: None,
                        }],
                    },
                ],
            }),
            label: None,
            children: &[],
        }]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "cc",
        alternate: Some(Alternate {
            atom_type: "cd",
            pass: AlternatePass::Conjugated,
        }),
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
                        prop: AtomProp::SbAny,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::DbAny,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::Ar2,
                        relation: None,
                    }],
                },
            ],
        }),
        environment: Some(&[AtomPattern {
            atom: PatternAtom::Wild(WILD_XD),
            degree: Some(4),
            property: Some(PropExpr {
                constraints: &[
                    PropConstraint {
                        units: &[PropUnit {
                            count: None,
                            prop: AtomProp::SbAny,
                            relation: Some(PropRelation::BondedToPrev),
                        }],
                    },
                    PropConstraint {
                        units: &[PropUnit {
                            count: None,
                            prop: AtomProp::DbAny,
                            relation: None,
                        }],
                    },
                ],
            }),
            label: None,
            children: &[],
        }]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "cc",
        alternate: Some(Alternate {
            atom_type: "cd",
            pass: AlternatePass::Conjugated,
        }),
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
                        prop: AtomProp::SbAny,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::DbAny,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::Ar3,
                        relation: None,
                    }],
                },
            ],
        }),
        environment: Some(&[AtomPattern {
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
        }]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "cc",
        alternate: Some(Alternate {
            atom_type: "cd",
            pass: AlternatePass::Conjugated,
        }),
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
                        prop: AtomProp::SbAny,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::DbAny,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::Ar3,
                        relation: None,
                    }],
                },
            ],
        }),
        environment: Some(&[AtomPattern {
            atom: PatternAtom::Element(6),
            degree: Some(3),
            property: None,
            label: None,
            children: &[AtomPattern {
                atom: PatternAtom::Element(6),
                degree: Some(2),
                property: None,
                label: None,
                children: &[],
            }],
        }]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "cc",
        alternate: Some(Alternate {
            atom_type: "cd",
            pass: AlternatePass::Conjugated,
        }),
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
                        prop: AtomProp::SbAny,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::DbAny,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::Ar3,
                        relation: None,
                    }],
                },
            ],
        }),
        environment: Some(&[AtomPattern {
            atom: PatternAtom::Element(6),
            degree: Some(3),
            property: None,
            label: None,
            children: &[AtomPattern {
                atom: PatternAtom::Wild(WILD_XB),
                degree: Some(2),
                property: None,
                label: None,
                children: &[],
            }],
        }]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "cc",
        alternate: Some(Alternate {
            atom_type: "cd",
            pass: AlternatePass::Conjugated,
        }),
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
                        prop: AtomProp::SbAny,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::DbAny,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::Ar3,
                        relation: None,
                    }],
                },
            ],
        }),
        environment: Some(&[AtomPattern {
            atom: PatternAtom::Wild(WILD_XB),
            degree: Some(2),
            property: None,
            label: None,
            children: &[AtomPattern {
                atom: PatternAtom::Wild(WILD_XB),
                degree: Some(2),
                property: None,
                label: None,
                children: &[],
            }],
        }]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "cc",
        alternate: Some(Alternate {
            atom_type: "cd",
            pass: AlternatePass::Conjugated,
        }),
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
                        prop: AtomProp::SbAny,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::DbAny,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::Ar3,
                        relation: None,
                    }],
                },
            ],
        }),
        environment: Some(&[AtomPattern {
            atom: PatternAtom::Wild(WILD_XB),
            degree: Some(2),
            property: None,
            label: None,
            children: &[AtomPattern {
                atom: PatternAtom::Element(6),
                degree: Some(2),
                property: None,
                label: None,
                children: &[],
            }],
        }]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "cc",
        alternate: Some(Alternate {
            atom_type: "cd",
            pass: AlternatePass::Conjugated,
        }),
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
                        prop: AtomProp::SbAny,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::DbAny,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::Ar3,
                        relation: None,
                    }],
                },
            ],
        }),
        environment: Some(&[AtomPattern {
            atom: PatternAtom::Wild(WILD_XB),
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
        }]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "cc",
        alternate: Some(Alternate {
            atom_type: "cd",
            pass: AlternatePass::Conjugated,
        }),
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
                        prop: AtomProp::SbAny,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::DbAny,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::Ar3,
                        relation: None,
                    }],
                },
            ],
        }),
        environment: Some(&[AtomPattern {
            atom: PatternAtom::Element(6),
            degree: Some(3),
            property: Some(PropExpr {
                constraints: &[PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::SbAny,
                        relation: Some(PropRelation::BondedToPrev),
                    }],
                }],
            }),
            label: None,
            children: &[],
        }]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "cc",
        alternate: Some(Alternate {
            atom_type: "cd",
            pass: AlternatePass::Conjugated,
        }),
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
                        prop: AtomProp::SbAny,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::DbAny,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::Ar3,
                        relation: None,
                    }],
                },
            ],
        }),
        environment: Some(&[AtomPattern {
            atom: PatternAtom::Wild(WILD_XB),
            degree: Some(2),
            property: Some(PropExpr {
                constraints: &[PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::SbAny,
                        relation: Some(PropRelation::BondedToPrev),
                    }],
                }],
            }),
            label: None,
            children: &[],
        }]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "cc",
        alternate: Some(Alternate {
            atom_type: "cd",
            pass: AlternatePass::Conjugated,
        }),
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
                        prop: AtomProp::SbAny,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::DbAny,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::Ar3,
                        relation: None,
                    }],
                },
            ],
        }),
        environment: Some(&[AtomPattern {
            atom: PatternAtom::Wild(WILD_XD),
            degree: Some(3),
            property: Some(PropExpr {
                constraints: &[
                    PropConstraint {
                        units: &[PropUnit {
                            count: None,
                            prop: AtomProp::SbAny,
                            relation: Some(PropRelation::BondedToPrev),
                        }],
                    },
                    PropConstraint {
                        units: &[PropUnit {
                            count: None,
                            prop: AtomProp::DbAny,
                            relation: None,
                        }],
                    },
                ],
            }),
            label: None,
            children: &[],
        }]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "cc",
        alternate: Some(Alternate {
            atom_type: "cd",
            pass: AlternatePass::Conjugated,
        }),
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
                        prop: AtomProp::SbAny,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::DbAny,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::Ar3,
                        relation: None,
                    }],
                },
            ],
        }),
        environment: Some(&[AtomPattern {
            atom: PatternAtom::Wild(WILD_XD),
            degree: Some(4),
            property: Some(PropExpr {
                constraints: &[
                    PropConstraint {
                        units: &[PropUnit {
                            count: None,
                            prop: AtomProp::SbAny,
                            relation: Some(PropRelation::BondedToPrev),
                        }],
                    },
                    PropConstraint {
                        units: &[PropUnit {
                            count: None,
                            prop: AtomProp::DbAny,
                            relation: None,
                        }],
                    },
                ],
            }),
            label: None,
            children: &[],
        }]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "cc",
        alternate: Some(Alternate {
            atom_type: "cd",
            pass: AlternatePass::Conjugated,
        }),
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
                        prop: AtomProp::SbAny,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::DbAny,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::Ar2,
                        relation: None,
                    }],
                },
            ],
        }),
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "cc",
        alternate: Some(Alternate {
            atom_type: "cd",
            pass: AlternatePass::Conjugated,
        }),
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
                        prop: AtomProp::SbAny,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::DbAny,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::Ar3,
                        relation: None,
                    }],
                },
            ],
        }),
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "ce",
        alternate: Some(Alternate {
            atom_type: "cf",
            pass: AlternatePass::Conjugated,
        }),
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
                        prop: AtomProp::SbAny,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::DbAny,
                        relation: None,
                    }],
                },
            ],
        }),
        environment: Some(&[AtomPattern {
            atom: PatternAtom::Element(6),
            degree: Some(3),
            property: Some(PropExpr {
                constraints: &[PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::SbStrict,
                        relation: Some(PropRelation::BondedToPrev),
                    }],
                }],
            }),
            label: None,
            children: &[],
        }]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "ce",
        alternate: Some(Alternate {
            atom_type: "cf",
            pass: AlternatePass::Conjugated,
        }),
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
                        prop: AtomProp::SbAny,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::DbAny,
                        relation: None,
                    }],
                },
            ],
        }),
        environment: Some(&[AtomPattern {
            atom: PatternAtom::Element(6),
            degree: Some(2),
            property: Some(PropExpr {
                constraints: &[PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::SbStrict,
                        relation: Some(PropRelation::BondedToPrev),
                    }],
                }],
            }),
            label: None,
            children: &[],
        }]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "ce",
        alternate: Some(Alternate {
            atom_type: "cf",
            pass: AlternatePass::Conjugated,
        }),
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
                        prop: AtomProp::SbAny,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::DbAny,
                        relation: None,
                    }],
                },
            ],
        }),
        environment: Some(&[AtomPattern {
            atom: PatternAtom::Wild(WILD_XB),
            degree: Some(2),
            property: Some(PropExpr {
                constraints: &[PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::SbStrict,
                        relation: Some(PropRelation::BondedToPrev),
                    }],
                }],
            }),
            label: None,
            children: &[],
        }]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "ce",
        alternate: Some(Alternate {
            atom_type: "cf",
            pass: AlternatePass::Conjugated,
        }),
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
                        prop: AtomProp::SbAny,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::DbAny,
                        relation: None,
                    }],
                },
            ],
        }),
        environment: Some(&[AtomPattern {
            atom: PatternAtom::Wild(WILD_XD),
            degree: Some(3),
            property: Some(PropExpr {
                constraints: &[
                    PropConstraint {
                        units: &[PropUnit {
                            count: None,
                            prop: AtomProp::SbStrict,
                            relation: Some(PropRelation::BondedToPrev),
                        }],
                    },
                    PropConstraint {
                        units: &[PropUnit {
                            count: None,
                            prop: AtomProp::DbAny,
                            relation: None,
                        }],
                    },
                ],
            }),
            label: None,
            children: &[],
        }]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "ce",
        alternate: Some(Alternate {
            atom_type: "cf",
            pass: AlternatePass::Conjugated,
        }),
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
                        prop: AtomProp::SbAny,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::DbAny,
                        relation: None,
                    }],
                },
            ],
        }),
        environment: Some(&[AtomPattern {
            atom: PatternAtom::Wild(WILD_XD),
            degree: Some(4),
            property: Some(PropExpr {
                constraints: &[
                    PropConstraint {
                        units: &[PropUnit {
                            count: None,
                            prop: AtomProp::SbStrict,
                            relation: Some(PropRelation::BondedToPrev),
                        }],
                    },
                    PropConstraint {
                        units: &[PropUnit {
                            count: None,
                            prop: AtomProp::DbAny,
                            relation: None,
                        }],
                    },
                ],
            }),
            label: None,
            children: &[],
        }]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "cu",
        alternate: None,
        residue: "*",
        atomic_number: Some(6),
        degree: Some(3),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: Some(PropExpr {
            constraints: &[PropConstraint {
                units: &[PropUnit {
                    count: None,
                    prop: AtomProp::Rg3,
                    relation: None,
                }],
            }],
        }),
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "cv",
        alternate: None,
        residue: "*",
        atomic_number: Some(6),
        degree: Some(3),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: Some(PropExpr {
            constraints: &[PropConstraint {
                units: &[PropUnit {
                    count: None,
                    prop: AtomProp::Rg4,
                    relation: None,
                }],
            }],
        }),
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "c2",
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
        atom_type: "cg",
        alternate: Some(Alternate {
            atom_type: "ch",
            pass: AlternatePass::Conjugated,
        }),
        residue: "*",
        atomic_number: Some(6),
        degree: Some(2),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: Some(PropExpr {
            constraints: &[
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::SbAny,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::TbAny,
                        relation: None,
                    }],
                },
            ],
        }),
        environment: Some(&[AtomPattern {
            atom: PatternAtom::Element(6),
            degree: Some(2),
            property: Some(PropExpr {
                constraints: &[PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::SbStrict,
                        relation: Some(PropRelation::BondedToPrev),
                    }],
                }],
            }),
            label: None,
            children: &[],
        }]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "cg",
        alternate: Some(Alternate {
            atom_type: "ch",
            pass: AlternatePass::Conjugated,
        }),
        residue: "*",
        atomic_number: Some(6),
        degree: Some(2),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: Some(PropExpr {
            constraints: &[
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::SbAny,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::TbAny,
                        relation: None,
                    }],
                },
            ],
        }),
        environment: Some(&[AtomPattern {
            atom: PatternAtom::Element(6),
            degree: Some(3),
            property: Some(PropExpr {
                constraints: &[PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::SbStrict,
                        relation: Some(PropRelation::BondedToPrev),
                    }],
                }],
            }),
            label: None,
            children: &[],
        }]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "cg",
        alternate: Some(Alternate {
            atom_type: "ch",
            pass: AlternatePass::Conjugated,
        }),
        residue: "*",
        atomic_number: Some(6),
        degree: Some(2),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: Some(PropExpr {
            constraints: &[
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::SbAny,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::TbAny,
                        relation: None,
                    }],
                },
            ],
        }),
        environment: Some(&[AtomPattern {
            atom: PatternAtom::Element(7),
            degree: Some(1),
            property: Some(PropExpr {
                constraints: &[PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::SbStrict,
                        relation: Some(PropRelation::BondedToPrev),
                    }],
                }],
            }),
            label: None,
            children: &[],
        }]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "cg",
        alternate: Some(Alternate {
            atom_type: "ch",
            pass: AlternatePass::Conjugated,
        }),
        residue: "*",
        atomic_number: Some(6),
        degree: Some(2),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: Some(PropExpr {
            constraints: &[
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::SbAny,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::TbAny,
                        relation: None,
                    }],
                },
            ],
        }),
        environment: Some(&[AtomPattern {
            atom: PatternAtom::Wild(WILD_XB),
            degree: Some(2),
            property: Some(PropExpr {
                constraints: &[PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::SbStrict,
                        relation: Some(PropRelation::BondedToPrev),
                    }],
                }],
            }),
            label: None,
            children: &[],
        }]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "c1",
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
        atom_type: "c1",
        alternate: None,
        residue: "*",
        atomic_number: Some(6),
        degree: Some(1),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "hn",
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
        atom_type: "ho",
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
        atom_type: "hs",
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
        atom_type: "hp",
        alternate: None,
        residue: "*",
        atomic_number: Some(1),
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
        atom_type: "hx",
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
        atom_type: "hw",
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
        atom_type: "h3",
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
        atom_type: "h2",
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
        atom_type: "h1",
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
        atom_type: "hc",
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
        atom_type: "h5",
        alternate: None,
        residue: "*",
        atomic_number: Some(1),
        degree: Some(1),
        hydrogen_count: None,
        ewd_count: Some(2),
        atom_property: None,
        environment: Some(&[AtomPattern {
            atom: PatternAtom::Element(6),
            degree: Some(3),
            property: None,
            label: None,
            children: &[],
        }]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "h4",
        alternate: None,
        residue: "*",
        atomic_number: Some(1),
        degree: Some(1),
        hydrogen_count: None,
        ewd_count: Some(1),
        atom_property: None,
        environment: Some(&[AtomPattern {
            atom: PatternAtom::Element(6),
            degree: Some(3),
            property: None,
            label: None,
            children: &[],
        }]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "ha",
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
        atom_type: "f",
        alternate: None,
        residue: "*",
        atomic_number: Some(9),
        degree: None,
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "cl",
        alternate: None,
        residue: "*",
        atomic_number: Some(17),
        degree: None,
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "br",
        alternate: None,
        residue: "*",
        atomic_number: Some(35),
        degree: None,
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "i",
        alternate: None,
        residue: "*",
        atomic_number: Some(53),
        degree: None,
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "pc",
        alternate: Some(Alternate {
            atom_type: "pd",
            pass: AlternatePass::Conjugated,
        }),
        residue: "*",
        atomic_number: Some(15),
        degree: Some(2),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: Some(PropExpr {
            constraints: &[
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::SbAny,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::DbAny,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::Ar2,
                        relation: None,
                    }],
                },
            ],
        }),
        environment: Some(&[AtomPattern {
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
        }]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "pc",
        alternate: Some(Alternate {
            atom_type: "pd",
            pass: AlternatePass::Conjugated,
        }),
        residue: "*",
        atomic_number: Some(15),
        degree: Some(2),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: Some(PropExpr {
            constraints: &[
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::SbAny,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::DbAny,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::Ar2,
                        relation: None,
                    }],
                },
            ],
        }),
        environment: Some(&[AtomPattern {
            atom: PatternAtom::Element(6),
            degree: Some(3),
            property: None,
            label: None,
            children: &[AtomPattern {
                atom: PatternAtom::Element(6),
                degree: Some(2),
                property: None,
                label: None,
                children: &[],
            }],
        }]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "pc",
        alternate: Some(Alternate {
            atom_type: "pd",
            pass: AlternatePass::Conjugated,
        }),
        residue: "*",
        atomic_number: Some(15),
        degree: Some(2),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: Some(PropExpr {
            constraints: &[
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::SbAny,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::DbAny,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::Ar2,
                        relation: None,
                    }],
                },
            ],
        }),
        environment: Some(&[AtomPattern {
            atom: PatternAtom::Element(6),
            degree: Some(3),
            property: None,
            label: None,
            children: &[AtomPattern {
                atom: PatternAtom::Wild(WILD_XB),
                degree: Some(2),
                property: None,
                label: None,
                children: &[],
            }],
        }]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "pc",
        alternate: Some(Alternate {
            atom_type: "pd",
            pass: AlternatePass::Conjugated,
        }),
        residue: "*",
        atomic_number: Some(15),
        degree: Some(2),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: Some(PropExpr {
            constraints: &[
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::SbAny,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::DbAny,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::Ar2,
                        relation: None,
                    }],
                },
            ],
        }),
        environment: Some(&[AtomPattern {
            atom: PatternAtom::Wild(WILD_XB),
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
        }]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "pc",
        alternate: Some(Alternate {
            atom_type: "pd",
            pass: AlternatePass::Conjugated,
        }),
        residue: "*",
        atomic_number: Some(15),
        degree: Some(2),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: Some(PropExpr {
            constraints: &[
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::SbAny,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::DbAny,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::Ar2,
                        relation: None,
                    }],
                },
            ],
        }),
        environment: Some(&[AtomPattern {
            atom: PatternAtom::Wild(WILD_XB),
            degree: Some(2),
            property: None,
            label: None,
            children: &[AtomPattern {
                atom: PatternAtom::Element(6),
                degree: Some(2),
                property: None,
                label: None,
                children: &[],
            }],
        }]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "pc",
        alternate: Some(Alternate {
            atom_type: "pd",
            pass: AlternatePass::Conjugated,
        }),
        residue: "*",
        atomic_number: Some(15),
        degree: Some(2),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: Some(PropExpr {
            constraints: &[
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::SbAny,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::DbAny,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::Ar2,
                        relation: None,
                    }],
                },
            ],
        }),
        environment: Some(&[AtomPattern {
            atom: PatternAtom::Wild(WILD_XB),
            degree: Some(2),
            property: None,
            label: None,
            children: &[AtomPattern {
                atom: PatternAtom::Wild(WILD_XB),
                degree: Some(2),
                property: None,
                label: None,
                children: &[],
            }],
        }]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "pc",
        alternate: Some(Alternate {
            atom_type: "pd",
            pass: AlternatePass::Conjugated,
        }),
        residue: "*",
        atomic_number: Some(15),
        degree: Some(2),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: Some(PropExpr {
            constraints: &[
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::SbAny,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::DbAny,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::Ar2,
                        relation: None,
                    }],
                },
            ],
        }),
        environment: Some(&[AtomPattern {
            atom: PatternAtom::Element(6),
            degree: Some(3),
            property: Some(PropExpr {
                constraints: &[PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::SbAny,
                        relation: Some(PropRelation::BondedToPrev),
                    }],
                }],
            }),
            label: None,
            children: &[],
        }]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "pc",
        alternate: Some(Alternate {
            atom_type: "pd",
            pass: AlternatePass::Conjugated,
        }),
        residue: "*",
        atomic_number: Some(15),
        degree: Some(2),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: Some(PropExpr {
            constraints: &[
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::SbAny,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::DbAny,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::Ar2,
                        relation: None,
                    }],
                },
            ],
        }),
        environment: Some(&[AtomPattern {
            atom: PatternAtom::Element(6),
            degree: Some(2),
            property: Some(PropExpr {
                constraints: &[PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::SbAny,
                        relation: Some(PropRelation::BondedToPrev),
                    }],
                }],
            }),
            label: None,
            children: &[],
        }]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "pc",
        alternate: Some(Alternate {
            atom_type: "pd",
            pass: AlternatePass::Conjugated,
        }),
        residue: "*",
        atomic_number: Some(15),
        degree: Some(2),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: Some(PropExpr {
            constraints: &[
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::SbAny,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::DbAny,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::Ar2,
                        relation: None,
                    }],
                },
            ],
        }),
        environment: Some(&[AtomPattern {
            atom: PatternAtom::Wild(WILD_XB),
            degree: Some(2),
            property: Some(PropExpr {
                constraints: &[PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::SbAny,
                        relation: Some(PropRelation::BondedToPrev),
                    }],
                }],
            }),
            label: None,
            children: &[],
        }]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "pc",
        alternate: Some(Alternate {
            atom_type: "pd",
            pass: AlternatePass::Conjugated,
        }),
        residue: "*",
        atomic_number: Some(15),
        degree: Some(2),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: Some(PropExpr {
            constraints: &[
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::SbAny,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::DbAny,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::Ar2,
                        relation: None,
                    }],
                },
            ],
        }),
        environment: Some(&[AtomPattern {
            atom: PatternAtom::Wild(WILD_XD),
            degree: Some(3),
            property: Some(PropExpr {
                constraints: &[
                    PropConstraint {
                        units: &[PropUnit {
                            count: None,
                            prop: AtomProp::SbAny,
                            relation: Some(PropRelation::BondedToPrev),
                        }],
                    },
                    PropConstraint {
                        units: &[PropUnit {
                            count: None,
                            prop: AtomProp::DbAny,
                            relation: None,
                        }],
                    },
                ],
            }),
            label: None,
            children: &[],
        }]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "pc",
        alternate: Some(Alternate {
            atom_type: "pd",
            pass: AlternatePass::Conjugated,
        }),
        residue: "*",
        atomic_number: Some(15),
        degree: Some(2),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: Some(PropExpr {
            constraints: &[
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::SbAny,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::DbAny,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::Ar2,
                        relation: None,
                    }],
                },
            ],
        }),
        environment: Some(&[AtomPattern {
            atom: PatternAtom::Wild(WILD_XD),
            degree: Some(4),
            property: Some(PropExpr {
                constraints: &[
                    PropConstraint {
                        units: &[PropUnit {
                            count: None,
                            prop: AtomProp::SbAny,
                            relation: Some(PropRelation::BondedToPrev),
                        }],
                    },
                    PropConstraint {
                        units: &[PropUnit {
                            count: None,
                            prop: AtomProp::DbAny,
                            relation: None,
                        }],
                    },
                ],
            }),
            label: None,
            children: &[],
        }]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "pc",
        alternate: Some(Alternate {
            atom_type: "pd",
            pass: AlternatePass::Conjugated,
        }),
        residue: "*",
        atomic_number: Some(15),
        degree: Some(2),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: Some(PropExpr {
            constraints: &[
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::SbAny,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::DbAny,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::Ar3,
                        relation: None,
                    }],
                },
            ],
        }),
        environment: Some(&[AtomPattern {
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
        }]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "pc",
        alternate: Some(Alternate {
            atom_type: "pd",
            pass: AlternatePass::Conjugated,
        }),
        residue: "*",
        atomic_number: Some(15),
        degree: Some(2),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: Some(PropExpr {
            constraints: &[
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::SbAny,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::DbAny,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::Ar3,
                        relation: None,
                    }],
                },
            ],
        }),
        environment: Some(&[AtomPattern {
            atom: PatternAtom::Element(6),
            degree: Some(3),
            property: None,
            label: None,
            children: &[AtomPattern {
                atom: PatternAtom::Element(6),
                degree: Some(2),
                property: None,
                label: None,
                children: &[],
            }],
        }]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "pc",
        alternate: Some(Alternate {
            atom_type: "pd",
            pass: AlternatePass::Conjugated,
        }),
        residue: "*",
        atomic_number: Some(15),
        degree: Some(2),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: Some(PropExpr {
            constraints: &[
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::SbAny,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::DbAny,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::Ar3,
                        relation: None,
                    }],
                },
            ],
        }),
        environment: Some(&[AtomPattern {
            atom: PatternAtom::Element(6),
            degree: Some(3),
            property: None,
            label: None,
            children: &[AtomPattern {
                atom: PatternAtom::Wild(WILD_XB),
                degree: Some(2),
                property: None,
                label: None,
                children: &[],
            }],
        }]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "pc",
        alternate: Some(Alternate {
            atom_type: "pd",
            pass: AlternatePass::Conjugated,
        }),
        residue: "*",
        atomic_number: Some(15),
        degree: Some(2),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: Some(PropExpr {
            constraints: &[
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::SbAny,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::DbAny,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::Ar3,
                        relation: None,
                    }],
                },
            ],
        }),
        environment: Some(&[AtomPattern {
            atom: PatternAtom::Wild(WILD_XB),
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
        }]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "pc",
        alternate: Some(Alternate {
            atom_type: "pd",
            pass: AlternatePass::Conjugated,
        }),
        residue: "*",
        atomic_number: Some(15),
        degree: Some(2),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: Some(PropExpr {
            constraints: &[
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::SbAny,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::DbAny,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::Ar3,
                        relation: None,
                    }],
                },
            ],
        }),
        environment: Some(&[AtomPattern {
            atom: PatternAtom::Wild(WILD_XB),
            degree: Some(2),
            property: None,
            label: None,
            children: &[AtomPattern {
                atom: PatternAtom::Element(6),
                degree: Some(2),
                property: None,
                label: None,
                children: &[],
            }],
        }]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "pc",
        alternate: Some(Alternate {
            atom_type: "pd",
            pass: AlternatePass::Conjugated,
        }),
        residue: "*",
        atomic_number: Some(15),
        degree: Some(2),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: Some(PropExpr {
            constraints: &[
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::SbAny,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::DbAny,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::Ar3,
                        relation: None,
                    }],
                },
            ],
        }),
        environment: Some(&[AtomPattern {
            atom: PatternAtom::Wild(WILD_XB),
            degree: Some(2),
            property: None,
            label: None,
            children: &[AtomPattern {
                atom: PatternAtom::Wild(WILD_XB),
                degree: Some(2),
                property: None,
                label: None,
                children: &[],
            }],
        }]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "pc",
        alternate: Some(Alternate {
            atom_type: "pd",
            pass: AlternatePass::Conjugated,
        }),
        residue: "*",
        atomic_number: Some(15),
        degree: Some(2),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: Some(PropExpr {
            constraints: &[
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::SbAny,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::DbAny,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::Ar3,
                        relation: None,
                    }],
                },
            ],
        }),
        environment: Some(&[AtomPattern {
            atom: PatternAtom::Element(6),
            degree: Some(3),
            property: Some(PropExpr {
                constraints: &[PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::SbAny,
                        relation: Some(PropRelation::BondedToPrev),
                    }],
                }],
            }),
            label: None,
            children: &[],
        }]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "pc",
        alternate: Some(Alternate {
            atom_type: "pd",
            pass: AlternatePass::Conjugated,
        }),
        residue: "*",
        atomic_number: Some(15),
        degree: Some(2),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: Some(PropExpr {
            constraints: &[
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::SbAny,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::DbAny,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::Ar3,
                        relation: None,
                    }],
                },
            ],
        }),
        environment: Some(&[AtomPattern {
            atom: PatternAtom::Element(6),
            degree: Some(2),
            property: Some(PropExpr {
                constraints: &[PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::SbAny,
                        relation: Some(PropRelation::BondedToPrev),
                    }],
                }],
            }),
            label: None,
            children: &[],
        }]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "pc",
        alternate: Some(Alternate {
            atom_type: "pd",
            pass: AlternatePass::Conjugated,
        }),
        residue: "*",
        atomic_number: Some(15),
        degree: Some(2),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: Some(PropExpr {
            constraints: &[
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::SbAny,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::DbAny,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::Ar3,
                        relation: None,
                    }],
                },
            ],
        }),
        environment: Some(&[AtomPattern {
            atom: PatternAtom::Wild(WILD_XB),
            degree: Some(2),
            property: Some(PropExpr {
                constraints: &[PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::SbAny,
                        relation: Some(PropRelation::BondedToPrev),
                    }],
                }],
            }),
            label: None,
            children: &[],
        }]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "pc",
        alternate: Some(Alternate {
            atom_type: "pd",
            pass: AlternatePass::Conjugated,
        }),
        residue: "*",
        atomic_number: Some(15),
        degree: Some(2),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: Some(PropExpr {
            constraints: &[
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::SbAny,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::DbAny,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::Ar3,
                        relation: None,
                    }],
                },
            ],
        }),
        environment: Some(&[AtomPattern {
            atom: PatternAtom::Wild(WILD_XD),
            degree: Some(3),
            property: Some(PropExpr {
                constraints: &[
                    PropConstraint {
                        units: &[PropUnit {
                            count: None,
                            prop: AtomProp::SbAny,
                            relation: Some(PropRelation::BondedToPrev),
                        }],
                    },
                    PropConstraint {
                        units: &[PropUnit {
                            count: None,
                            prop: AtomProp::DbAny,
                            relation: None,
                        }],
                    },
                ],
            }),
            label: None,
            children: &[],
        }]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "pc",
        alternate: Some(Alternate {
            atom_type: "pd",
            pass: AlternatePass::Conjugated,
        }),
        residue: "*",
        atomic_number: Some(15),
        degree: Some(2),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: Some(PropExpr {
            constraints: &[
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::SbAny,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::DbAny,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::Ar3,
                        relation: None,
                    }],
                },
            ],
        }),
        environment: Some(&[AtomPattern {
            atom: PatternAtom::Wild(WILD_XD),
            degree: Some(4),
            property: Some(PropExpr {
                constraints: &[
                    PropConstraint {
                        units: &[PropUnit {
                            count: None,
                            prop: AtomProp::SbAny,
                            relation: Some(PropRelation::BondedToPrev),
                        }],
                    },
                    PropConstraint {
                        units: &[PropUnit {
                            count: None,
                            prop: AtomProp::DbAny,
                            relation: None,
                        }],
                    },
                ],
            }),
            label: None,
            children: &[],
        }]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "pb",
        alternate: None,
        residue: "*",
        atomic_number: Some(15),
        degree: Some(2),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: Some(PropExpr {
            constraints: &[PropConstraint {
                units: &[PropUnit {
                    count: None,
                    prop: AtomProp::Ar1,
                    relation: None,
                }],
            }],
        }),
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "pe",
        alternate: Some(Alternate {
            atom_type: "pf",
            pass: AlternatePass::Conjugated,
        }),
        residue: "*",
        atomic_number: Some(15),
        degree: Some(2),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: Some(PropExpr {
            constraints: &[
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::SbAny,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::DbAny,
                        relation: None,
                    }],
                },
            ],
        }),
        environment: Some(&[AtomPattern {
            atom: PatternAtom::Element(6),
            degree: Some(3),
            property: Some(PropExpr {
                constraints: &[PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::SbAny,
                        relation: Some(PropRelation::BondedToPrev),
                    }],
                }],
            }),
            label: None,
            children: &[],
        }]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "pe",
        alternate: Some(Alternate {
            atom_type: "pf",
            pass: AlternatePass::Conjugated,
        }),
        residue: "*",
        atomic_number: Some(15),
        degree: Some(2),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: Some(PropExpr {
            constraints: &[
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::SbAny,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::DbAny,
                        relation: None,
                    }],
                },
            ],
        }),
        environment: Some(&[AtomPattern {
            atom: PatternAtom::Element(6),
            degree: Some(2),
            property: Some(PropExpr {
                constraints: &[PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::SbStrict,
                        relation: Some(PropRelation::BondedToPrev),
                    }],
                }],
            }),
            label: None,
            children: &[],
        }]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "pe",
        alternate: Some(Alternate {
            atom_type: "pf",
            pass: AlternatePass::Conjugated,
        }),
        residue: "*",
        atomic_number: Some(15),
        degree: Some(2),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: Some(PropExpr {
            constraints: &[
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::SbAny,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::DbAny,
                        relation: None,
                    }],
                },
            ],
        }),
        environment: Some(&[AtomPattern {
            atom: PatternAtom::Wild(WILD_XA),
            degree: Some(1),
            property: Some(PropExpr {
                constraints: &[PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::SbStrict,
                        relation: Some(PropRelation::BondedToPrev),
                    }],
                }],
            }),
            label: None,
            children: &[],
        }]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "pe",
        alternate: Some(Alternate {
            atom_type: "pf",
            pass: AlternatePass::Conjugated,
        }),
        residue: "*",
        atomic_number: Some(15),
        degree: Some(2),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: Some(PropExpr {
            constraints: &[
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::SbAny,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::DbAny,
                        relation: None,
                    }],
                },
            ],
        }),
        environment: Some(&[AtomPattern {
            atom: PatternAtom::Wild(WILD_XB),
            degree: Some(2),
            property: Some(PropExpr {
                constraints: &[PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::SbStrict,
                        relation: Some(PropRelation::BondedToPrev),
                    }],
                }],
            }),
            label: None,
            children: &[],
        }]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "pe",
        alternate: Some(Alternate {
            atom_type: "pf",
            pass: AlternatePass::Conjugated,
        }),
        residue: "*",
        atomic_number: Some(15),
        degree: Some(2),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: Some(PropExpr {
            constraints: &[
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::SbAny,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::DbAny,
                        relation: None,
                    }],
                },
            ],
        }),
        environment: Some(&[AtomPattern {
            atom: PatternAtom::Wild(WILD_XD),
            degree: Some(3),
            property: Some(PropExpr {
                constraints: &[
                    PropConstraint {
                        units: &[PropUnit {
                            count: None,
                            prop: AtomProp::SbStrict,
                            relation: Some(PropRelation::BondedToPrev),
                        }],
                    },
                    PropConstraint {
                        units: &[PropUnit {
                            count: None,
                            prop: AtomProp::DbStrict,
                            relation: None,
                        }],
                    },
                ],
            }),
            label: None,
            children: &[],
        }]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "pe",
        alternate: Some(Alternate {
            atom_type: "pf",
            pass: AlternatePass::Conjugated,
        }),
        residue: "*",
        atomic_number: Some(15),
        degree: Some(2),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: Some(PropExpr {
            constraints: &[
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::SbAny,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::DbAny,
                        relation: None,
                    }],
                },
            ],
        }),
        environment: Some(&[AtomPattern {
            atom: PatternAtom::Wild(WILD_XD),
            degree: Some(4),
            property: Some(PropExpr {
                constraints: &[
                    PropConstraint {
                        units: &[PropUnit {
                            count: None,
                            prop: AtomProp::SbStrict,
                            relation: Some(PropRelation::BondedToPrev),
                        }],
                    },
                    PropConstraint {
                        units: &[PropUnit {
                            count: None,
                            prop: AtomProp::DbStrict,
                            relation: None,
                        }],
                    },
                ],
            }),
            label: None,
            children: &[],
        }]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "p2",
        alternate: None,
        residue: "*",
        atomic_number: Some(15),
        degree: Some(2),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "p2",
        alternate: None,
        residue: "*",
        atomic_number: Some(15),
        degree: Some(1),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "px",
        alternate: None,
        residue: "*",
        atomic_number: Some(15),
        degree: Some(3),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: Some(PropExpr {
            constraints: &[PropConstraint {
                units: &[PropUnit {
                    count: None,
                    prop: AtomProp::DbAny,
                    relation: None,
                }],
            }],
        }),
        environment: Some(&[AtomPattern {
            atom: PatternAtom::Wild(WILD_XB),
            degree: Some(2),
            property: Some(PropExpr {
                constraints: &[PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::SbAny,
                        relation: Some(PropRelation::BondedToPrev),
                    }],
                }],
            }),
            label: None,
            children: &[],
        }]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "px",
        alternate: None,
        residue: "*",
        atomic_number: Some(15),
        degree: Some(3),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: Some(PropExpr {
            constraints: &[PropConstraint {
                units: &[PropUnit {
                    count: None,
                    prop: AtomProp::DbAny,
                    relation: None,
                }],
            }],
        }),
        environment: Some(&[AtomPattern {
            atom: PatternAtom::Element(6),
            degree: Some(3),
            property: Some(PropExpr {
                constraints: &[PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::SbAny,
                        relation: Some(PropRelation::BondedToPrev),
                    }],
                }],
            }),
            label: None,
            children: &[],
        }]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "px",
        alternate: None,
        residue: "*",
        atomic_number: Some(15),
        degree: Some(3),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: Some(PropExpr {
            constraints: &[PropConstraint {
                units: &[PropUnit {
                    count: None,
                    prop: AtomProp::DbAny,
                    relation: None,
                }],
            }],
        }),
        environment: Some(&[AtomPattern {
            atom: PatternAtom::Wild(WILD_XD),
            degree: Some(3),
            property: Some(PropExpr {
                constraints: &[
                    PropConstraint {
                        units: &[PropUnit {
                            count: None,
                            prop: AtomProp::SbAny,
                            relation: Some(PropRelation::BondedToPrev),
                        }],
                    },
                    PropConstraint {
                        units: &[PropUnit {
                            count: None,
                            prop: AtomProp::DbAny,
                            relation: None,
                        }],
                    },
                ],
            }),
            label: None,
            children: &[],
        }]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "px",
        alternate: None,
        residue: "*",
        atomic_number: Some(15),
        degree: Some(3),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: Some(PropExpr {
            constraints: &[PropConstraint {
                units: &[PropUnit {
                    count: None,
                    prop: AtomProp::DbAny,
                    relation: None,
                }],
            }],
        }),
        environment: Some(&[AtomPattern {
            atom: PatternAtom::Wild(WILD_XD),
            degree: Some(4),
            property: Some(PropExpr {
                constraints: &[
                    PropConstraint {
                        units: &[PropUnit {
                            count: None,
                            prop: AtomProp::SbAny,
                            relation: Some(PropRelation::BondedToPrev),
                        }],
                    },
                    PropConstraint {
                        units: &[PropUnit {
                            count: None,
                            prop: AtomProp::DbAny,
                            relation: None,
                        }],
                    },
                ],
            }),
            label: None,
            children: &[],
        }]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "p4",
        alternate: None,
        residue: "*",
        atomic_number: Some(15),
        degree: Some(3),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: Some(PropExpr {
            constraints: &[PropConstraint {
                units: &[PropUnit {
                    count: None,
                    prop: AtomProp::DbAny,
                    relation: None,
                }],
            }],
        }),
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
        atom_type: "p3",
        alternate: None,
        residue: "*",
        atomic_number: Some(15),
        degree: Some(3),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "py",
        alternate: None,
        residue: "*",
        atomic_number: Some(15),
        degree: Some(4),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: Some(PropExpr {
            constraints: &[PropConstraint {
                units: &[PropUnit {
                    count: None,
                    prop: AtomProp::DbAny,
                    relation: None,
                }],
            }],
        }),
        environment: Some(&[AtomPattern {
            atom: PatternAtom::Wild(WILD_XB),
            degree: Some(2),
            property: Some(PropExpr {
                constraints: &[PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::SbAny,
                        relation: Some(PropRelation::BondedToPrev),
                    }],
                }],
            }),
            label: None,
            children: &[],
        }]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "py",
        alternate: None,
        residue: "*",
        atomic_number: Some(15),
        degree: Some(4),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: Some(PropExpr {
            constraints: &[PropConstraint {
                units: &[PropUnit {
                    count: None,
                    prop: AtomProp::DbAny,
                    relation: None,
                }],
            }],
        }),
        environment: Some(&[AtomPattern {
            atom: PatternAtom::Element(6),
            degree: Some(3),
            property: Some(PropExpr {
                constraints: &[PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::SbAny,
                        relation: Some(PropRelation::BondedToPrev),
                    }],
                }],
            }),
            label: None,
            children: &[],
        }]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "py",
        alternate: None,
        residue: "*",
        atomic_number: Some(15),
        degree: Some(4),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: Some(PropExpr {
            constraints: &[PropConstraint {
                units: &[PropUnit {
                    count: None,
                    prop: AtomProp::DbAny,
                    relation: None,
                }],
            }],
        }),
        environment: Some(&[AtomPattern {
            atom: PatternAtom::Wild(WILD_XD),
            degree: Some(3),
            property: Some(PropExpr {
                constraints: &[
                    PropConstraint {
                        units: &[PropUnit {
                            count: None,
                            prop: AtomProp::SbAny,
                            relation: Some(PropRelation::BondedToPrev),
                        }],
                    },
                    PropConstraint {
                        units: &[PropUnit {
                            count: None,
                            prop: AtomProp::DbAny,
                            relation: None,
                        }],
                    },
                ],
            }),
            label: None,
            children: &[],
        }]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "py",
        alternate: None,
        residue: "*",
        atomic_number: Some(15),
        degree: Some(4),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: Some(PropExpr {
            constraints: &[PropConstraint {
                units: &[PropUnit {
                    count: None,
                    prop: AtomProp::DbAny,
                    relation: None,
                }],
            }],
        }),
        environment: Some(&[AtomPattern {
            atom: PatternAtom::Wild(WILD_XD),
            degree: Some(4),
            property: Some(PropExpr {
                constraints: &[
                    PropConstraint {
                        units: &[PropUnit {
                            count: None,
                            prop: AtomProp::SbAny,
                            relation: Some(PropRelation::BondedToPrev),
                        }],
                    },
                    PropConstraint {
                        units: &[PropUnit {
                            count: None,
                            prop: AtomProp::DbAny,
                            relation: None,
                        }],
                    },
                ],
            }),
            label: None,
            children: &[],
        }]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "p5",
        alternate: None,
        residue: "*",
        atomic_number: Some(15),
        degree: Some(4),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "p5",
        alternate: None,
        residue: "*",
        atomic_number: Some(15),
        degree: Some(5),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "p5",
        alternate: None,
        residue: "*",
        atomic_number: Some(15),
        degree: Some(6),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "ni",
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
                    prop: AtomProp::Rg3,
                    relation: None,
                }],
            }],
        }),
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
        atom_type: "nj",
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
                    prop: AtomProp::Rg4,
                    relation: None,
                }],
            }],
        }),
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
        atom_type: "n",
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
        atom_type: "nk",
        alternate: None,
        residue: "*",
        atomic_number: Some(7),
        degree: Some(4),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: Some(PropExpr {
            constraints: &[PropConstraint {
                units: &[PropUnit {
                    count: None,
                    prop: AtomProp::Rg3,
                    relation: None,
                }],
            }],
        }),
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "nl",
        alternate: None,
        residue: "*",
        atomic_number: Some(7),
        degree: Some(4),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: Some(PropExpr {
            constraints: &[PropConstraint {
                units: &[PropUnit {
                    count: None,
                    prop: AtomProp::Rg4,
                    relation: None,
                }],
            }],
        }),
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "n4",
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
        atom_type: "no",
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
        atom_type: "na",
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
        atom_type: "nm",
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
                    prop: AtomProp::Rg3,
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
        atom_type: "nm",
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
                    prop: AtomProp::Rg3,
                    relation: None,
                }],
            }],
        }),
        environment: Some(&[AtomPattern {
            atom: PatternAtom::Element(6),
            degree: Some(3),
            property: Some(PropExpr {
                constraints: &[PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::DbStrict,
                        relation: None,
                    }],
                }],
            }),
            label: None,
            children: &[],
        }]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "nm",
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
                    prop: AtomProp::Rg3,
                    relation: None,
                }],
            }],
        }),
        environment: Some(&[AtomPattern {
            atom: PatternAtom::Element(7),
            degree: Some(2),
            property: Some(PropExpr {
                constraints: &[PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::DbStrict,
                        relation: None,
                    }],
                }],
            }),
            label: None,
            children: &[],
        }]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "nm",
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
                    prop: AtomProp::Rg3,
                    relation: None,
                }],
            }],
        }),
        environment: Some(&[AtomPattern {
            atom: PatternAtom::Element(15),
            degree: Some(2),
            property: Some(PropExpr {
                constraints: &[PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::DbStrict,
                        relation: None,
                    }],
                }],
            }),
            label: None,
            children: &[],
        }]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "nn",
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
                    prop: AtomProp::Rg4,
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
        atom_type: "nn",
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
                    prop: AtomProp::Rg4,
                    relation: None,
                }],
            }],
        }),
        environment: Some(&[AtomPattern {
            atom: PatternAtom::Element(6),
            degree: Some(3),
            property: Some(PropExpr {
                constraints: &[PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::DbStrict,
                        relation: None,
                    }],
                }],
            }),
            label: None,
            children: &[],
        }]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "nn",
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
                    prop: AtomProp::Rg4,
                    relation: None,
                }],
            }],
        }),
        environment: Some(&[AtomPattern {
            atom: PatternAtom::Element(7),
            degree: Some(2),
            property: Some(PropExpr {
                constraints: &[PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::DbStrict,
                        relation: None,
                    }],
                }],
            }),
            label: None,
            children: &[],
        }]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "nn",
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
                    prop: AtomProp::Rg4,
                    relation: None,
                }],
            }],
        }),
        environment: Some(&[AtomPattern {
            atom: PatternAtom::Element(15),
            degree: Some(2),
            property: Some(PropExpr {
                constraints: &[PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::DbStrict,
                        relation: None,
                    }],
                }],
            }),
            label: None,
            children: &[],
        }]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "nh",
        alternate: None,
        residue: "*",
        atomic_number: Some(7),
        degree: Some(3),
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
        atom_type: "nh",
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
            property: Some(PropExpr {
                constraints: &[PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::DbStrict,
                        relation: None,
                    }],
                }],
            }),
            label: None,
            children: &[],
        }]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "nh",
        alternate: None,
        residue: "*",
        atomic_number: Some(7),
        degree: Some(3),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: Some(&[AtomPattern {
            atom: PatternAtom::Element(7),
            degree: Some(2),
            property: Some(PropExpr {
                constraints: &[PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::DbStrict,
                        relation: None,
                    }],
                }],
            }),
            label: None,
            children: &[],
        }]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "nh",
        alternate: None,
        residue: "*",
        atomic_number: Some(7),
        degree: Some(3),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: Some(&[AtomPattern {
            atom: PatternAtom::Element(15),
            degree: Some(2),
            property: Some(PropExpr {
                constraints: &[PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::DbStrict,
                        relation: None,
                    }],
                }],
            }),
            label: None,
            children: &[],
        }]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "np",
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
                    prop: AtomProp::Rg3,
                    relation: None,
                }],
            }],
        }),
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "nq",
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
                    prop: AtomProp::Rg4,
                    relation: None,
                }],
            }],
        }),
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "n3",
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
        atom_type: "nb",
        alternate: None,
        residue: "*",
        atomic_number: Some(7),
        degree: Some(2),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: Some(PropExpr {
            constraints: &[PropConstraint {
                units: &[PropUnit {
                    count: None,
                    prop: AtomProp::Ar1,
                    relation: None,
                }],
            }],
        }),
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "nc",
        alternate: Some(Alternate {
            atom_type: "nd",
            pass: AlternatePass::Conjugated,
        }),
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
                        prop: AtomProp::SbAny,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::DbAny,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::Ar2,
                        relation: None,
                    }],
                },
            ],
        }),
        environment: Some(&[AtomPattern {
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
        }]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "nc",
        alternate: Some(Alternate {
            atom_type: "nd",
            pass: AlternatePass::Conjugated,
        }),
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
                        prop: AtomProp::SbAny,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::DbAny,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::Ar2,
                        relation: None,
                    }],
                },
            ],
        }),
        environment: Some(&[AtomPattern {
            atom: PatternAtom::Element(6),
            degree: Some(3),
            property: None,
            label: None,
            children: &[AtomPattern {
                atom: PatternAtom::Element(6),
                degree: Some(2),
                property: None,
                label: None,
                children: &[],
            }],
        }]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "nc",
        alternate: Some(Alternate {
            atom_type: "nd",
            pass: AlternatePass::Conjugated,
        }),
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
                        prop: AtomProp::SbAny,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::DbAny,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::Ar2,
                        relation: None,
                    }],
                },
            ],
        }),
        environment: Some(&[AtomPattern {
            atom: PatternAtom::Element(6),
            degree: Some(3),
            property: None,
            label: None,
            children: &[AtomPattern {
                atom: PatternAtom::Wild(WILD_XB),
                degree: Some(2),
                property: None,
                label: None,
                children: &[],
            }],
        }]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "nc",
        alternate: Some(Alternate {
            atom_type: "nd",
            pass: AlternatePass::Conjugated,
        }),
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
                        prop: AtomProp::SbAny,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::DbAny,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::Ar2,
                        relation: None,
                    }],
                },
            ],
        }),
        environment: Some(&[AtomPattern {
            atom: PatternAtom::Wild(WILD_XB),
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
        }]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "nc",
        alternate: Some(Alternate {
            atom_type: "nd",
            pass: AlternatePass::Conjugated,
        }),
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
                        prop: AtomProp::SbAny,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::DbAny,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::Ar2,
                        relation: None,
                    }],
                },
            ],
        }),
        environment: Some(&[AtomPattern {
            atom: PatternAtom::Wild(WILD_XB),
            degree: Some(2),
            property: None,
            label: None,
            children: &[AtomPattern {
                atom: PatternAtom::Element(6),
                degree: Some(2),
                property: None,
                label: None,
                children: &[],
            }],
        }]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "nc",
        alternate: Some(Alternate {
            atom_type: "nd",
            pass: AlternatePass::Conjugated,
        }),
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
                        prop: AtomProp::SbAny,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::DbAny,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::Ar2,
                        relation: None,
                    }],
                },
            ],
        }),
        environment: Some(&[AtomPattern {
            atom: PatternAtom::Wild(WILD_XB),
            degree: Some(2),
            property: None,
            label: None,
            children: &[AtomPattern {
                atom: PatternAtom::Wild(WILD_XB),
                degree: Some(2),
                property: None,
                label: None,
                children: &[],
            }],
        }]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "nc",
        alternate: Some(Alternate {
            atom_type: "nd",
            pass: AlternatePass::Conjugated,
        }),
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
                        prop: AtomProp::SbAny,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::DbAny,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::Ar2,
                        relation: None,
                    }],
                },
            ],
        }),
        environment: Some(&[AtomPattern {
            atom: PatternAtom::Element(6),
            degree: Some(3),
            property: Some(PropExpr {
                constraints: &[PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::SbAny,
                        relation: Some(PropRelation::BondedToPrev),
                    }],
                }],
            }),
            label: None,
            children: &[],
        }]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "nc",
        alternate: Some(Alternate {
            atom_type: "nd",
            pass: AlternatePass::Conjugated,
        }),
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
                        prop: AtomProp::SbAny,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::DbAny,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::Ar2,
                        relation: None,
                    }],
                },
            ],
        }),
        environment: Some(&[AtomPattern {
            atom: PatternAtom::Wild(WILD_XB),
            degree: Some(2),
            property: Some(PropExpr {
                constraints: &[PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::SbAny,
                        relation: Some(PropRelation::BondedToPrev),
                    }],
                }],
            }),
            label: None,
            children: &[],
        }]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "nc",
        alternate: Some(Alternate {
            atom_type: "nd",
            pass: AlternatePass::Conjugated,
        }),
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
                        prop: AtomProp::SbAny,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::DbAny,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::Ar2,
                        relation: None,
                    }],
                },
            ],
        }),
        environment: Some(&[AtomPattern {
            atom: PatternAtom::Wild(WILD_XD),
            degree: Some(3),
            property: Some(PropExpr {
                constraints: &[
                    PropConstraint {
                        units: &[PropUnit {
                            count: None,
                            prop: AtomProp::SbAny,
                            relation: Some(PropRelation::BondedToPrev),
                        }],
                    },
                    PropConstraint {
                        units: &[PropUnit {
                            count: None,
                            prop: AtomProp::DbAny,
                            relation: None,
                        }],
                    },
                ],
            }),
            label: None,
            children: &[],
        }]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "nc",
        alternate: Some(Alternate {
            atom_type: "nd",
            pass: AlternatePass::Conjugated,
        }),
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
                        prop: AtomProp::SbAny,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::DbAny,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::Ar2,
                        relation: None,
                    }],
                },
            ],
        }),
        environment: Some(&[AtomPattern {
            atom: PatternAtom::Wild(WILD_XD),
            degree: Some(4),
            property: Some(PropExpr {
                constraints: &[
                    PropConstraint {
                        units: &[PropUnit {
                            count: None,
                            prop: AtomProp::SbAny,
                            relation: Some(PropRelation::BondedToPrev),
                        }],
                    },
                    PropConstraint {
                        units: &[PropUnit {
                            count: None,
                            prop: AtomProp::DbAny,
                            relation: None,
                        }],
                    },
                ],
            }),
            label: None,
            children: &[],
        }]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "nc",
        alternate: Some(Alternate {
            atom_type: "nd",
            pass: AlternatePass::Conjugated,
        }),
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
                        prop: AtomProp::SbAny,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::DbAny,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::Ar3,
                        relation: None,
                    }],
                },
            ],
        }),
        environment: Some(&[AtomPattern {
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
        }]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "nc",
        alternate: Some(Alternate {
            atom_type: "nd",
            pass: AlternatePass::Conjugated,
        }),
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
                        prop: AtomProp::SbAny,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::DbAny,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::Ar3,
                        relation: None,
                    }],
                },
            ],
        }),
        environment: Some(&[AtomPattern {
            atom: PatternAtom::Element(6),
            degree: Some(3),
            property: None,
            label: None,
            children: &[AtomPattern {
                atom: PatternAtom::Element(6),
                degree: Some(2),
                property: None,
                label: None,
                children: &[],
            }],
        }]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "nc",
        alternate: Some(Alternate {
            atom_type: "nd",
            pass: AlternatePass::Conjugated,
        }),
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
                        prop: AtomProp::SbAny,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::DbAny,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::Ar3,
                        relation: None,
                    }],
                },
            ],
        }),
        environment: Some(&[AtomPattern {
            atom: PatternAtom::Element(6),
            degree: Some(3),
            property: None,
            label: None,
            children: &[AtomPattern {
                atom: PatternAtom::Wild(WILD_XB),
                degree: Some(2),
                property: None,
                label: None,
                children: &[],
            }],
        }]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "nc",
        alternate: Some(Alternate {
            atom_type: "nd",
            pass: AlternatePass::Conjugated,
        }),
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
                        prop: AtomProp::SbAny,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::DbAny,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::Ar3,
                        relation: None,
                    }],
                },
            ],
        }),
        environment: Some(&[AtomPattern {
            atom: PatternAtom::Wild(WILD_XB),
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
        }]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "nc",
        alternate: Some(Alternate {
            atom_type: "nd",
            pass: AlternatePass::Conjugated,
        }),
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
                        prop: AtomProp::SbAny,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::DbAny,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::Ar3,
                        relation: None,
                    }],
                },
            ],
        }),
        environment: Some(&[AtomPattern {
            atom: PatternAtom::Wild(WILD_XB),
            degree: Some(2),
            property: None,
            label: None,
            children: &[AtomPattern {
                atom: PatternAtom::Element(6),
                degree: Some(2),
                property: None,
                label: None,
                children: &[],
            }],
        }]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "nc",
        alternate: Some(Alternate {
            atom_type: "nd",
            pass: AlternatePass::Conjugated,
        }),
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
                        prop: AtomProp::SbAny,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::DbAny,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::Ar3,
                        relation: None,
                    }],
                },
            ],
        }),
        environment: Some(&[AtomPattern {
            atom: PatternAtom::Wild(WILD_XB),
            degree: Some(2),
            property: None,
            label: None,
            children: &[AtomPattern {
                atom: PatternAtom::Wild(WILD_XB),
                degree: Some(2),
                property: None,
                label: None,
                children: &[],
            }],
        }]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "nc",
        alternate: Some(Alternate {
            atom_type: "nd",
            pass: AlternatePass::Conjugated,
        }),
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
                        prop: AtomProp::SbAny,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::DbAny,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::Ar3,
                        relation: None,
                    }],
                },
            ],
        }),
        environment: Some(&[AtomPattern {
            atom: PatternAtom::Element(6),
            degree: Some(3),
            property: Some(PropExpr {
                constraints: &[PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::SbAny,
                        relation: Some(PropRelation::BondedToPrev),
                    }],
                }],
            }),
            label: None,
            children: &[],
        }]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "nc",
        alternate: Some(Alternate {
            atom_type: "nd",
            pass: AlternatePass::Conjugated,
        }),
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
                        prop: AtomProp::SbAny,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::DbAny,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::Ar3,
                        relation: None,
                    }],
                },
            ],
        }),
        environment: Some(&[AtomPattern {
            atom: PatternAtom::Wild(WILD_XB),
            degree: Some(2),
            property: Some(PropExpr {
                constraints: &[PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::SbAny,
                        relation: Some(PropRelation::BondedToPrev),
                    }],
                }],
            }),
            label: None,
            children: &[],
        }]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "nc",
        alternate: Some(Alternate {
            atom_type: "nd",
            pass: AlternatePass::Conjugated,
        }),
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
                        prop: AtomProp::SbAny,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::DbAny,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::Ar3,
                        relation: None,
                    }],
                },
            ],
        }),
        environment: Some(&[AtomPattern {
            atom: PatternAtom::Wild(WILD_XD),
            degree: Some(3),
            property: Some(PropExpr {
                constraints: &[
                    PropConstraint {
                        units: &[PropUnit {
                            count: None,
                            prop: AtomProp::SbAny,
                            relation: Some(PropRelation::BondedToPrev),
                        }],
                    },
                    PropConstraint {
                        units: &[PropUnit {
                            count: None,
                            prop: AtomProp::DbAny,
                            relation: None,
                        }],
                    },
                ],
            }),
            label: None,
            children: &[],
        }]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "nc",
        alternate: Some(Alternate {
            atom_type: "nd",
            pass: AlternatePass::Conjugated,
        }),
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
                        prop: AtomProp::SbAny,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::DbAny,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::Ar3,
                        relation: None,
                    }],
                },
            ],
        }),
        environment: Some(&[AtomPattern {
            atom: PatternAtom::Wild(WILD_XD),
            degree: Some(4),
            property: Some(PropExpr {
                constraints: &[
                    PropConstraint {
                        units: &[PropUnit {
                            count: None,
                            prop: AtomProp::SbAny,
                            relation: Some(PropRelation::BondedToPrev),
                        }],
                    },
                    PropConstraint {
                        units: &[PropUnit {
                            count: None,
                            prop: AtomProp::DbAny,
                            relation: None,
                        }],
                    },
                ],
            }),
            label: None,
            children: &[],
        }]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "nc",
        alternate: Some(Alternate {
            atom_type: "nd",
            pass: AlternatePass::Conjugated,
        }),
        residue: "*",
        atomic_number: Some(7),
        degree: Some(2),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: Some(PropExpr {
            constraints: &[PropConstraint {
                units: &[PropUnit {
                    count: None,
                    prop: AtomProp::Rg5,
                    relation: None,
                }],
            }],
        }),
        environment: Some(&[AtomPattern {
            atom: PatternAtom::Element(7),
            degree: Some(2),
            property: Some(PropExpr {
                constraints: &[PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::Rg5,
                        relation: None,
                    }],
                }],
            }),
            label: None,
            children: &[AtomPattern {
                atom: PatternAtom::Element(7),
                degree: Some(2),
                property: Some(PropExpr {
                    constraints: &[PropConstraint {
                        units: &[PropUnit {
                            count: None,
                            prop: AtomProp::Rg5,
                            relation: None,
                        }],
                    }],
                }),
                label: None,
                children: &[AtomPattern {
                    atom: PatternAtom::Element(7),
                    degree: Some(2),
                    property: Some(PropExpr {
                        constraints: &[PropConstraint {
                            units: &[PropUnit {
                                count: None,
                                prop: AtomProp::Rg5,
                                relation: None,
                            }],
                        }],
                    }),
                    label: None,
                    children: &[],
                }],
            }],
        }]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "nc",
        alternate: Some(Alternate {
            atom_type: "nd",
            pass: AlternatePass::Conjugated,
        }),
        residue: "*",
        atomic_number: Some(7),
        degree: Some(2),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: Some(PropExpr {
            constraints: &[PropConstraint {
                units: &[PropUnit {
                    count: None,
                    prop: AtomProp::Rg5,
                    relation: None,
                }],
            }],
        }),
        environment: Some(&[
            AtomPattern {
                atom: PatternAtom::Element(7),
                degree: Some(2),
                property: Some(PropExpr {
                    constraints: &[PropConstraint {
                        units: &[PropUnit {
                            count: None,
                            prop: AtomProp::Rg5,
                            relation: None,
                        }],
                    }],
                }),
                label: None,
                children: &[],
            },
            AtomPattern {
                atom: PatternAtom::Element(7),
                degree: Some(2),
                property: Some(PropExpr {
                    constraints: &[PropConstraint {
                        units: &[PropUnit {
                            count: None,
                            prop: AtomProp::Rg5,
                            relation: None,
                        }],
                    }],
                }),
                label: None,
                children: &[AtomPattern {
                    atom: PatternAtom::Element(7),
                    degree: Some(2),
                    property: Some(PropExpr {
                        constraints: &[PropConstraint {
                            units: &[PropUnit {
                                count: None,
                                prop: AtomProp::Rg5,
                                relation: None,
                            }],
                        }],
                    }),
                    label: None,
                    children: &[],
                }],
            },
        ]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "ne",
        alternate: Some(Alternate {
            atom_type: "nf",
            pass: AlternatePass::Conjugated,
        }),
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
                        prop: AtomProp::SbAny,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::DbAny,
                        relation: None,
                    }],
                },
            ],
        }),
        environment: Some(&[AtomPattern {
            atom: PatternAtom::Element(6),
            degree: Some(2),
            property: Some(PropExpr {
                constraints: &[PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::SbStrict,
                        relation: Some(PropRelation::BondedToPrev),
                    }],
                }],
            }),
            label: None,
            children: &[],
        }]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "ne",
        alternate: Some(Alternate {
            atom_type: "nf",
            pass: AlternatePass::Conjugated,
        }),
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
                        prop: AtomProp::SbAny,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::DbAny,
                        relation: None,
                    }],
                },
            ],
        }),
        environment: Some(&[AtomPattern {
            atom: PatternAtom::Element(6),
            degree: Some(3),
            property: Some(PropExpr {
                constraints: &[PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::SbStrict,
                        relation: Some(PropRelation::BondedToPrev),
                    }],
                }],
            }),
            label: None,
            children: &[],
        }]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "ne",
        alternate: Some(Alternate {
            atom_type: "nf",
            pass: AlternatePass::Conjugated,
        }),
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
                        prop: AtomProp::SbAny,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::DbAny,
                        relation: None,
                    }],
                },
            ],
        }),
        environment: Some(&[AtomPattern {
            atom: PatternAtom::Wild(WILD_XA),
            degree: Some(1),
            property: Some(PropExpr {
                constraints: &[PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::SbStrict,
                        relation: Some(PropRelation::BondedToPrev),
                    }],
                }],
            }),
            label: None,
            children: &[],
        }]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "ne",
        alternate: Some(Alternate {
            atom_type: "nf",
            pass: AlternatePass::Conjugated,
        }),
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
                        prop: AtomProp::SbAny,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::DbAny,
                        relation: None,
                    }],
                },
            ],
        }),
        environment: Some(&[AtomPattern {
            atom: PatternAtom::Wild(WILD_XB),
            degree: Some(2),
            property: Some(PropExpr {
                constraints: &[PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::SbStrict,
                        relation: Some(PropRelation::BondedToPrev),
                    }],
                }],
            }),
            label: None,
            children: &[],
        }]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "ne",
        alternate: Some(Alternate {
            atom_type: "nf",
            pass: AlternatePass::Conjugated,
        }),
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
                        prop: AtomProp::SbAny,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::DbAny,
                        relation: None,
                    }],
                },
            ],
        }),
        environment: Some(&[AtomPattern {
            atom: PatternAtom::Wild(WILD_XD),
            degree: Some(3),
            property: Some(PropExpr {
                constraints: &[
                    PropConstraint {
                        units: &[PropUnit {
                            count: None,
                            prop: AtomProp::SbStrict,
                            relation: Some(PropRelation::BondedToPrev),
                        }],
                    },
                    PropConstraint {
                        units: &[PropUnit {
                            count: None,
                            prop: AtomProp::DbAny,
                            relation: None,
                        }],
                    },
                ],
            }),
            label: None,
            children: &[],
        }]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "ne",
        alternate: Some(Alternate {
            atom_type: "nf",
            pass: AlternatePass::Conjugated,
        }),
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
                        prop: AtomProp::SbAny,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::DbAny,
                        relation: None,
                    }],
                },
            ],
        }),
        environment: Some(&[AtomPattern {
            atom: PatternAtom::Wild(WILD_XD),
            degree: Some(4),
            property: Some(PropExpr {
                constraints: &[
                    PropConstraint {
                        units: &[PropUnit {
                            count: None,
                            prop: AtomProp::SbStrict,
                            relation: Some(PropRelation::BondedToPrev),
                        }],
                    },
                    PropConstraint {
                        units: &[PropUnit {
                            count: None,
                            prop: AtomProp::DbAny,
                            relation: None,
                        }],
                    },
                ],
            }),
            label: None,
            children: &[],
        }]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "n1",
        alternate: None,
        residue: "*",
        atomic_number: Some(7),
        degree: Some(2),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: Some(PropExpr {
            constraints: &[PropConstraint {
                units: &[PropUnit {
                    count: Some(2),
                    prop: AtomProp::DbAny,
                    relation: None,
                }],
            }],
        }),
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "n1",
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
                        prop: AtomProp::TbAny,
                        relation: None,
                    }],
                },
                PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::SbAny,
                        relation: None,
                    }],
                },
            ],
        }),
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "n2",
        alternate: None,
        residue: "*",
        atomic_number: Some(7),
        degree: Some(2),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "n1",
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
        atom_type: "o",
        alternate: None,
        residue: "*",
        atomic_number: Some(8),
        degree: Some(1),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "oh",
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
        atom_type: "oh",
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
        atom_type: "oh",
        alternate: None,
        residue: "*",
        atomic_number: Some(8),
        degree: Some(3),
        hydrogen_count: Some(1),
        ewd_count: None,
        atom_property: None,
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "oh",
        alternate: None,
        residue: "*",
        atomic_number: Some(8),
        degree: Some(3),
        hydrogen_count: Some(2),
        ewd_count: None,
        atom_property: None,
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "oh",
        alternate: None,
        residue: "*",
        atomic_number: Some(8),
        degree: Some(3),
        hydrogen_count: Some(3),
        ewd_count: None,
        atom_property: None,
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "op",
        alternate: None,
        residue: "*",
        atomic_number: Some(8),
        degree: Some(2),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: Some(PropExpr {
            constraints: &[PropConstraint {
                units: &[PropUnit {
                    count: None,
                    prop: AtomProp::Rg3,
                    relation: None,
                }],
            }],
        }),
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "oq",
        alternate: None,
        residue: "*",
        atomic_number: Some(8),
        degree: Some(2),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: Some(PropExpr {
            constraints: &[PropConstraint {
                units: &[PropUnit {
                    count: None,
                    prop: AtomProp::Rg4,
                    relation: None,
                }],
            }],
        }),
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "os",
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
        atom_type: "os",
        alternate: None,
        residue: "*",
        atomic_number: Some(8),
        degree: Some(3),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "os",
        alternate: None,
        residue: "*",
        atomic_number: Some(8),
        degree: None,
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "s",
        alternate: None,
        residue: "*",
        atomic_number: Some(16),
        degree: Some(1),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "s2",
        alternate: None,
        residue: "*",
        atomic_number: Some(16),
        degree: Some(2),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: Some(PropExpr {
            constraints: &[PropConstraint {
                units: &[PropUnit {
                    count: None,
                    prop: AtomProp::DbStrict,
                    relation: None,
                }],
            }],
        }),
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "s2",
        alternate: None,
        residue: "*",
        atomic_number: Some(16),
        degree: Some(2),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: Some(PropExpr {
            constraints: &[PropConstraint {
                units: &[PropUnit {
                    count: None,
                    prop: AtomProp::TbStrict,
                    relation: None,
                }],
            }],
        }),
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "sh",
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
        atom_type: "sh",
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
        atom_type: "sp",
        alternate: None,
        residue: "*",
        atomic_number: Some(16),
        degree: Some(2),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: Some(PropExpr {
            constraints: &[PropConstraint {
                units: &[PropUnit {
                    count: None,
                    prop: AtomProp::Rg3,
                    relation: None,
                }],
            }],
        }),
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "sq",
        alternate: None,
        residue: "*",
        atomic_number: Some(16),
        degree: Some(2),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: Some(PropExpr {
            constraints: &[PropConstraint {
                units: &[PropUnit {
                    count: None,
                    prop: AtomProp::Rg4,
                    relation: None,
                }],
            }],
        }),
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "ss",
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
        atom_type: "sx",
        alternate: None,
        residue: "*",
        atomic_number: Some(16),
        degree: Some(3),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: Some(PropExpr {
            constraints: &[PropConstraint {
                units: &[PropUnit {
                    count: None,
                    prop: AtomProp::DbAny,
                    relation: None,
                }],
            }],
        }),
        environment: Some(&[AtomPattern {
            atom: PatternAtom::Wild(WILD_XB),
            degree: Some(2),
            property: Some(PropExpr {
                constraints: &[PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::SbAny,
                        relation: Some(PropRelation::BondedToPrev),
                    }],
                }],
            }),
            label: None,
            children: &[],
        }]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "sx",
        alternate: None,
        residue: "*",
        atomic_number: Some(16),
        degree: Some(3),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: Some(PropExpr {
            constraints: &[PropConstraint {
                units: &[PropUnit {
                    count: None,
                    prop: AtomProp::DbAny,
                    relation: None,
                }],
            }],
        }),
        environment: Some(&[AtomPattern {
            atom: PatternAtom::Element(6),
            degree: Some(3),
            property: Some(PropExpr {
                constraints: &[PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::SbAny,
                        relation: Some(PropRelation::BondedToPrev),
                    }],
                }],
            }),
            label: None,
            children: &[],
        }]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "sx",
        alternate: None,
        residue: "*",
        atomic_number: Some(16),
        degree: Some(3),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: Some(PropExpr {
            constraints: &[PropConstraint {
                units: &[PropUnit {
                    count: None,
                    prop: AtomProp::DbAny,
                    relation: None,
                }],
            }],
        }),
        environment: Some(&[AtomPattern {
            atom: PatternAtom::Wild(WILD_XD),
            degree: Some(3),
            property: Some(PropExpr {
                constraints: &[
                    PropConstraint {
                        units: &[PropUnit {
                            count: None,
                            prop: AtomProp::SbAny,
                            relation: Some(PropRelation::BondedToPrev),
                        }],
                    },
                    PropConstraint {
                        units: &[PropUnit {
                            count: None,
                            prop: AtomProp::DbAny,
                            relation: None,
                        }],
                    },
                ],
            }),
            label: None,
            children: &[],
        }]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "sx",
        alternate: None,
        residue: "*",
        atomic_number: Some(16),
        degree: Some(3),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: Some(PropExpr {
            constraints: &[PropConstraint {
                units: &[PropUnit {
                    count: None,
                    prop: AtomProp::DbAny,
                    relation: None,
                }],
            }],
        }),
        environment: Some(&[AtomPattern {
            atom: PatternAtom::Wild(WILD_XD),
            degree: Some(4),
            property: Some(PropExpr {
                constraints: &[
                    PropConstraint {
                        units: &[PropUnit {
                            count: None,
                            prop: AtomProp::SbAny,
                            relation: Some(PropRelation::BondedToPrev),
                        }],
                    },
                    PropConstraint {
                        units: &[PropUnit {
                            count: None,
                            prop: AtomProp::DbAny,
                            relation: None,
                        }],
                    },
                ],
            }),
            label: None,
            children: &[],
        }]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "s4",
        alternate: None,
        residue: "*",
        atomic_number: Some(16),
        degree: Some(3),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "sy",
        alternate: None,
        residue: "*",
        atomic_number: Some(16),
        degree: Some(4),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: Some(PropExpr {
            constraints: &[PropConstraint {
                units: &[PropUnit {
                    count: None,
                    prop: AtomProp::DbAny,
                    relation: None,
                }],
            }],
        }),
        environment: Some(&[AtomPattern {
            atom: PatternAtom::Wild(WILD_XB),
            degree: Some(2),
            property: Some(PropExpr {
                constraints: &[PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::SbAny,
                        relation: Some(PropRelation::BondedToPrev),
                    }],
                }],
            }),
            label: None,
            children: &[],
        }]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "sy",
        alternate: None,
        residue: "*",
        atomic_number: Some(16),
        degree: Some(4),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: Some(PropExpr {
            constraints: &[PropConstraint {
                units: &[PropUnit {
                    count: None,
                    prop: AtomProp::DbAny,
                    relation: None,
                }],
            }],
        }),
        environment: Some(&[AtomPattern {
            atom: PatternAtom::Element(6),
            degree: Some(3),
            property: Some(PropExpr {
                constraints: &[PropConstraint {
                    units: &[PropUnit {
                        count: None,
                        prop: AtomProp::SbAny,
                        relation: Some(PropRelation::BondedToPrev),
                    }],
                }],
            }),
            label: None,
            children: &[],
        }]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "sy",
        alternate: None,
        residue: "*",
        atomic_number: Some(16),
        degree: Some(4),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: Some(PropExpr {
            constraints: &[PropConstraint {
                units: &[PropUnit {
                    count: None,
                    prop: AtomProp::DbAny,
                    relation: None,
                }],
            }],
        }),
        environment: Some(&[AtomPattern {
            atom: PatternAtom::Wild(WILD_XD),
            degree: Some(3),
            property: Some(PropExpr {
                constraints: &[
                    PropConstraint {
                        units: &[PropUnit {
                            count: None,
                            prop: AtomProp::SbAny,
                            relation: Some(PropRelation::BondedToPrev),
                        }],
                    },
                    PropConstraint {
                        units: &[PropUnit {
                            count: None,
                            prop: AtomProp::DbAny,
                            relation: None,
                        }],
                    },
                ],
            }),
            label: None,
            children: &[],
        }]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "sy",
        alternate: None,
        residue: "*",
        atomic_number: Some(16),
        degree: Some(4),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: Some(PropExpr {
            constraints: &[PropConstraint {
                units: &[PropUnit {
                    count: None,
                    prop: AtomProp::DbAny,
                    relation: None,
                }],
            }],
        }),
        environment: Some(&[AtomPattern {
            atom: PatternAtom::Wild(WILD_XD),
            degree: Some(4),
            property: Some(PropExpr {
                constraints: &[
                    PropConstraint {
                        units: &[PropUnit {
                            count: None,
                            prop: AtomProp::SbAny,
                            relation: Some(PropRelation::BondedToPrev),
                        }],
                    },
                    PropConstraint {
                        units: &[PropUnit {
                            count: None,
                            prop: AtomProp::DbAny,
                            relation: None,
                        }],
                    },
                ],
            }),
            label: None,
            children: &[],
        }]),
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "s6",
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
        atom_type: "s6",
        alternate: None,
        residue: "*",
        atomic_number: Some(16),
        degree: Some(5),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "s6",
        alternate: None,
        residue: "*",
        atomic_number: Some(16),
        degree: Some(6),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "He",
        alternate: None,
        residue: "*",
        atomic_number: Some(2),
        degree: None,
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "Li",
        alternate: None,
        residue: "*",
        atomic_number: Some(3),
        degree: None,
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "Be",
        alternate: None,
        residue: "*",
        atomic_number: Some(4),
        degree: None,
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "B",
        alternate: None,
        residue: "*",
        atomic_number: Some(5),
        degree: None,
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "Ne",
        alternate: None,
        residue: "*",
        atomic_number: Some(10),
        degree: None,
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "Na",
        alternate: None,
        residue: "*",
        atomic_number: Some(11),
        degree: None,
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "Mg",
        alternate: None,
        residue: "*",
        atomic_number: Some(12),
        degree: None,
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "Al",
        alternate: None,
        residue: "*",
        atomic_number: Some(13),
        degree: None,
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "Si",
        alternate: None,
        residue: "*",
        atomic_number: Some(14),
        degree: None,
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "Ar",
        alternate: None,
        residue: "*",
        atomic_number: Some(18),
        degree: None,
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "K",
        alternate: None,
        residue: "*",
        atomic_number: Some(19),
        degree: None,
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "Ca",
        alternate: None,
        residue: "*",
        atomic_number: Some(20),
        degree: None,
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "Sc",
        alternate: None,
        residue: "*",
        atomic_number: Some(21),
        degree: None,
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "Ti",
        alternate: None,
        residue: "*",
        atomic_number: Some(22),
        degree: None,
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "V",
        alternate: None,
        residue: "*",
        atomic_number: Some(23),
        degree: None,
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "Cr",
        alternate: None,
        residue: "*",
        atomic_number: Some(24),
        degree: None,
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "Mn",
        alternate: None,
        residue: "*",
        atomic_number: Some(25),
        degree: None,
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "Fe",
        alternate: None,
        residue: "*",
        atomic_number: Some(26),
        degree: None,
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "Co",
        alternate: None,
        residue: "*",
        atomic_number: Some(27),
        degree: None,
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "Ni",
        alternate: None,
        residue: "*",
        atomic_number: Some(28),
        degree: None,
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "Cu",
        alternate: None,
        residue: "*",
        atomic_number: Some(29),
        degree: None,
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "Zn",
        alternate: None,
        residue: "*",
        atomic_number: Some(30),
        degree: None,
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "Ga",
        alternate: None,
        residue: "*",
        atomic_number: Some(31),
        degree: None,
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "Ge",
        alternate: None,
        residue: "*",
        atomic_number: Some(32),
        degree: None,
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "As",
        alternate: None,
        residue: "*",
        atomic_number: Some(33),
        degree: None,
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "Se",
        alternate: None,
        residue: "*",
        atomic_number: Some(34),
        degree: None,
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "Kr",
        alternate: None,
        residue: "*",
        atomic_number: Some(36),
        degree: None,
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "Rb",
        alternate: None,
        residue: "*",
        atomic_number: Some(37),
        degree: None,
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "Sr",
        alternate: None,
        residue: "*",
        atomic_number: Some(38),
        degree: None,
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "Y",
        alternate: None,
        residue: "*",
        atomic_number: Some(39),
        degree: None,
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "Zr",
        alternate: None,
        residue: "*",
        atomic_number: Some(40),
        degree: None,
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "Nb",
        alternate: None,
        residue: "*",
        atomic_number: Some(41),
        degree: None,
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "Mo",
        alternate: None,
        residue: "*",
        atomic_number: Some(42),
        degree: None,
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "Tc",
        alternate: None,
        residue: "*",
        atomic_number: Some(43),
        degree: None,
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "Ru",
        alternate: None,
        residue: "*",
        atomic_number: Some(44),
        degree: None,
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "Rh",
        alternate: None,
        residue: "*",
        atomic_number: Some(45),
        degree: None,
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "Pd",
        alternate: None,
        residue: "*",
        atomic_number: Some(46),
        degree: None,
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "Ag",
        alternate: None,
        residue: "*",
        atomic_number: Some(47),
        degree: None,
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "Cd",
        alternate: None,
        residue: "*",
        atomic_number: Some(48),
        degree: None,
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "In",
        alternate: None,
        residue: "*",
        atomic_number: Some(49),
        degree: None,
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "Sn",
        alternate: None,
        residue: "*",
        atomic_number: Some(50),
        degree: None,
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "Sb",
        alternate: None,
        residue: "*",
        atomic_number: Some(51),
        degree: None,
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "Te",
        alternate: None,
        residue: "*",
        atomic_number: Some(52),
        degree: None,
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "Xe",
        alternate: None,
        residue: "*",
        atomic_number: Some(54),
        degree: None,
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "Cs",
        alternate: None,
        residue: "*",
        atomic_number: Some(55),
        degree: None,
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "Ba",
        alternate: None,
        residue: "*",
        atomic_number: Some(56),
        degree: None,
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "La",
        alternate: None,
        residue: "*",
        atomic_number: Some(57),
        degree: None,
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "Ce",
        alternate: None,
        residue: "*",
        atomic_number: Some(58),
        degree: None,
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "Pr",
        alternate: None,
        residue: "*",
        atomic_number: Some(59),
        degree: None,
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "Nd",
        alternate: None,
        residue: "*",
        atomic_number: Some(60),
        degree: None,
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "Pm",
        alternate: None,
        residue: "*",
        atomic_number: Some(61),
        degree: None,
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "Sm",
        alternate: None,
        residue: "*",
        atomic_number: Some(62),
        degree: None,
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "Eu",
        alternate: None,
        residue: "*",
        atomic_number: Some(63),
        degree: None,
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "Gd",
        alternate: None,
        residue: "*",
        atomic_number: Some(64),
        degree: None,
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "Tb",
        alternate: None,
        residue: "*",
        atomic_number: Some(65),
        degree: None,
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "Dy",
        alternate: None,
        residue: "*",
        atomic_number: Some(66),
        degree: None,
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "Ho",
        alternate: None,
        residue: "*",
        atomic_number: Some(67),
        degree: None,
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "Er",
        alternate: None,
        residue: "*",
        atomic_number: Some(68),
        degree: None,
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "Tm",
        alternate: None,
        residue: "*",
        atomic_number: Some(69),
        degree: None,
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "Yb",
        alternate: None,
        residue: "*",
        atomic_number: Some(70),
        degree: None,
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "Lu",
        alternate: None,
        residue: "*",
        atomic_number: Some(71),
        degree: None,
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "Hf",
        alternate: None,
        residue: "*",
        atomic_number: Some(72),
        degree: None,
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "Ta",
        alternate: None,
        residue: "*",
        atomic_number: Some(73),
        degree: None,
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "W",
        alternate: None,
        residue: "*",
        atomic_number: Some(74),
        degree: None,
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "Re",
        alternate: None,
        residue: "*",
        atomic_number: Some(75),
        degree: None,
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "Os",
        alternate: None,
        residue: "*",
        atomic_number: Some(76),
        degree: None,
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "Ir",
        alternate: None,
        residue: "*",
        atomic_number: Some(77),
        degree: None,
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "Pt",
        alternate: None,
        residue: "*",
        atomic_number: Some(78),
        degree: None,
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "Au",
        alternate: None,
        residue: "*",
        atomic_number: Some(79),
        degree: None,
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "Hg",
        alternate: None,
        residue: "*",
        atomic_number: Some(80),
        degree: None,
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "Tl",
        alternate: None,
        residue: "*",
        atomic_number: Some(81),
        degree: None,
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "Pb",
        alternate: None,
        residue: "*",
        atomic_number: Some(82),
        degree: None,
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "Bi",
        alternate: None,
        residue: "*",
        atomic_number: Some(83),
        degree: None,
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "Po",
        alternate: None,
        residue: "*",
        atomic_number: Some(84),
        degree: None,
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "At",
        alternate: None,
        residue: "*",
        atomic_number: Some(85),
        degree: None,
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "Rn",
        alternate: None,
        residue: "*",
        atomic_number: Some(86),
        degree: None,
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "Fr",
        alternate: None,
        residue: "*",
        atomic_number: Some(87),
        degree: None,
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "Ra",
        alternate: None,
        residue: "*",
        atomic_number: Some(88),
        degree: None,
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "Ac",
        alternate: None,
        residue: "*",
        atomic_number: Some(89),
        degree: None,
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "Th",
        alternate: None,
        residue: "*",
        atomic_number: Some(90),
        degree: None,
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "Pa",
        alternate: None,
        residue: "*",
        atomic_number: Some(91),
        degree: None,
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "U",
        alternate: None,
        residue: "*",
        atomic_number: Some(92),
        degree: None,
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "Np",
        alternate: None,
        residue: "*",
        atomic_number: Some(93),
        degree: None,
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "Pu",
        alternate: None,
        residue: "*",
        atomic_number: Some(94),
        degree: None,
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "Am",
        alternate: None,
        residue: "*",
        atomic_number: Some(95),
        degree: None,
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "Cm",
        alternate: None,
        residue: "*",
        atomic_number: Some(96),
        degree: None,
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "Bk",
        alternate: None,
        residue: "*",
        atomic_number: Some(97),
        degree: None,
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "Cf",
        alternate: None,
        residue: "*",
        atomic_number: Some(98),
        degree: None,
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "Es",
        alternate: None,
        residue: "*",
        atomic_number: Some(99),
        degree: None,
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "Fm",
        alternate: None,
        residue: "*",
        atomic_number: Some(100),
        degree: None,
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "Md",
        alternate: None,
        residue: "*",
        atomic_number: Some(101),
        degree: None,
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "No",
        alternate: None,
        residue: "*",
        atomic_number: Some(102),
        degree: None,
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "Lr",
        alternate: None,
        residue: "*",
        atomic_number: Some(103),
        degree: None,
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "Rf",
        alternate: None,
        residue: "*",
        atomic_number: Some(104),
        degree: None,
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "Db",
        alternate: None,
        residue: "*",
        atomic_number: Some(105),
        degree: None,
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "Sg",
        alternate: None,
        residue: "*",
        atomic_number: Some(106),
        degree: None,
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "Bh",
        alternate: None,
        residue: "*",
        atomic_number: Some(107),
        degree: None,
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "Hs",
        alternate: None,
        residue: "*",
        atomic_number: Some(108),
        degree: None,
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "Mt",
        alternate: None,
        residue: "*",
        atomic_number: Some(109),
        degree: None,
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "Ds",
        alternate: None,
        residue: "*",
        atomic_number: Some(103),
        degree: None,
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
        degree: Some(1),
        hydrogen_count: None,
        ewd_count: None,
        atom_property: None,
        environment: None,
        environment_bonds: None,
    },
    AtdRule {
        atom_type: "lp",
        alternate: None,
        residue: "*",
        atomic_number: Some(0),
        degree: Some(1),
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

/// `ATOMTYPE_GFF.DEF` as one typed table.
pub const ATOMTYPE_GFF: AtdTable = AtdTable {
    name: "ATOMTYPE_GFF.DEF",
    wildatoms: WILDATOMS,
    rules: RULES,
};
