//! Engine codecs in process (`ff-ir-02-protocol` §8): styles the engines
//! have no molrs arm for — registered at run time with a spec, or built in
//! with an expression — written by molrs's engine writers, read back, and
//! priced by molrs to the same energy.
//!
//! | Case | Format | What it holds |
//! |---|---|---|
//! | `fene` | LAMMPS | a run-time `bond fene` registered with `LammpsForm::Positional`: `bond_style fene` |
//! | `smooth` | LAMMPS | a run-time `pair lj/smooth/linear`, positional, mixed by `pair_modify mix arithmetic` |
//! | `metal` | LAMMPS | built-in `bond morse` and `pair buck` of a `real` field written in `metal`: the conversion per dimension (`E`, `1/L`, `E*L^6`) |
//! | `morse` | LAMMPS | built-in `pair morse`, positional, a row per pair |
//! | `openmm` | OpenMM XML | the `fene` bond (`CustomBondForce`), built-in `bond morse`, `angle class2` and `improper cvff` (`CustomBondForce`, `CustomAngleForce`, `CustomTorsionForce` with `ordering="charmm"`), a run-time compound category `urey_bradley` (`CustomCompoundBondForce`, by the XML's `<Script>`), the `lj/smooth/linear` pair (`CustomNonbondedForce`) |

use crate::core::Cell;
use crate::ff::potential::fixtures::OneShot;
use crate::io::writer::WrittenText;
use std::sync::Arc;

use ndarray::Array1;

use crate::ff::compile::PotentialCompiler;
use crate::ff::forcefield::ForceField;
use crate::ff::ir::{
    CategorySpec, Coordinate, EndpointOrder, LammpsForm, ParamCombination, ParamDimension,
    ParamSpec, ParamValue, SpecialClass, StyleSpec,
};
use crate::ff::ir::{Params, SpecialBonds};
use crate::ff::style_registry::Registry;
use crate::io::lammps::units::LammpsUnitConverter;
use crate::io::{lammps::LammpsForcefieldWriter, openmm_xml::OpenmmXmlWriter};
use molrs::core::Block;
use molrs::core::Frame;
use molrs::core::SimBox;
use molrs::core::TypeLabels;
use molrs::op::{F, Idx};

fn dim(s: &str) -> ParamDimension {
    s.parse().unwrap()
}

/// LAMMPS `bond_style fene` (Kremer–Grest): `K R0 epsilon sigma`, the
/// WCA repulsion inside `2^(1/6) σ`.
pub(crate) fn fene_spec() -> StyleSpec {
    StyleSpec::new("bond", "fene")
        .params(vec![
            ParamSpec::new("k", dim("E/L^2")),
            ParamSpec::new("r0", ParamDimension::LENGTH),
            ParamSpec::new("epsilon", ParamDimension::ENERGY),
            ParamSpec::new("sigma", ParamDimension::LENGTH),
        ])
        .expression(
            "-0.5*k*r0^2*log(1-(r/r0)^2)+step(2^(1/6)*sigma-r)*(4*epsilon*((sigma/r)^12-(sigma/r)^6)+epsilon)",
        )
        .lammps(LammpsForm::positional())
}

/// LAMMPS `pair_style lj/smooth/linear` (shifted force): the 12-6 less its
/// value and slope at the cutoff.
pub(crate) fn smooth_spec() -> StyleSpec {
    StyleSpec::new("pair", "lj/smooth/linear")
        .params(vec![
            ParamSpec::new("epsilon", ParamDimension::ENERGY).mix(ParamCombination::LjEpsilon {
                sigma: "sigma".into(),
            }),
            ParamSpec::new("sigma", ParamDimension::LENGTH).mix(ParamCombination::LjSigma {
                epsilon: "epsilon".into(),
            }),
        ])
        .style_params(vec![
            ParamSpec::new("cutoff", ParamDimension::LENGTH),
            ParamSpec::text("mixing", &["arithmetic", "geometric", "sixthpower"])
                .default_value(ParamValue::Text("arithmetic".into())),
        ])
        .special(SpecialClass::Vdw)
        .expression(
            "4*epsilon*((sigma/r)^12-(sigma/r)^6)-4*epsilon*((sigma/cutoff)^12-(sigma/cutoff)^6)\
             +(r-cutoff)*f; f=4*epsilon*(12*sigma^12/cutoff^13-6*sigma^6/cutoff^7)",
        )
        .lammps(LammpsForm::positional())
}

/// A run-time compound category: the Urey–Bradley 1-3 term of an angle,
/// its own block `urey_bradleys`.
pub(crate) fn urey_bradley() -> (CategorySpec, StyleSpec) {
    (
        CategorySpec::custom(
            "urey_bradley",
            3,
            Coordinate::Compound,
            EndpointOrder::Reversible,
        ),
        StyleSpec::new("urey_bradley", "harmonic")
            .params(vec![
                ParamSpec::new("k", dim("E/L^2")),
                ParamSpec::new("r0", ParamDimension::LENGTH),
            ])
            .expression("k*(distance(p1,p3)-r0)^2"),
    )
}

/// The registry of the cases: the built-ins and the run-time styles.
pub(crate) fn registry() -> Arc<Registry> {
    let mut r = Registry::builtin();
    r.register_style(fene_spec(), None).unwrap();
    r.register_style(smooth_spec(), None).unwrap();
    let (cat, ub) = urey_bradley();
    r.register_category(cat).unwrap();
    r.register_style(ub, None).unwrap();
    Arc::new(r)
}

/// One case: a force field and its frame.
pub(crate) struct Case {
    pub name: &'static str,
    pub ff: ForceField,
    pub frame: Frame,
}

fn strings(v: &[&str]) -> ndarray::ArrayD<String> {
    Array1::from_vec(v.iter().map(|s| (*s).to_owned()).collect()).into_dyn()
}

/// A frame of atoms `types` (masses `mass` by type), `relations` blocks of
/// `(atoms, type)` rows.
/// A relation block: its name and its `(atoms, type)` rows.
type Relation<'a> = (&'a str, &'a [(&'a [Idx], &'a str)]);

fn frame(types: &[&str], mass: &[(&str, F)], relations: &[Relation<'_>]) -> Frame {
    let n = types.len();
    let mut atoms = Block::new();
    atoms.insert("type", strings(types), None).unwrap();
    let m: Vec<F> = types
        .iter()
        .map(|t| mass.iter().find(|(k, _)| k == t).unwrap().1)
        .collect();
    atoms
        .insert("mass", Array1::from_vec(m).into_dyn(), None)
        .unwrap();
    atoms
        .insert("charge", Array1::from_vec(vec![0.0; n]).into_dyn(), None)
        .unwrap();
    let mol: Vec<Idx> = vec![1; n];
    atoms
        .insert("mol_id", Array1::from_vec(mol).into_dyn(), None)
        .unwrap();
    for key in ["x", "y", "z"] {
        atoms
            .insert(key, Array1::from_vec(vec![0.0; n]).into_dyn(), None)
            .unwrap();
    }
    let mut f = Frame::new();
    f.insert("atoms", atoms);
    for (block, rows) in relations {
        let mut b = Block::new();
        let arity = rows[0].0.len();
        for (k, key) in ["atomi", "atomj", "atomk", "atoml"][..arity]
            .iter()
            .enumerate()
        {
            b.insert(
                *key,
                Array1::from_vec(rows.iter().map(|r| r.0[k]).collect()).into_dyn(),
                None,
            )
            .unwrap();
        }
        b.insert(
            "type",
            strings(&rows.iter().map(|r| r.1).collect::<Vec<_>>()),
            None,
        )
        .unwrap();
        f.insert(*block, b);
    }
    f
}

fn atom_types(ff: &mut ForceField, types: &[(&str, F)]) {
    let s = ff.def_style("atom", "full", Params::new()).unwrap();
    for (t, m) in types {
        s.def_type(t, &[], Params::from_iter([("mass", *m), ("charge", 0.0)]))
            .unwrap();
    }
}

/// `frame` with its `pairs` block under `ff`'s special-bonds weights.
///
/// The 1-3 and 1-4 classes are the bond graph's, as an engine's are (OpenMM's
/// `bondCutoff`), not only the rows the frame's `angles` and `dihedrals`
/// blocks happen to hold.
fn with_pairs(mut frame: Frame) -> Frame {
    let pairs = molrs::core::Topology::from_frame(&graph_topology(&frame))
        .unwrap()
        .pairs_block();
    frame.insert("pairs", pairs);
    frame
}

/// `frame` with every angle and proper dihedral of its bond graph as its
/// `angles` and `dihedrals` rows (untyped: for the pair list only).
fn graph_topology(frame: &Frame) -> Frame {
    let mut out = frame.clone();
    let n = frame.get("atoms").unwrap().n_rows().unwrap();
    let mut adj = vec![Vec::new(); n];
    if let Some(b) = frame.get("bonds") {
        let col = |k: &str| b.get(k).unwrap().get::<u64>().unwrap().to_owned();
        let (i, j) = (col("atomi"), col("atomj"));
        for (&a, &b) in i.iter().zip(j.iter()) {
            adj[a as usize].push(b as usize);
            adj[b as usize].push(a as usize);
        }
    }
    let mut angles: Vec<[usize; 3]> = Vec::new();
    for (c, around) in adj.iter().enumerate() {
        for (x, &a) in around.iter().enumerate() {
            for &k in &around[x + 1..] {
                angles.push([a, c, k]);
            }
        }
    }
    let mut dihedrals: Vec<[usize; 4]> = Vec::new();
    for j in 0..n {
        for &k in adj[j].iter().filter(|&&k| k > j) {
            for &i in adj[j].iter().filter(|&&i| i != k) {
                for &l in adj[k].iter().filter(|&&l| l != j && l != i) {
                    dihedrals.push([i, j, k, l]);
                }
            }
        }
    }
    let block = |rows: Vec<Vec<usize>>, keys: &[&str]| {
        let mut b = Block::new();
        for (c, key) in keys.iter().enumerate() {
            let v: Vec<Idx> = rows.iter().map(|r| r[c] as Idx).collect();
            b.insert(*key, Array1::from_vec(v).into_dyn(), None)
                .unwrap();
        }
        b
    };
    out.insert(
        "angles",
        block(
            angles.iter().map(|r| r.to_vec()).collect(),
            &["atomi", "atomj", "atomk"],
        ),
    );
    out.insert(
        "dihedrals",
        block(
            dihedrals.iter().map(|r| r.to_vec()).collect(),
            &["atomi", "atomj", "atomk", "atoml"],
        ),
    );
    out
}

pub(crate) fn cases(reg: &Arc<Registry>) -> Vec<Case> {
    let mut out = Vec::new();
    let masses = [("A", 12.011), ("B", 14.007), ("C", 15.999), ("D", 32.06)];

    // fene: a 4-atom chain, three fene types, two inside the WCA range.
    let mut ff = ForceField::new("fene").with_registry(reg.clone());
    ff.set_units("real");
    ff.set_special_bonds(SpecialBonds {
        lj: [0.0, 1.0, 1.0],
        coul: [0.0, 1.0, 1.0],
    });
    atom_types(&mut ff, &masses[..2]);
    let fene = ff.def_style("bond", "fene", Params::new()).unwrap();
    for (name, ends, k, r0, eps, sigma) in [
        ("A-A", ["A", "A"], 30.0, 1.5, 1.0, 1.0),
        ("A-B", ["A", "B"], 25.0, 1.75, 0.8, 1.1),
        ("B-B", ["B", "B"], 40.0, 1.6, 1.2, 0.95),
    ] {
        fene.def_type(
            name,
            &ends,
            Params::from_iter([("k", k), ("r0", r0), ("epsilon", eps), ("sigma", sigma)]),
        )
        .unwrap();
    }
    out.push(Case {
        name: "fene",
        frame: with_pairs(frame(
            &["A", "A", "B", "B"],
            &masses,
            &[(
                "bonds",
                &[(&[0, 1], "A-A"), (&[1, 2], "A-B"), (&[2, 3], "B-B")],
            )],
        )),
        ff,
    });

    // smooth: five unbonded atoms of two types, every pair inside the cutoff.
    let mut ff = ForceField::new("smooth").with_registry(reg.clone());
    ff.set_units("real");
    ff.set_special_bonds(SpecialBonds {
        lj: [0.0, 0.0, 0.0],
        coul: [0.0, 0.0, 0.0],
    });
    atom_types(&mut ff, &masses[..2]);
    let mut style = Params::from_iter([("cutoff", 8.0)]);
    style.set("mixing", "arithmetic");
    let lj = ff.def_style("pair", "lj/smooth/linear", style).unwrap();
    lj.def_type(
        "A",
        &["A"],
        Params::from_iter([("epsilon", 0.2), ("sigma", 3.1)]),
    )
    .unwrap();
    lj.def_type(
        "B",
        &["B"],
        Params::from_iter([("epsilon", 0.15), ("sigma", 3.6)]),
    )
    .unwrap();
    out.push(Case {
        name: "smooth",
        frame: with_pairs(frame(&["A", "A", "B", "B", "B"], &masses, &[])),
        ff,
    });

    // metal: bond morse and pair buck of a real field, written in metal.
    let mut ff = ForceField::new("metal").with_registry(reg.clone());
    ff.set_units("real");
    ff.set_special_bonds(SpecialBonds {
        lj: [0.0, 1.0, 1.0],
        coul: [0.0, 1.0, 1.0],
    });
    atom_types(&mut ff, &masses[..2]);
    ff.def_style("bond", "morse", Params::new())
        .unwrap()
        .def_type(
            "A-B",
            &["A", "B"],
            Params::from_iter([("d0", 95.6), ("alpha", 2.1), ("r0", 1.53)]),
        )
        .unwrap();
    let buck = ff
        .def_style("pair", "buck", Params::from_iter([("cutoff", 9.0)]))
        .unwrap();
    for (name, ends, a, rho, c) in [
        ("A", vec!["A"], 32000.0, 0.27, 210.0),
        ("B", vec!["B"], 41000.0, 0.26, 180.0),
        ("A-B", vec!["A", "B"], 36000.0, 0.265, 195.0),
    ] {
        buck.def_type(
            name,
            &ends,
            Params::from_iter([("a", a), ("rho", rho), ("c", c)]),
        )
        .unwrap();
    }
    out.push(Case {
        name: "metal",
        frame: with_pairs(frame(
            &["A", "B", "A", "B"],
            &masses,
            &[("bonds", &[(&[0, 1], "A-B"), (&[2, 3], "A-B")])],
        )),
        ff,
    });

    // morse: pair morse, every pair its row (no mixing), four unbonded atoms.
    let mut ff = ForceField::new("morse").with_registry(reg.clone());
    ff.set_units("real");
    ff.set_special_bonds(SpecialBonds {
        lj: [0.0, 0.0, 0.0],
        coul: [0.0, 0.0, 0.0],
    });
    atom_types(&mut ff, &masses[..2]);
    let morse = ff
        .def_style("pair", "morse", Params::from_iter([("cutoff", 10.0)]))
        .unwrap();
    for (name, ends, d0, alpha, r0) in [
        ("A", vec!["A"], 0.25, 1.4, 3.6),
        ("B", vec!["B"], 0.18, 1.2, 3.9),
        ("A-B", vec!["A", "B"], 0.21, 1.3, 3.75),
    ] {
        morse
            .def_type(
                name,
                &ends,
                Params::from_iter([("d0", d0), ("alpha", alpha), ("r0", r0)]),
            )
            .unwrap();
    }
    out.push(Case {
        name: "morse",
        frame: with_pairs(frame(&["A", "B", "A", "B"], &masses, &[])),
        ff,
    });

    // openmm: a chain A-B-B-A, a D on the first B (a centre of three: an
    // improper), and a lone C.
    // 1-2, 1-3 and 1-4 pairs excluded: `bondCutoff="3"`.
    let mut ff = ForceField::new("openmm").with_registry(reg.clone());
    ff.set_units("real");
    ff.set_special_bonds(SpecialBonds {
        lj: [0.0, 0.0, 0.0],
        coul: [0.0, 0.0, 0.0],
    });
    atom_types(&mut ff, &masses);
    ff.def_style("bond", "fene", Params::new())
        .unwrap()
        .def_type(
            "A-B",
            &["A", "B"],
            Params::from_iter([("k", 25.0), ("r0", 1.9), ("epsilon", 0.8), ("sigma", 1.4)]),
        )
        .unwrap();
    ff.def_style("bond", "morse", Params::new())
        .unwrap()
        .def_type(
            "B-B",
            &["B", "B"],
            Params::from_iter([("d0", 95.6), ("alpha", 2.1), ("r0", 1.53)]),
        )
        .unwrap()
        .def_type(
            "B-D",
            &["B", "D"],
            Params::from_iter([("d0", 70.2), ("alpha", 1.9), ("r0", 1.41)]),
        )
        .unwrap();
    // LAMMPS's cvff: the dihedral of the row, its centre first, which
    // OpenMM's `ordering="charmm"` prices as written.
    ff.def_style("improper", "cvff", Params::new())
        .unwrap()
        .def_type(
            "B-A-B-D",
            &["B", "A", "B", "D"],
            Params::from_iter([("k", 1.1), ("sign", -1.0), ("periodicity", 2.0)]),
        )
        .unwrap();
    ff.def_style("angle", "class2", Params::new())
        .unwrap()
        .def_type(
            "A-B-B",
            &["A", "B", "B"],
            Params::from_iter([("theta0", 112.0), ("k2", 40.0), ("k3", -12.0), ("k4", 3.0)]),
        )
        .unwrap();
    ff.def_style("urey_bradley", "harmonic", Params::new())
        .unwrap()
        .def_type(
            "A-B-B",
            &["A", "B", "B"],
            Params::from_iter([("k", 22.5), ("r0", 2.45)]),
        )
        .unwrap();
    let mut style = Params::from_iter([("cutoff", 12.0)]);
    style.set("mixing", "arithmetic");
    let lj = ff.def_style("pair", "lj/smooth/linear", style).unwrap();
    for (t, eps, sigma) in [
        ("A", 0.2, 3.1),
        ("B", 0.15, 3.6),
        ("C", 0.25, 3.0),
        ("D", 0.3, 3.4),
    ] {
        lj.def_type(
            t,
            &[t],
            Params::from_iter([("epsilon", eps), ("sigma", sigma)]),
        )
        .unwrap();
    }
    out.push(Case {
        name: "openmm",
        frame: with_pairs(frame(
            &["A", "B", "B", "A", "C", "D"],
            &masses,
            &[
                (
                    "bonds",
                    &[
                        (&[0, 1], "A-B"),
                        (&[1, 2], "B-B"),
                        (&[2, 3], "A-B"),
                        (&[1, 5], "B-D"),
                    ],
                ),
                ("impropers", &[(&[1, 0, 2, 5], "B-A-B-D")]),
                ("angles", &[(&[0, 1, 2], "A-B-B"), (&[1, 2, 3], "A-B-B")]),
                (
                    "urey_bradleys",
                    &[(&[0, 1, 2], "A-B-B"), (&[1, 2, 3], "A-B-B")],
                ),
            ],
        )),
        ff,
    });
    out
}

/// `frame` with its atoms at `x`, in a box past every atom
/// (non-periodic).
fn at(frame: &Frame, x: &[F]) -> Frame {
    let mut out = frame.clone();
    let atoms = out.get_mut("atoms").unwrap();
    for (d, key) in ["x", "y", "z"].iter().enumerate() {
        let col: Vec<F> = x.iter().skip(d).step_by(3).copied().collect();
        atoms
            .insert(*key, Array1::from_vec(col).into_dyn(), None)
            .unwrap();
    }
    out.simbox =
        Some(SimBox::new(Cell::Lengths([100.0; 3]), [-50.0, -50.0, -50.0], [false; 3]).unwrap());
    out
}

// ── the codecs, in process ──────────────────────────────────────────────────

#[cfg(test)]
mod codecs {
    use super::*;
    use crate::ff::ir::expression::{Binding, Geometry, compile};
    use crate::ff::ir::{Engine, IrError, UnitScale, builtin_styles};
    use crate::io::amber::frcmod::AmberFrcmodWriter;
    use crate::io::gromacs::top_writer::GromacsTopForcefieldWriter;
    use crate::io::lammps::forcefield_reader::LammpsForcefieldReader;
    use crate::io::reader::ForceFieldReader;

    /// A row of `spec` a LAMMPS line holds: each parameter a value of its
    /// kind, the multiplicities and signs integers.
    fn sample(spec: &StyleSpec) -> Params {
        let pairs: &[(&str, F)] = match (spec.category.as_ref(), spec.name.as_ref()) {
            ("dihedral", "periodic") => &[
                ("k1", 0.5),
                ("periodicity1", 1.0),
                ("phase1", 180.0),
                ("k2", 0.25),
                ("periodicity2", 3.0),
                ("phase2", 0.0),
            ],
            ("dihedral", "nharmonic") => &[("a1", 1.0), ("a2", -2.0), ("a3", 0.5)],
            ("dihedral", "harmonic") | ("improper", "cvff") => {
                &[("k", 2.0), ("sign", -1.0), ("periodicity", 3.0)]
            }
            ("dihedral", "charmm") => &[
                ("k", 0.2),
                ("periodicity", 3.0),
                ("phase", 180.0),
                // w > 0 needs a lj/charmm beside it, which this one-style
                // field has not; the codec writes it the same either way.
                ("w", 0.0),
            ],
            ("improper", "periodic") => &[("k", 1.1), ("periodicity", 2.0), ("phase", 180.0)],
            _ => &[],
        };
        if !pairs.is_empty() {
            return Params::from_iter(pairs.iter().copied());
        }
        let mut p = Params::new();
        for (i, ps) in spec.params.iter().enumerate() {
            let v = if ps.dim == ParamDimension::ANGLE {
                100.0 + 5.0 * i as F
            } else {
                0.5 + 0.25 * i as F
            };
            p.set(&ps.name, v);
        }
        p
    }

    /// The built-in styles a LAMMPS coefficient line holds (not `fix cmap`,
    /// not the Coulomb halves).
    fn coefficient_styles() -> Vec<StyleSpec> {
        builtin_styles()
            .into_iter()
            .filter(|s| {
                s.lammps.codec().is_some() && s.category != "cmap" && !s.name.starts_with("coul/")
            })
            .collect()
    }

    /// Every built-in a LAMMPS line holds: the codec's tokens read back to
    /// the row, and the tokens are the identity.
    #[test]
    fn every_builtin_codec_reads_back_what_it_writes() {
        let mut seen = Vec::new();
        for spec in coefficient_styles() {
            let codec = spec.lammps.codec().unwrap();
            let row = sample(&spec);
            let written = codec.write(&spec, &row, &UnitScale::IDENTITY).unwrap();
            let tokens: Vec<String> = written.values.iter().map(|t| t.render(17)).collect();
            let strs: Vec<&str> = tokens.iter().map(String::as_str).collect();
            let back = codec
                .read(&spec, &strs)
                .unwrap_or_else(|e| panic!("{} {}: {e}", spec.category, spec.name));
            assert!(
                back.same_parameters(&row),
                "{} {}: {back:?} != {row:?}",
                spec.category,
                spec.name
            );
            let again = codec.write(&spec, &back, &UnitScale::IDENTITY).unwrap();
            assert_eq!(again, written, "{} {}", spec.category, spec.name);
            for (keyword, values) in &written.extra {
                let tokens: Vec<String> = values.iter().map(|t| t.render(6)).collect();
                let strs: Vec<&str> = tokens.iter().map(String::as_str).collect();
                codec
                    .parse_extra(&spec, keyword, &strs, &mut Params::new())
                    .unwrap();
            }
            seen.push(format!("{} {}", spec.category, spec.name));
        }
        for must in [
            "bond class2",
            "angle class2",
            "dihedral class2",
            "pair buck",
            "pair morse",
            "pair lj/class2",
        ] {
            assert!(seen.iter().any(|s| s == must), "{must} has no LAMMPS codec");
        }
    }

    /// A force field of one built-in style and the frame of one term (a
    /// pair style: two types, every pair a row), for a whole-file round
    /// trip.
    fn one_style(spec: &StyleSpec) -> (ForceField, Frame) {
        let mut ff = ForceField::new("one");
        ff.set_units("real");
        ff.set_special_bonds(SpecialBonds {
            lj: [0.0, 0.0, 0.0],
            coul: [0.0, 0.0, 0.0],
        });
        let masses = [("A", 12.011), ("B", 14.007)];
        atom_types(&mut ff, &masses);
        let cat = spec.category.as_ref();
        let mut style = Params::new();
        if cat == "pair" {
            style.set("cutoff", 9.0);
        }
        match spec.name.as_ref() {
            "lj/charmm" => style.set("inner", 7.0),
            "lj/class2" => style.set("mixing", "sixthpower"),
            "lj/cut" => style.set("mixing", "geometric"),
            _ => {}
        }
        let s = ff.def_style(cat, &spec.name, style).unwrap();
        let row = sample(spec);
        let x = vec![0.0, 0.0, 0.0, 1.5, 0.1, 0.0, 2.1, 1.4, 0.2, 3.4, 1.7, 1.1];
        let types = ["A", "B", "B", "A"];
        let frame = match cat {
            "pair" => {
                s.def_type("A", &["A"], row.clone()).unwrap();
                let mut b = Params::new();
                for (k, v) in row.iter().filter_map(|(k, v)| v.as_num().map(|x| (k, x))) {
                    b.set(k, v * 1.1);
                }
                s.def_type("B", &["B"], b).unwrap();
                if spec.params.iter().any(|p| p.mix == ParamCombination::None)
                    || spec.name == "lj/class2"
                {
                    let mut c = Params::new();
                    for (k, v) in row.iter().filter_map(|(k, v)| v.as_num().map(|x| (k, x))) {
                        c.set(k, v * 1.05);
                    }
                    s.def_type("A-B", &["A", "B"], c).unwrap();
                }
                frame(&types, &masses, &[])
            }
            "bond" => {
                s.def_type("A-B", &["A", "B"], row).unwrap();
                frame(&types, &masses, &[("bonds", &[(&[0, 1], "A-B")])])
            }
            "angle" => {
                s.def_type("A-B-B", &["A", "B", "B"], row).unwrap();
                frame(&types, &masses, &[("angles", &[(&[0, 1, 2], "A-B-B")])])
            }
            "dihedral" => {
                s.def_type("A-B-B-A", &["A", "B", "B", "A"], row).unwrap();
                frame(
                    &types,
                    &masses,
                    &[("dihedrals", &[(&[0, 1, 2, 3], "A-B-B-A")])],
                )
            }
            _ => {
                s.def_type("B-A-B-A", &["B", "A", "B", "A"], row).unwrap();
                frame(
                    &types,
                    &masses,
                    &[("impropers", &[(&[1, 0, 2, 3], "B-A-B-A")])],
                )
            }
        };
        let frame = with_pairs(frame);
        let frame = at(&frame, &x);
        (ff, frame)
    }

    fn include(ff: &ForceField, frame: &Frame) -> String {
        let labels = TypeLabels::from_frame(frame).unwrap();
        LammpsForcefieldWriter::new(&labels)
            .with_precision(17)
            .written(ff)
            .unwrap()
    }

    /// Every built-in a LAMMPS line holds, through a whole include: write →
    /// read → write is the identity on the text, the rows read back as
    /// written, and both force fields price the term alike.
    #[test]
    fn every_builtin_round_trips_through_an_include_at_the_same_energy() {
        for spec in coefficient_styles() {
            let what = format!("{} {}", spec.category, spec.name);
            if spec.name == "lj/charmm" {
                // LAMMPS has lj/charmm only beside its coul/charmm (the
                // equivalence check's CHARMM sources write and read it).
                continue;
            }
            let (ff, frame) = one_style(&spec);
            let text = include(&ff, &frame);
            let back = LammpsForcefieldReader::new()
                .read(text.as_bytes())
                .unwrap_or_else(|e| panic!("{what}: {e}\n{text}"));
            assert_eq!(include(&back, &frame), text, "{what}");
            // `improper periodic` is LAMMPS's `cvff`, which reads as molrs's
            // `improper cvff`: the same energy, another spelling.
            let Some(style) = back.get_style(&spec.category, &spec.name) else {
                assert_eq!(what, "improper periodic", "{text}");
                assert!(back.get_style("improper", "cvff").is_some(), "{text}");
                continue_energy(&what, &ff, &back, &frame);
                continue;
            };
            for (name, _, p) in ff
                .get_style(&spec.category, &spec.name)
                .unwrap()
                .type_rows()
            {
                let got = style
                    .type_params(name)
                    .unwrap_or_else(|| panic!("{what}: no type {name} read"));
                assert!(got.same_parameters(p), "{what} {name}: {got:?} != {p:?}");
            }
            continue_energy(&what, &ff, &back, &frame);
        }
    }

    /// Both force fields price `frame` alike, and not at zero.
    fn continue_energy(what: &str, ff: &ForceField, back: &ForceField, frame: &Frame) {
        let x: Vec<F> = {
            let atoms = frame.get("atoms").unwrap();
            let c = |k: &str| atoms.get(k).unwrap().get::<f64>().unwrap().to_owned();
            let (x, y, z) = (c("x"), c("y"), c("z"));
            (0..x.len())
                .flat_map(|i| [x[[i]], y[[i]], z[[i]]])
                .collect()
        };
        let e = |ff: &ForceField| {
            PotentialCompiler::new(ff)
                .compile(frame)
                .unwrap_or_else(|e| panic!("{what}: {e}"))
                .energy_at(&x)
        };
        let (a, b) = (e(ff), e(back));
        assert!(
            (a - b).abs() <= 1e-12 * a.abs(),
            "{what}: the energy read back, {a} != {b}"
        );
        assert!(
            a != 0.0,
            "{what}: a term that prices nothing proves nothing"
        );
    }

    /// `class2` carries its cross-term lines at zero; a non-zero cross term
    /// is refused by name, in an include and in a data file.
    #[test]
    fn class2_cross_terms_are_written_at_zero_and_refused_otherwise() {
        for name in ["angle", "dihedral"] {
            let spec = builtin_styles()
                .into_iter()
                .find(|s| s.category == name && s.name == "class2")
                .unwrap();
            let (ff, frame) = one_style(&spec);
            let text = include(&ff, &frame);
            let keywords: &[&str] = if name == "angle" {
                &["bb", "ba"]
            } else {
                &["mbt", "ebt", "at", "aat", "bb13"]
            };
            for k in keywords {
                assert!(
                    text.lines().any(|l| l.split_whitespace().nth(2) == Some(k)),
                    "{name} class2 `{k}` line missing:\n{text}"
                );
            }
            let nonzero = text.replacen(
                &format!("{} 0 ", keywords[0]),
                &format!("{} 3.5 ", keywords[0]),
                1,
            );
            let err = LammpsForcefieldReader::new()
                .read(nonzero.as_bytes())
                .unwrap_err()
                .to_string();
            assert!(err.contains("cross term"), "{err}");
            // The data file: the cross-term sections, read back.
            let labels = TypeLabels::from_frame(&frame).unwrap();
            let data = crate::io::writer::written_with(|b| {
                LammpsForcefieldWriter::new(&labels).write_data_coeffs(b, &ff)
            })
            .unwrap();
            let heading = if name == "angle" {
                "BondBond Coeffs"
            } else {
                "MiddleBondTorsion Coeffs"
            };
            assert!(data.contains(heading), "{data}");
            let back = LammpsForcefieldReader::new()
                .read_data_sections(&data, &Default::default(), "real")
                .unwrap();
            assert!(back.get_style(name, "class2").is_some(), "{data}");
        }
    }

    /// Unit conversion is per dimension: energy, length, their products and
    /// inverses; an angle value and a per-radian constant are unchanged.
    #[test]
    fn units_convert_per_dimension() {
        let sys = LammpsUnitConverter::canonical().unwrap();
        let scale = sys.scale("real", "metal").unwrap();
        let fe = sys.energy(1.0, "real", "metal").unwrap();
        let close = |a: F, b: F| assert!((a - b).abs() <= 1e-15 * b.abs(), "{a} != {b}");
        close(scale.apply(2.0, dim("E")), 2.0 * fe);
        close(scale.apply(2.0, dim("E/L^2")), 2.0 * fe);
        close(scale.apply(2.0, dim("E*L^6")), 2.0 * fe);
        close(scale.apply(2.0, dim("1/L")), 2.0);
        assert_eq!(scale.apply(109.5, ParamDimension::ANGLE), 109.5);
        close(scale.apply(2.0, dim("E/A^2")), 2.0 * fe);
        assert!(sys.scale("real", "real").unwrap().is_identity());
        // Through a writer: `buck`'s `c` (E·L⁶) is the energy's factor.
        let reference = crate::io::lammps::units::LammpsLjReference::canonical().unwrap();
        let sys = LammpsUnitConverter::with_reference(&reference).unwrap();
        let to_lj = sys.scale("real", "lj").unwrap();
        let l = sys.length(1.0, "real", "lj").unwrap();
        close(
            to_lj.apply(3.0, dim("E*L^6")),
            3.0 * sys.energy(1.0, "real", "lj").unwrap() * l.powi(6),
        );
    }

    /// The run-time `bond fene` (positional): written as `bond_style fene`
    /// and read back through the same registry; the process-wide registry
    /// (which has no `fene`) refuses the line by name.
    #[test]
    fn a_runtime_positional_style_reads_and_writes_through_its_registry() {
        let reg = registry();
        let case = cases(&reg).into_iter().find(|c| c.name == "fene").unwrap();
        let labels = TypeLabels::from_frame(&case.frame).unwrap();
        let text = LammpsForcefieldWriter::new(&labels)
            .written(&case.ff)
            .unwrap();
        assert!(text.contains("bond_style fene\n"), "{text}");
        assert!(
            text.contains("bond_coeff A-A 30.000000 1.500000 1.000000 1.000000\n"),
            "{text}"
        );
        let back = LammpsForcefieldReader::new()
            .with_registry(reg.clone())
            .read(text.as_bytes())
            .unwrap();
        let fene = back.get_style("bond", "fene").unwrap();
        assert!(
            fene.type_params("A-B").unwrap().same_parameters(
                case.ff
                    .get_style("bond", "fene")
                    .unwrap()
                    .type_params("A-B")
                    .unwrap()
            )
        );
        let err = LammpsForcefieldReader::new()
            .read(text.as_bytes())
            .unwrap_err()
            .to_string();
        assert!(err.contains("unsupported bond_style `fene`"), "{err}");
        // The same force field against the process-wide registry.
        let mut global = case.ff.clone();
        global.registry = crate::ff::style_registry::RegistryRef::Global;
        let err = LammpsForcefieldWriter::new(&labels)
            .written(&global)
            .unwrap_err();
        assert!(err.contains("LAMMPS has no form for bond `fene`"), "{err}");
    }

    /// Every way an engine refuses a style is `NoEngineForm`, by name.
    #[test]
    fn engines_refuse_what_they_cannot_hold_by_name() {
        // LAMMPS: an expression style without a LAMMPS form (no LEPTON).
        let mut reg = Registry::builtin();
        let mut expr_only = fene_spec();
        expr_only.lammps = LammpsForm::None;
        reg.register_style(expr_only, None).unwrap();
        let reg = Arc::new(reg);
        let case = cases(&registry())
            .into_iter()
            .find(|c| c.name == "fene")
            .unwrap();
        let labels = TypeLabels::from_frame(&case.frame).unwrap();
        let err = LammpsForcefieldWriter::new(&labels)
            .written(&case.ff.clone().with_registry(reg))
            .unwrap_err();
        assert!(
            err.contains("LAMMPS has no form for bond `fene`") && err.contains("LEPTON"),
            "{err}"
        );
        // A positional form the spec cannot have, at registration.
        let mut r = Registry::builtin();
        let mut indexed = StyleSpec::new("dihedral", "my_series")
            .params(vec![ParamSpec::new("k", ParamDimension::ENERGY).indexed()])
            .expression("k1*cos(phi)")
            .lammps(LammpsForm::positional());
        let err = r.register_style(indexed.clone(), None).unwrap_err();
        assert!(matches!(err, IrError::NoEngineForm { .. }), "{err:?}");
        let (cat, mut ub) = urey_bradley();
        r.register_category(cat).unwrap();
        ub.lammps = LammpsForm::positional();
        let err = r.register_style(ub, None).unwrap_err();
        assert!(err.to_string().contains("no `urey_bradley_style`"), "{err}");
        // register_engine_form: a form for a style registered without one;
        // another engine refused; a built-in sealed.
        indexed.lammps = LammpsForm::None;
        indexed.params = vec![ParamSpec::new("k", ParamDimension::ENERGY)];
        indexed.expression = Some("k*cos(phi)".into());
        r.register_style(indexed, None).unwrap();
        r.register_engine_form(
            Engine::Lammps,
            "dihedral",
            "my_series",
            LammpsForm::positional(),
        )
        .unwrap();
        assert_eq!(
            r.style("dihedral", "my_series").unwrap().0.lammps,
            LammpsForm::positional()
        );
        assert!(matches!(
            r.register_engine_form(
                Engine::Gromacs,
                "dihedral",
                "my_series",
                LammpsForm::positional()
            ),
            Err(IrError::NoEngineForm { .. })
        ));
        assert!(matches!(
            r.register_engine_form(Engine::Lammps, "bond", "harmonic", LammpsForm::None),
            Err(IrError::Sealed { .. })
        ));
        // GROMACS and frcmod: a style that is not built in.
        let mut ff = ForceField::new("x");
        ff.set_special_bonds(crate::ff::ir::SpecialBonds {
            lj: [0.0, 0.0, 0.5],
            coul: [0.0, 0.0, 1.0 / 1.2],
        });
        ff.set_units("real");
        let mut p = Params::new();
        p.set("expression", "k*(r-r0)^4");
        ff.def_style("bond", "quartic_custom", p)
            .unwrap()
            .def_type(
                "A-B",
                &["A", "B"],
                Params::from_iter([("k", 1.0), ("r0", 1.0)]),
            )
            .unwrap();
        let err = GromacsTopForcefieldWriter::new().written(&ff).unwrap_err();
        assert!(
            err.contains("GROMACS has no form for bond `quartic_custom`")
                && err.contains("not a built-in style"),
            "{err}"
        );
        let err = AmberFrcmodWriter::new()
            .written(&ff)
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("AMBER frcmod has no form for bond `quartic_custom`"),
            "{err}"
        );
        // LAMMPS: a run-time category has no `*_style` command, and its
        // energy is not dropped silently.
        let reg = registry();
        let case = cases(&reg)
            .into_iter()
            .find(|c| c.name == "openmm")
            .unwrap();
        let labels = TypeLabels::from_frame(&case.frame).unwrap();
        let err = LammpsForcefieldWriter::new(&labels)
            .written(&case.ff)
            .unwrap_err();
        assert!(
            err.contains("LAMMPS has no form for urey_bradley `harmonic`")
                && err.contains("no `urey_bradley_style`"),
            "{err}"
        );
        // OpenMM: an unregistered expression style is its Custom*Force.
        let xml = OpenmmXmlWriter::new().written(&ff).unwrap();
        assert!(
            xml.contains("<CustomBondForce energy=\"4.184*(k*(10*r-r0)^4)\">"),
            "{xml}"
        );
    }

    /// The OpenMM rewrite is exact: the written energy of OpenMM's
    /// coordinate (nm) is 4.184 times molrs's of the same distance in Å.
    #[test]
    fn the_openmm_rewrite_is_the_energy_in_openmm_units() {
        let reg = registry();
        let case = cases(&reg)
            .into_iter()
            .find(|c| c.name == "openmm")
            .unwrap();
        let xml = OpenmmXmlWriter::new().written(&case.ff).unwrap();
        let energy = xml
            .lines()
            .find(|l| l.contains("<CustomBondForce") && l.contains("log("))
            .and_then(|l| l.split('"').nth(1))
            .unwrap()
            .to_owned();
        let names = ["k", "r0", "epsilon", "sigma"];
        let openmm = compile(
            &energy,
            &Binding::new(Geometry::of_category("bond").unwrap(), &names, &[]),
        )
        .unwrap();
        let fene = fene_spec();
        let ir = compile(
            fene.expression.as_deref().unwrap(),
            &Binding::new(Geometry::of_category("bond").unwrap(), &names, &[]),
        )
        .unwrap();
        let p = [25.0, 1.9, 0.8, 1.4];
        for r in [1.1, 1.3, 1.45, 1.7] {
            let (e_omm, _) =
                openmm.eval_scalar_one(r * crate::core::unit_factors::ANGSTROM_TO_NM.get(), &p);
            let (e_ir, _) = ir.eval_scalar_one(r, &p);
            assert!(
                (e_omm - 4.184 * e_ir).abs() <= 1e-13 * e_omm.abs(),
                "r = {r}: {e_omm} != 4.184 × {e_ir}"
            );
        }
    }

    /// OpenMM's refusals: a compound category it generates no tuples for, a
    /// pair cross row, a 1-4 weight a bondCutoff cannot state.
    #[test]
    fn openmm_custom_forces_refuse_by_name() {
        let reg = registry();
        let mut case = cases(&reg)
            .into_iter()
            .find(|c| c.name == "openmm")
            .unwrap();
        case.ff.set_special_bonds(SpecialBonds {
            lj: [0.0, 0.0, 0.5],
            coul: [0.0, 0.0, 0.5],
        });
        let err = OpenmmXmlWriter::new().written(&case.ff).unwrap_err();
        assert!(err.contains("bondCutoff"), "{err}");
        let mut r = (*reg).clone();
        r.register_category(CategorySpec::custom(
            "quint",
            5,
            Coordinate::Compound,
            EndpointOrder::Reversible,
        ))
        .unwrap();
        r.register_style(
            StyleSpec::new("quint", "x")
                .params(vec![ParamSpec::new("k", ParamDimension::ENERGY)])
                .expression("k*distance(p1,p5)"),
            None,
        )
        .unwrap();
        let mut ff = ForceField::new("q").with_registry(Arc::new(r));
        ff.set_special_bonds(crate::ff::ir::SpecialBonds {
            lj: [0.0, 0.0, 0.5],
            coul: [0.0, 0.0, 1.0 / 1.2],
        });
        ff.set_units("real");
        ff.def_style("quint", "x", Params::new())
            .unwrap()
            .def_type(
                "t",
                &["A", "A", "A", "A", "A"],
                Params::from_iter([("k", 1.0)]),
            )
            .unwrap();
        let err = OpenmmXmlWriter::new().written(&ff).unwrap_err();
        assert!(
            err.contains("OpenMM XML has no form for quint `x`"),
            "{err}"
        );
    }

    /// A water-like A-B-A under bond and angle `harmonic`, `lj/cut` and
    /// `coul/cut` (its `delta` stated at the spec default), every bonded row
    /// carrying `extra` besides its parameters.
    fn with_row_extras(extra: &Params) -> (ForceField, Frame) {
        let mut ff = ForceField::new("extras");
        ff.set_special_bonds(crate::ff::ir::SpecialBonds {
            lj: [0.0, 0.0, 0.5],
            coul: [0.0, 0.0, 1.0 / 1.2],
        });
        ff.set_units("real");
        let atoms = ff.def_style("atom", "full", Params::new()).unwrap();
        for (t, m) in [("A", 16.0), ("B", 1.008)] {
            atoms
                .def_type(t, &[], Params::from_iter([("mass", m)].iter().copied()))
                .unwrap();
        }
        let row = |pairs: &[(&str, F)]| {
            let mut p = Params::from_iter((pairs).iter().copied());
            for (k, v) in extra.iter() {
                p.set(k, v.clone());
            }
            p
        };
        ff.def_style("bond", "harmonic", Params::new())
            .unwrap()
            .def_type("A-B", &["A", "B"], row(&[("k", 450.0), ("r0", 0.96)]))
            .unwrap();
        ff.def_style("angle", "harmonic", Params::new())
            .unwrap()
            .def_type(
                "B-A-B",
                &["B", "A", "B"],
                row(&[("k", 55.0), ("theta0", 104.5)]),
            )
            .unwrap();
        let mut lj_style = Params::from_iter([("cutoff", 10.0)].iter().copied());
        lj_style.set("mixing", "arithmetic");
        let lj = ff.def_style("pair", "lj/cut", lj_style).unwrap();
        for (t, eps, sigma) in [("A", 0.15, 3.15), ("B", 0.0, 0.0)] {
            lj.def_type(
                t,
                &[t],
                Params::from_iter([("epsilon", eps), ("sigma", sigma)].iter().copied()),
            )
            .unwrap();
        }
        ff.def_style(
            "pair",
            "coul/cut",
            Params::from_iter([("cutoff", 10.0), ("delta", 0.0)].iter().copied()),
        )
        .unwrap();
        let frame = with_pairs(frame(
            &["B", "A", "B"],
            &[("A", 16.0), ("B", 1.008)],
            &[
                ("bonds", &[(&[0, 1], "A-B"), (&[1, 2], "A-B")]),
                ("angles", &[(&[0, 1, 2], "B-A-B")]),
            ],
        ));
        (ff, frame)
    }

    /// Every engine writer's output of `ff`, or its refusal.
    fn every_writer(ff: &ForceField, frame: &Frame) -> Vec<(&'static str, Result<String, String>)> {
        let labels = TypeLabels::from_frame(frame).unwrap();
        let err = |e: crate::io::writer::ForceFieldWriteError| e.to_string();
        vec![
            (
                "LAMMPS",
                LammpsForcefieldWriter::new(&labels)
                    .written(ff)
                    .map_err(err),
            ),
            (
                "GROMACS",
                crate::io::writer::capture(|sink| {
                    GromacsTopForcefieldWriter::new().write_system(sink, ff, frame)
                })
                .map_err(err),
            ),
            ("OpenMM", OpenmmXmlWriter::new().written(ff).map_err(err)),
            (
                "frcmod",
                crate::io::writer::capture(|sink| {
                    use crate::io::writer::ForceFieldWriter;
                    crate::io::amber::AmberFrcmodWriter::new().write(sink, ff)
                })
                .map_err(err),
            ),
        ]
    }

    /// The keys that price nothing — an estimate's provenance, an
    /// annotation — pass every writer, and a parameter at its spec default
    /// (`coul/cut`'s `delta = 0`) needs no column; a numeric key no format
    /// has a column for is still refused by each, by name.
    #[test]
    fn writers_skip_what_prices_nothing_and_refuse_the_unknown() {
        let mut provenance = Params::new();
        crate::ff::typifier::Provenance::analogy(2.5, "c3-os").apply_to(&mut provenance);
        provenance.set("desc", "a water");
        provenance.set("doi", "10.0/x");
        let (ff, frame) = with_row_extras(&provenance);
        for (writer, out) in every_writer(&ff, &frame) {
            let text = out.unwrap_or_else(|e| panic!("{writer}: {e}"));
            assert!(!text.contains("estimate_"), "{writer}: {text}");
        }

        let (ff, frame) = with_row_extras(&Params::from_iter([("bogus", 1.0)].iter().copied()));
        for (writer, out) in every_writer(&ff, &frame) {
            match out {
                Err(e) => assert!(e.contains("bogus"), "{writer}: {e}"),
                Ok(text) => panic!("{writer} dropped `bogus`:\n{text}"),
            }
        }
    }

    /// A style parameter is refused unless at its spec default: `coul/cut`
    /// with `delta = 0` writes, with `delta = 0.5` (a soft core no engine
    /// file here states) is refused by name.
    #[test]
    fn a_style_parameter_off_its_default_is_refused() {
        let (mut ff, frame) = with_row_extras(&Params::new());
        ff.get_style_mut("pair", "coul/cut")
            .unwrap()
            .set_param("delta", 0.5);
        for (writer, out) in every_writer(&ff, &frame) {
            let e = out
                .err()
                .unwrap_or_else(|| panic!("{writer} dropped delta"));
            assert!(e.contains("delta"), "{writer}: {e}");
        }
    }
}
