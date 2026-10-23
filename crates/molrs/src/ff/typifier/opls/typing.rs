//! OPLS-AA SMARTS atom typing (dependency-aware, layered).
//!
//! [`typify_atoms`] drives the [`LayeredTypingEngine`]: it processes the type
//! defs level by level so that a def referencing a previously-assigned type via
//! a `%label` (e.g. benzene's aromatic-H type `opls_146` =
//! `[#1]-[c;%opls_145]`) is matched only after its dependency is resolved. The
//! engine returns the per-atom `opls_NNN` assignment, which this function turns
//! into the node [`Annotation`]s of a [`TypeAssignment`](crate::ff::typifier::TypeAssignment):
//! each type's `type` (with its `atom/full` row's `mass` and `charge`) and
//! `class`. It writes nothing; the typing base stamps the match.
//!
//! # Aromaticity on a private copy
//!
//! The engine never sees the caller's graph as given. It matches a clone
//! brought to the standard aromatic form by
//! [`assign_aromaticity`](molrs::perceive::assign_aromaticity),
//! which keeps every [`NodeId`], so a Kekulé ring and an aromatic-declared ring
//! type alike under the aromatic rules (`c`, `n`, `:`). Only the assigned type
//! names travel back; the caller's bond types and bond numbers are never
//! touched, so a Kekulé input stays Kekulé.
//!
//! # SMARTS reuse
//!
//! Matching uses the always-compiled molrs SMARTS engine
//! ([`SmartsPattern`](molrs::perceive::smarts::SmartsPattern)) with standard Daylight semantics
//! (an unmarked bond is single-or-aromatic, `[#1]` is a hydrogen atom, `H<n>`
//! a hydrogen count, `r<n>` the smallest ring) plus the context-label
//! extension ([`MatchOptions::labels`](molrs::perceive::smarts::MatchOptions::labels)): the
//! engine feeds back the current assignment map as the label context so
//! `%opls_NNN` predicates can read it. Each `def` is the SMARTS for the type's
//! *target* atom: by RDKit convention the engine roots a match at query atom 0,
//! so the typed atom of every match is `match[0]`.
//!
//! # Conflict resolution
//!
//! When several defs match the same atom *within a level*:
//! 1. candidates dominated by another candidate drop out — a type dominates
//!    another when it sits on a higher `layer`, or on the same layer overrides
//!    it directly or transitively (see [`OplsTypingMetadata`]);
//! 2. among the rest, higher explicit `priority` wins (absent = 0);
//! 3. then the more specific pattern (more query atoms);
//! 4. then the earlier sorted type name.
//!
//! Across levels, a later level's winner replaces an atom's current type unless
//! the current type dominates it.
//!
//! # Levels
//!
//! Standalone (no-`%label`) defs are level 0. `%label`-referencing defs
//! resolve in dependency order; mutually-dependent defs form a circular group
//! resolved by fixed-point iteration (see [`layered`](super::layered)).
//!
//! # Out of scope
//!
//! - Rows with no `def` are skipped (cannot be SMARTS-matched).
//! - Bonded-term (bond / angle / dihedral) labeling lives in
//!   [`assign`](super::assign).

use std::collections::HashMap;

use molrs::core::PropValue;
use molrs::core::{Atomistic, NodeId};
use molrs::perceive::assign_aromaticity;

use crate::ff::forcefield::ForceField;
use crate::ff::typifier::Annotation;

use super::layered::LayeredTypingEngine;
use super::typing_metadata::OplsTypingMetadata;

/// The OPLS-AA atom typing of one graph.
pub(crate) struct AtomTyping {
    /// The `opls_NNN` type of every atom a def typed; an atom no def typed is
    /// absent.
    pub(crate) types: HashMap<NodeId, String>,
    /// The node annotations, positional against `graph.atoms()`.
    pub(crate) nodes: Vec<Vec<(String, Annotation)>>,
}

/// Type the atoms of `mol` with OPLS-AA atom types.
///
/// Drives `engine` (built from `meta`), matching a privately
/// aromaticity-perceived copy of `mol` (same [`NodeId`]s; `mol` itself is never
/// changed). Every atom assigned a type gets two annotations:
/// - `type` → [`Annotation::Type`] under the `atom/full` style of `ff`, named
///   by the `opls_NNN` type, with that row's numeric params (`mass`, `charge`)
///   — or a plain [`Annotation::Value`] when `ff` has no such row. The row's
///   string metadata (`type_`, `smarts`, …, from the XML reader) is typing input
///   and is neither stamped nor defined;
/// - `class` → [`Annotation::Value`] of the type's class, when `meta` has one.
///
/// Atoms typed by no def get no annotation (and stay untyped); the strict
/// [`OplsAaTypifier`](super::OplsAaTypifier) refuses such a molecule.
///
/// # Errors
///
/// Returns `Err` if aromaticity cannot be perceived on `mol`.
pub(crate) fn typify_atoms(
    mol: &Atomistic,
    engine: &LayeredTypingEngine,
    meta: &OplsTypingMetadata,
    ff: &ForceField,
) -> Result<AtomTyping, String> {
    let mut perceived = mol.clone();
    assign_aromaticity(&mut perceived).map_err(|e| e.to_string())?;
    let types = engine.assign(&perceived);
    let atom_full = ff.get_style("atom", "full");

    let nodes = mol
        .atoms()
        .map(|(id, _)| {
            let Some(type_name) = types.get(&id) else {
                return Vec::new();
            };
            let typed = match atom_full.and_then(|s| s.get_atomtype(type_name)) {
                Some(row) => Annotation::Type {
                    style: "full".to_owned(),
                    name: type_name.clone(),
                    endpoints: Vec::new(),
                    params: row
                        .params
                        .iter()
                        .filter_map(|(k, v)| v.as_num().map(|x| (k, x)))
                        .collect(),
                },
                None => Annotation::Value(PropValue::Str(type_name.clone())),
            };
            let mut annotations = vec![("type".to_owned(), typed)];
            if let Some(row) = meta.get(type_name) {
                annotations.push((
                    "class".to_owned(),
                    Annotation::Value(PropValue::Str(row.class.clone())),
                ));
            }
            annotations
        })
        .collect();

    Ok(AtomTyping { types, nodes })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ff::ir::Params;
    use crate::ff::typifier::OplsTypeRow;
    use crate::ff::typifier::TypeAssignment;
    use crate::props;
    use molrs::core::BondOrder;
    use molrs::core::Element;
    use molrs::core::keys;

    /// `mol` with the node annotations of [`typify_atoms`] written onto it
    /// through the typing base's one execution path.
    fn typed_graph(
        mol: &Atomistic,
        meta: &OplsTypingMetadata,
        ff: &ForceField,
    ) -> Result<Atomistic, String> {
        let engine = LayeredTypingEngine::build(meta)?;
        let typing = typify_atoms(mol, &engine, meta, ff)?;
        let mut graph = mol.clone();
        let mut m = TypeAssignment {
            nodes: typing.nodes,
            ..TypeAssignment::default()
        };
        m.declare_styles_of(ff);
        m.apply_to(&mut graph, &mut ff.empty_like())?;
        Ok(graph)
    }

    /// Build a tiny ethane-like skeleton C-C with explicit H neighbours so the
    /// `[C;X4](C)(H)(H)H` style defs have something to match. (Pure-function
    /// unit fixture — typing of real molecules by the shipped rules is the
    /// golden tests of `opls/shipped_forcefield.rs`.)
    fn ethane() -> Atomistic {
        let mut g = Atomistic::new();
        let c0 = g
            .add_atom(props! {
                keys::ATOMIC_NUMBER => Element::C,
                keys::X => 0.0,
                keys::Y => 0.0,
                keys::Z => 0.0,
            })
            .unwrap();
        let c1 = g
            .add_atom(props! {
                keys::ATOMIC_NUMBER => Element::C,
                keys::X => 1.5,
                keys::Y => 0.0,
                keys::Z => 0.0,
            })
            .unwrap();
        g.add_bond(c0, c1, props! {}).unwrap();
        for c in [c0, c1] {
            for k in 0..3 {
                let h = g
                    .add_atom(props! {
                        keys::ATOMIC_NUMBER => Element::H,
                        keys::X => 0.3 * (k as f64 + 1.0),
                        keys::Y => 0.8,
                        keys::Z => 0.0,
                    })
                    .unwrap();
                g.add_bond(c, h, props! {}).unwrap();
            }
        }
        g
    }

    fn meta_with(rows: &[(&str, OplsTypeRow)]) -> OplsTypingMetadata {
        let mut m = OplsTypingMetadata::new();
        for (name, row) in rows {
            m.insert(*name, row.clone());
        }
        m
    }

    fn row(class: &str, def: Option<&str>, overrides: &[&str]) -> OplsTypeRow {
        OplsTypeRow {
            class: class.to_string(),
            def: def.map(str::to_string),
            overrides: overrides.iter().map(|s| s.to_string()).collect(),
            priority: None,
            layer: 0,
        }
    }

    #[test]
    fn typing_assigns_carbon_and_hydrogen() {
        let m = meta_with(&[
            ("opls_135", row("CT", Some("[C;X4](C)(H)(H)H"), &[])),
            ("opls_140", row("HC", Some("H[C;X4]"), &[])),
        ]);
        let ff = ForceField::new("OPLS-AA");
        let typed = typed_graph(&ethane(), &m, &ff).unwrap();

        let mut n_ct = 0;
        let mut n_hc = 0;
        for (_, a) in typed.atoms() {
            match a.get_str("type") {
                Some("opls_135") => {
                    n_ct += 1;
                    assert_eq!(a.get_str("class"), Some("CT"));
                }
                Some("opls_140") => n_hc += 1,
                _ => {}
            }
        }
        assert_eq!(n_ct, 2, "both methyl carbons typed CT");
        assert_eq!(n_hc, 6, "all six H typed HC");
    }

    #[test]
    fn higher_priority_overrides_wins() {
        // Two defs both match every C; opls_special overrides opls_generic, so it
        // gains priority and must win on the carbons.
        let m = meta_with(&[
            ("opls_generic", row("CG", Some("[C]"), &[])),
            ("opls_special", row("CS", Some("[C;X4]"), &["opls_generic"])),
        ]);
        let ff = ForceField::new("OPLS-AA");
        let typed = typed_graph(&ethane(), &m, &ff).unwrap();
        for (id, a) in typed.atoms() {
            if a.atomic_number() == Some(Element::C.z()) {
                assert_eq!(
                    a.get_str("type"),
                    Some("opls_special"),
                    "carbon {id:?} should take the higher-priority override"
                );
            }
        }
    }

    #[test]
    fn charge_written_from_forcefield() {
        let m = meta_with(&[("opls_140", row("HC", Some("H[C;X4]"), &[]))]);
        let mut ff = ForceField::new("OPLS-AA");
        ff.def_style("atom", "full", Params::new())
            .unwrap()
            .def_type(
                "opls_140",
                &[],
                Params::from_iter([("mass", 1.008), ("charge", 0.06)]),
            )
            .unwrap();
        let typed = typed_graph(&ethane(), &m, &ff).unwrap();
        let h = typed
            .atoms()
            .find(|(_, a)| a.get_str("type") == Some("opls_140"))
            .expect("a hydrogen was typed");
        assert_eq!(h.1.get_f64("charge"), Some(0.06));
    }

    #[test]
    fn malformed_def_fails_fast() {
        // Unbalanced bracket — a broken force-field def, must Err (never drop).
        let m = meta_with(&[("opls_bad", row("X", Some("[C"), &[]))]);
        let ff = ForceField::new("OPLS-AA");
        let err = typed_graph(&ethane(), &m, &ff).unwrap_err();
        assert!(err.contains("opls_bad"), "err names the type: {err}");
    }

    #[test]
    fn recursive_dollar_def_matches() {
        // Recursive $() SMARTS: an sp3 carbon bonded to another sp3 carbon.
        // Exercises the engine's recursive `$(...)` support (rooted at the
        // candidate atom). Both ethane carbons match.
        let m = meta_with(&[("opls_rec", row("CT", Some("[$([CX4][CX4])]"), &[]))]);
        let ff = ForceField::new("OPLS-AA");
        let typed = typed_graph(&ethane(), &m, &ff).unwrap();
        let n = typed
            .atoms()
            .filter(|(_, a)| a.get_str("type") == Some("opls_rec"))
            .count();
        assert_eq!(n, 2, "recursive def should match both sp3 carbons");
    }

    #[test]
    fn typed_atom_ref_def_without_dependency_types_nothing() {
        // A %opls_NNN def is supported (layered), not skipped. With no
        // matching dependency present (nothing is ever typed opls_145), the
        // `%opls_145` predicate never holds, so the def matches nothing — and
        // it is NOT an error. (Real layered typing is covered in
        // `opls/layered.rs` and the golden tests of `opls/shipped_forcefield.rs`.)
        let m = meta_with(&[("opls_ref", row("HA", Some("[H][C;%opls_145]"), &[]))]);
        let ff = ForceField::new("OPLS-AA");
        let typed = typed_graph(&ethane(), &m, &ff).unwrap();
        assert!(typed.atoms().all(|(_, a)| a.get_str("type").is_none()));
    }

    #[test]
    fn layered_dependency_def_types_after_its_dependency() {
        // opls_154 (alcohol O) then opls_155 (H[O;%opls_154]): the hydroxyl H
        // is typed only after the O is typed opls_154 — exercising the full
        // typify_atoms layered path end to end on a constructed ethanol.
        let mut g = Atomistic::new();
        let cm = g
            .add_atom(props! {
                keys::ATOMIC_NUMBER => Element::C,
                keys::X => 0.0,
                keys::Y => 0.0,
                keys::Z => 0.0,
            })
            .unwrap();
        let ch = g
            .add_atom(props! {
                keys::ATOMIC_NUMBER => Element::C,
                keys::X => 1.5,
                keys::Y => 0.0,
                keys::Z => 0.0,
            })
            .unwrap();
        let o = g
            .add_atom(props! {
                keys::ATOMIC_NUMBER => Element::O,
                keys::X => 2.5,
                keys::Y => 0.0,
                keys::Z => 0.0,
            })
            .unwrap();
        let ho = g
            .add_atom(props! {
                keys::ATOMIC_NUMBER => Element::H,
                keys::X => 3.3,
                keys::Y => 0.0,
                keys::Z => 0.0,
            })
            .unwrap();
        g.add_bond(cm, ch, props! {}).unwrap();
        g.add_bond(ch, o, props! {}).unwrap();
        g.add_bond(o, ho, props! {}).unwrap();
        for k in 0..3 {
            let h = g
                .add_atom(props! {
                    keys::ATOMIC_NUMBER => Element::H,
                    keys::X => 0.3 * (k as f64 + 1.0),
                    keys::Y => 0.9,
                    keys::Z => 0.0,
                })
                .unwrap();
            g.add_bond(cm, h, props! {}).unwrap();
        }
        for k in 0..2 {
            let h = g
                .add_atom(props! {
                    keys::ATOMIC_NUMBER => Element::H,
                    keys::X => 1.5 + 0.3 * (k as f64),
                    keys::Y => 0.9,
                    keys::Z => 0.0,
                })
                .unwrap();
            g.add_bond(ch, h, props! {}).unwrap();
        }

        let m = meta_with(&[
            ("opls_154", row("OH", Some("[O;X2](H)([!H])"), &[])),
            ("opls_155", row("HO", Some("H[O;%opls_154]"), &[])),
        ]);
        let ff = ForceField::new("OPLS-AA");
        let typed = typed_graph(&g, &m, &ff).unwrap();

        assert_eq!(
            typed.get_atom(o).unwrap().get_str("type"),
            Some("opls_154"),
            "alcohol O typed opls_154"
        );
        assert_eq!(
            typed.get_atom(ho).unwrap().get_str("type"),
            Some("opls_155"),
            "hydroxyl H typed opls_155 via the %opls_154 dependency"
        );
    }

    /// Kekulé benzene, hand-built: ring carbons 0..=5 bonded `C0=C1-C2=C3-C4=C5-C0`
    /// (bond types `Double` / `Single`, no `is_aromatic` flag), one hydrogen on
    /// each carbon (atoms 6..=11). Returns the graph and its six ring bonds.
    fn kekule_benzene() -> (Atomistic, Vec<molrs::core::RelationId>) {
        let mut g = Atomistic::new();
        let c: Vec<NodeId> = (0..6)
            .map(|k| {
                g.add_atom(props! {
                    keys::ATOMIC_NUMBER => Element::C,
                    keys::X => 1.4 * k as f64,
                    keys::Y => 0.0,
                    keys::Z => 0.0,
                })
                .unwrap()
            })
            .collect();
        let ring: Vec<molrs::core::RelationId> = (0..6)
            .map(|k| {
                let bond = g.add_bond(c[k], c[(k + 1) % 6], props! {}).unwrap();
                let order = if k % 2 == 0 {
                    BondOrder::Double
                } else {
                    BondOrder::Single
                };
                let number = order.implied_number().expect("a Kekulé fixture bond");
                g.set_bond_class(bond, order, number).unwrap();
                bond
            })
            .collect();
        for (k, &carbon) in c.iter().enumerate() {
            let h = g
                .add_atom(props! {
                    keys::ATOMIC_NUMBER => Element::H,
                    keys::X => 1.4 * k as f64,
                    keys::Y => 1.1,
                    keys::Z => 0.0,
                })
                .unwrap();
            g.add_bond(carbon, h, props! {}).unwrap();
        }
        (g, ring)
    }

    /// Typing perceives aromaticity on a private copy.
    ///
    /// Kekulé benzene declares no aromatic atom, yet it matches the
    /// aromatic-only rules `[c;X3;r6]` (every ring carbon) and `[#1]-[c]`
    /// (every hydrogen), so all 12 atoms are typed. The perceived copy stays
    /// private: the returned graph keeps the caller's `Double` / `Single` type
    /// and bond number on every ring bond.
    #[test]
    fn kekule_input_is_typed_by_aromatic_rules_and_keeps_its_bond_orders() {
        let m = meta_with(&[
            ("t_ca", row("CA", Some("[c;X3;r6]"), &[])),
            ("t_ha", row("HA", Some("[#1]-[c]"), &[])),
        ]);
        let ff = ForceField::new("OPLS-AA");
        let (mol, ring) = kekule_benzene();
        let typed = typed_graph(&mol, &m, &ff).unwrap();

        let count = |t: &str| {
            typed
                .atoms()
                .filter(|(_, a)| a.get_str("type") == Some(t))
                .count()
        };
        assert_eq!(count("t_ca"), 6, "all six ring carbons typed t_ca");
        assert_eq!(count("t_ha"), 6, "all six hydrogens typed t_ha");

        for (k, &bond) in ring.iter().enumerate() {
            let want = if k % 2 == 0 {
                BondOrder::Double
            } else {
                BondOrder::Single
            };
            assert_eq!(typed.bond_type(bond).unwrap(), want, "ring bond {k} type");
            assert_eq!(
                typed.bond_number(bond).unwrap(),
                mol.bond_number(bond).unwrap(),
                "ring bond {k} number"
            );
        }
    }
}
