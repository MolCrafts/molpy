//! OPLS-AA SMARTS atom typifier.

use std::collections::HashSet;

use molrs::core::{Atomistic, NodeId};

use crate::ff::forcefield::ForceField;

use crate::ff::typifier::{TypeAssignment, Typifier, TypingError};

mod assign;
mod dependency;
mod layered;
mod shipped_forcefield;
pub(crate) mod typing;
pub(crate) mod typing_metadata;

pub use assign::{CandidateTables, NoMatch};
pub use typing_metadata::{OplsTypeRow, OplsTypingMetadata};

use assign::typify_bonded_with;
use layered::LayeredTypingEngine;
use typing::typify_atoms;

/// OPLS-AA typifier — owns typing metadata and force-field parameters.
///
/// [`oplsaa`](Self::oplsaa) is the shipped set; [`new`](Self::new) takes typing
/// metadata ([`OplsTypingMetadata`]) and potential parameters ([`ForceField`])
/// — a caller's OPLS-AA / CL&P file (OpenMM's schema with Foyer's typing
/// annotations) gives both to
/// [`OpenmmXmlReader::read_opls_typifier`](crate::io::openmm_xml::OpenmmXmlReader::read_opls_typifier).
/// Each precomputes the bonded candidate tables once.
///
/// Mirrors [`mmff`](crate::ff::typifier::mmff): typing metadata
/// ([`OplsTypingMetadata`]) is kept *separately* from the potential [`ForceField`].
/// [`OplsAaTypifier`] owns both and implements [`Typifier`], assigning
/// `opls_NNN` atom types by SMARTS matching.
///
/// # How atoms are typed
///
/// - **Rules.** The shipped rules ([`crate::ff::params::OPLSAA_TYPING`]) are
///   molrs-owned Daylight SMARTS with explicit bonds, `[#1]` hydrogens and
///   lowercase aromatic atoms, matched with standard semantics against a
///   molecule with explicit hydrogens.
/// - **Aromaticity on a private copy.** The rules run on a clone brought to the
///   standard aromatic form by
///   [`assign_aromaticity`](molrs::perceive::assign_aromaticity)
///   (same atom ids), so a Kekulé ring and an aromatic-declared ring type
///   alike, while the caller's bond types and bond numbers are never changed.
///   Typifiers built by [`OplsAaTypifier::new`] over a caller's own file do
///   the same.
/// - **Ranking.** A type on a higher `layer`, or on the same layer overriding
///   another (directly or transitively), dominates it: it wins on an atom where
///   both match, and a later dependency level never replaces it. Candidates
///   nothing dominates rank by explicit `priority`, then pattern size, then
///   name.
///
/// After atom typing, its `assign` ([`Typifier`]) matches every bond /
/// angle / dihedral against the force field's bonded tables by OPLS
/// specificity + overlay layer (chain 2).
/// [`Typing`](crate::ff::typifier::Typing) runs the match: it stamps
/// the typed copy and defines exactly the types used in its output force field.
/// Callers compose `Typing::typify` → `to_frame` → pairs →
/// `PotentialCompiler::new(typing.forcefield()).compile`.
///
/// # Bonded parameters
///
/// Bonded terms are matched against the force field's class-keyed rows by
/// specificity and overlay layer (`assign`).
///
/// # Scope
///
/// Only types carrying a SMARTS `def` participate; the united-atom types
/// (`opls_001`–`opls_134`) carry none and are out of scope for auto-typing. Improper
/// matching is out of scope. Uncovered bonded terms follow the `NoMatch`
/// policy: an error in strict mode, left unparametrized otherwise.
pub struct OplsAaTypifier {
    meta: OplsTypingMetadata,
    /// The SMARTS defs of `meta`, compiled and layered once.
    engine: LayeredTypingEngine,
    ff: ForceField,
    tables: CandidateTables,
    /// No-match policy for bonded terms with no force-field candidate.
    no_match: NoMatch,
}

impl OplsAaTypifier {
    /// Build a typifier over the shipped canonical OPLS-AA parameter set.
    ///
    /// The parameters ([`crate::ff::params::OPLSAA_ATOMS`] and its sibling tables) are generated from
    /// GROMACS v2026.3 `share/top/oplsaa.ff` (LGPL-2.1-or-later): atom classes
    /// are GROMACS `bond_type`s, and the `pair/lj/cut` style declares
    /// OPLS-AA's **geometric** combining rule (σᵢⱼ = √(σᵢσⱼ), εᵢⱼ = √(εᵢεⱼ);
    /// Jorgensen et al. 1996, GROMACS comb-rule 3), so molrs's kernel and an
    /// exported LAMMPS input mix alike. The SMARTS typing rules
    /// ([`crate::ff::params::OPLSAA_TYPING`]) are molrs's own Daylight SMARTS;
    /// each takes its class from the atom row of the same name.
    ///
    /// The input molecule needs explicit hydrogens. Its rings may be written
    /// Kekulé or aromatic: typing perceives aromaticity on a private copy and
    /// leaves the caller's bond orders as they were.
    ///
    /// The parameters are compiled-in typed Rust, so this is the standalone
    /// path: the OPLS typifier needs no external file on disk and parses nothing
    /// at runtime. Mirrors
    /// [`Mmff94Typifier::new`](crate::ff::typifier::mmff::Mmff94Typifier::new).
    ///
    /// Infallible: the parameters are compile-time constants, the same policy
    /// as `Mmff94Typifier::new` and `UffTypifier::new`;
    /// `shipped_forcefield::tests::the_shipped_rules_build` proves they build.
    ///
    /// # Examples
    ///
    /// The library declares OPLS-AA's geometric mixing:
    ///
    /// ```
    /// use molrs::ff::typifier::Typifier;
    /// use molrs::ff::typifier::OplsAaTypifier;
    ///
    /// let typifier = OplsAaTypifier::oplsaa();
    /// let lj = typifier
    ///     .source_forcefield()
    ///     .get_style("pair", "lj/cut")
    ///     .expect("OPLS-AA declares lj/cut");
    /// assert_eq!(lj.params().get("mixing").and_then(|v| v.as_text()), Some("geometric"));
    /// ```
    ///
    /// Typing ethanol, `CH3-CH2-OH` with explicit hydrogens: the hydroxyl
    /// oxygen is the mono-alcohol O, `opls_154`.
    ///
    /// ```
    /// use molrs::ff::typifier::Typing;
    /// use molrs::ff::typifier::OplsAaTypifier;
    /// use molrs::core::{Atomistic, Element, keys};
    /// use molrs::props;
    ///
    /// let mut ethanol = Atomistic::new();
    /// let c1 = ethanol.add_atom(props! {
    ///     keys::ATOMIC_NUMBER => Element::C,
    ///     keys::X => 0.0,
    ///     keys::Y => 0.0,
    ///     keys::Z => 0.0,
    /// }).unwrap();
    /// let c2 = ethanol.add_atom(props! {
    ///     keys::ATOMIC_NUMBER => Element::C,
    ///     keys::X => 1.5,
    ///     keys::Y => 0.0,
    ///     keys::Z => 0.0,
    /// }).unwrap();
    /// let o = ethanol.add_atom(props! {
    ///     keys::ATOMIC_NUMBER => Element::O,
    ///     keys::X => 2.0,
    ///     keys::Y => 1.3,
    ///     keys::Z => 0.0,
    /// }).unwrap();
    /// ethanol.add_bond(c1, c2, props! {}).unwrap();
    /// ethanol.add_bond(c2, o, props! {}).unwrap();
    /// for (heavy, n) in [(c1, 3), (c2, 2), (o, 1)] {
    ///     for _ in 0..n {
    ///         let h = ethanol.add_atom(props! {
    ///             keys::ATOMIC_NUMBER => Element::H,
    ///             keys::X => 0.0,
    ///             keys::Y => 0.0,
    ///             keys::Z => 1.0,
    ///         }).unwrap();
    ///         ethanol.add_bond(heavy, h, props! {}).unwrap();
    ///     }
    /// }
    ///
    /// let typed = Typing::new(OplsAaTypifier::oplsaa().with_strict(false))
    ///     .typify(&ethanol)
    ///     .expect("ethanol types");
    /// assert_eq!(typed.get_atom(o).unwrap().get_str("type"), Some("opls_154"));
    /// ```
    pub fn oplsaa() -> Self {
        Self::new(
            shipped_forcefield::typing_meta(),
            shipped_forcefield::force_field(),
        )
        .expect(
            "the shipped OPLS-AA rules build — proved by \
             ff::typifier::opls::shipped_forcefield::tests::the_shipped_rules_build",
        )
    }

    /// Construct from already-parsed metadata and force field (strict bonded
    /// matching). The SMARTS defs are compiled and layered, and the bonded
    /// candidate tables built from `ff`, once, here.
    ///
    /// # Errors
    ///
    /// [`TypingError::Malformed`] naming the type for a malformed SMARTS
    /// `def`, naming both types for an override of a type absent from `meta`,
    /// naming every member for an overrides cycle.
    pub fn new(meta: OplsTypingMetadata, ff: ForceField) -> Result<Self, TypingError> {
        let engine = LayeredTypingEngine::build(&meta).map_err(TypingError::malformed)?;
        let tables = CandidateTables::build(&ff, &meta);
        Ok(Self {
            meta,
            engine,
            ff,
            tables,
            no_match: NoMatch::Error,
        })
    }

    /// Set the bonded no-match policy (chaining). `strict=true` →
    /// an unmatched bonded term is an error; `strict=false` → it is skipped.
    pub fn with_strict(mut self, strict: bool) -> Self {
        self.no_match = if strict {
            NoMatch::Error
        } else {
            NoMatch::Skip
        };
        self
    }

    /// Access the typing metadata.
    pub fn meta(&self) -> &OplsTypingMetadata {
        &self.meta
    }
}

impl Typifier for OplsAaTypifier {
    /// Type the atoms, then match every bonded term.
    ///
    /// Node annotations: `type` (a `Type` under `atom/full` with the row's
    /// `mass` and `charge`, or a plain value when the library has no row) and
    /// `class`. Bonded annotations: `type`, the matched class-keyed name with
    /// its library params. Styles: every library style, in library order. Pairs: the library's pair rows among the atom types used.
    ///
    /// # Errors
    ///
    /// Propagates atom-typing and bonded-matching errors. In strict mode
    /// (an unmatched bonded term is an error) also returns `Err` naming every atom no def typed,
    /// before any bonded term is matched.
    fn assign(&self, graph: &mut Atomistic) -> Result<TypeAssignment, TypingError> {
        let atoms = typify_atoms(graph, &self.engine, &self.meta, &self.ff)
            .map_err(TypingError::malformed)?;
        if self.no_match == NoMatch::Error {
            let untyped: Vec<NodeId> = graph
                .atoms()
                .map(|(id, _)| id)
                .filter(|id| !atoms.types.contains_key(id))
                .collect();
            if !untyped.is_empty() {
                return Err(TypingError::MissingAtomType {
                    table: "OPLS".to_owned(),
                    detail: crate::ff::typifier::name_atoms(graph, &untyped),
                });
            }
        }
        let mut m = typify_bonded_with(graph, &atoms.types, &self.tables, self.no_match)
            .map_err(|detail| TypingError::MissingTerms { detail })?;
        let used: HashSet<&str> = atoms.types.values().map(String::as_str).collect();
        m.nodes = atoms.nodes;
        m.declare_styles_of(&self.ff);
        m.add_pairs_among(&self.ff, &used);
        Ok(m)
    }

    /// The OPLS-AA force field this typifier matches against.
    fn source_forcefield(&self) -> &ForceField {
        &self.ff
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ff::ir::Params;
    use crate::ff::typifier::Typing;
    use crate::props;
    use molrs::core::Element;
    use molrs::core::PropValue;
    use molrs::core::Props;
    use molrs::core::keys;
    use std::collections::{BTreeMap, BTreeSet};

    /// Methylsilane `H3C-SiH3`, hand-built: C is atom 0, Si is atom 1, the
    /// hydrogens on C are 2..=4 and the hydrogens on Si are 5..=7.
    fn methylsilane() -> Atomistic {
        let mut g = Atomistic::new();
        let c = g
            .add_atom(props! {
                keys::ATOMIC_NUMBER => Element::C,
                keys::X => 0.0,
                keys::Y => 0.0,
                keys::Z => 0.0,
            })
            .unwrap();
        let si = g
            .add_atom(props! {
                keys::ATOMIC_NUMBER => Element::Si,
                keys::X => 1.9,
                keys::Y => 0.0,
                keys::Z => 0.0,
            })
            .unwrap();
        g.add_bond(c, si, props! {}).unwrap();
        for heavy in [c, si] {
            for k in 0..3 {
                let h = g
                    .add_atom(props! {
                        keys::ATOMIC_NUMBER => Element::H,
                        keys::X => 0.3 * k as f64,
                        keys::Y => 1.0,
                        keys::Z => 0.0,
                    })
                    .unwrap();
                g.add_bond(heavy, h, props! {}).unwrap();
            }
        }
        g
    }

    /// The name an untyped-atom error gives one atom: its 0-based index in graph
    /// atom order and its element, the ATD typifier's `atom {i} ({element})`.
    fn atom_label(index: usize, element: &str) -> String {
        format!("atom {index} ({element})")
    }

    /// Strict typing of a molecule with atoms no def matches is an `Err` that
    /// names every untyped atom — and only those.
    ///
    /// Hand-derived from the shipped rules: `[#1]-[C;X4]` (opls_140) types the
    /// three hydrogens on carbon (atoms 2..=4); no rule types a carbon bonded to
    /// silicon, silicon itself, or a hydrogen on silicon, so atoms 0, 1 and
    /// 5..=7 stay untyped.
    #[test]
    fn strict_typify_names_every_untyped_atom() {
        let err = Typing::new(OplsAaTypifier::oplsaa().with_strict(true))
            .typify(&methylsilane())
            .expect_err("strict typing must refuse a partly typed molecule")
            .to_string();

        for (i, el) in [(0, "C"), (1, "Si"), (5, "H"), (6, "H"), (7, "H")] {
            let label = atom_label(i, el);
            assert!(err.contains(&label), "err names {label}: {err}");
        }
        for i in 2..=4 {
            let label = atom_label(i, "H");
            assert!(
                !err.contains(&label),
                "err must not name typed {label}: {err}"
            );
        }
    }

    /// Non-strict typing of the same molecule keeps its current contract: `Ok`,
    /// untyped atoms left without a `type`.
    #[test]
    fn non_strict_typify_accepts_a_partly_typed_molecule() {
        let typed =
            Typing::new(OplsAaTypifier::oplsaa().with_strict(false)).typify(&methylsilane());
        assert!(typed.is_ok(), "non-strict typing stays Ok: {typed:?}");
    }

    /// One-type OPLS XML whose `opls_a` carries the given `overrides`
    /// attribute (none when `None`).
    fn one_type_xml(overrides: Option<&str>) -> String {
        let overrides = overrides.map_or(String::new(), |o| format!(r#" overrides="{o}""#));
        format!(
            r#"<ForceField name="OPLS-AA" combining_rule="geometric">
  <AtomTypes>
    <Type name="opls_a" class="CT" element="C" mass="12.011" def="[#6]"{overrides}/>
  </AtomTypes>
  <NonbondedForce coulomb14scale="0.5" lj14scale="0.5">
    <Atom type="opls_a" charge="0.0" sigma="0.35" epsilon="0.276144"/>
  </NonbondedForce>
</ForceField>"#
        )
    }

    /// The OPLS typing reader refuses a dangling override: `opls_a` overrides
    /// `opls_missing`, which the XML never declares, so no table — and so no
    /// typifier — is built, and the error names both. The same XML without
    /// the attribute builds.
    #[test]
    fn a_dangling_override_refuses_the_typing_table() {
        use crate::io::openmm_xml::OpenmmXmlReader;
        let reader = OpenmmXmlReader::new();
        reader
            .read_opls_typifier(one_type_xml(None).as_bytes())
            .expect("the XML without the override builds");
        let Err(e) = reader.read_opls_typifier(one_type_xml(Some("opls_missing")).as_bytes())
        else {
            panic!("a dangling override must refuse the table");
        };
        let e = e.to_string();
        assert!(e.contains("opls_a"), "err names the overriding type: {e}");
        assert!(e.contains("opls_missing"), "err names the absent type: {e}");
    }

    /// A typifier whose defs cover C and H only, over a force field whose
    /// all-wildcard bonded rows match every fully typed term — so a strict
    /// failure can come from atom coverage alone, never from a bonded miss.
    fn c_h_only_typifier() -> OplsAaTypifier {
        let mut meta = OplsTypingMetadata::new();
        let row = |class: &str, def: &str| OplsTypeRow {
            class: class.to_string(),
            def: Some(def.to_string()),
            overrides: Vec::new(),
            priority: None,
            layer: 0,
        };
        meta.insert("opls_c", row("CT", "[C;X4]"));
        meta.insert("opls_h", row("HC", "[H]"));

        let mut ff = ForceField::new("OPLS-AA");
        ff.def_style("bond", "harmonic", Params::new())
            .unwrap()
            .def_type(
                "X-X",
                &["X", "X"],
                Params::from_iter([("k", 1.0), ("r0", 1.0)]),
            )
            .unwrap();
        ff.def_style("angle", "harmonic", Params::new())
            .unwrap()
            .def_type(
                "X-X-X",
                &["X", "X", "X"],
                Params::from_iter([("k", 1.0), ("theta0", 108.9)]),
            )
            .unwrap();
        ff.def_style("dihedral", "opls", Params::new())
            .unwrap()
            .def_type(
                "X-X-X-X",
                &["X", "X", "X", "X"],
                Params::from_iter([("k1", 0.0), ("k2", 0.0), ("k3", 0.0), ("k4", 0.0)]),
            )
            .unwrap();
        OplsAaTypifier::new(meta, ff).unwrap().with_strict(true)
    }

    /// Methanol `CH3-OH`, hand-built: C is atom 0, O is atom 1, the three methyl
    /// hydrogens are 2..=4, the hydroxyl hydrogen is 5.
    fn methanol() -> Atomistic {
        let mut g = Atomistic::new();
        let c = g
            .add_atom(props! {
                keys::ATOMIC_NUMBER => Element::C,
                keys::X => 0.0,
                keys::Y => 0.0,
                keys::Z => 0.0,
            })
            .unwrap();
        let o = g
            .add_atom(props! {
                keys::ATOMIC_NUMBER => Element::O,
                keys::X => 1.4,
                keys::Y => 0.0,
                keys::Z => 0.0,
            })
            .unwrap();
        g.add_bond(c, o, props! {}).unwrap();
        for k in 0..3 {
            let h = g
                .add_atom(props! {
                    keys::ATOMIC_NUMBER => Element::H,
                    keys::X => -0.3 * k as f64,
                    keys::Y => 1.0,
                    keys::Z => 0.0,
                })
                .unwrap();
            g.add_bond(c, h, props! {}).unwrap();
        }
        let h = g
            .add_atom(props! {
                keys::ATOMIC_NUMBER => Element::H,
                keys::X => 1.8,
                keys::Y => 0.9,
                keys::Z => 0.0,
            })
            .unwrap();
        g.add_bond(o, h, props! {}).unwrap();
        g
    }

    // -- Typing<OplsAaTypifier> output (system-forcefield-07) -----------------

    /// Ethane `CH3-CH3`, hand-built: carbons are atoms 0 and 1, the three
    /// hydrogens on C0 are 2..=4, the three on C1 are 5..=7.
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
                        keys::X => 0.3 * k as f64,
                        keys::Y => 1.0,
                        keys::Z => 0.0,
                    })
                    .unwrap();
                g.add_bond(c, h, props! {}).unwrap();
            }
        }
        g
    }

    /// A two-type library (`opls_135` CT carbon, `opls_140` HC hydrogen) plus
    /// one extra type ethane never uses (`opls_154` OH oxygen) with its own
    /// atom, pair and bond rows.
    fn ethane_library_typifier() -> OplsAaTypifier {
        let mut meta = OplsTypingMetadata::new();
        let row = |class: &str, def: &str| OplsTypeRow {
            class: class.to_string(),
            def: Some(def.to_string()),
            overrides: Vec::new(),
            priority: None,
            layer: 0,
        };
        meta.insert("opls_135", row("CT", "[C;X4]"));
        meta.insert("opls_140", row("HC", "[H]"));
        meta.insert("opls_154", row("OH", "[O;X2]"));

        let mut ff = ForceField::new("OPLS-AA");
        ff.def_style("atom", "full", Params::new())
            .unwrap()
            .def_type(
                "opls_135",
                &[],
                Params::from_iter([("mass", 12.011), ("charge", -0.18)]),
            )
            .unwrap()
            .def_type(
                "opls_140",
                &[],
                Params::from_iter([("mass", 1.008), ("charge", 0.06)]),
            )
            .unwrap()
            .def_type(
                "opls_154",
                &[],
                Params::from_iter([("mass", 15.999), ("charge", -0.683)]),
            )
            .unwrap();
        ff.def_style("pair", "lj/cut", Params::from_iter([("cutoff", 10.0)]))
            .unwrap()
            .def_type(
                "opls_135",
                &["opls_135"],
                Params::from_iter([("epsilon", 0.066), ("sigma", 3.5)]),
            )
            .unwrap()
            .def_type(
                "opls_140",
                &["opls_140"],
                Params::from_iter([("epsilon", 0.03), ("sigma", 2.5)]),
            )
            .unwrap()
            .def_type(
                "opls_154",
                &["opls_154"],
                Params::from_iter([("epsilon", 0.17), ("sigma", 3.12)]),
            )
            .unwrap();
        ff.def_style("pair", "coul/cut", Params::from_iter([("cutoff", 10.0)]))
            .unwrap();
        ff.def_style("bond", "harmonic", Params::new())
            .unwrap()
            .def_type(
                "CT-HC",
                &["CT", "HC"],
                Params::from_iter([("k", 680.0), ("r0", 1.09)]),
            )
            .unwrap()
            .def_type(
                "CT-CT",
                &["CT", "CT"],
                Params::from_iter([("k", 536.0), ("r0", 1.529)]),
            )
            .unwrap()
            .def_type(
                "CT-OH",
                &["CT", "OH"],
                Params::from_iter([("k", 640.0), ("r0", 1.41)]),
            )
            .unwrap();
        ff.def_style("angle", "harmonic", Params::new())
            .unwrap()
            .def_type(
                "HC-CT-HC",
                &["HC", "CT", "HC"],
                Params::from_iter([("k", 66.0), ("theta0", 107.8)]),
            )
            .unwrap()
            .def_type(
                "CT-CT-HC",
                &["CT", "CT", "HC"],
                Params::from_iter([("k", 75.0), ("theta0", 110.7)]),
            )
            .unwrap();
        ff.def_style("dihedral", "opls", Params::new())
            .unwrap()
            .def_type(
                "HC-CT-CT-HC",
                &["HC", "CT", "CT", "HC"],
                Params::from_iter([("k1", 0.0), ("k2", 0.0), ("k3", 0.6276), ("k4", 0.0)]),
            )
            .unwrap();

        OplsAaTypifier::new(meta, ff).unwrap().with_strict(false)
    }

    /// Ethane typed through the base: the typed copy and the base.
    fn typed_ethane() -> (Atomistic, crate::ff::typifier::Typing<OplsAaTypifier>) {
        let mut typing = crate::ff::typifier::Typing::new(ethane_library_typifier());
        let typed = typing.typify(&ethane()).expect("ethane types");
        (typed, typing)
    }

    fn str_prop(props: &Props, key: &str) -> Option<String> {
        match props.get(key) {
            Some(PropValue::Str(s)) => Some(s.clone()),
            _ => None,
        }
    }

    /// Per `(category, style)` of `ff`, the set of type names; styles holding
    /// no type are left out.
    fn output_names(ff: &ForceField) -> BTreeMap<(String, String), BTreeSet<String>> {
        ff.styles()
            .iter()
            .filter_map(|s| {
                let names: BTreeSet<String> = s
                    .defs()
                    .collect_type_params()
                    .into_iter()
                    .map(|(name, _)| name)
                    .collect();
                (!names.is_empty()).then(|| ((s.category().to_owned(), s.name().to_owned()), names))
            })
            .collect()
    }

    /// Every output `(category, style)` holds exactly the labels stamped on the
    /// typed graph: the atom `type`s under `atom/full` and as the self rows of
    /// `pair/lj/cut`, the bond / angle / dihedral `type`s under their one style.
    #[test]
    fn typing_output_names_equal_the_stamped_labels() {
        let (typed, typing) = typed_ethane();

        let atoms: BTreeSet<String> = typed
            .atoms()
            .map(|(_, a)| a.get_str("type").expect("every atom typed").to_owned())
            .collect();
        let bonds: BTreeSet<String> = typed
            .bonds()
            .map(|(_, b)| str_prop(&b.props, "type").expect("every bond typed"))
            .collect();
        let angles: BTreeSet<String> = typed
            .angles()
            .map(|(_, a)| str_prop(&a.props, "type").expect("every angle typed"))
            .collect();
        let dihedrals: BTreeSet<String> = typed
            .dihedrals()
            .map(|(_, d)| str_prop(&d.props, "type").expect("every dihedral typed"))
            .collect();

        // Hand-derived from the defs: [C;X4] types both carbons, [H] every H.
        let expected_atoms: BTreeSet<String> = ["opls_135", "opls_140"].map(str::to_owned).into();
        assert_eq!(atoms, expected_atoms);

        let key = |c: &str, s: &str| (c.to_owned(), s.to_owned());
        let expected = BTreeMap::from([
            (key("atom", "full"), atoms.clone()),
            (key("pair", "lj/cut"), atoms),
            (key("bond", "harmonic"), bonds),
            (key("angle", "harmonic"), angles),
            (key("dihedral", "opls"), dihedrals),
        ]);
        assert_eq!(output_names(typing.forcefield()), expected);
    }

    /// The unused library type `opls_154` and its rows (`atom/full`, the
    /// `pair/lj/cut` self row, the `CT-OH` bond) do not reach the output.
    #[test]
    fn typing_output_omits_the_unused_library_rows() {
        let (_, typing) = typed_ethane();
        let out = typing.forcefield();

        let atom_full = out.get_style("atom", "full").expect("atom/full declared");
        assert!(atom_full.get_atomtype("opls_154").is_none());
        let lj = out
            .get_style("pair", "lj/cut")
            .expect("pair/lj/cut declared");
        assert!(lj.get_pairtype("opls_154", None).is_none());
        let bonds = out
            .get_style("bond", "harmonic")
            .expect("bond/harmonic declared");
        assert!(bonds.type_endpoints("CT-OH").is_none());
    }

    /// Exactly one heavy atom (the O) matches no def: strict typing is an `Err`
    /// naming that atom and no other.
    #[test]
    fn strict_typify_names_the_single_untyped_heavy_atom() {
        let err = Typing::new(c_h_only_typifier())
            .typify(&methanol())
            .expect_err("strict typing must refuse the untyped oxygen")
            .to_string();

        let oxygen = atom_label(1, "O");
        assert!(err.contains(&oxygen), "err names {oxygen}: {err}");
        for (i, el) in [(0, "C"), (2, "H"), (3, "H"), (4, "H"), (5, "H")] {
            let label = atom_label(i, el);
            assert!(
                !err.contains(&label),
                "err must not name typed {label}: {err}"
            );
        }
    }
}
