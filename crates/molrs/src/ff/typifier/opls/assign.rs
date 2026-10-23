//! OPLS-AA bonded-parameter matching (bonds / angles / dihedrals).
//!
//! Given the `opls_NNN` type of every atom the atom typing assigned, this module
//! enumerates every bond / angle / dihedral and resolves the most specific
//! matching bonded type from the potential [`ForceField`]'s `bond` / `angle` /
//! `dihedral` style tables. The winner becomes the term's `type`
//! [`Annotation::Type`] (its name, endpoint pattern and params); nothing is
//! written onto the graph here — the typing base stamps and defines the match.
//!
//! # Why a bespoke matcher (not [`Style::get_bondtype`](crate::ff::forcefield::Style::get_bondtype))
//!
//! OPLS-AA keys its bonded forces on **class** names (`CT`, `HC`, …) and uses
//! wildcard end atoms heavily (`X-CT-CT-X`). The force field's
//! [`get_bondtype`](crate::ff::forcefield::Style::get_bondtype) is *exact-key +
//! symmetric only* — no wildcard, no specificity ordering — so it cannot pick
//! the best of several overlapping candidates. Keeping `get_bondtype` exact is
//! deliberate (see the spec's "Out of scope"); the specificity ranking is the
//! **typifier's** job and lives here.
//!
//! The ranking is `(score, layer)`: [`end_score`] per endpoint, summed by
//! [`sequence_score`] over the better orientation, ties broken by the
//! candidate's overlay layer.
//!
//! # Wildcard
//!
//! The [`OpenmmXmlReader`](crate::io::openmm_xml::OpenmmXmlReader) transcribes
//! the `class*` attributes verbatim, so an OPLS wildcard end arrives here as the
//! **empty string** `""` — the one spelling the shipped tables use too.
//!
//! # No match
//!
//! A term that matches no candidate follows the configured [`NoMatch`] policy:
//! `Err` when strict, left unparametrized otherwise.

use std::collections::HashMap;

use molrs::core::schema::block_names::{ANGLES, BONDS, DIHEDRALS};
use molrs::core::{Atomistic, NodeId};

use crate::ff::forcefield::{ForceField, TypeDef};
use crate::ff::ir::Params;
use crate::ff::typifier::{Annotation, TypeAssignment};

use super::typing_metadata::OplsTypingMetadata;

/// Specificity of one bonded-type *end pattern* against one atom.
///
/// - exact `opls_NNN` type match → `3`;
/// - class match → `1`;
/// - wildcard (`""`) → `0`;
/// - no match → `None`.
///
/// `pattern` is the bonded-type endpoint name (a class name, a wildcard, or — in
/// principle — a type name). `atom_type` is the atom's `opls_NNN` type;
/// `atom_class` is its resolved class (`None` if the type has no class mapping).
fn end_score(pattern: &str, atom_type: &str, atom_class: Option<&str>) -> Option<i64> {
    if is_wildcard(pattern) {
        return Some(0);
    }
    if pattern == atom_type {
        return Some(3);
    }
    if let Some(cls) = atom_class
        && pattern == cls
    {
        return Some(1);
    }
    None
}

/// Best specificity of an ordered bonded-term pattern against an ordered atom
/// list, trying forward and end-for-end-reversed orientations.
///
/// Bonded terms are symmetric under reversal, so both orientations are scored and the larger total returned. An
/// orientation in which any end scores `None` is rejected; the whole call
/// returns `None` only if *both* orientations have a non-matching end.
///
/// `atoms` is `[(type, class), ...]`, the same length as `pattern`.
fn sequence_score(pattern: &[&str], atoms: &[(&str, Option<&str>)]) -> Option<i64> {
    debug_assert_eq!(pattern.len(), atoms.len());
    let score_in = |order: &mut dyn Iterator<Item = &(&str, Option<&str>)>| -> Option<i64> {
        let mut total = 0;
        for (pat, (at_type, at_class)) in pattern.iter().zip(order) {
            total += end_score(pat, at_type, *at_class)?;
        }
        Some(total)
    };
    let forward = score_in(&mut atoms.iter());
    let reversed = score_in(&mut atoms.iter().rev());
    match (forward, reversed) {
        (Some(a), Some(b)) => Some(a.max(b)),
        (Some(a), None) | (None, Some(a)) => Some(a),
        (None, None) => None,
    }
}

/// The bond style the candidate tables are read from.
const BOND_STYLE: &str = "harmonic";
/// The angle style the candidate tables are read from.
const ANGLE_STYLE: &str = "harmonic";
/// A bonded-term candidate from the force field: its type name, endpoint class
/// pattern (length 2 / 3 / 4), the precomputed overlay layer, and the params to
/// write on a match.
struct Candidate {
    /// Force-field type name (e.g. `"CT-CT"`), the term's `type`.
    name: String,
    /// Endpoint pattern (class names / wildcards), e.g. `["X", "CT", "CT", "X"]`.
    pattern: Vec<String>,
    /// Overlay layer = max layer over the pattern's classes (CL&P / CL&Pol).
    layer: u32,
    /// The type's params (e.g. `k`/`r0`), defined and stamped for the term.
    params: Params,
    /// The style the type is defined under.
    style: String,
}

/// Per-arity candidate tables, built once from the force field.
///
/// Built from the `bond/harmonic` and `angle/harmonic` styles and every
/// `dihedral` style of the force field (an OpenMM-XML read stores
/// `<RBTorsionForce>` rows as `multi/harmonic`).
pub struct CandidateTables {
    bonds: Vec<Candidate>,
    angles: Vec<Candidate>,
    dihedrals: Vec<Candidate>,
    /// `opls_NNN` → class (for resolving each atom's class at match time).
    type_to_class: HashMap<String, String>,
}

impl CandidateTables {
    /// Build the candidate tables and the type→class map from a force field +
    /// typing metadata.
    ///
    /// The class→layer map (used to compute each candidate's overlay layer) is
    /// derived from `meta`: `class → max(layer)` over every type carrying that
    /// class ([`build_type_class_layer`]).
    pub fn build(ff: &ForceField, meta: &OplsTypingMetadata) -> Self {
        let (type_to_class, class_to_layer) = build_type_class_layer(meta);

        let layer_of = |classes: &[&str]| -> u32 {
            classes
                .iter()
                .map(|c| class_to_layer.get(*c).copied().unwrap_or(0))
                .max()
                .unwrap_or(0)
        };

        let candidate = |t: &TypeDef, style: &str| {
            let classes: Vec<&str> = t.endpoints.iter().map(String::as_str).collect();
            Candidate {
                name: t.name.clone(),
                pattern: t.endpoints.to_vec(),
                layer: layer_of(&classes),
                params: t.params.clone(),
                style: style.to_owned(),
            }
        };
        let candidates = |category: &str, style: &str| -> Vec<Candidate> {
            ff.get_style(category, style)
                .map(|s| {
                    s.defs()
                        .types()
                        .iter()
                        .map(|t| candidate(t, style))
                        .collect()
                })
                .unwrap_or_default()
        };
        let bonds = candidates("bond", BOND_STYLE);
        let angles = candidates("angle", ANGLE_STYLE);
        let dihedrals = ff
            .get_styles("dihedral")
            .into_iter()
            .flat_map(|style| {
                style
                    .defs()
                    .types()
                    .iter()
                    .map(move |t| candidate(t, style.name()))
            })
            .collect();

        Self {
            bonds,
            angles,
            dihedrals,
            type_to_class,
        }
    }

    /// Resolve an atom's `(type, class)` pair from its `opls_NNN` type name.
    fn atom_of<'a>(&'a self, atom_type: &'a str) -> (&'a str, Option<&'a str>) {
        (
            atom_type,
            self.type_to_class.get(atom_type).map(String::as_str),
        )
    }

    /// Pick the best-ranked candidate for `atoms` from `table`. Highest
    /// `(score, layer)` wins; `None` if no candidate matches.
    fn best<'a>(table: &'a [Candidate], atoms: &[(&str, Option<&str>)]) -> Option<&'a Candidate> {
        let mut best_key: Option<(i64, u32)> = None;
        let mut best: Option<&'a Candidate> = None;
        for cand in table {
            let pat: Vec<&str> = cand.pattern.iter().map(String::as_str).collect();
            let Some(score) = sequence_score(&pat, atoms) else {
                continue;
            };
            let key = (score, cand.layer);
            if best_key.is_none_or(|cur| key > cur) {
                best_key = Some(key);
                best = Some(cand);
            }
        }
        best
    }

    /// The annotations of one bonded term with endpoint atoms `ends`, matched
    /// against `table`.
    ///
    /// A match is `type` → [`Annotation::Type`] named by the winning
    /// candidate, with its endpoint pattern and params. A term with an
    /// untyped endpoint, or that no candidate covers, follows `policy`: [`NoMatch::Skip`] gives it no annotation, [`NoMatch::Error`]
    /// is an `Err`.
    #[allow(clippy::too_many_arguments)]
    fn annotate(
        &self,
        graph: &Atomistic,
        types: &HashMap<NodeId, String>,
        ends: &[NodeId],
        table: &[Candidate],
        policy: NoMatch,
    ) -> Result<Vec<(String, Annotation)>, String> {
        let Some(names) = ends
            .iter()
            .map(|id| types.get(id).cloned())
            .collect::<Option<Vec<String>>>()
        else {
            untyped_endpoint(graph, types, ends, policy)?;
            return Ok(Vec::new());
        };
        let atoms: Vec<(&str, Option<&str>)> = names.iter().map(|t| self.atom_of(t)).collect();
        if let Some(cand) = Self::best(table, &atoms) {
            return Ok(vec![(
                "type".to_owned(),
                Annotation::Type {
                    style: cand.style.clone(),
                    name: cand.name.clone(),
                    endpoints: cand.pattern.clone(),
                    params: cand.params.clone(),
                },
            )]);
        }
        match policy {
            NoMatch::Error => Err(format!("OPLS: no bonded type for {}", names.join("-"))),
            NoMatch::Skip => Ok(Vec::new()), // leave the term unparametrized
        }
    }
}

/// Whether an endpoint pattern is the IR's wildcard, `""`: what the OpenMM
/// reader transcribes an absent class as and what the OPLS tables are
/// generated with.
fn is_wildcard(pattern: &str) -> bool {
    pattern.is_empty()
}

/// Map each `opls_NNN` type to its class, and each class to its highest overlay
/// layer, from the [`OplsTypingMetadata`] rows (each carries `class` + `layer`).
fn build_type_class_layer(
    meta: &OplsTypingMetadata,
) -> (HashMap<String, String>, HashMap<String, u32>) {
    let mut type_to_class = HashMap::new();
    let mut class_to_layer: HashMap<String, u32> = HashMap::new();
    for (name, row) in meta.iter() {
        type_to_class.insert(name.clone(), row.class.clone());
        if !is_wildcard(&row.class) {
            let e = class_to_layer.entry(row.class.clone()).or_insert(0);
            *e = (*e).max(row.layer);
        }
    }
    (type_to_class, class_to_layer)
}

/// Strict-mode policy for a term that matches no candidate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NoMatch {
    /// Strict: a missing bonded parameter is a hard error.
    Error,
    /// Lenient: leave the term unparametrized and continue.
    Skip,
}

/// The bonded annotations of a graph whose atoms carry the OPLS `types`,
/// choosing each term's type by the OPLS specificity + layer ranking.
///
/// For every enumerated bond / angle / dihedral:
/// 1. resolve each endpoint atom's `(type, class)`;
/// 2. scan the matching candidate table for the highest `(score, layer)`;
/// 3. on a match, annotate `type` with the winning type (name, endpoint
///    pattern, params);
/// 4. on no match, apply `policy`.
///
/// Angles and dihedrals are enumerated onto `graph` from the bond graph via
/// the shared typifier topology helper (clearing any pre-existing generated
/// ones), mirroring the MMFF typifier; the returned [`TypeAssignment`] holds `bonds`,
/// `angles` and `dihedrals`, positional against `graph` after that
/// enumeration. Only atoms in `types` participate: under [`NoMatch::Error`] a
/// term with any untyped endpoint is an `Err` naming that endpoint; under
/// [`NoMatch::Skip`] the term gets no annotation.
///
/// # Errors
///
/// Returns `Err` if `policy` is [`NoMatch::Error`] and a term has an untyped
/// endpoint or a term matches no candidate; or if topology enumeration
/// fails.
pub(crate) fn typify_bonded_with(
    graph: &mut Atomistic,
    types: &HashMap<NodeId, String>,
    tables: &CandidateTables,
    policy: NoMatch,
) -> Result<TypeAssignment, String> {
    let mut m = TypeAssignment::default();

    // --- bonds (already present from the input topology) ---
    let bonds: Vec<[NodeId; 2]> = graph
        .bonds()
        .map(|(_, b)| [b.nodes[0], b.nodes[1]])
        .collect();
    *m.link_mut(BONDS) = bonds
        .iter()
        .map(|ends| tables.annotate(graph, types, ends, &tables.bonds, policy))
        .collect::<Result<_, _>>()?;

    // --- enumerate angles + dihedrals from the bond graph (clear existing) ---
    graph
        .generate_topology(true, true, false, true)
        .map_err(|e| e.to_string())?;

    let angles: Vec<[NodeId; 3]> = graph
        .angles()
        .map(|(_, a)| [a.nodes[0], a.nodes[1], a.nodes[2]])
        .collect();
    *m.link_mut(ANGLES) = angles
        .iter()
        .map(|ends| tables.annotate(graph, types, ends, &tables.angles, policy))
        .collect::<Result<_, _>>()?;

    let dihedrals: Vec<[NodeId; 4]> = graph
        .dihedrals()
        .map(|(_, d)| [d.nodes[0], d.nodes[1], d.nodes[2], d.nodes[3]])
        .collect();
    *m.link_mut(DIHEDRALS) = dihedrals
        .iter()
        .map(|ends| tables.annotate(graph, types, ends, &tables.dihedrals, policy))
        .collect::<Result<_, _>>()?;

    Ok(m)
}

/// Apply `policy` to a bonded term whose endpoints `ends` include an untyped
/// atom: [`NoMatch::Skip`] accepts (the caller skips the term);
/// [`NoMatch::Error`] refuses, naming every untyped endpoint.
fn untyped_endpoint(
    mol: &Atomistic,
    types: &HashMap<NodeId, String>,
    ends: &[NodeId],
    policy: NoMatch,
) -> Result<(), String> {
    match policy {
        NoMatch::Skip => Ok(()),
        NoMatch::Error => {
            let untyped: Vec<NodeId> = ends
                .iter()
                .copied()
                .filter(|id| !types.contains_key(id))
                .collect();
            Err(format!(
                "OPLS: bonded term has untyped endpoint {}",
                crate::ff::typifier::name_atoms(mol, &untyped)
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::props;
    use molrs::core::Element;
    use molrs::core::keys;

    // --- end_score: 3 / 1 / 0 / None ---------------------------------------

    #[test]
    fn end_score_exact_type_is_3() {
        assert_eq!(end_score("opls_135", "opls_135", Some("CT")), Some(3));
    }

    #[test]
    fn end_score_class_is_1() {
        // Pattern is the class name (OPLS bonded forces key on class).
        assert_eq!(end_score("CT", "opls_135", Some("CT")), Some(1));
    }

    #[test]
    fn end_score_wildcard_is_0() {
        // The empty string is the wildcard; `*` and `X` are names like any other.
        assert_eq!(end_score("", "opls_135", Some("CT")), Some(0));
        assert_eq!(end_score("*", "opls_135", Some("CT")), None);
        assert_eq!(end_score("X", "opls_135", Some("CT")), None);
    }

    #[test]
    fn end_score_no_match_is_none() {
        assert_eq!(end_score("OH", "opls_135", Some("CT")), None);
        // Class given but does not equal the pattern, and no type match either.
        assert_eq!(end_score("HC", "opls_140", Some("CT")), None);
    }

    #[test]
    fn end_score_no_class_falls_through_to_none() {
        // No class resolved, pattern is neither the type nor a wildcard => None.
        assert_eq!(end_score("CT", "opls_135", None), None);
    }

    // --- sequence_score: additive, symmetric, None-propagating -------------

    #[test]
    fn sequence_score_sums_per_end() {
        // CT-CT bond against two CT atoms: 1 + 1 = 2.
        let atoms = [("opls_135", Some("CT")), ("opls_136", Some("CT"))];
        assert_eq!(sequence_score(&["CT", "CT"], &atoms), Some(2));
    }

    #[test]
    fn sequence_score_is_reversal_symmetric() {
        // Pattern CT-OH matched against atoms (OH, CT) only works reversed.
        let atoms = [("opls_154", Some("OH")), ("opls_135", Some("CT"))];
        assert_eq!(sequence_score(&["CT", "OH"], &atoms), Some(2));
        // Forward-only would fail (CT vs OH-class, OH vs CT-class) -> reversed wins.
    }

    #[test]
    fn sequence_score_returns_best_orientation() {
        // Forward: exact(3) + wildcard(0) = 3; reversed: wildcard(0)+class(1)=1.
        // Best = 3.
        let atoms = [("opls_135", Some("CT")), ("opls_140", Some("HC"))];
        assert_eq!(sequence_score(&["opls_135", ""], &atoms), Some(3));
    }

    #[test]
    fn sequence_score_none_when_no_orientation_matches() {
        let atoms = [("opls_135", Some("CT")), ("opls_140", Some("HC"))];
        // Pattern OH-OH matches neither end in either orientation.
        assert_eq!(sequence_score(&["OH", "OH"], &atoms), None);
    }

    // --- ranking: specificity beats wildcard; layer breaks ties ------------

    fn cand(pattern: &[&str], layer: u32, k: f64) -> Candidate {
        Candidate {
            name: pattern.join("-"),
            pattern: pattern.iter().map(|s| s.to_string()).collect(),
            layer,
            params: Params::from_iter([("k", k)]),
            style: BOND_STYLE.to_owned(),
        }
    }

    #[test]
    fn ranking_fully_resolved_beats_wildcard() {
        // Two bond candidates both match a CT-HC bond: the fully class-resolved
        // CT-HC (score 2) must beat the wildcard X-HC (score 0+1=1).
        let table = vec![cand(&["", "HC"], 0, 1.0), cand(&["CT", "HC"], 0, 2.0)];
        let atoms = [("opls_135", Some("CT")), ("opls_140", Some("HC"))];
        let best = CandidateTables::best(&table, &atoms).expect("a match");
        assert_eq!(
            best.params.get("k").and_then(|v| v.as_num()),
            Some(2.0),
            "fully-resolved candidate wins"
        );
    }

    #[test]
    fn ranking_equal_score_higher_layer_wins() {
        // Two CT-HC candidates with equal specificity; the higher overlay layer
        // (CL&P/CL&Pol) wins the tie.
        let table = vec![cand(&["CT", "HC"], 0, 1.0), cand(&["CT", "HC"], 2, 9.0)];
        let atoms = [("opls_135", Some("CT")), ("opls_140", Some("HC"))];
        let best = CandidateTables::best(&table, &atoms).expect("a match");
        assert_eq!(
            best.params.get("k").and_then(|v| v.as_num()),
            Some(9.0),
            "higher-layer candidate wins"
        );
    }

    #[test]
    fn ranking_no_candidate_matches_is_none() {
        let table = vec![cand(&["OH", "OH"], 0, 1.0)];
        let atoms = [("opls_135", Some("CT")), ("opls_140", Some("HC"))];
        assert!(CandidateTables::best(&table, &atoms).is_none());
    }

    // --- build_type_class_layer (class -> max layer) -----------------------

    #[test]
    fn class_to_layer_takes_the_max() {
        use crate::ff::typifier::OplsTypeRow;
        let mut meta = OplsTypingMetadata::new();
        let row = |class: &str, layer: u32| OplsTypeRow {
            class: class.to_string(),
            def: Some("[C]".into()),
            overrides: Vec::new(),
            priority: None,
            layer,
        };
        meta.insert("opls_a", row("CT", 0));
        meta.insert("opls_b", row("CT", 2)); // same class, higher layer
        let (t2c, c2l) = build_type_class_layer(&meta);
        assert_eq!(t2c.get("opls_a").map(String::as_str), Some("CT"));
        assert_eq!(c2l.get("CT"), Some(&2), "class layer is the max over types");
    }

    // --- untyped endpoints vs the no-match policy ---------------------------

    /// A force field whose one bond row is the all-wildcard `X-X`.
    fn wildcard_bond_ff() -> ForceField {
        let mut ff = ForceField::new("OPLS-AA");
        ff.def_style("bond", "harmonic", Params::new())
            .unwrap()
            .def_type(
                "X-X",
                &["X", "X"],
                Params::from_iter([("k", 1.0), ("r0", 1.5)]),
            )
            .unwrap();
        ff
    }

    /// Tables over [`wildcard_bond_ff`] and metadata that knows `opls_135`
    /// (class `CT`): any bond whose two endpoints are typed matches, so a
    /// failure can only come from an untyped endpoint.
    fn wildcard_bond_tables() -> CandidateTables {
        use crate::ff::typifier::OplsTypeRow;
        let mut meta = OplsTypingMetadata::new();
        meta.insert(
            "opls_135",
            OplsTypeRow {
                class: "CT".to_string(),
                def: Some("[C;X4]".into()),
                overrides: Vec::new(),
                priority: None,
                layer: 0,
            },
        );
        CandidateTables::build(&wildcard_bond_ff(), &meta)
    }

    /// A C-O bond where only the carbon (atom 0) is typed `opls_135`; the
    /// oxygen (atom 1) is untyped. Returns the graph and the atom types.
    fn half_typed_bond() -> (Atomistic, HashMap<NodeId, String>) {
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
        (g, HashMap::from([(c, "opls_135".to_string())]))
    }

    /// Strict policy refuses a bonded term with an untyped endpoint, naming the
    /// untyped atom (`atom {index} ({element})`, 0-based graph order).
    #[test]
    fn strict_policy_refuses_untyped_endpoint() {
        let (mut g, types) = half_typed_bond();
        let err = typify_bonded_with(&mut g, &types, &wildcard_bond_tables(), NoMatch::Error)
            .expect_err("strict bonded typing must refuse an untyped endpoint");
        assert!(err.contains("atom 1 (O)"), "err names the untyped O: {err}");
    }

    /// Lenient policy keeps skipping the term: `Ok`, and the bond stays
    /// unlabelled even though a wildcard row would match any typed pair.
    #[test]
    fn skip_policy_leaves_untyped_endpoint_bond_unlabelled() {
        let (mut out, types) = half_typed_bond();
        let mut m = typify_bonded_with(&mut out, &types, &wildcard_bond_tables(), NoMatch::Skip)
            .expect("lenient bonded typing accepts an untyped endpoint");
        m.declare_styles_of(&wildcard_bond_ff());
        m.apply_to(&mut out, &mut ForceField::new("out"))
            .expect("the match writes");
        let (_, bond) = out.bonds().next().expect("the one bond");
        assert_eq!(bond.props.get("type"), None, "bond stays untyped");
    }
}
