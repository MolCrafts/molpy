//! Layered, dependency-aware OPLS-AA atom typing.
//!
//! [`LayeredTypingEngine`] drives atom typing level by level, using the
//! [`OplsDependencyAnalyzer`] to order defs so that a def referencing a type
//! name through a `%label` context predicate is only matched *after* that type
//! has been assigned in an earlier level. It accumulates a
//! `HashMap<NodeId, String>` of assigned types and feeds it back as the SMARTS
//! **context-label map** (so `%opls_NNN` predicates can read the current
//! assignments).
//!
//! - Levels are processed in ascending order ([`LayeredTypingEngine::assign`]).
//! - A normal level is resolved in a single pass: every def of the level is
//!   matched under the current assignment context, the per-atom candidates are
//!   ranked, and the level's winners are merged into the assignment.
//! - A level containing a circular-dependency group is resolved by fixed-point
//!   iteration (`max_iterations = 10`, converging on assignment-map equality).
//!
//! # Ranking: pairwise dominance
//!
//! Type `a` *dominates* type `b` iff `a` sits on a higher overlay `layer`, or
//! the two share a layer and `a` overrides `b` — directly or through a chain of
//! declared `overrides`. Dominance is built once per engine; an overrides cycle,
//! or an override naming a type absent from the metadata, is a build error.
//!
//! - **Within a level**, the winners on an atom are the candidates no other
//!   candidate dominates. Among those, the explicit `priority` (absent = 0)
//!   decides, then the number of query atoms, then the earlier sorted type name.
//!   The result depends on no iteration order.
//! - **Across levels**, a level's winner replaces an atom's current type unless
//!   the current type dominates it.

use std::collections::{HashMap, HashSet};

use molrs::core::{Atomistic, NodeId};
use molrs::perceive::smarts::{MatchOptions, SmartsPattern};

use super::dependency::OplsDependencyAnalyzer;
use super::typing_metadata::OplsTypingMetadata;

/// Maximum fixed-point iterations for a circular-dependency level.
pub const MAX_CIRCULAR_ITERATIONS: usize = 10;

/// A compiled, ranked def: its SMARTS pattern plus the tie-break inputs used
/// among candidates that no other candidate dominates.
struct RankedDef {
    name: String,
    pattern: SmartsPattern,
    /// Explicit `priority`, 0 when absent.
    priority: i64,
    /// Number of query atoms.
    specificity: usize,
    /// Stable definition order (index after sorting type names).
    order: usize,
}

impl RankedDef {
    /// The tie-break key among undominated candidates; the greater key wins:
    /// higher priority, then more query atoms, then the EARLIER sorted name.
    fn tie_break(&self) -> (i64, usize, std::cmp::Reverse<usize>) {
        (
            self.priority,
            self.specificity,
            std::cmp::Reverse(self.order),
        )
    }
}

/// The pairwise dominance relation over the typing metadata.
///
/// `dominates(a, b)` holds iff `layer(a) > layer(b)`, or the layers are equal
/// and `b` lies in the transitive closure of `a`'s declared overrides.
pub(super) struct Dominance {
    /// Type name → its overlay layer.
    layer: HashMap<String, u32>,
    /// Type name → every type it overrides, directly or transitively.
    closure: HashMap<String, HashSet<String>>,
}

impl Dominance {
    /// Build the relation from `meta`.
    ///
    /// # Errors
    ///
    /// `Err` naming both types when an override names a type absent from
    /// `meta`, and `Err` naming every member when the declared overrides form
    /// a cycle.
    pub(super) fn new(meta: &OplsTypingMetadata) -> Result<Self, String> {
        let mut named: Vec<(&String, &super::typing_metadata::OplsTypeRow)> = meta.iter().collect();
        named.sort_by(|a, b| a.0.cmp(b.0));
        for (name, row) in &named {
            if let Some(missing) = row.overrides.iter().find(|o| meta.get(o).is_none()) {
                return Err(format!(
                    "OPLS type {name} overrides {missing}, which is not a type of this force field"
                ));
            }
        }

        let mut closure: HashMap<String, HashSet<String>> = HashMap::with_capacity(named.len());
        for (name, _) in &named {
            let mut reached: HashSet<String> = HashSet::new();
            let mut stack: Vec<&str> = vec![name.as_str()];
            while let Some(current) = stack.pop() {
                let Some(row) = meta.get(current) else {
                    continue;
                };
                for next in &row.overrides {
                    if reached.insert(next.clone()) {
                        stack.push(next);
                    }
                }
            }
            closure.insert((*name).clone(), reached);
        }

        let mut cyclic: Vec<&str> = named
            .iter()
            .map(|(name, _)| name.as_str())
            .filter(|name| closure[*name].contains(*name))
            .collect();
        if !cyclic.is_empty() {
            cyclic.sort_unstable();
            return Err(format!(
                "OPLS overrides form a cycle among {}",
                cyclic.join(", ")
            ));
        }

        let layer = named
            .iter()
            .map(|(name, row)| ((*name).clone(), row.layer))
            .collect();
        Ok(Self { layer, closure })
    }

    /// Whether type `a` dominates type `b`. A name outside the metadata
    /// dominates nothing and is dominated by nothing.
    fn dominates(&self, a: &str, b: &str) -> bool {
        let (Some(la), Some(lb)) = (self.layer.get(a), self.layer.get(b)) else {
            return false;
        };
        la > lb || (la == lb && self.closure.get(a).is_some_and(|c| c.contains(b)))
    }
}

/// Dependency-aware, level-by-level OPLS atom typing engine.
pub struct LayeredTypingEngine {
    analyzer: OplsDependencyAnalyzer,
    dominance: Dominance,
    /// Compiled defs grouped by topological level (index = level).
    by_level: Vec<Vec<RankedDef>>,
}

impl LayeredTypingEngine {
    /// Build the engine from typing metadata.
    ///
    /// Compiles every def-carrying type's SMARTS, computes dependency levels,
    /// buckets the compiled defs by level, and builds the dominance relation.
    /// A bare element symbol (`Li`) is read as its bracket atom (`[Li]`).
    ///
    /// # Errors
    ///
    /// `Err` naming the type for a malformed SMARTS `def`; naming both types
    /// for an override of a type absent from `meta`; naming every member for
    /// an overrides cycle.
    pub fn build(meta: &OplsTypingMetadata) -> Result<Self, String> {
        let dominance = Dominance::new(meta)?;
        let analyzer = OplsDependencyAnalyzer::new(meta);

        // Deterministic definition order: sort by type name.
        let mut named: Vec<(&String, &super::typing_metadata::OplsTypeRow)> = meta.iter().collect();
        named.sort_by(|a, b| a.0.cmp(b.0));

        let max_level = analyzer.max_level().unwrap_or(0);
        let mut by_level: Vec<Vec<RankedDef>> = (0..=max_level).map(|_| Vec::new()).collect();

        for (order, (name, row)) in named.into_iter().enumerate() {
            let Some(def) = row.def.as_deref() else {
                continue; // no-def row
            };
            let pattern = compile_def(def).map_err(|e| {
                format!("OPLS type {name:?}: failed to parse SMARTS def {def:?}: {e}")
            })?;
            let specificity = pattern.n_query_atoms();
            let level = analyzer.level(name).unwrap_or(0);
            if level >= by_level.len() {
                by_level.resize_with(level + 1, Vec::new);
            }
            by_level[level].push(RankedDef {
                name: name.clone(),
                pattern,
                priority: row.priority.unwrap_or(0),
                specificity,
                order,
            });
        }

        Ok(Self {
            analyzer,
            dominance,
            by_level,
        })
    }

    /// Resolve every atom's OPLS type, returning `atom → opls_NNN`.
    ///
    /// Levels are processed in ascending order; the accumulated assignment map
    /// is threaded into each level's SMARTS matching as the context-label map,
    /// so `%opls_NNN` defs see the prior levels' results. A level whose defs lie
    /// in a circular-dependency group is resolved by fixed-point iteration.
    pub fn assign(&self, mol: &Atomistic) -> HashMap<NodeId, String> {
        let mut assignments: HashMap<NodeId, String> = HashMap::new();
        for level in 0..self.by_level.len() {
            let defs = &self.by_level[level];
            if defs.is_empty() {
                continue;
            }
            let is_circular = defs.iter().any(|d| self.analyzer.is_circular(&d.name));
            assignments = if is_circular {
                self.resolve_circular(defs, mol, assignments)
            } else {
                self.resolve_level(defs, mol, assignments)
            };
        }
        assignments
    }

    /// Resolve a single (non-circular) level: match every def at this level
    /// against `mol` under the `current` label context, pick each atom's winner
    /// among this level's candidates, then merge — the winner replaces the
    /// atom's current type unless the current type dominates it.
    fn resolve_level(
        &self,
        defs: &[RankedDef],
        mol: &Atomistic,
        current: HashMap<NodeId, String>,
    ) -> HashMap<NodeId, String> {
        // Candidate defs per atom, each def at most once.
        let mut candidates: HashMap<NodeId, Vec<usize>> = HashMap::new();
        for (k, d) in defs.iter().enumerate() {
            for m in d.pattern.find(
                mol,
                MatchOptions {
                    labels: Some(&current),
                    root: None,
                    limit: None,
                },
            ) {
                // Target atom is the root (query atom 0), per RDKit convention.
                let Some(&target) = m.atoms.first() else {
                    continue;
                };
                let entry = candidates.entry(target).or_default();
                if !entry.contains(&k) {
                    entry.push(k);
                }
            }
        }

        let mut result = current;
        for (atom, cands) in candidates {
            let winner = cands
                .iter()
                .map(|&k| &defs[k])
                .filter(|c| {
                    !cands
                        .iter()
                        .any(|&k| self.dominance.dominates(&defs[k].name, &c.name))
                })
                .max_by_key(|c| c.tie_break());
            let Some(winner) = winner else {
                continue;
            };
            let keep = result
                .get(&atom)
                .is_some_and(|t| self.dominance.dominates(t, &winner.name));
            if !keep {
                result.insert(atom, winner.name.clone());
            }
        }
        result
    }

    /// Resolve a circular-dependency level by fixed-point iteration: repeatedly
    /// run [`resolve_level`](Self::resolve_level) (each pass feeding the prior
    /// pass's assignments back as the label context) until the assignment map
    /// stops changing or `MAX_CIRCULAR_ITERATIONS` is reached. Each pass merges
    /// under the same dominance rule.
    fn resolve_circular(
        &self,
        defs: &[RankedDef],
        mol: &Atomistic,
        current: HashMap<NodeId, String>,
    ) -> HashMap<NodeId, String> {
        let mut assignments = current;
        for _ in 0..MAX_CIRCULAR_ITERATIONS {
            let prev = assignments.clone();
            assignments = self.resolve_level(defs, mol, assignments);
            if assignments == prev {
                break;
            }
        }
        assignments
    }

    /// Access the underlying dependency analyzer (levels / circular groups).
    #[cfg(test)]
    pub fn analyzer(&self) -> &OplsDependencyAnalyzer {
        &self.analyzer
    }
}

/// Compile a single SMARTS `def`, reading a bare element symbol (`Li`, which
/// SMARTS only admits in brackets) as its bracket atom (`[Li]`) for XML inputs.
fn compile_def(def: &str) -> Result<SmartsPattern, molrs::core::MolRsError> {
    match SmartsPattern::parse(def) {
        Ok(p) => Ok(p),
        Err(e) => {
            if is_bare_element_symbol(def) {
                SmartsPattern::parse(&format!("[{def}]"))
            } else {
                Err(e)
            }
        }
    }
}

/// Whether `def` is exactly one element symbol (`Li`, `Na`, `Br`, `C`).
fn is_bare_element_symbol(def: &str) -> bool {
    let b = def.as_bytes();
    match b.len() {
        1 => b[0].is_ascii_uppercase(),
        2 => b[0].is_ascii_uppercase() && b[1].is_ascii_lowercase(),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ff::typifier::OplsTypeRow;
    use crate::props;
    use molrs::core::Element;
    use molrs::core::keys;

    fn row(class: &str, def: Option<&str>, overrides: &[&str]) -> OplsTypeRow {
        OplsTypeRow {
            class: class.to_string(),
            def: def.map(str::to_string),
            overrides: overrides.iter().map(|s| s.to_string()).collect(),
            priority: None,
            layer: 0,
        }
    }

    fn meta_with(rows: &[(&str, OplsTypeRow)]) -> OplsTypingMetadata {
        let mut m = OplsTypingMetadata::new();
        for (name, r) in rows {
            m.insert(*name, r.clone());
        }
        m
    }

    /// Ethanol skeleton C-C-O with explicit Hs; returns (graph, O id, H-on-O id).
    fn ethanol() -> (Atomistic, NodeId, NodeId) {
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
        for c in [cm, ch] {
            let n = if c == cm { 3 } else { 2 };
            for k in 0..n {
                let h = g
                    .add_atom(props! {
                        keys::ATOMIC_NUMBER => Element::H,
                        keys::X => 0.3 * (k as f64 + 1.0),
                        keys::Y => 0.9,
                        keys::Z => 0.0,
                    })
                    .unwrap();
                g.add_bond(c, h, props! {}).unwrap();
            }
        }
        (g, o, ho)
    }

    #[test]
    fn level_zero_only_matches_chain1_behaviour() {
        // No %-defs: everything is level 0, resolved in one pass.
        let meta = meta_with(&[
            ("opls_135", row("CT", Some("[C;X4](C)(H)(H)H"), &[])),
            ("opls_140", row("HC", Some("H[C;X4]"), &[])),
        ]);
        let engine = LayeredTypingEngine::build(&meta).unwrap();
        assert_eq!(engine.analyzer().max_level(), Some(0));

        let (g, _o, _ho) = ethanol();
        let assigned = engine.assign(&g);
        // The methyl carbon (CH3 on a C) types opls_135; its 3 H type opls_140.
        let n135 = assigned.values().filter(|t| *t == "opls_135").count();
        let n140 = assigned.values().filter(|t| *t == "opls_140").count();
        assert_eq!(n135, 1, "one CH3 carbon");
        assert!(n140 >= 1, "methyl Hs typed");
    }

    #[test]
    fn dependent_def_resolves_after_its_dependency() {
        // opls_154 (alcohol O, level 0) then opls_155 (H[O;%opls_154], level 1):
        // the hydroxyl H must type opls_155 only after the O is opls_154.
        let meta = meta_with(&[
            ("opls_154", row("OH", Some("[O;X2](H)([!H])"), &[])),
            ("opls_155", row("HO", Some("H[O;%opls_154]"), &[])),
        ]);
        let engine = LayeredTypingEngine::build(&meta).unwrap();
        assert_eq!(engine.analyzer().level("opls_154"), Some(0));
        assert_eq!(engine.analyzer().level("opls_155"), Some(1));

        let (g, o, ho) = ethanol();
        let assigned = engine.assign(&g);
        assert_eq!(
            assigned.get(&o).map(String::as_str),
            Some("opls_154"),
            "alcohol O typed at level 0"
        );
        assert_eq!(
            assigned.get(&ho).map(String::as_str),
            Some("opls_155"),
            "hydroxyl H typed at level 1 via %opls_154"
        );
    }

    #[test]
    fn missing_dependency_leaves_dependent_untyped() {
        // Without the level-0 O def, %opls_154 is never satisfied, so opls_155
        // never matches (the H stays untyped) — and the engine still terminates.
        let meta = meta_with(&[("opls_155", row("HO", Some("H[O;%opls_154]"), &[]))]);
        let engine = LayeredTypingEngine::build(&meta).unwrap();
        let (g, _o, ho) = ethanol();
        let assigned = engine.assign(&g);
        assert!(
            !assigned.contains_key(&ho),
            "no dependency assigned -> dependent stays untyped"
        );
    }

    #[test]
    fn circular_level_iterates_to_fixed_point() {
        // A constructed two-cycle that still terminates: opls_x and opls_y are a
        // circular group at max_level+1. With a level-0 seed (opls_base on the
        // methyl C), the fixed-point loop converges (here neither cyclic def can
        // actually fire on ethanol, so it converges immediately to the seed) and
        // the engine returns without exhausting the iteration cap.
        let meta = meta_with(&[
            ("opls_base", row("CT", Some("[C;X4](C)(H)(H)H"), &[])),
            ("opls_x", row("X", Some("[C;%opls_base][O;%opls_y]"), &[])),
            ("opls_y", row("Y", Some("[N;%opls_x]"), &[])),
        ]);
        let engine = LayeredTypingEngine::build(&meta).unwrap();
        assert_eq!(engine.analyzer().circular_groups().len(), 1);
        let (g, _o, _ho) = ethanol();
        let assigned = engine.assign(&g);
        // The base def still types the methyl carbon; the cyclic defs (needing
        // an O/N neighbour with cyclic types) never fire on ethanol.
        assert_eq!(
            assigned.values().filter(|t| *t == "opls_base").count(),
            1,
            "level-0 seed survives the circular level"
        );
        assert!(
            !assigned.values().any(|t| t == "opls_x" || t == "opls_y"),
            "cyclic defs cannot fire on ethanol"
        );
    }

    #[test]
    fn malformed_def_fails_fast() {
        let meta = meta_with(&[("opls_bad", row("X", Some("[C"), &[]))]);
        match LayeredTypingEngine::build(&meta) {
            Ok(_) => panic!("malformed def should fail fast"),
            Err(e) => assert!(e.contains("opls_bad"), "err names the type: {e}"),
        }
    }

    // -- Pairwise dominance (opls-gromacs-03 § 2) ------------------------------
    //
    // Every case below types the oxygen of the hand-built `ethanol()`: `[#8]`
    // (one query atom) and `[#8]-[#1]` (two) both match it, and `[#7]` / `[#9]`
    // match nothing in ethanol, so a row carrying those defs is never a
    // candidate. All rows are level 0 unless their def carries a `%label`.

    /// A typing row with an explicit priority and layer.
    fn ranked(def: &str, overrides: &[&str], priority: Option<i64>, layer: u32) -> OplsTypeRow {
        OplsTypeRow {
            class: "X".to_string(),
            def: Some(def.to_string()),
            overrides: overrides.iter().map(|s| s.to_string()).collect(),
            priority,
            layer,
        }
    }

    /// The type the engine built from `meta` assigns to ethanol's oxygen.
    fn oxygen_type(meta: &OplsTypingMetadata) -> Option<String> {
        let engine = LayeredTypingEngine::build(meta).expect("engine builds");
        let (g, o, _ho) = ethanol();
        engine.assign(&g).get(&o).cloned()
    }

    /// `opls_a` (`[#8]`, one query atom) overrides the more specific `opls_b`
    /// (`[#8]-[#1]`, two query atoms), so `opls_a` wins on the oxygen.
    ///
    /// `opls_b` also overrides two non-candidates (`[#7]`), which a collapsed
    /// score would count as +2, tying the two and letting specificity decide.
    /// Dominance does not count: `opls_a` dominates `opls_b`, full stop.
    #[test]
    fn override_beats_specificity() {
        let meta = meta_with(&[
            ("opls_a", ranked("[#8]", &["opls_b"], None, 0)),
            (
                "opls_b",
                ranked("[#8]-[#1]", &["opls_n1", "opls_n2"], None, 0),
            ),
            ("opls_n1", ranked("[#7]", &[], None, 0)),
            ("opls_n2", ranked("[#7]", &[], None, 0)),
        ]);
        assert_eq!(oxygen_type(&meta).as_deref(), Some("opls_a"));
    }

    /// `opls_a` overrides `opls_b`, which overrides `opls_c`; `opls_b` never
    /// matches, so the candidates are `opls_a` (`[#8]`) and `opls_c`
    /// (`[#8]-[#1]`). By the transitive closure `opls_a` dominates `opls_c`.
    ///
    /// Collapsed scores would pick `opls_c`: it overrides two non-candidates
    /// (+2) and is overridden once (−1), while `opls_a` overrides one (+1) and
    /// is overridden by two non-candidates (−2).
    #[test]
    fn override_dominance_is_transitive() {
        let meta = meta_with(&[
            ("opls_a", ranked("[#8]", &["opls_b"], None, 0)),
            ("opls_b", ranked("[#9]", &["opls_c"], None, 0)),
            (
                "opls_c",
                ranked("[#8]-[#1]", &["opls_d1", "opls_d2"], None, 0),
            ),
            ("opls_d1", ranked("[#7]", &[], None, 0)),
            ("opls_d2", ranked("[#7]", &[], None, 0)),
            ("opls_x1", ranked("[#7]", &["opls_a"], None, 0)),
            ("opls_x2", ranked("[#7]", &["opls_a"], None, 0)),
        ]);
        assert_eq!(oxygen_type(&meta).as_deref(), Some("opls_a"));
    }

    /// A layer-1 candidate dominates a layer-0 candidate even when the layer-0
    /// one declares it overrides the layer-1 one: overrides only relate types
    /// of the same layer.
    #[test]
    fn higher_layer_beats_an_override() {
        let meta = meta_with(&[
            ("opls_hi", ranked("[#8]", &[], None, 1)),
            ("opls_lo", ranked("[#8]-[#1]", &["opls_hi"], None, 0)),
        ]);
        assert_eq!(oxygen_type(&meta).as_deref(), Some("opls_hi"));
    }

    /// A layer-1 candidate dominates a layer-0 candidate whatever the latter's
    /// explicit priority: priority only orders candidates nothing dominates.
    #[test]
    fn higher_layer_beats_an_explicit_priority() {
        let meta = meta_with(&[
            ("opls_hi", ranked("[#8]", &[], None, 1)),
            ("opls_lo", ranked("[#8]-[#1]", &[], Some(5000), 0)),
        ]);
        assert_eq!(oxygen_type(&meta).as_deref(), Some("opls_hi"));
    }

    /// Unrelated candidates rank first by explicit priority (absent = 0): the
    /// less specific `opls_a` with priority 1 beats `opls_b` with none.
    #[test]
    fn unrelated_candidates_rank_by_priority_first() {
        let meta = meta_with(&[
            ("opls_a", ranked("[#8]", &[], Some(1), 0)),
            ("opls_b", ranked("[#8]-[#1]", &[], None, 0)),
        ]);
        assert_eq!(oxygen_type(&meta).as_deref(), Some("opls_a"));
    }

    /// Equal priority: the candidate with more query atoms wins.
    #[test]
    fn unrelated_candidates_rank_by_query_atom_count_second() {
        let meta = meta_with(&[
            ("opls_a", ranked("[#8]", &[], None, 0)),
            ("opls_b", ranked("[#8]-[#1]", &[], None, 0)),
        ]);
        assert_eq!(oxygen_type(&meta).as_deref(), Some("opls_b"));
    }

    /// Equal priority and size: the earlier sorted name wins.
    #[test]
    fn unrelated_candidates_rank_by_name_last() {
        let meta = meta_with(&[
            ("opls_b", ranked("[#8]", &[], None, 0)),
            ("opls_a", ranked("[#8]", &[], None, 0)),
        ]);
        assert_eq!(oxygen_type(&meta).as_deref(), Some("opls_a"));
    }

    /// An override of a type that is not a candidate relates `opls_a` to
    /// nothing on this atom, so it does not rank `opls_a` above the unrelated,
    /// more specific `opls_b` (both priority 0; size decides).
    #[test]
    fn an_override_of_a_non_candidate_does_not_rank() {
        let meta = meta_with(&[
            ("opls_a", ranked("[#8]", &["opls_n"], None, 0)),
            ("opls_b", ranked("[#8]-[#1]", &[], None, 0)),
            ("opls_n", ranked("[#7]", &[], None, 0)),
        ]);
        assert_eq!(oxygen_type(&meta).as_deref(), Some("opls_b"));
    }

    /// `opls_t0` (`[#8]`, level 0) overrides `opls_t1` (`[#8;%opls_t0]`,
    /// level 1). Level 1 matches the oxygen, but `opls_t0` dominates the
    /// newcomer, so the level-0 assignment is kept.
    #[test]
    fn later_level_keeps_a_dominating_type() {
        let meta = meta_with(&[
            ("opls_t0", ranked("[#8]", &["opls_t1"], None, 0)),
            ("opls_t1", ranked("[#8;%opls_t0]", &[], None, 0)),
        ]);
        let engine = LayeredTypingEngine::build(&meta).expect("engine builds");
        assert_eq!(engine.analyzer().level("opls_t1"), Some(1));
        let (g, o, _ho) = ethanol();
        assert_eq!(
            engine.assign(&g).get(&o).map(String::as_str),
            Some("opls_t0")
        );
    }

    /// The same two levels without the override: `opls_t0` does not dominate
    /// `opls_t1`, so the level-1 winner replaces the level-0 assignment.
    #[test]
    fn later_level_replaces_a_non_dominating_type() {
        let meta = meta_with(&[
            ("opls_t0", ranked("[#8]", &[], None, 0)),
            ("opls_t1", ranked("[#8;%opls_t0]", &[], None, 0)),
        ]);
        let engine = LayeredTypingEngine::build(&meta).expect("engine builds");
        assert_eq!(engine.analyzer().level("opls_t1"), Some(1));
        let (g, o, _ho) = ethanol();
        assert_eq!(
            engine.assign(&g).get(&o).map(String::as_str),
            Some("opls_t1")
        );
    }

    /// `opls_a` → `opls_b` → `opls_c` → `opls_a` is an overrides cycle: build
    /// is `Err`, naming every member.
    #[test]
    fn override_cycle_is_a_build_error_naming_its_members() {
        let meta = meta_with(&[
            ("opls_a", ranked("[#8]", &["opls_b"], None, 0)),
            ("opls_b", ranked("[#8]", &["opls_c"], None, 0)),
            ("opls_c", ranked("[#8]", &["opls_a"], None, 0)),
        ]);
        let Err(e) = LayeredTypingEngine::build(&meta) else {
            panic!("an overrides cycle must be a build error");
        };
        for member in ["opls_a", "opls_b", "opls_c"] {
            assert!(e.contains(member), "err names cycle member {member}: {e}");
        }
    }

    /// `opls_a` overrides `opls_missing`, which is absent from the metadata:
    /// build is `Err`, naming both.
    #[test]
    fn dangling_override_is_a_build_error_naming_both_types() {
        let meta = meta_with(&[("opls_a", ranked("[#8]", &["opls_missing"], None, 0))]);
        let Err(e) = LayeredTypingEngine::build(&meta) else {
            panic!("an override naming an absent type must be a build error");
        };
        assert!(e.contains("opls_a"), "err names the overriding type: {e}");
        assert!(e.contains("opls_missing"), "err names the absent type: {e}");
    }
}
