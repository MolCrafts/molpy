//! Matching the two mini-languages an ATD rule is written in.
//!
//! A rule constrains an atom in two ways beyond its bare counts:
//!
//! * a **property expression** — `[RG5.RG6,AR1.AR2.AR3]`, an AND of ORs over
//!   [`AtomProp`] counts, optionally suffixed `'` / `''` to also constrain the
//!   bond back to the atom the match arrived from;
//! * an **environment** — `C3[AR1](O1,O1)`, a forest of neighbour patterns
//!   walked outward from the candidate, plus `a:b:TYPE` bonds that close a ring
//!   between two `<label>`ed atoms.
//!
//! Both arrive here **pre-parsed** as `&'static` data from
//! [`crate::ff::params`]: this module walks a tree, it never reads a table.
//!
//! The environment walk is a backtracking match, not a greedy one. Sibling
//! patterns compete for the same neighbours, so committing to the first
//! neighbour that satisfies pattern *k* can starve pattern *k+1* of the only
//! atom that could have satisfied it — `used` and the cloned label map exist to
//! let the walk take that choice back.

use std::collections::HashMap;

use molrs::core::NodeId;

use super::facts::MolFacts;
use crate::ff::params::{
    AtomPattern, AtomProp, EnvBond, EnvBondType, PatternAtom, PropExpr, PropRelation, PropUnit,
};

/// The `<label>` → atom bindings of one environment match.
type Labels = HashMap<&'static str, NodeId>;

/// The rest of an environment match, run on the labels bound so far.
type Rest<'a> = dyn FnMut(&mut Labels) -> bool + 'a;

/// One pattern list and where it hangs: the patterns must match distinct
/// neighbours of `parent`, none of them `prev`.
struct Siblings {
    parent: NodeId,
    prev: Option<NodeId>,
    patterns: &'static [AtomPattern],
}

impl MolFacts {
    /// Walk a pre-parsed `[...]` expression: an AND of ORs.
    ///
    /// `prev` is the atom the pattern arrived from, and is `None` for a rule's
    /// own top-level property — where a `'` suffix has no bond to speak of.
    pub(super) fn atom_property_matches(
        &self,
        aid: NodeId,
        prev: Option<NodeId>,
        expr: &PropExpr,
    ) -> bool {
        expr.constraints.iter().all(|constraint| {
            constraint
                .units
                .iter()
                .any(|unit| self.atom_property_unit_matches(aid, prev, unit))
        })
    }

    /// One `[count]PROP['|'']` unit.
    fn atom_property_unit_matches(
        &self,
        aid: NodeId,
        prev: Option<NodeId>,
        unit: &PropUnit,
    ) -> bool {
        let Ok(i) = self.index_of(aid) else {
            return false;
        };
        let count = self.props[i].count(unit.prop);
        // `apcheck` reads a bare `PROP` as count -1 and accepts
        // `count == -1 || value > 0`, so a bare `AR2` also holds where the
        // AM1-BCC indole rule left the count at exactly -1 (acenaphthylene's
        // five-ring carbons are BCC type 16 by it).
        let count_matches = unit.count.map_or(count > 0 || count == -1, |n| {
            i32::try_from(n).is_ok_and(|n| count == n)
        });
        if !count_matches {
            return false;
        }
        match unit.relation {
            Some(PropRelation::BondedToPrev) => {
                prev.is_some_and(|prev| self.bond_to_prev_matches(aid, prev, unit.prop))
            }
            Some(PropRelation::NotBondedToPrev) => {
                prev.is_some_and(|prev| !self.bond_to_prev_matches(aid, prev, unit.prop))
            }
            None => true,
        }
    }

    /// Is the bond from `aid` back to `prev` itself of type `prop`?
    fn bond_to_prev_matches(&self, aid: NodeId, prev: NodeId, prop: AtomProp) -> bool {
        let Ok(i) = self.index_of(aid) else {
            return false;
        };
        self.neighbors[i]
            .iter()
            .find(|(nb, _, _)| *nb == prev)
            .is_some_and(|(_, bond_type, _)| bond_type_matches_property(*bond_type, prop))
    }

    /// Does the neighbourhood of `aid` satisfy `patterns` (and any `env_bonds`)?
    ///
    /// The `env_bonds` take part in the search: an assignment whose labelled
    /// atoms are not bonded as required is abandoned for the next one, as
    /// `atomtype.c`'s `dccheck` tries every combination of matched chains.
    /// (Carbazole's N is BCC type 23 by `XX<a1>, XB(XB(XX<a2>)) a1:a2:any`
    /// through its second carbon path, not its first.)
    pub(super) fn environment_matches(
        &self,
        aid: NodeId,
        patterns: &'static [AtomPattern],
        env_bonds: Option<&'static [EnvBond]>,
    ) -> bool {
        let mut labels = HashMap::new();
        self.match_pattern_list(aid, None, patterns, &mut labels, &mut |labels| {
            env_bonds.is_none_or(|bonds| self.environment_bonds_match(bonds, labels))
        })
    }

    /// The `a:b:TYPE` constraints between `<label>`ed environment atoms.
    fn environment_bonds_match(&self, bonds: &[EnvBond], labels: &Labels) -> bool {
        for constraint in bonds {
            let Some(&a) = labels.get(constraint.a) else {
                return false;
            };
            let Some(&b) = labels.get(constraint.b) else {
                return false;
            };
            let Ok(i) = self.index_of(a) else {
                return false;
            };
            let Some((_, bond_type, _)) = self.neighbors[i].iter().find(|(nb, _, _)| *nb == b)
            else {
                return false;
            };
            if !environment_bond_type_matches(*bond_type, constraint.bond) {
                return false;
            }
        }
        true
    }

    /// Match every pattern of `patterns` against a distinct neighbour of
    /// `parent`, then hand the labels to `rest` — the remainder of the whole
    /// match. `true` once `rest` accepts some assignment.
    fn match_pattern_list(
        &self,
        parent: NodeId,
        prev: Option<NodeId>,
        patterns: &'static [AtomPattern],
        labels: &mut Labels,
        rest: &mut Rest<'_>,
    ) -> bool {
        let siblings = Siblings {
            parent,
            prev,
            patterns,
        };
        self.match_siblings(&siblings, 0, labels, &mut Vec::new(), rest)
    }

    /// Backtracking assignment of `siblings.patterns[pos..]` to unused
    /// neighbours of `siblings.parent`.
    fn match_siblings(
        &self,
        siblings: &Siblings,
        pos: usize,
        labels: &mut Labels,
        used: &mut Vec<NodeId>,
        rest: &mut Rest<'_>,
    ) -> bool {
        if pos == siblings.patterns.len() {
            return rest(labels);
        }
        let Ok(parent_i) = self.index_of(siblings.parent) else {
            return false;
        };
        for (candidate, _, _) in &self.neighbors[parent_i] {
            if Some(*candidate) == siblings.prev || used.contains(candidate) {
                continue;
            }
            let mut labels_next = labels.clone();
            used.push(*candidate);
            let matched = self.atom_pattern_matches(
                *candidate,
                siblings.parent,
                &siblings.patterns[pos],
                &mut labels_next,
                &mut |labels_after| {
                    self.match_siblings(siblings, pos + 1, labels_after, used, rest)
                },
            );
            if matched {
                *labels = labels_next;
                return true;
            }
            used.pop();
        }
        false
    }

    /// One node of the environment forest, plus its children, then `rest`.
    fn atom_pattern_matches(
        &self,
        aid: NodeId,
        prev: NodeId,
        pattern: &'static AtomPattern,
        labels: &mut Labels,
        rest: &mut Rest<'_>,
    ) -> bool {
        let Ok(i) = self.index_of(aid) else {
            return false;
        };
        if !self.pattern_atom_matches(i, pattern.atom) {
            return false;
        }
        if pattern
            .degree
            .is_some_and(|degree| degree != self.degree[i])
        {
            return false;
        }
        if let Some(prop) = pattern.property
            && !self.atom_property_matches(aid, Some(prev), &prop)
        {
            return false;
        }
        if let Some(label) = pattern.label
            && labels.insert(label, aid).is_some_and(|old| old != aid)
        {
            return false;
        }
        self.match_pattern_list(aid, Some(prev), pattern.children, labels, rest)
    }

    /// Match a pattern atom whose name the generator already resolved.
    ///
    /// `EW` / `WILDATOM` / element resolution happened at table-generation time,
    /// so there is no name table to consult here — and no per-table special case
    /// either: `XB` means one thing in `ATOMTYPE_BCC.DEF` and another in
    /// `ATOMTYPE_GFF.DEF`, but both arrive as the same expanded [`PatternAtom`].
    fn pattern_atom_matches(&self, atom_index: usize, atom: PatternAtom) -> bool {
        match atom {
            PatternAtom::ElectronWithdrawing => self.ewd[atom_index],
            PatternAtom::Wild(specs) => specs.iter().any(|spec| {
                spec.z == self.atomic_number[atom_index]
                    && spec
                        .degree
                        .is_none_or(|degree| degree == self.degree[atom_index])
            }),
            PatternAtom::Element(z) => z == self.atomic_number[atom_index],
        }
    }
}

/// Does an antechamber bond type satisfy a bond-valued [`AtomProp`]?
///
/// Ring and aromaticity properties (`RG*`, `AR*`, `NR`) are atom-valued, not
/// bond-valued, so they never constrain a bond.
fn bond_type_matches_property(bond_type: i32, prop: AtomProp) -> bool {
    match prop {
        AtomProp::SbStrict => matches!(bond_type, 1 | 9),
        AtomProp::SbAny => matches!(bond_type, 1 | 7 | 9 | 10),
        AtomProp::DbStrict => matches!(bond_type, 2 | 9),
        AtomProp::DbAny => matches!(bond_type, 2 | 8 | 9 | 10),
        AtomProp::TbStrict | AtomProp::TbAny => bond_type == 3,
        AtomProp::Dl => bond_type == 9,
        AtomProp::Ab => matches!(bond_type, 7 | 8 | 10),
        AtomProp::Rg
        | AtomProp::Rg3
        | AtomProp::Rg4
        | AtomProp::Rg5
        | AtomProp::Rg6
        | AtomProp::Rg7
        | AtomProp::Rg8
        | AtomProp::Rg9
        | AtomProp::Rg10
        | AtomProp::Nr
        | AtomProp::Ar1
        | AtomProp::Ar2
        | AtomProp::Ar3
        | AtomProp::Ar4
        | AtomProp::Ar5 => false,
    }
}

/// Does an antechamber bond type satisfy an `a:b:TYPE` environment-bond constraint?
fn environment_bond_type_matches(bond_type: i32, bond: EnvBondType) -> bool {
    match bond {
        EnvBondType::Any => true,
        EnvBondType::Single => bond_type_matches_property(bond_type, AtomProp::SbAny),
        EnvBondType::Double => bond_type_matches_property(bond_type, AtomProp::DbAny),
        EnvBondType::Triple => bond_type_matches_property(bond_type, AtomProp::TbAny),
        EnvBondType::Aromatic => bond_type_matches_property(bond_type, AtomProp::Ab),
    }
}
