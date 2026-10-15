//! What the cascade searches: one parameter table, flattened into candidates.
//!
//! A [`Candidate`] is one row (or one whole multi-cosine group) of a bonded
//! table: the type name it is known by, its endpoint **patterns** (atom-type
//! names, class names, or the `X` wildcard), and the params to copy on a match.
//! That is all the cascade ever needs, which is what lets one estimator serve a
//! [`ForceField`] and AMBER's `parm` tables alike — the caller flattens its own
//! table and the cascade stays force-field agnostic.
//!
//! # Units and key names are the table's, not ours
//!
//! A candidate's [`params`](Candidate::params) are copied **verbatim** on a
//! match, so they arrive in whatever convention the table that produced them
//! uses, under whatever key names its consumer reads. The estimator neither
//! converts nor inspects them (with one exception: the empirical angle formula
//! reads a neighbour angle's `theta0`, which is why angles must reach the
//! estimator in molrs's radians — see [`empirical`](super::empirical)).

use crate::ff::forcefield::{ForceField, StyleDefs};
use crate::ff::ir::Params;

/// One row of a bonded table, as the cascade sees it.
#[derive(Debug, Clone)]
pub struct Candidate {
    /// The force-field type name (`c3-oh`), or a row's dash form with `X` in its
    /// wildcard slots (`X-X-ca-ha`). Reported as the estimate's `estimate_analog`.
    pub name: String,
    /// The endpoint patterns: an atom-type name, a class name, or a wildcard
    /// (`""` / `"*"` / `"X"`). Its length is the term's arity.
    pub pattern: Vec<String>,
    /// The params to copy verbatim on a match.
    pub params: Params,
}

impl Candidate {
    /// Build a candidate from an endpoint pattern and the params to copy.
    pub fn new(
        name: impl Into<String>,
        pattern: impl IntoIterator<Item = impl Into<String>>,
        params: Params,
    ) -> Self {
        Self {
            name: name.into(),
            pattern: pattern.into_iter().map(Into::into).collect(),
            params,
        }
    }

    /// How many of the endpoint slots are wildcards.
    ///
    /// The cascade ranks a **specific** row above a wildcard one: a row written
    /// for these very atom types is a better analog than a generic one that
    /// happens to cover them.
    pub fn wildcards(&self) -> usize {
        self.pattern.iter().filter(|p| is_wildcard(p)).count()
    }
}

/// The candidate tables of one parameter set, by arity.
///
/// Impropers are a first-class arity here, not an afterthought: molrs's own
/// force-field categories are bonds / angles / dihedrals / **impropers**, and a
/// planar centre with no improper term is a silent hole in the potential, not a
/// missing nicety.
#[derive(Debug, Clone, Default)]
pub struct CandidateSet {
    /// Two-atom candidates.
    pub bonds: Vec<Candidate>,
    /// Three-atom candidates, vertex in the middle.
    pub angles: Vec<Candidate>,
    /// Four-atom candidates along the bonded chain. One candidate per quartet —
    /// a multi-cosine torsion is ONE candidate whose params hold the whole group.
    pub dihedrals: Vec<Candidate>,
    /// Four-atom candidates with the centre third, peripherals unordered.
    pub impropers: Vec<Candidate>,
}

impl CandidateSet {
    /// Flatten every bonded style of a [`ForceField`] into candidates.
    ///
    /// **Style-agnostic**: the arity comes from the style's
    /// [`StyleDefs`] variant, never from its *name*. Asking for
    /// `("dihedral", "opls")` by name would be blind to GAFF, whose dihedral
    /// style is `periodic`: a fully populated force field would show the
    /// estimator an empty table.
    pub fn from_forcefield(ff: &ForceField) -> Self {
        let mut out = Self::default();
        for style in ff.styles() {
            match style.defs() {
                StyleDefs::Bond(types) => out.bonds.extend(
                    types
                        .iter()
                        .map(|t| Candidate::new(&t.name, [&t.itom, &t.jtom], t.params.clone())),
                ),
                StyleDefs::Angle(types) => out.angles.extend(types.iter().map(|t| {
                    Candidate::new(&t.name, [&t.itom, &t.jtom, &t.ktom], t.params.clone())
                })),
                StyleDefs::Dihedral(types) => out.dihedrals.extend(types.iter().map(|t| {
                    Candidate::new(
                        &t.name,
                        [&t.itom, &t.jtom, &t.ktom, &t.ltom],
                        t.params.clone(),
                    )
                })),
                StyleDefs::Improper(types) => out.impropers.extend(types.iter().map(|t| {
                    Candidate::new(
                        &t.name,
                        [&t.itom, &t.jtom, &t.ktom, &t.ltom],
                        t.params.clone(),
                    )
                })),
                StyleDefs::Atom(_)
                | StyleDefs::Pair(_)
                | StyleDefs::Cmap(_)
                | StyleDefs::Relation { .. } => {}
            }
        }
        out
    }
}

/// Whether an endpoint pattern is a wildcard.
///
/// Three spellings, because three vocabularies meet here: the OPLS XML reader
/// transcribes an absent class as the empty string, molpy normalises it to `*`,
/// and AMBER's `parm` files write `X`.
pub fn is_wildcard(pattern: &str) -> bool {
    pattern.is_empty() || pattern == "*" || pattern == "X"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_spelling_of_a_wildcard_is_one() {
        assert!(is_wildcard(""));
        assert!(is_wildcard("*"));
        assert!(is_wildcard("X"));
        assert!(!is_wildcard("c3"));
    }

    #[test]
    fn extraction_reads_the_style_kind_not_its_name() {
        // A GAFF-shaped force field: dihedral style `periodic`, not `opls`. A
        // by-name extractor would see nothing here.
        let mut ff = ForceField::new("gaff-shaped");
        ff.def_style("bond", "harmonic", Params::new())
            .unwrap()
            .def_type(
                "c3-hc",
                &["c3", "hc"],
                Params::from_iter([("k", 330.6), ("r0", 1.0969)]),
            )
            .unwrap();
        ff.def_style("dihedral", "periodic", Params::new())
            .unwrap()
            .def_type(
                "X-c3-c3-X",
                &["X", "c3", "c3", "X"],
                Params::from_iter([("k1", 0.16), ("periodicity1", 3.0), ("phase1", 0.0)]),
            )
            .unwrap();
        ff.def_style("improper", "periodic", Params::new())
            .unwrap()
            .def_type(
                "X-X-ca-ha",
                &["X", "X", "ca", "ha"],
                Params::from_iter([("k", 1.1), ("periodicity", 2.0), ("phase", 180.0)]),
            )
            .unwrap();

        let set = CandidateSet::from_forcefield(&ff);
        assert_eq!(set.bonds.len(), 1);
        assert_eq!(set.dihedrals.len(), 1, "a `periodic` dihedral style counts");
        assert_eq!(set.impropers.len(), 1, "impropers are a first-class arity");
        assert_eq!(set.angles.len(), 0);
        assert_eq!(set.dihedrals[0].wildcards(), 2);
        assert_eq!(set.impropers[0].pattern, ["X", "X", "ca", "ha"]);
    }
}
