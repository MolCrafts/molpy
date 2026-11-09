//! The junction stage: what a graph's interaction lists and atom types owe
//! the bonds a link made.
//!
//! A [`link`](crate::core::Atomistic::link) changes connectivity and nothing
//! else it can avoid: the leaving groups go, with every term that named one of
//! their atoms, and the anchors are bonded. What the new bonds imply is this
//! stage's concern, in one place for every caller — the
//! [`Assembler`](crate::builder::Assembler), a hand-made ring closure, a
//! crosslink:
//!
//! - **Atom types**, by the [`Retype`] policy. [`Retype::Keep`] changes none:
//!   the caller's templates were typed in the environment the product gives
//!   them, so their types stay right.
//! - **Graph-derived terms.** The angles, dihedrals and 1-2/1-3/1-4 pair rows
//!   of every kind the graph already carries are made to agree with the new
//!   bond graph. A link only adds bonds, so every existing angle and dihedral
//!   is still a path and stays, with its parameters; the terms through a new
//!   bond are added, bare.
//! - **Reclassified pairs.** Adding a bond can shorten a path: a ring closure
//!   makes the two ends of a former 1-4 pair bonded. Such a pair row gets its
//!   new bond distance and loses every cell but it — its overrides were for
//!   the old relation — so a later parameter assignment prices it afresh.
//! - **Impropers** are not graph-derived (which centres carry one is force
//!   field data); `link` itself continues an improper centred on an anchor
//!   onto the partner, bare.
//!
//! Charges are not this stage's concern: the [`Junction`]s it reports carry
//! the leaving-group charge each link folded onto its anchors, for a charge
//! step to read.

use std::collections::HashMap;
use std::fmt;

use crate::core::keys;
use crate::core::schema::block_names::PAIRS;
use crate::core::{Atomistic, Junction, MolRsError, RelationId};

/// How the atom types of a linked graph follow its new bonds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Retype {
    /// Change no atom type: the templates were typed in the chemical
    /// environment the product gives them (an in-chain residue, a monomer
    /// typed with its neighbours mimicked). The caller answers for that.
    #[default]
    Keep,
}

/// What [`update_junctions`] did.
#[derive(Debug, Clone, PartialEq)]
pub struct JunctionReport {
    /// The junctions it was given: the new bonds, their anchors and the
    /// charge each link folded onto each anchor.
    pub junctions: Vec<Junction>,
    /// Pair rows whose bond distance the new bonds shortened; each holds only
    /// its new `n_bonds` now.
    pub reclassified_pairs: Vec<RelationId>,
    /// Angles added through the new bonds.
    pub added_angles: usize,
    /// Dihedrals added through the new bonds.
    pub added_dihedrals: usize,
    /// Pair rows added through the new bonds.
    pub added_pairs: usize,
}

/// Why [`update_junctions`] refused its input.
#[derive(Debug)]
pub enum JunctionError {
    /// Junction `junction`'s bond is not a live bond of the graph joining its
    /// two anchors.
    NotABond {
        /// The junction's index in the slice given.
        junction: usize,
    },
    /// The graph refused an edit.
    Graph(MolRsError),
}

impl fmt::Display for JunctionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotABond { junction } => write!(
                f,
                "junction {junction}: its bond is not a live bond joining its two anchors"
            ),
            Self::Graph(error) => write!(f, "{error}"),
        }
    }
}

impl std::error::Error for JunctionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Graph(error) => Some(error),
            Self::NotABond { .. } => None,
        }
    }
}

impl From<MolRsError> for JunctionError {
    fn from(error: MolRsError) -> Self {
        Self::Graph(error)
    }
}

/// Bring `world`'s atom types and graph-derived terms up to the bonds of
/// `junctions`, by `retype`: atom types by the policy, then the
/// graph-derived terms and the reclassified pairs as the module docs say.
///
/// Only the term kinds `world` already carries are generated: a graph whose
/// templates had no angles gets none. Calling it again with the same
/// junctions changes nothing.
///
/// # Errors
///
/// [`JunctionError::NotABond`] when a junction's bond is not a live bond
/// between its anchors (checked before any write);
/// [`JunctionError::Graph`] when the graph refuses an edit.
pub fn update_junctions(
    world: &mut Atomistic,
    junctions: &[Junction],
    retype: &Retype,
) -> Result<JunctionReport, JunctionError> {
    for (index, junction) in junctions.iter().enumerate() {
        let joins = world.get_bond(junction.bond).is_ok_and(|bond| {
            let mut ends = bond.nodes.to_vec();
            let mut anchors = junction.anchors.to_vec();
            ends.sort();
            anchors.sort();
            ends == anchors
        });
        if !joins {
            return Err(JunctionError::NotABond { junction: index });
        }
    }
    match retype {
        Retype::Keep => {}
    }

    let with_pairs = world.n_terms(PAIRS)? > 0;
    let before: HashMap<RelationId, i64> = if with_pairs {
        world
            .term_ids(PAIRS)?
            .map(|id| {
                let n = world.get_term(PAIRS, id)?.props.get_int(keys::N_BONDS);
                Ok((id, n.map_or(-1, i64::from)))
            })
            .collect::<Result<_, MolRsError>>()?
    } else {
        HashMap::new()
    };
    let (added_angles, added_dihedrals, _, added_pairs) = world.generate_topology(
        world.n_angles() > 0,
        world.n_dihedrals() > 0,
        false,
        with_pairs,
        false,
    )?;

    let mut reclassified_pairs = Vec::new();
    for (&id, &n_before) in &before {
        let row = world.get_term(PAIRS, id)?;
        let n_after = row.props.get_int(keys::N_BONDS).map_or(-1, i64::from);
        if n_after != n_before {
            for key in row.props.keys().filter(|&key| key != keys::N_BONDS) {
                world.clear_term(PAIRS, id, key)?;
            }
            reclassified_pairs.push(id);
        }
    }
    reclassified_pairs.sort();

    Ok(JunctionReport {
        junctions: junctions.to_vec(),
        reclassified_pairs,
        added_angles,
        added_dihedrals,
        added_pairs,
    })
}

/// The graph types an [`Assembler`](crate::builder::Assembler) builds, and
/// what each owes its junctions: an [`Atomistic`] runs [`update_junctions`];
/// a [`MolGraph`](crate::core::MolGraph) or a
/// [`CoarseGrain`](crate::core::CoarseGrain) has no atom types or
/// graph-derived terms to bring up and only reports its junctions. Sealed.
pub trait Settle: crate::core::Graph + sealed::Sealed {
    /// Bring `self` up to `junctions` by `retype`.
    ///
    /// # Errors
    ///
    /// As [`update_junctions`].
    fn settle(
        &mut self,
        junctions: &[Junction],
        retype: &Retype,
    ) -> Result<JunctionReport, JunctionError>;
}

mod sealed {
    pub trait Sealed {}
    impl Sealed for crate::core::Atomistic {}
    impl Sealed for crate::core::MolGraph {}
    impl Sealed for crate::core::CoarseGrain {}
}

impl Settle for Atomistic {
    fn settle(
        &mut self,
        junctions: &[Junction],
        retype: &Retype,
    ) -> Result<JunctionReport, JunctionError> {
        update_junctions(self, junctions, retype)
    }
}

/// The report of a graph that has nothing to bring up.
fn reported(junctions: &[Junction]) -> JunctionReport {
    JunctionReport {
        junctions: junctions.to_vec(),
        reclassified_pairs: Vec::new(),
        added_angles: 0,
        added_dihedrals: 0,
        added_pairs: 0,
    }
}

impl Settle for crate::core::MolGraph {
    fn settle(
        &mut self,
        junctions: &[Junction],
        _: &Retype,
    ) -> Result<JunctionReport, JunctionError> {
        Ok(reported(junctions))
    }
}

impl Settle for crate::core::CoarseGrain {
    fn settle(
        &mut self,
        junctions: &[Junction],
        _: &Retype,
    ) -> Result<JunctionReport, JunctionError> {
        Ok(reported(junctions))
    }
}

#[cfg(test)]
mod tests;
