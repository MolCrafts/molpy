//! Close contacts: the pairs of a neighbor table that the bond graph does not
//! already join within a few bonds.
//!
//! A structure built by placing fragments can put two atoms on top of each
//! other that no bond relates; a relaxation pushes most such pairs apart, but
//! counting them is how a builder knows how much it left to the relaxation.
//! Bonded neighbors (1-2), and usually the ends of a bond angle (1-3), sit
//! closer than any non-bonded contact by construction, so they are not
//! contacts.

use molrs::core::{Neighbors, Topology};
use molrs::op::F;
use ndarray::{Array1, Array2};

use crate::compute::require::require_dist_sq;
use crate::compute::{Compute, ComputeError, require_self_query};

/// The close contacts of one snapshot: every pair of a half-shell self-query
/// neighbor table — the table's cutoff is the contact distance — whose two
/// atoms are more than `exclude_bonds` bonds apart in the bond graph, or in
/// different components of it.
///
/// Configuration: `exclude_bonds`, the bond distance up to which a pair is
/// not a contact (0 excludes nothing, 1 bonded pairs, 2 also angle ends, 3
/// also torsion ends — the 1-2 / 1-3 / 1-4 exclusions of a force field). Data:
/// the table and the bonds, as atom index pairs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CloseContacts {
    exclude_bonds: usize,
}

impl CloseContacts {
    /// Pairs at most `exclude_bonds` bonds apart are not contacts.
    pub fn new(exclude_bonds: usize) -> Self {
        Self { exclude_bonds }
    }

    /// The bond distance up to which a pair is not a contact.
    pub fn exclude_bonds(&self) -> usize {
        self.exclude_bonds
    }
}

/// The data a [`CloseContacts`] analyses: one snapshot.
#[derive(Clone, Copy)]
pub struct CloseContactsInput<'a> {
    /// Half-shell self-query table with a `dist_sq` column.
    pub pairs: &'a Neighbors,
    /// Bonds as `(i, j)` atom index pairs.
    pub bonds: &'a [[usize; 2]],
}

/// The close contacts, sorted by `(i, j)`.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct CloseContactsResult {
    /// The contacts, one `(i, j)` row each, `i < j`.
    pub pairs: Array2<usize>,
    /// Distance of each contact (Å, minimum image as the table measured it).
    pub distances: Array1<F>,
}

impl CloseContactsResult {
    /// Number of contacts.
    pub fn n_contacts(&self) -> usize {
        self.distances.len()
    }
}

impl Compute for CloseContacts {
    type Input<'a> = CloseContactsInput<'a>;
    type Output = CloseContactsResult;

    /// # Errors
    /// [`ComputeError::BadShape`] for a cross-query table or one without
    /// `dist_sq`; [`ComputeError::OutOfRange`] for a bond index past the
    /// table's point count.
    fn compute(&self, input: CloseContactsInput<'_>) -> Result<CloseContactsResult, ComputeError> {
        let n = require_self_query(input.pairs)?;
        require_dist_sq(input.pairs)?;
        let dist_sq = input.pairs.dist_sq().expect("checked by require_dist_sq");
        for &[i, j] in input.bonds {
            if i >= n || j >= n {
                return Err(ComputeError::OutOfRange {
                    field: "CloseContacts bond index",
                    value: format!("({i}, {j}) with {n} atoms"),
                });
            }
        }
        let mut graph = Topology::with_atoms(n);
        graph.add_bonds(input.bonds);
        let excluded = graph.close_pairs(self.exclude_bonds);

        let mut found: Vec<(usize, usize, F)> = input
            .pairs
            .query_point_indices()
            .iter()
            .zip(input.pairs.point_indices())
            .zip(dist_sq)
            .map(|((&a, &b), &d2)| {
                let (a, b) = (a as usize, b as usize);
                (a.min(b), a.max(b), d2.sqrt())
            })
            .filter(|&(i, j, _)| !excluded.contains_key(&(i, j)))
            .collect();
        found.sort_by_key(|&(i, j, _)| (i, j));
        let pairs = Array2::from_shape_fn((found.len(), 2), |(row, col)| {
            if col == 0 { found[row].0 } else { found[row].1 }
        });
        let distances = found.iter().map(|&(_, _, d)| d).collect();
        Ok(CloseContactsResult { pairs, distances })
    }
}

crate::compute::spec::compute_spec! {
    CloseContacts as "CloseContacts";
    params |_s| {
        exclude_bonds: Int;
    }
    inputs {
        pairs: Neighbors;
        bonds: IndexPairs;
    }
}

#[cfg(test)]
mod tests;
