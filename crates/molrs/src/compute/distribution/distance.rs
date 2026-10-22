//! Distance observable: minimum-image separation of selected atom pairs.

use molrs::op::F;

use crate::compute::ComputeError;

use super::observable::{Observable, check_tuples, displacement};
use crate::op::vec3::norm;
use molrs::core::Mic;
use ndarray::ArrayView2;

/// Distance between the two atoms of each pair (arity 2), minimum-image under PBC.
///
/// The pairwise distance distribution generalizes the RDF to arbitrary
/// user-selected pairs (reference implementation distance DF, `src/tddf.cpp` /
/// `CTimeDiff`/`CDF` distance mode). The minimum-image convention is delegated
/// to the frame's [`Mic`] so a
/// distance DF and [`Rdf`](crate::compute::Rdf) return the same value
/// for the same pair under PBC.
#[derive(Debug, Clone, Default)]
pub struct DistanceObservable;

impl Observable for DistanceObservable {
    fn arity(&self) -> usize {
        2
    }

    fn sample_into(
        &self,
        positions: ArrayView2<'_, F>,
        mic: &Mic,
        tuples: ArrayView2<'_, usize>,
        out: &mut Vec<F>,
    ) -> Result<(), ComputeError> {
        check_tuples(self.arity(), positions, tuples)?;
        out.clear();
        out.reserve(tuples.nrows());
        for t in tuples.outer_iter() {
            let d = displacement(mic, positions, t[0], t[1]);
            out.push(norm(d));
        }
        Ok(())
    }
}
