//! The `Observable` abstraction: positions + atom-index tuples → one scalar
//! sample per tuple.

use molrs::core::Mic;
use molrs::op::F;
use ndarray::ArrayView2;

use crate::compute::ComputeError;
use crate::op::vec3::sub;

/// A per-snapshot geometric extractor: each atom-index tuple → one scalar.
///
/// The tuples are an `n_tuples × arity` index array — 2 columns for a
/// distance, 3 for an angle (vertex in the middle), 4 for a dihedral — passed
/// as data with the positions; the observable holds no selection. `&self` is
/// an immutable parameter bag and identical inputs yield identical samples.
///
/// Mirrors the reference implementation's separation between *which* atoms an
/// analysis runs over (its observation lists, `src/tddf.cpp`,
/// `src/geodens.cpp`) and the geometric quantity extracted per tuple.
pub trait Observable {
    /// Atom indices consumed per sample (2 / 3 / 4).
    fn arity(&self) -> usize;

    /// Whether samples are angular (radians), enabling the sin θ ADF correction.
    fn is_angular(&self) -> bool {
        false
    }

    /// The closed `[min, max]` range of the sample, used to size the histogram
    /// when the caller does not override it (distance has no natural max, so it
    /// returns `None`).
    fn natural_range(&self) -> Option<(F, F)> {
        None
    }

    /// Extract one scalar per tuple into `out` (cleared first, its allocation
    /// reused across frames).
    ///
    /// `positions` is `N × 3`; `mic` is the frame's minimum-image convention
    /// ([`Mic::Free`] for open boundaries); `tuples` is `n_tuples ×`
    /// [`arity`](Self::arity).
    ///
    /// # Errors
    /// [`ComputeError::BadShape`] for a tuple array of the wrong arity or
    /// positions without 3 columns; [`ComputeError::OutOfRange`] for an atom
    /// index outside the positions; degenerate-geometry errors (e.g.
    /// zero-length vectors).
    fn sample_into(
        &self,
        positions: ArrayView2<'_, F>,
        mic: &Mic,
        tuples: ArrayView2<'_, usize>,
        out: &mut Vec<F>,
    ) -> Result<(), ComputeError>;
}

/// Check that `tuples` has `arity` columns and indexes rows of the `N × 3`
/// `positions` — the guard every observable runs before reading a position.
pub(super) fn check_tuples(
    arity: usize,
    positions: ArrayView2<'_, F>,
    tuples: ArrayView2<'_, usize>,
) -> Result<(), ComputeError> {
    if positions.ncols() != 3 {
        return Err(ComputeError::BadShape {
            expected: "positions N × 3".into(),
            got: format!("N × {}", positions.ncols()),
        });
    }
    if tuples.ncols() != arity {
        return Err(ComputeError::BadShape {
            expected: format!("atom-index tuples of arity {arity}"),
            got: format!("arity {}", tuples.ncols()),
        });
    }
    let n_atoms = positions.nrows();
    if let Some(&bad) = tuples.iter().find(|&&i| i >= n_atoms) {
        return Err(ComputeError::OutOfRange {
            field: "atom index",
            value: format!("{bad} (positions hold {n_atoms} atoms)"),
        });
    }
    Ok(())
}

/// Minimum-image displacement `r_b − r_a` under the frame's [`Mic`], resolved
/// once per frame by the caller. The one minimum-image use across the
/// observables, so a distance DF agrees with [`Rdf`](crate::compute::Rdf) on
/// the same pair (ac-003).
#[inline]
pub(super) fn displacement(mic: &Mic, positions: ArrayView2<'_, F>, a: usize, b: usize) -> [F; 3] {
    let from = [positions[[a, 0]], positions[[a, 1]], positions[[a, 2]]];
    let to = [positions[[b, 0]], positions[[b, 1]], positions[[b, 2]]];
    mic.apply(sub(to, from))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn tuples_are_checked_against_arity_and_positions() {
        let positions = array![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        let pairs = array![[0usize, 1]];
        assert!(check_tuples(2, positions.view(), pairs.view()).is_ok());
        assert!(matches!(
            check_tuples(3, positions.view(), pairs.view()),
            Err(ComputeError::BadShape { .. })
        ));
        // An index past the positions used to panic on the read; it is an
        // error.
        let out_of_range = array![[0usize, 2]];
        assert!(matches!(
            check_tuples(2, positions.view(), out_of_range.view()),
            Err(ComputeError::OutOfRange {
                field: "atom index",
                ..
            })
        ));
        let flat = array![[0.0, 0.0], [1.0, 0.0]];
        assert!(matches!(
            check_tuples(2, flat.view(), pairs.view()),
            Err(ComputeError::BadShape { .. })
        ));
    }
}
