//! Input requirements for [`Neighbors`] tables — the one place a compute
//! kernel turns "this table cannot answer the question I am asking" into a
//! [`ComputeError::BadShape`]. Two kinds of requirement live here: a *column*
//! the table never stored, and a *query mode* whose pair ordering the kernel
//! cannot consume.
//!
//! A [`Neighbors`] table is the list of particle pairs within a cutoff
//! distance, one row per pair. Besides the two particle indices it may carry
//! two *physical* columns, both taken under the **minimum-image convention**
//! (with periodic boundaries a pair has one separation per periodic copy of the
//! box; only the shortest is ever reported): `dist_sq`, the squared pair
//! distance `|r_j − r_i|²` in Å², and `disp`, the displacement `r_j − r_i`
//! itself in Å, pointing from `i` to `j` and left unnormalized. Which of them
//! a table holds is decided when it is materialized, by
//! [`NeighborColumns`](molrs::core::NeighborColumns).
//!
//! A table reports a column it never stored as `None`, never as a fabricated
//! zero: `disp()` on an indices-only table is `None`, not a view full of
//! `(0, 0, 0)`, because a zero displacement means two coincident particles and
//! would answer confidently and wrongly. Every kernel that needs a column
//! therefore has to reject `None` itself, and each rejection carries the same
//! two facts — which column was missing, and how many pairs went unanswered
//! because of it.
//!
//! The mode requirement is the other half. A table also records *how* it was
//! searched, in its [`QueryMode`]: a **self-query** is half-shell, holding each
//! unordered pair exactly once as `(i, j)` with `i < j`, while a **cross-query**
//! is directed over two point sets and holds a row per (query point, reference
//! point). A kernel that visits each row once and updates *both* of its
//! endpoints — exploiting that the same bond seen from `j` is just the reversed
//! vector — is counting on the half-shell guarantee, and quietly double-counts
//! without it. That is a wrong number rather than a missing one, so it too is
//! refused rather than computed.
//!
//! The table helpers are free functions rather than methods on [`Neighbors`] /
//! [`PairSource`]:
//! the owning type lives in `core`, one layer below this one, and must not learn
//! about [`ComputeError`] to answer a question `core` never asks.
//!
//! The orientation guards are the other input every orientation kernel
//! validates the same way: an orientation is a rotation (a quaternion), a
//! direction (a director) or a planar angle, and a zero quaternion or a zero
//! director names none of them. Each is an error here, once, instead of the
//! identity, a skipped particle or a zero angle at each kernel.

use molrs::core::{Neighbors, PairSource, QueryMode};
use molrs::op::{F, Fnx3View, Quat, Vec3};
use ndarray::{ArrayView1, ArrayView2};

use super::error::ComputeError;

/// The minimum-image displacement column (Å), or [`ComputeError::BadShape`].
///
/// On success the view has exactly [`n_pairs()`](Neighbors::n_pairs) rows, so a
/// caller may index rows `0..n_pairs` without further checks — which is the
/// point: reading an absent column would otherwise index an empty buffer (an
/// out-of-bounds panic natively, a `unreachable` trap under WASM). Row `k` is
/// `(dx, dy, dz)` for pair `k` in Å, pointing from `i` to `j` and
/// **unnormalized**: its length is the pair distance.
///
/// # Errors
///
/// [`ComputeError::BadShape`] when the table was materialized without
/// [`NeighborColumns::disp`](molrs::core::NeighborColumns::disp)
/// — that is, with any policy other than `DISP` or `FULL`. The `expected` text
/// names the missing column and the pair count it was needed for, and `got`
/// describes the table that arrived; nothing is substituted for the column.
pub(crate) fn require_disp(nlist: &Neighbors) -> Result<Fnx3View<'_>, ComputeError> {
    nlist
        .disp()
        .ok_or_else(|| missing_column("disp", nlist.n_pairs()))
}

/// Require that a pair source reports the squared-distance column (Å²), or
/// [`ComputeError::BadShape`].
///
/// On success every pair [`visit_pairs`](PairSource::visit_pairs) reports
/// carries `Some(dist_sq)`: the squared minimum-image distance in Å², the
/// square root left to the caller (most consumers compare against a squared
/// cutoff, and a binning kernel takes it once per pair). A built
/// [`NeighborList`](molrs::core::NeighborList) always qualifies; a
/// materialized table does when it kept the column.
///
/// # Errors
///
/// [`ComputeError::BadShape`] when the source was materialized without
/// [`NeighborColumns::dist_sq`](molrs::core::NeighborColumns::dist_sq)
/// — that is, with any policy other than `DIST_SQ` or `FULL`. The `expected`
/// text names the missing column and the pair count it was needed for, and
/// `got` describes the table that arrived.
pub(crate) fn require_dist_sq(source: &dyn PairSource) -> Result<(), ComputeError> {
    if source.columns().dist_sq {
        return Ok(());
    }
    let mut n_pairs = 0usize;
    source.visit_pairs(&mut |_, _, _, _| n_pairs += 1);
    Err(missing_column("dist_sq", n_pairs))
}

/// Assert the table is a half-shell [`QueryMode::SelfQuery`], for kernels that
/// update *both* endpoints of every row they visit.
///
/// Such a kernel reads a row once and credits the bond to `i` and to `j`,
/// reversing the direction for the `j` side. That is exact on a self-query,
/// where each unordered pair is present exactly once, and wrong on a
/// cross-query, where the reversed row is present too: every neighbor is then
/// counted twice, and any per-particle count derived from the same traversal
/// (`N_i`, a solid-bond tally, a Lechner-Dellago `1 + N_i` denominator) is off
/// by that factor. Nothing about the rows themselves reveals this, so the mode
/// is the only thing that can be checked.
///
/// Kernels that *adapt* to the mode instead of refusing it — [`BondOrientationalOrder`] and
/// [`LocalDensity`] add the symmetric contribution only when the table is
/// half-shell — match on [`Neighbors::mode`] directly and do not call this.
///
/// [`BondOrientationalOrder`]: crate::compute::BondOrientationalOrder
/// [`LocalDensity`]: crate::compute::LocalDensity
///
/// # Errors
///
/// [`ComputeError::BadShape`] when the table is a
/// [`QueryMode::CrossQuery`]. The `expected` text names the mode the kernel
/// needs and the pair count it was reading, and `got` says why a cross table is
/// not a substitute.
///
/// On success, the size of the one point set both indices address — the
/// particle count of the snapshot, read off the table rather than off a frame.
pub(crate) fn require_self_query(nlist: &Neighbors) -> Result<usize, ComputeError> {
    match nlist.mode() {
        QueryMode::SelfQuery { n_points } => Ok(n_points),
        QueryMode::CrossQuery { .. } => Err(ComputeError::BadShape {
            expected: format!(
                "SelfQuery half-shell Neighbors for {} pairs — this kernel updates both \
                 endpoints of every row it visits, which needs each unordered pair exactly once",
                nlist.n_pairs()
            ),
            got: "CrossQuery table: directed over two point sets, so the reversed row is \
                  present too and every neighbor would be counted twice"
                .to_string(),
        }),
    }
}

/// Whether a table is half-shell: a [`QueryMode::SelfQuery`] holds each
/// unordered pair once, so a kernel that adapts to the mode (rather than
/// refusing a cross table) credits the reversed bond itself.
pub(crate) fn is_half_shell(mode: QueryMode) -> bool {
    matches!(mode, QueryMode::SelfQuery { .. })
}

/// Unit quaternions `(w, x, y, z)` from an `N × 4` array, each divided by its
/// norm.
///
/// # Errors
/// [`ComputeError::DimensionMismatch`] when the array does not have 4
/// columns; [`ComputeError::NonFinite`] at the first non-finite component
/// (flat index); [`ComputeError::OutOfRange`] for a row of zero norm — a zero
/// quaternion is no rotation, and reading it as the identity would be a
/// silent answer.
pub(crate) fn require_quaternions(
    q: ArrayView2<'_, F>,
    field: &'static str,
) -> Result<Vec<Quat>, ComputeError> {
    unit_rows::<4>(q, field)
}

/// Unit directions from an `N × 3` array, each divided by its length.
///
/// # Errors
/// [`ComputeError::DimensionMismatch`] when the array does not have 3
/// columns; [`ComputeError::NonFinite`] at the first non-finite component
/// (flat index); [`ComputeError::OutOfRange`] for a row of zero length — a
/// zero vector has no direction.
pub(crate) fn require_directions(
    v: ArrayView2<'_, F>,
    field: &'static str,
) -> Result<Vec<Vec3>, ComputeError> {
    unit_rows::<3>(v, field)
}

/// Planar orientation angles (radians), one per particle.
///
/// # Errors
/// [`ComputeError::NonFinite`] at the first non-finite angle.
pub(crate) fn require_angles(
    angles: ArrayView1<'_, F>,
    field: &'static str,
) -> Result<Vec<F>, ComputeError> {
    if let Some(index) = angles.iter().position(|a| !a.is_finite()) {
        return Err(ComputeError::NonFinite {
            where_: field,
            index,
        });
    }
    Ok(angles.to_vec())
}

/// The rows of an `N × D` array, each normalised; the shared body of
/// [`require_quaternions`] and [`require_directions`].
fn unit_rows<const D: usize>(
    rows: ArrayView2<'_, F>,
    field: &'static str,
) -> Result<Vec<[F; D]>, ComputeError> {
    if rows.ncols() != D {
        return Err(ComputeError::DimensionMismatch {
            expected: D,
            got: rows.ncols(),
            what: field,
        });
    }
    rows.outer_iter()
        .enumerate()
        .map(|(item, row)| {
            let mut unit = [0.0; D];
            for (c, &v) in row.iter().enumerate() {
                if !v.is_finite() {
                    return Err(ComputeError::NonFinite {
                        where_: field,
                        index: item * D + c,
                    });
                }
                unit[c] = v;
            }
            let norm = unit.iter().map(|v| v * v).sum::<F>().sqrt();
            if norm == 0.0 {
                return Err(ComputeError::OutOfRange {
                    field,
                    value: format!("row {item} has zero norm; an orientation needs one"),
                });
            }
            Ok(unit.map(|v| v / norm))
        })
        .collect()
}

/// The shared error text: `expected` names the column *and* the pair count it
/// was needed for (a table with zero pairs is a different, harmless story),
/// `got` describes the table that arrived.
fn missing_column(column: &'static str, n_pairs: usize) -> ComputeError {
    ComputeError::BadShape {
        expected: format!("Neighbors with the {column} column for {n_pairs} pairs"),
        got: "indices-only / lean neighbor table".to_string(),
    }
}
