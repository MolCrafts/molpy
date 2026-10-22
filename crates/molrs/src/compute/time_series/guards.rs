//! Input guards shared by the batch and streaming correlator (and the
//! wrapped-trajectory check of [`VanHove`](crate::compute::VanHove)).

use molrs::core::SimBox;
use molrs::op::F;
use ndarray::ArrayView2;

use crate::compute::ComputeError;

/// [`ComputeError::NonFinite`] at the first NaN / ±∞.
pub(super) fn require_finite<'a>(
    values: impl IntoIterator<Item = &'a F>,
    where_: &'static str,
) -> Result<(), ComputeError> {
    match values.into_iter().position(|v| !v.is_finite()) {
        Some(index) => Err(ComputeError::NonFinite { where_, index }),
        None => Ok(()),
    }
}

/// A presence series (the input `field`) holds only `0` and `1`.
pub(super) fn require_presence(
    values: impl IntoIterator<Item = F>,
    field: &'static str,
) -> Result<(), ComputeError> {
    match values
        .into_iter()
        .enumerate()
        .find(|(_, v)| *v != 0.0 && *v != 1.0)
    {
        Some((index, value)) => Err(ComputeError::OutOfRange {
            field,
            value: format!("presence value {value} at index {index} is neither 0 nor 1"),
        }),
        None => Ok(()),
    }
}

/// The first item whose step from `prev` to `now` is longer than half the
/// shortest perpendicular height of `simbox` along its periodic axes; `None`
/// if every step is shorter or the box has no periodic axis.
pub(crate) fn first_wrapped_item(
    prev: ArrayView2<'_, F>,
    now: ArrayView2<'_, F>,
    simbox: &SimBox,
) -> Result<Option<usize>, ComputeError> {
    if now.ncols() != 3 {
        return Err(ComputeError::DimensionMismatch {
            expected: 3,
            got: now.ncols(),
            what: "components per item (the wrapped-trajectory check is 3-D)",
        });
    }
    let pbc = simbox.pbc();
    let heights = simbox.nearest_plane_distance();
    let shortest = (0..3)
        .filter(|&axis| pbc[axis])
        .map(|axis| heights[axis])
        .fold(F::INFINITY, F::min);
    if !shortest.is_finite() {
        return Ok(None);
    }
    let bound_sq = (0.5 * shortest) * (0.5 * shortest);
    Ok(prev.outer_iter().zip(now.outer_iter()).position(|(p, n)| {
        p.iter()
            .zip(n.iter())
            .map(|(a, b)| (b - a) * (b - a))
            .sum::<F>()
            > bound_sq
    }))
}
