//! Zero-copy access to the `atoms` block's `x`/`y`/`z` columns, shared by the
//! compute kernels: [`get_positions_ref`] returns borrowed views when the
//! underlying columns are contiguous (the common case for `Frame` and
//! `FrameView`), avoiding the per-atom copies of an owned `Vec`.
//!
//! Minimum image is not here: it is [`SimBox::mic`](molrs::core::SimBox::mic)
//! / [`Mic`](molrs::core::Mic), resolved once per frame by the
//! caller.

use molrs::core::FrameAccess;
use molrs::op::F;
use ndarray::Array3;

use super::error::ComputeError;

// ---------------------------------------------------------------------------
// Position access — borrow when contiguous, copy only if forced
// ---------------------------------------------------------------------------

/// Position storage: either a borrow from a contiguous column (zero copy) or
/// an owned `Vec` (required when the view is non-contiguous). Callers reach
/// the raw data via `AsRef<[F]>` / [`Positions::slice`].
#[derive(Debug)]
pub(crate) enum Positions<'a> {
    Borrowed(&'a [F]),
    Owned(Vec<F>),
}

impl<'a> Positions<'a> {
    #[inline]
    pub(crate) fn slice(&self) -> &[F] {
        match self {
            Positions::Borrowed(s) => s,
            Positions::Owned(v) => v.as_slice(),
        }
    }
}

fn column_to_positions<'a, FA: FrameAccess>(
    frame: &'a FA,
    col: &'static str,
) -> Result<Positions<'a>, ComputeError> {
    let view = frame
        .column("atoms", col)
        .and_then(|c| c.as_float())
        .ok_or(ComputeError::MissingColumn {
            block: "atoms",
            col,
        })?;
    match view.as_slice() {
        Some(s) => {
            // SAFETY: the slice refers to the same buffer backing the view,
            // which lives for 'a (tied to `&'a frame`). The `as_slice()`
            // return's own lifetime is a local reborrow; extending it is
            // sound because ArrayView in `ViewRepr<&'a A>` owns its data
            // pointer for 'a.
            let ptr = s.as_ptr();
            let len = s.len();
            let slice: &'a [F] = unsafe { std::slice::from_raw_parts(ptr, len) };
            Ok(Positions::Borrowed(slice))
        }
        None => Ok(Positions::Owned(view.iter().copied().collect())),
    }
}

/// Zero-copy position triplet when the frame exposes contiguous columns.
///
/// Returns a triple of [`Positions`]. Each can be dereferenced to `&[F]` via
/// [`slice`](Positions::slice). For `Frame` and `FrameView` the inner
/// variant is always [`Positions::Borrowed`], so the per-atom loop sees the
/// same cache-friendly layout as the stored block.
pub(crate) fn get_positions_ref<'a, FA: FrameAccess>(
    frame: &'a FA,
) -> Result<(Positions<'a>, Positions<'a>, Positions<'a>), ComputeError> {
    let xs = column_to_positions(frame, "x")?;
    let ys = column_to_positions(frame, "y")?;
    let zs = column_to_positions(frame, "z")?;
    Ok((xs, ys, zs))
}

/// Positions of a frame that may carry fewer than three spatial axes: an
/// absent `y` or `z` reads as zeros.
///
/// For quantities that are **dimension-agnostic** — a displacement, a
/// correlation — a one- or two-dimensional series is a legitimate input, and a
/// caller should not have to fabricate zero columns to be allowed to ask.
///
/// Deliberately *not* what [`get_positions_ref`] does, and it must not become
/// so. A hydrogen-bond geometry or a Steinhardt order parameter on a frame
/// missing `z` is a broken frame, not a two-dimensional system: there the
/// missing column has to stay an error, because zeros would answer confidently
/// and wrongly. `x` is always required — a frame with no positions at all is
/// not a low-dimensional frame.
pub(crate) fn get_positions_ref_any_dim<'a, FA: FrameAccess>(
    frame: &'a FA,
) -> Result<(Positions<'a>, Positions<'a>, Positions<'a>), ComputeError> {
    let xs = column_to_positions(frame, "x")?;
    let n = xs.slice().len();
    let axis = |col: &'static str| match column_to_positions(frame, col) {
        Ok(p) => Ok(p),
        Err(ComputeError::MissingColumn { .. }) => Ok(Positions::Owned(vec![0.0; n])),
        Err(e) => Err(e),
    };
    let ys = axis("y")?;
    let zs = axis("z")?;
    Ok((xs, ys, zs))
}

/// The `(n_frames, n_atoms, 3)` position series of a trajectory: the
/// `atoms` block's `x`/`y`/`z` of every frame, in frame order, an absent `y`
/// or `z` reading as zeros (a one- or two-dimensional series is a legitimate
/// input to a displacement; a frame with no `x` is not).
///
/// This is the series the displacement correlations run on —
/// [`Msd`](crate::compute::Msd) in window mode,
/// [`EinsteinDiffusion`](crate::compute::EinsteinDiffusion) — and what a
/// caller hands to [`displacement_correlate`](crate::compute::displacement_correlate)
/// for a lag range or origin stride those front ends do not expose.
///
/// # Errors
///
/// [`ComputeError::EmptyInput`] for no frames;
/// [`ComputeError::DimensionMismatch`] when a frame's atom count differs from
/// the first frame's; whatever the column read refuses (a missing `x`).
pub fn position_series<FA: FrameAccess>(frames: &[&FA]) -> Result<Array3<F>, ComputeError> {
    let n_frames = frames.len();
    if n_frames == 0 {
        return Err(ComputeError::EmptyInput);
    }
    let (xs0, _, _) = get_positions_ref_any_dim(frames[0])?;
    let n_atoms = xs0.slice().len();
    let mut positions = Array3::<F>::zeros((n_frames, n_atoms, 3));
    for (k, frame) in frames.iter().enumerate() {
        let (xp, yp, zp) = get_positions_ref_any_dim(*frame)?;
        let (xs, ys, zs) = (xp.slice(), yp.slice(), zp.slice());
        if xs.len() != n_atoms {
            return Err(ComputeError::DimensionMismatch {
                expected: n_atoms,
                got: xs.len(),
                what: "position_series atom count",
            });
        }
        for i in 0..n_atoms {
            positions[[k, i, 0]] = xs[i];
            positions[[k, i, 1]] = ys[i];
            positions[[k, i, 2]] = zs[i];
        }
    }
    Ok(positions)
}
