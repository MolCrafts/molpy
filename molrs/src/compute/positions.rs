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
