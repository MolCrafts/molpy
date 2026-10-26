use crate::io::mrec::zarr_storage::frame_io::{canonical_width, check_local_references, quantized};
use crate::io::mrec::zarr_storage::record_io::zerr;
use molrs::core::Frame;
use molrs::core::MolRsError;
use molrs::core::SimBox;
use molrs::core::{Block, Column, DType};
use ndarray::{ArrayD, Axis, Slice};

use super::schema::{BlockSchema, SequenceSchema};

/// Bitwise column equality — the predicate behind a section's `step_index`.
///
/// Bitwise rather than numeric: a NaN coordinate has to count as *unchanged*,
/// or a section that never moves would earn an entry at every step. Compared
/// element by element on the typed arrays, so a comparison allocates nothing.
fn same_column(left: &Column, right: &Column) -> bool {
    if left.dtype() != right.dtype() || left.shape() != right.shape() {
        return false;
    }
    macro_rules! bits {
        ($a:expr, $b:expr) => {
            $a.iter()
                .zip($b.iter())
                .all(|(x, y)| x.to_bits() == y.to_bits())
        };
    }
    macro_rules! exact {
        ($a:expr, $b:expr) => {
            $a.iter().zip($b.iter()).all(|(x, y)| x == y)
        };
    }
    match (left, right) {
        (Column::Float(a), Column::Float(b)) => bits!(a, b),
        (Column::C64(a), Column::C64(b)) => a
            .iter()
            .zip(b.iter())
            .all(|(x, y)| x.re.to_bits() == y.re.to_bits() && x.im.to_bits() == y.im.to_bits()),
        (Column::C128(a), Column::C128(b)) => a
            .iter()
            .zip(b.iter())
            .all(|(x, y)| x.re.to_bits() == y.re.to_bits() && x.im.to_bits() == y.im.to_bits()),
        (Column::I8(a), Column::I8(b)) => exact!(a, b),
        (Column::I16(a), Column::I16(b)) => exact!(a, b),
        (Column::Int(a), Column::Int(b)) => exact!(a, b),
        (Column::I64(a), Column::I64(b)) => exact!(a, b),
        (Column::U8(a), Column::U8(b)) => exact!(a, b),
        (Column::U16(a), Column::U16(b)) => exact!(a, b),
        (Column::U32(a), Column::U32(b)) => exact!(a, b),
        (Column::Uint(a), Column::Uint(b)) => exact!(a, b),
        (Column::Bool(a), Column::Bool(b)) => exact!(a, b),
        (Column::String(a), Column::String(b)) => exact!(a, b),
        _ => false,
    }
}

/// Whether two blocks carry the same content, column for column.
///
/// The validity masks count as content: two frames whose values agree while
/// one holds nulls under them are different frames, and letting the second
/// earn no update would carry the first frame's mask forward over it.
pub(super) fn same_block(left: &Block, right: &Block) -> bool {
    left.len() == right.len()
        && left.n_rows() == right.n_rows()
        && left.structural_shape() == right.structural_shape()
        && left.iter().all(|(name, column)| {
            right
                .get(name)
                .is_some_and(|other| same_column(column, other))
                && left.validity(name) == right.validity(name)
        })
}

/// `block` as a writer stores it under `declared`: every column whose
/// precision is declared rounded to its grid ([`quantized`]), carrying the
/// declaration; every other column as presented. Borrowed when the block
/// declares no precision.
pub(super) fn stored_presentation<'a>(
    declared: &BlockSchema,
    block: &'a Block,
) -> Result<std::borrow::Cow<'a, Block>, MolRsError> {
    let mut stored: Option<Block> = None;
    // A canonical key at another width of its family lands at its declared
    // dtype.
    for column in declared.columns.keys() {
        let Some(values) = block.get(column) else {
            continue;
        };
        if let Some(converted) = canonical_width(column, values)? {
            let target = stored.get_or_insert_with(|| block.clone());
            let mask = target.validity(column).map(<[bool]>::to_vec);
            target.insert(column.as_str(), converted, None)?;
            if let Some(mask) = mask {
                target.set_validity(column, mask)?;
            }
        }
    }
    for (column, schema) in &declared.columns {
        let Some(precision) = schema.precision else {
            continue;
        };
        let Some(values) = block.get(column).and_then(Column::get::<f64>) else {
            continue;
        };
        let rounded = Column::from(quantized(values, precision)?);
        let target = stored.get_or_insert_with(|| block.clone());
        // `Block::insert` drops the mask with the old column; the rows did not
        // move, so it goes back on.
        let mask = target.validity(column).map(<[bool]>::to_vec);
        target.insert(column.as_str(), rounded, None)?;
        if let Some(mask) = mask {
            target.set_validity(column, mask)?;
        }
        target.set_precision(column, precision)?;
    }
    // The pinned row-reference targets travel on the stored block, so the
    // resolved-frame check sees them whether the frame stated them or not.
    if declared
        .targets
        .iter()
        .any(|(column, target)| block.contains_key(column) && block.target(column) != Some(target))
    {
        let target_block = stored.get_or_insert_with(|| block.clone());
        for (column, target) in &declared.targets {
            if target_block.contains_key(column) {
                target_block.set_target(column, target)?;
            }
        }
    }
    Ok(match stored {
        Some(block) => std::borrow::Cow::Owned(block),
        None => std::borrow::Cow::Borrowed(block),
    })
}

/// Check a resolved trajectory frame against the rules that span blocks:
/// every aligned block present there has its target present with the same
/// row count, and every same-frame row reference the schema pins resolves.
/// `resolve(name)` is the block at this ordinal, or `None` where it is
/// absent.
pub(super) fn check_resolved_frame<'a>(
    schema: &SequenceSchema,
    resolve: impl Fn(&str) -> Option<&'a Block>,
    ordinal: u64,
) -> Result<(), MolRsError> {
    for (aligned, target) in schema.aligned_pairs() {
        let Some(block) = resolve(aligned) else {
            continue;
        };
        let rows = block.n_rows().unwrap_or(0);
        match resolve(target) {
            None => {
                return Err(MolRsError::zarr(format!(
                    "trajectory frame {ordinal}: block {aligned:?} ({rows} rows) is aligned with \
                     {target:?}, which is absent there"
                )));
            }
            Some(with) if with.n_rows().unwrap_or(0) != rows => {
                return Err(MolRsError::zarr(format!(
                    "trajectory frame {ordinal}: block {aligned:?} has {rows} rows but the block \
                     it is aligned with, {target:?}, has {}; restate {aligned:?} when \
                     {target:?} changes its row count",
                    with.n_rows().unwrap_or(0)
                )));
            }
            Some(_) => {}
        }
    }
    let mut resolved = Frame::new();
    for name in schema.blocks.keys() {
        if let Some(block) = resolve(name) {
            resolved.insert(name.clone(), block.clone());
        }
    }
    check_local_references(&resolved, &format!("trajectory frame {ordinal}"))
}

/// Whether two cells are the same cell, bit for bit.
pub(super) fn same_simbox(left: &SimBox, right: &SimBox) -> bool {
    left.is_cell_defined() == right.is_cell_defined()
        && left.pbc() == right.pbc()
        && left
            .matrix()
            .as_flattened()
            .iter()
            .map(|value| value.to_bits())
            .eq(right
                .matrix()
                .as_flattened()
                .iter()
                .map(|value| value.to_bits()))
        && left
            .origin()
            .iter()
            .map(|value| value.to_bits())
            .eq(right.origin().iter().map(|value| value.to_bits()))
}

/// Origin is optional on disk: all zeros is the default and is not written.
pub(super) fn origin_is_default(cell: &SimBox) -> bool {
    cell.origin().iter().all(|value| *value == 0.0)
}

/// Boundary is optional on disk: all-periodic is the default and is not written.
pub(super) fn boundary_is_default(cell: &SimBox) -> bool {
    cell.pbc() == [true, true, true]
}

/// Rows `start..end` of `column`, as a new column.
///
/// The whole column comes back as a cheap Arc clone; any other range is one
/// copy of exactly those rows.
pub(super) fn column_rows(column: &Column, start: usize, end: usize) -> Column {
    let rows = column.n_rows().unwrap_or(0);
    if start == 0 && end == rows {
        return column.clone();
    }
    macro_rules! slice {
        ($holder:expr) => {
            Column::from(
                $holder
                    .slice_axis(Axis(0), Slice::from(start..end))
                    .to_owned(),
            )
        };
    }
    match column {
        Column::Float(h) => slice!(h),
        Column::I8(h) => slice!(h),
        Column::I16(h) => slice!(h),
        Column::Int(h) => slice!(h),
        Column::I64(h) => slice!(h),
        Column::U8(h) => slice!(h),
        Column::U16(h) => slice!(h),
        Column::U32(h) => slice!(h),
        Column::Uint(h) => slice!(h),
        Column::Bool(h) => slice!(h),
        Column::String(h) => slice!(h),
        Column::C64(h) => slice!(h),
        Column::C128(h) => slice!(h),
    }
}

/// A zero-row column of `dtype` with `trailing` axes — the columns of a block
/// that is present and empty.
pub(super) fn empty_column(dtype: DType, trailing: &[u64]) -> Result<Column, MolRsError> {
    let mut shape = Vec::with_capacity(trailing.len() + 1);
    shape.push(0usize);
    shape.extend(trailing.iter().map(|&n| n as usize));
    macro_rules! empty {
        ($ty:ty) => {
            Column::from(ArrayD::<$ty>::from_shape_vec(shape, Vec::new()).map_err(zerr)?)
        };
    }
    Ok(match dtype {
        DType::Float => empty!(f64),
        DType::I8 => empty!(i8),
        DType::I16 => empty!(i16),
        DType::Int => empty!(i32),
        DType::I64 => empty!(i64),
        DType::U8 => empty!(u8),
        DType::U16 => empty!(u16),
        DType::U32 => empty!(u32),
        DType::Uint => empty!(u64),
        DType::Bool => empty!(bool),
        DType::String => empty!(String),
        DType::C64 => empty!(num_complex::Complex<f32>),
        DType::C128 => empty!(num_complex::Complex<f64>),
    })
}
