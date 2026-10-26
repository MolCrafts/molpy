//! The frame handle for the C++ engine — the CXX face of `molrs::core::Frame`
//! (through `molrs_ffi::FrameRef`): introspection, typed column readers and
//! create-or-update writers, exact-dtype metadata and the simulation box.

use molrs::core::{Block, BlockDtype, Frame, MetaValue};
use molrs::core::{Cell, Element, SimBox};
use ndarray::{Array1, ArrayD};

use crate::bridge;

// ── Frame bridge ─────────────────────────────────────────────────────────────
//
// `molrs_ffi::FrameRef` is the handle type used by every molrs language
// binding (python, wasm). It pairs a `FrameId` with the shared
// `FrameArenaCell` (`Rc<RefCell<FrameArena>>`) that owns the frame — cloning
// is two `Rc` bumps and keeps the arena alive. Atomiverse C++ reads/writes a `molrs.core.Frame` entirely through this
// handle, so the SCF / MD pipeline never copies a whole frame.
//
// cxx `extern "Rust"` opaque types are crate-local by convention, so the
// bridged type is a thin newtype around `molrs_ffi::FrameRef` rather than a
// re-export. All `frame_*` functions deref through `.0`.

/// CXX-bridged opaque handle to a `molrs.core.Frame`.
///
/// Newtype around [`molrs_ffi::FrameRef`]; the inner handle carries the
/// shared frame arena, so cloning this and round-tripping it through a
/// `Box`/raw-pointer (e.g. a Python `PyCapsule`) keeps the same underlying
/// frame data — mutations through one handle are visible through all clones.
///
/// `#[repr(transparent)]` guarantees this newtype has exactly the layout of
/// the wrapped `molrs_ffi::FrameRef`, so a `*const molrs_ffi::FrameRef` (e.g.
/// the pointer carried by molpy's `_ffi_frameref_capsule`) and a
/// `*const FrameRef` are layout-compatible. `frame_clone_from_addr` relies on
/// this only as a defensive guarantee — it still dereferences the pointer as
/// the *real* `molrs_ffi::FrameRef` type, never as a layout cast.
#[repr(transparent)]
pub struct FrameRef(pub molrs_ffi::FrameRef);

/// Create a fresh standalone frame (new arena + empty `Frame`).
///
/// @return boxed opaque handle owning a fresh frame arena
pub(crate) fn frame_new() -> Box<FrameRef> {
    Box::new(FrameRef(molrs_ffi::FrameRef::default()))
}

/// Clone a `molrs_ffi::FrameRef` reachable through a `molrs.FrameRef`
/// PyCapsule address into a bridge-side [`FrameRef`] handle.
///
/// This is the cross-extension ingress point. molpy's
/// `Frame._ffi_frameref_capsule()` produces a `PyCapsule` named
/// `molrs_ffi::abi::frameref_capsule_name()` — `"molrs.FrameRef/<major.minor>"`,
/// carrying the ABI line. PyO3 heap-boxes the capsule payload, and that payload
/// is a `#[repr(transparent)]` `FrameRefPtr` — itself a `*mut FrameRef`
/// (a clone of the Python frame's handle). The capsule's `void*` is
/// therefore `*mut *mut molrs_ffi::FrameRef`.
///
/// `addr` is `PyCapsule_GetPointer` cast to `usize`. This function resolves
/// it as `*const *const molrs_ffi::FrameRef`, dereferences once to reach the
/// cloned `*const FrameRef`, then `.clone()`s it — two cheap `Rc` bumps —
/// producing a handle onto the *same* shared frame arena. Reads and writes
/// through the returned handle are visible in the originating Python
/// `molrs.core.Frame`.
///
/// # Safety
///
/// `addr` must be the pointer returned by `PyCapsule_GetPointer` on a
/// capsule from `molpy.Frame._ffi_frameref_capsule()`, asked for with the
/// name THIS build accepts — [`frameref_capsule_name`], never a name the
/// producer reports — and valid for the duration of this call. The name
/// carries the molrs minor line and the build fingerprint (compiler and
/// handle layout), so a producer that may lay a handle out differently fails
/// the name check instead of being dereferenced here. The molrs FFI frame arena is
/// single-threaded and GIL-guarded; the caller must hold the GIL (or
/// otherwise guarantee exclusive access) while calling.
///
/// @param addr `molrs.FrameRef` capsule pointer (a `*mut *mut FrameRef`)
/// @return boxed bridge handle sharing the same arena as the source frame
pub(crate) unsafe fn frame_clone_from_addr(addr: usize) -> Box<FrameRef> {
    let pp = addr as *const *const molrs_ffi::FrameRef;
    let p = unsafe { *pp };
    let cloned = unsafe { (*p).clone() };
    Box::new(FrameRef(cloned))
}

/// The capsule name this build accepts:
/// `molrs_ffi::abi::frameref_capsule_name()`.
pub(crate) fn frameref_capsule_name() -> String {
    molrs_ffi::abi::frameref_capsule_name()
        .to_string_lossy()
        .into_owned()
}

/// List the block keys present in the frame.
///
/// @param fref frame handle
/// @return block names; an error for a handle that names no frame
pub(crate) fn frame_block_names(fref: &FrameRef) -> Result<Vec<String>, String> {
    fref.0
        .with(|f| f.keys().map(|s| s.to_string()).collect())
        .map_err(|e| format!("frame_block_names: {e}"))
}

/// Test whether the frame contains a block.
///
/// @param fref  frame handle
/// @param block block key
/// @return true if the block exists; an error for a handle that names no frame
pub(crate) fn frame_has_block(fref: &FrameRef, block: &str) -> Result<bool, String> {
    fref.0
        .has_block(block)
        .map_err(|e| format!("frame_has_block: {e}"))
}

/// List the column keys of a block.
///
/// @param fref  frame handle
/// @param block block key
/// @return column names; an error naming the block when it is absent
pub(crate) fn frame_block_columns(fref: &FrameRef, block: &str) -> Result<Vec<String>, String> {
    fref.0
        .block(block)
        .and_then(|blk| blk.keys())
        .map_err(|e| format!("frame_block_columns {block}: {e}"))
}

/// Row count of a block.
///
/// @param fref  frame handle
/// @param block block key
/// @return number of rows (0 for a block with no columns); an error naming
///         the block when it is absent
pub(crate) fn frame_block_n_rows(fref: &FrameRef, block: &str) -> Result<i64, String> {
    fref.0
        .block(block)
        .and_then(|blk| blk.n_rows())
        .map(|n| n as i64)
        .map_err(|e| format!("frame_block_n_rows {block}: {e}"))
}

/// Test whether the frame has a box.
///
/// @param fref frame handle
/// @return true if it has one; false when the frame is live and has none;
///         an error for a handle that names no frame
pub(crate) fn frame_has_box(fref: &FrameRef) -> Result<bool, String> {
    match fref.0.box_clone() {
        Ok(Some(_)) => Ok(true),
        Ok(None) => Ok(false),
        Err(e) => Err(format!("frame_has_box: {e}")),
    }
}

/// A [`bridge::ffi::KeyedMetaValue`] of `dtype` with every payload empty.
pub(crate) fn empty_keyed_meta_value(
    key: String,
    dtype: bridge::ffi::MetaType,
) -> bridge::ffi::KeyedMetaValue {
    bridge::ffi::KeyedMetaValue {
        key,
        dtype,
        bool_value: false,
        i32_value: 0,
        i64_value: 0,
        u32_value: 0,
        u64_value: 0,
        f64_value: 0.0,
        string_value: String::new(),
        bool_values: Vec::new(),
        i32_values: Vec::new(),
        i64_values: Vec::new(),
        u32_values: Vec::new(),
        u64_values: Vec::new(),
        f64_values: Vec::new(),
    }
}

/// The keys of the frame's metadata, in insertion order.
///
/// @param fref frame handle
/// @return metadata keys; empty when the live frame has none; an error for a
///         handle that names no frame
pub(crate) fn frame_meta_keys(fref: &FrameRef) -> Result<Vec<String>, String> {
    fref.0
        .with(|frame| frame.meta.keys().map(|key| key.to_string()).collect())
        .map_err(|e| format!("frame_meta_keys: {e}"))
}

/// One metadata value with its exact dtype and native payload.
///
/// @param fref frame handle
/// @param key  metadata key
/// @return the keyed value; an error when the key is absent
pub(crate) fn frame_get_meta(
    fref: &FrameRef,
    key: &str,
) -> Result<bridge::ffi::KeyedMetaValue, String> {
    fref.0
        .with(|frame| {
            frame
                .meta
                .get(key)
                .map(|value| keyed_meta_value(key, value))
        })
        .map_err(|e| format!("frame_get_meta: {e}"))?
        .ok_or_else(|| format!("frame_get_meta: no metadata key {key:?}"))
}

/// Marshal one metadata value into its bridge record.
pub(crate) fn keyed_meta_value(key: &str, value: &MetaValue) -> bridge::ffi::KeyedMetaValue {
    use bridge::ffi::MetaType;
    let dtype = match value {
        MetaValue::Bool(_) => MetaType::Bool,
        MetaValue::I32(_) => MetaType::I32,
        MetaValue::I64(_) => MetaType::I64,
        MetaValue::U32(_) => MetaType::U32,
        MetaValue::U64(_) => MetaType::U64,
        MetaValue::F64(_) => MetaType::F64,
        MetaValue::String(_) => MetaType::String,
        MetaValue::Bool3(_) => MetaType::Bool3,
        MetaValue::I32x3(_) => MetaType::I32x3,
        MetaValue::I64x3(_) => MetaType::I64x3,
        MetaValue::U32x3(_) => MetaType::U32x3,
        MetaValue::U64x3(_) => MetaType::U64x3,
        MetaValue::F64x3(_) => MetaType::F64x3,
        MetaValue::F64x6(_) => MetaType::F64x6,
        MetaValue::F64x9(_) => MetaType::F64x9,
        MetaValue::Json(_) => MetaType::String,
    };
    let mut entry = empty_keyed_meta_value(key.to_string(), dtype);
    match value {
        MetaValue::Bool(v) => entry.bool_value = *v,
        MetaValue::I32(v) => entry.i32_value = *v,
        MetaValue::I64(v) => entry.i64_value = *v,
        MetaValue::U32(v) => entry.u32_value = *v,
        MetaValue::U64(v) => entry.u64_value = *v,
        MetaValue::F64(v) => entry.f64_value = *v,
        MetaValue::String(v) => entry.string_value = v.clone(),
        MetaValue::Bool3(v) => entry.bool_values = v.iter().map(|v| u8::from(*v)).collect(),
        MetaValue::I32x3(v) => entry.i32_values = v.to_vec(),
        MetaValue::I64x3(v) => entry.i64_values = v.to_vec(),
        MetaValue::U32x3(v) => entry.u32_values = v.to_vec(),
        MetaValue::U64x3(v) => entry.u64_values = v.to_vec(),
        MetaValue::F64x3(v) => entry.f64_values = v.to_vec(),
        MetaValue::F64x6(v) => entry.f64_values = v.to_vec(),
        MetaValue::F64x9(v) => entry.f64_values = v.to_vec(),
        MetaValue::Json(v) => entry.string_value = v.to_string(),
    }
    entry
}

fn fixed<T, const N: usize>(values: Vec<T>, dtype: &str) -> Result<[T; N], String> {
    values.try_into().map_err(|values: Vec<T>| {
        format!("metadata {dtype} expects {N} values, got {}", values.len())
    })
}

/// Unmarshal a bridge metadata record; a payload whose length does not fit
/// its dtype is an error.
pub(crate) fn meta_from_keyed_value(
    entry: bridge::ffi::KeyedMetaValue,
) -> Result<(String, MetaValue), String> {
    use bridge::ffi::MetaType;
    if entry.key.is_empty() {
        return Err("metadata key must not be empty".into());
    }
    let value = match entry.dtype {
        MetaType::Bool => MetaValue::Bool(entry.bool_value),
        MetaType::I32 => MetaValue::I32(entry.i32_value),
        MetaType::I64 => MetaValue::I64(entry.i64_value),
        MetaType::U32 => MetaValue::U32(entry.u32_value),
        MetaType::U64 => MetaValue::U64(entry.u64_value),
        MetaType::F64 => MetaValue::F64(entry.f64_value),
        MetaType::String => MetaValue::String(entry.string_value),
        MetaType::Bool3 => {
            if entry.bool_values.iter().any(|&v| v > 1) {
                return Err("metadata bool3 values must be 0 or 1".into());
            }
            MetaValue::Bool3(fixed::<_, 3>(
                entry.bool_values.into_iter().map(|v| v != 0).collect(),
                "bool3",
            )?)
        }
        MetaType::I32x3 => MetaValue::I32x3(fixed(entry.i32_values, "i32x3")?),
        MetaType::I64x3 => MetaValue::I64x3(fixed(entry.i64_values, "i64x3")?),
        MetaType::U32x3 => MetaValue::U32x3(fixed(entry.u32_values, "u32x3")?),
        MetaType::U64x3 => MetaValue::U64x3(fixed(entry.u64_values, "u64x3")?),
        MetaType::F64x3 => MetaValue::F64x3(fixed(entry.f64_values, "f64x3")?),
        MetaType::F64x6 => MetaValue::F64x6(fixed(entry.f64_values, "f64x6")?),
        MetaType::F64x9 => MetaValue::F64x9(fixed(entry.f64_values, "f64x9")?),
        _ => return Err("unknown metadata dtype".into()),
    };
    Ok((entry.key, value))
}

/// Insert or replace one exact-dtype metadata value.
pub(crate) fn frame_set_meta(
    fref: &mut FrameRef,
    entry: bridge::ffi::KeyedMetaValue,
) -> Result<(), String> {
    let (key, value) = meta_from_keyed_value(entry)?;
    fref.0
        .with_meta_mut(|meta| {
            meta.insert(key, value);
        })
        .map_err(|err| err.to_string())
}

/// Declare the precision of an existing `f64` column: an absolute tolerance
/// in the column's units. A `*.mrec` writer (`write_mrec_frame`, an
/// `MrecWriterRef` minted from this frame) stores the column rounded to the largest
/// power of two not above it, shuffled and compressed; the in-memory values
/// are untouched.
///
/// @param fref      frame handle
/// @param block     block key
/// @param col       column key (must exist and be `f64`)
/// @param precision finite, within `[2^-1000, 2^1000]`
pub(crate) fn frame_set_precision(
    fref: &mut FrameRef,
    block: &str,
    col: &str,
    precision: f64,
) -> Result<(), String> {
    fref.0
        .with_mut(|frame| {
            frame
                .get_mut(block)
                .ok_or_else(|| format!("frame_set_precision: no block {block:?}"))?
                .set_precision(col, precision)
                .map_err(|e| format!("frame_set_precision {col}: {e}"))
        })
        .map_err(|e| format!("frame_set_precision: {e}"))?
}

/// Copy a column of dtype `T` out of a block — the one reader behind the four
/// per-dtype bridge doors (cxx has no generics).
///
/// An absent block or column is an error naming it, and a column of another
/// dtype one naming both dtypes: never an empty read.
fn frame_column<T: BlockDtype>(
    fref: &FrameRef,
    block: &str,
    col: &str,
    what: &str,
) -> Result<Vec<T>, String> {
    let blk = fref
        .0
        .block(block)
        .map_err(|e| format!("{what} {block}.{col}: {e}"))?;
    match blk.copy::<T>(col) {
        Ok(Some((data, _shape))) => Ok(data),
        Ok(None) => Err(format!("{what} {block}.{col}: no column '{col}'")),
        Err(e) => Err(format!("{what} {block}.{col}: {e}")),
    }
}

/// Copy an `f64` column out of a block.
///
/// @param fref  frame handle
/// @param block block key
/// @param col   column key
/// @return owned column data. An absent block, an absent column, a column
///         of another dtype, and a dropped frame are all errors.
pub(crate) fn frame_column_f64(
    fref: &FrameRef,
    block: &str,
    col: &str,
) -> Result<Vec<f64>, String> {
    frame_column(fref, block, col, "frame_column_f64")
}

/// Copy an `i32` column out of a block.
///
/// @param fref  frame handle
/// @param block block key
/// @param col   column key
/// @return owned column data. An absent block, an absent column, a column
///         of another dtype, and a dropped frame are all errors.
pub(crate) fn frame_column_i32(
    fref: &FrameRef,
    block: &str,
    col: &str,
) -> Result<Vec<i32>, String> {
    frame_column(fref, block, col, "frame_column_i32")
}

/// Copy a domain-uint (`u64` / `Idx`) column out of a block.
///
/// @param fref  frame handle
/// @param block block key
/// @param col   column key
/// @return owned column data. An absent block, an absent column, a column
///         of another dtype, and a dropped frame are all errors.
pub(crate) fn frame_column_u64(
    fref: &FrameRef,
    block: &str,
    col: &str,
) -> Result<Vec<u64>, String> {
    frame_column(fref, block, col, "frame_column_u64")
}

/// Copy a string column out of a block.
///
/// @param fref  frame handle
/// @param block block key
/// @param col   column key
/// @return owned column data. An absent block, an absent column, a column
///         of another dtype, and a dropped frame are all errors.
pub(crate) fn frame_column_str(
    fref: &FrameRef,
    block: &str,
    col: &str,
) -> Result<Vec<String>, String> {
    frame_column(fref, block, col, "frame_column_str")
}

/// Read the frame's box: its cell matrix H as 9 row-major `f64` (columns are
/// the lattice vectors, as molrs `SimBox::matrix` and LAMMPS write H; empty
/// for a box whose cell is undefined — molrs `Cell::Undefined` stores an
/// identity placeholder that is not a cell), its origin and its periodic
/// axes.
///
/// @param fref frame handle
/// @return the box; an error when the frame has none
pub(crate) fn frame_box(fref: &FrameRef) -> Result<bridge::ffi::BoxData, String> {
    let sb = fref
        .0
        .box_clone()
        .map_err(|e| format!("frame_box: {e}"))?
        .ok_or("frame_box: the frame has no box (ask frame_has_box first)")?;
    Ok(bridge::ffi::BoxData {
        h: if sb.is_cell_defined() {
            sb.matrix().into_iter().flatten().collect()
        } else {
            Vec::new()
        },
        origin: sb.origin().to_vec(),
        pbc: sb.pbc().to_vec(),
    })
}

/// Get-or-create a block by key, then run fallible `f` to populate it.
pub(crate) fn with_block_inserted_res<T, E>(
    frame: &mut Frame,
    block: &str,
    f: impl FnOnce(&mut Block) -> Result<T, E>,
) -> Result<T, E> {
    if let Some(blk) = frame.get_mut(block) {
        f(blk)
    } else {
        let mut blk = Block::new();
        let out = f(&mut blk)?;
        frame.insert(block, blk);
        Ok(out)
    }
}

/// Create or overwrite an `f64` column on a block.
///
/// @param fref  frame handle
/// @param block block key (created if absent)
/// @param col   column key (overwritten if present)
/// @param data  column values
pub(crate) fn frame_set_column_f64(
    fref: &mut FrameRef,
    block: &str,
    col: &str,
    data: &[f64],
) -> Result<(), String> {
    fref.0
        .with_mut(|frame| {
            with_block_inserted_res(frame, block, |blk| {
                blk.insert(col, Array1::from_vec(data.to_vec()).into_dyn(), None)
                    .map_err(|e| format!("frame_set_column_f64 insert {col}: {e}"))
            })
        })
        .map_err(|e| format!("frame_set_column_f64: {e}"))?
}

/// Create or overwrite an `i32` column on a block.
///
/// @param fref  frame handle
/// @param block block key (created if absent)
/// @param col   column key (overwritten if present)
/// @param data  column values
pub(crate) fn frame_set_column_i32(
    fref: &mut FrameRef,
    block: &str,
    col: &str,
    data: &[i32],
) -> Result<(), String> {
    fref.0
        .with_mut(|frame| {
            with_block_inserted_res(frame, block, |blk| {
                blk.insert(col, Array1::from_vec(data.to_vec()).into_dyn(), None)
                    .map_err(|e| format!("frame_set_column_i32 insert {col}: {e}"))
            })
        })
        .map_err(|e| format!("frame_set_column_i32: {e}"))?
}

/// Create or overwrite a domain-uint (`u64` / `Idx`) column on a block.
///
/// @param fref  frame handle
/// @param block block key (created if absent)
/// @param col   column key (overwritten if present)
/// @param data  column values
pub(crate) fn frame_set_column_u64(
    fref: &mut FrameRef,
    block: &str,
    col: &str,
    data: &[u64],
) -> Result<(), String> {
    fref.0
        .with_mut(|frame| {
            with_block_inserted_res(frame, block, |blk| {
                blk.insert(col, Array1::from_vec(data.to_vec()).into_dyn(), None)
                    .map_err(|e| format!("frame_set_column_u64 insert {col}: {e}"))
            })
        })
        .map_err(|e| format!("frame_set_column_u64: {e}"))?
}

/// Create or overwrite a string column on a block.
///
/// @param fref  frame handle
/// @param block block key (created if absent)
/// @param col   column key (overwritten if present)
/// @param data  column values
pub(crate) fn frame_set_column_str(
    fref: &mut FrameRef,
    block: &str,
    col: &str,
    data: &[String],
) -> Result<(), String> {
    fref.0
        .with_mut(|frame| {
            with_block_inserted_res(frame, block, |blk| {
                blk.insert(
                    col,
                    Array1::from_vec(data.to_vec()).into_dyn() as ArrayD<String>,
                    None,
                )
                .map_err(|e| format!("frame_set_column_str insert {col}: {e}"))
            })
        })
        .map_err(|e| format!("frame_set_column_str: {e}"))?
}

/// Write `atoms.atomic_number` from the `atoms.element` symbols — molrs
/// [`Element::fill_atomic_numbers`], the explicit element → Z conversion
/// (`*` is Z = 0). `element` is left in place.
///
/// @param fref frame handle
/// @return an error when there is no `atoms.element` string column or a row
///         holds a symbol that is not an element (the row is named)
pub(crate) fn frame_fill_atomic_numbers(fref: &mut FrameRef) -> Result<(), String> {
    fref.0
        .with_mut(Element::fill_atomic_numbers)
        .map_err(|e| format!("frame_fill_atomic_numbers: {e}"))?
        .map_err(|e| format!("frame_fill_atomic_numbers: {e}"))
}

/// Write `atoms.element` symbols from `atoms.atomic_number` — molrs
/// [`Element::fill_symbols`], the explicit Z → element conversion (Z = 0
/// becomes `*`). `atomic_number` is left in place.
///
/// @param fref frame handle
/// @return an error when there is no `atoms.atomic_number` Uint column or a
///         row holds a number outside 0..=118 (the row is named)
pub(crate) fn frame_fill_symbols(fref: &mut FrameRef) -> Result<(), String> {
    fref.0
        .with_mut(Element::fill_symbols)
        .map_err(|e| format!("frame_fill_symbols: {e}"))?
        .map_err(|e| format!("frame_fill_symbols: {e}"))
}

/// A periodic [`SimBox`] at the origin from a row-major 3×3 cell matrix H.
///
/// @param h 9-element row-major H
/// @return the box; an error on a length other than 9 or a singular H
pub(crate) fn simbox_from_h(h: &[f64]) -> Result<SimBox, String> {
    let rows: [[f64; 3]; 3] = match h.as_chunks::<3>() {
        (&[a, b, c], []) => [a, b, c],
        _ => return Err(format!("H must have 9 elements, got {}", h.len())),
    };
    SimBox::new(Cell::Matrix(rows), [0.0; 3], [true; 3])
        .map_err(|e| format!("singular cell matrix H: {e}"))
}

/// Set the frame's box from its cell matrix H (9 row-major `f64`), its
/// origin and its periodic axes.
///
/// @param fref   frame handle
/// @param h      9-element row-major H
/// @param origin 3-element origin
/// @param pbc    3 periodic flags (x, y, z)
pub(crate) fn frame_set_box(
    fref: &mut FrameRef,
    h: &[f64],
    origin: &[f64],
    pbc: &[bool],
) -> Result<(), String> {
    let refuse = |e: String| format!("frame_set_box: {e}");
    let cell = simbox_from_h(h).map_err(refuse)?;
    let origin: [f64; 3] = origin
        .try_into()
        .map_err(|_| refuse(format!("origin must have 3 elements, got {}", origin.len())))?;
    let pbc: [bool; 3] = pbc
        .try_into()
        .map_err(|_| refuse(format!("pbc must have 3 flags, got {}", pbc.len())))?;
    let simbox = SimBox::new(Cell::Matrix(cell.matrix()), origin, pbc)
        .map_err(|e| refuse(format!("{e:?}")))?;
    fref.0
        .set_box(Some(simbox))
        .map_err(|e| refuse(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::rc::Rc;

    /// Round-trip: write every column dtype + simbox through the bridge,
    /// then read them back and assert element-wise equality. Also exercises
    /// the introspection surface (`names`, `has`, `columns`, `nrows`).
    #[test]
    fn frame_roundtrip_all_dtypes() {
        let mut fref = frame_new();

        // One column per dtype, each on a key whose *schema* dtype matches it.
        // The vocabulary binds a key's dtype wherever it appears, so writing an
        // i32 into `id` (a UInt) is refused — which is the schema working, not
        // the bridge failing.
        let xs: Vec<f64> = vec![0.0, 1.5, -2.25];
        let spins: Vec<i32> = vec![-1, 0, 1];
        let ids: Vec<u64> = vec![10, 20, 30];
        let elems: Vec<String> = vec!["H".into(), "H".into(), "O".into()];

        frame_set_column_f64(&mut fref, "atoms", "x", &xs).unwrap();
        frame_set_column_i32(&mut fref, "atoms", "spin", &spins).unwrap();
        frame_set_column_u64(&mut fref, "atoms", "id", &ids).unwrap();
        frame_set_column_str(&mut fref, "atoms", "element", &elems).unwrap();

        // 9-elem row-major 3x3 H matrix.
        let h: Vec<f64> = vec![12.0, 0.0, 0.0, 0.0, 13.0, 0.0, 0.0, 0.0, 14.0];
        frame_set_box(&mut fref, &h, &[1.0, -2.0, 0.5], &[true, true, false]).unwrap();

        // ── Introspection ──
        let names = frame_block_names(&fref).unwrap();
        assert_eq!(names, vec!["atoms".to_string()]);
        assert!(frame_has_block(&fref, "atoms").unwrap());
        assert!(!frame_has_block(&fref, "bonds").unwrap());

        let mut cols = frame_block_columns(&fref, "atoms").unwrap();
        cols.sort();
        assert_eq!(cols, vec!["element", "id", "spin", "x"]);
        assert_eq!(frame_block_n_rows(&fref, "atoms").unwrap(), 3);
        assert!(frame_block_n_rows(&fref, "missing").is_err());

        // ── Readers ──
        assert_eq!(frame_column_f64(&fref, "atoms", "x").unwrap(), xs);
        assert_eq!(frame_column_i32(&fref, "atoms", "spin").unwrap(), spins);
        assert_eq!(frame_column_u64(&fref, "atoms", "id").unwrap(), ids);
        assert_eq!(frame_column_str(&fref, "atoms", "element").unwrap(), elems);
        // The box crosses whole: H, origin and periodicity.
        let sb = frame_box(&fref).unwrap();
        assert_eq!(sb.h, h);
        assert_eq!(sb.origin, [1.0, -2.0, 0.5]);
        assert_eq!(sb.pbc, [true, true, false]);

        // ── Absent block / column → an error naming it, never an empty read ──
        let err = frame_column_f64(&fref, "atoms", "nope").unwrap_err();
        assert!(err.contains("nope"), "{err}");
        assert!(frame_column_f64(&fref, "missing", "x").is_err());
        assert!(frame_column_i32(&fref, "missing", "spin").is_err());
        assert!(frame_column_u64(&fref, "missing", "id").is_err());
        assert!(frame_column_str(&fref, "missing", "element").is_err());
        assert!(frame_block_columns(&fref, "missing").is_err());

        // ── A column of another dtype is an error, never an empty read ──
        let err = frame_column_i32(&fref, "atoms", "x").unwrap_err();
        assert!(err.contains("atoms.x"), "{err}");
        assert!(frame_column_u64(&fref, "atoms", "spin").is_err());
        assert!(frame_column_str(&fref, "atoms", "id").is_err());
        assert!(frame_column_f64(&fref, "atoms", "element").is_err());

        // ── Update path: overwrite an existing column ──
        let xs2: Vec<f64> = vec![9.0, 8.0, 7.0];
        frame_set_column_f64(&mut fref, "atoms", "x", &xs2).unwrap();
        assert_eq!(frame_column_f64(&fref, "atoms", "x").unwrap(), xs2);
        // other columns survive the update
        assert_eq!(frame_column_u64(&fref, "atoms", "id").unwrap(), ids);
    }

    #[test]
    fn frame_metadata_roundtrip_preserves_exact_native_types() {
        use bridge::ffi::MetaType;
        let mut fref = frame_new();

        let mut tag = empty_keyed_meta_value("tag".into(), MetaType::I64);
        tag.i64_value = 9_007_199_254_740_993;
        frame_set_meta(&mut fref, tag).unwrap();

        let mut stress = empty_keyed_meta_value("stress".into(), MetaType::F64x6);
        stress.f64_values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        frame_set_meta(&mut fref, stress).unwrap();

        assert_eq!(frame_meta_keys(&fref).unwrap().len(), 2);
        let tag = frame_get_meta(&fref, "tag").unwrap();
        assert!(tag.dtype == MetaType::I64);
        assert_eq!(tag.i64_value, 9_007_199_254_740_993);
        let stress = frame_get_meta(&fref, "stress").unwrap();
        assert!(stress.dtype == MetaType::F64x6);
        assert_eq!(stress.f64_values, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        assert!(frame_get_meta(&fref, "absent").is_err());

        let mut malformed = empty_keyed_meta_value("bad".into(), MetaType::F64x6);
        malformed.f64_values = vec![1.0; 5];
        assert!(frame_set_meta(&mut fref, malformed).is_err());
    }

    #[test]
    fn frame_meta_keys_follow_insertion_order() {
        use bridge::ffi::MetaType;
        let mut fref = frame_new();

        // Not alphabetical: a reintroduced sort must fail this case.
        let mut tag = empty_keyed_meta_value("tag".into(), MetaType::I64);
        tag.i64_value = 1;
        frame_set_meta(&mut fref, tag).unwrap();

        let mut stress = empty_keyed_meta_value("stress".into(), MetaType::F64);
        stress.f64_value = 2.0;
        frame_set_meta(&mut fref, stress).unwrap();

        let mut run = empty_keyed_meta_value("run".into(), MetaType::String);
        run.string_value = "third".into();
        frame_set_meta(&mut fref, run).unwrap();

        let keys = frame_meta_keys(&fref).unwrap();
        assert_eq!(
            keys,
            vec!["tag".to_string(), "stress".to_string(), "run".to_string()]
        );
    }

    /// Capsule / shared-arena semantics: a cloned `FrameRef` shares the
    /// same frame arena. Round-tripping through `Box::into_raw`/`from_raw`
    /// (the PyCapsule pattern) must preserve `Rc::ptr_eq`, and a mutation
    /// through one handle must be visible through the other.
    #[test]
    fn cloned_frameref_shares_arena() {
        let mut original = frame_new();
        frame_set_column_f64(&mut original, "atoms", "x", &[1.0, 2.0, 3.0]).unwrap();

        // Clone + simulate the PyCapsule carry: Box::into_raw → Box::from_raw.
        let cloned: FrameRef = FrameRef(original.0.clone());
        let raw: *mut FrameRef = Box::into_raw(Box::new(cloned));
        let mut recovered: Box<FrameRef> = unsafe { Box::from_raw(raw) };

        // Same arena behind both handles.
        assert!(Rc::ptr_eq(&original.0.arena, &recovered.0.arena));

        // Mutate through the recovered handle, observe through the original.
        frame_set_column_f64(&mut recovered, "atoms", "x", &[7.0, 8.0, 9.0]).unwrap();
        assert_eq!(
            frame_column_f64(&original, "atoms", "x").unwrap(),
            vec![7.0, 8.0, 9.0]
        );

        // ...and vice versa.
        frame_set_column_u64(&mut original, "atoms", "id", &[1, 2, 3]).unwrap();
        assert_eq!(
            frame_column_u64(&recovered, "atoms", "id").unwrap(),
            vec![1, 2, 3]
        );
    }

    /// A dropped frame is an error on every probe. Absence of a block, a box
    /// or metadata stays `Ok` on a live frame.
    #[test]
    fn a_dropped_frame_is_an_error_on_every_probe() {
        let fref = frame_new();
        let stale = FrameRef(fref.0.clone());
        fref.0.clone().drop_frame().unwrap();
        let err = frame_has_block(&stale, "atoms").unwrap_err();
        assert!(err.contains("Invalid frame"), "{err}");
        let err = frame_has_box(&stale).unwrap_err();
        assert!(err.contains("Invalid frame"), "{err}");
        let err = frame_meta_keys(&stale).unwrap_err();
        assert!(err.contains("Invalid frame"), "{err}");
        let err = frame_block_names(&stale).unwrap_err();
        assert!(err.contains("Invalid frame"), "{err}");
    }

    /// A frame without a box is an error to read; a box without a defined
    /// cell carries an identity placeholder, which is not a cell: it reads
    /// as no H.
    #[test]
    fn a_box_without_a_defined_cell_reads_as_no_h() {
        let fref = frame_new();
        assert!(!frame_has_box(&fref).unwrap());
        assert!(frame_box(&fref).is_err());
        let undefined = SimBox::new(Cell::Undefined, [0.0; 3], [false; 3]).unwrap();
        fref.0.set_box(Some(undefined)).unwrap();
        assert!(frame_box(&fref).unwrap().h.is_empty());
        let mut fref = fref;
        let h = [2.0, 0.5, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 4.0];
        frame_set_box(&mut fref, &h, &[0.0; 3], &[true; 3]).unwrap();
        assert_eq!(frame_box(&fref).unwrap().h, h);
        assert!(frame_set_box(&mut fref, &[0.0; 9], &[0.0; 3], &[true; 3]).is_err());
        assert!(frame_set_box(&mut fref, &[1.0; 8], &[0.0; 3], &[true; 3]).is_err());
        assert!(frame_set_box(&mut fref, &h, &[0.0; 2], &[true; 3]).is_err());
        assert!(frame_set_box(&mut fref, &h, &[0.0; 3], &[true; 2]).is_err());
    }

    #[test]
    fn species_conversions_are_explicit_and_name_the_bad_row() {
        let mut fref = frame_new();
        frame_set_column_str(
            &mut fref,
            "atoms",
            "element",
            &["O".into(), "h".into(), "*".into()],
        )
        .unwrap();
        frame_fill_atomic_numbers(&mut fref).unwrap();
        assert_eq!(
            frame_column_u64(&fref, "atoms", "atomic_number").unwrap(),
            [8u64, 1, 0]
        );

        let mut zs = frame_new();
        frame_set_column_u64(&mut zs, "atoms", "atomic_number", &[6, 0, 1]).unwrap();
        frame_fill_symbols(&mut zs).unwrap();
        assert_eq!(
            frame_column_str(&zs, "atoms", "element").unwrap(),
            ["C", "*", "H"]
        );

        let mut bad = frame_new();
        frame_set_column_str(&mut bad, "atoms", "element", &["C".into(), "Xx".into()]).unwrap();
        let err = frame_fill_atomic_numbers(&mut bad).unwrap_err();
        assert!(err.contains("row 1") && err.contains("Xx"), "{err}");
        assert!(
            frame_fill_symbols(&mut bad).is_err(),
            "no atomic_number column"
        );
        frame_set_column_u64(&mut bad, "atoms", "atomic_number", &[6, 119]).unwrap();
        let err = frame_fill_symbols(&mut bad).unwrap_err();
        assert!(err.contains("119"), "{err}");
    }
}
