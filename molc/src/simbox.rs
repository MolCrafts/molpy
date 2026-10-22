//! `extern "C"` functions for Box (simulation cell) operations.
//!
//! A **Box** defines a periodic simulation cell via a 3x3 cell matrix
//! **H** (columns are lattice vectors), an origin point, and per-axis
//! periodic boundary condition (PBC) flags.
//!
//! All lengths are in Angstrom.  The cell matrix is stored and
//! transmitted in **row-major** order as a flat array of 9 floats.
//!
//! # Coordinate conventions
//!
//! * Cell matrix **H**: `h9[3 * i + j]` is `H[i][j]`, row `i`, column `j`;
//!   lattice vector `j` is column `j`, so **a** is `(h9[0], h9[3], h9[6])`.
//! * Origin: lower corner of the simulation box in Angstrom.
//! * PBC flags: `[px, py, pz]` -- `true` if periodic along that axis.
//! * Tilts: off-diagonal elements `[xy, xz, yz]` in Angstrom.
//!
//! # Constructing a cell
//!
//! [`molrs_box_new`] and
//! [`molrs_region_parallelepiped`](crate::molrs_region_parallelepiped) take
//! the cell the way molrs's `Cell` states it: a [`MolrsCellKind`] and the
//! values that kind reads.

use ndarray::ArrayView2;

use molrs::core::{BoxError, Cell, SimBox};

use crate::error::{self, MolrsStatus};
use crate::handle::{MolrsBoxHandle, box_key_to_handle, handle_to_box_key};
use crate::handle_registry::lock_registry;
use crate::{ffi_try, null_check};
use molrs::op::F;

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

/// How a cell is stated — the C face of molrs's `Cell`. The kind fixes how
/// many values the accompanying `cell` array holds and what they mean.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MolrsCellKind {
    /// 9 values: the cell matrix `H` row-major, lattice vectors as columns
    /// (`cell[3 * i + j]` is `H[i][j]`).
    Matrix = 0,
    /// 3 values: orthorhombic edge lengths `[a, b, c]` (Å), all positive.
    Lengths = 1,
    /// 6 values: edge lengths `[a, b, c]` (Å) then the angles
    /// `[alpha, beta, gamma]` (degrees) of a restricted-triclinic cell.
    LengthsAngles = 2,
    /// 6 values: diagonal sizes `[lx, ly, lz]` (Å) then the tilts
    /// `[xy, xz, yz]` (Å) of a restricted-triclinic cell — the LAMMPS form.
    LengthsTilts = 3,
    /// No values: no geometric cell (a no-cell box of undefined volume);
    /// `cell` is not read and may be `NULL`.
    Undefined = 4,
}

/// The `Cell` that `kind` and the `cell` values state.
///
/// # Safety
///
/// For every kind but `Undefined`, `cell` points to the number of readable
/// `F` values [`MolrsCellKind`] names.
pub(crate) unsafe fn cell_from_c(kind: MolrsCellKind, cell: *const F) -> Result<Cell, MolrsStatus> {
    let read = |n: usize| -> Result<&[F], MolrsStatus> {
        if cell.is_null() {
            error::set_last_error(format!("null pointer: cell ({kind:?} reads {n} values)"));
            return Err(MolrsStatus::NullPointer);
        }
        Ok(unsafe { std::slice::from_raw_parts(cell, n) })
    };
    let triple = |v: &[F]| [v[0], v[1], v[2]];
    Ok(match kind {
        MolrsCellKind::Matrix => {
            let v = read(9)?;
            Cell::Matrix([triple(&v[0..3]), triple(&v[3..6]), triple(&v[6..9])])
        }
        MolrsCellKind::Lengths => Cell::Lengths(triple(read(3)?)),
        MolrsCellKind::LengthsAngles => {
            let v = read(6)?;
            Cell::LengthsAngles {
                lengths: triple(&v[0..3]),
                angles: triple(&v[3..6]),
            }
        }
        MolrsCellKind::LengthsTilts => {
            let v = read(6)?;
            Cell::LengthsTilts {
                lengths: triple(&v[0..3]),
                tilts: triple(&v[3..6]),
            }
        }
        MolrsCellKind::Undefined => Cell::Undefined,
    })
}

/// Create a Box from its cell, origin and per-axis periodicity — the C face
/// of molrs's `SimBox::new(cell, origin, pbc)`.
///
/// # C signature
///
/// ```c
/// MolrsStatus molrs_box_new(MolrsCellKind kind,
///                           const F* cell,
///                           const F origin3[3],
///                           const bool pbc3[3],
///                           MolrsBoxHandle* out);
/// ```
///
/// # Arguments
///
/// * `kind` -- How `cell` states the cell (see [`MolrsCellKind`]): a cube of
///   edge `a` is `MOLRS_CELL_KIND_LENGTHS` with `{a, a, a}`, a general cell
///   `MOLRS_CELL_KIND_MATRIX` with the 9 values of `H`.
/// * `cell` -- The values `kind` reads (9, 3, 6, 6 or none).
/// * `origin3` -- Box origin `[ox, oy, oz]` in Angstrom (3 floats).
/// * `pbc3` -- Per-axis periodic boundary flags `[px, py, pz]` (3 bools).
/// * `out` -- On success, receives the new Box handle.
///
/// # Returns
///
/// * `MolrsStatus::Ok` on success.
/// * `MolrsStatus::NullPointer` if `origin3`, `pbc3`, `out`, or (for a kind
///   that reads values) `cell` is null.
/// * `MolrsStatus::SingularCell` if a `MATRIX` cell is singular
///   (determinant = 0).
/// * `MolrsStatus::InvalidArgument` if a length is zero, negative or not
///   finite, or lengths and angles describe no cell.
///
/// # Safety
///
/// * `cell` must point to the number of readable `F` values `kind` names
///   (it is not read for `MOLRS_CELL_KIND_UNDEFINED`).
/// * `origin3` must point to at least 3 readable `F` values.
/// * `pbc3` must point to at least 3 readable `bool` values.
/// * `out` must point to a writable `MolrsBoxHandle`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn molrs_box_new(
    kind: MolrsCellKind,
    cell: *const F,
    origin3: *const F,
    pbc3: *const bool,
    out: *mut MolrsBoxHandle,
) -> MolrsStatus {
    ffi_try!({
        null_check!(origin3);
        null_check!(pbc3);
        null_check!(out);
        let cell = match unsafe { cell_from_c(kind, cell) } {
            Ok(c) => c,
            Err(status) => return status,
        };
        let o = unsafe { std::slice::from_raw_parts(origin3, 3) };
        let p = unsafe { std::slice::from_raw_parts(pbc3, 3) };
        let sb = match SimBox::new(cell, [o[0], o[1], o[2]], [p[0], p[1], p[2]]) {
            Ok(sb) => sb,
            Err(e) => {
                let status = match e {
                    BoxError::SingularCell => MolrsStatus::SingularCell,
                    _ => MolrsStatus::InvalidArgument,
                };
                error::set_last_error(e.to_string());
                return status;
            }
        };
        let mut registry = lock_registry();
        let key = registry.simboxes.insert(sb);
        unsafe { *out = box_key_to_handle(key) };
        MolrsStatus::Ok
    })
}

/// Destroy a Box and invalidate its handle.
///
/// # C signature
///
/// ```c
/// MolrsStatus molrs_box_drop(MolrsBoxHandle handle);
/// ```
///
/// # Arguments
///
/// * `handle` -- The Box to destroy.
///
/// # Returns
///
/// * `MolrsStatus::Ok` on success.
/// * `MolrsStatus::InvalidBoxHandle` if `handle` is stale.
///
/// # Safety
///
/// The caller must not use `handle` after this call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn molrs_box_drop(handle: MolrsBoxHandle) -> MolrsStatus {
    ffi_try!({
        let mut registry = lock_registry();
        let key = handle_to_box_key(handle);
        match registry.simboxes.remove(key) {
            Some(_) => MolrsStatus::Ok,
            None => {
                error::set_last_error("invalid simbox handle");
                MolrsStatus::InvalidBoxHandle
            }
        }
    })
}

// ---------------------------------------------------------------------------
// Queries
// ---------------------------------------------------------------------------

/// Helper: get a reference to a Box by handle, returning error if invalid.
macro_rules! get_simbox {
    ($registry:expr, $handle:expr) => {
        match $registry.simboxes.get(handle_to_box_key($handle)) {
            Some(sb) => sb,
            None => {
                error::set_last_error("invalid simbox handle");
                return MolrsStatus::InvalidBoxHandle;
            }
        }
    };
}

/// Copy the 3x3 cell matrix into a caller buffer (row-major, 9 floats,
/// lattice vectors as columns: `out9[3 * i + j]` is `H[i][j]`).
///
/// The values are in Angstrom.
///
/// # C signature
///
/// ```c
/// MolrsStatus molrs_box_h(MolrsBoxHandle box_handle, F out9[9]);
/// ```
///
/// # Arguments
///
/// * `box_handle` -- Box handle.
/// * `out9` -- Buffer of at least 9 floats; receives the cell matrix.
///
/// # Returns
///
/// * `MolrsStatus::Ok` on success.
/// * `MolrsStatus::NullPointer` if `out9` is null.
/// * `MolrsStatus::InvalidBoxHandle` if `box_handle` is stale.
///
/// # Safety
///
/// * `box_handle` must be a live Box handle.
/// * `out9` must point to at least 9 writable `F` values.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn molrs_box_h(box_handle: MolrsBoxHandle, out9: *mut F) -> MolrsStatus {
    ffi_try!({
        null_check!(out9);
        let registry = lock_registry();
        let sb = get_simbox!(registry, box_handle);
        let out_slice = unsafe { std::slice::from_raw_parts_mut(out9, 9) };
        out_slice.copy_from_slice(sb.matrix().as_flattened());
        MolrsStatus::Ok
    })
}

/// Copy the box origin into a caller buffer (3 floats, in Angstrom).
///
/// # C signature
///
/// ```c
/// MolrsStatus molrs_box_origin(MolrsBoxHandle box_handle,
///                                  F out3[3]);
/// ```
///
/// # Arguments
///
/// * `box_handle` -- Box handle.
/// * `out3` -- Buffer of at least 3 floats; receives `[ox, oy, oz]`.
///
/// # Returns
///
/// * `MolrsStatus::Ok` on success.
/// * `MolrsStatus::NullPointer` if `out3` is null.
/// * `MolrsStatus::InvalidBoxHandle` if `box_handle` is stale.
///
/// # Safety
///
/// * `box_handle` must be a live Box handle.
/// * `out3` must point to at least 3 writable floats.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn molrs_box_origin(box_handle: MolrsBoxHandle, out3: *mut F) -> MolrsStatus {
    ffi_try!({
        null_check!(out3);
        let registry = lock_registry();
        let sb = get_simbox!(registry, box_handle);
        let out_slice = unsafe { std::slice::from_raw_parts_mut(out3, 3) };
        out_slice.copy_from_slice(&sb.origin());
        MolrsStatus::Ok
    })
}

/// Copy the per-axis periodic boundary condition flags (3 bools).
///
/// # C signature
///
/// ```c
/// MolrsStatus molrs_box_pbc(MolrsBoxHandle box_handle, bool out3[3]);
/// ```
///
/// # Arguments
///
/// * `box_handle` -- Box handle.
/// * `out3` -- Buffer of at least 3 bools; receives `[px, py, pz]`.
///
/// # Returns
///
/// * `MolrsStatus::Ok` on success.
/// * `MolrsStatus::NullPointer` if `out3` is null.
/// * `MolrsStatus::InvalidBoxHandle` if `box_handle` is stale.
///
/// # Safety
///
/// * `box_handle` must be a live Box handle.
/// * `out3` must point to at least 3 writable `bool` values.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn molrs_box_pbc(box_handle: MolrsBoxHandle, out3: *mut bool) -> MolrsStatus {
    ffi_try!({
        null_check!(out3);
        let registry = lock_registry();
        let sb = get_simbox!(registry, box_handle);
        let out_slice = unsafe { std::slice::from_raw_parts_mut(out3, 3) };
        out_slice.copy_from_slice(&sb.pbc());
        MolrsStatus::Ok
    })
}

/// Compute the cell volume in Angstrom^3.
///
/// The volume is `|det(H)|` where H is the 3x3 cell matrix.
///
/// # C signature
///
/// ```c
/// MolrsStatus molrs_box_volume(MolrsBoxHandle box_handle,
///                                  F* out);
/// ```
///
/// # Arguments
///
/// * `box_handle` -- Box handle.
/// * `out` -- On success, receives the volume in Angstrom^3.
///
/// # Returns
///
/// * `MolrsStatus::Ok` on success.
/// * `MolrsStatus::NullPointer` if `out` is null.
/// * `MolrsStatus::InvalidBoxHandle` if `box_handle` is stale.
///
/// # Safety
///
/// * `box_handle` must be a live Box handle.
/// * `out` must point to a writable float.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn molrs_box_volume(box_handle: MolrsBoxHandle, out: *mut F) -> MolrsStatus {
    ffi_try!({
        null_check!(out);
        let registry = lock_registry();
        let sb = get_simbox!(registry, box_handle);
        unsafe { *out = sb.volume() };
        MolrsStatus::Ok
    })
}

/// Get lattice vector lengths `[|a|, |b|, |c|]` in Angstrom.
///
/// # C signature
///
/// ```c
/// MolrsStatus molrs_box_lengths(MolrsBoxHandle box_handle,
///                                   F out3[3]);
/// ```
///
/// # Arguments
///
/// * `box_handle` -- Box handle.
/// * `out3` -- Buffer of at least 3 floats; receives the lengths.
///
/// # Returns
///
/// * `MolrsStatus::Ok` on success.
/// * `MolrsStatus::NullPointer` if `out3` is null.
/// * `MolrsStatus::InvalidBoxHandle` if `box_handle` is stale.
///
/// # Safety
///
/// * `box_handle` must be a live Box handle.
/// * `out3` must point to at least 3 writable floats.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn molrs_box_lengths(
    box_handle: MolrsBoxHandle,
    out3: *mut F,
) -> MolrsStatus {
    ffi_try!({
        null_check!(out3);
        let registry = lock_registry();
        let sb = get_simbox!(registry, box_handle);
        let lengths = sb.lengths();
        let out_slice = unsafe { std::slice::from_raw_parts_mut(out3, 3) };
        for (i, &val) in lengths.iter().enumerate() {
            out_slice[i] = val;
        }
        MolrsStatus::Ok
    })
}

/// Get off-diagonal tilt factors `[xy, xz, yz]` in Angstrom.
///
/// For an orthorhombic box all tilts are zero.  Non-zero tilts indicate
/// a triclinic cell (e.g. for crystal simulations).
///
/// # C signature
///
/// ```c
/// MolrsStatus molrs_box_tilts(MolrsBoxHandle box_handle,
///                                 F out3[3]);
/// ```
///
/// # Arguments
///
/// * `box_handle` -- Box handle.
/// * `out3` -- Buffer of at least 3 floats; receives `[xy, xz, yz]`.
///
/// # Returns
///
/// * `MolrsStatus::Ok` on success.
/// * `MolrsStatus::NullPointer` if `out3` is null.
/// * `MolrsStatus::InvalidBoxHandle` if `box_handle` is stale.
///
/// # Safety
///
/// * `box_handle` must be a live Box handle.
/// * `out3` must point to at least 3 writable floats.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn molrs_box_tilts(box_handle: MolrsBoxHandle, out3: *mut F) -> MolrsStatus {
    ffi_try!({
        null_check!(out3);
        let registry = lock_registry();
        let sb = get_simbox!(registry, box_handle);
        let tilts = sb.tilts();
        let out_slice = unsafe { std::slice::from_raw_parts_mut(out3, 3) };
        for (i, &val) in tilts.iter().enumerate() {
            out_slice[i] = val;
        }
        MolrsStatus::Ok
    })
}

// ---------------------------------------------------------------------------
// Batch coordinate operations
// ---------------------------------------------------------------------------

/// Wrap Cartesian coordinates into the unit cell (minimum-image).
///
/// Atoms outside the periodic box are folded back in.  Non-periodic
/// axes are unaffected.
///
/// # C signature
///
/// ```c
/// MolrsStatus molrs_box_wrap(MolrsBoxHandle box_handle,
///                                const F* xyz_in,
///                                F* xyz_out,
///                                size_t n_atoms);
/// ```
///
/// # Arguments
///
/// * `box_handle` -- Box handle.
/// * `xyz_in` -- Input coordinates, flat `[x0,y0,z0, x1,y1,z1, ...]`
///   (`n_atoms * 3` floats, in Angstrom).
/// * `xyz_out` -- Output buffer (`n_atoms * 3` floats).  May alias
///   `xyz_in` for in-place wrapping.
/// * `n_atoms` -- Number of atoms.  If 0, the function is a no-op.
///
/// # Returns
///
/// * `MolrsStatus::Ok` on success.
/// * `MolrsStatus::NullPointer` if `xyz_in` or `xyz_out` is null.
/// * `MolrsStatus::InvalidBoxHandle` if `box_handle` is stale.
/// * `MolrsStatus::InvalidArgument` if the input array shape is bad.
///
/// # Safety
///
/// * `box_handle` must be a live Box handle.
/// * `xyz_in` must point to at least `n_atoms * 3` readable floats.
/// * `xyz_out` must point to at least `n_atoms * 3` writable floats.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn molrs_box_wrap(
    box_handle: MolrsBoxHandle,
    xyz_in: *const F,
    xyz_out: *mut F,
    n_atoms: usize,
) -> MolrsStatus {
    ffi_try!({
        null_check!(xyz_in);
        null_check!(xyz_out);
        if n_atoms == 0 {
            return MolrsStatus::Ok;
        }
        let registry = lock_registry();
        let sb = get_simbox!(registry, box_handle);

        let in_slice = unsafe { std::slice::from_raw_parts(xyz_in, n_atoms * 3) };
        let in_arr = match ArrayView2::from_shape((n_atoms, 3), in_slice) {
            Ok(a) => a,
            Err(e) => {
                error::set_last_error(format!("invalid input array: {e}"));
                return MolrsStatus::InvalidArgument;
            }
        };
        // The image shifts `wrap` also reports are not part of this door.
        let (wrapped, _shifts) = sb.wrap(in_arr);
        let out_slice = unsafe { std::slice::from_raw_parts_mut(xyz_out, n_atoms * 3) };
        let wrapped_slice = wrapped
            .as_slice_memory_order()
            .expect("wrapped result is contiguous");
        out_slice.copy_from_slice(wrapped_slice);
        MolrsStatus::Ok
    })
}

/// Row-wise displacement vectors `r2[i] - r1[i]` for `n_pairs` pairs,
/// minimum-imaged when `minimum_image` is set — the C face of molrs's
/// `SimBox::delta`.
///
/// # C signature
///
/// ```c
/// MolrsStatus molrs_box_delta(
///     MolrsBoxHandle box_handle,
///     const F* r1,
///     const F* r2,
///     bool minimum_image,
///     F* dr_out,
///     size_t n_pairs);
/// ```
///
/// # Arguments
///
/// * `box_handle` -- Box handle.
/// * `r1` -- First set of positions, flat `[x0,y0,z0, ...]`
///   (`n_pairs * 3` floats, in Angstrom).
/// * `r2` -- Second set of positions (`n_pairs * 3` floats, in Angstrom).
/// * `minimum_image` -- Apply the minimum-image convention along the
///   periodic axes; `false` returns the plain difference.
/// * `dr_out` -- Output displacements (`n_pairs * 3` floats, in Angstrom).
/// * `n_pairs` -- Number of pairs.  If 0, the function is a no-op.
///
/// # Returns
///
/// * `MolrsStatus::Ok` on success.
/// * `MolrsStatus::NullPointer` if any pointer is null.
/// * `MolrsStatus::InvalidBoxHandle` if `box_handle` is stale.
///
/// # Safety
///
/// * `box_handle` must be a live Box handle.
/// * `r1`, `r2` must each point to at least `n_pairs * 3` readable floats.
/// * `dr_out` must point to at least `n_pairs * 3` writable floats.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn molrs_box_delta(
    box_handle: MolrsBoxHandle,
    r1: *const F,
    r2: *const F,
    minimum_image: bool,
    dr_out: *mut F,
    n_pairs: usize,
) -> MolrsStatus {
    ffi_try!({
        null_check!(r1);
        null_check!(r2);
        null_check!(dr_out);
        if n_pairs == 0 {
            return MolrsStatus::Ok;
        }
        let registry = lock_registry();
        let sb = get_simbox!(registry, box_handle);

        let r1_slice = unsafe { std::slice::from_raw_parts(r1, n_pairs * 3) };
        let r2_slice = unsafe { std::slice::from_raw_parts(r2, n_pairs * 3) };
        let a = ArrayView2::from_shape((n_pairs, 3), r1_slice).expect("n_pairs * 3 floats");
        let b = ArrayView2::from_shape((n_pairs, 3), r2_slice).expect("n_pairs * 3 floats");
        let dr = sb.delta(a, b, minimum_image);
        let out_slice = unsafe { std::slice::from_raw_parts_mut(dr_out, n_pairs * 3) };
        out_slice.copy_from_slice(dr.as_slice().expect("delta result is standard layout"));
        MolrsStatus::Ok
    })
}
