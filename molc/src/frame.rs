//! `extern "C"` functions for Frame lifecycle and access.
//!
//! A **Frame** is the top-level data container in molrs.  It maps string
//! keys (e.g. `"atoms"`, `"bonds"`, `"angles"`) to [`Block`]s, carries
//! an optional [`Box`](molrs::core::Box) for periodic
//! boundary conditions, and stores exact-dtype scalar/fixed-vector metadata.
//!
//! # Typical column layout
//!
//! | Block key  | Column   | C type      | Description                     |
//! |------------|----------|-------------|---------------------------------|
//! | `"atoms"`  | `"atomic_number"` | `uint64_t` | Atomic number Z (0..=118, 0 = wildcard `*`) — the one species field molrs reads |
//! | `"atoms"`  | `"element"` | string   | Optional raw element symbol ("C", "N", ...) as a symbol format spells it; nothing reads it — convert with `Element::fill_atomic_numbers` / `Element::fill_symbols` |
//! | `"atoms"`  | `"x"`    | `F` (double) | Cartesian x coordinate in Angstrom |
//! | `"atoms"`  | `"y"`    | `F` (double) | Cartesian y coordinate in Angstrom |
//! | `"atoms"`  | `"z"`    | `F` (double) | Cartesian z coordinate in Angstrom |
//! | `"atoms"`  | `"mass"` | `F` (double) | Atomic mass in amu             |
//! | `"bonds"`  | `"atomi"` | `uint64_t` | First atom index (0-based)      |
//! | `"bonds"`  | `"atomj"` | `uint64_t` | Second atom index (0-based)     |
//! | `"bonds"`  | `"bond_type"`  | `uint64_t` | 0 unknown, 1 single, 2 double, 3 triple, 4 aromatic |
//! | `"bonds"`  | `"bond_number"`| `uint64_t` | Localized Lewis/Kekulé integer (never fractional) |

use std::ffi::{CStr, CString, c_char};

use molrs::core::Block;
use molrs::core::MetaValue;

use crate::error::{self, MolcStatus, ffi_err_to_status};
use crate::handle::{
    MolcBlockHandle, MolcBoxHandle, MolcFrameHandle, block_handle_to_c, box_key_to_handle,
    frame_id_to_handle, handle_to_box_key, handle_to_frame_id,
};
use crate::handle_registry::lock_registry;
use crate::{ffi_try, null_check};

/// Exact frame-metadata dtype exposed by the C API.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MolcMetaType {
    Bool = 0,
    I32 = 1,
    I64 = 2,
    U32 = 3,
    U64 = 4,
    F64 = 6,
    String = 7,
    Bool3 = 8,
    I32x3 = 9,
    I64x3 = 10,
    U32x3 = 11,
    U64x3 = 12,
    F64x3 = 14,
    F64x6 = 16,
    F64x9 = 18,
}

/// Tagged exact-dtype frame metadata value.
///
/// Only the field selected by `dtype` is read. A string returned by
/// `molc_frame_get_meta` is owned by the caller and must be released with
/// `molc_free_string`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct MolcMetaValue {
    pub dtype: MolcMetaType,
    pub bool_value: bool,
    pub i32_value: i32,
    pub i64_value: i64,
    pub u32_value: u32,
    pub u64_value: u64,
    pub f64_value: f64,
    pub string_value: *mut c_char,
    pub bool3: [bool; 3],
    pub i32x3: [i32; 3],
    pub i64x3: [i64; 3],
    pub u32x3: [u32; 3],
    pub u64x3: [u64; 3],
    pub f64x9: [f64; 9],
}

impl MolcMetaValue {
    fn zeroed(dtype: MolcMetaType) -> Self {
        Self {
            dtype,
            bool_value: false,
            i32_value: 0,
            i64_value: 0,
            u32_value: 0,
            u64_value: 0,
            f64_value: 0.0,
            string_value: std::ptr::null_mut(),
            bool3: [false; 3],
            i32x3: [0; 3],
            i64x3: [0; 3],
            u32x3: [0; 3],
            u64x3: [0; 3],
            f64x9: [0.0; 9],
        }
    }
}

impl Default for MolcMetaValue {
    fn default() -> Self {
        Self::zeroed(MolcMetaType::Bool)
    }
}

/// Create a new, empty frame with no blocks, no Box, and no metadata.
///
/// # C signature
///
/// ```c
/// MolcStatus molc_frame_new(MolcFrameHandle* out);
/// ```
///
/// # Arguments
///
/// * `out` -- On success, receives the new frame handle.
///
/// # Returns
///
/// * `MolcStatus::Ok` on success.
/// * `MolcStatus::NullPointer` if `out` is null.
///
/// # Safety
///
/// `out` must point to a writable `MolcFrameHandle`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn molc_frame_new(out: *mut MolcFrameHandle) -> MolcStatus {
    ffi_try!({
        null_check!(out);
        let mut registry = lock_registry();
        let id = registry.frames.frame_new();
        unsafe { *out = frame_id_to_handle(id) };
        MolcStatus::Ok
    })
}

/// Drop a frame and invalidate all handles that reference it.
///
/// Any [`MolcBlockHandle`] derived from this frame becomes invalid
/// after this call.
///
/// # C signature
///
/// ```c
/// MolcStatus molc_frame_drop(MolcFrameHandle handle);
/// ```
///
/// # Arguments
///
/// * `handle` -- The frame to destroy.
///
/// # Returns
///
/// * `MolcStatus::Ok` on success.
/// * `MolcStatus::InvalidFrameHandle` if `handle` is stale or unknown.
///
/// # Safety
///
/// The caller must not use `handle` or any block handle derived from it
/// after this call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn molc_frame_drop(handle: MolcFrameHandle) -> MolcStatus {
    ffi_try!({
        let mut registry = lock_registry();
        let id = handle_to_frame_id(handle);
        match registry.frames.drop_frame(id) {
            Ok(()) => MolcStatus::Ok,
            Err(e) => ffi_err_to_status(&e),
        }
    })
}

/// Deep-clone a frame, returning a new independent handle.
///
/// The cloned frame is a complete copy of all blocks, columns, Box,
/// and metadata.  Modifications to the clone do not affect the original.
///
/// # C signature
///
/// ```c
/// MolcStatus molc_frame_clone(MolcFrameHandle src,
///                                MolcFrameHandle* out);
/// ```
///
/// # Arguments
///
/// * `src` -- The source frame to clone.
/// * `out` -- On success, receives the handle to the new frame.
///
/// # Returns
///
/// * `MolcStatus::Ok` on success.
/// * `MolcStatus::NullPointer` if `out` is null.
/// * `MolcStatus::InvalidFrameHandle` if `src` is stale or unknown.
///
/// # Safety
///
/// `out` must point to a writable `MolcFrameHandle`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn molc_frame_clone(
    src: MolcFrameHandle,
    out: *mut MolcFrameHandle,
) -> MolcStatus {
    ffi_try!({
        null_check!(out);
        let mut registry = lock_registry();
        let src_id = handle_to_frame_id(src);
        let cloned = match registry.frames.clone_frame(src_id) {
            Ok(f) => f,
            Err(e) => return ffi_err_to_status(&e),
        };
        let new_id = registry.frames.frame_new();
        if let Err(e) = registry.frames.set_frame(new_id, cloned) {
            return ffi_err_to_status(&e);
        }
        unsafe { *out = frame_id_to_handle(new_id) };
        MolcStatus::Ok
    })
}

/// Insert an empty block into a frame under the given interned key.
///
/// If a block with the same key already exists it is replaced.
///
/// # C signature
///
/// ```c
/// MolcStatus molc_frame_set_block(MolcFrameHandle frame,
///                                    uint32_t key_id);
/// ```
///
/// # Arguments
///
/// * `frame` -- Target frame.
/// * `key_id` -- Interned key (see [`molc_intern_key`](crate::molc_intern_key)).
///   Conventional keys: `"atoms"`, `"bonds"`, `"angles"`, `"dihedrals"`.
///
/// # Returns
///
/// * `MolcStatus::Ok` on success.
/// * `MolcStatus::KeyNotFound` if `key_id` has not been interned.
/// * `MolcStatus::InvalidFrameHandle` if `frame` is stale.
///
/// # Safety
///
/// `frame` must be a live frame handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn molc_frame_set_block(frame: MolcFrameHandle, key_id: u32) -> MolcStatus {
    ffi_try!({
        let mut registry = lock_registry();
        let frame_id = handle_to_frame_id(frame);
        let key_str = match registry.key_str(key_id) {
            Some(s) => s.to_owned(),
            None => {
                error::set_last_error(format!("unknown key_id {key_id}"));
                return MolcStatus::KeyNotFound;
            }
        };
        let block = Block::new();
        match registry.frames.set_block(frame_id, &key_str, block) {
            Ok(()) => MolcStatus::Ok,
            Err(e) => ffi_err_to_status(&e),
        }
    })
}

/// Delete a block from a frame by its interned key.
///
/// All [`MolcBlockHandle`]s that reference this block become invalid.
///
/// # C signature
///
/// ```c
/// MolcStatus molc_frame_del_block(MolcFrameHandle frame,
///                                    uint32_t key_id);
/// ```
///
/// # Arguments
///
/// * `frame` -- Target frame.
/// * `key_id` -- Interned key of the block to delete.
///
/// # Returns
///
/// * `MolcStatus::Ok` on success.
/// * `MolcStatus::KeyNotFound` if `key_id` was not interned or the
///   block does not exist.
/// * `MolcStatus::InvalidFrameHandle` if `frame` is stale.
///
/// # Safety
///
/// `frame` must be a live frame handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn molc_frame_del_block(frame: MolcFrameHandle, key_id: u32) -> MolcStatus {
    ffi_try!({
        let mut registry = lock_registry();
        let frame_id = handle_to_frame_id(frame);
        let key_str = match registry.key_str(key_id) {
            Some(s) => s.to_owned(),
            None => {
                error::set_last_error(format!("unknown key_id {key_id}"));
                return MolcStatus::KeyNotFound;
            }
        };
        match registry.frames.del_block(frame_id, &key_str) {
            Ok(()) => MolcStatus::Ok,
            Err(e) => ffi_err_to_status(&e),
        }
    })
}

/// Obtain a block handle for a named block inside a frame.
///
/// The returned handle can be used with `molc_block_*` functions to
/// inspect and modify individual columns.
///
/// # C signature
///
/// ```c
/// MolcStatus molc_frame_get_block(MolcFrameHandle frame,
///                                    uint32_t key_id,
///                                    MolcBlockHandle* out);
/// ```
///
/// # Arguments
///
/// * `frame` -- The owning frame.
/// * `key_id` -- Interned key of the desired block.
/// * `out` -- On success, receives the block handle.
///
/// # Returns
///
/// * `MolcStatus::Ok` on success.
/// * `MolcStatus::NullPointer` if `out` is null.
/// * `MolcStatus::KeyNotFound` if `key_id` was not interned or no
///   block with that key exists in the frame.
/// * `MolcStatus::InvalidFrameHandle` if `frame` is stale.
///
/// # Safety
///
/// * `frame` must be a live frame handle.
/// * `out` must point to a writable `MolcBlockHandle`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn molc_frame_get_block(
    frame: MolcFrameHandle,
    key_id: u32,
    out: *mut MolcBlockHandle,
) -> MolcStatus {
    ffi_try!({
        null_check!(out);
        let registry = lock_registry();
        let frame_id = handle_to_frame_id(frame);
        let key_str = match registry.key_str(key_id) {
            Some(s) => s.to_owned(),
            None => {
                error::set_last_error(format!("unknown key_id {key_id}"));
                return MolcStatus::KeyNotFound;
            }
        };
        let bh = match registry.frames.get_block(frame_id, &key_str) {
            Ok(h) => h,
            Err(e) => return ffi_err_to_status(&e),
        };
        let c_handle = match block_handle_to_c(&bh, &registry.key_to_id) {
            Some(h) => h,
            None => {
                error::set_last_error("failed to intern block key");
                return MolcStatus::InternalError;
            }
        };
        unsafe { *out = c_handle };
        MolcStatus::Ok
    })
}

/// Associate a Box with a frame.
///
/// The Box is cloned from the handle registry into the frame.
/// Changes to the original Box handle after this call do not affect
/// the frame's copy.
///
/// # C signature
///
/// ```c
/// MolcStatus molc_frame_set_box(MolcFrameHandle frame,
///                                     MolcBoxHandle simbox);
/// ```
///
/// # Arguments
///
/// * `frame` -- Target frame.
/// * `simbox` -- A live Box handle to clone into the frame.
///
/// # Returns
///
/// * `MolcStatus::Ok` on success.
/// * `MolcStatus::InvalidFrameHandle` if `frame` is stale.
/// * `MolcStatus::InvalidBoxHandle` if `simbox` is stale.
///
/// # Safety
///
/// Both `frame` and `simbox` must be live handles.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn molc_frame_set_box(
    frame: MolcFrameHandle,
    box_handle: MolcBoxHandle,
) -> MolcStatus {
    ffi_try!({
        let mut registry = lock_registry();
        let frame_id = handle_to_frame_id(frame);
        let sb_key = handle_to_box_key(box_handle);
        let sb = match registry.simboxes.get(sb_key) {
            Some(sb) => sb.clone(),
            None => {
                error::set_last_error("invalid simbox handle");
                return MolcStatus::InvalidBoxHandle;
            }
        };
        match registry.frames.set_frame_box(frame_id, Some(sb)) {
            Ok(()) => MolcStatus::Ok,
            Err(e) => ffi_err_to_status(&e),
        }
    })
}

/// Delete the Box from a frame, leaving it with no periodic cell.
///
/// # C signature
///
/// ```c
/// MolcStatus molc_frame_del_box(MolcFrameHandle frame);
/// ```
///
/// # Arguments
///
/// * `frame` -- Target frame.
///
/// # Returns
///
/// * `MolcStatus::Ok` on success (including if the frame had no Box).
/// * `MolcStatus::InvalidFrameHandle` if `frame` is stale.
///
/// # Safety
///
/// `frame` must be a live frame handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn molc_frame_del_box(frame: MolcFrameHandle) -> MolcStatus {
    ffi_try!({
        let mut registry = lock_registry();
        let frame_id = handle_to_frame_id(frame);
        match registry.frames.set_frame_box(frame_id, None) {
            Ok(()) => MolcStatus::Ok,
            Err(e) => ffi_err_to_status(&e),
        }
    })
}

/// Extract the Box from a frame, cloning it into the box registry.
///
/// A new Box handle is created each time this function is called.
/// The caller is responsible for freeing it with
/// [`molc_box_drop`](crate::molc_box_drop).
///
/// # C signature
///
/// ```c
/// MolcStatus molc_frame_get_box(MolcFrameHandle frame,
///                                     MolcBoxHandle* out);
/// ```
///
/// # Arguments
///
/// * `frame` -- Source frame.
/// * `out` -- On success, receives a new Box handle.
///
/// # Returns
///
/// * `MolcStatus::Ok` on success.
/// * `MolcStatus::NullPointer` if `out` is null.
/// * `MolcStatus::KeyNotFound` if the frame has no Box.
/// * `MolcStatus::InvalidFrameHandle` if `frame` is stale.
///
/// # Safety
///
/// * `frame` must be a live frame handle.
/// * `out` must point to a writable `MolcBoxHandle`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn molc_frame_get_box(
    frame: MolcFrameHandle,
    out: *mut MolcBoxHandle,
) -> MolcStatus {
    ffi_try!({
        null_check!(out);
        let mut registry = lock_registry();
        let frame_id = handle_to_frame_id(frame);
        let sb_clone = match registry.frames.with_frame(frame_id, |f| f.simbox.clone()) {
            Ok(Some(sb)) => sb,
            Ok(None) => {
                error::set_last_error("frame has no simbox");
                return MolcStatus::KeyNotFound;
            }
            Err(e) => return ffi_err_to_status(&e),
        };
        let key = registry.simboxes.insert(sb_clone);
        unsafe { *out = box_key_to_handle(key) };
        MolcStatus::Ok
    })
}

fn meta_to_c(value: &MetaValue) -> Result<MolcMetaValue, MolcStatus> {
    macro_rules! base {
        ($dtype:ident) => {
            MolcMetaValue::zeroed(MolcMetaType::$dtype)
        };
    }
    Ok(match value {
        MetaValue::Bool(v) => {
            let mut out = base!(Bool);
            out.bool_value = *v;
            out
        }
        MetaValue::I32(v) => {
            let mut out = base!(I32);
            out.i32_value = *v;
            out
        }
        MetaValue::I64(v) => {
            let mut out = base!(I64);
            out.i64_value = *v;
            out
        }
        MetaValue::U32(v) => {
            let mut out = base!(U32);
            out.u32_value = *v;
            out
        }
        MetaValue::U64(v) => {
            let mut out = base!(U64);
            out.u64_value = *v;
            out
        }
        MetaValue::F64(v) => {
            let mut out = base!(F64);
            out.f64_value = *v;
            out
        }
        MetaValue::String(v) => {
            let mut out = base!(String);
            out.string_value = CString::new(v.as_str())
                .map_err(|_| {
                    error::set_last_error("string metadata contains an interior NUL byte");
                    MolcStatus::InvalidArgument
                })?
                .into_raw();
            out
        }
        MetaValue::Bool3(v) => {
            let mut out = base!(Bool3);
            out.bool3 = *v;
            out
        }
        MetaValue::I32x3(v) => {
            let mut out = base!(I32x3);
            out.i32x3 = *v;
            out
        }
        MetaValue::I64x3(v) => {
            let mut out = base!(I64x3);
            out.i64x3 = *v;
            out
        }
        MetaValue::U32x3(v) => {
            let mut out = base!(U32x3);
            out.u32x3 = *v;
            out
        }
        MetaValue::U64x3(v) => {
            let mut out = base!(U64x3);
            out.u64x3 = *v;
            out
        }
        MetaValue::F64x3(v) => {
            let mut out = base!(F64x3);
            out.f64x9[..3].copy_from_slice(v);
            out
        }
        MetaValue::F64x6(v) => {
            let mut out = base!(F64x6);
            out.f64x9[..6].copy_from_slice(v);
            out
        }
        MetaValue::F64x9(v) => {
            let mut out = base!(F64x9);
            out.f64x9 = *v;
            out
        }
        MetaValue::Json(v) => {
            let mut out = base!(String);
            let owned =
                CString::new(v.to_string()).unwrap_or_else(|_| CString::new("null").unwrap());
            out.string_value = owned.into_raw();
            out
        }
    })
}

unsafe fn meta_from_c(value: &MolcMetaValue) -> Result<MetaValue, MolcStatus> {
    Ok(match value.dtype {
        MolcMetaType::Bool => MetaValue::Bool(value.bool_value),
        MolcMetaType::I32 => MetaValue::I32(value.i32_value),
        MolcMetaType::I64 => MetaValue::I64(value.i64_value),
        MolcMetaType::U32 => MetaValue::U32(value.u32_value),
        MolcMetaType::U64 => MetaValue::U64(value.u64_value),
        MolcMetaType::F64 => MetaValue::F64(value.f64_value),
        MolcMetaType::String => {
            if value.string_value.is_null() {
                error::set_last_error("string metadata has a null payload");
                return Err(MolcStatus::NullPointer);
            }
            let text = unsafe { CStr::from_ptr(value.string_value) }
                .to_str()
                .map_err(|_| {
                    error::set_last_error("string metadata is not valid UTF-8");
                    MolcStatus::Utf8Error
                })?;
            MetaValue::String(text.to_owned())
        }
        MolcMetaType::Bool3 => MetaValue::Bool3(value.bool3),
        MolcMetaType::I32x3 => MetaValue::I32x3(value.i32x3),
        MolcMetaType::I64x3 => MetaValue::I64x3(value.i64x3),
        MolcMetaType::U32x3 => MetaValue::U32x3(value.u32x3),
        MolcMetaType::U64x3 => MetaValue::U64x3(value.u64x3),
        MolcMetaType::F64x3 => MetaValue::F64x3(value.f64x9[..3].try_into().unwrap()),
        MolcMetaType::F64x6 => MetaValue::F64x6(value.f64x9[..6].try_into().unwrap()),
        MolcMetaType::F64x9 => MetaValue::F64x9(value.f64x9),
    })
}

/// Insert or replace one exact-dtype metadata entry.
///
/// # C signature
///
/// ```c
/// MolcStatus molc_frame_set_meta(MolcFrameHandle frame, const char* key,
///                                  const MolcMetaValue* value);
/// ```
///
/// Block handles on the frame stay valid: only the metadata changes.
///
/// # Returns
///
/// * `MolcStatus::Ok` on success.
/// * `MolcStatus::NullPointer` if `key` or `value` (or a string payload) is null.
/// * `MolcStatus::InvalidArgument` if `key` is empty.
/// * `MolcStatus::Utf8Error` if `key` or a string payload is not UTF-8.
/// * `MolcStatus::InvalidFrameHandle` if `frame` is stale.
///
/// # Safety
///
/// `key` must be a NUL-terminated C string and `value` must point to a
/// readable `MolcMetaValue`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn molc_frame_set_meta(
    frame: MolcFrameHandle,
    key: *const c_char,
    value: *const MolcMetaValue,
) -> MolcStatus {
    ffi_try!({
        null_check!(key);
        null_check!(value);
        let key = match unsafe { CStr::from_ptr(key) }.to_str() {
            Ok(key) if !key.is_empty() => key.to_owned(),
            Ok(_) => {
                error::set_last_error("metadata key must not be empty");
                return MolcStatus::InvalidArgument;
            }
            Err(_) => {
                error::set_last_error("metadata key is not valid UTF-8");
                return MolcStatus::Utf8Error;
            }
        };
        let value = match unsafe { meta_from_c(&*value) } {
            Ok(value) => value,
            Err(status) => return status,
        };
        let mut registry = lock_registry();
        match registry
            .frames
            .with_frame_meta_mut(handle_to_frame_id(frame), |meta| {
                meta.insert(key, value);
            }) {
            Ok(()) => MolcStatus::Ok,
            Err(e) => ffi_err_to_status(&e),
        }
    })
}

/// Read one exact-dtype metadata entry.
///
/// # C signature
///
/// ```c
/// MolcStatus molc_frame_get_meta(MolcFrameHandle frame, const char* key,
///                                  MolcMetaValue* out);
/// ```
///
/// A string value (`MOLC_META_TYPE_STRING`; a JSON value is returned as its
/// text) is owned by the caller: free `out->string_value` with
/// `molc_free_string`.
///
/// # Returns
///
/// * `MolcStatus::Ok` on success.
/// * `MolcStatus::KeyNotFound` if the frame has no entry `key`.
/// * `MolcStatus::NullPointer`, `Utf8Error`, `InvalidFrameHandle` as usual.
///
/// # Safety
///
/// `key` must be a NUL-terminated C string and `out` a writable
/// `MolcMetaValue`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn molc_frame_get_meta(
    frame: MolcFrameHandle,
    key: *const c_char,
    out: *mut MolcMetaValue,
) -> MolcStatus {
    ffi_try!({
        null_check!(key);
        null_check!(out);
        let key = match unsafe { CStr::from_ptr(key) }.to_str() {
            Ok(key) => key,
            Err(_) => {
                error::set_last_error("metadata key is not valid UTF-8");
                return MolcStatus::Utf8Error;
            }
        };
        let registry = lock_registry();
        let frame = match registry.frames.clone_frame(handle_to_frame_id(frame)) {
            Ok(frame) => frame,
            Err(e) => return ffi_err_to_status(&e),
        };
        let Some(value) = frame.meta.get(key) else {
            error::set_last_error(format!("metadata key '{key}' not found"));
            return MolcStatus::KeyNotFound;
        };
        let value = match meta_to_c(value) {
            Ok(value) => value,
            Err(status) => return status,
        };
        unsafe { *out = value };
        MolcStatus::Ok
    })
}

/// Number of metadata entries; with [`molc_frame_meta_key`] this lists the
/// keys (the C shape of `meta_keys`).
///
/// # C signature
///
/// ```c
/// MolcStatus molc_frame_n_meta(MolcFrameHandle frame, size_t* out);
/// ```
///
/// # Safety
///
/// `out` must point to a writable `size_t`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn molc_frame_n_meta(frame: MolcFrameHandle, out: *mut usize) -> MolcStatus {
    ffi_try!({
        null_check!(out);
        let registry = lock_registry();
        let frame = match registry.frames.clone_frame(handle_to_frame_id(frame)) {
            Ok(frame) => frame,
            Err(e) => return ffi_err_to_status(&e),
        };
        unsafe {
            *out = frame.meta.len();
        }
        MolcStatus::Ok
    })
}

/// The metadata key at `index` (`0 <= index < n_meta`), in insertion order.
///
/// # C signature
///
/// ```c
/// MolcStatus molc_frame_meta_key(MolcFrameHandle frame, size_t index,
///                                  char** out);
/// ```
///
/// The caller owns `*out`: free it with `molc_free_string`. An index past
/// the end is `InvalidArgument`.
///
/// # Safety
///
/// `out` must point to a writable `char*`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn molc_frame_meta_key(
    frame: MolcFrameHandle,
    index: usize,
    out: *mut *mut c_char,
) -> MolcStatus {
    ffi_try!({
        null_check!(out);
        let registry = lock_registry();
        let frame = match registry.frames.clone_frame(handle_to_frame_id(frame)) {
            Ok(frame) => frame,
            Err(e) => return ffi_err_to_status(&e),
        };
        let keys: Vec<_> = frame.meta.keys().collect();
        let Some(key) = keys.get(index) else {
            error::set_last_error(format!("metadata index {index} out of range"));
            return MolcStatus::InvalidArgument;
        };
        let key = match CString::new(key.as_str()) {
            Ok(key) => key,
            Err(_) => {
                error::set_last_error("metadata key contains an interior NUL byte");
                return MolcStatus::InvalidArgument;
            }
        };
        unsafe { *out = key.into_raw() };
        MolcStatus::Ok
    })
}
