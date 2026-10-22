#![allow(clippy::missing_safety_doc)]
//! C API for molrs -- stable ABI for C and C++ integration.
//!
//! This crate exposes a flat, handle-based C API for creating and
//! manipulating molecular simulation data (frames, blocks, simulation
//! boxes, and force fields).
//!
//! # Design
//!
//! * **Handle-based** -- all objects live in one global, mutex-protected
//!   handle registry.  C callers receive opaque `MolcFrameHandle`,
//!   `MolcBlockHandle`, `MolcBoxHandle`, `MolcForceFieldHandle`, or
//!   `MolcRegionHandle` values (plain `repr(C)` structs that fit in two
//!   machine words).
//! * **Status codes** -- every function returns [`MolcStatus`].  On
//!   failure the last error message is stored in a thread-local buffer
//!   and retrieved via [`molc_last_error`].
//! * **Panic safety** -- every `extern "C"` body is wrapped in
//!   `catch_unwind` so Rust panics never unwind across the FFI boundary.
//!
//! # Float / integer widths
//!
//! Widths are fixed. The header declares `typedef double F;`.
//!
//! | Rust | C type     | Column dtype                 |
//! |------|------------|------------------------------|
//! | `F`  | `F` (`double`) | `MOLC_D_TYPE_FLOAT` (f64) |
//! | `I`  | `int32_t`  | `MOLC_D_TYPE_INT` (i32)     |
//! | `Idx`| `uint64_t` | `MOLC_D_TYPE_UINT` (u64)    |
//!
//! Each `MolcDType` constant is `MOLC_D_TYPE_` + the upper-cased core
//! dtype name (`DType::name()`: `float`, `int`, `uint`, `i64`, `u8`, …).
//! Every fixed-width dtype is written with `molc_block_set` (the dtype is
//! an argument) and read with `molc_block_get`, which reports the dtype as
//! a `MolcDType`.
//!
//! # Typical C usage
//!
//! ```c
//! #include "molc.h"
//!
//! molc_init();
//!
//! MolcFrameHandle frame;
//! molc_frame_new(&frame);
//!
//! // ... populate frame with blocks, columns, simbox ...
//!
//! molc_frame_drop(frame);
//! molc_shutdown();
//! ```
//!
//! # Safety
//!
//! All `extern "C"` functions are unsafe because they accept raw pointers
//! from C callers.  See per-function documentation for pointer validity
//! and lifetime requirements.

// The modules are private and every C-facing item is re-exported flat here:
// one Rust path per symbol, `molc::<name>`, as C spells it.
mod block;
mod error;
mod forcefield;
mod frame;
mod handle;
mod handle_registry;
mod region;
mod schema;
mod simbox;
mod smiles;

pub use block::*;
pub use error::{MolcDType, MolcStatus};
pub use forcefield::*;
pub use frame::*;
pub use handle::{
    MolcBlockHandle, MolcBoxHandle, MolcForceFieldHandle, MolcFrameHandle, MolcRegionHandle,
};
pub use region::*;
pub use schema::*;
pub use simbox::*;
pub use smiles::*;

use std::ffi::{CStr, CString, c_char};

use handle_registry::lock_registry;

// The float scalar is molrs's own `molrs::op::F` (always `f64`); the C
// header spells it `typedef double F` (cbindgen.toml `after_includes`, since
// cbindgen does not parse dependencies).

// ---------------------------------------------------------------------------
// Internal helper: catch panics at the FFI boundary
// ---------------------------------------------------------------------------

macro_rules! ffi_try {
    ($body:expr) => {{
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| $body)) {
            Ok(status) => status,
            Err(_) => {
                error::set_last_error("internal panic in FFI call");
                MolcStatus::InternalError
            }
        }
    }};
}
pub(crate) use ffi_try;

/// Null-pointer check helper. Returns `NullPointer` status if null.
macro_rules! null_check {
    ($ptr:expr) => {
        if $ptr.is_null() {
            error::set_last_error(concat!("null pointer: ", stringify!($ptr)));
            return MolcStatus::NullPointer;
        }
    };
}
pub(crate) use null_check;

// ---------------------------------------------------------------------------
// Lifecycle & Utilities
// ---------------------------------------------------------------------------

/// Initialize the handle registry.
///
/// Safe to call multiple times -- the second and subsequent calls are
/// no-ops.  Must be called before any other `molrs_*` function.
///
/// # C signature
///
/// ```c
/// void molc_init(void);
/// ```
///
/// # Safety
///
/// No pointer arguments.  This function only initialises the handle
/// registry and cannot violate memory safety.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn molc_init() {
    // Force lazy initialization of the handle registry.
    drop(lock_registry());
}

/// Report the `molcrafts-molrs` core version compiled into this library.
///
/// Returns a pointer to a static null-terminated UTF-8 string, e.g.
/// `"0.17.0"`. Informational — this identifies the exact molrs release for
/// diagnostics.
///
/// # C signature
///
/// ```c
/// const char* molc_version(void);
/// ```
///
/// # Safety
///
/// The caller must not write through the returned pointer. The pointer is
/// valid for the lifetime of the process.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn molc_version() -> *const c_char {
    static VERSION: std::sync::OnceLock<CString> = std::sync::OnceLock::new();
    VERSION
        .get_or_init(|| CString::new(env!("CARGO_PKG_VERSION")).expect("version has no NUL"))
        .as_ptr()
}

/// Destroy all objects and reset the handle registry.
///
/// Every handle obtained before this call becomes invalid.
/// It is safe (but unnecessary) to call [`molc_init`] again afterwards.
///
/// # C signature
///
/// ```c
/// void molc_shutdown(void);
/// ```
///
/// # Safety
///
/// After this call, any previously obtained handle (frame, block,
/// box, force field, region) is dangling.  Using a stale handle will return
/// `MolcStatus::Invalid*Handle`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn molc_shutdown() {
    let mut registry = lock_registry();
    registry.clear();
}

/// Retrieve the last error message for the calling thread.
///
/// Returns a pointer to a null-terminated UTF-8 string describing the
/// most recent error.  If no error has occurred, an empty string (`""`)
/// is returned (never a null pointer).
///
/// # C signature
///
/// ```c
/// const char* molc_last_error(void);
/// ```
///
/// # Lifetime
///
/// The returned pointer is valid until the **next** `molrs_*` call that
/// sets an error **on the same thread**.  Copy the string immediately if
/// you need to keep it.
///
/// # Safety
///
/// The caller must not write through the returned pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn molc_last_error() -> *const c_char {
    error::last_error_ptr()
}

/// Intern a key string, returning a compact integer identifier.
///
/// Interning avoids repeated string-to-key lookups when the same block
/// or column name is used across many calls.  The same input string
/// always returns the same `key_id` (idempotent).
///
/// # C signature
///
/// ```c
/// MolcStatus molc_intern_key(const char* key, uint32_t* out_key_id);
/// ```
///
/// # Arguments
///
/// * `key` -- Null-terminated UTF-8 string (e.g. `"atoms"`, `"x"`).
/// * `out_key_id` -- On success, receives the interned key identifier.
///
/// # Returns
///
/// * `MolcStatus::Ok` on success.
/// * `MolcStatus::NullPointer` if `key` or `out_key_id` is null.
/// * `MolcStatus::Utf8Error` if `key` is not valid UTF-8.
///
/// # Safety
///
/// * `key` must be a valid, null-terminated C string.
/// * `out_key_id` must point to a writable `uint32_t`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn molc_intern_key(key: *const c_char, out_key_id: *mut u32) -> MolcStatus {
    ffi_try!({
        null_check!(key);
        null_check!(out_key_id);
        let c_str = unsafe { CStr::from_ptr(key) };
        let key_str = match c_str.to_str() {
            Ok(s) => s,
            Err(_) => {
                error::set_last_error("key is not valid UTF-8");
                return MolcStatus::Utf8Error;
            }
        };
        let mut registry = lock_registry();
        let id = registry.intern(key_str);
        unsafe { *out_key_id = id };
        MolcStatus::Ok
    })
}

/// Look up the string name for a previously interned `key_id`.
///
/// # C signature
///
/// ```c
/// const char* molc_key_name(uint32_t key_id);
/// ```
///
/// # Arguments
///
/// * `key_id` -- An identifier obtained from [`molc_intern_key`].
///
/// # Returns
///
/// A pointer to a null-terminated UTF-8 string, or `NULL` if `key_id`
/// is unknown.
///
/// # Lifetime
///
/// The returned pointer is valid until [`molc_shutdown`] is called.
///
/// # Safety
///
/// The caller must not write through or free the returned pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn molc_key_name(key_id: u32) -> *const c_char {
    let registry = lock_registry();
    match registry.interned_keys.get(key_id as usize) {
        Some(cstr) => cstr.as_ptr(),
        None => std::ptr::null(),
    }
}

/// Free a string that was allocated by the C API.
///
/// Several functions (e.g. [`molc_forcefield_to_json`],
/// [`molc_frame_get_meta`])
/// return heap-allocated C strings that the caller owns.  Pass those
/// pointers to this function when they are no longer needed.
///
/// Passing `NULL` is a safe no-op.
///
/// # C signature
///
/// ```c
/// void molc_free_string(char* s);
/// ```
///
/// # Arguments
///
/// * `s` -- A string pointer previously returned by a `molrs_*` function,
///   or `NULL`.
///
/// # Safety
///
/// * `s` must have been allocated by this library (via `CString::into_raw`).
/// * `s` must not be used after this call.
/// * Double-free is undefined behaviour.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn molc_free_string(s: *mut c_char) {
    if !s.is_null() {
        unsafe {
            drop(CString::from_raw(s));
        }
    }
}
