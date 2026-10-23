//! `extern "C"` functions for ForceField operations.
//!
//! A **ForceField** is a collection of interaction styles (atom, bond,
//! angle, dihedral, improper, pair, cmap) and per-type parameter sets.
//! This module exposes functions to build a force field programmatically
//! from C, serialize/deserialize it as JSON, and query its contents.
//!
//! The JSON is molrs's one force-field serialization: the core `forcefield`
//! record section (`ForceFieldSection::from_forcefield` / `ForceFieldSection::to_forcefield`) in
//! its serde form, `{"document": {…}, "tables": {<block>: Block}}` — the
//! same section an `*.mrec` record stores. The C API has no format of its
//! own.
//!
//! # Typical workflow
//!
//! ```c
//! MolcForceFieldHandle ff;
//! molc_forcefield_new("my_ff", &ff);
//!
//! // Define a pair style with a global cutoff parameter
//! const char* pk[] = {"cutoff"};
//! double      pv[] = {12.0};
//! molc_forcefield_def_style(ff, "pair", "lj/cut", pk, pv, 1);
//!
//! // Define a type under that style: its name, then its endpoints
//! const char* ow[] = {"OW"};
//! const char* tk[] = {"epsilon", "sigma"};
//! double      tv[] = {0.1553, 3.166};
//! molc_forcefield_def_type(ff, "pair", "lj/cut", "OW", ow, 1, tk, tv, 2);
//!
//! // Serialize to JSON (the core forcefield section) for storage
//! char*  json;
//! size_t json_len;
//! molc_forcefield_to_json(ff, &json, &json_len);
//! // ... write json to file ...
//! molc_free_string(json);
//!
//! molc_forcefield_drop(ff);
//! ```
//!
//! # Parameter values
//!
//! All parameter values are `double` (`f64`) regardless of the float
//! precision feature, because force field parameters require full
//! precision for numerical stability.
//!
//! The `molc_forcefield_def_*` calls take numeric parameters only. String
//! parameters (a pair style's `mixing`, an atom type's `element`) reach a
//! force field from C through [`molc_forcefield_from_json`], whose section carries
//! both.

use std::ffi::{CStr, CString, c_char};

use molrs::ff::forcefield::{DefError, ForceField};
use molrs::ff::ir::Params;
use molrs::io::mrec::ForceFieldSection;

use crate::error::{self, MolcStatus};
use crate::handle::{MolcForceFieldHandle, forcefield_key_to_handle, handle_to_forcefield_key};
use crate::handle_registry::lock_registry;
use crate::{ffi_try, null_check};

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

/// Create a new, empty ForceField with the given name.
///
/// # C signature
///
/// ```c
/// MolcStatus molc_forcefield_new(const char* name,
///                           MolcForceFieldHandle* out);
/// ```
///
/// # Arguments
///
/// * `name` -- Null-terminated UTF-8 force field name (e.g. `"OPLS"`,
///   `"MMFF94"`).
/// * `out` -- On success, receives the new ForceField handle.
///
/// # Returns
///
/// * `MolcStatus::Ok` on success.
/// * `MolcStatus::NullPointer` if `name` or `out` is null.
/// * `MolcStatus::Utf8Error` if `name` is not valid UTF-8.
///
/// # Safety
///
/// * `name` must be a valid, null-terminated C string.
/// * `out` must point to a writable `MolcForceFieldHandle`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn molc_forcefield_new(
    name: *const c_char,
    out: *mut MolcForceFieldHandle,
) -> MolcStatus {
    ffi_try!({
        null_check!(name);
        null_check!(out);
        let name_str = match unsafe { CStr::from_ptr(name) }.to_str() {
            Ok(s) => s,
            Err(_) => {
                error::set_last_error("name is not valid UTF-8");
                return MolcStatus::Utf8Error;
            }
        };
        let ff = ForceField::new(name_str);
        let mut registry = lock_registry();
        let key = registry.forcefields.insert(ff);
        unsafe { *out = forcefield_key_to_handle(key) };
        MolcStatus::Ok
    })
}

/// Destroy a ForceField and invalidate its handle.
///
/// # C signature
///
/// ```c
/// MolcStatus molc_forcefield_drop(MolcForceFieldHandle handle);
/// ```
///
/// # Arguments
///
/// * `handle` -- The ForceField to destroy.
///
/// # Returns
///
/// * `MolcStatus::Ok` on success.
/// * `MolcStatus::InvalidForceFieldHandle` if `handle` is stale.
///
/// # Safety
///
/// The caller must not use `handle` after this call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn molc_forcefield_drop(handle: MolcForceFieldHandle) -> MolcStatus {
    ffi_try!({
        let mut registry = lock_registry();
        let key = handle_to_forcefield_key(handle);
        match registry.forcefields.remove(key) {
            Some(_) => MolcStatus::Ok,
            None => {
                error::set_last_error("invalid forcefield handle");
                MolcStatus::InvalidForceFieldHandle
            }
        }
    })
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Extract a ForceField ref from the registry.
macro_rules! get_forcefield {
    ($registry:expr, $handle:expr) => {
        match $registry.forcefields.get(handle_to_forcefield_key($handle)) {
            Some(ff) => ff,
            None => {
                error::set_last_error("invalid forcefield handle");
                return MolcStatus::InvalidForceFieldHandle;
            }
        }
    };
}

macro_rules! get_forcefield_mut {
    ($registry:expr, $handle:expr) => {
        match $registry
            .forcefields
            .get_mut(handle_to_forcefield_key($handle))
        {
            Some(ff) => ff,
            None => {
                error::set_last_error("invalid forcefield handle");
                return MolcStatus::InvalidForceFieldHandle;
            }
        }
    };
}

/// Parse the C key / value arrays into numeric [`Params`].
unsafe fn parse_params(
    param_keys: *const *const c_char,
    param_values: *const f64,
    n_params: usize,
) -> Result<Params, MolcStatus> {
    if n_params == 0 {
        return Ok(Params::new());
    }
    if param_keys.is_null() || param_values.is_null() {
        error::set_last_error("null param_keys or param_values");
        return Err(MolcStatus::NullPointer);
    }
    let keys_slice = unsafe { std::slice::from_raw_parts(param_keys, n_params) };
    let vals_slice = unsafe { std::slice::from_raw_parts(param_values, n_params) };
    let mut params = Params::new();
    for i in 0..n_params {
        if keys_slice[i].is_null() {
            error::set_last_error(format!("param_keys[{i}] is null"));
            return Err(MolcStatus::NullPointer);
        }
        let key = match unsafe { CStr::from_ptr(keys_slice[i]) }.to_str() {
            Ok(s) => s,
            Err(_) => {
                error::set_last_error(format!("param_keys[{i}] is not valid UTF-8"));
                return Err(MolcStatus::Utf8Error);
            }
        };
        params.set(key, vals_slice[i]);
    }
    Ok(params)
}

// ---------------------------------------------------------------------------
// Definition: def_style / def_type
// ---------------------------------------------------------------------------

/// Read a required C string argument as UTF-8.
unsafe fn arg_str<'a>(ptr: *const c_char, what: &str) -> Result<&'a str, MolcStatus> {
    if ptr.is_null() {
        error::set_last_error(format!("{what} is null"));
        return Err(MolcStatus::NullPointer);
    }
    unsafe { CStr::from_ptr(ptr) }.to_str().map_err(|_| {
        error::set_last_error(format!("{what} is not valid UTF-8"));
        MolcStatus::Utf8Error
    })
}

/// Parse an array of `n` C strings (type endpoints).
unsafe fn parse_strings<'a>(
    ptrs: *const *const c_char,
    n: usize,
    what: &str,
) -> Result<Vec<&'a str>, MolcStatus> {
    if n == 0 {
        return Ok(Vec::new());
    }
    if ptrs.is_null() {
        error::set_last_error(format!("{what} is null"));
        return Err(MolcStatus::NullPointer);
    }
    let slice = unsafe { std::slice::from_raw_parts(ptrs, n) };
    slice
        .iter()
        .enumerate()
        .map(|(i, &p)| unsafe { arg_str(p, &format!("{what}[{i}]")) })
        .collect()
}

/// Unwrap a `Result<T, MolcStatus>`, returning the status from the calling
/// `extern "C"` function on error.
macro_rules! or_return {
    ($e:expr) => {
        match $e {
            Ok(v) => v,
            Err(status) => return status,
        }
    };
}

/// Map a definition error to `InvalidArgument`, recording its message.
fn invalid_argument(e: DefError) -> MolcStatus {
    error::set_last_error(e.to_string());
    MolcStatus::InvalidArgument
}

/// Define the `category` style named `name` on a ForceField.
///
/// A style is identified by `(category, name)`: re-defining it with equal
/// params keeps the existing style; with different params it is
/// `InvalidArgument` and the first definition is kept.
///
/// # C signature
///
/// ```c
/// MolcStatus molc_forcefield_def_style(MolcForceFieldHandle ff,
///                                 const char* category,
///                                 const char* name,
///                                 const char** param_keys,
///                                 const double* param_values,
///                                 size_t n_params);
/// ```
///
/// # Arguments
///
/// * `ff` -- ForceField handle.
/// * `category` -- One of `"atom"`, `"bond"`, `"angle"`, `"dihedral"`,
///   `"improper"`, `"pair"`, `"cmap"`, or any other category the force-field
///   IR registry declares (`"drude"`, a registered custom category) or the
///   force field already holds.
/// * `name` -- Style name (e.g. `"harmonic"`, `"lj/cut"`).
/// * `param_keys` -- Array of `n_params` style-level parameter names
///   (e.g. `"cutoff"`). May be `NULL` if `n_params == 0`.
/// * `param_values` -- Array of `n_params` `double` values. May be `NULL`
///   if `n_params == 0`.
/// * `n_params` -- Number of style-level parameters.
///
/// Parameters are numeric only; string params (e.g. `mixing`) reach a
/// force field from C through [`molc_forcefield_from_json`].
///
/// # Returns
///
/// * `MolcStatus::Ok` on success.
/// * `MolcStatus::NullPointer` if a required pointer is null.
/// * `MolcStatus::Utf8Error` if any string is not valid UTF-8.
/// * `MolcStatus::InvalidArgument` if `category` is none of the above, or
///   the style is already defined with different params.
/// * `MolcStatus::InvalidForceFieldHandle` if `ff` is stale.
///
/// # Safety
///
/// * `ff` must be a live ForceField handle.
/// * `category` and `name` must be valid, null-terminated C strings.
/// * `param_keys` (if non-null) must point to `n_params` valid C string
///   pointers; `param_values` (if non-null) to `n_params` doubles.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn molc_forcefield_def_style(
    ff: MolcForceFieldHandle,
    category: *const c_char,
    name: *const c_char,
    param_keys: *const *const c_char,
    param_values: *const f64,
    n_params: usize,
) -> MolcStatus {
    ffi_try!({
        let category = or_return!(unsafe { arg_str(category, "category") });
        let name = or_return!(unsafe { arg_str(name, "name") });
        let params = or_return!(unsafe { parse_params(param_keys, param_values, n_params) });
        let mut registry = lock_registry();
        let ff = get_forcefield_mut!(registry, ff);
        match ff.def_style(category, name, params) {
            Ok(_) => MolcStatus::Ok,
            Err(e) => invalid_argument(e),
        }
    })
}

/// Declare the unit system the force field's numbers are in (`"real"`,
/// `"metal"`, …, as LAMMPS names them). A force field states its units
/// before it is serialized ([`molc_forcefield_to_json`]); they are never
/// assumed.
///
/// # C signature
///
/// ```c
/// MolcStatus molc_forcefield_set_units(MolcForceFieldHandle ff, const char* units);
/// ```
///
/// # Returns
///
/// * `MolcStatus::Ok` on success.
/// * `MolcStatus::NullPointer` if `units` is null.
/// * `MolcStatus::Utf8Error` if `units` is not valid UTF-8.
/// * `MolcStatus::InvalidForceFieldHandle` if `ff` is stale.
///
/// # Safety
///
/// * `ff` must be a live ForceField handle.
/// * `units` must be a valid, null-terminated C string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn molc_forcefield_set_units(
    ff: MolcForceFieldHandle,
    units: *const c_char,
) -> MolcStatus {
    ffi_try!({
        let units = or_return!(unsafe { arg_str(units, "units") });
        let mut registry = lock_registry();
        let ff = get_forcefield_mut!(registry, ff);
        ff.set_units(units);
        MolcStatus::Ok
    })
}

/// Define a type on an existing style: its name and its endpoints.
///
/// The style `(style_category, style_name)` must already exist (see
/// [`molc_forcefield_def_style`]); it is never created implicitly. The name is an
/// opaque identifier stored verbatim and never split into endpoints (`CT-OH`
/// and MMFF's `0_1_5` alike); the endpoints are the ones given.
///
/// # C signature
///
/// ```c
/// MolcStatus molc_forcefield_def_type(MolcForceFieldHandle ff,
///                                const char* style_category,
///                                const char* style_name,
///                                const char* type_name,
///                                const char** endpoints,
///                                size_t n_endpoints,
///                                const char** param_keys,
///                                const double* param_values,
///                                size_t n_params);
/// ```
///
/// # Arguments
///
/// * `ff` -- ForceField handle.
/// * `style_category` / `style_name` -- An existing style.
/// * `type_name` -- The type name, stored as given.
/// * `endpoints` -- Array of `n_endpoints` atom-type names. The count must
///   match the category: atom 0, pair 1 (a self pair) or 2, bond 2, angle 3,
///   dihedral and improper 4, cmap 5, any other category its arity. May be
///   `NULL` if `n_endpoints == 0`.
/// * `param_keys` / `param_values` / `n_params` -- Numeric type parameters;
///   the arrays may be `NULL` if `n_params == 0`.
///
/// # Returns
///
/// * `MolcStatus::Ok` on success.
/// * `MolcStatus::NullPointer` if a required pointer is null.
/// * `MolcStatus::Utf8Error` if any string is not valid UTF-8.
/// * `MolcStatus::InvalidArgument` if the style does not exist, the
///   endpoint count does not match the category, or the style already
///   defines `type_name` with other endpoints or params (an identical
///   re-definition is `Ok` and changes nothing).
/// * `MolcStatus::InvalidForceFieldHandle` if `ff` is stale.
///
/// # Safety
///
/// * `ff` must be a live ForceField handle.
/// * All string arguments must be valid, null-terminated C strings.
/// * `endpoints` (if non-null) must point to `n_endpoints` valid C string
///   pointers.
/// * `param_keys` (if non-null) must point to `n_params` valid C string
///   pointers; `param_values` (if non-null) to `n_params` doubles.
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn molc_forcefield_def_type(
    ff: MolcForceFieldHandle,
    style_category: *const c_char,
    style_name: *const c_char,
    type_name: *const c_char,
    endpoints: *const *const c_char,
    n_endpoints: usize,
    param_keys: *const *const c_char,
    param_values: *const f64,
    n_params: usize,
) -> MolcStatus {
    ffi_try!({
        let category = or_return!(unsafe { arg_str(style_category, "style_category") });
        let style_name = or_return!(unsafe { arg_str(style_name, "style_name") });
        let type_name = or_return!(unsafe { arg_str(type_name, "type_name") });
        let endpoints = or_return!(unsafe { parse_strings(endpoints, n_endpoints, "endpoints") });
        let params = or_return!(unsafe { parse_params(param_keys, param_values, n_params) });
        let mut registry = lock_registry();
        let ff = get_forcefield_mut!(registry, ff);
        let Some(style) = ff.get_style_mut(category, style_name) else {
            return invalid_argument(DefError::UnknownStyle {
                category: category.to_owned(),
                name: style_name.to_owned(),
            });
        };
        match style.def_type(type_name, &endpoints, params) {
            Ok(_) => MolcStatus::Ok,
            Err(e) => invalid_argument(e),
        }
    })
}

// ---------------------------------------------------------------------------
// Queries
// ---------------------------------------------------------------------------

/// Get the number of interaction styles defined in a ForceField.
///
/// # C signature
///
/// ```c
/// MolcStatus molc_forcefield_n_styles(MolcForceFieldHandle ff,
///                                   size_t* out);
/// ```
///
/// # Arguments
///
/// * `ff` -- ForceField handle.
/// * `out` -- On success, receives the style count.
///
/// # Returns
///
/// * `MolcStatus::Ok` on success.
/// * `MolcStatus::NullPointer` if `out` is null.
/// * `MolcStatus::InvalidForceFieldHandle` if `ff` is stale.
///
/// # Safety
///
/// * `ff` must be a live ForceField handle.
/// * `out` must point to a writable `size_t`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn molc_forcefield_n_styles(
    ff: MolcForceFieldHandle,
    out: *mut usize,
) -> MolcStatus {
    ffi_try!({
        null_check!(out);
        let registry = lock_registry();
        let ff = get_forcefield!(registry, ff);
        unsafe { *out = ff.styles().len() };
        MolcStatus::Ok
    })
}

/// Get the category and name of a style by positional index.
///
/// Use [`molc_forcefield_n_styles`] to determine the valid index range
/// `[0, count)`.
///
/// Both returned strings are heap-allocated and must be freed with
/// [`molc_free_string`](crate::molc_free_string).
///
/// # C signature
///
/// ```c
/// MolcStatus molc_forcefield_style_name(MolcForceFieldHandle ff,
///                                      size_t index,
///                                      char** out_category,
///                                      char** out_name);
/// ```
///
/// # Arguments
///
/// * `ff` -- ForceField handle.
/// * `index` -- Zero-based style index.
/// * `out_category` -- Receives a heap-allocated category string
///   (e.g. `"pair"`, `"bond"`).
/// * `out_name` -- Receives a heap-allocated style name string
///   (e.g. `"lj/cut"`, `"harmonic"`).
///
/// # Returns
///
/// * `MolcStatus::Ok` on success.
/// * `MolcStatus::NullPointer` if `out_category` or `out_name` is null.
/// * `MolcStatus::InvalidArgument` if `index >= style_count`.
/// * `MolcStatus::InvalidForceFieldHandle` if `ff` is stale.
///
/// # Safety
///
/// * `ff` must be a live ForceField handle.
/// * `out_category` and `out_name` must each point to a writable `char*`.
/// * The caller owns both returned strings and must free them.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn molc_forcefield_style_name(
    ff: MolcForceFieldHandle,
    index: usize,
    out_category: *mut *mut c_char,
    out_name: *mut *mut c_char,
) -> MolcStatus {
    ffi_try!({
        null_check!(out_category);
        null_check!(out_name);
        let registry = lock_registry();
        let ff = get_forcefield!(registry, ff);
        let styles = ff.styles();
        if index >= styles.len() {
            error::set_last_error(format!(
                "style index {} out of range (count={})",
                index,
                styles.len()
            ));
            return MolcStatus::InvalidArgument;
        }
        let style = &styles[index];
        let cat = CString::new(style.category()).unwrap_or_default();
        let name = CString::new(style.name()).unwrap_or_default();
        unsafe {
            *out_category = cat.into_raw();
            *out_name = name.into_raw();
        }
        MolcStatus::Ok
    })
}

// ---------------------------------------------------------------------------
// JSON serialization
// ---------------------------------------------------------------------------

/// Serialize a ForceField to a JSON string: its core `forcefield` section
/// (`ForceFieldSection::from_forcefield`) in serde form,
/// `{"document": {…}, "tables": {<block>: Block}}` — molrs's one force-field
/// serialization, the section an `*.mrec` record stores.
///
/// The returned string is heap-allocated and must be freed with
/// [`molc_free_string`](crate::molc_free_string).
///
/// # C signature
///
/// ```c
/// MolcStatus molc_forcefield_to_json(MolcForceFieldHandle ff,
///                               char** out_json,
///                               size_t* out_len);
/// ```
///
/// # Arguments
///
/// * `ff` -- ForceField handle.
/// * `out_json` -- Receives a heap-allocated, null-terminated JSON string.
/// * `out_len` -- Receives the byte length of the JSON string
///   (not counting the null terminator).
///
/// # Returns
///
/// * `MolcStatus::Ok` on success.
/// * `MolcStatus::NullPointer` if `out_json` or `out_len` is null.
/// * `MolcStatus::InvalidForceFieldHandle` if `ff` is stale.
/// * `MolcStatus::InvalidArgument` if the force field has no section form
///   (`ForceFieldSection::from_forcefield` refuses it — e.g. units that are no preset).
///
/// # Safety
///
/// * `ff` must be a live ForceField handle.
/// * `out_json` must point to a writable `char*`.
/// * `out_len` must point to a writable `size_t`.
/// * The caller owns the returned string and must free it.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn molc_forcefield_to_json(
    ff: MolcForceFieldHandle,
    out_json: *mut *mut c_char,
    out_len: *mut usize,
) -> MolcStatus {
    ffi_try!({
        null_check!(out_json);
        null_check!(out_len);
        let registry = lock_registry();
        let ff = get_forcefield!(registry, ff);

        let json = match ff_to_json_string(ff) {
            Ok(json) => json,
            Err(msg) => {
                error::set_last_error(msg);
                return MolcStatus::InvalidArgument;
            }
        };
        let len = json.len();
        let c_json = CString::new(json).unwrap_or_default();
        unsafe {
            *out_json = c_json.into_raw();
            *out_len = len;
        }
        MolcStatus::Ok
    })
}

/// Deserialize a ForceField from a JSON string: a core `forcefield`
/// section in the serde form [`molc_forcefield_to_json`] writes, turned into a
/// force field by `ForceFieldSection::to_forcefield`.
///
/// The section is carried whole or refused: JSON that is no section (a
/// missing `document`, an unknown top-level key, a malformed table) or a
/// section `ForceFieldSection::to_forcefield` refuses is `InvalidArgument` --
/// nothing is skipped.
///
/// # C signature
///
/// ```c
/// MolcStatus molc_forcefield_from_json(const char* json,
///                                 MolcForceFieldHandle* out);
/// ```
///
/// # Arguments
///
/// * `json` -- Null-terminated JSON string.
/// * `out` -- On success, receives the new ForceField handle.
///
/// # Returns
///
/// * `MolcStatus::Ok` on success.
/// * `MolcStatus::NullPointer` if `json` or `out` is null.
/// * `MolcStatus::Utf8Error` if `json` is not valid UTF-8.
/// * `MolcStatus::InvalidArgument` if the JSON is malformed, is no
///   section, or is a section with no force-field form.
///
/// # Safety
///
/// * `json` must be a valid, null-terminated C string.
/// * `out` must point to a writable `MolcForceFieldHandle`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn molc_forcefield_from_json(
    json: *const c_char,
    out: *mut MolcForceFieldHandle,
) -> MolcStatus {
    ffi_try!({
        null_check!(json);
        null_check!(out);
        let json_str = match unsafe { CStr::from_ptr(json) }.to_str() {
            Ok(s) => s,
            Err(_) => {
                error::set_last_error("JSON is not valid UTF-8");
                return MolcStatus::Utf8Error;
            }
        };
        let ff = match ff_from_json_string(json_str) {
            Ok(ff) => ff,
            Err(msg) => {
                error::set_last_error(msg);
                return MolcStatus::InvalidArgument;
            }
        };
        let mut registry = lock_registry();
        let key = registry.forcefields.insert(ff);
        unsafe { *out = forcefield_key_to_handle(key) };
        MolcStatus::Ok
    })
}

// ---------------------------------------------------------------------------
// JSON: the core forcefield section
// ---------------------------------------------------------------------------

/// The force field's core section (`ForceFieldSection::from_forcefield`) as JSON.
fn ff_to_json_string(ff: &ForceField) -> Result<String, String> {
    let section = ForceFieldSection::from_forcefield(ff)?;
    serde_json::to_string(&section).map_err(|e| format!("JSON encode error: {e}"))
}

/// The force field the JSON form of a core section describes
/// (`ForceFieldSection::to_forcefield`).
fn ff_from_json_string(json: &str) -> Result<ForceField, String> {
    let section: ForceFieldSection =
        serde_json::from_str(json).map_err(|e| format!("JSON parse error: {e}"))?;
    section.to_forcefield()
}

#[cfg(test)]
mod tests {
    use super::*;
    use molrs::ff::ir::{ParamValue, Params, SpecialBonds};

    fn round_trip(ff: &ForceField) -> ForceField {
        ff_from_json_string(&ff_to_json_string(ff).unwrap()).unwrap()
    }

    #[test]
    fn json_round_trip_keeps_a_string_style_param() {
        let mut params: Params = [("cutoff", 10.0)].into_iter().collect();
        params.set("mixing", "geometric");
        let mut ff = ForceField::new("rt");
        ff.set_units("real");
        ff.def_style("pair", "lj/cut", params).unwrap();

        let back = round_trip(&ff);
        let style = back.get_style("pair", "lj/cut").unwrap();
        assert_eq!(
            style.params().get("mixing").and_then(ParamValue::as_text),
            Some("geometric")
        );
    }

    #[test]
    fn json_round_trip_keeps_bond_style_params() {
        let mut ff = ForceField::new("rt");
        ff.set_units("real");
        ff.def_style("bond", "harmonic", [("scale", 0.5)].into_iter().collect())
            .unwrap();

        let back = round_trip(&ff);
        let style = back.get_style("bond", "harmonic").unwrap();
        assert_eq!(
            style.params().get("scale").and_then(ParamValue::as_num),
            Some(0.5)
        );
    }

    /// A type's endpoints are data, not its name: MMFF's `0_1_5` comes back
    /// on `1`, `5` only because the document carries them.
    #[test]
    fn json_round_trip_keeps_the_given_endpoints() {
        let mut ff = ForceField::new("rt");
        ff.set_units("real");
        ff.def_style("bond", "mmff", Params::new())
            .unwrap()
            .def_type("0_1_5", &["1", "5"], [("kb", 4.258)].into_iter().collect())
            .unwrap();

        let back = round_trip(&ff);
        let style = back.get_style("bond", "mmff").unwrap();
        assert_eq!(
            style.type_endpoints("0_1_5"),
            Some(vec!["1".to_string(), "5".to_string()])
        );
    }

    /// An array param comes back at its shape, on a cmap type with five
    /// endpoints.
    #[test]
    fn json_round_trip_keeps_a_cmap_grid() {
        let grid =
            ndarray::ArrayD::from_shape_fn(vec![3, 3], |ix| (ix[0] * 3 + ix[1]) as f64 / 7.0);
        let mut params = Params::new();
        params.set("grid", grid.clone());
        let mut ff = ForceField::new("rt");
        ff.set_units("real");
        ff.def_style("cmap", "charmm", Params::new())
            .unwrap()
            .def_type("c", &["C", "N", "CA", "C", "N"], params)
            .unwrap();

        let back = round_trip(&ff);
        let cmap = back.get_types("cmap")[0];
        assert_eq!(
            cmap.params.get("grid").and_then(ParamValue::as_array),
            Some(&grid)
        );
        assert_eq!(cmap.endpoints[4], "N");
    }

    #[test]
    fn json_round_trip_keeps_special_bonds() {
        let special = SpecialBonds {
            lj: [0.0, 0.0, 0.5],
            coul: [0.0, 0.0, 0.8333],
        };
        let mut ff = ForceField::new("rt");
        ff.set_units("real");
        ff.set_special_bonds(special);

        let back = round_trip(&ff);
        assert_eq!(back.special_bonds(), Some(&special));
    }

    #[test]
    fn json_round_trip_keeps_units() {
        let mut ff = ForceField::new("rt");
        ff.set_units("real");
        ff.set_units("lj");

        let back = round_trip(&ff);
        assert_eq!(back.units(), Some("lj"));
    }

    /// Seam only: the Rust conflict rule is unit-tested in molrs; here a
    /// conflicting `molc_forcefield_def_type` must surface as `InvalidArgument`.
    #[test]
    fn def_type_conflicting_redefinition_is_invalid_argument() {
        let name = CString::new("conflict").unwrap();
        let bond = CString::new("bond").unwrap();
        let harmonic = CString::new("harmonic").unwrap();
        let ct_oh = CString::new("CT-OH").unwrap();
        let ct = CString::new("CT").unwrap();
        let oh = CString::new("OH").unwrap();
        let ends = [ct.as_ptr(), oh.as_ptr()];
        let k = CString::new("k").unwrap();
        let r0 = CString::new("r0").unwrap();
        let keys = [k.as_ptr(), r0.as_ptr()];
        let first = [300.0, 1.4];
        let second = [310.0, 1.4];
        let mut ff = MolcForceFieldHandle { idx: 0, version: 0 };

        // SAFETY: every pointer is a live CString / array of the stated length,
        // and `ff` is written by `molc_forcefield_new` before use.
        unsafe {
            assert_eq!(molc_forcefield_new(name.as_ptr(), &mut ff), MolcStatus::Ok);
            assert_eq!(
                molc_forcefield_def_style(
                    ff,
                    bond.as_ptr(),
                    harmonic.as_ptr(),
                    std::ptr::null(),
                    std::ptr::null(),
                    0
                ),
                MolcStatus::Ok
            );
            assert_eq!(
                molc_forcefield_def_type(
                    ff,
                    bond.as_ptr(),
                    harmonic.as_ptr(),
                    ct_oh.as_ptr(),
                    ends.as_ptr(),
                    2,
                    keys.as_ptr(),
                    first.as_ptr(),
                    2
                ),
                MolcStatus::Ok
            );
            let status = molc_forcefield_def_type(
                ff,
                bond.as_ptr(),
                harmonic.as_ptr(),
                ct_oh.as_ptr(),
                ends.as_ptr(),
                2,
                keys.as_ptr(),
                second.as_ptr(),
                2,
            );
            assert_eq!(molc_forcefield_drop(ff), MolcStatus::Ok);
            assert_eq!(status, MolcStatus::InvalidArgument);
        }
    }

    /// The JSON is the core section's serde form: what `ForceFieldSection::from_forcefield` gives,
    /// key for key.
    #[test]
    fn json_is_the_core_forcefield_section() {
        let mut ff = ForceField::new("rt");
        ff.set_units("real");
        ff.def_style("bond", "harmonic", Params::new())
            .unwrap()
            .def_type(
                "A-B",
                &["A", "B"],
                [("k", 300.0), ("r0", 1.5)].into_iter().collect(),
            )
            .unwrap();
        let json = ff_to_json_string(&ff).unwrap();
        let expected =
            serde_json::to_string(&ForceFieldSection::from_forcefield(&ff).unwrap()).unwrap();
        assert_eq!(json, expected);
    }

    /// JSON that is no section is refused whole: an unknown top-level key
    /// and a missing `document`.
    #[test]
    fn json_that_is_no_section_is_refused() {
        let mut ff = ForceField::new("rt");
        ff.set_units("real");
        ff.set_units("real");
        let mut doc: serde_json::Value =
            serde_json::from_str(&ff_to_json_string(&ff).unwrap()).unwrap();
        assert!(ff_from_json_string(&doc.to_string()).is_ok());
        doc["bogus"] = 1.into();
        assert!(ff_from_json_string(&doc.to_string()).is_err());
        assert!(ff_from_json_string(r#"{"tables": {}}"#).is_err());
    }
}
