//! `extern "C"` door for SMILES text: the C face of molrs's
//! `SmilesIr::parse(..)?.to_atomistic()`.

use std::ffi::{CStr, c_char};

use molrs::io::smiles::SmilesIr;

use crate::error::{self, MolcStatus, ffi_err_to_status};
use crate::handle::{MolcFrameHandle, frame_id_to_handle};
use crate::handle_registry::lock_registry;
use crate::{ffi_try, null_check};

/// Read one molecule from SMILES text into a new frame (`SmilesIr::parse`,
/// `SmilesIr::to_atomistic`, then `Atomistic::to_frame`).
///
/// Connectivity only: hydrogens implicit in the SMILES are **not** added and
/// **no coordinates** are generated, so the frame has no `x`/`y`/`z`
/// columns. The `"atoms"` block carries `atomic_number` (`uint64_t`, the Z
/// every molrs consumer reads; 0 for the wildcard `*`) and `mass` (absent on
/// a wildcard), plus `h_count`, `formal_charge`, `isotope`, `atom_map` (the
/// atom class) and `stereo` where a bracket atom states them and
/// `is_aromatic` for an aromatic atom. No `element` symbols are written:
/// a caller that needs them (a symbol-spelling writer) converts explicitly
/// with molrs `Element::fill_symbols`;
/// the `"bonds"` block carries `atomi` / `atomj` (`uint64_t`, 0-based),
/// `bond_type` (0 unknown, 1 single, 2 double, 3 triple, 4 aromatic) and
/// `bond_number` (the localized Lewis order).
///
/// # C signature
///
/// ```c
/// MolcStatus molc_read_smiles(const char* smiles,
///                               MolcFrameHandle* out);
/// ```
///
/// # Arguments
///
/// * `smiles` -- Null-terminated SMILES string (e.g. `"CCO"`). One
///   molecule: a `.`-separated set of components is refused.
/// * `out` -- On success, receives the new frame handle.
///
/// # Returns
///
/// * `MolcStatus::Ok` on success.
/// * `MolcStatus::NullPointer` if either pointer is null.
/// * `MolcStatus::Utf8Error` if `smiles` is not valid UTF-8.
/// * `MolcStatus::ParseError` if the SMILES string is malformed or names
///   more than one molecule.
/// * `MolcStatus::InternalError` if the parsed molecule cannot be expressed
///   as a frame (a property contradicting the Frame schema).
///
/// # Safety
///
/// * `smiles` must be a valid, null-terminated UTF-8 C string.
/// * `out` must point to a writable `MolcFrameHandle`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn molc_read_smiles(
    smiles: *const c_char,
    out: *mut MolcFrameHandle,
) -> MolcStatus {
    ffi_try!({
        null_check!(smiles);
        null_check!(out);
        let Ok(smiles_str) = (unsafe { CStr::from_ptr(smiles) }).to_str() else {
            error::set_last_error("SMILES string is not valid UTF-8");
            return MolcStatus::Utf8Error;
        };
        let ir = match SmilesIr::parse(smiles_str) {
            Ok(ir) => ir,
            Err(e) => {
                error::set_last_error(format!("{e}"));
                return MolcStatus::ParseError;
            }
        };
        if ir.components.len() != 1 {
            error::set_last_error(format!(
                "SMILES '{smiles_str}' names {} molecules ('.'-separated components); \
                 molc_read_smiles reads one -- read each component separately",
                ir.components.len()
            ));
            return MolcStatus::ParseError;
        }
        let mol = match ir.to_atomistic() {
            Ok(m) => m,
            Err(e) => {
                error::set_last_error(format!("{e}"));
                return MolcStatus::ParseError;
            }
        };
        let frame = match mol.to_frame() {
            Ok(f) => f,
            Err(e) => {
                error::set_last_error(format!("{e}"));
                return MolcStatus::InternalError;
            }
        };

        let mut registry = lock_registry();
        let id = registry.frames.frame_new();
        if let Err(e) = registry.frames.set_frame(id, frame) {
            return ffi_err_to_status(&e);
        }
        unsafe { *out = frame_id_to_handle(id) };
        MolcStatus::Ok
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::handle_to_frame_id;

    #[test]
    fn ethanol_has_the_documented_columns() {
        let mut out = MolcFrameHandle { idx: 0, version: 0 };
        let status = unsafe { molc_read_smiles(c"CCO".as_ptr(), &mut out) };
        assert_eq!(status, MolcStatus::Ok);
        let frame = lock_registry()
            .frames
            .clone_frame(handle_to_frame_id(out))
            .unwrap();
        let atoms = frame.get("atoms").unwrap();
        assert_eq!(atoms.n_rows(), Some(3));
        for key in ["atomic_number", "mass"] {
            assert!(atoms.contains_key(key), "atoms.{key}");
        }
        let z: Vec<u64> = atoms
            .get("atomic_number")
            .and_then(|c| c.get::<u64>())
            .expect("atomic_number is a uint column")
            .iter()
            .copied()
            .collect();
        assert_eq!(z, [6, 6, 8]);
        assert!(
            !atoms.contains_key("element"),
            "SMILES writes atomic_number, not element symbols"
        );
        assert!(!atoms.contains_key("x"), "SMILES carries no coordinates");
        let bonds = frame.get("bonds").unwrap();
        for key in ["atomi", "atomj", "bond_type", "bond_number"] {
            assert!(bonds.contains_key(key), "bonds.{key}");
        }
    }

    #[test]
    fn two_molecules_are_a_parse_error() {
        let mut out = MolcFrameHandle { idx: 0, version: 0 };
        let status = unsafe { molc_read_smiles(c"CCO.O".as_ptr(), &mut out) };
        assert_eq!(status, MolcStatus::ParseError);
    }
}
