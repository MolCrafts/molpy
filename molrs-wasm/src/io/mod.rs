//! File I/O and format conversion for the WASM API.
//!
//! One module per format, as in `molrs::io`: each holds that format's
//! in-memory doors and its reader class (a whole-content `*Reader` class or a
//! chunk-fed `*Stream` class, never both) where molrs has one.
//!
//! | Module | JS classes / functions | Format |
//! |--------|------------------------|--------|
//! | [`xyz`] | `XyzStream`, `readXyzStr`, `readXyzBytes`, `writeXyzStr` | XYZ / Extended XYZ |
//! | [`pdb`] | `PdbStream`, `readPdbStr`, `readPdbBytes`, `writePdbStr` | Protein Data Bank |
//! | [`gro`] | `GroReader`, `readGroStr`, `writeGroStr` | GROMACS GRO (nm ↔ Å at the molrs boundary) |
//! | [`sdf`] | `SdfStream`, `readSdfStr`, `readSdfBytes` | MDL SDF / MOL (read-only: molrs has no SDF writer) |
//! | [`mol2`] | `Mol2Reader`, `readMol2Str`, `writeMol2Str` | Tripos MOL2 |
//! | [`cif`] | `CifReader`, `readCifStr`, `writeCifStr` | Crystallographic Information File |
//! | [`vasp`] | `VaspPoscarReader`, `readPoscarStr`, `writePoscarStr`, `readChgcarStr` | VASP POSCAR / CONTCAR, CHGCAR (read-only) |
//! | [`dcd`] | `DcdStream`, `readDcdBytes`, `writeDcdBytes` | DCD trajectory |
//! | [`trr`] | `TrrStream`, `readTrrBytes`, `writeTrrBytes` | GROMACS TRR |
//! | [`xtc`] | `XtcStream`, `readXtcBytes`, `writeXtcBytes` | GROMACS XTC |
//! | [`lammps`] | `LammpsDataStream`, `LammpsDumpStream`, `readLammpsDataStr`, `readLammpsDataBytes`, `writeLammpsDataStr`, `readLammpsDumpStr`, `readLammpsDumpBytes`, `writeLammpsDumpStr`; `readLammpsLogStr`, `isLammpsLog` ([`lammps::log`]) | LAMMPS data, dump, run log |
//! | [`amber`] | `readInpcrdStr`, `readAcStr`, `readPrmtopStr` | AMBER inpcrd / restrt, Antechamber AC, prmtop structure |
//! | `smiles` | `readSmilesStr`, `writeSmilesStr`, `readCgsmilesStr`, `SmilesIr.parse` | SMILES and CGsmiles strings (`smiles` feature) |
//! | [`csv`] | `readCsvBlockStr`, `writeCsvBlockStr` | CSV tables as a `Block` |
//! | [`mrec`] | `MrecReader`, `readMrecFrameBytes` / `readMrecFrameFiles`, `sectionNamesBytes` / `sectionNamesFiles` | `*.mrec` scientific records (Zarr V3) |
//! | [`cube`] | `readCubeStr`, `writeCubeStr` | Gaussian Cube |
//! | [`xsf`] | `readXsfStr`, `writeXsfStr` | XCrySDen XSF |
//! | [`stl`] | `readStlBytes` | STL surface meshes (ASCII or binary) — produces a `TriMesh`, not a `Frame` |
//! | `frame_encoding` | `readMsgpackFrameBytes`, `writeMsgpackFrameBytes`, `readJsonFrameStr`, `writeJsonFrameStr` | `molrs::stream` wire encodings (`stream` feature) |
//! | [`frame_index`] | `FrameOffset` | Shared by every `*Stream`: the chunk-fed `FrameIndexBuilder` protocol |
//!
//! Every function is a `molrs::io` door of the same name, camelCased — the
//! in-memory doors `read_<fmt>_str` / `_bytes` and `write_<fmt>_str` /
//! `_bytes`, which Python's `molrs.io` carries too — and calls that door; a
//! class is a `molrs::io::<fmt>` reader class (`GroReader`, `Mol2Reader`,
//! `CifReader`, `VaspPoscarReader`) or a chunk-fed stream over the format's
//! `read_<fmt>_bytes`. No reader takes a file handle, since WASM has no
//! filesystem access, so no path door is bound. No export picks a format
//! from a string.

/// A `read_<fmt>_str` / `read_<fmt>_bytes` export: the molrs door of the
/// same name, its frame handed to JS.
macro_rules! read_door {
    ($(#[$doc:meta])* $js:ident => $name:ident($input:ident: $ty:ty), $what:literal) => {
        $(#[$doc])*
        #[wasm_bindgen(js_name = $js)]
        pub fn $name($input: $ty) -> Result<$crate::core::frame::Frame, JsValue> {
            let frame = molrs::io::$name($input)
                .map_err(|e| JsValue::from_str(&format!("{} read error: {e}", $what)))?;
            $crate::core::frame::Frame::from_rs(frame)
        }
    };
}

/// A `write_<fmt>_str` / `write_<fmt>_bytes` export: the molrs door of the
/// same name on `frame`.
macro_rules! write_door {
    ($(#[$doc:meta])* $js:ident => $name:ident -> $out:ty, $what:literal) => {
        $(#[$doc])*
        #[wasm_bindgen(js_name = $js)]
        pub fn $name(frame: &$crate::core::frame::Frame) -> Result<$out, JsValue> {
            frame.with_frame(|f| {
                molrs::io::$name(f)
                    .map_err(|e| JsValue::from_str(&format!("{} writing error: {e}", $what)))
            })
        }
    };
}

#[macro_use]
pub mod frame_index;

pub mod amber;
pub mod cif;
pub mod csv;
pub mod cube;
pub mod dcd;
#[cfg(feature = "stream")]
pub mod frame_encoding;
pub mod gro;
pub mod lammps;
pub mod mol2;
pub mod mrec;
pub mod pdb;
pub mod sdf;
#[cfg(feature = "smiles")]
pub mod smiles;
pub mod stl;
pub mod trr;
pub mod vasp;
pub mod xsf;
pub mod xtc;
pub mod xyz;

pub use amber::*;
pub use cif::*;
pub use csv::*;
pub use cube::*;
pub use dcd::*;
#[cfg(feature = "stream")]
pub use frame_encoding::*;
pub use frame_index::*;
pub use gro::*;
pub use lammps::*;
pub use mol2::*;
pub use mrec::*;
pub use pdb::*;
pub use sdf::*;
#[cfg(feature = "smiles")]
pub use smiles::*;
pub use stl::*;
pub use trr::*;
pub use vasp::*;
pub use xsf::*;
pub use xtc::*;
pub use xyz::*;

#[cfg(test)]
mod test_fixtures {
    use crate::core::frame::Frame;
    use crate::core::nd_array::JsFloatArray;
    use wasm_bindgen::JsCast;
    use wasm_bindgen::prelude::*;

    /// Owned `f64` column `key` of `block`.
    pub(super) fn float_col(block: &crate::core::Block, key: &str) -> js_sys::Float64Array {
        JsValue::from(block.copy(key, None).expect(key)).unchecked_into()
    }

    /// A two-atom (H, O) frame with `x`/`y`/`z` and `element` columns.
    pub(super) fn two_atom_frame() -> Frame {
        let frame = Frame::new();
        let mut atoms = frame.create_block("atoms").expect("atoms block");
        for (key, values) in [("x", [0.0, 1.0]), ("y", [0.0, 0.0]), ("z", [0.0, 0.5])] {
            let col = JsFloatArray::from(&values[..]);
            atoms
                .set(key, JsValue::from(col).unchecked_into(), None)
                .expect(key);
        }
        let elements = js_sys::Array::new();
        elements.push(&JsValue::from_str("H"));
        elements.push(&JsValue::from_str("O"));
        atoms
            .set("element", JsValue::from(elements).unchecked_into(), None)
            .expect("element");
        frame
    }
}
