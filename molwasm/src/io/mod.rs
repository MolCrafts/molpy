//! File I/O and format conversion for the WASM API.
//!
//! One module per format, as in `molrs::io`: each holds that format's one
//! door per direction — `read<Fmt>(source)` and `write<Fmt>(frame)` — and
//! its reader class (a whole-content `*Reader` class or a chunk-fed
//! `*Stream` class, never both) where molrs has one.
//!
//! A read door takes a [`Source`](JsSource): a whole file's content, as a
//! `string` (always the file's **text** — WASM has no filesystem, so a string
//! is never a path) or a `Uint8Array` (its bytes). A binary format's door
//! (DCD, TRR, XTC) takes the bytes alone. A write door returns the whole
//! file: a `string` for a text format, a `Uint8Array` for a binary one.
//!
//! | Module | JS classes / functions | Format |
//! |--------|------------------------|--------|
//! | [`xyz`] | `XyzStream`, `readXyz`, `writeXyz` | XYZ / Extended XYZ |
//! | [`pdb`] | `PdbStream`, `readPdb`, `writePdb` | Protein Data Bank |
//! | [`gro`] | `GroReader`, `readGro`, `writeGro` | GROMACS GRO (nm ↔ Å at the molrs boundary) |
//! | [`sdf`] | `SdfStream`, `readSdf` | MDL SDF / MOL (read-only: molrs has no SDF writer) |
//! | [`mol2`] | `Mol2Reader`, `readMol2`, `writeMol2` | Tripos MOL2 |
//! | [`cif`] | `CifReader`, `readCif`, `writeCif` | Crystallographic Information File |
//! | [`vasp`] | `VaspPoscarReader`, `readPoscar`, `writePoscar`, `readChgcar` | VASP POSCAR / CONTCAR, CHGCAR (read-only) |
//! | [`dcd`] | `DcdStream`, `readDcd`, `writeDcd` | DCD trajectory |
//! | [`trr`] | `TrrStream`, `readTrr`, `writeTrr` | GROMACS TRR |
//! | [`xtc`] | `XtcStream`, `readXtc`, `writeXtc` | GROMACS XTC |
//! | [`lammps`] | `LammpsDataStream`, `LammpsDumpStream`, `readLammpsData`, `writeLammpsData`, `readLammpsDump`, `writeLammpsDump`; `readLammpsLog`, `isLammpsLog` ([`lammps::log`]) | LAMMPS data, dump, run log |
//! | [`amber`] | `readInpcrd`, `readAc`, `readPrmtop` | AMBER inpcrd / restrt, Antechamber AC, prmtop structure |
//! | `smiles` | `SmilesIr`, `CgSmilesIr` | SMILES and CGsmiles line notations (`smiles` feature) |
//! | [`csv`] | `readCsvBlock`, `writeCsvBlock` | CSV tables as a `Block` |
//! | [`mrec`] | `MrecReader`, `MrecFile` | `*.mrec` scientific records (Zarr V3) |
//! | [`cube`] | `readCube`, `writeCube` | Gaussian Cube |
//! | [`xsf`] | `readXsf`, `writeXsf` | XCrySDen XSF |
//! | [`stl`] | `readStl` | STL surface meshes (ASCII or binary) — produces a `TriMesh`, not a `Frame` |
//! | `frame_encoding` | `FrameEncoding`, `readWireFrame`, `writeWireFrame` | `molrs::stream` wire encodings (`stream` feature) |
//! | [`frame_index`] | `FrameOffset` | Shared by every `*Stream`: the chunk-fed `FrameIndexBuilder` protocol |
//!
//! Every function is the `molrs::io` door of the same name, camelCased —
//! `read_<fmt>(Source::Bytes)` and `write_<fmt>(Sink::Buffer, frame)` — and
//! calls that door; a class is a `molrs::io::<fmt>` reader class
//! (`GroReader`, `Mol2Reader`, `CifReader`, `VaspPoscarReader`) or a
//! chunk-fed stream over the format's indexer and `read_<fmt>`. No export
//! picks a format from a string.

use wasm_bindgen::prelude::*;

#[wasm_bindgen(typescript_custom_section)]
const SOURCE_TYPE: &'static str = r#"
/**
 * A whole file's content, as every text-format read door takes it: a
 * `string` is always the file's text (WASM has no filesystem, so a string is
 * never a path), a `Uint8Array` the file's bytes.
 */
export type Source = string | Uint8Array;
"#;

#[wasm_bindgen]
extern "C" {
    /// A whole file's content: see the `Source` TypeScript type.
    #[wasm_bindgen(typescript_type = "Source")]
    pub type JsSource;
}

/// The bytes of `source`: a string's UTF-8 text, or a `Uint8Array`'s bytes.
/// `op` names the door in the error.
pub(crate) fn source_bytes(op: &str, source: &JsSource) -> Result<Vec<u8>, JsValue> {
    if let Some(text) = source.as_string() {
        return Ok(text.into_bytes());
    }
    if let Some(bytes) = wasm_bindgen::JsCast::dyn_ref::<js_sys::Uint8Array>(&**source) {
        return Ok(bytes.to_vec());
    }
    Err(JsValue::from_str(&format!(
        "{op}: source must be the file's text (a string) or its bytes (a Uint8Array)"
    )))
}

/// A `read_<fmt>` export over a [`Source`](JsSource): the molrs door of the
/// same name on the source's bytes, its frame handed to JS.
macro_rules! read_door {
    ($(#[$doc:meta])* $js:ident => $name:ident, $what:literal) => {
        $(#[$doc])*
        #[wasm_bindgen(js_name = $js)]
        pub fn $name(
            source: &$crate::io::JsSource,
        ) -> Result<$crate::core::frame::Frame, JsValue> {
            let bytes = $crate::io::source_bytes(stringify!($js), source)?;
            let frame = molrs::io::$name(bytes.as_slice())
                .map_err(|e| JsValue::from_str(&format!("{} read error: {e}", $what)))?;
            $crate::core::frame::Frame::from_rs(frame)
        }
    };
}

/// A `read_<fmt>` export for a binary format: the molrs door of the same
/// name on the file's bytes.
macro_rules! read_bytes_door {
    ($(#[$doc:meta])* $js:ident => $name:ident, $what:literal) => {
        $(#[$doc])*
        #[wasm_bindgen(js_name = $js)]
        pub fn $name(bytes: &[u8]) -> Result<$crate::core::frame::Frame, JsValue> {
            let frame = molrs::io::$name(bytes)
                .map_err(|e| JsValue::from_str(&format!("{} read error: {e}", $what)))?;
            $crate::core::frame::Frame::from_rs(frame)
        }
    };
}

/// The whole file `write` puts into a buffer, `what` naming the format in
/// the error.
pub(crate) fn written<E: std::fmt::Display>(
    what: &str,
    write: impl FnOnce(&mut Vec<u8>) -> Result<(), E>,
) -> Result<Vec<u8>, JsValue> {
    let mut buffer = Vec::new();
    write(&mut buffer).map_err(|e| JsValue::from_str(&format!("{what} writing error: {e}")))?;
    Ok(buffer)
}

/// The text a text-format writer put into its buffer.
pub(crate) fn text_of(what: &str, bytes: Vec<u8>) -> Result<String, JsValue> {
    String::from_utf8(bytes)
        .map_err(|e| JsValue::from_str(&format!("{what} writing error: not UTF-8 text: {e}")))
}

/// A `write_<fmt>` export for a text format: the molrs door of the same name
/// into a buffer, returned as the file's text.
macro_rules! write_door {
    ($(#[$doc:meta])* $js:ident => $name:ident, $what:literal) => {
        $(#[$doc])*
        #[wasm_bindgen(js_name = $js)]
        pub fn $name(frame: &$crate::core::frame::Frame) -> Result<String, JsValue> {
            let bytes = frame.with_frame(|f| {
                $crate::io::written($what, |buf| molrs::io::$name(buf, f))
            })?;
            $crate::io::text_of($what, bytes)
        }
    };
}

/// A `write_<fmt>` export for a binary format: the molrs door of the same
/// name into a buffer, returned as the file's bytes.
macro_rules! write_bytes_door {
    ($(#[$doc:meta])* $js:ident => $name:ident, $what:literal) => {
        $(#[$doc])*
        #[wasm_bindgen(js_name = $js)]
        pub fn $name(frame: &$crate::core::frame::Frame) -> Result<Vec<u8>, JsValue> {
            frame.with_frame(|f| $crate::io::written($what, |buf| molrs::io::$name(buf, f)))
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

    /// `text` as a read door's `Source`.
    pub(super) fn text(text: &str) -> crate::io::JsSource {
        JsValue::from_str(text).unchecked_into()
    }

    /// `bytes` as a read door's `Source`.
    pub(super) fn bytes(bytes: &[u8]) -> crate::io::JsSource {
        JsValue::from(js_sys::Uint8Array::from(bytes)).unchecked_into()
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
