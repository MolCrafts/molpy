//! VASP for the WASM API — the face of `molrs::io::vasp`.
//!
//! | JS | molrs |
//! |----|-------|
//! | `VaspPoscarReader` | `read_poscar` behind the reader-class protocol (`readFrame` / `len`; POSCAR / CONTCAR is one frame) |
//! | `readPoscar` | `read_poscar` |
//! | `writePoscar` | `write_poscar` (needs a `box`) |
//! | `readChgcar` | `read_chgcar` (CHGCAR / CHGDIF; functions only in molrs) |
//!
//! molrs has no CHGCAR writer, so CHGCAR is read-only.

use wasm_bindgen::prelude::*;

use crate::core::frame::Frame;
use crate::io::{JsSource, source_bytes};

/// VASP POSCAR / CONTCAR structure reader.
///
/// POSCAR describes a single crystalline cell. Coordinates are returned as
/// Cartesian angstrom (`Direct` files are converted on read by molrs).
/// Produces an `"atoms"` block (`x`/`y`/`z`, optional `symbol`,
/// selective-dynamics flags, velocities) and a periodic `box`.
/// Single-frame: any `step != 0` returns `undefined`.
#[wasm_bindgen(js_name = VaspPoscarReader)]
pub struct VaspPoscarReader {
    content: Vec<u8>,
    cached_len: Option<usize>,
}

#[wasm_bindgen(js_class = VaspPoscarReader)]
impl VaspPoscarReader {
    /// A POSCAR reader over the file's content: its text (a `string`) or
    /// its bytes (a `Uint8Array`), copied once.
    ///
    /// # Errors
    ///
    /// Throws if `source` is neither a string nor a `Uint8Array`.
    #[wasm_bindgen(constructor)]
    pub fn new(source: &JsSource) -> Result<VaspPoscarReader, JsValue> {
        Ok(VaspPoscarReader {
            content: source_bytes("new VaspPoscarReader", source)?,
            cached_len: None,
        })
    }

    /// Read the frame at `index`. POSCAR is single-frame, so any `index != 0`
    /// returns `undefined`.
    #[wasm_bindgen(js_name = readFrame)]
    pub fn read_frame(&mut self, index: usize) -> Result<Option<Frame>, JsValue> {
        if index > 0 {
            return Ok(None);
        }
        let rs_frame = molrs::io::read_poscar(self.content.as_slice())
            .map_err(|e| JsValue::from_str(&format!("POSCAR read error: {}", e)))?;
        Ok(Some(Frame::from_rs(rs_frame)?))
    }

    /// Return the number of frames (always 0 or 1 for POSCAR files).
    #[wasm_bindgen]
    pub fn len(&mut self) -> Result<usize, JsValue> {
        if let Some(n) = self.cached_len {
            return Ok(n);
        }
        let n = if self.read_frame(0)?.is_some() { 1 } else { 0 };
        self.cached_len = Some(n);
        Ok(n)
    }

    /// Check whether the file contains no valid frame.
    #[wasm_bindgen(js_name = isEmpty)]
    pub fn is_empty(&mut self) -> Result<bool, JsValue> {
        Ok(self.len()? == 0)
    }
}

read_door!(
    /// Read VASP POSCAR / CONTCAR, the file's text or bytes (Cartesian Å;
    /// `Direct` files are converted on read).
    readPoscar => read_poscar, "POSCAR"
);

read_door!(
    /// Read VASP CHGCAR / CHGDIF (the file's text or bytes) into a frame with:
    /// - `"atoms"` block: `element` (string), `x`/`y`/`z` (F, Cartesian Å).
    /// - `"grid"` block: structural shape `[nx, ny, nz]`, columns
    ///   `total` (always) and `diff` (when ISPIN=2).
    /// - `box`: triclinic POSCAR lattice in Å, fully periodic.
    ///
    /// ```js
    /// const frame = readChgcar(await file.text());
    /// const total = frame.get("grid").copy("total");
    /// ```
    readChgcar => read_chgcar, "CHGCAR"
);

write_door!(
    /// Write `frame` as VASP POSCAR text (needs a box).
    writePoscar => write_poscar, "POSCAR"
);
