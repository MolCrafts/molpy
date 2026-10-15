//! VASP for the WASM API — the face of `molrs::io::vasp`.
//!
//! | JS | molrs |
//! |----|-------|
//! | `VaspPoscarReader` | `VaspPoscarReader` (whole-content; POSCAR / CONTCAR) |
//! | `readPoscarStr` | `read_poscar_str` |
//! | `writePoscarStr` | `write_poscar_str` (needs a `box`) |
//! | `readChgcarStr` | `read_chgcar_str` (CHGCAR / CHGDIF; functions only in molrs) |
//!
//! molrs has no CHGCAR writer, so CHGCAR is read-only.

use wasm_bindgen::prelude::*;

use crate::core::frame::Frame;

/// VASP POSCAR / CONTCAR structure reader.
///
/// POSCAR describes a single crystalline cell. Coordinates are returned as
/// Cartesian angstrom (`Direct` files are converted on read by molrs).
/// Produces an `"atoms"` block (`x`/`y`/`z`, optional `symbol`,
/// selective-dynamics flags, velocities) and a periodic `box`.
/// Single-frame: any `step != 0` returns `undefined`.
#[wasm_bindgen(js_name = VaspPoscarReader)]
pub struct VaspPoscarReader {
    content: String,
    cached_len: Option<usize>,
}

#[wasm_bindgen(js_class = VaspPoscarReader)]
impl VaspPoscarReader {
    /// Create a new POSCAR reader from a string containing the file content.
    #[wasm_bindgen(constructor)]
    pub fn new(content: &str) -> VaspPoscarReader {
        VaspPoscarReader {
            content: content.to_string(),
            cached_len: None,
        }
    }

    /// Read the frame at `index`. POSCAR is single-frame, so any `index != 0`
    /// returns `undefined`.
    #[wasm_bindgen(js_name = readFrame)]
    pub fn read_frame(&mut self, index: usize) -> Result<Option<Frame>, JsValue> {
        if index > 0 {
            return Ok(None);
        }
        let rs_frame = molrs::io::read_poscar_str(&self.content)
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
    /// Read VASP POSCAR / CONTCAR text (Cartesian Å; `Direct` files are
    /// converted on read).
    readPoscarStr => read_poscar_str(text: &str), "POSCAR"
);

read_door!(
    /// Read VASP CHGCAR / CHGDIF text into a frame with:
    /// - `"atoms"` block: `element` (string), `x`/`y`/`z` (F, Cartesian Å).
    /// - `"grid"` block: structural shape `[nx, ny, nz]`, columns
    ///   `total` (always) and `diff` (when ISPIN=2).
    /// - `box`: triclinic POSCAR lattice in Å, fully periodic.
    ///
    /// ```js
    /// const frame = readChgcarStr(await file.text());
    /// const total = frame.get("grid").copy("total");
    /// ```
    readChgcarStr => read_chgcar_str(text: &str), "CHGCAR"
);

write_door!(
    /// Write `frame` as VASP POSCAR text (needs a box).
    writePoscarStr => write_poscar_str -> String, "POSCAR"
);
