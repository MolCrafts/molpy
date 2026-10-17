//! Crystallographic Information File for the WASM API — the face of `molrs::io::cif`.
//!
//! | JS | molrs |
//! |----|-------|
//! | `CifReader` | `CifReader::open` (random access by `data_` block) |
//! | `readCif`, `writeCif` | `read_cif`, `write_cif` |

use molrs::io::reader::{ReadSeek, TrajectoryReader};
use wasm_bindgen::prelude::*;

use crate::core::frame::Frame;
use crate::io::{JsSource, source_bytes};
use molrs::io::cif::CifReader as RsCifReader;

/// Crystallographic Information File (CIF / mmCIF) reader.
///
/// Each `data_*` block in the file becomes one [`Frame`]. Most CIF files
/// contain a single block (one structure), but multi-block files (e.g.
/// polymorphs of the same compound) are also supported and are exposed
/// as a multi-frame sequence. The unit cell parameters
/// (`_cell_length_a`, `_b`, `_c`, `_cell_angle_alpha`, `_beta`,
/// `_gamma`) are converted to a 3x3 h-matrix on the Rust side and
/// surface on the JS side as `frame.box`.
///
/// Produces a [`Frame`] with an `"atoms"` block containing
/// `element` (string), `x`, `y`, `z` (F, angstrom in Cartesian
/// coordinates) and (when present in the file) `label`, `occupancy`,
/// `bfactor` columns.
///
/// # Example (JavaScript)
///
/// ```js
/// const content = await file.text();
/// const reader = new CifReader(content);
/// const frame  = reader.readFrame(0);
/// const atoms  = frame.get("atoms");
/// const box    = frame.box;           // populated from the unit cell
/// ```
#[wasm_bindgen(js_name = CifReader)]
pub struct CifReader {
    inner: RsCifReader<Box<dyn ReadSeek>>,
}

#[wasm_bindgen(js_class = CifReader)]
impl CifReader {
    /// Open the file's content — its text (a `string`) or its bytes (a
    /// `Uint8Array`) — for random access by frame (molrs `CifReader::open`). The
    /// content is copied once.
    ///
    /// # Errors
    ///
    /// Throws if `source` is neither a string nor a `Uint8Array`.
    #[wasm_bindgen(constructor)]
    pub fn new(source: &JsSource) -> Result<CifReader, JsValue> {
        let bytes = source_bytes("new CifReader", source)?;
        RsCifReader::open(bytes.as_slice())
            .map(|inner| CifReader { inner })
            .map_err(|e| JsValue::from_str(&format!("CIF read error: {e}")))
    }

    /// The frame at `index` (0-based), or `undefined` past the end.
    ///
    /// # Errors
    ///
    /// Throws when the frame cannot be parsed.
    #[wasm_bindgen(js_name = readFrame)]
    pub fn read_frame(&mut self, index: usize) -> Result<Option<Frame>, JsValue> {
        match self
            .inner
            .read_frame(index)
            .map_err(|e| JsValue::from_str(&format!("CIF read error: {e}")))?
        {
            Some(frame) => Ok(Some(Frame::from_rs(frame)?)),
            None => Ok(None),
        }
    }

    /// The number of frames in the file.
    ///
    /// # Errors
    ///
    /// Throws when the file cannot be indexed.
    #[wasm_bindgen]
    pub fn len(&mut self) -> Result<usize, JsValue> {
        self.inner
            .len()
            .map_err(|e| JsValue::from_str(&format!("CIF read error: {e}")))
    }

    /// Whether the file holds no frame.
    ///
    /// # Errors
    ///
    /// As [`len`](Self::len).
    #[wasm_bindgen(js_name = isEmpty)]
    pub fn is_empty(&mut self) -> Result<bool, JsValue> {
        Ok(self.len()? == 0)
    }
}

read_door!(
    /// Read the first `data_` block of CIF, the file's text or bytes.
    readCif => read_cif, "CIF"
);
write_door!(
    /// Write `frame` as single-block CIF text.
    writeCif => write_cif, "CIF"
);
