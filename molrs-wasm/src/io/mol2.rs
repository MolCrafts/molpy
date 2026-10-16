//! Tripos MOL2 for the WASM API — the face of `molrs::io::mol2`.
//!
//! | JS | molrs |
//! |----|-------|
//! | `Mol2Reader` | `Mol2Reader::open` (random access by molecule) |
//! | `readMol2`, `writeMol2` | `read_mol2`, `write_mol2` |

use molrs::io::reader::{ReadSeek, TrajectoryReader};
use wasm_bindgen::prelude::*;

use crate::core::frame::Frame;
use crate::io::{JsSource, source_bytes};
use molrs::io::mol2::Mol2Reader as RsMol2Reader;

/// Tripos MOL2 reader.
///
/// MOL2 is a section-delimited (`@<TRIPOS>...`) text format. Multi-molecule
/// files expose each `MOLECULE` record as a frame via `readFrame(index)`.
/// Coordinates are already in angstrom. Produces an `"atoms"` block (`id`,
/// `name`, `x`/`y`/`z`, `atom_type`, optional `subst_id`/`subst_name`/
/// `charge`) and, when present, a `"bonds"` block (`atomi`/`atomj` 0-based,
/// `bond_type`).
#[wasm_bindgen(js_name = Mol2Reader)]
pub struct Mol2Reader {
    inner: RsMol2Reader<Box<dyn ReadSeek>>,
}

#[wasm_bindgen(js_class = Mol2Reader)]
impl Mol2Reader {
    /// Open the file's content — its text (a `string`) or its bytes (a
    /// `Uint8Array`) — for random access by frame (molrs `Mol2Reader::open`). The
    /// content is copied once.
    ///
    /// # Errors
    ///
    /// Throws if `source` is neither a string nor a `Uint8Array`.
    #[wasm_bindgen(constructor)]
    pub fn new(source: &JsSource) -> Result<Mol2Reader, JsValue> {
        let bytes = source_bytes("new Mol2Reader", source)?;
        RsMol2Reader::open(bytes.as_slice())
            .map(|inner| Mol2Reader { inner })
            .map_err(|e| JsValue::from_str(&format!("MOL2 read error: {e}")))
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
            .map_err(|e| JsValue::from_str(&format!("MOL2 read error: {e}")))?
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
            .map_err(|e| JsValue::from_str(&format!("MOL2 read error: {e}")))
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
    /// Read the first molecule of Tripos MOL2, the file's text or bytes.
    readMol2 => read_mol2, "MOL2"
);
write_door!(
    /// Write `frame` as Tripos MOL2 text.
    writeMol2 => write_mol2, "MOL2"
);
