//! GROMACS GRO for the WASM API — the face of `molrs::io::gro`.
//!
//! | JS | molrs |
//! |----|-------|
//! | `GroReader` | `GroReader::open` (random access by frame) |
//! | `readGro`, `writeGro` | `read_gro`, `write_gro` (nm ↔ Å) |

use molrs::io::reader::{ReadSeek, TrajectoryReader};
use wasm_bindgen::prelude::*;

use crate::core::frame::Frame;
use crate::io::{JsSource, source_bytes};
use molrs::io::gro::GroReader as RsGroReader;

/// GROMACS GRO structure / trajectory reader.
///
/// GRO is a fixed-column text format for GROMACS structures and
/// single-precision trajectories. Multi-frame files expose each frame via
/// `readFrame(index)`. Coordinates and box are GROMACS-native nm in the file and
/// arrive in angstrom — the molrs GRO reader normalises at its own boundary,
/// so this binder scales nothing. Each frame produces an `"atoms"` block
/// (`res_id`, `res_name`, `name`, `element`, `id`, `x`/`y`/`z`, optional
/// `vx`/`vy`/`vz`) and a `box` from the box-vector line.
#[wasm_bindgen(js_name = GroReader)]
pub struct GroReader {
    inner: RsGroReader<Box<dyn ReadSeek>>,
}

#[wasm_bindgen(js_class = GroReader)]
impl GroReader {
    /// Open the file's content — its text (a `string`) or its bytes (a
    /// `Uint8Array`) — for random access by frame (molrs `GroReader::open`). The
    /// content is copied once.
    ///
    /// # Errors
    ///
    /// Throws if `source` is neither a string nor a `Uint8Array`.
    #[wasm_bindgen(constructor)]
    pub fn new(source: &JsSource) -> Result<GroReader, JsValue> {
        let bytes = source_bytes("new GroReader", source)?;
        RsGroReader::open(bytes.as_slice())
            .map(|inner| GroReader { inner })
            .map_err(|e| JsValue::from_str(&format!("GRO read error: {e}")))
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
            .map_err(|e| JsValue::from_str(&format!("GRO read error: {e}")))?
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
            .map_err(|e| JsValue::from_str(&format!("GRO read error: {e}")))
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
    /// Read the first frame of GROMACS GRO, the file's text or bytes (nm → Å).
    readGro => read_gro, "GRO"
);
write_door!(
    /// Write `frame` as GROMACS GRO text (Å → nm).
    writeGro => write_gro, "GRO"
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::test_fixtures::text;
    use wasm_bindgen_test::*;

    const GRO: &str = "two frames\n    1\n    1SOL     OW    1   0.100   0.200   0.300\n   1.0   1.0   1.0\n\
two frames\n    1\n    1SOL     OW    1   0.400   0.500   0.600\n   1.0   1.0   1.0\n";

    #[wasm_bindgen_test]
    fn gro_reader_reads_each_frame_by_index() {
        let mut reader = GroReader::new(&text(GRO)).expect("open");
        assert_eq!(reader.len().unwrap(), 2);
        let second = reader.read_frame(1).unwrap().expect("frame 1");
        let x = crate::io::test_fixtures::float_col(&second.get("atoms").unwrap(), "x");
        assert!((x.get_index(0) - 4.0).abs() < 1e-9, "nm → Å");
        assert!(reader.read_frame(2).unwrap().is_none());
        assert!(read_gro(&text(GRO)).is_ok());
    }
}
