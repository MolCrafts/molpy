//! The `molrs::stream` wire encodings for the WASM API — the face of
//! `molrs::io::frame_encoding` (`stream` feature).
//!
//! A page subscribed to a live run decodes payloads with these and never
//! re-derives the layout in JavaScript.
//!
//! | JS | molrs |
//! |----|-------|
//! | `FrameEncoding` | `stream::FrameEncoding` (`MessagePack`, `Json`) |
//! | `readWireFrame`, `writeWireFrame` | `read_wire_frame`, `write_wire_frame` |

use molrs::stream::FrameEncoding as RsFrameEncoding;
use wasm_bindgen::prelude::*;

use crate::core::frame::Frame;
use crate::io::{JsSource, source_bytes};

/// The encoding of a frame on the `molrs::stream` wire —
/// `stream::FrameEncoding`.
#[wasm_bindgen]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FrameEncoding {
    /// MessagePack: compact binary, what a publisher sends by default.
    MessagePack = "msgpack",
    /// JSON: UTF-8 text, for debugging and interop.
    Json = "json",
}

impl FrameEncoding {
    fn rs(self) -> Result<RsFrameEncoding, JsValue> {
        match self {
            FrameEncoding::MessagePack => Ok(RsFrameEncoding::MessagePack),
            FrameEncoding::Json => Ok(RsFrameEncoding::Json),
            FrameEncoding::__Invalid => Err(JsValue::from_str(
                "FrameEncoding must be \"msgpack\" or \"json\"",
            )),
        }
    }
}

/// Rebuild a [`Frame`] from its `molrs::stream` wire form in `encoding` —
/// what a publisher puts on the socket. The inverse of `writeWireFrame`.
///
/// `source` is the payload: a `Uint8Array` (either encoding) or a `string`
/// (JSON text).
///
/// # Errors
///
/// Throws if `source` is neither a string nor a `Uint8Array`, or is not a
/// frame in `encoding`.
///
/// # Example (JavaScript)
///
/// ```js
/// socket.onmessage = (ev) => {
///   const frame = readWireFrame(new Uint8Array(ev.data), "msgpack");
/// };
/// ```
#[wasm_bindgen(js_name = readWireFrame)]
pub fn read_wire_frame(source: &JsSource, encoding: FrameEncoding) -> Result<Frame, JsValue> {
    let bytes = source_bytes("readWireFrame", source)?;
    let rs_frame = molrs::io::read_wire_frame(&bytes, encoding.rs()?)
        .map_err(|e| JsValue::from_str(&format!("wire frame read error: {e}")))?;
    Frame::from_rs(rs_frame)
}

/// Encode `frame` in the `molrs::stream` wire `encoding` — what a publisher
/// puts on the socket; for JSON, UTF-8 text (`new TextDecoder().decode(…)`).
/// The inverse of `readWireFrame`.
#[wasm_bindgen(js_name = writeWireFrame)]
pub fn write_wire_frame(frame: &Frame, encoding: FrameEncoding) -> Result<Vec<u8>, JsValue> {
    let encoding = encoding.rs()?;
    frame.with_frame(|f| {
        molrs::io::write_wire_frame(f, encoding)
            .map_err(|e| JsValue::from_str(&format!("wire frame writing error: {e}")))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::test_fixtures::{bytes, float_col, text};
    use wasm_bindgen_test::*;

    #[wasm_bindgen_test]
    fn a_frame_round_trips_through_both_encodings() {
        use crate::core::nd_array::JsFloatArray;

        let frame = Frame::new();
        let mut atoms = frame.create_block("atoms").expect("atoms block");
        let x = JsFloatArray::from(&[1.0, 4.0][..]);
        atoms
            .set(
                "x",
                wasm_bindgen::JsCast::unchecked_into(JsValue::from(x)),
                None,
            )
            .expect("x");

        let msgpack = write_wire_frame(&frame, FrameEncoding::MessagePack).expect("encode");
        let json = write_wire_frame(&frame, FrameEncoding::Json).expect("encode");
        let json_text = String::from_utf8(json.clone()).expect("JSON is UTF-8");
        for back in [
            read_wire_frame(&bytes(&msgpack), FrameEncoding::MessagePack).expect("decode"),
            read_wire_frame(&bytes(&json), FrameEncoding::Json).expect("decode"),
            read_wire_frame(&text(&json_text), FrameEncoding::Json).expect("decode"),
        ] {
            let x = float_col(&back.get("atoms").expect("atoms"), "x");
            assert_eq!(x.to_vec(), vec![1.0, 4.0]);
        }
    }
}
