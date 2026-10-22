//! Live streaming of `Frame`s.
//!
//! The core model serializes **directly**: the real [`Frame`] — with its
//! `Block`s, `Column`s, and `SimBox` — implements
//! `serde::Serialize`/`Deserialize` via the `serde` feature (which `stream`
//! enables); the impls live in the crate-private `serialize` module. There is no
//! separate wire type. This module only adds the transport encoding: MessagePack
//! (default) or JSON.

//! Beyond the encoding this module also carries the live transport that uses
//! it: [`crate::stream::ControlCommand`] (WASM-clean) and [`crate::stream::Publisher`] (native only). They
//! live here rather than under `io` because they pull third-party runtime
//! dependencies — tokio, tungstenite, rmp-serde — that `io` must not acquire.

mod control;

pub use control::ControlCommand;

#[cfg(not(target_arch = "wasm32"))]
mod publisher;

#[cfg(not(target_arch = "wasm32"))]
pub use publisher::{Publisher, PublisherConfig, SendError};

use crate::core::Frame;

/// Encoding used for a streamed `Frame`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameEncoding {
    /// MessagePack (compact binary; default).
    MessagePack,
    /// JSON (text; debugging / interop).
    Json,
}

/// Error from encoding or decoding a streamed `Frame`.
#[derive(Debug)]
pub enum StreamError {
    /// Serialization to bytes failed.
    Encode(String),
    /// Deserialization from bytes failed (bad bytes, or a payload that does not
    /// rebuild into a valid `Frame`).
    Decode(String),
}

impl std::fmt::Display for StreamError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StreamError::Encode(m) => write!(f, "stream encode error: {m}"),
            StreamError::Decode(m) => write!(f, "stream decode error: {m}"),
        }
    }
}

impl std::error::Error for StreamError {}

// The frame codecs. Their one public path is `crate::io`
// (`io::read_msgpack_frame_bytes`, …), which wraps these; they stay here
// because `stream` builds without `io`.

/// Encode a [`Frame`] as MessagePack bytes.
#[cfg_attr(not(feature = "io"), allow(dead_code))]
pub(crate) fn encode_msgpack_frame(frame: &Frame) -> Result<Vec<u8>, StreamError> {
    rmp_serde::to_vec_named(frame).map_err(|e| StreamError::Encode(e.to_string()))
}

/// Decode a [`Frame`] from MessagePack bytes.
#[cfg_attr(not(feature = "io"), allow(dead_code))]
pub(crate) fn decode_msgpack_frame(bytes: &[u8]) -> Result<Frame, StreamError> {
    rmp_serde::from_slice(bytes).map_err(|e| StreamError::Decode(e.to_string()))
}

/// Encode a [`Frame`] as JSON text.
#[cfg_attr(not(feature = "io"), allow(dead_code))]
pub(crate) fn encode_json_frame(frame: &Frame) -> Result<String, StreamError> {
    serde_json::to_string(frame).map_err(|e| StreamError::Encode(e.to_string()))
}

/// Decode a [`Frame`] from JSON text.
#[cfg_attr(not(feature = "io"), allow(dead_code))]
pub(crate) fn decode_json_frame(text: &str) -> Result<Frame, StreamError> {
    serde_json::from_str(text).map_err(|e| StreamError::Decode(e.to_string()))
}

/// The bytes a publisher configured for `format` puts on the wire.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn encode_frame(frame: &Frame, format: FrameEncoding) -> Result<Vec<u8>, StreamError> {
    match format {
        FrameEncoding::MessagePack => encode_msgpack_frame(frame),
        FrameEncoding::Json => encode_json_frame(frame).map(String::into_bytes),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::Block;
    use crate::core::Cell;
    use crate::core::SimBox;
    use crate::op::{F, I, Idx};
    use ndarray::Array1;

    /// Build a full Frame used by the net-streaming lossless round-trip contract
    /// (ac-002): atoms x/y/z + serial + type, bonds i/j + order, SimBox, meta.
    fn rich_frame() -> Frame {
        let mut atoms = Block::new();
        atoms
            .insert(
                "x",
                Array1::from_vec(vec![1.0 as F, 2.0 as F, 3.0 as F]).into_dyn(),
                None,
            )
            .unwrap();
        atoms
            .insert(
                "y",
                Array1::from_vec(vec![0.0 as F, 1.0 as F, 2.0 as F]).into_dyn(),
                None,
            )
            .unwrap();
        atoms
            .insert(
                "z",
                Array1::from_vec(vec![0.5 as F, 1.5 as F, 2.5 as F]).into_dyn(),
                None,
            )
            .unwrap();
        atoms
            .insert(
                "serial",
                Array1::from_vec(vec![10 as I, 20 as I, 30 as I]).into_dyn(),
                None,
            )
            .unwrap();
        atoms
            .insert(
                "atype",
                Array1::from_vec(vec![6u8, 1u8, 8u8]).into_dyn(),
                None,
            )
            .unwrap();

        let mut bonds = Block::new();
        bonds
            .insert(
                "i",
                Array1::from_vec(vec![0 as Idx, 1 as Idx]).into_dyn(),
                None,
            )
            .unwrap();
        bonds
            .insert(
                "j",
                Array1::from_vec(vec![1 as Idx, 2 as Idx]).into_dyn(),
                None,
            )
            .unwrap();
        bonds
            .insert("order", Array1::from_vec(vec![1u8, 1u8]).into_dyn(), None)
            .unwrap();

        let mut frame = Frame::new();
        frame.insert("atoms", atoms);
        frame.insert("bonds", bonds);
        frame.simbox = Some(
            SimBox::new(
                Cell::Lengths([10.0; 3]),
                [0.0, 0.0, 0.0],
                [true, true, true],
            )
            .expect("simbox"),
        );
        frame.meta.insert("title", "stream-roundtrip");
        frame.meta.insert("step", 42i64);
        frame
    }

    fn assert_frame_eq(a: &Frame, b: &Frame) {
        assert_eq!(a.len(), b.len());
        assert!(a.contains_key("atoms"));
        assert!(b.contains_key("atoms"));
        assert!(a.contains_key("bonds"));
        assert!(b.contains_key("bonds"));

        let ax = a["atoms"].get("x").and_then(|c| c.get::<f64>()).unwrap();
        let bx = b["atoms"].get("x").and_then(|c| c.get::<f64>()).unwrap();
        assert_eq!(ax.len(), bx.len());
        for (u, v) in ax.iter().zip(bx.iter()) {
            assert!((u - v).abs() < f64::EPSILON);
        }
        let ay = a["atoms"].get("y").and_then(|c| c.get::<f64>()).unwrap();
        let by = b["atoms"].get("y").and_then(|c| c.get::<f64>()).unwrap();
        for (u, v) in ay.iter().zip(by.iter()) {
            assert!((u - v).abs() < f64::EPSILON);
        }
        let az = a["atoms"].get("z").and_then(|c| c.get::<f64>()).unwrap();
        let bz = b["atoms"].get("z").and_then(|c| c.get::<f64>()).unwrap();
        for (u, v) in az.iter().zip(bz.iter()) {
            assert!((u - v).abs() < f64::EPSILON);
        }

        let aserial = a["atoms"]
            .get("serial")
            .and_then(|c| c.get::<i32>())
            .unwrap();
        let bserial = b["atoms"]
            .get("serial")
            .and_then(|c| c.get::<i32>())
            .unwrap();
        assert_eq!(aserial.as_slice().unwrap(), bserial.as_slice().unwrap());

        let atype = a["atoms"].get("atype").and_then(|c| c.get::<u8>()).unwrap();
        let btype = b["atoms"].get("atype").and_then(|c| c.get::<u8>()).unwrap();
        assert_eq!(atype.as_slice().unwrap(), btype.as_slice().unwrap());

        let bi = a["bonds"].get("i").and_then(|c| c.get::<u64>()).unwrap();
        let bj = b["bonds"].get("i").and_then(|c| c.get::<u64>()).unwrap();
        assert_eq!(bi.as_slice().unwrap(), bj.as_slice().unwrap());

        assert!(a.simbox.is_some());
        assert!(b.simbox.is_some());
        assert_eq!(
            a.meta.get("title").and_then(|m| m.as_str()),
            b.meta.get("title").and_then(|m| m.as_str())
        );
        assert_eq!(
            a.meta.get("step").and_then(|m| m.as_i64()),
            b.meta.get("step").and_then(|m| m.as_i64())
        );
    }

    #[test]
    fn frame_messagepack_roundtrip() {
        let frame = rich_frame();
        let bytes = encode_msgpack_frame(&frame).expect("encode");
        let back = decode_msgpack_frame(&bytes).expect("decode");
        assert_frame_eq(&frame, &back);
    }

    #[test]
    fn frame_json_roundtrip() {
        let frame = rich_frame();
        let text = encode_json_frame(&frame).expect("encode");
        let back = decode_json_frame(&text).expect("decode");
        assert_frame_eq(&frame, &back);
    }
}
