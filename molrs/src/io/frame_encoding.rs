//! A [`Frame`] as bytes in memory, in the encodings [`crate::stream`] puts
//! on the wire: MessagePack (the default) and JSON.
//!
//! The codecs live in `stream` (it builds without `io`); these are their one
//! public path.

use crate::core::Frame;
use crate::io::invalid_data;
use crate::stream::FrameEncoding;

/// Encode a [`Frame`] as `encoding` says — the bytes a
/// [`Publisher`](crate::stream::Publisher) sends (MessagePack by default; JSON
/// for debugging and interop, UTF-8 text).
///
/// ```
/// use molrs::io::{read_wire_frame, read_xyz, write_wire_frame};
/// use molrs::stream::FrameEncoding;
///
/// let frame = read_xyz(b"1\n\nH 0 0 0\n")?;
/// for encoding in [FrameEncoding::MessagePack, FrameEncoding::Json] {
///     let bytes = write_wire_frame(&frame, encoding)?;
///     assert_eq!(read_wire_frame(&bytes, encoding)?.keys().count(), frame.keys().count());
/// }
/// # Ok::<(), std::io::Error>(())
/// ```
pub fn write_wire_frame(frame: &Frame, encoding: FrameEncoding) -> std::io::Result<Vec<u8>> {
    match encoding {
        FrameEncoding::MessagePack => crate::stream::encode_msgpack_frame(frame),
        FrameEncoding::Json => crate::stream::encode_json_frame(frame).map(String::into_bytes),
    }
    .map_err(invalid_data)
}

/// Decode a [`Frame`] from `bytes` in `encoding` — the inverse of
/// [`write_wire_frame`].
///
/// # Errors
///
/// `InvalidData` for bytes that are not a frame in `encoding` (JSON that is
/// not UTF-8 included).
pub fn read_wire_frame(bytes: &[u8], encoding: FrameEncoding) -> std::io::Result<Frame> {
    match encoding {
        FrameEncoding::MessagePack => crate::stream::decode_msgpack_frame(bytes),
        FrameEncoding::Json => {
            crate::stream::decode_json_frame(std::str::from_utf8(bytes).map_err(invalid_data)?)
        }
    }
    .map_err(invalid_data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_frame_round_trips_through_both_encodings() {
        let frame = crate::io::read_xyz(b"1\nH\nH 0 0 0\n").unwrap();
        for encoding in [FrameEncoding::MessagePack, FrameEncoding::Json] {
            let bytes = write_wire_frame(&frame, encoding).unwrap();
            let back = read_wire_frame(&bytes, encoding).unwrap();
            assert_eq!(back.keys().count(), frame.keys().count());
        }
    }

    #[test]
    fn malformed_input_is_invalid_data() {
        let err = read_wire_frame(b"\xc1", FrameEncoding::MessagePack).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
        let err = read_wire_frame(b"{", FrameEncoding::Json).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
        let err = read_wire_frame(b"\xff", FrameEncoding::Json).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
    }
}
