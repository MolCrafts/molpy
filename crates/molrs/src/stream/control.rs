//! Control commands sent from streaming clients back to the simulation.

use serde::{Deserialize, Serialize};

use crate::stream::{FrameEncoding, StreamError};

/// A bidirectional control message from a visualization client to the server.
///
/// Wire format is externally tagged by `"type"` with snake_case variant names,
/// so a browser client can send `{"type":"pause"}` or MessagePack with the same
/// shape.
///
/// It is WASM-clean: it depends only on `serde` (and optional
/// MessagePack / JSON codecs at the call site). No tokio or socket types.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ControlCommand {
    /// Pause the producing simulation loop (interpretation is caller-defined).
    Pause,
    /// Resume after [`Self::Pause`].
    Resume,
    /// Request a maximum stream rate in frames per second.
    SetFrameRate {
        /// Desired frames per second.
        hz: f64,
    },
    /// Restrict subsequent frames to a subset of atom indices.
    SetSubset {
        /// Atom ids to keep, in caller-defined order (preserved on the wire).
        atom_ids: Vec<u32>,
    },
    /// Ask the producer to emit a full key frame on the next send.
    RequestKeyFrame,
}

impl ControlCommand {
    /// Encode this command in `format`.
    ///
    /// Shares [`FrameEncoding`] with [`crate::stream`] so a client speaks one
    /// encoding for both directions of the socket.
    pub fn to_bytes(&self, format: FrameEncoding) -> Result<Vec<u8>, StreamError> {
        match format {
            FrameEncoding::Json => {
                serde_json::to_vec(self).map_err(|e| StreamError::Encode(e.to_string()))
            }
            FrameEncoding::MessagePack => {
                rmp_serde::to_vec_named(self).map_err(|e| StreamError::Encode(e.to_string()))
            }
        }
    }

    /// Decode a command written in `format`.
    pub fn from_bytes(bytes: &[u8], format: FrameEncoding) -> Result<Self, StreamError> {
        match format {
            FrameEncoding::Json => {
                serde_json::from_slice(bytes).map_err(|e| StreamError::Decode(e.to_string()))
            }
            FrameEncoding::MessagePack => {
                rmp_serde::from_slice(bytes).map_err(|e| StreamError::Decode(e.to_string()))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn all_variants() -> Vec<ControlCommand> {
        vec![
            ControlCommand::Pause,
            ControlCommand::Resume,
            ControlCommand::SetFrameRate { hz: 42.0 },
            ControlCommand::SetSubset {
                atom_ids: vec![1, 3, 5],
            },
            ControlCommand::RequestKeyFrame,
        ]
    }

    #[test]
    fn control_command_json_roundtrip() {
        for cmd in all_variants() {
            let bytes = serde_json::to_vec(&cmd).expect("json encode");
            let back: ControlCommand = serde_json::from_slice(&bytes).expect("json decode");
            assert_eq!(back, cmd);
        }
    }

    #[test]
    fn control_command_rmp_roundtrip() {
        for cmd in all_variants() {
            let bytes = rmp_serde::to_vec_named(&cmd).expect("rmp encode");
            let back: ControlCommand = rmp_serde::from_slice(&bytes).expect("rmp decode");
            assert_eq!(back, cmd);
        }
    }

    #[test]
    fn control_command_json_type_tags() {
        let pause = serde_json::to_value(ControlCommand::Pause).unwrap();
        assert_eq!(pause["type"], "pause");
        let resume = serde_json::to_value(ControlCommand::Resume).unwrap();
        assert_eq!(resume["type"], "resume");
        let rate = serde_json::to_value(ControlCommand::SetFrameRate { hz: 10.0 }).unwrap();
        assert_eq!(rate["type"], "set_frame_rate");
        let subset = serde_json::to_value(ControlCommand::SetSubset {
            atom_ids: vec![2, 4],
        })
        .unwrap();
        assert_eq!(subset["type"], "set_subset");
        assert_eq!(subset["atom_ids"], serde_json::json!([2, 4]));
        let key = serde_json::to_value(ControlCommand::RequestKeyFrame).unwrap();
        assert_eq!(key["type"], "request_key_frame");
    }

    #[test]
    fn to_bytes_round_trips_every_variant_in_both_formats() {
        for format in [FrameEncoding::Json, FrameEncoding::MessagePack] {
            for cmd in all_variants() {
                let bytes = cmd.to_bytes(format).expect("encode");
                let back = ControlCommand::from_bytes(&bytes, format).expect("decode");
                assert_eq!(back, cmd, "{format:?}");
            }
        }
    }

    #[test]
    fn to_bytes_emits_the_same_wire_as_bare_serde() {
        // The codec methods exist so callers stop hand-rolling the encoding.
        // That only holds if they produce byte-identical output to the serde
        // calls they replace — otherwise a mixed-vintage client silently
        // fails to parse.
        for cmd in all_variants() {
            assert_eq!(
                cmd.to_bytes(FrameEncoding::Json).unwrap(),
                serde_json::to_vec(&cmd).unwrap()
            );
            assert_eq!(
                cmd.to_bytes(FrameEncoding::MessagePack).unwrap(),
                rmp_serde::to_vec_named(&cmd).unwrap()
            );
        }
    }

    #[test]
    fn from_bytes_rejects_a_payload_in_the_other_format() {
        let json = ControlCommand::Pause.to_bytes(FrameEncoding::Json).unwrap();
        assert!(ControlCommand::from_bytes(&json, FrameEncoding::MessagePack).is_err());
    }

    #[test]
    fn set_subset_preserves_atom_id_order() {
        let cmd = ControlCommand::SetSubset {
            atom_ids: vec![9, 1, 7, 3],
        };
        let bytes = rmp_serde::to_vec_named(&cmd).unwrap();
        let back: ControlCommand = rmp_serde::from_slice(&bytes).unwrap();
        match back {
            ControlCommand::SetSubset { atom_ids } => assert_eq!(atom_ids, vec![9, 1, 7, 3]),
            other => panic!("unexpected variant: {other:?}"),
        }
    }
}
