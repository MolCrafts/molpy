//! A decode-only `zstd` codec over `ruzstd` (pure Rust), for builds that
//! cannot link the C `zstd` library — wasm32 above all.
//!
//! molrec puts `zstd` in the must-decode codec set (it is zarr-python's own
//! default compressor, and the reference compressor of a precision column),
//! so a reader built without `zarr-codecs` still has to open such an array.
//! zarrs' own `zstd` codec binds the C library through `zstd-sys`, which does
//! not cross-compile to `wasm32-unknown-unknown` without a wasm-capable
//! clang. This codec decodes the same frames in pure Rust and registers under
//! the Zarr V3 name `zstd` through zarrs' codec plugin inventory, exactly the
//! way zarrs registers its built-in codecs; `encode` refuses by name, since
//! nothing in such a build writes `zstd` (a precision column there is
//! shuffled and gzipped instead).
//!
//! The plugin is registered only when `zarr-codecs` is off: with it on, zarrs'
//! C-backed codec owns the name. The type itself is also compiled for the
//! test suite, which checks it against frames the C encoder wrote.

use std::borrow::Cow;
use std::io::Read;
use std::sync::Arc;

use zarrs::array::codec::api::{
    BytesToBytesCodecTraits, Codec, CodecError, CodecMetadataOptions, CodecOptions, CodecTraits,
    CodecTraitsV3, PartialDecoderCapability, PartialEncoderCapability, RecommendedConcurrency,
};
use zarrs::array::{ArrayBytesRaw, BytesRepresentation};
use zarrs::metadata::Configuration;
use zarrs::metadata::v3::MetadataV3;
use zarrs::metadata_ext::codec::zstd::ZstdCodecConfiguration;
use zarrs::plugin::{PluginCreateError, ZarrVersion};

/// The `zstd` bytes-to-bytes codec, decode only.
#[derive(Clone, Debug)]
pub(crate) struct ZstdDecodeCodec {
    /// The configuration the array metadata carried, handed back unchanged
    /// when the metadata is serialized again.
    configuration: ZstdCodecConfiguration,
}

zarrs::plugin::impl_extension_aliases!(ZstdDecodeCodec, v3: "zstd");

#[cfg(not(feature = "zarr-codecs"))]
inventory::submit! {
    zarrs::array::codec::api::CodecPluginV3::new::<ZstdDecodeCodec>()
}

impl CodecTraitsV3 for ZstdDecodeCodec {
    fn create(metadata: &MetadataV3) -> Result<Codec, PluginCreateError> {
        let configuration: ZstdCodecConfiguration = metadata.to_typed_configuration()?;
        Ok(Codec::BytesToBytes(Arc::new(Self { configuration })))
    }
}

impl CodecTraits for ZstdDecodeCodec {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn configuration(
        &self,
        _version: ZarrVersion,
        _options: &CodecMetadataOptions,
    ) -> Option<Configuration> {
        Some(self.configuration.clone().into())
    }

    fn partial_decoder_capability(&self) -> PartialDecoderCapability {
        PartialDecoderCapability {
            partial_read: false,
            partial_decode: false,
        }
    }

    fn partial_encoder_capability(&self) -> PartialEncoderCapability {
        PartialEncoderCapability {
            partial_encode: false,
        }
    }
}

impl BytesToBytesCodecTraits for ZstdDecodeCodec {
    fn into_dyn(self: Arc<Self>) -> Arc<dyn BytesToBytesCodecTraits> {
        self as Arc<dyn BytesToBytesCodecTraits>
    }

    fn recommended_concurrency(
        &self,
        _decoded_representation: &BytesRepresentation,
    ) -> Result<RecommendedConcurrency, CodecError> {
        Ok(RecommendedConcurrency::new_maximum(1))
    }

    fn encode<'a>(
        &self,
        _decoded_value: ArrayBytesRaw<'a>,
        _options: &CodecOptions,
    ) -> Result<ArrayBytesRaw<'a>, CodecError> {
        Err(CodecError::Other(
            "zstd: this build decodes zstd (pure-Rust ruzstd) but cannot encode it; enable the \
             `zarr-codecs` feature, or write with gzip"
                .to_string(),
        ))
    }

    fn decode<'a>(
        &self,
        encoded_value: ArrayBytesRaw<'a>,
        decoded_representation: &BytesRepresentation,
        _options: &CodecOptions,
    ) -> Result<ArrayBytesRaw<'a>, CodecError> {
        decode_frames(&encoded_value, decoded_representation.size()).map(Cow::Owned)
    }

    fn encoded_representation(
        &self,
        decoded_representation: &BytesRepresentation,
    ) -> BytesRepresentation {
        // The bound zarrs' own codec states: the frame header and checksum,
        // plus a block header per kilobyte of input.
        decoded_representation
            .size()
            .map_or(BytesRepresentation::UnboundedSize, |size| {
                const HEADER_TRAILER_OVERHEAD: u64 = 4 + 14 + 4;
                const MIN_WINDOW_SIZE: u64 = 1000;
                const BLOCK_OVERHEAD: u64 = 3;
                let blocks_overhead = BLOCK_OVERHEAD * size.div_ceil(MIN_WINDOW_SIZE);
                BytesRepresentation::BoundedSize(size + HEADER_TRAILER_OVERHEAD + blocks_overhead)
            })
    }
}

/// Decode every zstd frame of `encoded`, concatenated.
///
/// With the decoded size known — every fixed-size chunk — the frames decode
/// straight into a buffer of that size; otherwise frame by frame through the
/// streaming decoder.
fn decode_frames(encoded: &[u8], size: Option<u64>) -> Result<Vec<u8>, CodecError> {
    let failed = |e: &dyn std::fmt::Display| CodecError::Other(format!("zstd decode: {e}"));
    if let Some(size) = size {
        let mut out = Vec::with_capacity(size as usize);
        ruzstd::decoding::FrameDecoder::new()
            .decode_all_to_vec(encoded, &mut out)
            .map_err(|e| failed(&e))?;
        return Ok(out);
    }
    let mut out = Vec::new();
    let mut input = encoded;
    while !input.is_empty() {
        let mut frame =
            ruzstd::decoding::StreamingDecoder::new(&mut input).map_err(|e| failed(&e))?;
        frame.read_to_end(&mut out).map_err(|e| failed(&e))?;
    }
    Ok(out)
}

#[cfg(all(test, feature = "zarr-codecs"))]
mod tests {
    use super::*;
    use zarrs::array::codec::ZstdCodec;

    fn payload() -> Vec<u8> {
        // Shuffled, rounded coordinates look like this: long zero runs beside
        // a few busy bytes.
        (0..40_000u32)
            .flat_map(|i| {
                let x = (f64::from(i % 977) * 2f64.powi(-10)).to_le_bytes();
                x.into_iter()
            })
            .collect()
    }

    fn c_encoded(level: i32, checksum: bool) -> Vec<u8> {
        ZstdCodec::new(level, checksum)
            .encode(Cow::Owned(payload()), &CodecOptions::default())
            .unwrap()
            .into_owned()
    }

    #[test]
    fn decodes_what_the_c_encoder_wrote_with_and_without_a_known_size() {
        let expected = payload();
        for (level, checksum) in [(3, false), (1, true), (19, false)] {
            let encoded = c_encoded(level, checksum);
            assert!(encoded.len() < expected.len() / 4);
            for size in [Some(expected.len() as u64), None] {
                assert_eq!(decode_frames(&encoded, size).unwrap(), expected);
            }
        }
    }

    #[test]
    fn decodes_concatenated_frames() {
        let mut encoded = c_encoded(3, false);
        encoded.extend(c_encoded(3, false));
        let mut expected = payload();
        expected.extend(payload());
        assert_eq!(decode_frames(&encoded, None).unwrap(), expected);
        assert_eq!(
            decode_frames(&encoded, Some(expected.len() as u64)).unwrap(),
            expected
        );
    }

    #[test]
    fn garbage_is_a_codec_error_and_encoding_is_refused() {
        assert!(decode_frames(b"not a zstd frame", None).is_err());
        let codec = ZstdDecodeCodec {
            configuration: ZstdCodecConfiguration::V1(
                zarrs::metadata_ext::codec::zstd::ZstdCodecConfigurationV1::new(3.into(), false),
            ),
        };
        let err = codec
            .encode(Cow::Owned(payload()), &CodecOptions::default())
            .unwrap_err()
            .to_string();
        assert!(err.contains("cannot encode"), "{err}");
    }

    #[test]
    fn creates_from_v3_metadata_and_echoes_its_configuration() {
        let metadata: MetadataV3 = serde_json::from_str(
            r#"{"name": "zstd", "configuration": {"level": 3, "checksum": false}}"#,
        )
        .unwrap();
        let Codec::BytesToBytes(codec) =
            <ZstdDecodeCodec as CodecTraitsV3>::create(&metadata).unwrap()
        else {
            panic!("zstd is a bytes-to-bytes codec");
        };
        let configuration = codec
            .configuration(ZarrVersion::V3, &CodecMetadataOptions::default())
            .unwrap();
        assert_eq!(
            serde_json::to_value(configuration).unwrap(),
            serde_json::json!({"level": 3, "checksum": false})
        );
    }
}

/// The build this codec exists for: no C `zstd`, so an array whose pipeline
/// names `zstd` opens through the plugin registered above.
#[cfg(all(test, not(feature = "zarr-codecs")))]
mod plugin_tests {
    use std::sync::Arc;

    use zarrs::array::{Array, ArraySubset};
    use zarrs::storage::store::MemoryStore;
    use zarrs::storage::{StoreKey, WritableStorageTraits};

    /// An array laid down by hand the way the reference writer lays a
    /// precision column: `bytes`, `numcodecs.shuffle` (8), `zstd`, encoded
    /// here by ruzstd's own encoder since this build has no other.
    #[test]
    fn a_shuffled_zstd_array_reads_back_through_the_plugin() {
        let values: Vec<f64> = (0..64).map(|i| f64::from(i) * 2f64.powi(-10)).collect();
        let raw: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let mut shuffled = vec![0u8; raw.len()];
        for (element, bytes) in raw.as_chunks::<8>().0.iter().enumerate() {
            for (byte, value) in bytes.iter().enumerate() {
                shuffled[byte * values.len() + element] = *value;
            }
        }
        let encoded = ruzstd::encoding::compress_to_vec(
            &shuffled[..],
            ruzstd::encoding::CompressionLevel::Fastest,
        );

        let store = Arc::new(MemoryStore::new());
        let metadata = serde_json::json!({
            "zarr_format": 3,
            "node_type": "array",
            "shape": [values.len()],
            "data_type": "float64",
            "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [values.len()]}},
            "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
            "fill_value": 0.0,
            "codecs": [
                {"name": "bytes", "configuration": {"endian": "little"}},
                {"name": "numcodecs.shuffle", "configuration": {"elementsize": 8}},
                {"name": "zstd", "configuration": {"level": 3, "checksum": false}}
            ]
        });
        store
            .set(
                &StoreKey::new("x/zarr.json").unwrap(),
                serde_json::to_vec(&metadata).unwrap().into(),
            )
            .unwrap();
        store
            .set(&StoreKey::new("x/c/0").unwrap(), encoded.into())
            .unwrap();

        let array = Array::open(store, "/x").unwrap();
        let back: Vec<f64> = array
            .retrieve_array_subset(&ArraySubset::new_with_shape(vec![values.len() as u64]))
            .unwrap();
        assert_eq!(back, values);
    }
}
