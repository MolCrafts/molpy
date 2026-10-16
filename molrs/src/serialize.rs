//! `serde` support for the core model, enabled by the `serde` feature.
//!
//! The real [`Frame`], [`Block`], [`Column`], and [`SimBox`] implement
//! `Serialize`/`Deserialize` **directly** — there is no parallel "wire" type.
//! Transport encodings (MessagePack / JSON) live in [`crate::stream`], which
//! turns this feature on.
//!
//! On-wire shape (the general MolRec model, no privileged fields):
//!
//! - `Frame`  -> `{ blocks: { <name>: Block }, meta: { k: {dtype,value} }, box?: SimBox }`
//! - `Block`  -> `{ shape: [usize], columns: { <name>: Column },
//!   validity?: { <name>: [bool] } }` — `validity` carries the per-row masks
//!   of the nullable columns only, and is omitted when no column has one, so
//!   a payload without it reads back as a block whose every cell is filled.
//!   Blocks, columns and meta keys are written and read back in insertion
//!   order; nothing is sorted.
//! - `Column` -> `{ dtype, shape: [usize], data }` — `data` is raw
//!   little-endian bytes for numeric dtypes, or a string list for `string`.
//! - `SimBox` -> `{ vectors: [[f64;3];3], origin, boundary, cell_defined }`
//!
//! The record's force-field section (`io::mrec::ForceFieldSection`) carries
//! its own impls beside its type.
//!
//! The small private `*Repr` structs and visitors below are serde
//! deserialization scaffolding (derive needs owned fields); they are not part
//! of the public model.

use crate::core::Cell;
use indexmap::IndexMap;

use ndarray::{ArrayD, IxDyn};
use serde::de::{self, MapAccess, SeqAccess, Visitor};
use serde::ser::SerializeStruct;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::core::Frame;
use crate::core::SimBox;
use crate::core::{Block, Column, DType};
use crate::core::{MetaMap, MetaValue};

// ===== MetaValue ===========================================================

impl Serialize for MetaValue {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.to_json_value().serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for MetaValue {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let value = serde_json::Value::deserialize(deserializer)?;
        MetaValue::from_json_value(&value).map_err(de::Error::custom)
    }
}

// ===== Column ===============================================================

/// Serialize a byte slice via serde's `bytes` (a compact `bin` in MessagePack;
/// an integer array in JSON).
struct RawBytes<'a>(&'a [u8]);

impl Serialize for RawBytes<'_> {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_bytes(self.0)
    }
}

impl Serialize for Column {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let mut st = s.serialize_struct("Column", 3)?;
        st.serialize_field("dtype", self.dtype().name())?;
        st.serialize_field("shape", self.shape())?;
        match self {
            Column::String(h) => {
                let v: Vec<&String> = h.array().iter().collect();
                st.serialize_field("data", &v)?;
            }
            _ => {
                let bytes = self
                    .raw_bytes()
                    .expect("a non-string column always yields raw_bytes");
                st.serialize_field("data", &RawBytes(&bytes))?;
            }
        }
        st.end()
    }
}

/// Decoded column payload, chosen by dtype.
enum ColumnPayload {
    Bytes(Vec<u8>),
    Strings(Vec<String>),
}

/// Untyped payload as it appears on the wire. The dtype field may arrive before
/// or after data, so we defer interpretation until the whole Column map is read.
enum WirePayload {
    Bytes(Vec<u8>),
    Strings(Vec<String>),
}

impl WirePayload {
    fn into_typed(self, dtype: DType) -> Result<ColumnPayload, String> {
        match (dtype, self) {
            (DType::String, WirePayload::Strings(s)) => Ok(ColumnPayload::Strings(s)),
            (DType::String, WirePayload::Bytes(b)) if b.is_empty() => {
                Ok(ColumnPayload::Strings(Vec::new()))
            }
            (DType::String, WirePayload::Bytes(_)) => {
                Err("string column payload must be a string array".to_string())
            }
            (_, WirePayload::Bytes(b)) => Ok(ColumnPayload::Bytes(b)),
            (_, WirePayload::Strings(_)) => {
                Err("numeric column payload must be a byte buffer".to_string())
            }
        }
    }
}

enum WireElement {
    Byte(u8),
    String(String),
}

impl<'de> Deserialize<'de> for WireElement {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct ElemVisitor;
        impl Visitor<'_> for ElemVisitor {
            type Value = WireElement;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("a byte value or a string")
            }
            fn visit_u64<E: de::Error>(self, v: u64) -> Result<WireElement, E> {
                u8::try_from(v)
                    .map(WireElement::Byte)
                    .map_err(|_| de::Error::custom(format!("byte value out of range: {v}")))
            }
            fn visit_i64<E: de::Error>(self, v: i64) -> Result<WireElement, E> {
                u8::try_from(v)
                    .map(WireElement::Byte)
                    .map_err(|_| de::Error::custom(format!("byte value out of range: {v}")))
            }
            fn visit_str<E: de::Error>(self, v: &str) -> Result<WireElement, E> {
                Ok(WireElement::String(v.to_string()))
            }
            fn visit_string<E: de::Error>(self, v: String) -> Result<WireElement, E> {
                Ok(WireElement::String(v))
            }
        }
        d.deserialize_any(ElemVisitor)
    }
}

/// Accept numeric payloads as MessagePack `bin` (`visit_bytes`/`visit_byte_buf`)
/// or JSON integer arrays, and string payloads as string arrays.
struct WirePayloadVisitor;

impl<'de> Visitor<'de> for WirePayloadVisitor {
    type Value = WirePayload;
    fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.write_str("a byte buffer or a string array")
    }
    fn visit_bytes<E: de::Error>(self, v: &[u8]) -> Result<WirePayload, E> {
        Ok(WirePayload::Bytes(v.to_vec()))
    }
    fn visit_byte_buf<E: de::Error>(self, v: Vec<u8>) -> Result<WirePayload, E> {
        Ok(WirePayload::Bytes(v))
    }
    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<WirePayload, A::Error> {
        let Some(first) = seq.next_element::<WireElement>()? else {
            return Ok(WirePayload::Bytes(Vec::new()));
        };
        match first {
            WireElement::Byte(b) => {
                let mut out = Vec::with_capacity(seq.size_hint().unwrap_or(0) + 1);
                out.push(b);
                while let Some(next) = seq.next_element::<WireElement>()? {
                    match next {
                        WireElement::Byte(b) => out.push(b),
                        WireElement::String(_) => {
                            return Err(de::Error::custom(
                                "mixed string and byte values in column payload",
                            ));
                        }
                    }
                }
                Ok(WirePayload::Bytes(out))
            }
            WireElement::String(s) => {
                let mut out = Vec::with_capacity(seq.size_hint().unwrap_or(0) + 1);
                out.push(s);
                while let Some(next) = seq.next_element::<WireElement>()? {
                    match next {
                        WireElement::String(s) => out.push(s),
                        WireElement::Byte(_) => {
                            return Err(de::Error::custom(
                                "mixed string and byte values in column payload",
                            ));
                        }
                    }
                }
                Ok(WirePayload::Strings(out))
            }
        }
    }
}

impl<'de> Deserialize<'de> for WirePayload {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        d.deserialize_any(WirePayloadVisitor)
    }
}

impl<'de> Deserialize<'de> for Column {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Column, D::Error> {
        struct ColumnVisitor;
        impl<'de> Visitor<'de> for ColumnVisitor {
            type Value = Column;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("a column map {dtype, shape, data}")
            }
            fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Column, A::Error> {
                let mut dtype: Option<DType> = None;
                let mut shape: Option<Vec<usize>> = None;
                let mut data: Option<WirePayload> = None;
                while let Some(key) = map.next_key::<String>()? {
                    match key.as_str() {
                        "dtype" => {
                            let tag: String = map.next_value()?;
                            dtype = Some(DType::from_name(&tag).ok_or_else(|| {
                                de::Error::custom(format!("unknown dtype {tag:?}"))
                            })?);
                        }
                        "shape" => shape = Some(map.next_value()?),
                        "data" => data = Some(map.next_value()?),
                        _ => {
                            let _: de::IgnoredAny = map.next_value()?;
                        }
                    }
                }
                let dtype = dtype.ok_or_else(|| de::Error::missing_field("dtype"))?;
                let shape = shape.ok_or_else(|| de::Error::missing_field("shape"))?;
                let data = data
                    .ok_or_else(|| de::Error::missing_field("data"))?
                    .into_typed(dtype)
                    .map_err(de::Error::custom)?;
                build_column(dtype, &shape, data).map_err(de::Error::custom)
            }
        }
        d.deserialize_struct("Column", &["dtype", "shape", "data"], ColumnVisitor)
    }
}

fn build_column(dtype: DType, shape: &[usize], data: ColumnPayload) -> Result<Column, String> {
    let n: usize = shape.iter().product();
    let ix = IxDyn(shape);
    let shape_err = |ty: &str| format!("{ty} column: element count does not match shape {shape:?}");
    match (dtype, data) {
        (DType::Float, ColumnPayload::Bytes(b)) => {
            if b.len() != n * 8 {
                return Err(format!("float column: {} byte(s) is not {n} × 8", b.len()));
            }
            let v = le::<8, _>(&b, n, f64::from_le_bytes)?;
            Ok(Column::from(
                ArrayD::from_shape_vec(ix, v).map_err(|e| e.to_string())?,
            ))
        }
        (DType::Int, ColumnPayload::Bytes(b)) => {
            let v = le::<4, _>(&b, n, i32::from_le_bytes)?;
            Ok(Column::from(
                ArrayD::from_shape_vec(ix, v).map_err(|e| e.to_string())?,
            ))
        }
        (DType::I8, ColumnPayload::Bytes(b)) => {
            if b.len() != n {
                return Err(shape_err("i8"));
            }
            let v: Vec<i8> = b.iter().map(|&x| x as i8).collect();
            Ok(Column::from(
                ArrayD::from_shape_vec(ix, v).map_err(|e| e.to_string())?,
            ))
        }
        (DType::I16, ColumnPayload::Bytes(b)) => {
            let v = le::<2, _>(&b, n, i16::from_le_bytes)?;
            Ok(Column::from(
                ArrayD::from_shape_vec(ix, v).map_err(|e| e.to_string())?,
            ))
        }
        (DType::I64, ColumnPayload::Bytes(b)) => {
            let v = le::<8, _>(&b, n, i64::from_le_bytes)?;
            Ok(Column::from(
                ArrayD::from_shape_vec(ix, v).map_err(|e| e.to_string())?,
            ))
        }
        (DType::Uint, ColumnPayload::Bytes(b)) => {
            let v = le::<8, _>(&b, n, u64::from_le_bytes)?;
            Ok(Column::from(
                ArrayD::from_shape_vec(ix, v).map_err(|e| e.to_string())?,
            ))
        }
        (DType::U8, ColumnPayload::Bytes(b)) => {
            if b.len() != n {
                return Err(shape_err("u8"));
            }
            Ok(Column::from(
                ArrayD::from_shape_vec(ix, b).map_err(|e| e.to_string())?,
            ))
        }
        (DType::U16, ColumnPayload::Bytes(b)) => {
            let v = le::<2, _>(&b, n, u16::from_le_bytes)?;
            Ok(Column::from(
                ArrayD::from_shape_vec(ix, v).map_err(|e| e.to_string())?,
            ))
        }
        (DType::U32, ColumnPayload::Bytes(b)) => {
            let v = le::<4, _>(&b, n, u32::from_le_bytes)?;
            Ok(Column::from(
                ArrayD::from_shape_vec(ix, v).map_err(|e| e.to_string())?,
            ))
        }
        (DType::Bool, ColumnPayload::Bytes(b)) => {
            if b.len() != n {
                return Err(shape_err("bool"));
            }
            let v: Vec<bool> = b.iter().map(|&x| x != 0).collect();
            Ok(Column::from(
                ArrayD::from_shape_vec(ix, v).map_err(|e| e.to_string())?,
            ))
        }
        (DType::String, ColumnPayload::Strings(s)) => {
            if s.len() != n {
                return Err(shape_err("string"));
            }
            Ok(Column::from(
                ArrayD::from_shape_vec(ix, s).map_err(|e| e.to_string())?,
            ))
        }
        (DType::C64, ColumnPayload::Bytes(b)) => {
            let v = le::<8, _>(&b, n, |bytes| {
                num_complex::Complex::<f32>::new(
                    f32::from_le_bytes(bytes[..4].try_into().unwrap()),
                    f32::from_le_bytes(bytes[4..].try_into().unwrap()),
                )
            })?;
            Ok(Column::from(
                ArrayD::from_shape_vec(ix, v).map_err(|e| e.to_string())?,
            ))
        }
        (DType::C128, ColumnPayload::Bytes(b)) => {
            let v = le::<16, _>(&b, n, |bytes| {
                num_complex::Complex::<f64>::new(
                    f64::from_le_bytes(bytes[..8].try_into().unwrap()),
                    f64::from_le_bytes(bytes[8..].try_into().unwrap()),
                )
            })?;
            Ok(Column::from(
                ArrayD::from_shape_vec(ix, v).map_err(|e| e.to_string())?,
            ))
        }
        _ => Err("dtype does not match its data payload".to_string()),
    }
}

/// Parse `n` little-endian values of `N` bytes each from `bytes`.
fn le<const N: usize, T>(
    bytes: &[u8],
    n: usize,
    read: impl Fn([u8; N]) -> T,
) -> Result<Vec<T>, String> {
    if bytes.len() != n * N {
        return Err(format!(
            "numeric payload has {} byte(s), expected {}",
            bytes.len(),
            n * N
        ));
    }
    Ok(bytes.as_chunks::<N>().0.iter().map(|c| read(*c)).collect())
}

// ===== Block ================================================================

impl Serialize for Block {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        // A validity mask is part of the column's data: dropping it here would
        // turn "this cell holds nothing" into the filled default on the far
        // side. The field is written only when some column carries a mask, so
        // a payload from a block with none is byte-for-byte what it was before
        // nullable columns existed — and a payload written then still reads,
        // because an absent `validity` means no masks.
        let masks: IndexMap<&str, &[bool]> = self
            .keys()
            .filter_map(|key| self.validity(key).map(|mask| (key, mask)))
            .collect();
        let mut st = s.serialize_struct("Block", if masks.is_empty() { 2 } else { 3 })?;
        st.serialize_field("shape", &self.shape())?;
        let columns: IndexMap<&str, &Column> = self.iter().collect();
        st.serialize_field("columns", &columns)?;
        if !masks.is_empty() {
            st.serialize_field("validity", &masks)?;
        }
        st.end()
    }
}

#[derive(Deserialize)]
struct BlockRepr {
    #[serde(default)]
    shape: Vec<usize>,
    #[serde(default)]
    columns: IndexMap<String, Column>,
    #[serde(default)]
    validity: IndexMap<String, Vec<bool>>,
}

impl<'de> Deserialize<'de> for Block {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Block, D::Error> {
        let r = BlockRepr::deserialize(d)?;
        let shape = r.shape;
        let has_columns = !r.columns.is_empty();
        let mut block = Block::new();
        for (name, col) in r.columns {
            block.insert(name, col, None).map_err(de::Error::custom)?;
        }
        // Restore an N-D structural shape (e.g. a volumetric block); a plain
        // table's `[count]` is already established by the inserts, but an empty
        // schema-only table needs its explicit row count restored.
        if !shape.is_empty() && (!has_columns || shape.len() > 1) {
            block.set_shape(&shape).map_err(de::Error::custom)?;
        }
        // Masks are restored after the columns, which is what `set_validity`
        // exists for: the decoder holds `Column` values, not typed arrays.
        for (name, mask) in r.validity {
            block.set_validity(&name, mask).map_err(de::Error::custom)?;
        }
        Ok(block)
    }
}

// ===== SimBox ===============================================================

impl Serialize for SimBox {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let o = self.origin();
        let mut st = s.serialize_struct("SimBox", 4)?;
        st.serialize_field("vectors", &self.matrix())?;
        st.serialize_field("origin", &[o[0], o[1], o[2]])?;
        st.serialize_field("boundary", &self.pbc())?;
        st.serialize_field("cell_defined", &self.is_cell_defined())?;
        st.end()
    }
}

#[derive(Deserialize)]
struct SimBoxRepr {
    vectors: [[f64; 3]; 3],
    origin: [f64; 3],
    boundary: [bool; 3],
    cell_defined: bool,
}

impl<'de> Deserialize<'de> for SimBox {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<SimBox, D::Error> {
        let r = SimBoxRepr::deserialize(d)?;
        SimBox::new(
            if r.cell_defined {
                Cell::Matrix(r.vectors)
            } else {
                Cell::Undefined
            },
            r.origin,
            r.boundary,
        )
        .map_err(|e| de::Error::custom(format!("{e:?}")))
    }
}

// ===== Frame ================================================================

impl Serialize for Frame {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let has_box = self.simbox.is_some();
        let mut st = s.serialize_struct("Frame", 2 + has_box as usize)?;
        let blocks: IndexMap<&str, &Block> = self.iter().collect();
        st.serialize_field("blocks", &blocks)?;
        let meta: IndexMap<&str, &MetaValue> =
            self.meta.iter().map(|(k, v)| (k.as_str(), v)).collect();
        st.serialize_field("meta", &meta)?;
        if let Some(sb) = &self.simbox {
            st.serialize_field("box", sb)?;
        }
        st.end()
    }
}

#[derive(Deserialize)]
struct FrameRepr {
    #[serde(default)]
    blocks: IndexMap<String, Block>,
    #[serde(default)]
    meta: IndexMap<String, MetaValue>,
    #[serde(default, rename = "box")]
    simbox: Option<SimBox>,
}

impl<'de> Deserialize<'de> for Frame {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Frame, D::Error> {
        let r = FrameRepr::deserialize(d)?;
        let mut frame: Frame = r.blocks.into_iter().collect();
        frame.meta = MetaMap::with_capacity(r.meta.len());
        frame.meta.extend(r.meta);
        frame.simbox = r.simbox;
        Ok(frame)
    }
}

#[cfg(test)]
mod tests {
    use crate::core::Block;
    use crate::core::Cell;
    use crate::core::Frame;
    use crate::core::MetaValue;
    use crate::core::SimBox;
    use ndarray::{Array1, array};

    fn atoms() -> Block {
        let mut b = Block::new();
        b.insert(
            "x",
            Array1::from_vec(vec![0.5, -1.25, 3.0]).into_dyn(),
            None,
        )
        .unwrap();
        b.insert("seq", Array1::from_vec(vec![1i32, 2, 3]).into_dyn(), None)
            .unwrap();
        b.insert(
            "name",
            Array1::from_vec(vec!["O".to_string(), "H".to_string(), "Ω".to_string()]).into_dyn(),
            None,
        )
        .unwrap();
        b
    }

    #[test]
    fn a_block_round_trips_every_column_at_its_dtype() {
        let json = serde_json::to_string(&atoms()).unwrap();
        let back: Block = serde_json::from_str(&json).unwrap();
        assert_eq!(back.n_rows(), Some(3));
        assert_eq!(
            back.get("x")
                .and_then(|c| c.get::<f64>())
                .unwrap()
                .as_slice()
                .unwrap(),
            &[0.5, -1.25, 3.0]
        );
        assert_eq!(
            back.get("seq")
                .and_then(|c| c.get::<i32>())
                .unwrap()
                .as_slice()
                .unwrap(),
            &[1, 2, 3]
        );
        assert_eq!(
            back.get("name").and_then(|c| c.get::<String>()).unwrap()[2],
            "Ω"
        );
    }

    #[test]
    fn a_frame_keeps_its_blocks_typed_meta_and_box() {
        let mut frame = Frame::new();
        frame.insert("atoms", atoms());
        frame.meta.insert("timestep", MetaValue::I64(42));
        frame.meta.insert("label", MetaValue::String("run".into()));
        frame.simbox = Some(
            SimBox::new(
                Cell::Lengths([10.0; 3]),
                [1.0, 2.0, 3.0],
                [true, true, false],
            )
            .unwrap(),
        );

        let json = serde_json::to_string(&frame).unwrap();
        // The envelope carries no version field: nothing gates compatibility
        // before 1.0.0.
        assert!(!json.contains("\"version\""));
        let back: Frame = serde_json::from_str(&json).unwrap();
        assert_eq!(back.get("atoms").unwrap().n_rows(), Some(3));
        assert_eq!(back.meta.get("timestep"), Some(&MetaValue::I64(42)));
        assert_eq!(
            back.meta.get("label"),
            Some(&MetaValue::String("run".into()))
        );
        let bx = back.simbox.as_ref().expect("box survives");
        assert_eq!(bx.pbc(), [true, true, false]);
        assert_eq!(bx.origin(), [1.0, 2.0, 3.0]);
        assert_eq!(bx.lengths(), array![10.0, 10.0, 10.0]);
    }

    /// A validity mask is part of the column's data, not a view over it: a
    /// frame that travelled through the transport encoding must still spell
    /// "this cell holds nothing" the same way.
    #[test]
    fn a_nullable_column_keeps_its_validity_mask() {
        let mut block = Block::new();
        block
            .insert(
                "x",
                Array1::from_vec(vec![0.5, 0.0, 3.0]).into_dyn(),
                Some(vec![true, false, true]),
            )
            .unwrap();
        let json = serde_json::to_string(&block).unwrap();
        let back: Block = serde_json::from_str(&json).unwrap();
        assert_eq!(back.validity("x"), Some(&[true, false, true][..]));
    }

    #[test]
    fn a_column_with_a_bad_dtype_tag_is_refused() {
        let json = r#"{"dtype":"quaternion","shape":[1],"data":[0]}"#;
        assert!(serde_json::from_str::<crate::core::Column>(json).is_err());
    }

    /// Column order is the file's order; a round trip must not sort it.
    #[test]
    fn a_block_round_trips_columns_in_insertion_order() {
        let mut block = Block::new();
        for key in ["c", "a", "b"] {
            block
                .insert(key, Array1::from_vec(vec![0.0]).into_dyn(), None)
                .unwrap();
        }
        let json = serde_json::to_string(&block).unwrap();
        let back: Block = serde_json::from_str(&json).unwrap();
        assert_eq!(back.keys().collect::<Vec<_>>(), ["c", "a", "b"]);
    }

    /// Block order is the file's order too.
    #[test]
    fn a_frame_round_trips_blocks_in_insertion_order() {
        let mut frame = Frame::new();
        for key in ["z", "a", "m"] {
            frame.insert(key, Block::new());
        }
        let json = serde_json::to_string(&frame).unwrap();
        let back: Frame = serde_json::from_str(&json).unwrap();
        assert_eq!(back.keys().collect::<Vec<_>>(), ["z", "a", "m"]);
    }

    /// Meta keys are part of the frame document, and a round trip must not
    /// reshuffle them into alphabetical or hash order.
    #[test]
    fn a_frame_round_trips_meta_keys_in_insertion_order() {
        let mut frame = Frame::new();
        frame.meta.insert("z", MetaValue::String("Z".into()));
        frame.meta.insert("a", MetaValue::String("A".into()));
        frame.meta.insert("m", MetaValue::String("M".into()));

        let json = serde_json::to_string(&frame).unwrap();
        let back: Frame = serde_json::from_str(&json).unwrap();
        assert_eq!(
            back.meta.keys().map(String::as_str).collect::<Vec<_>>(),
            vec!["z", "a", "m"]
        );
    }
}
