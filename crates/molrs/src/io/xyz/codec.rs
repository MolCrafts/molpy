//! The XYZ / extended XYZ codec: the comment-line grammar, the reader and
//! writer classes, the path doors.

use crate::core::Cell;
use crate::io::reader::{FrameReader, Reader, TrajectoryReader};
use crate::io::writer::{FrameWriter, Writer};
use crate::io::{Sink, SinkWriter, Source, invalid_data};
use molrs::core::Block;
use molrs::core::Frame;
use molrs::core::FrameAccess;
use molrs::core::MetaValue;
use molrs::core::SimBox;
use molrs::op::{F, I, Idx};
use ndarray::{Array1, Array2, ArrayD};
use std::collections::HashMap;
use std::io::{BufRead, Seek, SeekFrom, Write};
use std::sync::OnceLock;

/// A scalar value of an extended XYZ comment line (string, integer, real or logical)
#[derive(Debug, Clone, PartialEq)]
enum ExtxyzScalar {
    Str(String),
    Int(i64),
    Real(f64),
    Logical(bool),
}

/// An extended XYZ comment value: a scalar, or a 1-D or 2-D array of them
#[derive(Debug, Clone, PartialEq)]
enum ExtxyzValue {
    Scalar(ExtxyzScalar),
    Array1(Vec<ExtxyzScalar>),
    Array2(Vec<Vec<ExtxyzScalar>>),
}

/// The kind of a `Properties` column: `S` (string), `I` (integer), `R` (real), `L` (logical)
#[derive(Debug, Clone, Copy, PartialEq)]
enum ExtxyzPropertyKind {
    String,
    Integer,
    Real,
    Logical,
}

/// Property specification: name, type and multiplicity
#[derive(Debug, Clone, PartialEq)]
struct PropertySpec {
    /// Property name
    pub name: String,
    /// Property type
    pub ty: ExtxyzPropertyKind,
    /// Multiplicity (1 for scalar)
    pub m: usize,
}

/// Parsed XYZ comment line with optional extended fields
#[derive(Debug, Clone, PartialEq)]
struct ExtxyzComment {
    /// Key-value pairs
    pub kv: HashMap<String, ExtxyzValue>,
    /// Parsed properties (from key "Properties"), if present
    pub properties: Option<Vec<PropertySpec>>, // parsed from key "Properties" if present
    /// Original comment line when treated as extxyz
    pub comment: Option<String>, // original comment line when treated as extxyz
    /// True if treated as plain XYZ
    pub is_plain_xyz: bool,
}

fn parse_logical_token(tok: &str) -> Option<bool> {
    match tok.to_ascii_lowercase().as_str() {
        "t" | "true" => Some(true),
        "f" | "false" => Some(false),
        _ => None,
    }
}

fn parse_scalar_token(tok: &str) -> ExtxyzScalar {
    if let Some(b) = parse_logical_token(tok) {
        ExtxyzScalar::Logical(b)
    } else if tok.contains('.') || tok.contains('e') || tok.contains('E') {
        match tok.parse::<f64>() {
            Ok(v) => ExtxyzScalar::Real(v),
            Err(_) => ExtxyzScalar::Str(tok.to_string()),
        }
    } else {
        match tok.parse::<i64>() {
            Ok(v) => ExtxyzScalar::Int(v),
            Err(_) => ExtxyzScalar::Str(tok.to_string()),
        }
    }
}

fn parse_array_from_quoted(s: &str) -> ExtxyzValue {
    // Try 2D using row separators ';' or '|' or comma between rows
    let has_row_sep = s.contains(';') || s.contains('|') || s.contains('\n');
    if has_row_sep {
        let rows: Vec<Vec<ExtxyzScalar>> = s
            .split([';', '|', '\n'])
            .filter(|row| !row.trim().is_empty())
            .map(|row| row.split_whitespace().map(parse_scalar_token).collect())
            .collect();
        return ExtxyzValue::Array2(rows);
    }

    // Try comma-separated values or whitespace-separated
    let elements: Vec<&str> = if s.contains(',') {
        s.split(',').collect()
    } else {
        s.split_whitespace().collect()
    };
    if elements.len() > 1 {
        let values: Vec<ExtxyzScalar> = elements
            .into_iter()
            .map(|t| parse_scalar_token(t.trim()))
            .collect();
        if values
            .iter()
            .all(|value| matches!(value, ExtxyzScalar::Str(_)))
        {
            ExtxyzValue::Scalar(ExtxyzScalar::Str(s.to_string()))
        } else {
            ExtxyzValue::Array1(values)
        }
    } else {
        ExtxyzValue::Scalar(ExtxyzScalar::Str(s.to_string()))
    }
}

fn parse_properties(spec: &str) -> Option<Vec<PropertySpec>> {
    // Expect a colon-separated stream of triplets: name:T:m: name2:T:m: ...
    let parts: Vec<&str> = spec.split(':').filter(|s| !s.is_empty()).collect();
    if parts.len() < 3 || !parts.len().is_multiple_of(3) {
        return None;
    }
    let mut out = Vec::new();
    let mut i = 0;
    while i + 2 < parts.len() {
        let name = parts[i].to_string();
        let ty = match parts[i + 1] {
            "S" => ExtxyzPropertyKind::String,
            "I" => ExtxyzPropertyKind::Integer,
            "R" => ExtxyzPropertyKind::Real,
            "L" => ExtxyzPropertyKind::Logical,
            other => {
                // try tolerate lower-case
                match other.to_ascii_uppercase().as_str() {
                    "S" => ExtxyzPropertyKind::String,
                    "I" => ExtxyzPropertyKind::Integer,
                    "R" => ExtxyzPropertyKind::Real,
                    "L" => ExtxyzPropertyKind::Logical,
                    _ => return None,
                }
            }
        };
        let m = match parts[i + 2].parse::<usize>() {
            Ok(v) if v > 0 => v,
            _ => return None,
        };
        out.push(PropertySpec { name, ty, m });
        i += 3;
    }
    Some(out)
}

fn parse_comment_line(line: &str) -> std::result::Result<ExtxyzComment, String> {
    let original = line.to_string();
    let input = line.trim();

    // Quick check: if no '=' present, treat as plain xyz comment
    if !input.contains('=') {
        let mut kv = HashMap::new();
        kv.insert(
            "comment".to_string(),
            ExtxyzValue::Scalar(ExtxyzScalar::Str(original)),
        );
        return Ok(ExtxyzComment {
            kv,
            properties: None,
            comment: None,
            is_plain_xyz: true,
        });
    }

    let bytes = input.as_bytes();
    let mut idx = 0usize;
    let len = bytes.len();
    let mut kv: HashMap<String, ExtxyzValue> = HashMap::new();
    let mut properties: Option<Vec<PropertySpec>> = None;

    let skip_ws = |pos: &mut usize| {
        while *pos < len && bytes[*pos].is_ascii_whitespace() {
            *pos += 1;
        }
    };

    while idx < len {
        skip_ws(&mut idx);
        if idx >= len {
            break;
        }

        let key = if bytes[idx] == b'"' {
            idx += 1;
            let start = idx;
            while idx < len && bytes[idx] != b'"' {
                idx += 1;
            }
            if idx >= len {
                return Err("unterminated quoted key".to_string());
            }
            let key = input[start..idx].to_string();
            idx += 1;
            key
        } else {
            let start = idx;
            while idx < len && !bytes[idx].is_ascii_whitespace() && bytes[idx] != b'=' {
                idx += 1;
            }
            if start == idx {
                return Err("missing key".to_string());
            }
            input[start..idx].to_string()
        };

        skip_ws(&mut idx);
        if idx >= len || bytes[idx] != b'=' {
            // Bare boolean key (no '=' follows) — valid EXTXYZ, treat as true.
            kv.insert(key, ExtxyzValue::Scalar(ExtxyzScalar::Logical(true)));
            continue;
        }
        idx += 1;
        skip_ws(&mut idx);
        if idx >= len {
            return Err(format!("missing value for key '{key}'"));
        }

        let value = if bytes[idx] == b'"' {
            idx += 1;
            let start = idx;
            while idx < len && bytes[idx] != b'"' {
                idx += 1;
            }
            if idx >= len {
                return Err(format!("unterminated quoted value for key '{key}'"));
            }
            let value_str = &input[start..idx];
            idx += 1;
            parse_array_from_quoted(value_str)
        } else {
            let start = idx;
            while idx < len && !bytes[idx].is_ascii_whitespace() {
                idx += 1;
            }
            let token = &input[start..idx];
            ExtxyzValue::Scalar(parse_scalar_token(token))
        };

        if key.eq_ignore_ascii_case("properties") {
            let spec_str = match &value {
                ExtxyzValue::Scalar(ExtxyzScalar::Str(s)) => s.clone(),
                ExtxyzValue::Array1(vs) => vs
                    .iter()
                    .map(|p| match p {
                        ExtxyzScalar::Str(s) => s.clone(),
                        _ => "".into(),
                    })
                    .collect::<Vec<_>>()
                    .join(" "),
                _ => String::new(),
            };
            properties = parse_properties(&spec_str);
        }
        kv.insert(key, value);
    }

    if properties.is_none() {
        kv.insert(
            "comment".to_string(),
            ExtxyzValue::Scalar(ExtxyzScalar::Str(original)),
        );
        Ok(ExtxyzComment {
            kv,
            properties: None,
            comment: None,
            is_plain_xyz: true,
        })
    } else {
        Ok(ExtxyzComment {
            kv,
            properties,
            comment: Some(original),
            is_plain_xyz: false,
        })
    }
}

/// The extxyz property name of the canonical `res_name` column (ASE's
/// spelling), mapped at the I/O boundary in both directions.
const EXTXYZ_RESNAME: &str = "resname";

/// The extxyz property name of the canonical `element` column; the writer
/// always declares it first (`species:S:1`).
const EXTXYZ_SPECIES: &str = "species";

/// Row `row` of `column` as EXTXYZ property tokens: one per value of the
/// row, logicals as `T` / `F`.
fn row_tokens(column: &molrs::core::ColumnView<'_>, row: usize) -> Vec<String> {
    use ndarray::Axis;
    fn shown<T: std::fmt::Display>(a: &ndarray::ArrayViewD<'_, T>, row: usize) -> Vec<String> {
        a.index_axis(Axis(0), row)
            .iter()
            .map(T::to_string)
            .collect()
    }
    match column {
        molrs::core::ColumnView::Bool(a) => a
            .index_axis(Axis(0), row)
            .iter()
            .map(|v| if *v { "T" } else { "F" }.to_string())
            .collect(),
        molrs::core::ColumnView::Float(a) => shown(a, row),
        molrs::core::ColumnView::I8(a) => shown(a, row),
        molrs::core::ColumnView::I16(a) => shown(a, row),
        molrs::core::ColumnView::Int(a) => shown(a, row),
        molrs::core::ColumnView::I64(a) => shown(a, row),
        molrs::core::ColumnView::U8(a) => shown(a, row),
        molrs::core::ColumnView::U16(a) => shown(a, row),
        molrs::core::ColumnView::U32(a) => shown(a, row),
        molrs::core::ColumnView::Uint(a) => shown(a, row),
        molrs::core::ColumnView::String(a) => shown(a, row),
        molrs::core::ColumnView::C64(a) => shown(a, row),
        molrs::core::ColumnView::C128(a) => shown(a, row),
    }
}

/// The extxyz property a frame column is written as.
fn extxyz_property_name(column: &str) -> &str {
    if column == molrs::core::keys::RES_NAME {
        EXTXYZ_RESNAME
    } else {
        column
    }
}

/// One frame column an extxyz `Properties` entry becomes: its name, the
/// property type, and how many values per row it holds.
struct XyzColumn {
    name: String,
    ty: ExtxyzPropertyKind,
    width: usize,
}

/// The frame columns of a `Properties` declaration, in declaration order.
///
/// Canonical names cross the boundary here, in one place:
///
/// - `pos:R:3` becomes the three columns `x`, `y`, `z`;
/// - `species:S:1` becomes `element` (unless the file also declares an
///   `element` property, which then keeps its own name);
/// - `type:I:1` becomes `type_id` (an ExtXYZ integer type is an ordinal);
/// - `resname` becomes `res_name`;
/// - any other `name:T:m` with `m > 1` is **one** `(N, m)` column `name` — the
///   shape the writer writes back as `name:T:m`.
fn property_columns(props: &[PropertySpec]) -> Vec<XyzColumn> {
    use molrs::core::schema::consts;
    let declares_element = props.iter().any(|p| p.name == consts::ELEMENT);
    let mut cols = Vec::new();
    for p in props {
        if p.m == 3 && p.ty == ExtxyzPropertyKind::Real && p.name.eq_ignore_ascii_case("pos") {
            for axis in [consts::X, consts::Y, consts::Z] {
                cols.push(XyzColumn {
                    name: axis.to_string(),
                    ty: ExtxyzPropertyKind::Real,
                    width: 1,
                });
            }
            continue;
        }
        let name = if p.m != 1 {
            p.name.clone()
        } else if p.name.eq_ignore_ascii_case("type") && p.ty == ExtxyzPropertyKind::Integer {
            consts::TYPE_ID.to_string()
        } else if p.name == EXTXYZ_RESNAME {
            consts::RES_NAME.to_string()
        } else if p.name == EXTXYZ_SPECIES
            && p.ty == ExtxyzPropertyKind::String
            && !declares_element
        {
            consts::ELEMENT.to_string()
        } else {
            p.name.clone()
        };
        cols.push(XyzColumn {
            name,
            ty: p.ty,
            width: p.m.max(1),
        });
    }
    cols
}

fn parse_bool_token(tok: &str) -> Option<bool> {
    match tok.to_ascii_lowercase().as_str() {
        "t" | "true" => Some(true),
        "f" | "false" => Some(false),
        _ => None,
    }
}

fn line_to_tokens(line: &str) -> Vec<&str> {
    line.split_whitespace().collect()
}

/// Build schema from parsed properties
fn build_complete_schema(ec: &ExtxyzComment) -> Vec<PropertySpec> {
    // If no Properties key, return plain XYZ schema (4 columns: element, x, y, z)
    // Otherwise, return the properties as-is
    ec.properties.clone().unwrap_or_else(|| {
        vec![
            PropertySpec {
                name: "element".into(),
                ty: ExtxyzPropertyKind::String,
                m: 1,
            },
            PropertySpec {
                name: "x".into(),
                ty: ExtxyzPropertyKind::Real,
                m: 1,
            },
            PropertySpec {
                name: "y".into(),
                ty: ExtxyzPropertyKind::Real,
                m: 1,
            },
            PropertySpec {
                name: "z".into(),
                ty: ExtxyzPropertyKind::Real,
                m: 1,
            },
        ]
    })
}

fn build_block_from_props(
    n: usize,
    lines: &[String],
    props: &[PropertySpec],
) -> Result<Block, String> {
    let cols = property_columns(props);
    let m_total: usize = cols.iter().map(|c| c.width).sum();
    if lines.len() != n {
        return Err("insufficient atom lines".into());
    }

    // One buffer per column, `width` values per row, row-major.
    enum ColBuf {
        S(Vec<String>),
        I(Vec<I>),
        R(Vec<F>),
        L(Vec<bool>),
    }
    let mut buffers: Vec<ColBuf> = cols
        .iter()
        .map(|c| match c.ty {
            ExtxyzPropertyKind::String => ColBuf::S(Vec::with_capacity(n * c.width)),
            ExtxyzPropertyKind::Integer => ColBuf::I(Vec::with_capacity(n * c.width)),
            ExtxyzPropertyKind::Real => ColBuf::R(Vec::with_capacity(n * c.width)),
            ExtxyzPropertyKind::Logical => ColBuf::L(Vec::with_capacity(n * c.width)),
        })
        .collect();

    for (row_i, line) in lines.iter().enumerate() {
        let toks = line_to_tokens(line);
        if toks.len() < m_total {
            return Err(format!(
                "line {}: expected at least {} tokens, got {}",
                row_i,
                m_total,
                toks.len()
            ));
        }
        let mut tok_idx = 0;
        for (col, buf) in cols.iter().zip(buffers.iter_mut()) {
            for _ in 0..col.width {
                let tok = toks[tok_idx];
                match buf {
                    ColBuf::S(v) => v.push(tok.to_string()),
                    ColBuf::I(v) => v.push(tok.parse::<I>().map_err(|_| {
                        format!("line {} col {}: invalid int '{}", row_i, tok_idx, tok)
                    })?),
                    ColBuf::R(v) => v.push(tok.parse::<F>().map_err(|_| {
                        format!("line {} col {}: invalid float '{}", row_i, tok_idx, tok)
                    })?),
                    ColBuf::L(v) => v.push(parse_bool_token(tok).ok_or_else(|| {
                        format!("line {} col {}: invalid bool '{}", row_i, tok_idx, tok)
                    })?),
                }
                tok_idx += 1;
            }
        }
    }

    /// `(n,)` for a one-value column, `(n, width)` for a wider one.
    fn shaped<T>(values: Vec<T>, n: usize, width: usize) -> Result<ArrayD<T>, String> {
        if width == 1 {
            Ok(Array1::from_vec(values).into_dyn())
        } else {
            Array2::from_shape_vec((n, width), values)
                .map(|a| a.into_dyn())
                .map_err(|e| e.to_string())
        }
    }

    let mut block = Block::new();
    for (XyzColumn { name, width, .. }, buf) in cols.into_iter().zip(buffers) {
        match buf {
            ColBuf::I(v) => {
                // Extended-XYZ declares `I` for any integral column, but a
                // canonical key's dtype is fixed by the vocabulary — `id` is
                // unsigned there, and an Int column under that name would be
                // invisible to every consumer reading it as unsigned.
                if molrs::core::schema::column(&name).map(|c| c.dtype)
                    == Some(molrs::core::DType::Uint)
                {
                    let unsigned: Vec<molrs::op::Idx> = v
                        .iter()
                        .map(|&x| {
                            Idx::try_from(x).map_err(|_| {
                                format!("column '{name}' is unsigned in the Frame schema, got {x}")
                            })
                        })
                        .collect::<Result<_, String>>()?;
                    let arr = shaped(unsigned, n, width)?;
                    block.insert(name, arr, None).map_err(|e| e.to_string())?;
                } else {
                    let arr = shaped(v, n, width)?;
                    block.insert(name, arr, None).map_err(|e| e.to_string())?;
                }
            }
            ColBuf::R(v) => {
                let arr = shaped(v, n, width)?;
                block.insert(name, arr, None).map_err(|e| e.to_string())?;
            }
            ColBuf::L(v) => {
                let arr = shaped(v, n, width)?;
                block.insert(name, arr, None).map_err(|e| e.to_string())?;
            }
            ColBuf::S(v) => {
                let arr = shaped(v, n, width)?;
                block.insert(name, arr, None).map_err(|e| e.to_string())?;
            }
        }
    }

    Ok(block)
}

/// One number of a comment-line vector value, or a refusal naming it.
fn vector_number(name: &str, scalar: &ExtxyzScalar) -> std::io::Result<F> {
    match scalar {
        ExtxyzScalar::Real(r) => Ok(*r as F),
        ExtxyzScalar::Int(i) => Ok(*i as F),
        ExtxyzScalar::Str(s) => s
            .parse::<F>()
            .map_err(|_| invalid_data(format!("{name} holds `{s}`, not a number"))),
        ExtxyzScalar::Logical(b) => Err(invalid_data(format!("{name} holds `{b}`, not a number"))),
    }
}

/// The `n` numbers of a comment-line vector value (`Lattice`, `Origin`). A
/// value the line states is read whole or refused by name — an element that
/// is not a number, a count other than `n` — never read as "no box".
fn vector_values(name: &str, v: &ExtxyzValue, n: usize) -> std::io::Result<Vec<F>> {
    let mut values = Vec::with_capacity(n);
    match v {
        ExtxyzValue::Array1(vals) => {
            for scalar in vals {
                values.push(vector_number(name, scalar)?);
            }
        }
        ExtxyzValue::Scalar(ExtxyzScalar::Str(s)) => {
            for tok in s.split_whitespace() {
                values
                    .push(tok.parse::<F>().map_err(|_| {
                        invalid_data(format!("{name} holds `{tok}`, not a number"))
                    })?);
            }
        }
        ExtxyzValue::Scalar(scalar) => values.push(vector_number(name, scalar)?),
        ExtxyzValue::Array2(_) => {
            return Err(invalid_data(format!(
                "{name} must be a flat list of {n} numbers, not a 2-D array"
            )));
        }
    }
    if values.len() != n {
        return Err(invalid_data(format!(
            "{name} must hold {n} numbers, got {}",
            values.len()
        )));
    }
    Ok(values)
}

/// Parse the MolCrafts `Connct` XYZ comment extension.
///
/// The value is a flat, zero-based list of atom-index pairs, for example
/// `Connct="[0,1,0,2]"` describes bonds 0-1 and 0-2. Bond order is implicitly
/// one. Brackets are required by the public convention but are accepted
/// leniently here so older hand-written inputs remain readable.
fn parse_connct(value: &ExtxyzValue, n_atoms: usize) -> Result<Vec<(Idx, Idx)>, String> {
    fn append_primitive(raw: &mut String, value: &ExtxyzScalar) -> Result<(), String> {
        if !raw.is_empty() {
            raw.push(',');
        }
        match value {
            ExtxyzScalar::Int(value) => raw.push_str(&value.to_string()),
            ExtxyzScalar::Str(value) => raw.push_str(value),
            ExtxyzScalar::Real(_) | ExtxyzScalar::Logical(_) => {
                return Err("Connct accepts integer atom indices only".to_string());
            }
        }
        Ok(())
    }

    let mut raw = String::new();
    match value {
        ExtxyzValue::Scalar(value) => append_primitive(&mut raw, value)?,
        ExtxyzValue::Array1(values) => {
            for value in values {
                append_primitive(&mut raw, value)?;
            }
        }
        ExtxyzValue::Array2(_) => {
            return Err("Connct must be a flat list of atom indices".to_string());
        }
    }

    let indices = raw
        .split(|ch: char| ch == '[' || ch == ']' || ch == ',' || ch.is_whitespace())
        .filter(|token| !token.is_empty())
        .map(|token| {
            token
                .parse::<Idx>()
                .map_err(|_| format!("Connct contains invalid atom index '{token}'"))
        })
        .collect::<Result<Vec<_>, _>>()?;

    if !indices.len().is_multiple_of(2) {
        return Err(format!(
            "Connct requires atom-index pairs, but received {} indices",
            indices.len()
        ));
    }

    let mut pairs = Vec::with_capacity(indices.len() / 2);
    for pair in indices.as_chunks::<2>().0 {
        let (atomi, atomj) = (pair[0], pair[1]);
        if atomi as usize >= n_atoms || atomj as usize >= n_atoms {
            return Err(format!(
                "Connct bond {atomi}-{atomj} is out of range for {n_atoms} atoms"
            ));
        }
        pairs.push((atomi, atomj));
    }
    Ok(pairs)
}

fn connct_block(value: &ExtxyzValue, n_atoms: usize) -> Result<Option<Block>, String> {
    let pairs = parse_connct(value, n_atoms)?;
    if pairs.is_empty() {
        return Ok(None);
    }

    let (atomi, atomj): (Vec<Idx>, Vec<Idx>) = pairs.into_iter().unzip();
    let mut block = Block::new();
    block
        .insert("atomi", Array1::from_vec(atomi).into_dyn(), None)
        .map_err(|error| error.to_string())?;
    block
        .insert("atomj", Array1::from_vec(atomj).into_dyn(), None)
        .map_err(|error| error.to_string())?;
    Ok(Some(block))
}

/// Build a SimBox from Lattice + optional Origin ExtValues.
///
/// `Origin` defaults to `[0, 0, 0]` when absent, matching the extxyz convention.
///
/// # Errors
///
/// `InvalidData` for a `Lattice` or `Origin` that does not read as its
/// numbers ([`vector_values`]), and for lattice vectors no cell has.
fn parse_simbox(lattice: &ExtxyzValue, origin: Option<&ExtxyzValue>) -> std::io::Result<SimBox> {
    let h_vals = vector_values("Lattice", lattice, 9)?;
    // extxyz lists the three lattice vectors one after another (R1 R2 R3);
    // `SimBox` keeps them as the *columns* of H, so the row-major reshape is
    // transposed.
    let h = std::array::from_fn(|r| std::array::from_fn(|c| h_vals[3 * c + r]));
    let origin = match origin {
        Some(o) => {
            let v = vector_values("Origin", o, 3)?;
            [v[0], v[1], v[2]]
        }
        None => [0.0; 3],
    };
    SimBox::new(Cell::Matrix(h), origin, [true, true, true]).map_err(|e| {
        invalid_data(format!(
            "Lattice={h_vals:?}: no cell has these lattice vectors ({e:?})"
        ))
    })
}

/// Read one XYZ/EXTXYZ frame from the current position of a buffered reader.
/// Returns Ok(None) on EOF before the first line.
fn read_frame_from<R: BufRead>(reader: &mut R) -> std::io::Result<Option<Frame>> {
    // Read first non-empty line as atom count
    let mut line = String::new();
    let n = loop {
        line.clear();
        let bytes = reader.read_line(&mut line)?;
        if bytes == 0 {
            return Ok(None); // EOF
        }
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        match trimmed.parse::<usize>() {
            Ok(v) => break v,
            Err(_) => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("invalid atom count line: {}", trimmed),
                ));
            }
        }
    };

    // Read comment line (can be empty)
    line.clear();
    let _ = reader.read_line(&mut line)?; // if EOF after count, it's malformed but we allow empty
    let comment = line.trim_end_matches(['\r', '\n']);

    // Parse comment to metadata and properties
    let ec = parse_comment_line(comment)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    let mut kv_meta: HashMap<String, ExtxyzValue> = HashMap::new();
    let mut lattice_value: Option<ExtxyzValue> = None;
    let mut origin_value: Option<ExtxyzValue> = None;
    let mut connct_value: Option<ExtxyzValue> = None;
    for (k, v) in ec.kv.iter() {
        if k.eq_ignore_ascii_case("Properties") {
            continue;
        }
        if k.eq_ignore_ascii_case("Lattice") {
            lattice_value = Some(v.clone());
            continue;
        }
        if k.eq_ignore_ascii_case("Origin") {
            origin_value = Some(v.clone());
            continue;
        }
        if k.eq_ignore_ascii_case("Connct") {
            connct_value = Some(v.clone());
            continue;
        }
        kv_meta.insert(k.clone(), v.clone());
    }

    // Read N atom lines
    let mut atom_lines: Vec<String> = Vec::with_capacity(n);
    for _ in 0..n {
        line.clear();
        let bytes = reader.read_line(&mut line)?;
        if bytes == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "unexpected EOF in atom lines",
            ));
        }
        atom_lines.push(line.trim_end_matches(['\r', '\n']).to_string());
    }

    // Build complete schema (base properties + derived columns)
    let schema = build_complete_schema(&ec);

    // Parse columns according to schema -> atoms block
    let atoms_block = build_block_from_props(n, &atom_lines, &schema)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

    let mut frame = Frame::new();
    frame.insert("atoms", atoms_block);
    if let Some(ref connct) = connct_value
        && let Some(bonds) = connct_block(connct, n)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?
    {
        frame.insert("bonds", bonds);
    }
    for (k, v) in kv_meta.into_iter() {
        frame.meta.insert(
            k,
            ext_value_to_meta(v)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?,
        );
    }

    // Build SimBox from Lattice + optional Origin
    if let Some(ref lat) = lattice_value {
        frame.simbox = Some(parse_simbox(lat, origin_value.as_ref())?);
    }

    // Every door reads through here, so every door checks here: a frame that
    // violates the vocabulary is a malformed file or a reader bug.
    crate::io::reader::check_read_frame(&frame)?;
    Ok(Some(frame))
}

/// One frame of XYZ text through the stream parser — the tests' shorthand.
#[cfg(test)]
fn parse_frame_str(s: &str) -> std::result::Result<Frame, String> {
    read_frame_from(&mut s.as_bytes())
        .map_err(|e| e.to_string())?
        .ok_or_else(|| "no frame in the text".to_owned())
}

fn ext_value_to_meta(value: ExtxyzValue) -> std::result::Result<MetaValue, String> {
    match value {
        ExtxyzValue::Scalar(ExtxyzScalar::Str(value)) => Ok(MetaValue::String(value)),
        ExtxyzValue::Scalar(ExtxyzScalar::Int(value)) => Ok(MetaValue::I64(value)),
        ExtxyzValue::Scalar(ExtxyzScalar::Real(value)) => Ok(MetaValue::F64(value)),
        ExtxyzValue::Scalar(ExtxyzScalar::Logical(value)) => Ok(MetaValue::Bool(value)),
        ExtxyzValue::Array2(rows) => ext_array_to_meta(rows.into_iter().flatten().collect()),
        ExtxyzValue::Array1(values) => ext_array_to_meta(values),
    }
}

fn ext_array_to_meta(values: Vec<ExtxyzScalar>) -> std::result::Result<MetaValue, String> {
    if values.len() == 3 && values.iter().all(|v| matches!(v, ExtxyzScalar::Logical(_))) {
        let values: Vec<bool> = values
            .into_iter()
            .map(|v| match v {
                ExtxyzScalar::Logical(v) => v,
                _ => unreachable!(),
            })
            .collect();
        return Ok(MetaValue::Bool3(values.try_into().unwrap()));
    }
    if values.len() == 3 && values.iter().all(|v| matches!(v, ExtxyzScalar::Int(_))) {
        let values: Vec<i64> = values
            .into_iter()
            .map(|v| match v {
                ExtxyzScalar::Int(v) => v,
                _ => unreachable!(),
            })
            .collect();
        return Ok(MetaValue::I64x3(values.try_into().unwrap()));
    }
    if matches!(values.len(), 3 | 6 | 9)
        && values.iter().all(|v| matches!(v, ExtxyzScalar::Real(_)))
    {
        let values: Vec<f64> = values
            .into_iter()
            .map(|v| match v {
                ExtxyzScalar::Real(v) => v,
                _ => unreachable!(),
            })
            .collect();
        return Ok(match values.len() {
            3 => MetaValue::F64x3(values.try_into().unwrap()),
            6 => MetaValue::F64x6(values.try_into().unwrap()),
            9 => MetaValue::F64x9(values.try_into().unwrap()),
            _ => unreachable!(),
        });
    }
    Err(format!(
        "extended XYZ metadata array has unsupported type/length {}; expected a homogeneous numeric vector of length 3, 6, or 9, or bool[3]",
        values.len()
    ))
}

fn meta_to_extxyz(value: &MetaValue) -> String {
    macro_rules! joined {
        ($values:expr) => {
            $values
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(" ")
        };
    }
    let raw = match value {
        MetaValue::Bool(v) => v.to_string(),
        MetaValue::I32(v) => v.to_string(),
        MetaValue::I64(v) => v.to_string(),
        MetaValue::U32(v) => v.to_string(),
        MetaValue::U64(v) => v.to_string(),
        MetaValue::F64(v) => v.to_string(),
        MetaValue::String(v) => v.clone(),
        MetaValue::Bool3(v) => joined!(v),
        MetaValue::I32x3(v) => joined!(v),
        MetaValue::I64x3(v) => joined!(v),
        MetaValue::U32x3(v) => joined!(v),
        MetaValue::U64x3(v) => joined!(v),
        MetaValue::F64x3(v) => joined!(v),
        MetaValue::F64x6(v) => joined!(v),
        MetaValue::F64x9(v) => joined!(v),
        MetaValue::Json(v) => v.to_string(),
    };
    if raw.contains(char::is_whitespace) {
        format!("\"{raw}\"")
    } else {
        raw
    }
}

// =============== XyzReader ===============

/// Unified XYZ/ExtXYZ reader treating all files as trajectories
///
/// Single-frame files are treated as 1-step trajectories. This reader
/// implements lazy indexing: the first `read_frame(0)` call reads immediately
/// without scanning the file, while accessing later frames triggers index
/// building for efficient random access.
///
/// # Examples
///
/// ```no_run
/// use molrs::io::xyz::XyzReader;
/// use molrs::io::reader::TrajectoryReader;
/// use std::path::Path;
///
/// # fn main() -> std::io::Result<()> {
/// let mut reader = XyzReader::open(Path::new("trajectory.xyz"))?;
///
/// // Read first frame (no indexing)
/// if let Some(frame) = reader.read_frame(0)? {
///     println!("First frame loaded");
/// }
///
/// // Read frame 5 (triggers indexing)
/// if let Some(frame) = reader.read_frame(5)? {
///     println!("Frame 5 loaded");
/// }
///
/// // Iterate all frames
/// for result in reader.iter() {
///     let frame = result?;
///     // Process frame
/// }
/// # Ok(())
/// # }
/// ```
pub struct XyzReader<R: BufRead> {
    reader: R,
    index: OnceLock<Vec<FrameOffset>>,
}

impl XyzReader<Box<dyn crate::io::reader::ReadSeek>> {
    /// Open an XYZ file, or the bytes of one, for random access by frame.
    pub fn open<'a>(src: impl Into<Source<'a>>) -> std::io::Result<Self> {
        Ok(Self::new(src.into().owned_reader()?))
    }
}

impl<R: BufRead + Seek> XyzReader<R> {
    /// Create a new XYZ reader from a buffered reader
    pub fn new(reader: R) -> Self {
        Self {
            reader,
            index: OnceLock::new(),
        }
    }

    /// Index the file's frames ([`XyzIndexBuilder`], the one XYZ indexer).
    fn build_index(&mut self) -> std::io::Result<()> {
        if self.index.get().is_some() {
            return Ok(());
        }
        let offsets = crate::io::frame_index::index_stream(
            &mut self.reader,
            Box::new(XyzIndexBuilder::new()),
        )?;
        self.index
            .set(offsets)
            .map_err(|_| std::io::Error::other("failed to set index"))
    }

    /// Read frame at specific offset
    fn read_at_offset(&mut self, offset: u64) -> std::io::Result<Option<Frame>> {
        self.reader.seek(SeekFrom::Start(offset))?;
        read_frame_from(&mut self.reader)
    }
}

impl<R: BufRead + Seek> Reader for XyzReader<R> {
    type R = R;

    fn new(reader: Self::R) -> Self {
        Self {
            reader,
            index: OnceLock::new(),
        }
    }
}

impl<R: BufRead + Seek> FrameReader for XyzReader<R> {
    fn read(&mut self) -> std::io::Result<Option<Frame>> {
        // Always read the first frame from the current stream position.
        read_frame_from(&mut self.reader)
    }
}

impl<R: BufRead + Seek> TrajectoryReader for XyzReader<R> {
    fn build_index(&mut self) -> std::io::Result<()> {
        self.build_index()
    }

    fn read_frame(&mut self, index: usize) -> std::io::Result<Option<Frame>> {
        // Fast path for first frame: read immediately without indexing
        if index == 0
            && self.index.get().is_none()
            && let Ok(start_pos) = self.reader.stream_position()
        {
            let frame = read_frame_from(&mut self.reader)?;
            self.reader.seek(SeekFrom::Start(start_pos))?;
            return Ok(frame);
        }

        if self.index.get().is_none() {
            self.build_index()?;
        }

        let offsets = self.index.get().unwrap();
        if index >= offsets.len() {
            return Ok(None);
        }

        let offset = offsets[index].byte_offset;
        self.read_at_offset(offset)
    }

    fn len(&mut self) -> std::io::Result<usize> {
        if self.index.get().is_none() {
            self.build_index()?;
        }
        Ok(self.index.get().unwrap().len())
    }
}

/// Read the first frame of an (extended) XYZ file.
///
/// # Examples
///
/// ```
/// use molrs::io::read_xyz;
///
/// let frame = read_xyz(b"2\n\nO 0 0 0\nH 0.96 0 0\n")?;
/// assert_eq!(frame.get("atoms").and_then(|b| b.n_rows()), Some(2));
/// # Ok::<(), std::io::Error>(())
/// ```
pub fn read_xyz<'a>(src: impl Into<Source<'a>>) -> std::io::Result<Frame> {
    let src = src.into();
    XyzReader::new(src.reader()?)
        .read_frame(0)?
        .ok_or_else(|| invalid_data(format!("{}: no XYZ frame", src.name())))
}

/// Write one frame as an (extended) XYZ file.
pub fn write_xyz<'a>(sink: impl Into<Sink<'a>>, frame: &impl FrameAccess) -> std::io::Result<()> {
    sink.into().put_with(|out| write_frame_to(out, frame))
}

use crate::io::frame_index::{FrameIndexBuilder, FrameOffset, LineAccumulator};

/// Phase of the XYZ indexer state machine.
#[derive(Debug, Clone, Copy)]
enum XyzPhase {
    /// Looking for the natoms count line.
    AwaitingNatoms,
    /// natoms parsed; the next line is the comment line.
    AwaitingComment { natoms: usize },
    /// Consuming `remaining` atom lines.
    ConsumingAtoms { remaining: usize },
}

/// Streaming frame indexer for XYZ / Extended-XYZ files.
///
/// State machine: `AwaitingNatoms` → `AwaitingComment` → `ConsumingAtoms` →
/// emit and reset to `AwaitingNatoms`. Blank lines in `AwaitingNatoms` are
/// skipped; any other line there must be an atom count, and the last frame
/// must be whole — the rules of `read_frame_from`, so an index and a reader
/// never disagree about a file.
pub struct XyzIndexBuilder {
    lines: LineAccumulator,
    /// The first malformed line `feed` met, reported by `finish`.
    error: Option<std::io::Error>,
    phase: XyzPhase,
    pending_frame_start: Option<u64>,
    pending_entries: Vec<FrameOffset>,
    /// Total bytes scanned so far (line offset + line len of the most
    /// recently completed atom line) — needed to compute byte_len on
    /// frame completion.
    last_line_end: u64,
}

impl Default for XyzIndexBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl XyzIndexBuilder {
    pub fn new() -> Self {
        Self {
            lines: LineAccumulator::new(),
            error: None,
            phase: XyzPhase::AwaitingNatoms,
            pending_frame_start: None,
            pending_entries: Vec::new(),
            last_line_end: 0,
        }
    }

    fn process_line(&mut self, line: &str, line_offset: u64, line_len: u32) -> std::io::Result<()> {
        let line_end = line_offset + line_len as u64;
        self.last_line_end = line_end;
        match self.phase {
            XyzPhase::AwaitingNatoms => {
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    // Blank line — skip in the AwaitingNatoms state, as
                    // `read_frame_from` does.
                    return Ok(());
                }
                let n = trimmed.parse::<usize>().map_err(|_| {
                    std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!("invalid atom count line at byte {line_offset}: {trimmed}"),
                    )
                })?;
                self.pending_frame_start = Some(line_offset);
                self.phase = XyzPhase::AwaitingComment { natoms: n };
                Ok(())
            }
            XyzPhase::AwaitingComment { natoms } => {
                if natoms == 0 {
                    // 0-atom frame — emit now, spanning the count + comment.
                    let start = self.pending_frame_start.take().unwrap_or(line_offset);
                    let span = line_end - start;
                    self.push_frame_span(start, span)?;
                    self.phase = XyzPhase::AwaitingNatoms;
                } else {
                    self.phase = XyzPhase::ConsumingAtoms { remaining: natoms };
                }
                Ok(())
            }
            XyzPhase::ConsumingAtoms { remaining } => {
                let new_remaining = remaining - 1;
                if new_remaining == 0 {
                    let start = self.pending_frame_start.take().unwrap_or(line_offset);
                    let span = line_end - start;
                    self.push_frame_span(start, span)?;
                    self.phase = XyzPhase::AwaitingNatoms;
                } else {
                    self.phase = XyzPhase::ConsumingAtoms {
                        remaining: new_remaining,
                    };
                }
                Ok(())
            }
        }
    }

    fn push_frame_span(&mut self, byte_offset: u64, span: u64) -> std::io::Result<()> {
        if span > u32::MAX as u64 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "XYZ frame size exceeds 4 GiB",
            ));
        }
        self.pending_entries.push(FrameOffset {
            byte_offset,
            byte_len: span as u32,
        });
        Ok(())
    }
}

impl FrameIndexBuilder for XyzIndexBuilder {
    fn feed(&mut self, chunk: &[u8], global_offset: u64) {
        // Drain into a buffer first, then process — the closure can't borrow
        // self mutably and call self.process_line. Collect pending lines and
        // process post-loop.
        let mut staged: Vec<(String, u64, u32)> = Vec::new();
        self.lines
            .feed(chunk, global_offset, |line, line_offset, line_len| {
                staged.push((line.to_string(), line_offset, line_len));
            });
        for (line, off, len) in staged {
            if self.error.is_some() {
                return;
            }
            // `feed` has no error channel: the first malformed line is kept
            // and `finish` reports it.
            if let Err(e) = self.process_line(&line, off, len) {
                self.error = Some(e);
            }
        }
    }

    fn drain(&mut self) -> Vec<FrameOffset> {
        std::mem::take(&mut self.pending_entries)
    }

    fn finish(mut self: Box<Self>) -> std::io::Result<Vec<FrameOffset>> {
        if let Some(e) = self.error.take() {
            return Err(e);
        }
        self.lines.check_line_budget()?;
        let mut staged: Vec<(String, u64, u32)> = Vec::new();
        self.lines.finish(|line, line_offset, line_len| {
            staged.push((line.to_string(), line_offset, line_len));
        });
        for (line, off, len) in staged {
            self.process_line(&line, off, len)?;
        }
        // A frame cut short at EOF is a malformed file, as `read_frame_from`
        // finds it.
        if let Some(start) = self.pending_frame_start {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                format!("XYZ frame at byte {start} ends before its atoms do"),
            ));
        }
        Ok(std::mem::take(&mut self.pending_entries))
    }

    fn bytes_seen(&self) -> u64 {
        self.lines.bytes_seen()
    }
}

#[cfg(test)]
#[allow(clippy::items_after_test_module)]
mod tests {
    use super::*;
    use crate::core::Cell;

    /// The index reads a file by the reader's rules: a line that is no atom
    /// count, or a last frame cut short, is refused rather than skipped.
    #[test]
    fn the_index_refuses_what_the_reader_refuses() {
        for (text, kind) in [
            (
                "1\n\nO 0 0 0\ngarbage\n1\n\nO 0 0 0\n",
                std::io::ErrorKind::InvalidData,
            ),
            (
                "1\n\nO 0 0 0\n2\n\nO 0 0 0\n",
                std::io::ErrorKind::UnexpectedEof,
            ),
        ] {
            let mut reader = XyzReader::new(std::io::Cursor::new(text.as_bytes()));
            assert_eq!(reader.len().unwrap_err().kind(), kind, "{text:?}");
        }
        let mut whole =
            XyzReader::new(std::io::Cursor::new(&b"1\n\nO 0 0 0\n\n1\n\nH 0 0 0\n"[..]));
        assert_eq!(whole.len().unwrap(), 2);
    }

    /// The single-frame door checks the Frame schema as the streaming writer
    /// does, and refuses before it touches its sink.
    #[test]
    fn write_xyz_refuses_a_frame_off_the_schema_and_keeps_the_file() {
        let mut frame = parse_frame_str("1\n\nO 0 0 0\n").unwrap();
        // A `bonds` block without its second endpoint column.
        let mut bonds = Block::new();
        bonds
            .insert(
                "atomi",
                ndarray::Array1::from_vec(vec![0_u64]).into_dyn(),
                None,
            )
            .unwrap();
        frame.insert("bonds", bonds);
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("a.xyz");
        std::fs::write(&path, b"kept").unwrap();
        let err = write_xyz(&path, &frame).unwrap_err();
        assert!(err.to_string().contains("Frame schema"), "{err}");
        assert_eq!(std::fs::read(&path).unwrap(), b"kept");
    }

    /// A `Lattice` or `Origin` the comment line states but that does not
    /// read as its numbers is refused by name, never read as "no box".
    #[test]
    fn a_malformed_lattice_or_origin_is_refused() {
        let good = "1\nLattice=\"10 0 0 0 10 0 0 0 10\" Properties=species:S:1:pos:R:3\nC 0 0 0\n";
        assert!(read_xyz(good.as_bytes()).unwrap().simbox.is_some());
        for (bad, names) in [
            (
                "1\nLattice=\"10 0 0 0 1x 0 0 0 10\" Properties=species:S:1:pos:R:3\nC 0 0 0\n",
                "1x",
            ),
            (
                "1\nLattice=\"10 0 0 0 10 0\" Properties=species:S:1:pos:R:3\nC 0 0 0\n",
                "9",
            ),
            (
                "1\nLattice=\"1 2 3 2 4 6 3 6 9\" Properties=species:S:1:pos:R:3\nC 0 0 0\n",
                "Lattice",
            ),
            (
                "1\nLattice=\"10 0 0 0 10 0 0 0 10\" Origin=\"0 0\" Properties=species:S:1:pos:R:3\nC 0 0 0\n",
                "Origin",
            ),
        ] {
            let err = read_xyz(bad.as_bytes()).unwrap_err().to_string();
            assert!(err.contains(names), "{bad:?}: {err}");
        }
    }

    /// extxyz `Lattice="R1 R2 R3"` lists the vectors in sequence; `SimBox`
    /// keeps them as the columns of H.
    #[test]
    fn lattice_vectors_become_the_box_columns() {
        let frame = parse_frame_str(
            "1\nLattice=\"10 0 0 2 11 0 3 4 12\" Properties=species:S:1:pos:R:3\nH 0 0 0\n",
        )
        .expect("parse XYZ");
        let simbox = frame.simbox.as_ref().expect("Lattice sets the box");
        assert_eq!(simbox.lattice(0).to_vec(), vec![10.0, 0.0, 0.0]);
        assert_eq!(simbox.lattice(1).to_vec(), vec![2.0, 11.0, 0.0]);
        assert_eq!(simbox.lattice(2).to_vec(), vec![3.0, 4.0, 12.0]);
    }

    /// ExtXYZ `type:I:1` is a numeric ordinal; the Frame schema stores those
    /// in `type_id`. `type` is reserved for string labels.
    #[test]
    fn extxyz_integer_type_column_lands_as_type_id() {
        let frame =
            parse_frame_str("1\nProperties=species:S:1:pos:R:3:type:I:1\nC 0.0 0.0 0.0 3\n")
                .expect("parse XYZ");
        let atoms = frame.get("atoms").expect("atoms block");
        assert_eq!(
            atoms
                .get("type_id")
                .and_then(|c| c.get::<u64>())
                .unwrap()
                .as_slice()
                .unwrap(),
            &[3_u64]
        );
        assert!(atoms.get("type").and_then(|c| c.get::<String>()).is_none());
        assert!(!atoms.contains_key("type"));
    }

    /// ExtXYZ `type:S:1` is already a label; it stays `type`.
    #[test]
    fn extxyz_string_type_column_stays_type() {
        let frame =
            parse_frame_str("1\nProperties=species:S:1:pos:R:3:type:S:1\nC 0.0 0.0 0.0 C_3\n")
                .expect("parse XYZ");
        let atoms = frame.get("atoms").expect("atoms block");
        assert_eq!(
            atoms
                .get("type")
                .and_then(|c| c.get::<String>())
                .unwrap()
                .as_slice()
                .unwrap(),
            &["C_3".to_string()]
        );
        assert!(atoms.get("type_id").and_then(|c| c.get::<u64>()).is_none());
        assert!(!atoms.contains_key("type_id"));
    }

    #[test]
    fn writer_lists_the_lattice_vectors_in_sequence() {
        let mut frame = parse_frame_str(
            "1\nLattice=\"10 0 0 2 11 0 3 4 12\" Properties=species:S:1:pos:R:3\nH 0 0 0\n",
        )
        .expect("parse XYZ");
        let h = [[10.0, 2.0, 3.0], [0.0, 11.0, 4.0], [0.0, 0.0, 12.0]];
        frame.simbox =
            Some(SimBox::new(Cell::Matrix(h), [0.0, 0.0, 0.0], [true, true, true]).expect("cell"));
        let mut output = Vec::new();
        write_frame_to(&mut output, &frame).expect("write XYZ");
        let output = String::from_utf8(output).expect("UTF-8 XYZ");
        assert!(
            output.contains("Lattice=\"10 0 0 2 11 0 3 4 12\""),
            "comment line: {output}"
        );
    }

    #[test]
    fn parse_properties_triplets() {
        let line = "Properties=species:S:1:pos:R:3:mass:R:1";
        let ec = parse_comment_line(line).expect("parse");
        assert!(ec.properties.is_some());
        let props = ec.properties.unwrap();
        assert_eq!(props.len(), 3);
        assert_eq!(
            props[0],
            PropertySpec {
                name: "species".into(),
                ty: ExtxyzPropertyKind::String,
                m: 1
            }
        );
        assert_eq!(
            props[1],
            PropertySpec {
                name: "pos".into(),
                ty: ExtxyzPropertyKind::Real,
                m: 3
            }
        );
        assert_eq!(
            props[2],
            PropertySpec {
                name: "mass".into(),
                ty: ExtxyzPropertyKind::Real,
                m: 1
            }
        );
        assert!(!ec.is_plain_xyz);
    }

    #[test]
    fn parse_comment_with_properties() {
        let line = r#"Lattice="8.43116035 0.0 0.0 0.158219155128 14.5042431863 0.0 1.16980663624 4.4685149855 14.9100096405" Properties=species:S:1:pos:R:3:CS:R:2 ENERGY=-2069.84934116 Natoms=192 NAME=COBHUW"#;
        let ec = parse_comment_line(line).expect("parse");
        assert!(ec.properties.is_some());
        assert!(!ec.is_plain_xyz);
        // Lattice
        match ec.kv.get("Lattice").unwrap() {
            ExtxyzValue::Array1(v) => {
                assert_eq!(v.len(), 9);
                assert!(
                    matches!(v[0], ExtxyzScalar::Real(_)) || matches!(v[0], ExtxyzScalar::Int(_))
                );
            }
            other => panic!("unexpected Lattice value: {other:?}"),
        }
        // energy
        match ec.kv.get("ENERGY").unwrap() {
            ExtxyzValue::Scalar(ExtxyzScalar::Real(x)) => {
                assert!((x - -2069.84934116).abs() < 1e-6)
            }
            other => panic!("unexpected energy value: {other:?}"),
        }
    }

    #[test]
    fn quoted_metadata_with_spaces_stays_a_string() {
        let data = b"1\ntitle=\"Water box\" Properties=species:S:1:pos:R:3\nH 0 0 0\n";
        let mut cursor = std::io::Cursor::new(&data[..]);
        let frame = read_frame_from(&mut cursor)
            .expect("read XYZ")
            .expect("one frame");

        assert_eq!(
            frame.meta.get("title").and_then(MetaValue::as_str),
            Some("Water box")
        );
    }

    #[test]
    fn connct_comment_builds_zero_based_bonds() {
        let data = b"3\nname=water Connct=\"[0,1,0,2]\"\nO 0 0 0\nH 1 0 0\nH 0 1 0\n";
        let mut cursor = std::io::Cursor::new(&data[..]);
        let frame = read_frame_from(&mut cursor)
            .expect("read XYZ")
            .expect("one frame");
        let bonds = frame.get("bonds").expect("Connct creates bonds block");

        assert_eq!(
            bonds
                .get("atomi")
                .and_then(|c| c.get::<u64>())
                .unwrap()
                .as_slice()
                .unwrap(),
            &[0, 0]
        );
        assert_eq!(
            bonds
                .get("atomj")
                .and_then(|c| c.get::<u64>())
                .unwrap()
                .as_slice()
                .unwrap(),
            &[1, 2]
        );
        assert!(!frame.meta.contains_key("Connct"));
    }

    #[test]
    fn xyz_writer_preserves_connct_bonds() {
        let frame =
            parse_frame_str("3\nname=water Connct=\"[0,1,0,2]\"\nO 0 0 0\nH 1 0 0\nH 0 1 0\n")
                .expect("parse XYZ");
        let mut output = Vec::new();

        write_frame_to(&mut output, &frame).expect("write XYZ");
        let output = String::from_utf8(output).expect("UTF-8 XYZ");

        assert!(
            output
                .lines()
                .nth(1)
                .unwrap()
                .contains("Connct=\"[0,1,0,2]\"")
        );
        let round_trip = parse_frame_str(&output).expect("read written XYZ");
        assert_eq!(
            round_trip
                .get("bonds")
                .unwrap()
                .get("atomj")
                .and_then(|c| c.get::<u64>())
                .unwrap()
                .as_slice()
                .unwrap(),
            &[1, 2]
        );
    }

    #[test]
    fn connct_comment_rejects_invalid_pairs() {
        let odd = "3\nConnct=\"[0,1,0]\"\nO 0 0 0\nH 1 0 0\nH 0 1 0\n";
        assert!(
            parse_frame_str(odd)
                .unwrap_err()
                .contains("atom-index pairs")
        );

        let out_of_range = "3\nConnct=\"[0,3]\"\nO 0 0 0\nH 1 0 0\nH 0 1 0\n";
        assert!(
            parse_frame_str(out_of_range)
                .unwrap_err()
                .contains("out of range")
        );
    }

    /// A frame shaped like what the GRO reader produces — an `id` column, and
    /// extra string/int columns that sort *before* `id` — plus the `element`
    /// column the XYZ writer requires (GRO itself spells no species).
    fn gro_shaped_frame() -> Frame {
        use molrs::core::Block;
        use ndarray::Array1;

        let floats = |v: [f64; 3]| Array1::from_vec(v.to_vec()).into_dyn();
        let uints = |v: [u64; 3]| Array1::from_vec(v.to_vec()).into_dyn();
        let strings = |v: [&str; 3]| {
            Array1::from_vec(v.iter().map(|s| (*s).to_owned()).collect::<Vec<_>>()).into_dyn()
        };

        let mut atoms = Block::new();
        atoms.insert("x", floats([0.0, 1.0, 0.0]), None).unwrap();
        atoms.insert("y", floats([0.0, 0.0, 1.0]), None).unwrap();
        atoms.insert("z", floats([0.0, 0.0, 0.0]), None).unwrap();
        atoms.insert("id", uints([1, 2, 3]), None).unwrap();
        atoms
            .insert("name", strings(["OW", "HW1", "HW2"]), None)
            .unwrap();
        atoms
            .insert("element", strings(["O", "H", "H"]), None)
            .unwrap();
        atoms.insert("res_id", uints([1, 1, 1]), None).unwrap();
        atoms
            .insert("res_name", strings(["WAT", "WAT", "WAT"]), None)
            .unwrap();

        let mut frame = Frame::new();
        frame.insert("atoms", atoms);
        frame
    }

    /// Two frames written as one trajectory read back as two frames, in
    /// order, each with its own coordinates.
    #[test]
    fn write_xyz_traj_round_trips_every_frame() {
        let first = gro_shaped_frame();
        let mut second = gro_shaped_frame();
        second
            .get_mut("atoms")
            .unwrap()
            .insert(
                "x",
                ndarray::Array1::from_vec(vec![5.0, 6.0, 7.0]).into_dyn(),
                None,
            )
            .unwrap();
        let mut out = Vec::new();
        {
            use crate::io::writer::{FrameWriter, Writer};
            let mut writer = XyzWriter::new(&mut out);
            writer.write(&first).expect("write XYZ frame 1");
            writer.write(&second).expect("write XYZ frame 2");
        }

        let frames: Vec<Frame> = XyzReader::new(std::io::Cursor::new(out))
            .iter()
            .collect::<std::io::Result<_>>()
            .expect("read XYZ trajectory");
        assert_eq!(frames.len(), 2);
        let x1 = frames[1]
            .get("atoms")
            .unwrap()
            .get("x")
            .and_then(|c| c.get::<f64>())
            .unwrap();
        assert_eq!(x1[[0]], 5.0);
        assert_eq!(x1[[2]], 7.0);
    }

    #[test]
    fn xyz_writer_row_order_matches_the_properties_header() {
        // The Properties header and the data rows are two views of one column
        // order. `id` is written before the alphabetically-earlier `name`
        // in the header, so the rows must do the same — or every consumer reads
        // an atom name where the header promised an integer id.
        let mut output = Vec::new();
        write_frame_to(&mut output, &gro_shaped_frame()).expect("write XYZ");
        let output = String::from_utf8(output).expect("UTF-8 XYZ");

        let comment = output.lines().nth(1).expect("comment line");
        let props = comment
            .split_whitespace()
            .find_map(|tok| tok.strip_prefix("Properties="))
            .expect("Properties in comment");
        let declared: Vec<&str> = props.split(':').step_by(3).collect();
        assert_eq!(
            declared,
            ["species", "pos", "id", "name", "res_id", "resname"]
        );

        // species + 3 coordinates, then one field per remaining declared column.
        let first_row: Vec<&str> = output
            .lines()
            .nth(2)
            .expect("first atom line")
            .split_whitespace()
            .collect();
        assert_eq!(first_row.len(), 8, "row: {first_row:?}");
        assert_eq!(
            &first_row[4..],
            ["1", "OW", "1", "WAT"],
            "row: {first_row:?}"
        );
    }

    #[test]
    fn xyz_writer_round_trips_every_declared_column() {
        // The header/row contract, stated as the thing a reader actually needs:
        // each column comes back carrying its own values, not its neighbour's.
        let mut output = Vec::new();
        write_frame_to(&mut output, &gro_shaped_frame()).expect("write XYZ");
        let text = String::from_utf8(output).expect("UTF-8 XYZ");

        let back = parse_frame_str(&text).expect("read written XYZ");
        let atoms = back.get("atoms").expect("atoms block");
        assert_eq!(
            atoms
                .get("name")
                .and_then(|c| c.get::<String>())
                .unwrap()
                .as_slice()
                .unwrap(),
            &["OW".to_string(), "HW1".to_string(), "HW2".to_string()]
        );
        assert_eq!(
            atoms
                .get("res_name")
                .and_then(|c| c.get::<String>())
                .unwrap()
                .as_slice()
                .unwrap(),
            &["WAT".to_string(), "WAT".to_string(), "WAT".to_string()]
        );
        assert_eq!(
            atoms
                .get("id")
                .and_then(|c| c.get::<u64>())
                .unwrap()
                .as_slice()
                .unwrap(),
            &[1_u64, 2, 3]
        );
        assert_eq!(
            atoms
                .get("res_id")
                .and_then(|c| c.get::<u64>())
                .unwrap()
                .as_slice()
                .unwrap(),
            &[1_u64, 1, 1]
        );
    }

    #[test]
    fn test_xyz_invalid_atom_count() {
        use std::io::Cursor;

        let data = b"abc\nComment\nH 0 0 0\n";
        let mut cursor = Cursor::new(&data[..]);
        let err = read_frame_from(&mut cursor).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
    }

    // -----------------------------------------------------------------
    // Streaming index tests
    // -----------------------------------------------------------------

    fn xyz_build_chunked(bytes: &[u8], chunk_size: usize) -> Vec<FrameOffset> {
        let mut builder = Box::new(XyzIndexBuilder::new());
        let mut offset: u64 = 0;
        let mut out: Vec<FrameOffset> = Vec::new();
        for piece in bytes.chunks(chunk_size.max(1)) {
            builder.feed(piece, offset);
            offset += piece.len() as u64;
            out.extend(builder.drain());
        }
        out.extend(builder.finish().expect("finish"));
        out
    }

    const TWO_FRAME_XYZ: &str =
        "2\nframe 0\nH 0 0 0\nH 1 0 0\n3\nframe 1\nO 0 0 0\nH 1 0 0\nH -1 0 0\n";

    #[test]
    fn xyz_streaming_single_shot_matches_chunks() {
        let bytes = TWO_FRAME_XYZ.as_bytes();
        let one_shot = xyz_build_chunked(bytes, bytes.len());
        assert_eq!(one_shot.len(), 2);
        for cs in [1usize, 3, 7, 13, 31, 64, 1024] {
            let chunked = xyz_build_chunked(bytes, cs);
            assert_eq!(
                one_shot, chunked,
                "chunk size {} produced different index",
                cs
            );
        }

        // Each entry slice must be parseable.
        for entry in &one_shot {
            let lo = entry.byte_offset as usize;
            let hi = lo + entry.byte_len as usize;
            read_xyz(&bytes[lo..hi]).expect("read_xyz");
        }
    }

    /// Edge: chunk boundary inside the natoms count line.
    #[test]
    fn xyz_streaming_boundary_inside_natoms_count() {
        // Multi-digit count.
        let s = "12\nframe\n".to_string()
            + &(0..12).map(|i| format!("H 0 0 {i}\n")).collect::<String>()
            + "8\nframe2\n"
            + &(0..8).map(|i| format!("H 0 0 {i}\n")).collect::<String>();
        let bytes = s.as_bytes();
        // Split at byte 1 — middle of "12".
        let mut builder = Box::new(XyzIndexBuilder::new());
        builder.feed(&bytes[..1], 0);
        builder.feed(&bytes[1..], 1);
        let mut got = builder.drain();
        got.extend(builder.finish().expect("finish"));
        let one_shot = xyz_build_chunked(bytes, bytes.len());
        assert_eq!(one_shot.len(), 2);
        assert_eq!(got, one_shot);
    }

    /// Edge: blank/garbage lines in `AwaitingNatoms` are skipped, not crashed.
    #[test]
    fn xyz_streaming_skips_blanks_in_awaiting_natoms() {
        let s = "\n\n  \n2\nframe 0\nH 0 0 0\nH 1 0 0\n";
        let bytes = s.as_bytes();
        let entries = xyz_build_chunked(bytes, bytes.len());
        assert_eq!(entries.len(), 1);
        // Frame must start at the "2\n" line, not at byte 0.
        let lo = entries[0].byte_offset as usize;
        assert!(bytes[lo..].starts_with(b"2\n"));
        let hi = lo + entries[0].byte_len as usize;
        let frame = read_xyz(&bytes[lo..hi]).expect("parse");
        assert_eq!(frame.get("atoms").unwrap().n_rows().unwrap(), 2);
    }

    /// `XyzReader::build_index` (used by `len()` / random-access
    /// `read_frame`) tolerates blank lines between frames and trailing
    /// blanks — the same AwaitingNatoms rule as the streaming index — rather
    /// than reporting `XYZ len error: invalid atom count:`.
    #[test]
    fn xyz_random_access_index_skips_inter_frame_and_trailing_blanks() {
        use crate::io::reader::TrajectoryReader;
        use std::io::{BufReader, Cursor};

        let s = "\
2
frame 0
H 0 0 0
H 1 0 0

2
frame 1
H 0 0 1
H 1 0 1

";
        let mut reader = XyzReader::new(BufReader::new(Cursor::new(s.as_bytes())));
        assert_eq!(reader.len().expect("len"), 2);

        let f0 = reader.read_frame(0).expect("step0").expect("some");
        let f1 = reader.read_frame(1).expect("step1").expect("some");
        assert_eq!(f0.get("atoms").unwrap().n_rows().unwrap(), 2);
        assert_eq!(f1.get("atoms").unwrap().n_rows().unwrap(), 2);
        let z0 = f0
            .get("atoms")
            .unwrap()
            .get("z")
            .and_then(|c| c.get::<f64>())
            .unwrap();
        let z1 = f1
            .get("atoms")
            .unwrap()
            .get("z")
            .and_then(|c| c.get::<f64>())
            .unwrap();
        assert!((z0[0] - 0.0).abs() < 1e-12);
        assert!((z1[0] - 1.0).abs() < 1e-12);
    }

    /// Leading blanks before the first frame must not confuse the random-access index.
    #[test]
    fn xyz_random_access_index_skips_leading_blanks() {
        use crate::io::reader::TrajectoryReader;
        use std::io::{BufReader, Cursor};

        let s = "\n\n  \n2\nframe 0\nH 0 0 0\nH 1 0 0\n";
        let mut reader = XyzReader::new(BufReader::new(Cursor::new(s.as_bytes())));
        assert_eq!(reader.len().expect("len"), 1);
        let f0 = reader.read_frame(0).expect("step0").expect("some");
        assert_eq!(f0.get("atoms").unwrap().n_rows().unwrap(), 2);
    }
}

// =============== XyzWriter ===============

/// A writer of (extended) XYZ frames over any byte sink.
pub struct XyzWriter<W: Write> {
    writer: W,
}

impl<W: Write> Writer for XyzWriter<W> {
    type W = W;

    fn new(writer: Self::W) -> Self {
        Self { writer }
    }
}

impl<'a> XyzWriter<SinkWriter<'a>> {
    /// Create an XYZ file, or fill a buffer, frame by frame through
    /// [`FrameWriter::write`].
    pub fn create(sink: impl Into<Sink<'a>>) -> std::io::Result<Self> {
        Ok(Self {
            writer: sink.into().writer()?,
        })
    }
}

impl<W: Write> FrameWriter for XyzWriter<W> {
    fn write(&mut self, frame: &Frame) -> std::io::Result<()> {
        write_frame_to(&mut self.writer, frame)?;
        self.writer.flush()
    }
}

/// Write a single frame to the writer in Extended XYZ format.
///
/// Accepts any type implementing [`FrameAccess`], including both [`Frame`] and
/// [`FrameView`](molrs::core::FrameView). Existing callers passing `&Frame`
/// continue to work without changes.
fn write_frame_to<W: Write>(writer: &mut W, frame: &impl FrameAccess) -> std::io::Result<()> {
    crate::io::null_cells::refuse_null_cells_in_block(frame, "atoms")?;
    use molrs::core::DType;

    // Every door writes through here, so every door checks here: a frame that
    // violates the vocabulary makes a file that looks fine and is wrong.
    crate::io::writer::check_write_frame(frame)?;

    // 1. Build per-atom data from the atoms block via visit_block.
    //    We collect everything we need into owned data structures inside the closure,
    //    then write outside it.
    struct AtomRows {
        n: usize,
        properties_str: String,
        elements: Vec<String>,
        /// Per-row, per-column values (excluding element). Outer = rows, inner = values.
        row_values: Vec<Vec<String>>,
    }

    let atom_rows: Option<std::io::Result<AtomRows>> = frame.visit_block("atoms", |atoms| {
        let n = atoms.n_rows().unwrap_or(0);

        // Collect and sort keys
        let mut keys = atoms.column_keys();
        keys.sort_by(|a, b| {
            let rank = |s: &str| match s {
                "x" => 0,
                "y" => 1,
                "z" => 2,
                _ => 3,
            };
            let ra = rank(a);
            let rb = rank(b);
            if ra != rb { ra.cmp(&rb) } else { a.cmp(b) }
        });

        let has_xyz = keys.contains(&"x") && keys.contains(&"y") && keys.contains(&"z");
        let priority_keys = ["id", "mol"];
        let is_pos = |k: &str| has_xyz && matches!(k, "x" | "y" | "z");

        // ONE ordered column list drives both the Properties header and every
        // data row. Building them from two independent walks is how the header
        // came to hoist `id` / `mol` to the front while the rows kept them in
        // alphabetical position: a GRO-shaped frame then declared
        // `…:pos:R:3:id:I:1:atom_name:S:1` and wrote `… 0 0 0 OW 1 …`, handing
        // every reader an atom name where an integer was promised.
        //
        // Columns the block cannot describe are dropped here rather than inside
        // each loop, so there is no per-loop condition left to disagree on.
        let mut columns: Vec<&str> = Vec::new();
        if has_xyz {
            columns.extend(["x", "y", "z"]);
        }
        columns.extend(priority_keys.iter().copied().filter(|pk| keys.contains(pk)));
        columns.extend(
            keys.iter()
                .copied()
                .filter(|k| *k != "element" && !priority_keys.contains(k) && !is_pos(k)),
        );
        columns.retain(|k| atoms.column_dtype(k).is_some() && atoms.column_shape(k).is_some());

        let dtype_to_char = |dt: DType| -> &'static str {
            match dt {
                DType::Float | DType::C64 | DType::C128 => "R",
                DType::Int
                | DType::I8
                | DType::I16
                | DType::I64
                | DType::Uint
                | DType::U8
                | DType::U16
                | DType::U32 => "I",
                DType::Bool => "L",
                DType::String => "S",
            }
        };

        // Build the Properties string from that list. The x/y/z triple is
        // declared once as `pos:R:3`; every other column declares itself.
        let mut props_parts = vec!["species:S:1".to_string()];
        if has_xyz {
            props_parts.push("pos:R:3".to_string());
        }
        for k in &columns {
            if is_pos(k) {
                continue;
            }
            let (dt, shape) = (
                atoms.column_dtype(k).expect("retained above"),
                atoms.column_shape(k).expect("retained above"),
            );
            let m: usize = shape.iter().skip(1).product();
            let m = if m == 0 { 1 } else { m };
            props_parts.push(format!(
                "{}:{}:{}",
                extxyz_property_name(k),
                dtype_to_char(dt),
                m
            ));
        }
        let properties_str = props_parts.join(":");

        // The species column XYZ writes is the `element` symbol, read as is;
        // a frame that carries only `atomic_number` converts explicitly first.
        let elements: Vec<String> = match atoms.column("element").and_then(|c| c.get::<String>()) {
            Some(arr) => arr.iter().cloned().collect(),
            None if n == 0 => Vec::new(),
            None => {
                return Err(crate::io::invalid_data(
                    "XYZ writer: atoms block has no string 'element' column; a frame \
                     that carries only atomic_number needs \
                     `Element::fill_symbols(&mut frame)` first",
                ));
            }
        };

        // Build per-row values — same list, same order, no second policy.
        let mut row_values: Vec<Vec<String>> = Vec::with_capacity(n);
        for i in 0..n {
            let mut line_parts = Vec::new();
            for k in &columns {
                if let Some(column) = atoms.column(k) {
                    line_parts.extend(row_tokens(&column, i));
                }
            }
            row_values.push(line_parts);
        }

        Ok(AtomRows {
            n,
            properties_str,
            elements,
            row_values,
        })
    });

    let atom_rows = match atom_rows {
        Some(d) => d?,
        None => {
            writeln!(writer, "0")?;
            writeln!(writer)?;
            return Ok(());
        }
    };

    writeln!(writer, "{}", atom_rows.n)?;

    // 2. Construct comment line using FrameAccess for simbox and meta
    let mut comment_parts = Vec::new();
    if let Some(simbox) = frame.simbox_ref() {
        let h = ndarray::arr2(&simbox.matrix());
        let mut lattice_values = Vec::with_capacity(9);
        // Column k of H is lattice vector k; extxyz writes R1 R2 R3 in sequence.
        for k in 0..3 {
            for j in 0..3 {
                lattice_values.push(format!("{}", h[[j, k]]));
            }
        }
        comment_parts.push(format!("Lattice=\"{}\"", lattice_values.join(" ")));

        let o = simbox.origin();
        if o[0] != 0.0 || o[1] != 0.0 || o[2] != 0.0 {
            comment_parts.push(format!("Origin=\"{} {} {}\"", o[0], o[1], o[2]));
        }
    }
    for (k, v) in frame.meta_ref() {
        if k == "Lattice"
            || k == "Origin"
            || k == "Properties"
            || k == "comment"
            || k == "elements"
            || k.eq_ignore_ascii_case("Connct")
        {
            continue;
        }
        let val_str = meta_to_extxyz(v);
        comment_parts.push(format!("{}={}", k, val_str));
    }
    if let (Some(atomi), Some(atomj)) = (
        frame.column("bonds", "atomi").and_then(|c| c.get::<u64>()),
        frame.column("bonds", "atomj").and_then(|c| c.get::<u64>()),
    ) && atomi.len() == atomj.len()
        && !atomi.is_empty()
    {
        let indices = atomi
            .iter()
            .zip(atomj.iter())
            .flat_map(|(atomi, atomj)| [atomi.to_string(), atomj.to_string()])
            .collect::<Vec<_>>()
            .join(",");
        comment_parts.push(format!("Connct=\"[{indices}]\""));
    }
    comment_parts.push(format!("Properties={}", atom_rows.properties_str));
    writeln!(writer, "{}", comment_parts.join(" "))?;

    // 3. Write atom lines
    for i in 0..atom_rows.n {
        let mut line_parts = vec![atom_rows.elements[i].clone()];
        line_parts.extend(atom_rows.row_values[i].iter().cloned());
        writeln!(writer, "{}", line_parts.join(" "))?;
    }

    Ok(())
}

#[cfg(test)]
mod canonical_column_tests {
    use super::*;

    const WIDE: &str = "2\nProperties=species:S:1:pos:R:3:forces:R:3:tags:I:2\n\
                        O 0 0 0 0.1 0.2 0.3 1 2\nH 1 0 0 -0.1 -0.2 -0.3 3 4\n";

    #[test]
    fn species_reads_as_element() {
        let frame = parse_frame_str(WIDE).unwrap();
        let atoms = frame.get("atoms").unwrap();
        assert!(!atoms.contains_key("species"));
        let el = atoms
            .get("element")
            .and_then(|c| c.get::<String>())
            .unwrap();
        assert_eq!((el[[0]].as_str(), el[[1]].as_str()), ("O", "H"));
    }

    #[test]
    fn a_wide_property_is_one_column_of_that_width() {
        let frame = parse_frame_str(WIDE).unwrap();
        let atoms = frame.get("atoms").unwrap();
        let forces = atoms.get("forces").and_then(|c| c.get::<f64>()).unwrap();
        assert_eq!(forces.shape(), &[2, 3]);
        assert_eq!(forces[[1, 2]], -0.3);
        let tags = atoms.get("tags").and_then(|c| c.get::<i32>()).unwrap();
        assert_eq!(tags.shape(), &[2, 2]);
        assert_eq!(tags[[1, 0]], 3);
        assert!(!atoms.contains_key("forces_1"));
    }

    #[test]
    fn wide_columns_and_element_round_trip_through_the_writer() {
        let frame = parse_frame_str(WIDE).unwrap();
        let mut out = Vec::new();
        write_frame_to(&mut out, &frame).unwrap();
        let text = String::from_utf8(out).unwrap();
        assert!(text.contains("forces:R:3"), "{text}");
        let back = parse_frame_str(&text).unwrap();
        let atoms = back.get("atoms").unwrap();
        assert_eq!(
            atoms
                .get("forces")
                .and_then(|c| c.get::<f64>())
                .unwrap()
                .shape(),
            &[2, 3]
        );
        assert!(atoms.contains_key("element"));
    }

    #[test]
    fn writer_requires_an_element_column() {
        let mut frame = parse_frame_str(WIDE).unwrap();
        let mut atoms = frame.get("atoms").unwrap().clone();
        atoms.remove("element");
        frame.insert("atoms", atoms);
        let err = write_frame_to(&mut Vec::new(), &frame).unwrap_err();
        assert!(err.to_string().contains("fill_symbols"), "{err}");
    }

    #[test]
    fn a_declared_element_property_keeps_species_apart() {
        let text = "1\nProperties=species:S:1:pos:R:3:element:S:1\nX 0 0 0 C\n";
        let frame = parse_frame_str(text).unwrap();
        let atoms = frame.get("atoms").unwrap();
        assert_eq!(
            atoms
                .get("element")
                .and_then(|c| c.get::<String>())
                .unwrap()[[0]],
            "C"
        );
        assert!(atoms.contains_key("species"));
    }
}
