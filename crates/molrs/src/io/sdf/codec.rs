//! The SDF / MDL molfile codec (V2000 CTAB).

use crate::io::reader::{FrameReader, ReadSeek, Reader, TrajectoryReader};
use crate::io::{Source, invalid_data};
use molrs::core::Block;
use molrs::core::Frame;
use molrs::op::{F, Idx};
use ndarray::{Array1, IxDyn};
use std::io::{BufRead, Seek};

fn to_array_float(vec: Vec<F>, len: usize) -> std::io::Result<ndarray::ArrayD<F>> {
    Ok(Array1::from_vec(vec)
        .into_shape_with_order(IxDyn(&[len]))
        .map_err(invalid_data)?
        .into_dyn())
}

fn to_array_uint(vec: Vec<Idx>, len: usize) -> std::io::Result<ndarray::ArrayD<Idx>> {
    Ok(Array1::<Idx>::from_vec(vec)
        .into_shape_with_order(IxDyn(&[len]))
        .map_err(invalid_data)?
        .into_dyn())
}

/// Slice `[start..end)` clamped to `s.len()`.
fn substr(s: &str, start: usize, end: usize) -> &str {
    let len = s.len();
    if start >= len {
        return "";
    }
    &s[start..end.min(len)]
}

/// Parsed V2000 counts line.
#[derive(Debug, Clone, Copy)]
struct Counts {
    atoms: usize,
    bonds: usize,
}

fn parse_counts_line(line: &str) -> std::io::Result<Counts> {
    if line.len() < 6 {
        return Err(invalid_data("counts line too short"));
    }
    let version = substr(line, 33, 39).trim();
    if version.eq_ignore_ascii_case("V3000") {
        return Err(invalid_data("V3000 SDF not supported"));
    }
    let atoms = substr(line, 0, 3)
        .trim()
        .parse::<usize>()
        .map_err(invalid_data)?;
    let bonds = substr(line, 3, 6)
        .trim()
        .parse::<usize>()
        .map_err(invalid_data)?;
    Ok(Counts { atoms, bonds })
}

#[derive(Debug, Clone)]
struct SdfAtom {
    element: String,
    x: F,
    y: F,
    z: F,
}

fn parse_atom_line(line: &str) -> std::io::Result<SdfAtom> {
    if line.len() < 34 {
        return Err(invalid_data("atom line too short"));
    }
    let x = substr(line, 0, 10)
        .trim()
        .parse::<F>()
        .map_err(invalid_data)?;
    let y = substr(line, 10, 20)
        .trim()
        .parse::<F>()
        .map_err(invalid_data)?;
    let z = substr(line, 20, 30)
        .trim()
        .parse::<F>()
        .map_err(invalid_data)?;
    // The atom symbol (columns 32-34) is a mandatory V2000 field, kept as the
    // file spells it. A blank one is a malformed record, not a placeholder.
    let element = substr(line, 31, 34).trim().to_string();
    if element.is_empty() {
        return Err(invalid_data(format!(
            "atom line has a blank atom symbol (columns 32-34): {line:?}"
        )));
    }
    Ok(SdfAtom { element, x, y, z })
}

#[derive(Debug, Clone, Copy)]
struct SdfBond {
    i: Idx, // 1-based in source
    j: Idx,
    order: Idx,
}

fn parse_bond_line(line: &str) -> std::io::Result<SdfBond> {
    if line.len() < 9 {
        return Err(invalid_data("bond line too short"));
    }
    let i = substr(line, 0, 3)
        .trim()
        .parse::<Idx>()
        .map_err(invalid_data)?;
    let j = substr(line, 3, 6)
        .trim()
        .parse::<Idx>()
        .map_err(invalid_data)?;
    let order = substr(line, 6, 9)
        .trim()
        .parse::<Idx>()
        .map_err(invalid_data)?;
    Ok(SdfBond { i, j, order })
}

fn build_frame(atoms: &[SdfAtom], bonds: &[SdfBond]) -> std::io::Result<Frame> {
    let n = atoms.len();
    let mut x_vec = Vec::with_capacity(n);
    let mut y_vec = Vec::with_capacity(n);
    let mut z_vec = Vec::with_capacity(n);
    let mut id_vec: Vec<Idx> = Vec::with_capacity(n);
    let mut elements = Vec::with_capacity(n);

    for (i, a) in atoms.iter().enumerate() {
        x_vec.push(a.x);
        y_vec.push(a.y);
        z_vec.push(a.z);
        id_vec.push((i as Idx) + 1);
        elements.push(a.element.clone());
    }

    let mut atoms_block = Block::new();
    atoms_block
        .insert("x", to_array_float(x_vec, n)?, None)
        .map_err(invalid_data)?;
    atoms_block
        .insert("y", to_array_float(y_vec, n)?, None)
        .map_err(invalid_data)?;
    atoms_block
        .insert("z", to_array_float(z_vec, n)?, None)
        .map_err(invalid_data)?;
    atoms_block
        .insert("id", to_array_uint(id_vec, n)?, None)
        .map_err(invalid_data)?;
    let elements_arr = Array1::from_vec(elements)
        .into_shape_with_order(IxDyn(&[n]))
        .map_err(invalid_data)?
        .into_dyn();
    atoms_block
        .insert("element", elements_arr, None)
        .map_err(invalid_data)?;

    let mut frame = Frame::new();
    frame.insert("atoms", atoms_block);

    if !bonds.is_empty() {
        let bn = bonds.len();
        let mut i_vec: Vec<Idx> = Vec::with_capacity(bn);
        let mut j_vec: Vec<Idx> = Vec::with_capacity(bn);
        let mut order_vec: Vec<Idx> = Vec::with_capacity(bn);
        for b in bonds {
            if b.i == 0 || b.j == 0 || (b.i as usize) > n || (b.j as usize) > n {
                return Err(invalid_data(format!(
                    "bond references out-of-range atom: {}-{}",
                    b.i, b.j
                )));
            }
            // Convert 1-based to 0-based indices into the atoms block.
            i_vec.push(b.i - 1);
            j_vec.push(b.j - 1);
            order_vec.push(b.order);
        }
        let mut bonds_block = Block::new();
        bonds_block
            .insert("atomi", to_array_uint(i_vec, bn)?, None)
            .map_err(invalid_data)?;
        bonds_block
            .insert("atomj", to_array_uint(j_vec, bn)?, None)
            .map_err(invalid_data)?;
        // §14: an MDL bond block states integer orders (4 = aromatic in the
        // SDF query alphabet), so it maps onto both facts. An aromatic entry
        // carries no Kekulé phase — that is left `Unknown` for standardization.
        let types: Vec<Idx> = order_vec
            .iter()
            .map(|&v| if v == 4 { 4 } else { v.min(3) })
            .collect();
        let numbers: Vec<Idx> = order_vec
            .iter()
            .map(|&v| if v == 4 { 0 } else { v.min(4) })
            .collect();
        bonds_block
            .insert("bond_type", to_array_uint(types, bn)?, None)
            .map_err(invalid_data)?;
        bonds_block
            .insert("bond_number", to_array_uint(numbers, bn)?, None)
            .map_err(invalid_data)?;
        frame.insert("bonds", bonds_block);
    }

    Ok(frame)
}

/// V2000 SDF / MDL molfile reader.
///
/// Multi-record SDF files are supported; `read_frame` returns one record
/// per call and advances past the `$$$$` terminator.
///
/// Parses the connection table (CTAB) inside MDL `.mol` files and the record
/// blocks of `.sdf` files. V2000 only — V3000 records are rejected.
///
/// Each record is a [`Frame`] with:
/// - `"atoms"` block: `element` (string, the raw atom symbol; no
///   `atomic_number` is synthesized — call `Element::fill_atomic_numbers`),
///   `id` (u32, 1-based),
///   `x`, `y`, `z` (F, angstrom)
/// - `"bonds"` block (if any): `atomi`, `atomj` (u32, 0-based indices
///   into the atoms block), `order` (u32)
///
/// Over a seekable stream it is also a [`TrajectoryReader`]: records by
/// index, located by [`SdfIndexBuilder`] on first use.
pub struct SdfReader<R: BufRead> {
    reader: R,
    index: Option<Vec<FrameOffset>>,
}

impl<R: BufRead> SdfReader<R> {
    /// A reader over any buffered byte stream.
    pub fn new(reader: R) -> Self {
        Self {
            reader,
            index: None,
        }
    }

    fn read_single_record(&mut self) -> std::io::Result<Option<Frame>> {
        // Header (3 lines): title, program info, comment.
        let mut header = [String::new(), String::new(), String::new()];
        for slot in &mut header {
            if self.reader.read_line(slot)? == 0 {
                return Ok(None);
            }
        }

        // Counts line.
        let mut counts_line = String::new();
        if self.reader.read_line(&mut counts_line)? == 0 {
            return Err(invalid_data("missing counts line"));
        }
        let counts = parse_counts_line(&counts_line)?;
        if counts.atoms == 0 {
            return Err(invalid_data("SDF record has zero atoms"));
        }

        // Atom block.
        let mut atoms = Vec::with_capacity(counts.atoms);
        for _ in 0..counts.atoms {
            let mut line = String::new();
            if self.reader.read_line(&mut line)? == 0 {
                return Err(invalid_data("unexpected EOF in atom block"));
            }
            atoms.push(parse_atom_line(&line)?);
        }

        // Bond block.
        let mut bonds = Vec::with_capacity(counts.bonds);
        for _ in 0..counts.bonds {
            let mut line = String::new();
            if self.reader.read_line(&mut line)? == 0 {
                return Err(invalid_data("unexpected EOF in bond block"));
            }
            bonds.push(parse_bond_line(&line)?);
        }

        // Drain remaining lines of this record (properties, data items)
        // until `M  END` or `$$$$` (record terminator in multi-record SDF).
        let mut line = String::new();
        loop {
            line.clear();
            if self.reader.read_line(&mut line)? == 0 {
                break;
            }
            let trimmed = line.trim_end();
            if trimmed == "$$$$" {
                break;
            }
        }

        let frame = build_frame(&atoms, &bonds)?;
        // Every door reads through here, so every door checks here.
        crate::io::reader::check_read_frame(&frame)?;
        Ok(Some(frame))
    }
}

impl<R: BufRead> Reader for SdfReader<R> {
    type R = R;

    fn new(reader: R) -> Self {
        SdfReader::new(reader)
    }
}

impl SdfReader<Box<dyn ReadSeek>> {
    /// Open an SDF file, or the bytes of one, for random access by record.
    pub fn open<'a>(src: impl Into<Source<'a>>) -> std::io::Result<Self> {
        Ok(Self::new(src.into().owned_reader()?))
    }
}

impl<R: BufRead + Seek> TrajectoryReader for SdfReader<R> {
    fn build_index(&mut self) -> std::io::Result<()> {
        if self.index.is_none() {
            self.index = Some(crate::io::frame_index::index_stream(
                &mut self.reader,
                Box::new(SdfIndexBuilder::new()),
            )?);
        }
        Ok(())
    }

    fn read_frame(&mut self, index: usize) -> std::io::Result<Option<Frame>> {
        self.build_index()?;
        let Some(at) = self.index.as_ref().and_then(|o| o.get(index).copied()) else {
            return Ok(None);
        };
        let bytes = crate::io::frame_index::frame_bytes(&mut self.reader, at)?;
        read_sdf(bytes.as_slice()).map(Some)
    }

    fn len(&mut self) -> std::io::Result<usize> {
        self.build_index()?;
        Ok(self.index.as_ref().map_or(0, Vec::len))
    }
}

impl<R: BufRead> FrameReader for SdfReader<R> {
    fn read(&mut self) -> std::io::Result<Option<Frame>> {
        self.read_single_record()
    }
}

/// Read the first record of an SDF / MDL molfile.
pub fn read_sdf<'a>(src: impl Into<Source<'a>>) -> std::io::Result<Frame> {
    let src = src.into();
    SdfReader::new(src.reader()?)
        .read()?
        .ok_or_else(|| invalid_data(format!("{}: no SDF record", src.name())))
}

// ============================================================================
// Streaming
// ============================================================================

use crate::io::frame_index::{FrameIndexBuilder, FrameOffset, LineAccumulator};

/// Streaming frame indexer for SDF / MDL multi-record files.
///
/// Detects record terminators (`$$$$` lines) and emits one
/// [`FrameOffset`] per record. The terminator line is included in the
/// preceding frame's byte range.
pub struct SdfIndexBuilder {
    lines: LineAccumulator,
    /// Offset of the next record's first byte (i.e. the byte after the
    /// most-recently emitted terminator's line). `0` initially.
    next_record_start: u64,
    pending_entries: Vec<FrameOffset>,
}

impl Default for SdfIndexBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl SdfIndexBuilder {
    pub fn new() -> Self {
        Self {
            lines: LineAccumulator::new(),
            next_record_start: 0,
            pending_entries: Vec::new(),
        }
    }
}

impl FrameIndexBuilder for SdfIndexBuilder {
    fn feed(&mut self, chunk: &[u8], global_offset: u64) {
        let next_record_start = &mut self.next_record_start;
        let pending_entries = &mut self.pending_entries;
        self.lines
            .feed(chunk, global_offset, |line, line_offset, line_len| {
                if line.trim_end() != "$$$$" {
                    return;
                }
                let line_end = line_offset + line_len as u64;
                let start = *next_record_start;
                let span = line_end - start;
                // span fits in u32 because per-record SDF size is always
                // small (< 1 MiB typical). Defensive saturate just in case.
                let len = span.min(u32::MAX as u64) as u32;
                pending_entries.push(FrameOffset {
                    byte_offset: start,
                    byte_len: len,
                });
                *next_record_start = line_end;
            });
    }

    fn drain(&mut self) -> Vec<FrameOffset> {
        std::mem::take(&mut self.pending_entries)
    }

    fn finish(mut self: Box<Self>) -> std::io::Result<Vec<FrameOffset>> {
        self.lines.check_line_budget()?;
        let next_record_start = &mut self.next_record_start;
        let pending_entries = &mut self.pending_entries;
        self.lines.finish(|line, line_offset, line_len| {
            if line.trim_end() != "$$$$" {
                return;
            }
            let line_end = line_offset + line_len as u64;
            let start = *next_record_start;
            let span = line_end - start;
            let len = span.min(u32::MAX as u64) as u32;
            pending_entries.push(FrameOffset {
                byte_offset: start,
                byte_len: len,
            });
            *next_record_start = line_end;
        });

        let bytes_seen = self.lines.bytes_seen();
        // Trailing record without `$$$$`: a single-record `.mol` file has
        // no terminator. Treat the trailing
        // bytes as one final frame iff they look non-empty.
        if self.next_record_start < bytes_seen {
            let span = bytes_seen - self.next_record_start;
            if span > u32::MAX as u64 {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "SDF record exceeds 4 GiB",
                ));
            }
            self.pending_entries.push(FrameOffset {
                byte_offset: self.next_record_start,
                byte_len: span as u32,
            });
        }

        Ok(std::mem::take(&mut self.pending_entries))
    }

    fn bytes_seen(&self) -> u64 {
        self.lines.bytes_seen()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use molrs::core::FrameAccess;
    use std::io::Cursor;

    const WATER_SDF: &str = "962\n  -OEChem-\n\n  3  2  0     0  0  0  0  0  0999 V2000\n    0.0000    0.0000    0.1173 O   0  0  0  0  0  0  0  0  0  0  0  0\n    0.7572    0.0000   -0.4692 H   0  0  0  0  0  0  0  0  0  0  0  0\n   -0.7572    0.0000   -0.4692 H   0  0  0  0  0  0  0  0  0  0  0  0\n  1  2  1  0  0  0  0\n  1  3  1  0  0  0  0\nM  END\n$$$$\n";

    #[test]
    fn reads_water_record() {
        let mut reader = SdfReader::new(Cursor::new(WATER_SDF.as_bytes()));
        let frame = reader.read().expect("read ok").expect("frame present");

        let x = frame
            .column("atoms", "x")
            .and_then(|c| c.get::<f64>())
            .expect("x column");
        assert_eq!(x.shape(), &[3]);
        assert!((x[[0]] - 0.0).abs() < 1e-6);
        assert!((x[[1]] - 0.7572).abs() < 1e-6);

        let atomi = frame
            .column("bonds", "atomi")
            .and_then(|c| c.get::<u64>())
            .expect("atomi column");
        assert_eq!(atomi.shape(), &[2]);
        assert_eq!(atomi[[0]], 0); // 1-based -> 0-based
        assert_eq!(atomi[[1]], 0);
        let atomj = frame
            .column("bonds", "atomj")
            .and_then(|c| c.get::<u64>())
            .expect("atomj column");
        assert_eq!(atomj[[0]], 1);
        assert_eq!(atomj[[1]], 2);
    }

    #[test]
    fn rejects_a_blank_atom_symbol() {
        let bad = WATER_SDF.replacen("0.1173 O ", "0.1173   ", 1);
        let mut reader = SdfReader::new(Cursor::new(bad.as_bytes()));
        let err = reader.read().unwrap_err();
        assert!(err.to_string().contains("blank atom symbol"), "{err}");
    }

    #[test]
    fn rejects_v3000() {
        let bad = "name\n\n\n  0  0  0  0  0  0  0  0  0  0999 V3000\n";
        let mut reader = SdfReader::new(Cursor::new(bad.as_bytes()));
        assert!(reader.read().is_err());
    }

    // -----------------------------------------------------------------
    // Streaming index tests
    // -----------------------------------------------------------------

    fn sdf_build_chunked(bytes: &[u8], cs: usize) -> Vec<FrameOffset> {
        let mut b = Box::new(SdfIndexBuilder::new());
        let mut off: u64 = 0;
        let mut out: Vec<FrameOffset> = Vec::new();
        for piece in bytes.chunks(cs.max(1)) {
            b.feed(piece, off);
            off += piece.len() as u64;
            out.extend(b.drain());
        }
        out.extend(b.finish().expect("finish"));
        out
    }

    fn make_two_record_sdf() -> String {
        format!("{WATER_SDF}{WATER_SDF}")
    }

    /// `SdfReader::open` reads records by index, in any order.
    #[test]
    fn sdf_reader_open_reads_records_by_index() {
        let s = make_two_record_sdf();
        let mut reader = SdfReader::open(s.as_bytes()).unwrap();
        assert_eq!(reader.len().unwrap(), 2);
        assert!(reader.read_frame(1).unwrap().is_some());
        assert!(reader.read_frame(0).unwrap().is_some());
        assert!(reader.read_frame(2).unwrap().is_none());
    }

    #[test]
    fn sdf_streaming_two_records_match_chunks() {
        let s = make_two_record_sdf();
        let bytes = s.as_bytes();
        let one = sdf_build_chunked(bytes, bytes.len());
        assert_eq!(one.len(), 2);
        for cs in [1usize, 7, 13, 31, 64, 1024] {
            let chunked = sdf_build_chunked(bytes, cs);
            assert_eq!(one, chunked, "chunk size {}", cs);
        }
        for entry in &one {
            let lo = entry.byte_offset as usize;
            let hi = lo + entry.byte_len as usize;
            read_sdf(&bytes[lo..hi]).expect("parse SDF record");
        }
    }

    /// Edge: chunk boundary lands inside the literal `$$$$`.
    #[test]
    fn sdf_streaming_boundary_in_terminator() {
        let s = make_two_record_sdf();
        let bytes = s.as_bytes();
        // Find first `$$$$` and split inside it.
        let term_pos = bytes.windows(4).position(|w| w == b"$$$$").expect("$$$$");
        let split = term_pos + 2;
        let mut b = Box::new(SdfIndexBuilder::new());
        b.feed(&bytes[..split], 0);
        b.feed(&bytes[split..], split as u64);
        let mut got = b.drain();
        got.extend(b.finish().expect("finish"));
        let one_shot = sdf_build_chunked(bytes, bytes.len());
        assert_eq!(got, one_shot);
    }

    /// Edge: SDF without trailing `$$$$` — a single-record `.mol` file. Indexer
    /// should still emit one frame from `finish`.
    #[test]
    fn sdf_streaming_no_terminator() {
        // Strip the trailing $$$$\n
        let s = WATER_SDF.replace("$$$$\n", "");
        let bytes = s.as_bytes();
        let entries = sdf_build_chunked(bytes, bytes.len());
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].byte_offset, 0);
        assert_eq!(entries[0].byte_len as usize, bytes.len());
        read_sdf(bytes).expect("parse no-terminator SDF");
    }
}
