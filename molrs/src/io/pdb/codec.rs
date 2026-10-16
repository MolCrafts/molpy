//! The PDB codec: records, the reader and writer classes, the path doors.
//!
//! Implements PDB 3.3 specification for coordinate section records:
//! <https://www.wwpdb.org/documentation/file-format-content/format33/sect9.html>

use crate::core::Cell;
use crate::io::reader::{FrameReader, ReadSeek, Reader, TrajectoryReader};
use crate::io::writer::FrameWriter;
use crate::io::{Sink, SinkWriter, Source, invalid_data};
use molrs::core::Block;
use molrs::core::Frame;
use molrs::core::FrameAccess;
use molrs::core::SimBox;
use molrs::op::{F, I, Idx};
use ndarray::{Array1, IxDyn};
use std::collections::{HashMap, HashSet};
use std::io::{BufRead, Seek, Write};

// ============================================================================
// Record Structs (per PDB 3.3 spec)
// ============================================================================

/// ATOM record - standard amino acid/nucleotide atoms
/// Columns per spec: 1-6 record, 7-11 serial, 13-16 name, 17 altLoc,
/// 18-20 resName, 22 chainID, 23-26 resSeq, 27 iCode, 31-38 x, 39-46 y,
/// 47-54 z, 55-60 occupancy, 61-66 tempFactor, 77-78 element, 79-80 charge
#[derive(Debug, Clone, PartialEq)]
struct AtomRecord {
    pub serial: i32,
    pub name: String,
    pub alt_loc: char,
    pub res_name: String,
    pub chain_id: char,
    pub res_seq: i32,
    pub i_code: char,
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub occupancy: f32,
    pub temp_factor: f32,
    pub element: String,
    pub charge: String,
}

/// CONECT record - bond connectivity
#[derive(Debug, Clone, PartialEq)]
struct ConectRecord {
    pub serial: i32,
    pub bonded: Vec<i32>,
}

/// CRYST1 record - unit cell parameters
#[derive(Debug, Clone, PartialEq)]
struct Cryst1Record {
    pub a: f32,
    pub b: f32,
    pub c: f32,
    pub alpha: f32,
    pub beta: f32,
    pub gamma: f32,
    pub space_group: String,
    pub z: i32,
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Get char at position, defaulting to space
fn char_at(s: &str, idx: usize) -> char {
    s.chars().nth(idx).unwrap_or(' ')
}

/// Get substring safely, returning empty string if out of bounds
fn substr(s: &str, start: usize, end: usize) -> &str {
    let len = s.len();
    if start >= len {
        return "";
    }
    let end = end.min(len);
    &s[start..end]
}

/// The periodic-table spelling of a PDB element field: surrounding blanks
/// dropped, first letter upper case, the rest lower case (`" C"` → `"C"`,
/// `"FE"` → `"Fe"`). Empty when the field is blank.
fn element_symbol(field: &str) -> String {
    let mut chars = field.trim().chars();
    let Some(first) = chars.next() else {
        return String::new();
    };
    let mut symbol = first.to_ascii_uppercase().to_string();
    symbol.extend(chars.map(|c| c.to_ascii_lowercase()));
    symbol
}

// ============================================================================
// Record Parsers
// ============================================================================

/// Internal implementation for parsing ATOM or HETATM records
fn parse_atom_or_hetatm_impl(
    line: &str,
    expected_record: &str,
) -> std::io::Result<Option<AtomRecord>> {
    // Minimum length check
    if line.len() < 54 {
        return Ok(None);
    }

    // Check record type (columns 1-6, 0-indexed: 0-6)
    let record = substr(line, 0, 6).trim();
    if record != expected_record {
        return Ok(None);
    }

    // Parse fixed-width fields (PDB uses 1-indexed columns, we use 0-indexed)
    let serial = substr(line, 6, 11)
        .trim()
        .parse::<i32>()
        .map_err(invalid_data)?;
    let name = substr(line, 12, 16).trim().to_string();
    let alt_loc = char_at(line, 16);
    let res_name = substr(line, 17, 20).trim().to_string();
    let chain_id = char_at(line, 21);
    let res_seq_str = substr(line, 22, 26).trim();
    if res_seq_str.is_empty() {
        return Err(invalid_data("missing res_seq"));
    }
    let res_seq = res_seq_str.parse::<i32>().map_err(invalid_data)?;
    let i_code = char_at(line, 26);
    let x = substr(line, 30, 38)
        .trim()
        .parse::<f32>()
        .map_err(invalid_data)?;
    let y = substr(line, 38, 46)
        .trim()
        .parse::<f32>()
        .map_err(invalid_data)?;
    let z = substr(line, 46, 54)
        .trim()
        .parse::<f32>()
        .map_err(invalid_data)?;

    // Optional fields
    let occupancy_str = substr(line, 54, 60).trim();
    let occupancy = if occupancy_str.is_empty() {
        1.0
    } else {
        occupancy_str.parse::<f32>().map_err(invalid_data)?
    };
    // make sure occupancy is between 0 and 1
    if !(0.0..=1.0).contains(&occupancy) {
        return Err(invalid_data(
            "occupancy out of range (0.0 - 1.0) in ".to_string() + line,
        ));
    }
    let temp_factor_str = substr(line, 60, 66).trim();
    let temp_factor = if temp_factor_str.is_empty() {
        0.0
    } else {
        temp_factor_str.parse::<f32>().map_err(invalid_data)?
    };
    // make sure temp_factor is non-negative
    if temp_factor < 0.0 {
        return Err(invalid_data("temp_factor negative in ".to_string() + line));
    }

    // Element (columns 77-78, 0-indexed: 76-78): right-justified and upper
    // case in the file (" C", "FE"), stored as the periodic-table spelling
    // ("C", "Fe") so no caller has to trim or re-case it. A blank field stays
    // empty: the species is never guessed from the atom name.
    let element = if line.len() >= 78 {
        element_symbol(substr(line, 76, 78))
    } else {
        String::new()
    };

    // Charge (columns 79-80)
    let charge = if line.len() >= 80 {
        substr(line, 78, 80).trim().to_string()
    } else {
        String::new()
    };

    Ok(Some(AtomRecord {
        serial,
        name,
        alt_loc,
        res_name,
        chain_id,
        res_seq,
        i_code,
        x,
        y,
        z,
        occupancy,
        temp_factor,
        element,
        charge,
    }))
}

/// Parse ATOM record from line
fn parse_atom_record(line: &str) -> std::io::Result<Option<AtomRecord>> {
    parse_atom_or_hetatm_impl(line, "ATOM")
}

/// Parse HETATM record from line (same format as ATOM)
fn parse_hetatm_record(line: &str) -> std::io::Result<Option<AtomRecord>> {
    parse_atom_or_hetatm_impl(line, "HETATM")
}

/// Parse CONECT record
fn parse_conect_record(line: &str) -> Option<ConectRecord> {
    if !line.starts_with("CONECT") {
        return None;
    }

    // Parse serial numbers from columns 7-11, 12-16, 17-21, 22-26, 27-31
    let mut bonded = Vec::new();
    let serial = substr(line, 6, 11).trim().parse::<i32>().ok()?;

    // Parse bonded atoms (up to 4 per line in standard CONECT)
    for i in 0..4 {
        let start = 11 + i * 5;
        let end = start + 5;
        if line.len() >= end
            && let Ok(bonded_serial) = substr(line, start, end).trim().parse::<i32>()
            && bonded_serial > 0
        {
            bonded.push(bonded_serial);
        }
    }

    Some(ConectRecord { serial, bonded })
}

/// Parse CRYST1 record
fn parse_cryst1_record(line: &str) -> Option<Cryst1Record> {
    if !line.starts_with("CRYST1") {
        return None;
    }

    if line.len() < 54 {
        return None;
    }

    let a = substr(line, 6, 15).trim().parse::<f32>().ok()?;
    let b = substr(line, 15, 24).trim().parse::<f32>().ok()?;
    let c = substr(line, 24, 33).trim().parse::<f32>().ok()?;
    let alpha = substr(line, 33, 40).trim().parse::<f32>().unwrap_or(90.0);
    let beta = substr(line, 40, 47).trim().parse::<f32>().unwrap_or(90.0);
    let gamma = substr(line, 47, 54).trim().parse::<f32>().unwrap_or(90.0);

    let space_group = if line.len() >= 66 {
        substr(line, 55, 66).trim().to_string()
    } else {
        String::new()
    };

    let z = if line.len() >= 70 {
        substr(line, 66, 70).trim().parse::<i32>().unwrap_or(1)
    } else {
        1
    };

    Some(Cryst1Record {
        a,
        b,
        c,
        alpha,
        beta,
        gamma,
        space_group,
        z,
    })
}

/// Check if line is ENDMDL
fn is_endmdl(line: &str) -> bool {
    line.trim().starts_with("ENDMDL")
}

/// Check if line is END
fn is_end(line: &str) -> bool {
    line.trim() == "END"
}

// ============================================================================
// Frame Building
// ============================================================================

/// Build a Frame from parsed records
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

fn to_array_string(vec: Vec<String>, len: usize) -> std::io::Result<ndarray::ArrayD<String>> {
    Ok(Array1::from_vec(vec)
        .into_shape_with_order(IxDyn(&[len]))
        .map_err(invalid_data)?
        .into_dyn())
}

fn build_atoms_block(atoms: &[AtomRecord]) -> std::io::Result<(Block, String, HashMap<i32, Idx>)> {
    let n = atoms.len();
    let mut x_vec = Vec::with_capacity(n);
    let mut y_vec = Vec::with_capacity(n);
    let mut z_vec = Vec::with_capacity(n);
    let mut ids_vec: Vec<Idx> = Vec::with_capacity(n);
    let mut elements = Vec::with_capacity(n);
    let mut names: Vec<String> = Vec::with_capacity(n);
    let mut res_names: Vec<String> = Vec::with_capacity(n);
    let mut res_seqs: Vec<I> = Vec::with_capacity(n);
    let mut chains: Vec<String> = Vec::with_capacity(n);
    let mut altlocs: Vec<String> = Vec::with_capacity(n);
    let mut icodes: Vec<String> = Vec::with_capacity(n);
    let mut occupancies: Vec<F> = Vec::with_capacity(n);
    let mut b_factors: Vec<F> = Vec::with_capacity(n);
    let mut serial_map: HashMap<i32, Idx> = HashMap::with_capacity(n);

    for (i, atom) in atoms.iter().enumerate() {
        x_vec.push(atom.x as F);
        y_vec.push(atom.y as F);
        z_vec.push(atom.z as F);
        ids_vec.push(atom.serial as Idx);
        elements.push(atom.element.clone());
        names.push(atom.name.clone());
        res_names.push(atom.res_name.clone());
        res_seqs.push(atom.res_seq);
        // Chain ID is a single character per the PDB format spec; expose it
        // as a String so the block schema is uniform across formats and so
        // mmCIF files (multi-char chain IDs) can drop in later without
        // breaking downstream consumers.
        chains.push(blank_as_empty(atom.chain_id));
        altlocs.push(blank_as_empty(atom.alt_loc));
        icodes.push(blank_as_empty(atom.i_code));
        occupancies.push(atom.occupancy as F);
        b_factors.push(atom.temp_factor as F);
        serial_map.insert(atom.serial, i as Idx);
    }

    let unique_elements = collect_unique_elements(&elements);
    // A blank element field is an absent species, not a placeholder one: the
    // row is null in the validity mask (its value reads ""), so the explicit
    // `Element::fill_atomic_numbers` step reports it by row. A file that
    // spells no element at all gets no `element` column.
    let element_valid: Vec<bool> = elements.iter().map(|e| !e.is_empty()).collect();
    let any_element = element_valid.iter().any(|&v| v);

    let mut block = Block::new();
    block
        .insert("x", to_array_float(x_vec, n)?, None)
        .map_err(invalid_data)?;
    block
        .insert("y", to_array_float(y_vec, n)?, None)
        .map_err(invalid_data)?;
    block
        .insert("z", to_array_float(z_vec, n)?, None)
        .map_err(invalid_data)?;
    block
        .insert("id", to_array_uint(ids_vec, n)?, None)
        .map_err(invalid_data)?;
    if any_element {
        block
            .insert(
                "element",
                to_array_string(elements, n)?,
                Some(element_valid),
            )
            .map_err(invalid_data)?;
    }
    block
        .insert("name", to_array_string(names, n)?, None)
        .map_err(invalid_data)?;
    block
        .insert("res_name", to_array_string(res_names, n)?, None)
        .map_err(invalid_data)?;
    // Emit the canonical `res_id` directly rather than a format-native
    // `res_seq` that something downstream renames: the rename would be a write
    // into a UInt key, so an Int column could not survive it anyway.
    let res_ids: Vec<Idx> = res_seqs
        .iter()
        .map(|&v| {
            Idx::try_from(v).map_err(|_| {
                invalid_data(format!(
                    "PDB residue sequence number {v} is negative; residue ids are unsigned"
                ))
            })
        })
        .collect::<std::io::Result<_>>()?;
    block
        .insert("res_id", to_array_uint(res_ids, n)?, None)
        .map_err(invalid_data)?;
    block
        .insert("chain", to_array_string(chains, n)?, None)
        .map_err(invalid_data)?;
    block
        .insert("icode", to_array_string(icodes, n)?, None)
        .map_err(invalid_data)?;
    block
        .insert("altloc", to_array_string(altlocs, n)?, None)
        .map_err(invalid_data)?;
    block
        .insert("occupancy", to_array_float(occupancies, n)?, None)
        .map_err(invalid_data)?;
    block
        .insert("b_factor", to_array_float(b_factors, n)?, None)
        .map_err(invalid_data)?;

    Ok((block, unique_elements, serial_map))
}

/// A one-character PDB field as a string, `""` for the blank that means
/// "none" (chain, altLoc, iCode).
fn blank_as_empty(c: char) -> String {
    if c == ' ' {
        String::new()
    } else {
        c.to_string()
    }
}

fn collect_unique_elements(elements: &[String]) -> String {
    let mut unique = Vec::new();
    for elem in elements.iter().filter(|e| !e.is_empty()) {
        if !unique.contains(elem) {
            unique.push(elem.clone());
        }
    }
    unique.join("|")
}

fn build_bonds_block(
    conects: &[ConectRecord],
    serial_map: &HashMap<i32, Idx>,
) -> std::io::Result<Option<Block>> {
    if conects.is_empty() {
        return Ok(None);
    }

    // A CONECT pair is listed once from each end (and some writers repeat a
    // partner to mark a multiple bond); a bond is the undirected pair, kept
    // once, at its first listing.
    let mut seen: HashSet<(Idx, Idx)> = HashSet::new();
    let mut i_indices: Vec<Idx> = Vec::new();
    let mut j_indices: Vec<Idx> = Vec::new();

    for conect in conects {
        if let Some(&idx1) = serial_map.get(&conect.serial) {
            for &bonded_serial in &conect.bonded {
                if let Some(&idx2) = serial_map.get(&bonded_serial)
                    && seen.insert((idx1.min(idx2), idx1.max(idx2)))
                {
                    i_indices.push(idx1);
                    j_indices.push(idx2);
                }
            }
        }
    }

    if i_indices.is_empty() {
        return Ok(None);
    }

    let bn = i_indices.len();
    let mut block = Block::new();
    block
        .insert("atomi", to_array_uint(i_indices, bn)?, None)
        .map_err(invalid_data)?;
    block
        .insert("atomj", to_array_uint(j_indices, bn)?, None)
        .map_err(invalid_data)?;

    Ok(Some(block))
}

fn add_simbox_from_cryst1(frame: &mut Frame, cryst1: Option<&Cryst1Record>) {
    if let Some(cryst) = cryst1
        && cryst.a > 0.0
        && cryst.b > 0.0
        && cryst.c > 0.0
    {
        let lengths = [cryst.a as F, cryst.b as F, cryst.c as F];
        let origin = [0.0 as F, 0.0, 0.0];
        let pbc = [true, true, true];
        if let Ok(simbox) = SimBox::new(Cell::Lengths(lengths), origin, pbc) {
            frame.simbox = Some(simbox);
        }
    }
}

fn build_frame(
    atoms: &[AtomRecord],
    cryst1: Option<&Cryst1Record>,
    conects: &[ConectRecord],
) -> std::io::Result<Frame> {
    if atoms.is_empty() {
        return Ok(Frame::new());
    }

    let mut frame = Frame::new();

    let (atoms_block, elements_metadata, serial_map) = build_atoms_block(atoms)?;
    frame.insert("atoms", atoms_block);
    frame.meta.insert("elements".to_string(), elements_metadata);

    if let Some(bonds_block) = build_bonds_block(conects, &serial_map)? {
        frame.insert("bonds", bonds_block);
    }

    add_simbox_from_cryst1(&mut frame, cryst1);

    Ok(frame)
}

// ============================================================================
// Reader (single-frame)
// ============================================================================

/// PDB reader: [`FrameReader::read`] streams one `MODEL` per call; over a
/// seekable stream it is also a [`TrajectoryReader`] (random access by model,
/// indexed by [`PdbIndexBuilder`] on first use).
pub struct PdbReader<R: BufRead> {
    reader: R,
    index: Option<Vec<FrameOffset>>,
}

impl PdbReader<Box<dyn ReadSeek>> {
    /// Open a PDB file, or the bytes of one, for random access by model.
    pub fn open<'a>(src: impl Into<Source<'a>>) -> std::io::Result<Self> {
        Ok(Self::new(src.into().owned_reader()?))
    }
}

impl<R: BufRead + Seek> TrajectoryReader for PdbReader<R> {
    fn build_index(&mut self) -> std::io::Result<()> {
        if self.index.is_none() {
            let offsets = crate::io::frame_index::index_stream(
                &mut self.reader,
                Box::new(PdbIndexBuilder::new()),
            )?;
            self.index = Some(offsets);
        }
        Ok(())
    }

    fn read_frame(&mut self, index: usize) -> std::io::Result<Option<Frame>> {
        self.build_index()?;
        let Some(&at) = self.index.as_ref().and_then(|offsets| offsets.get(index)) else {
            return Ok(None);
        };
        let bytes = crate::io::frame_index::frame_bytes(&mut self.reader, at)?;
        crate::io::reader::check_read_frame(Some(read_pdb(bytes.as_slice())?))
    }

    fn len(&mut self) -> std::io::Result<usize> {
        self.build_index()?;
        Ok(self.index.as_ref().map_or(0, Vec::len))
    }
}

impl<R: BufRead> PdbReader<R> {
    pub fn new(reader: R) -> Self {
        Self {
            reader,
            index: None,
        }
    }

    fn read_single_frame(&mut self) -> std::io::Result<Option<Frame>> {
        let mut atoms = Vec::new();
        let mut conects = Vec::new();
        let mut cryst1 = None;
        let mut line = String::new();

        loop {
            line.clear();
            if self.reader.read_line(&mut line)? == 0 {
                break;
            }
            let trimmed = line.trim();

            if is_endmdl(trimmed) || is_end(trimmed) {
                break;
            } else if let Some(cryst) = parse_cryst1_record(&line) {
                cryst1 = Some(cryst);
            } else if let Some(atom) = parse_atom_record(&line)? {
                atoms.push(atom);
            } else if let Some(hetatm) = parse_hetatm_record(&line)? {
                atoms.push(hetatm);
            } else if let Some(conect) = parse_conect_record(&line) {
                conects.push(conect);
            }
        }

        if atoms.is_empty() {
            return Ok(None);
        }

        Ok(Some(build_frame(&atoms, cryst1.as_ref(), &conects)?))
    }
}

impl<R: BufRead> Reader for PdbReader<R> {
    type R = R;

    fn new(reader: R) -> Self {
        PdbReader::new(reader)
    }
}

impl<R: BufRead> FrameReader for PdbReader<R> {
    fn read(&mut self) -> std::io::Result<Option<Frame>> {
        // Validate on the way out: a frame that violates the vocabulary
        // is a malformed file or a reader bug, not a result to return.
        crate::io::reader::check_read_frame(self.read_single_frame()?)
    }
}

// ============================================================================
// Writer
// ============================================================================

/// A multi-model PDB writer: the first [`FrameWriter::write`] writes the
/// `CRYST1` record of that frame, and every write one `MODEL n` … `ENDMDL`
/// block — what [`PdbReader`] reads back model by model.
///
/// The writer ends no model with `END`: it cannot know which model is the
/// last. A one-frame PDB file with `END` is [`write_pdb`]'s.
pub struct PdbWriter<W: Write> {
    writer: W,
    models: usize,
}

impl<W: Write> crate::io::writer::Writer for PdbWriter<W> {
    type W = W;
    fn new(writer: W) -> Self {
        Self { writer, models: 0 }
    }
}

impl<W: Write> FrameWriter for PdbWriter<W> {
    fn write(&mut self, frame: &Frame) -> std::io::Result<()> {
        // Refuse to emit a frame that violates the vocabulary: a bad file
        // looks fine and is found wrong later, by whatever reads it.
        crate::io::writer::check_write_frame(frame)?;
        if self.models == 0 {
            write_cryst1(&mut self.writer, frame)?;
        }
        self.models += 1;
        write_model(&mut self.writer, self.models, frame)?;
        self.writer.flush()
    }
}

impl<'a> PdbWriter<SinkWriter<'a>> {
    /// Create a PDB file, or fill a buffer, one `MODEL` per
    /// [`FrameWriter::write`].
    pub fn create(sink: impl Into<Sink<'a>>) -> std::io::Result<Self> {
        Ok(Self {
            writer: sink.into().writer()?,
            models: 0,
        })
    }
}

/// Write the `CRYST1` record from the frame's simulation box, if any.
fn write_cryst1<W: Write>(writer: &mut W, frame: &impl FrameAccess) -> std::io::Result<()> {
    // Always emit CRYST1: PDB readers and OpenMM decks expect a cell line even
    // when the frame has no simbox, which is written as a unit cube.
    let (a, b, c) = if let Some(simbox) = frame.simbox_ref() {
        let lengths = simbox.lengths();
        (lengths[0], lengths[1], lengths[2])
    } else {
        (1.0, 1.0, 1.0)
    };
    writeln!(
        writer,
        "CRYST1{:9.3}{:9.3}{:9.3}{:7.2}{:7.2}{:7.2} {:<11}{:>4}",
        a, b, c, 90.00, 90.00, 90.00, "P 1", 1
    )?;
    Ok(())
}

/// Write `ATOM` records (PDB v3.3 column layout) plus `CONECT` records derived
/// from the `bonds` block. Emits neither `CRYST1` nor a frame terminator
/// (`END`/`ENDMDL`) — callers wrap as needed (see [`write_pdb`](crate::io::write_pdb) and
/// [`PdbWriter`]).
fn write_atom_conect_records<W: Write>(
    writer: &mut W,
    frame: &impl FrameAccess,
) -> std::io::Result<()> {
    let x = frame
        .column("atoms", "x")
        .and_then(|c| c.get::<f64>())
        .ok_or_else(|| invalid_data("Missing 'x' column"))?;
    let y = frame
        .column("atoms", "y")
        .and_then(|c| c.get::<f64>())
        .ok_or_else(|| invalid_data("Missing 'y' column"))?;
    let z = frame
        .column("atoms", "z")
        .and_then(|c| c.get::<f64>())
        .ok_or_else(|| invalid_data("Missing 'z' column"))?;
    let n = x
        .shape()
        .first()
        .copied()
        .ok_or_else(|| invalid_data("Empty 'atoms' block"))?;

    let x_slice = x
        .as_slice_memory_order()
        .ok_or_else(|| invalid_data("Non-contiguous 'x' column"))?;
    let y_slice = y
        .as_slice_memory_order()
        .ok_or_else(|| invalid_data("Non-contiguous 'y' column"))?;
    let z_slice = z
        .as_slice_memory_order()
        .ok_or_else(|| invalid_data("Non-contiguous 'z' column"))?;

    // Optional per-atom string columns (snake_case, as emitted by the reader).
    let owned_str = |col: &str| -> Vec<String> {
        frame
            .column("atoms", col)
            .and_then(|c| c.get::<String>())
            .as_ref()
            .and_then(|arr| arr.as_slice().map(|s| s.to_vec()))
            .unwrap_or_default()
    };
    let names = owned_str("name");
    let res_names = owned_str("res_name");
    let chains = owned_str("chain");
    let altlocs = owned_str("altloc");
    let icodes = owned_str("icode");
    let owned_f64 = |col: &str| -> Vec<F> {
        frame
            .column("atoms", col)
            .and_then(|c| c.get::<f64>())
            .as_ref()
            .and_then(|arr| arr.as_slice().map(|s| s.to_vec()))
            .unwrap_or_default()
    };
    let occupancies = owned_f64("occupancy");
    let b_factors = owned_f64("b_factor");
    // PDB spells species as a symbol (columns 77-78); the writer reads the
    // `element` column and nothing else. A row whose element is empty writes
    // the blank field PDB itself uses for "not stated".
    let elements: Vec<String> = frame
        .column("atoms", "element")
        .and_then(|c| c.get::<String>())
        .map(|arr| arr.iter().cloned().collect())
        .ok_or_else(|| {
            invalid_data(
                "PDB writer: atoms block has no string 'element' column; a frame \
                 that carries only atomic_number needs \
                 `Element::fill_symbols(&mut frame)` first",
            )
        })?;

    let res_seqs: Vec<Idx> = frame
        .column("atoms", "res_id")
        .and_then(|c| c.get::<u64>())
        .as_ref()
        .and_then(|arr| arr.as_slice().map(|s| s.to_vec()))
        .unwrap_or_default();

    let ids: Vec<Idx> = frame
        .column("atoms", "id")
        .and_then(|c| c.get::<u64>())
        .as_ref()
        .and_then(|arr| arr.as_slice().map(|s| s.to_vec()))
        .unwrap_or_default();
    let has_ids = !ids.is_empty();

    let mut serials = Vec::with_capacity(n);
    for i in 0..n {
        let serial = if has_ids { ids[i] as usize } else { i + 1 };
        serials.push(serial);

        let elem_raw = elements[i].trim();

        // Atom name (cols 13-16): single-char names are offset one column. A
        // frame with no `name` column writes the field blank — the name is
        // not made up from the element.
        let name_raw = names.get(i).map(|s| s.trim()).unwrap_or("");
        let name_field = if name_raw.chars().count() == 1 {
            format!(" {:<3}", name_raw)
        } else {
            let truncated: String = name_raw.chars().take(4).collect();
            format!("{:<4}", truncated)
        };

        let res_name: String = res_names
            .get(i)
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .unwrap_or("UNK")
            .chars()
            .take(3)
            .collect();

        let one_char = |values: &[String]| {
            values
                .get(i)
                .and_then(|s| s.trim().chars().next())
                .unwrap_or(' ')
        };
        let chain = one_char(&chains);
        let altloc = one_char(&altlocs);
        let icode = one_char(&icodes);
        let occupancy = occupancies.get(i).copied().unwrap_or(1.0);
        let b_factor = b_factors.get(i).copied().unwrap_or(0.0);

        let res_seq = res_seqs.get(i).copied().unwrap_or(1);

        let elem_field: String = elem_raw
            .chars()
            .take(2)
            .collect::<String>()
            .to_ascii_uppercase();

        // PDB v3.3 ATOM record. occupancy/tempFactor default to 1.00/0.00
        // when the frame carries no `occupancy` / `b_factor`. Element
        // right-justified in cols 77-78, then one charge pad space; every
        // line is padded to 79 printable columns + newline.
        let mut line = format!(
            "ATOM  {:>5} {}{}{:<3} {}{:>4}{}   {:>8.3}{:>8.3}{:>8.3}{:>6.2}{:>6.2}          {:>2}  ",
            serial,
            name_field,
            altloc,
            res_name,
            chain,
            res_seq,
            icode,
            x_slice[i],
            y_slice[i],
            z_slice[i],
            occupancy,
            b_factor,
            elem_field,
        );
        if line.len() < 79 {
            line = format!("{:<79}", line);
        } else if line.len() > 79 {
            line.truncate(79);
        }
        writeln!(writer, "{line}")?;
    }

    // Write CONECT records from bonds block
    if frame.contains_block("bonds") {
        let bond_data: Option<Result<Vec<Vec<usize>>, std::io::Error>> =
            frame.visit_block("bonds", |bonds| {
                let bn = bonds.n_rows().unwrap_or(0);
                if bn == 0 {
                    return Ok(vec![Vec::new(); n]);
                }

                let i_arr = bonds
                    .column("atomi")
                    .and_then(|c| c.get::<u64>())
                    .ok_or_else(|| invalid_data("Bonds block missing 'atomi' column"))?;
                let j_arr = bonds
                    .column("atomj")
                    .and_then(|c| c.get::<u64>())
                    .ok_or_else(|| invalid_data("Bonds block missing 'atomj' column"))?;

                let i_slice = i_arr
                    .as_slice_memory_order()
                    .ok_or_else(|| invalid_data("Non-contiguous bonds 'atomi' column"))?;
                let j_slice = j_arr
                    .as_slice_memory_order()
                    .ok_or_else(|| invalid_data("Non-contiguous bonds 'atomj' column"))?;

                let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
                for b in 0..bn {
                    let idx_i = i_slice[b] as usize;
                    let idx_j = j_slice[b] as usize;
                    if idx_i >= n || idx_j >= n {
                        return Err(invalid_data("Bond index out of range for atoms"));
                    }
                    adj[idx_i].push(serials[idx_j]);
                    adj[idx_j].push(serials[idx_i]);
                }
                Ok(adj)
            });

        if let Some(adj_result) = bond_data {
            let mut adj = adj_result?;
            for (atom_idx, neighbors) in adj.iter_mut().enumerate() {
                if neighbors.is_empty() {
                    continue;
                }
                neighbors.sort_unstable();
                neighbors.dedup();

                let serial = serials[atom_idx];
                for chunk in neighbors.chunks(4) {
                    write!(writer, "CONECT{:>5}", serial)?;
                    for &bond_serial in chunk {
                        write!(writer, "{:>5}", bond_serial)?;
                    }
                    writeln!(writer)?;
                }
            }
        }
    }

    Ok(())
}

/// Write a single frame in PDB format (`CRYST1` + `ATOM`/`CONECT` + `END`).
///
/// Accepts any type implementing [`FrameAccess`], including both [`Frame`] and
/// [`FrameView`](molrs::core::FrameView).
fn write_frame_to<W: Write>(writer: &mut W, frame: &impl FrameAccess) -> std::io::Result<()> {
    // REMARK: meta "name", else "MOL" (molpy / OpenMM deck convention).
    let title = frame
        .meta_ref()
        .get("name")
        .and_then(|v| v.as_str())
        .unwrap_or("MOL");
    writeln!(writer, "REMARK  {title}")?;
    write_cryst1(writer, frame)?;
    write_atom_conect_records(writer, frame)?;
    // Blank line before END matches long-standing molpy PDB writer output.
    writeln!(writer)?;
    writeln!(writer, "END")?;
    Ok(())
}

/// One `MODEL serial` … `ENDMDL` block.
fn write_model<W: Write>(
    writer: &mut W,
    serial: usize,
    frame: &impl FrameAccess,
) -> std::io::Result<()> {
    writeln!(writer, "MODEL     {:>4}", serial)?;
    write_atom_conect_records(writer, frame)?;
    writeln!(writer, "ENDMDL")
}

// ============================================================================
// Convenience Functions
// ============================================================================

/// Read the first frame (`MODEL`) of a PDB file.
///
/// # Examples
///
/// ```
/// use molrs::io::read_pdb;
///
/// let text = "ATOM      1  O   HOH A   1       0.000   0.000   0.000  1.00  0.00           O\nEND\n";
/// let frame = read_pdb(text.as_bytes())?;
/// assert_eq!(frame.get("atoms").and_then(|b| b.n_rows()), Some(1));
/// # Ok::<(), std::io::Error>(())
/// ```
pub fn read_pdb<'a>(src: impl Into<Source<'a>>) -> std::io::Result<Frame> {
    let src = src.into();
    PdbReader::new(src.reader()?)
        .read()?
        .ok_or_else(|| invalid_data(format!("{}: no PDB frame", src.name())))
}

/// Write one frame as a PDB file (`REMARK`, `CRYST1`, `ATOM`/`CONECT`, `END`).
pub fn write_pdb<'a>(sink: impl Into<Sink<'a>>, frame: &impl FrameAccess) -> std::io::Result<()> {
    let mut out = sink.into().writer()?;
    write_frame_to(&mut out, frame)?;
    out.flush()
}

// ============================================================================
// Streaming
// ============================================================================

use crate::io::frame_index::{FrameIndexBuilder, FrameOffset, LineAccumulator};

#[derive(Debug, Clone, Copy)]
enum PdbMode {
    /// No `MODEL` line has been observed; the entire file will be one frame.
    Single,
    /// `MODEL` has been observed; frame boundaries are `MODEL` / `ENDMDL`.
    Multi,
}

/// Streaming frame indexer for PDB files.
///
/// The state machine tracks two modes:
/// * `Single`: no `MODEL` record has been seen — the whole file is one
///   frame, emitted by `finish`.
/// * `Multi`: `MODEL` opens a frame, `ENDMDL` (or another `MODEL` without
///   a preceding `ENDMDL`) closes it.
///
/// The builder tolerates chunk boundaries inside the `MODEL`/`ENDMDL`
/// literals.
pub struct PdbIndexBuilder {
    lines: LineAccumulator,
    mode: PdbMode,
    pending_frame_start: Option<u64>,
    pending_entries: Vec<FrameOffset>,
}

impl Default for PdbIndexBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl PdbIndexBuilder {
    pub fn new() -> Self {
        Self {
            lines: LineAccumulator::new(),
            mode: PdbMode::Single,
            pending_frame_start: None,
            pending_entries: Vec::new(),
        }
    }
}

impl FrameIndexBuilder for PdbIndexBuilder {
    fn feed(&mut self, chunk: &[u8], global_offset: u64) {
        let mode = &mut self.mode;
        let pending_frame_start = &mut self.pending_frame_start;
        let pending_entries = &mut self.pending_entries;
        self.lines
            .feed(chunk, global_offset, |line, line_offset, line_len| {
                let trimmed = line.trim_start();
                if trimmed.starts_with("MODEL") {
                    *mode = PdbMode::Multi;
                    if let Some(prev) = pending_frame_start.replace(line_offset) {
                        // A MODEL without a preceding ENDMDL — close the
                        // previous frame at the new MODEL's offset.
                        let len = (line_offset - prev) as u32;
                        pending_entries.push(FrameOffset {
                            byte_offset: prev,
                            byte_len: len,
                        });
                    }
                } else if trimmed.starts_with("ENDMDL")
                    && let Some(prev) = pending_frame_start.take()
                {
                    let line_end = line_offset + line_len as u64;
                    let len = (line_end - prev) as u32;
                    pending_entries.push(FrameOffset {
                        byte_offset: prev,
                        byte_len: len,
                    });
                }
            });
    }

    fn drain(&mut self) -> Vec<FrameOffset> {
        std::mem::take(&mut self.pending_entries)
    }

    fn finish(mut self: Box<Self>) -> std::io::Result<Vec<FrameOffset>> {
        self.lines.check_line_budget()?;
        let mode = &mut self.mode;
        let pending_frame_start = &mut self.pending_frame_start;
        let pending_entries = &mut self.pending_entries;
        self.lines.finish(|line, line_offset, line_len| {
            let trimmed = line.trim_start();
            if trimmed.starts_with("MODEL") {
                *mode = PdbMode::Multi;
                if let Some(prev) = pending_frame_start.replace(line_offset) {
                    let len = (line_offset - prev) as u32;
                    pending_entries.push(FrameOffset {
                        byte_offset: prev,
                        byte_len: len,
                    });
                }
            } else if trimmed.starts_with("ENDMDL")
                && let Some(prev) = pending_frame_start.take()
            {
                let line_end = line_offset + line_len as u64;
                let len = (line_end - prev) as u32;
                pending_entries.push(FrameOffset {
                    byte_offset: prev,
                    byte_len: len,
                });
            }
        });

        let bytes_seen = self.lines.bytes_seen();
        match self.mode {
            PdbMode::Single => {
                if bytes_seen > 0 {
                    if bytes_seen > u32::MAX as u64 {
                        return Err(std::io::Error::new(
                            std::io::ErrorKind::InvalidData,
                            "PDB frame size exceeds 4 GiB",
                        ));
                    }
                    self.pending_entries.push(FrameOffset {
                        byte_offset: 0,
                        byte_len: bytes_seen as u32,
                    });
                }
            }
            PdbMode::Multi => {
                // If a MODEL was opened but never closed (no ENDMDL),
                // include it as a trailing frame spanning to EOF.
                if let Some(prev) = self.pending_frame_start.take() {
                    let span = bytes_seen.saturating_sub(prev);
                    if span > u32::MAX as u64 {
                        return Err(std::io::Error::new(
                            std::io::ErrorKind::InvalidData,
                            "PDB frame size exceeds 4 GiB",
                        ));
                    }
                    self.pending_entries.push(FrameOffset {
                        byte_offset: prev,
                        byte_len: span as u32,
                    });
                }
            }
        }

        Ok(std::mem::take(&mut self.pending_entries))
    }

    fn bytes_seen(&self) -> u64 {
        self.lines.bytes_seen()
    }
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_ATOM_LINE: &str =
        "ATOM      1  N   ALA A   1       1.000   2.000   3.000  1.00 20.00           N  ";
    const SAMPLE_HETATM_LINE: &str =
        "HETATM  100  O   HOH A 501      10.000  20.000  30.000  1.00  0.00           O  ";
    const SAMPLE_HETATM_LINE_NO_OCC: &str =
        "HETATM    1  O   HOH     1      10.203   7.604  12.673";
    const SAMPLE_CRYST1_LINE: &str =
        "CRYST1   50.000   60.000   70.000  90.00  90.00  90.00 P 1           1";
    const SAMPLE_CONECT_LINE: &str = "CONECT    1    2    3    4";

    #[test]
    fn test_parse_atom_record() {
        let atom = parse_atom_record(SAMPLE_ATOM_LINE)
            .expect("Failed to parse ATOM")
            .expect("Missing ATOM record");
        assert_eq!(atom.serial, 1);
        assert_eq!(atom.name, "N");
        assert_eq!(atom.res_name, "ALA");
        assert_eq!(atom.chain_id, 'A');
        assert_eq!(atom.res_seq, 1);
        assert!((atom.x - 1.0).abs() < 0.001);
        assert!((atom.y - 2.0).abs() < 0.001);
        assert!((atom.z - 3.0).abs() < 0.001);
        assert!((atom.occupancy - 1.0).abs() < 0.001);
        assert!((atom.temp_factor - 20.0).abs() < 0.001);
        assert_eq!(atom.element, "N");
    }

    #[test]
    fn element_field_is_trimmed_and_cased() {
        assert_eq!(element_symbol(" C"), "C");
        assert_eq!(element_symbol("FE"), "Fe");
        assert_eq!(element_symbol("cl"), "Cl");
        assert_eq!(element_symbol("  "), "");
    }

    #[test]
    fn two_letter_element_column_reads_as_its_symbol() {
        // An iron HETATM: columns 77-78 hold "FE", as the PDB format writes it.
        let line =
            "HETATM  100 FE   HEM A 501      10.000  20.000  30.000  1.00  0.00          FE  ";
        let atom = parse_hetatm_record(line)
            .expect("parse HETATM")
            .expect("a HETATM record");
        assert_eq!(atom.element, "Fe");
    }

    #[test]
    fn test_parse_hetatm_record() {
        let atom = parse_hetatm_record(SAMPLE_HETATM_LINE)
            .expect("Failed to parse HETATM")
            .expect("Missing HETATM record");
        assert_eq!(atom.serial, 100);
        assert_eq!(atom.name, "O");
        assert_eq!(atom.res_name, "HOH");
        assert!((atom.x - 10.0).abs() < 0.001);
        assert_eq!(atom.element, "O");
    }

    #[test]
    fn test_parse_hetatm_record_missing_occupancy_temp() {
        let atom = parse_hetatm_record(SAMPLE_HETATM_LINE_NO_OCC)
            .expect("Failed to parse HETATM")
            .expect("Missing HETATM record");
        assert_eq!(atom.serial, 1);
        assert_eq!(atom.name, "O");
        assert!((atom.occupancy - 1.0).abs() < 0.001);
        assert!((atom.temp_factor - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_parse_cryst1_record() {
        let cryst = parse_cryst1_record(SAMPLE_CRYST1_LINE).expect("Failed to parse CRYST1");
        assert!((cryst.a - 50.0).abs() < 0.001);
        assert!((cryst.b - 60.0).abs() < 0.001);
        assert!((cryst.c - 70.0).abs() < 0.001);
        assert!((cryst.alpha - 90.0).abs() < 0.001);
        assert_eq!(cryst.space_group, "P 1");
        assert_eq!(cryst.z, 1);
    }

    #[test]
    fn test_parse_conect_record() {
        let conect = parse_conect_record(SAMPLE_CONECT_LINE).expect("Failed to parse CONECT");
        assert_eq!(conect.serial, 1);
        assert_eq!(conect.bonded, vec![2, 3, 4]);
    }

    #[test]
    fn test_write_pdb_frame_conect_from_bonds() {
        use molrs::core::Block;
        use molrs::core::Frame;
        use ndarray::{Array1, IxDyn};

        let mut frame = Frame::new();
        let mut atoms = Block::new();
        let n = 3;

        let x = Array1::from_vec(vec![0.0 as F, 1.0 as F, 2.0 as F])
            .into_shape_with_order(IxDyn(&[n]))
            .unwrap()
            .into_dyn();
        let y = Array1::from_vec(vec![0.0 as F, 0.0 as F, 0.0 as F])
            .into_shape_with_order(IxDyn(&[n]))
            .unwrap()
            .into_dyn();
        let z = Array1::from_vec(vec![0.0 as F, 0.0 as F, 0.0 as F])
            .into_shape_with_order(IxDyn(&[n]))
            .unwrap()
            .into_dyn();
        let elements = Array1::from_vec(vec!["C".to_string(), "O".to_string(), "N".to_string()])
            .into_shape_with_order(IxDyn(&[n]))
            .unwrap()
            .into_dyn();
        let ids = Array1::from_vec(vec![10 as Idx, 20 as Idx, 30 as Idx])
            .into_shape_with_order(IxDyn(&[n]))
            .unwrap()
            .into_dyn();

        atoms.insert("x", x, None).unwrap();
        atoms.insert("y", y, None).unwrap();
        atoms.insert("z", z, None).unwrap();
        atoms.insert("element", elements, None).unwrap();
        atoms.insert("id", ids, None).unwrap();
        frame.insert("atoms", atoms);

        let mut bonds = Block::new();
        let atom_i = Array1::from_vec(vec![0 as Idx, 1 as Idx])
            .into_shape_with_order(IxDyn(&[2]))
            .unwrap()
            .into_dyn();
        let atom_j = Array1::from_vec(vec![2 as Idx, 2 as Idx])
            .into_shape_with_order(IxDyn(&[2]))
            .unwrap()
            .into_dyn();
        bonds.insert("atomi", atom_i, None).unwrap();
        bonds.insert("atomj", atom_j, None).unwrap();
        frame.insert("bonds", bonds);

        let mut out = Vec::new();
        write_frame_to(&mut out, &frame).expect("write pdb");
        let output = String::from_utf8(out).expect("utf8");
        assert!(output.contains("CONECT   10   30"));
        assert!(output.contains("CONECT   20   30"));
        assert!(output.contains("CONECT   30   10   20"));
    }

    #[test]
    fn test_parse_pdb_blank_element_is_null_not_guessed() {
        // Row 1 states its element; row 2 leaves columns 77-78 blank. The
        // name "FE" must not be read as a species: the row is null.
        let pdb_content = concat!(
            "ATOM      1  C   ALA A   1       1.000   2.000   3.000  1.00 20.00           C\n",
            "ATOM      2 FE   HEM A   1       2.000   3.000   4.000  1.00 20.00\n",
            "END\n",
        );

        let mut reader = PdbReader::new(std::io::Cursor::new(pdb_content));
        let mut frame = reader.read().expect("IO error").expect("No frame");

        let atom_block = frame.get("atoms").expect("No atoms block");
        let elements = atom_block
            .get("element")
            .and_then(|c| c.get::<String>())
            .expect("No element column");
        assert_eq!(elements[[0]], "C");
        assert_eq!(elements[[1]], "");
        assert_eq!(atom_block.validity("element"), Some(&[true, false][..]));
        assert!(atom_block.get("atomic_number").is_none());

        // The explicit conversion is where the absence surfaces, by row.
        let err = molrs::core::Element::fill_atomic_numbers(&mut frame).unwrap_err();
        assert!(err.to_string().contains("row 1"), "{err}");
    }

    #[test]
    fn test_parse_pdb_without_any_element_has_no_element_column() {
        let pdb_content = concat!(
            "ATOM      1  C   ALA A   1       1.000   2.000   3.000  1.00 20.00\n",
            "END\n",
        );
        let frame = read_pdb(pdb_content.as_bytes()).unwrap();
        assert!(frame.get("atoms").unwrap().get("element").is_none());
    }

    #[test]
    fn writer_requires_an_element_column() {
        let mut frame = read_pdb(SAMPLE_ATOM_LINE.as_bytes()).unwrap();
        let mut atoms = frame.get("atoms").unwrap().clone();
        atoms.remove("element");
        frame.insert("atoms", atoms);
        let err = write_pdb(&mut Vec::new(), &frame).unwrap_err();
        assert!(err.to_string().contains("fill_symbols"), "{err}");
    }

    #[test]
    fn test_atom_line_short() {
        // Line too short should return None
        let short_line = "ATOM      1  N";
        assert!(
            parse_atom_record(short_line)
                .expect("Parse error")
                .is_none()
        );
    }

    // -----------------------------------------------------------------
    // Streaming index tests
    // -----------------------------------------------------------------

    fn pdb_build_chunked(bytes: &[u8], chunk_size: usize) -> Vec<FrameOffset> {
        let mut builder = Box::new(PdbIndexBuilder::new());
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

    const SINGLE_PDB: &str = concat!(
        "CRYST1   10.000   10.000   10.000  90.00  90.00  90.00 P 1           1\n",
        "ATOM      1  N   ALA A   1       1.000   2.000   3.000  1.00 20.00           N  \n",
        "ATOM      2  CA  ALA A   1       2.000   3.000   4.000  1.00 20.00           C  \n",
        "END\n",
    );

    const MULTI_PDB: &str = concat!(
        "MODEL        1\n",
        "ATOM      1  N   ALA A   1       1.000   2.000   3.000  1.00 20.00           N  \n",
        "ATOM      2  CA  ALA A   1       2.000   3.000   4.000  1.00 20.00           C  \n",
        "ENDMDL\n",
        "MODEL        2\n",
        "ATOM      1  N   ALA A   1       1.500   2.500   3.500  1.00 20.00           N  \n",
        "ATOM      2  CA  ALA A   1       2.500   3.500   4.500  1.00 20.00           C  \n",
        "ENDMDL\n",
    );

    #[test]
    fn pdb_streaming_single_frame_no_model() {
        let bytes = SINGLE_PDB.as_bytes();
        for cs in [1usize, 7, 16, 64, bytes.len()] {
            let entries = pdb_build_chunked(bytes, cs);
            assert_eq!(entries.len(), 1, "chunk size {}", cs);
            assert_eq!(entries[0].byte_offset, 0);
            assert_eq!(entries[0].byte_len as usize, bytes.len());
            read_pdb(&bytes[..entries[0].byte_len as usize]).expect("parse single-frame PDB");
        }
    }

    #[test]
    fn pdb_streaming_multi_model() {
        let bytes = MULTI_PDB.as_bytes();
        let one_shot = pdb_build_chunked(bytes, bytes.len());
        assert_eq!(one_shot.len(), 2);
        for cs in [1usize, 7, 16, 64, 256] {
            let chunked = pdb_build_chunked(bytes, cs);
            assert_eq!(
                one_shot, chunked,
                "chunk size {} produced different index",
                cs
            );
        }
        for entry in &one_shot {
            let lo = entry.byte_offset as usize;
            let hi = lo + entry.byte_len as usize;
            let frame = read_pdb(&bytes[lo..hi]).expect("parse model");
            assert_eq!(frame.get("atoms").unwrap().n_rows().unwrap(), 2);
        }
    }

    // -----------------------------------------------------------------
    // Trajectory (multi-MODEL) reader/writer
    // -----------------------------------------------------------------

    fn read_all_models(text: &str) -> Vec<Frame> {
        let mut reader = PdbReader::new(std::io::Cursor::new(text.to_string()));
        let mut frames = Vec::new();
        while let Some(f) = reader.read().expect("read frame") {
            frames.push(f);
        }
        frames
    }

    #[test]
    fn a_conect_pair_listed_from_both_ends_is_one_bond() {
        let conects = [
            ConectRecord {
                serial: 1,
                bonded: vec![2, 2],
            },
            ConectRecord {
                serial: 2,
                bonded: vec![1, 3],
            },
        ];
        let serial_map: HashMap<i32, Idx> = [(1, 0), (2, 1), (3, 2)].into_iter().collect();
        let bonds = build_bonds_block(&conects, &serial_map).unwrap().unwrap();
        let i: Vec<Idx> = bonds
            .get("atomi")
            .and_then(|c| c.get::<u64>())
            .unwrap()
            .iter()
            .copied()
            .collect();
        let j: Vec<Idx> = bonds
            .get("atomj")
            .and_then(|c| c.get::<u64>())
            .unwrap()
            .iter()
            .copied()
            .collect();
        assert_eq!((i, j), (vec![0, 1], vec![1, 2]));
    }

    #[test]
    fn multi_model_yields_one_frame_per_model() {
        // The same iteration PdbReader::read performs: one frame per MODEL,
        // reusing read_single_frame (stops at ENDMDL, ignores MODEL header).
        let frames = read_all_models(MULTI_PDB);
        assert_eq!(frames.len(), 2);
        for f in &frames {
            assert_eq!(f.get("atoms").unwrap().n_rows().unwrap(), 2);
        }
    }

    /// Over a seekable stream the reader is a trajectory: models by index,
    /// in any order, and `None` past the last.
    #[test]
    fn pdb_reader_reads_models_by_index() {
        use crate::io::reader::TrajectoryReader;
        let mut reader = PdbReader::new(std::io::Cursor::new(MULTI_PDB.as_bytes()));
        assert_eq!(reader.len().unwrap(), 2);
        let second = reader.read_frame(1).unwrap().expect("model 2");
        let first = reader.read_frame(0).unwrap().expect("model 1");
        let models = read_all_models(MULTI_PDB);
        let x = |f: &Frame| {
            f.get("atoms")
                .unwrap()
                .get("x")
                .unwrap()
                .get::<f64>()
                .unwrap()
                .iter()
                .copied()
                .collect::<Vec<_>>()
        };
        assert_eq!(x(&first), x(&models[0]));
        assert_eq!(x(&second), x(&models[1]));
        assert!(reader.read_frame(2).unwrap().is_none());
    }

    /// `PdbWriter::create` writes one `MODEL` per frame into a buffer, and
    /// `PdbReader::open` reads them back by index.
    #[test]
    fn pdb_writer_create_and_reader_open_round_trip_models() {
        use crate::io::reader::TrajectoryReader;
        let frames = read_all_models(MULTI_PDB);
        let mut out = Vec::new();
        let mut writer = PdbWriter::create(&mut out).unwrap();
        for f in &frames {
            writer.write(f).unwrap();
        }
        drop(writer);
        let text = String::from_utf8(out.clone()).unwrap();
        assert_eq!(text.matches("CRYST1").count(), 1, "{text}");
        assert_eq!(text.matches("ENDMDL").count(), 2, "{text}");
        let mut reader = PdbReader::open(&out).unwrap();
        assert_eq!(reader.len().unwrap(), 2);
        assert_eq!(
            reader.read_frame(1).unwrap().unwrap()["atoms"].n_rows(),
            Some(2)
        );
    }

    #[test]
    fn write_pdb_traj_roundtrips() {
        let frames = read_all_models(MULTI_PDB);
        let mut out = Vec::new();
        {
            use crate::io::writer::{FrameWriter, Writer};
            let mut writer = PdbWriter::new(&mut out);
            for frame in &frames {
                writer.write(frame).expect("write model");
            }
        }
        let text = String::from_utf8(out).expect("utf8");
        assert_eq!(text.matches("MODEL ").count(), 2, "{text}");
        assert_eq!(text.matches("ENDMDL").count(), 2, "{text}");

        let reparsed = read_all_models(&text);
        assert_eq!(reparsed.len(), 2);
        assert_eq!(reparsed[0].get("atoms").unwrap().n_rows().unwrap(), 2);
    }

    #[test]
    fn write_pdb_frame_uses_atom_columns() {
        use molrs::core::Block;
        use molrs::core::Frame;
        use ndarray::{Array1, IxDyn};

        let n = 1;
        let mut atoms = Block::new();
        atoms
            .insert(
                "x",
                Array1::from_vec(vec![1.0 as F])
                    .into_shape_with_order(IxDyn(&[n]))
                    .unwrap()
                    .into_dyn(),
                None,
            )
            .unwrap();
        atoms
            .insert(
                "y",
                Array1::from_vec(vec![2.0 as F])
                    .into_shape_with_order(IxDyn(&[n]))
                    .unwrap()
                    .into_dyn(),
                None,
            )
            .unwrap();
        atoms
            .insert(
                "z",
                Array1::from_vec(vec![3.0 as F])
                    .into_shape_with_order(IxDyn(&[n]))
                    .unwrap()
                    .into_dyn(),
                None,
            )
            .unwrap();
        atoms
            .insert(
                "name",
                Array1::from_vec(vec!["CA".to_string()])
                    .into_shape_with_order(IxDyn(&[n]))
                    .unwrap()
                    .into_dyn(),
                None,
            )
            .unwrap();
        atoms
            .insert(
                "res_name",
                Array1::from_vec(vec!["ALA".to_string()])
                    .into_shape_with_order(IxDyn(&[n]))
                    .unwrap()
                    .into_dyn(),
                None,
            )
            .unwrap();
        atoms
            .insert(
                "res_id",
                Array1::from_vec(vec![5 as Idx])
                    .into_shape_with_order(IxDyn(&[n]))
                    .unwrap()
                    .into_dyn(),
                None,
            )
            .unwrap();
        atoms
            .insert(
                "chain",
                Array1::from_vec(vec!["B".to_string()])
                    .into_shape_with_order(IxDyn(&[n]))
                    .unwrap()
                    .into_dyn(),
                None,
            )
            .unwrap();
        atoms
            .insert(
                "element",
                Array1::from_vec(vec!["C".to_string()])
                    .into_shape_with_order(IxDyn(&[n]))
                    .unwrap()
                    .into_dyn(),
                None,
            )
            .unwrap();

        let mut frame = Frame::new();
        frame.insert("atoms", atoms);

        let mut out = Vec::new();
        write_frame_to(&mut out, &frame).expect("write frame");
        let text = String::from_utf8(out).expect("utf8");
        let atom_line = text
            .lines()
            .find(|l| l.starts_with("ATOM"))
            .expect("atom line");

        // PDB v3.3 column checks: name (13-16), resName (18-20), chain (22),
        // resSeq (23-26).
        assert_eq!(atom_line[12..16].trim(), "CA", "name: {atom_line}");
        assert_eq!(atom_line[17..20].trim(), "ALA", "resName: {atom_line}");
        assert_eq!(&atom_line[21..22], "B", "chain: {atom_line}");
        assert_eq!(atom_line[22..26].trim(), "5", "resSeq: {atom_line}");
        // columns 77-78: element (1-based); 0-based [76..78]
        assert!(
            atom_line.len() >= 78,
            "line too short: {atom_line:?} len={}",
            atom_line.len()
        );
        assert_eq!(atom_line[76..78].trim(), "C", "element: {atom_line:?}");
    }

    #[test]
    fn chain_icode_altloc_occupancy_and_b_factor_round_trip() {
        let pdb = concat!(
            "ATOM      1  CA AALA B  12A     11.104   6.134  -6.504  0.50 17.25           C\n",
            "ATOM      2  CB  ALA B  13      12.104   6.134  -6.504  1.00  0.00           C\n",
            "END\n",
        );
        let frame = PdbReader::new(std::io::Cursor::new(pdb.as_bytes()))
            .read_single_frame()
            .unwrap()
            .unwrap();
        let atoms = &frame["atoms"];
        let strings = |key: &str| -> Vec<String> {
            atoms
                .get(key)
                .and_then(|c| c.get::<String>())
                .unwrap()
                .iter()
                .cloned()
                .collect()
        };
        let floats = |key: &str| -> Vec<F> {
            atoms
                .get(key)
                .and_then(|c| c.get::<f64>())
                .unwrap()
                .iter()
                .copied()
                .collect()
        };
        assert_eq!(strings("chain"), ["B", "B"]);
        assert_eq!(strings("altloc"), ["A", ""]);
        assert_eq!(strings("icode"), ["A", ""]);
        assert_eq!(floats("occupancy"), [0.5, 1.0]);
        assert_eq!(floats("b_factor"), [17.25, 0.0]);
        assert!(atoms.get("chain_id").is_none());

        let mut out = Vec::new();
        write_frame_to(&mut out, &frame).unwrap();
        let back = PdbReader::new(std::io::Cursor::new(out.as_slice()))
            .read_single_frame()
            .unwrap()
            .unwrap();
        for key in ["chain", "altloc", "icode"] {
            assert_eq!(
                back["atoms"]
                    .get(key)
                    .and_then(|c| c.get::<String>())
                    .unwrap(),
                atoms.get(key).and_then(|c| c.get::<String>()).unwrap(),
                "{key}"
            );
        }
        for key in ["occupancy", "b_factor"] {
            assert_eq!(
                back["atoms"].get(key).and_then(|c| c.get::<f64>()).unwrap(),
                atoms.get(key).and_then(|c| c.get::<f64>()).unwrap(),
                "{key}"
            );
        }
    }
}
