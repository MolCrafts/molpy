//! The PDB codec: records, the reader and writer classes, the path doors.
//!
//! Implements PDB 3.3 specification for coordinate section records:
//! <https://www.wwpdb.org/documentation/file-format-content/format33/sect9.html>

use crate::io::invalid_data;
use crate::io::pdb::labels::AtomLabels;
use crate::io::reader::{FrameReader, ReadSeek, Reader, TrajectoryReader};
use crate::io::writer::FrameWriter;
use molrs::core::Block;
use molrs::core::Frame;
use molrs::core::FrameAccess;
use molrs::core::SimBox;
use molrs::core::keys;
use molrs::op::{F, I, Idx};
use ndarray::{Array1, IxDyn, array};
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader, Seek, Write};
use std::path::Path;

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

fn infer_element_from_atom_name(name_raw: &str) -> Option<String> {
    let mut chars = name_raw.chars();
    let first = chars.next()?;
    let second = chars.next();

    if first.is_whitespace() {
        if let Some(c) = second
            && c.is_ascii_alphabetic()
        {
            return Some(c.to_ascii_uppercase().to_string());
        }
        return None;
    }

    if first.is_ascii_alphabetic() {
        let mut symbol = String::new();
        symbol.push(first.to_ascii_uppercase());
        if let Some(c) = second
            && c.is_ascii_alphabetic()
        {
            symbol.push(c.to_ascii_lowercase());
        }
        return Some(symbol);
    }

    None
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
    let name_raw = substr(line, 12, 16);
    let name = name_raw.trim().to_string();
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
    // ("C", "Fe") so no caller has to trim or re-case it.
    let mut element = if line.len() >= 78 {
        element_symbol(substr(line, 76, 78))
    } else {
        String::new()
    };
    if element.is_empty() {
        if let Some(inferred) = infer_element_from_atom_name(name_raw) {
            element = inferred;
        } else {
            element = "X".to_string();
        }
    }

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

    // `substr` clamps to the line, so a record whose trailing Z field (and
    // blanks) were cut off still keeps its symbol.
    let space_group = substr(line, 55, 66).trim().to_string();

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
        elements.push(if atom.element.trim().is_empty() {
            "X".to_string()
        } else {
            atom.element.clone()
        });
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

    let mut block = Block::new();
    block
        .insert("x", to_array_float(x_vec, n)?)
        .map_err(invalid_data)?;
    block
        .insert("y", to_array_float(y_vec, n)?)
        .map_err(invalid_data)?;
    block
        .insert("z", to_array_float(z_vec, n)?)
        .map_err(invalid_data)?;
    block
        .insert("id", to_array_uint(ids_vec, n)?)
        .map_err(invalid_data)?;
    block
        .insert("element", to_array_string(elements, n)?)
        .map_err(invalid_data)?;
    block
        .insert("name", to_array_string(names, n)?)
        .map_err(invalid_data)?;
    block
        .insert("res_name", to_array_string(res_names, n)?)
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
        .insert("res_id", to_array_uint(res_ids, n)?)
        .map_err(invalid_data)?;
    block
        .insert("chain", to_array_string(chains, n)?)
        .map_err(invalid_data)?;
    block
        .insert("icode", to_array_string(icodes, n)?)
        .map_err(invalid_data)?;
    block
        .insert("altloc", to_array_string(altlocs, n)?)
        .map_err(invalid_data)?;
    block
        .insert("occupancy", to_array_float(occupancies, n)?)
        .map_err(invalid_data)?;
    block
        .insert("b_factor", to_array_float(b_factors, n)?)
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
    for elem in elements {
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
        .insert("atomi", to_array_uint(i_indices, bn)?)
        .map_err(invalid_data)?;
    block
        .insert("atomj", to_array_uint(j_indices, bn)?)
        .map_err(invalid_data)?;

    Ok(Some(block))
}

/// The frame's box and space-group symbol from the `CRYST1` record: the
/// symbol goes to [`keys::SPACE_GROUP`] when the record states one.
///
/// The box is the cell of `a`, `b`, `c`, `α`, `β`, `γ` in PDB's standard
/// orthogonalization (`a` along x, `b` in the xy plane), whose lattice vectors
/// are the columns of H ([`SimBox::matrix_from_lengths_angles`]), at the
/// origin and periodic on every axis. A record with a non-positive length
/// states no box, and neither does PDB's placeholder for a structure
/// determined without a cell, `1 1 1 90 90 90 P 1` ([`is_cell_placeholder`]):
/// the frame then has neither a box nor a space group.
///
/// # Errors
///
/// Angles no cell has (each in `(0°, 180°)` and together closing a cell),
/// naming the record's six numbers.
fn add_simbox_from_cryst1(frame: &mut Frame, cryst1: Option<&Cryst1Record>) -> std::io::Result<()> {
    let Some(cryst) = cryst1.filter(|c| !is_cell_placeholder(c)) else {
        return Ok(());
    };
    if !cryst.space_group.is_empty() {
        frame
            .meta
            .insert(keys::SPACE_GROUP, cryst.space_group.clone());
    }
    if cryst.a <= 0.0 || cryst.b <= 0.0 || cryst.c <= 0.0 {
        return Ok(());
    }
    let lengths = [cryst.a as F, cryst.b as F, cryst.c as F];
    let angles = [cryst.alpha as F, cryst.beta as F, cryst.gamma as F];
    let refuse = |e| {
        invalid_data(format!(
            "CRYST1 a b c = {lengths:?}, alpha beta gamma = {angles:?}: no cell has these \
             parameters ({e:?})"
        ))
    };
    let h = SimBox::matrix_from_lengths_angles(lengths, angles).map_err(refuse)?;
    let simbox = SimBox::new(h, array![0.0 as F, 0.0, 0.0], [true, true, true]).map_err(refuse)?;
    frame.simbox = Some(simbox);
    Ok(())
}

/// PDB's `CRYST1` for a structure not determined by crystallography: "the
/// unitary values" `a = b = c = 1.0`, `α = β = γ = 90°`, space group `P 1`
/// (wwPDB format 3.3, `CRYST1`). It states no cell.
fn is_cell_placeholder(cryst: &Cryst1Record) -> bool {
    [cryst.a, cryst.b, cryst.c] == [1.0; 3]
        && [cryst.alpha, cryst.beta, cryst.gamma] == [90.0; 3]
        && matches!(cryst.space_group.as_str(), "" | "P 1")
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

    add_simbox_from_cryst1(&mut frame, cryst1)?;

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
    /// Open a PDB file for random access by model.
    pub fn open<P: AsRef<Path>>(path: P) -> std::io::Result<Self> {
        Ok(Self::new(crate::io::reader::open_seekable(path)?))
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
        crate::io::reader::check_read_frame(Some(read_pdb_bytes(&bytes)?))
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

pub struct PdbWriter<W: Write> {
    writer: W,
}

impl<W: Write> crate::io::writer::Writer for PdbWriter<W> {
    type W = W;
    fn new(writer: W) -> Self {
        Self { writer }
    }
}

impl<W: Write> FrameWriter for PdbWriter<W> {
    fn write(&mut self, frame: &Frame) -> std::io::Result<()> {
        // Refuse to emit a frame that violates the vocabulary: a bad file
        // looks fine and is found wrong later, by whatever reads it.
        crate::io::writer::check_write_frame(frame)?;
        write_frame_to(&mut self.writer, frame)
    }
}

impl<W: Write> PdbWriter<W> {
    /// Create a new PDB writer from an output sink.
    pub fn new(writer: W) -> Self {
        Self { writer }
    }
}

/// Write the `CRYST1` record from the frame's simulation box, if any: the
/// lengths of its lattice vectors and the angles between them
/// ([`SimBox::lengths`], [`SimBox::angles`]).
///
/// CRYST1 states a cell's shape, not its orientation: a reader rebuilds it in
/// PDB's standard orthogonalization (`a` along x, `b` in the xy plane), the
/// orientation every molrs reader and LAMMPS's restricted triclinic box use.
/// A box rotated away from it is written with its own lengths and angles, and
/// read back in that standard orientation.
///
/// A frame without a box gets no `CRYST1`, as OpenMM's `PDBFile` writes
/// none: any cell line would read back as a box (OpenMM takes even PDB's
/// `1 1 1` placeholder as a 1 Å cell).
fn write_cryst1<W: Write>(writer: &mut W, frame: &impl FrameAccess) -> std::io::Result<()> {
    let Some(simbox) = frame.simbox_ref() else {
        return Ok(());
    };
    let (lengths, angles) = (simbox.lengths(), simbox.angles());
    writeln!(
        writer,
        "CRYST1{:9.3}{:9.3}{:9.3}{:7.2}{:7.2}{:7.2} {:<11}{:>4}",
        lengths[0], lengths[1], lengths[2], angles[0], angles[1], angles[2], "P 1", 1
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
        .and_then(|c| c.as_float())
        .ok_or_else(|| invalid_data("Missing 'x' column"))?;
    let y = frame
        .column("atoms", "y")
        .and_then(|c| c.as_float())
        .ok_or_else(|| invalid_data("Missing 'y' column"))?;
    let z = frame
        .column("atoms", "z")
        .and_then(|c| c.as_float())
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
            .and_then(|c| c.as_string())
            .as_ref()
            .and_then(|arr| arr.as_slice().map(|s| s.to_vec()))
            .unwrap_or_default()
    };
    // A stated serial or coordinate has no stand-in (`null_cells`).
    crate::io::null_cells::refuse_null_cells(frame, "atoms", &["id", "x", "y", "z"])?;
    // Names and residues: the one definition the OpenMM system writer's
    // residue templates also use.
    let labels = AtomLabels::pdb(frame, n)?;
    let altlocs = owned_str("altloc");
    let owned_f64 = |col: &str| -> Vec<F> {
        frame
            .column("atoms", col)
            .and_then(|c| c.as_float())
            .as_ref()
            .and_then(|arr| arr.as_slice().map(|s| s.to_vec()))
            .unwrap_or_default()
    };
    // A null occupancy or B-factor is unset: the default an absent column
    // takes (1.00, 0.00), not the cell's filler.
    let unset = |col: &str| crate::io::null_cells::null_rows(frame, "atoms", col);
    let (null_occupancy, null_b_factor) = (unset("occupancy"), unset("b_factor"));
    let occupancies = owned_f64("occupancy");
    let b_factors = owned_f64("b_factor");
    let elements = owned_str("element");

    let ids: Vec<Idx> = frame
        .column("atoms", "id")
        .and_then(|c| c.as_uint())
        .as_ref()
        .and_then(|arr| arr.as_slice().map(|s| s.to_vec()))
        .unwrap_or_default();
    let has_ids = !ids.is_empty();

    let mut serials = Vec::with_capacity(n);
    for i in 0..n {
        let serial = if has_ids { ids[i] as usize } else { i + 1 };
        serials.push(serial);

        let elem_raw = elements
            .get(i)
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .unwrap_or("X");

        // Atom name (cols 13-16): single-char names are offset one column.
        let name_raw = labels.names[i].as_str();
        let name_field = if name_raw.chars().count() == 1 {
            format!(" {:<3}", name_raw)
        } else {
            format!("{:<4}", name_raw)
        };
        let res_name = labels.res_names[i].as_str();
        let chain = labels.chains[i];
        let altloc = altlocs
            .get(i)
            .and_then(|s| s.trim().chars().next())
            .unwrap_or(' ');
        let icode = labels.icodes[i];
        let stated = |values: &[F], nulls: &[usize]| {
            values
                .get(i)
                .copied()
                .filter(|_| nulls.binary_search(&i).is_err())
        };
        let occupancy = stated(&occupancies, &null_occupancy).unwrap_or(1.0);
        let b_factor = stated(&b_factors, &null_b_factor).unwrap_or(0.0);
        let res_seq = labels.res_seqs[i];

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
                    .and_then(|c| c.as_uint())
                    .ok_or_else(|| invalid_data("Bonds block missing 'atomi' column"))?;
                let j_arr = bonds
                    .column("atomj")
                    .and_then(|c| c.as_uint())
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

/// Write a trajectory as a multi-`MODEL` PDB.
///
/// A shared `CRYST1` is written once from the first frame, then each frame
/// becomes one `MODEL`/`ENDMDL` block, reusing the same record writer as
/// [`write_frame_to`]. The inverse of [`read_pdb_trajectory`].
fn write_trajectory_to<W: Write, FA: FrameAccess>(
    writer: &mut W,
    frames: &[FA],
) -> std::io::Result<()> {
    if let Some(first) = frames.first() {
        write_cryst1(writer, first)?;
    }
    for (i, frame) in frames.iter().enumerate() {
        writeln!(writer, "MODEL     {:>4}", i + 1)?;
        write_atom_conect_records(writer, frame)?;
        writeln!(writer, "ENDMDL")?;
    }
    writeln!(writer, "END")?;
    Ok(())
}

// ============================================================================
// Convenience Functions
// ============================================================================

/// Read a single frame from a PDB file
///
/// # Examples
///
/// ```no_run
/// use molrs::io::read_pdb;
///
/// # fn main() -> std::io::Result<()> {
/// let frame = read_pdb("protein.pdb")?;
/// # Ok(())
/// # }
/// ```
pub fn read_pdb<P: AsRef<Path>>(path: P) -> std::io::Result<Frame> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut pdb_reader = PdbReader::new(reader);
    pdb_reader.read()?.ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "No frame found in PDB file",
        )
    })
}

/// Read every `MODEL` of a PDB file as a trajectory — one [`Frame`] per model.
///
/// Reuses [`PdbReader`]'s private single-frame parser: each
/// `MODEL`/`ENDMDL` block parses to one frame (`read_single_frame` stops at
/// `ENDMDL` and ignores the `MODEL` header). A single-model or MODEL-less PDB
/// yields a one-frame trajectory. The inverse of [`write_pdb_trajectory`].
///
/// # Examples
///
/// ```no_run
/// use molrs::io::read_pdb_trajectory;
///
/// # fn main() -> std::io::Result<()> {
/// let frames = read_pdb_trajectory("ensemble.pdb")?;
/// # Ok(())
/// # }
/// ```
pub fn read_pdb_trajectory<P: AsRef<Path>>(path: P) -> std::io::Result<Vec<Frame>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut pdb_reader = PdbReader::new(reader);
    let mut frames = Vec::new();
    while let Some(frame) = pdb_reader.read()? {
        frames.push(frame);
    }
    Ok(frames)
}

/// Write one frame as a PDB file (`REMARK`, `CRYST1`, `ATOM`/`CONECT`, `END`).
///
/// `CRYST1` states the frame's box, and is not written for a frame without
/// one, which then reads back without a box.
///
/// The atoms' `name`, `res_name`, `res_id`, `chain` and `icode` columns are
/// written as given. Without `res_id`, each bonded molecule (a connected
/// component of `bonds`) is one residue; without `res_name`, residues are
/// `MOL`; an atom without a `name` is its element numbered within its
/// residue (`C1`, `C2`, `H1`, …). Residue numbers wrap at 10 000. These are
/// the residues and names
/// [`write_openmm_system`](crate::io::write_openmm_system) gives its
/// templates.
pub fn write_pdb<P: AsRef<Path>>(path: P, frame: &impl FrameAccess) -> std::io::Result<()> {
    let mut out = std::io::BufWriter::new(File::create(path)?);
    write_frame_to(&mut out, frame)?;
    out.flush()
}

/// Write frames as a multi-`MODEL` PDB file — the inverse of
/// [`read_pdb_trajectory`].
pub fn write_pdb_trajectory<P: AsRef<Path>, FA: FrameAccess>(
    path: P,
    frames: &[FA],
) -> std::io::Result<()> {
    let mut out = std::io::BufWriter::new(File::create(path)?);
    write_trajectory_to(&mut out, frames)?;
    out.flush()
}

/// Read the first frame of PDB `text` — [`read_pdb`] on text in memory.
pub fn read_pdb_str(text: &str) -> std::io::Result<Frame> {
    PdbReader::new(std::io::Cursor::new(text.as_bytes()))
        .read()?
        .ok_or_else(|| invalid_data("No frame found in PDB text"))
}

/// Write one frame as PDB text — [`write_pdb`] into memory.
pub fn write_pdb_str(frame: &impl FrameAccess) -> std::io::Result<String> {
    let mut buf = Vec::new();
    write_frame_to(&mut buf, frame)?;
    String::from_utf8(buf).map_err(invalid_data)
}

// ============================================================================
// Streaming
// ============================================================================

use crate::io::frame_index::{FrameIndexBuilder, FrameOffset, LineAccumulator};
use std::io::Cursor;

/// Parse exactly one PDB frame from a tightly-bounded byte slice. The slice
/// must be a `[byte_offset, byte_offset + byte_len)` window produced by
/// [`PdbIndexBuilder`] — for multi-MODEL files this is a single
/// `MODEL ... ENDMDL` block; for single-model files this is the entire file.
pub fn read_pdb_bytes(bytes: &[u8]) -> std::io::Result<Frame> {
    let cursor = Cursor::new(bytes);
    let mut reader = PdbReader::new(cursor);
    reader.read()?.ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "PDB frame slice contained no atoms",
        )
    })
}

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

    const TRICLINIC_PDB: &str = concat!(
        "CRYST1   10.000   11.000   12.000  70.00  80.00  65.00 P 1           1\n",
        "ATOM      1  O   HOH A   1       1.000   2.000   3.000  1.00  0.00           O  \n",
        "END\n",
    );

    /// CRYST1's angles shape the box: its lattice vectors (the columns of H)
    /// are PDB's standard orthogonalization — `a` along x, `b` in the xy
    /// plane — with the stated lengths and angles.
    #[test]
    fn cryst1_angles_build_a_triclinic_box() {
        let frame = read_pdb_str(TRICLINIC_PDB).unwrap();
        let simbox = frame.simbox.as_ref().expect("CRYST1 states a box");
        let lengths = simbox.lengths();
        let angles = simbox.angles();
        for (got, want) in lengths.iter().zip([10.0, 11.0, 12.0]) {
            assert!((got - want).abs() < 1e-9, "lengths {lengths:?}");
        }
        for (got, want) in angles.iter().zip([70.0, 80.0, 65.0]) {
            assert!((got - want).abs() < 1e-9, "angles {angles:?}");
        }
        let a = simbox.lattice(0);
        let b = simbox.lattice(1);
        assert!((a[0] - 10.0).abs() < 1e-12 && a[1] == 0.0 && a[2] == 0.0);
        assert_eq!(b[2], 0.0, "b lies in the xy plane");
        assert_eq!(simbox.style(), "triclinic");
    }

    /// The writer states the box's real angles, so a triclinic box survives
    /// a write and a read.
    #[test]
    fn a_triclinic_box_round_trips_through_cryst1() {
        let frame = read_pdb_str(TRICLINIC_PDB).unwrap();
        let text = write_pdb_str(&frame).unwrap();
        let cryst1 = text
            .lines()
            .find(|l| l.starts_with("CRYST1"))
            .expect("a CRYST1 line");
        assert_eq!(
            cryst1,
            "CRYST1   10.000   11.000   12.000  70.00  80.00  65.00 P 1           1"
        );
        let back = read_pdb_str(&text).unwrap();
        assert!(
            back.simbox
                .as_ref()
                .unwrap()
                .approx_eq(frame.simbox.as_ref().unwrap(), 1e-9)
        );
    }

    /// A frame without a box is written without `CRYST1` (as OpenMM's
    /// `PDBFile` writes one) and reads back without a box, in one frame and
    /// in a trajectory. PDB's own placeholder for a structure without a cell
    /// — the unit cube `1 1 1 90 90 90 P 1 1`, which molrs wrote before —
    /// states no box either.
    #[test]
    fn a_frame_without_a_box_has_no_cryst1_and_reads_back_without_one() {
        let mut frame = read_pdb_str(TRICLINIC_PDB).unwrap();
        frame.simbox = None;
        let text = write_pdb_str(&frame).unwrap();
        assert!(!text.contains("CRYST1"), "{text}");
        assert!(read_pdb_str(&text).unwrap().simbox.is_none());

        let mut out = Vec::new();
        write_trajectory_to(&mut out, &[frame.clone(), frame.clone()]).unwrap();
        let text = String::from_utf8(out).unwrap();
        assert!(!text.contains("CRYST1"), "{text}");

        let unit = TRICLINIC_PDB.replace(
            "   10.000   11.000   12.000  70.00  80.00  65.00",
            "    1.000    1.000    1.000  90.00  90.00  90.00",
        );
        let back = read_pdb_str(&unit).unwrap();
        assert!(back.simbox.is_none());
        assert!(back.meta.get(keys::SPACE_GROUP).is_none());
    }

    /// Angles no cell has (they sum past 360°) are a malformed record, not an
    /// orthogonal box.
    #[test]
    fn cryst1_angles_no_cell_has_are_refused() {
        let bad = TRICLINIC_PDB.replace("70.00  80.00  65.00", "170.00170.00170.00");
        let err = read_pdb_str(&bad).expect_err("no cell has these angles");
        assert!(err.to_string().contains("CRYST1"), "{err}");
    }

    #[test]
    fn cryst1_space_group_lands_in_frame_meta() {
        let text = concat!(
            "CRYST1   52.000   58.600   61.900  90.00  90.00  90.00 P 21 21 21    8\n",
            "ATOM      1  N   ALA A   1       1.000   2.000   3.000  1.00 20.00           N  \n",
            "END\n",
        );
        let frame = read_pdb_str(text).unwrap();
        assert_eq!(
            frame.meta.get(keys::SPACE_GROUP).and_then(|v| v.as_str()),
            Some("P 21 21 21")
        );

        // A record cut short after the symbol keeps it; a blank symbol and
        // a file with no CRYST1 state none.
        let short = text.replace("P 21 21 21    8", "P 21 21 21");
        let frame = read_pdb_str(&short).unwrap();
        assert_eq!(
            frame.meta.get(keys::SPACE_GROUP).and_then(|v| v.as_str()),
            Some("P 21 21 21")
        );
        let blank = text.replace("P 21 21 21    8", "");
        assert!(
            read_pdb_str(&blank)
                .unwrap()
                .meta
                .get(keys::SPACE_GROUP)
                .is_none()
        );
        let bare = text.lines().skip(1).collect::<Vec<_>>().join("\n");
        assert!(
            read_pdb_str(&bare)
                .unwrap()
                .meta
                .get(keys::SPACE_GROUP)
                .is_none()
        );
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

        atoms.insert("x", x).unwrap();
        atoms.insert("y", y).unwrap();
        atoms.insert("z", z).unwrap();
        atoms.insert("element", elements).unwrap();
        atoms.insert("id", ids).unwrap();
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
        bonds.insert("atomi", atom_i).unwrap();
        bonds.insert("atomj", atom_j).unwrap();
        frame.insert("bonds", bonds);

        let mut out = Vec::new();
        write_frame_to(&mut out, &frame).expect("write pdb");
        let output = String::from_utf8(out).expect("utf8");
        assert!(output.contains("CONECT   10   30"));
        assert!(output.contains("CONECT   20   30"));
        assert!(output.contains("CONECT   30   10   20"));
    }

    #[test]
    fn test_parse_pdb_missing_element_infers_from_name() {
        let pdb_content = r#"ATOM      1  C   ALA A   1       1.000   2.000   3.000  1.00 20.00
ATOM      2 FE   HEM A   1       2.000   3.000   4.000  1.00 20.00
END
"#;

        let mut reader = PdbReader::new(std::io::Cursor::new(pdb_content));
        let frame = reader.read().expect("IO error").expect("No frame");

        let atom_block = frame.get("atoms").expect("No atoms block");
        let elements = atom_block
            .get("element")
            .and_then(|c| c.as_string())
            .expect("No element column");
        assert_eq!(elements[[0]], "C");
        assert_eq!(elements[[1]], "Fe");
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
            read_pdb_bytes(&bytes[..entries[0].byte_len as usize]).expect("parse single-frame PDB");
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
            let frame = read_pdb_bytes(&bytes[lo..hi]).expect("parse model");
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
            .and_then(|c| c.as_uint())
            .unwrap()
            .iter()
            .copied()
            .collect();
        let j: Vec<Idx> = bonds
            .get("atomj")
            .and_then(|c| c.as_uint())
            .unwrap()
            .iter()
            .copied()
            .collect();
        assert_eq!((i, j), (vec![0, 1], vec![1, 2]));
    }

    #[test]
    fn multi_model_yields_one_frame_per_model() {
        // The same iteration read_pdb_trajectory performs: one frame per MODEL,
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
                .as_float()
                .unwrap()
                .iter()
                .copied()
                .collect::<Vec<_>>()
        };
        assert_eq!(x(&first), x(&models[0]));
        assert_eq!(x(&second), x(&models[1]));
        assert!(reader.read_frame(2).unwrap().is_none());
    }

    #[test]
    fn write_pdb_traj_roundtrips() {
        let frames = read_all_models(MULTI_PDB);
        let mut out = Vec::new();
        write_trajectory_to(&mut out, &frames).expect("write traj");
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
            )
            .unwrap();
        atoms
            .insert(
                "y",
                Array1::from_vec(vec![2.0 as F])
                    .into_shape_with_order(IxDyn(&[n]))
                    .unwrap()
                    .into_dyn(),
            )
            .unwrap();
        atoms
            .insert(
                "z",
                Array1::from_vec(vec![3.0 as F])
                    .into_shape_with_order(IxDyn(&[n]))
                    .unwrap()
                    .into_dyn(),
            )
            .unwrap();
        atoms
            .insert(
                "name",
                Array1::from_vec(vec!["CA".to_string()])
                    .into_shape_with_order(IxDyn(&[n]))
                    .unwrap()
                    .into_dyn(),
            )
            .unwrap();
        atoms
            .insert(
                "res_name",
                Array1::from_vec(vec!["ALA".to_string()])
                    .into_shape_with_order(IxDyn(&[n]))
                    .unwrap()
                    .into_dyn(),
            )
            .unwrap();
        atoms
            .insert(
                "res_id",
                Array1::from_vec(vec![5 as Idx])
                    .into_shape_with_order(IxDyn(&[n]))
                    .unwrap()
                    .into_dyn(),
            )
            .unwrap();
        atoms
            .insert(
                "chain",
                Array1::from_vec(vec!["B".to_string()])
                    .into_shape_with_order(IxDyn(&[n]))
                    .unwrap()
                    .into_dyn(),
            )
            .unwrap();
        atoms
            .insert(
                "element",
                Array1::from_vec(vec!["C".to_string()])
                    .into_shape_with_order(IxDyn(&[n]))
                    .unwrap()
                    .into_dyn(),
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
                .and_then(|c| c.as_string())
                .unwrap()
                .iter()
                .cloned()
                .collect()
        };
        let floats = |key: &str| -> Vec<F> {
            atoms
                .get(key)
                .and_then(|c| c.as_float())
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
                back["atoms"].get(key).and_then(|c| c.as_string()).unwrap(),
                atoms.get(key).and_then(|c| c.as_string()).unwrap(),
                "{key}"
            );
        }
        for key in ["occupancy", "b_factor"] {
            assert_eq!(
                back["atoms"].get(key).and_then(|c| c.as_float()).unwrap(),
                atoms.get(key).and_then(|c| c.as_float()).unwrap(),
                "{key}"
            );
        }
    }
}
