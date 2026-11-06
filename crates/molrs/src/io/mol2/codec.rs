//! The Tripos MOL2 codec.

use crate::io::frame_columns::{insert_column, insert_column_masked};
use crate::io::frame_index::FrameOffset;
use crate::io::{Sink, SinkWriter, Source, invalid_data};
use std::io::{BufRead, Result, Seek, Write};

use molrs::core::Block;
use molrs::core::Frame;
use molrs::core::SimBox;
use molrs::core::schema::consts as keys;
use molrs::op::{F, I, Idx};

use crate::io::reader::{FrameReader, ReadSeek, Reader, TrajectoryReader};
use crate::io::writer::{FrameWriter, Writer};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Parsed records
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct Mol2Atom {
    id: I,
    name: String,
    x: F,
    y: F,
    z: F,
    atom_type: String,
    subst_id: Option<I>,
    subst_name: Option<String>,
    charge: Option<F>,
}

fn parse_atom(line: &str, line_no: usize) -> Result<Mol2Atom> {
    let toks: Vec<&str> = line.split_whitespace().collect();
    if toks.len() < 6 {
        return Err(invalid_data(format!(
            "line {}: ATOM row needs ≥6 fields, got {}",
            line_no,
            toks.len()
        )));
    }
    let id: I = toks[0]
        .parse()
        .map_err(|_| invalid_data(format!("line {}: bad atom id '{}'", line_no, toks[0])))?;
    let name = toks[1].to_string();
    let x: F = toks[2]
        .parse()
        .map_err(|_| invalid_data(format!("line {}: bad x '{}'", line_no, toks[2])))?;
    let y: F = toks[3]
        .parse()
        .map_err(|_| invalid_data(format!("line {}: bad y '{}'", line_no, toks[3])))?;
    let z: F = toks[4]
        .parse()
        .map_err(|_| invalid_data(format!("line {}: bad z '{}'", line_no, toks[4])))?;
    let atom_type = toks[5].to_string();
    let subst_id = toks.get(6).and_then(|t| t.parse::<I>().ok());
    let subst_name = toks.get(7).map(|t| t.to_string());
    let charge = toks.get(8).and_then(|t| t.parse::<F>().ok());
    Ok(Mol2Atom {
        id,
        name,
        x,
        y,
        z,
        atom_type,
        subst_id,
        subst_name,
        charge,
    })
}

#[derive(Debug, Clone)]
struct Mol2Bond {
    _id: I,
    atom_i: I,
    atom_j: I,
    bond_type: String,
}

fn parse_bond(line: &str, line_no: usize) -> Result<Mol2Bond> {
    let toks: Vec<&str> = line.split_whitespace().collect();
    if toks.len() < 4 {
        return Err(invalid_data(format!(
            "line {}: BOND row needs ≥4 fields, got {}",
            line_no,
            toks.len()
        )));
    }
    Ok(Mol2Bond {
        _id: toks[0].parse().unwrap_or(0),
        atom_i: toks[1]
            .parse()
            .map_err(|_| invalid_data(format!("line {}: bad atom_i '{}'", line_no, toks[1])))?,
        atom_j: toks[2]
            .parse()
            .map_err(|_| invalid_data(format!("line {}: bad atom_j '{}'", line_no, toks[2])))?,
        bond_type: toks[3].to_string(),
    })
}

// ---------------------------------------------------------------------------
// Reader
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq)]
enum Section {
    None,
    Molecule,
    Atom,
    Bond,
    Crysin,
    Other,
}

fn section_for(line: &str) -> Section {
    let trimmed = line.trim_end();
    let upper = trimmed.trim_start_matches("@<TRIPOS>");
    if !trimmed.starts_with("@<TRIPOS>") {
        return Section::None;
    }
    match upper {
        "MOLECULE" => Section::Molecule,
        "ATOM" => Section::Atom,
        "BOND" => Section::Bond,
        "CRYSIN" => Section::Crysin,
        _ => Section::Other,
    }
}

/// State machine for reading a single MOL2 record.
fn read_one_record<R: BufRead>(
    reader: &mut R,
    pending_first_line: &mut Option<String>,
) -> Result<Option<Frame>> {
    // Find the start of a MOLECULE block.
    let mut buf = String::new();

    loop {
        if let Some(line) = pending_first_line.take() {
            if section_for(&line) == Section::Molecule {
                break;
            }
            // Otherwise discard and keep scanning.
            continue;
        }
        buf.clear();
        let bytes = reader.read_line(&mut buf)?;
        if bytes == 0 {
            return Ok(None);
        }
        if section_for(&buf) == Section::Molecule {
            break;
        }
    }

    // ---- MOLECULE block ----
    // Line 1: name
    buf.clear();
    if reader.read_line(&mut buf)? == 0 {
        return Err(invalid_data("EOF after @<TRIPOS>MOLECULE"));
    }
    let name = buf.trim().to_string();

    // Line 2: counts
    buf.clear();
    if reader.read_line(&mut buf)? == 0 {
        return Err(invalid_data("EOF before MOLECULE counts"));
    }
    let counts: Vec<&str> = buf.split_whitespace().collect();
    if counts.is_empty() {
        return Err(invalid_data("empty MOLECULE counts line"));
    }
    let n_atoms: usize = counts[0]
        .parse()
        .map_err(|_| invalid_data(format!("bad MOLECULE atom count '{}'", counts[0])))?;
    let n_bonds: usize = counts.get(1).and_then(|s| s.parse().ok()).unwrap_or(0);

    // The next several lines (mol_type, charge_type, [status_bits], [comment])
    // are optional and we don't need them. Skip until we hit the next section
    // marker or a blank line followed by a section marker.
    let mut atoms: Vec<Mol2Atom> = Vec::with_capacity(n_atoms);
    let mut bonds: Vec<Mol2Bond> = Vec::with_capacity(n_bonds);
    let mut current = Section::Molecule;
    let mut atoms_remaining = n_atoms;
    let mut bonds_remaining = n_bonds;
    let mut crysin: Option<[F; 6]> = None;

    let mut line_no = 2usize;
    loop {
        buf.clear();
        let bytes = reader.read_line(&mut buf)?;
        if bytes == 0 {
            // EOF — only valid if we collected enough atoms/bonds
            break;
        }
        line_no += 1;

        // Section change?
        let sec = section_for(&buf);
        match sec {
            Section::None => {}
            Section::Molecule => {
                // Next molecule begins. Push back this line so the next
                // read_one_record sees it.
                *pending_first_line = Some(buf.clone());
                break;
            }
            other => {
                current = other;
                continue;
            }
        }

        let trimmed = buf.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }

        match current {
            Section::Atom if atoms_remaining > 0 => {
                atoms.push(parse_atom(&buf, line_no)?);
                atoms_remaining -= 1;
            }
            Section::Bond if bonds_remaining > 0 => {
                bonds.push(parse_bond(&buf, line_no)?);
                bonds_remaining -= 1;
            }
            Section::Crysin if crysin.is_none() => {
                crysin = Some(parse_crysin(trimmed, line_no)?);
            }
            _ => {} // tolerate / skip
        }
    }

    if atoms.is_empty() {
        return Err(invalid_data(format!(
            "MOLECULE '{}' yielded zero atoms (declared {})",
            name, n_atoms
        )));
    }

    let mut frame = build_frame(name, atoms, bonds)?;
    if let Some(cell) = crysin {
        frame.simbox = Some(simbox_from_crysin(cell)?);
    }
    // Every door reads through here, so every door checks here.
    crate::io::reader::check_read_frame(&frame)?;
    Ok(Some(frame))
}

/// The six cell numbers of a `CRYSIN` line: `a b c alpha beta gamma`, then
/// the space group and setting, which molrs does not keep.
fn parse_crysin(line: &str, line_no: usize) -> Result<[F; 6]> {
    let toks: Vec<&str> = line.split_whitespace().collect();
    if toks.len() < 6 {
        return Err(invalid_data(format!(
            "line {line_no}: CRYSIN needs a b c alpha beta gamma, got `{line}`"
        )));
    }
    let mut cell = [0.0 as F; 6];
    for (slot, tok) in cell.iter_mut().zip(&toks[..6]) {
        *slot = tok.parse().map_err(|_| {
            invalid_data(format!(
                "line {line_no}: CRYSIN value `{tok}` is not a number"
            ))
        })?;
    }
    Ok(cell)
}

/// The periodic box a `CRYSIN` record states, in the standard orientation
/// (`a` along x, `b` in the xy plane), as the PDB reader builds `CRYST1`.
fn simbox_from_crysin(cell: [F; 6]) -> Result<SimBox> {
    let lengths = [cell[0], cell[1], cell[2]];
    let angles = [cell[3], cell[4], cell[5]];
    SimBox::new(
        crate::core::Cell::LengthsAngles { lengths, angles },
        [0.0; 3],
        [true; 3],
    )
    .map_err(|e| {
        invalid_data(format!(
            "CRYSIN a b c = {lengths:?}, alpha beta gamma = {angles:?}: no cell has these \
             parameters ({e:?})"
        ))
    })
}

fn build_frame(name: String, atoms: Vec<Mol2Atom>, bonds: Vec<Mol2Bond>) -> Result<Frame> {
    let n = atoms.len();
    let mut block = Block::new();
    let mut id = Vec::with_capacity(n);
    let mut a_name = Vec::with_capacity(n);
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    let mut z = Vec::with_capacity(n);
    let mut a_type = Vec::with_capacity(n);
    let mut subst_id = Vec::with_capacity(n);
    let mut subst_name = Vec::with_capacity(n);
    let mut charge = Vec::with_capacity(n);
    let mut have_subst = false;
    let mut have_charge = false;
    for a in &atoms {
        id.push(a.id);
        a_name.push(a.name.clone());
        x.push(a.x);
        y.push(a.y);
        z.push(a.z);
        a_type.push(a.atom_type.clone());
        subst_id.push(a.subst_id.unwrap_or(0));
        subst_name.push(a.subst_name.clone().unwrap_or_default());
        charge.push(a.charge.unwrap_or(0.0));
        if a.subst_id.is_some() {
            have_subst = true;
        }
        if a.charge.is_some() {
            have_charge = true;
        }
    }
    insert_column(&mut block, "id", id.iter().map(|&v| v as Idx).collect())?;
    insert_column(&mut block, "name", a_name)?;
    insert_column(&mut block, "x", x)?;
    insert_column(&mut block, "y", y)?;
    insert_column(&mut block, "z", z)?;
    let (elements, element_valid): (Vec<String>, Vec<bool>) = atoms
        .iter()
        .map(|a| match atom_element(&a.atom_type) {
            Some(symbol) => (symbol.to_string(), true),
            None => (String::new(), false),
        })
        .unzip();
    insert_column(&mut block, keys::TYPE, a_type)?;
    // A row whose species the file does not spell is null, not guessed; a file
    // that spells none at all gets no `element` column (as the PDB reader).
    if element_valid.iter().all(|&v| v) {
        insert_column(&mut block, keys::ELEMENT, elements)?;
    } else if element_valid.iter().any(|&v| v) {
        insert_column_masked(&mut block, keys::ELEMENT, elements, element_valid)?;
    }
    if have_subst {
        // `res_id` is unsigned in the vocabulary; a negative `subst_id`
        // (unheard of in practice) is clamped to 0.
        insert_column(
            &mut block,
            keys::RES_ID,
            subst_id.iter().map(|&v| v.max(0) as Idx).collect(),
        )?;
        insert_column(&mut block, keys::RES_NAME, subst_name)?;
    }
    if have_charge {
        insert_column(&mut block, "charge", charge)?;
    }

    let mut frame = Frame::new();
    if !name.is_empty() {
        frame.meta.insert("title", name);
    }
    frame.insert("atoms", block);

    if !bonds.is_empty() {
        let bn = bonds.len();
        let mut atomi = Vec::with_capacity(bn);
        let mut atomj = Vec::with_capacity(bn);
        let mut btype = Vec::with_capacity(bn);
        let mut btype_src: Vec<String> = Vec::with_capacity(bn);
        for b in &bonds {
            if b.atom_i <= 0 || b.atom_j <= 0 {
                return Err(invalid_data(format!(
                    "non-positive atom index in bond: {} - {}",
                    b.atom_i, b.atom_j
                )));
            }
            if (b.atom_i as usize) > n || (b.atom_j as usize) > n {
                return Err(invalid_data(format!(
                    "bond references out-of-range atom: {} - {} (have {} atoms)",
                    b.atom_i, b.atom_j, n
                )));
            }
            atomi.push((b.atom_i as Idx) - 1);
            atomj.push((b.atom_j as Idx) - 1);
            btype.push(b.bond_type.clone());
            btype_src.push(b.bond_type.clone());
        }
        let mut bblock = Block::new();
        insert_column(&mut bblock, "atomi", atomi)?;
        insert_column(&mut bblock, "atomj", atomj)?;
        // `ar` declares delocalization and no Kekulé phase, so its number is
        // left unknown; `am` is the amide C–N, a plain single bond. The
        // SYBYL token is stored as `type` only when those two codes do not
        // spell it back (`am`, `du`, `un`, `nc`): `1` / `2` / `3` / `ar`
        // would be a second name for the same bond.
        let canonical_type: Vec<Idx> = btype_src
            .iter()
            .map(|t| match t.as_str() {
                "ar" => 4,
                "2" => 2,
                "3" => 3,
                _ => 1,
            })
            .collect();
        let canonical_number: Vec<Idx> = btype_src
            .iter()
            .map(|t| match t.as_str() {
                "ar" => 0,
                "2" => 2,
                "3" => 3,
                _ => 1,
            })
            .collect();
        insert_column(&mut bblock, "bond_type", canonical_type)?;
        insert_column(&mut bblock, "bond_number", canonical_number)?;
        let extra: Vec<bool> = btype_src
            .iter()
            .map(|t| !sybyl_token_is_canonical(t))
            .collect();
        if extra.iter().any(|&kept| kept) {
            if extra.iter().all(|&kept| kept) {
                insert_column(&mut bblock, keys::TYPE, btype)?;
            } else {
                insert_column_masked(&mut bblock, keys::TYPE, btype, extra)?;
            }
        }
        frame.insert("bonds", bblock);
    }

    Ok(frame)
}

/// SYBYL atom types that name no element: lone pair, dummy (`Du`, `Du.C`)
/// and the query wildcards.
const SYBYL_NON_ATOMS: [&str; 6] = ["LP", "Du", "Any", "Hal", "Het", "Hev"];

/// The element symbol a MOL2 atom row states, if it states one.
///
/// Only the SYBYL atom type states a species: its part before the dot
/// (`C.ar` → `C`, `N.pl3` → `N`), or the whole token (`H`, `Br`, `Cl`), when
/// that is an element symbol written in its own capitalization. Case is the
/// evidence that the column holds SYBYL types: force-field types are written
/// in lower case (GAFF `c3`, `ca`, `br`) or upper case (Amber `CT`, `CA`,
/// `HO`) and state no element. A SYBYL non-atom (`LP`, `Du`, `Du.C`, `Any`,
/// `Hal`, `Het`, `Hev`) has none either.
///
/// The atom name is a label and is never read as a species. `None` leaves
/// the row without an element: the file did not state one.
fn atom_element(atom_type: &str) -> Option<&'static str> {
    let prefix = atom_type.split('.').next().unwrap_or("");
    if SYBYL_NON_ATOMS.contains(&prefix) {
        return None;
    }
    let element = molrs::core::Element::by_symbol(prefix)?;
    (element.symbol() == prefix).then_some(element.symbol())
}

/// Read the first molecule of a `.mol2` file, checked as [`Mol2Reader`]
/// checks every frame it reads.
pub fn read_mol2<'a>(src: impl Into<Source<'a>>) -> Result<Frame> {
    let src = src.into();
    Mol2Reader::new(src.reader()?)
        .read()?
        .ok_or_else(|| invalid_data(format!("{}: no MOL2 MOLECULE block", src.name())))
}

/// `FrameReader`-trait wrapper. Each call returns the next molecule or `None`
/// at EOF.
///
/// MOL2 is a section-delimited text format used widely in cheminformatics.
/// Sections are introduced by lines starting with `@<TRIPOS>`. The four
/// sections this reader honours:
///
/// - `MOLECULE` — name + counts line (`n_atoms n_bonds n_subst n_feat n_sets`)
/// - `ATOM` — `id name x y z atom_type [subst_id subst_name [charge]]`
/// - `BOND` — `id atom_i atom_j type` (type ∈ `1 2 3 am ar du un nc`)
/// - All other sections are tolerantly skipped.
///
/// # Output Frame
///
/// Columns carry the canonical vocabulary names; the MOL2 field each one
/// comes from is in parentheses.
///
/// - `"atoms"` block: `id` (uint, 1-based; `atom_id`), `name` (str;
///   `atom_name`), `x`/`y`/`z` (F), `type` (str; the SYBYL `atom_type`),
///   `res_id` (uint; `subst_id`) and `res_name` (str; `subst_name`) when
///   present, `charge` (F) when present.
/// - `element` (str): MOL2 has no element field, so it is derived — from a
///   SYBYL atom type (`C.ar` → `C`, `Br` → `Br`), else from the atom name's
///   leading letters (`C1` → `C`, `Br12` → `Br`) when the type is a force-field
///   type (GAFF `c3`, Amber `CT`). A row neither spells unambiguously is null,
///   not guessed (`CA` could be calcium or Cα); a file that spells no element
///   has no `element` column. See `atom_element` for the exact rules.
/// - `"bonds"` block (when present): `atomi`/`atomj` (uint, 0-based indices),
///   `bond_type` / `bond_number` (the chemical class and Kekulé number).
///   `type` is the SYBYL token only when the codes do not spell it (`am`,
///   `du`, `un`, `nc`); `1` / `2` / `3` / `ar` are the codes alone.
/// - `frame.meta["title"]` = molecule name.
///
/// [`Mol2Writer`] reads the same names back.
///
/// Over a seekable stream it is also a [`TrajectoryReader`]: molecules by
/// index, each starting at its `@<TRIPOS>MOLECULE` line.
pub struct Mol2Reader<R: BufRead> {
    reader: R,
    pending: Option<String>,
    index: Option<Vec<FrameOffset>>,
}

impl<R: BufRead> Reader for Mol2Reader<R> {
    type R = R;
    fn new(reader: R) -> Self {
        Self {
            reader,
            pending: None,
            index: None,
        }
    }
}

impl Mol2Reader<Box<dyn ReadSeek>> {
    /// Open a `.mol2` file, or the bytes of one, for random access by
    /// molecule.
    pub fn open<'a>(src: impl Into<Source<'a>>) -> Result<Self> {
        Ok(Self::new(src.into().owned_reader()?))
    }
}

impl<R: BufRead + Seek> TrajectoryReader for Mol2Reader<R> {
    fn build_index(&mut self) -> Result<()> {
        if self.index.is_none() {
            self.index = Some(crate::io::frame_index::index_line_records(
                &mut self.reader,
                |line| line.trim_start().starts_with("@<TRIPOS>MOLECULE"),
            )?);
        }
        Ok(())
    }

    fn read_frame(&mut self, index: usize) -> Result<Option<Frame>> {
        self.build_index()?;
        let Some(at) = self.index.as_ref().and_then(|o| o.get(index).copied()) else {
            return Ok(None);
        };
        let bytes = crate::io::frame_index::frame_bytes(&mut self.reader, at)?;
        read_mol2(bytes.as_slice()).map(Some)
    }

    fn len(&mut self) -> Result<usize> {
        self.build_index()?;
        Ok(self.index.as_ref().map_or(0, Vec::len))
    }
}

impl<R: BufRead> FrameReader for Mol2Reader<R> {
    fn read(&mut self) -> Result<Option<Frame>> {
        // Validate on the way out: a frame that violates the vocabulary
        // is a malformed file or a reader bug, not a result to return.
        read_one_record(&mut self.reader, &mut self.pending)
    }
}

// ---------------------------------------------------------------------------
// Writer
// ---------------------------------------------------------------------------

/// Write a Frame as a single-molecule `.mol2` file.
///
/// The atom `type` column is written as given — SYBYL (`C.3`) or force-field
/// (GAFF `c3`). A frame without the column is written with the bare element
/// symbol as the atom type (`C`, `Br`), which SYBYL allows and [`read_mol2`]
/// reads back as the element; the symbol comes from `element`, else from
/// `atomic_number`, and a frame that states none of the three is refused. A
/// hybridized type (`C.3`, `C.ar`) is never guessed from the element. The
/// `element` column itself has no MOL2 field; [`read_mol2`] derives it back
/// from the type or the name (a frame without `name` is written with
/// element-based names, `C1`, `Br12`, which read back as their elements).
///
/// A bond's SYBYL type is its `type` cell when that cell is a SYBYL bond type
/// (`1`, `2`, `3`, `am`, `ar`, `du`, `un`, `nc`), else the canonical
/// `bond_type` / `bond_number` codes mapped back (`4` → `ar`). So a frame read
/// from MOL2 keeps its aromatic and double bonds without `type`, and a
/// force-field bond type name in `type` (GAFF `ca-ca`) never lands in the
/// order field.
pub fn write_mol2<'a>(sink: impl Into<Sink<'a>>, frame: &Frame) -> Result<()> {
    sink.into().put_with(|w| write_frame_to(w, frame))
}

/// Emit one frame in MOL2 format.
fn write_frame_to<W: Write>(writer: &mut W, frame: &Frame) -> Result<()> {
    crate::io::null_cells::refuse_null_cells(
        frame,
        "atoms",
        &[
            "id",
            "x",
            "y",
            "z",
            "charge",
            "type",
            keys::RES_ID,
            keys::RES_NAME,
        ],
    )?;
    // A null bond `type` is a token the codes already spell (`1`, `ar`, …).
    // The writer rebuilds it. Endpoints and the codes themselves are stated.
    crate::io::null_cells::refuse_null_cells(
        frame,
        "bonds",
        &["atomi", "atomj", keys::BOND_TYPE, keys::BOND_NUMBER],
    )?;
    // Every door writes through here, so every door checks here.
    crate::io::writer::check_write_frame(frame)?;
    let atoms = frame
        .get("atoms")
        .ok_or_else(|| invalid_data("MOL2 write: frame has no atoms block"))?;
    let n = atoms
        .n_rows()
        .ok_or_else(|| invalid_data("MOL2 write: atoms block has no rows"))?;
    let bonds = frame.get("bonds");
    let n_bonds = bonds.and_then(|b| b.n_rows()).unwrap_or(0);

    let title = frame
        .meta
        .get("title")
        .and_then(|value| value.as_str())
        .unwrap_or("molrs");

    writeln!(writer, "@<TRIPOS>MOLECULE")?;
    writeln!(writer, "{}", title)?;
    writeln!(writer, "{} {} 0 0 0", n, n_bonds)?;
    writeln!(writer, "SMALL")?;
    let charge_col = atoms.get("charge").and_then(|c| c.get::<f64>());
    writeln!(
        writer,
        "{}",
        if charge_col.is_some() {
            "USER_CHARGES"
        } else {
            "NO_CHARGES"
        }
    )?;
    writeln!(writer)?;
    writeln!(writer)?;

    writeln!(writer, "@<TRIPOS>ATOM")?;
    let id_col = atoms.get("id").and_then(|c| c.get::<u64>());
    // The Tripos atom type is where mol2 spells the species (`C.3`, `O.2`, or
    // a force-field type): `type` as given, else the bare element symbol. The
    // atom name is a label: an absent column, or a null cell, is named as
    // antechamber names atoms, element symbol then 1-based row (`C1`, `H7`).
    let names = atom_names(atoms, n)?;
    let types = atom_types(atoms, n)?;
    let xs = atoms
        .get("x")
        .and_then(|c| c.get::<f64>())
        .ok_or_else(|| invalid_data("atoms.x missing"))?;
    let ys = atoms
        .get("y")
        .and_then(|c| c.get::<f64>())
        .ok_or_else(|| invalid_data("atoms.y missing"))?;
    let zs = atoms
        .get("z")
        .and_then(|c| c.get::<f64>())
        .ok_or_else(|| invalid_data("atoms.z missing"))?;
    let subst_id_col = atoms.get(keys::RES_ID).and_then(|c| c.get::<u64>());
    let subst_name_col = atoms.get(keys::RES_NAME).and_then(|c| c.get::<String>());
    for i in 0..n {
        let id = id_col.map(|c| c[[i]]).unwrap_or((i as Idx) + 1);
        let name = names[i].as_str();
        let atype = types[i].as_str();
        let sid = subst_id_col.map(|c| c[[i]]).unwrap_or(1);
        let sname = subst_name_col.map(|c| c[[i]].as_str()).unwrap_or("MOL");
        write!(
            writer,
            "{:>7} {:<8} {:>10.4} {:>10.4} {:>10.4} {:<6} {} {:<8}",
            id,
            name,
            xs[[i]],
            ys[[i]],
            zs[[i]],
            atype,
            sid,
            sname
        )?;
        if let Some(c) = charge_col {
            write!(writer, " {:>9.4}", c[[i]])?;
        }
        writeln!(writer)?;
    }

    if let Some(b) = bonds
        && n_bonds > 0
    {
        writeln!(writer, "@<TRIPOS>BOND")?;
        let atomi = b
            .get("atomi")
            .and_then(|c| c.get::<u64>())
            .ok_or_else(|| invalid_data("bonds.atomi missing"))?;
        let atomj = b
            .get("atomj")
            .and_then(|c| c.get::<u64>())
            .ok_or_else(|| invalid_data("bonds.atomj missing"))?;
        let btype_col = b.get(keys::TYPE).and_then(|c| c.get::<String>());
        let type_valid = b.validity(keys::TYPE);
        let class_col = b.get(keys::BOND_TYPE).and_then(|c| c.get::<u64>());
        let number_col = b.get(keys::BOND_NUMBER).and_then(|c| c.get::<u64>());
        for i in 0..n_bonds {
            let stated = btype_col.as_ref().and_then(|column| {
                let live = type_valid.map(|mask| mask[i]).unwrap_or(true);
                // Only a SYBYL token is a MOL2 bond type: a force-field bond
                // type name (GAFF `ca-ca`) says nothing about the order, so
                // the codes speak for that bond instead.
                if live && is_sybyl_bond_token(&column[[i]]) {
                    Some(column[[i]].clone())
                } else {
                    None
                }
            });
            let bt = match stated {
                Some(token) => token,
                None => sybyl_bond_token(
                    class_col.as_ref().map(|column| column[[i]]),
                    number_col.as_ref().map(|column| column[[i]]),
                )?
                .to_string(),
            };
            writeln!(
                writer,
                "{:>6} {:>6} {:>6} {}",
                i + 1,
                atomi[[i]] + 1,
                atomj[[i]] + 1,
                bt
            )?;
        }
    }
    // The cell, when the frame has one: lengths and angles, space group P1
    // (number 1), setting 1. A frame without a box writes no CRYSIN.
    if let Some(simbox) = frame.simbox.as_ref() {
        let (lengths, angles) = (simbox.lengths(), simbox.angles());
        writeln!(writer, "@<TRIPOS>CRYSIN")?;
        writeln!(
            writer,
            "{:.4} {:.4} {:.4} {:.4} {:.4} {:.4} 1 1",
            lengths[0], lengths[1], lengths[2], angles[0], angles[1], angles[2]
        )?;
    }
    Ok(())
}

/// The MOL2 atom name of every row: the `name` cell where the frame states
/// one, else the element symbol (the `element` column, or the symbol of
/// `atomic_number`) followed by the 1-based row.
///
/// # Errors
///
/// `InvalidData` when a row needs a generated name and the frame states its
/// species in neither column, or `atomic_number` holds no element's number.
fn atom_names(atoms: &Block, n: usize) -> Result<Vec<String>> {
    let stated = atoms.get("name").and_then(|c| c.get::<String>());
    let valid = atoms.validity("name");
    let symbol = species_symbol(atoms);
    (0..n)
        .map(|i| {
            if let Some(column) = stated
                && valid.is_none_or(|mask| mask[i])
            {
                return Ok(column[[i]].clone());
            }
            let symbol = symbol(i)?.ok_or_else(|| {
                invalid_data(format!(
                    "MOL2 write: atoms row {i} has no name and the frame states \
                     no 'element' or 'atomic_number' to name it from"
                ))
            })?;
            Ok(format!("{symbol}{}", i + 1))
        })
        .collect()
}

/// The MOL2 atom type of every row: the `type` cell where the frame has the
/// column (SYBYL `C.3` or force-field `c3`, as given), else the bare element
/// symbol (the `element` column, or the symbol of `atomic_number`), which
/// SYBYL allows as an atom type and [`read_mol2`] reads back as the element.
/// Hybridization is never guessed: a frame without types gets `C`, not `C.3`.
///
/// # Errors
///
/// `InvalidData` when the frame has no `type` column and a row states its
/// species in neither column, or `atomic_number` holds no element's number.
fn atom_types(atoms: &Block, n: usize) -> Result<Vec<String>> {
    if let Some(column) = atoms.get(keys::TYPE).and_then(|c| c.get::<String>()) {
        return Ok((0..n).map(|i| column[[i]].clone()).collect());
    }
    let symbol = species_symbol(atoms);
    (0..n)
        .map(|i| {
            symbol(i)?.ok_or_else(|| {
                invalid_data(format!(
                    "MOL2 write: atoms block has no string 'type' column and row {i} \
                     states no 'element' or 'atomic_number' to write as the atom type; \
                     set one of them before writing"
                ))
            })
        })
        .collect()
}

/// The element symbol `atoms` states for a row, looked up on demand: the live,
/// non-empty `element` cell, else the symbol of a live `atomic_number`; `None`
/// for a row that states neither. What a missing name or type is spelled from.
///
/// # Errors
///
/// `InvalidData` when the row's `atomic_number` names no element.
fn species_symbol(atoms: &Block) -> impl Fn(usize) -> Result<Option<String>> + '_ {
    let symbols = atoms.get(keys::ELEMENT).and_then(|c| c.get::<String>());
    let symbol_valid = atoms.validity(keys::ELEMENT);
    let numbers = atoms.get(keys::ATOMIC_NUMBER).and_then(|c| c.get::<u64>());
    let number_valid = atoms.validity(keys::ATOMIC_NUMBER);
    move |i| {
        if let Some(column) = symbols
            && symbol_valid.is_none_or(|mask| mask[i])
            && !column[[i]].is_empty()
        {
            return Ok(Some(column[[i]].clone()));
        }
        let Some(number) = numbers
            .filter(|_| number_valid.is_none_or(|mask| mask[i]))
            .map(|column| column[[i]])
        else {
            return Ok(None);
        };
        u8::try_from(number)
            .ok()
            .and_then(molrs::core::Element::by_number)
            .map(|e| Some(e.symbol().to_string()))
            .ok_or_else(|| {
                invalid_data(format!(
                    "MOL2 write: atoms row {i}: atomic_number {number} names no element"
                ))
            })
    }
}

/// `true` when `token` is one of the SYBYL bond types a MOL2 `BOND` record
/// may carry (`1`, `2`, `3`, `am`, `ar`, `du`, `un`, `nc`).
fn is_sybyl_bond_token(token: &str) -> bool {
    matches!(token, "1" | "2" | "3" | "am" | "ar" | "du" | "un" | "nc")
}

/// `true` when `token` is not the spelling of [`sybyl_bond_token`] on the
/// codes the reader stores for it (`1`, `2`, `3`, `ar`).
fn sybyl_token_is_canonical(token: &str) -> bool {
    matches!(token, "1" | "2" | "3" | "ar")
}

/// The SYBYL bond token for a bond with no `type` column: the inverse of
/// the reader's map, from the canonical [`keys::BOND_TYPE`] class (aromatic
/// is `ar`), else the [`keys::BOND_NUMBER`] count when the class is unknown
/// or absent. A bond that states neither column is written single, as SYBYL
/// files without orders are; one whose columns say unknown (`0`) is `un`.
///
/// # Errors
///
/// `InvalidData` for a code outside the schema's range, or a quadruple bond,
/// which SYBYL cannot spell.
fn sybyl_bond_token(class: Option<Idx>, number: Option<Idx>) -> Result<&'static str> {
    let token = |code: Idx, key: &str| -> Result<Option<&'static str>> {
        match code {
            0 => Ok(None),
            1 => Ok(Some("1")),
            2 => Ok(Some("2")),
            3 => Ok(Some("3")),
            4 if key == keys::BOND_TYPE => Ok(Some("ar")),
            _ => Err(invalid_data(format!(
                "MOL2 write: bonds.{key} = {code} has no SYBYL bond type"
            ))),
        }
    };
    if let Some(found) = class
        .map(|c| token(c, keys::BOND_TYPE))
        .transpose()?
        .flatten()
    {
        return Ok(found);
    }
    if let Some(found) = number
        .map(|n| token(n, keys::BOND_NUMBER))
        .transpose()?
        .flatten()
    {
        return Ok(found);
    }
    Ok(if class.is_none() && number.is_none() {
        "1"
    } else {
        "un"
    })
}

/// `FrameWriter`-trait wrapper.
pub struct Mol2Writer<W: Write> {
    writer: W,
}

impl<'a> Mol2Writer<SinkWriter<'a>> {
    /// Create a `.mol2` file, or fill a buffer, molecule by molecule.
    pub fn create(sink: impl Into<Sink<'a>>) -> Result<Self> {
        Ok(Self {
            writer: sink.into().writer()?,
        })
    }
}

impl<W: Write> Writer for Mol2Writer<W> {
    type W = W;
    fn new(writer: W) -> Self {
        Self { writer }
    }
}

impl<W: Write> FrameWriter for Mol2Writer<W> {
    fn write(&mut self, frame: &Frame) -> Result<()> {
        write_frame_to(&mut self.writer, frame)?;
        self.writer.flush()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn crysin_round_trips_the_box() {
        let mut frame = read_mol2(ETHANE_MIN.as_bytes()).unwrap();
        assert!(frame.simbox.is_none());
        let cell = crate::core::Cell::LengthsAngles {
            lengths: [68.105, 66.514, 102.898],
            angles: [90.0, 90.0, 90.0],
        };
        frame.simbox = Some(SimBox::new(cell, [0.0; 3], [true; 3]).unwrap());
        let mut out = Vec::new();
        write_mol2(&mut out, &frame).unwrap();
        let text = String::from_utf8(out).unwrap();
        assert!(
            text.contains(
                "@<TRIPOS>CRYSIN\n68.1050 66.5140 102.8980 90.0000 90.0000 90.0000 1 1\n"
            ),
            "{text}"
        );
        let back = read_mol2(text.as_bytes()).unwrap();
        let lengths = back
            .simbox
            .as_ref()
            .expect("CRYSIN reads back as a box")
            .lengths();
        assert!((lengths[0] - 68.105).abs() < 1e-3 && (lengths[2] - 102.898).abs() < 1e-3);
    }

    #[test]
    fn a_frame_without_a_box_writes_no_crysin() {
        let frame = read_mol2(ETHANE_MIN.as_bytes()).unwrap();
        let mut out = Vec::new();
        write_mol2(&mut out, &frame).unwrap();
        assert!(!String::from_utf8(out).unwrap().contains("CRYSIN"));
    }

    const ETHANE_MIN: &str = "@<TRIPOS>MOLECULE\nETH\n2 1 1 0 0\nSMALL\nNO_CHARGES\n@<TRIPOS>ATOM\n1 C1 0.0 0.0 0.0 c3 1 ETH 0.0\n2 C2 1.5 0.0 0.0 c3 1 ETH 0.0\n@<TRIPOS>BOND\n1 1 2 1\n";

    /// `Mol2Writer::create` writes one molecule per frame; `Mol2Reader::open`
    /// reads them back by index.
    #[test]
    fn mol2_writer_create_and_reader_open_round_trip() {
        let frame = read_mol2(ETHANE_MIN.as_bytes()).unwrap();
        let mut out = Vec::new();
        let mut writer = Mol2Writer::create(&mut out).unwrap();
        writer.write(&frame).unwrap();
        writer.write(&frame).unwrap();
        drop(writer);
        let mut reader = Mol2Reader::open(&out).unwrap();
        assert_eq!(reader.len().unwrap(), 2);
        assert_eq!(
            reader.read_frame(1).unwrap().unwrap()["atoms"].n_rows(),
            Some(2)
        );
        assert!(reader.read_frame(2).unwrap().is_none());
    }

    #[test]
    fn reads_minimal_mol2() {
        let mut reader = Mol2Reader::new(Cursor::new(ETHANE_MIN.as_bytes()));
        let frame = reader.read().unwrap().unwrap();
        let atoms = frame.get("atoms").unwrap();
        assert_eq!(atoms.n_rows(), Some(2));
        let xs = atoms.get("x").and_then(|c| c.get::<f64>()).unwrap();
        assert!((xs[[1]] - 1.5).abs() < 1e-9);
        let bonds = frame.get("bonds").unwrap();
        assert_eq!(bonds.n_rows(), Some(1));
        let atomi = bonds.get("atomi").and_then(|c| c.get::<u64>()).unwrap();
        let atomj = bonds.get("atomj").and_then(|c| c.get::<u64>()).unwrap();
        assert_eq!(atomi[[0]], 0);
        assert_eq!(atomj[[0]], 1);
    }

    /// A frame with no `name` column writes antechamber-style names built from
    /// the species: `element` where stated, else the symbol of `atomic_number`.
    #[test]
    fn absent_atom_names_are_generated_from_the_species() {
        use ndarray::Array1;
        let mut atoms = Block::new();
        for (key, v) in [("x", [0.0, 0.96]), ("y", [0.0, 0.0]), ("z", [0.0, 0.0])] {
            atoms
                .insert(key, Array1::from_vec(v.to_vec()).into_dyn(), None)
                .unwrap();
        }
        atoms
            .insert(
                keys::TYPE,
                Array1::from_vec(vec!["o".to_string(), "ho".to_string()]).into_dyn(),
                None,
            )
            .unwrap();
        atoms
            .insert(
                keys::ATOMIC_NUMBER,
                Array1::from_vec(vec![8_u64, 1]).into_dyn(),
                None,
            )
            .unwrap();
        let mut frame = Frame::new();
        frame.insert("atoms", atoms);

        let mut buf = Vec::new();
        write_frame_to(&mut buf, &frame).unwrap();
        let back = Mol2Reader::new(Cursor::new(&buf)).read().unwrap().unwrap();
        let names = back["atoms"]
            .get("name")
            .and_then(|c| c.get::<String>())
            .unwrap();
        assert_eq!((names[[0]].as_str(), names[[1]].as_str()), ("O1", "H2"));

        // Without either species column there is nothing to name a row from.
        frame
            .get_mut("atoms")
            .unwrap()
            .remove(keys::ATOMIC_NUMBER)
            .unwrap();
        let err = write_frame_to(&mut Vec::new(), &frame).expect_err("no species");
        assert!(err.to_string().contains("no name"), "{err}");
    }

    /// A frame that states its species but has no `type` column is written
    /// with the bare element symbol as the SYBYL atom type (`C`, `Br`, never
    /// a guessed `C.3`), which reads back as the element; one that states
    /// neither `type`, `element` nor `atomic_number` is refused.
    #[test]
    fn absent_atom_types_are_the_bare_element_symbols() {
        use ndarray::Array1;
        let mut atoms = Block::new();
        for (key, v) in [("x", [0.0, 1.9, 3.0]), ("y", [0.0; 3]), ("z", [0.0; 3])] {
            atoms
                .insert(key, Array1::from_vec(v.to_vec()).into_dyn(), None)
                .unwrap();
        }
        let symbols = ["C", "Br", "H"].map(str::to_string);
        atoms
            .insert(
                keys::ELEMENT,
                Array1::from_vec(symbols.to_vec()).into_dyn(),
                None,
            )
            .unwrap();
        let mut bonds = Block::new();
        for (key, v) in [("atomi", [0_u64, 0]), ("atomj", [1_u64, 2])] {
            bonds
                .insert(key, Array1::from_vec(v.to_vec()).into_dyn(), None)
                .unwrap();
        }
        let mut frame = Frame::new();
        frame.insert("atoms", atoms);
        frame.insert("bonds", bonds);

        let types_of = |frame: &Frame| -> Vec<String> {
            let mut buf = Vec::new();
            write_frame_to(&mut buf, frame).unwrap();
            let text = String::from_utf8(buf).unwrap();
            assert!(text.contains("@<TRIPOS>ATOM\n"), "{text}");
            assert!(text.contains("@<TRIPOS>BOND\n"), "{text}");
            let back = read_mol2(text.as_bytes()).unwrap();
            assert_eq!(back["bonds"].n_rows(), Some(2));
            assert_eq!(elements_of(&back), symbols.clone().map(Some));
            let types = back["atoms"]
                .get(keys::TYPE)
                .and_then(|c| c.get::<String>())
                .unwrap();
            types.iter().cloned().collect()
        };
        assert_eq!(types_of(&frame), symbols);

        // `atomic_number` spells the symbol when `element` is absent too.
        let atoms = frame.get_mut("atoms").unwrap();
        atoms.remove(keys::ELEMENT).unwrap();
        atoms
            .insert(
                keys::ATOMIC_NUMBER,
                Array1::from_vec(vec![6_u64, 35, 1]).into_dyn(),
                None,
            )
            .unwrap();
        assert_eq!(types_of(&frame), symbols);

        // Neither column: nothing spells the type, so the write is refused
        // (named atoms, so that the type is what is missing).
        let atoms = frame.get_mut("atoms").unwrap();
        atoms.remove(keys::ATOMIC_NUMBER).unwrap();
        atoms
            .insert(
                "name",
                Array1::from_vec(["C1", "Br2", "H3"].map(str::to_string).to_vec()).into_dyn(),
                None,
            )
            .unwrap();
        let err = write_frame_to(&mut Vec::new(), &frame).expect_err("no type");
        assert!(err.to_string().contains("'type'"), "{err}");
    }

    /// The `element` column of a read frame, `None` for a null row.
    fn elements_of(frame: &Frame) -> Vec<Option<String>> {
        let atoms = &frame["atoms"];
        let column = atoms
            .get(keys::ELEMENT)
            .and_then(|c| c.get::<String>())
            .expect("an element column");
        let valid = atoms.validity(keys::ELEMENT);
        (0..column.len())
            .map(|i| valid.is_none_or(|m| m[i]).then(|| column[[i]].clone()))
            .collect()
    }

    /// A one-molecule MOL2 of `(name, type)` atoms, no bonds.
    fn mol2_of(atoms: &[(&str, &str)]) -> String {
        let mut text = format!(
            "@<TRIPOS>MOLECULE\nM\n{} 0 0 0 0\nSMALL\nNO_CHARGES\n@<TRIPOS>ATOM\n",
            atoms.len()
        );
        for (i, (name, atype)) in atoms.iter().enumerate() {
            text += &format!("{} {name} {}.0 0.0 0.0 {atype} 1 MOL 0.0\n", i + 1, i);
        }
        text
    }

    /// GAFF types are not element symbols (`ca` is not calcium, `br` is not
    /// read case-blind), and the antechamber-style name is a label: no element.
    #[test]
    fn a_gaff_typed_file_states_no_elements() {
        // Force-field types state no species, and a name is a label.
        let text = mol2_of(&[("C1", "c3"), ("H4", "ho"), ("Br12", "br"), ("ZN1", "zn")]);
        let frame = read_mol2(text.as_bytes()).unwrap();
        assert!(!frame["atoms"].contains_key(keys::ELEMENT));
    }

    /// SYBYL types spell the element before the dot, or as the whole token.
    #[test]
    fn a_sybyl_typed_file_takes_its_elements_from_the_types() {
        // Names that say nothing (or the wrong thing) do not override the type.
        let text = mol2_of(&[
            ("CA", "C.3"),
            ("X1", "C.ar"),
            ("OXT", "O.3"),
            ("HO", "H"),
            ("BR", "Br"),
            ("Q", "N.pl3"),
            ("CL9", "Cl"),
        ]);
        let frame = read_mol2(text.as_bytes()).unwrap();
        let want = ["C", "C", "O", "H", "Br", "N", "Cl"];
        assert_eq!(elements_of(&frame), want.map(|s| Some(s.to_string())));
    }

    /// A row that spells no element unambiguously is null, never guessed; a
    /// file that spells none has no column at all.
    #[test]
    fn an_unspelled_element_is_left_out_not_guessed() {
        let text = mol2_of(&[
            ("CA", "CT"),   // Amber type
            ("HO1", "ho"),  // GAFF type
            ("LP1", "LP"),  // SYBYL lone pair
            ("C7", "Du.C"), // SYBYL dummy
            ("Na2", "Na+"), // Amber ion type
            ("X", "C.ar"),  // SYBYL type: the one stated species
        ]);
        let frame = read_mol2(text.as_bytes()).unwrap();
        let want = [None, None, None, None, None, Some("C")];
        assert_eq!(elements_of(&frame), want.map(|s| s.map(str::to_string)));

        let none = read_mol2(mol2_of(&[("Q1", "CT"), ("Q2", "c3")]).as_bytes()).unwrap();
        assert!(!none["atoms"].contains_key(keys::ELEMENT));
    }

    /// Writing and reading back keeps the elements a SYBYL-typed file states.
    #[test]
    fn elements_survive_a_write_read_round_trip() {
        let text = mol2_of(&[("A", "C.ar"), ("B", "Br"), ("C", "H")]);
        let frame = read_mol2(text.as_bytes()).unwrap();
        let mut buf = Vec::new();
        write_frame_to(&mut buf, &frame).unwrap();
        let back = read_mol2(buf.as_slice()).unwrap();
        assert_eq!(elements_of(&back), elements_of(&frame));
        assert_eq!(elements_of(&back)[1].as_deref(), Some("Br"));
    }

    #[test]
    fn columns_carry_canonical_names() {
        let mut reader = Mol2Reader::new(Cursor::new(ETHANE_MIN.as_bytes()));
        let frame = reader.read().unwrap().unwrap();
        let atoms = frame.get("atoms").unwrap();
        for key in ["type", "res_id", "res_name"] {
            assert!(atoms.contains_key(key), "atoms.{key}");
        }
        for key in ["atom_type", "subst_id", "subst_name"] {
            assert!(!atoms.contains_key(key), "atoms.{key} is format-native");
        }
        let bonds = frame.get("bonds").unwrap();
        assert!(!bonds.contains_key(keys::TYPE));
        assert!(!bonds.contains_key("sybyl_bond_type"));
        assert_eq!(
            bonds
                .get(keys::BOND_TYPE)
                .and_then(|c| c.get::<u64>())
                .unwrap()[[0]],
            1
        );
        assert_eq!(
            bonds
                .get(keys::BOND_NUMBER)
                .and_then(|c| c.get::<u64>())
                .unwrap()[[0]],
            1
        );
    }

    #[test]
    fn residues_and_types_round_trip() {
        const TWO_RES: &str = "@<TRIPOS>MOLECULE\nX\n2 1 2 0 0\nSMALL\nNO_CHARGES\n@<TRIPOS>ATOM\n1 C1 0.0 0.0 0.0 C.ar 7 BEN 0.0\n2 N1 1.4 0.0 0.0 N.am 9 AMD 0.0\n@<TRIPOS>BOND\n1 1 2 am\n";
        let frame = Mol2Reader::new(Cursor::new(TWO_RES.as_bytes()))
            .read()
            .unwrap()
            .unwrap();
        let mut buf = Vec::new();
        write_frame_to(&mut buf, &frame).unwrap();
        let back = Mol2Reader::new(Cursor::new(&buf)).read().unwrap().unwrap();
        let atoms = back.get("atoms").unwrap();
        let res_id: Vec<Idx> = atoms
            .get("res_id")
            .and_then(|c| c.get::<u64>())
            .unwrap()
            .iter()
            .copied()
            .collect();
        assert_eq!(res_id, [7, 9]);
        let types = atoms.get("type").and_then(|c| c.get::<String>()).unwrap();
        assert_eq!((types[[0]].as_str(), types[[1]].as_str()), ("C.ar", "N.am"));
        let res_name = atoms
            .get("res_name")
            .and_then(|c| c.get::<String>())
            .unwrap();
        assert_eq!(res_name[[1]], "AMD");
        let btype = back
            .get("bonds")
            .unwrap()
            .get("type")
            .and_then(|c| c.get::<String>())
            .unwrap();
        assert_eq!(btype[[0]], "am");
    }

    #[test]
    fn bond_orders_round_trip_without_the_type_column() {
        const BENZALDEHYDE_ISH: &str = "@<TRIPOS>MOLECULE\nX\n4 3 1 0 0\nSMALL\nNO_CHARGES\n@<TRIPOS>ATOM\n1 C1 0.0 0.0 0.0 C.ar 1 X\n2 C2 1.4 0.0 0.0 C.ar 1 X\n3 C3 2.8 0.0 0.0 C.2 1 X\n4 O1 4.0 0.0 0.0 O.2 1 X\n@<TRIPOS>BOND\n1 1 2 ar\n2 2 3 1\n3 3 4 2\n";
        let frame = Mol2Reader::new(Cursor::new(BENZALDEHYDE_ISH.as_bytes()))
            .read()
            .unwrap()
            .unwrap();
        // `ar` / `1` / `2` are the codes themselves, so the reader stores no
        // second `type` column. The writer spells the tokens back from them.
        assert!(!frame.get("bonds").unwrap().contains_key(keys::TYPE));
        let text = crate::io::writer::capture(|sink| crate::io::write_mol2(sink, &frame)).unwrap();
        let bond_rows: Vec<&str> = text
            .lines()
            .skip_while(|line| *line != "@<TRIPOS>BOND")
            .skip(1)
            .take(3)
            .collect();
        assert_eq!(
            bond_rows
                .iter()
                .map(|line| line.split_whitespace().last().unwrap())
                .collect::<Vec<_>>(),
            ["ar", "1", "2"]
        );
        let back = Mol2Reader::new(Cursor::new(text.as_bytes()))
            .read()
            .unwrap()
            .unwrap();
        let bonds = back.get("bonds").unwrap();
        assert!(!bonds.contains_key(keys::TYPE));
        let class: Vec<u64> = bonds
            .get(keys::BOND_TYPE)
            .and_then(|c| c.get::<u64>())
            .unwrap()
            .iter()
            .copied()
            .collect();
        let number: Vec<u64> = bonds
            .get(keys::BOND_NUMBER)
            .and_then(|c| c.get::<u64>())
            .unwrap()
            .iter()
            .copied()
            .collect();
        assert_eq!(class, [4, 1, 2]);
        assert_eq!(number, [0, 1, 2]);
    }

    #[test]
    fn sybyl_bond_token_maps_the_canonical_codes() {
        assert_eq!(sybyl_bond_token(Some(4), Some(2)).unwrap(), "ar");
        assert_eq!(sybyl_bond_token(Some(2), None).unwrap(), "2");
        assert_eq!(sybyl_bond_token(None, Some(3)).unwrap(), "3");
        assert_eq!(sybyl_bond_token(Some(0), Some(2)).unwrap(), "2");
        assert_eq!(sybyl_bond_token(Some(0), Some(0)).unwrap(), "un");
        assert_eq!(sybyl_bond_token(None, None).unwrap(), "1");
        assert!(sybyl_bond_token(Some(5), None).is_err());
        assert!(sybyl_bond_token(None, Some(4)).is_err());
    }

    #[test]
    fn round_trip_minimal_mol2() {
        let frame = {
            let mut reader = Mol2Reader::new(Cursor::new(ETHANE_MIN.as_bytes()));
            reader.read().unwrap().unwrap()
        };
        let mut buf = Vec::new();
        write_frame_to(&mut buf, &frame).unwrap();
        let mut reader2 = Mol2Reader::new(Cursor::new(&buf));
        let frame2 = reader2.read().unwrap().unwrap();
        let xs1 = frame
            .get("atoms")
            .unwrap()
            .get("x")
            .and_then(|c| c.get::<f64>())
            .unwrap();
        let xs2 = frame2
            .get("atoms")
            .unwrap()
            .get("x")
            .and_then(|c| c.get::<f64>())
            .unwrap();
        for i in 0..xs1.len() {
            assert!((xs1[[i]] - xs2[[i]]).abs() < 1e-3);
        }
    }
}
