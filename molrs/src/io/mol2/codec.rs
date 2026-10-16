//! The Tripos MOL2 codec.

use crate::io::frame_columns::insert_column;
use crate::io::invalid_data;
use std::io::{BufRead, BufWriter, Result, Write};
use std::path::Path;

use molrs::core::Block;
use molrs::core::Frame;
use molrs::core::schema::consts as keys;
use molrs::op::{F, I, Idx};

use crate::io::reader::{FrameReader, Reader};
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
            _ => {} // tolerate / skip
        }
    }

    if atoms.is_empty() {
        return Err(invalid_data(format!(
            "MOLECULE '{}' yielded zero atoms (declared {})",
            name, n_atoms
        )));
    }

    Ok(Some(build_frame(name, atoms, bonds)?))
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
    insert_column(&mut block, keys::TYPE, a_type)?;
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
        // SYBYL's own alphabet ("1", "2", "ar", "am") is the bond's type label,
        // so it is the string `type` column; `bond_type` is the chemical class,
        // a uint code, and two quantities never share one key.
        insert_column(&mut bblock, keys::TYPE, btype)?;

        // …and the canonical pair it maps onto. `ar` declares delocalization
        // and no Kekulé phase, so its number is left unknown for
        // standardization; `am` is the amide C–N, a plain single bond.
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
        frame.insert("bonds", bblock);
    }

    Ok(frame)
}

/// Read the first molecule of the `.mol2` file at `path`, checked as
/// [`Mol2Reader`] checks every frame it reads.
pub fn read_mol2<P: AsRef<Path>>(path: P) -> Result<Frame> {
    let file = std::fs::File::open(path.as_ref())?;
    let mut reader = std::io::BufReader::new(file);
    let mut pending = None;
    crate::io::reader::check_read_frame(read_one_record(&mut reader, &mut pending)?)?
        .ok_or_else(|| invalid_data("MOL2 file has no MOLECULE block"))
}

/// Read every molecule of a multi-molecule `.mol2` file at `path`.
pub fn read_mol2_trajectory<P: AsRef<Path>>(path: P) -> Result<Vec<Frame>> {
    let file = std::fs::File::open(path.as_ref())?;
    let reader = std::io::BufReader::new(file);
    let mut mr = Mol2Reader::new(reader);
    crate::io::reader::collect_frames(&mut mr)
}

/// Read the first molecule of MOL2 `text` — [`read_mol2`] on text in memory.
pub fn read_mol2_str(text: &str) -> Result<Frame> {
    let mut reader = text.as_bytes();
    let mut pending = None;
    crate::io::reader::check_read_frame(read_one_record(&mut reader, &mut pending)?)?
        .ok_or_else(|| invalid_data("MOL2 text has no MOLECULE block"))
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
/// - `"bonds"` block (when present): `atomi`/`atomj` (uint, 0-based indices),
///   `type` (str; the SYBYL `bond_type` token `1`, `2`, `ar`, `am`, …), plus
///   the chemical `bond_type` / `bond_number` codes it maps onto.
/// - `frame.meta["title"]` = molecule name.
///
/// [`Mol2Writer`] reads the same names back.
pub struct Mol2Reader<R: BufRead> {
    reader: R,
    pending: Option<String>,
}

impl<R: BufRead> Reader for Mol2Reader<R> {
    type R = R;
    fn new(reader: R) -> Self {
        Self {
            reader,
            pending: None,
        }
    }
}

impl<R: BufRead> FrameReader for Mol2Reader<R> {
    fn read(&mut self) -> Result<Option<Frame>> {
        // Validate on the way out: a frame that violates the vocabulary
        // is a malformed file or a reader bug, not a result to return.
        crate::io::reader::check_read_frame(read_one_record(&mut self.reader, &mut self.pending)?)
    }
}

// ---------------------------------------------------------------------------
// Writer
// ---------------------------------------------------------------------------

/// Write a Frame as a single-molecule `.mol2` file.
///
/// A bond's SYBYL type is its `type` column when the frame has one, else the
/// canonical `bond_type` / `bond_number` codes mapped back (`4` → `ar`), so a
/// frame read from MOL2 keeps its aromatic and double bonds without `type`.
pub fn write_mol2<P: AsRef<Path>>(path: P, frame: &Frame) -> Result<()> {
    let file = std::fs::File::create(path.as_ref())?;
    let mut w = BufWriter::new(file);
    write_frame_to(&mut w, frame)?;
    w.flush()
}

/// Write a frame as single-molecule MOL2 text — [`write_mol2`] into memory.
pub fn write_mol2_str(frame: &Frame) -> Result<String> {
    let mut buf = Vec::new();
    write_frame_to(&mut buf, frame)?;
    String::from_utf8(buf).map_err(invalid_data)
}

/// Emit one frame in MOL2 format.
fn write_frame_to<W: Write>(writer: &mut W, frame: &Frame) -> Result<()> {
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
    let charge_col = atoms.get("charge").and_then(|c| c.as_float());
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
    let id_col = atoms.get("id").and_then(|c| c.as_uint());
    let name_col = atoms.get("name").and_then(|c| c.as_string());
    let xs = atoms
        .get("x")
        .and_then(|c| c.as_float())
        .ok_or_else(|| invalid_data("atoms.x missing"))?;
    let ys = atoms
        .get("y")
        .and_then(|c| c.as_float())
        .ok_or_else(|| invalid_data("atoms.y missing"))?;
    let zs = atoms
        .get("z")
        .and_then(|c| c.as_float())
        .ok_or_else(|| invalid_data("atoms.z missing"))?;
    let type_col = atoms.get(keys::TYPE).and_then(|c| c.as_string());
    let subst_id_col = atoms.get(keys::RES_ID).and_then(|c| c.as_uint());
    let subst_name_col = atoms.get(keys::RES_NAME).and_then(|c| c.as_string());
    for i in 0..n {
        let id = id_col.map(|c| c[[i]]).unwrap_or((i as Idx) + 1);
        let name = name_col.map(|c| c[[i]].as_str()).unwrap_or("X");
        let atype = type_col.map(|c| c[[i]].as_str()).unwrap_or("X");
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
            .and_then(|c| c.as_uint())
            .ok_or_else(|| invalid_data("bonds.atomi missing"))?;
        let atomj = b
            .get("atomj")
            .and_then(|c| c.as_uint())
            .ok_or_else(|| invalid_data("bonds.atomj missing"))?;
        let btype_col = b.get(keys::TYPE).and_then(|c| c.as_string());
        let class_col = b.get(keys::BOND_TYPE).and_then(|c| c.as_uint());
        let number_col = b.get(keys::BOND_NUMBER).and_then(|c| c.as_uint());
        for i in 0..n_bonds {
            let bt = match btype_col {
                Some(c) => c[[i]].as_str(),
                None => sybyl_bond_token(class_col.map(|c| c[[i]]), number_col.map(|c| c[[i]]))?,
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
    Ok(())
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

impl<W: Write> Writer for Mol2Writer<W> {
    type W = W;
    fn new(writer: W) -> Self {
        Self { writer }
    }
}

impl<W: Write> FrameWriter for Mol2Writer<W> {
    fn write(&mut self, frame: &Frame) -> Result<()> {
        // Refuse to emit a frame that violates the vocabulary: a bad file
        // looks fine and is found wrong later, by whatever reads it.
        crate::io::writer::check_write_frame(frame)?;
        write_frame_to(&mut self.writer, frame)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    const ETHANE_MIN: &str = "@<TRIPOS>MOLECULE\nETH\n2 1 1 0 0\nSMALL\nNO_CHARGES\n@<TRIPOS>ATOM\n1 C1 0.0 0.0 0.0 c3 1 ETH 0.0\n2 C2 1.5 0.0 0.0 c3 1 ETH 0.0\n@<TRIPOS>BOND\n1 1 2 1\n";

    #[test]
    fn reads_minimal_mol2() {
        let mut reader = Mol2Reader::new(Cursor::new(ETHANE_MIN.as_bytes()));
        let frame = reader.read().unwrap().unwrap();
        let atoms = frame.get("atoms").unwrap();
        assert_eq!(atoms.n_rows(), Some(2));
        let xs = atoms.get("x").and_then(|c| c.as_float()).unwrap();
        assert!((xs[[1]] - 1.5).abs() < 1e-9);
        let bonds = frame.get("bonds").unwrap();
        assert_eq!(bonds.n_rows(), Some(1));
        let atomi = bonds.get("atomi").and_then(|c| c.as_uint()).unwrap();
        let atomj = bonds.get("atomj").and_then(|c| c.as_uint()).unwrap();
        assert_eq!(atomi[[0]], 0);
        assert_eq!(atomj[[0]], 1);
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
        assert_eq!(
            bonds.get("type").and_then(|c| c.as_string()).unwrap()[[0]],
            "1"
        );
        assert!(!bonds.contains_key("sybyl_bond_type"));
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
            .and_then(|c| c.as_uint())
            .unwrap()
            .iter()
            .copied()
            .collect();
        assert_eq!(res_id, [7, 9]);
        let types = atoms.get("type").and_then(|c| c.as_string()).unwrap();
        assert_eq!((types[[0]].as_str(), types[[1]].as_str()), ("C.ar", "N.am"));
        let res_name = atoms.get("res_name").and_then(|c| c.as_string()).unwrap();
        assert_eq!(res_name[[1]], "AMD");
        let btype = back
            .get("bonds")
            .unwrap()
            .get("type")
            .and_then(|c| c.as_string())
            .unwrap();
        assert_eq!(btype[[0]], "am");
    }

    #[test]
    fn bond_orders_round_trip_without_the_type_column() {
        const BENZALDEHYDE_ISH: &str = "@<TRIPOS>MOLECULE\nX\n4 3 1 0 0\nSMALL\nNO_CHARGES\n@<TRIPOS>ATOM\n1 C1 0.0 0.0 0.0 C.ar 1 X\n2 C2 1.4 0.0 0.0 C.ar 1 X\n3 C3 2.8 0.0 0.0 C.2 1 X\n4 O1 4.0 0.0 0.0 O.2 1 X\n@<TRIPOS>BOND\n1 1 2 ar\n2 2 3 1\n3 3 4 2\n";
        let mut frame = Mol2Reader::new(Cursor::new(BENZALDEHYDE_ISH.as_bytes()))
            .read()
            .unwrap()
            .unwrap();
        // What a caller does when its `type` column is a force-field label:
        // drop it, leaving the canonical columns the reader wrote.
        frame.get_mut("bonds").unwrap().remove(keys::TYPE);
        let text = write_mol2_str(&frame).unwrap();
        let back = Mol2Reader::new(Cursor::new(text.as_bytes()))
            .read()
            .unwrap()
            .unwrap();
        let types: Vec<String> = back
            .get("bonds")
            .unwrap()
            .get(keys::TYPE)
            .and_then(|c| c.as_string())
            .unwrap()
            .iter()
            .cloned()
            .collect();
        assert_eq!(types, ["ar", "1", "2"]);
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
            .and_then(|c| c.as_float())
            .unwrap();
        let xs2 = frame2
            .get("atoms")
            .unwrap()
            .get("x")
            .and_then(|c| c.as_float())
            .unwrap();
        for i in 0..xs1.len() {
            assert!((xs1[[i]] - xs2[[i]]).abs() < 1e-3);
        }
    }
}
