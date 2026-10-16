//! XSF (XCrySDen Structure File) structure reader and writer.

use crate::core::Cell;
use crate::io::frame_columns::insert_column;
use crate::io::{Sink, Source, invalid_data};
use std::io::{BufRead, Result, Write};

use molrs::core::Block;
use molrs::core::Frame;
use molrs::core::SimBox;
use molrs::core::keys;
use molrs::op::{F, Idx};

// ---------------------------------------------------------------------------
// Error helpers
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Reader
// ---------------------------------------------------------------------------

/// Read one XSF structure file from `path`.
///
/// Supports crystal and molecular structures:
///
/// - `CRYSTAL` / `MOLECULE` keywords
/// - `PRIMVEC` / `CONVVEC` — three lattice-vector lines (Å)
/// - `PRIMCOORD` — atom count / multiplicity, then `Z x y z` rows
///
/// The returned [`Frame`] has:
///
/// - `"atoms"` block: `atomic_number` (U, the raw Z the file spells; no
///   `element` column is synthesized — call `Element::fill_symbols` for one),
///   `x`/`y`/`z` (F)
/// - `frame.simbox` — periodic cell from PRIMVEC (else CONVVEC) for `CRYSTAL`,
///   or a free (no-cell) box for `MOLECULE` / unspecified
pub fn read_xsf<'a>(src: impl Into<Source<'a>>) -> Result<Frame> {
    read_frame_from(src.into().reader()?)
}

/// Read one XSF structure from a buffered reader.
fn read_frame_from<R: BufRead>(mut reader: R) -> Result<Frame> {
    let mut raw_lines = Vec::new();
    let mut buf = String::new();
    loop {
        buf.clear();
        let n = reader.read_line(&mut buf)?;
        if n == 0 {
            break;
        }
        let trimmed = buf.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        raw_lines.push(trimmed.to_string());
    }

    if raw_lines.is_empty() {
        return Err(invalid_data("Empty XSF file"));
    }

    let mut structure_type: Option<&str> = None;
    let mut primvec: Option<[[F; 3]; 3]> = None;
    let mut convvec: Option<[[F; 3]; 3]> = None;
    let mut atomic_numbers: Vec<Idx> = Vec::new();
    let mut xs: Vec<F> = Vec::new();
    let mut ys: Vec<F> = Vec::new();
    let mut zs: Vec<F> = Vec::new();

    let mut i = 0usize;
    while i < raw_lines.len() {
        let line = raw_lines[i].as_str();
        let upper = line.to_ascii_uppercase();
        match upper.as_str() {
            "CRYSTAL" => {
                structure_type = Some("CRYSTAL");
                i += 1;
            }
            "MOLECULE" => {
                structure_type = Some("MOLECULE");
                i += 1;
            }
            "PRIMVEC" => {
                primvec = Some(parse_vectors(&raw_lines[i + 1..])?);
                i += 4;
            }
            "CONVVEC" => {
                convvec = Some(parse_vectors(&raw_lines[i + 1..])?);
                i += 4;
            }
            "PRIMCOORD" => {
                if i + 1 >= raw_lines.len() {
                    return Err(invalid_data("PRIMCOORD section incomplete"));
                }
                let header: Vec<&str> = raw_lines[i + 1].split_whitespace().collect();
                if header.len() < 2 {
                    return Err(invalid_data("Invalid PRIMCOORD header"));
                }
                let n_atoms: usize = header[0].parse().map_err(|_| {
                    invalid_data(format!("Invalid PRIMCOORD header: {}", raw_lines[i + 1]))
                })?;
                // Multiplicity (header[1]) is accepted and ignored.
                let start = i + 2;
                let end = start + n_atoms;
                if end > raw_lines.len() {
                    return Err(invalid_data("PRIMCOORD section incomplete"));
                }
                parse_atoms(
                    &raw_lines[start..end],
                    &mut atomic_numbers,
                    &mut xs,
                    &mut ys,
                    &mut zs,
                )?;
                i = end;
            }
            _ => {
                i += 1;
            }
        }
    }

    let mut frame = Frame::new();

    if !atomic_numbers.is_empty() {
        let mut atoms = Block::new();
        insert_column(&mut atoms, keys::ATOMIC_NUMBER, atomic_numbers)?;
        insert_column(&mut atoms, "x", xs)?;
        insert_column(&mut atoms, "y", ys)?;
        insert_column(&mut atoms, "z", zs)?;
        frame.insert("atoms", atoms);
    }

    frame.simbox = Some(match structure_type {
        Some("CRYSTAL") => {
            let mat =
                primvec
                    .or(convvec)
                    .unwrap_or([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
            // PRIMVEC lines are lattice vectors; H columns are lattice vectors.
            let h = std::array::from_fn(|r| std::array::from_fn(|c| mat[c][r]));
            SimBox::new(Cell::Matrix(h), [0.0; 3], [true, true, true])
                .map_err(|e| invalid_data(format!("{:?}", e)))?
        }
        _ => free_simbox()?,
    });

    Ok(frame)
}

fn free_simbox() -> Result<SimBox> {
    SimBox::new(Cell::Undefined, [0.0; 3], [false, false, false])
        .map_err(|e| invalid_data(format!("{:?}", e)))
}

/// Parse three lattice-vector lines (row-per-vector in the file).
fn parse_vectors(lines: &[String]) -> Result<[[F; 3]; 3]> {
    if lines.len() < 3 {
        return Err(invalid_data("Incomplete vector specification"));
    }
    let mut matrix = [[0.0 as F; 3]; 3];
    for (i, line) in lines.iter().take(3).enumerate() {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 3 {
            return Err(invalid_data(format!("Invalid vector line: {line}")));
        }
        for j in 0..3 {
            matrix[i][j] = parts[j]
                .parse::<F>()
                .map_err(|_| invalid_data(format!("Invalid vector coordinates: {line}")))?;
        }
    }
    Ok(matrix)
}

fn parse_atoms(
    lines: &[String],
    atomic_numbers: &mut Vec<Idx>,
    xs: &mut Vec<F>,
    ys: &mut Vec<F>,
    zs: &mut Vec<F>,
) -> Result<()> {
    for line in lines {
        if line.trim().is_empty() {
            continue;
        }
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 4 {
            return Err(invalid_data(format!("Invalid atom line: {line}")));
        }
        // The file spells species as Z: keep it raw (0 = wildcard) and refuse
        // anything outside 0..=118 rather than invent a species.
        let z_u: Idx = parts[0]
            .parse::<u8>()
            .ok()
            .filter(|&z| z <= 118)
            .map(Idx::from)
            .ok_or_else(|| {
                invalid_data(format!(
                    "Invalid atom data: atomic number '{}' is not in 0..=118: {line}",
                    parts[0]
                ))
            })?;
        let x: F = parts[1]
            .parse()
            .map_err(|_| invalid_data(format!("Invalid atom data: {line}")))?;
        let y: F = parts[2]
            .parse()
            .map_err(|_| invalid_data(format!("Invalid atom data: {line}")))?;
        let zc: F = parts[3]
            .parse()
            .map_err(|_| invalid_data(format!("Invalid atom data: {line}")))?;

        atomic_numbers.push(z_u);
        xs.push(x);
        ys.push(y);
        zs.push(zc);
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Writer
// ---------------------------------------------------------------------------

/// Write a frame as an XSF file.
pub fn write_xsf<'a>(sink: impl Into<Sink<'a>>, frame: &Frame) -> Result<()> {
    let mut w = sink.into().writer()?;
    write_frame_to(&mut w, frame)?;
    w.flush()
}

/// Emit one Frame in XSF format.
fn write_frame_to<W: Write>(writer: &mut W, frame: &Frame) -> Result<()> {
    writeln!(writer, "# XSF file generated by molpy")?;

    let has_crystal = frame
        .simbox
        .as_ref()
        .is_some_and(|sb| sb.is_cell_defined() && !sb.is_free());

    if has_crystal {
        let sb = frame.simbox.as_ref().unwrap();
        let h = ndarray::arr2(&sb.matrix());
        writeln!(writer, "CRYSTAL")?;
        writeln!(writer, "PRIMVEC")?;
        for i in 0..3 {
            // Lattice vector i = column i of H → written as one PRIMVEC row.
            writeln!(
                writer,
                "    {:12.8}    {:12.8}    {:12.8}",
                h[[0, i]],
                h[[1, i]],
                h[[2, i]]
            )?;
        }
        writeln!(writer, "CONVVEC")?;
        for i in 0..3 {
            writeln!(
                writer,
                "    {:12.8}    {:12.8}    {:12.8}",
                h[[0, i]],
                h[[1, i]],
                h[[2, i]]
            )?;
        }
    } else {
        writeln!(writer, "MOLECULE")?;
    }

    if let Some(atoms) = frame.get("atoms") {
        let n = atoms.n_rows().unwrap_or(0);
        let xs = atoms
            .get("x")
            .and_then(|c| c.get::<f64>())
            .ok_or_else(|| invalid_data("XSF write: atoms.x missing"))?;
        let ys = atoms
            .get("y")
            .and_then(|c| c.get::<f64>())
            .ok_or_else(|| invalid_data("XSF write: atoms.y missing"))?;
        let zs = atoms
            .get("z")
            .and_then(|c| c.get::<f64>())
            .ok_or_else(|| invalid_data("XSF write: atoms.z missing"))?;
        // One Z dtype, the schema's: unsigned. A symbol-only frame converts
        // first with `Element::fill_atomic_numbers`.
        let z_u = atoms
            .get(keys::ATOMIC_NUMBER)
            .and_then(|c| c.get::<u64>())
            .ok_or_else(|| {
                invalid_data(format!(
                    "XSF write: atoms block has no unsigned '{}' column; a frame that \
                     spells species as symbols needs `Element::fill_atomic_numbers(&mut frame)` \
                     first",
                    keys::ATOMIC_NUMBER
                ))
            })?;

        writeln!(writer, "PRIMCOORD")?;
        writeln!(writer, "       {n} 1")?;
        for idx in 0..n {
            let an = z_u[[idx]];
            if an > 118 {
                return Err(invalid_data(format!(
                    "XSF write: atoms row {idx}: {an} is not an atomic number (0..=118)"
                )));
            }
            writeln!(
                writer,
                "{:2}    {:12.8}    {:12.8}    {:12.8}",
                an,
                xs[[idx]],
                ys[[idx]],
                zs[[idx]]
            )?;
        }
    } else {
        writeln!(writer, "PRIMCOORD")?;
        writeln!(writer, "        0 1")?;
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn read_crystal_orthogonal() {
        let text = "\
CRYSTAL
PRIMVEC
3.0 0.0 0.0
0.0 3.0 0.0
0.0 0.0 3.0
PRIMCOORD
2 1
1  0.0  0.0  0.0
8  1.5  1.5  1.5
";
        let frame = read_frame_from(Cursor::new(text)).expect("parse");
        let atoms = frame.get("atoms").expect("atoms");
        assert_eq!(atoms.n_rows(), Some(2));
        let z = atoms
            .get("atomic_number")
            .and_then(|c| c.get::<u64>())
            .unwrap();
        assert_eq!(z[[0]], 1);
        assert_eq!(z[[1]], 8);
        let sb = frame.simbox.as_ref().unwrap();
        assert!(sb.is_cell_defined());
        assert!(!sb.is_free());
        assert_eq!(sb.style(), "orthogonal");
    }

    #[test]
    fn read_molecule_free() {
        let text = "\
MOLECULE
PRIMCOORD
2 1
1  0.0  0.0  0.0
1  1.0  0.0  0.0
";
        let frame = read_frame_from(Cursor::new(text)).expect("parse");
        let atoms = frame.get("atoms").expect("atoms");
        assert_eq!(atoms.n_rows(), Some(2));
        let sb = frame.simbox.as_ref().unwrap();
        assert!(!sb.is_cell_defined());
        assert!(sb.is_free());
        assert_eq!(sb.style(), "free");
    }

    #[test]
    fn empty_file_errors() {
        let err = read_frame_from(Cursor::new("")).unwrap_err();
        assert!(err.to_string().contains("Empty XSF file"));
    }

    #[test]
    fn roundtrip_crystal() {
        let text = "\
CRYSTAL
PRIMVEC
5.0 0.0 0.0
0.0 5.0 0.0
0.0 0.0 5.0
PRIMCOORD
2 1
6  0.0  0.0  0.0
1  1.0  0.0  0.0
";
        let frame = read_frame_from(Cursor::new(text)).unwrap();
        let mut out = Vec::new();
        write_frame_to(&mut out, &frame).unwrap();
        let back = read_frame_from(Cursor::new(out)).unwrap();
        let z0 = frame
            .get("atoms")
            .unwrap()
            .get("atomic_number")
            .and_then(|c| c.get::<u64>())
            .unwrap();
        let z1 = back
            .get("atoms")
            .unwrap()
            .get("atomic_number")
            .and_then(|c| c.get::<u64>())
            .unwrap();
        assert_eq!(z0[[0]], z1[[0]]);
        assert_eq!(z0[[1]], z1[[1]]);
    }

    #[test]
    fn reader_keeps_raw_z_and_synthesizes_no_element() {
        let text = "MOLECULE\nPRIMCOORD\n1 1\n0  0.0  0.0  0.0\n";
        let frame = read_frame_from(Cursor::new(text)).unwrap();
        let atoms = frame.get("atoms").unwrap();
        assert_eq!(
            atoms
                .get(keys::ATOMIC_NUMBER)
                .and_then(|c| c.get::<u64>())
                .unwrap()[[0]],
            0
        );
        assert!(atoms.get(keys::ELEMENT).is_none());
    }

    #[test]
    fn reader_rejects_out_of_range_z() {
        for bad in ["119", "-1", "Fe"] {
            let text = format!("MOLECULE\nPRIMCOORD\n1 1\n{bad}  0.0  0.0  0.0\n");
            let err = read_frame_from(Cursor::new(text)).unwrap_err();
            assert_eq!(err.kind(), std::io::ErrorKind::InvalidData, "{bad}");
        }
    }

    #[test]
    fn writer_requires_atomic_number() {
        let mut frame = Frame::new();
        let mut atoms = Block::new();
        insert_column(&mut atoms, keys::ELEMENT, vec!["O".to_string()]).unwrap();
        for k in ["x", "y", "z"] {
            insert_column(&mut atoms, k, vec![0.0 as F]).unwrap();
        }
        frame.insert("atoms", atoms);
        let err = write_xsf(&mut Vec::new(), &frame).unwrap_err();
        assert!(err.to_string().contains("fill_atomic_numbers"), "{err}");
    }

    #[test]
    fn malformed_primcoord_header() {
        let text = "MOLECULE\nPRIMCOORD\ninvalid_number 1\n";
        let err = read_frame_from(Cursor::new(text)).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
    }
}
