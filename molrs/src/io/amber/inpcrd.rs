//! AMBER ASCII inpcrd / restrt coordinate reader.

use crate::io::frame_columns::insert_column;
use crate::io::invalid_data;
use std::io::{BufRead, Result};
use std::path::Path;

use ndarray::{Array2, array};

use molrs::core::Block;
use molrs::core::Frame;
use molrs::core::SimBox;
use molrs::op::{F, Idx};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Parse a line with fixed-width columns (default width 12).
///
/// AMBER writes Fortran ``6F12.7``; a negative sign may abut the previous field
/// (e.g. ``50.5413286-100.7101036``), so whitespace splitting is wrong.
fn parse_fixed_width(raw: &str, width: usize) -> Result<Vec<F>> {
    let mut out = Vec::new();
    let mut i = 0;
    let bytes = raw.as_bytes();
    while i < bytes.len() {
        let end = (i + width).min(bytes.len());
        let token = std::str::from_utf8(&bytes[i..end])
            .map_err(|_| invalid_data("non-utf8 in fixed-width field"))?
            .trim();
        if !token.is_empty() {
            let v: F = token
                .parse()
                .map_err(|_| invalid_data(format!("bad float in fixed-width field: {token:?}")))?;
            out.push(v);
        }
        i += width;
    }
    Ok(out)
}

/// Parse floats from one coordinate/velocity/box line.
///
/// Prefer fixed-width 12-char fields; fall back to whitespace split if that fails.
fn parse_floats_line(raw: &str) -> Result<Vec<F>> {
    match parse_fixed_width(raw, 12) {
        Ok(v) if !v.is_empty() || raw.trim().is_empty() => Ok(v),
        Ok(_) | Err(_) => {
            let mut out = Vec::new();
            for tok in raw.split_whitespace() {
                let v: F = tok
                    .parse()
                    .map_err(|_| invalid_data(format!("bad float token: {tok:?}")))?;
                out.push(v);
            }
            Ok(out)
        }
    }
}

/// Collect floats from a slice of lines into an `[n_atoms × 3]` flat vector.
fn eat_section(lines: &[String], n_atoms: usize) -> Result<Vec<F>> {
    let need = n_atoms * 3;
    let mut data = Vec::with_capacity(need);
    for line in lines {
        let raw = line.trim_end_matches(['\n', '\r']);
        if raw.trim().is_empty() {
            continue;
        }
        data.extend(parse_floats_line(raw)?);
    }
    if data.len() < need {
        return Err(invalid_data(format!(
            "expected {need} floats for {n_atoms} atoms, got {}",
            data.len()
        )));
    }
    data.truncate(need);
    Ok(data)
}

// ---------------------------------------------------------------------------
// Public reader API
// ---------------------------------------------------------------------------

/// Read an AMBER ASCII inpcrd / restrt file at `path`.
pub fn read_inpcrd<P: AsRef<Path>>(path: P) -> Result<Frame> {
    let file = std::fs::File::open(path.as_ref())?;
    read_frame_from(std::io::BufReader::new(file))
}

/// Merge the coordinates an AMBER inpcrd / restrt holds — `coordinates`, as
/// [`read_inpcrd`](crate::io::read_inpcrd) read it — into an
/// existing `frame`: the structure's other half, as a prmtop pairs with its
/// inpcrd.
///
/// The file's `x` / `y` / `z` (and `vel`, when it has velocities) replace
/// those columns of `frame["atoms"]`; every other atom column stays. The
/// file's box replaces the frame's (a box-less file clears it), and the
/// file's meta keys (`title`, `timestep`) are set, keeping every other meta
/// key. A frame with no `atoms` block receives the file's whole `atoms`
/// block.
///
/// # Errors
///
/// [`std::io::ErrorKind::InvalidData`] when `frame["atoms"]` has a different
/// row count than the file has atoms. `frame` is unchanged on error.
pub fn merge_inpcrd(frame: &mut Frame, coordinates: Frame) -> Result<()> {
    let mut loaded = coordinates;
    let source = loaded
        .remove("atoms")
        .ok_or_else(|| invalid_data("inpcrd produced no atoms block"))?;
    match frame.get_mut("atoms") {
        None => {
            frame.insert("atoms", source);
        }
        Some(atoms) => {
            if atoms.n_rows() != source.n_rows() {
                return Err(invalid_data(format!(
                    "atoms block has {} rows, but the inpcrd has {} atoms",
                    atoms.n_rows().unwrap_or(0),
                    source.n_rows().unwrap_or(0)
                )));
            }
            for key in ["x", "y", "z", "vel"] {
                if let Some(col) = source.get(key) {
                    atoms
                        .insert_column(key, col.clone())
                        .map_err(invalid_data)?;
                }
            }
        }
    }
    frame.simbox = loaded.simbox.take();
    frame
        .meta
        .extend(loaded.meta.iter().map(|(k, v)| (k.clone(), v.clone())));
    Ok(())
}

/// Read an AMBER ASCII inpcrd / restrt from any [`BufRead`].
fn read_frame_from<R: BufRead>(mut reader: R) -> Result<Frame> {
    let mut raw_lines: Vec<String> = Vec::new();
    let mut buf = String::new();
    loop {
        buf.clear();
        let n = reader.read_line(&mut buf)?;
        if n == 0 {
            break;
        }
        // Keep original line content (minus only the trailing newline) so
        // fixed-width columns that include leading spaces stay intact.
        let line = buf.trim_end_matches(['\n', '\r']).to_string();
        raw_lines.push(line);
    }

    if raw_lines.len() < 2 {
        return Err(invalid_data("inpcrd too short"));
    }

    let title = raw_lines[0].trim().to_string();
    let header_tokens: Vec<&str> = raw_lines[1].split_whitespace().collect();
    if header_tokens.is_empty() {
        return Err(invalid_data("inpcrd header missing atom count"));
    }
    let n_atoms: usize = header_tokens[0]
        .parse()
        .map_err(|_| invalid_data(format!("bad atom count: {:?}", header_tokens[0])))?;
    let time: Option<F> = if header_tokens.len() > 1 {
        Some(
            header_tokens[1]
                .parse()
                .map_err(|_| invalid_data(format!("bad time token: {:?}", header_tokens[1])))?,
        )
    } else {
        None
    };

    // AMBER: 6 values per line → ceil(n_atoms * 3 / 6) coordinate lines.
    let n_coord_lines = n_atoms.saturating_mul(3).saturating_add(5) / 6;
    if raw_lines.len() < 2 + n_coord_lines {
        return Err(invalid_data(format!(
            "Not enough lines for {n_atoms} atoms: need {n_coord_lines} coordinate lines, got {}",
            raw_lines.len().saturating_sub(2)
        )));
    }

    let mut cursor = 2 + n_coord_lines;
    let coords_flat = eat_section(&raw_lines[2..cursor], n_atoms)?;

    // Velocities only appear in restart files (timestamp on the header line).
    let mut velocity_flat: Option<Vec<F>> = None;
    if time.is_some() {
        let non_blank_remaining = raw_lines[cursor..]
            .iter()
            .filter(|l| !l.trim().is_empty())
            .count();
        if non_blank_remaining >= n_coord_lines {
            let maybe = eat_section(&raw_lines[cursor..cursor + n_coord_lines], n_atoms)?;
            if maybe.len() == n_atoms * 3 {
                velocity_flat = Some(maybe);
                cursor += n_coord_lines;
            }
        }
    }

    // Optional box line (3–6 floats; first three → diagonal lengths).
    let mut box_lengths: Option<[F; 3]> = None;
    if cursor < raw_lines.len() {
        let raw = raw_lines[cursor].as_str();
        if !raw.trim().is_empty() {
            let box_floats = parse_floats_line(raw)?;
            if box_floats.len() >= 3 {
                box_lengths = Some([box_floats[0], box_floats[1], box_floats[2]]);
            }
        }
    }

    // ---- Build Frame -------------------------------------------------------
    let mut block = Block::new();
    let mut ids = Vec::with_capacity(n_atoms);
    let mut names = Vec::with_capacity(n_atoms);
    let mut xs = Vec::with_capacity(n_atoms);
    let mut ys = Vec::with_capacity(n_atoms);
    let mut zs = Vec::with_capacity(n_atoms);
    for i in 0..n_atoms {
        ids.push((i as Idx) + 1);
        names.push(format!("ATM{}", i + 1));
        xs.push(coords_flat[i * 3]);
        ys.push(coords_flat[i * 3 + 1]);
        zs.push(coords_flat[i * 3 + 2]);
    }
    insert_column(&mut block, "id", ids)?;
    insert_column(&mut block, "name", names)?;
    insert_column(&mut block, "x", xs)?;
    insert_column(&mut block, "y", ys)?;
    insert_column(&mut block, "z", zs)?;

    if let Some(vel) = velocity_flat {
        let arr = Array2::from_shape_vec((n_atoms, 3), vel)
            .map_err(invalid_data)?
            .into_dyn();
        block.insert("vel", arr).map_err(invalid_data)?;
    }

    let mut frame = Frame::new();
    frame.meta.insert("title", title);
    if let Some(t) = time {
        frame.meta.insert("timestep", t as i64);
    }
    frame.insert("atoms", block);

    if let Some(lens) = box_lengths {
        let origin = array![0.0 as F, 0.0, 0.0];
        let simbox = SimBox::ortho(array![lens[0], lens[1], lens[2]], origin, [true; 3])
            .map_err(|e| invalid_data(format!("{:?}", e)))?;
        frame.simbox = Some(simbox);
    }

    Ok(frame)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// Read an AMBER ASCII inpcrd / restrt from its text.
///
/// The layout is fixed-width Fortran ``6F12.7`` (old-style ``*.inpcrd`` /
/// restart):
///
/// ```text
/// line 1   : title
/// line 2   : natom [time]
/// next     : coordinates — 6 floats per line, 12 chars each
/// optional : velocities (same layout; only when `time` is present on line 2)
/// optional : box line — 3–6 floats (first three → orthorhombic diagonal)
/// ```
///
/// # Output Frame
///
/// - `"atoms"` block: `id` (uint, 1-based), `name` (str, `"ATM{i}"`),
///   `x`/`y`/`z` (F), optional `vel` (F, shape `[n, 3]`)
/// - `frame.meta["title"]` (string); optional `frame.meta["timestep"]` (i64,
///   truncates the header time)
/// - `frame.simbox`: orthorhombic cell from the first three box floats when present
pub fn read_inpcrd_str(text: &str) -> Result<Frame> {
    read_frame_from(text.as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;
    use std::io::Cursor;

    #[test]
    fn into_replaces_coordinates_and_keeps_the_rest() {
        let path =
            std::env::temp_dir().join(format!("molrs-inpcrd-into-{}.inpcrd", std::process::id()));
        std::fs::write(
            &path,
            "two atoms\n  2\n  1.0000000   2.0000000   3.0000000   4.0000000   5.0000000   6.0000000\n  10.0000000  11.0000000  12.0000000  90.0000000  90.0000000  90.0000000\n",
        )
        .unwrap();
        let mut frame = Frame::new();
        let mut atoms = Block::new();
        atoms
            .insert("x", Array1::from_vec(vec![0.0 as F, 0.0]).into_dyn())
            .unwrap();
        atoms
            .insert("charge", Array1::from_vec(vec![0.5 as F, -0.5]).into_dyn())
            .unwrap();
        frame.insert("atoms", atoms);
        frame.meta.insert("source", "prmtop");
        merge_inpcrd(&mut frame, read_inpcrd(&path).unwrap()).unwrap();

        let atoms = frame.get("atoms").unwrap();
        let z = atoms.get("z").and_then(|c| c.as_float()).unwrap();
        assert_eq!(z.iter().copied().collect::<Vec<_>>(), [3.0, 6.0]);
        assert!(atoms.contains_key("charge"));
        assert!(frame.simbox.is_some());
        assert_eq!(
            frame.meta.get("title").and_then(|v| v.as_str()),
            Some("two atoms")
        );
        assert_eq!(
            frame.meta.get("source").and_then(|v| v.as_str()),
            Some("prmtop")
        );

        let mut three = Frame::new();
        let mut atoms = Block::new();
        atoms
            .insert("x", Array1::from_vec(vec![0.0 as F; 3]).into_dyn())
            .unwrap();
        three.insert("atoms", atoms);
        let err = merge_inpcrd(&mut three, read_inpcrd(&path).unwrap()).unwrap_err();
        let _ = std::fs::remove_file(&path);
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
        assert!(
            !three.get("atoms").unwrap().contains_key("y"),
            "unchanged on error"
        );
    }

    fn frame_from(s: &str) -> Frame {
        read_frame_from(Cursor::new(s.as_bytes())).expect("parse")
    }

    #[test]
    fn basic_coords_only() {
        let text = "\
Simple 3-atom system
  3
  0.0000000   1.0000000   2.0000000   3.0000000   4.0000000   5.0000000
  6.0000000   7.0000000   8.0000000
";
        let frame = frame_from(text);
        let atoms = frame.get("atoms").unwrap();
        assert_eq!(atoms.n_rows(), Some(3));
        assert_eq!(
            frame.meta.get("title").and_then(|v| v.as_str()),
            Some("Simple 3-atom system")
        );
        let x = atoms.get("x").and_then(|c| c.as_float()).unwrap();
        let y = atoms.get("y").and_then(|c| c.as_float()).unwrap();
        let z = atoms.get("z").and_then(|c| c.as_float()).unwrap();
        assert!((x[[0]] - 0.0).abs() < 1e-9);
        assert!((y[[1]] - 4.0).abs() < 1e-9);
        assert!((z[[2]] - 8.0).abs() < 1e-9);
        assert!(frame.simbox.is_none());
    }

    #[test]
    fn with_time() {
        let text = "\
Test with time
  2   100.5
  1.0         2.0         3.0         4.0         5.0         6.0
";
        let frame = frame_from(text);
        assert_eq!(
            frame.meta.get("timestep").and_then(|v| v.as_i64()),
            Some(100)
        );
    }

    #[test]
    fn with_velocities() {
        let text = "\
Test with velocities
  2   25.0
  1.0         2.0         3.0         4.0         5.0         6.0
  0.1         0.2         0.3         0.4         0.5         0.6
";
        let frame = frame_from(text);
        let atoms = frame.get("atoms").unwrap();
        let vel = atoms
            .get("vel")
            .and_then(|c| c.as_float())
            .expect("vel column");
        assert_eq!(vel.shape(), &[2, 3]);
        assert!((vel[[0, 0]] - 0.1).abs() < 1e-9);
        assert!((vel[[1, 2]] - 0.6).abs() < 1e-9);
    }

    #[test]
    fn with_box() {
        let text = "\
Test with box
  2
  1.0         2.0         3.0         4.0         5.0         6.0
 10.0        20.0        30.0        90.0        90.0        90.0
";
        let frame = frame_from(text);
        let sb = frame.simbox.as_ref().expect("box");
        let lens = sb.lengths();
        assert!((lens[0] - 10.0).abs() < 1e-9);
        assert!((lens[1] - 20.0).abs() < 1e-9);
        assert!((lens[2] - 30.0).abs() < 1e-9);
    }

    #[test]
    fn abutting_negatives() {
        // Fortran 6F12.7 can place a minus flush against the previous field.
        let abutting = "  50.5413286-100.7101036  12.3456789 -44.5678901  88.8888888  -0.1234567";
        let text = format!("Abutting negatives\n  2\n{abutting}\n");
        let frame = frame_from(&text);
        let atoms = frame.get("atoms").unwrap();
        let x = atoms.get("x").and_then(|c| c.as_float()).unwrap();
        let y = atoms.get("y").and_then(|c| c.as_float()).unwrap();
        let z = atoms.get("z").and_then(|c| c.as_float()).unwrap();
        assert!((x[[0]] - 50.5413286).abs() < 1e-6);
        assert!((y[[0]] + 100.7101036).abs() < 1e-6);
        assert!((z[[0]] - 12.3456789).abs() < 1e-6);
        assert!((x[[1]] + 44.5678901).abs() < 1e-6);
    }

    #[test]
    fn too_short() {
        let err = read_frame_from(Cursor::new(b"Only title\n")).unwrap_err();
        assert!(err.to_string().contains("too short"));
    }

    #[test]
    fn insufficient_coords() {
        let text = "\
Insufficient data
  5
  1.0         2.0         3.0         4.0         5.0         6.0
";
        let err = read_frame_from(Cursor::new(text.as_bytes())).unwrap_err();
        assert!(err.to_string().contains("Not enough lines"));
    }
}
