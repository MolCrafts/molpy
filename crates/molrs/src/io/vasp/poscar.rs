//! VASP POSCAR / CONTCAR structure file reader and writer.

use crate::io::frame_columns::insert_column;
use crate::io::{Sink, Source, invalid_data};
use std::io::{BufRead, Result, Write};

use ndarray::{Array1, IxDyn};

use molrs::core::Block;
use molrs::core::Frame;
use molrs::core::keys;
use molrs::op::F;

use super::header::{AtomRow, CoordMode, expand_symbols, parse_atom_row, read_coords, read_header};
use crate::io::reader::{FrameReader, Reader};
use crate::io::writer::{FrameWriter, Writer};

// ---------------------------------------------------------------------------
// Reader
// ---------------------------------------------------------------------------

/// Read one POSCAR file from `path`.
///
/// POSCAR is the VASP input format describing a crystalline cell and the atoms
/// within it. Format outline:
///
/// ```text
/// line 1   : comment / system name              → frame.meta["title"]
/// line 2   : global scale factor (Å)
/// line 3-5 : lattice vectors (row-per-line, Å after scaling)
/// line 6   : element symbols  (VASP5 only — omitted in VASP4)
/// line 7   : atom counts per element
/// line 8?  : "Selective dynamics" (optional)
/// line N   : "Direct" or "Cartesian"
/// line N+1+: atom coordinates (3 floats; + T/F flags if selective dynamics)
/// [opt]    : blank line then "Cartesian"/"Direct" + N velocity rows
/// ```
///
/// The returned [`Frame`] contains:
///
/// - `"atoms"` block:
///   - `x`, `y`, `z` — Cartesian Å (`Direct` files are converted on read).
///   - `element` — the species symbol as the file spells it (omitted when the
///     file did not declare them, VASP4). No `atomic_number` is synthesized:
///     call `Element::fill_atomic_numbers` for one.
///   - `sd_x`, `sd_y`, `sd_z` — selective-dynamics flags, if present.
///   - `vx`, `vy`, `vz` — atomic velocities, if present.
/// - `frame.simbox` — periodic [`SimBox`](molrs::core::SimBox) from the lattice vectors.
/// - `frame.meta` — `title`, plus `poscar_mode = "direct" | "cartesian"`.
pub fn read_poscar<'a>(src: impl Into<Source<'a>>) -> Result<Frame> {
    read_frame_from(src.into().reader()?)
}

/// Read one POSCAR-format frame from `reader`.
fn read_frame_from<R: BufRead>(mut reader: R) -> Result<Frame> {
    let mut line_no = 0usize;
    let header = read_header(&mut reader, &mut line_no)?;
    let n = header.total_atoms();
    if n == 0 {
        return Err(invalid_data("POSCAR declares zero atoms"));
    }

    let (raw_x, raw_y, raw_z, sd_flags) =
        read_coords(&mut reader, n, &mut line_no, header.selective_dynamics)?;

    let simbox = header.simbox()?;
    let (cart_x, cart_y, cart_z) = header.cartesian(&simbox, &raw_x, &raw_y, &raw_z);

    // Optional velocity block after a blank line. Best-effort: silently stop
    // if the file does not contain one or stops early.
    let velocities = try_read_velocities(&mut reader, n, &mut line_no);

    // ---------------------------------------------------------------------
    // Assemble Frame
    // ---------------------------------------------------------------------
    let mut atoms = Block::new();
    insert_column(&mut atoms, "x", cart_x)?;
    insert_column(&mut atoms, "y", cart_y)?;
    insert_column(&mut atoms, "z", cart_z)?;

    let symbols = expand_symbols(&header.symbols, &header.counts);
    if !symbols.is_empty() {
        let arr = Array1::from_vec(symbols)
            .into_shape_with_order(IxDyn(&[n]))
            .map_err(invalid_data)?
            .into_dyn();
        atoms
            .insert(keys::ELEMENT, arr, None)
            .map_err(invalid_data)?;
    }

    if let Some(flags) = sd_flags {
        let (sx, sy, sz): (Vec<bool>, Vec<bool>, Vec<bool>) = flags.iter().fold(
            (
                Vec::with_capacity(n),
                Vec::with_capacity(n),
                Vec::with_capacity(n),
            ),
            |(mut x, mut y, mut z), f| {
                x.push(f[0]);
                y.push(f[1]);
                z.push(f[2]);
                (x, y, z)
            },
        );
        insert_column(&mut atoms, "sd_x", sx)?;
        insert_column(&mut atoms, "sd_y", sy)?;
        insert_column(&mut atoms, "sd_z", sz)?;
    }

    if let Some((vx, vy, vz)) = velocities {
        insert_column(&mut atoms, "vx", vx)?;
        insert_column(&mut atoms, "vy", vy)?;
        insert_column(&mut atoms, "vz", vz)?;
    }

    let mut frame = Frame::new();
    if !header.title.is_empty() {
        frame.meta.insert("title", header.title);
    }
    frame.meta.insert(
        "poscar_mode",
        match header.mode {
            CoordMode::Direct => "direct",
            CoordMode::Cartesian => "cartesian",
        },
    );

    frame.simbox = Some(simbox);

    frame.insert("atoms", atoms);
    // Every door reads through here, so every door checks here.
    crate::io::reader::check_read_frame(&frame)?;
    Ok(frame)
}

/// Best-effort velocity read. Returns `None` if no velocity block present.
fn try_read_velocities<R: BufRead>(
    reader: &mut R,
    n: usize,
    line_no: &mut usize,
) -> Option<(Vec<F>, Vec<F>, Vec<F>)> {
    // Consume any blank line(s).
    let mut buf = String::new();
    loop {
        buf.clear();
        let bytes = reader.read_line(&mut buf).ok()?;
        if bytes == 0 {
            return None;
        }
        *line_no += 1;
        if !buf.trim().is_empty() {
            break;
        }
    }
    // `buf` now holds the first non-blank line. It is either a velocity row
    // (3 floats) or a coord-mode keyword for the velocity block.
    let trimmed = buf.trim();
    let first_char = trimmed.chars().next()?;
    let first_row_is_keyword = matches!(first_char, 'D' | 'd' | 'C' | 'c' | 'K' | 'k');

    let mut vx = Vec::with_capacity(n);
    let mut vy = Vec::with_capacity(n);
    let mut vz = Vec::with_capacity(n);

    if !first_row_is_keyword {
        let row = parse_atom_row(&buf, *line_no, false).ok()?;
        let AtomRow { x, y, z, .. } = row;
        vx.push(x);
        vy.push(y);
        vz.push(z);
    }

    while vx.len() < n {
        buf.clear();
        let bytes = reader.read_line(&mut buf).ok()?;
        if bytes == 0 {
            return None;
        }
        *line_no += 1;
        if buf.trim().is_empty() {
            continue;
        }
        let row = parse_atom_row(&buf, *line_no, false).ok()?;
        vx.push(row.x);
        vy.push(row.y);
        vz.push(row.z);
    }
    Some((vx, vy, vz))
}

/// `FrameReader`-trait wrapper for `read_frame_from`.
///
/// POSCAR holds at most one frame. The first `read_frame` call returns it;
/// subsequent calls return `Ok(None)`.
pub struct VaspPoscarReader<R: BufRead> {
    reader: R,
    consumed: bool,
}

impl<R: BufRead> Reader for VaspPoscarReader<R> {
    type R = R;
    fn new(reader: R) -> Self {
        Self {
            reader,
            consumed: false,
        }
    }
}

impl<R: BufRead> FrameReader for VaspPoscarReader<R> {
    fn read(&mut self) -> Result<Option<Frame>> {
        if self.consumed {
            return Ok(None);
        }
        self.consumed = true;
        read_frame_from(&mut self.reader).map(Some)
    }
}

// ---------------------------------------------------------------------------
// Writer
// ---------------------------------------------------------------------------

/// Write a frame as a POSCAR file.
pub fn write_poscar<'a>(sink: impl Into<Sink<'a>>, frame: &Frame) -> Result<()> {
    sink.into().put_with(|w| write_frame_to(w, frame))
}

/// Write a Frame in POSCAR format to any writer.
///
/// The output mode (Direct / Cartesian) follows `frame.meta["poscar_mode"]`,
/// defaulting to Cartesian when absent.
fn write_frame_to<W: Write>(writer: &mut W, frame: &Frame) -> Result<()> {
    // Every door writes through here, so every door checks here.
    crate::io::writer::check_write_frame(frame)?;
    crate::io::null_cells::refuse_null_cells(frame, "atoms", &["x", "y", "z", keys::ELEMENT])?;
    let atoms = frame
        .get("atoms")
        .ok_or_else(|| invalid_data("POSCAR write: frame has no atoms block"))?;
    let n = atoms.n_rows().unwrap_or(0);
    if n == 0 {
        return Err(invalid_data("POSCAR write: atoms block is empty"));
    }
    let simbox = frame
        .simbox
        .as_ref()
        .ok_or_else(|| invalid_data("POSCAR write: frame has no SimBox"))?;

    let title = frame
        .meta
        .get("title")
        .and_then(|value| value.as_str())
        .unwrap_or("molrs POSCAR");
    writeln!(writer, "{}", title)?;
    writeln!(writer, "1.0")?;

    // Lattice rows: h columns are lattice vectors → row i = (h[0,i], h[1,i], h[2,i]).
    let h = ndarray::arr2(&simbox.matrix());
    for i in 0..3 {
        writeln!(
            writer,
            "  {:.10}  {:.10}  {:.10}",
            h[(0, i)],
            h[(1, i)],
            h[(2, i)]
        )?;
    }

    // Group atoms by `element` so we can emit POSCAR's per-element runs. The
    // species line is optional in POSCAR (VASP4 takes species from POTCAR),
    // so a frame without `element` writes the VASP4 form — no species line,
    // not a made-up one.
    let symbol_col = atoms.get(keys::ELEMENT).and_then(|c| c.get::<String>());
    let SpeciesRuns {
        symbols: symbols_line,
        counts: counts_line,
        order,
    } = group_by_symbol(n, symbol_col)?;

    if let Some(syms) = &symbols_line {
        writeln!(writer, "{}", syms.join(" "))?;
    }
    writeln!(
        writer,
        "{}",
        counts_line
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(" ")
    )?;

    // Mode: Cartesian (default) or Direct based on frame.meta.
    let direct = frame
        .meta
        .get("poscar_mode")
        .and_then(|value| value.as_str())
        == Some("direct");
    writeln!(writer, "{}", if direct { "Direct" } else { "Cartesian" })?;

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

    if direct {
        // Fractions of the cell, from its origin (`SimBox::to_frac`).
        let xyz = ndarray::Array2::from_shape_fn((n, 3), |(i, k)| [xs, ys, zs][k][[i]]);
        let frac = simbox.to_frac(xyz.view());
        for &i in &order {
            writeln!(
                writer,
                "  {:.10}  {:.10}  {:.10}",
                frac[(i, 0)],
                frac[(i, 1)],
                frac[(i, 2)]
            )?;
        }
    } else {
        for &i in &order {
            writeln!(
                writer,
                "  {:.10}  {:.10}  {:.10}",
                xs[[i]],
                ys[[i]],
                zs[[i]]
            )?;
        }
    }
    Ok(())
}

/// Atoms grouped into species runs, as a POSCAR writes them.
struct SpeciesRuns {
    /// One symbol per run; `None` when the frame has no `element` column.
    symbols: Option<Vec<String>>,
    /// Atom count of each run.
    counts: Vec<usize>,
    /// Permutation of the original atom indices that puts each run together.
    order: Vec<usize>,
}

/// Group atoms by their `element` column (if present).
///
/// With symbols present, one run per maximal block of equal symbols; with no
/// element column, a single run of all `n` atoms in their original order.
///
/// # Errors
///
/// `InvalidData` naming the row when an `element` value is empty — a POSCAR
/// species line has no spelling for "unknown".
fn group_by_symbol(n: usize, symbol_col: Option<&ndarray::ArrayD<String>>) -> Result<SpeciesRuns> {
    if let Some(col) = symbol_col {
        let mut runs: Vec<(String, Vec<usize>)> = Vec::new();
        for i in 0..n {
            let s = col[[i]].trim().to_string();
            if s.is_empty() {
                return Err(invalid_data(format!(
                    "POSCAR write: atoms row {i} has an empty 'element'; set every \
                     species (e.g. `Element::fill_symbols(&mut frame)`) before writing"
                )));
            }
            if let Some((sym, idxs)) = runs.last_mut()
                && *sym == s
            {
                idxs.push(i);
                continue;
            }
            runs.push((s, vec![i]));
        }
        let symbols = runs.iter().map(|(s, _)| s.clone()).collect();
        let counts = runs.iter().map(|(_, v)| v.len()).collect();
        let order: Vec<usize> = runs.into_iter().flat_map(|(_, v)| v).collect();
        Ok(SpeciesRuns {
            symbols: Some(symbols),
            counts,
            order,
        })
    } else {
        Ok(SpeciesRuns {
            symbols: None,
            counts: vec![n],
            order: (0..n).collect(),
        })
    }
}

/// Convenience writer: implements [`FrameWriter`].
pub struct VaspPoscarWriter<W: Write> {
    writer: W,
}

impl<W: Write> Writer for VaspPoscarWriter<W> {
    type W = W;
    fn new(writer: W) -> Self {
        Self { writer }
    }
}

impl<W: Write> FrameWriter for VaspPoscarWriter<W> {
    fn write(&mut self, frame: &Frame) -> Result<()> {
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

    const POSCAR_BN: &str = "BN bulk\n\
1.0\n\
2.5  0.0  0.0\n\
0.0  2.5  0.0\n\
0.0  0.0  2.5\n\
B N\n\
1 1\n\
Direct\n\
0.0 0.0 0.0\n\
0.5 0.5 0.5\n";

    #[test]
    fn reads_basic_poscar() {
        let frame = read_frame_from(Cursor::new(POSCAR_BN.as_bytes())).unwrap();
        let atoms = frame.get("atoms").unwrap();
        assert_eq!(atoms.n_rows(), Some(2));
        assert!(frame.simbox.is_some());

        let xs = atoms.get("x").and_then(|c| c.get::<f64>()).unwrap();
        // (0.5, 0.5, 0.5) fractional with 2.5Å cube → (1.25, 1.25, 1.25)
        assert!((xs[[1]] - 1.25).abs() < 1e-10);
    }

    #[test]
    fn round_trip_basic_poscar() {
        let frame = read_frame_from(Cursor::new(POSCAR_BN.as_bytes())).unwrap();
        let mut buf = Vec::new();
        write_frame_to(&mut buf, &frame).unwrap();
        let frame2 = read_frame_from(Cursor::new(&buf)).unwrap();

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
            assert!(
                (xs1[[i]] - xs2[[i]]).abs() < 1e-6,
                "x[{}]: {} vs {}",
                i,
                xs1[[i]],
                xs2[[i]]
            );
        }
    }
}
