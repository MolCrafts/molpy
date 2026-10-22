//! The GRO codec: the GROMACS fixed-column structure / trajectory format.
use crate::io::{Sink, SinkWriter, Source};

use crate::core::Cell;
use crate::io::frame_columns::insert_column;
use crate::io::invalid_data;
use crate::io::pdb::labels::{AtomLabels, LabelWidths};
use std::io::{BufRead, Result, Seek, SeekFrom, Write};

use ndarray::Array2;

use crate::core::unit_factors::NM_TO_ANGSTROM;
use crate::io::reader::{FrameIndex, FrameReader, ReadSeek, Reader, TrajectoryReader};
use crate::io::writer::{FrameWriter, Writer};
use molrs::core::Block;
use molrs::core::Frame;
use molrs::core::SimBox;
use molrs::op::{F, I, Idx};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn substr(s: &str, start: usize, end: usize) -> &str {
    let len = s.len();
    if start >= len {
        return "";
    }
    &s[start..end.min(len)]
}

/// Insert an unsigned column, rejecting negatives with a message that names
/// the key — used for the canonical identifier columns.
fn insert_uint_column_from_signed(block: &mut Block, key: &str, vals: Vec<I>) -> Result<()> {
    let unsigned: Vec<molrs::op::Idx> = vals
        .iter()
        .map(|&v| {
            Idx::try_from(v).map_err(|_| {
                invalid_data(format!(
                    "GRO column '{key}' is unsigned in the Frame schema, got {v}"
                ))
            })
        })
        .collect::<Result<_>>()?;
    insert_column(block, key, unsigned)
}

// ---------------------------------------------------------------------------
// Parsed atom record
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct GroAtom {
    resid: I,
    resname: String,
    atom_name: String,
    atom_id: I,
    x: F,
    y: F,
    z: F,
    velocity: Option<[F; 3]>,
}

fn parse_atom_line(line: &str, line_no: usize) -> Result<GroAtom> {
    // Strip trailing newline only — leading/trailing spaces inside fields are significant.
    let line = line.trim_end_matches(['\r', '\n']);
    if line.len() < 44 {
        return Err(invalid_data(format!(
            "line {}: GRO atom record too short ({} chars; need ≥44)",
            line_no,
            line.len()
        )));
    }
    let resid = substr(line, 0, 5)
        .trim()
        .parse::<I>()
        .map_err(|_| invalid_data(format!("line {}: bad residue number", line_no)))?;
    let resname = substr(line, 5, 10).trim().to_string();
    let atom_name = substr(line, 10, 15).trim().to_string();
    let atom_id = substr(line, 15, 20)
        .trim()
        .parse::<I>()
        .map_err(|_| invalid_data(format!("line {}: bad atom number", line_no)))?;
    let x = substr(line, 20, 28)
        .trim()
        .parse::<F>()
        .map_err(|_| invalid_data(format!("line {}: bad x", line_no)))?;
    let y = substr(line, 28, 36)
        .trim()
        .parse::<F>()
        .map_err(|_| invalid_data(format!("line {}: bad y", line_no)))?;
    let z = substr(line, 36, 44)
        .trim()
        .parse::<F>()
        .map_err(|_| invalid_data(format!("line {}: bad z", line_no)))?;

    let velocity = if line.len() >= 68 {
        let vx = substr(line, 44, 52)
            .trim()
            .parse::<F>()
            .map_err(|_| invalid_data(format!("line {}: bad vx", line_no)))?;
        let vy = substr(line, 52, 60)
            .trim()
            .parse::<F>()
            .map_err(|_| invalid_data(format!("line {}: bad vy", line_no)))?;
        let vz = substr(line, 60, 68)
            .trim()
            .parse::<F>()
            .map_err(|_| invalid_data(format!("line {}: bad vz", line_no)))?;
        Some([vx, vy, vz])
    } else {
        None
    };

    Ok(GroAtom {
        resid,
        resname,
        atom_name,
        atom_id,
        x,
        y,
        z,
        velocity,
    })
}

/// Parse the GROMACS box-vector line (3 floats orthorhombic, 9 triclinic).
/// Returns the 3x3 H matrix with H columns = lattice vectors.
fn parse_box_line(line: &str, line_no: usize) -> Result<[[F; 3]; 3]> {
    let tokens: Vec<&str> = line.split_whitespace().collect();
    if tokens.len() != 3 && tokens.len() != 9 {
        return Err(invalid_data(format!(
            "line {}: box vector line must have 3 or 9 floats, got {}",
            line_no,
            tokens.len()
        )));
    }
    let parse = |idx: usize| -> Result<F> {
        tokens[idx]
            .parse::<F>()
            .map_err(|_| invalid_data(format!("line {}: bad box float '{}'", line_no, tokens[idx])))
    };
    if tokens.len() == 3 {
        let v1x = parse(0)?;
        let v2y = parse(1)?;
        let v3z = parse(2)?;
        return Ok([[v1x, 0.0, 0.0], [0.0, v2y, 0.0], [0.0, 0.0, v3z]]);
    }
    // GROMACS triclinic box order:
    // v1(x) v2(y) v3(z) v1(y) v1(z) v2(x) v2(z) v3(x) v3(y)
    let v1x = parse(0)?;
    let v2y = parse(1)?;
    let v3z = parse(2)?;
    let v1y = parse(3)?;
    let v1z = parse(4)?;
    let v2x = parse(5)?;
    let v2z = parse(6)?;
    let v3x = parse(7)?;
    let v3y = parse(8)?;
    Ok([[v1x, v1y, v1z], [v2x, v2y, v2z], [v3x, v3y, v3z]])
}

/// Format a 3x3 H matrix back into a GRO box line. Emits 3 numbers if the box
/// is orthorhombic (off-diagonals all zero), else the 9-number triclinic form.
fn format_box_line(h: &Array2<F>) -> String {
    let v1x = h[(0, 0)];
    let v2y = h[(1, 1)];
    let v3z = h[(2, 2)];
    let v1y = h[(1, 0)];
    let v1z = h[(2, 0)];
    let v2x = h[(0, 1)];
    let v2z = h[(2, 1)];
    let v3x = h[(0, 2)];
    let v3y = h[(1, 2)];
    let off_diag = [v1y, v1z, v2x, v2z, v3x, v3y];
    let is_ortho = off_diag.iter().all(|v| v.abs() < 1e-10);
    if is_ortho {
        format!("{:10.5}{:10.5}{:10.5}", v1x, v2y, v3z)
    } else {
        format!(
            "{:10.5}{:10.5}{:10.5}{:10.5}{:10.5}{:10.5}{:10.5}{:10.5}{:10.5}",
            v1x, v2y, v3z, v1y, v1z, v2x, v2z, v3x, v3y
        )
    }
}

// ---------------------------------------------------------------------------
// Public reader API
// ---------------------------------------------------------------------------

/// Read the first frame of a `.gro` file.
pub fn read_gro<'a>(src: impl Into<Source<'a>>) -> Result<Frame> {
    let src = src.into();
    GroReader::new(src.reader()?)
        .read()?
        .ok_or_else(|| invalid_data(format!("{}: no GRO frame", src.name())))
}

/// Read a single GRO frame from any [`BufRead`]. Returns `Ok(None)` at EOF.
fn read_frame_from<R: BufRead>(reader: &mut R) -> Result<Option<Frame>> {
    let mut buf = String::new();

    // Title
    buf.clear();
    if reader.read_line(&mut buf)? == 0 {
        return Ok(None);
    }
    let title = buf.trim().to_string();

    // Atom count
    buf.clear();
    if reader.read_line(&mut buf)? == 0 {
        return Err(invalid_data("missing atom-count line"));
    }
    let n_atoms: usize = buf
        .trim()
        .parse()
        .map_err(|_| invalid_data(format!("bad atom-count line: {:?}", buf.trim())))?;

    // Atoms
    let mut atoms = Vec::with_capacity(n_atoms);
    let mut have_velocities: Option<bool> = None;
    for i in 0..n_atoms {
        buf.clear();
        if reader.read_line(&mut buf)? == 0 {
            return Err(invalid_data(format!(
                "unexpected EOF after {} of {} atom records",
                i, n_atoms
            )));
        }
        let atom = parse_atom_line(&buf, i + 3)?;
        // Velocity presence must be consistent across all rows.
        let has_v = atom.velocity.is_some();
        match have_velocities {
            None => have_velocities = Some(has_v),
            Some(expected) if expected != has_v => {
                return Err(invalid_data(format!(
                    "atom {} has {} velocities; expected {} based on first row",
                    i + 1,
                    if has_v { "" } else { "no" },
                    if expected { "yes" } else { "no" }
                )));
            }
            _ => {}
        }
        atoms.push(atom);
    }

    // Box vectors
    buf.clear();
    if reader.read_line(&mut buf)? == 0 {
        return Err(invalid_data("missing box-vector line"));
    }
    let cell_rows = parse_box_line(&buf, 3 + n_atoms)?;

    // -----------------------------------------------------------------------
    // Build the Frame
    // -----------------------------------------------------------------------
    let mut block = Block::new();
    let mut resid = Vec::with_capacity(n_atoms);
    let mut resname = Vec::with_capacity(n_atoms);
    let mut atom_name = Vec::with_capacity(n_atoms);
    let mut atom_id = Vec::with_capacity(n_atoms);
    let mut x = Vec::with_capacity(n_atoms);
    let mut y = Vec::with_capacity(n_atoms);
    let mut z = Vec::with_capacity(n_atoms);
    let mut vx = Vec::with_capacity(n_atoms);
    let mut vy = Vec::with_capacity(n_atoms);
    let mut vz = Vec::with_capacity(n_atoms);
    for a in &atoms {
        resid.push(a.resid);
        resname.push(a.resname.clone());
        atom_name.push(a.atom_name.clone());
        atom_id.push(a.atom_id);
        x.push(a.x * NM_TO_ANGSTROM.get());
        y.push(a.y * NM_TO_ANGSTROM.get());
        z.push(a.z * NM_TO_ANGSTROM.get());
        if let Some(v) = a.velocity {
            // nm/ps → Å/ps: the time unit is untouched, so the length scale is
            // the whole conversion.
            vx.push(v[0] * NM_TO_ANGSTROM.get());
            vy.push(v[1] * NM_TO_ANGSTROM.get());
            vz.push(v[2] * NM_TO_ANGSTROM.get());
        }
    }
    insert_uint_column_from_signed(&mut block, "res_id", resid)?;
    insert_column(&mut block, "res_name", resname)?;
    insert_column(&mut block, "name", atom_name)?;
    insert_uint_column_from_signed(&mut block, "id", atom_id)?;
    insert_column(&mut block, "x", x)?;
    insert_column(&mut block, "y", y)?;
    insert_column(&mut block, "z", z)?;
    if have_velocities == Some(true) {
        insert_column(&mut block, "vx", vx)?;
        insert_column(&mut block, "vy", vy)?;
        insert_column(&mut block, "vz", vz)?;
    }

    let mut frame = Frame::new();
    if !title.is_empty() {
        frame.meta.insert("title", title);
    }
    frame.insert("atoms", block);

    // SimBox: H columns = lattice vectors. cell_rows[i] = lattice vector i.
    let h =
        std::array::from_fn(|i| std::array::from_fn(|j| cell_rows[j][i] * NM_TO_ANGSTROM.get()));
    let origin = [0.0 as F, 0.0, 0.0];
    // A zero box line states that the structure has no geometric cell.
    if h.iter().flatten().any(|&value| value != 0.0) {
        frame.simbox = Some(
            SimBox::new(Cell::Matrix(h), origin, [true; 3])
                .map_err(|e| invalid_data(format!("{e:?}")))?,
        );
    }

    Ok(Some(frame))
}

/// GRO reader: [`FrameReader::read`] streams one frame per call; over a
/// seekable stream it is also a [`TrajectoryReader`] (random access by frame,
/// indexed on first use by one pass over the frame headers).
///
/// GRO is a fixed-column text format used by GROMACS for input structures and
/// single-precision trajectories. One frame layout:
///
/// ```text
/// line 1   : title comment                        → frame.meta["title"]
/// line 2   : atom count `n` (decimal integer)
/// line 3..3+n : atom records (fixed columns; see below)
/// line 3+n : box vectors (3 floats orthorhombic; 9 floats triclinic; nm)
/// ```
///
/// Multi-frame `.gro` files concatenate this layout. [`GroReader::read`]
/// returns one frame per call.
///
/// # Atom record columns (1-indexed)
///
/// | Cols | Meaning                | Example       |
/// |------|------------------------|---------------|
/// | 1-5  | Residue number (i32)   | `    1`       |
/// | 6-10 | Residue name (str)     | `LIG  `       |
/// | 11-15| Atom name (str)        | `   CA`       |
/// | 16-20| Atom number (i32)      | `    1`       |
/// | 21-28| x (nm, %8.3f)          | `   0.310`    |
/// | 29-36| y                      | `   0.862`    |
/// | 37-44| z                      | `   1.316`    |
/// | 45-52| vx (optional, %8.4f)   |               |
/// | 53-60| vy                     |               |
/// | 61-68| vz                     |               |
///
/// # GRO triclinic box convention (line 3+n)
///
/// Tokens, in file order: `v1x v2y v3z v1y v1z v2x v2z v3x v3y`. When only 3
/// tokens are present, the box is orthorhombic: off-diagonals = 0.
///
/// # Output Frame
///
/// - `"atoms"` block: `res_id` (uint), `res_name` (str), `name` (str),
///   `id` (uint),
///   `x`/`y`/`z` (F, **Å**), and optional `vx`/`vy`/`vz` (F, **Å/ps**).
///   GRO has no species field, so there is no `element` / `atomic_number`
///   column: species is not guessed from the atom name. A consumer that
///   needs one sets `element` itself and runs `Element::fill_atomic_numbers`.
/// - `frame.simbox`: triclinic [`SimBox`] from the box-vector line, in Å.
/// - `frame.meta["title"]`.
///
/// # Units
///
/// GRO is nm; molrs is Å. The reader multiplies by 10 and the
/// writer divides by it, so a frame in memory is never in nm and no consumer
/// has to ask where it came from. There is no `gro_units` tag: a unit that is
/// normalised at the boundary is not a property of the frame.
pub struct GroReader<R: BufRead> {
    reader: R,
    index: Option<FrameIndex>,
}

impl<R: BufRead> GroReader<R> {
    /// A reader over any buffered byte stream.
    pub fn new(reader: R) -> Self {
        Self {
            reader,
            index: None,
        }
    }
}

impl GroReader<Box<dyn ReadSeek>> {
    /// Open a `.gro` file, or the bytes of one, for random access by frame.
    pub fn open<'a>(src: impl Into<Source<'a>>) -> Result<Self> {
        Ok(Self::new(src.into().owned_reader()?))
    }
}

impl<R: BufRead> Reader for GroReader<R> {
    type R = R;
    fn new(reader: R) -> Self {
        GroReader::new(reader)
    }
}

impl<R: BufRead + Seek> TrajectoryReader for GroReader<R> {
    fn build_index(&mut self) -> Result<()> {
        if self.index.is_some() {
            return Ok(());
        }
        let start = self.reader.stream_position()?;
        self.reader.seek(SeekFrom::Start(0))?;
        let mut index = FrameIndex::new();
        let mut offset = 0u64;
        let mut line = String::new();
        loop {
            // A frame: title, atom count, that many atom lines, box line.
            line.clear();
            let n = self.reader.read_line(&mut line)?;
            if n == 0 {
                break;
            }
            index.add_frame(offset);
            offset += n as u64;
            line.clear();
            let n = self.reader.read_line(&mut line)?;
            if n == 0 {
                return Err(invalid_data("GRO frame ends before its atom-count line"));
            }
            offset += n as u64;
            let atoms: usize = line
                .trim()
                .parse()
                .map_err(|_| invalid_data(format!("GRO atom count {:?}", line.trim())))?;
            for _ in 0..=atoms {
                line.clear();
                let n = self.reader.read_line(&mut line)?;
                if n == 0 {
                    return Err(invalid_data("GRO frame ends before its box line"));
                }
                offset += n as u64;
            }
        }
        self.reader.seek(SeekFrom::Start(start))?;
        self.index = Some(index);
        Ok(())
    }

    fn read_frame(&mut self, index: usize) -> Result<Option<Frame>> {
        self.build_index()?;
        let Some(at) = self.index.as_ref().and_then(|offsets| offsets.get(index)) else {
            return Ok(None);
        };
        self.reader.seek(SeekFrom::Start(at))?;
        crate::io::reader::check_read_frame(read_frame_from(&mut self.reader)?)
    }

    fn len(&mut self) -> Result<usize> {
        self.build_index()?;
        Ok(self.index.as_ref().map_or(0, FrameIndex::len))
    }
}

impl<R: BufRead> FrameReader for GroReader<R> {
    fn read(&mut self) -> Result<Option<Frame>> {
        // Validate on the way out: a frame that violates the vocabulary
        // is a malformed file or a reader bug, not a result to return.
        crate::io::reader::check_read_frame(read_frame_from(&mut self.reader)?)
    }
}

// ---------------------------------------------------------------------------
// Public writer API
// ---------------------------------------------------------------------------

/// Write a frame as `.gro` at `path`.
///
/// The residue number, residue name and atom name are the PDB writer's
/// labels at GROMACS's five columns
/// ([`write_pdb`](crate::io::write_pdb)): the frame's own, else — no
/// column, or an empty or null cell — the atom's molecule's number (bonded
/// components of `bonds`), `MOL`, and the element numbered within its
/// residue (`C1`, `C2`, `H1`, …), as
/// [`GromacsTopForcefieldWriter::write_system`](crate::io::gromacs::GromacsTopForcefieldWriter::write_system) names them too. Atom
/// numbers are the `id` column, else 1..N.
///
/// # Errors
///
/// A frame without `atoms` or its `x`/`y`/`z`; a null cell in `id`,
/// `x`/`y`/`z` or `vx`/`vy`/`vz` (named by column and rows: no placeholder
/// is written for it); a `bonds` row naming an atom past the frame.
pub fn write_gro<'a>(sink: impl Into<Sink<'a>>, frame: &Frame) -> Result<()> {
    let mut w = sink.into().writer()?;
    write_frame_to(&mut w, frame)?;
    w.flush()
}

/// Write a single frame in GRO format.
fn write_frame_to<W: Write>(writer: &mut W, frame: &Frame) -> Result<()> {
    let atoms = frame
        .get("atoms")
        .ok_or_else(|| invalid_data("GRO write: frame has no atoms block"))?;
    let n = atoms
        .n_rows()
        .ok_or_else(|| invalid_data("GRO write: atoms block has no rows"))?;

    let title = frame
        .meta
        .get("title")
        .and_then(|value| value.as_str())
        .unwrap_or("molrs GRO");
    writeln!(writer, "{}", title)?;
    writeln!(writer, "{:>5}", n)?;

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
    let vx = atoms.get("vx").and_then(|c| c.get::<f64>());
    let vy = atoms.get("vy").and_then(|c| c.get::<f64>());
    let vz = atoms.get("vz").and_then(|c| c.get::<f64>());
    let atom_id = atoms.get("id").and_then(|c| c.get::<u64>());
    // A stated number or coordinate has no stand-in; an unset name is
    // generated (`null_cells`).
    crate::io::null_cells::refuse_null_cells(
        frame,
        "atoms",
        &["id", "x", "y", "z", "vx", "vy", "vz"],
    )?;
    // Names and residues as the `.top` writer labels them, so `grompp`
    // finds the two alike: a frame without names (XYZ, say) gets its
    // elements numbered within each residue.
    let labels = AtomLabels::with_widths(frame, n, LabelWidths::GROMACS)?;

    for i in 0..n {
        let aid = atom_id.map(|c| c[[i]]).unwrap_or((i as molrs::op::Idx) + 1);
        // GROMACS truncates the atom number at 5 digits via modulo (the
        // labels' residue number already is).
        let aid_mod = aid.rem_euclid(100_000);
        // Å → nm on the way out, mirroring the reader's normalisation.
        write!(
            writer,
            "{:>5}{:<5}{:>5}{:>5}{:>8.3}{:>8.3}{:>8.3}",
            labels.res_seqs[i],
            labels.res_names[i],
            labels.names[i],
            aid_mod,
            xs[[i]] / NM_TO_ANGSTROM.get(),
            ys[[i]] / NM_TO_ANGSTROM.get(),
            zs[[i]] / NM_TO_ANGSTROM.get()
        )?;
        if let (Some(vxc), Some(vyc), Some(vzc)) = (vx, vy, vz) {
            write!(
                writer,
                "{:>8.4}{:>8.4}{:>8.4}",
                vxc[[i]] / NM_TO_ANGSTROM.get(),
                vyc[[i]] / NM_TO_ANGSTROM.get(),
                vzc[[i]] / NM_TO_ANGSTROM.get()
            )?;
        }
        writeln!(writer)?;
    }

    let h = frame
        .simbox
        .as_ref()
        .map(|sb| ndarray::arr2(&sb.matrix()) / NM_TO_ANGSTROM.get())
        .unwrap_or_else(|| Array2::<F>::zeros((3, 3)));
    writeln!(writer, "{}", format_box_line(&h))?;

    Ok(())
}

/// `FrameWriter`-trait wrapper.
pub struct GroWriter<W: Write> {
    writer: W,
}

impl<'a> GroWriter<SinkWriter<'a>> {
    /// Create a `.gro` file, or fill a buffer, frame by frame.
    pub fn create(sink: impl Into<Sink<'a>>) -> Result<Self> {
        Ok(Self {
            writer: sink.into().writer()?,
        })
    }
}

impl<W: Write> Writer for GroWriter<W> {
    type W = W;
    fn new(writer: W) -> Self {
        Self { writer }
    }
}

impl<W: Write> FrameWriter for GroWriter<W> {
    fn write(&mut self, frame: &Frame) -> Result<()> {
        // Refuse to emit a frame that violates the vocabulary: a bad file
        // looks fine and is found wrong later, by whatever reads it.
        crate::io::writer::check_write_frame(frame)?;
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

    fn water_gro() -> String {
        // Each atom line must be at least 44 chars, fixed-column.
        // resid(5) + resname(5) + atom_name(5) + atom_id(5) + x(8) + y(8) + z(8) = 44
        let lines = [
            "Water box",
            "    3",
            "    1WAT     OW    1   0.000   0.000   0.000",
            "    1WAT    HW1    2   0.100   0.000   0.000",
            "    1WAT    HW2    3   0.000   0.100   0.000",
            "   2.00000   2.00000   2.00000",
        ];
        let mut out = String::new();
        for l in lines {
            out.push_str(l);
            out.push('\n');
        }
        out
    }

    /// Two frames written as one trajectory read back as two frames.
    /// Over a seekable stream the reader is a trajectory: frames by index.
    #[test]
    fn gro_reader_reads_frames_by_index() {
        use crate::io::reader::TrajectoryReader;
        let text = water_gro() + &water_gro().replacen("0.100", "0.200", 1);
        let mut reader = GroReader::new(Cursor::new(text.into_bytes()));
        assert_eq!(reader.len().unwrap(), 2);
        let x = |f: &Frame| {
            f.get("atoms")
                .unwrap()
                .get("x")
                .unwrap()
                .get::<f64>()
                .unwrap()[[1]]
        };
        let second = reader.read_frame(1).unwrap().expect("frame 2");
        let first = reader.read_frame(0).unwrap().expect("frame 1");
        assert!((x(&first) - 1.0).abs() < 1e-9);
        assert!((x(&second) - 2.0).abs() < 1e-9);
        assert!(reader.read_frame(2).unwrap().is_none());
    }

    #[test]
    fn write_gro_trajectory_round_trips_every_frame() {
        let frame = read_frame_from(&mut Cursor::new(water_gro().into_bytes()))
            .unwrap()
            .unwrap();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("traj.gro");
        let mut writer = GroWriter::create(&path).expect("create GRO trajectory");
        writer.write(&frame).expect("frame 1");
        writer.write(&frame).expect("frame 2");
        drop(writer);
        let back = GroReader::open(&path)
            .expect("open GRO trajectory")
            .iter()
            .collect::<Result<Vec<_>>>()
            .expect("read GRO trajectory");
        assert_eq!(back.len(), 2);
        assert_eq!(back[1].get("atoms").unwrap().n_rows(), Some(3));
    }

    #[test]
    fn reads_basic_gro() {
        let frame = read_frame_from(&mut Cursor::new(water_gro().into_bytes()))
            .unwrap()
            .unwrap();
        let atoms = frame.get("atoms").unwrap();
        assert_eq!(atoms.n_rows(), Some(3));
        let xs = atoms.get("x").and_then(|c| c.get::<f64>()).unwrap();
        assert!((xs[[1]] - 1.0).abs() < 1e-9); // 0.100 nm → 1.0 Å
        let names = atoms.get("name").and_then(|c| c.get::<String>()).unwrap();
        assert_eq!(names[[0]], "OW");
        assert_eq!(names[[1]], "HW1");
        assert!(frame.simbox.is_some());
    }

    #[test]
    fn reads_lengths_as_angstrom() {
        // GRO is nm; every other molrs format is Å. The reader normalises at
        // its boundary, so nothing downstream has to know where a frame came
        // from — 0.100 nm is 1.0 Å, and the 2 nm box is 20 Å.
        let frame = read_frame_from(&mut Cursor::new(water_gro().into_bytes()))
            .unwrap()
            .unwrap();
        let xs = frame
            .get("atoms")
            .unwrap()
            .get("x")
            .and_then(|c| c.get::<f64>())
            .unwrap();
        assert!((xs[[1]] - 1.0).abs() < 1e-9, "x[1] = {}", xs[[1]]);
        let h = ndarray::arr2(&frame.simbox.as_ref().unwrap().matrix());
        assert!((h[[0, 0]] - 20.0).abs() < 1e-9, "box = {}", h[[0, 0]]);
        assert!(
            !frame.meta.contains_key("gro_units"),
            "the frame is in Å; a nm tag would be a lie"
        );
    }

    #[test]
    fn reads_velocities_as_angstrom_per_ps() {
        // A velocity is a length per time: leaving it in nm/ps beside an Å
        // position is what makes `x += v * dt` silently wrong.
        let with_v = "Water box\n    1\n    1WAT     OW    1   0.000   0.000   0.000  0.1000  0.2000  0.3000\n   2.00000   2.00000   2.00000\n";
        let frame = read_frame_from(&mut Cursor::new(with_v.as_bytes().to_vec()))
            .unwrap()
            .unwrap();
        let vx = frame
            .get("atoms")
            .unwrap()
            .get("vx")
            .and_then(|c| c.get::<f64>())
            .unwrap();
        assert!((vx[[0]] - 1.0).abs() < 1e-9, "vx = {}", vx[[0]]);
    }

    #[test]
    fn reads_no_species_column_from_atom_names() {
        // A GRO file has no species field. `CA` is a C-alpha in one residue
        // and calcium in another; the reader keeps the name and invents no
        // element for it.
        let frame = read_frame_from(&mut Cursor::new(water_gro().into_bytes()))
            .unwrap()
            .unwrap();
        let atoms = frame.get("atoms").unwrap();
        assert!(atoms.get("element").is_none());
        assert!(atoms.get("atomic_number").is_none());
        let names = atoms.get("name").and_then(|c| c.get::<String>()).unwrap();
        assert_eq!(names[[0]], "OW");
    }

    #[test]
    fn gro_preserves_names_without_guessing_species() {
        // GRO spells names, not elements: keep protein C-alpha and calcium labels verbatim.
        let ions = concat!(
            "ions\n",
            "    6\n",
            "    1ALA      N    1   0.000   0.000   0.000\n",
            "    1ALA     CA    2   0.100   0.000   0.000\n",
            "    1ALA     CB    3   0.200   0.000   0.000\n",
            "    2CA      CA    4   1.000   0.000   0.000\n",
            "    3CL      CL    5   0.000   1.000   0.000\n",
            "    4SOL     OW    6   0.000   0.000   1.000\n",
            "   2.00000   2.00000   2.00000\n",
        );
        let frame = read_frame_from(&mut Cursor::new(ions.as_bytes().to_vec()))
            .unwrap()
            .unwrap();
        let atoms = frame.get("atoms").unwrap();
        let names = atoms.get("name").and_then(|c| c.get::<String>()).unwrap();
        assert_eq!(
            names.as_slice().unwrap(),
            ["N", "CA", "CB", "CA", "CL", "OW"]
        );
        assert!(atoms.get("element").is_none());
        assert!(atoms.get("atomic_number").is_none());
    }

    #[test]
    fn names_an_unnamed_atom_by_its_element() {
        // The mirror of the read-side inference: a frame that came from a
        // format without atom names still knows its elements, and "X" would
        // throw away the identity the next reader is expected to recover.
        // The element is numbered within the residue, as the `.top` and PDB
        // writers name it.
        use molrs::core::Block;
        use ndarray::Array1;

        let mut atoms = Block::new();
        for (key, v) in [("x", 1.0 as F), ("y", 0.0), ("z", 0.0)] {
            atoms
                .insert(key, Array1::from_vec(vec![v]).into_dyn(), None)
                .unwrap();
        }
        atoms
            .insert(
                "element",
                Array1::from_vec(vec!["Na".to_string()]).into_dyn(),
                None,
            )
            .unwrap();
        let mut frame = Frame::new();
        frame.insert("atoms", atoms);

        let mut buf = Vec::new();
        write_frame_to(&mut buf, &frame).unwrap();
        let text = String::from_utf8(buf.clone()).unwrap();
        let atom_line = text.lines().nth(2).expect("atom line");
        assert_eq!(
            substr(atom_line, 10, 15).trim(),
            "Na1",
            "line: {atom_line:?}"
        );
        let back = read_frame_from(&mut Cursor::new(buf)).unwrap().unwrap();
        assert_eq!(
            back["atoms"].get("name").unwrap().get::<String>().unwrap()[[0]],
            "Na1"
        );
    }

    #[test]
    fn round_trip_basic_gro() {
        let frame = read_frame_from(&mut Cursor::new(water_gro().into_bytes()))
            .unwrap()
            .unwrap();
        let mut buf = Vec::new();
        write_frame_to(&mut buf, &frame).unwrap();
        let frame2 = read_frame_from(&mut Cursor::new(&buf)).unwrap().unwrap();
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
        assert_eq!(xs1.len(), xs2.len());
        for i in 0..xs1.len() {
            assert!((xs1[[i]] - xs2[[i]]).abs() < 1e-3);
        }
    }
}
