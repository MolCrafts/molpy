//! Gaussian Cube file reader and writer.

use std::io::{BufRead, BufReader, Write};
use std::path::Path;

use ndarray::{Array1, ArrayD, IxDyn};

use crate::core::unit_factors::BOHR_TO_ANGSTROM;

use molrs::core::Block;
use molrs::core::Element;
use molrs::core::Frame;
use molrs::core::MolRsError;
use molrs::core::SimBox;
use molrs::op::{F, Idx};

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Read a Gaussian Cube file from disk.
pub fn read_cube<P: AsRef<Path>>(path: P) -> Result<Frame, MolRsError> {
    let file = std::fs::File::open(path.as_ref()).map_err(MolRsError::Io)?;
    read_frame_from(BufReader::new(file))
}

/// Read a Gaussian Cube file from any [`BufRead`] source.
fn read_frame_from<R: BufRead>(mut reader: R) -> Result<Frame, MolRsError> {
    let mut line_no = 0usize;
    // Reused across every header/atom line so the parse loop allocates no
    // per-line String (the volumetric reader below reuses its own buffer too).
    let mut line_buf = String::new();

    macro_rules! next_line {
        () => {{
            line_buf.clear();
            reader.read_line(&mut line_buf).map_err(MolRsError::Io)?;
            line_no += 1;
            line_buf.as_str()
        }};
    }

    macro_rules! parse_err {
        ($msg:expr) => {
            MolRsError::parse_error(line_no, $msg)
        };
    }

    // -----------------------------------------------------------------------
    // Comment lines
    // -----------------------------------------------------------------------
    let comment1 = next_line!().trim_end().to_string();
    let comment2 = next_line!().trim_end().to_string();

    // -----------------------------------------------------------------------
    // Line 3: NATOMS  origin_x  origin_y  origin_z
    // -----------------------------------------------------------------------
    let line3 = next_line!();
    let toks3: Vec<&str> = line3.split_whitespace().collect();
    if toks3.len() < 4 {
        return Err(parse_err!("expected NATOMS origin_x origin_y origin_z"));
    }
    let natoms_signed: i32 = toks3[0]
        .parse()
        .map_err(|_| parse_err!("bad NATOMS integer"))?;
    let has_mo = natoms_signed < 0;
    let n_atoms = natoms_signed.unsigned_abs() as usize;
    let origin = [
        parse_f64(toks3[1], line_no)?,
        parse_f64(toks3[2], line_no)?,
        parse_f64(toks3[3], line_no)?,
    ];

    // -----------------------------------------------------------------------
    // Lines 4-6: voxel axes  N_i  vx vy vz
    // -----------------------------------------------------------------------
    let mut dims = [0usize; 3];
    let mut voxel_vecs = [[0.0f64; 3]; 3];
    let mut is_angstrom = false;

    for i in 0..3 {
        let line = next_line!();
        let toks: Vec<&str> = line.split_whitespace().collect();
        if toks.len() < 4 {
            return Err(parse_err!("expected N vx vy vz for voxel axis"));
        }
        let n_signed: i32 = toks[0].parse().map_err(|_| parse_err!("bad voxel count"))?;
        if i == 0 && n_signed < 0 {
            is_angstrom = true;
        }
        dims[i] = n_signed.unsigned_abs() as usize;
        voxel_vecs[i][0] = parse_f64(toks[1], line_no)?;
        voxel_vecs[i][1] = parse_f64(toks[2], line_no)?;
        voxel_vecs[i][2] = parse_f64(toks[3], line_no)?;
    }

    // -----------------------------------------------------------------------
    // Atom lines: Z  charge  x  y  z
    // -----------------------------------------------------------------------
    let mut atomic_numbers = Vec::with_capacity(n_atoms);
    let mut charges = Vec::with_capacity(n_atoms);
    let mut xs = Vec::with_capacity(n_atoms);
    let mut ys = Vec::with_capacity(n_atoms);
    let mut zs = Vec::with_capacity(n_atoms);
    let mut symbols = Vec::with_capacity(n_atoms);

    for _ in 0..n_atoms {
        let line = next_line!();
        let toks: Vec<&str> = line.split_whitespace().collect();
        if toks.len() < 5 {
            return Err(parse_err!("expected Z charge x y z for atom"));
        }
        let z: i32 = toks[0]
            .parse()
            .map_err(|_| parse_err!("bad atomic number"))?;
        atomic_numbers.push(z);
        charges.push(parse_f64(toks[1], line_no)?);
        xs.push(parse_f64(toks[2], line_no)?);
        ys.push(parse_f64(toks[3], line_no)?);
        zs.push(parse_f64(toks[4], line_no)?);

        let sym = if z > 0 && z <= 118 {
            Element::by_number(z as u8)
                .map(|e| e.symbol().to_string())
                .unwrap_or_else(|| "X".to_string())
        } else {
            "X".to_string()
        };
        symbols.push(sym);
    }

    // -----------------------------------------------------------------------
    // Optional MO header: NVALS  idx1  idx2  …
    // -----------------------------------------------------------------------
    let mut mo_indices: Vec<usize> = Vec::new();

    let n_vals_per_point: usize = if has_mo {
        let line = next_line!();
        let toks: Vec<&str> = line.split_whitespace().collect();
        if toks.is_empty() {
            return Err(parse_err!("expected MO header (NVALS idx1 idx2 …)"));
        }
        let nvals: usize = toks[0]
            .parse()
            .map_err(|_| parse_err!("bad NVALS in MO header"))?;
        for &t in &toks[1..] {
            let idx: usize = t.parse().map_err(|_| parse_err!("bad MO index"))?;
            mo_indices.push(idx);
        }
        if mo_indices.len() != nvals {
            return Err(parse_err!(format!(
                "NVALS={} but {} indices given",
                nvals,
                mo_indices.len()
            )));
        }
        nvals
    } else {
        1
    };

    // -----------------------------------------------------------------------
    // Volumetric data
    // -----------------------------------------------------------------------
    let n_voxels = dims[0] * dims[1] * dims[2];
    let total_values = n_voxels * n_vals_per_point;
    let flat_data = read_volumetric_data(&mut reader, total_values, &mut line_no)?;

    // -----------------------------------------------------------------------
    // Normalise to Å. Cube files store atom positions and voxel vectors in
    // the same unit system declared by the sign of N1; we convert here so
    // every downstream consumer (simbox, marching cubes, atom rendering)
    // can treat the frame as Å without further bookkeeping.
    // -----------------------------------------------------------------------
    let unit_scale: f64 = if is_angstrom {
        1.0
    } else {
        BOHR_TO_ANGSTROM.get()
    };
    let origin_ang: [f64; 3] = [
        origin[0] * unit_scale,
        origin[1] * unit_scale,
        origin[2] * unit_scale,
    ];
    let voxel_vecs_ang: [[f64; 3]; 3] = [
        [
            voxel_vecs[0][0] * unit_scale,
            voxel_vecs[0][1] * unit_scale,
            voxel_vecs[0][2] * unit_scale,
        ],
        [
            voxel_vecs[1][0] * unit_scale,
            voxel_vecs[1][1] * unit_scale,
            voxel_vecs[1][2] * unit_scale,
        ],
        [
            voxel_vecs[2][0] * unit_scale,
            voxel_vecs[2][1] * unit_scale,
            voxel_vecs[2][2] * unit_scale,
        ],
    ];

    // cell column i = voxel_vec[i] * N_i — total span along voxel axis i.
    let cell_cols: [[F; 3]; 3] = [
        [
            (voxel_vecs_ang[0][0] * dims[0] as f64) as F,
            (voxel_vecs_ang[0][1] * dims[0] as f64) as F,
            (voxel_vecs_ang[0][2] * dims[0] as f64) as F,
        ],
        [
            (voxel_vecs_ang[1][0] * dims[1] as f64) as F,
            (voxel_vecs_ang[1][1] * dims[1] as f64) as F,
            (voxel_vecs_ang[1][2] * dims[1] as f64) as F,
        ],
        [
            (voxel_vecs_ang[2][0] * dims[2] as f64) as F,
            (voxel_vecs_ang[2][1] * dims[2] as f64) as F,
            (voxel_vecs_ang[2][2] * dims[2] as f64) as F,
        ],
    ];

    // -----------------------------------------------------------------------
    // Build the volumetric Block ("grid"): one f64 column per scalar field,
    // structural shape = [nx, ny, nz] so consumers can unflatten the row
    // index back into voxel coordinates.
    // -----------------------------------------------------------------------
    let mut grid_block = Block::new();
    if has_mo {
        // Deinterleave: file stores [v0_pt0, v1_pt0, …, v0_pt1, v1_pt1, …].
        // Each orbital becomes its own column.
        for (k, &idx) in mo_indices.iter().enumerate() {
            let col_data: Vec<F> = (0..n_voxels)
                .map(|i| flat_data[i * n_vals_per_point + k])
                .collect();
            grid_block
                .insert(format!("mo_{}", idx), Array1::from_vec(col_data).into_dyn())
                .map_err(MolRsError::Block)?;
        }
    } else {
        grid_block
            .insert("density", Array1::from_vec(flat_data).into_dyn())
            .map_err(MolRsError::Block)?;
    }
    grid_block
        .set_shape(&[dims[0], dims[1], dims[2]])
        .map_err(MolRsError::Block)?;

    // -----------------------------------------------------------------------
    // Build atoms Block. Coordinates are in the file's native unit; convert
    // to Å so simbox + atoms share the same world frame.
    // -----------------------------------------------------------------------
    let mut atoms = Block::new();
    atoms
        .insert(
            "x",
            Array1::from_vec(
                xs.iter()
                    .map(|&v| (v * unit_scale) as F)
                    .collect::<Vec<_>>(),
            )
            .into_dyn(),
        )
        .map_err(MolRsError::Block)?;
    atoms
        .insert(
            "y",
            Array1::from_vec(
                ys.iter()
                    .map(|&v| (v * unit_scale) as F)
                    .collect::<Vec<_>>(),
            )
            .into_dyn(),
        )
        .map_err(MolRsError::Block)?;
    atoms
        .insert(
            "z",
            Array1::from_vec(
                zs.iter()
                    .map(|&v| (v * unit_scale) as F)
                    .collect::<Vec<_>>(),
            )
            .into_dyn(),
        )
        .map_err(MolRsError::Block)?;
    atoms
        .insert(
            "atomic_number",
            // UInt, not Int: the Frame schema declares this column unsigned, and
            // an insert that disagrees is rejected outright.
            Array1::from_vec(atomic_numbers.iter().map(|&v| v as Idx).collect::<Vec<_>>())
                .into_dyn(),
        )
        .map_err(MolRsError::Block)?;
    atoms
        .insert(
            "charge",
            Array1::from_vec(charges.iter().map(|&v| v as F).collect::<Vec<_>>()).into_dyn(),
        )
        .map_err(MolRsError::Block)?;
    atoms
        .insert(
            "element",
            ArrayD::from_shape_vec(IxDyn(&[n_atoms]), symbols)
                .expect("shape matches")
                .into_dyn(),
        )
        .map_err(MolRsError::Block)?;

    // -----------------------------------------------------------------------
    // Build SimBox: column i of h = i-th voxel-axis × dim, origin in Å.
    // PBC defaults off — cube files describe a finite voxel grid, not a
    // periodic cell. Hosts that know better can flip pbc on later.
    // -----------------------------------------------------------------------
    let h = ndarray::Array2::from_shape_fn((3, 3), |(i, j)| cell_cols[j][i]);
    let origin_arr = ndarray::array![origin_ang[0] as F, origin_ang[1] as F, origin_ang[2] as F];
    let simbox = SimBox::new(h, origin_arr, [false; 3])
        .map_err(|e| MolRsError::parse(format!("invalid cube cell: {:?}", e)))?;

    // -----------------------------------------------------------------------
    // Assemble Frame
    // -----------------------------------------------------------------------
    let mut frame = Frame::new();
    if !comment1.is_empty() {
        frame.meta.insert("comment1", comment1);
    }
    if !comment2.is_empty() {
        frame.meta.insert("comment2", comment2);
    }
    frame
        .meta
        .insert("cube_units", if is_angstrom { "angstrom" } else { "bohr" });
    if has_mo {
        let indices_str = mo_indices
            .iter()
            .map(|i| i.to_string())
            .collect::<Vec<_>>()
            .join(",");
        frame.meta.insert("cube_mo_indices", indices_str);
    }

    frame.simbox = Some(simbox);
    frame.insert("atoms", atoms);
    frame.insert("grid", grid_block);

    Ok(frame)
}

/// Write a Gaussian Cube file to disk.
pub fn write_cube<P: AsRef<Path>>(path: P, frame: &Frame) -> Result<(), MolRsError> {
    let file = std::fs::File::create(path.as_ref()).map_err(MolRsError::Io)?;
    let mut writer = std::io::BufWriter::new(file);
    write_frame_to(&mut writer, frame)
}

/// Read one Gaussian cube from its text.
///
/// # File layout
///
/// ```text
/// <comment line 1>                    ← free text → frame.meta["comment1"]
/// <comment line 2>                    ← free text → frame.meta["comment2"]
/// NATOMS  origin_x  origin_y  origin_z
///   N1    vx1_x  vx1_y  vx1_z         ← voxel step vector for axis 1
///   N2    vx2_x  vx2_y  vx2_z         ← voxel step vector for axis 2
///   N3    vx3_x  vx3_y  vx3_z         ← voxel step vector for axis 3
///   Z  charge  x  y  z                ← one atom per line
///   …
/// [MO only: NVALS  idx1  idx2  …]     ← present if NATOMS < 0
/// <val> <val> …                        ← volumetric data, 6 per line
/// ```
///
/// # Sign conventions
///
/// - **NATOMS**: negative → MO mode (extra header line after atoms listing
///   orbital count and indices).
/// - **N1** (first voxel count): positive → file units are Bohr; negative →
///   Angstrom. The absolute value is the actual grid count.
///
/// # Data layout
///
/// Volumetric data is stored with x as the outermost loop and z as the
/// innermost (row-major). This matches the molvis marching-cubes consumer's
/// `data[ix*ny*nz + iy*nz + iz]` layout, so **no reordering** is needed
/// (unlike CHGCAR).
///
/// # Unit handling
///
/// Atom coordinates and the simulation box are normalised to **Å** on read
/// (bohr → Å through the unit registry — the Bohr radius, CODATA 2018
/// 0.529 177 210 903 Å — when the file uses Bohr units).
/// The original unit system is recorded in `frame.meta["cube_units"]` so
/// [`write_cube`] can round-trip the file without surprising the
/// producing toolchain.
///
/// # Frame contents
///
/// | Block        | Key(s)            | Content                        |
/// |--------------|-------------------|--------------------------------|
/// | `"atoms"`    | element, x, y, z, atomic_number, charge | Atom data (Å) |
/// | `"grid"`     | `"density"` or `"mo_<idx>"`             | Volumetric scalar fields, shape `[nx, ny, nz]` |
/// | `simbox`     | (h-matrix, origin)                      | Voxel cell × dims, in Å |
pub fn read_cube_str(text: &str) -> Result<Frame, MolRsError> {
    read_frame_from(text.as_bytes())
}

/// Write one frame as Gaussian cube text — the inverse of [`read_cube_str`].
pub fn write_cube_str(frame: &Frame) -> Result<String, MolRsError> {
    let mut buf = Vec::new();
    write_frame_to(&mut buf, frame)?;
    String::from_utf8(buf).map_err(|e| MolRsError::parse(e.to_string()))
}

/// Read a concatenated multi-cube trajectory file into one [`Frame`] per cube.
///
/// AIMD spectroscopy workflows emit one volumetric electron-density cube per
/// step, concatenated back-to-back with no blank separator lines (as the
/// reference `bqb_cubeframe.cpp` consumes them); each is parsed by the
/// single-cube reader. Reading stops cleanly at EOF. Every frame is normalised
/// to Å; the volumetric block stays in its native `e/Bohr³` (give the Voronoi
/// integrator `LengthUnit::Bohr` and it converts to `e/Å³`).
pub fn read_cube_trajectory<P: AsRef<Path>>(path: P) -> Result<Vec<Frame>, MolRsError> {
    let file = std::fs::File::open(path).map_err(MolRsError::Io)?;
    let mut reader = BufReader::new(file);
    let mut frames = Vec::new();
    loop {
        // EOF when the buffered reader has nothing more to hand out.
        let at_eof = reader.fill_buf().map_err(MolRsError::Io)?.is_empty();
        if at_eof {
            break;
        }
        frames.push(read_frame_from(&mut reader)?);
    }
    Ok(frames)
}

/// Write a Gaussian Cube file to any [`Write`] destination.
///
/// Reads from the modern Frame layout: `frame.get("grid")` (Block with
/// `set_shape([nx, ny, nz])` and one f64 column per scalar field),
/// `frame.get("atoms")` for geometry, and `frame.simbox` for the cell.
fn write_frame_to<W: Write>(writer: &mut W, frame: &Frame) -> Result<(), MolRsError> {
    let grid_block = frame
        .get("grid")
        .ok_or_else(|| MolRsError::validation("frame has no 'grid' block"))?;
    let atoms = frame
        .get("atoms")
        .ok_or_else(|| MolRsError::validation("frame has no 'atoms' block"))?;
    let simbox = frame
        .simbox
        .as_ref()
        .ok_or_else(|| MolRsError::validation("frame has no simbox; cannot recover cube cell"))?;

    let n_atoms = atoms.n_rows().unwrap_or(0);
    let grid_shape = grid_block.shape();
    if grid_shape.len() != 3 {
        return Err(MolRsError::validation(format!(
            "grid block must have 3-D shape, got {:?}",
            grid_shape
        )));
    }
    let (nx, ny, nz) = (grid_shape[0], grid_shape[1], grid_shape[2]);

    // Determine MO mode from metadata
    let mo_indices: Option<Vec<usize>> = frame
        .meta
        .get("cube_mo_indices")
        .and_then(|value| value.as_str())
        .map(|s| {
            s.split(',')
                .filter_map(|t| t.trim().parse::<usize>().ok())
                .collect()
        });
    let has_mo = mo_indices.is_some();

    // Determine unit sign convention. Stored cell/atom coordinates are in
    // Å regardless; the meta key only controls how we write the file back
    // out so producers consuming the round-tripped file see what they
    // expect.
    let is_angstrom = frame
        .meta
        .get("cube_units")
        .and_then(|value| value.as_str())
        .is_some_and(|u| u == "angstrom");
    let unit_scale: f64 = if is_angstrom {
        1.0
    } else {
        1.0 / BOHR_TO_ANGSTROM.get()
    };

    // Recover origin and cell columns from simbox, in the writer's unit.
    let origin_arr = simbox.origin_view().to_owned();
    let origin = [
        origin_arr[0] * unit_scale,
        origin_arr[1] * unit_scale,
        origin_arr[2] * unit_scale,
    ];
    let h = simbox.h_view();

    // Comment lines
    let c1 = frame
        .meta
        .get("comment1")
        .and_then(|value| value.as_str())
        .unwrap_or("");
    let c2 = frame
        .meta
        .get("comment2")
        .and_then(|value| value.as_str())
        .unwrap_or("");
    writeln!(writer, "{}", c1).map_err(MolRsError::Io)?;
    writeln!(writer, "{}", c2).map_err(MolRsError::Io)?;

    // NATOMS line
    let natoms_signed: i32 = if has_mo {
        -(n_atoms as i32)
    } else {
        n_atoms as i32
    };
    writeln!(
        writer,
        "{:5}{:12.6}{:12.6}{:12.6}",
        natoms_signed, origin[0], origin[1], origin[2]
    )
    .map_err(MolRsError::Io)?;

    // Voxel axis lines: voxel_vec = cell_col / N (in writer's unit).
    let dims = [nx, ny, nz];
    for i in 0..3 {
        let n = dims[i];
        let n_signed: i32 = if i == 0 && is_angstrom {
            -(n as i32)
        } else {
            n as i32
        };
        let vx = (h[[0, i]] * unit_scale) / n as f64;
        let vy = (h[[1, i]] * unit_scale) / n as f64;
        let vz = (h[[2, i]] * unit_scale) / n as f64;
        writeln!(writer, "{:5}{:12.6}{:12.6}{:12.6}", n_signed, vx, vy, vz)
            .map_err(MolRsError::Io)?;
    }

    // Atom lines (positions in writer's unit).
    let atom_x = atoms.get("x").and_then(|c| c.as_float());
    let atom_y = atoms.get("y").and_then(|c| c.as_float());
    let atom_z = atoms.get("z").and_then(|c| c.as_float());
    let atom_z_num = atoms.get("atomic_number").and_then(|c| c.as_uint());
    let atom_charge = atoms.get("charge").and_then(|c| c.as_float());
    let atom_symbol = atoms.get("element").and_then(|c| c.as_string());

    for i in 0..n_atoms {
        let z_num = atom_z_num
            .map(|a| a[[i]])
            .or_else(|| atom_symbol.and_then(|s| Element::by_symbol(&s[[i]]).map(|e| e.z() as Idx)))
            .unwrap_or(0);
        let charge = atom_charge.map(|a| a[[i]]).unwrap_or(z_num as F);
        let x = atom_x.map(|a| a[[i]]).unwrap_or(0.0) * unit_scale as F;
        let y = atom_y.map(|a| a[[i]]).unwrap_or(0.0) * unit_scale as F;
        let z = atom_z.map(|a| a[[i]]).unwrap_or(0.0) * unit_scale as F;
        writeln!(
            writer,
            "{:5}{:12.6}{:12.6}{:12.6}{:12.6}",
            z_num, charge, x, y, z
        )
        .map_err(MolRsError::Io)?;
    }

    // MO header line
    if let Some(ref indices) = mo_indices {
        let parts: Vec<String> = std::iter::once(indices.len().to_string())
            .chain(indices.iter().map(|i| format!("{:4}", i)))
            .collect();
        writeln!(writer, "{}", parts.join("  ")).map_err(MolRsError::Io)?;
    }

    // Volumetric data
    let n_voxels = nx * ny * nz;
    if let Some(ref indices) = mo_indices {
        // Interleave MO data, deinterleaved on read into one column per orbital.
        let n_vals = indices.len();
        let columns: Vec<Vec<F>> = indices
            .iter()
            .map(|idx| {
                grid_block
                    .get(&format!("mo_{}", idx))
                    .and_then(|c| c.as_float())
                    .map(|a| a.iter().copied().collect::<Vec<_>>())
                    .unwrap_or_default()
            })
            .collect();
        let mut col = 0;
        for i in 0..n_voxels {
            for (k, arr) in columns.iter().enumerate() {
                let val = arr.get(i).copied().unwrap_or(0.0);
                write!(writer, "{:13.5E}", val).map_err(MolRsError::Io)?;
                col += 1;
                if col % 6 == 0 || (i == n_voxels - 1 && k == n_vals - 1) {
                    writeln!(writer).map_err(MolRsError::Io)?;
                    col = 0;
                }
            }
        }
        if col != 0 {
            writeln!(writer).map_err(MolRsError::Io)?;
        }
    } else {
        // Single density field
        let data = grid_block
            .get("density")
            .and_then(|c| c.as_float())
            .ok_or_else(|| MolRsError::validation("cube grid has no 'density' column"))?;
        let mut col = 0;
        for v in data.iter() {
            write!(writer, "{:13.5E}", v).map_err(MolRsError::Io)?;
            col += 1;
            if col % 6 == 0 {
                writeln!(writer).map_err(MolRsError::Io)?;
                col = 0;
            }
        }
        if col != 0 {
            writeln!(writer).map_err(MolRsError::Io)?;
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn parse_f64(s: &str, line_no: usize) -> Result<f64, MolRsError> {
    s.parse::<f64>()
        .map_err(|_| MolRsError::parse_error(line_no, format!("expected float, got '{}'", s)))
}

/// Read `count` floating-point values from `reader`, ignoring line structure.
fn read_volumetric_data<R: BufRead>(
    reader: &mut R,
    count: usize,
    line_no: &mut usize,
) -> Result<Vec<F>, MolRsError> {
    let mut values = Vec::with_capacity(count);
    // Reuse one buffer across the (potentially millions of) data lines instead
    // of allocating a fresh String per line.
    let mut line = String::new();
    while values.len() < count {
        line.clear();
        let bytes = reader.read_line(&mut line).map_err(MolRsError::Io)?;
        *line_no += 1;
        if bytes == 0 {
            return Err(MolRsError::parse_error(
                *line_no,
                format!(
                    "unexpected EOF while reading volumetric data (got {}/{} values)",
                    values.len(),
                    count
                ),
            ));
        }
        for tok in line.split_whitespace() {
            if values.len() >= count {
                break;
            }
            let v = tok.parse::<f64>().map_err(|_| {
                MolRsError::parse_error(
                    *line_no,
                    format!("expected float in volumetric data, got '{}'", tok),
                )
            })?;
            values.push(v as F);
        }
    }
    Ok(values)
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    #[test]
    fn parse_f64_valid() {
        assert!((super::parse_f64("1.23E+02", 1).unwrap() - 123.0).abs() < 1e-10);
        assert!((super::parse_f64("-0.5", 1).unwrap() - (-0.5)).abs() < 1e-10);
    }

    #[test]
    fn parse_f64_invalid() {
        assert!(super::parse_f64("abc", 1).is_err());
    }
}
