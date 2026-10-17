//! VASP CHGCAR / CHGDIF volumetric data file reader.

use std::io::BufRead;

use crate::io::Source;

use ndarray::Array1;

use super::header::{expand_symbols, parse_usize_vec, read_coords, read_header};
use molrs::core::Block;
use molrs::core::Frame;
use molrs::core::MolRsError;
use molrs::core::keys;
use molrs::op::F;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Read a CHGCAR (or CHGDIF) file.
///
/// Returns a [`Frame`] with:
/// - `"atoms"` block: `element` (str, the raw species symbols of the header;
///   absent for a VASP4 header — no `atomic_number` is synthesized),
///   `x`/`y`/`z` (float, Å Cartesian)
/// - `simbox`: triclinic periodic box derived from the POSCAR header
/// - `"grid"` block: a [`Block`] of shape `[nx, ny, nz]` carrying the `"total"`
///   column and, for spin-polarized files, `"diff"`
///
/// # File layout
///
/// ```text
/// <comment>                        ← system name → frame.meta["title"]
/// <scale>                          ← uniform scaling factor
/// <a1x> <a1y> <a1z>               ← lattice vector a1 (Å after scaling)
/// <a2x> <a2y> <a2z>               ← lattice vector a2
/// <a3x> <a3y> <a3z>               ← lattice vector a3
/// <Elem1> <Elem2> …               ← element symbols
/// <n1> <n2> …                     ← element counts
/// Direct | Cartesian               ← coordinate mode
/// <s1> <s2> <s3>                  ← one atom per line
/// …
///                                  ← blank line
/// <nx> <ny> <nz>                  ← grid dimensions
/// <val> <val> …                   ← nx*ny*nz values, 5 per line, VASP column-major (x fastest)
///                                  ← optional: augmentation occupancies (skipped)
///                                  ← optional: blank line + nx ny nz + spin data
/// ```
///
/// # Grid stored in Frame
///
/// The returned [`Frame`] carries a `"grid"` [`Block`] with `set_shape([nx,
/// ny, nz])` and one f64 column per scalar field:
///
/// | Column     | Content                                   | Always? |
/// |------------|-------------------------------------------|---------|
/// | `"total"`  | Total charge density (raw: ρ·V_cell, e)  | yes     |
/// | `"diff"`   | Spin density α−β (raw: ρ·V_cell, e)      | ISPIN=2 |
///
/// The values are stored **as-is** from the file (ρ × V_cell).
/// To convert to charge density in e/Å³: divide by `simbox.volume()`.
///
/// # Grid axis convention
///
/// The VASP/FORTRAN data is x-fastest (column-major).
/// `read_chgcar` converts to C row-major `(ix, iy, iz)` order so that
/// the column data at index `ix*ny*nz + iy*nz + iz` is `ρ(ix, iy, iz) × V`.
///
/// # Atom positions
///
/// Atom positions are returned in Cartesian Å.  If the file uses `Direct`
/// coordinates they are multiplied by the lattice matrix on read.
///
/// # Errors
///
/// The frame doors' error: the OS error of a path that cannot be opened,
/// [`InvalidData`](std::io::ErrorKind::InvalidData) for a file that is not a
/// CHGCAR, its message naming why.
pub fn read_chgcar<'a>(src: impl Into<Source<'a>>) -> std::io::Result<Frame> {
    read_frame_from(src.into().reader()?).map_err(|e| match e {
        MolRsError::Io(e) => e,
        other => crate::io::invalid_data(other),
    })
}

/// Read a CHGCAR from any [`BufRead`] source.
fn read_frame_from<R: BufRead>(mut reader: R) -> Result<Frame, MolRsError> {
    let mut line_no = 0usize;

    macro_rules! next_line {
        () => {{
            let mut s = String::new();
            reader.read_line(&mut s).map_err(|e| MolRsError::Io(e))?;
            line_no += 1;
            s
        }};
    }

    macro_rules! parse_err {
        ($msg:expr) => {
            MolRsError::parse_error(line_no, $msg)
        };
    }

    // -----------------------------------------------------------------------
    // Header and atoms (the POSCAR preamble)
    // -----------------------------------------------------------------------
    let header = read_header(&mut reader, &mut line_no).map_err(MolRsError::Io)?;
    let n_atoms = header.total_atoms();
    let (frac_x, frac_y, frac_z, _) = read_coords(
        &mut reader,
        n_atoms,
        &mut line_no,
        header.selective_dynamics,
    )
    .map_err(MolRsError::Io)?;
    let simbox = header.simbox().map_err(MolRsError::Io)?;
    let (cart_x, cart_y, cart_z) = header.cartesian(&simbox, &frac_x, &frac_y, &frac_z);
    let title = header.title;
    let atom_syms = expand_symbols(&header.symbols, &header.counts);

    let mut atoms = Block::new();
    atoms
        .insert("x", Array1::from_vec(cart_x).into_dyn(), None)
        .map_err(MolRsError::Block)?;
    atoms
        .insert("y", Array1::from_vec(cart_y).into_dyn(), None)
        .map_err(MolRsError::Block)?;
    atoms
        .insert("z", Array1::from_vec(cart_z).into_dyn(), None)
        .map_err(MolRsError::Block)?;
    if !atom_syms.is_empty() {
        atoms
            .insert(keys::ELEMENT, Array1::from_vec(atom_syms).into_dyn(), None)
            .map_err(MolRsError::Block)?;
    }

    // -----------------------------------------------------------------------
    // Skip blank line(s) before grid header
    // -----------------------------------------------------------------------
    skip_blank_lines(&mut reader, &mut line_no)?;

    // -----------------------------------------------------------------------
    // Grid header: nx ny nz
    // -----------------------------------------------------------------------
    let dim_line = next_line!();
    let dims = parse_usize_vec(&dim_line, line_no).map_err(MolRsError::Io)?;
    if dims.len() < 3 {
        return Err(parse_err!("expected 'nx ny nz' grid dimensions"));
    }
    let [nx, ny, nz] = [dims[0], dims[1], dims[2]];
    let n_voxels = nx * ny * nz;

    // -----------------------------------------------------------------------
    // Build the volumetric Block ("grid"): row-major (z fastest),
    // shape = [nx, ny, nz]. Spin and SOC channels are stored as additional
    // f64 columns alongside "total".
    // -----------------------------------------------------------------------
    let mut grid_block = Block::new();

    // Total charge density: VASP column-major → row-major.
    let total_vasp = read_volumetric_data(&mut reader, n_voxels, &mut line_no)?;
    let total = vasp_to_row_major(total_vasp, nx, ny, nz);
    grid_block
        .insert("total", Array1::from_vec(total).into_dyn(), None)
        .map_err(MolRsError::Block)?;

    // Optional: augmentation occupancies (skip) + spin density.
    if let Some(diff) = try_read_optional_grid(&mut reader, n_voxels, nx, ny, nz, &mut line_no)? {
        grid_block
            .insert("diff", Array1::from_vec(diff).into_dyn(), None)
            .map_err(MolRsError::Block)?;
    }

    grid_block
        .set_shape(&[nx, ny, nz])
        .map_err(MolRsError::Block)?;

    // -----------------------------------------------------------------------
    // Assemble Frame
    // -----------------------------------------------------------------------
    let mut frame = Frame::new();
    if !title.is_empty() {
        frame.meta.insert("title", title);
    }
    frame.simbox = Some(simbox);
    frame.insert("atoms", atoms);
    frame.insert("grid", grid_block);

    Ok(frame)
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Read `n_voxels` floating-point values from `reader`, ignoring line structure.
fn read_volumetric_data<R: BufRead>(
    reader: &mut R,
    n_voxels: usize,
    line_no: &mut usize,
) -> Result<Vec<F>, MolRsError> {
    let mut values = Vec::with_capacity(n_voxels);
    while values.len() < n_voxels {
        let mut line = String::new();
        let bytes = reader.read_line(&mut line).map_err(MolRsError::Io)?;
        *line_no += 1;
        if bytes == 0 {
            return Err(MolRsError::parse_error(
                *line_no,
                format!(
                    "unexpected EOF while reading grid data (got {}/{} values)",
                    values.len(),
                    n_voxels
                ),
            ));
        }
        for tok in line.split_whitespace() {
            if values.len() >= n_voxels {
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

/// Convert VASP column-major (x fastest: flat index = ix + nx*iy + nx*ny*iz)
/// to our row-major (z fastest: flat index = ix*ny*nz + iy*nz + iz).
fn vasp_to_row_major(vasp: Vec<F>, nx: usize, ny: usize, nz: usize) -> Vec<F> {
    let n = nx * ny * nz;
    let mut out = vec![0.0 as F; n];
    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                let src = ix + nx * iy + nx * ny * iz;
                let dst = ix * ny * nz + iy * nz + iz;
                out[dst] = vasp[src];
            }
        }
    }
    out
}

/// Skip blank lines and lines that consist only of whitespace.
fn skip_blank_lines<R: BufRead>(reader: &mut R, line_no: &mut usize) -> Result<(), MolRsError> {
    loop {
        // Peek by filling the buffer without consuming
        let buf = reader.fill_buf().map_err(MolRsError::Io)?;
        if buf.is_empty() {
            return Ok(()); // EOF
        }
        // Find first newline
        let newline_pos = buf.iter().position(|&b| b == b'\n');
        let line_is_blank = match newline_pos {
            None => buf.iter().all(|&b| b == b' ' || b == b'\t' || b == b'\r'),
            Some(pos) => buf[..pos]
                .iter()
                .all(|&b| b == b' ' || b == b'\t' || b == b'\r'),
        };
        if !line_is_blank {
            return Ok(());
        }
        // Consume the blank line
        let consume = newline_pos.map_or(buf.len(), |p| p + 1);
        reader.consume(consume);
        *line_no += 1;
    }
}

/// After the total charge density block, try to read an optional spin-density block.
///
/// Skips any "augmentation occupancies" lines, then looks for another `nx ny nz`
/// header followed by volumetric data.
///
/// Returns `Ok(Some(data))` if a second grid was found, `Ok(None)` otherwise.
fn try_read_optional_grid<R: BufRead>(
    reader: &mut R,
    n_voxels: usize,
    nx: usize,
    ny: usize,
    nz: usize,
    line_no: &mut usize,
) -> Result<Option<Vec<F>>, MolRsError> {
    // Skip blank lines and augmentation occupancy blocks
    loop {
        skip_blank_lines(reader, line_no)?;

        // Peek at the next non-blank line
        let buf = reader.fill_buf().map_err(MolRsError::Io)?;
        if buf.is_empty() {
            return Ok(None);
        }

        // Read the line to inspect it
        let mut line = String::new();
        reader.read_line(&mut line).map_err(MolRsError::Io)?;
        *line_no += 1;

        let trimmed = line.trim().to_ascii_lowercase();

        if trimmed.starts_with("augmentation") {
            // Skip this line and any following data lines (all-numeric)
            // until we hit another blank or "augmentation" or the dim line
            loop {
                skip_blank_lines(reader, line_no)?;
                let buf = reader.fill_buf().map_err(MolRsError::Io)?;
                if buf.is_empty() {
                    return Ok(None);
                }
                // Peek: is the next line a dim line, aug line, or data?
                let mut peek = String::new();
                reader.read_line(&mut peek).map_err(MolRsError::Io)?;
                *line_no += 1;
                let peek_trim = peek.trim().to_ascii_lowercase();
                if peek_trim.starts_with("augmentation") {
                    continue; // another aug block
                }
                if peek_trim.is_empty() {
                    break; // blank → back to outer loop
                }
                // If this line contains only integers that match nx ny nz, it's the dim header
                if is_dim_line(&peek, nx, ny, nz) {
                    let diff_vasp = read_volumetric_data(reader, n_voxels, line_no)?;
                    return Ok(Some(vasp_to_row_major(diff_vasp, nx, ny, nz)));
                }
                // Otherwise it's data we're skipping
            }
            continue;
        }

        // Check if this is a dim line matching nx ny nz
        if is_dim_line(&line, nx, ny, nz) {
            let diff_vasp = read_volumetric_data(reader, n_voxels, line_no)?;
            return Ok(Some(vasp_to_row_major(diff_vasp, nx, ny, nz)));
        }

        // Something else → no second grid
        return Ok(None);
    }
}

fn is_dim_line(line: &str, nx: usize, ny: usize, nz: usize) -> bool {
    let toks: Vec<&str> = line.split_whitespace().collect();
    if toks.len() < 3 {
        return false;
    }
    toks[0].parse::<usize>().ok() == Some(nx)
        && toks[1].parse::<usize>().ok() == Some(ny)
        && toks[2].parse::<usize>().ok() == Some(nz)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[allow(clippy::identity_op, clippy::erasing_op)]
    fn vasp_to_row_major_reorder() {
        // VASP order (x fastest): flat[k] at k = ix + nx*iy + nx*ny*iz
        // Our order (z fastest):  flat[k] at k = ix*ny*nz + iy*nz + iz
        // With nx=ny=nz=2, values 1..8:
        // VASP flat: k=0 → (0,0,0), k=1 → (1,0,0), k=2 → (0,1,0), k=3 → (1,1,0)
        //            k=4 → (0,0,1), k=5 → (1,0,1), k=6 → (0,1,1), k=7 → (1,1,1)
        let vasp: Vec<F> = (1..=8).map(|v| v as F).collect();
        let row_major = vasp_to_row_major(vasp, 2, 2, 2);
        // (ix=0,iy=0,iz=0) → VASP k=0 → value 1 → row-major dst=0*4+0*2+0=0
        assert_eq!(row_major[0 * 4 + 0 * 2 + 0], 1.0); // (0,0,0)
        assert_eq!(row_major[1 * 4 + 0 * 2 + 0], 2.0); // (1,0,0)
        assert_eq!(row_major[0 * 4 + 1 * 2 + 0], 3.0); // (0,1,0)
        assert_eq!(row_major[0 * 4 + 0 * 2 + 1], 5.0); // (0,0,1)
    }
}
