//! The in-memory doors of `molrs.io`: every frame format's
//! `read_<fmt>_str` / `write_<fmt>_str` (text) and `read_<fmt>_bytes` /
//! `write_<fmt>_bytes` (binary trajectories, and the byte-window readers the
//! chunked streams use) — each one its path twin of [`super::structure`] /
//! [`super::trajectory_readers`] on memory, the same Rust door underneath and
//! the same name in Rust, Python and JS (camelCased there).

use std::collections::BTreeMap;

use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};

use crate::core::frame::PyFrame;
use crate::error::{io_error_to_pyerr, molrs_error_to_pyerr};

use super::structure::{prep_residue_to_pydict, py_to_prep_residue};

/// A `read_<fmt>_str` / `read_<fmt>_bytes` door: `$input` in, a frame out.
macro_rules! read_door {
    ($(#[$doc:meta])* $name:ident($input:ident: $ty:ty) => $rs:path, $err:path) => {
        $(#[$doc])*
        #[pyfunction]
        pub fn $name($input: $ty) -> PyResult<PyFrame> {
            PyFrame::from_core_frame($rs($input).map_err($err)?)
        }
    };
}

/// A `write_<fmt>_str` door: a frame in, text out.
macro_rules! write_str_door {
    ($(#[$doc:meta])* $name:ident => $rs:path, $err:path) => {
        $(#[$doc])*
        #[pyfunction]
        pub fn $name(frame: &PyFrame) -> PyResult<String> {
            frame.with_frame(|f| $rs(f).map_err($err))?
        }
    };
}

/// A `write_<fmt>_bytes` door: a frame in, a one-frame file's bytes out.
macro_rules! write_bytes_door {
    ($(#[$doc:meta])* $name:ident => $rs:path) => {
        $(#[$doc])*
        #[pyfunction]
        pub fn $name<'py>(py: Python<'py>, frame: &PyFrame) -> PyResult<Bound<'py, PyBytes>> {
            let bytes = frame.with_frame(|f| $rs(f).map_err(io_error_to_pyerr))??;
            Ok(PyBytes::new(py, &bytes))
        }
    };
}

read_door!(
    /// Read the first frame of PDB text — :func:`read_pdb` on a string.
    read_pdb_str(text: &str) => molrs::io::read_pdb_str, io_error_to_pyerr
);
read_door!(
    /// Read one PDB frame from bytes (a ``MODEL`` block or a whole file).
    read_pdb_bytes(data: &[u8]) => molrs::io::read_pdb_bytes, io_error_to_pyerr
);
write_str_door!(
    /// Write a Frame as PDB text — :func:`write_pdb` into a string.
    write_pdb_str => molrs::io::write_pdb_str, io_error_to_pyerr
);
read_door!(
    /// Read the first frame of (extended) XYZ text — :func:`read_xyz` on a string.
    read_xyz_str(text: &str) => molrs::io::read_xyz_str, io_error_to_pyerr
);
read_door!(
    /// Read one (extended) XYZ frame from bytes.
    read_xyz_bytes(data: &[u8]) => molrs::io::read_xyz_bytes, io_error_to_pyerr
);
write_str_door!(
    /// Write a Frame as (extended) XYZ text — :func:`write_xyz` into a string.
    write_xyz_str => molrs::io::write_xyz_str, io_error_to_pyerr
);
read_door!(
    /// Read the first record of SDF / MDL molfile text — :func:`read_sdf` on a string.
    read_sdf_str(text: &str) => molrs::io::read_sdf_str, io_error_to_pyerr
);
read_door!(
    /// Read one SDF record from bytes.
    read_sdf_bytes(data: &[u8]) => molrs::io::read_sdf_bytes, io_error_to_pyerr
);
read_door!(
    /// Read the first frame of GRO text — :func:`read_gro` on a string (nm → Å).
    read_gro_str(text: &str) => molrs::io::read_gro_str, io_error_to_pyerr
);
write_str_door!(
    /// Write a Frame as GRO text — :func:`write_gro` into a string (Å → nm).
    write_gro_str => molrs::io::write_gro_str, io_error_to_pyerr
);
read_door!(
    /// Read the first molecule of MOL2 text — :func:`read_mol2` on a string.
    read_mol2_str(text: &str) => molrs::io::read_mol2_str, io_error_to_pyerr
);
write_str_door!(
    /// Write a Frame as MOL2 text — :func:`write_mol2` into a string.
    write_mol2_str => molrs::io::write_mol2_str, io_error_to_pyerr
);
read_door!(
    /// Read the first ``data_`` block of CIF text — :func:`read_cif` on a string.
    read_cif_str(text: &str) => molrs::io::read_cif_str, io_error_to_pyerr
);
write_str_door!(
    /// Write a Frame as CIF text — :func:`write_cif` into a string.
    write_cif_str => molrs::io::write_cif_str, io_error_to_pyerr
);
read_door!(
    /// Read XSF text — :func:`read_xsf` on a string.
    read_xsf_str(text: &str) => molrs::io::read_xsf_str, io_error_to_pyerr
);
write_str_door!(
    /// Write a Frame as XSF text — :func:`write_xsf` into a string.
    write_xsf_str => molrs::io::write_xsf_str, io_error_to_pyerr
);
read_door!(
    /// Read Gaussian cube text — :func:`read_cube` on a string.
    read_cube_str(text: &str) => molrs::io::read_cube_str, molrs_error_to_pyerr
);
write_str_door!(
    /// Write a Frame as Gaussian cube text — :func:`write_cube` into a string.
    write_cube_str => molrs::io::write_cube_str, molrs_error_to_pyerr
);
read_door!(
    /// Read VASP POSCAR / CONTCAR text — :func:`read_poscar` on a string.
    read_poscar_str(text: &str) => molrs::io::read_poscar_str, io_error_to_pyerr
);
write_str_door!(
    /// Write a Frame as VASP POSCAR text — :func:`write_poscar` into a string.
    write_poscar_str => molrs::io::write_poscar_str, io_error_to_pyerr
);
read_door!(
    /// Read VASP CHGCAR / CHGDIF text — :func:`read_chgcar` on a string.
    read_chgcar_str(text: &str) => molrs::io::read_chgcar_str, molrs_error_to_pyerr
);
read_door!(
    /// Read antechamber ``.ac`` text — :func:`read_ac` on a string.
    read_ac_str(text: &str) => molrs::io::read_ac_str, io_error_to_pyerr
);
read_door!(
    /// Read AMBER prmtop structure text — :func:`read_prmtop` on a string.
    read_prmtop_str(text: &str) => molrs::io::read_prmtop_str, io_error_to_pyerr
);
read_door!(
    /// Read one LAMMPS data frame from bytes.
    read_lammps_data_bytes(data: &[u8]) => molrs::io::read_lammps_data_bytes, io_error_to_pyerr
);
read_door!(
    /// Read the first snapshot of LAMMPS dump text.
    read_lammps_dump_str(text: &str) => molrs::io::read_lammps_dump_str, io_error_to_pyerr
);
read_door!(
    /// Read one LAMMPS dump snapshot from bytes.
    read_lammps_dump_bytes(data: &[u8]) => molrs::io::read_lammps_dump_bytes, io_error_to_pyerr
);
read_door!(
    /// Read one GROMACS TRR frame from bytes (nm → Å).
    read_trr_bytes(data: &[u8]) => molrs::io::read_trr_bytes, io_error_to_pyerr
);
read_door!(
    /// Read one GROMACS XTC frame from bytes (nm → Å).
    read_xtc_bytes(data: &[u8]) => molrs::io::read_xtc_bytes, io_error_to_pyerr
);
write_bytes_door!(
    /// Write a Frame as a one-frame DCD file's bytes — :func:`write_dcd_trajectory`
    /// of ``[frame]`` into memory.
    write_dcd_bytes => molrs::io::write_dcd_bytes
);
write_bytes_door!(
    /// Write a Frame as a one-frame TRR file's bytes (Å → nm) —
    /// :func:`write_trr_trajectory` of ``[frame]`` into memory.
    write_trr_bytes => molrs::io::write_trr_bytes
);
write_bytes_door!(
    /// Write a Frame as a one-frame XTC file's bytes (Å → nm) —
    /// :func:`write_xtc_trajectory` of ``[frame]`` into memory.
    write_xtc_bytes => molrs::io::write_xtc_bytes
);

/// Read one DCD frame from bytes: a whole one-frame file, or a frame body
/// with the ``decoder_state`` (the header) it is decoded with.
#[pyfunction]
#[pyo3(signature = (data, decoder_state = None))]
pub fn read_dcd_bytes(data: &[u8], decoder_state: Option<&[u8]>) -> PyResult<PyFrame> {
    PyFrame::from_core_frame(
        molrs::io::read_dcd_bytes(data, decoder_state).map_err(io_error_to_pyerr)?,
    )
}

/// Read AMBER inpcrd / restrt text — :func:`read_inpcrd` on a string,
/// ``frame`` as there.
#[pyfunction]
#[pyo3(signature = (text, frame = None))]
pub fn read_inpcrd_str<'py>(
    py: Python<'py>,
    text: &str,
    frame: Option<Bound<'py, PyFrame>>,
) -> PyResult<Bound<'py, PyAny>> {
    let coordinates = molrs::io::read_inpcrd_str(text).map_err(io_error_to_pyerr)?;
    match frame {
        None => Ok(Bound::new(py, PyFrame::from_core_frame(coordinates)?)?.into_any()),
        Some(target) => {
            target
                .borrow()
                .with_frame_mut(|f| molrs::io::amber::merge_inpcrd(f, coordinates))?
                .map_err(io_error_to_pyerr)?;
            Ok(target.into_any())
        }
    }
}

/// Read Amber prep text into a nested dict — :func:`read_prep` on a string.
#[pyfunction]
pub fn read_prep_str<'py>(py: Python<'py>, text: &str) -> PyResult<Bound<'py, PyDict>> {
    let res = molrs::io::read_prep_str(text).map_err(io_error_to_pyerr)?;
    prep_residue_to_pydict(py, &res)
}

/// Write an Amber prep residue (a nested dict) as text —
/// :func:`write_prep` into a string.
#[pyfunction]
pub fn write_prep_str(residue: &Bound<'_, PyAny>) -> PyResult<String> {
    Ok(molrs::io::write_prep_str(&py_to_prep_residue(residue)?))
}

/// Read LAMMPS data text — :func:`read_lammps_data` on a string,
/// ``atom_style`` as there.
#[pyfunction]
#[pyo3(signature = (text, atom_style = None))]
pub fn read_lammps_data_str(text: &str, atom_style: Option<&str>) -> PyResult<PyFrame> {
    use molrs::io::lammps::LammpsDataReader;
    use molrs::io::reader::FrameReader;
    let frame = match atom_style {
        None => molrs::io::read_lammps_data_str(text).map_err(io_error_to_pyerr)?,
        Some(style) => LammpsDataReader::new(std::io::Cursor::new(text.as_bytes()))
            .with_atom_style(style)
            .map_err(io_error_to_pyerr)?
            .read()
            .map_err(io_error_to_pyerr)?
            .ok_or_else(|| pyo3::exceptions::PyIOError::new_err("no frame in LAMMPS data text"))?,
    };
    PyFrame::from_core_frame(frame)
}

/// Write a Frame as LAMMPS data text — :func:`write_lammps_data` into a
/// string, ``type_labels``, ``atom_style`` and ``units`` as there.
#[pyfunction]
#[pyo3(signature = (frame, *, type_labels = None, atom_style = None, units = None))]
pub fn write_lammps_data_str(
    frame: &PyFrame,
    type_labels: Option<BTreeMap<String, Vec<String>>>,
    atom_style: Option<String>,
    units: Option<String>,
) -> PyResult<String> {
    let options = super::structure::lammps_data_options(atom_style, units)?;
    match type_labels {
        None => frame.with_frame(|f| {
            molrs::io::write_lammps_data_str_with(f, &options).map_err(io_error_to_pyerr)
        })?,
        Some(extra) => {
            let work = super::structure::with_declared_labels(frame, &extra)?;
            molrs::io::write_lammps_data_str_with(&work, &options).map_err(io_error_to_pyerr)
        }
    }
}

/// Write a Frame as one LAMMPS dump snapshot in a string — one frame of
/// :func:`write_lammps_dump_trajectory`, ``columns`` as there.
#[pyfunction]
#[pyo3(signature = (frame, columns = None))]
pub fn write_lammps_dump_str(frame: &PyFrame, columns: Option<Vec<String>>) -> PyResult<String> {
    let chosen: Option<Vec<&str>> = columns
        .as_ref()
        .map(|c| c.iter().map(String::as_str).collect());
    frame.with_frame(|f| {
        molrs::io::write_lammps_dump_str(f, chosen.as_deref()).map_err(io_error_to_pyerr)
    })?
}

/// Register this module's functions on `molrs.io`.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    for door in [
        wrap_pyfunction!(read_pdb_str, m)?,
        wrap_pyfunction!(read_pdb_bytes, m)?,
        wrap_pyfunction!(write_pdb_str, m)?,
        wrap_pyfunction!(read_xyz_str, m)?,
        wrap_pyfunction!(read_xyz_bytes, m)?,
        wrap_pyfunction!(write_xyz_str, m)?,
        wrap_pyfunction!(read_sdf_str, m)?,
        wrap_pyfunction!(read_sdf_bytes, m)?,
        wrap_pyfunction!(read_gro_str, m)?,
        wrap_pyfunction!(write_gro_str, m)?,
        wrap_pyfunction!(read_mol2_str, m)?,
        wrap_pyfunction!(write_mol2_str, m)?,
        wrap_pyfunction!(read_cif_str, m)?,
        wrap_pyfunction!(write_cif_str, m)?,
        wrap_pyfunction!(read_xsf_str, m)?,
        wrap_pyfunction!(write_xsf_str, m)?,
        wrap_pyfunction!(read_cube_str, m)?,
        wrap_pyfunction!(write_cube_str, m)?,
        wrap_pyfunction!(read_poscar_str, m)?,
        wrap_pyfunction!(write_poscar_str, m)?,
        wrap_pyfunction!(read_chgcar_str, m)?,
        wrap_pyfunction!(read_ac_str, m)?,
        wrap_pyfunction!(read_inpcrd_str, m)?,
        wrap_pyfunction!(read_prmtop_str, m)?,
        wrap_pyfunction!(read_prep_str, m)?,
        wrap_pyfunction!(write_prep_str, m)?,
        wrap_pyfunction!(read_lammps_data_str, m)?,
        wrap_pyfunction!(read_lammps_data_bytes, m)?,
        wrap_pyfunction!(write_lammps_data_str, m)?,
        wrap_pyfunction!(read_lammps_dump_str, m)?,
        wrap_pyfunction!(read_lammps_dump_bytes, m)?,
        wrap_pyfunction!(write_lammps_dump_str, m)?,
        wrap_pyfunction!(read_dcd_bytes, m)?,
        wrap_pyfunction!(write_dcd_bytes, m)?,
        wrap_pyfunction!(read_trr_bytes, m)?,
        wrap_pyfunction!(write_trr_bytes, m)?,
        wrap_pyfunction!(read_xtc_bytes, m)?,
        wrap_pyfunction!(write_xtc_bytes, m)?,
    ] {
        crate::add_function(m, "molrs.io", door)?;
    }
    Ok(())
}
