//! Python bindings for `molrs::io` (`molpy.io`): every file reader and
//! writer, one Python door per Rust door, the same name.
//!
//! | binding                 | what                                                     |
//! |-------------------------|----------------------------------------------------------|
//! | [`source`]              | `molpy.io.Text`, and the source / sink every door takes   |
//! | [`structure`]           | the one-frame `read_<fmt>(source)` / `write_<fmt>(sink, frame)` doors |
//! | [`multi_frame`]         | the `<Fmt>Reader(source)` / `<Fmt>Writer(sink)` classes  |
//! | [`stl`]                 | STL                                                      |
//! | [`csv`]                 | a `Block` as CSV                                         |
//! | [`clpol`]               | the CL&Pol `alpha.ff` polarisation table                 |
//! | [`forcefield`]          | the force-field reader and writer classes                |
//! | [`frame_encoding`]      | a frame in the stream's wire encodings                   |
//! | [`smiles`], [`cgsmiles`] | the SMILES and CGsmiles line notations                  |
//! | [`lammps_log`]          | LAMMPS log files                                         |
//! | [`lammps_bond_react`]   | LAMMPS `fix bond/react` templates                        |
//! | [`mrec`]                | `*.mrec` scientific records                              |
//!
//! A reader takes a source — an ``os.PathLike``, ``bytes`` or a
//! ``molpy.io.Text``, never a bare ``str`` — and a writer a sink — an
//! ``os.PathLike`` or a ``bytearray``. No door picks a format for the caller.

pub mod cgsmiles;
mod clpol;
mod csv;
mod forcefield;
mod frame_encoding;
mod json_to_py;
pub mod lammps_bond_react;
// The log parser is `std::fs` plus an in-memory parser. It does not use the
// zarr `filesystem` feature, so it is part of every build, including Pyodide.
mod lammps_log;
// `*.mrec` is a zarr store. It needs `fs` (`molrs/filesystem`) and is absent
// from a `--no-default-features` Pyodide build.
#[cfg(feature = "fs")]
pub mod mrec;
mod multi_frame;
pub(crate) mod smiles;
mod source;
mod stl;
mod structure;

use pyo3::exceptions::{PyFileNotFoundError, PyIOError, PyValueError};
use pyo3::prelude::*;

pub(crate) use source::{PySink, PySource};

/// A door's [`std::io::Error`] as Python sees it when the door's refusals
/// are value errors (CSV, CL&Pol, force fields, wire payloads): a file the
/// format refuses is a ``ValueError``, a missing file a
/// ``FileNotFoundError``, any other OS failure an ``OSError``.
pub(crate) fn data_error(e: std::io::Error) -> PyErr {
    use std::io::ErrorKind;
    match e.kind() {
        ErrorKind::InvalidData | ErrorKind::InvalidInput | ErrorKind::Unsupported => {
            PyValueError::new_err(e.to_string())
        }
        ErrorKind::NotFound => PyFileNotFoundError::new_err(e.to_string()),
        _ => PyIOError::new_err(e.to_string()),
    }
}

/// Register every `molrs::io` binding on the native module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<source::PyText>()?;
    structure::register(m)?;
    multi_frame::register(m)?;
    stl::register(m)?;
    csv::register(m)?;
    clpol::register(m)?;
    forcefield::register(m)?;
    frame_encoding::register(m)?;
    smiles::register(m)?;
    cgsmiles::register(m)?;
    lammps_bond_react::register(m)?;
    lammps_log::register(m)?;
    #[cfg(feature = "fs")]
    {
        mrec::register_doors(m)?;
        crate::add_submodule(m, "mrec", "molpy.io.mrec", mrec::register)?;
    }
    Ok(())
}
