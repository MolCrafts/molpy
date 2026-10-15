//! Python bindings for `molrs::io` (`molrs.io`): every file reader and
//! writer, one Rust door per Python door, the same name.
//!
//! | binding                 | what                                                     |
//! |-------------------------|----------------------------------------------------------|
//! | [`structure`]           | one-frame doors: PDB, XYZ, GRO, SDF, MOL2, CIF, XSF, cube, VASP, LAMMPS data / molecule, AMBER |
//! | [`in_memory`]           | the frame formats' `read_<fmt>_str` / `_bytes` and `write_<fmt>_str` / `_bytes` |
//! | [`trajectory_readers`]  | `read_<fmt>_trajectory` and the lazy `<Fmt>Reader` classes; the trajectory writers |
//! | [`stl`]                 | STL                                                      |
//! | [`csv`]                 | a `Block` as CSV                                         |
//! | [`clpol`]               | the CL&Pol `alpha.ff` polarisation table                 |
//! | [`forcefield`]          | force-field files                                        |
//! | [`frame_encoding`]      | a frame as MessagePack bytes / JSON text                 |
//! | [`smiles`], [`cgsmiles`] | the SMILES and CGsmiles line notations                  |
//! | [`lammps_log`]          | LAMMPS log files                                         |
//! | [`lammps_bond_react`]   | LAMMPS `fix bond/react` templates                        |
//! | [`mrec`]                | `*.mrec` scientific records                              |
//!
//! Every factory has one shape: a function flat on `molrs.io`
//! (`read_<fmt>[_<what>]` / `write_<fmt>[_<what>]`, `_str` / `_bytes` for
//! memory), or a class of the format's own submodule
//! (`molrs.io.<fmt>.<Fmt>Reader`). No door picks a format for the caller.

pub mod cgsmiles;
mod clpol;
mod csv;
mod forcefield;
mod frame_encoding;
mod in_memory;
mod json_to_py;
pub mod lammps_bond_react;
// The log parser is `std::fs` plus an in-memory door. It does not use the
// zarr `filesystem` feature, so it is part of every build, including Pyodide.
mod lammps_log;
// `*.mrec` is a zarr store. It needs `fs` (`molrs/filesystem`) and is absent
// from a `--no-default-features` Pyodide build.
#[cfg(feature = "fs")]
pub mod mrec;
pub(crate) mod smiles;
mod stl;
mod structure;
mod trajectory_readers;

use pyo3::prelude::*;

/// Register every `molrs::io` binding on the native module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    structure::register(m)?;
    in_memory::register(m)?;
    trajectory_readers::register(m)?;
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
        crate::add_submodule(m, "mrec", "molrs.io.mrec", mrec::register)?;
    }
    Ok(())
}
