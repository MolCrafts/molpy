//! LAMMPS's files: the data file, molecule templates, dump trajectories,
//! `fix bond/react` maps, run logs, and force-field coefficient includes
//! (`*.ff`) with `fix cmap` grids.
//!
//! The doors are functions of [`crate::io`]:
//!
//! | file | doors |
//! |---|---|
//! | data file | [`read_lammps_data`](crate::io::read_lammps_data), [`write_lammps_data`](crate::io::write_lammps_data), [`LammpsDataWriter`] for explicit style and units |
//! | molecule template | [`read_lammps_molecule`](crate::io::read_lammps_molecule), [`read_lammps_molecule_json`](crate::io::read_lammps_molecule_json), [`write_lammps_molecule`](crate::io::write_lammps_molecule), [`write_lammps_molecule_json`](crate::io::write_lammps_molecule_json) |
//! | dump | [`read_lammps_dump`](crate::io::read_lammps_dump), [`write_lammps_dump`](crate::io::write_lammps_dump), [`LammpsDumpReader::open`] / [`LammpsDumpWriter::create`] for every snapshot, [`write_lammps_dump_local`](crate::io::write_lammps_dump_local) |
//! | `fix bond/react` | [`write_lammps_bond_react_map`](crate::io::write_lammps_bond_react_map), [`write_lammps_bond_react_system`](crate::io::write_lammps_bond_react_system) |
//! | log | [`read_lammps_log`](crate::io::read_lammps_log) |
//! | force-field include `*.ff` | [`LammpsForcefieldReader`]'s `read`, [`LammpsForcefieldWriter`]'s `write` |
//! | data-file `* Coeffs` | [`LammpsForcefieldReader::read_data_coeffs`] (of the data file's frame), [`LammpsForcefieldWriter::write_data_coeffs`] |
//! | `fix cmap` grid | [`LammpsForcefieldReader::read_cmap`], [`LammpsForcefieldWriter::write_cmap`] |
//!
//! This module holds the family's classes and records: the readers and
//! writers ([`LammpsDataReader`] / [`LammpsDataWriter`] with its
//! [`LammpsDataWriteOptions`] and [`infer_lammps_atom_style`],
//! [`LammpsDumpReader`] /
//! [`LammpsDumpWriter`], [`LammpsForcefieldReader`] /
//! [`LammpsForcefieldWriter`]), the type numbering both writers share
//! ([`lammps_type_labels`]: a `drudes` row is a LAMMPS bond), the chunked indexers, the bond/react records
//! ([`BondReactTemplate`], [`BondReactSystem`], [`DroppedRows`]), the log
//! records ([`LammpsLog`] and its parts) with the log sniffer
//! [`is_lammps_log`], the `fix cmap` grid size ([`LAMMPS_CMAP_DIM`],
//! [`LAMMPS_CMAP_MAX`]), and the `units` adapter ([`parse_lammps_units_style`],
//! [`LammpsUnitConverter`]). The primitives the data and dump readers share
//! (atom-style layouts, box bounds, field parsing) are crate-private.

pub(crate) mod atom_style;
pub(crate) mod bond_react;
#[cfg(feature = "ff")]
pub(crate) mod bond_react_system;
pub(crate) mod box_bounds;
pub(crate) mod columns;
pub(crate) mod data;
pub(crate) mod data_options;
pub(crate) mod drude;
pub(crate) mod dump;
pub(crate) mod fields;
#[cfg(feature = "ff")]
pub(crate) mod forcefield_reader;
#[cfg(feature = "ff")]
pub(crate) mod forcefield_writer;
pub(crate) mod log;
pub(crate) mod molecule;
pub(crate) mod units;

pub use bond_react::BondReactTemplate;
#[cfg(feature = "ff")]
pub use bond_react_system::{BondReactSystem, DroppedRows};
pub use data::{LammpsDataIndexBuilder, LammpsDataReader, LammpsDataWriter};
pub use data_options::{LammpsDataWriteOptions, infer_lammps_atom_style};
pub use drude::lammps_type_labels;
pub use dump::{LammpsDumpIndexBuilder, LammpsDumpReader, LammpsDumpWriter};
#[cfg(feature = "ff")]
pub use forcefield_reader::{LAMMPS_CMAP_DIM, LAMMPS_CMAP_MAX, LammpsForcefieldReader};
#[cfg(feature = "ff")]
pub use forcefield_writer::{LammpsForcefieldWriter, refuse_pair_overrides};
pub use log::{
    LammpsCpuUse, LammpsLoadBalance, LammpsLog, LammpsLogHeader, LammpsLoopTime, LammpsMemoryUsage,
    LammpsNeighborStatistics, LammpsPerformance, LammpsRun, LammpsThermo, LammpsTimingBreakdown,
    LammpsTimingRow, LammpsWarning, is_lammps_log,
};
pub use units::{LammpsLjReference, LammpsUnitConverter, parse_lammps_units_style};
