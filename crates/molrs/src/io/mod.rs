//! File I/O, one module per file format.
//!
//! Every format has one door per direction:
//!
//! - `read_<fmt>(src)` reads a file's first (or only) frame, `src` a
//!   [`Source`]: a path ([`Source::Path`]) or the bytes of a whole file
//!   ([`Source::Bytes`]). There is no conversion from `&str` — a string is
//!   ambiguous between a path and text — so a caller says `Path::new(p)` or
//!   `text.as_bytes()`.
//! - `write_<fmt>(sink, frame)` writes one frame as a whole file, `sink` a
//!   [`Sink`]: a path ([`Sink::Path`], created or truncated) or a buffer
//!   ([`Sink::Buffer`], cleared, then holding exactly the file's bytes).
//! - A multi-frame format's class reads and writes every frame:
//!   `io::<fmt>::<Fmt>Reader::open(src)` is a
//!   [`TrajectoryReader`](reader::TrajectoryReader) (`len`, `read_frame(i)`,
//!   `iter`), and `<Fmt>Writer::create(sink)` a
//!   [`FrameWriter`](writer::FrameWriter) that writes one frame per call. Both
//!   classes also wrap any byte stream (`new(reader)` / `new(writer)`).
//!
//! The frame doors fail with [`std::io::Error`]: `InvalidData` for a file
//! that is not the format, the OS error for a path that cannot be opened. No
//! door picks a format for the caller: every door names its format.
//!
//! | module | format |
//! |---|---|
//! | [`pdb`], [`xyz`], [`gro`], [`sdf`], [`mol2`], [`cif`] | structure files (PDB, XYZ / extended XYZ, GRO, MDL SDF, Tripos MOL2, CIF) |
//! | XSF, STL | functions only: `read_xsf` / `write_xsf`, `read_stl` |
//! | Gaussian cube | `read_cube` / `write_cube`, and [`CubeReader`] for cubes back to back |
//! | [`vasp`] | POSCAR / CONTCAR, CHGCAR |
//! | [`dcd`], [`trr`], [`xtc`] | binary trajectories (CHARMM/NAMD DCD, GROMACS TRR and XTC) |
//! | [`lammps`] | LAMMPS data, molecule, dump, log, `fix bond/react`, force-field `*.ff`, `fix cmap` |
//! | [`amber`] | AMBER prmtop, inpcrd, antechamber `.ac`, prep, frcmod |
//! | [`gromacs`] | GROMACS `.top` / `.itp` |
//! | [`openmm_xml`] | OpenMM force-field XML |
//! | force-field XML | [`XmlForcefieldReader`] / [`XmlForcefieldWriter`] — every style of a force field as one element, in its declared units |
//! | [`clpol`] | CL&Pol `alpha.ff`: `read_clpol_alpha` |
//! | [`smiles`], [`cgsmiles`] | the SMILES and CGsmiles line notations |
//! | [`mrec`] | MolRec scientific records (`*.mrec`): `write_mrec`, [`mrec::MrecFile`] |
//! | CSV | functions only: `read_csv_block` / `write_csv_block` |
//! | a frame in `stream`'s wire encodings (feature `stream`) | `read_wire_frame` / `write_wire_frame` |
//!
//! [`reader`] and [`writer`] hold the contracts the classes implement
//! ([`FrameReader`](reader::FrameReader), [`TrajectoryReader`](reader::TrajectoryReader),
//! [`FrameWriter`](writer::FrameWriter), [`ForceFieldReader`](reader::ForceFieldReader),
//! [`ForceFieldWriter`](writer::ForceFieldWriter)); [`frame_index`] the chunked
//! frame indexing the `*IndexBuilder` classes share — each indexed byte range
//! is a whole one-frame file its format's `read_<fmt>` reads.
//!
//! Force-field files map a file onto a [`ForceField`](crate::ff::forcefield::ForceField),
//! the data model `ff` owns (feature `ff`): a reader owns the translation from
//! its format — names, **and unit and factor normalization** — into the
//! force-field IR (adopts the LAMMPS standard), its writer the inverse, so unit
//! conversion stays at one boundary pair.
//!
//! A force-field format's door is its class, not a function: every one has
//! settings a function would take as an options bag (LAMMPS's type labels,
//! units and registry, GROMACS's include directories and skipped directives,
//! a precision), so the class carries them — one constructor and setters —
//! and the one operation is the trait method,
//! [`ForceFieldReader::read(src)`](reader::ForceFieldReader::read) /
//! [`ForceFieldWriter::write(sink, &ff)`](writer::ForceFieldWriter::write),
//! over the same [`Source`] / [`Sink`] as the frame doors and failing with
//! the same [`std::io::Error`] (a writer's typed refusal is a
//! [`ForceFieldWriteError`](writer::ForceFieldWriteError), which converts into
//! one). What else a format reads or writes is a method of the same class
//! with the same arguments: `read_system(src)` (GROMACS, AMBER prmtop),
//! `write_system(sink, &ff, &frame)` (GROMACS), `read_cmap(src)` /
//! `write_cmap(sink, &ff)` and `write_data_coeffs(sink, &ff)` (LAMMPS),
//! `read_opls_typifier(src)` (OpenMM). A table that is no force field, CL&Pol's
//! `alpha.ff`, is a function like the frame doors: `read_clpol_alpha(src)`.

pub(crate) mod frame_columns;
pub mod frame_index;
pub(crate) mod null_cells;
pub mod reader;
mod source;
pub mod writer;
pub(crate) mod xdr;

pub mod amber;
pub mod cif;
#[cfg(feature = "ff")]
pub mod clpol;
mod csv;
mod cube;
pub mod dcd;
#[cfg(feature = "stream")]
mod frame_encoding;
pub mod gro;
#[cfg(feature = "ff")]
pub mod gromacs;
pub mod lammps;
pub mod mol2;
#[cfg(feature = "ff")]
pub mod openmm_xml;
pub mod pdb;
pub mod sdf;
mod stl;
pub mod trr;
pub mod vasp;
#[cfg(feature = "ff")]
mod xml;
#[cfg(feature = "ff")]
mod xml_attribute;
mod xsf;
pub mod xtc;
pub mod xyz;

#[cfg(feature = "smiles")]
pub mod cgsmiles;
#[cfg(feature = "zarr")]
pub mod mrec;
#[cfg(feature = "smiles")]
pub mod smiles;

pub use amber::ac::read_ac;
pub use amber::inpcrd::read_inpcrd;
pub use amber::prep::{read_prep, write_prep};
pub use amber::prmtop::read_prmtop;
pub use cif::codec::{read_cif, write_cif};
#[cfg(feature = "ff")]
pub use clpol::codec::read_clpol_alpha;
pub use csv::{read_csv_block, write_csv_block};
pub use cube::{CubeReader, read_cube, write_cube};
pub use dcd::codec::{read_dcd, write_dcd};
#[cfg(feature = "stream")]
pub use frame_encoding::{read_wire_frame, write_wire_frame};
pub use gro::codec::{read_gro, write_gro};
pub use lammps::bond_react::write_lammps_bond_react_map;
#[cfg(feature = "ff")]
pub use lammps::bond_react_system::write_lammps_bond_react_system;
pub use lammps::data::{read_lammps_data, write_lammps_data};
pub use lammps::dump::{read_lammps_dump, write_lammps_dump, write_lammps_dump_local};
pub use lammps::log::read_lammps_log;
pub use lammps::molecule::{
    read_lammps_molecule, read_lammps_molecule_json, write_lammps_molecule,
    write_lammps_molecule_json,
};
pub use mol2::codec::{read_mol2, write_mol2};
pub use pdb::codec::{read_pdb, write_pdb};
pub use sdf::codec::read_sdf;
pub use stl::read_stl;
pub use trr::codec::{read_trr, write_trr};
pub use vasp::chgcar::read_chgcar;
pub use vasp::poscar::{read_poscar, write_poscar};
#[cfg(feature = "ff")]
pub use xml::{XmlForcefieldReader, XmlForcefieldWriter};
pub use xsf::{read_xsf, write_xsf};
pub use xtc::codec::{read_xtc, write_xtc};
pub use xyz::codec::{read_xyz, write_xyz};

#[cfg(feature = "filesystem")]
pub use mrec::zarr_storage::write_mrec;
#[cfg(feature = "zarr")]
pub use mrec::zarr_storage::write_mrec_storage;

pub use source::{Sink, SinkWriter, Source};

/// The one `InvalidData` error of the io readers and writers: a parse or
/// shape failure carrying `e`'s message.
pub(crate) fn invalid_data<E: std::fmt::Display>(e: E) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string())
}

#[cfg(test)]
mod tests {
    //! Every door reads a path and bytes alike, and writes the same bytes to
    //! a path as into a buffer.

    use super::*;
    use crate::core::Frame;
    use crate::io::reader::TrajectoryReader;
    use crate::io::writer::FrameWriter;
    use std::path::Path;

    /// Two atoms with element, type, name, mass and coordinates in an
    /// orthorhombic box, read from extended XYZ — every field some format's
    /// writer requires (GRO the name, LAMMPS data the mass), stated rather
    /// than made up by a writer.
    fn water_pair() -> Frame {
        read_xyz(
            b"2\nLattice=\"10 0 0 0 11 0 0 0 12\" \
             Properties=species:S:1:pos:R:3:type:S:1:name:S:1:mass:R:1 pbc=\"T T T\"\n\
             O 1.0 2.0 3.0 OW OW 15.999\nH 1.5 2.0 3.0 HW HW1 1.008\n",
        )
        .expect("xyz text")
    }

    fn n_atoms(frame: &Frame) -> usize {
        frame
            .get("atoms")
            .and_then(|b| b.n_rows())
            .expect("atoms block")
    }

    type Write = fn(Sink<'_>, &Frame) -> std::io::Result<()>;
    type Read = fn(Source<'_>) -> std::io::Result<Frame>;

    /// `write` into a file and into a buffer gives the same bytes, and `read`
    /// of the file and of the bytes gives the frame back.
    fn check(name: &str, frame: &Frame, write: Write, read: Read) {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join(name);
        write(Sink::from(path.as_path()), frame).expect("path writer");
        let mut bytes = b"stale".to_vec();
        write(Sink::from(&mut bytes), frame).expect("buffer writer");
        assert_eq!(std::fs::read(&path).unwrap(), bytes, "{name}");
        let from_file = read(Source::from(path.as_path())).expect("path reader");
        let from_bytes = read(Source::from(bytes.as_slice())).expect("bytes reader");
        assert_eq!(n_atoms(&from_file), n_atoms(&from_bytes), "{name}");
        assert_eq!(n_atoms(&from_bytes), n_atoms(frame), "{name}");
    }

    #[test]
    fn every_door_reads_paths_and_bytes_and_writes_paths_and_buffers_alike() {
        let f = water_pair();
        check("a.pdb", &f, |s, f| write_pdb(s, f), |s| read_pdb(s));
        check("a.xyz", &f, |s, f| write_xyz(s, f), |s| read_xyz(s));
        check("a.gro", &f, |s, f| write_gro(s, f), |s| read_gro(s));
        check("a.mol2", &f, |s, f| write_mol2(s, f), |s| read_mol2(s));
        check("a.cif", &f, |s, f| write_cif(s, f), |s| read_cif(s));
        check(
            "a.data",
            &f,
            |s, f| write_lammps_data(s, f),
            |s| read_lammps_data(s),
        );
        check(
            "a.dump",
            &f,
            |s, f| write_lammps_dump(s, f, None),
            |s| read_lammps_dump(s),
        );
        check("a.dcd", &f, |s, f| write_dcd(s, f), |s| read_dcd(s));
        check("a.trr", &f, |s, f| write_trr(s, f), |s| read_trr(s));
        check("a.xtc", &f, |s, f| write_xtc(s, f), |s| read_xtc(s));
    }

    /// `<Fmt>Writer::create` writes one frame per call, into a file or a
    /// buffer alike, and `<Fmt>Reader::open` reads each back by index.
    #[test]
    fn trajectory_classes_write_and_read_every_frame() {
        macro_rules! round_trip {
            ($name:expr, $writer:ty, $reader:ty) => {{
                let name = $name;
                let f = water_pair();
                let dir = tempfile::tempdir().expect("tempdir");
                let path = dir.path().join(name);
                let mut bytes = Vec::new();
                for sink in [Sink::from(path.as_path()), Sink::from(&mut bytes)] {
                    let mut writer = <$writer>::create(sink).expect("create");
                    writer.write(&f).expect("frame 1");
                    writer.write(&f).expect("frame 2");
                }
                assert_eq!(std::fs::read(&path).unwrap(), bytes, "{name}");
                for src in [Source::from(path.as_path()), Source::from(bytes.as_slice())] {
                    let mut reader = <$reader>::open(src).expect("open");
                    assert_eq!(reader.len().unwrap(), 2, "{name}");
                    let second = reader.read_frame(1).unwrap().unwrap();
                    assert_eq!(n_atoms(&second), 2, "{name}");
                    assert!(reader.read_frame(2).unwrap().is_none(), "{name}");
                }
            }};
        }
        round_trip!("a.xyz", xyz::XyzWriter<_>, xyz::XyzReader<_>);
        round_trip!("a.pdb", pdb::PdbWriter<_>, pdb::PdbReader<_>);
        round_trip!("a.gro", gro::GroWriter<_>, gro::GroReader<_>);
        round_trip!("a.mol2", mol2::Mol2Writer<_>, mol2::Mol2Reader<_>);
        round_trip!("a.cif", cif::CifWriter<_>, cif::CifReader<_>);
        round_trip!("a.dcd", dcd::DcdWriter<_>, dcd::DcdReader<_>);
        round_trip!("a.trr", trr::TrrWriter<_>, trr::TrrReader<_>);
        round_trip!("a.xtc", xtc::XtcWriter<_>, xtc::XtcReader<_>);
        round_trip!(
            "a.dump",
            lammps::LammpsDumpWriter<_>,
            lammps::LammpsDumpReader<_>
        );
    }

    #[test]
    fn a_missing_file_is_the_os_error_and_garbage_is_invalid_data() {
        let err = read_xyz(Path::new("/no/such/dir/a.xyz")).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::NotFound);
        let err = read_xyz(b"not a count\n").unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
    }
}
