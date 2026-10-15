//! File I/O, one module per file format.
//!
//! Every reader and writer of a format has one of two shapes:
//!
//! - a function at the top of this module, `read_<fmt>[_<what>]` /
//!   `write_<fmt>[_<what>]` — a path; `read_<fmt>_str` / `write_<fmt>_str` for
//!   text in memory and `read_<fmt>_bytes` / `write_<fmt>_bytes` for bytes
//!   (Python and JS carry the same in-memory doors, JS camelCased);
//!   `read_<fmt>_trajectory`
//!   / `write_<fmt>_trajectory` for every frame of a multi-frame file;
//! - a class of the format's own module, `io::<fmt>::<Fmt>Reader` /
//!   `<Fmt>Writer`, over any byte stream (a trajectory reader's `open(path)`
//!   opens a file for random access), with the format's other records.
//!
//! No door picks a format for the caller: every door names its format.
//!
//! | module | format |
//! |---|---|
//! | [`pdb`], [`xyz`], [`gro`], [`sdf`], [`mol2`], [`cif`] | structure files (PDB, XYZ / extended XYZ, GRO, MDL SDF, Tripos MOL2, CIF) |
//! | XSF, Gaussian cube, STL | functions only: `read_xsf` …, `read_cube` …, `read_stl` … |
//! | [`vasp`] | POSCAR / CONTCAR, CHGCAR |
//! | [`dcd`], [`trr`], [`xtc`] | binary trajectories (CHARMM/NAMD DCD, GROMACS TRR and XTC) |
//! | [`lammps`] | LAMMPS data, molecule, dump, log, `fix bond/react`, force-field `*.ff`, `fix cmap` |
//! | [`amber`] | AMBER prmtop, inpcrd, antechamber `.ac`, prep, frcmod |
//! | [`gromacs`] | GROMACS `.top` / `.itp` |
//! | [`openmm_xml`] | OpenMM force-field XML: `read_openmm_forcefield` … |
//! | force-field XML | functions only: `read_xml_forcefield` … — every style of a force field as one element, in its declared units |
//! | [`clpol`] | CL&Pol `alpha.ff` |
//! | [`smiles`], [`cgsmiles`] | the SMILES and CGsmiles line notations |
//! | [`mrec`] | MolRec scientific records (`*.mrec`) |
//! | CSV | functions only: `read_csv_block` … |
//! | a frame in `stream`'s wire encodings (feature `stream`) | functions only: `read_msgpack_frame_bytes` …, `read_json_frame_str` … |
//!
//! [`reader`] and [`writer`] hold the contracts the classes implement
//! ([`FrameReader`](reader::FrameReader), [`TrajectoryReader`](reader::TrajectoryReader),
//! [`FrameWriter`](writer::FrameWriter), [`ForceFieldReader`](reader::ForceFieldReader),
//! [`ForceFieldWriter`](writer::ForceFieldWriter)); [`frame_index`] the chunked
//! frame indexing the `*IndexBuilder` classes and `read_<fmt>_bytes` doors
//! share.
//!
//! Force-field files map a file onto a [`ForceField`](crate::ff::forcefield::ForceField),
//! the data model `ff` owns (feature `ff`): a reader owns the translation from
//! its format — names, **and unit and factor normalization** — into the
//! force-field IR (adopts the LAMMPS standard), its writer the inverse, so unit
//! conversion stays at one boundary pair.

pub(crate) mod frame_columns;
pub mod frame_index;
pub mod reader;
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

pub use amber::ac::{read_ac, read_ac_str};
pub use amber::inpcrd::{read_inpcrd, read_inpcrd_str};
pub use amber::prep::{read_prep, read_prep_str, write_prep, write_prep_str};
pub use amber::prmtop::{read_prmtop, read_prmtop_str};
#[cfg(feature = "ff")]
pub use amber::{
    frcmod::{write_frcmod, write_frcmod_str},
    prmtop_forcefield::{read_prmtop_forcefield, read_prmtop_system},
};
pub use cif::codec::{read_cif, read_cif_str, read_cif_trajectory, write_cif, write_cif_str};
#[cfg(feature = "ff")]
pub use clpol::codec::{read_clpol_alpha, read_clpol_alpha_str};
pub use csv::{read_csv_block, read_csv_block_str, write_csv_block, write_csv_block_str};
pub use cube::{read_cube, read_cube_str, read_cube_trajectory, write_cube, write_cube_str};
pub use dcd::codec::{read_dcd_bytes, read_dcd_trajectory, write_dcd_bytes, write_dcd_trajectory};
#[cfg(feature = "stream")]
pub use frame_encoding::{
    read_json_frame_str, read_msgpack_frame_bytes, write_json_frame_str, write_msgpack_frame_bytes,
};
pub use gro::codec::{
    read_gro, read_gro_str, read_gro_trajectory, write_gro, write_gro_str, write_gro_trajectory,
};
#[cfg(feature = "ff")]
pub use gromacs::{
    top_reader::{read_top_forcefield, read_top_system},
    top_writer::{write_top_forcefield, write_top_system},
};
pub use lammps::bond_react::write_lammps_bond_react_map;
#[cfg(feature = "ff")]
pub use lammps::bond_react_system::write_lammps_bond_react_system;
pub use lammps::data::{
    read_lammps_data, read_lammps_data_bytes, read_lammps_data_str, write_lammps_data,
    write_lammps_data_str,
};
pub use lammps::dump::{
    read_lammps_dump_bytes, read_lammps_dump_str, read_lammps_dump_trajectory,
    write_lammps_dump_local, write_lammps_dump_str, write_lammps_dump_trajectory,
};
pub use lammps::log::{read_lammps_log, read_lammps_log_str};
pub use lammps::molecule::{
    read_lammps_molecule, read_lammps_molecule_json, write_lammps_molecule,
    write_lammps_molecule_json,
};
#[cfg(feature = "ff")]
pub use lammps::{
    forcefield_reader::{
        read_lammps_cmap_forcefield, read_lammps_cmap_str, read_lammps_data_coeffs,
        read_lammps_data_coeffs_str, read_lammps_forcefield, read_lammps_forcefield_str,
    },
    forcefield_writer::{
        write_lammps_cmap_forcefield, write_lammps_cmap_str, write_lammps_data_coeffs_str,
        write_lammps_forcefield, write_lammps_forcefield_str,
    },
};
pub use mol2::codec::{read_mol2, read_mol2_str, read_mol2_trajectory, write_mol2, write_mol2_str};
#[cfg(feature = "ff")]
pub use openmm_xml::{
    reader::{read_openmm_forcefield, read_openmm_forcefield_str},
    writer::{write_openmm_forcefield, write_openmm_forcefield_str},
};
pub use pdb::codec::{
    read_pdb, read_pdb_bytes, read_pdb_str, read_pdb_trajectory, write_pdb, write_pdb_str,
    write_pdb_trajectory,
};
pub use sdf::codec::{read_sdf, read_sdf_bytes, read_sdf_str, read_sdf_trajectory};
pub use stl::{read_stl, read_stl_bytes};
pub use trr::codec::{read_trr_bytes, read_trr_trajectory, write_trr_bytes, write_trr_trajectory};
pub use vasp::chgcar::{read_chgcar, read_chgcar_str};
pub use vasp::poscar::{read_poscar, read_poscar_str, write_poscar, write_poscar_str};
#[cfg(feature = "ff")]
pub use xml::{
    read_xml_forcefield, read_xml_forcefield_str, write_xml_forcefield, write_xml_forcefield_str,
};
pub use xsf::{read_xsf, read_xsf_str, write_xsf, write_xsf_str};
pub use xtc::codec::{read_xtc_bytes, read_xtc_trajectory, write_xtc_bytes, write_xtc_trajectory};
pub use xyz::codec::{
    read_xyz, read_xyz_bytes, read_xyz_str, read_xyz_trajectory, write_xyz, write_xyz_str,
    write_xyz_trajectory,
};

#[cfg(feature = "smiles")]
pub use cgsmiles::to_atomistic::read_cgsmiles_str;
#[cfg(feature = "smiles")]
pub use smiles::{ir_from_atomistic::write_smiles_str, ir_to_atomistic::read_smiles_str};

#[cfg(feature = "filesystem")]
pub use mrec::zarr_storage::{
    read_mrec, read_mrec_forcefield, read_mrec_frame, read_mrec_meta, read_mrec_system,
    read_mrec_trajectory, write_mrec, write_mrec_forcefield, write_mrec_frame, write_mrec_system,
    write_mrec_trajectory,
};
#[cfg(feature = "zarr")]
pub use mrec::zarr_storage::{read_mrec_frame_storage, read_mrec_storage, write_mrec_storage};

/// The one `InvalidData` error of the io readers and writers: a parse or
/// shape failure carrying `e`'s message.
pub(crate) fn invalid_data<E: std::fmt::Display>(e: E) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string())
}

#[cfg(test)]
mod tests {
    //! Every in-memory door is its path door into or out of memory: the
    //! `write_<fmt>_str` / `_bytes` output is the file the path door writes,
    //! and `read_<fmt>_str` / `_bytes` reads it back to the frame the path
    //! door reads.

    use super::*;
    use crate::core::Frame;

    /// Two atoms with element, type and coordinates in an orthorhombic box,
    /// read from extended XYZ.
    fn water_pair() -> Frame {
        read_xyz_str(
            "2\nLattice=\"10 0 0 0 11 0 0 0 12\" Properties=species:S:1:pos:R:3:type:S:1 pbc=\"T T T\"\n\
             O 1.0 2.0 3.0 OW\nH 1.5 2.0 3.0 HW\n",
        )
        .expect("xyz text")
    }

    fn n_atoms(frame: &Frame) -> usize {
        frame
            .get("atoms")
            .and_then(|b| b.n_rows())
            .expect("atoms block")
    }

    /// `write_str(frame)` equals the file `write_path` writes, and
    /// `read_str` of it reads what `read_path` reads.
    fn check_text<E: std::fmt::Debug, F: std::fmt::Debug>(
        name: &str,
        frame: &Frame,
        write_path: fn(&std::path::Path, &Frame) -> Result<(), E>,
        write_str: fn(&Frame) -> Result<String, E>,
        read_path: fn(&std::path::Path) -> Result<Frame, F>,
        read_str: fn(&str) -> Result<Frame, F>,
    ) {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join(name);
        write_path(&path, frame).expect("path writer");
        let text = write_str(frame).expect("str writer");
        assert_eq!(std::fs::read_to_string(&path).unwrap(), text, "{name}");
        let from_file = read_path(&path).expect("path reader");
        let from_text = read_str(&text).expect("str reader");
        assert_eq!(n_atoms(&from_file), n_atoms(&from_text), "{name}");
        assert_eq!(n_atoms(&from_text), n_atoms(frame), "{name}");
    }

    #[test]
    fn text_doors_are_the_path_doors_in_memory() {
        let f = water_pair();
        check_text(
            "a.pdb",
            &f,
            |p, f| write_pdb(p, f),
            write_pdb_str,
            |p| read_pdb(p),
            read_pdb_str,
        );
        check_text(
            "a.xyz",
            &f,
            |p, f| write_xyz(p, f),
            write_xyz_str,
            |p| read_xyz(p),
            read_xyz_str,
        );
        check_text(
            "a.gro",
            &f,
            |p, f| write_gro(p, f),
            write_gro_str,
            |p| read_gro(p),
            read_gro_str,
        );
        check_text(
            "a.mol2",
            &f,
            |p, f| write_mol2(p, f),
            write_mol2_str,
            |p| read_mol2(p),
            read_mol2_str,
        );
        check_text(
            "a.cif",
            &f,
            |p, f| write_cif(p, f),
            write_cif_str,
            |p| read_cif(p),
            read_cif_str,
        );
        check_text(
            "a.data",
            &f,
            |p, f| write_lammps_data(p, f),
            write_lammps_data_str,
            |p| read_lammps_data(p),
            read_lammps_data_str,
        );
        check_text(
            "a.dump",
            &f,
            |p, f| write_lammps_dump_trajectory(p, std::slice::from_ref(f), None),
            |f| write_lammps_dump_str(f, None),
            |p| read_lammps_dump_trajectory(p).map(|mut v| v.remove(0)),
            read_lammps_dump_str,
        );
    }

    #[test]
    fn text_doors_agree_with_the_bytes_doors() {
        let f = water_pair();
        let pdb = write_pdb_str(&f).unwrap();
        assert_eq!(n_atoms(&read_pdb_bytes(pdb.as_bytes()).unwrap()), 2);
        let xyz = write_xyz_str(&f).unwrap();
        assert_eq!(n_atoms(&read_xyz_bytes(xyz.as_bytes()).unwrap()), 2);
        let data = write_lammps_data_str(&f).unwrap();
        assert_eq!(
            n_atoms(&read_lammps_data_bytes(data.as_bytes()).unwrap()),
            2
        );
        let dump = write_lammps_dump_str(&f, None).unwrap();
        assert_eq!(
            n_atoms(&read_lammps_dump_bytes(dump.as_bytes()).unwrap()),
            2
        );
        let sdf = "water\n  molrs\n\n  2  1  0  0  0  0  0  0  0  0999 V2000\n\
                   \x20   1.0000    2.0000    3.0000 O   0  0  0  0  0  0  0  0  0  0  0  0\n\
                   \x20   1.5000    2.0000    3.0000 H   0  0  0  0  0  0  0  0  0  0  0  0\n\
                   \x20 1  2  1  0\nM  END\n$$$$\n";
        assert_eq!(n_atoms(&read_sdf_str(sdf).unwrap()), 2);
        assert_eq!(n_atoms(&read_sdf_bytes(sdf.as_bytes()).unwrap()), 2);
    }

    #[test]
    fn bytes_doors_are_one_frame_files() {
        let f = water_pair();
        let dir = tempfile::tempdir().expect("tempdir");
        type Pair = (
            &'static str,
            fn(&Frame) -> std::io::Result<Vec<u8>>,
            fn(&std::path::Path, &[Frame]) -> std::io::Result<()>,
            fn(&[u8]) -> std::io::Result<Frame>,
        );
        let doors: [Pair; 3] = [
            (
                "a.dcd",
                |f| write_dcd_bytes(f),
                |p, fs| write_dcd_trajectory(p, fs),
                |b| read_dcd_bytes(b, None),
            ),
            (
                "a.trr",
                |f| write_trr_bytes(f),
                |p, fs| write_trr_trajectory(p, fs),
                read_trr_bytes,
            ),
            (
                "a.xtc",
                |f| write_xtc_bytes(f),
                |p, fs| write_xtc_trajectory(p, fs),
                read_xtc_bytes,
            ),
        ];
        for (name, write_bytes, write_path, read_bytes) in doors {
            let path = dir.path().join(name);
            write_path(&path, std::slice::from_ref(&f)).expect("path writer");
            let bytes = write_bytes(&f).expect("bytes writer");
            assert_eq!(std::fs::read(&path).unwrap(), bytes, "{name}");
            assert_eq!(
                n_atoms(&read_bytes(&bytes).expect("bytes reader")),
                2,
                "{name}"
            );
        }
    }
}
