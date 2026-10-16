//! The one-frame doors of `molrs.io`, `molrs::io`'s `read_<fmt>(src)` /
//! `write_<fmt>(sink, frame)`: PDB, XYZ, GRO, SDF, MOL2, CIF, XSF, cube,
//! VASP POSCAR / CHGCAR, LAMMPS data / dump / molecule templates, DCD, TRR,
//! XTC, AMBER inpcrd / prmtop / antechamber ``.ac`` / prep.
//!
//! A reader takes a [`PySource`] (an ``os.PathLike``, ``bytes`` or a
//! ``molrs.io.Text``) and reads the file's first (or only) frame; a writer
//! takes a [`PySink`] (an ``os.PathLike`` or a ``bytearray``) and writes one
//! frame as a whole file. Every reader emits the canonical column names
//! (`molrs.core.keys`); the format's own spelling never reaches Python.

use std::collections::BTreeMap;

use molrs::io::amber::{PrepAtom, PrepResidue};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use serde_json::Value as JsonValue;

use super::json_to_py::json_object_to_pydict;
use super::source::{PySink, PySource};
use crate::core::frame::PyFrame;
use crate::error::io_error_to_pyerr;

/// A `read_<fmt>(source) -> Frame` door over the core function `$rs`.
macro_rules! frame_reader {
    ($(#[$doc:meta])* $name:ident => $rs:path) => {
        $(#[$doc])*
        #[pyfunction]
        pub fn $name(source: PySource<'_>) -> PyResult<PyFrame> {
            PyFrame::from_core_frame(source.read($rs).map_err(io_error_to_pyerr)?)
        }
    };
}

/// A `write_<fmt>(sink, frame)` door over the core function `$rs`.
macro_rules! frame_writer {
    ($(#[$doc:meta])* $name:ident => $rs:path) => {
        $(#[$doc])*
        #[pyfunction]
        pub fn $name(sink: PySink<'_>, frame: &PyFrame) -> PyResult<()> {
            frame.with_frame(|f| sink.put(|s| $rs(s, f), io_error_to_pyerr))?
        }
    };
}

frame_reader!(
    /// Read the first model of a PDB file.
    ///
    /// The ``"atoms"`` block carries ``element`` (str), ``x``/``y``/``z``
    /// (float), ``id`` and ``res_id`` (uint), ``name``, ``res_name``,
    /// ``chain``, ``icode`` and ``altloc`` (str; ``""`` for none),
    /// ``occupancy`` and ``b_factor`` (float). A ``CRYST1`` record gives
    /// ``frame.box``. Every model: :class:`molrs.io.pdb.PdbReader`.
    ///
    /// Parameters
    /// ----------
    /// source : os.PathLike | bytes | molrs.io.Text
    ///     The file, or its whole content.
    ///
    /// Returns
    /// -------
    /// Frame
    ///
    /// Raises
    /// ------
    /// TypeError
    ///     If ``source`` is a ``str`` (pass ``pathlib.Path(s)`` or
    ///     ``molrs.io.Text(s)``).
    /// OSError
    ///     If the file cannot be opened or parsed.
    ///
    /// Examples
    /// --------
    /// >>> frame = molrs.io.read_pdb(pathlib.Path("molecule.pdb"))  # doctest: +SKIP
    read_pdb => molrs::io::read_pdb
);
frame_writer!(
    /// Write a frame as a one-model PDB file.
    ///
    /// Parameters
    /// ----------
    /// sink : os.PathLike | bytearray
    ///     The file (created or truncated), or a buffer (cleared, then
    ///     holding exactly the file's bytes).
    /// frame : Frame
    ///     The frame to write.
    ///
    /// Raises
    /// ------
    /// OSError
    ///     If the frame cannot be written.
    write_pdb => molrs::io::write_pdb
);
frame_reader!(
    /// Read the first frame of an (extended) XYZ file. Every frame:
    /// :class:`molrs.io.xyz.XyzReader`.
    ///
    /// Parameters
    /// ----------
    /// source : os.PathLike | bytes | molrs.io.Text
    ///     The file, or its whole content.
    ///
    /// Returns
    /// -------
    /// Frame
    ///
    /// Examples
    /// --------
    /// >>> frame = molrs.io.read_xyz(molrs.io.Text("1\n\nO 0 0 0\n"))
    /// >>> frame["atoms"].nrows
    /// 1
    read_xyz => molrs::io::read_xyz
);
frame_writer!(
    /// Write a frame as a one-frame extended XYZ file.
    ///
    /// Parameters
    /// ----------
    /// sink : os.PathLike | bytearray
    ///     The file, or a buffer that then holds exactly its bytes.
    /// frame : Frame
    ///     The frame to write.
    write_xyz => molrs::io::write_xyz
);
frame_reader!(
    /// Read the first frame of a GROMACS GRO file.
    ///
    /// The ``"atoms"`` block carries ``res_id``, ``res_name``, ``name``,
    /// ``element`` (inferred from the atom name), ``id`` and ``x``/``y``/``z``
    /// in Å (converted from the file's nm), plus ``vx``/``vy``/``vz`` in Å/ps
    /// when the file has velocities. The box is ``frame.box``. Every frame:
    /// :class:`molrs.io.gro.GroReader`.
    ///
    /// Parameters
    /// ----------
    /// source : os.PathLike | bytes | molrs.io.Text
    ///     The file, or its whole content.
    ///
    /// Returns
    /// -------
    /// Frame
    read_gro => molrs::io::read_gro
);
frame_writer!(
    /// Write a frame as a one-frame GROMACS GRO file (Å → nm).
    ///
    /// Reads ``x``/``y``/``z`` from the ``"atoms"`` block and, when present,
    /// ``res_id``, ``res_name``, ``name``, ``id`` and ``vx``/``vy``/``vz``.
    /// The box is ``frame.box``.
    ///
    /// Parameters
    /// ----------
    /// sink : os.PathLike | bytearray
    ///     The file, or a buffer that then holds exactly its bytes.
    /// frame : Frame
    ///     The frame to write.
    write_gro => molrs::io::write_gro
);
frame_reader!(
    /// Read the first molecule of a Tripos MOL2 file.
    ///
    /// Atoms carry ``id``, ``name``, ``x``/``y``/``z``, ``type`` (the SYBYL
    /// atom type), optional ``res_id``/``res_name`` (the substructure) and
    /// ``charge``. Bonds carry ``atomi``/``atomj`` (0-based), ``type`` (the
    /// SYBYL bond type token) and the chemical ``bond_type``/``bond_number``.
    /// Every molecule: :class:`molrs.io.mol2.Mol2Reader`.
    ///
    /// Parameters
    /// ----------
    /// source : os.PathLike | bytes | molrs.io.Text
    ///     The file, or its whole content.
    ///
    /// Returns
    /// -------
    /// Frame
    read_mol2 => molrs::io::read_mol2
);
frame_writer!(
    /// Write a frame as a one-molecule Tripos MOL2 file.
    ///
    /// Parameters
    /// ----------
    /// sink : os.PathLike | bytearray
    ///     The file, or a buffer that then holds exactly its bytes.
    /// frame : Frame
    ///     The frame to write.
    write_mol2 => molrs::io::write_mol2
);
frame_reader!(
    /// Read the first ``data_`` block of a CIF file. Every block:
    /// :class:`molrs.io.cif.CifReader`.
    ///
    /// Parameters
    /// ----------
    /// source : os.PathLike | bytes | molrs.io.Text
    ///     The file, or its whole content.
    ///
    /// Returns
    /// -------
    /// Frame
    read_cif => molrs::io::read_cif
);
frame_writer!(
    /// Write a frame as a single-block CIF file.
    ///
    /// Parameters
    /// ----------
    /// sink : os.PathLike | bytearray
    ///     The file, or a buffer that then holds exactly its bytes.
    /// frame : Frame
    ///     The frame to write.
    write_cif => molrs::io::write_cif
);
frame_reader!(
    /// Read the first record of an MDL SDF / molfile (V2000). Every record:
    /// :class:`molrs.io.sdf.SdfReader`.
    ///
    /// Parameters
    /// ----------
    /// source : os.PathLike | bytes | molrs.io.Text
    ///     The file, or its whole content.
    ///
    /// Returns
    /// -------
    /// Frame
    ///     ``atoms`` (``element``, ``id``, ``x``/``y``/``z``) and, when the
    ///     record has any, ``bonds``.
    read_sdf => molrs::io::read_sdf
);
frame_reader!(
    /// Read an XSF (XCrySDen Structure File).
    ///
    /// Crystal structures (``CRYSTAL`` + ``PRIMVEC``/``CONVVEC``) yield a
    /// periodic box; molecular structures (``MOLECULE``) a free box. Atoms
    /// carry ``atomic_number`` and ``x``/``y``/``z`` (Å).
    ///
    /// Parameters
    /// ----------
    /// source : os.PathLike | bytes | molrs.io.Text
    ///     The file, or its whole content.
    ///
    /// Returns
    /// -------
    /// Frame
    read_xsf => molrs::io::read_xsf
);
frame_writer!(
    /// Write a frame as an XSF file: ``CRYSTAL`` for a defined periodic box,
    /// otherwise ``MOLECULE``. Atoms need ``atomic_number`` and
    /// ``x``/``y``/``z``.
    ///
    /// Parameters
    /// ----------
    /// sink : os.PathLike | bytearray
    ///     The file, or a buffer that then holds exactly its bytes.
    /// frame : Frame
    ///     The frame to write.
    write_xsf => molrs::io::write_xsf
);
frame_reader!(
    /// Read a Gaussian cube file.
    ///
    /// The frame holds an ``"atoms"`` block (``atomic_number``, ``x``,
    /// ``y``, ``z``, ``charge``) and the grid ``"cube"``: ``"density"`` (a
    /// scalar field) or ``"mo_<idx>"`` arrays. Values are as stored in the
    /// file; ``frame.meta["cube_units"]`` records ``"bohr"`` or
    /// ``"angstrom"``. Cubes back to back: :class:`molrs.io.cube.CubeReader`.
    ///
    /// Parameters
    /// ----------
    /// source : os.PathLike | bytes | molrs.io.Text
    ///     The file, or its whole content.
    ///
    /// Returns
    /// -------
    /// Frame
    read_cube => molrs::io::read_cube
);
frame_writer!(
    /// Write a frame as a Gaussian cube file. The frame needs a ``"cube"``
    /// grid and an ``"atoms"`` block.
    ///
    /// Parameters
    /// ----------
    /// sink : os.PathLike | bytearray
    ///     The file, or a buffer that then holds exactly its bytes.
    /// frame : Frame
    ///     The frame to write.
    write_cube => molrs::io::write_cube
);
frame_reader!(
    /// Read a VASP POSCAR / CONTCAR structure file.
    ///
    /// Parameters
    /// ----------
    /// source : os.PathLike | bytes | molrs.io.Text
    ///     The file, or its whole content.
    ///
    /// Returns
    /// -------
    /// Frame
    read_poscar => molrs::io::read_poscar
);
frame_writer!(
    /// Write a frame as a VASP POSCAR file.
    ///
    /// Parameters
    /// ----------
    /// sink : os.PathLike | bytearray
    ///     The file, or a buffer that then holds exactly its bytes.
    /// frame : Frame
    ///     The frame to write.
    write_poscar => molrs::io::write_poscar
);
frame_reader!(
    /// Read a VASP CHGCAR or CHGDIF file.
    ///
    /// The frame holds an ``"atoms"`` block (``element``, Cartesian
    /// ``x``/``y``/``z`` in Å), a triclinic periodic box and the grid
    /// ``"chgcar"`` with ``"total"`` (and ``"diff"`` for ISPIN=2). The values
    /// are raw (ρ × V_cell, units e): divide by ``frame.box.volume()`` for
    /// e/Å³.
    ///
    /// Parameters
    /// ----------
    /// source : os.PathLike | bytes | molrs.io.Text
    ///     The file, or its whole content.
    ///
    /// Returns
    /// -------
    /// Frame
    read_chgcar => molrs::io::read_chgcar
);
frame_reader!(
    /// Read an antechamber ``.ac`` file.
    ///
    /// Parameters
    /// ----------
    /// source : os.PathLike | bytes | molrs.io.Text
    ///     The file, or its whole content.
    ///
    /// Returns
    /// -------
    /// Frame
    read_ac => molrs::io::read_ac
);
frame_reader!(
    /// Read an AMBER prmtop **structure**: atoms (name, type, charge in
    /// electron units, mass, ``atomic_number`` when the file has it,
    /// ``res_id``), bonds / angles / dihedrals with 0-based indices and type
    /// labels, and the POINTERS meta. The force field is
    /// :class:`molrs.io.amber.AmberPrmtopForcefieldReader`'s.
    ///
    /// Parameters
    /// ----------
    /// source : os.PathLike | bytes | molrs.io.Text
    ///     The file, or its whole content.
    ///
    /// Returns
    /// -------
    /// Frame
    read_prmtop => molrs::io::read_prmtop
);
frame_reader!(
    /// Read a LAMMPS molecule template in its native text format.
    ///
    /// Parameters
    /// ----------
    /// source : os.PathLike | bytes | molrs.io.Text
    ///     The file, or its whole content.
    ///
    /// Returns
    /// -------
    /// Frame
    read_lammps_molecule => molrs::io::read_lammps_molecule
);
frame_writer!(
    /// Write a frame as a LAMMPS molecule template in the native text
    /// format.
    ///
    /// Parameters
    /// ----------
    /// sink : os.PathLike | bytearray
    ///     The file, or a buffer that then holds exactly its bytes.
    /// frame : Frame
    ///     The template.
    write_lammps_molecule => molrs::io::write_lammps_molecule
);
frame_reader!(
    /// Read a LAMMPS molecule template in its JSON format
    /// (``format: "molecule"``).
    ///
    /// Parameters
    /// ----------
    /// source : os.PathLike | bytes | molrs.io.Text
    ///     The file, or its whole content.
    ///
    /// Returns
    /// -------
    /// Frame
    read_lammps_molecule_json => molrs::io::read_lammps_molecule_json
);
frame_writer!(
    /// Write a frame as a LAMMPS molecule template in its JSON format.
    ///
    /// Parameters
    /// ----------
    /// sink : os.PathLike | bytearray
    ///     The file, or a buffer that then holds exactly its bytes.
    /// frame : Frame
    ///     The template.
    write_lammps_molecule_json => molrs::io::write_lammps_molecule_json
);
frame_reader!(
    /// Read the first snapshot of a LAMMPS dump. Every snapshot:
    /// :class:`molrs.io.lammps.LammpsDumpReader`.
    ///
    /// Parameters
    /// ----------
    /// source : os.PathLike | bytes | molrs.io.Text
    ///     The file, or its whole content.
    ///
    /// Returns
    /// -------
    /// Frame
    read_lammps_dump => molrs::io::read_lammps_dump
);
frame_reader!(
    /// Read the first frame of a DCD trajectory. Every frame:
    /// :class:`molrs.io.dcd.DcdReader`.
    ///
    /// Parameters
    /// ----------
    /// source : os.PathLike | bytes
    ///     The file, or its whole content.
    ///
    /// Returns
    /// -------
    /// Frame
    read_dcd => molrs::io::read_dcd
);
frame_writer!(
    /// Write a frame as a one-frame DCD file (NAMD-compatible,
    /// little-endian). A frame with a ``w`` column (4D dynamics) is refused.
    ///
    /// Parameters
    /// ----------
    /// sink : os.PathLike | bytearray
    ///     The file, or a buffer that then holds exactly its bytes.
    /// frame : Frame
    ///     The frame to write.
    write_dcd => molrs::io::write_dcd
);
frame_reader!(
    /// Read the first frame of a GROMACS TRR trajectory (nm → Å). Every
    /// frame: :class:`molrs.io.trr.TrrReader`.
    ///
    /// Parameters
    /// ----------
    /// source : os.PathLike | bytes
    ///     The file, or its whole content.
    ///
    /// Returns
    /// -------
    /// Frame
    read_trr => molrs::io::read_trr
);
frame_writer!(
    /// Write a frame as a one-frame GROMACS TRR file (Å → nm, single
    /// precision): ``x``/``y``/``z``, and ``vx``/``vy``/``vz`` and
    /// ``fx``/``fy``/``fz`` when present.
    ///
    /// Parameters
    /// ----------
    /// sink : os.PathLike | bytearray
    ///     The file, or a buffer that then holds exactly its bytes.
    /// frame : Frame
    ///     The frame to write.
    write_trr => molrs::io::write_trr
);
frame_reader!(
    /// Read the first frame of a GROMACS XTC trajectory (nm → Å). Every
    /// frame: :class:`molrs.io.xtc.XtcReader`.
    ///
    /// Parameters
    /// ----------
    /// source : os.PathLike | bytes
    ///     The file, or its whole content.
    ///
    /// Returns
    /// -------
    /// Frame
    read_xtc => molrs::io::read_xtc
);
frame_writer!(
    /// Write a frame as a one-frame GROMACS XTC file (Å → nm, lossy). The
    /// quantization precision is ``frame.meta["precision"]`` when present.
    ///
    /// Parameters
    /// ----------
    /// sink : os.PathLike | bytearray
    ///     The file, or a buffer that then holds exactly its bytes.
    /// frame : Frame
    ///     The frame to write.
    write_xtc => molrs::io::write_xtc
);

/// Read a LAMMPS data file.
///
/// Every typed block (``atoms``, ``bonds``, ``angles``, ``dihedrals``,
/// ``impropers``) carries the numeric ``type_id`` and the string ``type``:
/// the file's type label, or the id spelled as a label when the file has no
/// ``* Type Labels`` section. The ``* Coeffs`` sections are kept verbatim in
/// ``frame.meta["lammps_coeffs_text"]``
/// (:meth:`molrs.io.lammps.LammpsForcefieldReader.read_data_coeffs` reads
/// them into a force field); the type-label inventories, header counts,
/// unit style and the box axes the header named are in ``frame.meta`` too.
///
/// Parameters
/// ----------
/// source : os.PathLike | bytes | molrs.io.Text
///     The file, or its whole content.
/// atom_style : str, optional
///     The ``Atoms`` column layout (``"full"``, ``"atomic"``, ``"charge"``,
///     …), as LAMMPS's ``atom_style`` governs ``read_data``. Without it the
///     section's ``# style`` comment decides, and without that the column
///     count.
///
/// Returns
/// -------
/// Frame
///
/// Raises
/// ------
/// OSError
///     If the file cannot be opened or parsed, or ``atom_style`` is not an
///     atom style.
#[pyfunction]
#[pyo3(signature = (source, atom_style = None))]
pub fn read_lammps_data(source: PySource<'_>, atom_style: Option<&str>) -> PyResult<PyFrame> {
    use molrs::io::lammps::LammpsDataReader;
    use molrs::io::reader::{FrameReader, ReadSeek};
    let frame = source
        .read(|src| match atom_style {
            None => molrs::io::read_lammps_data(src),
            Some(style) => {
                let input: Box<dyn ReadSeek + '_> = match src {
                    molrs::io::Source::Path(path) => molrs::io::reader::open_seekable(path)?,
                    molrs::io::Source::Bytes(bytes) => Box::new(std::io::Cursor::new(bytes)),
                };
                LammpsDataReader::new(input)
                    .with_atom_style(style)?
                    .read()?
                    .ok_or_else(|| {
                        std::io::Error::new(
                            std::io::ErrorKind::InvalidData,
                            "no frame in the LAMMPS data file",
                        )
                    })
            }
        })
        .map_err(io_error_to_pyerr)?;
    PyFrame::from_core_frame(frame)
}

/// Write a frame as a LAMMPS data file.
///
/// The box is ``frame.box``. A frame without one is a non-periodic system,
/// written inside the bounds of its coordinates widened by 1 length unit on
/// every side: read it with ``boundary s s s``. Atoms, topology, ``Masses``
/// and the ``* Type Labels`` sections are written; ``* Coeffs`` are not
/// (:meth:`molrs.io.lammps.LammpsForcefieldWriter.write_data_coeffs` writes
/// them for the same labels). A system with Drude particles gets a header
/// comment with the ``fix drude`` C/D/N flags in atom-type order.
///
/// Parameters
/// ----------
/// sink : os.PathLike | bytearray
///     The file, or a buffer that then holds exactly its bytes.
/// frame : Frame
///     The frame to write.
/// type_labels : dict[str, list[str]], optional
///     Extra type labels per block (``{"atoms": [...], "bonds": [...]}``),
///     declared in the file even when no row uses them; ids stay dense and
///     follow the sorted labels. ``frame`` itself is not modified.
///
/// Raises
/// ------
/// ValueError
///     If a declared label is empty or contains ``,``, or a block is not
///     ``atoms`` / ``bonds`` / ``angles`` / ``dihedrals`` / ``impropers`` /
///     ``cmaps``.
/// OSError
///     If the frame cannot be written.
#[pyfunction]
#[pyo3(signature = (sink, frame, *, type_labels = None))]
pub fn write_lammps_data(
    sink: PySink<'_>,
    frame: &PyFrame,
    type_labels: Option<BTreeMap<String, Vec<String>>>,
) -> PyResult<()> {
    match type_labels {
        None => frame
            .with_frame(|f| sink.put(|s| molrs::io::write_lammps_data(s, f), io_error_to_pyerr))?,
        Some(extra) => {
            let mut work = frame.clone_core_frame()?;
            for (block, labels) in &extra {
                molrs::core::TypeLabels::declare(&mut work, block, labels)
                    .map_err(PyValueError::new_err)?;
            }
            sink.put(
                |s| molrs::io::write_lammps_data(s, &work),
                io_error_to_pyerr,
            )
        }
    }
}

/// Write a frame as a one-snapshot LAMMPS dump. Several snapshots:
/// :class:`molrs.io.lammps.LammpsDumpWriter`.
///
/// Parameters
/// ----------
/// sink : os.PathLike | bytearray
///     The file, or a buffer that then holds exactly its bytes.
/// frame : Frame
///     The frame to write.
/// columns : list[str], optional
///     The ``dump custom`` column line, e.g. ``["id", "element", "x", "y",
///     "z"]``, written in the order given; a name the ``atoms`` block cannot
///     supply raises. Without it every column the block holds is written.
///
/// Notes
/// -----
/// The dump's ``type`` field is ``type_id`` when the block has it, otherwise
/// the string ``type`` labels. Values are formatted from each column's
/// stored dtype; a complex column, one with more than one value per row, or
/// a string that is empty or contains whitespace raises.
#[pyfunction]
#[pyo3(signature = (sink, frame, columns = None))]
pub fn write_lammps_dump(
    sink: PySink<'_>,
    frame: &PyFrame,
    columns: Option<Vec<String>>,
) -> PyResult<()> {
    let chosen: Option<Vec<&str>> = columns
        .as_ref()
        .map(|c| c.iter().map(String::as_str).collect());
    frame.with_frame(|f| {
        sink.put(
            |s| molrs::io::write_lammps_dump(s, f, chosen.as_deref()),
            io_error_to_pyerr,
        )
    })?
}

/// Write frames as a LAMMPS ``dump local`` file (OVITO's bond input):
/// ``ITEM: NUMBER OF ENTRIES`` + ``ITEM: ENTRIES batom1 batom2 [btype]`` per
/// frame, rows from an ``entries`` block if present, otherwise from the
/// canonical ``bonds``.
///
/// Parameters
/// ----------
/// sink : os.PathLike | bytearray
///     The file, or a buffer that then holds exactly its bytes.
/// frames : list[Frame]
///     The frames, in order.
#[pyfunction]
pub fn write_lammps_dump_local(sink: PySink<'_>, frames: Vec<PyRef<'_, PyFrame>>) -> PyResult<()> {
    let core: Vec<_> = frames
        .iter()
        .map(|f| f.clone_core_frame())
        .collect::<PyResult<_>>()?;
    sink.put(
        |s| molrs::io::write_lammps_dump_local(s, &core),
        io_error_to_pyerr,
    )
}

/// Read an AMBER ASCII inpcrd / restrt coordinate file.
///
/// Fixed-width Fortran ``6F12.7`` layout. Without ``frame``, returns a new
/// frame with ``id``, ``name``, ``x``/``y``/``z``, optional ``vel`` (shape
/// ``[n, 3]``), optional box, and meta ``title`` / ``timestep``.
///
/// With ``frame`` (say, the structure :func:`read_prmtop` read), the file's
/// coordinates go into that frame in place and it is returned: ``x`` /
/// ``y`` / ``z`` (and ``vel``) replace those atom columns, every other
/// column stays; the file's box and meta keys are set. A frame without an
/// ``atoms`` block receives the file's whole block.
///
/// Parameters
/// ----------
/// source : os.PathLike | bytes | molrs.io.Text
///     The file, or its whole content.
/// frame : Frame, optional
///     Frame to receive the coordinates.
///
/// Returns
/// -------
/// Frame
///     ``frame`` itself when given, else a new frame.
///
/// Raises
/// ------
/// OSError
///     If the file cannot be opened or parsed, or ``frame``'s atom count
///     differs from the file's (``frame`` is then unchanged).
#[pyfunction]
#[pyo3(signature = (source, frame = None))]
pub fn read_inpcrd<'py>(
    py: Python<'py>,
    source: PySource<'_>,
    frame: Option<Bound<'py, PyFrame>>,
) -> PyResult<Bound<'py, PyAny>> {
    let coordinates = source
        .read(molrs::io::read_inpcrd)
        .map_err(io_error_to_pyerr)?;
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

/// Read an AMBER prep residue into its nested-dict form (the serde JSON
/// shape of the residue: ``name``, ``atoms`` — each with ``index``,
/// ``name``, ``atom_type``, ``tree_type``, ``na``/``nb``/``nc``, ``r``,
/// ``theta``, ``phi``, ``charge``, ``element`` — and ``impropers``).
///
/// Parameters
/// ----------
/// source : os.PathLike | bytes | molrs.io.Text
///     The file, or its whole content.
///
/// Returns
/// -------
/// dict
#[pyfunction]
pub fn read_prep<'py>(py: Python<'py>, source: PySource<'_>) -> PyResult<Bound<'py, PyDict>> {
    let res = source
        .read(molrs::io::read_prep)
        .map_err(io_error_to_pyerr)?;
    let value = serde_json::to_value(&res)
        .map_err(|e| PyValueError::new_err(format!("failed to serialize prep residue: {e}")))?;
    match value {
        JsonValue::Object(map) => json_object_to_pydict(py, &map),
        _ => Err(PyValueError::new_err(
            "internal error: prep residue did not serialize to an object",
        )),
    }
}

/// Write an AMBER prep residue from its nested-dict form (the shape
/// :func:`read_prep` returns).
///
/// Parameters
/// ----------
/// sink : os.PathLike | bytearray
///     The file, or a buffer that then holds exactly its bytes.
/// residue : dict
///     The residue. Every atom field is required.
///
/// Raises
/// ------
/// KeyError
///     If a field is missing (no default stands in for it).
#[pyfunction]
pub fn write_prep(sink: PySink<'_>, residue: &Bound<'_, PyAny>) -> PyResult<()> {
    let res = py_to_prep_residue(residue)?;
    sink.put(|s| molrs::io::write_prep(s, &res), io_error_to_pyerr)
}

/// A prep residue from its nested-dict form. Every atom field is required: a
/// missing one raises ``KeyError`` naming it rather than standing in a
/// default (a zero geometry, a zero charge, an empty element).
fn py_to_prep_residue(residue: &Bound<'_, PyAny>) -> PyResult<PrepResidue> {
    let name: String = residue.get_item("name")?.extract()?;
    let mut atoms = Vec::new();
    for item in residue.get_item("atoms")?.try_iter()? {
        let d = item?;
        atoms.push(PrepAtom {
            index: d.get_item("index")?.extract()?,
            name: d.get_item("name")?.extract()?,
            atom_type: d.get_item("atom_type")?.extract()?,
            tree_type: d.get_item("tree_type")?.extract()?,
            na: d.get_item("na")?.extract()?,
            nb: d.get_item("nb")?.extract()?,
            nc: d.get_item("nc")?.extract()?,
            r: d.get_item("r")?.extract()?,
            theta: d.get_item("theta")?.extract()?,
            phi: d.get_item("phi")?.extract()?,
            charge: d.get_item("charge")?.extract()?,
            element: d.get_item("element")?.extract()?,
        });
    }
    let mut impropers = Vec::new();
    if let Ok(imps) = residue.get_item("impropers") {
        for item in imps.try_iter()? {
            impropers.push(item?.extract::<Vec<String>>()?);
        }
    }
    Ok(PrepResidue {
        name,
        atoms,
        head_atom: None,
        tail_atom: None,
        impropers,
    })
}

/// Register this module's doors flat on the native module, as `molrs.io`'s.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    for door in [
        wrap_pyfunction!(read_pdb, m)?,
        wrap_pyfunction!(write_pdb, m)?,
        wrap_pyfunction!(read_xyz, m)?,
        wrap_pyfunction!(write_xyz, m)?,
        wrap_pyfunction!(read_gro, m)?,
        wrap_pyfunction!(write_gro, m)?,
        wrap_pyfunction!(read_mol2, m)?,
        wrap_pyfunction!(write_mol2, m)?,
        wrap_pyfunction!(read_cif, m)?,
        wrap_pyfunction!(write_cif, m)?,
        wrap_pyfunction!(read_sdf, m)?,
        wrap_pyfunction!(read_xsf, m)?,
        wrap_pyfunction!(write_xsf, m)?,
        wrap_pyfunction!(read_cube, m)?,
        wrap_pyfunction!(write_cube, m)?,
        wrap_pyfunction!(read_poscar, m)?,
        wrap_pyfunction!(write_poscar, m)?,
        wrap_pyfunction!(read_chgcar, m)?,
        wrap_pyfunction!(read_ac, m)?,
        wrap_pyfunction!(read_prmtop, m)?,
        wrap_pyfunction!(read_inpcrd, m)?,
        wrap_pyfunction!(read_prep, m)?,
        wrap_pyfunction!(write_prep, m)?,
        wrap_pyfunction!(read_lammps_data, m)?,
        wrap_pyfunction!(write_lammps_data, m)?,
        wrap_pyfunction!(read_lammps_dump, m)?,
        wrap_pyfunction!(write_lammps_dump, m)?,
        wrap_pyfunction!(write_lammps_dump_local, m)?,
        wrap_pyfunction!(read_lammps_molecule, m)?,
        wrap_pyfunction!(write_lammps_molecule, m)?,
        wrap_pyfunction!(read_lammps_molecule_json, m)?,
        wrap_pyfunction!(write_lammps_molecule_json, m)?,
        wrap_pyfunction!(read_dcd, m)?,
        wrap_pyfunction!(write_dcd, m)?,
        wrap_pyfunction!(read_trr, m)?,
        wrap_pyfunction!(write_trr, m)?,
        wrap_pyfunction!(read_xtc, m)?,
        wrap_pyfunction!(write_xtc, m)?,
    ] {
        crate::add_function(m, "molrs.io", door)?;
    }
    Ok(())
}
