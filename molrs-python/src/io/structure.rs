//! The structure-file doors of `molrs.io`: one frame in, one frame out —
//! PDB, XYZ, GRO, SDF, MOL2, CIF, XSF, cube, VASP POSCAR / CHGCAR, LAMMPS
//! data and molecule templates, AMBER inpcrd / prmtop structure, antechamber
//! ``.ac`` and ``prep``. Every reader emits the canonical column names
//! (`molrs.core.keys`); the format's own spelling never reaches Python.

use crate::core::frame::PyFrame;
use crate::error::{io_error_to_pyerr, molrs_error_to_pyerr};
use crate::path::path_str;
use molrs::io::amber::{PrepAtom, PrepResidue};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use super::json_to_py::json_object_to_pydict;
use pyo3::types::PyDict;
use serde_json::Value as JsonValue;
use std::fs::File;
use std::path::PathBuf;

/// Read a PDB file and return a Frame.
///
/// The resulting frame contains an ``"atoms"`` block with columns ``element``
/// (str), ``x``/``y``/``z`` (float), ``id`` and ``res_id`` (uint), ``name``,
/// ``res_name``, ``chain``, ``icode`` and ``altloc`` (str; ``""`` for none),
/// ``occupancy`` and ``b_factor`` (float). If CRYST1 records are present a
/// ``Box`` is also attached.
///
/// Parameters
/// ----------
/// path : str
///     Path to a ``.pdb`` file on disk.
///
/// Returns
/// -------
/// Frame
///     Parsed molecular data.
///
/// Raises
/// ------
/// IOError
///     If the file cannot be opened or parsed.
///
/// Examples
/// --------
/// >>> frame = molrs.io.read_pdb("molecule.pdb")
/// >>> atoms = frame["atoms"]
/// >>> symbols = atoms["symbol"]
#[pyfunction]
pub fn read_pdb(path: PathBuf) -> PyResult<PyFrame> {
    let path = path_str(&path)?;
    let frame = molrs::io::read_pdb(path)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("{path}: {e}")))?;
    PyFrame::from_core_frame(frame)
}

/// Read an XYZ file and return a single Frame.
///
/// Parameters
/// ----------
/// path : str
///     Path to a ``.xyz`` file on disk.
///
/// Returns
/// -------
/// Frame
#[pyfunction]
pub fn read_xyz(path: PathBuf) -> PyResult<PyFrame> {
    let path = path_str(&path)?;
    let frame = molrs::io::read_xyz(path).map_err(io_error_to_pyerr)?;
    PyFrame::from_core_frame(frame)
}

/// Read a LAMMPS data file and return a Frame.
///
/// Every typed block (``atoms``, ``bonds``, ``angles``, ``dihedrals``,
/// ``impropers``) carries the numeric ``type_id`` and the string ``type``: the
/// file's type label, or the id spelled as a label when the file has no
/// ``* Type Labels`` section. The ``* Coeffs`` sections are kept verbatim in
/// ``frame.meta["lammps_coeffs_text"]`` (``molrs.io.read_lammps_data_coeffs(path)``
/// reads them into a force field); the type-label inventories,
/// header counts, unit style and the box axes the header named are in
/// ``frame.meta`` too.
///
/// Parameters
/// ----------
/// path : str
///     Path to a LAMMPS data file on disk.
/// atom_style : str, optional
///     The ``Atoms`` column layout (``"full"``, ``"atomic"``, ``"charge"``,
///     …), as LAMMPS's ``atom_style`` governs ``read_data``. Without it the
///     section's ``# style`` comment decides, and without that the column
///     count.
///
/// Returns
/// -------
/// Frame
///     Parsed molecular data with atoms, bonds, and box metadata.
///
/// Raises
/// ------
/// IOError
///     If the file cannot be opened or parsed, or ``atom_style`` is not an
///     atom style.
///
/// Examples
/// --------
/// >>> frame = molrs.io.read_lammps_data("system.data")
/// >>> atoms = frame["atoms"]
#[pyfunction]
#[pyo3(signature = (path, atom_style = None))]
pub fn read_lammps_data(path: PathBuf, atom_style: Option<&str>) -> PyResult<PyFrame> {
    use molrs::io::lammps::LammpsDataReader;
    use molrs::io::reader::FrameReader;
    let path = path_str(&path)?;
    let frame = match atom_style {
        None => molrs::io::read_lammps_data(path).map_err(io_error_to_pyerr)?,
        Some(style) => {
            let file = File::open(path).map_err(io_error_to_pyerr)?;
            LammpsDataReader::new(std::io::BufReader::new(file))
                .with_atom_style(style)
                .map_err(io_error_to_pyerr)?
                .read()
                .map_err(io_error_to_pyerr)?
                .ok_or_else(|| {
                    pyo3::exceptions::PyIOError::new_err("no frame in LAMMPS data file")
                })?
        }
    };
    PyFrame::from_core_frame(frame)
}

/// Read the first frame of a GROMACS GRO file.
///
/// The ``"atoms"`` block carries ``res_id``, ``res_name``, ``name``,
/// ``element`` (inferred from the atom name), ``id`` and ``x``/``y``/``z`` in
/// Å (converted from the file's nm), plus ``vx``/``vy``/``vz`` in Å/ps when
/// the file has velocities. The box is ``frame.box``. Every frame of a
/// multi-frame file: :func:`read_gro_trajectory`.
///
/// Raises
/// ------
/// IOError
///     If the file cannot be opened or parsed, or holds no frame.
#[pyfunction]
pub fn read_gro(path: PathBuf) -> PyResult<PyFrame> {
    let path = path_str(&path)?;
    let frame = molrs::io::read_gro(path).map_err(io_error_to_pyerr)?;
    PyFrame::from_core_frame(frame)
}

/// Write a Frame to a GROMACS GRO file.
///
/// Reads ``x``/``y``/``z`` (Å, written as nm) from the ``"atoms"`` block and,
/// when present, ``res_id``, ``res_name``, ``name`` (else ``element``),
/// ``id`` and ``vx``/``vy``/``vz``. The box is taken from ``frame.box``.
///
/// Raises
/// ------
/// IOError
///     If the file cannot be written.
/// ValueError
///     If the frame is missing the ``"atoms"`` block or coordinate columns.
#[pyfunction]
pub fn write_gro(path: PathBuf, frame: &PyFrame) -> PyResult<()> {
    let path = path_str(&path)?;
    frame.with_frame(|f| molrs::io::write_gro(path, f).map_err(io_error_to_pyerr))?
}

/// Write Frames as one multi-frame GROMACS GRO trajectory, each as
/// :func:`write_gro` writes it. Inverse of :func:`read_gro_trajectory`.
#[pyfunction]
pub fn write_gro_trajectory(path: PathBuf, frames: Vec<PyRef<'_, PyFrame>>) -> PyResult<()> {
    let path = path_str(&path)?;
    let core_frames: Vec<_> = frames
        .iter()
        .map(|f| f.clone_core_frame())
        .collect::<PyResult<_>>()?;
    molrs::io::write_gro_trajectory(path, &core_frames).map_err(io_error_to_pyerr)
}

/// Read a VASP CHGCAR or CHGDIF file.
///
/// Returns a Frame containing:
///
/// - ``"atoms"`` block with ``symbol``, ``x``, ``y``, ``z`` (Cartesian Å)
/// - ``box``: triclinic periodic box
/// - grid ``"chgcar"``: a :class:`Grid` with at least ``"total"`` (and
///   ``"diff"`` for spin-polarised ISPIN=2 calculations)
///
/// The volumetric values are stored **raw** (ρ × V_cell, units e).
/// Divide by ``simbox.volume()`` to get charge density in e/Å³.
///
/// Parameters
/// ----------
/// path : str
///     Path to a CHGCAR or CHGDIF file.
///
/// Returns
/// -------
/// Frame
///
/// Raises
/// ------
/// ValueError
///     On parse errors.
/// IOError
///     If the file cannot be opened.
///
/// Examples
/// --------
/// >>> frame = molrs.io.read_chgcar("CHGCAR")
/// >>> grid = frame["chgcar"]
/// >>> total = grid["total"]          # shape (nx, ny, nz)
/// >>> density = total / frame.box.volume()
#[pyfunction]
pub fn read_chgcar(path: PathBuf) -> PyResult<PyFrame> {
    let path = path_str(&path)?;
    let frame = molrs::io::read_chgcar(path).map_err(molrs_error_to_pyerr)?;
    PyFrame::from_core_frame(frame)
}

/// Read a Gaussian Cube file.
///
/// Returns a Frame containing:
///
/// - ``"atoms"`` block with ``element``, ``x``, ``y``, ``z``,
///   ``atomic_number``, ``charge``
/// - grid ``"cube"``: a :class:`Grid` with ``"density"`` (scalar field)
///   or ``"mo_<idx>"`` arrays (MO variant)
///
/// Values are stored as-is from the file (no unit conversion).
/// The unit system is recorded in ``frame.meta["cube_units"]``
/// (``"bohr"`` or ``"angstrom"``).
///
/// Parameters
/// ----------
/// path : str
///     Path to a ``.cube`` file.
///
/// Returns
/// -------
/// Frame
///
/// Raises
/// ------
/// ValueError
///     On parse errors.
/// IOError
///     If the file cannot be opened.
///
/// Examples
/// --------
/// >>> frame = molrs.io.read_cube("density.cube")
/// >>> grid = frame["cube"]
/// >>> density = grid["density"]       # shape (nx, ny, nz)
#[pyfunction]
pub fn read_cube(path: PathBuf) -> PyResult<PyFrame> {
    let path = path_str(&path)?;
    let frame = molrs::io::read_cube(path).map_err(molrs_error_to_pyerr)?;
    PyFrame::from_core_frame(frame)
}

/// Write a Frame to a Gaussian Cube file.
///
/// The Frame must contain a ``"cube"`` grid and an ``"atoms"`` block.
///
/// Parameters
/// ----------
/// path : str
///     Output file path.
/// frame : Frame
///     Frame to write.
#[pyfunction]
pub fn write_cube(path: PathBuf, frame: &PyFrame) -> PyResult<()> {
    let path = path_str(&path)?;
    frame.with_frame(|f| molrs::io::write_cube(path, f).map_err(molrs_error_to_pyerr))?
}

/// Read a Tripos MOL2 file and return the first molecule as a Frame.
///
/// Canonical columns on atoms: ``id``, ``name``, ``x``/``y``/``z``, ``type``
/// (the SYBYL atom type), optional ``res_id``/``res_name`` (the MOL2
/// substructure) and ``charge``. Bonds carry ``atomi``/``atomj`` (0-based),
/// ``type`` (the SYBYL bond type token), and the chemical
/// ``bond_type``/``bond_number`` codes.
///
/// Parameters
/// ----------
/// path : str
///     Path to a ``.mol2`` file.
///
/// Returns
/// -------
/// Frame
#[pyfunction]
pub fn read_mol2(path: PathBuf) -> PyResult<PyFrame> {
    let path = path_str(&path)?;
    let frame = molrs::io::read_mol2(path).map_err(io_error_to_pyerr)?;
    PyFrame::from_core_frame(frame)
}

/// Read an AMBER ASCII inpcrd / restrt coordinate file.
///
/// Fixed-width Fortran ``6F12.7`` layout. Without ``frame``, returns a new
/// Frame with ``id``, ``name``, ``x``/``y``/``z``, optional ``vel`` (shape
/// ``[n, 3]``), optional box, and meta ``title`` / ``timestep``.
///
/// With ``frame`` (say, the structure :func:`read_prmtop` read), the
/// file's coordinates go into that frame in place and it is returned: ``x`` /
/// ``y`` / ``z`` (and ``vel``) replace those atom columns, every other column
/// stays; the file's box and meta keys are set. A frame without an ``atoms``
/// block receives the file's whole block.
///
/// Parameters
/// ----------
/// path : str
///     Path to a ``.inpcrd`` or restart file.
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
/// IOError
///     If the file cannot be opened or parsed, or ``frame``'s atom count
///     differs from the file's (``frame`` is then unchanged).
#[pyfunction]
#[pyo3(signature = (path, frame = None))]
pub fn read_inpcrd<'py>(
    py: Python<'py>,
    path: PathBuf,
    frame: Option<Bound<'py, PyFrame>>,
) -> PyResult<Bound<'py, PyAny>> {
    let path = path_str(&path)?;
    match frame {
        None => {
            let frame = molrs::io::read_inpcrd(path).map_err(io_error_to_pyerr)?;
            Ok(Bound::new(py, PyFrame::from_core_frame(frame)?)?.into_any())
        }
        Some(target) => {
            let coordinates = molrs::io::read_inpcrd(path).map_err(io_error_to_pyerr)?;
            target
                .borrow()
                .with_frame_mut(|f| molrs::io::amber::merge_inpcrd(f, coordinates))?
                .map_err(io_error_to_pyerr)?;
            Ok(target.into_any())
        }
    }
}

/// Read an AMBER prmtop **structure** file into a Frame.
///
/// Structure / connectivity only: atoms (name, type, charge in electron units,
/// mass, optional atomic_number/element, res_id), bonds/angles/dihedrals with
/// 0-based indices and type labels, plus POINTERS meta. Force-field parameter
/// tables are not assembled.
///
/// Parameters
/// ----------
/// path : str
///     Path to a ``.prmtop`` / ``.parm7`` file.
///
/// Returns
/// -------
/// Frame
///
/// Raises
/// ------
/// IOError
///     If the file cannot be opened or parsed.
#[pyfunction]
pub fn read_prmtop(path: PathBuf) -> PyResult<PyFrame> {
    let path = path_str(&path)?;
    let frame = molrs::io::read_prmtop(path).map_err(io_error_to_pyerr)?;
    PyFrame::from_core_frame(frame)
}

/// Read an Antechamber ``.ac`` file into a Frame.
#[pyfunction]
pub fn read_ac(path: PathBuf) -> PyResult<PyFrame> {
    let path = path_str(&path)?;
    let frame = molrs::io::read_ac(path).map_err(io_error_to_pyerr)?;
    PyFrame::from_core_frame(frame)
}

/// Read an Amber prep file into a nested dict (serde JSON shape).
#[pyfunction]
pub fn read_prep<'py>(py: Python<'py>, path: PathBuf) -> PyResult<Bound<'py, PyDict>> {
    let path = path_str(&path)?;
    let res = molrs::io::read_prep(path).map_err(io_error_to_pyerr)?;
    prep_residue_to_pydict(py, &res)
}

/// Write an Amber prep residue from a nested dict.
#[pyfunction]
pub fn write_prep(path: PathBuf, residue: &Bound<'_, PyAny>) -> PyResult<()> {
    let path = path_str(&path)?;
    let res = py_to_prep_residue(residue)?;
    molrs::io::write_prep(path, &res).map_err(io_error_to_pyerr)
}

pub(super) fn py_to_prep_residue(residue: &Bound<'_, PyAny>) -> PyResult<PrepResidue> {
    let name: String = residue.get_item("name")?.extract()?;
    let atoms_list = residue.get_item("atoms")?;
    let mut atoms = Vec::new();
    for item in atoms_list.try_iter()? {
        let d = item?;
        atoms.push(PrepAtom {
            index: d.get_item("index")?.extract()?,
            name: d.get_item("name")?.extract()?,
            atom_type: d.get_item("atom_type")?.extract()?,
            tree_type: d
                .get_item("tree_type")
                .ok()
                .and_then(|v| v.extract().ok())
                .unwrap_or_else(|| "M".into()),
            na: d
                .get_item("na")
                .ok()
                .and_then(|v| v.extract().ok())
                .unwrap_or(0),
            nb: d
                .get_item("nb")
                .ok()
                .and_then(|v| v.extract().ok())
                .unwrap_or(0),
            nc: d
                .get_item("nc")
                .ok()
                .and_then(|v| v.extract().ok())
                .unwrap_or(0),
            r: d.get_item("r")
                .ok()
                .and_then(|v| v.extract().ok())
                .unwrap_or(0.0),
            theta: d
                .get_item("theta")
                .ok()
                .and_then(|v| v.extract().ok())
                .unwrap_or(0.0),
            phi: d
                .get_item("phi")
                .ok()
                .and_then(|v| v.extract().ok())
                .unwrap_or(0.0),
            charge: d
                .get_item("charge")
                .ok()
                .and_then(|v| v.extract().ok())
                .unwrap_or(0.0),
            element: d
                .get_item("element")
                .ok()
                .and_then(|v| v.extract().ok())
                .unwrap_or_default(),
        });
    }
    let mut impropers = Vec::new();
    if let Ok(imps) = residue.get_item("impropers") {
        for item in imps.try_iter()? {
            let row: Vec<String> = item?.extract()?;
            impropers.push(row);
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

pub(super) fn prep_residue_to_pydict<'py>(
    py: Python<'py>,
    res: &PrepResidue,
) -> PyResult<Bound<'py, PyDict>> {
    let value = serde_json::to_value(res)
        .map_err(|e| PyValueError::new_err(format!("failed to serialize prep residue: {e}")))?;
    match value {
        JsonValue::Object(map) => json_object_to_pydict(py, &map),
        _ => Err(PyValueError::new_err(
            "internal error: prep residue did not serialize to an object",
        )),
    }
}

/// Write a Frame to a Tripos MOL2 file.
///
/// Expects format-native atom columns (``atom_type``, optional
/// ``subst_id``/``subst_name``). Canonical renames are applied by the
/// :mod:`molrs.io` façade before calling this binding.
///
/// Parameters
/// ----------
/// path : str
///     Output file path.
/// frame : Frame
///     Frame to write.
#[pyfunction]
pub fn write_mol2(path: PathBuf, frame: &PyFrame) -> PyResult<()> {
    let path = path_str(&path)?;
    frame.with_frame(|f| molrs::io::write_mol2(path, f).map_err(io_error_to_pyerr))?
}

/// Read a LAMMPS molecule template (native ``.mol`` or JSON).
#[pyfunction]
pub fn read_lammps_molecule(path: PathBuf) -> PyResult<PyFrame> {
    let path = path_str(&path)?;
    let frame = molrs::io::read_lammps_molecule(path).map_err(io_error_to_pyerr)?;
    PyFrame::from_core_frame(frame)
}

/// Read an XSF (XCrySDen Structure File) and return a Frame.
///
/// Crystal structures (`CRYSTAL` + `PRIMVEC`/`CONVVEC`) yield a periodic box;
/// molecular structures (`MOLECULE`) yield a free box. Atoms carry
/// ``atomic_number``, ``element``, and ``x``/``y``/``z`` (Å).
///
/// Parameters
/// ----------
/// path : str
///     Path to a ``.xsf`` file.
///
/// Returns
/// -------
/// Frame
///
/// Raises
/// ------
/// IOError
///     If the file cannot be opened or parsed.
#[pyfunction]
pub fn read_xsf(path: PathBuf) -> PyResult<PyFrame> {
    let path = path_str(&path)?;
    let frame = molrs::io::read_xsf(path).map_err(io_error_to_pyerr)?;
    PyFrame::from_core_frame(frame)
}

/// Write a Frame to an XSF (XCrySDen Structure File).
///
/// A defined periodic box produces `CRYSTAL` + `PRIMVEC`/`CONVVEC`; otherwise
/// the structure is written as `MOLECULE`. Atoms need ``atomic_number`` and
/// ``x``/``y``/``z``.
///
/// Parameters
/// ----------
/// path : str
///     Output file path.
/// frame : Frame
///     Frame to write.
///
/// Raises
/// ------
/// IOError
///     If the file cannot be written.
#[pyfunction]
pub fn write_xsf(path: PathBuf, frame: &PyFrame) -> PyResult<()> {
    let path = path_str(&path)?;
    frame.with_frame(|f| molrs::io::write_xsf(path, f).map_err(io_error_to_pyerr))?
}

/// Write a Frame as a LAMMPS molecule template, in the native text format
/// (:func:`write_lammps_molecule_json` writes the JSON one).
///
/// Parameters
/// ----------
/// path : str
///     Output path.
/// frame : Frame
///     Molecule frame.
#[pyfunction]
pub fn write_lammps_molecule(path: PathBuf, frame: &PyFrame) -> PyResult<()> {
    let path = path_str(&path)?;
    frame.with_frame(|f| molrs::io::write_lammps_molecule(path, f).map_err(io_error_to_pyerr))?
}

/// Read a LAMMPS molecule template in its JSON format (``format: "molecule"``).
///
/// Parameters
/// ----------
/// path : str | os.PathLike
///     Path to the ``.json`` molecule file.
///
/// Returns
/// -------
/// Frame
#[pyfunction]
pub fn read_lammps_molecule_json(path: PathBuf) -> PyResult<PyFrame> {
    let path = path_str(&path)?;
    let frame = molrs::io::read_lammps_molecule_json(path).map_err(io_error_to_pyerr)?;
    PyFrame::from_core_frame(frame)
}

/// Write a frame as a LAMMPS molecule template in its JSON format — the
/// inverse of :func:`read_lammps_molecule_json`.
///
/// Parameters
/// ----------
/// path : str | os.PathLike
///     File to write.
/// frame : Frame
///     The template.
#[pyfunction]
pub fn write_lammps_molecule_json(path: PathBuf, frame: &PyFrame) -> PyResult<()> {
    let path = path_str(&path)?;
    frame
        .with_frame(|f| molrs::io::write_lammps_molecule_json(path, f).map_err(io_error_to_pyerr))?
}

// ============================================================================
// Writers
// ============================================================================

/// Write a Frame to a PDB file.
///
/// Parameters
/// ----------
/// path : str
///     Output file path.
/// frame : Frame
///     Frame to write.
#[pyfunction]
pub fn write_pdb(path: PathBuf, frame: &PyFrame) -> PyResult<()> {
    let path = path_str(&path)?;
    frame.with_frame(|f| molrs::io::write_pdb(path, f).map_err(io_error_to_pyerr))?
}

/// Write a list of Frames to a multi-MODEL PDB trajectory.
///
/// Each frame becomes one ``MODEL``/``ENDMDL`` block; a shared ``CRYST1`` is
/// written once from the first frame. Inverse of :func:`read_pdb_trajectory`.
///
/// Parameters
/// ----------
/// path : str
///     Output file path.
/// frames : list[Frame]
///     Frames to write, in order.
#[pyfunction]
pub fn write_pdb_trajectory(path: PathBuf, frames: Vec<PyFrame>) -> PyResult<()> {
    let path = path_str(&path)?;
    let core_frames: Vec<_> = frames
        .iter()
        .map(|f| f.clone_core_frame())
        .collect::<PyResult<_>>()?;
    molrs::io::write_pdb_trajectory(path, &core_frames).map_err(io_error_to_pyerr)
}

/// Write a Frame to an XYZ file.
///
/// Parameters
/// ----------
/// path : str
///     Output file path.
/// frame : Frame
///     Frame to write.
#[pyfunction]
pub fn write_xyz(path: PathBuf, frame: &PyFrame) -> PyResult<()> {
    let path = path_str(&path)?;
    frame.with_frame(|f| molrs::io::write_xyz(path, f).map_err(io_error_to_pyerr))?
}

/// Write Frames as one multi-frame Extended XYZ trajectory, each as
/// :func:`write_xyz` writes it. Inverse of :func:`read_xyz_trajectory`.
#[pyfunction]
pub fn write_xyz_trajectory(path: PathBuf, frames: Vec<PyRef<'_, PyFrame>>) -> PyResult<()> {
    let path = path_str(&path)?;
    let core_frames: Vec<_> = frames
        .iter()
        .map(|f| f.clone_core_frame())
        .collect::<PyResult<_>>()?;
    molrs::io::write_xyz_trajectory(path, &core_frames).map_err(io_error_to_pyerr)
}

/// Write a Frame to a LAMMPS data file.
///
/// The box is ``frame.box``. A frame without one is a non-periodic system and
/// is written inside the bounds of its coordinates widened by 1 length unit on
/// every side (``BOXLESS_MARGIN``): read it with ``boundary s s s``, which
/// shrink-wraps that box to the atoms. A frame meant to be periodic carries
/// its box.
///
/// Atoms, topology, ``Masses`` and the ``* Type Labels`` sections are
/// written; ``* Coeffs`` are not (:func:`molrs.io.write_lammps_data_coeffs_str`
/// returns them for the same labels). A system with Drude particles (a
/// ``drudes`` block, or atoms whose ``vsite`` is ``"drude"``) gets a header
/// comment with the ``fix drude`` C/D/N flags in atom-type order.
///
/// Parameters
/// ----------
/// path : str
///     Output file path.
/// frame : Frame
///     Frame to write.
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
/// IOError
///     If the frame cannot be written.
#[pyfunction]
#[pyo3(signature = (path, frame, *, type_labels = None))]
pub fn write_lammps_data(
    path: PathBuf,
    frame: &PyFrame,
    type_labels: Option<std::collections::BTreeMap<String, Vec<String>>>,
) -> PyResult<()> {
    let path = path_str(&path)?;
    match type_labels {
        None => frame
            .with_frame(|f| molrs::io::write_lammps_data(path, f).map_err(io_error_to_pyerr))?,
        Some(extra) => {
            let mut work = frame.clone_core_frame()?;
            for (block, labels) in &extra {
                molrs::core::TypeLabels::declare(&mut work, block, labels)
                    .map_err(pyo3::exceptions::PyValueError::new_err)?;
            }
            molrs::io::write_lammps_data(path, &work).map_err(io_error_to_pyerr)
        }
    }
}

/// Register this module's classes and functions.
/// Read the first record of an MDL SDF / molfile (V2000).
///
/// Parameters
/// ----------
/// path : str | os.PathLike
///     Path to a ``.sdf`` / ``.mol`` file.
///
/// Returns
/// -------
/// Frame
///     ``atoms`` (``element``, ``id``, ``x``/``y``/``z``) and, when the record
///     has any, ``bonds``.
#[pyfunction]
pub fn read_sdf(path: PathBuf) -> PyResult<PyFrame> {
    let path = path_str(&path)?;
    let frame = molrs::io::read_sdf(path).map_err(io_error_to_pyerr)?;
    PyFrame::from_core_frame(frame)
}

/// Read the first ``data_`` block of a CIF file.
///
/// Parameters
/// ----------
/// path : str | os.PathLike
///     Path to a ``.cif`` file.
///
/// Returns
/// -------
/// Frame
#[pyfunction]
pub fn read_cif(path: PathBuf) -> PyResult<PyFrame> {
    let path = path_str(&path)?;
    let frame = molrs::io::read_cif(path).map_err(io_error_to_pyerr)?;
    PyFrame::from_core_frame(frame)
}

/// Write a frame as a single-block CIF file — the inverse of :func:`read_cif`.
#[pyfunction]
pub fn write_cif(path: PathBuf, frame: &PyFrame) -> PyResult<()> {
    let path = path_str(&path)?;
    frame.with_frame(|f| molrs::io::write_cif(path, f).map_err(io_error_to_pyerr))?
}

/// Read a VASP POSCAR / CONTCAR structure file.
///
/// Parameters
/// ----------
/// path : str | os.PathLike
///     Path to a POSCAR / CONTCAR / ``.vasp`` file.
///
/// Returns
/// -------
/// Frame
#[pyfunction]
pub fn read_poscar(path: PathBuf) -> PyResult<PyFrame> {
    let path = path_str(&path)?;
    let frame = molrs::io::read_poscar(path).map_err(io_error_to_pyerr)?;
    PyFrame::from_core_frame(frame)
}

/// Write a frame as a VASP POSCAR file — the inverse of
/// :func:`read_poscar`.
#[pyfunction]
pub fn write_poscar(path: PathBuf, frame: &PyFrame) -> PyResult<()> {
    let path = path_str(&path)?;
    frame.with_frame(|f| molrs::io::write_poscar(path, f).map_err(io_error_to_pyerr))?
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    crate::add_function(m, "molrs.io", wrap_pyfunction!(read_ac, m)?)?;
    crate::add_function(m, "molrs.io", wrap_pyfunction!(read_inpcrd, m)?)?;
    crate::add_function(m, "molrs.io", wrap_pyfunction!(read_prep, m)?)?;
    crate::add_function(m, "molrs.io", wrap_pyfunction!(read_prmtop, m)?)?;
    crate::add_function(m, "molrs.io", wrap_pyfunction!(read_cif, m)?)?;
    crate::add_function(m, "molrs.io", wrap_pyfunction!(read_cube, m)?)?;
    crate::add_function(m, "molrs.io", wrap_pyfunction!(read_gro, m)?)?;
    crate::add_function(m, "molrs.io", wrap_pyfunction!(read_lammps_data, m)?)?;
    crate::add_function(m, "molrs.io", wrap_pyfunction!(read_lammps_molecule, m)?)?;
    crate::add_function(
        m,
        "molrs.io",
        wrap_pyfunction!(read_lammps_molecule_json, m)?,
    )?;
    crate::add_function(m, "molrs.io", wrap_pyfunction!(read_mol2, m)?)?;
    crate::add_function(m, "molrs.io", wrap_pyfunction!(read_pdb, m)?)?;
    crate::add_function(m, "molrs.io", wrap_pyfunction!(read_sdf, m)?)?;
    crate::add_function(m, "molrs.io", wrap_pyfunction!(read_chgcar, m)?)?;
    crate::add_function(m, "molrs.io", wrap_pyfunction!(read_poscar, m)?)?;
    crate::add_function(m, "molrs.io", wrap_pyfunction!(read_xsf, m)?)?;
    crate::add_function(m, "molrs.io", wrap_pyfunction!(read_xyz, m)?)?;
    crate::add_function(m, "molrs.io", wrap_pyfunction!(write_prep, m)?)?;
    crate::add_function(m, "molrs.io", wrap_pyfunction!(write_cif, m)?)?;
    crate::add_function(m, "molrs.io", wrap_pyfunction!(write_cube, m)?)?;
    crate::add_function(m, "molrs.io", wrap_pyfunction!(write_gro, m)?)?;
    crate::add_function(m, "molrs.io", wrap_pyfunction!(write_gro_trajectory, m)?)?;
    crate::add_function(m, "molrs.io", wrap_pyfunction!(write_lammps_data, m)?)?;
    crate::add_function(m, "molrs.io", wrap_pyfunction!(write_lammps_molecule, m)?)?;
    crate::add_function(
        m,
        "molrs.io",
        wrap_pyfunction!(write_lammps_molecule_json, m)?,
    )?;
    crate::add_function(m, "molrs.io", wrap_pyfunction!(write_mol2, m)?)?;
    crate::add_function(m, "molrs.io", wrap_pyfunction!(write_pdb, m)?)?;
    crate::add_function(m, "molrs.io", wrap_pyfunction!(write_pdb_trajectory, m)?)?;
    crate::add_function(m, "molrs.io", wrap_pyfunction!(write_poscar, m)?)?;
    crate::add_function(m, "molrs.io", wrap_pyfunction!(write_xsf, m)?)?;
    crate::add_function(m, "molrs.io", wrap_pyfunction!(write_xyz, m)?)?;
    crate::add_function(m, "molrs.io", wrap_pyfunction!(write_xyz_trajectory, m)?)?;
    Ok(())
}
