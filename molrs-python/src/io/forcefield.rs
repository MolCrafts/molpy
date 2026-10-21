//! The force-field file classes of `molrs.io`, each `molrs::io`'s class of
//! the same name: a reader's `read(source) -> ForceField` is
//! `ForceFieldReader::read`, a writer's `write(sink, forcefield)` is
//! `ForceFieldWriter::write`, and what else a format reads or writes is a
//! method of the same class (`read_system`, `write_system`, `read_cmap`,
//! `write_cmap`, `read_data_coeffs`, `write_data_coeffs`). The format's
//! settings are the constructor's keyword arguments — the core class's
//! `with_*` setters.
//!
//! | class | Python module |
//! |---|---|
//! | `LammpsForcefieldReader` / `LammpsForcefieldWriter` | `molrs.io.lammps` |
//! | `GromacsTopForcefieldReader` / `GromacsTopForcefieldWriter` | `molrs.io.gromacs` |
//! | `AmberPrmtopForcefieldReader` / `AmberFrcmodWriter` | `molrs.io.amber` |
//! | `OpenmmXmlReader` / `OpenmmXmlWriter` | `molrs.io.openmm_xml` |
//! | `XmlForcefieldReader` / `XmlForcefieldWriter` | `molrs.io` |

use std::path::PathBuf;

use molrs::core::TypeLabels;
use molrs::io::reader::ForceFieldReader;
use molrs::io::writer::{ForceFieldWriteError, ForceFieldWriter};
use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;

use super::{PySink, PySource, data_error};
use crate::core::frame::PyFrame;
use crate::ff::forcefield::PyForceField;

/// A force-field writer's error: a refusal of the force field as
/// [`crate::ff::ir::writer_err`] raises it (the typed IR refusal's subclass,
/// else ``ValueError``), the sink's OS failure as ``OSError``.
fn write_error(e: ForceFieldWriteError) -> PyErr {
    use std::io::ErrorKind;
    match e.kind() {
        ErrorKind::InvalidData | ErrorKind::InvalidInput | ErrorKind::Unsupported => {
            crate::ff::ir::writer_err(e)
        }
        _ if e.ir().is_some() => crate::ff::ir::writer_err(e),
        _ => PyIOError::new_err(e.to_string()),
    }
}

/// A read force field as its Python object.
fn read_into(read: std::io::Result<molrs::ff::forcefield::ForceField>) -> PyResult<PyForceField> {
    read.map(|inner| PyForceField { inner }).map_err(data_error)
}

/// A read `(force field, frame)` pair as its Python objects.
fn system_into(
    read: std::io::Result<(molrs::ff::forcefield::ForceField, molrs::core::Frame)>,
) -> PyResult<(PyForceField, PyFrame)> {
    let (inner, frame) = read.map_err(data_error)?;
    Ok((PyForceField { inner }, PyFrame::from_core_frame(frame)?))
}

// ---------------------------------------------------------------------------
// LAMMPS
// ---------------------------------------------------------------------------

/// Reader of LAMMPS force-field files — ``molrs::io::lammps::LammpsForcefieldReader``.
///
/// :meth:`read` parses a force-field include (``*.ff``): the
/// ``pair_style``/``pair_coeff`` + ``bond_style``/``angle_style``/
/// ``dihedral_style``/``improper_style`` lines :class:`LammpsForcefieldWriter`
/// writes. The force-field IR follows the LAMMPS standard, so every
/// coefficient is stored as written (``K``, degrees) and the force field
/// declares the file's ``units`` (``real`` for a file without a ``units``
/// line); the ``special_bonds`` line is recorded on the force field (an
/// include that omits it is refused). Per-atom charge and mass live in the
/// data file, not the include.
///
/// :meth:`read_data_coeffs` reads the ``* Coeffs`` sections of a data file,
/// :meth:`read_cmap` a ``fix cmap`` file.
///
/// Examples
/// --------
/// >>> ff = LammpsForcefieldReader().read(pathlib.Path("system.ff"))  # doctest: +SKIP
#[pyclass(module = "molrs.io.lammps", name = "LammpsForcefieldReader", frozen)]
pub struct PyLammpsForcefieldReader {
    inner: molrs::io::lammps::LammpsForcefieldReader,
}

#[pymethods]
impl PyLammpsForcefieldReader {
    #[new]
    fn new() -> Self {
        Self {
            inner: molrs::io::lammps::LammpsForcefieldReader::new(),
        }
    }

    /// Read the force-field include ``source`` holds.
    ///
    /// Parameters
    /// ----------
    /// source : os.PathLike | bytes | molrs.io.Text
    ///     The file, or its whole content.
    ///
    /// Returns
    /// -------
    /// ForceField
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     On an unsupported style, a coefficient before its style
    ///     declaration, a wrong-arity type label, a non-numeric parameter or
    ///     a missing ``special_bonds`` line (reading is total — never a
    ///     silent skip).
    fn read(&self, source: PySource<'_>) -> PyResult<PyForceField> {
        read_into(source.read(|s| self.inner.read(s)))
    }

    /// Read the force field a LAMMPS data file's ``* Coeffs`` sections
    /// define, from the frame :func:`molrs.io.read_lammps_data` returned —
    /// the file is read once, for the system and its coefficients alike.
    ///
    /// A row's numeric type id is named by the label the file's
    /// ``* Type Labels`` section gave it, else by the id itself. A section's
    /// ``# style`` comment selects the style (an accelerator suffix removed);
    /// a section without one is ``harmonic`` (bonded) or ``lj/cut`` (pair,
    /// cutoff 10.0 in the file's units).
    ///
    /// Parameters
    /// ----------
    /// frame : Frame
    ///     The data file's frame (``frame.meta["lammps_coeffs_text"]``).
    /// units : str, optional
    ///     The unit style the coefficients are in; the one the file's title
    ///     line stated when omitted.
    ///
    /// Returns
    /// -------
    /// ForceField
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     A frame with no ``* Coeffs`` sections, a ``units`` that disagrees
    ///     with the file's, a malformed type-label inventory, an unsupported
    ///     hinted style, or a malformed coefficient row.
    #[pyo3(signature = (frame, units = None))]
    fn read_data_coeffs(&self, frame: &PyFrame, units: Option<&str>) -> PyResult<PyForceField> {
        read_into(frame.with_frame(|f| self.inner.read_data_coeffs(f, units))?)
    }

    /// Read a LAMMPS ``fix cmap`` file (CHARMM's 24×24 grids) into a force
    /// field of ``cmap charmm`` rows — the inverse of
    /// :meth:`LammpsForcefieldWriter.write_cmap`.
    ///
    /// Parameters
    /// ----------
    /// source : os.PathLike | bytes | molrs.io.Text
    ///     The file, or its whole content.
    ///
    /// Returns
    /// -------
    /// ForceField
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     A non-numeric token, a file with no map, or a ``UNITS:`` tag that
    ///     is no LAMMPS unit style.
    fn read_cmap(&self, source: PySource<'_>) -> PyResult<PyForceField> {
        read_into(source.read(|s| self.inner.read_cmap(s)))
    }

    fn __repr__(&self) -> &'static str {
        "LammpsForcefieldReader()"
    }
}

/// Writer of LAMMPS force-field files for a typed system —
/// ``molrs::io::lammps::LammpsForcefieldWriter``.
///
/// The frame's type labels (``TypeLabels.from_frame``) number the rows, so
/// the include matches the data file :func:`molrs.io.write_lammps_data`
/// writes for the same frame. :meth:`write` writes the force-field include
/// (``*.ff``), :meth:`write_data_coeffs` the data file's ``* Coeffs``
/// sections, :meth:`write_cmap` the ``fix cmap`` file.
///
/// Parameters
/// ----------
/// frame : Frame
///     The typed system whose labels number the rows.
/// precision : int
///     Decimal places for floating-point coefficients.
/// units : str
///     The LAMMPS ``units`` style the file is written in.
/// skip_pair_style : bool
///     Omit the ``pair_style`` line (the input script sets its own).
/// skip_special_bonds : bool
///     Omit ``special_bonds`` (the input script states its own 1-4 weights).
/// skip_units : bool
///     Omit the ``units`` line.
/// cmap_file : str, optional
///     The ``fix cmap`` file the include names, as the input script will
///     find it; needed exactly when the system has CMAP crossterms.
///
/// Raises
/// ------
/// ValueError
///     If ``units`` is no LAMMPS unit style or the frame's type-label
///     inventory is malformed.
#[pyclass(module = "molrs.io.lammps", name = "LammpsForcefieldWriter", frozen)]
pub struct PyLammpsForcefieldWriter {
    labels: TypeLabels,
    /// Why the frame's `pairs` block cannot be written as an include (a
    /// per-pair override column), raised by `write`.
    pair_refusal: Option<String>,
    precision: usize,
    units: &'static str,
    skip_pair_style: bool,
    skip_special_bonds: bool,
    skip_units: bool,
    cmap_file: Option<String>,
}

impl PyLammpsForcefieldWriter {
    fn writer(&self) -> molrs::io::lammps::LammpsForcefieldWriter<'_> {
        let writer = molrs::io::lammps::LammpsForcefieldWriter::new(&self.labels)
            .with_precision(self.precision)
            .with_units(self.units)
            .with_skip_pair_style(self.skip_pair_style)
            .with_skip_special_bonds(self.skip_special_bonds)
            .with_skip_units(self.skip_units);
        match &self.cmap_file {
            Some(file) => writer.with_cmap_file(file.clone()),
            None => writer,
        }
    }
}

#[pymethods]
impl PyLammpsForcefieldWriter {
    #[new]
    #[pyo3(signature = (
        frame,
        *,
        precision = 6,
        units = "real",
        skip_pair_style = false,
        skip_special_bonds = false,
        skip_units = false,
        cmap_file = None,
    ))]
    #[allow(clippy::too_many_arguments, reason = "Public Python keyword arguments")]
    fn new(
        frame: &PyFrame,
        precision: usize,
        units: &str,
        skip_pair_style: bool,
        skip_special_bonds: bool,
        skip_units: bool,
        cmap_file: Option<String>,
    ) -> PyResult<Self> {
        let units =
            molrs::io::lammps::parse_lammps_units_style(units).map_err(PyValueError::new_err)?;
        let (labels, pair_refusal) = frame.with_frame(|f| {
            (
                molrs::io::lammps::lammps_type_labels(f),
                molrs::io::lammps::refuse_pair_overrides(f).err(),
            )
        })?;
        Ok(Self {
            labels: labels.map_err(PyValueError::new_err)?,
            pair_refusal,
            precision,
            units,
            skip_pair_style,
            skip_special_bonds,
            skip_units,
            cmap_file,
        })
    }

    /// Write ``forcefield`` as the LAMMPS force-field include (``*.ff``).
    ///
    /// Parameters
    /// ----------
    /// sink : os.PathLike | bytearray
    ///     The file, or a buffer that then holds exactly its bytes.
    /// forcefield : ForceField
    ///     The force field.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     A per-pair 1-4 override column on the frame's ``pairs`` (LAMMPS
    ///     has no per-pair exception), or what the format cannot express
    ///     (the typed IR refusal's subclass).
    /// OSError
    ///     If the file cannot be written.
    fn write(&self, sink: PySink<'_>, forcefield: &PyForceField) -> PyResult<()> {
        if let Some(refusal) = &self.pair_refusal {
            return Err(crate::ff::ir::writer_err(refusal.clone().into()));
        }
        sink.put(|s| self.writer().write(s, &forcefield.inner), write_error)
    }

    /// Write the data file's ``* Coeffs`` sections only (no ``units`` /
    /// ``*_style`` lines): a fragment for the caller to splice into a data
    /// file. ``Pair Coeffs`` holds self pairs only, so a used explicit cross
    /// pair is refused here (write it through :meth:`write`).
    ///
    /// Parameters
    /// ----------
    /// sink : os.PathLike | bytearray
    ///     The file, or a buffer that then holds exactly its bytes.
    /// forcefield : ForceField
    ///     The force field.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     What the include refuses, and a used explicit cross pair.
    /// OSError
    ///     If the file cannot be written.
    fn write_data_coeffs(&self, sink: PySink<'_>, forcefield: &PyForceField) -> PyResult<()> {
        sink.put(
            |s| self.writer().write_data_coeffs(s, &forcefield.inner),
            write_error,
        )
    }

    /// Write the LAMMPS ``fix cmap`` file of the ``cmaps`` labels — the
    /// inverse of :meth:`LammpsForcefieldReader.read_cmap`.
    ///
    /// Parameters
    /// ----------
    /// sink : os.PathLike | bytearray
    ///     The file, or a buffer that then holds exactly its bytes.
    /// forcefield : ForceField
    ///     The force field.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     No ``cmaps`` label, a label with no cmap type, a grid that is not
    ///     24×24, or more than six maps.
    /// OSError
    ///     If the file cannot be written.
    fn write_cmap(&self, sink: PySink<'_>, forcefield: &PyForceField) -> PyResult<()> {
        sink.put(
            |s| self.writer().write_cmap(s, &forcefield.inner),
            write_error,
        )
    }

    fn __repr__(&self) -> String {
        format!(
            "LammpsForcefieldWriter(precision={}, units={:?})",
            self.precision, self.units
        )
    }
}

// ---------------------------------------------------------------------------
// GROMACS
// ---------------------------------------------------------------------------

/// Reader of GROMACS topologies (``.top`` / ``.itp``) —
/// ``molrs::io::gromacs::GromacsTopForcefieldReader``.
///
/// :meth:`read` reads the force-field directives (``[ defaults ]``,
/// ``[ atomtypes ]``, ``[ nonbond_params ]``, ``[ pairtypes ]``,
/// ``[ bondtypes ]``, ``[ angletypes ]``, ``[ dihedraltypes ]``,
/// ``[ cmaptypes ]``), converting GROMACS units (nm, kJ/mol, ``½k``) to the
/// force-field IR (LAMMPS standard, ``real``); :meth:`read_system` reads the
/// molecule sections too. See the Force-field IR guide, "GROMACS
/// topologies".
///
/// Parameters
/// ----------
/// include : bool
///     Follow ``#include`` directives, relative to the including file and
///     then each of ``include_dirs`` (false: they are ignored).
/// include_dirs : Sequence[os.PathLike]
///     Directories an ``#include`` is resolved against, in order (GROMACS's
///     ``-I``, or its ``share/gromacs/top``).
/// skip_directives : Sequence[str]
///     Sections read past, rows and all, instead of refused (bracket-less,
///     case-insensitive, e.g. ``"constrainttypes"``).
#[pyclass(
    module = "molrs.io.gromacs",
    name = "GromacsTopForcefieldReader",
    frozen
)]
pub struct PyGromacsTopForcefieldReader {
    inner: molrs::io::gromacs::GromacsTopForcefieldReader,
}

#[pymethods]
impl PyGromacsTopForcefieldReader {
    #[new]
    #[pyo3(signature = (*, include = false, include_dirs = Vec::new(), skip_directives = Vec::new()))]
    fn new(include: bool, include_dirs: Vec<PathBuf>, skip_directives: Vec<String>) -> Self {
        let mut inner = molrs::io::gromacs::GromacsTopForcefieldReader::new().with_include(include);
        for dir in include_dirs {
            inner = inner.with_include_dir(dir);
        }
        for name in &skip_directives {
            inner = inner.with_skipped_directive(name);
        }
        Self { inner }
    }

    /// Read the force-field directives ``source`` holds.
    ///
    /// Parameters
    /// ----------
    /// source : os.PathLike | bytes | molrs.io.Text
    ///     The topology, or its whole content (``#include`` needs a path).
    ///
    /// Returns
    /// -------
    /// ForceField
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     What the IR cannot hold, by name: an unsupported function code or
    ///     comb-rule, ``[ constrainttypes ]``, any unknown section, and every
    ///     molecule section (read a whole topology with :meth:`read_system`,
    ///     or skip them).
    fn read(&self, source: PySource<'_>) -> PyResult<PyForceField> {
        read_into(source.read(|s| self.inner.read(s)))
    }

    /// Read a whole topology — its directives and its molecules — into a
    /// force field and a typed frame.
    ///
    /// The frame holds ``atoms`` (``type``, ``charge``, ``mass``, ``name``,
    /// ``res_id``, ``res_name``, ``mol_id``), ``bonds``, ``angles``,
    /// ``dihedrals``, ``impropers``, ``cmaps``, ``constraints``,
    /// ``exclusions`` and ``pairs`` (0-based indices). Coordinates come from
    /// the ``.gro`` (:func:`molrs.io.read_gro`).
    ///
    /// Parameters
    /// ----------
    /// source : os.PathLike | bytes | molrs.io.Text
    ///     The topology, or its whole content.
    ///
    /// Returns
    /// -------
    /// tuple[ForceField, Frame]
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     Anything the IR cannot hold, naming the file, line and row.
    fn read_system(&self, source: PySource<'_>) -> PyResult<(PyForceField, PyFrame)> {
        system_into(source.read(|s| self.inner.read_system(s)))
    }

    fn __repr__(&self) -> &'static str {
        "GromacsTopForcefieldReader()"
    }
}

/// Writer of GROMACS topologies — ``molrs::io::gromacs::GromacsTopForcefieldWriter``.
///
/// Parameters
/// ----------
/// precision : int
///     Decimal places for floating-point coefficients.
#[pyclass(
    module = "molrs.io.gromacs",
    name = "GromacsTopForcefieldWriter",
    frozen
)]
pub struct PyGromacsTopForcefieldWriter {
    inner: molrs::io::gromacs::GromacsTopForcefieldWriter,
}

#[pymethods]
impl PyGromacsTopForcefieldWriter {
    #[new]
    #[pyo3(signature = (*, precision = 6))]
    fn new(precision: usize) -> Self {
        Self {
            inner: molrs::io::gromacs::GromacsTopForcefieldWriter::new().with_precision(precision),
        }
    }

    /// Write ``forcefield``'s directives as a topology.
    ///
    /// Parameters
    /// ----------
    /// sink : os.PathLike | bytearray
    ///     The file, or a buffer that then holds exactly its bytes.
    /// forcefield : ForceField
    ///     The force field.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     What GROMACS cannot express (the typed IR refusal's subclass).
    /// OSError
    ///     If the file cannot be written.
    fn write(&self, sink: PySink<'_>, forcefield: &PyForceField) -> PyResult<()> {
        sink.put(|s| self.inner.write(s, &forcefield.inner), write_error)
    }

    /// Write a whole topology — the directives and the frame's molecules —
    /// the inverse of :meth:`GromacsTopForcefieldReader.read_system`.
    ///
    /// Parameters
    /// ----------
    /// sink : os.PathLike | bytearray
    ///     The file, or a buffer that then holds exactly its bytes.
    /// forcefield : ForceField
    ///     The force field.
    /// frame : Frame
    ///     The typed system.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     What GROMACS cannot express; nothing is written then.
    /// OSError
    ///     If the file cannot be written.
    fn write_system(
        &self,
        sink: PySink<'_>,
        forcefield: &PyForceField,
        frame: &PyFrame,
    ) -> PyResult<()> {
        frame.with_frame(|f| {
            sink.put(
                |s| self.inner.write_system(s, &forcefield.inner, f),
                write_error,
            )
        })?
    }

    fn __repr__(&self) -> &'static str {
        "GromacsTopForcefieldWriter()"
    }
}

// ---------------------------------------------------------------------------
// AMBER
// ---------------------------------------------------------------------------

/// Reader of an AMBER prmtop's force field —
/// ``molrs::io::amber::AmberPrmtopForcefieldReader``.
///
/// :meth:`read` parses the parameter tables: harmonic bonds and angles
/// (``k = K``; θ₀ to degrees), periodic dihedrals (phases to degrees),
/// impropers in AMBER's atom order, and LJ A/B → σ/ε, in LAMMPS ``real``
/// units. The structure alone is :func:`molrs.io.read_prmtop`'s;
/// :meth:`read_system` reads both.
#[pyclass(
    module = "molrs.io.amber",
    name = "AmberPrmtopForcefieldReader",
    frozen
)]
pub struct PyAmberPrmtopForcefieldReader {
    inner: molrs::io::amber::AmberPrmtopForcefieldReader,
}

#[pymethods]
impl PyAmberPrmtopForcefieldReader {
    #[new]
    fn new() -> Self {
        Self {
            inner: molrs::io::amber::AmberPrmtopForcefieldReader::new(),
        }
    }

    /// Read the force field of the prmtop ``source`` holds.
    ///
    /// Parameters
    /// ----------
    /// source : os.PathLike | bytes | molrs.io.Text
    ///     The file, or its whole content.
    ///
    /// Returns
    /// -------
    /// ForceField
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     On a table the IR cannot hold.
    fn read(&self, source: PySource<'_>) -> PyResult<PyForceField> {
        read_into(source.read(|s| self.inner.read(s)))
    }

    /// Read a whole prmtop (or ParmEd chamber) topology into a force field
    /// and a typed frame.
    ///
    /// The frame is :func:`molrs.io.read_prmtop`'s plus a ``pairs`` block
    /// holding the 1-4 pairs sander weighs otherwise than the field's
    /// ``special_bonds``, each with its own ``coul_scale`` / ``lj_scale``
    /// where it differs (no ``pairs`` block when every 1-4 pair agrees).
    /// Coordinates come from the restart (:func:`molrs.io.read_inpcrd`).
    ///
    /// Parameters
    /// ----------
    /// source : os.PathLike | bytes | molrs.io.Text
    ///     The file, or its whole content.
    ///
    /// Returns
    /// -------
    /// tuple[ForceField, Frame]
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     What either reader refuses, and a 1-4 row on a bonded pair or an
    ///     angle's ends.
    fn read_system(&self, source: PySource<'_>) -> PyResult<(PyForceField, PyFrame)> {
        system_into(source.read(|s| self.inner.read_system(s)))
    }

    fn __repr__(&self) -> &'static str {
        "AmberPrmtopForcefieldReader()"
    }
}

/// Writer of an AMBER ``frcmod`` parameter file —
/// ``molrs::io::amber::AmberFrcmodWriter``.
#[pyclass(module = "molrs.io.amber", name = "AmberFrcmodWriter", frozen)]
pub struct PyAmberFrcmodWriter {
    inner: molrs::io::amber::AmberFrcmodWriter,
}

#[pymethods]
impl PyAmberFrcmodWriter {
    #[new]
    fn new() -> Self {
        Self {
            inner: molrs::io::amber::AmberFrcmodWriter::new(),
        }
    }

    /// Write ``forcefield`` as a ``frcmod`` file.
    ///
    /// Parameters
    /// ----------
    /// sink : os.PathLike | bytearray
    ///     The file, or a buffer that then holds exactly its bytes.
    /// forcefield : ForceField
    ///     The force field.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     What ``frcmod`` cannot express (the typed IR refusal's subclass).
    /// OSError
    ///     If the file cannot be written.
    fn write(&self, sink: PySink<'_>, forcefield: &PyForceField) -> PyResult<()> {
        sink.put(|s| self.inner.write(s, &forcefield.inner), write_error)
    }

    fn __repr__(&self) -> &'static str {
        "AmberFrcmodWriter()"
    }
}

// ---------------------------------------------------------------------------
// OpenMM XML
// ---------------------------------------------------------------------------

/// Reader of OpenMM force-field XML — ``molrs::io::openmm_xml::OpenmmXmlReader``.
///
/// Reads OpenMM's own ``<ForceField>`` schema — CHARMM36, AMBER and OPLS-AA
/// ports alike (nm, kJ/mol, radians, ``½k`` harmonic terms) — into the
/// force-field IR (Å, kcal/mol, degrees, un-halved ``K``): harmonic bonds and
/// angles, Urey-Bradley, periodic and Ryckaert-Bellemans torsions, periodic
/// and CHARMM harmonic impropers, CMAP, ``NonbondedForce`` and
/// ``LennardJonesForce`` with NBFIX and 1-4 parameters. molrs's own
/// force-field XML is :class:`molrs.io.xml.XmlForcefieldReader`'s.
#[pyclass(module = "molrs.io.openmm_xml", name = "OpenmmXmlReader", frozen)]
pub struct PyOpenmmXmlReader {
    inner: molrs::io::openmm_xml::OpenmmXmlReader,
}

#[pymethods]
impl PyOpenmmXmlReader {
    #[new]
    fn new() -> Self {
        Self {
            inner: molrs::io::openmm_xml::OpenmmXmlReader::new(),
        }
    }

    /// Read the OpenMM force field ``source`` holds.
    ///
    /// Parameters
    /// ----------
    /// source : os.PathLike | bytes | molrs.io.Text
    ///     The file (``charmm36.xml``, ``oplsaa.xml``, …), or its whole
    ///     content.
    ///
    /// Returns
    /// -------
    /// ForceField
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     On a malformed document, a section or row with no IR form (named),
    ///     or a missing / non-numeric required attribute.
    fn read(&self, source: PySource<'_>) -> PyResult<PyForceField> {
        read_into(source.read(|s| self.inner.read(s)))
    }

    fn __repr__(&self) -> &'static str {
        "OpenmmXmlReader()"
    }
}

/// Writer of OpenMM force-field XML — ``molrs::io::openmm_xml::OpenmmXmlWriter``,
/// the inverse of :class:`OpenmmXmlReader`.
///
/// Parameters
/// ----------
/// precision : int, optional
///     Decimals of every number; the shortest form that reads back to the
///     same float when omitted.
#[pyclass(module = "molrs.io.openmm_xml", name = "OpenmmXmlWriter", frozen)]
pub struct PyOpenmmXmlWriter {
    inner: molrs::io::openmm_xml::OpenmmXmlWriter,
}

#[pymethods]
impl PyOpenmmXmlWriter {
    #[new]
    #[pyo3(signature = (*, precision = None))]
    fn new(precision: Option<usize>) -> Self {
        Self {
            inner: molrs::io::openmm_xml::OpenmmXmlWriter::new().with_precision(precision),
        }
    }

    /// Write ``forcefield`` as an OpenMM force-field XML document.
    ///
    /// Parameters
    /// ----------
    /// sink : os.PathLike | bytearray
    ///     The file, or a buffer that then holds exactly its bytes.
    /// forcefield : ForceField
    ///     The force field.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     What OpenMM cannot express (the typed IR refusal's subclass).
    /// OSError
    ///     If the file cannot be written.
    fn write(&self, sink: PySink<'_>, forcefield: &PyForceField) -> PyResult<()> {
        sink.put(|s| self.inner.write(s, &forcefield.inner), write_error)
    }

    /// Write the force field and the residue templates matching this frame's PDB.
    fn write_system(
        &self,
        sink: PySink<'_>,
        forcefield: &PyForceField,
        frame: &PyFrame,
    ) -> PyResult<()> {
        frame.with_frame(|f| {
            sink.put(
                |s| self.inner.write_system(s, &forcefield.inner, f),
                write_error,
            )
        })?
    }

    fn __repr__(&self) -> &'static str {
        "OpenmmXmlWriter()"
    }
}

// ---------------------------------------------------------------------------
// molrs force-field XML
// ---------------------------------------------------------------------------

/// Reader of molrs's force-field XML — ``molrs::io::XmlForcefieldReader``:
/// every style one ``<{Category}Style name=…>`` element with its ``<Type>``
/// rows, in the units the ``<ForceField units=…>`` root declares. OpenMM's
/// own force-field XML is :class:`molrs.io.openmm_xml.OpenmmXmlReader`'s.
#[pyclass(module = "molrs.io.xml", name = "XmlForcefieldReader", frozen)]
pub struct PyXmlForcefieldReader {
    inner: molrs::io::XmlForcefieldReader,
}

#[pymethods]
impl PyXmlForcefieldReader {
    #[new]
    fn new() -> Self {
        Self {
            inner: molrs::io::XmlForcefieldReader::new(),
        }
    }

    /// Read the force field ``source`` holds.
    ///
    /// Parameters
    /// ----------
    /// source : os.PathLike | bytes | molrs.io.Text
    ///     The file, or its whole content.
    ///
    /// Returns
    /// -------
    /// ForceField
    ///     With the root's ``units`` declared.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     On a malformed document or an element this layout does not have.
    fn read(&self, source: PySource<'_>) -> PyResult<PyForceField> {
        read_into(source.read(|s| self.inner.read(s)))
    }

    fn __repr__(&self) -> &'static str {
        "XmlForcefieldReader()"
    }
}

/// Writer of molrs's force-field XML — ``molrs::io::XmlForcefieldWriter``,
/// the inverse of :class:`XmlForcefieldReader`: every style of the force
/// field as one element, in its declared units.
#[pyclass(module = "molrs.io.xml", name = "XmlForcefieldWriter", frozen)]
pub struct PyXmlForcefieldWriter {
    inner: molrs::io::XmlForcefieldWriter,
}

#[pymethods]
impl PyXmlForcefieldWriter {
    #[new]
    fn new() -> Self {
        Self {
            inner: molrs::io::XmlForcefieldWriter::new(),
        }
    }

    /// Write ``forcefield`` as a force-field XML document.
    ///
    /// Parameters
    /// ----------
    /// sink : os.PathLike | bytearray
    ///     The file, or a buffer that then holds exactly its bytes.
    /// forcefield : ForceField
    ///     The force field.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     A style the layout cannot hold.
    /// OSError
    ///     If the file cannot be written.
    fn write(&self, sink: PySink<'_>, forcefield: &PyForceField) -> PyResult<()> {
        sink.put(|s| self.inner.write(s, &forcefield.inner), write_error)
    }

    fn __repr__(&self) -> &'static str {
        "XmlForcefieldWriter()"
    }
}

/// Register the force-field reader and writer classes on the native module.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyLammpsForcefieldReader>()?;
    m.add_class::<PyLammpsForcefieldWriter>()?;
    m.add_class::<PyGromacsTopForcefieldReader>()?;
    m.add_class::<PyGromacsTopForcefieldWriter>()?;
    m.add_class::<PyAmberPrmtopForcefieldReader>()?;
    m.add_class::<PyAmberFrcmodWriter>()?;
    m.add_class::<PyOpenmmXmlReader>()?;
    m.add_class::<PyOpenmmXmlWriter>()?;
    m.add_class::<PyXmlForcefieldReader>()?;
    m.add_class::<PyXmlForcefieldWriter>()?;
    Ok(())
}
