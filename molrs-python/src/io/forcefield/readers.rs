//! The force-field file readers of `molrs.io` (`read_<fmt>_forcefield`, …):
//! each file format's force-field directives into a [`PyForceField`].

use std::path::PathBuf;

use pyo3::prelude::*;

use crate::core::frame::PyFrame;
use crate::ff::forcefield::PyForceField;
use crate::path::path_str;
use molrs::io::gromacs::GromacsTopReadOptions;

/// Read a force field from a force-field XML file — every style one
/// ``<{Category}Style name=…>`` element (``<BondStyle>``, ``<PairStyle>``,
/// ``<CmapStyle>``, …) with its ``<Type>`` rows, in the units the
/// ``<ForceField units=…>`` root declares; the inverse of
/// :func:`write_xml_forcefield`. OpenMM's own force-field XML is a different
/// document: :func:`read_openmm_forcefield`.
///
/// Parameters
/// ----------
/// path : str or os.PathLike
///     Path to a force-field XML file.
///
/// Returns
/// -------
/// ForceField
///     With the root's ``units`` declared.
///
/// Raises
/// ------
/// ValueError
///     On a malformed document or an element this layout does not have (an
///     OpenMM ``<…Force>`` names :func:`read_openmm_forcefield`).
#[pyfunction]
#[pyo3(name = "read_xml_forcefield")]
pub fn read_xml_forcefield_py(path: PathBuf) -> PyResult<PyForceField> {
    let forcefield = molrs::io::read_xml_forcefield(path_str(&path)?)
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    Ok(PyForceField { inner: forcefield })
}

/// Read an OpenMM force-field XML file into a :class:`ForceField`.
///
/// Reads OpenMM's own ``<ForceField>`` schema — CHARMM36, AMBER and OPLS-AA
/// ports alike (nm, kJ/mol, radians, ``½k`` harmonic terms) — into the
/// force-field IR, whose definitions follow LAMMPS (Å, kcal/mol, degrees,
/// un-halved ``K``): harmonic bonds and angles, Urey-Bradley
/// (``angle charmm``), periodic and Ryckaert-Bellemans torsions
/// (``dihedral periodic`` / ``multi/harmonic``), periodic impropers (stored in
/// the order OpenMM prices them) and CHARMM's harmonic ``CustomTorsionForce``
/// impropers, CMAP (``cmap charmm``), ``NonbondedForce`` (``lj/cut`` +
/// ``coul/cut``) and ``LennardJonesForce`` with NBFIX and 1-4 parameters
/// (``lj/charmm`` + ``coul/charmm``). Coulomb styles state OpenMM's own
/// constant. A field whose ``lj/charmm`` declares ``one_four="epsilon14"``
/// needs :meth:`ForceField.materialize_one_four` on its frames before it
/// compiles. The inverse of :func:`write_openmm_forcefield`; molrs's own
/// force-field XML is :func:`read_xml_forcefield`'s.
///
/// Parameters
/// ----------
/// path : str or os.PathLike
///     Path to an OpenMM force-field XML (``charmm36.xml``, ``oplsaa.xml``, …).
///
/// Returns
/// -------
/// ForceField
///
/// Raises
/// ------
/// ValueError
///     On a malformed document, a section or row with no IR form (named:
///     ``Custom*Force`` other than the harmonic improper, ``<Script>``,
///     ``ordering="smirnoff"``, …), or a missing/non-numeric required
///     attribute (reading is total — never a silent skip).
#[pyfunction]
#[pyo3(name = "read_openmm_forcefield")]
pub fn read_openmm_forcefield_py(path: PathBuf) -> PyResult<PyForceField> {
    let forcefield = molrs::io::read_openmm_forcefield(path_str(&path)?)
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    Ok(PyForceField { inner: forcefield })
}

/// Read a LAMMPS force-field include (``*.ff``) into a :class:`ForceField`.
///
/// Parses the ``pair_style``/``pair_coeff`` + ``bond_style``/``angle_style``/
/// ``dihedral_style``/``improper_style`` include that
/// :func:`write_lammps_forcefield` emits. the force-field IR follows the LAMMPS standard, so
/// every coefficient is stored as written (``K``, degrees) and the force field
/// declares the file's ``units``; the ``fourier`` dihedral is molrs's
/// ``periodic``. The ``special_bonds`` line is recorded on the force field.
/// Distinct from :func:`read_xml_forcefield` (molrs's force-field XML) and
/// :func:`read_openmm_forcefield` (OpenMM's).
///
/// Per-atom charge and mass live in the LAMMPS *data* file, not this include, so
/// they are not read here: Coulomb charges are drawn from the frame at evaluation
/// time and masses are irrelevant to geometry relaxation.
///
/// Parameters
/// ----------
/// path : str or os.PathLike
///     Path to a LAMMPS force-field include (``*.ff``).
///
/// Returns
/// -------
/// ForceField
///
/// Raises
/// ------
/// ValueError
///     On an unsupported style, a coefficient before its style declaration, a
///     wrong-arity type label, or a non-numeric parameter (reading is total —
///     never a silent skip).
#[pyfunction]
#[pyo3(name = "read_lammps_forcefield")]
pub fn read_lammps_forcefield_py(path: PathBuf) -> PyResult<PyForceField> {
    let forcefield = molrs::io::read_lammps_forcefield(&path)
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    Ok(PyForceField { inner: forcefield })
}

/// Read AMBER prmtop force-field parameter tables into a :class:`ForceField`.
///
/// Structure/connectivity is :func:`molrs.io.read_prmtop`; this parses
/// harmonic bond/angle tables (``k = K``, AMBER's and LAMMPS's form; θ₀ to
/// degrees), periodic dihedrals (phases to degrees), impropers in AMBER's atom
/// order, and LJ A/B → σ/ε, in LAMMPS ``real`` units.
#[pyfunction]
#[pyo3(name = "read_prmtop_forcefield")]
pub fn read_prmtop_forcefield_py(path: PathBuf) -> PyResult<PyForceField> {
    let forcefield =
        molrs::io::read_prmtop_forcefield(path).map_err(pyo3::exceptions::PyValueError::new_err)?;
    Ok(PyForceField { inner: forcefield })
}

/// Read a whole AMBER prmtop (or ParmEd chamber) topology into a
/// :class:`ForceField` and a typed :class:`Frame`.
///
/// The force field as :func:`read_prmtop_forcefield` reads it, and the
/// structure as :func:`molrs.io.read_prmtop` reads it, plus a ``pairs``
/// block holding the 1-4 pairs sander weighs otherwise than the field's
/// ``special_bonds`` (a torsion type whose ``SCEE`` / ``SCNB`` differ from
/// the divisor most 1-4 rows carry, as GLYCAM's beside ff14SB's; a pair two
/// rows list), each with its own ``coul_scale`` (Σ 1/SCEE) / ``lj_scale``
/// (Σ 1/SCNB) where it differs, null where it agrees. No ``pairs`` block when
/// every 1-4 pair agrees. It is not a pair list: build the full one with
/// :func:`molrs.ff.potential.intramolecular_pairs`, which keeps these cells.
/// Coordinates come from the restart (:func:`molrs.io.read_inpcrd`).
///
/// Returns ``(forcefield, frame)``. What either reader refuses, and a 1-4
/// row on a bonded pair or an angle's ends, raises ``ValueError``.
#[pyfunction]
#[pyo3(name = "read_prmtop_system")]
pub fn read_prmtop_system_py(path: PathBuf) -> PyResult<(PyForceField, PyFrame)> {
    let (forcefield, frame) =
        molrs::io::read_prmtop_system(&path).map_err(pyo3::exceptions::PyValueError::new_err)?;
    Ok((
        PyForceField { inner: forcefield },
        PyFrame::from_core_frame(frame)?,
    ))
}

/// Read the force-field directives of a GROMACS topology into a
/// :class:`ForceField`.
///
/// Reads ``[ defaults ]`` (nbfunc 1; comb-rule 2 or 3 → ``mixing``
/// ``arithmetic`` / ``geometric``; gen-pairs, fudgeLJ, fudgeQQ →
/// ``special_bonds``), ``[ atomtypes ]``, ``[ nonbond_params ]`` (explicit
/// cross rows), ``[ pairtypes ]`` (``lj/charmm`` ``epsilon14`` / ``sigma14``,
/// declared ``one_four = "epsilon14"``), ``[ bondtypes ]`` (funct 1, 3),
/// ``[ angletypes ]`` (funct 1, 5 → ``angle charmm``), ``[ dihedraltypes ]``
/// (funct 1, 9 → ``dihedral periodic``; 3 → ``multi/harmonic`` /
/// ``nharmonic``; 5 → ``opls``; 2, 4 → impropers) and ``[ cmaptypes ]``
/// (``cmap charmm``), converting GROMACS units (nm, kJ/mol, ``½k`` harmonic
/// terms) to the force-field IR (LAMMPS standard, ``real``: Å, kcal/mol,
/// ``K = k/2``; degrees stay degrees). See the Force-field IR guide,
/// "GROMACS topologies".
///
/// What the IR cannot hold raises ``ValueError`` naming it: an unsupported
/// function code or comb-rule, ``[ constrainttypes ]``,
/// ``[ implicit_genborn_params ]``, any unknown section, and every molecule
/// section — read a whole topology with :func:`read_top_system`, or skip
/// them here.
///
/// ``include`` follows ``#include`` relative to the including file and then
/// each of ``include_dirs`` (default false: ignored). Each name in
/// ``skip_directives`` (bracket-less, case-insensitive, e.g.
/// ``"constrainttypes"``) is read past, rows and all, instead of refused.
#[pyfunction]
#[pyo3(
    name = "read_top_forcefield",
    signature = (path, include = false, *, include_dirs = Vec::new(), skip_directives = Vec::new())
)]

pub fn read_top_forcefield_py(
    path: PathBuf,
    include: bool,
    include_dirs: Vec<PathBuf>,
    skip_directives: Vec<String>,
) -> PyResult<PyForceField> {
    let options = GromacsTopReadOptions {
        include,
        include_dirs,
        skip_directives,
    };
    let forcefield = molrs::io::read_top_forcefield(&path, &options)
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    Ok(PyForceField { inner: forcefield })
}

/// Read a whole GROMACS topology into a :class:`ForceField` and a typed
/// :class:`Frame`.
///
/// The directives as :func:`read_top_forcefield` reads them, and the molecule
/// sections: ``atoms`` (``type``, ``charge``, ``mass``, ``name``, ``res_id``,
/// ``res_name``, ``mol_id``), ``bonds``, ``angles``, ``dihedrals``,
/// ``impropers``, ``cmaps`` (each row's ``type`` the force-field type
/// GROMACS's own lookup picks; a row with parameters of its own gets a type
/// of its own), ``constraints``, ``exclusions`` and ``pairs`` (every
/// intramolecular pair GROMACS prices; the ``[ pairs ]`` rows flagged
/// ``is_14``, with per-pair override columns where they carry parameters).
/// Atom indices are 0-based. ``#include`` is followed, relative to the
/// including file and then each of ``include_dirs`` (GROMACS's
/// ``share/gromacs/top`` for ``#include "charmm27.ff/forcefield.itp"``).
/// Coordinates come from the ``.gro`` (:func:`molrs.io.read_gro`).
///
/// Returns ``(forcefield, frame)``. Anything the IR cannot hold (virtual
/// sites, restraints, free-energy B states, …) raises ``ValueError`` naming
/// it and where.
#[pyfunction]
#[pyo3(
    name = "read_top_system",
    signature = (path, *, include_dirs = Vec::new(), skip_directives = Vec::new())
)]

pub fn read_top_system_py(
    path: PathBuf,
    include_dirs: Vec<PathBuf>,
    skip_directives: Vec<String>,
) -> PyResult<(PyForceField, PyFrame)> {
    let options = GromacsTopReadOptions {
        include: true,
        include_dirs,
        skip_directives,
    };
    let (forcefield, frame) = molrs::io::read_top_system(&path, &options)
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    Ok((
        PyForceField { inner: forcefield },
        PyFrame::from_core_frame(frame)?,
    ))
}

/// Read the :class:`ForceField` a LAMMPS data file's ``* Coeffs`` sections
/// define.
///
/// The file is read as :func:`molrs.io.read_lammps_data` reads it; a row's
/// numeric type id is named by the label the file's ``* Type Labels``
/// section gave it, else by the id itself. ``units`` is the unit style the
/// coefficients are in: the one the file's ``write_data`` title line stated
/// when ``None``, else ``"real"``. Styles come from the section headers'
/// ``# style`` comments, default harmonic / ``lj/cut/coul/cut``.
///
/// Raises ``ValueError`` for an unreadable data file, one with no
/// ``* Coeffs`` sections, a ``units`` that disagrees with the file's, an
/// unsupported style, or a row that does not parse.
#[pyfunction]
#[pyo3(name = "read_lammps_data_coeffs", signature = (path, *, units = None))]
pub fn read_lammps_data_coeffs_py(path: PathBuf, units: Option<&str>) -> PyResult<PyForceField> {
    let forcefield = molrs::io::read_lammps_data_coeffs(&path, units)
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    Ok(PyForceField { inner: forcefield })
}

/// :func:`read_lammps_data_coeffs` on the text of a whole LAMMPS data file.
#[pyfunction]
#[pyo3(name = "read_lammps_data_coeffs_str", signature = (text, *, units = None))]
pub fn read_lammps_data_coeffs_str_py(text: &str, units: Option<&str>) -> PyResult<PyForceField> {
    let forcefield = molrs::io::read_lammps_data_coeffs_str(text, units)
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    Ok(PyForceField { inner: forcefield })
}

/// Read a LAMMPS ``fix cmap`` file (CHARMM format) into a :class:`ForceField`
/// of one ``cmap charmm`` style.
///
/// Map ``t`` of the file (1-based: the crossterm type a data file's ``CMAP``
/// section gives) is the row named ``"t"``, with the synthetic endpoints
/// ``t-t-t-t-t``, its ``grid`` the 24×24 map, φ-major, as written. The force
/// field is in the file's ``UNITS:`` tag, else ``real``. An incomplete map, a
/// seventh map, or a line running past a map's end raises ``ValueError``
/// (LAMMPS would drop the values).
#[pyfunction]
#[pyo3(name = "read_lammps_cmap_forcefield")]
pub fn read_lammps_cmap_forcefield_py(path: PathBuf) -> PyResult<PyForceField> {
    let forcefield = molrs::io::read_lammps_cmap_forcefield(&path)
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    Ok(PyForceField { inner: forcefield })
}

/// The force field of a text door's result, a reader error raised as
/// ``ValueError``.
fn forcefield_of(
    read: Result<molrs::ff::forcefield::ForceField, String>,
) -> PyResult<PyForceField> {
    read.map(|inner| PyForceField { inner })
        .map_err(pyo3::exceptions::PyValueError::new_err)
}

/// Read force-field XML text — :func:`read_xml_forcefield` on a string.
#[pyfunction]
#[pyo3(name = "read_xml_forcefield_str")]
pub fn read_xml_forcefield_str_py(text: &str) -> PyResult<PyForceField> {
    forcefield_of(molrs::io::read_xml_forcefield_str(text))
}

/// Read OpenMM force-field XML text — :func:`read_openmm_forcefield` on a
/// string.
#[pyfunction]
#[pyo3(name = "read_openmm_forcefield_str")]
pub fn read_openmm_forcefield_str_py(text: &str) -> PyResult<PyForceField> {
    forcefield_of(molrs::io::read_openmm_forcefield_str(text))
}

/// Read a LAMMPS force-field include's text — :func:`read_lammps_forcefield`
/// on a string.
#[pyfunction]
#[pyo3(name = "read_lammps_forcefield_str")]
pub fn read_lammps_forcefield_str_py(text: &str) -> PyResult<PyForceField> {
    forcefield_of(molrs::io::read_lammps_forcefield_str(text))
}

/// Register the force-field readers.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    crate::add_function(m, "molrs.io", wrap_pyfunction!(read_xml_forcefield_py, m)?)?;
    crate::add_function(
        m,
        "molrs.io",
        wrap_pyfunction!(read_openmm_forcefield_py, m)?,
    )?;
    crate::add_function(
        m,
        "molrs.io",
        wrap_pyfunction!(read_lammps_forcefield_py, m)?,
    )?;
    crate::add_function(
        m,
        "molrs.io",
        wrap_pyfunction!(read_prmtop_forcefield_py, m)?,
    )?;
    crate::add_function(m, "molrs.io", wrap_pyfunction!(read_prmtop_system_py, m)?)?;
    crate::add_function(m, "molrs.io", wrap_pyfunction!(read_top_forcefield_py, m)?)?;
    crate::add_function(m, "molrs.io", wrap_pyfunction!(read_top_system_py, m)?)?;
    crate::add_function(
        m,
        "molrs.io",
        wrap_pyfunction!(read_lammps_data_coeffs_py, m)?,
    )?;
    crate::add_function(
        m,
        "molrs.io",
        wrap_pyfunction!(read_lammps_data_coeffs_str_py, m)?,
    )?;
    crate::add_function(
        m,
        "molrs.io",
        wrap_pyfunction!(read_lammps_cmap_forcefield_py, m)?,
    )?;
    for door in [
        wrap_pyfunction!(read_xml_forcefield_str_py, m)?,
        wrap_pyfunction!(read_openmm_forcefield_str_py, m)?,
        wrap_pyfunction!(read_lammps_forcefield_str_py, m)?,
    ] {
        crate::add_function(m, "molrs.io", door)?;
    }
    Ok(())
}
