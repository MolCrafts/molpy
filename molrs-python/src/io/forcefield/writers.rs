//! Force-field file writers (`molrs::io::writers`): a
//! [`PyForceField`] (and, for a whole-system format, a typed frame) out to
//! each engine's force-field text.

use std::path::PathBuf;

use pyo3::prelude::*;

use crate::core::frame::PyFrame;
use crate::ff::forcefield::PyForceField;
use crate::path::path_str;

/// Write a ForceField as GROMACS force-field directives.
///
/// Writes ``[ defaults ]``, ``[ atomtypes ]``, ``[ nonbond_params ]``,
/// ``[ pairtypes ]``, ``[ bondtypes ]``, ``[ angletypes ]``,
/// ``[ dihedraltypes ]`` and ``[ cmaptypes ]`` in GROMACS units (nm, kJ/mol,
/// degrees) — the inverse of :func:`read_top_forcefield`. No molecule section
/// is written: a force field holds no molecule. A style or parameter GROMACS
/// directives cannot express raises ``ValueError`` naming it. ``precision`` is
/// the number of decimal places for floating coefficients.
#[pyfunction]
#[pyo3(name = "write_top_forcefield", signature = (path, forcefield, precision = 6))]
pub fn write_top_forcefield_py(
    path: PathBuf,
    forcefield: &PyForceField,
    precision: usize,
) -> PyResult<()> {
    molrs::io::write_top_forcefield(&path, &forcefield.inner, precision)
        .map_err(crate::ff::ir::writer_err)
}

/// Write a ForceField as an AMBER frcmod file.
///
/// Writes ``MASS``, ``BOND``, ``ANGLE``, ``DIHE``, ``IMPROPER`` and ``NONBON``
/// in AMBER's conventions (``RK = k``, ``TK = k``, degrees — molrs's own —
/// and ``R*/2`` from sigma), so tleap can ``loadamberparams`` it. A style or parameter a frcmod
/// cannot express raises ``ValueError`` naming it.
#[pyfunction]
#[pyo3(name = "write_frcmod", signature = (path, forcefield))]
pub fn write_frcmod_py(path: PathBuf, forcefield: &PyForceField) -> PyResult<()> {
    molrs::io::write_frcmod(path_str(&path)?, &forcefield.inner).map_err(crate::ff::ir::writer_err)
}

/// Write a ForceField to OpenMM force-field XML.
///
/// Every style is written in OpenMM's own schema and units (nm, kJ/mol,
/// radians, ½k harmonic terms), or refused naming the style when OpenMM's
/// tags cannot hold it (see the force-field IR guide, "OpenMM").
///
/// Geometric
/// mixing (OPLS-AA) adds foyer's ``combining_rule="geometric"`` and the
/// ``<Script>`` that makes ``createSystem`` apply it, which OpenMM's
/// attribute alone does not.
///
/// Parameters
/// ----------
/// path : str or os.PathLike
///     Output file.
/// forcefield : ForceField
/// precision : int, optional
///     Decimals per number; ``None`` (default) writes each number in the
///     shortest form that reads back to the same float.
///
/// Raises
/// ------
/// ValueError
///     A style, parameter or 1-4 setting with no OpenMM form, named.
#[pyfunction]
#[pyo3(name = "write_openmm_forcefield", signature = (path, forcefield, precision = None))]
pub fn write_openmm_forcefield_py(
    path: PathBuf,
    forcefield: &PyForceField,
    precision: Option<usize>,
) -> PyResult<()> {
    molrs::io::write_openmm_forcefield(path_str(&path)?, &forcefield.inner, precision)
        .map_err(crate::ff::ir::writer_err)
}

/// Write a :class:`ForceField` to a LAMMPS force-field include (``*.ff``).
///
/// Coefficient writing, keyed by the system's type labels: ``frame``'s
/// ``atoms`` / ``bonds`` / ``angles`` / ``dihedrals`` / ``impropers`` type
/// labels are walked in id order and each is looked up in ``forcefield``
/// (every label matched to a type name exactly).
/// Force-field types no label uses are not written.
///
/// Inverse of :func:`read_lammps_forcefield`, and the identity on coefficients:
/// the force-field IR follows the LAMMPS standard. A force field declared in another LAMMPS
/// unit style than ``units`` has its energies and lengths converted through
/// the lj reduced hub — never hard-coded eV/kcal factors. A split ``lj/cut`` +
/// ``coul/cut`` pair is recombined as ``lj/cut/coul/cut`` so geometric mixing
/// is not defeated by a hybrid wildcard. Every style is written through its
/// LAMMPS form in the IR registry (``molrs.ff.style_registry``, ``StyleSpec.lammps``):
/// the built-ins LAMMPS has (``dihedral periodic`` as ``fourier``,
/// ``improper periodic`` as ``cvff``, the ``class2`` styles with their
/// cross-term lines at zero, …) and a style registered with
/// ``lammps="positional"``, each parameter converted by its dimension.
///
/// A Drude spring (a ``drude harmonic`` type, ``k*r^2``, used by the frame's
/// ``drudes`` rows) is a LAMMPS harmonic bond: it is written as a
/// ``bond_coeff`` of its label with ``K = k`` and ``r0 = 0``, under the bond
/// type id :func:`write_lammps_data` gives the ``drudes`` rows. Its ``alpha``
/// and ``thole`` are listed in a comment for the input script's
/// ``pair_style thole`` and ``fix drude``, which are not written.
///
/// Parameters
/// ----------
/// path : str
///     Destination path for the include.
/// forcefield : ForceField
///     Force field, in the force-field IR (adopts the LAMMPS standard).
/// frame : Frame
///     The system whose type labels select the coefficients.
/// precision : int, optional
///     Decimal places for floating coefficients (default 6).
/// skip_pair_style : bool, optional
///     When true, omit the ``pair_style`` line: the input script sets its own
///     before the include. Only that line is skipped: ``special_bonds`` and
///     ``pair_modify mix`` / ``shift`` are the force field's and stay
///     (LAMMPS's defaults, ``0 0 0`` and ``geometric`` for ``lj/cut``, are not
///     molrs's); ``pair_modify`` needs a pair style, so read such an include
///     after the input's ``pair_style``.
/// skip_special_bonds : bool, optional
///     When true, omit ``special_bonds``: the input script states its own 1-4
///     weights, which an include read after them would override.
/// skip_units : bool, optional
///     When true, omit the ``units`` line so the include can follow ``units``
///     already set in the input script.
/// units : str, optional
///     LAMMPS ``units`` style for the written file: ``"real"`` (default),
///     ``"metal"``, or ``"lj"``.
/// cmap_file : str, optional
///     The ``fix cmap`` file the include names on its ``fix cmap all cmap
///     <file>`` line (where :func:`write_lammps_cmap_forcefield` saved it). Required
///     exactly when ``frame`` has a ``cmaps`` block; that fix must reach
///     LAMMPS before ``read_data <data> fix cmap crossterm CMAP``.
///
/// Raises
/// ------
/// ValueError
///     On a frame type label the force field does not define (the message
///     names the block and the label), a style without a LAMMPS form holding
///     a used type ("LAMMPS has no form for …"), a bad units keyword, or
///     missing required parameters.
#[pyfunction]
#[pyo3(
    name = "write_lammps_forcefield",
    signature = (
        path,
        forcefield,
        frame,
        *,
        precision = 6,
        skip_pair_style = false,
        skip_special_bonds = false,
        skip_units = false,
        units = "real",
        cmap_file = None,
    )
)]
#[allow(clippy::too_many_arguments)]
pub fn write_lammps_forcefield_py(
    path: PathBuf,
    forcefield: &PyForceField,
    frame: &PyFrame,
    precision: usize,
    skip_pair_style: bool,
    skip_special_bonds: bool,
    skip_units: bool,
    units: &str,
    cmap_file: Option<String>,
) -> PyResult<()> {
    let units = molrs::io::lammps::parse_lammps_units_style(units)
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    let options = molrs::io::lammps::LammpsForcefieldWriteOptions {
        precision,
        skip_pair_style,
        skip_special_bonds,
        skip_units,
        units,
        cmap_file,
    };
    frame
        .with_frame(|frame| {
            molrs::io::write_lammps_forcefield(&path, &forcefield.inner, frame, options)
        })?
        .map_err(crate::ff::ir::writer_err)
}

/// Serialize a :class:`ForceField` to a LAMMPS force-field include string
/// (same labels, format and unit conversion as :func:`write_lammps_forcefield`).
#[pyfunction]
#[pyo3(
    name = "write_lammps_forcefield_str",
    signature = (
        forcefield,
        frame,
        *,
        precision = 6,
        skip_pair_style = false,
        skip_special_bonds = false,
        skip_units = false,
        units = "real",
        cmap_file = None,
    )
)]
#[allow(clippy::too_many_arguments)]
pub fn write_lammps_forcefield_str_py(
    forcefield: &PyForceField,
    frame: &PyFrame,
    precision: usize,
    skip_pair_style: bool,
    skip_special_bonds: bool,
    skip_units: bool,
    units: &str,
    cmap_file: Option<String>,
) -> PyResult<String> {
    let units = molrs::io::lammps::parse_lammps_units_style(units)
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    let options = molrs::io::lammps::LammpsForcefieldWriteOptions {
        precision,
        skip_pair_style,
        skip_special_bonds,
        skip_units,
        units,
        cmap_file,
    };
    frame
        .with_frame(|frame| {
            molrs::io::write_lammps_forcefield_str(&forcefield.inner, frame, options)
        })?
        .map_err(crate::ff::ir::writer_err)
}

/// Serialize a :class:`ForceField` to LAMMPS data-file ``* Coeffs`` sections.
///
/// Same labels, form map and ``units`` conversion as
/// :func:`write_lammps_forcefield`, but emits ``Pair Coeffs`` / ``Bond Coeffs``
/// / … blocks whose integer ids are ``frame``'s type-label ids, not
/// input-script ``*_coeff`` lines. ``Pair Coeffs`` holds self pairs only; a
/// used explicit cross pair raises ``ValueError``.
#[pyfunction]
#[pyo3(
    name = "write_lammps_data_coeffs_str",
    signature = (
        forcefield,
        frame,
        *,
        precision = 6,
        units = "real",
    )
)]

pub fn write_lammps_data_coeffs_str_py(
    forcefield: &PyForceField,
    frame: &PyFrame,
    precision: usize,
    units: &str,
) -> PyResult<String> {
    let options = lammps_options(precision, units)?;
    frame
        .with_frame(|frame| {
            molrs::io::write_lammps_data_coeffs_str(&forcefield.inner, frame, options)
        })?
        .map_err(crate::ff::ir::writer_err)
}

/// Write the LAMMPS ``fix cmap`` file of ``frame``'s CMAP crossterms.
///
/// The ``cmaps`` block's type labels are walked in id order — the crossterm
/// types :func:`molrs.io.write_lammps_data` gives the ``CMAP`` section — and
/// each label's ``cmap charmm`` grid is written in CHARMM's layout (a
/// ``# UNITS:`` line, ``# <φ>`` rows of five ``precision``-decimal values),
/// energies converted to ``units``. A CHARMM file read with
/// :func:`read_lammps_cmap_forcefield` is written back line for line.
///
/// Raises
/// ------
/// ValueError
///     No ``cmaps`` label, a label without a row, a grid that is not 24×24,
///     more than six maps, or a style other than ``charmm``.
#[pyfunction]
#[pyo3(
    name = "write_lammps_cmap_forcefield",
    signature = (path, forcefield, frame, *, precision = 6, units = "real")
)]

pub fn write_lammps_cmap_forcefield_py(
    path: PathBuf,
    forcefield: &PyForceField,
    frame: &PyFrame,
    precision: usize,
    units: &str,
) -> PyResult<()> {
    let options = lammps_options(precision, units)?;
    frame
        .with_frame(|frame| {
            molrs::io::write_lammps_cmap_forcefield(&path, &forcefield.inner, frame, options)
        })?
        .map_err(crate::ff::ir::writer_err)
}

/// Write a force field and a typed frame as one GROMACS topology — the
/// inverse of :func:`read_top_system`.
///
/// The force-field directives as :func:`write_top_forcefield` writes them,
/// then one ``[ moleculetype ]`` per molecule of ``frame`` (its ``atoms``
/// ``type`` / ``charge`` / ``mass`` and the relation blocks, typed by the
/// force field's labels), ``[ system ]`` and ``[ molecules ]``. Coordinates
/// go to a ``.gro`` (:func:`molrs.io.write_gro`).
///
/// Parameters
/// ----------
/// path : str | os.PathLike
///     The ``.top`` file to write.
/// forcefield : ForceField
///     Holds a type for every label ``frame`` uses.
/// frame : Frame
///     Typed system (string ``type`` columns).
/// precision : int
///     Decimal places for floating coefficients.
///
/// Raises
/// ------
/// ValueError
///     A style, parameter or row GROMACS cannot express, or a label the force
///     field lacks.
/// OSError
///     The file cannot be written.
#[pyfunction]
#[pyo3(name = "write_top_system", signature = (path, forcefield, frame, *, precision = 6))]
pub fn write_top_system_py(
    path: PathBuf,
    forcefield: PyRef<'_, PyForceField>,
    frame: &PyFrame,
    precision: usize,
) -> PyResult<()> {
    frame
        .with_frame(|f| molrs::io::write_top_system(&path, &forcefield.inner, f, precision))?
        .map_err(crate::ff::ir::writer_err)
}

/// Write a force field as a force-field XML file — the inverse of
/// :func:`read_xml_forcefield`: every style one element, every value as the
/// force field holds it, under ``<ForceField units=…>`` (the declared units,
/// or the ``real`` default). Reading it back gives the same force field with
/// its units declared.
///
/// Parameters
/// ----------
/// path : str or os.PathLike
///     File to write.
/// forcefield : ForceField
///     The force field.
///
/// Raises
/// ------
/// ValueError
///     What the layout cannot hold, by name: a parameter called ``name`` (or
///     ``class1`` … on a type row); a string parameter whose value reads back
///     as a number; a style of an undeclared category with no type rows.
#[pyfunction]
#[pyo3(name = "write_xml_forcefield", signature = (path, forcefield))]
pub fn write_xml_forcefield_py(path: PathBuf, forcefield: &PyForceField) -> PyResult<()> {
    molrs::io::write_xml_forcefield(path_str(&path)?, &forcefield.inner)
        .map_err(pyo3::exceptions::PyValueError::new_err)
}

/// Write a ForceField as AMBER frcmod text — :func:`write_frcmod` into
/// a string.
#[pyfunction]
#[pyo3(name = "write_frcmod_str")]
pub fn write_frcmod_str_py(forcefield: &PyForceField) -> PyResult<String> {
    molrs::io::write_frcmod_str(&forcefield.inner).map_err(crate::ff::ir::writer_err)
}

/// Write a ForceField as OpenMM force-field XML text —
/// :func:`write_openmm_forcefield` into a string, ``precision`` as there.
#[pyfunction]
#[pyo3(name = "write_openmm_forcefield_str", signature = (forcefield, precision = None))]
pub fn write_openmm_forcefield_str_py(
    forcefield: &PyForceField,
    precision: Option<usize>,
) -> PyResult<String> {
    molrs::io::write_openmm_forcefield_str(&forcefield.inner, precision)
        .map_err(crate::ff::ir::writer_err)
}

/// Write a ForceField and the residue templates of a typed Frame as OpenMM
/// force-field XML: what ``openmm.app.ForceField(path).createSystem(
/// PDBFile(pdb).topology)`` builds a System from, for the PDB
/// :func:`write_pdb` writes of the same frame.
///
/// The force field is written as :func:`write_openmm_forcefield` writes it,
/// plus ``<Residues>``: one ``<Residue>`` per distinct residue of the frame
/// (same residue name, atom names, types, charges and bonds), with an
/// ``<Atom name type charge>`` per atom, a ``<Bond>`` per bond inside it and
/// an ``<ExternalBond>`` per bond leaving it.
///
/// - Residues are the ones OpenMM reads from that PDB: runs of consecutive
///   atoms with the same ``chain``, ``res_id``, ``icode`` and ``res_name``.
///   With no ``res_id`` column, each bonded molecule (a connected component
///   of ``bonds``) is one residue, named ``MOL``.
/// - Atom names are the PDB's: the frame's ``name``, else the element
///   numbered within the residue (``C1``, ``C2``, ``H1``, …).
/// - Charges are the frame's ``charge`` column, on the templates
///   (``<UseAttributeFromResidue name="charge"/>``); without that column,
///   the atom types' own charges.
/// - Every atom type the frame uses gets an ``element`` (its own, else its
///   atoms'), which OpenMM matches residues by.
///
/// Parameters
/// ----------
/// path : str or os.PathLike
///     Output file.
/// forcefield : ForceField
///     Holds an atom type for every ``type`` the frame's atoms name.
/// frame : Frame
///     Typed system: string ``type`` and ``element`` atom columns, ``bonds``.
/// precision : int, optional
///     Decimals per number; ``None`` (default) writes each number in the
///     shortest form that reads back to the same float.
///
/// Raises
/// ------
/// ValueError
///     What :func:`write_openmm_forcefield` refuses, and: a missing
///     ``type`` or ``element`` column, a type the force field lacks, one
///     type on two elements, two atoms of one residue with the same name, no
///     charges in the frame or on the types.
/// OSError
///     The file cannot be written.
#[pyfunction]
#[pyo3(
    name = "write_openmm_system",
    signature = (path, forcefield, frame, *, precision = None)
)]
pub fn write_openmm_system_py(
    path: PathBuf,
    forcefield: PyRef<'_, PyForceField>,
    frame: &PyFrame,
    precision: Option<usize>,
) -> PyResult<()> {
    let path = path_str(&path)?;
    frame
        .with_frame(|f| molrs::io::write_openmm_system(path, &forcefield.inner, f, precision))?
        .map_err(crate::ff::ir::writer_err)
}

/// :func:`write_openmm_system` into a string, ``precision`` as there.
#[pyfunction]
#[pyo3(
    name = "write_openmm_system_str",
    signature = (forcefield, frame, *, precision = None)
)]
pub fn write_openmm_system_str_py(
    forcefield: PyRef<'_, PyForceField>,
    frame: &PyFrame,
    precision: Option<usize>,
) -> PyResult<String> {
    frame
        .with_frame(|f| molrs::io::write_openmm_system_str(&forcefield.inner, f, precision))?
        .map_err(crate::ff::ir::writer_err)
}

/// Write a ForceField as force-field XML text — :func:`write_xml_forcefield`
/// into a string.
#[pyfunction]
#[pyo3(name = "write_xml_forcefield_str")]
pub fn write_xml_forcefield_str_py(forcefield: &PyForceField) -> PyResult<String> {
    molrs::io::write_xml_forcefield_str(&forcefield.inner)
        .map_err(pyo3::exceptions::PyValueError::new_err)
}

/// The LAMMPS write options of `precision` decimals in `units`.
fn lammps_options(
    precision: usize,
    units: &str,
) -> PyResult<molrs::io::lammps::LammpsForcefieldWriteOptions> {
    let units = molrs::io::lammps::parse_lammps_units_style(units)
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    Ok(molrs::io::lammps::LammpsForcefieldWriteOptions {
        precision,
        units,
        ..Default::default()
    })
}

/// Register the force-field writers.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    crate::add_function(m, "molrs.io", wrap_pyfunction!(write_top_forcefield_py, m)?)?;
    crate::add_function(m, "molrs.io", wrap_pyfunction!(write_top_system_py, m)?)?;
    crate::add_function(m, "molrs.io", wrap_pyfunction!(write_frcmod_py, m)?)?;
    crate::add_function(
        m,
        "molrs.io",
        wrap_pyfunction!(write_openmm_forcefield_py, m)?,
    )?;
    crate::add_function(m, "molrs.io", wrap_pyfunction!(write_xml_forcefield_py, m)?)?;
    crate::add_function(m, "molrs.io", wrap_pyfunction!(write_openmm_system_py, m)?)?;
    crate::add_function(
        m,
        "molrs.io",
        wrap_pyfunction!(write_openmm_system_str_py, m)?,
    )?;
    crate::add_function(
        m,
        "molrs.io",
        wrap_pyfunction!(write_lammps_forcefield_py, m)?,
    )?;
    crate::add_function(
        m,
        "molrs.io",
        wrap_pyfunction!(write_lammps_forcefield_str_py, m)?,
    )?;
    crate::add_function(
        m,
        "molrs.io",
        wrap_pyfunction!(write_lammps_data_coeffs_str_py, m)?,
    )?;
    crate::add_function(
        m,
        "molrs.io",
        wrap_pyfunction!(write_lammps_cmap_forcefield_py, m)?,
    )?;
    for door in [
        wrap_pyfunction!(write_frcmod_str_py, m)?,
        wrap_pyfunction!(write_openmm_forcefield_str_py, m)?,
        wrap_pyfunction!(write_xml_forcefield_str_py, m)?,
    ] {
        crate::add_function(m, "molrs.io", door)?;
    }
    Ok(())
}
