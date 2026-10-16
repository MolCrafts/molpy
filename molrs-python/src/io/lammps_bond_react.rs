//! LAMMPS `fix bond/react` file sets: `molrs.io.lammps.BondReactTemplate`,
//! `molrs.io.write_lammps_bond_react_map` and
//! `write_lammps_bond_react_system`, over `molrs::io::lammps`.

use std::path::PathBuf;

use molrs::core::Frame;
use molrs::core::keys::REACT_ID;
use molrs::io::lammps::BondReactTemplate;
use molrs::io::writer::ForceFieldWriteError;
use molrs::io::{
    write_lammps_bond_react_map as write_map_rs, write_lammps_bond_react_system as write_system_rs,
};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use super::{PySink, data_error};
use crate::core::frame::PyFrame;
use crate::error::io_error_to_pyerr;
use crate::ff::forcefield::PyForceField;

/// One ``fix bond/react`` reaction: the pre-reaction template, the same
/// atoms after the reaction, and the atoms its map file names.
///
/// ``pre`` and ``post`` are :class:`~molrs.core.Atomistic` graphs (or
/// :class:`~molrs.core.Frame` s) whose atoms carry an integer ``react_id`` pairing
/// them. ``initiator_atoms`` (exactly two), ``edge_atoms`` and
/// ``deleted_atoms`` are atoms of ``pre`` — or their ``react_id`` values.
/// The objects are kept as given (nothing is copied or renumbered); the
/// writers read them when they write.
///
/// Serialized by :func:`molrs.io.write_lammps_bond_react_map` (``{name}.map``) and
/// :func:`write_lammps_bond_react_system` (also ``{name}_pre.mol`` /
/// ``{name}_post.mol``). https://docs.lammps.org/fix_bond_react.html
#[pyclass(module = "molrs.io.lammps", name = "BondReactTemplate")]
pub struct PyBondReactTemplate {
    /// Pre-reaction template.
    #[pyo3(get, set)]
    pre: Py<PyAny>,
    /// Post-reaction template.
    #[pyo3(get, set)]
    post: Py<PyAny>,
    /// The two atoms that initiate the reaction.
    #[pyo3(get, set)]
    initiator_atoms: Py<PyAny>,
    /// Atoms bonded to atoms outside the template.
    #[pyo3(get, set)]
    edge_atoms: Py<PyAny>,
    /// Atoms the reaction deletes.
    #[pyo3(get, set)]
    deleted_atoms: Py<PyAny>,
}

#[pymethods]
impl PyBondReactTemplate {
    #[new]
    #[pyo3(signature = (pre, post, initiator_atoms, edge_atoms = None, deleted_atoms = None))]
    fn new(
        py: Python<'_>,
        pre: Py<PyAny>,
        post: Py<PyAny>,
        initiator_atoms: Py<PyAny>,
        edge_atoms: Option<Py<PyAny>>,
        deleted_atoms: Option<Py<PyAny>>,
    ) -> PyResult<Self> {
        let empty = || -> PyResult<Py<PyAny>> { Ok(PyList::empty(py).into_any().unbind()) };
        Ok(Self {
            pre,
            post,
            initiator_atoms,
            edge_atoms: match edge_atoms {
                Some(v) => v,
                None => empty()?,
            },
            deleted_atoms: match deleted_atoms {
                Some(v) => v,
                None => empty()?,
            },
        })
    }

    fn __repr__(&self, py: Python<'_>) -> String {
        let n = |obj: &Py<PyAny>| obj.bind(py).len().unwrap_or(0);
        format!(
            "BondReactTemplate(initiators={}, edges={}, deleted={})",
            n(&self.initiator_atoms),
            n(&self.edge_atoms),
            n(&self.deleted_atoms)
        )
    }
}

/// The frame behind a template object: a `Frame` itself, or whatever its
/// `to_frame()` returns (an `Atomistic`).
fn frame_of(obj: &Bound<'_, PyAny>, which: &str) -> PyResult<Frame> {
    if let Ok(frame) = obj.extract::<PyRef<'_, PyFrame>>() {
        return frame.clone_core_frame();
    }
    let converted = obj.call_method0("to_frame").map_err(|_| {
        PyTypeError::new_err(format!(
            "the {which} template must be a Frame or have to_frame() (an Atomistic)"
        ))
    })?;
    converted
        .extract::<PyRef<'_, PyFrame>>()?
        .clone_core_frame()
}

/// The `react_id` an entry names: an int, or an atom's `react_id`.
fn react_id_of(obj: &Bound<'_, PyAny>, what: &str) -> PyResult<i64> {
    if !obj.is_instance_of::<pyo3::types::PyBool>()
        && let Ok(id) = obj.extract::<i64>()
    {
        return Ok(id);
    }
    obj.get_item(REACT_ID)
        .and_then(|v| v.extract::<i64>())
        .map_err(|_| {
            PyValueError::new_err(format!(
                "every {what} must be an atom with an integer '{REACT_ID}' (or that id)"
            ))
        })
}

fn react_ids(obj: &Bound<'_, PyAny>, what: &str) -> PyResult<Vec<i64>> {
    obj.try_iter()?
        .map(|item| react_id_of(&item?, what))
        .collect()
}

impl PyBondReactTemplate {
    fn to_rust(&self, py: Python<'_>) -> PyResult<BondReactTemplate> {
        let initiators = react_ids(self.initiator_atoms.bind(py), "initiator atom")?;
        let initiators: [i64; 2] = initiators.as_slice().try_into().map_err(|_| {
            PyValueError::new_err(format!(
                "fix bond/react requires exactly 2 initiator atoms, got {}.",
                initiators.len()
            ))
        })?;
        Ok(BondReactTemplate {
            pre: frame_of(self.pre.bind(py), "pre")?,
            post: frame_of(self.post.bind(py), "post")?,
            initiators,
            edges: react_ids(self.edge_atoms.bind(py), "edge atom")?,
            deleted: react_ids(self.deleted_atoms.bind(py), "deleted atom")?,
        })
    }
}

/// Write the ``fix bond/react`` map file of ``template``.
///
/// Parameters
/// ----------
/// sink : os.PathLike | bytearray
///     The file (conventionally ``{name}.map``), or a buffer that then holds
///     exactly its bytes.
/// template : BondReactTemplate
///     The reaction.
///
/// Raises
/// ------
/// ValueError
///     The templates lack an integer ``react_id`` or repeat one, ``pre`` and
///     ``post`` hold different ``react_id`` s, there are not exactly two
///     initiators, or an initiator is not in ``pre``.
/// OSError
///     The file cannot be written.
#[pyfunction]
pub fn write_lammps_bond_react_map(
    py: Python<'_>,
    sink: PySink<'_>,
    template: PyRef<'_, PyBondReactTemplate>,
) -> PyResult<()> {
    let rust = template.to_rust(py)?;
    sink.put(|s| write_map_rs(s, &rust), data_error)
}

/// Write the whole file set of a ``fix bond/react`` run into ``workdir``.
///
/// ``{stem}.data`` (the system), ``{stem}.ff`` (``forcefield``'s styles and
/// coefficients, without a ``units`` line: the input sets ``units`` and
/// ``atom_style``, reads the data file, then includes it), and per template
/// ``{name}_pre.mol``, ``{name}_post.mol`` and ``{name}.map``; ``stem`` is
/// ``workdir``'s own name. Every check runs before the first file is
/// written. Every type label
/// the system and the templates use is declared in the data file and covered
/// by the ``.ff`` include, so the templates' type ids match the system's.
/// Template topology rows without a type label are left out of the molecule
/// files, with a :class:`UserWarning` per template and block.
///
/// Parameters
/// ----------
/// workdir : str | os.PathLike
///     Output directory (created if missing).
/// frame : Frame
///     The system.
/// forcefield : ForceField
///     Holds a type for every label used.
/// templates : dict[str, BondReactTemplate] | Sequence[BondReactTemplate]
///     Named templates; a sequence is named ``rxn1``, ``rxn2``, ….
///
/// Raises
/// ------
/// ValueError
///     A malformed template, an untyped template atom, a label the force
///     field lacks, or system labels that cannot share the numbering.
#[pyfunction]
pub fn write_lammps_bond_react_system(
    py: Python<'_>,
    workdir: PathBuf,
    frame: &PyFrame,
    forcefield: PyRef<'_, PyForceField>,
    templates: &Bound<'_, PyAny>,
) -> PyResult<()> {
    let named: Vec<(String, Py<PyBondReactTemplate>)> = if let Ok(dict) = templates.cast::<PyDict>()
    {
        dict.iter()
            .map(|(k, v)| Ok((k.extract::<String>()?, v.extract()?)))
            .collect::<PyResult<_>>()?
    } else {
        templates
            .try_iter()?
            .enumerate()
            .map(|(i, v)| Ok((format!("rxn{}", i + 1), v?.extract()?)))
            .collect::<PyResult<_>>()?
    };
    let rust: Vec<(String, BondReactTemplate)> = named
        .iter()
        .map(|(name, t)| Ok((name.clone(), t.borrow(py).to_rust(py)?)))
        .collect::<PyResult<_>>()?;
    let system = frame.clone_core_frame()?;
    let written = write_system_rs(&workdir, &system, &forcefield.inner, &rust).map_err(|e| {
        if let Some(refusal) = e
            .get_ref()
            .and_then(|inner| inner.downcast_ref::<ForceFieldWriteError>())
        {
            crate::ff::ir::writer_err(refusal.clone())
        } else if e.kind() == std::io::ErrorKind::InvalidData
            || e.kind() == std::io::ErrorKind::InvalidInput
        {
            PyValueError::new_err(e.to_string())
        } else {
            io_error_to_pyerr(e)
        }
    })?;
    let warnings = py.import("warnings")?;
    for d in &written.dropped {
        warnings.call_method1(
            "warn",
            (format!(
                "Dropped {} {} entries with unrecognized types from template '{}'. Ensure all \
                 template topology is typed (pass a typifier to the assembler).",
                d.rows, d.block, d.template
            ),),
        )?;
    }
    Ok(())
}

/// Register the LAMMPS ``fix bond/react`` template and writers.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyBondReactTemplate>()?;
    crate::add_function(
        m,
        "molrs.io",
        wrap_pyfunction!(write_lammps_bond_react_map, m)?,
    )?;
    crate::add_function(
        m,
        "molrs.io",
        wrap_pyfunction!(write_lammps_bond_react_system, m)?,
    )?;
    Ok(())
}
