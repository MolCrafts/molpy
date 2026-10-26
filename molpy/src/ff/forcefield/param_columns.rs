//! `ForceField.materialize_params`: a force field's parameters of a typed frame,
//! written onto the frame as per-row and per-atom columns
//! (`molrs::ff::forcefield::param_columns`).

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use super::PyForceField;
use crate::core::frame::PyFrame;
use crate::error::py_value_err;

#[pymethods]
impl PyForceField {
    /// Write the parameters this force field gives each row of ``frame`` as
    /// columns ``<prefix><parameter>``, in place.
    ///
    /// Each row of ``bonds``, ``angles``, ``dihedrals``, ``impropers`` and
    /// ``cmaps`` gets the numeric parameters of the type its ``type`` column
    /// names (``bonds.<prefix>k``, ``bonds.<prefix>r0``,
    /// ``dihedrals.<prefix>k1``, ``<prefix>periodicity1``, ``<prefix>phase1``,
    /// …); each atom gets those of its ``atoms.type`` under every ``atom``
    /// style (``<prefix>mass``) and those of its type's self row under every
    /// ``pair`` style with per-type rows (``<prefix>epsilon``,
    /// ``<prefix>sigma`` under ``lj/cut``). The columns are the parameters the
    /// field stores, whatever the style — nothing here is per style — in the
    /// force-field IR's units (LAMMPS standard: degrees, ``K`` without a ½,
    /// the field's ``units`` preset).
    ///
    /// A row whose type lacks a parameter another row's type has (a two-term
    /// torsion beside a three-term one; ``estimate_penalty`` on an estimated
    /// term only) is a null cell: ``Block.validity(column)`` marks it.
    /// String and array parameters, a pair style without per-type rows
    /// (``coul/cut``) and cross rows (NBFIX) are not written. A column
    /// already present under a written name is replaced. Charges are frame
    /// data, not a type parameter (unless the field stores them per type):
    /// a charge model writes ``atoms.charge``.
    ///
    /// Parameters
    /// ----------
    /// frame : Frame
    ///     A frame typed by this force field (``atoms.type``, and a ``type``
    ///     column on every relation block with rows), modified in place.
    /// prefix : str
    ///     Prepended to every parameter name, e.g. ``"ref_"``; ``""`` writes
    ///     the bare names.
    ///
    /// Returns
    /// -------
    /// dict[str, list[str]]
    ///     Block name → the columns written to it, in order.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     A relation block with rows but no string ``type`` column; a row or
    ///     atom whose type no style of its category defines, or two styles
    ///     define; an atom whose type has no self row in a pair style that has
    ///     rows; two styles writing one ``atoms`` column.
    ///
    /// Examples
    /// --------
    /// >>> gaff = molpy.ff.typifier.GaffTypifier(parameter_set="gaff2")
    /// >>> frame = gaff.typify(labelled).to_frame()
    /// >>> gaff.forcefield().materialize_params(frame, prefix="gaff2_")
    /// {'bonds': ['gaff2_k', 'gaff2_r0'], 'angles': [...], ..., 'atoms': ['gaff2_epsilon', 'gaff2_mass', 'gaff2_sigma']}
    /// >>> k = frame["bonds"].get("gaff2_k")
    #[pyo3(signature = (frame, *, prefix))]
    fn materialize_params<'py>(
        &self,
        py: Python<'py>,
        frame: &PyFrame,
        prefix: &str,
    ) -> PyResult<Bound<'py, PyDict>> {
        let written = frame
            .with_frame_mut(|core| self.inner.materialize_params(core, prefix))?
            .map_err(py_value_err)?;
        let out = PyDict::new(py);
        for (block, column) in written {
            match out.get_item(&block)? {
                Some(list) => list.cast::<PyList>()?.append(column)?,
                None => out.set_item(block, PyList::new(py, [column])?)?,
            }
        }
        Ok(out)
    }
}
