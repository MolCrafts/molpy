//! Python binding for the GAFF / GAFF2 typifier (`molrs::ff::typifier::gaff`).
//!
//! `GaffTypifier` matches the bonded terms of a molecule whose atoms already carry
//! GAFF atom types — what `AtdTypifier(parameter_set="gaff")` / `"gaff2"` stamps —
//! against the compiled `gaff.dat` / `gaff2.dat` table, estimating what the table
//! does not cover the way `parmchk2` does. Typing atoms is the ATD typifier's job;
//! the two compose, as in Rust:
//!
//! ```text
//! labelled = AtdTypifier(parameter_set="gaff2").typify(mol)
//! typed = GaffTypifier(parameter_set="gaff2").typify(labelled)
//! ```
//!
//! The parameter set is named, never defaulted, and spelled like the ATD table
//! that types for it (antechamber's `-at` flag): typing with one table and
//! parameterising with the other is the category error the shared spelling makes
//! visible.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use molrs::ff::typifier::{GaffParameterSet, GaffTypifier};

use super::PyTypifier;

/// GAFF / GAFF2 bonded-term typifier — ``molrs.ff.typifier.GaffTypifier``.
///
/// Matches a molecule whose atoms already carry GAFF atom types (``keys.TYPE``,
/// as :class:`AtdTypifier` with the same ``parameter_set`` stamps them) against
/// the AMBER ``gaff.dat`` / ``gaff2.dat`` table compiled into molrs. Angles and
/// dihedrals are regenerated from the bond graph; impropers are built as
/// tleap builds them, wherever tleap finds a row (or ``parmchk2`` an estimate,
/// at the centres ``PARMCHK.DAT`` flags as planar) for a triple of an atom's
/// neighbours, in the atom order tleap gives them (centre third). Every term
/// is looked up first against the table's exact rows, then against its
/// wildcard rows, then estimated as ``parmchk2`` estimates it; an estimated
/// term's type carries the provenance
/// params ``estimated``, ``estimate_penalty``, ``estimate_method`` and
/// ``estimate_analog``. They price nothing: every engine writer passes them
/// over (the frcmod writer as the row's comment) and ``ForceField.merge``
/// ignores them.
///
/// The output (:meth:`forcefield`) holds the styles ``atom full``,
/// ``pair lj/cut``, ``pair coul/cut`` (AMBER's Coulomb constant),
/// ``bond harmonic``, ``angle harmonic``, ``dihedral periodic`` and
/// ``improper periodic`` with AMBER's 1-4 weights (1/2 Lennard-Jones,
/// 1/1.2 Coulomb), in the force-field IR's units. Charges are not a GAFF
/// parameter: set ``atoms.charge`` with a charge model (:class:`BccModel`,
/// :class:`GasteigerModel`) before pricing Coulomb.
///
/// Parameters
/// ----------
/// parameter_set : str
///     ``"gaff"`` (GAFF 1.81, ``gaff.dat``) or ``"gaff2"`` (``gaff2.dat``).
///     Required; it must name the table the atoms were typed with.
///
/// Raises
/// ------
/// ValueError
///     If ``parameter_set`` names no set; from :meth:`typify`, when an atom has
///     no type, a type the table does not declare, or a term neither the table
///     nor the estimator covers (every such term is listed).
///
/// Examples
/// --------
/// >>> labelled = molrs.ff.typifier.AtdTypifier(parameter_set="gaff2").typify(mol)
/// >>> gaff = molrs.ff.typifier.GaffTypifier(parameter_set="gaff2")
/// >>> frame = gaff.typify(labelled).to_frame()
/// >>> frame["pairs"] = molrs.ff.potential.intramolecular_pairs(frame)
/// >>> pots = molrs.ff.compile.PotentialCompiler(gaff.forcefield()).compile(frame)
#[pyclass(module = "molrs.ff.typifier", name = "GaffTypifier", extends = PyTypifier, subclass)]
#[derive(Debug)]
pub struct PyGaffTypifier {
    parameter_set: GaffParameterSet,
}

#[pymethods]
impl PyGaffTypifier {
    /// Bind the typifier to the table `parameter_set` names.
    #[new]
    #[pyo3(signature = (*, parameter_set))]
    fn new(parameter_set: &str) -> PyResult<(Self, PyTypifier)> {
        let parameter_set =
            GaffParameterSet::from_name(parameter_set).map_err(PyValueError::new_err)?;
        Ok((
            Self { parameter_set },
            PyTypifier::native(GaffTypifier::new(parameter_set)),
        ))
    }

    /// The name of the table this typifier matches against: ``"gaff"`` or
    /// ``"gaff2"``.
    #[getter]
    fn parameter_set(&self) -> &'static str {
        self.parameter_set.name()
    }

    fn __repr__(&self) -> String {
        format!("GaffTypifier(parameter_set='{}')", self.parameter_set())
    }
}
