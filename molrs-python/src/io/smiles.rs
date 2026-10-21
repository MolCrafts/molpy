//! SMILES (`molrs::io::smiles`): `read_smiles(text)` reads SMILES into a
//! [`PySmilesIr`], `write_smiles(mol)` writes a molecule as SMILES text, and
//! `read_smiles_fragment(body)` reads a fragment body with bonding descriptors
//! into a [`PyFragmentIr`]. An IR is only ever handed out, never constructed
//! by the caller.
//! SMARTS — parsing, matching, and a pattern written from a molecule — is
//! `molrs.perceive`'s.

use std::collections::HashMap;

use crate::core::molgraph::PyAtomistic;
use crate::error::smiles_error_to_pyerr;
use crate::io::cgsmiles::PyBondingDescriptor;
use molrs::core::NodeId;
use molrs::io::smiles::{FragmentIr, SmilesIr};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Intermediate representation of a parsed SMILES string: the syntax tree
/// the parser produces, one chain per ``'.'``-separated component.
///
/// :func:`read_smiles` hands it out; it has no constructor of its own.
/// ``str(ir)`` writes it back as text (SMILES, or SMARTS where it holds query
/// atoms or bond operators) and :meth:`to_atomistic` turns it into a
/// molecular graph. A fragment body with bonding descriptors is
/// :func:`read_smiles_fragment`'s, a :class:`FragmentIr`.
///
/// Examples
/// --------
/// >>> ir = molrs.io.smiles.read_smiles("CCO")
/// >>> ir.n_components
/// 1
/// >>> ir.to_atomistic().n_atoms
/// 3
/// >>> str(ir)
/// 'CCO'
#[pyclass(module = "molrs.io.smiles", name = "SmilesIr", frozen)]
pub struct PySmilesIr {
    inner: SmilesIr,
}

/// `ir` as its text, `ValueError` for an IR neither the parser nor
/// `SmilesIr::from_atomistic` produces (no components, or bonding descriptors).
fn smiles_text(ir: &SmilesIr) -> PyResult<String> {
    use std::fmt::Write;
    let mut text = String::new();
    write!(text, "{ir}").map_err(|_| {
        PyValueError::new_err("this SMILES IR has no text form (no components, or descriptors)")
    })?;
    Ok(text)
}

#[pymethods]
impl PySmilesIr {
    /// Number of ``'.'``-separated components.
    #[getter]
    fn n_components(&self) -> usize {
        self.inner.components.len()
    }

    /// The molecular graph the IR states: atoms and bonds, connectivity only
    /// — hydrogens implicit in the text are **not** added
    /// (:func:`molrs.perceive.add_hydrogens` fills open valences) and no
    /// coordinates are made. Every component lands in the one graph;
    /// :meth:`components` gives them apart.
    ///
    /// Returns
    /// -------
    /// Atomistic
    ///
    /// Raises
    /// ------
    /// SmilesError
    ///     A ring-closure digit never closed, or SMARTS query atoms or bonds
    ///     (which have no single atomistic reading).
    fn to_atomistic(&self, py: Python<'_>) -> PyResult<Py<PyAtomistic>> {
        let mol = self.inner.to_atomistic().map_err(smiles_error_to_pyerr)?;
        PyAtomistic::from_core(py, mol)
    }

    /// One graph per component, in input order. Splitting happens on the
    /// parsed components, not by cutting the text on ``'.'``.
    ///
    /// Returns
    /// -------
    /// list[Atomistic]
    fn components(&self, py: Python<'_>) -> PyResult<Vec<Py<PyAtomistic>>> {
        self.inner
            .components
            .iter()
            .map(|chain| {
                let one = SmilesIr {
                    components: vec![chain.clone()],
                    span: self.inner.span,
                };
                let mol = one.to_atomistic().map_err(smiles_error_to_pyerr)?;
                PyAtomistic::from_core(py, mol)
            })
            .collect()
    }

    /// The IR as text — ``molrs::io::smiles::SmilesIr``'s ``Display``.
    fn __str__(&self) -> PyResult<String> {
        smiles_text(&self.inner)
    }

    fn __repr__(&self) -> String {
        match smiles_text(&self.inner) {
            Ok(text) => format!("SmilesIr({text:?})"),
            Err(_) => format!("SmilesIr(<{} components>)", self.inner.components.len()),
        }
    }
}

/// Read SMILES text into its IR.
///
/// Parameters
/// ----------
/// smiles : str
///     SMILES text (``"CCO"`` for ethanol, ``"c1ccccc1"``).
///
/// Returns
/// -------
/// SmilesIr
///
/// Raises
/// ------
/// SmilesError
///     (a ``ValueError``) for text that is not SMILES: a syntax error, an
///     unknown element, a bonding descriptor (a fragment body goes through
///     :func:`read_smiles_fragment`).
///
/// Examples
/// --------
/// >>> molrs.io.smiles.read_smiles("CCO").to_atomistic().n_atoms
/// 3
#[pyfunction]
fn read_smiles(smiles: &str) -> PyResult<PySmilesIr> {
    let inner = SmilesIr::parse(smiles).map_err(smiles_error_to_pyerr)?;
    Ok(PySmilesIr { inner })
}

/// Read a fragment body — SMILES plus ``CGsmiles`` / ``BigSMILES`` bonding
/// descriptors (``[<]OCC[>]``, ``[$]COC[$]``) — into its IR.
///
/// Parameters
/// ----------
/// body : str
///     The fragment body.
///
/// Returns
/// -------
/// FragmentIr
///
/// Raises
/// ------
/// SmilesError
///     (a ``ValueError``) for text that is not fragment notation.
///
/// Examples
/// --------
/// >>> molrs.io.smiles.read_smiles_fragment("[<]OCC[>]").to_template().n_ports
/// 2
#[pyfunction]
fn read_smiles_fragment(body: &str) -> PyResult<PyFragmentIr> {
    let inner = SmilesIr::from_fragment(body).map_err(smiles_error_to_pyerr)?;
    Ok(PyFragmentIr { inner })
}

/// Write a molecule as SMILES text.
///
/// Every science or representation choice is an explicit keyword, the
/// fields of ``molrs::io::smiles::SmilesEmitOptions``.
///
/// Parameters
/// ----------
/// mol : Atomistic
///     The molecule.
/// canonical : bool
///     Use the canonical atom order for root selection and branch order.
/// root : Atom, optional
///     The atom the string starts at (overrides the canonical root).
/// aromatic : {"as_marked", "kekule_only"}
///     Honour aromatic marks, or require integer bond numbers.
/// hydrogens : {"organic_subset", "explicit_all", "as_stored"}
///     How hydrogens appear.
/// include_stereo : bool
///     Write tetrahedral / double-bond stereo marks the graph holds.
/// multi_component : {"error_if_multiple", "join_dot", "first_only"}
///     What a molecule of several connected components becomes.
/// organic_subset : bool
///     Prefer the Daylight organic-subset letters when legal.
///
/// Returns
/// -------
/// str
///
/// Raises
/// ------
/// ValueError
///     An unknown option spelling, or ``root`` is not an atom.
/// SmilesError
///     If the molecule cannot be written under these options.
///
/// Examples
/// --------
/// >>> molrs.io.smiles.write_smiles(molrs.io.smiles.read_smiles("OCC").to_atomistic())
/// 'CCO'
#[pyfunction]
#[pyo3(signature = (
    mol,
    *,
    canonical = true,
    root = None,
    aromatic = "as_marked",
    hydrogens = "organic_subset",
    include_stereo = false,
    multi_component = "error_if_multiple",
    organic_subset = true,
))]
#[allow(clippy::too_many_arguments, reason = "Public Python keyword arguments")]
fn write_smiles(
    mol: &PyAtomistic,
    canonical: bool,
    root: Option<&Bound<'_, PyAny>>,
    aromatic: &str,
    hydrogens: &str,
    include_stereo: bool,
    multi_component: &str,
    organic_subset: bool,
) -> PyResult<String> {
    let root = root
        .map(|atom| {
            atom.getattr("handle")
                .and_then(|h| h.extract::<u64>())
                .map(NodeId::from)
                .map_err(|_| PyValueError::new_err("root must be an Atom of mol"))
        })
        .transpose()?;
    let opts = emit_options(
        canonical,
        root,
        aromatic,
        hydrogens,
        include_stereo,
        multi_component,
        organic_subset,
    )?;
    let ir = SmilesIr::from_atomistic(mol.core(), &opts).map_err(smiles_error_to_pyerr)?;
    smiles_text(&ir)
}

/// The emit options the keywords of [`write_smiles`] spell.
fn emit_options(
    canonical: bool,
    root: Option<NodeId>,
    aromatic: &str,
    hydrogens: &str,
    include_stereo: bool,
    multi_component: &str,
    organic_subset: bool,
) -> PyResult<molrs::io::smiles::SmilesEmitOptions> {
    use molrs::io::smiles::{AromaticEmit, HydrogenEmit, MultiComponentEmit, SmilesEmitOptions};

    let aromatic = match aromatic {
        "as_marked" => AromaticEmit::AsMarked,
        "kekule_only" => AromaticEmit::KekuleOnly,
        other => {
            return Err(PyValueError::new_err(format!(
                "aromatic must be 'as_marked' or 'kekule_only', got {other:?}"
            )));
        }
    };
    let hydrogens = match hydrogens {
        "organic_subset" => HydrogenEmit::OrganicSubset,
        "explicit_all" => HydrogenEmit::ExplicitAll,
        "as_stored" => HydrogenEmit::AsStored,
        other => {
            return Err(PyValueError::new_err(format!(
                "hydrogens must be 'organic_subset', 'explicit_all', or 'as_stored', got {other:?}"
            )));
        }
    };
    let multi_component = match multi_component {
        "error_if_multiple" => MultiComponentEmit::ErrorIfMultiple,
        "join_dot" => MultiComponentEmit::JoinDot,
        "first_only" => MultiComponentEmit::FirstOnly,
        other => {
            return Err(PyValueError::new_err(format!(
                "multi_component must be 'error_if_multiple', 'join_dot', or 'first_only', got {other:?}"
            )));
        }
    };
    Ok(SmilesEmitOptions {
        canonical,
        root,
        aromatic,
        hydrogens,
        include_stereo,
        multi_component,
        organic_subset,
    })
}

/// A SMILES fragment body — SMILES plus bonding descriptors (``[<]OCC[>]``)
/// — as :func:`read_smiles_fragment` reads it, and as a ``CGsmiles``
/// fragment table holds an atomistic body
/// (:attr:`molrs.io.cgsmiles.CgFragmentDef.body`).
///
/// Its own class, because its conversion is not :meth:`SmilesIr.to_atomistic`:
/// the descriptors are data the graph does not hold, so :meth:`to_atomistic`
/// returns them beside it. ``str(body)`` writes the body back as text.
///
/// Examples
/// --------
/// >>> body = molrs.io.smiles.read_smiles_fragment("[<]OCC[>]")
/// >>> mol, descriptors = body.to_atomistic()
/// >>> [d.kind for _, d in descriptors]
/// ['<', '>']
#[pyclass(module = "molrs.io.smiles", name = "FragmentIr", frozen)]
pub struct PyFragmentIr {
    inner: FragmentIr,
}

/// Each bonding descriptor of a fragment beside the atom view it binds to.
type DescriptorSites<'py> = Vec<(Bound<'py, PyAny>, PyBondingDescriptor)>;

impl PyFragmentIr {
    /// A Python `FragmentIr` over a core one (`CgFragmentDef.body`).
    pub(crate) fn from_core(inner: FragmentIr) -> Self {
        Self { inner }
    }

    fn text(&self) -> PyResult<String> {
        use std::fmt::Write;
        let mut text = String::new();
        write!(text, "{}", self.inner)
            .map_err(|_| PyValueError::new_err("this fragment body has no text form"))?;
        Ok(text)
    }
}

#[pymethods]
impl PyFragmentIr {
    /// The body's graph and its bonding descriptors.
    ///
    /// The graph is built by the same walk as :meth:`SmilesIr.to_atomistic`;
    /// each descriptor is paired with the atom it binds to, in atom-visit
    /// order, then each atom's own descriptor order. Descriptors are data,
    /// not atoms: no hydrogens are added and no bond is made.
    ///
    /// Returns
    /// -------
    /// tuple[Atomistic, list[tuple[Atom, BondingDescriptor]]]
    ///
    /// Raises
    /// ------
    /// SmilesError
    ///     A ring-closure digit never closed, or SMARTS query atoms or bonds.
    fn to_atomistic<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(Py<PyAtomistic>, DescriptorSites<'py>)> {
        let (mol, descriptors) = self.inner.to_atomistic().map_err(smiles_error_to_pyerr)?;
        let mol = PyAtomistic::from_core(py, mol)?;
        let mut views: HashMap<u64, Bound<'py, PyAny>> = HashMap::new();
        for atom in mol.bind(py).getattr("atoms")?.try_iter()? {
            let atom = atom?;
            let handle: u64 = atom.getattr("handle")?.extract()?;
            views.insert(handle, atom);
        }
        let pairs = descriptors
            .into_iter()
            .map(|(node, descriptor)| {
                let atom = views.get(&u64::from(node)).cloned().ok_or_else(|| {
                    PyValueError::new_err("internal error: a descriptor names no atom of the body")
                })?;
                Ok((atom, PyBondingDescriptor::from_core(descriptor)))
            })
            .collect::<PyResult<Vec<_>>>()?;
        Ok((mol, pairs))
    }

    /// Build the ported :class:`~molrs.core.Atomistic` template of this body:
    /// the heavy atoms, plus one capping hydrogen *handle* and one port per
    /// bonding descriptor (with its label and bond order). No coordinates,
    /// no ``frag_id``; a body without descriptors gives a template without
    /// ports. The one-unit form of :meth:`CgSmilesIr.templates`.
    ///
    /// Returns
    /// -------
    /// Atomistic
    ///
    /// Raises
    /// ------
    /// SmilesError
    ///     If the body does not convert.
    ///
    /// Examples
    /// --------
    /// >>> eo = molrs.io.smiles.read_smiles_fragment("[<]OCC[>]").to_template()
    /// >>> eo.n_atoms, eo.n_ports
    /// (5, 2)
    fn to_template(&self, py: Python<'_>) -> PyResult<Py<PyAtomistic>> {
        let mol = self.inner.to_template().map_err(smiles_error_to_pyerr)?;
        PyAtomistic::from_core(py, mol)
    }

    /// The body as text, descriptors in the canonical trailing form —
    /// ``molrs::io::smiles::FragmentIr``'s ``Display``.
    fn __str__(&self) -> PyResult<String> {
        self.text()
    }

    fn __repr__(&self) -> String {
        match self.text() {
            Ok(text) => format!("FragmentIr({text:?})"),
            Err(_) => "FragmentIr(<no text form>)".to_owned(),
        }
    }
}

/// Register `SmilesError`, `SmilesIr`, `FragmentIr` and the three doors
/// `read_smiles`, `read_smiles_fragment` and `write_smiles`.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    for door in [
        wrap_pyfunction!(read_smiles, m)?,
        wrap_pyfunction!(read_smiles_fragment, m)?,
        wrap_pyfunction!(write_smiles, m)?,
    ] {
        crate::add_function(m, "molrs.io.smiles", door)?;
    }
    m.add(
        "SmilesError",
        m.py().get_type::<crate::error::SmilesError>(),
    )?;
    m.add_class::<PySmilesIr>()?;
    m.add_class::<PyFragmentIr>()?;
    Ok(())
}
