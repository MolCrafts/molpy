//! SMARTS matching and reaction transforms (`molrs::perceive::smarts`):
//! [`PySmartsPattern`] and the [`PySmartsMatch`] it yields, and the
//! reaction-SMARTS [`PyReaction`]. A pattern is a query over a *perceived*
//! graph — matching needs ring membership and aromaticity — so SMARTS is
//! perception's, wholly: parsing, matching, and a pattern generated from an
//! atom's environment ([`PySmartsPattern::from_environment`]).

use std::collections::HashMap;

use molrs::core::NodeId;
use molrs::perceive::smarts::{
    EnvironmentOptions, MatchOptions, NeighborStyle, Reaction, RingPrimitive, SmartsPattern,
};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple, PyType};

use super::atom_views;
use crate::core::graph_views::{Leaf, PyNodeRef};
use crate::core::molgraph::PyAtomistic;
use crate::error::molrs_error_to_pyerr;

/// The atom `atom` names, checked to be an atom of `leaf`.
fn own_atom(leaf: &Leaf<'_>, atom: &Bound<'_, PyAny>) -> PyResult<NodeId> {
    let atom = atom.cast::<PyNodeRef>().map_err(|_| {
        PyTypeError::new_err(format!(
            "expected an Atom view of the molecule, got {}",
            atom.get_type()
                .name()
                .map(|n| n.to_string())
                .unwrap_or_default()
        ))
    })?;
    leaf.own_node(atom)
}

/// A ``{Atom: label}`` dict as the core's `%LABEL` context.
fn own_labels(
    leaf: &Leaf<'_>,
    labels: Option<&Bound<'_, PyDict>>,
) -> PyResult<Option<HashMap<NodeId, String>>> {
    labels
        .map(|labels| {
            labels
                .iter()
                .map(|(atom, label)| Ok((own_atom(leaf, &atom)?, label.extract::<String>()?)))
                .collect()
        })
        .transpose()
}

/// A ``{map_number: Atom}`` binding as the core's.
fn own_binding(leaf: &Leaf<'_>, binding: &Bound<'_, PyDict>) -> PyResult<HashMap<u32, NodeId>> {
    binding
        .iter()
        .map(|(label, atom)| Ok((label.extract::<u32>()?, own_atom(leaf, &atom)?)))
        .collect()
}

// ---------------------------------------------------------------------------
// PySmartsMatch / PySmartsPattern — atom-map-aware SMARTS matcher over Atomistic
// ---------------------------------------------------------------------------

/// One SMARTS match, exposed to Python as `molrs.perceive.SmartsMatch`.
///
/// ``atoms`` holds the molecule's atom views in query-atom order. ``mapping``
/// is the Daylight atom-map projection (``:1`` → atom view), empty when the
/// query carries no map labels.
#[pyclass(module = "molrs.perceive", name = "SmartsMatch", frozen, subclass)]
pub struct PySmartsMatch {
    atoms: Vec<Py<PyAny>>,
    mapping: Vec<(u32, Py<PyAny>)>,
}

#[pymethods]
impl PySmartsMatch {
    /// The molecule's atoms in query-atom order.
    #[getter]
    fn atoms(&self, py: Python<'_>) -> Vec<Py<PyAny>> {
        self.atoms.iter().map(|a| a.clone_ref(py)).collect()
    }

    /// Daylight atom-map projection (``:n`` label -> molecule atom).
    #[getter]
    fn mapping<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let out = PyDict::new(py);
        for (label, atom) in &self.mapping {
            out.set_item(label, atom)?;
        }
        Ok(out)
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        Ok(format!(
            "SmartsMatch(atoms={}, mapping={})",
            pyo3::types::PyList::new(py, &self.atoms)?.repr()?,
            self.mapping(py)?.repr()?
        ))
    }
}

/// Compiled SMARTS query, exposed to Python as `molrs.perceive.SmartsPattern`.
///
/// A thin wrapper over the core [`SmartsPattern`] (`crates/molrs/src/core/chem/smarts`)
/// — the same backtracking subgraph-isomorphism engine that drives the OPLS-AA
/// typifier. Matching is non-uniquified (RDKit ``uniquify=False``): every
/// distinct query-atom → mol-atom embedding is reported as a
/// :class:`SmartsMatch`.
///
/// Daylight atom maps (``[C:1]``) are parsed and carried through but add **no**
/// match constraint (they are "ignored in molecule SMARTS"); each match's
/// :attr:`SmartsMatch.mapping` is its ``{map_number: Atom}`` dict.
///
/// Examples
/// --------
/// >>> pat = molrs.perceive.SmartsPattern("[C:1][O:2][H:3]")
/// >>> pat.find_matches(methanol)[0].mapping
/// {1: <C>, 2: <O>, 3: <H>}
#[pyclass(module = "molrs.perceive", name = "SmartsPattern", subclass)]
pub struct PySmartsPattern {
    inner: SmartsPattern,
}

#[pymethods]
impl PySmartsPattern {
    /// Parse a SMARTS string. Raises ``ValueError`` on a syntax error.
    #[new]
    fn new(smarts: &str) -> PyResult<Self> {
        let inner = SmartsPattern::parse(smarts).map_err(molrs_error_to_pyerr)?;
        Ok(Self { inner })
    }

    /// Whether at least one match exists in `mol`. ``labels`` and ``root``
    /// are as for :meth:`find_matches`.
    #[pyo3(signature = (mol, *, labels=None, root=None))]
    fn has_match(
        &self,
        mol: &Bound<'_, PyAtomistic>,
        labels: Option<&Bound<'_, PyDict>>,
        root: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<bool> {
        let leaf = Leaf::of(mol.as_any())?;
        let labels = own_labels(&leaf, labels)?;
        let root = root.map(|r| own_atom(&leaf, r)).transpose()?;
        Ok(self.inner.has_match(
            mol.try_borrow()?.core(),
            MatchOptions {
                labels: labels.as_ref(),
                root,
                limit: None,
            },
        ))
    }

    /// All matches, each a :class:`SmartsMatch`. ``labels`` (``{Atom:
    /// label}``) supplies the ``%LABEL`` context, ``root`` (an atom) pins
    /// query atom 0, and ``limit`` stops after N embeddings.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     A ``labels`` key or ``root`` is an atom of another molecule.
    #[pyo3(signature = (mol, *, labels=None, root=None, limit=None))]
    fn find_matches(
        &self,
        mol: &Bound<'_, PyAtomistic>,
        labels: Option<&Bound<'_, PyDict>>,
        root: Option<&Bound<'_, PyAny>>,
        limit: Option<usize>,
    ) -> PyResult<Vec<PySmartsMatch>> {
        let leaf = Leaf::of(mol.as_any())?;
        let labels = own_labels(&leaf, labels)?;
        let root = root.map(|r| own_atom(&leaf, r)).transpose()?;
        let matches = self.inner.find(
            mol.try_borrow()?.core(),
            MatchOptions {
                labels: labels.as_ref(),
                root,
                limit,
            },
        );
        matches
            .iter()
            .map(|m| {
                let mut mapping = self
                    .inner
                    .mapped(m)
                    .into_iter()
                    .map(|(label, atom)| Ok((label, leaf.node(u64::from(atom))?.unbind())))
                    .collect::<PyResult<Vec<_>>>()?;
                mapping.sort_by_key(|(label, _)| *label);
                Ok(PySmartsMatch {
                    atoms: atom_views(&leaf, m.atoms())?
                        .into_iter()
                        .map(Bound::unbind)
                        .collect(),
                    mapping,
                })
            })
            .collect()
    }

    /// Number of query atoms in the pattern.
    #[getter]
    fn n_query_atoms(&self) -> usize {
        self.inner.n_query_atoms()
    }

    /// Longest shortest-path length (bonds) on the query atom graph.
    ///
    /// Isolated atoms → ``0``. Pure syntax fact for molpy region typing.
    #[getter]
    fn max_bond_depth(&self) -> usize {
        self.inner.max_bond_depth()
    }

    /// Ring primitives used in this pattern (syntax only; no boundedness).
    ///
    /// Each item is ``(kind, n)`` where ``kind`` is one of
    /// ``"sized"`` / ``"membership"`` / ``"ring_count"`` / ``"ring_bond_count"``
    /// and ``n`` is ``None`` for membership.
    #[getter]
    fn ring_primitives(&self) -> Vec<(String, Option<u32>)> {
        self.inner
            .ring_primitives()
            .into_iter()
            .map(|p| match p {
                RingPrimitive::Sized(n) => ("sized".into(), Some(n)),
                RingPrimitive::Membership => ("membership".into(), None),
                RingPrimitive::RingCount(n) => ("ring_count".into(), Some(n)),
                RingPrimitive::RingBondCount(n) => ("ring_bond_count".into(), Some(n)),
            })
            .collect()
    }

    /// The ``:n`` atom-map label of query atom `query_atom` (``None`` if
    /// unlabelled / out of range).
    fn map_label(&self, query_atom: usize) -> Option<u32> {
        self.inner.map_label(query_atom)
    }

    /// The pattern that states the local environment of ``center`` in
    /// ``mol``: its element, and as many of degree, hydrogen count, charge,
    /// aromaticity, ring membership and neighbours out to ``reach`` bonds as
    /// the flags ask for. The pattern matches ``mol`` at ``center`` (as query
    /// atom 0); ``str(pattern)`` is its SMARTS text.
    ///
    /// Parameters
    /// ----------
    /// mol : Atomistic
    ///     The molecule.
    /// center : Atom
    ///     The centre atom, an atom of ``mol``.
    /// reach : int, default 1
    ///     Bond depth from the centre; at least 1.
    /// atomic_number : bool, default True
    ///     Write the centre as ``[#Z]`` rather than its element symbol.
    /// include_degree, include_h_count, include_charge, include_aromatic : bool, default True
    /// include_ring_membership, include_ring_size : bool, default False
    /// include_explicit_h_atoms : bool, default False
    /// include_bond_orders : bool, default True
    /// neighbor_style : {"chain", "recursive"}, default "chain"
    ///     Neighbours as spanning-tree branches, or as recursive ``$(...)``
    ///     environments of the centre.
    /// canonical_neighbor_order : bool, default True
    ///
    /// Returns
    /// -------
    /// SmartsPattern
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     ``reach`` is 0, ``center`` is not an atom of ``mol``, an atom has
    ///     no known element, or ``neighbor_style`` is not one of the two.
    #[classmethod]
    #[pyo3(signature = (
        mol,
        center,
        *,
        reach = 1,
        atomic_number = true,
        include_degree = true,
        include_h_count = true,
        include_charge = true,
        include_aromatic = true,
        include_ring_membership = false,
        include_ring_size = false,
        include_explicit_h_atoms = false,
        include_bond_orders = true,
        neighbor_style = "chain",
        canonical_neighbor_order = true,
    ))]
    #[allow(clippy::too_many_arguments, reason = "Public Python keyword arguments")]
    fn from_environment(
        _cls: &Bound<'_, PyType>,
        mol: &Bound<'_, PyAtomistic>,
        center: &Bound<'_, PyAny>,
        reach: u32,
        atomic_number: bool,
        include_degree: bool,
        include_h_count: bool,
        include_charge: bool,
        include_aromatic: bool,
        include_ring_membership: bool,
        include_ring_size: bool,
        include_explicit_h_atoms: bool,
        include_bond_orders: bool,
        neighbor_style: &str,
        canonical_neighbor_order: bool,
    ) -> PyResult<Self> {
        let neighbor_style = match neighbor_style {
            "chain" => NeighborStyle::Chain,
            "recursive" => NeighborStyle::Recursive,
            other => {
                return Err(PyValueError::new_err(format!(
                    "neighbor_style must be 'chain' or 'recursive', got {other:?}"
                )));
            }
        };
        let options = EnvironmentOptions {
            reach,
            atomic_number,
            include_degree,
            include_h_count,
            include_charge,
            include_aromatic,
            include_ring_membership,
            include_ring_size,
            include_explicit_h_atoms,
            include_bond_orders,
            neighbor_style,
            canonical_neighbor_order,
        };
        let center = own_atom(&Leaf::of(mol.as_any())?, center)?;
        let inner = SmartsPattern::from_environment(mol.try_borrow()?.core(), center, &options)
            .map_err(molrs_error_to_pyerr)?;
        Ok(Self { inner })
    }

    /// The pattern's SMARTS text.
    fn __str__(&self) -> String {
        self.inner.to_string()
    }

    fn __repr__(&self) -> String {
        format!("SmartsPattern({:?})", self.inner.to_string())
    }
}

// ---------------------------------------------------------------------------
// PyReaction — Daylight reaction-SMARTS (SMIRKS) transform over an Atomistic
// ---------------------------------------------------------------------------

/// Compiled reaction SMARTS, exposed to Python as `molrs.perceive.Reaction`.
///
/// A thin wrapper over the core [`Reaction`] (`crates/molrs/src/core/chem/smarts`).
/// Parses ``reactants >> products`` (tolerating an ignored ``>agent>`` field),
/// derives the graph edit from the Daylight atom-map diff, and applies it to one
/// matched occurrence in place. Reacting atoms may carry SMARTS queries
/// (RDKit-style reaction SMARTS); only concrete product atoms are addable.
///
/// Examples
/// --------
/// >>> rxn = molrs.perceive.Reaction("[N;H2:1].[C:2](=O)OC >> [N:1][C:2]=O")
/// >>> rxn.forming_bonds                 # [(1, 2)]
/// >>> binding = {}                       # match each reactant component ...
/// >>> for pat in rxn.reactant_patterns:  # ... and merge the map->atom dicts
/// ...     binding.update(pat.find_matches(mol)[0].mapping)
/// >>> rxn.apply(mol, binding)            # edits `mol` in place
#[pyclass(module = "molrs.perceive", name = "Reaction", subclass)]
pub struct PyReaction {
    inner: Reaction,
}

#[pymethods]
impl PyReaction {
    /// Parse a reaction SMARTS. Raises ``ValueError`` on a syntax or
    /// map-consistency error (e.g. an atom map that appears on only one side).
    #[new]
    fn new(reaction_smarts: &str) -> PyResult<Self> {
        let inner = Reaction::parse(reaction_smarts).map_err(molrs_error_to_pyerr)?;
        Ok(Self { inner })
    }

    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyTuple>> {
        crate::pickle::reduce_via_type(slf.as_any(), (slf.borrow().inner.source().to_owned(),))
    }

    /// The reactant components (LHS), one :class:`SmartsPattern` per top-level
    /// ``.`` component, for matching / pairing each independently.
    #[getter]
    fn reactant_patterns(&self) -> Vec<PySmartsPattern> {
        self.inner
            .reactants()
            .iter()
            .map(|p| PySmartsPattern { inner: p.clone() })
            .collect()
    }

    /// The ``(map_a, map_b)`` pairs of newly formed bonds between preserved
    /// atoms — the distance criterion for picking a reacting occurrence. Bonds
    /// that merely change order, and bonds to added atoms, are excluded.
    #[getter]
    fn forming_bonds(&self) -> Vec<(u32, u32)> {
        self.inner.forming_bonds()
    }

    /// Apply the transform to `mol` in place at the occurrence pinned by
    /// `binding` (``{map_number: Atom}``). Deletes unmapped-LHS atoms,
    /// adds unmapped-RHS atoms (no coordinates), forms/breaks bonds, then
    /// regenerates angle/dihedral topology and re-perceives aromaticity.
    ///
    /// Returns the deduplicated, deterministically-ordered list of *surviving*
    /// touched atoms (formed/broken/order-changed bond endpoints, added
    /// atoms, deleted atoms' surviving neighbours, and prop-set atoms). Deleted
    /// atoms are never included. The caller expands this seed set into a
    /// retype-safe region. ``labels`` (``{Atom: label}``) is the ``%LABEL``
    /// context the reactant patterns read.
    ///
    /// ``refresh=False`` skips the per-apply whole-graph angle/dihedral
    /// regeneration + aromaticity re-perception: a batch caller (crosslinking a
    /// melt with many edits) passes it and refreshes ONCE at the end, turning an
    /// O(edits × N) cost into O(edits × local). Matching only needs bonds, which
    /// are updated in place regardless.
    #[pyo3(signature = (mol, binding, labels=None, refresh=true))]
    fn apply<'py>(
        &self,
        mol: &Bound<'py, PyAtomistic>,
        binding: &Bound<'py, PyDict>,
        labels: Option<&Bound<'py, PyDict>>,
        refresh: bool,
    ) -> PyResult<Vec<Bound<'py, PyAny>>> {
        let leaf = Leaf::of(mol.as_any())?;
        let binding = own_binding(&leaf, binding)?;
        let labels = own_labels(&leaf, labels)?.unwrap_or_default();
        let touched = self
            .inner
            .apply(mol.try_borrow_mut()?.core_mut(), &binding, &labels, refresh)
            .map_err(molrs_error_to_pyerr)?;
        atom_views(&leaf, &touched)
    }

    /// Compile every binding against the intact graph, then apply the
    /// disjoint transforms as one batch (leaving groups deleted with one
    /// relation scan). ``labels`` and ``refresh`` are as for :meth:`apply`.
    ///
    /// Returns
    /// -------
    /// ReactionBatch
    ///     Per binding, the touched atoms (as :meth:`apply` returns them) and
    ///     the atoms the product created, in product creation order.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     Two bindings resolve the same leaving atom, a binding misses an
    ///     atom map, or an edit is refused.
    #[pyo3(signature = (mol, bindings, labels=None, refresh=true))]
    fn apply_many(
        &self,
        mol: &Bound<'_, PyAtomistic>,
        bindings: Vec<Bound<'_, PyDict>>,
        labels: Option<&Bound<'_, PyDict>>,
        refresh: bool,
    ) -> PyResult<PyReactionBatch> {
        let leaf = Leaf::of(mol.as_any())?;
        let bindings = bindings
            .iter()
            .map(|binding| own_binding(&leaf, binding))
            .collect::<PyResult<Vec<_>>>()?;
        let labels = own_labels(&leaf, labels)?.unwrap_or_default();
        let batch = self
            .inner
            .apply_many(
                mol.try_borrow_mut()?.core_mut(),
                &bindings,
                &labels,
                refresh,
            )
            .map_err(molrs_error_to_pyerr)?;
        let views = |sets: &[Vec<NodeId>]| -> PyResult<Vec<Vec<Py<PyAny>>>> {
            sets.iter()
                .map(|set| {
                    Ok(atom_views(&leaf, set)?
                        .into_iter()
                        .map(Bound::unbind)
                        .collect())
                })
                .collect()
        };
        Ok(PyReactionBatch {
            touched: views(&batch.touched)?,
            created: views(&batch.created)?,
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "Reaction(reactants={}, forming_bonds={:?})",
            self.inner.reactants().len(),
            self.inner.forming_bonds()
        )
    }
}

/// The result of :meth:`Reaction.apply_many` —
/// ``molrs::perceive::smarts::ReactionBatch``.
#[pyclass(module = "molrs.perceive", name = "ReactionBatch", frozen)]
pub struct PyReactionBatch {
    touched: Vec<Vec<Py<PyAny>>>,
    created: Vec<Vec<Py<PyAny>>>,
}

fn clone_groups(py: Python<'_>, groups: &[Vec<Py<PyAny>>]) -> Vec<Vec<Py<PyAny>>> {
    groups
        .iter()
        .map(|g| g.iter().map(|a| a.clone_ref(py)).collect())
        .collect()
}

#[pymethods]
impl PyReactionBatch {
    /// Per binding: the surviving touched atoms, deduplicated and ordered as
    /// :meth:`Reaction.apply` returns them.
    #[getter]
    fn touched(&self, py: Python<'_>) -> Vec<Vec<Py<PyAny>>> {
        clone_groups(py, &self.touched)
    }

    /// Per binding: the atoms the product added, in product creation order —
    /// the stable correspondence for replaying annotations onto RHS-added
    /// atoms (sorting handles cannot give it once deleted slots are reused).
    #[getter]
    fn created(&self, py: Python<'_>) -> Vec<Vec<Py<PyAny>>> {
        clone_groups(py, &self.created)
    }

    fn __repr__(&self) -> String {
        format!("ReactionBatch(n_bindings={})", self.touched.len())
    }
}
