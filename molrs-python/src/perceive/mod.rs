//! Python bindings for the chemical-perception layer (`molrs::perceive`,
//! Python `molrs.perceive`).
//!
//! Every perception is a free function at the Rust name, in one of two shapes:
//! `perceive_<fact>(mol)` is a **query** — it reports a side table (here
//! `perceive_rings` → `RingSet`, `perceive_rotatable_bonds` →
//! `[RotatableBond]`) and leaves the graph alone; `assign_<fact>(mol)` is a
//! **mutation** — it writes the fact onto `mol` itself, in place, as atom /
//! bond props, exactly as the core `assign_*(&mut Atomistic)` does, and
//! returns `None` (`assign_kekule_bond_orders` returns the core's `bool`).
//! A caller that wants the input kept copies first (`copy.copy(mol)`).
//! `add_hydrogens` is an edit of the graph and keeps its verb: like the core
//! function it returns a new graph.
//!
//! Chemical perception is all-atom, so every function is typed against
//! `Atomistic`; a `CoarseGrain` leaf is a `TypeError` from PyO3's own
//! extraction, not a wrong answer.
//!
//! [`PySubgraphMatcher`] (`molrs.perceive.SubgraphMatcher`) is the
//! coarse-grained counterpart: it snapshots a bead pattern and lists every
//! occurrence of it in a target `CoarseGrain` as bead-view groups, in
//! pattern order and without partitioning overlaps; it is typed against
//! `CoarseGrain` the same way, so an `Atomistic` target is a `TypeError`.

mod rings;
mod smarts;

use pyo3::prelude::*;

use molrs::perceive::{
    BondOrderSource, EquivalenceOptions, SubgraphMatcher, UnknownBondPolicy, add_hydrogens,
    assign_aromaticity, assign_bcc_bond_types, assign_bond_orders, assign_equivalence_classes,
    assign_kekule_bond_orders, assign_rings, assign_rotatable_bonds, assign_stereo,
    perceive_rotatable_bonds,
};

use crate::core::graph_views::Leaf;
use crate::core::molgraph::{PyAtomistic, PyCoarseGrain};

use crate::error::molrs_error_to_pyerr;

/// The views of `ids` on `leaf`, in order.
pub(crate) fn atom_views<'py>(
    leaf: &Leaf<'py>,
    ids: &[molrs::core::NodeId],
) -> PyResult<Vec<Bound<'py, PyAny>>> {
    ids.iter().map(|&id| leaf.node(u64::from(id))).collect()
}

/// Run the in-place `assign` on `mol`'s core graph.
fn assign_in_place<R>(
    mol: &Bound<'_, PyAtomistic>,
    assign: impl FnOnce(&mut molrs::core::Atomistic) -> Result<R, molrs::core::MolRsError>,
) -> PyResult<R> {
    assign(mol.try_borrow_mut()?.core_mut()).map_err(molrs_error_to_pyerr)
}

/// Where :func:`assign_bcc_bond_types` (and :class:`~molrs.ff.typifier.AtdTypifier`)
/// take bond orders from — ``molrs::perceive::BondOrderSource``.
///
/// ``Stated``: the orders the molecule states (aromatic bonds without a
/// number are kekulized). ``Connectivity``: the orders antechamber judges from
/// the connectivity alone (``bondtype -j full``), whatever the molecule
/// states — what ``antechamber`` does by default; every hydrogen must be
/// drawn.
#[pyclass(
    module = "molrs.perceive",
    name = "BondOrderSource",
    eq,
    eq_int,
    frozen,
    hash,
    from_py_object
)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PyBondOrderSource {
    Stated = 0,
    Connectivity = 1,
}

impl From<PyBondOrderSource> for BondOrderSource {
    fn from(source: PyBondOrderSource) -> Self {
        match source {
            PyBondOrderSource::Stated => BondOrderSource::Stated,
            PyBondOrderSource::Connectivity => BondOrderSource::Connectivity,
        }
    }
}

impl From<BondOrderSource> for PyBondOrderSource {
    fn from(source: BondOrderSource) -> Self {
        match source {
            BondOrderSource::Stated => PyBondOrderSource::Stated,
            BondOrderSource::Connectivity => PyBondOrderSource::Connectivity,
        }
    }
}

/// What a bond with no ``bond_type`` written counts as when rotatable bonds
/// are perceived — ``molrs::perceive::UnknownBondPolicy``.
///
/// ``NotRotatable`` (the default): only a bond that *is* single rotates.
/// ``AsSingle``: an unclassified bond counts as single, then the degree and
/// ring rules apply (all-single graphs whose orders were never written).
#[pyclass(
    module = "molrs.perceive",
    name = "UnknownBondPolicy",
    eq,
    eq_int,
    frozen,
    hash,
    from_py_object
)]
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum PyUnknownBondPolicy {
    NotRotatable = 0,
    AsSingle = 1,
}

impl From<PyUnknownBondPolicy> for UnknownBondPolicy {
    fn from(policy: PyUnknownBondPolicy) -> Self {
        match policy {
            PyUnknownBondPolicy::NotRotatable => UnknownBondPolicy::NotRotatable,
            PyUnknownBondPolicy::AsSingle => UnknownBondPolicy::AsSingle,
        }
    }
}

/// One rotatable bond, ``j``–``k``, with the atoms that rotate about it —
/// ``molrs::perceive::RotatableBond``, its positional indices resolved to the
/// molecule's atom views.
#[pyclass(module = "molrs.perceive", name = "RotatableBond", frozen)]
pub struct PyRotatableBond {
    j: Py<PyAny>,
    k: Py<PyAny>,
    downstream: Vec<Py<PyAny>>,
}

#[pymethods]
impl PyRotatableBond {
    /// The axis start atom.
    #[getter]
    fn j(&self, py: Python<'_>) -> Py<PyAny> {
        self.j.clone_ref(py)
    }

    /// The axis end atom.
    #[getter]
    fn k(&self, py: Python<'_>) -> Py<PyAny> {
        self.k.clone_ref(py)
    }

    /// The atoms on the ``k`` side, rotated by a torsion about the bond.
    #[getter]
    fn downstream(&self, py: Python<'_>) -> Vec<Py<PyAny>> {
        self.downstream.iter().map(|a| a.clone_ref(py)).collect()
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        Ok(format!(
            "RotatableBond(j={}, k={}, downstream={})",
            self.j.bind(py).repr()?,
            self.k.bind(py).repr()?,
            self.downstream.len()
        ))
    }
}

/// Perceive the rotatable bonds of ``mol`` — a query: ``mol`` is left
/// untouched.
///
/// A bond is rotatable when its class is single, both endpoints have degree
/// > 1, and it lies on no ring. :func:`assign_rotatable_bonds` writes the same
/// answer onto the bonds.
///
/// Parameters
/// ----------
/// mol : Atomistic
/// unknown_bond : UnknownBondPolicy, default ``UnknownBondPolicy.NotRotatable``
///     What a bond with no ``bond_type`` written counts as.
///
/// Returns
/// -------
/// list[RotatableBond]
///     ``j`` / ``k`` / ``downstream`` are ``mol``'s atom views.
#[pyfunction(name = "perceive_rotatable_bonds")]
#[pyo3(signature = (mol, *, unknown_bond = PyUnknownBondPolicy::NotRotatable))]
fn perceive_rotatable_bonds_py(
    mol: &Bound<'_, PyAtomistic>,
    unknown_bond: PyUnknownBondPolicy,
) -> PyResult<Vec<PyRotatableBond>> {
    let found = perceive_rotatable_bonds(mol.try_borrow()?.core(), unknown_bond.into());
    let leaf = Leaf::of(mol.as_any())?;
    let ids = leaf.node_ids()?;
    found
        .into_iter()
        .map(|bond| {
            let downstream = bond
                .downstream
                .iter()
                .map(|&i| leaf.node(ids[i]).map(Bound::unbind))
                .collect::<PyResult<Vec<_>>>()?;
            Ok(PyRotatableBond {
                j: leaf.node(ids[bond.j])?.unbind(),
                k: leaf.node(ids[bond.k])?.unbind(),
                downstream,
            })
        })
        .collect()
}

/// Perceive rings (SSSR) and write them onto ``mol``, in place.
///
/// Every atom and every bond receives ``is_in_ring`` (0/1) and ``n_rings`` —
/// including the acyclic ones, which are explicitly flagged ``0`` rather than
/// left unset. For the ring list itself use :func:`perceive_rings`.
///
/// Examples
/// --------
/// >>> molrs.perceive.assign_rings(mol)
/// >>> atom["is_in_ring"]
/// 1
#[pyfunction(name = "assign_rings")]
fn assign_rings_py(mol: &Bound<'_, PyAtomistic>) -> PyResult<()> {
    assign_in_place(mol, assign_rings)
}

/// Bring ``mol`` to the standard aromatic representation, in place.
///
/// On return every aromatic atom carries ``is_aromatic``, every aromatic bond
/// carries ``bond_type = 4``, and every bond carries an integer
/// ``bond_number`` — the localized Lewis structure. Nothing carries a
/// fractional order: aromaticity is a bond *type*, not the number 1.5.
///
/// An input that already declares its aromatic bonds (a lowercase SMILES) is
/// *kekulized*; one that does not is *perceived* from its integer bond
/// numbers, rings, valences and electron counts, and any assignment it already
/// stated is kept. Hydrogens are neither added nor required.
#[pyfunction(name = "assign_aromaticity")]
fn assign_aromaticity_py(mol: &Bound<'_, PyAtomistic>) -> PyResult<()> {
    assign_in_place(mol, assign_aromaticity)
}

/// Add the hydrogens implied by each heavy atom's open valence.
///
/// A graph edit, not an assignment: returns a new graph — the heavy-atom
/// skeleton of ``mol`` plus the perceived hydrogens and their bonds; ``mol``
/// is left untouched.
///
/// Raises
/// ------
/// ValueError
///     If repletion reports a stale atom handle on the graph it built — an
///     invariant no molecule built through this package can break.
#[pyfunction(name = "add_hydrogens")]
fn add_hydrogens_py(py: Python<'_>, mol: &PyAtomistic) -> PyResult<Py<PyAtomistic>> {
    let out = add_hydrogens(mol.core()).map_err(molrs_error_to_pyerr)?;
    mol.derive(py, out)
}

/// Perceive stereochemistry from 3-D coordinates and write it onto ``mol``,
/// in place: a ``stereo`` prop appears only where a real descriptor was
/// perceived — ``"CW"`` / ``"CCW"`` on atoms, ``"E"`` / ``"Z"`` /
/// ``"either"`` on bonds.
#[pyfunction(name = "assign_stereo")]
fn assign_stereo_py(mol: &Bound<'_, PyAtomistic>) -> PyResult<()> {
    assign_in_place(mol, assign_stereo)
}

/// Perceive rotatable bonds and write them onto ``mol``, in place.
///
/// The :func:`perceive_rotatable_bonds` answer on the bonds: every bond
/// receives ``is_rotatable`` (0/1).
///
/// Parameters
/// ----------
/// mol : Atomistic
///     The molecule to annotate.
/// unknown_bond : UnknownBondPolicy, default ``UnknownBondPolicy.NotRotatable``
///     What a bond with no ``bond_type`` written counts as.
#[pyfunction(name = "assign_rotatable_bonds")]
#[pyo3(signature = (mol, *, unknown_bond = PyUnknownBondPolicy::NotRotatable))]
fn assign_rotatable_bonds_py(
    mol: &Bound<'_, PyAtomistic>,
    unknown_bond: PyUnknownBondPolicy,
) -> PyResult<()> {
    assign_in_place(mol, |m| assign_rotatable_bonds(m, unknown_bond.into()))
}

/// Perceive antechamber's BCC bond types and write them onto ``mol``, in
/// place: every bond receives a ``bcc_bond_type`` in ``{1, 2, 3, 6, 7, 8,
/// 9}`` — the alphabet AM1-BCC's atom-type rules and correction table are
/// keyed on, which distinguishes aromatic bonds (7/8) and *delocalized* ones
/// (9, e.g. a carboxylate's two equivalent C–O bonds) from plain orders. The
/// bond's ``type`` — the caller's force-field label — is neither read nor
/// written.
///
/// Parameters
/// ----------
/// mol : Atomistic
/// source : BondOrderSource
///     Which bond orders the types follow: ``BondOrderSource.Stated`` (the
///     molecule's own) or ``BondOrderSource.Connectivity`` (judged from the
///     connectivity alone, as antechamber runs it; every hydrogen must be
///     drawn). Required.
#[pyfunction(name = "assign_bcc_bond_types")]
fn assign_bcc_bond_types_py(
    mol: &Bound<'_, PyAtomistic>,
    source: PyBondOrderSource,
) -> PyResult<()> {
    assign_in_place(mol, |m| assign_bcc_bond_types(m, source.into()))
}

/// Judge every bond's order from the connectivity alone, as antechamber's
/// ``bondtype -j full`` does, and write it onto ``mol``, in place: every
/// judged bond gets a localized ``bond_number`` (1/2/3) and the ``bond_type``
/// it implies. The answer follows the atom and bond order, as antechamber's
/// does; the bonds of a residue no valence state closes are left untouched.
#[pyfunction(name = "assign_bond_orders")]
fn assign_bond_orders_py(mol: &Bound<'_, PyAtomistic>) -> PyResult<()> {
    assign_in_place(mol, assign_bond_orders)
}

/// Assign a localized (Kekulé) ``bond_number`` to every aromatic bond of
/// ``mol``, in place, and return whether every aromatic bond came out with a
/// legal number.
///
/// Kekulization and nothing else: a molecule whose aromatic bonds are not
/// marked yet is left unchanged (``True``), because deciding *which* bonds are
/// aromatic belongs to :func:`assign_aromaticity`. An aromatic bond that
/// already carries a legal number keeps it; a system with no legal assignment
/// is left entirely unchanged rather than half-assigned (``False``).
#[pyfunction(name = "assign_kekule_bond_orders")]
fn assign_kekule_bond_orders_py(mol: &Bound<'_, PyAtomistic>) -> PyResult<bool> {
    assign_in_place(mol, assign_kekule_bond_orders)
}

/// Perceive charge-equivalence classes and write them onto ``mol``, in place:
/// every atom receives an ``equiv_class`` id, antechamber's default ``-eq 1``
/// partition (the one AM1-BCC averages its AM1 charges over). Whether to
/// average is a property of the charge model (:meth:`BccModel.needs_equivalencing`),
/// not of the graph.
#[pyfunction(name = "assign_equivalence_classes")]
fn assign_equivalence_classes_py(mol: &Bound<'_, PyAtomistic>) -> PyResult<()> {
    assign_in_place(mol, |m| {
        assign_equivalence_classes(m, EquivalenceOptions::default())
    })
}

/// Bead-group pattern matching over coarse-grained graphs —
/// `molrs.perceive.SubgraphMatcher`.
///
/// Snapshots a bead pattern once; :meth:`find` lists every induced occurrence
/// of it in a target :class:`~molrs.core.CoarseGrain`. Beads match on equal
/// ``bead_type``; bonds match on adjacency.
///
/// ``find`` does **not** partition: overlapping groups are all returned, and a
/// caller that needs disjoint groups selects among them.
///
/// Parameters
/// ----------
/// pattern : CoarseGrain
///     The bead pattern, e.g. ``CgSmilesIr.parse("{[#1][#4]}").to_coarsegrain()``.
///     It is copied, so later edits to it do not affect the matcher.
///
/// Raises
/// ------
/// TypeError
///     If ``pattern`` is not a :class:`~molrs.core.CoarseGrain`.
///
/// Examples
/// --------
/// The two groups below share the middle bead:
///
/// >>> pattern = molrs.io.cgsmiles.CgSmilesIr.parse("{[#1][#4]}").to_coarsegrain()
/// >>> target = molrs.io.cgsmiles.CgSmilesIr.parse("{[#1][#4][#1]}").to_coarsegrain()
/// >>> len(molrs.perceive.SubgraphMatcher(pattern).find(target))
/// 2
#[pyclass(module = "molrs.perceive", name = "SubgraphMatcher", frozen)]
pub struct PySubgraphMatcher {
    inner: SubgraphMatcher,
}

#[pymethods]
impl PySubgraphMatcher {
    #[new]
    fn new(pattern: PyRef<'_, PyCoarseGrain>) -> Self {
        Self {
            inner: SubgraphMatcher::new(pattern.core()),
        }
    }

    /// Every induced occurrence of the pattern in ``target``.
    ///
    /// One group per distinct bead set: ``group[i]`` is the target bead
    /// matched to pattern bead ``i``, in the pattern's bead order. Groups are
    /// not partitioned, so two groups may share beads. The GIL is released
    /// while matching.
    ///
    /// Parameters
    /// ----------
    /// target : CoarseGrain
    ///     The bead graph to search.
    ///
    /// Returns
    /// -------
    /// list[list[Bead]]
    ///     The target's beads, one group per occurrence; ``[]`` when there is
    ///     no occurrence (or the pattern or target is empty).
    ///
    /// Raises
    /// ------
    /// TypeError
    ///     If ``target`` is not a :class:`~molrs.core.CoarseGrain`.
    fn find<'py>(
        &self,
        py: Python<'py>,
        target: &Bound<'py, PyCoarseGrain>,
    ) -> PyResult<Vec<Vec<Bound<'py, PyAny>>>> {
        let groups = {
            let graph = target.try_borrow()?;
            let (matcher, graph) = (&self.inner, graph.core());
            py.detach(|| matcher.find(graph))
        };
        let leaf = Leaf::of(target.as_any())?;
        groups
            .iter()
            .map(|group| atom_views(&leaf, group))
            .collect()
    }

    fn __repr__(&self) -> String {
        "SubgraphMatcher()".to_string()
    }
}

/// Register `molrs.perceive`.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    for f in [
        wrap_pyfunction!(assign_rings_py, m)?,
        wrap_pyfunction!(assign_aromaticity_py, m)?,
        wrap_pyfunction!(add_hydrogens_py, m)?,
        wrap_pyfunction!(assign_stereo_py, m)?,
        wrap_pyfunction!(assign_rotatable_bonds_py, m)?,
        wrap_pyfunction!(assign_bcc_bond_types_py, m)?,
        wrap_pyfunction!(perceive_rotatable_bonds_py, m)?,
        wrap_pyfunction!(assign_bond_orders_py, m)?,
        wrap_pyfunction!(assign_kekule_bond_orders_py, m)?,
        wrap_pyfunction!(assign_equivalence_classes_py, m)?,
        wrap_pyfunction!(rings::perceive_rings_py, m)?,
    ] {
        crate::add_function(m, "molrs.perceive", f)?;
    }
    m.add_class::<PySubgraphMatcher>()?;
    m.add_class::<PyBondOrderSource>()?;
    m.add_class::<PyUnknownBondPolicy>()?;
    m.add_class::<PyRotatableBond>()?;
    m.add_class::<rings::PyRingSet>()?;
    m.add_class::<smarts::PySmartsPattern>()?;
    m.add_class::<smarts::PySmartsMatch>()?;
    m.add_class::<smarts::PyReaction>()?;
    m.add_class::<smarts::PyReactionBatch>()?;
    Ok(())
}
