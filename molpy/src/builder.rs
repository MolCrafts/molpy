//! Python bindings for `molrs::builder`: the structure generators
//! `CarbonTubeBuilder` and `GrapheneBuilder`, each building a fresh `Frame`,
//! and site-graph assembly — `Assembler(library, SitePlacer(),
//! AxisOrienter()).assemble(sites)`, which orients, places and links one world
//! `Fragment` — and coarse-graining: `Coarsener(source).coarsen(groups,
//! names)` maps disjoint node groups of a held `CoarseGrain` or `Atomistic`
//! onto the sites of a new `CoarseGrain` (centre of mass, summed mass, one
//! type per group).

use std::collections::HashMap;

use molrs::builder::{
    AssembleError, Assembler, AxisOrienter, GrowthPlacer, Orienter, Placer, SitePlacer,
};
use molrs::builder::{CarbonTubeBuilder, Coarsener, GrapheneBuilder, JunctionReport, TubeExtent};
use molrs::core::{Atomistic, CoarseGrain, Graph, MolGraph, NodeId};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyMapping;

use crate::core::frame::PyFrame;
use crate::core::graph_views::{Leaf, PyRefs};
use crate::core::molgraph::{PyAtomistic, PyCoarseGrain, PyMolGraph};
use crate::core::simbox::PyBox;

/// Exact single-wall carbon nanotube builder.
#[pyclass(module = "molpy.builder", name = "CarbonTubeBuilder", subclass)]
pub struct PyCarbonTubeBuilder {
    inner: CarbonTubeBuilder,
}

#[pymethods]
impl PyCarbonTubeBuilder {
    #[new]
    #[pyo3(signature = (n, m, *, length=None, cells=None, bond_length=1.42, periodic=false, vacuum=10.0))]
    fn new(
        n: u32,
        m: u32,
        length: Option<f64>,
        cells: Option<usize>,
        bond_length: f64,
        periodic: bool,
        vacuum: f64,
    ) -> PyResult<Self> {
        if length.is_some() && cells.is_some() {
            return Err(PyTypeError::new_err(
                "length and cells are mutually exclusive",
            ));
        }

        let mut inner = CarbonTubeBuilder::new(n, m)
            .map_err(|error| PyValueError::new_err(error.to_string()))?
            .with_bond_length(bond_length)
            .map_err(|error| PyValueError::new_err(error.to_string()))?
            .with_periodic(periodic)
            .with_vacuum(vacuum)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        let extent = match (cells, length) {
            (Some(cells), _) => Some(TubeExtent::Cells(cells)),
            (None, Some(length)) => Some(TubeExtent::Length(length)),
            (None, None) => None,
        };
        if let Some(extent) = extent {
            inner = inner
                .with_extent(extent)
                .map_err(|error| PyValueError::new_err(error.to_string()))?;
        }
        inner
            .validate()
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        Ok(Self { inner })
    }

    /// Build a fresh frame containing atoms, bonds, and the simulation box.
    #[pyo3(signature = (*, atom_type=None, charge=0.0))]
    fn build(&self, atom_type: Option<String>, charge: f64) -> PyResult<PyFrame> {
        let mut builder = self
            .inner
            .clone()
            .with_charge(charge)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        if let Some(atom_type) = atom_type {
            builder = builder
                .with_atom_type(atom_type)
                .map_err(|error| PyValueError::new_err(error.to_string()))?;
        }
        PyFrame::from_core_frame(
            builder
                .build()
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        )
    }

    /// Return the matching simulation cell, optionally overriding vacuum.
    #[pyo3(signature = (*, vacuum=None))]
    fn cell(&self, vacuum: Option<f64>) -> PyResult<PyBox> {
        let builder = match vacuum {
            Some(vacuum) => self
                .inner
                .clone()
                .with_vacuum(vacuum)
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
            None => self.inner.clone(),
        };
        Ok(PyBox {
            inner: builder
                .cell()
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        })
    }

    #[getter]
    fn n(&self) -> u32 {
        self.inner.n()
    }

    #[getter]
    fn m(&self) -> u32 {
        self.inner.m()
    }

    #[getter]
    fn cells(&self) -> usize {
        self.inner.cells()
    }

    #[getter]
    fn bond_length(&self) -> f64 {
        self.inner.bond_length()
    }

    #[getter]
    fn periodic(&self) -> bool {
        self.inner.periodic()
    }
}

/// Rectangular graphene (honeycomb) sheet builder.
#[pyclass(module = "molpy.builder", name = "GrapheneBuilder", subclass)]
pub struct PyGrapheneBuilder {
    inner: GrapheneBuilder,
}

#[pymethods]
impl PyGrapheneBuilder {
    #[new]
    #[pyo3(signature = (nx, ny, *, bond_length=1.42, vacuum=10.0, periodic_xy=true))]
    fn new(nx: u32, ny: u32, bond_length: f64, vacuum: f64, periodic_xy: bool) -> PyResult<Self> {
        let inner = GrapheneBuilder::new(nx, ny)
            .map_err(|e| PyValueError::new_err(e.to_string()))?
            .with_bond_length(bond_length)
            .map_err(|e| PyValueError::new_err(e.to_string()))?
            .with_vacuum(vacuum)
            .map_err(|e| PyValueError::new_err(e.to_string()))?
            .with_periodic_xy(periodic_xy);
        Ok(Self { inner })
    }

    #[pyo3(signature = (*, atom_type=None, charge=0.0))]
    fn build(&self, atom_type: Option<String>, charge: f64) -> PyResult<PyFrame> {
        let mut builder = self
            .inner
            .clone()
            .with_charge(charge)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        if let Some(atom_type) = atom_type {
            builder = builder
                .with_atom_type(atom_type)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
        }
        PyFrame::from_core_frame(
            builder
                .build()
                .map_err(|e| PyValueError::new_err(e.to_string()))?,
        )
    }

    #[pyo3(signature = (*, vacuum=None))]
    fn cell(&self, vacuum: Option<f64>) -> PyResult<PyBox> {
        let builder = match vacuum {
            Some(v) => self
                .inner
                .clone()
                .with_vacuum(v)
                .map_err(|e| PyValueError::new_err(e.to_string()))?,
            None => self.inner.clone(),
        };
        Ok(PyBox {
            inner: builder
                .cell()
                .map_err(|e| PyValueError::new_err(e.to_string()))?,
        })
    }

    #[getter]
    fn nx(&self) -> u32 {
        self.inner.nx()
    }

    #[getter]
    fn ny(&self) -> u32 {
        self.inner.ny()
    }

    #[getter]
    fn bond_length(&self) -> f64 {
        self.inner.bond_length()
    }

    #[getter]
    fn periodic_xy(&self) -> bool {
        self.inner.periodic_xy()
    }
}

/// The translation-only placer — ``molpy.builder.SitePlacer``.
///
/// Each copy's centroid (the mean of its atom positions, Å) lands on its site;
/// rotation is the orienter's job. Takes no arguments.
///
/// Examples
/// --------
/// >>> assembler = molpy.builder.Assembler(
/// ...     library, molpy.builder.SitePlacer(), molpy.builder.AxisOrienter()
/// ... )
#[pyclass(module = "molpy.builder", name = "SitePlacer", frozen)]
pub struct PySitePlacer;

#[pymethods]
impl PySitePlacer {
    #[new]
    fn new() -> Self {
        Self
    }

    fn __repr__(&self) -> String {
        "SitePlacer()".to_owned()
    }
}

/// The growth placer — ``molpy.builder.GrowthPlacer``.
///
/// Grows each molecule copy by copy in the assembler's walk order: the first
/// copy keeps its template pose (its centroid on the site when the
/// site graph has positions), and every later copy is rotated and moved so
/// the anchor of its port toward its parent points back along the parent's
/// leaving bond, one covalent bond length (the two anchors' covalent radii)
/// from the parent's anchor. Needs no site positions; bond angles and
/// overlaps are left to a later minimisation. Takes no arguments.
#[pyclass(module = "molpy.builder", name = "GrowthPlacer", frozen)]
pub struct PyGrowthPlacer;

#[pymethods]
impl PyGrowthPlacer {
    #[new]
    fn new() -> Self {
        Self
    }

    fn __repr__(&self) -> String {
        "GrowthPlacer()".to_owned()
    }
}

/// The axis orienter — ``molpy.builder.AxisOrienter``.
///
/// Turns each copy about its template's centroid. A chain site (only
/// ``<`` / ``>`` ports, a two-port template) matches the template's
/// backbone-to-centre direction to the site axis (``CoarseGrain.axes``) and
/// its two joining atoms to the site's bond line. Any other bonded site fits
/// the template's port directions to its bond directions. A site with no
/// bond is not turned. Takes no arguments.
#[pyclass(module = "molpy.builder", name = "AxisOrienter", frozen)]
pub struct PyAxisOrienter;

#[pymethods]
impl PyAxisOrienter {
    #[new]
    fn new() -> Self {
        Self
    }

    fn __repr__(&self) -> String {
        "AxisOrienter()".to_owned()
    }
}

/// One placed, linked world from a site graph — ``molpy.builder.Assembler``.
///
/// Each bead of the site graph is one unit: a copy of
/// ``library[bead_type]``, turned by the orienter (when given) and given its
/// pose by the placer.
/// Each site bond joins one port of each end's copy (``<`` with ``>``, ``$``
/// with ``$``); the leaving groups are removed. Any topology works: chains,
/// branches, rings. Every atom gets ``frag_id`` (the site's ordinal),
/// ``mol_id`` (its connected component's ordinal + 1), ``template`` (the
/// library name it was copied from) and ``template_atom`` (its row in that
/// template, never renumbered).
///
/// Parameters
/// ----------
/// library : Mapping[str, MolGraph | Atomistic | CoarseGrain]
///     Name → template, copied at construction, every template of ONE graph
///     type — the type :meth:`assemble` builds the world as — with ports
///     where a site bonds. A template without ports can only fill an
///     unbonded site.
/// placer : SitePlacer | GrowthPlacer
///     ``SitePlacer`` moves each copy's centroid onto its site;
///     ``GrowthPlacer`` grows each molecule onto its parents' ports and needs
///     no site positions.
/// orienter : AxisOrienter, optional
///     Turns each copy about its centroid before it is placed; needs
///     site positions.
///
/// Raises
/// ------
/// TypeError
///     If ``library`` is not a mapping of ``str`` to graphs of one type,
///     ``placer`` is not a :class:`SitePlacer` or :class:`GrowthPlacer`, or
///     ``orienter`` is not an :class:`AxisOrienter`.
/// ValueError
///     If ``library`` is empty (it names the world's graph type).
///
/// Examples
/// --------
/// >>> sites = molpy.builder.Coarsener(cg).coarsen(groups, names)
/// >>> world = molpy.builder.Assembler(
/// ...     {"PMA": pma, "Li": li},
/// ...     molpy.builder.SitePlacer(),
/// ...     molpy.builder.AxisOrienter(),
/// ... ).assemble(sites)
#[pyclass(module = "molpy.builder", name = "Assembler", frozen)]
pub struct PyAssembler {
    inner: AnyAssembler,
}

/// An assembler over one graph type: the library's.
enum AnyAssembler {
    MolGraph(Assembler<MolGraph>),
    Atomistic(Assembler<Atomistic>),
    CoarseGrain(Assembler<CoarseGrain>),
}

/// Which placer an `Assembler` was given: the type is the whole choice.
#[derive(Clone, Copy)]
enum PlacerKind {
    Site,
    Growth,
}

impl PlacerKind {
    fn of(placer: &Bound<'_, PyAny>) -> PyResult<Self> {
        if placer.cast::<PySitePlacer>().is_ok() {
            Ok(Self::Site)
        } else if placer.cast::<PyGrowthPlacer>().is_ok() {
            Ok(Self::Growth)
        } else {
            Err(PyTypeError::new_err(format!(
                "placer must be a SitePlacer or a GrowthPlacer, not {}",
                placer.get_type().name()?
            )))
        }
    }

    fn build<G: Graph>(self) -> Box<dyn Placer<G>> {
        match self {
            Self::Site => Box::new(SitePlacer::new()),
            Self::Growth => Box::new(GrowthPlacer::new()),
        }
    }
}

/// The library's templates as one graph type, copied.
fn templates<G>(
    library: &[(String, Bound<'_, PyAny>)],
    copy: impl Fn(&Bound<'_, PyAny>) -> Option<PyResult<G>>,
    class: &str,
) -> PyResult<HashMap<String, G>> {
    library
        .iter()
        .map(|(name, value)| {
            let template = copy(value).ok_or_else(|| {
                PyTypeError::new_err(format!(
                    "library['{name}'] must be a {class} like the rest of the library: \
                     every template is of the world's one graph type"
                ))
            })??;
            Ok((name.clone(), template))
        })
        .collect()
}

fn assembler<G: Graph>(
    library: HashMap<String, G>,
    placer: PlacerKind,
    orienter: bool,
) -> Assembler<G> {
    let orienter = orienter.then(|| Box::new(AxisOrienter::new()) as Box<dyn Orienter<G>>);
    Assembler::new(library, placer.build(), orienter)
}

#[pymethods]
impl PyAssembler {
    #[new]
    #[pyo3(signature = (library, placer, orienter=None))]
    fn new(
        library: &Bound<'_, PyMapping>,
        placer: &Bound<'_, PyAny>,
        orienter: Option<&Bound<'_, PyAxisOrienter>>,
    ) -> PyResult<Self> {
        let placer = PlacerKind::of(placer)?;
        let orienter = orienter.is_some();
        let library: Vec<(String, Bound<'_, PyAny>)> = library
            .items()?
            .iter()
            .map(|item| item.extract())
            .collect::<PyResult<_>>()?;
        let Some((_, first)) = library.first() else {
            return Err(PyValueError::new_err(
                "an Assembler needs at least one template: the library's graph type is the world's",
            ));
        };
        let inner = if first.cast::<PyAtomistic>().is_ok() {
            let copy = |value: &Bound<'_, PyAny>| {
                let graph = value.cast::<PyAtomistic>().ok()?;
                Some(
                    graph
                        .try_borrow()
                        .map(|g| g.core().clone())
                        .map_err(PyErr::from),
                )
            };
            AnyAssembler::Atomistic(assembler(
                templates(&library, copy, "Atomistic")?,
                placer,
                orienter,
            ))
        } else if first.cast::<PyCoarseGrain>().is_ok() {
            let copy = |value: &Bound<'_, PyAny>| {
                let graph = value.cast::<PyCoarseGrain>().ok()?;
                Some(
                    graph
                        .try_borrow()
                        .map(|g| g.core().clone())
                        .map_err(PyErr::from),
                )
            };
            AnyAssembler::CoarseGrain(assembler(
                templates(&library, copy, "CoarseGrain")?,
                placer,
                orienter,
            ))
        } else if first.cast::<PyMolGraph>().is_ok() {
            let copy = |value: &Bound<'_, PyAny>| {
                let graph = value.cast::<PyMolGraph>().ok()?;
                Some(
                    graph
                        .try_borrow()
                        .map(|g| g.core().clone())
                        .map_err(PyErr::from),
                )
            };
            AnyAssembler::MolGraph(assembler(
                templates(&library, copy, "MolGraph")?,
                placer,
                orienter,
            ))
        } else {
            return Err(PyTypeError::new_err(format!(
                "library values must be graphs (MolGraph, Atomistic, CoarseGrain), not {}",
                first.get_type().name()?
            )));
        };
        Ok(Self { inner })
    }

    /// Place and join one copy per site; return the world.
    ///
    /// The GIL is released while assembling.
    ///
    /// Parameters
    /// ----------
    /// sites : CoarseGrain
    ///     The site graph: each bead's ``bead_type`` names its template and
    ///     each bond joins two copies; its optional position (Å) and axis
    ///     are read by the placer and the orienter.
    ///
    /// Returns
    /// -------
    /// MolGraph | Atomistic | CoarseGrain
    ///     The world, of the library's graph type; ports without a site bond
    ///     stay on it. Empty when ``sites`` is empty.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     Naming the site at fault: a bead lacks its type or position, a name
    ///     is not in the library, no accepting port exists for every bond of
    ///     a site, a site cannot be oriented or placed, a join is refused, or
    ///     the world breaks the graph type's invariant.
    fn assemble(&self, py: Python<'_>, sites: PyRef<'_, PyCoarseGrain>) -> PyResult<Py<PyAny>> {
        let sites = sites.core().clone();
        let refuse = |e: AssembleError| PyValueError::new_err(e.to_string());
        Ok(match &self.inner {
            AnyAssembler::MolGraph(a) => {
                let world = py.detach(|| a.assemble(&sites)).map_err(refuse)?;
                PyMolGraph::from_core(py, world)?.into_any()
            }
            AnyAssembler::Atomistic(a) => {
                let world = py.detach(|| a.assemble(&sites)).map_err(refuse)?;
                PyAtomistic::from_core(py, world)?.into_any()
            }
            AnyAssembler::CoarseGrain(a) => {
                let world = py.detach(|| a.assemble(&sites)).map_err(refuse)?;
                PyCoarseGrain::from_core(py, world)?.into_any()
            }
        })
    }

    /// :meth:`assemble`, and what its junction stage did.
    ///
    /// For an ``Atomistic`` library only: the junction stage brings the
    /// linked world's graph-derived terms up to its new bonds — the angles,
    /// dihedrals and pair rows of every kind the templates carry, the pairs a
    /// ring closure shortens reclassified and stripped to their ``n_bonds`` —
    /// and changes no atom type.
    ///
    /// Returns
    /// -------
    /// tuple[Atomistic, JunctionReport]
    ///
    /// Raises
    /// ------
    /// TypeError
    ///     If the library is not of ``Atomistic`` templates.
    /// ValueError
    ///     As :meth:`assemble`.
    fn assemble_with_report<'py>(
        &self,
        py: Python<'py>,
        sites: PyRef<'_, PyCoarseGrain>,
    ) -> PyResult<(Py<PyAny>, PyJunctionReport)> {
        let AnyAssembler::Atomistic(a) = &self.inner else {
            return Err(PyTypeError::new_err(
                "assemble_with_report needs a library of Atomistic templates",
            ));
        };
        let sites = sites.core().clone();
        let (world, report) = py
            .detach(|| a.assemble_with_report(&sites))
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let world = PyAtomistic::from_core(py, world)?.into_any();
        Ok((
            world.clone_ref(py),
            PyJunctionReport {
                world,
                inner: report,
            },
        ))
    }

    fn __repr__(&self) -> String {
        let class = match self.inner {
            AnyAssembler::MolGraph(_) => "MolGraph",
            AnyAssembler::Atomistic(_) => "Atomistic",
            AnyAssembler::CoarseGrain(_) => "CoarseGrain",
        };
        format!("Assembler(<{class} library>)")
    }
}

/// The graph a [`PyCoarsener`] holds: either public leaf that carries nodes
/// with positions and masses.
enum CoarsenSource {
    CoarseGrain(Py<PyCoarseGrain>),
    Atomistic(Py<PyAtomistic>),
}

impl CoarsenSource {
    fn object<'py>(&self, py: Python<'py>) -> &Bound<'py, PyAny> {
        match self {
            Self::CoarseGrain(cg) => cg.bind(py).as_any(),
            Self::Atomistic(mol) => mol.bind(py).as_any(),
        }
    }
}

/// Coarse-graining of a held source graph — `molpy.builder.Coarsener`.
///
/// Site ``I`` of the result stands for ``groups[I]``: it sits at the group's
/// mass-weighted centre (Å; no periodic imaging, so unwrap first), carries the
/// group's summed ``mass`` and ``bead_type = names[I]``, and records the
/// group's handles as its members. Two sites are bonded once when a source
/// bond joins their groups. The groups must be disjoint; ``SubgraphMatcher``
/// returns overlapping ones, so the caller selects first.
///
/// Parameters
/// ----------
/// source : CoarseGrain or Atomistic
///     The graph whose nodes are grouped. The object is held, not copied, and
///     read at each :meth:`coarsen` call.
///
/// Raises
/// ------
/// TypeError
///     If ``source`` is neither a :class:`~molpy.core.CoarseGrain` nor an
///     :class:`~molpy.core.Atomistic`.
///
/// Examples
/// --------
/// >>> groups = molpy.perceive.SubgraphMatcher(pattern).find(cg)
/// >>> sites = molpy.builder.Coarsener(cg).coarsen(groups, ["PMA"] * len(groups))
#[pyclass(module = "molpy.builder", name = "Coarsener", frozen)]
pub struct PyCoarsener {
    source: CoarsenSource,
}

#[pymethods]
impl PyCoarsener {
    #[new]
    fn new(source: &Bound<'_, PyAny>) -> PyResult<Self> {
        let source = if let Ok(cg) = source.cast::<PyCoarseGrain>() {
            CoarsenSource::CoarseGrain(cg.clone().unbind())
        } else if let Ok(mol) = source.cast::<PyAtomistic>() {
            CoarsenSource::Atomistic(mol.clone().unbind())
        } else {
            return Err(PyTypeError::new_err(format!(
                "Coarsener source must be a CoarseGrain or an Atomistic, not {}",
                source.get_type().name()?
            )));
        };
        Ok(Self { source })
    }

    /// A new :class:`~molpy.core.CoarseGrain` with one site per group.
    ///
    /// The GIL is released while mapping.
    ///
    /// Parameters
    /// ----------
    /// groups : Sequence[Iterable[Atom | Bead]]
    ///     Disjoint, non-empty node groups of the source (views or ``Refs``).
    /// names : Sequence[str]
    ///     One site ``bead_type`` per group.
    ///
    /// Returns
    /// -------
    /// CoarseGrain
    ///     The sites, in group order; empty when ``groups`` is empty. Each
    ///     site's members (:meth:`CoarseGrain.bead_members`) are its group's
    ///     nodes of the source.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If ``groups`` and ``names`` differ in length, a group is empty, a
    ///     node belongs to another graph or is listed twice, or a group has
    ///     no centre (a missing coordinate or mass, a non-positive total
    ///     mass); nodes are named by their handle.
    fn coarsen(
        &self,
        py: Python<'_>,
        groups: Vec<Bound<'_, PyAny>>,
        names: Vec<String>,
    ) -> PyResult<Py<PyCoarseGrain>> {
        let source = self.source.object(py);
        let leaf = Leaf::of(source)?;
        let groups: Vec<Vec<NodeId>> = groups
            .iter()
            .map(|group| leaf.own_nodes(group))
            .collect::<PyResult<_>>()?;
        let names: Vec<&str> = names.iter().map(String::as_str).collect();
        let sites = match &self.source {
            CoarsenSource::CoarseGrain(cg) => {
                let cg = cg.bind(py).try_borrow()?;
                let graph = cg.core();
                py.detach(|| Coarsener::new(graph).coarsen(&groups, &names))
            }
            CoarsenSource::Atomistic(mol) => {
                let mol = mol.bind(py).try_borrow()?;
                let graph = mol.core();
                py.detach(|| Coarsener::new(graph).coarsen(&groups, &names))
            }
        }
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        PyCoarseGrain::with_member_world(py, sites, Some(source.clone().unbind()))
    }

    fn __repr__(&self) -> String {
        let source = match self.source {
            CoarsenSource::CoarseGrain(_) => "CoarseGrain",
            CoarsenSource::Atomistic(_) => "Atomistic",
        };
        format!("Coarsener(<{source}>)")
    }
}

/// Register `molpy.builder`.
/// What the junction stage of an assembly did
/// (``molrs::builder::JunctionReport``), as views of the world it was made
/// for.
///
/// ``bonds[k]`` is the bond junction ``k`` made, ``anchors[k]`` its two
/// anchors, ``folds[k]`` the leaving-group charge (e) folded onto each anchor
/// (``None`` for a side that carried no charges).
/// ``reclassified_pairs`` are the pair rows whose bond distance the new bonds
/// shortened (a closed ring's former 1-4 pair); each now holds only its
/// ``n_bonds``.
#[pyclass(module = "molpy.builder", name = "JunctionReport", frozen)]
pub(crate) struct PyJunctionReport {
    world: Py<PyAny>,
    inner: JunctionReport,
}

#[pymethods]
impl PyJunctionReport {
    /// The new bonds, one per junction.
    #[getter]
    fn bonds(&self, py: Python<'_>) -> PyRefs {
        let handles = self
            .inner
            .junctions
            .iter()
            .map(|j| u64::from(j.bond))
            .collect();
        PyRefs::of_relations(self.world.bind(py), "bonds", handles)
    }

    /// The two anchors of each junction.
    #[getter]
    fn anchors<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Vec<(Bound<'py, PyAny>, Bound<'py, PyAny>)>> {
        let leaf = Leaf::of(self.world.bind(py))?;
        self.inner
            .junctions
            .iter()
            .map(|j| {
                Ok((
                    leaf.node(u64::from(j.anchors[0]))?,
                    leaf.node(u64::from(j.anchors[1]))?,
                ))
            })
            .collect()
    }

    /// The charge (e) folded onto each anchor of each junction.
    #[getter]
    fn folds(&self) -> Vec<(Option<f64>, Option<f64>)> {
        self.inner
            .junctions
            .iter()
            .map(|j| (j.folds[0], j.folds[1]))
            .collect()
    }

    /// Pair rows whose bond distance the new bonds shortened.
    #[getter]
    fn reclassified_pairs(&self, py: Python<'_>) -> PyRefs {
        let handles = self
            .inner
            .reclassified_pairs
            .iter()
            .map(|&id| u64::from(id))
            .collect();
        PyRefs::of_relations(self.world.bind(py), "pairs", handles)
    }

    /// Angles added through the new bonds.
    #[getter]
    fn n_added_angles(&self) -> usize {
        self.inner.added_angles
    }

    /// Dihedrals added through the new bonds.
    #[getter]
    fn n_added_dihedrals(&self) -> usize {
        self.inner.added_dihedrals
    }

    /// Pair rows added through the new bonds.
    #[getter]
    fn n_added_pairs(&self) -> usize {
        self.inner.added_pairs
    }

    fn __repr__(&self) -> String {
        format!(
            "JunctionReport(junctions={}, n_added_angles={}, n_added_dihedrals={}, \
             n_added_pairs={}, reclassified_pairs={})",
            self.inner.junctions.len(),
            self.inner.added_angles,
            self.inner.added_dihedrals,
            self.inner.added_pairs,
            self.inner.reclassified_pairs.len()
        )
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyCarbonTubeBuilder>()?;
    m.add_class::<PyGrapheneBuilder>()?;
    m.add_class::<PySitePlacer>()?;
    m.add_class::<PyGrowthPlacer>()?;
    m.add_class::<PyAxisOrienter>()?;
    m.add_class::<PyAssembler>()?;
    m.add_class::<PyJunctionReport>()?;
    m.add_class::<PyCoarsener>()?;
    crate::polymer::register(m)?;
    crate::symmetry::register(m)?;
    crate::lattice::register(m)?;
    crate::virtual_sites::register(m)
}
