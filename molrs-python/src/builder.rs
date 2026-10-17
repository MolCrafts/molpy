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
    AssembleError, Assembler, AxisOrienter, GrowthPlacer, OrientError, Orienter, PlaceError,
    Placer, SitePlacer,
};
use molrs::builder::{CarbonTubeBuilder, CoarsenError, Coarsener, GrapheneBuilder, TubeExtent};
use molrs::core::{Atomistic, CoarseGrain, Graph, LinkManyError, MolGraph, NodeId};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyMapping;

use crate::core::frame::PyFrame;
use crate::core::graph_views::Leaf;
use crate::core::molgraph::{
    PyAtomistic, PyCoarseGrain, PyMolGraph, center_error_message, link_error_message,
};
use crate::core::simbox::PyBox;

/// Exact single-wall carbon nanotube builder.
#[pyclass(module = "molrs.builder", name = "CarbonTubeBuilder", subclass)]
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
#[pyclass(module = "molrs.builder", name = "GrapheneBuilder", subclass)]
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

/// The translation-only placer — ``molrs.builder.SitePlacer``.
///
/// Each copy's centre of mass (Å, weights ``mass``) lands on its site;
/// rotation is the orienter's job. Takes no arguments.
///
/// Examples
/// --------
/// >>> assembler = molrs.builder.Assembler(
/// ...     library, molrs.builder.SitePlacer(), molrs.builder.AxisOrienter()
/// ... )
#[pyclass(module = "molrs.builder", name = "SitePlacer", frozen)]
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

/// The growth placer — ``molrs.builder.GrowthPlacer``.
///
/// Grows each molecule copy by copy in the assembler's walk order: the first
/// copy keeps its template pose (its centre of mass on the site when the
/// site graph has positions), and every later copy is rotated and moved so
/// the anchor of its port toward its parent lands on the parent's leaving
/// handle, pointing back along that bond. Needs no site positions; bond
/// lengths and overlaps are left to a later minimisation. Takes no
/// arguments.
#[pyclass(module = "molrs.builder", name = "GrowthPlacer", frozen)]
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

/// The axis orienter — ``molrs.builder.AxisOrienter``.
///
/// Turns each copy about its template's centre of mass. A chain site (only
/// ``<`` / ``>`` ports, a two-port template) matches the template's
/// backbone-to-centre direction to the site axis (``CoarseGrain.axes``) and
/// its two joining atoms to the site's bond line. Any other bonded site fits
/// the template's port directions to its bond directions. A site with no
/// bond is not turned. Takes no arguments.
#[pyclass(module = "molrs.builder", name = "AxisOrienter", frozen)]
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

/// One placed, linked world from a site graph — ``molrs.builder.Assembler``.
///
/// Each bead of the site graph is one unit: a copy of
/// ``library[bead_type]``, turned by the orienter (when given) and given its
/// pose by the placer.
/// Each site bond joins one port of each end's copy (``<`` with ``>``, ``$``
/// with ``$``); the leaving groups are removed. Any topology works: chains,
/// branches, rings. Every atom gets ``frag_id`` (the site's ordinal) and
/// ``mol_id`` (its connected component's ordinal + 1).
///
/// Parameters
/// ----------
/// library : Mapping[str, MolGraph | Atomistic | CoarseGrain]
///     Name → template, copied at construction, every template of ONE graph
///     type — the type :meth:`assemble` builds the world as — with ports
///     where a site bonds. A template without ports can only fill an
///     unbonded site.
/// placer : SitePlacer | GrowthPlacer
///     ``SitePlacer`` moves each copy's centre of mass onto its site;
///     ``GrowthPlacer`` grows each molecule onto its parents' ports and needs
///     no site positions.
/// orienter : AxisOrienter, optional
///     Turns each copy about its centre of mass before it is placed; needs
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
/// >>> sites = molrs.builder.Coarsener(cg).coarsen(groups, names)
/// >>> world = molrs.builder.Assembler(
/// ...     {"PMA": pma, "Li": li},
/// ...     molrs.builder.SitePlacer(),
/// ...     molrs.builder.AxisOrienter(),
/// ... ).assemble(sites)
#[pyclass(module = "molrs.builder", name = "Assembler", frozen)]
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
        let refuse = |e| PyValueError::new_err(assemble_error_message(e));
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

    fn __repr__(&self) -> String {
        let class = match self.inner {
            AnyAssembler::MolGraph(_) => "MolGraph",
            AnyAssembler::Atomistic(_) => "Atomistic",
            AnyAssembler::CoarseGrain(_) => "CoarseGrain",
        };
        format!("Assembler(<{class} library>)")
    }
}

/// The message of an [`AssembleError`] as Python sees it: node and port ids
/// as int handles, never `NodeId(..)` / `RelationId(..)`. Every variant is
/// matched by name; the wording of the outer sentence is the core's.
fn assemble_error_message(e: AssembleError) -> String {
    match e {
        AssembleError::Place { name, site, source } => {
            let reason = match source {
                PlaceError::Template(center) => format!(
                    "the template has no centre of mass: {}",
                    center_error_message(center)
                ),
                other @ (PlaceError::NoPosition
                | PlaceError::NonFinitePoint
                | PlaceError::Port(_)) => other.to_string(),
            };
            format!("site {site} ('{name}') cannot be placed: {reason}")
        }
        AssembleError::Orient { name, site, source } => {
            let reason = match source {
                OrientError::Center(center) => format!(
                    "the template has no centre of mass: {}",
                    center_error_message(center)
                ),
                other @ (OrientError::Template(_)
                | OrientError::NoAxis { .. }
                | OrientError::Frame { .. }) => other.to_string(),
            };
            format!("site {site} ('{name}') cannot be oriented: {reason}")
        }
        AssembleError::Link {
            site,
            partner,
            source,
        } => {
            let reason = match source {
                LinkManyError::Pair { pair, source } => {
                    format!("pair {pair} is refused: {}", link_error_message(source))
                }
                batch @ (LinkManyError::PortReused { .. }
                | LinkManyError::DuplicateBond { .. }
                | LinkManyError::BranchesOverlap { .. }) => batch.to_string(),
            };
            format!("site {site} cannot join site {partner}: {reason}")
        }
        e @ (AssembleError::TooManyUnits { .. }
        | AssembleError::Output(_)
        | AssembleError::Sites(_)
        | AssembleError::UnknownName { .. }
        | AssembleError::Template { .. }
        | AssembleError::Ports { .. }
        | AssembleError::Replicate { .. }) => e.to_string(),
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

/// Coarse-graining of a held source graph — `molrs.builder.Coarsener`.
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
///     If ``source`` is neither a :class:`~molrs.core.CoarseGrain` nor an
///     :class:`~molrs.core.Atomistic`.
///
/// Examples
/// --------
/// >>> groups = molrs.perceive.SubgraphMatcher(pattern).find(cg)
/// >>> sites = molrs.builder.Coarsener(cg).coarsen(groups, ["PMA"] * len(groups))
#[pyclass(module = "molrs.builder", name = "Coarsener", frozen)]
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

    /// A new :class:`~molrs.core.CoarseGrain` with one site per group.
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
        .map_err(|e| PyValueError::new_err(coarsen_error_message(e)))?;
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

/// The message of a [`CoarsenError`] as Python sees it: node ids as int
/// handles, never `NodeId(..)`. Every variant is matched by name.
fn coarsen_error_message(e: CoarsenError) -> String {
    match e {
        CoarsenError::Center { group, source } => {
            format!("group {group}: {}", center_error_message(source))
        }
        e @ (CoarsenError::LengthMismatch { .. }
        | CoarsenError::EmptyGroup { .. }
        | CoarsenError::Overlap { .. }) => e.to_string(),
    }
}

/// Register `molrs.builder`.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyCarbonTubeBuilder>()?;
    m.add_class::<PyGrapheneBuilder>()?;
    m.add_class::<PySitePlacer>()?;
    m.add_class::<PyGrowthPlacer>()?;
    m.add_class::<PyAxisOrienter>()?;
    m.add_class::<PyAssembler>()?;
    m.add_class::<PyCoarsener>()?;
    Ok(())
}
