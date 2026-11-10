//! Python bindings for `molrs::builder`'s polymer ensembles (generative
//! BigSMILES): `Distribution`, `ChainSize`, `Fragment`, `Terminal`,
//! `StochasticObject`, `Molecule`, `Amount`, `System` and the records of a
//! system sample. Registered into `molpy.builder`.
//!
//! Every type wraps its molrs value; validation and sampling are molrs's.
//! Enums the user picks a variant of cross as Python complex enums
//! (`Distribution.FlorySchulz(a=0.1)`, `Amount.Percent(percent=20.0)`); the
//! measure crosses as its lowercase name, a port kind as its glyph and a bond
//! order as its number, as everywhere else on this seam.

use std::collections::HashMap;

use molrs::builder::{
    Amount, ChainSize, Distribution, Fragment, Measure, Molecule, MoleculeRecord, Part, SiteWeight,
    StochasticObject, System, Terminal,
};
use molrs::core::{BondNumber, PortKind};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::core::molgraph::{PyAtomistic, PyCoarseGrain};
use crate::error::polymer_error_to_pyerr as polymer_err;

// -- size ---------------------------------------------------------------------------

/// A chain-size distribution of generative BigSMILES, its parameters named
/// as the notation writes them (``|schulz_zimm(Mw, Mn)|`` is
/// ``Distribution.SchulzZimm(mw=…, mn=…)``). What a drawn value counts is
/// the :class:`ChainSize` measure it sits in.
#[pyclass(
    module = "molpy.builder",
    name = "Distribution",
    frozen,
    from_py_object,
    eq
)]
#[derive(Clone, PartialEq)]
pub enum PyDistribution {
    /// ``P(k) = a² k (1 − a)^(k − 1)``, ``0 < a < 1``.
    FlorySchulz { a: f64 },
    /// Gamma of mean ``mn`` and weight average ``mw``, ``mw > mn > 0``.
    SchulzZimm { mw: f64, mn: f64 },
    /// Normal, ``sigma ≥ 0``.
    Gauss { mu: f64, sigma: f64 },
    /// Uniform over ``[low, high]``, ``0 ≤ low ≤ high``.
    Uniform { low: f64, high: f64 },
    /// Log-normal of number average ``mn`` and dispersity ``d > 1``.
    LogNormal { mn: f64, d: f64 },
    /// Poisson of mean ``n > 0``.
    Poisson { n: f64 },
}

impl From<PyDistribution> for Distribution {
    fn from(d: PyDistribution) -> Self {
        match d {
            PyDistribution::FlorySchulz { a } => Distribution::FlorySchulz { a },
            PyDistribution::SchulzZimm { mw, mn } => Distribution::SchulzZimm { mw, mn },
            PyDistribution::Gauss { mu, sigma } => Distribution::Gauss { mu, sigma },
            PyDistribution::Uniform { low, high } => Distribution::Uniform { low, high },
            PyDistribution::LogNormal { mn, d } => Distribution::LogNormal { mn, d },
            PyDistribution::Poisson { n } => Distribution::Poisson { n },
        }
    }
}

impl From<Distribution> for PyDistribution {
    fn from(d: Distribution) -> Self {
        match d {
            Distribution::FlorySchulz { a } => PyDistribution::FlorySchulz { a },
            Distribution::SchulzZimm { mw, mn } => PyDistribution::SchulzZimm { mw, mn },
            Distribution::Gauss { mu, sigma } => PyDistribution::Gauss { mu, sigma },
            Distribution::Uniform { low, high } => PyDistribution::Uniform { low, high },
            Distribution::LogNormal { mn, d } => PyDistribution::LogNormal { mn, d },
            Distribution::Poisson { n } => PyDistribution::Poisson { n },
        }
    }
}

fn measure(name: &str) -> PyResult<Measure> {
    match name {
        "repeat_units" => Ok(Measure::RepeatUnits),
        "heavy_atom_mass" => Ok(Measure::HeavyAtomMass),
        "mass" => Ok(Measure::Mass),
        other => Err(PyValueError::new_err(format!(
            "measure must be 'repeat_units', 'heavy_atom_mass' or 'mass', not '{other}'"
        ))),
    }
}

fn measure_name(m: Measure) -> &'static str {
    match m {
        Measure::RepeatUnits => "repeat_units",
        Measure::HeavyAtomMass => "heavy_atom_mass",
        Measure::Mass => "mass",
    }
}

/// Size targets drawn from a :class:`Distribution` and counted in a
/// measure: ``"repeat_units"``, ``"heavy_atom_mass"`` (g/mol, the measure
/// generative BigSMILES writes) or ``"mass"`` (g/mol, hydrogens included).
#[pyclass(module = "molpy.builder", name = "ChainSize", frozen, from_py_object)]
#[derive(Clone)]
pub struct PyChainSize {
    inner: ChainSize,
}

#[pymethods]
impl PyChainSize {
    #[new]
    fn new(distribution: PyDistribution, measure: &str) -> PyResult<Self> {
        let inner =
            ChainSize::new(distribution.into(), self::measure(measure)?).map_err(polymer_err)?;
        Ok(Self { inner })
    }

    #[getter]
    fn distribution(&self) -> PyDistribution {
        self.inner.distribution().into()
    }

    #[getter]
    fn measure(&self) -> &'static str {
        measure_name(self.inner.measure())
    }
}

// -- fragments ----------------------------------------------------------------------------

/// The weight a port carries: a float (its reactivity) or a list of floats
/// (its pairing weight with each port of the stochastic object).
#[derive(FromPyObject)]
enum WeightArg {
    Single(f64),
    List(Vec<f64>),
}

/// One ported template under the name sites refer to it by, with a
/// generation weight per port (``None``: weight 1.0 on every port).
#[pyclass(module = "molpy.builder", name = "Fragment", frozen, from_py_object)]
#[derive(Clone)]
pub struct PyFragment {
    inner: Fragment,
}

#[pymethods]
impl PyFragment {
    #[new]
    #[pyo3(signature = (name, template, weights=None))]
    fn new(
        name: String,
        template: &PyAtomistic,
        weights: Option<Vec<WeightArg>>,
    ) -> PyResult<Self> {
        let template = template.core().clone();
        let weights = match weights {
            None => vec![SiteWeight::default(); template.n_ports()],
            Some(ws) => ws
                .into_iter()
                .map(|w| match w {
                    WeightArg::Single(x) => SiteWeight::Single(x),
                    WeightArg::List(xs) => SiteWeight::List(xs),
                })
                .collect(),
        };
        let inner = Fragment::new(name, template, weights).map_err(polymer_err)?;
        Ok(Self { inner })
    }

    #[getter]
    fn name(&self) -> &str {
        self.inner.name()
    }

    /// A copy of the ported template.
    #[getter]
    fn template(&self, py: Python<'_>) -> PyResult<Py<PyAtomistic>> {
        PyAtomistic::from_core(py, self.inner.template().clone())
    }

    #[getter]
    fn n_ports(&self) -> usize {
        self.inner.ports().len()
    }

    /// Summed non-hydrogen atomic mass, g/mol.
    #[getter]
    fn heavy_atom_mass(&self) -> f64 {
        self.inner.heavy_atom_mass()
    }

    /// Mass with every port bonded: heavy atoms plus kept hydrogens, g/mol.
    #[getter]
    fn mass(&self) -> f64 {
        self.inner.mass()
    }

    fn __repr__(&self) -> String {
        format!(
            "Fragment('{}', n_ports={})",
            self.inner.name(),
            self.inner.ports().len()
        )
    }
}

/// The descriptor at one end of a stochastic object, standing for the port
/// of the part on that side (``kind`` ``"$"``, ``"<"`` or ``">"``).
#[pyclass(module = "molpy.builder", name = "Terminal", frozen, from_py_object)]
#[derive(Clone)]
pub struct PyTerminal {
    inner: Terminal,
}

#[pymethods]
impl PyTerminal {
    #[new]
    #[pyo3(
        signature = (kind, label=String::new(), order=1, weights=None),
        text_signature = "(kind, label='', order=1, weights=None)"
    )]
    fn new(kind: &str, label: String, order: u32, weights: Option<Vec<f64>>) -> PyResult<Self> {
        let kind: PortKind = kind
            .parse()
            .map_err(|e: molrs::core::MolRsError| PyValueError::new_err(e.to_string()))?;
        let inner = Terminal::new(kind, label, BondNumber::from_code(order), weights)
            .map_err(polymer_err)?;
        Ok(Self { inner })
    }

    #[getter]
    fn kind(&self) -> &'static str {
        self.inner.kind().as_str()
    }

    #[getter]
    fn label(&self) -> &str {
        self.inner.label()
    }

    #[getter]
    fn order(&self) -> u32 {
        self.inner.order().code()
    }

    #[getter]
    fn weights(&self) -> Option<Vec<f64>> {
        self.inner.weights().map(<[f64]>::to_vec)
    }
}

/// Repeat units and end groups grown to a size drawn from ``size``,
/// entered through ``left`` and left through ``right``. Without a ``size``
/// it describes chemistry but cannot be sampled.
#[pyclass(
    module = "molpy.builder",
    name = "StochasticObject",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyStochasticObject {
    inner: StochasticObject,
}

#[pymethods]
impl PyStochasticObject {
    #[new]
    #[pyo3(signature = (repeat_units, end_groups=Vec::new(), *, left=None, right=None, size=None))]
    fn new(
        repeat_units: Vec<PyFragment>,
        end_groups: Vec<PyFragment>,
        left: Option<PyTerminal>,
        right: Option<PyTerminal>,
        size: Option<PyChainSize>,
    ) -> PyResult<Self> {
        let fragments = |us: Vec<PyFragment>| us.into_iter().map(|u| u.inner).collect();
        let inner = StochasticObject::new(
            fragments(repeat_units),
            fragments(end_groups),
            left.map(|t| t.inner),
            right.map(|t| t.inner),
            size.map(|s| s.inner),
        )
        .map_err(polymer_err)?;
        Ok(Self { inner })
    }

    #[getter]
    fn repeat_units(&self) -> Vec<PyFragment> {
        wrap_fragments(self.inner.repeat_units())
    }

    #[getter]
    fn end_groups(&self) -> Vec<PyFragment> {
        wrap_fragments(self.inner.end_groups())
    }

    #[getter]
    fn size(&self) -> Option<PyChainSize> {
        self.inner.size().map(|s| PyChainSize { inner: *s })
    }
}

fn wrap_fragments(fragments: &[Fragment]) -> Vec<PyFragment> {
    fragments
        .iter()
        .map(|u| PyFragment { inner: u.clone() })
        .collect()
}

// -- molecule -----------------------------------------------------------------------

/// A piece of a molecule: a :class:`Fragment` placed once, or a
/// :class:`StochasticObject`.
#[derive(FromPyObject)]
enum PartArg {
    Fixed(PyFragment),
    Stochastic(PyStochasticObject),
}

/// Fixed fragments and stochastic objects joined end to end.
#[pyclass(module = "molpy.builder", name = "Molecule", frozen, from_py_object)]
#[derive(Clone)]
pub struct PyMolecule {
    pub(crate) inner: Molecule,
}

#[pymethods]
impl PyMolecule {
    #[new]
    fn new(parts: Vec<PartArg>) -> PyResult<Self> {
        let parts = parts
            .into_iter()
            .map(|p| match p {
                PartArg::Fixed(u) => Part::Fixed(u.inner),
                PartArg::Stochastic(o) => Part::Stochastic(o.inner),
            })
            .collect();
        Ok(Self {
            inner: Molecule::new(parts).map_err(polymer_err)?,
        })
    }

    /// The parts, in written order (``Fragment`` or ``StochasticObject``).
    #[getter]
    fn parts(&self, py: Python<'_>) -> PyResult<Vec<Py<PyAny>>> {
        self.inner
            .parts()
            .iter()
            .map(|p| match p {
                Part::Fixed(u) => Ok(Py::new(py, PyFragment { inner: u.clone() })?.into_any()),
                Part::Stochastic(o) => {
                    Ok(Py::new(py, PyStochasticObject { inner: o.clone() })?.into_any())
                }
            })
            .collect()
    }

    /// Every fragment the molecule may place, by name — the ``Assembler``
    /// library (add hydrogens and coordinates first).
    fn library(&self, py: Python<'_>) -> PyResult<HashMap<String, Py<PyAtomistic>>> {
        library(py, self.inner.library())
    }

    /// One sample as a site graph (``bead_type`` the fragment name, a bond per
    /// join with its two port ordinals). Deterministic in ``seed``.
    fn sample(&self, py: Python<'_>, seed: u64) -> PyResult<Py<PyCoarseGrain>> {
        let sites = self.inner.sample(seed).map_err(polymer_err)?;
        PyCoarseGrain::from_core(py, sites)
    }
}

fn library(
    py: Python<'_>,
    templates: HashMap<String, molrs::core::Atomistic>,
) -> PyResult<HashMap<String, Py<PyAtomistic>>> {
    templates
        .into_iter()
        .map(|(name, t)| Ok((name, PyAtomistic::from_core(py, t)?)))
        .collect()
}

// -- system ---------------------------------------------------------------------------

/// How much of a system one molecule kind makes up, in the system's
/// measure: ``Amount.Total(value=…)`` or ``Amount.Percent(percent=…)`` of the
/// whole.
#[pyclass(module = "molpy.builder", name = "Amount", frozen, from_py_object, eq)]
#[derive(Clone, PartialEq)]
pub enum PyAmount {
    /// An absolute amount, ``> 0``.
    Total { value: f64 },
    /// A share of the whole system, in percent, ``0 < percent < 100``.
    Percent { percent: f64 },
}

impl From<PyAmount> for Amount {
    fn from(a: PyAmount) -> Self {
        match a {
            PyAmount::Total { value } => Amount::Total(value),
            PyAmount::Percent { percent } => Amount::Percent(percent),
        }
    }
}

impl From<Amount> for PyAmount {
    fn from(a: Amount) -> Self {
        match a {
            Amount::Total(value) => PyAmount::Total { value },
            Amount::Percent(percent) => PyAmount::Percent { percent },
        }
    }
}

/// What one sampled molecule is: its component and its size.
#[pyclass(
    module = "molpy.builder",
    name = "MoleculeRecord",
    frozen,
    skip_from_py_object,
    get_all
)]
#[derive(Clone)]
pub struct PyMoleculeRecord {
    /// Index of the component it was sampled from.
    component: usize,
    /// Fragments placed — its sites, in site order.
    n_fragments: usize,
    /// How many of them are repeat units.
    n_repeat_units: usize,
    /// Summed non-hydrogen atomic mass, g/mol.
    heavy_atom_mass: f64,
    /// Summed mass with kept hydrogens, g/mol.
    mass: f64,
}

impl From<&MoleculeRecord> for PyMoleculeRecord {
    fn from(r: &MoleculeRecord) -> Self {
        Self {
            component: r.component,
            n_fragments: r.n_fragments,
            n_repeat_units: r.n_repeat_units,
            heavy_atom_mass: r.heavy_atom_mass,
            mass: r.mass,
        }
    }
}

/// One sample of a system: every molecule's sites in one site graph and a
/// record per molecule, in site order.
#[pyclass(module = "molpy.builder", name = "SystemSample", frozen, get_all)]
pub struct PySystemSample {
    sites: Py<PyCoarseGrain>,
    molecules: Vec<PyMoleculeRecord>,
}

/// Molecule kinds sampled until each reaches its :class:`Amount` of the
/// total, counted in ``measure``.
#[pyclass(module = "molpy.builder", name = "System", frozen)]
pub struct PySystem {
    pub(crate) inner: System,
}

#[pymethods]
impl PySystem {
    #[new]
    #[pyo3(signature = (components, *, measure))]
    fn new(components: Vec<(PyMolecule, PyAmount)>, measure: &str) -> PyResult<Self> {
        let components = components
            .into_iter()
            .map(|(m, a)| (m.inner, a.into()))
            .collect();
        let inner = System::new(components, self::measure(measure)?).map_err(polymer_err)?;
        Ok(Self { inner })
    }

    /// The molecule kinds and their amounts.
    #[getter]
    fn components(&self) -> Vec<(PyMolecule, PyAmount)> {
        self.inner
            .components()
            .iter()
            .map(|(m, a)| (PyMolecule { inner: m.clone() }, (*a).into()))
            .collect()
    }

    #[getter]
    fn measure(&self) -> &'static str {
        measure_name(self.inner.measure())
    }

    /// Each component's target, in the system's measure.
    fn targets(&self) -> Vec<f64> {
        self.inner.targets()
    }

    /// Every fragment any molecule may place, by name.
    fn library(&self, py: Python<'_>) -> PyResult<HashMap<String, Py<PyAtomistic>>> {
        library(py, self.inner.library())
    }

    /// Molecules of each component until it reaches its target.
    /// Deterministic in ``seed``.
    fn sample(&self, py: Python<'_>, seed: u64) -> PyResult<PySystemSample> {
        let sample = self.inner.sample(seed).map_err(polymer_err)?;
        Ok(PySystemSample {
            sites: PyCoarseGrain::from_core(py, sample.sites)?,
            molecules: sample.molecules.iter().map(Into::into).collect(),
        })
    }
}

/// Register the polymer types into `molpy.builder`.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add(
        "PolymerError",
        m.py().get_type::<crate::error::PolymerError>(),
    )?;
    m.add_class::<PyDistribution>()?;
    m.add_class::<PyChainSize>()?;
    m.add_class::<PyFragment>()?;
    m.add_class::<PyTerminal>()?;
    m.add_class::<PyStochasticObject>()?;
    m.add_class::<PyMolecule>()?;
    m.add_class::<PyAmount>()?;
    m.add_class::<PyMoleculeRecord>()?;
    m.add_class::<PySystemSample>()?;
    m.add_class::<PySystem>()
}
