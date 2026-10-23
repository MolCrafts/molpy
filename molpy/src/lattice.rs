//! `molpy.builder.Lattice` / `Site` — molrs `builder::{Lattice, Site}`.

use molrs::builder::{Lattice, LatticeError, Site};
use molrs::op::F;
use numpy::{IntoPyArray, PyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

use crate::core::frame::PyFrame;
use crate::core::molgraph::{PyAtomistic, props_to_py, py_props};
use crate::core::region::extract_region;
use crate::core::simbox::PyBox;
use crate::symmetry::PySpaceGroup;

fn lattice_err(e: LatticeError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// `repeats` as counts, every one positive.
fn counts(repeats: [i64; 3]) -> PyResult<[usize; 3]> {
    if repeats.iter().any(|&r| r < 1) {
        return Err(PyValueError::new_err(format!(
            "repeats must be three positive integers, got {repeats:?}"
        )));
    }
    Ok(repeats.map(|r| r as usize))
}

/// A lattice basis site in fractional coordinates of the cell.
///
/// Parameters
/// ----------
/// label
///     Site identifier (``"A"``, ``"B1"``); each built atom carries it as
///     ``label``.
/// species
///     Element symbol of the atom placed there.
/// fractional
///     Fractional coordinates ``(u, v, w)``.
/// charge
///     Partial charge.
/// attrs
///     Further per-atom fields (``bool`` / ``int`` / ``float`` / ``str``)
///     every built atom of this site carries; every site of a lattice
///     declares the same keys.
#[pyclass(module = "molpy.builder", name = "Site", frozen, eq, from_py_object)]
#[derive(Clone, PartialEq)]
pub struct PySite {
    pub(crate) inner: Site,
}

#[pymethods]
impl PySite {
    #[new]
    #[pyo3(signature = (label, species, fractional, charge=0.0, attrs=None))]
    fn new(
        label: String,
        species: String,
        fractional: [F; 3],
        charge: F,
        attrs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let mut inner = Site::new(label, species, fractional);
        inner.charge = charge;
        inner.attrs = py_props(attrs)?;
        Ok(Self { inner })
    }

    #[getter]
    fn label(&self) -> &str {
        &self.inner.label
    }

    #[getter]
    fn species(&self) -> &str {
        &self.inner.species
    }

    #[getter]
    fn fractional(&self) -> (F, F, F) {
        let [u, v, w] = self.inner.frac;
        (u, v, w)
    }

    #[getter]
    fn charge(&self) -> F {
        self.inner.charge
    }

    #[getter]
    fn attrs<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        props_to_py(py, &self.inner.attrs)
    }

    fn __repr__(&self) -> String {
        format!(
            "Site(label={:?}, species={:?}, fractional={:?})",
            self.inner.label, self.inner.species, self.inner.frac
        )
    }
}

/// A Bravais lattice: a :class:`~molpy.Box` unit cell and its basis
/// :class:`Site` objects (molrs ``Lattice``).
///
/// The box's matrix holds the lattice vectors as columns and owns fractional
/// ↔ Cartesian conversion. Construct directly with a box, or use
/// :meth:`from_vectors` / :meth:`from_cif` / :meth:`from_spacegroup` /
/// :meth:`sc` / :meth:`bcc` / :meth:`fcc` / :meth:`rocksalt`.
///
/// Raises
/// ------
/// ValueError
///     If ``box`` has no cell.
#[pyclass(
    module = "molpy.builder",
    name = "Lattice",
    frozen,
    skip_from_py_object
)]
pub struct PyLattice {
    inner: Lattice,
}

fn sites(basis: Option<Vec<PyRef<'_, PySite>>>) -> Vec<Site> {
    basis
        .unwrap_or_default()
        .iter()
        .map(|s| s.inner.clone())
        .collect()
}

#[pymethods]
impl PyLattice {
    #[new]
    #[pyo3(signature = (r#box, basis=None))]
    fn new(r#box: PyRef<'_, PyBox>, basis: Option<Vec<PyRef<'_, PySite>>>) -> PyResult<Self> {
        Ok(Self {
            inner: Lattice::new(r#box.inner.clone(), sites(basis)).map_err(lattice_err)?,
        })
    }

    /// A lattice from its three lattice vectors (the box's columns).
    #[staticmethod]
    #[pyo3(signature = (a1, a2, a3, basis=None))]
    fn from_vectors(
        a1: [F; 3],
        a2: [F; 3],
        a3: [F; 3],
        basis: Option<Vec<PyRef<'_, PySite>>>,
    ) -> PyResult<Self> {
        let h = std::array::from_fn(|row| [a1[row], a2[row], a3[row]]);
        let cell = molrs::core::SimBox::new(molrs::core::Cell::Matrix(h), [0.0; 3], [true; 3])
            .map_err(crate::core::simbox::box_error_to_pyerr)?;
        Ok(Self {
            inner: Lattice::new(cell, sites(basis)).map_err(lattice_err)?,
        })
    }

    /// The unit cell.
    #[getter(r#box)]
    fn cell(&self) -> PyBox {
        PyBox {
            inner: self.inner.cell().clone(),
        }
    }

    /// The basis sites, in order.
    #[getter]
    fn basis<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        let sites = self
            .inner
            .basis()
            .iter()
            .map(|s| PySite { inner: s.clone() })
            .collect::<Vec<_>>();
        PyTuple::new(py, sites)
    }

    /// First lattice vector (column 0 of ``box.matrix``).
    #[getter]
    fn a1<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<F>> {
        self.vector(py, 0)
    }

    /// Second lattice vector (column 1 of ``box.matrix``).
    #[getter]
    fn a2<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<F>> {
        self.vector(py, 1)
    }

    /// Third lattice vector (column 2 of ``box.matrix``).
    #[getter]
    fn a3<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<F>> {
        self.vector(py, 2)
    }

    /// A new lattice with ``site`` appended to the basis.
    fn with_site(&self, site: PyRef<'_, PySite>) -> Self {
        Self {
            inner: self.inner.with_site(site.inner.clone()),
        }
    }

    /// The box ``repeats`` unit cells span: each lattice vector scaled by its
    /// repeat, the origin and periodicity the unit cell's. Set it on the
    /// built structure's frame when it becomes a simulation.
    fn supercell(&self, repeats: [i64; 3]) -> PyResult<PyBox> {
        Ok(PyBox {
            inner: self
                .inner
                .supercell(counts(repeats)?)
                .map_err(lattice_err)?,
        })
    }

    /// Simple cubic: one atom per cell.
    #[staticmethod]
    fn sc(a: F, species: &str) -> PyResult<Self> {
        Ok(Self {
            inner: Lattice::sc(a, species).map_err(lattice_err)?,
        })
    }

    /// Body-centred cubic: two atoms per cell.
    #[staticmethod]
    fn bcc(a: F, species: &str) -> PyResult<Self> {
        Ok(Self {
            inner: Lattice::bcc(a, species).map_err(lattice_err)?,
        })
    }

    /// Face-centred cubic: four atoms per cell.
    #[staticmethod]
    fn fcc(a: F, species: &str) -> PyResult<Self> {
        Ok(Self {
            inner: Lattice::fcc(a, species).map_err(lattice_err)?,
        })
    }

    /// Rocksalt (NaCl): two interpenetrating face-centred cubic sublattices.
    #[staticmethod]
    fn rocksalt(a: F, species_a: &str, species_b: &str) -> PyResult<Self> {
        Ok(Self {
            inner: Lattice::rocksalt(a, species_a, species_b).map_err(lattice_err)?,
        })
    }

    /// Expand an asymmetric unit into the full cell: every symmetry image of
    /// every site in ``spacegroup``, images on a special position collapsed
    /// within ``symprec`` (fractional). Image ``k`` of site ``s`` is labelled
    /// ``"s_k"`` and keeps the site's species, charge and attrs.
    #[staticmethod]
    #[pyo3(signature = (r#box, sites, spacegroup, *, symprec=1e-5))]
    fn from_spacegroup(
        r#box: PyRef<'_, PyBox>,
        sites: Vec<PyRef<'_, PySite>>,
        spacegroup: PyRef<'_, PySpaceGroup>,
        symprec: F,
    ) -> PyResult<Self> {
        let sites: Vec<Site> = sites.iter().map(|s| s.inner.clone()).collect();
        Ok(Self {
            inner: Lattice::from_spacegroup(
                r#box.inner.clone(),
                &sites,
                &spacegroup.inner,
                symprec,
            )
            .map_err(lattice_err)?,
        })
    }

    /// The full-cell lattice of a frame from :func:`molpy.io.read_cif`: the
    /// cell is the frame's box, the asymmetric unit its ``atoms`` (label from
    /// ``name``, species from ``element``), the space group the operator loop
    /// kept in ``meta["symops"]``.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     A frame without a cell, operators, or the ``name`` / ``element`` /
    ///     coordinate columns; a partially occupied site (the lattice places
    ///     one atom per site, so resolve the disorder first).
    #[staticmethod]
    #[pyo3(signature = (frame, *, symprec=1e-5))]
    fn from_cif(frame: PyRef<'_, PyFrame>, symprec: F) -> PyResult<Self> {
        let lattice = frame.with_frame(|f| Lattice::from_cif(f, symprec))?;
        Ok(Self {
            inner: lattice.map_err(lattice_err)?,
        })
    }

    /// Tile the lattice and keep the atoms inside ``region``.
    ///
    /// ``repeats`` tiles cells ``0 <= (i, j, k) < repeats``; without it the
    /// tile is every cell — negative indices included — whose atoms can fall
    /// inside ``region.bounds()``. Each atom carries ``x``/``y``/``z``,
    /// ``element``, ``atomic_number``, ``charge``, ``label`` and its site's
    /// attrs. The tiled cell is :meth:`supercell`.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     Neither ``region`` nor ``repeats``, a repeat that is not positive,
    ///     an unbounded region without ``repeats``, sites declaring different
    ///     attrs, or a species that is no element.
    #[pyo3(signature = (region=None, *, repeats=None))]
    fn build(
        &self,
        py: Python<'_>,
        region: Option<&Bound<'_, PyAny>>,
        repeats: Option<[i64; 3]>,
    ) -> PyResult<Py<PyAtomistic>> {
        let region = region.map(extract_region).transpose()?;
        let repeats = repeats.map(counts).transpose()?;
        let built = self
            .inner
            .build(
                region.as_deref().map(|r| r as &dyn molrs::core::Region),
                repeats,
            )
            .map_err(lattice_err)?;
        PyAtomistic::from_core(py, built)
    }
}

impl PyLattice {
    fn vector<'py>(&self, py: Python<'py>, column: usize) -> Bound<'py, PyArray1<F>> {
        let h = self.inner.cell().matrix();
        vec![h[0][column], h[1][column], h[2][column]].into_pyarray(py)
    }
}

/// Register the lattice classes on `molpy.builder`.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySite>()?;
    m.add_class::<PyLattice>()?;
    Ok(())
}
