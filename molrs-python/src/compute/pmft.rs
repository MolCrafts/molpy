//! Potentials of mean force and torque (`molrs::compute::Pmft<C>`): one class
//! per registered pair coordinate system — `PmftR12`, `PmftXy`, `PmftXyt`,
//! `PmftXyz` — all returning a `PmftResult`.

use crate::array::{ArrayArg1, ArrayArg2};
use molrs::compute::{Compute, PairCoordinate, Pmft, PmftInput, R12, Xy, Xyt, Xyz};
use ndarray::{Array1, ArrayD};

use pyo3::prelude::*;

use super::error::Result;
use super::output::py_result;
use super::spec::described;
use crate::core::neighborlist::PyNeighbors;

/// A `PmftResult<C>` with its dimension erased, so the four coordinate
/// systems share one Python result class.
#[derive(Clone)]
pub(crate) struct PmftArrays {
    density: ArrayD<f64>,
    raw_counts: ArrayD<u64>,
    pmf: ArrayD<f64>,
    edges: Vec<Array1<f64>>,
}

impl<C: PairCoordinate> From<molrs::compute::PmftResult<C>> for PmftArrays {
    fn from(result: molrs::compute::PmftResult<C>) -> Self {
        Self {
            density: result.density.into_dyn(),
            raw_counts: result.raw_counts.into_dyn(),
            pmf: result.pmf.into_dyn(),
            edges: result.edges,
        }
    }
}

py_result! {
    /// A potential of mean force on a pair-coordinate grid
    /// (``molrs::compute::PmftResult``): pair ``density`` and ``raw_counts``
    /// per bin, ``pmf = −ln(ρ / ρ_ref)`` (``+inf`` where nothing was binned)
    /// and the bin ``edges`` of each axis.
    PyPmftResult(PmftArrays) as "PmftResult" { density, raw_counts, pmf, edges }
}

/// Declare the class of one `Pmft<C>` instantiation: its constructor
/// arguments and the array type of its orientations.
macro_rules! pmft_class {
    (
        $(#[$meta:meta])*
        $py:ident($coord:ty) as $name:literal,
        ($($arg:ident: $ty:ty),+),
        $orientations:ty
    ) => {
        $(#[$meta])*
        #[pyclass(module = "molrs.compute", name = $name, frozen)]
        pub(crate) struct $py {
            inner: Pmft<$coord>,
        }

        #[pymethods]
        impl $py {
            #[new]
            fn new($($arg: $ty),+) -> Result<Self> {
                Ok(Self {
                    inner: Pmft::<$coord>::new($($arg),+)?,
                })
            }

            #[pyo3(signature = (*, pairs, orientations = None))]
            fn compute(
                &self,
                pairs: PyRef<'_, PyNeighbors>,
                orientations: Option<$orientations>,
            ) -> Result<PyPmftResult> {
                let out = self.inner.compute(PmftInput {
                    pairs: &pairs.inner,
                    orientations: orientations.as_ref().map(|o| o.as_array()),
                })?;
                Ok(PmftArrays::from(out).into())
            }
        }

        described!($py, Pmft<$coord>);
    };
}

pmft_class! {
    /// PMFT in ``(r, θ1, θ2)`` of 2-D pairs (``molrs::compute::Pmft<R12>``).
    ///
    /// ``compute(pairs=, orientations=None)`` with ``(N,)`` angles.
    PyPmftR12(R12) as "PmftR12",
    (r_max: f64, n_r: usize, n_t1: usize, n_t2: usize),
    ArrayArg1<'_, f64>
}

pmft_class! {
    /// PMFT in the ``(x, y)`` frame of a 2-D particle
    /// (``molrs::compute::Pmft<Xy>``).
    ///
    /// ``compute(pairs=, orientations=None)`` with ``(N,)`` angles.
    PyPmftXy(Xy) as "PmftXy",
    (x_max: f64, y_max: f64, n_x: usize, n_y: usize),
    ArrayArg1<'_, f64>
}

pmft_class! {
    /// PMFT in ``(x, y, θ)`` of 2-D pairs (``molrs::compute::Pmft<Xyt>``).
    ///
    /// ``compute(pairs=, orientations=None)`` with ``(N,)`` angles.
    PyPmftXyt(Xyt) as "PmftXyt",
    (x_max: f64, y_max: f64, n_x: usize, n_y: usize, n_t: usize),
    ArrayArg1<'_, f64>
}

pmft_class! {
    /// PMFT in the ``(x, y, z)`` frame of a 3-D particle
    /// (``molrs::compute::Pmft<Xyz>``).
    ///
    /// ``compute(pairs=, orientations=None)`` with ``(N, 4)`` quaternions.
    PyPmftXyz(Xyz) as "PmftXyz",
    (x_max: f64, y_max: f64, z_max: f64, n_x: usize, n_y: usize, n_z: usize),
    ArrayArg2<'_, f64>
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyPmftResult>()?;
    m.add_class::<PyPmftR12>()?;
    m.add_class::<PyPmftXy>()?;
    m.add_class::<PyPmftXyt>()?;
    m.add_class::<PyPmftXyz>()?;
    Ok(())
}
