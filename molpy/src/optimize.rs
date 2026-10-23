//! Python bindings for `molrs::optimize` (`molpy.optimize`): the L-BFGS
//! minimizer ([`PyLbfgs`], the mirror of `molrs::optimize::Lbfgs`) and the
//! report it returns ([`PyOptimizationReport`]).
//!
//! An optimizer minimizes a `molpy.ff.potential.Potentials` over a coordinate
//! vector; it neither builds the potentials nor owns a force field. Like the
//! core object, each door relaxes its input **in place** and returns the
//! report(s): `minimize` a `Frame` (honouring its `atoms.free` mask),
//! `minimize_coords` one `(N, 3)` array, `minimize_batch` a `(B, N, 3)`
//! stack.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use molrs::ff::compile::PotentialCompiler;
use molrs::ff::potential::Potentials;
use molrs::optimize::{Lbfgs, LbfgsSettings, OptimizationReport, OptimizeError, Optimizer};

use crate::core::frame::PyFrame;
use crate::ff::ir;
use crate::ff::potential::{PotBacking, PyPotentials, potentials_moved_err, take_err};

use numpy::{PyArray2, PyArray3, PyArrayMethods, PyUntypedArrayMethods};

/// Outcome of a minimization, exposed to Python as
/// `molpy.optimize.OptimizationReport`.
#[pyclass(module = "molpy.optimize", name = "OptimizationReport", subclass)]
pub struct PyOptimizationReport {
    inner: OptimizationReport,
}

#[pymethods]
impl PyOptimizationReport {
    /// Whether ``fmax`` convergence was reached within ``max_steps``.
    #[getter]
    fn converged(&self) -> bool {
        self.inner.converged
    }

    /// Number of outer L-BFGS iterations performed.
    #[getter]
    fn n_steps(&self) -> usize {
        self.inner.n_steps
    }

    /// Potential energy at the returned geometry (kcal/mol).
    #[getter]
    fn final_energy(&self) -> f64 {
        self.inner.final_energy
    }

    /// Maximum per-atom force magnitude at the returned geometry
    /// (kcal/mol/angstrom).
    #[getter]
    fn final_fmax(&self) -> f64 {
        self.inner.final_fmax
    }

    /// Root-mean-square gradient component at the returned geometry
    /// (kcal/mol/angstrom).
    #[getter]
    fn final_grad_rms(&self) -> f64 {
        self.inner.final_grad_rms
    }

    fn __repr__(&self) -> String {
        format!(
            "OptimizationReport(converged={}, n_steps={}, final_energy={:.6}, final_fmax={:.6})",
            if self.inner.converged {
                "True"
            } else {
                "False"
            },
            self.inner.n_steps,
            self.inner.final_energy,
            self.inner.final_fmax
        )
    }
}

impl From<OptimizationReport> for PyOptimizationReport {
    fn from(inner: OptimizationReport) -> Self {
        Self { inner }
    }
}

/// L-BFGS geometry optimizer, exposed as `molpy.optimize.Lbfgs` — the mirror
/// of ``molrs::optimize::Lbfgs``.
///
/// Construct with potentials + knobs (defaults are Rust's
/// ``LbfgsSettings::DEFAULT``). Every door relaxes its input **in place** and
/// returns the report:
///
/// * :meth:`minimize` — a :class:`Frame`, honouring its ``atoms.free`` mask
///   (fixed atoms still enter the energy but do not move);
/// * :meth:`minimize_coords` — one ``(N, 3)`` float64 array, every atom free;
/// * :meth:`minimize_batch` — a ``(B, N, 3)`` stack, each structure
///   independently.
///
/// Examples
/// --------
/// >>> pots = molpy.ff.compile.PotentialCompiler(ff).compile(frame)
/// >>> opt = molpy.optimize.Lbfgs(pots, fmax=0.05, max_steps=500)
/// >>> report = opt.minimize(frame)                  # frame relaxed in place
/// >>> report = opt.minimize_coords(coords)          # (N, 3), in place
#[pyclass(module = "molpy.optimize", name = "Lbfgs", subclass)]
pub struct PyLbfgs {
    potentials: Py<PyPotentials>,
    settings: LbfgsSettings,
}

impl PyLbfgs {
    /// Run `f` on an optimizer over the compiled potentials, then re-raise a
    /// Python member's parked exception.
    fn with_compiled<R>(
        &self,
        py: Python<'_>,
        f: impl FnOnce(&Lbfgs<&Potentials>) -> Result<R, OptimizeError>,
    ) -> PyResult<R> {
        let pots = self.potentials.borrow(py);
        let out = f(&Lbfgs::new(pots.inner.compiled()?, self.settings));
        take_err(&pots.err_slots)?;
        out.map_err(optimize_err)
    }
}

/// A refused minimization, as a ``ValueError``.
fn optimize_err(err: OptimizeError) -> PyErr {
    PyValueError::new_err(err.to_string())
}

#[pymethods]
impl PyLbfgs {
    #[new]
    #[pyo3(signature = (
        potentials,
        *,
        fmax = LbfgsSettings::DEFAULT.fmax,
        max_steps = LbfgsSettings::DEFAULT.max_steps,
        max_step = LbfgsSettings::DEFAULT.max_step,
        memory = LbfgsSettings::DEFAULT.memory,
    ))]
    fn new(
        potentials: Py<PyPotentials>,
        fmax: f64,
        max_steps: usize,
        max_step: f64,
        memory: usize,
    ) -> Self {
        Self {
            potentials,
            settings: LbfgsSettings {
                fmax,
                max_steps,
                max_step,
                memory,
            },
        }
    }

    /// Relax ``frame``'s ``atoms`` coordinates in place
    /// (``Optimizer::minimize``).
    ///
    /// A bool ``atoms.free`` column marks the movable atoms: a fixed atom
    /// stays where it is but still enters every energy and force evaluation.
    /// Without the column every atom moves. A deferred :class:`Potentials`
    /// is bound to the frame's topology first.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     The frame has no coordinates, ``atoms.free`` is not one bool per
    ///     atom, or the potentials were moved.
    fn minimize(&self, py: Python<'_>, frame: &PyFrame) -> PyResult<PyOptimizationReport> {
        let pots = self.potentials.borrow(py);
        let report = frame.with_frame_mut(|core| -> PyResult<OptimizationReport> {
            let compiled;
            let pot: &Potentials = match &pots.inner {
                PotBacking::Compiled(p) => p,
                PotBacking::Deferred(ff) => {
                    compiled = PotentialCompiler::new(ff)
                        .compile(core)
                        .map_err(ir::compile_err)?;
                    &compiled
                }
                PotBacking::Moved => return Err(potentials_moved_err()),
            };
            Lbfgs::new(pot, self.settings)
                .minimize(core)
                .map_err(optimize_err)
        })??;
        take_err(&pots.err_slots)?;
        Ok(report.into())
    }

    /// Relax one structure's coordinates in place, every atom free
    /// (``Lbfgs::minimize_coords``).
    ///
    /// ``coords`` is a writeable C-contiguous float64 ``(N, 3)`` array.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     ``coords`` is not ``(N, 3)``, or is not writeable and C-contiguous
    ///     (pass ``np.ascontiguousarray(coords, dtype=np.float64)``).
    fn minimize_coords(
        &self,
        py: Python<'_>,
        coords: &Bound<'_, PyArray2<f64>>,
    ) -> PyResult<PyOptimizationReport> {
        if coords.shape()[1] != 3 {
            return Err(PyValueError::new_err(format!(
                "coords must be (N, 3), got {:?}",
                coords.shape()
            )));
        }
        let mut rw = coords
            .try_readwrite()
            .map_err(|e| PyValueError::new_err(format!("coords must be a writeable array: {e}")))?;
        let flat = rw.as_slice_mut().map_err(|_| contiguous_err())?;
        self.with_compiled(py, |opt| opt.minimize_coords(flat))
            .map(Into::into)
    }

    /// Relax a homogeneous batch in place — ``B`` structures of ``N`` atoms,
    /// each independently (``Lbfgs::minimize_batch``) — and return one report
    /// per structure.
    ///
    /// ``coords`` is a writeable C-contiguous float64 ``(B, N, 3)`` array.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     ``coords`` is not ``(B, N, 3)``, ``N`` differs from the
    ///     potentials' atom count, or ``coords`` is not writeable and
    ///     C-contiguous.
    fn minimize_batch(
        &self,
        py: Python<'_>,
        coords: &Bound<'_, PyArray3<f64>>,
    ) -> PyResult<Vec<PyOptimizationReport>> {
        let shape = coords.shape().to_vec();
        if shape[2] != 3 {
            return Err(PyValueError::new_err(format!(
                "batch coords must be (B, N, 3), got {shape:?}"
            )));
        }
        let (b, n) = (shape[0], shape[1]);
        let expected = self.potentials.borrow(py).inner.compiled()?.n_atoms();
        if expected != 0 && n != expected {
            return Err(PyValueError::new_err(format!(
                "structure atom count N={n} does not match this Potentials' atom count {expected}"
            )));
        }
        let mut rw = coords
            .try_readwrite()
            .map_err(|e| PyValueError::new_err(format!("coords must be a writeable array: {e}")))?;
        let flat = rw.as_slice_mut().map_err(|_| contiguous_err())?;
        let reports = self.with_compiled(py, |opt| opt.minimize_batch(flat, n, b))?;
        Ok(reports.into_iter().map(Into::into).collect())
    }

    fn __repr__(&self) -> String {
        format!(
            "Lbfgs(fmax={}, max_steps={}, max_step={}, memory={})",
            self.settings.fmax,
            self.settings.max_steps,
            self.settings.max_step,
            self.settings.memory
        )
    }
}

fn contiguous_err() -> PyErr {
    PyValueError::new_err(
        "coords must be C-contiguous float64: pass np.ascontiguousarray(coords, dtype=np.float64)",
    )
}

/// Register `molpy.optimize`.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyOptimizationReport>()?;
    m.add_class::<PyLbfgs>()?;
    Ok(())
}
