//! Python bindings for `molrs::optimize` (`molrs.optimize`): the L-BFGS
//! minimizer ([`PyLbfgs`]) and the report it returns
//! ([`PyOptimizationReport`]).
//!
//! An optimizer minimizes a `molrs.ff.potential.Potentials` over a coordinate
//! vector; it neither builds the potentials nor owns a force field.

use crate::array::ArrayArgDyn;
use pyo3::prelude::*;

use molrs::ff::compile::PotentialCompiler;
use molrs::optimize::{LbfgsSettings, OptimizationReport, minimize_lbfgs, minimize_lbfgs_batch};

use crate::core::frame::PyFrame;
use crate::ff::ir;
use crate::ff::potential::{PotBacking, PyPotentials, potentials_moved_err};

use ndarray::{Array2, Array3};
use numpy::{PyArray2, PyArray3, ToPyArray};

/// Outcome of a minimization, exposed to Python as
/// `molrs.optimize.OptimizationReport`.
#[pyclass(module = "molrs.optimize", name = "OptimizationReport", subclass)]
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

/// L-BFGS geometry optimizer, exposed as `molrs.optimize.Lbfgs`.
///
/// Construct with potentials + knobs on ``new`` (defaults are Rust's
/// ``LbfgsSettings::DEFAULT``), then ``minimize`` a :class:`Frame` (primary) or
/// a coordinate array (single / batch by rank).
///
/// Examples
/// --------
/// >>> pots = molrs.ff.compile.PotentialCompiler(molrs.ff.typifier.Mmff94Typifier().forcefield()).compile(frame)
/// >>> opt = molrs.optimize.Lbfgs(pots, fmax=0.05, max_steps=500)
/// >>> frame, report = opt.minimize(frame)
/// >>> coords, report = opt.minimize(coords)         # (N, 3)
#[pyclass(module = "molrs.optimize", name = "Lbfgs", subclass)]
pub struct PyLbfgs {
    potentials: Py<PyPotentials>,
    settings: LbfgsSettings,
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

    /// Relax a :class:`Frame` or coordinates by L-BFGS.
    ///
    /// * ``Frame`` → ``(Frame, OptimizationReport)`` (a new Python frame
    ///   object is returned with the minimized coords).
    /// * ``(N, 3)`` / ``(3N,)`` → ``((N, 3) array, OptimizationReport)``
    /// * ``(B, N, 3)`` → ``((B, N, 3) array, list[OptimizationReport])``
    fn minimize<'py>(
        &self,
        py: Python<'py>,
        arg: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        // Frame path (primary).
        if let Ok(frame) = arg.extract::<PyRef<'_, PyFrame>>() {
            let mut core = frame.clone_core_frame()?;
            let pots = self.potentials.borrow(py);
            // Compile against this frame if deferred, then minimize with free mask.
            let compiled;
            let pot: &dyn molrs::ff::potential::Potential = match &pots.inner {
                PotBacking::Compiled(p) => p,
                PotBacking::Deferred(ff) => {
                    compiled = PotentialCompiler::new(ff)
                        .compile(&core)
                        .map_err(ir::compile_err)?;
                    &compiled
                }
                PotBacking::Moved => return Err(potentials_moved_err()),
            };
            // Borrowed one-shot on flat coords extracted from frame, then write back.
            let mut xyz = core
                .coords()
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
            // `Frame::coords` is a fresh row-major N×3 array: its buffer is the
            // flat `[x0, y0, z0, …]` the minimizer takes.
            let flat = xyz
                .as_slice_mut()
                .expect("Frame::coords returns a standard-layout array");
            let report = minimize_lbfgs(pot, flat, &self.settings)
                .map_err(pyo3::exceptions::PyValueError::new_err)?;
            crate::ff::potential::take_err(&pots.err_slots)?;
            core.set_coords(xyz.view())
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
            let out_frame = PyFrame::from_core_frame(core)?;
            return Ok((out_frame, PyOptimizationReport::from(report))
                .into_pyobject(py)?
                .into_any());
        }

        let pots = self.potentials.borrow(py);
        let pot = pots.inner.compiled()?;
        let readonly = arg.extract::<ArrayArgDyn<'_, f64>>()?;
        let arr = readonly.as_array();
        let shape = arr.shape();
        match shape.len() {
            1 | 2 => {
                let mut flat: Vec<f64> = arr.iter().copied().collect();
                let n_elem = flat.len();
                if !n_elem.is_multiple_of(3) {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "coords has {n_elem} elements, not a multiple of 3 (expected (N, 3) or (3N,))"
                    )));
                }
                let report = minimize_lbfgs(pot, &mut flat, &self.settings)
                    .map_err(pyo3::exceptions::PyValueError::new_err)?;
                crate::ff::potential::take_err(&pots.err_slots)?;
                let out: Bound<'py, PyArray2<f64>> = Array2::from_shape_vec((n_elem / 3, 3), flat)
                    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?
                    .to_pyarray(py);
                Ok((out, PyOptimizationReport::from(report))
                    .into_pyobject(py)?
                    .into_any())
            }
            3 => {
                if shape[2] != 3 {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "batch coords must be (B, N, 3); trailing axis is {} not 3",
                        shape[2]
                    )));
                }
                let (b, n) = (shape[0], shape[1]);
                let expected = pot.n_atoms();
                if expected != 0 && n != expected {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "structure atom count N={n} does not match this Potentials' atom count {expected}"
                    )));
                }
                let mut flat: Vec<f64> = arr.iter().copied().collect();
                let reports = minimize_lbfgs_batch(pot, &mut flat, n, b, &self.settings)
                    .map_err(pyo3::exceptions::PyValueError::new_err)?;
                crate::ff::potential::take_err(&pots.err_slots)?;
                let out: Bound<'py, PyArray3<f64>> = Array3::from_shape_vec((b, n, 3), flat)
                    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?
                    .to_pyarray(py);
                let reports: Vec<PyOptimizationReport> = reports
                    .into_iter()
                    .map(PyOptimizationReport::from)
                    .collect();
                Ok((out, reports).into_pyobject(py)?.into_any())
            }
            other => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "arg must be Frame, 1-D (3N,), 2-D (N, 3), or 3-D (B, N, 3); got {other}-D array"
            ))),
        }
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

/// Register `molrs.optimize`.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyOptimizationReport>()?;
    m.add_class::<PyLbfgs>()?;
    Ok(())
}
