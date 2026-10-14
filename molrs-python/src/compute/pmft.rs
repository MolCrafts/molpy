//! Potentials of mean force and torque (`molrs::compute::pmft`): `PmftXy`.

#![allow(clippy::type_complexity)]

use super::{collect_frames, collect_neighbors};
use crate::error::py_value_err;
use molrs::compute::{LegacyCompute, PmftXy, PmftXyArgs, planar_orientation_angles};
use molrs::core::Frame as CoreFrame;
use molrs::op::F;
use numpy::{IntoPyArray, PyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyAny;

// ---------------------------------------------------------------------------
// PmftXy
// ---------------------------------------------------------------------------

#[pyclass(module = "molrs.compute", name = "PmftXy")]
pub struct PyPmftXy {
    inner: PmftXy,
}

#[pymethods]
impl PyPmftXy {
    #[new]
    fn new(x_max: f64, y_max: f64, n_x: usize, n_y: usize) -> PyResult<Self> {
        Ok(Self {
            inner: PmftXy::new(x_max, y_max, n_x, n_y).map_err(py_value_err)?,
        })
    }

    /// Returns per-frame `(raw_counts, density, pmf)`. If the first frame states
    /// per-particle orientations — the quaternion columns (`quatw`..`quatk`) on
    /// `atoms`, or an `"orientations"` topology block of `(head, tail)` atom
    /// pairs, one per query particle — every bond is rotated into that
    /// particle's local frame (molrs `compute::planar_orientation_angles`,
    /// per frame). Without one the analyzer works in the lab frame.
    fn compute<'py>(
        &self,
        py: Python<'py>,
        frames: &Bound<'py, PyAny>,
        nlists: &Bound<'py, PyAny>,
    ) -> PyResult<
        Vec<(
            Bound<'py, PyArray2<u64>>,
            Bound<'py, PyArray2<f64>>,
            Bound<'py, PyArray2<f64>>,
        )>,
    > {
        let owned = collect_frames(frames)?;
        let refs: Vec<&CoreFrame> = owned.iter().collect();
        let neighbors = collect_neighbors(nlists)?;
        let first = refs
            .first()
            .copied()
            .ok_or_else(|| PyValueError::new_err("no frames provided"))?;
        // Per-frame per-particle orientation angles (quaternion z-rotations or
        // `orientations` head–tail axes), or `None` (lab frame) when the first
        // frame states none.
        let orient_angles: Option<Vec<Vec<F>>> = if planar_orientation_angles(first)
            .map_err(py_value_err)?
            .is_some()
        {
            let per_frame = refs
                .iter()
                .map(|f| {
                    planar_orientation_angles(f)
                        .map_err(py_value_err)?
                        .ok_or_else(|| {
                            PyValueError::new_err(
                                "every frame must state the orientations the first one does",
                            )
                        })
                })
                .collect::<PyResult<Vec<_>>>()?;
            Some(per_frame)
        } else {
            None
        };
        let args = PmftXyArgs {
            nlists: &neighbors,
            query_orientations: orient_angles.as_deref(),
        };
        let results = self.inner.compute(&refs, args).map_err(py_value_err)?;
        Ok(results
            .into_iter()
            .map(|r| {
                (
                    r.raw_counts.into_pyarray(py),
                    r.density.into_pyarray(py),
                    r.pmf.into_pyarray(py),
                )
            })
            .collect())
    }
}

/// Register this domain's classes and functions.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyPmftXy>()?;
    Ok(())
}
