//! A Python object as a [`molrs::md::Calculator`]: the seam a torch model,
//! or any force provider of its own, plugs into.
//!
//! The object supplies the exchange buffers once, at ``bind``; every step the
//! core writes the positions into one and reads the forces back out of the
//! other, so no buffer is reallocated per step and the buffers can be the
//! pinned host memory a device copy needs. One Python call per step.

use molrs::core::{Frame, UnitPreset, Virial};
use molrs::md::{Calculator, CalculatorError, CalculatorResult, CalculatorStep};
use molrs::op::F;
use numpy::{PyArray1, PyArray2, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;

use crate::core::frame::PyFrame;
use crate::core::simbox::PyBox;
use crate::core::units::preset_arg;

/// The buffers a calculator handed over at ``bind``.
struct Buffers {
    positions: Py<PyArray2<f64>>,
    forces: Py<PyArray2<f64>>,
    virial: Option<Py<PyArray1<f64>>>,
}

/// A ``molpy.md.Calculator``: ``units``, ``cutoff``, ``bind(frame, skin)`` and
/// ``compute(step, rebuild, box)``.
pub(crate) struct PyCalculator {
    obj: Py<PyAny>,
    name: String,
    units: UnitPreset,
    cutoff: Option<F>,
    buffers: Option<Buffers>,
}

impl PyCalculator {
    /// Read the declared ``units`` and ``cutoff`` of `obj` and check it has
    /// ``bind`` and ``compute``.
    pub(crate) fn new(obj: &Bound<'_, PyAny>) -> PyResult<Self> {
        for method in ["bind", "compute"] {
            if !obj.hasattr(method)? || !obj.getattr(method)?.is_callable() {
                return Err(PyTypeError::new_err(format!(
                    "a calculator has a callable {method}(); {} does not",
                    obj.get_type().name()?
                )));
            }
        }
        let declared = obj.getattr("units").map_err(|_| {
            PyTypeError::new_err(
                "a calculator declares the units it returns: set a `units` attribute \
                 (a preset name such as \"metal\", or a molpy.core.UnitPreset)",
            )
        })?;
        let cutoff = obj.getattr("cutoff").map_err(|_| {
            PyTypeError::new_err(
                "a calculator declares the cutoff of its own neighbour search: set a \
                 `cutoff` attribute (a length, or None when it searches none)",
            )
        })?;
        let cutoff: Option<F> = cutoff
            .extract()
            .map_err(|_| PyTypeError::new_err("calculator.cutoff must be a float or None"))?;
        if let Some(c) = cutoff
            && !(c > 0.0 && c.is_finite())
        {
            return Err(PyValueError::new_err(format!(
                "calculator.cutoff must be a positive length, got {c}"
            )));
        }
        Ok(Self {
            obj: obj.clone().unbind(),
            name: obj.get_type().qualname()?.to_string(),
            units: preset_arg(&declared)?,
            cutoff,
            buffers: None,
        })
    }

    fn take_buffers(out: &Bound<'_, PyAny>, n: usize) -> PyResult<Buffers> {
        let shape_err = || {
            PyTypeError::new_err(
                "Calculator.bind must return (positions, forces) or (positions, forces, \
                 virial): float64 arrays of shape (N, 3), (N, 3) and (6,)",
            )
        };
        let items: Vec<Bound<'_, PyAny>> = out.extract().map_err(|_| shape_err())?;
        if !(2..=3).contains(&items.len()) {
            return Err(shape_err());
        }
        let nx3 = |item: &Bound<'_, PyAny>, what: &str| -> PyResult<Py<PyArray2<f64>>> {
            let array = item.cast::<PyArray2<f64>>().map_err(|_| {
                PyTypeError::new_err(format!(
                    "calculator {what} buffer must be a float64 (N, 3) array"
                ))
            })?;
            require_buffer(
                array.shape(),
                &[n, 3],
                array.is_c_contiguous(),
                array.try_readwrite().is_ok(),
                what,
            )?;
            Ok(array.clone().unbind())
        };
        let positions = nx3(&items[0], "positions")?;
        let forces = nx3(&items[1], "forces")?;
        let virial = items
            .get(2)
            .map(|item| -> PyResult<Py<PyArray1<f64>>> {
                let array = item.cast::<PyArray1<f64>>().map_err(|_| {
                    PyTypeError::new_err("calculator virial buffer must be a float64 (6,) array")
                })?;
                require_buffer(
                    array.shape(),
                    &[6],
                    array.is_c_contiguous(),
                    array.try_readwrite().is_ok(),
                    "virial",
                )?;
                Ok(array.clone().unbind())
            })
            .transpose()?;
        let span = |start: *mut f64, len: usize| (start as usize, start as usize + len * 8);
        let mut spans = vec![
            span(positions.bind(out.py()).data(), 3 * n),
            span(forces.bind(out.py()).data(), 3 * n),
        ];
        if let Some(w) = &virial {
            spans.push(span(w.bind(out.py()).data(), 6));
        }
        for (k, a) in spans.iter().enumerate() {
            if spans[k + 1..].iter().any(|b| a.0 < b.1 && b.0 < a.1) {
                return Err(PyValueError::new_err(
                    "calculator buffers must not share memory: the driver writes the \
                     positions and reads the forces through them",
                ));
            }
        }
        Ok(Buffers {
            positions,
            forces,
            virial,
        })
    }

    fn call(
        &mut self,
        py: Python<'_>,
        step: &CalculatorStep<'_>,
        positions: &[F],
        forces: &mut [F],
    ) -> PyResult<CalculatorResult> {
        let buffers = self
            .buffers
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("the calculator was not bound"))?;
        {
            let array = buffers.positions.bind(py);
            let mut rw = array.try_readwrite()?;
            let dst = rw.as_slice_mut()?;
            still_sized(dst.len(), positions.len(), "positions")?;
            dst.copy_from_slice(positions);
        }
        let cell = step.cell.map(|simbox| PyBox {
            inner: simbox.clone(),
        });
        let energy = self
            .obj
            .bind(py)
            .call_method1("compute", (step.step, step.rebuild, cell))?;
        let energy: F = energy.extract().map_err(|_| {
            PyTypeError::new_err(format!(
                "{}.compute must return the energy as a float, got {}",
                self.name,
                energy
                    .get_type()
                    .name()
                    .map_or_else(|_| "?".into(), |n| n.to_string())
            ))
        })?;
        let ro = buffers.forces.bind(py).try_readonly()?;
        let src = ro.as_slice()?;
        still_sized(src.len(), forces.len(), "forces")?;
        forces.copy_from_slice(src);
        let virial = match &buffers.virial {
            Some(array) => {
                let ro = array.bind(py).try_readonly()?;
                let w = ro.as_slice()?;
                still_sized(w.len(), 6, "virial")?;
                Some(Virial {
                    components: [w[0], w[1], w[2], w[3], w[4], w[5]],
                })
            }
            None => None,
        };
        Ok(CalculatorResult { energy, virial })
    }
}

/// A bound buffer still holds the `want` values it was bound with: one the
/// calculator resized in place is refused, not indexed past its end.
fn still_sized(have: usize, want: usize, what: &str) -> PyResult<()> {
    if have != want {
        return Err(PyValueError::new_err(format!(
            "calculator {what} buffer now holds {have} values, it was bound with {want}; \
             fill the buffers bind returned in place, never resize them"
        )));
    }
    Ok(())
}

fn require_buffer(
    shape: &[usize],
    want: &[usize],
    contiguous: bool,
    writeable: bool,
    what: &str,
) -> PyResult<()> {
    if shape != want {
        return Err(PyValueError::new_err(format!(
            "calculator {what} buffer has shape {shape:?}, the run needs {want:?}"
        )));
    }
    if !contiguous {
        return Err(PyValueError::new_err(format!(
            "calculator {what} buffer must be C-contiguous"
        )));
    }
    if !writeable {
        return Err(PyValueError::new_err(format!(
            "calculator {what} buffer must be writeable"
        )));
    }
    Ok(())
}

impl Calculator for PyCalculator {
    fn name(&self) -> &str {
        &self.name
    }

    fn units(&self) -> &UnitPreset {
        &self.units
    }

    fn cutoff(&self) -> Option<F> {
        self.cutoff
    }

    fn bind(&mut self, frame: &Frame, skin: F) -> Result<(), CalculatorError> {
        let n = frame
            .get("atoms")
            .and_then(|atoms| atoms.n_rows())
            .ok_or("the frame has no atoms to bind")?;
        let buffers = Python::attach(|py| -> PyResult<Buffers> {
            let snapshot = Py::new(py, PyFrame::from_core_frame(frame.clone())?)?;
            let out = self.obj.bind(py).call_method1("bind", (snapshot, skin))?;
            Self::take_buffers(&out, n)
        })?;
        self.buffers = Some(buffers);
        Ok(())
    }

    fn compute(
        &mut self,
        step: &CalculatorStep<'_>,
        positions: &[F],
        forces: &mut [F],
    ) -> Result<CalculatorResult, CalculatorError> {
        Ok(Python::attach(|py| self.call(py, step, positions, forces))?)
    }
}
