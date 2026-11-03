//! A Python object as a potential: the one ``molpy.ff.potential.Potential``
//! protocol, behind both consumers.
//!
//! - MD mounts one as a [`molrs::md::BoundPotential`] (``MD.set_evaluator``).
//!   The driver hands it the step, the rebuild decision of its own neighbour
//!   search, and the cell when it changed. A failure stops the run at its step.
//! - ``Potentials`` and ``Lbfgs`` take one as a member ([`PythonMember`]): a
//!   coordinates-in, energy-and-forces-out [`Potential`] that is told to
//!   rebuild on every call.
//!
//! The object supplies the exchange buffers once, at ``bind``. Every call
//! writes the positions into one and reads the forces out of the other, so no
//! buffer is reallocated per call and the buffers can be the pinned host
//! memory a device copy needs. One Python call per evaluation.

use std::sync::{Arc, Mutex};

use molrs::core::{Block, Frame, SimBox, UnitPreset, Virial};
use molrs::ff::potential::{Accumulator, Potential};
use molrs::op::F;
use ndarray::Array1;
use numpy::{PyArray1, PyArray2, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;

use super::potential::ErrSlot;
use crate::core::frame::PyFrame;
use crate::core::simbox::PyBox;
use crate::core::units::preset_arg;

/// The buffers a potential handed over at ``bind``.
struct Buffers {
    positions: Py<PyArray2<f64>>,
    forces: Py<PyArray2<f64>>,
    virial: Option<Py<PyArray1<f64>>>,
    n_atoms: usize,
}

/// A ``molpy.ff.potential.Potential``: ``units``, ``cutoff``,
/// ``bind(frame, skin)`` and ``compute(step, rebuild, box)``.
pub(crate) struct PythonPotential {
    obj: Py<PyAny>,
    name: String,
    units: UnitPreset,
    cutoff: Option<F>,
    buffers: Option<Buffers>,
}

impl PythonPotential {
    /// Read the declared ``units`` and ``cutoff`` of `obj` and check it has
    /// ``bind`` and ``compute``.
    pub(crate) fn new(obj: &Bound<'_, PyAny>) -> PyResult<Self> {
        let kind = obj.get_type().name()?;
        for method in ["bind", "compute"] {
            if !obj.hasattr(method)? || !obj.getattr(method)?.is_callable() {
                return Err(PyTypeError::new_err(format!(
                    "expected a potential: a native kernel (PairLjCut, Potentials, \
                     WeightedTerms) or a molpy.ff.potential.Potential, which has a \
                     callable {method}(); {kind} does not"
                )));
            }
        }
        let declared = obj.getattr("units").map_err(|_| {
            PyTypeError::new_err(
                "a potential declares the units it returns: set a `units` attribute \
                 (a preset name such as \"metal\", or a molpy.core.UnitPreset)",
            )
        })?;
        let cutoff = obj.getattr("cutoff").map_err(|_| {
            PyTypeError::new_err(
                "a potential declares the cutoff of its own neighbour search: set a \
                 `cutoff` attribute (a length, or None when it searches none)",
            )
        })?;
        let cutoff: Option<F> = cutoff
            .extract()
            .map_err(|_| PyTypeError::new_err("potential.cutoff must be a float or None"))?;
        if let Some(c) = cutoff
            && !(c > 0.0 && c.is_finite())
        {
            return Err(PyValueError::new_err(format!(
                "potential.cutoff must be a positive length, got {c}"
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

    pub(crate) fn name(&self) -> &str {
        &self.name
    }

    pub(crate) fn units(&self) -> &UnitPreset {
        &self.units
    }

    pub(crate) fn cutoff(&self) -> Option<F> {
        self.cutoff
    }

    /// The atom count of the last bind, or `None` before one.
    fn bound_atoms(&self) -> Option<usize> {
        self.buffers.as_ref().map(|b| b.n_atoms)
    }

    /// Call ``bind(frame, skin)`` on a snapshot of `frame` and take the
    /// buffers it returns.
    pub(crate) fn bind(&mut self, frame: &Frame, skin: F) -> PyResult<()> {
        let n = frame
            .get("atoms")
            .and_then(|atoms| atoms.n_rows())
            .ok_or_else(|| PyValueError::new_err("the frame has no atoms to bind"))?;
        let buffers = Python::attach(|py| -> PyResult<Buffers> {
            let snapshot = Py::new(py, PyFrame::from_core_frame(frame.clone())?)?;
            let out = self.obj.bind(py).call_method1("bind", (snapshot, skin))?;
            take_buffers(&out, n)
        })?;
        self.buffers = Some(buffers);
        Ok(())
    }

    /// Write `positions` into the bound buffer, call
    /// ``compute(step, rebuild, box)``, and read `forces` (and the virial)
    /// back.
    pub(crate) fn compute(
        &mut self,
        step: u64,
        rebuild: bool,
        cell: Option<&SimBox>,
        positions: &[F],
        forces: &mut [F],
    ) -> PyResult<(F, Option<Virial>)> {
        Python::attach(|py| {
            let buffers = self
                .buffers
                .as_ref()
                .ok_or_else(|| PyValueError::new_err("the potential was not bound"))?;
            {
                let array = buffers.positions.bind(py);
                let mut rw = array.try_readwrite()?;
                let dst = rw.as_slice_mut()?;
                still_sized(dst.len(), positions.len(), "positions")?;
                dst.copy_from_slice(positions);
            }
            let cell = cell.map(|simbox| PyBox {
                inner: simbox.clone(),
            });
            let energy = self
                .obj
                .bind(py)
                .call_method1("compute", (step, rebuild, cell))?;
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
            Ok((energy, virial))
        })
    }
}

fn take_buffers(out: &Bound<'_, PyAny>, n: usize) -> PyResult<Buffers> {
    let shape_err = || {
        PyTypeError::new_err(
            "Potential.bind must return (positions, forces) or (positions, forces, \
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
                "potential {what} buffer must be a float64 (N, 3) array"
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
                PyTypeError::new_err("potential virial buffer must be a float64 (6,) array")
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
                "potential buffers must not share memory: the caller writes the \
                 positions and reads the forces through them",
            ));
        }
    }
    Ok(Buffers {
        positions,
        forces,
        virial,
        n_atoms: n,
    })
}

/// A bound buffer still holds the `want` values it was bound with: one the
/// potential resized in place is refused, not indexed past its end.
fn still_sized(have: usize, want: usize, what: &str) -> PyResult<()> {
    if have != want {
        return Err(PyValueError::new_err(format!(
            "potential {what} buffer now holds {have} values, it was bound with {want}; \
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
            "potential {what} buffer has shape {shape:?}, the caller needs {want:?}"
        )));
    }
    if !contiguous {
        return Err(PyValueError::new_err(format!(
            "potential {what} buffer must be C-contiguous"
        )));
    }
    if !writeable {
        return Err(PyValueError::new_err(format!(
            "potential {what} buffer must be writeable"
        )));
    }
    Ok(())
}

/// The state of a [`PythonMember`], shared with the ``Potentials`` that holds
/// it so a caller with a frame can bind it there first.
pub(crate) struct MemberState {
    potential: PythonPotential,
    /// The cell of the last bind, handed to the first ``compute`` after it.
    pending_cell: Option<SimBox>,
    calls: u64,
}

impl MemberState {
    /// Bind to `frame` (skin 0: a member is told to rebuild on every call).
    pub(crate) fn bind(&mut self, frame: &Frame) -> PyResult<()> {
        self.potential.bind(frame, 0.0)?;
        self.pending_cell = frame.simbox.clone();
        Ok(())
    }

    fn evaluate(&mut self, coords: &[F]) -> PyResult<(F, Vec<F>)> {
        if self.potential.bound_atoms() != Some(coords.len() / 3) {
            self.bind(&coordinates_frame(coords)?)?;
        }
        let mut forces = vec![0.0; coords.len()];
        let cell = self.pending_cell.take();
        let step = self.calls;
        self.calls += 1;
        let (energy, _) = self
            .potential
            .compute(step, true, cell.as_ref(), coords, &mut forces)?;
        Ok((energy, forces))
    }
}

pub(crate) type SharedMember = Arc<Mutex<MemberState>>;

/// A Python potential as a member of a ``Potentials``: coordinates in, energy
/// and forces out, a neighbour rebuild on every call.
///
/// Evaluated from bare coordinates, it is bound to a frame of those
/// coordinates alone. A caller holding the frame (``Potentials.evaluate(frame)``,
/// ``Lbfgs.minimize``) binds it there first. The `Potential` trait has no error
/// channel, so an exception is parked in the member's [`ErrSlot`] and re-raised
/// by the caller that drove the evaluation.
pub(crate) struct PythonMember {
    state: SharedMember,
    error: ErrSlot,
}

impl PythonMember {
    /// The member, its shared state and its error slot.
    pub(crate) fn new(obj: &Bound<'_, PyAny>) -> PyResult<(Self, SharedMember, ErrSlot)> {
        let state = Arc::new(Mutex::new(MemberState {
            potential: PythonPotential::new(obj)?,
            pending_cell: None,
            calls: 0,
        }));
        let error: ErrSlot = Arc::default();
        let member = Self {
            state: Arc::clone(&state),
            error: Arc::clone(&error),
        };
        Ok((member, state, error))
    }
}

impl Potential for PythonMember {
    fn evaluate(&self, coords: &[F], acc: &mut Accumulator<'_>) {
        // `try_lock`: a potential whose `compute` evaluates the Potentials
        // that holds it would otherwise wait on itself forever.
        let out = match self.state.try_lock() {
            Ok(mut state) => state.evaluate(coords),
            Err(_) => Err(PyValueError::new_err(
                "a Python potential was evaluated again from inside its own compute",
            )),
        };
        let (energy, forces) = match out {
            Ok(out) => out,
            Err(err) => {
                *self.error.lock().expect("error slot poisoned") = Some(err);
                (F::NAN, vec![F::NAN; coords.len()])
            }
        };
        acc.add_energy(energy);
        // A member reports no virial here: adding through this door marks a
        // requested one lost.
        for (out, f) in acc.forces_without_virial().iter_mut().zip(forces) {
            *out += f;
        }
    }
}

/// A frame of `coords` (flat `[x0, y0, z0, …]`) and nothing else.
fn coordinates_frame(coords: &[F]) -> PyResult<Frame> {
    let n = coords.len() / 3;
    let mut atoms = Block::new();
    for (axis, key) in ["x", "y", "z"].into_iter().enumerate() {
        let column: Array1<F> = (0..n).map(|i| coords[3 * i + axis]).collect();
        atoms
            .insert(key, column.into_dyn(), None)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
    }
    let mut frame = Frame::new();
    frame.insert("atoms", atoms);
    Ok(frame)
}
