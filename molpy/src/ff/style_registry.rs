//! `molpy.ff.style_registry`: the force-field IR's style registry, from
//! Python.
//!
//! The IR (`molpy.ff.ir`) states what a category and a style are; the
//! registry (`molrs::ff::style_registry`) binds each style to the kernel
//! that prices it. [`PyGlobalRegistry`] is the process-wide registry every
//! compile reads, from Python, with nothing rebuilt:
//!
//! * `register_category` / `register_style` / `unregister_style` /
//!   `register_engine_form` — the registry calls, refusing what does not
//!   conform with an `IrError` subclass named after the Rust variant (D17);
//! * Python callables as Tier-2 kernels ([`PyScalarKernel`],
//!   [`PyCompoundKernel`]): one call per style per evaluation, an exception
//!   or a wrongly shaped result parked in the kernel error slot and
//!   re-raised by whatever drove the evaluation (`Potentials.calc_*`, a
//!   compile, an integrator);
//! * `styles` / `categories` / `evaluate` — introspection, and a style's
//!   form evaluated on a batch of coordinates (the core's `BatchForm`).
//!
//! The Python module's docs are `python/molpy/ff/style_registry.py`.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use ndarray::{Array3, Axis};
use numpy::{IntoPyArray, PyArray1, PyArrayDyn, PyArrayMethods, PyUntypedArrayMethods, ToPyArray};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

use molrs::ff::ir::{
    CategorySpec, Coordinate, Engine, IrError, ParamKind, SpecialClass, StyleSpec,
};
use molrs::ff::potential::form_kernel::{CompoundForm, ParamColumns, ScalarForm};
use molrs::ff::style_registry::{
    self, BatchCoordinates, BatchError, BatchForm, BatchGradient, BatchParamKind, BatchValue,
    Kernel,
};
use molrs::op::F;

use crate::ff::ir::engine::lammps_form;
use crate::ff::ir::{
    PyCategorySpec, PyStyleSpec, describe, parse_coordinate, parse_order, parse_params,
    parse_samples, py_shape, refuse,
};
use crate::ff::potential::ErrSlot;

// ---------------------------------------------------------------------------
// The kernel error slot
// ---------------------------------------------------------------------------

/// Where every Python kernel parks the first exception (or shape error) of
/// an evaluation: the `Potential` trait has no error channel, so the kernel
/// returns NaNs and the Python-facing caller re-raises from here.
fn kernel_slot() -> &'static ErrSlot {
    static SLOT: OnceLock<ErrSlot> = OnceLock::new();
    SLOT.get_or_init(ErrSlot::default)
}

/// The kernel error slot, for a caller's list of slots to check after an
/// evaluation (`Potentials`, the integrators).
pub(crate) fn kernel_err_slot() -> ErrSlot {
    Arc::clone(kernel_slot())
}

/// Re-raise the parked kernel exception, clearing the slot.
pub(crate) fn take_kernel_err() -> PyResult<()> {
    crate::ff::potential::take_err(std::slice::from_ref(kernel_slot()))
}

/// Forget a parked exception nobody re-raised (before a fresh compile or
/// registration, so a stale one is not blamed on it).
pub(crate) fn clear_kernel_err() {
    kernel_slot().lock().expect("error slot poisoned").take();
}

fn park(err: PyErr) {
    let mut slot = kernel_slot().lock().expect("error slot poisoned");
    if slot.is_none() {
        *slot = Some(err);
    }
}

// ---------------------------------------------------------------------------
// Python callables as Tier-2 kernels
// ---------------------------------------------------------------------------

/// The non-numeric inputs a Python kernel receives by name: array and text
/// per-type parameters, text style parameters. (Numeric ones are every
/// column the form kernel supplies.)
#[derive(Clone, Debug, Default)]
struct Extras {
    arrays: Vec<String>,
    texts: Vec<String>,
    style_texts: Vec<String>,
}

impl Extras {
    fn of(spec: &StyleSpec) -> Self {
        let mut out = Self::default();
        for p in &spec.params {
            match p.kind {
                ParamKind::Array { .. } => out.arrays.push(p.name.to_string()),
                ParamKind::Text { .. } => out.texts.push(p.name.to_string()),
                ParamKind::Scalar => {}
            }
        }
        for p in &spec.style_params {
            if let ParamKind::Text { .. } = p.kind {
                out.style_texts.push(p.name.to_string());
            }
        }
        out
    }

    /// The keyword arguments of one call: numeric columns as `ndarray[n]`,
    /// arrays `[n, S…]`, text per-type `list[str]`, text style `str`.
    fn kwargs<'py>(&self, py: Python<'py>, p: &ParamColumns<'_>) -> PyResult<Bound<'py, PyDict>> {
        let kw = PyDict::new(py);
        for name in p.names() {
            if let Some(col) = p.get(name) {
                kw.set_item(name, PyArray1::from_slice(py, col))?;
            }
        }
        for name in &self.arrays {
            if let Some(a) = p.array(name) {
                kw.set_item(name, a.to_pyarray(py))?;
            }
        }
        for name in &self.texts {
            if let Some(t) = p.text(name) {
                kw.set_item(name, PyList::new(py, t.iter().copied())?)?;
            }
        }
        for name in &self.style_texts {
            if let Some(t) = p.style_text(name) {
                kw.set_item(name, t)?;
            }
        }
        Ok(kw)
    }
}

/// `KernelShape` for `style`.
fn shape_err(style: &str, reason: String) -> PyErr {
    refuse(IrError::KernelShape {
        style: style.to_owned(),
        reason,
    })
}

/// A Python kernel's exception as `KernelShape`, the original its cause.
fn raised(py: Python<'_>, style: &str, err: PyErr) -> PyErr {
    let kind = err
        .get_type(py)
        .name()
        .map(|n| n.to_string())
        .unwrap_or_else(|_| "Exception".into());
    let wrapped = shape_err(
        style,
        format!("the Python kernel raised {kind}: {}", err.value(py)),
    );
    wrapped.set_cause(py, Some(err));
    wrapped
}

/// The two outputs of a kernel call, `(first, second)`.
fn two<'py>(
    style: &str,
    out: &Bound<'py, PyAny>,
    spelling: &str,
) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
    out.cast::<PyTuple>()
        .ok()
        .filter(|t| t.len() == 2)
        .map(|t| (t.get_item(0), t.get_item(1)))
        .and_then(|(a, b)| Some((a.ok()?, b.ok()?)))
        .ok_or_else(|| {
            shape_err(
                style,
                format!(
                    "the kernel must return a tuple {spelling}, got {}",
                    describe(out)
                ),
            )
        })
}

/// A kernel output `what`: a float64 ndarray of exactly `shape`.
fn output(style: &str, what: &str, obj: &Bound<'_, PyAny>, shape: &[usize]) -> PyResult<Vec<F>> {
    let wrong = || {
        shape_err(
            style,
            format!(
                "`{what}` must be a float64 ndarray of shape {}, got {}",
                py_shape(shape),
                describe(obj)
            ),
        )
    };
    let arr = obj.cast::<PyArrayDyn<F>>().map_err(|_| wrong())?;
    if arr.shape() != shape {
        return Err(wrong());
    }
    Ok(arr.readonly().as_array().iter().copied().collect())
}

/// A Python callable `f(q, **params) -> (e, de_dq)` as a [`ScalarForm`].
struct PyScalarKernel {
    func: Py<PyAny>,
    style: String,
    extras: Extras,
}

impl PyScalarKernel {
    fn call(&self, py: Python<'_>, q: &[F], p: &ParamColumns<'_>) -> PyResult<(Vec<F>, Vec<F>)> {
        let n = q.len();
        let kw = self.extras.kwargs(py, p)?;
        let out = self
            .func
            .bind(py)
            .call((PyArray1::from_slice(py, q),), Some(&kw))
            .map_err(|e| raised(py, &self.style, e))?;
        let (e, de) = two(&self.style, &out, "(e, de_dq)")?;
        Ok((
            output(&self.style, "e", &e, &[n])?,
            output(&self.style, "de_dq", &de, &[n])?,
        ))
    }
}

impl ScalarForm for PyScalarKernel {
    fn eval(&self, q: &[F], p: &ParamColumns<'_>, e: &mut [F], de_dq: &mut [F]) {
        Python::attach(|py| match self.call(py, q, p) {
            Ok((ev, dv)) => {
                e.copy_from_slice(&ev);
                de_dq.copy_from_slice(&dv);
            }
            Err(err) => {
                park(err);
                e.fill(F::NAN);
                de_dq.fill(F::NAN);
            }
        })
    }
}

/// A Python callable `f(x, **params) -> (e, grad)` as a [`CompoundForm`],
/// `x` and `grad` of shape `(n, arity, 3)`.
struct PyCompoundKernel {
    func: Py<PyAny>,
    style: String,
    extras: Extras,
}

impl PyCompoundKernel {
    fn call(
        &self,
        py: Python<'_>,
        x: &[[F; 3]],
        arity: usize,
        p: &ParamColumns<'_>,
    ) -> PyResult<(Vec<F>, Vec<F>)> {
        let n = x.len() / arity.max(1);
        let flat: Vec<F> = x.iter().flatten().copied().collect();
        let pos = Array3::from_shape_vec((n, arity, 3), flat)
            .expect("n_terms · arity points")
            .into_pyarray(py);
        let kw = self.extras.kwargs(py, p)?;
        let out = self
            .func
            .bind(py)
            .call((pos,), Some(&kw))
            .map_err(|e| raised(py, &self.style, e))?;
        let (e, grad) = two(&self.style, &out, "(e, grad)")?;
        Ok((
            output(&self.style, "e", &e, &[n])?,
            output(&self.style, "grad", &grad, &[n, arity, 3])?,
        ))
    }
}

impl CompoundForm for PyCompoundKernel {
    fn eval(
        &self,
        x: &[[F; 3]],
        arity: usize,
        p: &ParamColumns<'_>,
        e: &mut [F],
        grad: &mut [[F; 3]],
    ) {
        Python::attach(|py| match self.call(py, x, arity, p) {
            Ok((ev, gv)) => {
                e.copy_from_slice(&ev);
                let (points, _) = gv.as_chunks::<3>();
                grad.copy_from_slice(points);
            }
            Err(err) => {
                park(err);
                e.fill(F::NAN);
                grad.fill([F::NAN; 3]);
            }
        })
    }
}

/// The Python callables registered as kernels, by `(category, style)`: an
/// identical re-registration (the same callable, the same spec) reuses the
/// kernel, so the registry sees the same kernel and makes it a no-op.
type PyKernels = Mutex<HashMap<(String, String), (Py<PyAny>, Kernel)>>;

fn py_kernels() -> &'static PyKernels {
    static KERNELS: OnceLock<PyKernels> = OnceLock::new();
    KERNELS.get_or_init(Default::default)
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

/// The kernel a Python callable is registered as: reused when the same
/// callable is registered again for the style.
fn python_kernel(
    py: Python<'_>,
    spec: &StyleSpec,
    func: &Bound<'_, PyAny>,
    compound: bool,
    reuse: bool,
) -> PyResult<Kernel> {
    if !func.is_callable() {
        return Err(PyTypeError::new_err(format!(
            "kernel for {} `{}`: a callable, got {}",
            spec.category,
            spec.name,
            describe(func)
        )));
    }
    let key = (spec.category.to_string(), spec.name.to_string());
    if reuse {
        let kernels = py_kernels().lock().expect("kernel table poisoned");
        if let Some((known, kernel)) = kernels.get(&key)
            && matches!(kernel, Kernel::Compound(_)) == compound
            && known.bind(py).eq(func)?
        {
            return Ok(kernel.clone());
        }
    }
    let (func, style, extras) = (
        func.clone().unbind(),
        spec.name.to_string(),
        Extras::of(spec),
    );
    Ok(if compound {
        Kernel::Compound(Arc::new(PyCompoundKernel {
            func,
            style,
            extras,
        }))
    } else {
        Kernel::Scalar(Arc::new(PyScalarKernel {
            func,
            style,
            extras,
        }))
    })
}

// ---------------------------------------------------------------------------
// evaluate
// ---------------------------------------------------------------------------

/// `obj` as `n` numbers: a scalar broadcast, or a length-`n` sequence.
pub(crate) fn column(what: &str, obj: &Bound<'_, PyAny>, n: usize) -> PyResult<Vec<F>> {
    if let Ok(v) = obj.extract::<F>() {
        return Ok(vec![v; n]);
    }
    let col: Vec<F> = obj
        .extract()
        .map_err(|_| PyTypeError::new_err(format!("{what}: a number or a 1-D array of numbers")))?;
    if col.len() != n {
        return Err(PyValueError::new_err(format!(
            "{what}: {} values for {n} terms",
            col.len()
        )));
    }
    Ok(col)
}

/// The keyword parameters of an evaluation as the core's batch values, each
/// of the kind the style declares (an array of the declared rank is
/// broadcast to every term).
fn batch_params(
    py: Python<'_>,
    form: &BatchForm,
    who: &str,
    kwargs: Option<&Bound<'_, PyDict>>,
    n: usize,
) -> PyResult<Vec<(String, BatchValue)>> {
    let np = py.import("numpy")?;
    let mut out = Vec::new();
    for (k, v) in kwargs.into_iter().flat_map(|d| d.iter()) {
        let key: String = k.extract()?;
        let what = format!("{who}: `{key}`");
        let value = match form.param_kind(&key) {
            None => {
                return Err(PyTypeError::new_err(format!(
                    "{who} has no parameter `{key}`"
                )));
            }
            Some(BatchParamKind::Num) => BatchValue::Num(column(&what, &v, n)?),
            Some(BatchParamKind::StyleText) => BatchValue::StyleText(v.extract()?),
            Some(BatchParamKind::Text) => BatchValue::Text(match v.extract::<String>() {
                Ok(text) => vec![text; n],
                Err(_) => v.extract()?,
            }),
            Some(BatchParamKind::Array { rank }) => {
                let a = np
                    .call_method1("asarray", (v, "float64"))?
                    .cast_into::<PyArrayDyn<F>>()?
                    .readonly()
                    .as_array()
                    .to_owned();
                BatchValue::Array(if a.ndim() == rank as usize {
                    let views = vec![a.view(); n];
                    ndarray::stack(Axis(0), &views)
                        .map_err(|e| PyValueError::new_err(format!("{what}: {e}")))?
                } else {
                    a
                })
            }
        };
        out.push((key, value));
    }
    Ok(out)
}

/// A batch refusal as the Python error its kind is.
fn batch_err(error: BatchError) -> PyErr {
    match error {
        BatchError::Ir(e) => refuse(e),
        BatchError::Input(reason) => PyTypeError::new_err(reason),
        BatchError::Value(reason) => PyValueError::new_err(reason),
    }
}

/// The process-wide style registry every compile, reader and writer reads:
/// register a category, a style (an expression or a Python kernel) or a
/// style's engine form, with nothing rebuilt; list what is registered;
/// evaluate a style on a batch of terms. A handle that holds nothing.
///
/// Examples
/// --------
/// >>> registry = style_registry.GlobalRegistry()
/// >>> {s.kernel for s in registry.styles("bond") if s.name == "harmonic"}
/// {'scalar'}
#[pyclass(module = "molpy.ff.style_registry", name = "GlobalRegistry", frozen)]
pub struct PyGlobalRegistry;

#[pymethods]
impl PyGlobalRegistry {
    #[new]
    fn new() -> Self {
        Self
    }

    /// Register a new category of the force-field IR.
    ///
    /// Its terms name ``arity`` atoms and live in the Frame block
    /// ``f"{name}s"`` (molrec's rule, so a record alone locates them).
    ///
    /// Parameters
    /// ----------
    /// name : str
    ///     ``^[a-z][a-z0-9_]*$``.
    /// arity : int
    ///     Atoms per term, 2 to 5.
    /// coordinate : {"compound", "distance", "angle", "dihedral", "improper"}
    ///     What its energy is a function of: the atoms' positions (an expression
    ///     over ``p1 … pA``, or a compound kernel), or the geometric variable
    ///     its arity has (``distance`` 2, ``angle`` 3, ``dihedral`` /
    ///     ``improper`` 4).
    /// order : {"reversible", "ordered", "unordered"}
    ///     Which orders of a type row's endpoints name the same row.
    ///
    /// Raises
    /// ------
    /// BadName, Arity, CoordinateMismatch, Sealed, Conflict
    ///     A malformed name, an arity outside 2..=5, a coordinate the arity
    ///     cannot carry, a built-in's name, another category's name.
    ///
    /// Examples
    /// --------
    /// >>> style_registry.GlobalRegistry().register_category("urey_bradley", 3)
    /// >>> [c.block for c in style_registry.GlobalRegistry().categories() if c.name == "urey_bradley"]
    /// ['urey_bradleys']
    #[pyo3(signature = (name, arity, *, coordinate="compound", order="reversible"))]
    fn register_category(
        &self,
        name: String,
        arity: usize,
        coordinate: &str,
        order: &str,
    ) -> PyResult<()> {
        let coordinate = parse_coordinate(coordinate)?;
        let order = parse_order(order)?;
        let arity = u8::try_from(arity).map_err(|_| {
            refuse(IrError::Arity {
                category: name.clone(),
                arity,
            })
        })?;
        style_registry::GlobalRegistry
            .register_category(CategorySpec::custom(name, arity, coordinate, order))
            .map_err(refuse)
    }

    /// Register a new style of the force-field IR — priced by an expression, a
    /// Python kernel, or both (then they must agree).
    ///
    /// Every compile reads the registry, so the style prices at both compile
    /// doors (``compile``, ``compile_typed``) and in MD with nothing rebuilt.
    ///
    /// Parameters
    /// ----------
    /// category : str
    ///     A built-in category (``bond``, ``angle``, ``dihedral``, ``improper``,
    ///     ``pair``, ``cmap``, …) or one from :func:`register_category`.
    /// name : str
    ///     The style name.
    /// params : list of ParamSpec or dict of {name: dim}
    ///     The per-type parameters, **ordered** (the LAMMPS ``*_coeff`` order
    ///     when a LAMMPS style of the name exists).
    /// style_params : list of ParamSpec or dict of {name: dim}
    ///     Style-level parameters; ``cutoff`` (``L``), ``mixing`` and
    ///     ``special`` (text) keep their reserved meanings.
    /// expression : str, optional
    ///     The unweighted per-term energy, a Lepton expression over the
    ///     category's variable (``r``, ``theta``, ``phi``, ``chi``; points
    ///     ``p1 … pA`` in a compound category; ``q1``, ``q2``, ``x1``/``x2`` in
    ///     a pair) and the numeric parameters **as stored** (angle values in
    ///     degrees).
    /// kernel : callable, optional
    ///     A vectorised kernel, called once per style per evaluation. Scalar:
    ///     ``kernel(q, **params) -> (e, de_dq)``, ``q`` and every numeric
    ///     parameter an ``ndarray[n]``. Compound (``compound=True`` or a
    ///     ``"compound"`` category): ``kernel(x, **params) -> (e, grad)``, ``x``
    ///     and ``grad`` ``ndarray[n, arity, 3]``. Outputs are float64 arrays of
    ///     exactly those shapes (else :class:`KernelShapeError`); an exception it
    ///     raises is re-raised as :class:`KernelShapeError` with the exception as
    ///     ``__cause__``. A kernel must not call ``molpy.ff.ir`` itself.
    /// compound : bool, default False
    ///     ``kernel`` takes positions, not the category's coordinate.
    /// special : {"lj", "coul"}, optional
    ///     Pair styles: which special-bonds weights scale it (default ``lj``).
    /// samples : list of dict, optional
    ///     ``{"q": (lo, hi), <param>: value, …}``: the derivative (and
    ///     expression agreement) checks run on 16 seeded points of each at
    ///     registration. Without samples they run once per process at the
    ///     style's first compile.
    /// replace : bool, default False
    ///     Replace a style registered at run time under this name (a built-in
    ///     is :class:`SealedError` regardless). Without it, a different registration
    ///     under a taken name is :class:`ConflictError`, an identical one a no-op.
    /// lammps : {"positional", "positional:<name>"}, optional
    ///     The style's LAMMPS form: ``"positional"`` writes and reads it as
    ///     ``<category>_style <name>`` with ``params`` in order on the
    ///     ``*_coeff`` line, each converted by its dimension (``pair_style
    ///     <name> <cutoff>``, ``pair_modify mix <mixing>``); ``"positional:fene"``
    ///     under the LAMMPS style ``fene``. ``None`` (the default): LAMMPS
    ///     refuses it by name (:class:`NoEngineFormError`; the installed LAMMPS has
    ///     no LEPTON package for an expression). A positional form the spec
    ///     cannot have (a Text, Array or indexed parameter, a style parameter
    ///     other than ``cutoff`` / ``mixing``) is :class:`NoEngineFormError`.
    ///
    /// Raises
    /// ------
    /// IrError
    ///     The subclass naming what does not conform: :class:`UnknownCategoryError`,
    ///     :class:`BadNameError`, :class:`ReservedParamError`, :class:`DuplicateParamError`,
    ///     :class:`DimensionError`, :class:`ParseError`, :class:`UnboundVariableError`,
    ///     :class:`UnknownFunctionError`, :class:`FunctionArityError`, :class:`PointError`,
    ///     :class:`CoordinateMismatchError`, :class:`DerivativeError`, :class:`DisagreeError`,
    ///     :class:`AsymmetricError`, :class:`KernelShapeError`, :class:`SealedError`,
    ///     :class:`ConflictError`, :class:`NoKernelError`, :class:`MalformedError`.
    ///
    /// Examples
    /// --------
    /// LAMMPS ``bond_style fene``, by its expression:
    ///
    /// >>> style_registry.GlobalRegistry().register_style(
    /// ...     "bond", "fene",
    /// ...     params=[ir.ParamSpec("k", "E/L^2"), ir.ParamSpec("r0", "L"),
    /// ...             ir.ParamSpec("epsilon", "E"), ir.ParamSpec("sigma", "L")],
    /// ...     expression="-0.5*k*r0^2*log(1-(r/r0)^2)"
    /// ...                "+step(2^(1/6)*sigma-r)*(4*epsilon*((sigma/r)^12-(sigma/r)^6)+epsilon)",
    /// ... )
    #[pyo3(signature = (category, name, *, params=None, style_params=None, expression=None, kernel=None, compound=false, special=None, samples=None, replace=false, lammps=None))]
    #[allow(clippy::too_many_arguments)]
    fn register_style(
        &self,
        py: Python<'_>,
        category: String,
        name: String,
        params: Option<Bound<'_, PyAny>>,
        style_params: Option<Bound<'_, PyAny>>,
        expression: Option<String>,
        kernel: Option<Bound<'_, PyAny>>,
        compound: bool,
        special: Option<&str>,
        samples: Option<Bound<'_, PyAny>>,
        replace: bool,
        lammps: Option<&str>,
    ) -> PyResult<()> {
        let mut spec = StyleSpec::new(category.clone(), name.clone())
            .params(parse_params(params.as_ref())?)
            .style_params(parse_params(style_params.as_ref())?);
        spec.expression = expression;
        spec.samples = parse_samples(samples.as_ref())?;
        spec.lammps = lammps_form(lammps)?;
        spec.special = match special {
            None => None,
            Some("lj") => Some(SpecialClass::Vdw),
            Some("coul") => Some(SpecialClass::Coulomb),
            Some(other) => {
                return Err(PyValueError::new_err(format!(
                    "special={other:?}: 'lj' or 'coul'"
                )));
            }
        };
        let compound = compound
            || style_registry::GlobalRegistry.with(|r| {
                r.category(&category)
                    .is_some_and(|c| c.coordinate == Coordinate::Compound)
            });
        let kernel = kernel
            .filter(|k| !k.is_none())
            .map(|k| python_kernel(py, &spec, &k, compound, !replace).map(|kernel| (k, kernel)))
            .transpose()?;

        let key = (category.clone(), name.clone());
        // `replace=True` takes a run-time style out first, and puts it back if
        // the new one is refused.
        let previous = style_registry::GlobalRegistry.with(|r| {
            r.style(&category, &name)
                .filter(|_| !r.is_sealed(&category, &name))
                .map(|(s, k)| (s.clone(), k.cloned()))
        });
        let removed = match previous {
            Some(old) if replace => {
                style_registry::GlobalRegistry
                    .unregister_style(&category, &name)
                    .map_err(refuse)?;
                Some(old)
            }
            _ => None,
        };
        clear_kernel_err();
        let registered = style_registry::GlobalRegistry
            .register_style(spec, kernel.as_ref().map(|(_, k)| k.clone()));
        let outcome = match registered {
            Ok(()) => take_kernel_err().inspect_err(|_| {
                // The samples ran the kernel and it raised: take it out again.
                let _ = style_registry::GlobalRegistry.unregister_style(&category, &name);
            }),
            Err(e) => Err(refuse(e)),
        };
        if let Err(err) = outcome {
            if let Some((spec, kernel)) = removed {
                // It was registered before; it registers again.
                let _ = style_registry::GlobalRegistry.register_style(spec, kernel);
                clear_kernel_err();
            }
            return Err(err);
        }
        let mut kernels = py_kernels().lock().expect("kernel table poisoned");
        match kernel {
            Some((func, kernel)) => {
                kernels.insert(key, (func.unbind(), kernel));
            }
            None => {
                kernels.remove(&key);
            }
        }
        Ok(())
    }

    /// Remove a style registered at run time.
    ///
    /// Raises
    /// ------
    /// SealedError
    ///     A built-in style.
    /// NoKernelError
    ///     No style of that name is registered.
    fn unregister_style(&self, category: &str, name: &str) -> PyResult<()> {
        style_registry::GlobalRegistry
            .unregister_style(category, name)
            .map_err(refuse)?;
        py_kernels()
            .lock()
            .expect("kernel table poisoned")
            .remove(&(category.to_owned(), name.to_owned()));
        Ok(())
    }

    /// Give a registered style an engine form: today a LAMMPS form for a style
    /// registered without one.
    ///
    /// Parameters
    /// ----------
    /// engine : str
    ///     ``"lammps"`` (``"openmm"``, ``"gromacs"``, ``"prmtop"``,
    ///     ``"frcmod"`` take no registered form and raise ``NoEngineForm``: the
    ///     OpenMM XML writer writes an expression style's ``Custom*Force`` from
    ///     its expression, GROMACS and AMBER hold built-in styles only).
    /// category, name : str
    ///     The registered style.
    /// form : str
    ///     ``"positional"`` or ``"positional:<LAMMPS style name>"``.
    ///
    /// Raises
    /// ------
    /// NoKernelError
    ///     The style is not registered.
    /// NoEngineFormError
    ///     Another engine, or a positional form the style's spec cannot have (a
    ///     Text, Array or indexed parameter, a style parameter other than
    ///     ``cutoff`` / ``mixing``, a category without a ``*_style`` command).
    /// Sealed, Conflict
    ///     The style is built in, or has another form already.
    ///
    /// Examples
    /// --------
    /// >>> style_registry.GlobalRegistry().register_style("bond", "fene/doc", params={"k": "E/L^2", "r0": "L",
    /// ...     "epsilon": "E", "sigma": "L"},
    /// ...     expression="-0.5*k*r0^2*log(1-(r/r0)^2)")
    /// >>> style_registry.GlobalRegistry().register_engine_form("lammps", "bond", "fene/doc", "positional:fene")
    /// >>> [s.lammps for s in style_registry.GlobalRegistry().styles("bond") if s.name == "fene/doc"]
    /// ['positional:fene']
    /// >>> style_registry.GlobalRegistry().unregister_style("bond", "fene/doc")
    fn register_engine_form(
        &self,
        engine: &str,
        category: &str,
        name: &str,
        form: &str,
    ) -> PyResult<()> {
        let engine = Engine::parse(engine).map_err(PyValueError::new_err)?;
        let form = lammps_form(Some(form))?;
        style_registry::GlobalRegistry
            .register_engine_form(engine, category, name, form)
            .map_err(refuse)
    }

    /// Every registered style, or those of ``category``, sorted by
    /// ``(category, name)``.
    ///
    /// Examples
    /// --------
    /// >>> {s.kernel for s in style_registry.GlobalRegistry().styles("bond") if s.name == "harmonic"}
    /// {'scalar'}
    #[pyo3(signature = (category=None))]
    fn styles(&self, category: Option<&str>) -> Vec<PyStyleSpec> {
        style_registry::GlobalRegistry.with(|r| {
            r.styles(category)
                .map(|(s, k)| PyStyleSpec::of(r, s, k))
                .collect()
        })
    }

    /// Every registered category, sorted by name.
    fn categories(&self) -> Vec<PyCategorySpec> {
        let builtin: Vec<String> = molrs::ff::ir::builtin_categories()
            .into_iter()
            .map(|c| c.name.into_owned())
            .collect();
        style_registry::GlobalRegistry.with(|r| {
            r.categories()
                .map(|c| PyCategorySpec {
                    spec: c.clone(),
                    builtin: builtin.iter().any(|b| *b == c.name),
                })
                .collect()
        })
    }

    /// Evaluate a style's energy on a batch of terms, exactly as a compile
    /// would price them (unweighted, untruncated).
    ///
    /// Parameters
    /// ----------
    /// category, name : str
    ///     The style.
    /// q : array_like, shape (n,)
    ///     The coordinate per term, for a style priced by a scalar form: ``r``
    ///     (length), ``theta`` or ``phi`` (radians).
    /// x : array_like, shape (n, arity, 3), optional
    ///     The atoms' positions per term, for a style priced over positions (a
    ///     compound category, or an expression using ``distance``/``angle``/
    ///     ``dihedral``).
    /// **params
    ///     Each parameter **as stored** (angle values in degrees): a number
    ///     (broadcast) or one value per term; a text parameter a string; on a
    ///     pair ``q1``, ``q2`` and the self rows ``<name>1``, ``<name>2``
    ///     (default: the pair value). Missing ones take their default.
    ///
    /// Returns
    /// -------
    /// e : ndarray, shape (n,)
    ///     The energy per term.
    /// de_dq or grad : ndarray, shape (n,) or (n, arity, 3)
    ///     ``dE/dq``, or ``∂E/∂x`` (not the force).
    ///
    /// Raises
    /// ------
    /// MissingParamError
    ///     A parameter the form reads, not given and with no default.
    /// KernelShapeError
    ///     A Python kernel that raised or returned the wrong shape.
    /// TypeError
    ///     An undeclared parameter, or ``q``/``x`` not the form's input.
    ///
    /// Examples
    /// --------
    /// >>> e, de = style_registry.GlobalRegistry().evaluate("bond", "harmonic", [1.0, 1.5], k=300.0, r0=1.2)
    /// >>> e.tolist()
    /// [12.0, 27.0]
    #[pyo3(signature = (category, name, q=None, *, x=None, **params))]
    fn evaluate<'py>(
        &self,
        py: Python<'py>,
        category: &str,
        name: &str,
        q: Option<Bound<'py, PyAny>>,
        x: Option<Bound<'py, PyAny>>,
        params: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
        let who = format!("{category} `{name}`");
        let form = style_registry::GlobalRegistry
            .with(|r| r.batch_form(category, name))
            .map_err(batch_err)?;
        let np = py.import("numpy")?;
        let ascontiguous = |obj: Bound<'py, PyAny>| -> PyResult<Bound<'py, PyArrayDyn<F>>> {
            Ok(np
                .call_method1("ascontiguousarray", (obj, "float64"))?
                .cast_into::<PyArrayDyn<F>>()?)
        };
        let arity = form.arity();
        let (q, points, n) = match (q, x) {
            (Some(_), Some(_)) => {
                return Err(PyTypeError::new_err(format!(
                    "{who}: pass q or x, not both"
                )));
            }
            (Some(q), None) => {
                let q: Vec<F> = ascontiguous(q)?
                    .readonly()
                    .as_array()
                    .iter()
                    .copied()
                    .collect();
                let n = q.len();
                (Some(q), Vec::new(), n)
            }
            (None, Some(x)) => {
                let x = ascontiguous(x)?;
                let shape = x.shape().to_vec();
                if shape.len() != 3 || shape[1] != arity || shape[2] != 3 {
                    return Err(PyValueError::new_err(format!(
                        "{who}: x must have shape (n, {arity}, 3), got {shape:?}"
                    )));
                }
                let flat: Vec<F> = x.readonly().as_array().iter().copied().collect();
                (None, flat.as_chunks::<3>().0.to_vec(), shape[0])
            }
            (None, None) if form.is_compound() => {
                return Err(PyTypeError::new_err(format!(
                    "{who}: evaluate() needs x, the positions per term, shape (n, {arity}, 3)"
                )));
            }
            (None, None) => {
                return Err(PyTypeError::new_err(format!(
                    "{who}: evaluate() needs q, the coordinate per term"
                )));
            }
        };
        let coords = match &q {
            Some(q) => BatchCoordinates::Q(q),
            None => BatchCoordinates::X(&points),
        };
        let params = batch_params(py, &form, &who, params, n)?;
        clear_kernel_err();
        let result = form.evaluate(coords, params);
        take_kernel_err()?;
        let (e, gradient) = result.map_err(batch_err)?;
        let gradient = match gradient {
            BatchGradient::DeDq(de) => de.into_pyarray(py).into_any(),
            BatchGradient::Grad(grad) => {
                Array3::from_shape_vec((n, arity, 3), grad.into_iter().flatten().collect())
                    .expect("n · arity points")
                    .into_pyarray(py)
                    .into_any()
            }
        };
        Ok((e.into_pyarray(py).into_any(), gradient))
    }
}

/// Register `molpy.ff.style_registry` on `m`.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyGlobalRegistry>()?;
    Ok(())
}
