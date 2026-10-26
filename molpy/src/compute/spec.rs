//! The core compute descriptors, read-only: `ComputeSpec` (`all()`,
//! `find(name)`), `ParamSpec` and `InputSpec`, plus [`described!`], which
//! gives every compute class `spec()` and `tunables()`.
//!
//! These are what the binding's parity tests read: every class is named after
//! a spec, its `__init__` arguments are the spec's parameters, its `compute`
//! keywords are the spec's inputs, and on an instance built from the required
//! arguments alone `tunables()` equals the spec's defaults.

use molrs::compute::{
    ComputeSpec, InputForm, InputKind, InputSpec, ParamDefault, ParamKind, ParamSpec, ParamValue,
};
use pyo3::IntoPyObjectExt;
use pyo3::prelude::*;

/// A parameter value as Python sees it: a float, int, bool, name, list,
/// ``(start, end)`` tuple, or ``None`` for an option that is off.
pub(crate) fn param_value<'py>(py: Python<'py>, value: &ParamValue) -> PyResult<Bound<'py, PyAny>> {
    match value {
        ParamValue::Float(v) => v.into_bound_py_any(py),
        ParamValue::Int(v) => v.into_bound_py_any(py),
        ParamValue::Bool(v) => v.into_bound_py_any(py),
        ParamValue::Name(v) => v.into_bound_py_any(py),
        ParamValue::FloatList(v) => v.clone().into_bound_py_any(py),
        ParamValue::IntList(v) => v.clone().into_bound_py_any(py),
        ParamValue::Range(start, end) => (*start, *end).into_bound_py_any(py),
        ParamValue::Unset => Ok(py.None().into_bound(py)),
    }
}

/// `tunables()` of a core instance as ``[(name, value), …]``.
pub(crate) fn tunables<'py>(
    py: Python<'py>,
    values: Vec<(&'static str, ParamValue)>,
) -> PyResult<Vec<(&'static str, Bound<'py, PyAny>)>> {
    values
        .into_iter()
        .map(|(name, value)| Ok((name, param_value(py, &value)?)))
        .collect()
}

/// The descriptor of one compute type (``molrs::compute::ComputeSpec``).
#[pyclass(module = "molpy.compute", name = "ComputeSpec", frozen)]
pub(crate) struct PyComputeSpec {
    pub(crate) inner: ComputeSpec,
}

#[pymethods]
impl PyComputeSpec {
    /// The descriptor of every compute type in this build.
    #[staticmethod]
    fn all() -> Vec<Self> {
        ComputeSpec::all()
            .iter()
            .map(|inner| Self {
                inner: inner.clone(),
            })
            .collect()
    }

    /// The descriptor named ``name``, or ``None``.
    #[staticmethod]
    fn find(name: &str) -> Option<Self> {
        ComputeSpec::find(name).map(|inner| Self {
            inner: inner.clone(),
        })
    }

    /// The class name.
    #[getter]
    fn name(&self) -> &'static str {
        self.inner.name
    }

    /// The configuration parameters, in constructor-then-setter order.
    #[getter]
    fn params(&self) -> Vec<PyParamSpec> {
        self.inner
            .params
            .iter()
            .map(|inner| PyParamSpec {
                inner: inner.clone(),
            })
            .collect()
    }

    /// The fields of the input — the keyword arguments of ``compute``.
    #[getter]
    fn inputs(&self) -> Vec<PyInputSpec> {
        self.inner
            .inputs
            .iter()
            .map(|&inner| PyInputSpec { inner })
            .collect()
    }

    /// ``"single"`` (one input per call) or ``"frames"`` (one list per field,
    /// an entry per frame).
    #[getter]
    fn form(&self) -> &'static str {
        match self.inner.form {
            InputForm::Single => "single",
            InputForm::Frames => "frames",
        }
    }

    /// The parameter named ``name``, or ``None``.
    fn param(&self, name: &str) -> Option<PyParamSpec> {
        self.inner.param(name).map(|inner| PyParamSpec {
            inner: inner.clone(),
        })
    }

    /// The input field named ``name``, or ``None``.
    fn input(&self, name: &str) -> Option<PyInputSpec> {
        self.inner.input(name).map(|&inner| PyInputSpec { inner })
    }

    fn __repr__(&self) -> String {
        format!("ComputeSpec({:?})", self.inner.name)
    }
}

/// One configuration parameter of a compute (``molrs::compute::ParamSpec``).
#[pyclass(module = "molpy.compute", name = "ParamSpec", frozen)]
pub(crate) struct PyParamSpec {
    inner: ParamSpec,
}

#[pymethods]
impl PyParamSpec {
    /// The ``__init__`` argument name.
    #[getter]
    fn name(&self) -> &'static str {
        self.inner.name
    }

    /// What values it takes: ``"float"``, ``"int"``, ``"bool"``,
    /// ``"choice"``, ``"choice_list"``, ``"float_list"``, ``"int_list"``,
    /// ``"vec3_list"``, ``"axis"``, ``"axis_list"``, ``"grid"``, ``"range"``
    /// or ``"units"``.
    #[getter]
    fn kind(&self) -> &'static str {
        match self.inner.kind {
            ParamKind::Float => "float",
            ParamKind::Int => "int",
            ParamKind::Bool => "bool",
            ParamKind::Choice(_) => "choice",
            ParamKind::ChoiceList(_) => "choice_list",
            ParamKind::FloatList => "float_list",
            ParamKind::IntList => "int_list",
            ParamKind::Vec3List => "vec3_list",
            ParamKind::Axis => "axis",
            ParamKind::AxisList => "axis_list",
            ParamKind::Grid => "grid",
            ParamKind::Range => "range",
            ParamKind::Units => "units",
        }
    }

    /// The canonical names of a ``choice`` / ``choice_list``, else ``None``.
    #[getter]
    fn choices(&self) -> Option<Vec<&'static str>> {
        match self.inner.kind {
            ParamKind::Choice(names) | ParamKind::ChoiceList(names) => Some(names.to_vec()),
            _ => None,
        }
    }

    /// Whether the caller must give it.
    #[getter]
    fn required(&self) -> bool {
        self.inner.is_required()
    }

    /// For a parameter required by one of several constructor forms, the
    /// name of that form; else ``None``.
    #[getter]
    fn required_in(&self) -> Option<&'static str> {
        match self.inner.default {
            ParamDefault::RequiredIn(form) => Some(form),
            _ => None,
        }
    }

    /// The default value (``None`` for an option that is off, and for a
    /// parameter that is required or derived from another).
    #[getter]
    fn default<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        match &self.inner.default {
            ParamDefault::Value(value) => param_value(py, value),
            _ => Ok(py.None().into_bound(py)),
        }
    }

    /// The parameter whose value this one defaults to, else ``None``.
    #[getter]
    fn same_as(&self) -> Option<&'static str> {
        match self.inner.default {
            ParamDefault::SameAs(other) => Some(other),
            _ => None,
        }
    }

    /// ``(of, factor)`` when the default is ``factor`` times parameter
    /// ``of``, else ``None``.
    #[getter]
    fn scaled(&self) -> Option<(&'static str, f64)> {
        match self.inner.default {
            ParamDefault::Scaled { of, factor } => Some((of, factor)),
            _ => None,
        }
    }

    fn __repr__(&self) -> String {
        format!("ParamSpec({:?}, {:?})", self.inner.name, self.inner.default)
    }
}

/// One field of a compute's input (``molrs::compute::InputSpec``).
#[pyclass(module = "molpy.compute", name = "InputSpec", frozen)]
pub(crate) struct PyInputSpec {
    inner: InputSpec,
}

#[pymethods]
impl PyInputSpec {
    /// The ``compute`` keyword.
    #[getter]
    fn name(&self) -> &'static str {
        self.inner.name
    }

    /// What data it holds, e.g. ``"positions"``, ``"pair_source"``,
    /// ``"partition"``, ``"upstream"``.
    #[getter]
    fn kind(&self) -> &'static str {
        match self.inner.kind {
            InputKind::Positions => "positions",
            InputKind::Vectors => "vectors",
            InputKind::Trajectory => "trajectory",
            InputKind::Series => "series",
            InputKind::Scalars => "scalars",
            InputKind::Scalar => "scalar",
            InputKind::Index => "index",
            InputKind::Count => "count",
            InputKind::Indices => "indices",
            InputKind::IndexPairs => "index_pairs",
            InputKind::Tuples => "tuples",
            InputKind::TuplesList => "tuples_list",
            InputKind::Mask => "mask",
            InputKind::Images => "images",
            InputKind::Quaternions => "quaternions",
            InputKind::Angles => "angles",
            InputKind::SimBox => "simbox",
            InputKind::Boxes => "boxes",
            InputKind::Neighbors => "neighbors",
            InputKind::NeighborFrames => "neighbor_frames",
            InputKind::PairSource => "pair_source",
            InputKind::PairSourceFrames => "pair_source_frames",
            InputKind::Partition => "partition",
            InputKind::FeatureMatrix => "feature_matrix",
            InputKind::Grid => "grid",
            InputKind::Field => "field",
            InputKind::Choice(_) => "choice",
            InputKind::Upstream(_) => "upstream",
            InputKind::UpstreamList(_) => "upstream_list",
        }
    }

    /// The upstream result type of an ``upstream`` / ``upstream_list`` input,
    /// else ``None``.
    #[getter]
    fn upstream(&self) -> Option<&'static str> {
        match self.inner.kind {
            InputKind::Upstream(name) | InputKind::UpstreamList(name) => Some(name),
            _ => None,
        }
    }

    /// The canonical names of a ``choice`` input, else ``None``.
    #[getter]
    fn choices(&self) -> Option<Vec<&'static str>> {
        match self.inner.kind {
            InputKind::Choice(names) => Some(names.to_vec()),
            _ => None,
        }
    }

    /// Whether ``compute`` may be called without it.
    #[getter]
    fn optional(&self) -> bool {
        self.inner.optional
    }

    fn __repr__(&self) -> String {
        format!("InputSpec({:?})", self.inner.name)
    }
}

/// Give the compute class `$py` (holding its core value as `inner: $core`)
/// the descriptor methods: the static `spec()` and the instance `tunables()`.
macro_rules! described {
    ($py:ty, $core:ty) => {
        #[pymethods]
        impl $py {
            /// The descriptor of this compute type.
            #[staticmethod]
            fn spec() -> $crate::compute::spec::PyComputeSpec {
                $crate::compute::spec::PyComputeSpec {
                    inner: <$core as molrs::compute::Described>::spec(),
                }
            }

            /// The value this instance holds for every parameter that has a
            /// default, as ``[(name, value), …]`` in spec order.
            fn tunables<'py>(
                &self,
                py: pyo3::Python<'py>,
            ) -> pyo3::PyResult<Vec<(&'static str, pyo3::Bound<'py, pyo3::PyAny>)>> {
                $crate::compute::spec::tunables(
                    py,
                    molrs::compute::Described::tunables(&self.inner),
                )
            }
        }
    };
}

pub(crate) use described;

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyComputeSpec>()?;
    m.add_class::<PyParamSpec>()?;
    m.add_class::<PyInputSpec>()?;
    Ok(())
}
