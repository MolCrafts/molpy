//! Python bindings for `molrs::core` (`molpy.core`): the native unit
//! engine (`Unit`, `Quantity`, `UnitRegistry`, `UnitPreset`), its
//! `UnitsError`, and the physical constants other subsystems need by name
//! (`AMBER_COULOMB`).

use crate::error::units_error;
use molrs::core::{
    Dimension, Quantity, Unit, UnitDef, UnitPreset, UnitRegistry, lookup_unit_preset,
};
use pyo3::exceptions::{PyAttributeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyTuple;

#[pyclass(
    module = "molpy.core",
    name = "Unit",
    frozen,
    skip_from_py_object,
    subclass
)]
#[derive(Clone)]
pub struct PyUnit {
    inner: Unit,
}

impl PyUnit {
    fn new(inner: Unit) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyUnit {
    #[new]
    fn py_new(factor: f64, offset: f64, dimension: [i32; 7], name: String) -> Self {
        Self::new(Unit::from_parts(
            factor,
            offset,
            Dimension::from_exponents(dimension),
            name,
        ))
    }

    #[getter]
    fn dimension(&self) -> [i32; 7] {
        self.inner.dimension().exponents()
    }

    fn is_affine(&self) -> bool {
        self.inner.is_affine()
    }

    fn factor_to(&self, other: &PyUnit) -> PyResult<f64> {
        self.inner.factor_to(&other.inner).map_err(units_error)
    }

    fn __rmul__(&self, value: f64) -> PyQuantity {
        PyQuantity::new(Quantity::new(value, self.inner.clone()))
    }

    fn __mul__(&self, value: f64) -> PyQuantity {
        self.__rmul__(value)
    }

    fn __eq__(&self, other: &PyUnit) -> bool {
        if self.inner.is_affine() || other.inner.is_affine() {
            return self.inner == other.inner;
        }
        self.inner
            .factor_to(&other.inner)
            .is_ok_and(|factor| factor == 1.0)
    }

    fn __str__(&self) -> String {
        self.inner.to_string()
    }

    fn __repr__(&self) -> String {
        format!("<Unit('{}')>", self.inner)
    }

    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyTuple>> {
        let unit = &slf.borrow().inner;
        crate::pickle::reduce_via_type(
            slf.as_any(),
            (
                unit.factor(),
                unit.offset(),
                unit.dimension().exponents(),
                unit.name().to_owned(),
            ),
        )
    }
}

#[pyclass(
    module = "molpy.core",
    name = "Quantity",
    frozen,
    skip_from_py_object,
    subclass
)]
#[derive(Clone)]
pub struct PyQuantity {
    inner: Quantity,
}

impl PyQuantity {
    fn new(inner: Quantity) -> Self {
        Self { inner }
    }

    fn quantity_operand(value: &Bound<'_, PyAny>) -> Option<Quantity> {
        value
            .extract::<PyRef<'_, PyQuantity>>()
            .ok()
            .map(|quantity| quantity.inner.clone())
    }
}

#[pymethods]
impl PyQuantity {
    #[new]
    fn py_new(value: f64, unit: &PyUnit) -> Self {
        Self::new(Quantity::new(value, unit.inner.clone()))
    }

    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyTuple>> {
        let this = slf.borrow();
        crate::pickle::reduce_via_type(slf.as_any(), (this.value(), this.unit()))
    }

    #[getter]
    fn value(&self) -> f64 {
        self.inner.value()
    }

    #[getter]
    fn unit(&self) -> PyUnit {
        PyUnit::new(self.inner.unit().clone())
    }

    fn to(&self, target: &Bound<'_, PyAny>) -> PyResult<Self> {
        let converted = if let Ok(unit) = target.extract::<PyRef<'_, PyUnit>>() {
            self.inner.to(&unit.inner)
        } else if let Ok(expression) = target.extract::<String>() {
            self.inner.to_parsed(&expression)
        } else {
            return Err(PyTypeError::new_err(
                "target must be a Unit or unit expression",
            ));
        };
        converted.map(Self::new).map_err(units_error)
    }

    fn to_base_units(&self) -> Self {
        Self::new(self.inner.to_base_units())
    }

    fn __add__(&self, rhs: &PyQuantity) -> PyResult<Self> {
        self.inner
            .try_add(&rhs.inner)
            .map(Self::new)
            .map_err(units_error)
    }

    fn __sub__(&self, rhs: &PyQuantity) -> PyResult<Self> {
        self.inner
            .try_sub(&rhs.inner)
            .map(Self::new)
            .map_err(units_error)
    }

    fn __mul__(&self, rhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Some(quantity) = Self::quantity_operand(rhs) {
            return self
                .inner
                .try_mul(&quantity)
                .map(Self::new)
                .map_err(units_error);
        }
        if let Ok(value) = rhs.extract::<f64>() {
            return Ok(Self::new(self.inner.clone() * value));
        }
        Err(PyTypeError::new_err(
            "quantity can only multiply a scalar or Quantity",
        ))
    }

    fn __rmul__(&self, lhs: f64) -> Self {
        Self::new(self.inner.clone() * lhs)
    }

    fn __truediv__(&self, rhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Some(quantity) = Self::quantity_operand(rhs) {
            return self
                .inner
                .try_div(&quantity)
                .map(Self::new)
                .map_err(units_error);
        }
        if let Ok(value) = rhs.extract::<f64>() {
            return Ok(Self::new(self.inner.clone() / value));
        }
        Err(PyTypeError::new_err(
            "quantity can only divide by a scalar or Quantity",
        ))
    }

    fn __neg__(&self) -> Self {
        Self::new(-self.inner.clone())
    }

    fn __eq__(&self, other: &PyQuantity) -> bool {
        self.inner == other.inner
    }

    fn __str__(&self) -> String {
        self.inner.to_string()
    }

    fn __repr__(&self) -> String {
        format!(
            "<Quantity({}, '{}')>",
            self.inner.value(),
            self.inner.unit()
        )
    }
}

/// One unit definition the way Python hands it over — the positional shape of
/// [`UnitDef`]: `(name, aliases, symbol, factor, offset, dimension, prefixable)`.
type UnitDefTuple = (String, Vec<String>, String, f64, f64, [i32; 7], bool);

#[pyclass(module = "molpy.core", name = "UnitRegistry", subclass, dict)]
pub struct PyUnitRegistry {
    pub(crate) inner: UnitRegistry,
}

impl PyUnitRegistry {
    fn parse_inner(&self, expression: &str) -> PyResult<PyUnit> {
        self.inner
            .parse(expression)
            .map(PyUnit::new)
            .map_err(units_error)
    }
}

#[pymethods]
impl PyUnitRegistry {
    #[new]
    #[pyo3(signature = (definitions=None, *, empty=false))]
    fn new(definitions: Option<Vec<UnitDefTuple>>, empty: bool) -> PyResult<Self> {
        if let Some(definitions) = definitions {
            let definitions = definitions
                .into_iter()
                .map(
                    |(name, aliases, symbol, factor, offset, dimension, prefixable)| UnitDef {
                        name,
                        aliases,
                        symbol,
                        factor,
                        offset,
                        dimension: Dimension::from_exponents(dimension),
                        prefixable,
                    },
                )
                .collect();
            return UnitRegistry::from_definitions(definitions)
                .map(|inner| Self { inner })
                .map_err(units_error);
        }
        Ok(Self {
            inner: if empty {
                UnitRegistry::empty()
            } else {
                UnitRegistry::new()
            },
        })
    }

    fn parse(&self, expression: &str) -> PyResult<PyUnit> {
        self.parse_inner(expression)
    }

    /// The factor converting a value in ``from_unit`` to ``to_unit``
    /// (``value_to = value_from * factor``); a power of ten is the correctly
    /// rounded one (Å → nm is ``0.1``).
    fn factor(&self, from_unit: &str, to_unit: &str) -> PyResult<f64> {
        self.inner.factor(from_unit, to_unit).map_err(units_error)
    }

    fn quantity(&self, value: f64, expression: &str) -> PyResult<PyQuantity> {
        self.inner
            .quantity(value, expression)
            .map(PyQuantity::new)
            .map_err(units_error)
    }

    #[pyo3(signature = (name, factor, dimension, *, aliases = Vec::new(), symbol = None, offset = 0.0, prefixable = false))]
    #[allow(clippy::too_many_arguments, reason = "Public Python keyword arguments")]
    fn define(
        &mut self,
        name: String,
        factor: f64,
        dimension: [i32; 7],
        aliases: Vec<String>,
        symbol: Option<String>,
        offset: f64,
        prefixable: bool,
    ) -> PyResult<()> {
        self.inner
            .define(UnitDef {
                symbol: symbol.unwrap_or_else(|| name.clone()),
                name,
                aliases,
                factor,
                offset,
                dimension: Dimension::from_exponents(dimension),
                prefixable,
            })
            .map_err(units_error)
    }

    /// Define the reduced-LJ length unit ``lj_sigma`` alone.
    ///
    /// Only quantities of pure length dimension then have a reduced scale;
    /// every other ``lj_*`` unit stays unknown.
    ///
    /// Raises
    /// ------
    /// UnitsError
    ///     If ``sigma`` is not a finite positive length, or ``lj_sigma`` is
    ///     already defined.
    fn define_lj_sigma(&mut self, sigma: &PyQuantity) -> PyResult<()> {
        self.inner
            .define_lj_sigma(&sigma.inner)
            .map_err(units_error)
    }

    fn define_lj_units(
        &mut self,
        mass: &PyQuantity,
        sigma: &PyQuantity,
        epsilon: &PyQuantity,
    ) -> PyResult<()> {
        self.inner
            .define_lj_units(&mass.inner, &sigma.inner, &epsilon.inner)
            .map_err(units_error)
    }

    fn __getattr__(&self, name: &str) -> PyResult<PyUnit> {
        self.parse_inner(name)
            .map_err(|_| PyAttributeError::new_err(format!("unknown unit: {name}")))
    }

    fn __repr__(&self) -> &'static str {
        "<molpy.core.UnitRegistry>"
    }

    fn __reduce__<'py>(
        slf: &Bound<'py, Self>,
    ) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyTuple>, Bound<'py, PyAny>)> {
        let definitions: Vec<_> = slf
            .borrow()
            .inner
            .definitions()
            .map(|definition| {
                (
                    definition.name.clone(),
                    definition.aliases.clone(),
                    definition.symbol.clone(),
                    definition.factor,
                    definition.offset,
                    definition.dimension.exponents(),
                    definition.prefixable,
                )
            })
            .collect();
        Ok((
            slf.get_type().into_any(),
            PyTuple::new(slf.py(), [definitions])?,
            slf.getattr("__dict__")?,
        ))
    }
}

/// Named unit-system view (`"real"`, `"metal"`, …). Constants live in core;
/// this is the Python spelling of `molrs::core::UnitPreset`.
#[pyclass(
    module = "molpy.core",
    name = "UnitPreset",
    frozen,
    from_py_object,
    subclass
)]
#[derive(Clone)]
pub struct PyUnitPreset {
    pub(crate) inner: UnitPreset,
}

#[pymethods]
impl PyUnitPreset {
    #[new]
    fn new(name: &str) -> PyResult<Self> {
        lookup_unit_preset(name)
            .map(|inner| Self { inner })
            .ok_or_else(|| PyValueError::new_err(format!("unknown unit preset {name:?}")))
    }

    /// Every registered preset name, sorted: the LAMMPS styles (``real``,
    /// ``metal``, ``si``, ``cgs``, ``electron``, ``lj``, ``micro``, ``nano``),
    /// ``openmm``, and any :meth:`register` added.
    #[staticmethod]
    fn names() -> Vec<String> {
        molrs::core::unit_preset_names()
    }

    /// Register a preset of your own under ``name`` (process-wide) and return
    /// it; ``UnitPreset(name)`` finds it afterwards.
    ///
    /// Parameters
    /// ----------
    /// name : str
    ///     Preset name.
    /// units : dict[str, str]
    ///     The unit expression of each of the ten dimensions (``mass``,
    ///     ``length``, ``time``, ``energy``, ``temperature``, ``charge``,
    ///     ``pressure``, ``velocity``, ``force``, ``density``).
    /// boltzmann, coulomb : float
    ///     The two constants in this preset's units.
    /// overwrite : bool
    ///     Replace a preset already under ``name`` (a built-in included).
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     A dimension is missing or unknown, a unit is empty, or ``name`` is
    ///     taken and ``overwrite`` is false.
    #[staticmethod]
    #[pyo3(signature = (name, units, *, boltzmann, coulomb, overwrite = false))]
    fn register(
        name: &str,
        units: std::collections::HashMap<String, String>,
        boltzmann: f64,
        coulomb: f64,
        overwrite: bool,
    ) -> PyResult<Self> {
        let preset =
            UnitPreset::new(name, units, boltzmann, coulomb).map_err(PyValueError::new_err)?;
        if overwrite {
            molrs::core::replace_unit_preset(name, preset.clone())
                .map_err(PyValueError::new_err)?;
        } else {
            molrs::core::register_unit_preset(name, preset.clone())
                .map_err(PyValueError::new_err)?;
        }
        Ok(Self { inner: preset })
    }

    #[getter]
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyTuple>> {
        crate::pickle::reduce_via_type(slf.as_any(), (slf.borrow().name().to_owned(),))
    }

    /// Boltzmann constant **in this preset's energy / temperature units**
    /// (`"real"` gives kcal/mol/K), not in the amu / angstrom / fs system
    /// `molpy.md` integrates in. Convert it the same way you convert an
    /// energy, or the temperature the engine sees is off by the ratio.
    fn boltzmann(&self) -> f64 {
        self.inner.boltzmann()
    }

    /// Coulomb constant in this preset's own units — same caveat as
    /// [`boltzmann`](Self::boltzmann).
    fn coulomb(&self) -> f64 {
        self.inner.coulomb()
    }

    fn mass(&self) -> &str {
        self.inner.mass()
    }
    fn length(&self) -> &str {
        self.inner.length()
    }
    fn time(&self) -> &str {
        self.inner.time()
    }
    fn energy(&self) -> &str {
        self.inner.energy()
    }
    fn temperature(&self) -> &str {
        self.inner.temperature()
    }
    fn charge(&self) -> &str {
        self.inner.charge()
    }
    fn pressure(&self) -> &str {
        self.inner.pressure()
    }
    fn velocity(&self) -> &str {
        self.inner.velocity()
    }
    fn force(&self) -> &str {
        self.inner.force()
    }
    fn density(&self) -> &str {
        self.inner.density()
    }

    fn __repr__(&self) -> String {
        format!("UnitPreset({:?})", self.inner.name())
    }
}

/// Register `molpy.core.constants`: every constant of
/// `molrs::core::constants` ([`molrs::core::constants::ALL`]), by its Rust
/// name.
pub fn register_constants(m: &Bound<'_, PyModule>) -> PyResult<()> {
    for &(name, value) in molrs::core::constants::ALL {
        m.add(name, value)?;
    }
    Ok(())
}

/// Register the unit engine on `molpy.core`.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("UnitsError", m.py().get_type::<crate::error::UnitsError>())?;
    m.add_class::<PyUnit>()?;
    m.add_class::<PyQuantity>()?;
    m.add_class::<PyUnitRegistry>()?;
    m.add_class::<PyUnitPreset>()?;
    Ok(())
}
