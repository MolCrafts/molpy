//! `molpy.ff.ir`'s engine forms (`ff-ir-02-protocol` §8): a style's
//! LAMMPS form, from Python.
//!
//! * `style_registry.register_style(..., lammps="positional")` (and a
//!   `style_registry.StyleDeclaration`'s
//!   `lammps = "positional"`) gives a style the positional LAMMPS form: its
//!   `params` in order on the `*_coeff` line, each converted by its
//!   dimension; `"positional:<name>"` writes it under the LAMMPS style
//!   `<name>`; `None` (the default) gives it none, so LAMMPS refuses it by
//!   name ([`NoEngineFormError`](crate::ff::ir::errors::NoEngineFormError));
//! * `style_registry.register_engine_form(engine, category, name, form)`
//!   gives a style registered without one its form afterwards;
//! * `StyleSpec.lammps` names a style's form: `"positional"`,
//!   `"positional:<name>"`, `"custom:<name>"` (a codec of its own, the
//!   built-ins whose LAMMPS line is not positional) or `None`.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use molrs::ff::ir::{LammpsForm, StyleSpec};

/// The LAMMPS form a Python `lammps=` names: `None`, `"positional"` or
/// `"positional:<name>"`.
pub(crate) fn lammps_form(value: Option<&str>) -> PyResult<LammpsForm> {
    match value {
        None => Ok(LammpsForm::None),
        Some("positional") => Ok(LammpsForm::positional()),
        Some(v) => match v.strip_prefix("positional:") {
            Some(name) if !name.is_empty() => Ok(LammpsForm::positional_named(name.to_owned())),
            _ => Err(PyValueError::new_err(format!(
                "lammps={v:?}: None, \"positional\" or \"positional:<LAMMPS style name>\""
            ))),
        },
    }
}

/// How `spec`'s LAMMPS form is spelled in Python.
pub(crate) fn lammps_form_name(spec: &StyleSpec) -> Option<String> {
    match &spec.lammps {
        LammpsForm::None => None,
        LammpsForm::Positional { name: None } => Some("positional".into()),
        LammpsForm::Positional { name: Some(n) } => Some(format!("positional:{n}")),
        LammpsForm::Custom(_) => spec.lammps.lammps_name(spec).map(|n| format!("custom:{n}")),
    }
}
