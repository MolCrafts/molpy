//! Python bindings for `molrs::ff::params` (`molpy.ff.params`): the parameter
//! tables molrs ships: the CL&Pol `alpha.ff` polarizability table
//! (`clpol_polarizability`). The AMBER 1-4 divisors are engine constants,
//! `molpy.core.constants.AMBER_SCEE` / `AMBER_SCNB`.

use pyo3::prelude::*;
use std::collections::HashMap;

/// One row's fields under the names CL&Pol (and molpy) use.
fn fields(
    m_d: f64,
    q_d_sign: f64,
    k_d: f64,
    alpha: f64,
    a_thole: f64,
) -> HashMap<&'static str, f64> {
    HashMap::from([
        ("m_D", m_d),
        ("q_D_sign", q_d_sign),
        ("k_D", k_d),
        ("alpha", alpha),
        ("a_thole", a_thole),
    ])
}

/// CL&Pol Drude parameters per atom type: the ``alpha.ff`` table molrs ships
/// (paduagroup/clandpol ``alpha.ff`` 2024/06/05,
/// ``molrs::ff::params::CLPOL_POLARIZABILITY``). A caller's own ``alpha.ff``
/// is read by :func:`molpy.io.read_clpol_alpha`.
///
/// Returns
/// -------
/// dict[str, dict[str, float]]
///     ``type -> {"m_D", "q_D_sign", "k_D", "alpha", "a_thole"}`` in u, e
///     (sign), kJ/mol/Å² (``k/2 r²`` form), Å³ and dimensionless. A type
///     with ``k_D == 0`` carries no Drude particle.
#[pyfunction]
pub fn clpol_polarizability() -> HashMap<String, HashMap<&'static str, f64>> {
    molrs::ff::params::CLPOL_POLARIZABILITY
        .iter()
        .map(|r| {
            (
                r.type_name.to_owned(),
                fields(r.m_d, r.q_d_sign, r.k_d, r.alpha, r.a_thole),
            )
        })
        .collect()
}

/// Register `molpy.ff.params`.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    crate::add_function(
        m,
        "molpy.ff.params",
        wrap_pyfunction!(clpol_polarizability, m)?,
    )?;
    Ok(())
}
