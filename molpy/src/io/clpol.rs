//! `molpy.io.read_clpol_alpha` — `molrs::io`'s door of the same name: a
//! CL&Pol `alpha.ff` polarisation table read into rows.

use crate::error::io_error_to_pyerr;
use molrs::io::clpol::ClpolAlphaRow;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use super::PySource;

/// One row's fields under the names ``DrudeBuilder.polarizability`` uses, and
/// its ``type_name``.
fn row_dict(py: Python<'_>, row: ClpolAlphaRow) -> PyResult<Bound<'_, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("type_name", row.type_name)?;
    dict.set_item("m_D", row.m_d)?;
    dict.set_item("q_D_sign", row.q_d_sign)?;
    dict.set_item("k_D", row.k_d)?;
    dict.set_item("alpha", row.alpha)?;
    dict.set_item("a_thole", row.a_thole)?;
    Ok(dict)
}

/// Read a CL&Pol ``alpha.ff`` table into its rows, in file order.
///
/// The text is whitespace-separated rows ``type m_D q_D k_D alpha a_thole``;
/// ``#`` starts a comment and a line with fewer than six fields is not a
/// row. A type given twice keeps both rows; the later one is the file's last
/// word. The table molpy ships is ``mp.builder.DrudeBuilder().polarizability``.
///
/// Parameters
/// ----------
/// source : os.PathLike | bytes | molpy.io.Text
///     The file, or its whole content.
///
/// Returns
/// -------
/// list[dict]
///     One dict per row: ``type_name``, ``m_D`` (u), ``q_D_sign``, ``k_D``
///     (kJ/mol/Å², ``k/2 r²`` form), ``alpha`` (Å³) and ``a_thole``.
///
/// Raises
/// ------
/// ValueError
///     A row's numeric fields do not parse.
/// OSError
///     The file cannot be read.
#[pyfunction]
pub fn read_clpol_alpha<'py>(
    py: Python<'py>,
    source: PySource<'_>,
) -> PyResult<Vec<Bound<'py, PyDict>>> {
    let rows = source
        .read(molrs::io::read_clpol_alpha)
        .map_err(io_error_to_pyerr)?;
    rows.into_iter().map(|row| row_dict(py, row)).collect()
}

/// Register this module's door.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    crate::add_function(m, "molpy.io", wrap_pyfunction!(read_clpol_alpha, m)?)
}
