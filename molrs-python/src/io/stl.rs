//! STL, stereolithography triangle meshes, into a `molrs.core.TriMesh`
//! (`molrs.io.read_stl`).

use super::source::PySource;
use crate::core::mesh::PyTriMesh;
use crate::error::io_error_to_pyerr;
use pyo3::prelude::*;

/// Read an STL surface mesh (ASCII or binary).
///
/// The file's numbers are taken as they are; convert with
/// ``TriMesh.scaled`` when the file is not in the length unit you work in.
///
/// Parameters
/// ----------
/// source : os.PathLike | bytes | molrs.io.Text
///     The file, or its whole content.
///
/// Returns
/// -------
/// molrs.core.TriMesh
///
/// Raises
/// ------
/// OSError
///     If the file cannot be opened or is not STL.
#[pyfunction]
pub fn read_stl(source: PySource<'_>) -> PyResult<PyTriMesh> {
    let mesh = source
        .read(molrs::io::read_stl)
        .map_err(io_error_to_pyerr)?;
    Ok(PyTriMesh { inner: mesh })
}

/// Register this module's door.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    crate::add_function(m, "molrs.io", wrap_pyfunction!(read_stl, m)?)
}
