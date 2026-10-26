//! Python bindings for `molrs::core` (`molpy.core`): the column store and
//! the frame (`Block`, `Frame`, `Trajectory` and their metadata), the
//! simulation cell (`Box`), neighbour search, regions, meshes and point
//! paths, the molecular-graph hierarchy (`MolGraph`, `Atomistic`,
//! `CoarseGrain`) with its live views, elements and the index-only
//! `Topology`, and the unit engine — all flat on `molpy.core`, as in Rust.
//!
//! Three vocabularies are submodules, as in Rust: `molpy.core.keys`,
//! `molpy.core.schema` and `molpy.core.constants`.

pub mod block;
pub mod bond_order;
pub mod element;
pub mod frame;
pub mod graph_views;
pub mod mesh;
pub mod molgraph;
pub mod neighborlist;
pub mod region;
pub mod schema;
pub mod simbox;
pub mod topology;
pub mod trace;
pub mod trajectory;
pub mod units;

use pyo3::prelude::*;

/// Register `molpy.core`: its classes, `BlockDtypeError`, `UnitsError`, and
/// the `keys` / `schema` / `constants` submodules (each base class before
/// its subclasses).
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add(
        "BlockDtypeError",
        m.py().get_type::<crate::error::BlockDtypeError>(),
    )?;
    m.add_class::<block::PyBlock>()?;
    m.add_class::<frame::PyMetaValue>()?;
    m.add_class::<frame::PyMetaDocument>()?;
    m.add_class::<frame::PyFrameMeta>()?;
    m.add_class::<frame::PyFrame>()?;
    m.add_class::<trajectory::PyTrajectory>()?;
    m.add_class::<trajectory::PyObservableRecord>()?;

    m.add_class::<simbox::PyCell>()?;
    m.add_class::<simbox::PyBox>()?;
    m.add_class::<neighborlist::PyNeighborList>()?;
    m.add_class::<neighborlist::PyNeighbors>()?;
    m.add_class::<neighborlist::PyNeighborQuery>()?;
    m.add_class::<neighborlist::PyVerletSkin>()?;
    m.add_class::<mesh::PyTriMesh>()?;
    m.add_class::<region::PySphere>()?;
    m.add_class::<region::PyCuboid>()?;
    m.add_class::<region::PyParallelepiped>()?;
    m.add_class::<region::PyHalfSpace>()?;
    m.add_class::<region::PyCylinder>()?;
    m.add_class::<region::PyEllipsoid>()?;
    m.add_class::<region::PyPolyhedron>()?;
    m.add_class::<region::PySphereUnion>()?;
    m.add_class::<region::PyRegion>()?;
    m.add_class::<trace::PyTrace>()?;

    m.add_class::<element::PyElement>()?;
    m.add_class::<bond_order::PyBondOrder>()?;
    m.add_class::<bond_order::PyBondNumber>()?;
    m.add_class::<topology::PyTopology>()?;
    m.add_class::<molgraph::PyMolGraph>()?;
    m.add_class::<molgraph::PyAtomistic>()?;
    m.add_class::<molgraph::PyCoarseGrain>()?;
    m.add_class::<molgraph::PyExtractedSubgraph>()?;
    m.add_class::<graph_views::PyNodeRef>()?;
    m.add_class::<graph_views::PyAtom>()?;
    m.add_class::<graph_views::PyVirtualSite>()?;
    m.add_class::<graph_views::PyDrudeParticle>()?;
    m.add_class::<graph_views::PyMasslessSite>()?;
    m.add_class::<graph_views::PyBead>()?;
    m.add_class::<graph_views::PyRelationRef>()?;
    m.add_class::<graph_views::PyBond>()?;
    m.add_class::<graph_views::PyAngle>()?;
    m.add_class::<graph_views::PyDihedral>()?;
    m.add_class::<graph_views::PyImproper>()?;
    m.add_class::<graph_views::PyPort>()?;
    m.add_class::<graph_views::PyCgBond>()?;
    m.add_class::<graph_views::PyRefs>()?;

    units::register(m)?;

    crate::add_submodule(m, "keys", "molpy.core.keys", schema::register_keys)?;
    crate::add_submodule(m, "schema", "molpy.core.schema", schema::register_schema)?;
    crate::add_submodule(
        m,
        "constants",
        "molpy.core.constants",
        units::register_constants,
    )?;
    Ok(())
}
