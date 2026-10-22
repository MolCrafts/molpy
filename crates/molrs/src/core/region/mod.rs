//! Geometric regions: solids with a signed distance to their boundary (see [`Region`]).

mod cylinder;
mod ellipsoid;
mod half_space;
mod polyhedron;
mod primitives;
mod sphere_union;

pub use cylinder::Cylinder;
pub use ellipsoid::Ellipsoid;
pub use half_space::HalfSpace;
pub use polyhedron::{Polyhedron, PolyhedronError};
pub use primitives::{AndRegion, Cuboid, NotRegion, OrRegion, Parallelepiped, Region, Sphere};
pub use sphere_union::{SphereUnion, SphereUnionError};
