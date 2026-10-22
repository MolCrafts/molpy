//! Geometric regions for JavaScript — the same nine solids the Python binder
//! exposes, under the same names.
//!
//! A region is a solid that answers a **signed distance**: negative inside,
//! positive outside, zero on the boundary. `contains` is that sign, and
//! `bounds` is the axis-aligned box the solid fits in.
//!
//! JavaScript has no operator overloading, so the composition that Python
//! spells `a & b`, `a | b` and `~a` is spelled `a.and(b)`, `a.or(b)` and
//! `a.not()` here. Each returns a [`Region`], the composed solid, which
//! answers the same three questions — so `outer.and(inner.not())` is a shell.
//!
//! # Example (JavaScript)
//!
//! ```js
//! import { Sphere } from "@molcrafts/molrs";
//!
//! const ball  = Sphere.create([0, 0, 0], 3.0);
//! const shell = ball.and(Sphere.create([0, 0, 0], 2.0).not());
//!
//! shell.distance(new Float64Array([2.5, 0, 0]));   // Float64Array [-0.5]
//! shell.contains(new Float64Array([2.5, 0, 0]));   // Uint8Array [1]
//! shell.free();
//! ```

use std::sync::Arc;

use molrs::core::{
    AndRegion, Cuboid as RsCuboid, Cylinder as RsCylinder, Ellipsoid as RsEllipsoid,
    HalfSpace as RsHalfSpace, NotRegion, OrRegion, Parallelepiped as RsParallelepiped,
    Polyhedron as RsPolyhedron, Region as RegionTrait, Sphere as RsSphere,
    SphereUnion as RsSphereUnion,
};
use molrs::op::{F, F3, Fnx3};
use molrs_ffi::RegionRef;
use ndarray::Array2;
use wasm_bindgen::prelude::*;

use crate::core::mesh::TriMesh;
use crate::core::simbox::{Box as WasmBox, Cell};

/// Reshape a flat `[x0, y0, z0, x1, …]` array into the `N × 3` the core takes.
fn points_from_flat(points: &[F]) -> Result<Fnx3, JsValue> {
    if !points.len().is_multiple_of(3) {
        return Err(JsValue::from_str(&format!(
            "points must be a flat [x, y, z, …] array; got {} values, not a multiple of 3",
            points.len()
        )));
    }
    Array2::from_shape_vec((points.len() / 3, 3), points.to_vec())
        .map_err(|e| JsValue::from_str(&format!("points: {e}")))
}

fn distance_of(region: &dyn RegionTrait, points: &[F]) -> Result<Vec<F>, JsValue> {
    let pts = points_from_flat(points)?;
    Ok((0..pts.nrows())
        .map(|i| region.distance(&[pts[[i, 0]], pts[[i, 1]], pts[[i, 2]]]))
        .collect())
}

fn contains_of(region: &dyn RegionTrait, points: &[F]) -> Result<Vec<u8>, JsValue> {
    let pts = points_from_flat(points)?;
    Ok(region.contains(&pts).iter().map(|&b| u8::from(b)).collect())
}

/// `[xmin, xmax, ymin, ymax, zmin, zmax]`.
fn bounds_of(region: &dyn RegionTrait) -> Vec<F> {
    region.bounds().iter().copied().collect()
}

/// A composed region: the result of `and`, `or` or `not`.
///
/// It answers the same three questions as any single solid, so compositions
/// nest without a special case.
#[wasm_bindgen]
pub struct Region {
    inner: RegionRef,
}

impl Region {
    fn wrap(region: impl RegionTrait + 'static) -> Self {
        Self {
            inner: RegionRef::new(Arc::new(region)),
        }
    }
}

/// Every region class answers the same three questions and composes the same
/// three ways; only construction differs.
macro_rules! region_surface {
    ($ty:ty) => {
        #[wasm_bindgen]
        impl $ty {
            /// Signed distance of each point to the boundary: negative inside,
            /// positive outside, zero on it. `points` is flat `[x, y, z, …]`.
            pub fn distance(&self, points: &[F]) -> Result<Vec<F>, JsValue> {
                self.inner.with_region(|r| distance_of(r, points))
            }

            /// `1` for each point inside the solid, `0` outside. `points` is
            /// flat `[x, y, z, …]`.
            pub fn contains(&self, points: &[F]) -> Result<Vec<u8>, JsValue> {
                self.inner.with_region(|r| contains_of(r, points))
            }

            /// Axis-aligned bounding box as
            /// `[xmin, xmax, ymin, ymax, zmin, zmax]`.
            pub fn bounds(&self) -> Vec<F> {
                self.inner.with_region(bounds_of)
            }

            /// Intersection — what Python spells `self & other`.
            pub fn and(&self, other: &Region) -> Region {
                Region::wrap(AndRegion::new(self.inner.region(), other.inner.region()))
            }

            /// Union — what Python spells `self | other`.
            pub fn or(&self, other: &Region) -> Region {
                Region::wrap(OrRegion::new(self.inner.region(), other.inner.region()))
            }

            /// Complement — what Python spells `~self`. The outside of a shape
            /// is `shape.not()`; a shell is `outer.and(inner.not())`.
            #[wasm_bindgen(js_name = not)]
            pub fn not_region(&self) -> Region {
                Region::wrap(NotRegion::new(self.inner.region()))
            }

            /// This solid as a composed [`Region`], so it can be passed to
            /// `and` / `or`.
            #[wasm_bindgen(js_name = asRegion)]
            pub fn as_region(&self) -> Region {
                Region {
                    inner: self.inner.clone(),
                }
            }
        }
    };
}

region_surface!(Region);

/// Ball of `radius` about `center`.
#[wasm_bindgen]
pub struct Sphere {
    inner: RegionRef,
}

#[wasm_bindgen]
impl Sphere {
    /// `center` is `[x, y, z]`.
    pub fn create(center: &[F], radius: F) -> Result<Sphere, JsValue> {
        let c = triple("center", center)?;
        Ok(Sphere {
            inner: RegionRef::new(Arc::new(RsSphere::new(F3::from_vec(c.to_vec()), radius))),
        })
    }
}

region_surface!(Sphere);

/// Axis-aligned box with a corner at `origin` and the given edge `lengths`.
#[wasm_bindgen]
pub struct Cuboid {
    inner: RegionRef,
}

#[wasm_bindgen]
impl Cuboid {
    pub fn create(origin: &[F], lengths: &[F]) -> Result<Cuboid, JsValue> {
        let o = triple("origin", origin)?;
        let l = triple("lengths", lengths)?;
        Ok(Cuboid {
            inner: RegionRef::new(Arc::new(RsCuboid::new(
                F3::from_vec(o.to_vec()),
                F3::from_vec(l.to_vec()),
            ))),
        })
    }
}

region_surface!(Cuboid);

/// The solid a [`Cell`] spans from `origin` — `Parallelepiped::new(cell,
/// origin)`. One constructor: a cube is `Cell.lengths([a, a, a])`, an
/// orthorhombic cell `Cell.lengths([a, b, c])`, any other `Cell.matrix(h)`.
#[wasm_bindgen]
pub struct Parallelepiped {
    inner: RegionRef,
}

#[wasm_bindgen]
impl Parallelepiped {
    /// `origin` is `[x, y, z]` (Å).
    ///
    /// # Errors
    ///
    /// Throws if `origin` is not 3 values, or the cell spans no volume
    /// (`Cell.undefined()`, a singular matrix, a non-positive edge).
    pub fn create(cell: &Cell, origin: &[F]) -> Result<Parallelepiped, JsValue> {
        let o = triple("origin", origin)?;
        RsParallelepiped::new(cell.inner, F3::from_vec(o.to_vec()))
            .map(|r| Parallelepiped {
                inner: RegionRef::new(Arc::new(r)),
            })
            .map_err(|e| JsValue::from_str(&e))
    }
}

region_surface!(Parallelepiped);

/// Everything on the `normal` side of the plane through `point`.
#[wasm_bindgen]
pub struct HalfSpace {
    inner: RegionRef,
}

#[wasm_bindgen]
impl HalfSpace {
    pub fn create(normal: &[F], point: &[F]) -> Result<HalfSpace, JsValue> {
        let n = triple("normal", normal)?;
        let p = triple("point", point)?;
        RsHalfSpace::new(n, p)
            .map(|r| HalfSpace {
                inner: RegionRef::new(Arc::new(r)),
            })
            .map_err(|e| JsValue::from_str(&e))
    }
}

region_surface!(HalfSpace);

/// Finite cylinder: `base`, `axis`, `radius`, `length`.
#[wasm_bindgen]
pub struct Cylinder {
    inner: RegionRef,
}

#[wasm_bindgen]
impl Cylinder {
    pub fn create(base: &[F], axis: &[F], radius: F, length: F) -> Result<Cylinder, JsValue> {
        let b = triple("base", base)?;
        let a = triple("axis", axis)?;
        RsCylinder::new(b, a, radius, length)
            .map(|r| Cylinder {
                inner: RegionRef::new(Arc::new(r)),
            })
            .map_err(|e| JsValue::from_str(&e))
    }
}

region_surface!(Cylinder);

/// Ellipsoid about `center` with the given `semiAxes`.
#[wasm_bindgen]
pub struct Ellipsoid {
    inner: RegionRef,
}

#[wasm_bindgen]
impl Ellipsoid {
    #[wasm_bindgen(js_name = create)]
    pub fn create(center: &[F], semi_axes: &[F]) -> Result<Ellipsoid, JsValue> {
        let c = triple("center", center)?;
        let s = triple("semiAxes", semi_axes)?;
        RsEllipsoid::new(c, s)
            .map(|r| Ellipsoid {
                inner: RegionRef::new(Arc::new(r)),
            })
            .map_err(|e| JsValue::from_str(&e))
    }
}

region_surface!(Ellipsoid);

/// Solid bounded by a watertight triangle mesh — what `readStl` reads.
#[wasm_bindgen]
pub struct Polyhedron {
    inner: RegionRef,
}

#[wasm_bindgen]
impl Polyhedron {
    pub fn create(mesh: &TriMesh) -> Result<Polyhedron, JsValue> {
        RsPolyhedron::new(mesh.inner.clone())
            .map(|r| Polyhedron {
                inner: RegionRef::new(Arc::new(r)),
            })
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

region_surface!(Polyhedron);

/// Union of one sphere per site, under the minimum image of `box`.
///
/// This is atoms as a region: pass the centers flat and one radius each.
#[wasm_bindgen]
pub struct SphereUnion {
    inner: RegionRef,
    n_spheres: usize,
}

#[wasm_bindgen]
impl SphereUnion {
    pub fn create(centers: &[F], radii: &[F], bx: &WasmBox) -> Result<SphereUnion, JsValue> {
        let c = points_from_flat(centers)?;
        if c.nrows() != radii.len() {
            return Err(JsValue::from_str(&format!(
                "{} centers but {} radii",
                c.nrows(),
                radii.len()
            )));
        }
        RsSphereUnion::new(c.view(), radii, &bx.inner)
            .map(|r| SphereUnion {
                n_spheres: r.n_spheres(),
                inner: RegionRef::new(Arc::new(r)),
            })
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Number of spheres in the union.
    #[wasm_bindgen(js_name = nSpheres)]
    pub fn n_spheres(&self) -> usize {
        self.n_spheres
    }
}

region_surface!(SphereUnion);

fn triple(name: &str, v: &[F]) -> Result<[F; 3], JsValue> {
    v.try_into()
        .map_err(|_| JsValue::from_str(&format!("{name} must be 3 values, got {}", v.len())))
}

#[cfg(test)]
mod tests {
    use super::*;
    use wasm_bindgen_test::*;

    #[wasm_bindgen_test]
    fn sphere_distance_is_signed() {
        let ball = Sphere::create(&[0.0, 0.0, 0.0], 2.0).unwrap();
        let d = ball
            .distance(&[0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 2.0, 0.0, 0.0])
            .unwrap();
        assert!((d[0] + 2.0).abs() < 1e-12, "centre is -r: {}", d[0]);
        assert!((d[1] - 1.0).abs() < 1e-12, "outside is +1: {}", d[1]);
        assert!(d[2].abs() < 1e-12, "on the boundary is 0: {}", d[2]);
    }

    #[wasm_bindgen_test]
    fn contains_is_the_sign_of_distance() {
        let ball = Sphere::create(&[0.0, 0.0, 0.0], 2.0).unwrap();
        let inside = ball.contains(&[0.0, 0.0, 0.0, 3.0, 0.0, 0.0]).unwrap();
        assert_eq!(inside, vec![1, 0]);
    }

    #[wasm_bindgen_test]
    fn a_shell_is_outer_and_not_inner() {
        let outer = Sphere::create(&[0.0, 0.0, 0.0], 3.0).unwrap();
        let inner = Sphere::create(&[0.0, 0.0, 0.0], 2.0).unwrap();
        let shell = outer.and(&inner.not_region());
        // 2.5 is in the shell; 1.0 is in the hole; 4.0 is outside both.
        let hits = shell
            .contains(&[2.5, 0.0, 0.0, 1.0, 0.0, 0.0, 4.0, 0.0, 0.0])
            .unwrap();
        assert_eq!(hits, vec![1, 0, 0]);
    }

    #[wasm_bindgen_test]
    fn union_holds_either_side() {
        let a = Sphere::create(&[0.0, 0.0, 0.0], 1.0).unwrap();
        let b = Sphere::create(&[5.0, 0.0, 0.0], 1.0).unwrap();
        let both = a.or(&b.as_region());
        assert_eq!(
            both.contains(&[0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 2.5, 0.0, 0.0])
                .unwrap(),
            vec![1, 1, 0]
        );
    }

    #[wasm_bindgen_test]
    fn bounds_are_flat_min_max_pairs() {
        let b = Cuboid::create(&[1.0, 2.0, 3.0], &[1.0, 1.0, 1.0])
            .unwrap()
            .bounds();
        assert_eq!(b, vec![1.0, 2.0, 2.0, 3.0, 3.0, 4.0]);
    }

    #[wasm_bindgen_test]
    fn a_ragged_point_array_is_named_not_truncated() {
        let ball = Sphere::create(&[0.0, 0.0, 0.0], 1.0).unwrap();
        assert!(ball.distance(&[0.0, 0.0]).is_err());
    }

    #[wasm_bindgen_test]
    fn every_shape_constructs_and_answers() {
        let shapes: Vec<Region> = vec![
            Sphere::create(&[0.0, 0.0, 0.0], 1.0).unwrap().as_region(),
            Cuboid::create(&[0.0, 0.0, 0.0], &[1.0, 1.0, 1.0])
                .unwrap()
                .as_region(),
            Parallelepiped::create(&Cell::lengths(&[2.0; 3]).unwrap(), &[0.0, 0.0, 0.0])
                .unwrap()
                .as_region(),
            HalfSpace::create(&[0.0, 0.0, 1.0], &[0.0, 0.0, 0.0])
                .unwrap()
                .as_region(),
            Cylinder::create(&[0.0, 0.0, 0.0], &[0.0, 0.0, 1.0], 1.0, 2.0)
                .unwrap()
                .as_region(),
            Ellipsoid::create(&[0.0, 0.0, 0.0], &[1.0, 2.0, 3.0])
                .unwrap()
                .as_region(),
        ];
        for s in &shapes {
            assert_eq!(s.bounds().len(), 6);
            assert_eq!(s.contains(&[0.0, 0.0, 0.0]).unwrap().len(), 1);
        }
    }

    #[wasm_bindgen_test]
    fn sphere_union_counts_its_spheres() {
        let bx = WasmBox::new(
            &Cell::lengths(&[10.0; 3]).unwrap(),
            &[0.0; 3],
            &crate::core::simbox::tests::pbc([true; 3]),
        )
        .unwrap();
        let union = SphereUnion::create(
            &[1.0, 1.0, 1.0, 5.0, 5.0, 5.0, 8.0, 1.0, 1.0, 2.0, 7.0, 3.0],
            &[1.0, 1.0, 0.5, 0.5],
            &bx,
        )
        .unwrap();
        assert_eq!(union.n_spheres(), 4);
    }
}
