//! Geometric regions for the C++ engine — the CXX face of
//! `molrs::core::Region` (through `molrs_ffi::RegionRef`).

/// Bridge handle over a shared region.
///
/// The region is `Arc<dyn Region>` inside; the trait object never crosses the
/// bridge. C++ holds a `Box<RegionRef>` — the same `molrs_ffi::RegionRef` the
/// Python capsule and the C API carry, so a region built on any of those
/// surfaces answers identically here.
pub struct RegionRef(pub molrs_ffi::RegionRef);

impl RegionRef {
    fn wrap(region: std::sync::Arc<dyn molrs::core::Region + Send + Sync>) -> Box<Self> {
        Box::new(RegionRef(molrs_ffi::RegionRef::new(region)))
    }
}

/// `v` as a 3-vector, or an error naming `what` when it is not 3 long.
fn exact_triple(v: &[f64], what: &str) -> Result<[f64; 3], String> {
    <[f64; 3]>::try_from(v).map_err(|_| format!("{what} must have 3 elements, got {}", v.len()))
}

/// Ball of `radius` about `center` (3 values).
pub(crate) fn region_sphere(center: &[f64], radius: f64) -> Result<Box<RegionRef>, String> {
    let c = exact_triple(center, "region_sphere: center")?;
    Ok(RegionRef::wrap(std::sync::Arc::new(
        molrs::core::Sphere::new(molrs::op::F3::from_vec(c.to_vec()), radius),
    )))
}

/// Axis-aligned box with a corner at `origin` and edge `lengths` (3 values each).
pub(crate) fn region_cuboid(origin: &[f64], lengths: &[f64]) -> Result<Box<RegionRef>, String> {
    let o = exact_triple(origin, "region_cuboid: origin")?;
    let l = exact_triple(lengths, "region_cuboid: lengths")?;
    Ok(RegionRef::wrap(std::sync::Arc::new(
        molrs::core::Cuboid::new(
            molrs::op::F3::from_vec(o.to_vec()),
            molrs::op::F3::from_vec(l.to_vec()),
        ),
    )))
}

/// Everything on the `normal` side of the plane through `point`.
/// A degenerate normal is an error.
pub(crate) fn region_half_space(normal: &[f64], point: &[f64]) -> Result<Box<RegionRef>, String> {
    let normal = exact_triple(normal, "region_half_space: normal")?;
    let point = exact_triple(point, "region_half_space: point")?;
    molrs::core::HalfSpace::new(normal, point)
        .map(|r| RegionRef::wrap(std::sync::Arc::new(r)))
        .map_err(|e| e.to_string())
}

/// Finite cylinder of `radius` and `length` from `base` along `axis`.
pub(crate) fn region_cylinder(
    base: &[f64],
    axis: &[f64],
    radius: f64,
    length: f64,
) -> Result<Box<RegionRef>, String> {
    let base = exact_triple(base, "region_cylinder: base")?;
    let axis = exact_triple(axis, "region_cylinder: axis")?;
    molrs::core::Cylinder::new(base, axis, radius, length)
        .map(|r| RegionRef::wrap(std::sync::Arc::new(r)))
        .map_err(|e| e.to_string())
}

/// Ellipsoid about `center` with the given `semi_axes`.
pub(crate) fn region_ellipsoid(
    center: &[f64],
    semi_axes: &[f64],
) -> Result<Box<RegionRef>, String> {
    let center = exact_triple(center, "region_ellipsoid: center")?;
    let semi_axes = exact_triple(semi_axes, "region_ellipsoid: semi_axes")?;
    molrs::core::Ellipsoid::new(center, semi_axes)
        .map(|r| RegionRef::wrap(std::sync::Arc::new(r)))
        .map_err(|e| e.to_string())
}

/// Intersection of `a` and `b`. Both stay usable.
pub(crate) fn region_and(a: &RegionRef, b: &RegionRef) -> Box<RegionRef> {
    RegionRef::wrap(std::sync::Arc::new(molrs::core::AndRegion::new(
        a.0.region(),
        b.0.region(),
    )))
}

/// Union of `a` and `b`.
pub(crate) fn region_or(a: &RegionRef, b: &RegionRef) -> Box<RegionRef> {
    RegionRef::wrap(std::sync::Arc::new(molrs::core::OrRegion::new(
        a.0.region(),
        b.0.region(),
    )))
}

/// Everything `a` is not. A shell is `and(outer, not(inner))`.
pub(crate) fn region_not(a: &RegionRef) -> Box<RegionRef> {
    RegionRef::wrap(std::sync::Arc::new(molrs::core::NotRegion::new(
        a.0.region(),
    )))
}

/// `points` as `[x, y, z]` rows, or an error when its length is ragged.
fn point_rows<'a>(points: &'a [f64], what: &str) -> Result<&'a [[f64; 3]], String> {
    match points.as_chunks::<3>() {
        (rows, []) => Ok(rows),
        _ => Err(format!(
            "{what}: points must be flat [x, y, z, ...], got {} values",
            points.len()
        )),
    }
}

/// Signed distance of each point to the boundary: negative inside, positive
/// outside. `points` is flat `[x, y, z, …]`; a ragged length is an error.
pub(crate) fn region_distance(rref: &RegionRef, points: &[f64]) -> Result<Vec<f64>, String> {
    let rows = point_rows(points, "region_distance")?;
    Ok(rref
        .0
        .with_region(|r| rows.iter().map(|p| r.distance(p)).collect()))
}

/// `1` for each point inside the solid, `0` outside. cxx has no `Vec<bool>`.
pub(crate) fn region_contains(rref: &RegionRef, points: &[f64]) -> Result<Vec<u8>, String> {
    let rows = point_rows(points, "region_contains")?;
    Ok(rref
        .0
        .with_region(|r| rows.iter().map(|p| u8::from(r.contains_point(p))).collect()))
}

/// Axis-aligned bounds as `[xmin, xmax, ymin, ymax, zmin, zmax]`.
pub(crate) fn region_bounds(rref: &RegionRef) -> Vec<f64> {
    rref.0.with_region(|r| r.bounds().iter().copied().collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_shell_is_and_of_outer_and_not_inner() {
        let outer = region_sphere(&[0.0, 0.0, 0.0], 3.0).unwrap();
        let inner = region_sphere(&[0.0, 0.0, 0.0], 2.0).unwrap();
        let shell = region_and(&outer, &region_not(&inner));
        // 2.5 is in the shell, 1.0 is in the hole, 4.0 is outside both.
        let pts = [2.5, 0.0, 0.0, 1.0, 0.0, 0.0, 4.0, 0.0, 0.0];
        assert_eq!(region_contains(&shell, &pts).unwrap(), vec![1, 0, 0]);
        let d = region_distance(&outer, &pts).unwrap();
        assert!((d[0] + 0.5).abs() < 1e-12, "{d:?}");
        assert_eq!(region_bounds(&outer).len(), 6);
    }

    #[test]
    fn malformed_input_is_an_error_not_a_fallback() {
        let ball = region_sphere(&[0.0, 0.0, 0.0], 1.0).unwrap();
        assert!(region_distance(&ball, &[0.0, 0.0]).is_err());
        assert!(region_contains(&ball, &[0.0, 0.0]).is_err());
        assert!(region_sphere(&[], 1.0).is_err());
        assert!(region_cuboid(&[0.0; 3], &[1.0, 1.0]).is_err());
        assert!(region_cuboid(&[0.0; 2], &[1.0; 3]).is_err());
        assert!(region_cuboid(&[0.0; 3], &[1.0; 3]).is_ok());
    }
}
