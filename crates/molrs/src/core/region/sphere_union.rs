//! The union of many spheres — atoms as a region: [`SphereUnion`].

use super::primitives::Region;
use crate::core::bvh::Bvh;
use crate::core::{BoxError, SimBox};
use crate::op::vec3::{add, norm, sub};
use crate::op::{F, Fnx3, Fnx3View};
use ndarray::Array2;

/// Why a set of centres and radii cannot become a [`SphereUnion`].
#[derive(Debug, Clone, PartialEq)]
pub enum SphereUnionError {
    /// No spheres.
    Empty,
    /// `centers` and `radii` disagree on the count.
    LengthMismatch {
        /// Rows of `centers`.
        centers: usize,
        /// Entries of `radii`.
        radii: usize,
    },
    /// Sphere `index` has a non-finite centre or radius.
    NonFinite {
        /// Index into `centers` / `radii`.
        index: usize,
    },
    /// Sphere `index` has a radius `<= 0`.
    NonPositiveRadius {
        /// Index into `centers` / `radii`.
        index: usize,
    },
    /// The free box around the centres could not be built.
    Box {
        /// The [`BoxError`] as text.
        detail: String,
    },
}

impl std::fmt::Display for SphereUnionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Empty => write!(f, "sphere union has no spheres"),
            Self::LengthMismatch { centers, radii } => {
                write!(f, "{centers} centres but {radii} radii")
            }
            Self::NonFinite { index } => write!(f, "sphere {index} is not finite"),
            Self::NonPositiveRadius { index } => write!(f, "sphere {index} has radius <= 0"),
            Self::Box { detail } => write!(f, "free box around the centres: {detail}"),
        }
    }
}

impl std::error::Error for SphereUnionError {}

impl From<BoxError> for SphereUnionError {
    fn from(err: BoxError) -> Self {
        Self::Box {
            detail: format!("{err:?}"),
        }
    }
}

/// The union of spheres `‖x − c_i‖ ≤ r_i`, minimum-image on the periodic axes
/// of its box.
///
/// With one sphere per atom of radius `r_vdW + r_probe`, the union is the
/// solvent-accessible volume of Lee & Richards (*J. Mol. Biol.* 55, 379–400,
/// 1971): its boundary is the surface a probe of radius `r_probe` sweeps
/// while rolling over the atoms, and `NotRegion(SphereUnion)` is the space a
/// probe centre can occupy — the void a packer fills. The numerical form of
/// that surface is Shrake & Rupley (*J. Mol. Biol.* 79, 351–371, 1973); this
/// type needs no surface, only the distance to it. The solvent-*excluded*
/// (Connolly) surface is a different object and is not this type. The radii are
/// the caller's: molrs knows centres and lengths, not chemistry.
///
/// `distance(x) = min_{i, s} (‖w(x) + s − c_i‖ − r_i)` over the spheres `i`
/// and the lattice shifts `s` of the periodic axes (each of `−1, 0, +1` cells),
/// where `w` wraps the point into the cell and the centres are stored
/// wrapped. Outside the union this is the exact Euclidean distance to the
/// nearest sphere surface; inside it is the deepest single-sphere
/// penetration, a lower bound on the distance to the union's boundary. The
/// gradient is the radial unit vector of the winning sphere, zero at its
/// centre.
///
/// Queries walk a bounding-volume hierarchy over the spheres' boxes: one descent for the
/// unshifted image, and the other images only enter where the point is
/// within reach of a face.
#[derive(Debug, Clone)]
pub struct SphereUnion {
    centers: Vec<[F; 3]>,
    radii: Vec<F>,
    bx: SimBox,
    shifts: Vec<[F; 3]>,
    bvh: Bvh,
    bounds: Fnx3,
}

impl SphereUnion {
    /// Spheres at `centers` (N×3, Å) with per-sphere `radii` (Å) inside `bx`;
    /// centres are wrapped on the periodic axes of `bx` and distances use the
    /// minimum image there.
    ///
    /// # Errors
    ///
    /// Returns `Err` for no spheres, a count mismatch, a non-finite centre or
    /// radius, or a radius `<= 0`.
    pub fn new(centers: Fnx3View<'_>, radii: &[F], bx: &SimBox) -> Result<Self, SphereUnionError> {
        Self::check(centers, radii)?;
        let pbc = bx.pbc();
        let centers: Vec<[F; 3]> = centers
            .rows()
            .into_iter()
            .map(|r| bx.wrap_row([r[0], r[1], r[2]]))
            .collect();
        let radii = radii.to_vec();
        let boxes: Vec<_> = centers
            .iter()
            .zip(&radii)
            .map(|(c, &r)| {
                (
                    [c[0] - r, c[1] - r, c[2] - r],
                    [c[0] + r, c[1] + r, c[2] + r],
                )
            })
            .collect();
        let bvh = Bvh::build(&boxes);
        let mut shifts = vec![[0.0; 3]];
        let lattice: Vec<[F; 3]> = (0..3)
            .map(|k| {
                let v = bx.lattice(k);
                [v[0], v[1], v[2]]
            })
            .collect();
        let steps = |k: usize| -> Vec<i8> { if pbc[k] { vec![0, -1, 1] } else { vec![0] } };
        for i in steps(0) {
            for j in steps(1) {
                for k in steps(2) {
                    if i == 0 && j == 0 && k == 0 {
                        continue;
                    }
                    let mut s = [0.0; 3];
                    for d in 0..3 {
                        s[d] = i as F * lattice[0][d]
                            + j as F * lattice[1][d]
                            + k as F * lattice[2][d];
                    }
                    shifts.push(s);
                }
            }
        }
        let bounds = if pbc.iter().any(|&p| p) {
            bx.bounds()
        } else {
            let mut b = Array2::zeros((3, 2));
            for d in 0..3 {
                b[[d, 0]] = F::INFINITY;
                b[[d, 1]] = F::NEG_INFINITY;
            }
            for (lo, hi) in &boxes {
                for d in 0..3 {
                    b[[d, 0]] = b[[d, 0]].min(lo[d]);
                    b[[d, 1]] = b[[d, 1]].max(hi[d]);
                }
            }
            b
        };
        Ok(Self {
            centers,
            radii,
            bx: bx.clone(),
            shifts,
            bvh,
            bounds,
        })
    }

    /// Spheres in open space: the box is [`SimBox::bounding`] around the centres
    /// with twice the largest radius of padding, so every sphere lies inside
    /// it and no axis is periodic.
    ///
    /// # Errors
    ///
    /// The gates of [`new`](Self::new), plus a box that could not be built.
    pub fn free(centers: Fnx3View<'_>, radii: &[F]) -> Result<Self, SphereUnionError> {
        let r_max = Self::check(centers, radii)?;
        let bx = SimBox::bounding(centers, [2.0 * r_max; 3], [false; 3])?;
        Self::new(centers, radii, &bx)
    }

    /// Number of spheres.
    pub fn n_spheres(&self) -> usize {
        self.centers.len()
    }

    /// Sphere centres as stored: wrapped into the box on its periodic axes, Å.
    pub fn centers(&self) -> &[[F; 3]] {
        &self.centers
    }

    /// Sphere radii, Å, in the same order as [`centers`](Self::centers).
    pub fn radii(&self) -> &[F] {
        &self.radii
    }

    /// The box the union lives in.
    pub fn simbox(&self) -> &SimBox {
        &self.bx
    }

    /// Validate and return the largest radius.
    fn check(centers: Fnx3View<'_>, radii: &[F]) -> Result<F, SphereUnionError> {
        if centers.ncols() != 3 {
            return Err(SphereUnionError::LengthMismatch {
                centers: centers.nrows(),
                radii: radii.len(),
            });
        }
        if centers.nrows() != radii.len() {
            return Err(SphereUnionError::LengthMismatch {
                centers: centers.nrows(),
                radii: radii.len(),
            });
        }
        if radii.is_empty() {
            return Err(SphereUnionError::Empty);
        }
        let mut r_max: F = 0.0;
        for (index, (row, &r)) in centers.rows().into_iter().zip(radii).enumerate() {
            if !r.is_finite() || row.iter().any(|x| !x.is_finite()) {
                return Err(SphereUnionError::NonFinite { index });
            }
            if r <= 0.0 {
                return Err(SphereUnionError::NonPositiveRadius { index });
            }
            r_max = r_max.max(r);
        }
        Ok(r_max)
    }

    /// The nearest sphere over every periodic image: `(index, shifted query,
    /// signed distance)`.
    fn nearest(&self, point: &[F; 3]) -> (usize, [F; 3], F) {
        let w = self.bx.wrap_row(*point);
        let mut best = (usize::MAX, w, F::INFINITY);
        for s in &self.shifts {
            let q = add(w, *s);
            let hit = self.bvh.nearest_below(&q, best.2, |i| {
                norm(sub(q, self.centers[i as usize])) - self.radii[i as usize]
            });
            if let Some((i, d)) = hit {
                best = (i as usize, q, d);
            }
        }
        best
    }
}

impl Region for SphereUnion {
    /// The box on a periodic union (the union tiles it); the box of the
    /// spheres themselves in open space.
    fn bounds(&self) -> Fnx3 {
        self.bounds.clone()
    }

    fn distance(&self, point: &[F; 3]) -> F {
        self.nearest(point).2
    }

    fn distance_grad(&self, point: &[F; 3]) -> [F; 3] {
        let (i, q, _) = self.nearest(point);
        let d = sub(q, self.centers[i]);
        let n = norm(d);
        if n < 1e-12 {
            [0.0; 3]
        } else {
            [d[0] / n, d[1] / n, d[2] / n]
        }
    }

    /// Inside as soon as any image of any sphere reaches the point — a
    /// threshold query that stops at the first hit.
    fn contains_point(&self, point: &[F; 3]) -> bool {
        let w = self.bx.wrap_row(*point);
        self.shifts.iter().any(|s| {
            let q = add(w, *s);
            self.bvh.any_within(&q, 0.0, |i| {
                norm(sub(q, self.centers[i as usize])) - self.radii[i as usize]
            })
        })
    }
}

#[cfg(test)]
mod tests {
    use super::super::primitives::tests::check_region_contract;
    use super::*;
    use crate::core::Cell;
    use ndarray::array;

    /// Deterministic point cloud in `[0, L)³` — no rand dependency.
    fn cloud(n: usize, l: F) -> (Fnx3, Vec<F>) {
        let mut s: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 11) as F / (1u64 << 53) as F
        };
        let mut c = Array2::zeros((n, 3));
        let mut r = Vec::with_capacity(n);
        for i in 0..n {
            for d in 0..3 {
                c[[i, d]] = next() * l;
            }
            r.push(0.6 + next() * 1.2);
        }
        (c, r)
    }

    /// Brute force through the box's own minimum image.
    fn oracle(centers: &Fnx3, radii: &[F], bx: &SimBox, p: [F; 3]) -> F {
        centers
            .rows()
            .into_iter()
            .zip(radii)
            .map(|(c, &r)| {
                let d = bx.mic().apply(crate::op::vec3::sub([c[0], c[1], c[2]], p));
                norm(d) - r
            })
            .fold(F::INFINITY, F::min)
    }

    fn probes(l: F) -> Vec<[F; 3]> {
        let mut out = Vec::new();
        for i in 0..8 {
            for j in 0..5 {
                let t = i as F / 7.0;
                let u = j as F / 4.0;
                out.push([t * l * 1.3 - 0.15 * l, u * l, 0.37 * l + 0.2 * t * l]);
            }
        }
        out.push([-3.0, -3.0, -3.0]);
        out.push([l + 5.0, 0.5 * l, 0.5 * l]);
        out
    }

    #[test]
    fn matches_the_brute_force_oracle_in_every_box_kind() {
        let l = 12.0;
        let (c, r) = cloud(300, l);
        let boxes = [
            SimBox::new(Cell::Lengths([l; 3]), [0.0, 0.0, 0.0], [true, true, true]).unwrap(),
            SimBox::new(Cell::Lengths([l; 3]), [0.0, 0.0, 0.0], [true, true, false]).unwrap(),
            SimBox::new(
                Cell::Lengths([l; 3]),
                [0.0, 0.0, 0.0],
                [false, false, false],
            )
            .unwrap(),
            SimBox::new(
                Cell::Matrix([[l, 0.25 * l, 0.0], [0.0, l, 0.1 * l], [0.0, 0.0, l]]),
                [0.0, 0.0, 0.0],
                [true, true, true],
            )
            .unwrap(),
        ];
        for bx in &boxes {
            let u = SphereUnion::new(c.view(), &r, bx).unwrap();
            for p in probes(l) {
                let expect = oracle(&c, &r, bx, p);
                let got = u.distance(&p);
                assert!(
                    (got - expect).abs() < 1e-9,
                    "{got} vs oracle {expect} at {p:?} in {:?}",
                    bx.pbc()
                );
                assert_eq!(u.contains_point(&p), expect <= 0.0, "at {p:?}");
            }
        }
    }

    #[test]
    fn free_box_holds_every_sphere_and_reports_their_extent() {
        let c = array![[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]];
        let u = SphereUnion::free(c.view(), &[1.0, 2.0]).unwrap();
        assert!(!u.simbox().pbc().iter().any(|&p| p));
        let b = u.bounds();
        assert_eq!(b[[0, 0]], -1.0);
        assert_eq!(b[[0, 1]], 12.0);
        assert_eq!(u.n_spheres(), 2);
        // Far outside: exact Euclidean distance to the nearer sphere surface.
        assert!((u.distance(&[20.0, 0.0, 0.0]) - 8.0).abs() < 1e-12);
        assert_eq!(u.distance_grad(&[20.0, 0.0, 0.0]), [1.0, 0.0, 0.0]);
        // Between them, the union is not the hull.
        assert!(!u.contains_point(&[5.0, 0.0, 0.0]));
        assert!((u.distance(&[5.0, 0.0, 0.0]) - 3.0).abs() < 1e-12);
    }

    #[test]
    fn wrap_neighbour_across_face_is_found() {
        let l = 10.0;
        let bx = SimBox::new(Cell::Lengths([l; 3]), [0.0, 0.0, 0.0], [true, true, true]).unwrap();
        let c = array![[0.5, 5.0, 5.0]];
        let u = SphereUnion::new(c.view(), &[1.0], &bx).unwrap();
        // 9.8 is 0.7 from the centre through the x face.
        assert!(u.contains_point(&[9.8, 5.0, 5.0]));
        assert!((u.distance(&[9.8, 5.0, 5.0]) + 0.3).abs() < 1e-12);
        assert_eq!(u.distance_grad(&[9.8, 5.0, 5.0]), [-1.0, 0.0, 0.0]);
        // And a point outside the cell wraps first.
        assert!((u.distance(&[-0.2, 5.0, 5.0]) + 0.3).abs() < 1e-12);
    }

    #[test]
    fn inside_reports_the_deepest_sphere() {
        let c = array![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        let u = SphereUnion::free(c.view(), &[1.0, 3.0]).unwrap();
        // 0.5 from both centres: 0.5 deep in the small one, 2.5 in the big.
        assert!((u.distance(&[0.5, 0.0, 0.0]) + 2.5).abs() < 1e-12);
        assert_eq!(u.distance_grad(&[0.5, 0.0, 0.0]), [-1.0, 0.0, 0.0]);
    }

    #[test]
    fn contract_holds_off_the_ties() {
        let l = 12.0;
        let (c, r) = cloud(60, l);
        let bx = SimBox::new(Cell::Lengths([l; 3]), [0.0, 0.0, 0.0], [true, true, true]).unwrap();
        let u = SphereUnion::new(c.view(), &r, &bx).unwrap();
        // Keep probes away from centres and from ties between spheres, where
        // the max/min switches branch and the finite difference is one-sided.
        let smooth: Vec<[F; 3]> = probes(l)
            .into_iter()
            .filter(|p| {
                let (i, q, d) = u.nearest(p);
                let second = (0..u.n_spheres())
                    .filter(|&j| j != i)
                    .map(|j| norm(sub(q, u.centers[j])) - u.radii[j])
                    .fold(F::INFINITY, F::min);
                norm(sub(q, u.centers[i])) > 1e-3 && (second - d).abs() > 1e-3
            })
            .collect();
        assert!(smooth.len() > 20);
        check_region_contract(&u, &smooth);
    }

    /// molvis's `ball_field` splat, `f(p) = max_i (r_i − ‖p − x_i‖)`, is the
    /// negated distance sampled on a grid.
    #[test]
    fn ball_field_identity() {
        let c = array![[1.0, 1.0, 1.0], [2.5, 1.0, 1.0], [1.0, 2.5, 2.5]];
        let r = [0.8, 0.9, 1.1];
        let u = SphereUnion::free(c.view(), &r).unwrap();
        for ix in 0..5 {
            for iy in 0..5 {
                for iz in 0..5 {
                    let p = [ix as F * 0.7, iy as F * 0.7, iz as F * 0.7];
                    let splat = c
                        .rows()
                        .into_iter()
                        .zip(&r)
                        .map(|(x, &ri)| ri - norm(sub(p, [x[0], x[1], x[2]])))
                        .fold(F::NEG_INFINITY, F::max);
                    assert!((splat + u.distance(&p)).abs() < 1e-12, "at {p:?}");
                }
            }
        }
    }

    #[test]
    fn named_rejects() {
        let c = array![[0.0, 0.0, 0.0]];
        let bx = SimBox::new(Cell::Lengths([5.0; 3]), [0.0, 0.0, 0.0], [true; 3]).unwrap();
        let empty: Fnx3 = Array2::zeros((0, 3));
        assert_eq!(
            SphereUnion::new(empty.view(), &[], &bx).err(),
            Some(SphereUnionError::Empty)
        );
        assert!(matches!(
            SphereUnion::new(c.view(), &[1.0, 2.0], &bx),
            Err(SphereUnionError::LengthMismatch { .. })
        ));
        assert_eq!(
            SphereUnion::new(c.view(), &[0.0], &bx).err(),
            Some(SphereUnionError::NonPositiveRadius { index: 0 })
        );
        assert_eq!(
            SphereUnion::new(c.view(), &[F::NAN], &bx).err(),
            Some(SphereUnionError::NonFinite { index: 0 })
        );
    }
}
