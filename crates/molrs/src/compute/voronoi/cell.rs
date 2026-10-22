//! Per-cell geometry for the radical (Laguerre) Voronoi tessellation.

use crate::op::vec3::{cross, dot, norm, sub};
use molrs::op::F;

/// One bounding face of a cell: its area and the index of the neighbour cell
/// across it. `neighbor < 0` ([`VORONOI_BOUNDARY`]) marks a residual
/// initial-box face; a [`RadicalVoronoi`](crate::compute::RadicalVoronoi)
/// tessellation never leaves one, because every periodic cell is bounded by
/// real neighbours — its own images at the least.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct VoronoiFace {
    /// Face area (Å²).
    pub area: F,
    /// Index of the generator across this face (an atom/probe index), or a
    /// negative sentinel for an unclipped initial-box face.
    pub neighbor: i64,
}

/// Boundary sentinel for an initial-box face that was never clipped away.
pub const VORONOI_BOUNDARY: i64 = -1;

/// Result of a radical-Voronoi tessellation: one entry per generator.
///
/// A cell is built by *successive half-space clipping* of an initially
/// box-sized convex polyhedron against the radical plane of each candidate
/// neighbour — the same plane-cut semantics as voro++'s
/// `voronoicell_base::nplane` (`src/v_cell.cpp`), with the radical (power)
/// offset from `radius_poly` (`src/v_rad_option.h`). voro++ maintains an
/// incremental vertex/edge structure for speed; molrs uses a direct
/// convex-polyhedron clip (deliberate deviation, documented on
/// [`RadicalVoronoi`](crate::compute::RadicalVoronoi)) — slower but numerically identical: the cut plane,
/// the kept half-space, and the resulting face/area/neighbour relations match.
///
/// # References
/// - Rycroft, *Chaos* **2009**, 19, 041111 (voro++ cell-by-cell algorithm).
/// - Aurenhammer, *SIAM J. Comput.* **1987**, 16, 78 (power diagrams).
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct VoronoiCells {
    /// Cell volume per generator (Å³).
    pub volumes: Vec<F>,
    /// Bounding faces per generator.
    pub faces: Vec<Vec<VoronoiFace>>,
}

impl VoronoiCells {
    /// Sum of all cell volumes — equals the box volume for a correct periodic
    /// tessellation (the headline invariant).
    pub fn total_volume(&self) -> F {
        self.volumes.iter().sum()
    }

    /// Number of generators (cells).
    pub fn len(&self) -> usize {
        self.volumes.len()
    }

    /// Whether the tessellation is empty.
    pub fn is_empty(&self) -> bool {
        self.volumes.is_empty()
    }

    /// Sorted, de-duplicated neighbour generator indices of cell `i`
    /// (negative box sentinels dropped).
    pub fn neighbors(&self, i: usize) -> Vec<i64> {
        let mut v: Vec<i64> = self.faces[i]
            .iter()
            .map(|f| f.neighbor)
            .filter(|&n| n >= 0)
            .collect();
        v.sort_unstable();
        v.dedup();
        v
    }
}

/// A convex polyhedron as vertices + outward-oriented face loops (vertex
/// indices). Built as a box, then clipped by [`Poly::clip`].
pub(crate) struct Poly {
    verts: Vec<[F; 3]>,
    /// `(neighbor_id, ordered vertex-index loop)` — loop is CCW seen from
    /// outside (right-hand normal points outward).
    faces: Vec<(i64, Vec<usize>)>,
}

impl Poly {
    /// Axis-aligned box `[-hx,hx]×[-hy,hy]×[-hz,hz]` in cell-local coordinates
    /// (origin at the generator). All six faces carry the [`VORONOI_BOUNDARY`] id.
    pub(crate) fn box_cell(hx: F, hy: F, hz: F) -> Self {
        let verts = vec![
            [-hx, -hy, -hz], // 0
            [hx, -hy, -hz],  // 1
            [hx, hy, -hz],   // 2
            [-hx, hy, -hz],  // 3
            [-hx, -hy, hz],  // 4
            [hx, -hy, hz],   // 5
            [hx, hy, hz],    // 6
            [-hx, hy, hz],   // 7
        ];
        // Loops ordered so the right-hand normal points outward.
        let faces = vec![
            (VORONOI_BOUNDARY, vec![0, 3, 2, 1]), // -z (normal -z)
            (VORONOI_BOUNDARY, vec![4, 5, 6, 7]), // +z
            (VORONOI_BOUNDARY, vec![0, 1, 5, 4]), // -y
            (VORONOI_BOUNDARY, vec![2, 3, 7, 6]), // +y
            (VORONOI_BOUNDARY, vec![0, 4, 7, 3]), // -x
            (VORONOI_BOUNDARY, vec![1, 2, 6, 5]), // +x
        ];
        Poly { verts, faces }
    }

    /// Clip the polyhedron to the half-space `dot(n, x) <= off`, recording the
    /// new cap face with neighbour id `nid`. `n` need not be unit length.
    ///
    /// Ports the plane cut of `voronoicell_base::nplane` (`src/v_cell.cpp`):
    /// the kept half-space is the generator side of the radical plane.
    pub(crate) fn clip(&mut self, n: [F; 3], off: F, nid: i64) {
        let scale = norm(n).max(1.0);
        let eps = 1e-12 * scale;
        let s: Vec<F> = self.verts.iter().map(|&v| dot(n, v) - off).collect();
        let inside: Vec<bool> = s.iter().map(|&d| d <= eps).collect();

        if inside.iter().all(|&b| b) {
            return; // plane does not cut the cell
        }

        let mut new_verts: Vec<[F; 3]> = Vec::new();
        // old inside vertex -> new index
        let mut vmap: Vec<Option<usize>> = vec![None; self.verts.len()];
        // dedup edge intersections by unordered old-vertex pair
        use std::collections::HashMap;
        let mut cutmap: HashMap<(usize, usize), usize> = HashMap::new();

        let push_inside =
            |vmap: &mut Vec<Option<usize>>, new_verts: &mut Vec<[F; 3]>, i: usize| -> usize {
                if let Some(ni) = vmap[i] {
                    ni
                } else {
                    let ni = new_verts.len();
                    new_verts.push(self.verts[i]);
                    vmap[i] = Some(ni);
                    ni
                }
            };

        let interp = |a: usize, b: usize| -> [F; 3] {
            let t = s[a] / (s[a] - s[b]);
            let va = self.verts[a];
            let vb = self.verts[b];
            [
                va[0] + t * (vb[0] - va[0]),
                va[1] + t * (vb[1] - va[1]),
                va[2] + t * (vb[2] - va[2]),
            ]
        };

        let mut new_faces: Vec<(i64, Vec<usize>)> = Vec::new();
        let mut cap_verts: Vec<usize> = Vec::new();

        for (fid, lp) in &self.faces {
            let m = lp.len();
            let mut out: Vec<usize> = Vec::with_capacity(m + 2);
            for k in 0..m {
                let a = lp[k];
                let b = lp[(k + 1) % m];
                if inside[a] {
                    out.push(push_inside(&mut vmap, &mut new_verts, a));
                }
                if inside[a] != inside[b] {
                    let key = if a < b { (a, b) } else { (b, a) };
                    let ci = *cutmap.entry(key).or_insert_with(|| {
                        let ni = new_verts.len();
                        new_verts.push(interp(a, b));
                        ni
                    });
                    out.push(ci);
                    cap_verts.push(ci);
                }
            }
            // drop consecutive duplicates
            out.dedup();
            if out.len() >= 2 && out.first() == out.last() {
                out.pop();
            }
            if out.len() >= 3 {
                new_faces.push((*fid, out));
            }
        }

        // Build the cap face on the cut plane from the collected intersection
        // vertices, ordered CCW about their centroid with outward normal +n.
        cap_verts.sort_unstable();
        cap_verts.dedup();
        if cap_verts.len() >= 3 {
            let cap = order_cap(&new_verts, &cap_verts, n);
            new_faces.push((nid, cap));
        }

        self.verts = new_verts;
        self.faces = new_faces;
    }

    /// Distance from the generator (the cell-local origin) to the farthest
    /// vertex. A candidate neighbour whose radical plane lies beyond this
    /// radius cannot cut the cell — the basis for the O(N) candidate-search
    /// termination bound of [`RadicalVoronoi`](crate::compute::RadicalVoronoi).
    pub(crate) fn max_vertex_dist(&self) -> F {
        self.verts
            .iter()
            .map(|v| dot(*v, *v))
            .fold(0.0, F::max)
            .sqrt()
    }

    /// Cell volume via outward-face tetrahedra from the origin (the generator,
    /// which is interior): `V = (1/6) Σ_face Σ_fan v0·(va×vb)`.
    pub(crate) fn volume(&self) -> F {
        let mut v6 = 0.0;
        for (_, lp) in &self.faces {
            let p0 = self.verts[lp[0]];
            for k in 1..lp.len() - 1 {
                let pa = self.verts[lp[k]];
                let pb = self.verts[lp[k + 1]];
                v6 += dot(p0, cross(pa, pb));
            }
        }
        (v6 / 6.0).abs()
    }

    /// Bounding faces with their areas and neighbour ids.
    pub(crate) fn cell_faces(&self) -> Vec<VoronoiFace> {
        self.faces
            .iter()
            .map(|(nid, lp)| {
                let mut an = [0.0; 3];
                let m = lp.len();
                for k in 0..m {
                    let a = self.verts[lp[k]];
                    let b = self.verts[lp[(k + 1) % m]];
                    let c = cross(a, b);
                    an[0] += c[0];
                    an[1] += c[1];
                    an[2] += c[2];
                }
                VoronoiFace {
                    area: 0.5 * norm(an),
                    neighbor: *nid,
                }
            })
            .collect()
    }
}

/// Order cap vertices CCW about their centroid, in the plane with outward
/// normal `n`.
fn order_cap(verts: &[[F; 3]], idx: &[usize], n: [F; 3]) -> Vec<usize> {
    let nn = norm(n).max(1e-300);
    let nhat = [n[0] / nn, n[1] / nn, n[2] / nn];
    let mut c = [0.0; 3];
    for &i in idx {
        c[0] += verts[i][0];
        c[1] += verts[i][1];
        c[2] += verts[i][2];
    }
    let k = idx.len() as F;
    c = [c[0] / k, c[1] / k, c[2] / k];
    // in-plane basis (u, w) with w = n × u
    let mut u = sub(verts[idx[0]], c);
    let ul = norm(u);
    u = [u[0] / ul, u[1] / ul, u[2] / ul];
    let w = cross(nhat, u);
    let mut keyed: Vec<(F, usize)> = idx
        .iter()
        .map(|&i| {
            let d = sub(verts[i], c);
            (dot(d, w).atan2(dot(d, u)), i)
        })
        .collect();
    keyed.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    keyed.into_iter().map(|(_, i)| i).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn box_cell_volume_is_exact() {
        let p = Poly::box_cell(1.0, 2.0, 3.0);
        assert!((p.volume() - 8.0 * 6.0).abs() < 1e-12);
    }

    #[test]
    fn single_plane_clip_halves_the_box() {
        // Clip a unit-half-extent cube by x <= 0 → volume halves, new cap area = 4.
        let mut p = Poly::box_cell(1.0, 1.0, 1.0);
        p.clip([1.0, 0.0, 0.0], 0.0, 7);
        assert!((p.volume() - 4.0).abs() < 1e-12, "vol {}", p.volume());
        let cap = p
            .cell_faces()
            .into_iter()
            .find(|f| f.neighbor == 7)
            .unwrap();
        assert!((cap.area - 4.0).abs() < 1e-12, "cap area {}", cap.area);
    }
}
