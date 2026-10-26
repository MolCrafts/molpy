//! Bounding-volume hierarchy over boxed items.
//!
//! The mesh region asks two questions of its triangles — closest point, and
//! how many times a ray crosses them — and the sphere-union region asks one of
//! its spheres — which is nearest. Each is a scan over every item without an
//! index, which is fine for a twelve-triangle box and hopeless at melt scale,
//! where a packer's lattice mask alone asks about a million sites.
//!
//! One tree serves all three queries. It knows nothing about the items beyond
//! their bounding boxes: the caller supplies the per-item metric, and the tree
//! only decides which items are worth asking about. The descent prunes on the
//! distance from the query point to a node box, so it is exact whenever the
//! metric of an item is never smaller than that distance for a point outside
//! the item's box — true for the Euclidean distance to a triangle and for the
//! signed distance `‖x − c‖ − r` to a sphere boxed as `c ± r`. The BVH
//! changes what is *visited*, never what is *answered*.

use crate::op::F;

/// Items per leaf. A leaf scan is a handful of metric evaluations, cheaper
/// than the box tests that would separate them further.
const LEAF_SIZE: usize = 4;

#[derive(Debug, Clone, Default)]
struct Node {
    min: [F; 3],
    max: [F; 3],
    /// Interior node: index of the left child, with the right at `first + 1`.
    /// Leaf: index into [`Bvh::order`] of its first item.
    first: u32,
    /// Item count for a leaf; `0` marks an interior node.
    count: u32,
}

/// Median-split BVH. Empty for an empty item list; every query then returns
/// its empty answer.
#[derive(Debug, Clone)]
pub(crate) struct Bvh {
    nodes: Vec<Node>,
    /// Item indices permuted so that every leaf owns a contiguous run.
    order: Vec<u32>,
}

impl Bvh {
    /// Build over one axis-aligned box `(min, max)` per item.
    pub(crate) fn build(items: &[([F; 3], [F; 3])]) -> Self {
        let mut order: Vec<u32> = (0..items.len() as u32).collect();
        let centroids: Vec<[F; 3]> = items
            .iter()
            .map(|(lo, hi)| {
                [
                    0.5 * (lo[0] + hi[0]),
                    0.5 * (lo[1] + hi[1]),
                    0.5 * (lo[2] + hi[2]),
                ]
            })
            .collect();
        let mut nodes = vec![Node::default()];
        if !order.is_empty() {
            let n = order.len();
            build_node(&mut nodes, &mut order, items, &centroids, 0, 0, n);
        }
        Self { nodes, order }
    }

    /// The item with the smallest `metric`, as `(item, value)`.
    ///
    /// `metric(i)` is the caller's distance from `p` to item `i`; it may be
    /// negative (a signed distance). Nodes are pruned when their box is
    /// farther from `p` than the best value found so far, or, once the best
    /// is negative, when `p` is outside their box — a deeper item can only be
    /// one whose box contains `p`.
    pub(crate) fn nearest<M>(&self, p: &[F; 3], metric: M) -> Option<(u32, F)>
    where
        M: FnMut(u32) -> F,
    {
        self.nearest_below(p, F::INFINITY, metric)
    }

    /// [`nearest`](Self::nearest) restricted to items with `metric < bound`,
    /// `None` when there is no such item.
    ///
    /// A caller that already holds a candidate from another query — the
    /// sphere union asks the same tree once per periodic image — passes it as
    /// the bound, so an image that cannot improve on it prunes at the root.
    pub(crate) fn nearest_below<M>(&self, p: &[F; 3], bound: F, mut metric: M) -> Option<(u32, F)>
    where
        M: FnMut(u32) -> F,
    {
        if self.order.is_empty() {
            return None;
        }
        let mut best = (u32::MAX, bound);
        self.nearest_in(0, p, &mut metric, &mut best);
        (best.0 != u32::MAX).then_some(best)
    }

    fn nearest_in<M>(&self, n: usize, p: &[F; 3], metric: &mut M, best: &mut (u32, F))
    where
        M: FnMut(u32) -> F,
    {
        let node = &self.nodes[n];
        if node.count > 0 {
            let end = (node.first + node.count) as usize;
            for &i in &self.order[node.first as usize..end] {
                let d = metric(i);
                if d < best.1 {
                    *best = (i, d);
                }
            }
            return;
        }
        let left = node.first as usize;
        let dl = box_dist(&self.nodes[left], p);
        let dr = box_dist(&self.nodes[left + 1], p);
        // Nearer child first: it tightens `best` before the other is tested,
        // which is what turns the second test into a prune.
        let (near, far, dnear, dfar) = if dl <= dr {
            (left, left + 1, dl, dr)
        } else {
            (left + 1, left, dr, dl)
        };
        if dnear <= best.1.max(0.0) {
            self.nearest_in(near, p, metric, best);
        }
        if dfar <= best.1.max(0.0) {
            self.nearest_in(far, p, metric, best);
        }
    }

    /// The `k` items with the smallest `metric`, nearest first.
    ///
    /// The same descent as [`nearest`](Self::nearest) with a `k`-deep frontier
    /// instead of a single best: a node prunes once its box is farther than
    /// the worst of the `k` held so far, which only bites once the frontier is
    /// full. Fewer than `k` items yields all of them.
    pub(crate) fn knn<M>(&self, p: &[F; 3], k: usize, mut metric: M) -> Vec<(u32, F)>
    where
        M: FnMut(u32) -> F,
    {
        if k == 0 || self.order.is_empty() {
            return Vec::new();
        }
        let mut top: Vec<(u32, F)> = Vec::with_capacity(k + 1);
        self.knn_in(0, p, k, &mut metric, &mut top);
        top
    }

    fn knn_in<M>(&self, n: usize, p: &[F; 3], k: usize, metric: &mut M, top: &mut Vec<(u32, F)>)
    where
        M: FnMut(u32) -> F,
    {
        let node = &self.nodes[n];
        if node.count > 0 {
            let end = (node.first + node.count) as usize;
            for &i in &self.order[node.first as usize..end] {
                let d = metric(i);
                if top.len() == k && d >= top[k - 1].1 {
                    continue;
                }
                let at = top.partition_point(|&(_, other)| other <= d);
                top.insert(at, (i, d));
                top.truncate(k);
            }
            return;
        }
        let worst = if top.len() == k {
            top[k - 1].1
        } else {
            F::INFINITY
        };
        let left = node.first as usize;
        let dl = box_dist(&self.nodes[left], p);
        let dr = box_dist(&self.nodes[left + 1], p);
        let (near, far, dnear, dfar) = if dl <= dr {
            (left, left + 1, dl, dr)
        } else {
            (left + 1, left, dr, dl)
        };
        if dnear <= worst.max(0.0) {
            self.knn_in(near, p, k, metric, top);
        }
        // `worst` may have tightened while descending the nearer child.
        let worst = if top.len() == k {
            top[k - 1].1
        } else {
            F::INFINITY
        };
        if dfar <= worst.max(0.0) {
            self.knn_in(far, p, k, metric, top);
        }
    }

    /// Whether any item has `metric <= threshold`.
    ///
    /// The same descent as [`nearest`](Self::nearest) without the
    /// bookkeeping: a membership test only needs to know whether something is
    /// within reach, and answering that as a threshold query stops at the
    /// first hit instead of finding the closest one.
    pub(crate) fn any_within<M>(&self, p: &[F; 3], threshold: F, mut metric: M) -> bool
    where
        M: FnMut(u32) -> F,
    {
        !self.order.is_empty() && self.any_within_in(0, p, threshold, &mut metric)
    }

    fn any_within_in<M>(&self, n: usize, p: &[F; 3], threshold: F, metric: &mut M) -> bool
    where
        M: FnMut(u32) -> F,
    {
        let node = &self.nodes[n];
        if box_dist(node, p) > threshold.max(0.0) {
            return false;
        }
        if node.count > 0 {
            let end = (node.first + node.count) as usize;
            return self.order[node.first as usize..end]
                .iter()
                .any(|&i| metric(i) <= threshold);
        }
        let left = node.first as usize;
        self.any_within_in(left, p, threshold, metric)
            || self.any_within_in(left + 1, p, threshold, metric)
    }

    /// Visit every item whose metric is within `radius`.
    ///
    /// The same descent as [`any_within`](Self::any_within) without the
    /// short-circuit: a cutoff search wants all of them, not the first. The
    /// visitor is called once per item that passes, in whatever order the
    /// descent reaches it — a BVH walk owes no ordering, and a cutoff pair
    /// table does not depend on one.
    pub(crate) fn for_each_within<M, V>(&self, p: &[F; 3], radius: F, mut metric: M, mut visit: V)
    where
        M: FnMut(u32) -> F,
        V: FnMut(u32, F),
    {
        if !self.order.is_empty() {
            self.for_each_within_in(0, p, radius, &mut metric, &mut visit);
        }
    }

    fn for_each_within_in<M, V>(
        &self,
        n: usize,
        p: &[F; 3],
        radius: F,
        metric: &mut M,
        visit: &mut V,
    ) where
        M: FnMut(u32) -> F,
        V: FnMut(u32, F),
    {
        let node = &self.nodes[n];
        if box_dist(node, p) > radius.max(0.0) {
            return;
        }
        if node.count > 0 {
            let end = (node.first + node.count) as usize;
            for &i in &self.order[node.first as usize..end] {
                let d = metric(i);
                if d <= radius {
                    visit(i, d);
                }
            }
            return;
        }
        let left = node.first as usize;
        self.for_each_within_in(left, p, radius, metric, visit);
        self.for_each_within_in(left + 1, p, radius, metric, visit);
    }

    /// How many items `hit` reports along the ray from `origin`.
    ///
    /// `inv` is the componentwise reciprocal of the direction; the caller owns
    /// the direction, so it also owns keeping it free of zero components.
    pub(crate) fn count_hits<H>(&self, origin: &[F; 3], inv: &[F; 3], mut hit: H) -> u32
    where
        H: FnMut(u32) -> bool,
    {
        if self.order.is_empty() {
            return 0;
        }
        self.count_hits_in(0, origin, inv, &mut hit)
    }

    fn count_hits_in<H>(&self, n: usize, origin: &[F; 3], inv: &[F; 3], hit: &mut H) -> u32
    where
        H: FnMut(u32) -> bool,
    {
        let node = &self.nodes[n];
        if !ray_hits_box(node, origin, inv) {
            return 0;
        }
        if node.count > 0 {
            let end = (node.first + node.count) as usize;
            return self.order[node.first as usize..end]
                .iter()
                .filter(|&&i| hit(i))
                .count() as u32;
        }
        let left = node.first as usize;
        self.count_hits_in(left, origin, inv, hit) + self.count_hits_in(left + 1, origin, inv, hit)
    }
}

/// Fill node `n` with the bounds of `order[start..start + count]`, splitting
/// until a leaf is small enough.
fn build_node(
    nodes: &mut Vec<Node>,
    order: &mut [u32],
    items: &[([F; 3], [F; 3])],
    centroids: &[[F; 3]],
    n: usize,
    start: usize,
    count: usize,
) {
    let (min, max) = bounds(items, &order[start..start + count]);
    nodes[n].min = min;
    nodes[n].max = max;
    if count <= LEAF_SIZE {
        nodes[n].first = start as u32;
        nodes[n].count = count as u32;
        return;
    }
    // Median split on the widest spread of centroids: no surface-area
    // heuristic, because a mesh from a mesher or a bead cloud from a
    // simulation is already spatially coherent and the extra build cost buys
    // nothing measurable.
    let axis = widest_axis(centroids, &order[start..start + count]);
    let half = count / 2;
    order[start..start + count].select_nth_unstable_by(half, |&a, &b| {
        centroids[a as usize][axis].total_cmp(&centroids[b as usize][axis])
    });
    let left = nodes.len();
    nodes.push(Node::default());
    nodes.push(Node::default());
    nodes[n].first = left as u32;
    nodes[n].count = 0;
    build_node(nodes, order, items, centroids, left, start, half);
    build_node(
        nodes,
        order,
        items,
        centroids,
        left + 1,
        start + half,
        count - half,
    );
}

fn bounds(items: &[([F; 3], [F; 3])], idx: &[u32]) -> ([F; 3], [F; 3]) {
    let mut min = [F::INFINITY; 3];
    let mut max = [F::NEG_INFINITY; 3];
    for &i in idx {
        let (lo, hi) = &items[i as usize];
        for k in 0..3 {
            min[k] = min[k].min(lo[k]);
            max[k] = max[k].max(hi[k]);
        }
    }
    (min, max)
}

fn widest_axis(centroids: &[[F; 3]], idx: &[u32]) -> usize {
    let mut min = [F::INFINITY; 3];
    let mut max = [F::NEG_INFINITY; 3];
    for &i in idx {
        let c = centroids[i as usize];
        for k in 0..3 {
            min[k] = min[k].min(c[k]);
            max[k] = max[k].max(c[k]);
        }
    }
    let mut axis = 0;
    let mut span = max[0] - min[0];
    for (k, s) in (1..3).map(|k| (k, max[k] - min[k])) {
        if s > span {
            span = s;
            axis = k;
        }
    }
    axis
}

/// Distance from `p` to a node box; `0` when `p` is inside it.
fn box_dist(node: &Node, p: &[F; 3]) -> F {
    let mut d2 = 0.0;
    for (k, &pk) in p.iter().enumerate() {
        let v = if pk < node.min[k] {
            node.min[k] - pk
        } else if pk > node.max[k] {
            pk - node.max[k]
        } else {
            continue;
        };
        d2 += v * v;
    }
    d2.sqrt()
}

/// Slab test for a ray that starts at `origin` and never turns back.
fn ray_hits_box(node: &Node, origin: &[F; 3], inv: &[F; 3]) -> bool {
    let mut near: F = 0.0;
    let mut far = F::INFINITY;
    for (k, (&o, &iv)) in origin.iter().zip(inv.iter()).enumerate() {
        let a = (node.min[k] - o) * iv;
        let b = (node.max[k] - o) * iv;
        let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
        near = near.max(lo);
        far = far.min(hi);
        if far < near {
            return false;
        }
    }
    true
}

/// Box of one triangle, the item shape the mesh region indexes.
pub(crate) fn triangle_box(t: &[[F; 3]; 3]) -> ([F; 3], [F; 3]) {
    let mut min = t[0];
    let mut max = t[0];
    for v in &t[1..] {
        for k in 0..3 {
            min[k] = min[k].min(v[k]);
            max[k] = max[k].max(v[k]);
        }
    }
    (min, max)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::op::vec3::{dot, sub};

    /// Deterministic triangle soup — no rand dependency in a unit test.
    fn soup(n: usize) -> Vec<[[F; 3]; 3]> {
        let mut s: u64 = 0x2545_F491_4F6C_DD1D;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 11) as F / (1u64 << 53) as F
        };
        (0..n)
            .map(|_| {
                let o = [next() * 20.0, next() * 20.0, next() * 20.0];
                let mut v = || {
                    [
                        o[0] + next() * 3.0,
                        o[1] + next() * 3.0,
                        o[2] + next() * 3.0,
                    ]
                };
                [v(), v(), v()]
            })
            .collect()
    }

    fn dist2(a: [F; 3], b: [F; 3]) -> F {
        let d = sub(a, b);
        dot(d, d)
    }

    /// Closest point on a triangle by dense sampling: crude, but this test
    /// only needs a reference the tree must reproduce, not a fast one.
    fn closest(p: [F; 3], t: &[[F; 3]; 3]) -> [F; 3] {
        let mut best = t[0];
        let mut best_d = dist2(p, t[0]);
        let n = 24;
        for i in 0..=n {
            for j in 0..=(n - i) {
                let (u, v) = (i as F / n as F, j as F / n as F);
                let w = 1.0 - u - v;
                let q = [
                    u * t[0][0] + v * t[1][0] + w * t[2][0],
                    u * t[0][1] + v * t[1][1] + w * t[2][1],
                    u * t[0][2] + v * t[1][2] + w * t[2][2],
                ];
                let d = dist2(p, q);
                if d < best_d {
                    best_d = d;
                    best = q;
                }
            }
        }
        best
    }

    fn tree_of(tris: &[[[F; 3]; 3]]) -> Bvh {
        let boxes: Vec<_> = tris.iter().map(triangle_box).collect();
        Bvh::build(&boxes)
    }

    #[test]
    fn nearest_matches_the_linear_scan() {
        let tris = soup(300);
        let bvh = tree_of(&tris);
        for p in [
            [0.0, 0.0, 0.0],
            [10.0, 10.0, 10.0],
            [-5.0, 22.0, 3.0],
            [21.5, 21.5, 21.5],
        ] {
            let (_, d) = bvh
                .nearest(&p, |i| dist2(p, closest(p, &tris[i as usize])).sqrt())
                .unwrap();
            let brute = tris
                .iter()
                .map(|t| dist2(p, closest(p, t)).sqrt())
                .fold(F::INFINITY, F::min);
            assert!(
                (d - brute).abs() < 1e-12,
                "bvh {d} vs scan {brute} at {p:?}"
            );
        }
    }

    /// Spheres boxed as `c ± r` with the signed metric `‖p − c‖ − r`: once
    /// the query is inside one sphere the answer is the deepest one, which
    /// the box-containment prune must still reach.
    #[test]
    fn nearest_with_negative_metric_finds_the_deepest_sphere() {
        let centers: Vec<[F; 3]> = (0..40)
            .map(|i| {
                let t = i as F;
                [(t * 0.37) % 6.0, (t * 0.71) % 6.0, (t * 0.13) % 6.0]
            })
            .collect();
        let radii: Vec<F> = (0..40).map(|i| 0.5 + (i % 5) as F * 0.4).collect();
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
        for p in [
            [3.0, 3.0, 3.0],
            [0.1, 0.2, 0.3],
            [5.9, 0.1, 5.5],
            [9.0, 9.0, 9.0],
        ] {
            let metric = |i: u32| dist2(p, centers[i as usize]).sqrt() - radii[i as usize];
            let (_, d) = bvh.nearest(&p, metric).unwrap();
            let brute = (0..40u32).map(metric).fold(F::INFINITY, F::min);
            assert!(
                (d - brute).abs() < 1e-12,
                "bvh {d} vs scan {brute} at {p:?}"
            );
            assert_eq!(bvh.any_within(&p, 0.0, metric), brute <= 0.0);
        }
    }

    /// Möller–Trumbore, forward hits only — the same shape of test the region
    /// runs, so the tree may only skip what this would have rejected anyway.
    fn ray_hits(o: [F; 3], d: [F; 3], t: &[[F; 3]; 3]) -> bool {
        use crate::op::vec3::cross;
        let (e1, e2) = (sub(t[1], t[0]), sub(t[2], t[0]));
        let h = cross(d, e2);
        let a = dot(e1, h);
        if a.abs() < 1e-12 {
            return false;
        }
        let f = 1.0 / a;
        let s = sub(o, t[0]);
        let u = f * dot(s, h);
        if !(0.0..=1.0).contains(&u) {
            return false;
        }
        let q = cross(s, e1);
        let v = f * dot(d, q);
        if v < 0.0 || u + v > 1.0 {
            return false;
        }
        f * dot(e2, q) > 1e-12
    }

    #[test]
    fn count_hits_matches_the_linear_scan() {
        let tris = soup(300);
        let bvh = tree_of(&tris);
        let dir = {
            let d = [1.0, std::f64::consts::SQRT_2, std::f64::consts::PI];
            let n = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
            [d[0] / n, d[1] / n, d[2] / n]
        };
        let inv = [1.0 / dir[0], 1.0 / dir[1], 1.0 / dir[2]];
        for origin in [
            [0.0, 0.0, 0.0],
            [10.0, 10.0, 10.0],
            [-3.0, 7.5, 12.0],
            [19.0, 2.0, 8.0],
        ] {
            let counted =
                bvh.count_hits(&origin, &inv, |i| ray_hits(origin, dir, &tris[i as usize]));
            let brute = tris.iter().filter(|t| ray_hits(origin, dir, t)).count() as u32;
            assert_eq!(counted, brute, "at {origin:?}");
        }
    }

    #[test]
    fn box_distance_is_zero_inside_and_euclidean_outside() {
        let bvh = Bvh::build(&[([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])]);
        let node = &bvh.nodes[0];
        assert!(box_dist(node, &[0.5, 0.5, 0.5]).abs() < 1e-12);
        assert!((box_dist(node, &[2.0, 0.5, 0.5]) - 1.0).abs() < 1e-12);
        assert!((box_dist(node, &[2.0, 2.0, 2.0]) - (3.0 as F).sqrt()).abs() < 1e-12);
    }

    #[test]
    fn knn_matches_the_linear_scan() {
        // Points on a line at x = 0..4; the metric is the Euclidean distance.
        let pts: Vec<[F; 3]> = (0..5).map(|i| [i as F, 0.0, 0.0]).collect();
        let boxes: Vec<([F; 3], [F; 3])> = pts.iter().map(|&p| (p, p)).collect();
        let bvh = Bvh::build(&boxes);
        let q = [0.2, 0.0, 0.0];
        let metric = |i: u32| dist2(q, pts[i as usize]).sqrt();

        let got = bvh.knn(&q, 3, metric);
        let mut want: Vec<(u32, F)> = (0..5u32).map(|i| (i, metric(i))).collect();
        want.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        want.truncate(3);
        assert_eq!(got.len(), 3);
        for (g, w) in got.iter().zip(&want) {
            assert_eq!(g.0, w.0);
            assert!((g.1 - w.1).abs() < 1e-12);
        }
    }

    #[test]
    fn knn_asking_for_more_than_there_is_returns_everything() {
        let boxes = [
            ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
            ([1.0, 0.0, 0.0], [1.0, 0.0, 0.0]),
        ];
        let bvh = Bvh::build(&boxes);
        let got = bvh.knn(&[0.0, 0.0, 0.0], 10, |i| i as F);
        assert_eq!(got.len(), 2);
        assert!(Bvh::build(&[]).knn(&[0.0, 0.0, 0.0], 3, |_| 0.0).is_empty());
    }

    #[test]
    fn a_single_item_is_one_leaf_and_empty_is_none() {
        let tris = vec![[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]];
        let bvh = tree_of(&tris);
        let p = [0.0, 0.0, 2.0];
        let (i, d) = bvh
            .nearest(&p, |i| dist2(p, closest(p, &tris[i as usize])).sqrt())
            .unwrap();
        assert_eq!(i, 0);
        assert!((d - 2.0).abs() < 1e-12);
        assert!(Bvh::build(&[]).nearest(&p, |_| 0.0).is_none());
        assert!(!Bvh::build(&[]).any_within(&p, 1.0, |_| 0.0));
    }
}
