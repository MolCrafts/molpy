//! Native periodic radical (Laguerre) Voronoi tessellation.

use molrs::core::neighbors::Backend;
use molrs::core::{BoxKind, GhostSet, ImageRange, LinkCell, SimBox};
use molrs::op::F;
use ndarray::{Array2, ArrayView1, ArrayView2, array};

use super::cell::{Poly, VoronoiCells, VoronoiFace};
use crate::compute::{Compute, ComputeError};

/// The periodic radical-Voronoi tessellation of one snapshot. Orthorhombic,
/// fully periodic boxes only.
///
/// Configuration: none. Data: the generator positions, their box and one
/// radius per generator (all zero for a plain Voronoi diagram).
///
/// Builds one [`VoronoiCells`] entry per generator by clipping a box-sized
/// convex polyhedron against the radical plane of nearby periodic neighbours
/// (`Poly::clip`). This is the cell-by-cell strategy of voro++
/// (`container_periodic::compute_cell`, `src/v_container_prd.cpp`, driven as in
/// reference implementation `vorowrapper.cpp`), with the radical/power plane offset from
/// `radius_poly` (`src/v_rad_option.h`).
///
/// # Candidate search — O(N) at constant density
/// voro++ only clips a cell against particles near enough that their cut plane
/// can still reach the cell; it walks its container blocks outward and stops
/// once no closer plane can cut. molrs mirrors this with the core neighbour
/// search: the periodic copies within the search radius of the box are
/// materialised once ([`GhostSet`]) and indexed by a core cell list
/// ([`LinkCell`]) over the open box that holds them, so each generator's
/// candidates — *every* periodic image within the radius, not only the
/// minimum image, which a small box needs — come in O(1). The cell is clipped
/// in increasing candidate distance until the **termination bound** is
/// reached:
///
/// A neighbour `j` at centre distance `d` cuts the cell along the radical plane
/// at signed distance `p_j = (d² + Rᵢ² − Rⱼ²) / (2d)` from the generator. If
/// `p_j` exceeds the cell's farthest vertex radius `R_max`, the plane lies
/// wholly outside the cell and cannot cut it. Taking the largest radius in the
/// system `R_⋆` as the conservative worst case (`Rⱼ = R_⋆`), no neighbour past
/// `d_stop = R_max + √(R_max² + R_⋆² − Rᵢ²)` can ever cut the cell, so the
/// distance-ordered walk stops there. For equal radii this is the familiar
/// `d > 2·R_max` Voronoi cutoff. `R_max` only shrinks as clipping proceeds, so
/// the bound is monotone and safe.
///
/// If the search radius turns out too small to reach `d_stop` for a cell
/// (sparse regions, large voids, a slab's vacuum gap), that cell is rebuilt
/// exhaustively: every periodic image of every generator within the `d_stop`
/// of the cell the fast path left — which contains the true cell, so no plane
/// beyond it can cut — is clipped in a fixed order. The plane-cut semantics are
/// the same, so the result is identical; only the *candidate set* differs, and
/// each path is accepted only once it provably contains every cutting plane.
/// A cell whose exhaustive reach would need more periodic images than the core
/// enumeration allows ([`ImageRange::MAX_IMAGES`]) is an error, never a cell
/// built from a truncated image set. Every cell is bounded by real
/// neighbours (its own periodic images at the least), so no initial-box face
/// survives. Correctness is pinned by `Σ volume == box volume` and the
/// analytic radical-plane / single-atom / exhaustive-path tests.
///
/// # Radical plane
/// Between generators i, j at displacement `r = x_j − x_i` with radii `Rᵢ`,
/// `Rⱼ`, the radical plane is `|x|² − Rᵢ² = |x − r|² − Rⱼ²`, i.e.
/// `x·r = (|r|² + Rᵢ² − Rⱼ²)/2`; the cell keeps the generator side
/// (`x·r ≤ off`). Equal radii reduce to the plain Voronoi bisector at `r/2`.
///
/// ```
/// use molrs::compute::{Compute, RadicalVoronoi, RadicalVoronoiInput};
/// use molrs::core::SimBox;
/// use ndarray::array;
///
/// let simbox = SimBox::cube(4.0, array![0.0, 0.0, 0.0], [true; 3]).unwrap();
/// let positions = array![[1.0, 2.0, 2.0], [3.0, 2.0, 2.0]];
/// let cells = RadicalVoronoi::new().compute(RadicalVoronoiInput {
///     positions: positions.view(),
///     simbox: &simbox,
///     radii: array![0.0, 0.0].view(),
/// })?;
/// assert!((cells.total_volume() - 64.0).abs() < 1e-9);
/// # Ok::<(), molrs::compute::ComputeError>(())
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct RadicalVoronoi;

/// The data a [`RadicalVoronoi`] tessellates.
#[derive(Clone, Copy)]
pub struct RadicalVoronoiInput<'a> {
    /// Generator positions, `N × 3` (Å); wrapped into the box internally.
    pub positions: ArrayView2<'a, F>,
    /// The periodic orthorhombic box.
    pub simbox: &'a SimBox,
    /// One radius per generator (Å); all zero gives the plain Voronoi diagram.
    pub radii: ArrayView1<'a, F>,
}

/// One clipping candidate: `(d², displacement, generator, image key)`. The
/// image key — the search-point index (an owned point or one periodic copy) on
/// the fast path, `generator × n_shifts + shift` on the exhaustive one — makes
/// the distance order total, so equal distances clip in a fixed order.
type Candidate = (F, [F; 3], usize, u64);

/// One finished cell: its volume (Å³) and bounding faces.
type Cell = (F, Vec<VoronoiFace>);

/// Conservative termination radius: no neighbour farther than this can still
/// cut a cell whose farthest vertex is at `rmax`. `r_star2` is the largest
/// squared radius in the system (worst-case neighbour), `ri2` the generator's.
#[inline]
fn d_stop(rmax: F, ri2: F, r_star2: F) -> F {
    let disc = rmax * rmax + r_star2 - ri2;
    rmax + disc.max(0.0).sqrt()
}

impl RadicalVoronoi {
    /// No configuration.
    pub fn new() -> Self {
        Self
    }
}

impl Compute for RadicalVoronoi {
    type Input<'a> = RadicalVoronoiInput<'a>;
    type Output = VoronoiCells;

    /// # Errors
    /// [`ComputeError::DimensionMismatch`] when the positions are not `N × 3`
    /// or the radii are not one per generator; [`ComputeError::NonFinite`] for
    /// a non-finite position or radius; [`ComputeError::BadShape`] for a
    /// triclinic box or one that is not periodic along every axis (the
    /// tessellation is periodic and orthorhombic by construction);
    /// [`ComputeError::OutOfRange`] when the search radius, or the exhaustive
    /// reach of a cell the search radius cannot prove complete, would need
    /// more periodic copies than the core image enumeration allows.
    fn compute(&self, input: RadicalVoronoiInput<'_>) -> Result<VoronoiCells, ComputeError> {
        let RadicalVoronoiInput {
            positions,
            simbox,
            radii,
        } = input;
        let n = positions.nrows();
        if positions.ncols() != 3 {
            return Err(ComputeError::DimensionMismatch {
                expected: 3,
                got: positions.ncols(),
                what: "positions columns",
            });
        }
        if radii.len() != n {
            return Err(ComputeError::DimensionMismatch {
                expected: n,
                got: radii.len(),
                what: "radii length",
            });
        }
        if let Some(index) = positions.iter().position(|x| !x.is_finite()) {
            return Err(ComputeError::NonFinite {
                where_: "positions",
                index,
            });
        }
        if let Some(index) = radii.iter().position(|r| !r.is_finite()) {
            return Err(ComputeError::NonFinite {
                where_: "radii",
                index,
            });
        }
        let BoxKind::Ortho { len, .. } = simbox.kind() else {
            return Err(ComputeError::BadShape {
                expected: "an orthorhombic box".into(),
                got: "a triclinic box".into(),
            });
        };
        if simbox.pbc() != [true; 3] {
            return Err(ComputeError::BadShape {
                expected: "a box periodic along every axis".into(),
                got: format!("pbc {:?}", simbox.pbc()),
            });
        }
        let l = [len[0], len[1], len[2]];

        if n == 0 {
            return Ok(VoronoiCells {
                volumes: Vec::new(),
                faces: Vec::new(),
            });
        }

        let radii: Vec<F> = radii.to_vec();
        let r_star2 = radii.iter().map(|&r| r * r).fold(0.0, F::max);

        // Search radius: a few mean spacings plus the radius span comfortably
        // bounds a liquid-like Voronoi cell; cells that need more fall back.
        let mean_spacing = (l[0] * l[1] * l[2] / n as F).cbrt();
        let r_cut = 4.0 * mean_spacing + 2.0 * r_star2.sqrt();
        let search = ImageSearch::new(positions, &l, r_cut)?;

        // Each generator's cell is built independently from the shared read-only
        // search, so the per-cell loop is embarrassingly parallel: `map_init` gives
        // every worker its own reusable `cand` scratch buffer (matching the serial
        // buffer reuse) and `collect` preserves generator order, so the result is
        // bit-identical to the serial build — only the geometry algorithm's driver
        // loop is parallelized, never the geometry itself.
        let build = |i: usize, cand: &mut Vec<Candidate>| -> Result<Cell, ComputeError> {
            let ri2 = radii[i] * radii[i];
            match build_cell_fast(&search, i, ri2, r_star2, &radii, &l, r_cut, cand) {
                Ok(cell) => Ok(cell),
                Err(reach) => {
                    build_cell_exhaustive(&search, i, ri2, r_star2, &radii, &l, reach, cand)
                }
            }
        };

        // Collected in generator order, then the first failing cell's error —
        // so the outcome never depends on scheduling.
        #[cfg(feature = "rayon")]
        let cells: Vec<Result<Cell, ComputeError>> = {
            use rayon::prelude::*;
            (0..n)
                .into_par_iter()
                .map_init(Vec::new, |cand, i| build(i, cand))
                .collect()
        };
        #[cfg(not(feature = "rayon"))]
        let cells: Vec<Result<Cell, ComputeError>> = {
            let mut cand: Vec<Candidate> = Vec::new();
            (0..n).map(|i| build(i, &mut cand)).collect()
        };
        let cells: Vec<Cell> = cells.into_iter().collect::<Result<_, _>>()?;

        let mut volumes = Vec::with_capacity(n);
        let mut faces = Vec::with_capacity(n);
        for (vol, cf) in cells {
            volumes.push(vol);
            faces.push(cf);
        }

        Ok(VoronoiCells { volumes, faces })
    }
}

/// Fast path: clip against the search candidates in increasing distance,
/// stopping at the termination bound. When the search radius `r_cut` cannot be
/// proven to contain every cutting plane, returns `Err(reach)`: the
/// termination bound of the cell clipped so far, which contains the true cell,
/// so every plane that can still cut comes from within `reach`.
#[allow(clippy::too_many_arguments)]
fn build_cell_fast(
    search: &ImageSearch,
    i: usize,
    ri2: F,
    r_star2: F,
    radii: &[F],
    l: &[F; 3],
    r_cut: F,
    cand: &mut Vec<Candidate>,
) -> Result<Cell, F> {
    search.collect(i, r_cut, cand);
    let poly = clip_nearest_first(cand, ri2, r_star2, radii, l);

    // Accept only if r_cut provably reached the termination bound, i.e. no
    // unseen neighbour beyond r_cut could still cut this cell.
    let reach = d_stop(poly.max_vertex_dist(), ri2, r_star2);
    if r_cut + 1e-12 >= reach {
        Ok((poly.volume(), poly.cell_faces()))
    } else {
        Err(reach)
    }
}

/// Clip a box-sized cell against `cand` in increasing distance (ties by image
/// key), stopping at the termination bound — no farther candidate can cut a
/// cell whose bound has already been passed, and the bound only shrinks.
fn clip_nearest_first(cand: &mut [Candidate], ri2: F, r_star2: F, radii: &[F], l: &[F; 3]) -> Poly {
    cand.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.3.cmp(&b.3)));
    let mut poly = Poly::box_cell(l[0], l[1], l[2]);
    for &(d2, dr, j, _) in cand.iter() {
        let d = d2.sqrt();
        if d > d_stop(poly.max_vertex_dist(), ri2, r_star2) {
            break;
        }
        let rj2 = radii[j] * radii[j];
        let off = 0.5 * (d2 + ri2 - rj2);
        poly.clip(dr, off, j as i64);
    }
    poly
}

/// Exact exhaustive fallback for a cell the search radius could not prove
/// complete: the same nearest-first clipping over every periodic image, within
/// `reach` of the generator, of every generator.
///
/// The images come from the core [`ImageRange`] for `reach`: with every
/// generator wrapped into `[0, L)`, an image within `reach` along an axis is
/// at most `⌈reach / L⌉` cells away, which is the range's half-width. `reach`
/// is the termination bound of a cell that contains the true one, so every
/// plane that can cut is among these candidates.
///
/// # Errors
/// [`ComputeError::OutOfRange`] when the range would exceed
/// [`ImageRange::MAX_IMAGES`] — a cell is never built from a truncated image
/// set.
#[allow(clippy::too_many_arguments)]
fn build_cell_exhaustive(
    search: &ImageSearch,
    i: usize,
    ri2: F,
    r_star2: F,
    radii: &[F],
    l: &[F; 3],
    reach: F,
    cand: &mut Vec<Candidate>,
) -> Result<Cell, ComputeError> {
    let images = ImageRange::new(&search.cell, reach).map_err(|e| ComputeError::OutOfRange {
        field: "RadicalVoronoi exhaustive reach",
        value: format!("{reach} for generator {i}: {e}"),
    })?;
    let shifts = images.shifts();
    let gi = search.wrapped(i);
    let reach2 = reach * reach;
    cand.clear();
    for j in 0..radii.len() {
        let pj = search.wrapped(j);
        for (k, &[sx, sy, sz]) in shifts.iter().enumerate() {
            let dr = [
                pj[0] + sx as F * l[0] - gi[0],
                pj[1] + sy as F * l[1] - gi[1],
                pj[2] + sz as F * l[2] - gi[2],
            ];
            let d2 = dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2];
            // The generator itself (zero distance) and images past the reach
            // are not candidates.
            if d2 < 1e-18 || d2 > reach2 {
                continue;
            }
            cand.push((d2, dr, j, (j * shifts.len() + k) as u64));
        }
    }
    let poly = clip_nearest_first(cand, ri2, r_star2, radii, l);
    Ok((poly.volume(), poly.cell_faces()))
}

/// The generators wrapped into `[0, L)` plus every periodic copy within the
/// search radius of the box, indexed by a core [`LinkCell`] over the open box
/// that holds them all. Rows `0..n` of the point set are the generators, the
/// rest their copies.
struct ImageSearch {
    /// `[owned | copies]`, `n_all × 3` (Å).
    points: Array2<F>,
    /// Owning generator of every copy (row `n + g` copies `owner[g]`).
    owner: Vec<u32>,
    n_owned: usize,
    index: LinkCell,
    open_box: SimBox,
    /// The periodic cell `[0, L)` the generators are wrapped into.
    cell: SimBox,
}

impl ImageSearch {
    fn new(positions: ArrayView2<F>, l: &[F; 3], r_cut: F) -> Result<Self, ComputeError> {
        let n = positions.nrows();
        let mut wrapped = Array2::<F>::zeros((n, 3));
        for j in 0..n {
            for (a, &la) in l.iter().enumerate() {
                let mut p = positions[[j, a]].rem_euclid(la);
                // guard the rem_euclid == L edge from rounding
                if p >= la {
                    p -= la;
                }
                wrapped[[j, a]] = p;
            }
        }
        // The copies carry the periodicity: an origin-at-zero periodic cell for
        // the wrapped coordinates, then a search over an open box.
        let out_of_range = |e: &dyn std::fmt::Debug| ComputeError::OutOfRange {
            field: "RadicalVoronoi search radius",
            value: format!("{r_cut}: {e:?}"),
        };
        let cell = SimBox::ortho(array![l[0], l[1], l[2]], array![0.0, 0.0, 0.0], [true; 3])
            .map_err(|e| out_of_range(&e))?;
        let ghosts =
            GhostSet::borders(&cell, wrapped.view(), r_cut).map_err(|e| out_of_range(&e))?;
        let points = ghosts
            .combined(wrapped.view())
            .map_err(|e| out_of_range(&e))?;
        let open_box = SimBox::free(points.view(), r_cut).map_err(|e| out_of_range(&e))?;
        let mut index = LinkCell::new().cutoff(r_cut);
        index.build_index(points.view(), &open_box);
        Ok(Self {
            owner: ghosts.owner().to_vec(),
            points,
            n_owned: n,
            index,
            open_box,
            cell,
        })
    }

    /// Generator `i`, wrapped into `[0, L)`.
    fn wrapped(&self, i: usize) -> [F; 3] {
        [
            self.points[[i, 0]],
            self.points[[i, 1]],
            self.points[[i, 2]],
        ]
    }

    /// Collect every point (generator or copy) within `r_cut` of generator `i`
    /// as a [`Candidate`], excluding the generator itself (zero distance).
    /// `out` is cleared first.
    fn collect(&self, i: usize, r_cut: F, out: &mut Vec<Candidate>) {
        out.clear();
        let rc2 = r_cut * r_cut;
        self.index
            .visit_neighbors_of_pt(self.wrapped(i), &self.open_box, |p, d2, dr| {
                if d2 < 1e-18 || d2 > rc2 {
                    return;
                }
                let k = p as usize;
                let j = if k < self.n_owned {
                    k
                } else {
                    self.owner[k - self.n_owned] as usize
                };
                out.push((d2, dr, j, u64::from(p)));
            });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use molrs::core::SimBox;
    use ndarray::{Array1, array};

    fn periodic(l: [F; 3]) -> SimBox {
        SimBox::ortho(array![l[0], l[1], l[2]], array![0.0, 0.0, 0.0], [true; 3]).unwrap()
    }

    fn tessellate(positions: &Array2<F>, radii: &[F], simbox: &SimBox) -> VoronoiCells {
        RadicalVoronoi::new()
            .compute(RadicalVoronoiInput {
                positions: positions.view(),
                simbox,
                radii: ArrayView1::from(radii),
            })
            .unwrap()
    }

    /// A lone generator's cell is the whole box, bounded by six copies of
    /// itself.
    #[test]
    fn a_single_generator_fills_the_box() {
        let cells = tessellate(&array![[1.0, 2.0, 3.0]], &[0.5], &periodic([5.0, 6.0, 7.0]));
        assert!((cells.volumes[0] - 210.0).abs() < 1e-9);
        assert_eq!(cells.faces[0].len(), 6);
        assert!(cells.faces[0].iter().all(|f| f.neighbor == 0));
    }

    /// A simple cubic lattice: every cell is the unit cube with six faces.
    #[test]
    fn simple_cubic_cells_are_unit_cubes() {
        let mut p = Array2::zeros((27, 3));
        let mut m = 0;
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    p.row_mut(m)
                        .assign(&array![i as F + 0.5, j as F + 0.5, k as F + 0.5]);
                    m += 1;
                }
            }
        }
        let cells = tessellate(&p, &[0.0; 27], &periodic([3.0; 3]));
        for i in 0..27 {
            assert!((cells.volumes[i] - 1.0).abs() < 1e-12);
            assert_eq!(cells.neighbors(i).len(), 6);
        }
    }

    /// Two generators on the x axis: the radical plane sits at
    /// `x·r = (|r|² + R₁² − R₂²)/2`, so the volume split follows the radii.
    #[test]
    fn the_radical_plane_splits_by_power_distance() {
        let l = 4.0;
        let p = array![[1.0, 2.0, 2.0], [3.0, 2.0, 2.0]];
        let (r1, r2): (F, F) = (1.0, 0.5);
        let cells = tessellate(&p, &[r1, r2], &periodic([l; 3]));
        // Plane offsets from generator 0 along +x (d = 2) and −x (d = 2, image).
        let off = (4.0 + r1 * r1 - r2 * r2) / 4.0;
        let width0 = 2.0 * off;
        assert!((cells.volumes[0] - width0 * l * l).abs() < 1e-9);
        assert!((cells.total_volume() - l * l * l).abs() < 1e-9);
    }

    /// The headline invariant on a polydisperse random packing, generators
    /// partly outside the box: the cells tile the box.
    #[test]
    fn cells_tile_the_box() {
        let l = [10.0, 12.0, 9.0];
        let mut state = 42u64;
        let mut rng = || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((state >> 11) as F) / ((1u64 << 53) as F)
        };
        let n = 60;
        let mut p = Array2::zeros((n, 3));
        for i in 0..n {
            for k in 0..3 {
                p[[i, k]] = rng() * l[k] * 1.4 - 0.2 * l[k];
            }
        }
        let radii: Vec<F> = (0..n).map(|_| rng() * 1.2).collect();
        let cells = tessellate(&p, &radii, &periodic(l));
        assert!((cells.total_volume() - l[0] * l[1] * l[2]).abs() < 1e-8);
        for i in 0..n {
            for j in cells.neighbors(i) {
                assert!(
                    cells.neighbors(j as usize).contains(&(i as i64)),
                    "adjacency is symmetric"
                );
            }
        }
    }

    /// Two generators stacked in a tall box: each cell is a 4 × 4 × 20 column
    /// reaching across the empty half of the box, too long for the search
    /// radius to prove, so both cells are built on the exhaustive path — and
    /// are still exact, bounded by real neighbours only, and unchanged when the
    /// generators are given whole boxes away.
    #[test]
    fn sparse_cells_are_exact_on_the_exhaustive_path() {
        let simbox = periodic([4.0, 4.0, 40.0]);
        let p = array![[2.0, 2.0, 1.0], [2.0, 2.0, 3.0]];
        let radii = [0.0, 0.0];

        // The fast path cannot prove generator 0's cell (its search radius is
        // 4 · (640 / 2)^(1/3) ≈ 27 Å; the column needs ≈ 80 Å).
        let l = [4.0, 4.0, 40.0];
        let r_cut = 4.0 * (640.0_f64 / 2.0).cbrt();
        let search = ImageSearch::new(p.view(), &l, r_cut).unwrap();
        let mut cand = Vec::new();
        assert!(build_cell_fast(&search, 0, 0.0, 0.0, &radii, &l, r_cut, &mut cand).is_err());

        let cells = tessellate(&p, &radii, &simbox);
        for i in 0..2 {
            assert!(
                (cells.volumes[i] - 320.0).abs() < 1e-9,
                "{:?}",
                cells.volumes
            );
            assert!(cells.faces[i].iter().all(|f| f.neighbor >= 0));
        }
        assert_eq!(cells.neighbors(0), vec![0, 1]);

        let shifted = array![[-2.0, 6.0, 81.0], [2.0, 2.0, -77.0]];
        let moved = tessellate(&shifted, &radii, &simbox);
        assert_eq!(moved.volumes, cells.volumes);
    }

    /// A cell whose exhaustive reach would need more periodic images than the
    /// core enumeration allows is an error, not a cell from a truncated set.
    #[test]
    fn an_exhaustive_reach_beyond_the_image_limit_is_an_error() {
        // 2 × 2 × 200: each column is ~100 Å long, so proving it needs images
        // ~200 boxes away along x and y.
        let p = array![[1.0, 1.0, 1.0], [1.0, 1.0, 3.0]];
        let err = RadicalVoronoi::new()
            .compute(RadicalVoronoiInput {
                positions: p.view(),
                simbox: &periodic([2.0, 2.0, 200.0]),
                radii: array![0.0, 0.0].view(),
            })
            .unwrap_err();
        assert!(
            matches!(
                err,
                ComputeError::OutOfRange {
                    field: "RadicalVoronoi exhaustive reach",
                    ..
                }
            ),
            "{err:?}"
        );
    }

    #[test]
    fn non_finite_input_is_an_error() {
        let simbox = periodic([4.0; 3]);
        let mut p = array![[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]];
        p[[1, 2]] = F::NAN;
        let err = RadicalVoronoi::new()
            .compute(RadicalVoronoiInput {
                positions: p.view(),
                simbox: &simbox,
                radii: Array1::zeros(2).view(),
            })
            .unwrap_err();
        assert!(matches!(err, ComputeError::NonFinite { index: 5, .. }));

        let p = array![[1.0, 1.0, 1.0]];
        let err = RadicalVoronoi::new()
            .compute(RadicalVoronoiInput {
                positions: p.view(),
                simbox: &simbox,
                radii: array![F::INFINITY].view(),
            })
            .unwrap_err();
        assert!(matches!(
            err,
            ComputeError::NonFinite {
                where_: "radii",
                ..
            }
        ));
    }

    #[test]
    fn a_box_that_is_not_periodic_and_orthorhombic_is_an_error() {
        let p = array![[1.0, 1.0, 1.0]];
        let radii = array![0.0];
        let open = SimBox::ortho(
            array![4.0, 4.0, 4.0],
            array![0.0, 0.0, 0.0],
            [true, false, true],
        )
        .unwrap();
        let h = array![[4.0, 1.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]];
        let tilted = SimBox::new(h, array![0.0, 0.0, 0.0], [true; 3]).unwrap();
        for simbox in [&open, &tilted] {
            let err = RadicalVoronoi::new()
                .compute(RadicalVoronoiInput {
                    positions: p.view(),
                    simbox,
                    radii: radii.view(),
                })
                .unwrap_err();
            assert!(matches!(err, ComputeError::BadShape { .. }), "{err:?}");
        }
    }

    #[test]
    fn mismatched_radii_is_an_error() {
        let err = RadicalVoronoi::new()
            .compute(RadicalVoronoiInput {
                positions: array![[1.0, 1.0, 1.0]].view(),
                simbox: &periodic([4.0; 3]),
                radii: array![0.0, 0.0].view(),
            })
            .unwrap_err();
        assert!(matches!(err, ComputeError::DimensionMismatch { .. }));
    }
}
