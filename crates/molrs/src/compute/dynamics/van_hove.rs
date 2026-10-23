//! Van Hove correlation function `G(r, t)` — self and distinct parts.

use molrs::core::{NeighborQuery, SimBox};
use molrs::op::F;
use ndarray::{Array1, Array2, ArrayView3, Axis as NdAxis};

use crate::compute::time_series::guards::first_wrapped_item;
use crate::compute::{Axis, Compute, ComputeError, Histogram, Origins, UpperEdge};

/// Van Hove space–time correlation `G(r, t)`, self and distinct parts.
///
/// Configuration: the radial [`Axis`], the integer frame `lags` to evaluate
/// and the time [`Origins`] (all origins, optionally every `stride`-th, or
/// the first frame only). The data — an unwrapped `T × N × 3` trajectory and
/// one box per frame — goes to [`compute`](Compute::compute).
///
/// The Van Hove function is the space–time generalization of the radial
/// distribution function (L. van Hove, *Phys. Rev.* **1954**, 95, 249; Hansen
/// & McDonald, *Theory of Simple Liquids*, 4th ed., §7):
///
/// ```text
///   G_s(r,t) = (1/N) Σ_i      ⟨ δ(r − |r_i(t) − r_i(0)|) ⟩          (self)
///   G_d(r,t) = (1/N) Σ_i Σ_{j≠i} ⟨ δ(r − |r_j(t) − r_i(0)|) ⟩      (distinct)
/// ```
///
/// with the structural anchor `G_d(r,0) = ρ g(r)` (ρ = N/V the number density,
/// `g(r)` the RDF) and the dynamical anchor `∫ r² G_s(r,t) dr = Msd(t)`. The
/// two thus bridge [`Rdf`](crate::compute::Rdf) (structure) and
/// [`Msd`](crate::compute::Msd) (dynamics).
///
/// # Provenance
///
/// The reference implementation ships no dedicated Van Hove analyzer — only
/// its RDF and ACF machinery. The **definition** follows van Hove 1954 /
/// Hansen–McDonald; the **binning** is the one [`Axis`] rule (half-open
/// bins by default, like [`Rdf`](crate::compute::Rdf); the closed last bin
/// of the former VanHove, which keeps `d = r_max`, through
/// [`with_upper_edge`](VanHove::with_upper_edge)) and the distinct-part shell
/// normalization uses [`Axis::shell_volumes`]; the **multi-time-origin
/// averaging** mirrors the ACF origin accumulation of the reference
/// implementation (`src/reordyn.cpp` / `src/acf.cpp`).
///
/// # Conventions
///
/// - `g_self[t]` is the probability **density of the displacement magnitude**
///   `|Δr|` over the axis: `∫ g_self dr = 1` over the displacements that fall
///   inside it, so with an axis covering every displacement its second moment
///   `∫ r² g_self dr` is the MSD. (It is the radial-weighted self part
///   `4π r² G_s^{3D}`, the directly histogrammable quantity.)
/// - `g_distinct[t]` is the **number density** of other particles at distance
///   `r` from a reference particle, `ρ g(r)` at `t = 0`.
/// - Self displacements use the raw coordinate difference — exactly as
///   [`Msd`](crate::compute::Msd) does — so the trajectory must be unwrapped
///   (see [`Unwrap`](crate::compute::Unwrap)); a frame-to-frame step longer
///   than half the box is [`ComputeError::Wrapped`]. Distinct distances use
///   the minimum image of the origin frame's box, through core
///   [`NeighborQuery`] (any box shape): its pair set depends on the origin
///   and the lag, so the search is built here, once per origin, and not
///   taken from the caller.
#[derive(Debug, Clone)]
pub struct VanHove {
    axis: Axis,
    lags: Vec<usize>,
    origins: Origins,
    stride: usize,
    upper_edge: UpperEdge,
}

/// The trajectory a [`VanHove`] analyses.
#[derive(Debug, Clone, Copy)]
pub struct VanHoveInput<'a> {
    /// Unwrapped positions, `T × N × 3`, Å.
    pub positions: ArrayView3<'a, F>,
    /// One box per frame.
    pub boxes: &'a [SimBox],
}

/// Result of a [`VanHove`] computation.
///
/// Rows are lag times (`lags`), columns are r-bins (`r_centers`).
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct VanHoveResult {
    /// r-bin edges (`n_bins + 1`), Å.
    pub r_edges: Array1<F>,
    /// r-bin centers (`n_bins`), Å.
    pub r_centers: Array1<F>,
    /// The lag times evaluated (frame units), aligned with rows.
    pub lags: Vec<usize>,
    /// Self part `g_self[t]`: probability density of `|Δr|` over the axis.
    pub g_self: Array2<F>,
    /// Distinct part `g_distinct[t]`: number density of others at `r`
    /// (`ρ g(r)` at `t = 0`).
    pub g_distinct: Array2<F>,
    /// Bin width, Å.
    pub dr: F,
}

impl VanHove {
    /// The time-origin stride when not set: every frame.
    pub const DEFAULT_STRIDE: usize = 1;

    /// Bin distances on `axis` (Å, starting at `r ≥ 0`), at the integer frame
    /// `lags` (e.g. `[0, 1, 5, 10]`), over all time origins.
    ///
    /// # Errors
    /// [`ComputeError::OutOfRange`] for an axis starting below zero or an
    /// empty `lags`.
    pub fn new(axis: Axis, lags: Vec<usize>) -> Result<Self, ComputeError> {
        if axis.min() < 0.0 {
            return Err(ComputeError::OutOfRange {
                field: "VanHove::axis (a radial axis starts at r ≥ 0)",
                value: axis.min().to_string(),
            });
        }
        if lags.is_empty() {
            return Err(ComputeError::OutOfRange {
                field: "VanHove::lags",
                value: "empty".to_string(),
            });
        }
        Ok(Self {
            axis,
            lags,
            origins: Origins::default(),
            stride: Self::DEFAULT_STRIDE,
            upper_edge: UpperEdge::default(),
        })
    }

    /// Count a distance of exactly the axis maximum in the last bin
    /// ([`UpperEdge::Closed`]) or leave it out (default
    /// [`UpperEdge::Open`]).
    pub fn with_upper_edge(mut self, upper_edge: UpperEdge) -> Self {
        self.upper_edge = upper_edge;
        self
    }

    /// Average over these time origins (default [`Origins::All`]).
    ///
    /// # Errors
    /// [`ComputeError::OutOfRange`] for [`Origins::First`] with a stride
    /// other than 1 (a single origin has no stride).
    pub fn with_origins(mut self, origins: Origins) -> Result<Self, ComputeError> {
        self.origins = origins;
        self.check_origins()?;
        Ok(self)
    }

    /// Take every `stride`-th frame as a time origin (default
    /// [`DEFAULT_STRIDE`](Self::DEFAULT_STRIDE)).
    ///
    /// # Errors
    /// [`ComputeError::OutOfRange`] for `stride == 0`, or a stride other than
    /// 1 with [`Origins::First`].
    pub fn with_stride(mut self, stride: usize) -> Result<Self, ComputeError> {
        if stride == 0 {
            return Err(ComputeError::OutOfRange {
                field: "VanHove::stride",
                value: stride.to_string(),
            });
        }
        self.stride = stride;
        self.check_origins()?;
        Ok(self)
    }

    fn check_origins(&self) -> Result<(), ComputeError> {
        if self.origins == Origins::First && self.stride != 1 {
            return Err(ComputeError::OutOfRange {
                field: "VanHove::stride (a single first origin has no stride)",
                value: self.stride.to_string(),
            });
        }
        Ok(())
    }

    /// The radial axis.
    pub fn axis(&self) -> Axis {
        self.axis
    }

    /// The lags evaluated, in frames.
    pub fn lags(&self) -> &[usize] {
        &self.lags
    }

    /// The time origins.
    pub fn origins(&self) -> Origins {
        self.origins
    }

    /// The origin stride.
    pub fn stride(&self) -> usize {
        self.stride
    }

    /// Whether the axis maximum is in the last bin.
    pub fn upper_edge(&self) -> UpperEdge {
        self.upper_edge
    }

    /// Validate the trajectory; returns `(T, N)`.
    fn validate(&self, input: &VanHoveInput<'_>) -> Result<(usize, usize), ComputeError> {
        let (n_frames, n_particles, n_dim) = input.positions.dim();
        if n_frames == 0 || n_particles == 0 {
            return Err(ComputeError::EmptyInput);
        }
        if n_dim != 3 {
            return Err(ComputeError::DimensionMismatch {
                expected: 3,
                got: n_dim,
                what: "VanHove positions (T × N × 3)",
            });
        }
        if input.boxes.len() != n_frames {
            return Err(ComputeError::DimensionMismatch {
                expected: n_frames,
                got: input.boxes.len(),
                what: "VanHove boxes (one per frame)",
            });
        }
        if let Some(&lag) = self.lags.iter().find(|&&lag| lag >= n_frames) {
            return Err(ComputeError::OutOfRange {
                field: "VanHove::lags",
                value: format!(
                    "{lag} exceeds n_frames - 1 = {}: that lag has no time origin",
                    n_frames - 1
                ),
            });
        }
        if let Some(index) = input.positions.iter().position(|v| !v.is_finite()) {
            return Err(ComputeError::NonFinite {
                where_: "VanHove positions",
                index,
            });
        }
        let frames = input.positions.outer_iter();
        let steps = frames.clone().zip(frames.skip(1));
        for (frame, ((prev, now), simbox)) in steps.zip(&input.boxes[1..]).enumerate() {
            if let Some(item) = first_wrapped_item(prev, now, simbox)? {
                return Err(ComputeError::Wrapped {
                    frame: frame + 1,
                    item,
                });
            }
        }
        Ok((n_frames, n_particles))
    }

    /// The self and distinct histograms (one per lag) of one origin `tau`.
    ///
    /// The `NeighborQuery` over `r_i(τ)` is built once and serves every lag;
    /// each lag queries `r_j(τ + lag)` against it within `r_max` — every pair
    /// beyond `r_max` lies outside the axis anyway. Self pairs `i == j` are
    /// left out of the distinct part.
    fn origin_histograms(
        &self,
        input: &VanHoveInput<'_>,
        tau: usize,
    ) -> Result<Vec<(Histogram<1>, Histogram<1>)>, ComputeError> {
        let n_frames = input.positions.len_of(NdAxis(0));
        let reference = input.positions.index_axis(NdAxis(0), tau);
        let query =
            NeighborQuery::new(&input.boxes[tau], reference, self.axis.max()).map_err(|e| {
                ComputeError::OutOfRange {
                    field: "r_max",
                    value: e.cutoff.to_string(),
                }
            })?;
        let mut out = Vec::with_capacity(self.lags.len());
        for &lag in &self.lags {
            let mut self_part = Histogram::new([self.axis]);
            let mut distinct = Histogram::new([self.axis]);
            if tau + lag < n_frames {
                let later = input.positions.index_axis(NdAxis(0), tau + lag);
                for (r0, r1) in reference.outer_iter().zip(later.outer_iter()) {
                    let d2: F = r0
                        .iter()
                        .zip(r1.iter())
                        .map(|(a, b)| (b - a) * (b - a))
                        .sum();
                    self_part.add([d2.sqrt()], self.upper_edge)?;
                }
                let pairs = query.query(later);
                let d2 = pairs
                    .dist_sq()
                    .expect("NeighborQuery::query stores dist_sq");
                let reference_index = pairs.point_indices();
                let later_index = pairs.query_point_indices();
                for k in 0..pairs.n_pairs() {
                    if reference_index[k] != later_index[k] {
                        distinct.add([d2[k].sqrt()], self.upper_edge)?;
                    }
                }
            }
            out.push((self_part, distinct));
        }
        Ok(out)
    }
}

impl Compute for VanHove {
    type Input<'a> = VanHoveInput<'a>;
    type Output = VanHoveResult;

    /// # Errors
    /// * [`ComputeError::EmptyInput`] for no frames or no particles, and for
    ///   a lag at which no displacement falls inside the axis (its self
    ///   density is undefined — widen the axis).
    /// * [`ComputeError::DimensionMismatch`] for positions that are not
    ///   `T × N × 3` or a box count other than `T`.
    /// * [`ComputeError::OutOfRange`] for a lag `≥ T` (it has no origin).
    /// * [`ComputeError::NonFinite`] for a non-finite position.
    /// * [`ComputeError::Wrapped`] for a trajectory wrapped into the box.
    fn compute(&self, input: VanHoveInput<'_>) -> Result<VanHoveResult, ComputeError> {
        let (n_frames, n_particles) = self.validate(&input)?;
        let origins: Vec<usize> = match self.origins {
            Origins::All => (0..n_frames).step_by(self.stride).collect(),
            Origins::First => vec![0],
        };

        // Integer counts: merging the per-origin histograms in any order is
        // bit-identical, so the parallel and serial paths agree exactly.
        #[cfg(feature = "rayon")]
        let per_origin: Vec<Vec<(Histogram<1>, Histogram<1>)>> = {
            use rayon::prelude::*;
            origins
                .par_iter()
                .map(|&tau| self.origin_histograms(&input, tau))
                .collect::<Result<_, _>>()?
        };
        #[cfg(not(feature = "rayon"))]
        let per_origin: Vec<Vec<(Histogram<1>, Histogram<1>)>> = origins
            .iter()
            .map(|&tau| self.origin_histograms(&input, tau))
            .collect::<Result<_, _>>()?;

        let shells = self.axis.shell_volumes(3)?;
        let n_bins = self.axis.n_bins();
        let mut g_self = Array2::<F>::zeros((self.lags.len(), n_bins));
        let mut g_distinct = Array2::<F>::zeros((self.lags.len(), n_bins));
        for (li, &lag) in self.lags.iter().enumerate() {
            let mut self_part = Histogram::new([self.axis]);
            let mut distinct = Histogram::new([self.axis]);
            for histograms in &per_origin {
                self_part.merge(&histograms[li].0);
                distinct.merge(&histograms[li].1);
            }
            // Every lag is below T, so origin 0 always contributes.
            let n_origins = origins.iter().filter(|&&tau| tau + lag < n_frames).count();
            g_self.row_mut(li).assign(
                &self_part
                    .density()?
                    .into_dimensionality::<ndarray::Ix1>()
                    .expect("a 1-D histogram"),
            );
            let denom = (n_particles * n_origins) as F;
            for (k, g) in g_distinct.row_mut(li).iter_mut().enumerate() {
                *g = distinct.counts()[[k]] / (denom * shells[k]);
            }
        }

        Ok(VanHoveResult {
            r_edges: self.axis.edges(),
            r_centers: self.axis.centers(),
            lags: self.lags.clone(),
            g_self,
            g_distinct,
            dr: self.axis.width(),
        })
    }
}

crate::compute::spec::compute_spec! {
    VanHove as "VanHove";
    params |s| {
        axis: Axis;
        lags: IntList;
        origins: Choice(Origins::NAMES) = Origins::default() => s.origins;
        stride: Int = VanHove::DEFAULT_STRIDE => s.stride;
        upper_edge: Choice(UpperEdge::NAMES) = UpperEdge::default() => s.upper_edge;
    }
    inputs {
        positions: Trajectory;
        boxes: Boxes;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute::time_series::test_support::trajectory;
    use crate::core::Cell;
    use ndarray::Array3;

    fn cube(l: F) -> SimBox {
        SimBox::new(Cell::Lengths([l; 3]), [0.0, 0.0, 0.0], [true, true, true]).unwrap()
    }

    fn run(analysis: &VanHove, x: &Array3<F>, l: F) -> Result<VanHoveResult, ComputeError> {
        let boxes = vec![cube(l); x.len_of(NdAxis(0))];
        analysis.compute(VanHoveInput {
            positions: x.view(),
            boxes: &boxes,
        })
    }

    /// Two particles 1 Å apart; particle 0 steps +0.25 Å per frame along x,
    /// particle 1 sits still. Axis [0, 2) in 8 bins of 0.25.
    fn walker() -> Array3<F> {
        trajectory(&[
            vec![[1.0, 1.0, 1.0], [2.0, 1.0, 1.0]],
            vec![[1.25, 1.0, 1.0], [2.0, 1.0, 1.0]],
            vec![[1.5, 1.0, 1.0], [2.0, 1.0, 1.0]],
        ])
    }

    #[test]
    fn self_and_distinct_parts_by_hand() {
        let vh = VanHove::new(Axis::new(0.0, 2.0, 8).unwrap(), vec![0, 1, 2]).unwrap();
        let res = run(&vh, &walker(), 20.0).unwrap();
        let dr = 0.25;
        assert_eq!(res.dr, dr);
        assert_eq!(res.lags, vec![0, 1, 2]);

        // Self part. Lag 0: every |Δr| = 0 → bin 0. Lag 1: 0.25 (bin 1) and 0
        // (bin 0), over two origins. Lag 2: 0.5 (bin 2) and 0 (bin 0).
        let density = |counts: [F; 8], total: F| counts.map(|c| c / (total * dr));
        let lag0 = density([6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 6.0);
        let lag1 = density([2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 4.0);
        let lag2 = density([1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], 2.0);
        for (li, want) in [lag0, lag1, lag2].iter().enumerate() {
            assert_eq!(res.g_self.row(li).to_vec(), want.to_vec(), "self lag {li}");
        }

        // Distinct part at lag 0: the pair at d = 1.0, 0.75, 0.5 in frames
        // 0, 1, 2 — each seen from both ends: 2 counts per frame in bins 4,
        // 3, 2, over N = 2 particles and 3 origins.
        let shells = Axis::new(0.0, 2.0, 8).unwrap().shell_volumes(3).unwrap();
        for k in [2, 3, 4] {
            let want = 2.0 / (2.0 * 3.0 * shells[k]);
            assert!((res.g_distinct[[0, k]] - want).abs() < 1e-12, "bin {k}");
        }
        assert_eq!(
            res.g_distinct.row(0).iter().filter(|&&g| g != 0.0).count(),
            3
        );
    }

    /// Behaviour change 3: half-open bins, like `Rdf`. A displacement of
    /// exactly `r_max` is outside the axis; it used to be clamped into the
    /// last bin.
    #[test]
    fn a_distance_of_exactly_r_max_is_outside_the_axis() {
        // Particle 0 jumps exactly 2.0 = r_max; particle 1 stays.
        let x = trajectory(&[
            vec![[1.0, 1.0, 1.0], [5.0, 5.0, 5.0]],
            vec![[3.0, 1.0, 1.0], [5.0, 5.0, 5.0]],
        ]);
        let vh = VanHove::new(Axis::new(0.0, 2.0, 4).unwrap(), vec![1]).unwrap();
        let res = run(&vh, &x, 20.0).unwrap();
        // Only particle 1's zero displacement is binned: all weight in bin 0.
        assert_eq!(res.g_self.row(0).to_vec(), vec![2.0, 0.0, 0.0, 0.0]);
    }

    /// The former convention, selectable: a closed last bin keeps
    /// `d = r_max`. Same jump as above: bin 0 holds particle 1's zero
    /// displacement and the last bin particle 0's 2.0, so the density over
    /// `binned = 2`, `dr = 0.5` is `[1, 0, 0, 1]`. A distinct pair exactly
    /// `r_max` apart is in the last bin too, and outside the open axis.
    #[test]
    fn a_closed_upper_edge_keeps_r_max_in_the_last_bin() {
        let x = trajectory(&[
            vec![[1.0, 1.0, 1.0], [5.0, 5.0, 5.0]],
            vec![[3.0, 1.0, 1.0], [5.0, 5.0, 5.0]],
        ]);
        let axis = Axis::new(0.0, 2.0, 4).unwrap();
        let closed = VanHove::new(axis, vec![1])
            .unwrap()
            .with_upper_edge(UpperEdge::Closed);
        let res = run(&closed, &x, 20.0).unwrap();
        assert_eq!(res.g_self.row(0).to_vec(), vec![1.0, 0.0, 0.0, 1.0]);

        let pair = trajectory(&[vec![[1.0, 1.0, 1.0], [3.0, 1.0, 1.0]]]);
        let open = VanHove::new(axis, vec![0]).unwrap();
        let shut = open.clone().with_upper_edge(UpperEdge::Closed);
        let (open, shut) = (
            run(&open, &pair, 20.0).unwrap(),
            run(&shut, &pair, 20.0).unwrap(),
        );
        assert_eq!(open.g_distinct.row(0).to_vec(), vec![0.0; 4]);
        assert!(shut.g_distinct[[0, 3]] > 0.0);
        assert_eq!(shut.g_distinct.row(0).sum(), shut.g_distinct[[0, 3]]);
    }

    #[test]
    fn origin_stride_and_first_origin() {
        let x = walker();
        let axis = Axis::new(0.0, 2.0, 8).unwrap();
        // Stride 2: origins 0 and 2; lag 1 only has origin 0 (0.25 and 0).
        let strided = VanHove::new(axis, vec![1]).unwrap().with_stride(2).unwrap();
        let first = VanHove::new(axis, vec![1])
            .unwrap()
            .with_origins(Origins::First)
            .unwrap();
        let a = run(&strided, &x, 20.0).unwrap();
        let b = run(&first, &x, 20.0).unwrap();
        assert_eq!(a.g_self, b.g_self);
        assert_eq!(a.g_self[[0, 0]], 2.0);
        assert_eq!(a.g_self[[0, 1]], 2.0);
    }

    /// Every former fallback is an error with its own variant.
    #[test]
    fn former_fallbacks_are_errors() {
        let axis = Axis::new(0.0, 2.0, 8).unwrap();
        let x = walker();

        // A lag with no origin used to be dropped silently.
        let vh = VanHove::new(axis, vec![1, 3]).unwrap();
        assert!(matches!(
            run(&vh, &x, 20.0),
            Err(ComputeError::OutOfRange {
                field: "VanHove::lags",
                ..
            })
        ));

        // A zero stride used to be clamped to 1.
        assert!(matches!(
            VanHove::new(axis, vec![0]).unwrap().with_stride(0),
            Err(ComputeError::OutOfRange {
                field: "VanHove::stride",
                ..
            })
        ));
        assert!(matches!(
            VanHove::new(axis, vec![0])
                .unwrap()
                .with_stride(2)
                .unwrap()
                .with_origins(Origins::First),
            Err(ComputeError::OutOfRange { .. })
        ));

        // No box used to give an all-zero distinct part and a flag; the box
        // is now required data, one per frame.
        let vh = VanHove::new(axis, vec![0]).unwrap();
        assert!(matches!(
            vh.compute(VanHoveInput {
                positions: x.view(),
                boxes: &[],
            }),
            Err(ComputeError::DimensionMismatch { .. })
        ));

        // A lag at which nothing falls inside the axis has no self density.
        let far = trajectory(&[vec![[1.0, 1.0, 1.0]], vec![[4.0, 1.0, 1.0]]]);
        assert!(matches!(
            run(&VanHove::new(axis, vec![1]).unwrap(), &far, 20.0),
            Err(ComputeError::EmptyInput)
        ));
    }

    #[test]
    fn rejects_bad_configuration_and_data() {
        assert!(matches!(
            VanHove::new(Axis::new(0.0, 2.0, 8).unwrap(), vec![]),
            Err(ComputeError::OutOfRange { .. })
        ));
        assert!(matches!(
            VanHove::new(Axis::new(-1.0, 2.0, 8).unwrap(), vec![0]),
            Err(ComputeError::OutOfRange { .. })
        ));

        let vh = VanHove::new(Axis::new(0.0, 2.0, 8).unwrap(), vec![1]).unwrap();
        // A step of 6 Å in a 10 Å box: wrapped.
        let wrapped = trajectory(&[vec![[0.5, 1.0, 1.0]], vec![[9.5, 1.0, 1.0]]]);
        assert!(matches!(
            run(&vh, &wrapped, 10.0),
            Err(ComputeError::Wrapped { frame: 1, item: 0 })
        ));

        let mut nan = walker();
        nan[[1, 0, 0]] = F::NAN;
        assert!(matches!(
            run(&vh, &nan, 20.0),
            Err(ComputeError::NonFinite { .. })
        ));
    }
}
