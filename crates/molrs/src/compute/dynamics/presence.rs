//! [`Presence`] — which pairs exist at which frame — and [`PairPresence`],
//! the compute that reads it off per-frame neighbour tables.

use std::collections::{BTreeSet, HashMap};

use molrs::core::{PairSource, QueryMode};
use molrs::op::F;
use ndarray::Array2;

use crate::compute::{Compute, ComputeError, require_dist_sq};

/// A 0/1 presence series: one row per frame, one column per pair.
///
/// The output of [`PairPresence`] and of
/// [`HBondsResult::presence`](crate::compute::HBondsResult::presence), and
/// the input of [`Survival`](crate::compute::Survival): hand
/// `values.view()` to its `presence` field. A `Presence` always has at least
/// one pair — a run in which nothing ever formed has no presence series, and
/// its producer says so with `None`.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct Presence {
    /// The pair behind each column, in ascending order. What the two indices
    /// mean is the producer's: query / reference point for [`PairPresence`],
    /// (donor, acceptor) atom for a hydrogen bond.
    pub pairs: Vec<(u32, u32)>,
    /// `values[[t, k]]` is `1` when pair `k` is present at frame `t`, else
    /// `0`; shape `T × n_pairs`.
    pub values: Array2<F>,
}

/// Pair presence with hysteresis, read off one neighbour table per frame.
///
/// A pair *enters* when its distance falls to `r0` or below and *stays* while
/// it remains within `r1 ≥ r0`:
///
/// ```text
/// h(t) = 1   if d(t) ≤ r0,
///        1   if h(t − 1) = 1 and d(t) ≤ r1,
///        0   otherwise,
/// ```
///
/// with `d(t)` the pair distance in frame `t` (a pair missing from a table
/// is farther than its search cutoff). With `r1 = r0` (the default) this is
/// the plain distance criterion; with `r1 > r0` it is the stable-states
/// picture (Laage & Hynes, *Science* **311**, 832 (2006)), in which a pair
/// that rattles across a single cutoff does not break. The survival
/// correlations are [`Survival`](crate::compute::Survival) on the result —
/// continuous survival with `r1 > r0` is the "SSP" pair survival, the
/// intermittent one with `r1 = r0` the usual `⟨h(0) h(t)⟩` (Rapaport, *Mol.
/// Phys.* **50**, 1151 (1983); Luzar & Chandler, *Phys. Rev. Lett.* **76**,
/// 928 (1996)). Origins wherever the pair is in the bonded stable state, and
/// one curve per pair, are `Survival`'s defaults; its
/// [`SurvivalOrigin::Inner`](crate::compute::SurvivalOrigin::Inner) starts
/// runs only inside `r0` — pass the presence of a second `PairPresence` with
/// `r1 = r0` (same columns) as `inner` — and
/// [`SurvivalNormalization::PerReference`](crate::compute::SurvivalNormalization::PerReference)
/// averages per reference particle (`C(0)` the coordination number).
///
/// The pairs come from core neighbour search
/// ([`NeighborQuery`](molrs::core::NeighborQuery) cross-queries,
/// [`NeighborList`](molrs::core::NeighborList) self-queries), so any box
/// shape is handled there. **Each table must be searched with a cutoff of at
/// least `r1`**: a pair the search did not report counts as absent.
///
/// The columns are every pair that is present in at least one frame, in
/// ascending `(i, j)` order; a pair never within `r0` has no column. Adapted
/// from the pair-persistence recipe of *tame*
/// (<https://github.com/yqshao-archive/tame>).
///
/// ```
/// use molrs::compute::{Compute, PairPresence, PairPresenceInput, Survival, SurvivalInput, SurvivalKind};
/// use molrs::core::{Cell, NeighborQuery, PairSource, SimBox};
/// use ndarray::array;
///
/// let simbox = SimBox::new(Cell::Lengths([10.0; 3]), [0.0, 0.0, 0.0], [true, true, true]).unwrap();
/// let reference = array![[0.0, 0.0, 0.0]];
/// let tables: Vec<_> = [0.5, 1.5, 0.5]
///     .iter()
///     .map(|&x| {
///         NeighborQuery::new(&simbox, reference.view(), 2.0)
///             .unwrap()
///             .query(array![[x, 0.0, 0.0]].view())
///     })
///     .collect();
/// let frames: Vec<&(dyn PairSource + Sync)> = tables.iter().map(|t| t as _).collect();
///
/// let presence = PairPresence::new(1.0, false)?
///     .with_r1(2.0)?
///     .compute(PairPresenceInput { frames: &frames })?
///     .expect("the pair formed");
/// assert_eq!(presence.values.column(0).to_vec(), vec![1.0, 1.0, 1.0]);
///
/// let survival = Survival::new(SurvivalKind::Continuous, 2)
///     .compute(SurvivalInput {
///         presence: presence.values.view(),
///         dt: 1.0,
///         inner: None,
///         n_reference: None,
///     })?;
/// assert_eq!(survival.values[[2, 0]], 1.0);
/// # Ok::<(), molrs::compute::ComputeError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PairPresence {
    r0: F,
    r1: F,
    exclude_self: bool,
}

/// The data [`PairPresence`] reads: one neighbour table per frame.
#[derive(Clone, Copy)]
pub struct PairPresenceInput<'a> {
    /// One pair source per frame, all of one [`QueryMode`] (same pairing,
    /// same point counts), each with the `dist_sq` column and searched with
    /// a cutoff of at least `r1`.
    pub frames: &'a [&'a (dyn PairSource + Sync)],
}

impl PairPresence {
    /// Pairs within `r0` (Å, > 0) are present; `r1` starts equal to `r0`.
    ///
    /// `exclude_self` has no default: set it when every table is a
    /// cross-query of one point set against itself, to drop the `i == j`
    /// pair each point forms with itself; a table where that does not apply
    /// is an error at compute time, never silently ignored.
    ///
    /// # Errors
    /// [`ComputeError::OutOfRange`] for a non-finite or non-positive `r0`.
    pub fn new(r0: F, exclude_self: bool) -> Result<Self, ComputeError> {
        if !(r0.is_finite() && r0 > 0.0) {
            return Err(ComputeError::OutOfRange {
                field: "PairPresence::r0",
                value: r0.to_string(),
            });
        }
        Ok(Self {
            r0,
            r1: r0,
            exclude_self,
        })
    }

    /// A present pair stays present while within `r1` (Å, `r1 ≥ r0`).
    ///
    /// # Errors
    /// [`ComputeError::OutOfRange`] for a non-finite `r1` or `r1 < r0`.
    pub fn with_r1(mut self, r1: F) -> Result<Self, ComputeError> {
        if !(r1.is_finite() && r1 >= self.r0) {
            return Err(ComputeError::OutOfRange {
                field: "PairPresence::r1 (require r1 >= r0)",
                value: r1.to_string(),
            });
        }
        self.r1 = r1;
        Ok(self)
    }

    /// The entry distance, Å.
    pub fn r0(&self) -> F {
        self.r0
    }

    /// The exit distance, Å.
    pub fn r1(&self) -> F {
        self.r1
    }

    /// Whether the `i == j` self pairs are dropped.
    pub fn exclude_self(&self) -> bool {
        self.exclude_self
    }

    /// `exclude_self` applies only to a cross-query of a set against itself.
    fn check_exclude_self(&self, mode: QueryMode) -> Result<(), ComputeError> {
        if !self.exclude_self {
            return Ok(());
        }
        match mode {
            QueryMode::CrossQuery {
                n_query_points,
                n_points,
            } if n_query_points == n_points => Ok(()),
            _ => Err(ComputeError::OutOfRange {
                field: "PairPresence::exclude_self",
                value: format!(
                    "true, but the tables are {mode:?}: only a cross-query of one point \
                     set against itself has i == j self pairs"
                ),
            }),
        }
    }

    /// The `dist_sq` of every pair of one frame within `r1`.
    fn frame_pairs(&self, source: &dyn PairSource) -> Result<HashMap<(u32, u32), F>, ComputeError> {
        require_dist_sq(source)?;
        let r1_sq = self.r1 * self.r1;
        let mut within = HashMap::new();
        let mut failure = None;
        let mut k = 0usize;
        source.visit_pairs(&mut |i, j, dist_sq, _| {
            if failure.is_none() {
                // `require_dist_sq` has vouched for the column.
                match dist_sq {
                    Some(d2) if !d2.is_nan() => {
                        if d2 <= r1_sq && !(self.exclude_self && i == j) {
                            within.insert((i, j), d2);
                        }
                    }
                    _ => {
                        failure = Some(ComputeError::NonFinite {
                            where_: "pair dist_sq",
                            index: k,
                        })
                    }
                }
            }
            k += 1;
        });
        match failure {
            Some(err) => Err(err),
            None => Ok(within),
        }
    }
}

impl Compute for PairPresence {
    type Input<'a> = PairPresenceInput<'a>;
    type Output = Option<Presence>;

    /// The presence series, or `None` when no pair is ever within `r0`.
    ///
    /// # Errors
    /// * [`ComputeError::EmptyInput`] for no frames.
    /// * [`ComputeError::BadShape`] for a table without `dist_sq`, or tables
    ///   of different [`QueryMode`]s.
    /// * [`ComputeError::OutOfRange`] for `exclude_self` on tables that are
    ///   not a cross-query of one set against itself.
    /// * [`ComputeError::NonFinite`] for a NaN pair distance.
    fn compute(&self, input: PairPresenceInput<'_>) -> Result<Option<Presence>, ComputeError> {
        let Some(first) = input.frames.first() else {
            return Err(ComputeError::EmptyInput);
        };
        let mode = first.mode();
        self.check_exclude_self(mode)?;
        let mut per_frame = Vec::with_capacity(input.frames.len());
        for source in input.frames {
            if source.mode() != mode {
                return Err(ComputeError::BadShape {
                    expected: format!("every table of mode {mode:?}, like frame 0"),
                    got: format!("{:?}", source.mode()),
                });
            }
            per_frame.push(self.frame_pairs(*source)?);
        }

        let r0_sq = self.r0 * self.r0;
        let entered: BTreeSet<(u32, u32)> = per_frame
            .iter()
            .flat_map(|frame| frame.iter().filter(|(_, d2)| **d2 <= r0_sq))
            .map(|(pair, _)| *pair)
            .collect();
        if entered.is_empty() {
            return Ok(None);
        }
        let pairs: Vec<(u32, u32)> = entered.into_iter().collect();

        // Hysteresis is sequential in time, so the series is walked per pair.
        let mut values = Array2::<F>::zeros((per_frame.len(), pairs.len()));
        for (k, pair) in pairs.iter().enumerate() {
            let mut present = false;
            for (t, frame) in per_frame.iter().enumerate() {
                present = match frame.get(pair) {
                    Some(&d2) => d2 <= r0_sq || present,
                    None => false,
                };
                values[[t, k]] = if present { 1.0 } else { 0.0 };
            }
        }
        Ok(Some(Presence { pairs, values }))
    }
}

crate::compute::spec::compute_spec! {
    PairPresence as "PairPresence";
    params |s| {
        r0: Float;
        exclude_self: Bool;
        r1: Float = crate::compute::ParamDefault::SameAs("r0") => s.r1;
    }
    inputs {
        frames: PairSourceFrames;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute::{
        Survival, SurvivalInput, SurvivalKind, SurvivalNormalization, SurvivalOrigin,
        TimeCorrelation,
    };
    use crate::core::Cell;
    use molrs::core::{NeighborColumns, NeighborPair, NeighborQuery, Neighbors, SimBox};
    use ndarray::Array2;

    fn cube(l: F) -> SimBox {
        SimBox::new(Cell::Lengths([l; 3]), [0.0, 0.0, 0.0], [true, true, true]).unwrap()
    }

    /// One cross-query table per frame: species `j` (query) against species
    /// `i` (reference), searched to `cutoff`.
    fn tables(
        species_i: &Frames,
        species_j: &Frames,
        simbox: &SimBox,
        cutoff: F,
    ) -> Vec<Neighbors> {
        let rows =
            |points: &Vec<[F; 3]>| Array2::from_shape_fn((points.len(), 3), |(a, d)| points[a][d]);
        species_i
            .iter()
            .zip(species_j)
            .map(|(pi, pj)| {
                NeighborQuery::new(simbox, rows(pi).view(), cutoff)
                    .unwrap()
                    .query(rows(pj).view())
            })
            .collect()
    }

    fn presence(analysis: PairPresence, tables: &[Neighbors]) -> Option<Presence> {
        let frames: Vec<&(dyn PairSource + Sync)> = tables.iter().map(|t| t as _).collect();
        analysis
            .compute(PairPresenceInput { frames: &frames })
            .unwrap()
    }

    fn survival(kind: SurvivalKind, p: &Presence, max_lag: usize) -> TimeCorrelation {
        Survival::new(kind, max_lag)
            .compute(SurvivalInput {
                presence: p.values.view(),
                dt: 1.0,
                inner: None,
                n_reference: None,
            })
            .unwrap()
    }

    /// Points per frame.
    type Frames = Vec<Vec<[F; 3]>>;

    /// One `i` at the origin and one `j` on the x axis at `xs[t]`.
    fn one_pair(xs: &[F]) -> (Frames, Frames) {
        let i = xs.iter().map(|_| vec![[0.0, 0.0, 0.0]]).collect();
        let j = xs.iter().map(|&x| vec![[x, 0.0, 0.0]]).collect();
        (i, j)
    }

    #[test]
    fn permanently_bonded_pair_survives_fully() {
        let (ci, cj) = one_pair(&[0.5, 0.5, 0.5, 0.5]);
        let p = presence(
            PairPresence::new(1.0, false).unwrap(),
            &tables(&ci, &cj, &cube(100.0), 1.0),
        )
        .unwrap();
        let s = survival(SurvivalKind::Continuous, &p, 3);
        // Always bonded ⇒ C(τ) = 1 for all τ.
        for tau in 0..=3 {
            assert!((s.values[[tau, 0]] - 1.0).abs() < 1e-12, "C({tau})");
        }
    }

    #[test]
    fn continuous_breaks_permanently_intermittent_reforms() {
        // Within the cutoff at frames 0, 1; out at 2; back at 3.
        let (ci, cj) = one_pair(&[0.5, 0.5, 5.0, 0.5]);
        let p = presence(
            PairPresence::new(1.0, false).unwrap(),
            &tables(&ci, &cj, &cube(100.0), 1.0),
        )
        .unwrap();
        let cont = survival(SurvivalKind::Continuous, &p, 3);
        let imm = survival(SurvivalKind::Intermittent, &p, 3);
        // At τ = 3 only origin 0 exists: broken in between, present at the end.
        assert!(cont.values[[3, 0]].abs() < 1e-12);
        assert!((imm.values[[3, 0]] - 1.0).abs() < 1e-12);
    }

    /// Enter within `r0 = 1`, stay within `r1 = 2`. The pair at 0.5 then 1.5
    /// stays present under hysteresis (in the bonded stable state at both
    /// frames, so `C(0) = 1`) and breaks at frame 1 under a single cutoff.
    #[test]
    fn hysteresis_enters_at_r0_and_leaves_beyond_r1() {
        let (ci, cj) = one_pair(&[0.5, 1.5]);
        let t = tables(&ci, &cj, &cube(100.0), 2.0);

        let ssp = presence(
            PairPresence::new(1.0, false).unwrap().with_r1(2.0).unwrap(),
            &t,
        )
        .unwrap();
        assert_eq!(ssp.values.column(0).to_vec(), vec![1.0, 1.0]);
        let s = survival(SurvivalKind::Continuous, &ssp, 1);
        assert!((s.values[[0, 0]] - 1.0).abs() < 1e-12, "C(0)");
        assert!((s.values[[1, 0]] - 1.0).abs() < 1e-12, "C(1)");

        let single = presence(PairPresence::new(1.0, false).unwrap(), &t).unwrap();
        let c = survival(SurvivalKind::Continuous, &single, 1);
        assert!(c.values[[1, 0]].abs() < 1e-12);
    }

    /// A pair that only ever comes within `r1`, never within `r0`, never
    /// enters; and once out beyond `r1` it needs `r0` again.
    #[test]
    fn hysteresis_needs_r0_to_reenter() {
        let (ci, cj) = one_pair(&[1.5, 0.5, 1.5, 2.5, 1.5, 0.9]);
        let t = tables(&ci, &cj, &cube(100.0), 3.0);
        let p = presence(
            PairPresence::new(1.0, false).unwrap().with_r1(2.0).unwrap(),
            &t,
        )
        .unwrap();
        assert_eq!(
            p.values.column(0).to_vec(),
            vec![0.0, 1.0, 1.0, 0.0, 0.0, 1.0]
        );
    }

    /// One `i`, two `j` both within the cutoff: two pairs, each present.
    /// (`pair_survival_tcf` reported the coordination number, 2 partners per
    /// reference; the survival correlation is per pair, so `C(0) = ⟨h⟩ = 1`.)
    #[test]
    fn every_pair_is_a_column() {
        let ci = vec![vec![[0.0, 0.0, 0.0]]; 2];
        let cj = vec![vec![[0.5, 0.0, 0.0], [0.0, 0.5, 0.0]]; 2];
        let p = presence(
            PairPresence::new(1.0, false).unwrap(),
            &tables(&ci, &cj, &cube(100.0), 1.0),
        )
        .unwrap();
        assert_eq!(p.pairs, vec![(0, 0), (1, 0)]);
        let c = survival(SurvivalKind::Intermittent, &p, 1);
        assert!((c.values[[0, 0]] - 1.0).abs() < 1e-12, "C(0)");
    }

    /// The former convention, selectable: per reference particle, `C(0)` is
    /// the coordination number. `pair_survival_tcf`'s
    /// `coordination_number_at_lag_zero` on `dev`: one `i`, two `j` within
    /// the cutoff ⇒ `C(0) = 2.0`.
    #[test]
    fn per_reference_survival_is_the_coordination_number() {
        let ci = vec![vec![[0.0, 0.0, 0.0]]; 2];
        let cj = vec![vec![[0.5, 0.0, 0.0], [0.0, 0.5, 0.0]]; 2];
        let p = presence(
            PairPresence::new(1.0, false).unwrap(),
            &tables(&ci, &cj, &cube(100.0), 1.0),
        )
        .unwrap();
        let c = Survival::new(SurvivalKind::Intermittent, 1)
            .with_normalization(SurvivalNormalization::PerReference)
            .compute(SurvivalInput {
                presence: p.values.view(),
                dt: 1.0,
                inner: None,
                n_reference: Some(ci[0].len()),
            })
            .unwrap();
        assert!((c.values[[0, 0]] - 2.0).abs() < 1e-12, "C(0)");
    }

    /// The former SSP origin, selectable: runs start only where the pair is
    /// inside `r0` (the single-cutoff presence) and survive while within
    /// `r1` (the hysteresis presence). `pair_survival_tcf`'s
    /// `ssp_uses_inner_birth_outer_survival_cutoff` on `dev`: born only at
    /// frame 0 ⇒ `C(0) = 0.5`; still within `r1` at frame 1 ⇒ `C(1) = 1.0`.
    #[test]
    fn inner_origins_reproduce_the_former_ssp() {
        let (ci, cj) = one_pair(&[0.5, 1.5]);
        let t = tables(&ci, &cj, &cube(100.0), 2.0);
        let stable = presence(
            PairPresence::new(1.0, false).unwrap().with_r1(2.0).unwrap(),
            &t,
        )
        .unwrap();
        let inner = presence(PairPresence::new(1.0, false).unwrap(), &t).unwrap();
        assert_eq!(inner.pairs, stable.pairs);
        let ssp = Survival::new(SurvivalKind::Continuous, 1)
            .with_origin(SurvivalOrigin::Inner)
            .with_normalization(SurvivalNormalization::PerReference)
            .compute(SurvivalInput {
                presence: stable.values.view(),
                dt: 1.0,
                inner: Some(inner.values.view()),
                n_reference: Some(1),
            })
            .unwrap();
        assert!((ssp.values[[0, 0]] - 0.5).abs() < 1e-12, "C(0)");
        assert!((ssp.values[[1, 0]] - 1.0).abs() < 1e-12, "C(1)");
        // With a single cutoff 1.0 it would have died at τ = 1.
        let cont = survival(SurvivalKind::Continuous, &inner, 1);
        assert!(cont.values[[1, 0]].abs() < 1e-12);
    }

    #[test]
    fn minimum_image_wraps_across_boundary() {
        // 0.2 and 9.8 in a box of 10: 0.4 apart through the boundary.
        let ci = vec![vec![[0.2, 0.0, 0.0]]; 2];
        let cj = vec![vec![[9.8, 0.0, 0.0]]; 2];
        let p = presence(
            PairPresence::new(1.0, false).unwrap(),
            &tables(&ci, &cj, &cube(10.0), 1.0),
        )
        .unwrap();
        let c = survival(SurvivalKind::Continuous, &p, 1);
        assert!((c.values[[0, 0]] - 1.0).abs() < 1e-12);
    }

    /// The pairs come from core search, so a triclinic box needs nothing
    /// special: these two points are 0.5 apart only through the tilted face.
    #[test]
    fn triclinic_boxes_come_from_core_search() {
        // Lattice vectors are the columns: a = (10, 0, 0), b = (5, 10, 0).
        let h = [[10.0, 5.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]];
        let simbox = SimBox::new(Cell::Matrix(h), [0.0, 0.0, 0.0], [true, true, true]).unwrap();
        let ci = vec![vec![[0.0, 0.0, 0.0]]; 2];
        // (5, 9.5, 0) − b = (0, −0.5, 0).
        let cj = vec![vec![[5.0, 9.5, 0.0]]; 2];
        let p = presence(
            PairPresence::new(1.0, false).unwrap(),
            &tables(&ci, &cj, &simbox, 1.0),
        )
        .expect("the pair is 0.5 apart under the triclinic minimum image");
        assert_eq!(p.values.column(0).to_vec(), vec![1.0, 1.0]);
    }

    #[test]
    fn exclude_self_drops_the_diagonal() {
        // One species of two atoms, searched against itself.
        let c = vec![vec![[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]; 2];
        let t = tables(&c, &c, &cube(100.0), 1.0);
        let p = presence(PairPresence::new(1.0, true).unwrap(), &t).unwrap();
        assert_eq!(p.pairs, vec![(0, 1), (1, 0)]);
        // Each atom has exactly its one partner within the cutoff.
        let c = survival(SurvivalKind::Intermittent, &p, 1);
        assert!((c.values[[0, 0]] - 1.0).abs() < 1e-12, "C(0)");

        let kept = presence(PairPresence::new(1.0, false).unwrap(), &t).unwrap();
        assert_eq!(kept.pairs, vec![(0, 0), (0, 1), (1, 0), (1, 1)]);
    }

    /// `pair_survival_tcf` ignored `exclude_self` when the two species
    /// differed in size; now it is an error wherever it cannot apply.
    #[test]
    fn exclude_self_that_cannot_apply_is_an_error() {
        let ci = vec![vec![[0.0, 0.0, 0.0]]; 2];
        let cj = vec![vec![[0.5, 0.0, 0.0], [0.0, 0.5, 0.0]]; 2];
        let t = tables(&ci, &cj, &cube(100.0), 1.0);
        let frames: Vec<&(dyn PairSource + Sync)> = t.iter().map(|t| t as _).collect();
        let err = PairPresence::new(1.0, true)
            .unwrap()
            .compute(PairPresenceInput { frames: &frames })
            .unwrap_err();
        assert!(matches!(
            err,
            ComputeError::OutOfRange {
                field: "PairPresence::exclude_self",
                ..
            }
        ));

        let half_shell = Neighbors::from_pairs(
            [NeighborPair {
                i: 0,
                j: 1,
                dist_sq: 0.25,
                disp: [0.5, 0.0, 0.0],
            }],
            NeighborColumns::FULL,
            QueryMode::SelfQuery { n_points: 2 },
        );
        let frames: [&(dyn PairSource + Sync); 1] = [&half_shell];
        assert!(matches!(
            PairPresence::new(1.0, true)
                .unwrap()
                .compute(PairPresenceInput { frames: &frames }),
            Err(ComputeError::OutOfRange {
                field: "PairPresence::exclude_self",
                ..
            })
        ));
    }

    #[test]
    fn nothing_ever_present_is_none() {
        let (ci, cj) = one_pair(&[5.0, 5.0]);
        assert!(
            presence(
                PairPresence::new(1.0, false).unwrap(),
                &tables(&ci, &cj, &cube(100.0), 1.0),
            )
            .is_none()
        );
    }

    #[test]
    fn rejects_bad_configuration_and_data() {
        assert!(matches!(
            PairPresence::new(0.0, false),
            Err(ComputeError::OutOfRange { .. })
        ));
        assert!(matches!(
            PairPresence::new(2.0, false).unwrap().with_r1(1.0),
            Err(ComputeError::OutOfRange { .. })
        ));
        let none: [&(dyn PairSource + Sync); 0] = [];
        assert!(matches!(
            PairPresence::new(1.0, false)
                .unwrap()
                .compute(PairPresenceInput { frames: &none }),
            Err(ComputeError::EmptyInput)
        ));

        let lean = |mode| {
            Neighbors::from_pairs(
                [NeighborPair {
                    i: 0,
                    j: 1,
                    dist_sq: 0.25,
                    disp: [0.5, 0.0, 0.0],
                }],
                NeighborColumns::INDICES_ONLY,
                mode,
            )
        };
        let self_mode = QueryMode::SelfQuery { n_points: 2 };
        let no_dist = lean(self_mode);
        let frames: [&(dyn PairSource + Sync); 1] = [&no_dist];
        assert!(matches!(
            PairPresence::new(1.0, false)
                .unwrap()
                .compute(PairPresenceInput { frames: &frames }),
            Err(ComputeError::BadShape { .. })
        ));

        let full = |mode| {
            Neighbors::from_pairs(
                [NeighborPair {
                    i: 0,
                    j: 1,
                    dist_sq: 0.25,
                    disp: [0.5, 0.0, 0.0],
                }],
                NeighborColumns::FULL,
                mode,
            )
        };
        let a = full(self_mode);
        let b = full(QueryMode::SelfQuery { n_points: 3 });
        let frames: [&(dyn PairSource + Sync); 2] = [&a, &b];
        assert!(matches!(
            PairPresence::new(1.0, false)
                .unwrap()
                .compute(PairPresenceInput { frames: &frames }),
            Err(ComputeError::BadShape { .. })
        ));
    }
}
