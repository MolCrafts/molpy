//! [`Survival`] — survival correlation of 0/1 presence series.

use molrs::op::F;
use ndarray::{ArrayView1, ArrayView2, Axis};

use super::{
    Correlator, CorrelatorAccumulator, CorrelatorFrame, CorrelatorInput, LagKernel, TimeCorrelation,
};
use crate::compute::named_enum::named_enum;
use crate::compute::{Accumulate, Compute, ComputeError, Stream};

named_enum! {
    /// When a bond (pair, contact, …) counts as surviving a lag.
    pub enum SurvivalKind {
        /// Present at every frame of `[τ, τ + t]` — the continuous `S(t)`.
        Continuous => "continuous",
        /// Present at `τ` and at `τ + t`, gaps allowed — the intermittent
        /// `C(t) = ⟨h(0) h(t)⟩`.
        Intermittent => "intermittent",
    }
}

named_enum! {
    /// What a survival curve is averaged over.
    #[derive(Default)]
    pub enum SurvivalNormalization {
        /// The mean over the bonds (the presence columns): `C(0) = ⟨h⟩`.
        #[default]
        PerPair => "per_pair",
        /// The sum over the bonds per reference particle, so `C(0)` is the
        /// mean coordination number — the convention of the former
        /// `pair_survival_tcf`. The input gives the reference count
        /// ([`SurvivalInput::n_reference`]).
        PerReference => "per_reference",
    }
}

named_enum! {
    /// Which frames start a survival run — the time origins `τ` a bond counts
    /// from.
    #[derive(Default)]
    pub enum SurvivalOrigin {
        /// Every frame at which the bond is present: with a hysteresis
        /// presence, every frame of the bonded stable state.
        #[default]
        Present => "present",
        /// Only the frames at which a second 0/1 series, `inner`, is `1` —
        /// a pair inside the inner cutoff `r0` (a
        /// [`PairPresence`](crate::compute::PairPresence) with `r1 = r0`)
        /// while survival follows the hysteresis presence. This is the
        /// stable-states origin of the former `pair_survival_tcf` (`Ssp`):
        /// born inside `r0`, surviving while within `r1`.
        Inner => "inner",
    }
}

/// Survival time-correlation of presence series `h_b(t) ∈ {0, 1}`, one per
/// bond (Luzar & Chandler, *Phys. Rev. Lett.* **76**, 928 (1996)):
///
/// ```text
/// continuous:    S(t) = ⟨ h(τ) Π_{s ≤ t} h(τ + s) ⟩
/// intermittent:  C(t) = ⟨ h(τ) h(τ + t) ⟩
/// ```
///
/// averaged over bonds and every origin `τ`; `S(0) = C(0) = ⟨h⟩`. The
/// series is a [`Presence`](crate::compute::Presence)'s `values`, from
/// [`PairPresence`](crate::compute::PairPresence) (distance criterion, with
/// hysteresis for the stable-states picture) or
/// [`HBondsResult::presence`](crate::compute::HBondsResult::presence). Divide by
/// the value at lag 0 for the normalised curves, and integrate
/// ([`CumulativeTrapezoid`](crate::compute::CumulativeTrapezoid)) for a
/// lifetime. A [`Correlator`] with the [`LagKernel::Persistence`] kernel
/// (continuous) or the [`LagKernel::Product`] kernel (intermittent),
/// unbiased, averaged over bonds.
///
/// Two conventions are configuration, each defaulting to the per-bond
/// stable-states form:
///
/// - [`with_normalization`](Self::with_normalization): per bond (default) or
///   per reference particle ([`SurvivalNormalization`]); the latter is the
///   per-bond curve times `n_bonds / n_reference`.
/// - [`with_origin`](Self::with_origin): origins wherever the bond is present
///   (default) or only where the `inner` series is ([`SurvivalOrigin`]); the
///   latter correlates `inner(τ)` against the presence that must last — the
///   [`Correlator`] crossed with `a = inner`, `b = presence`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Survival {
    correlator: Correlator,
    normalization: SurvivalNormalization,
    origin: SurvivalOrigin,
}

/// The data [`Survival`] analyses.
#[derive(Debug, Clone, Copy)]
pub struct SurvivalInput<'a> {
    /// Presence per frame and bond, `T × n_bonds`, every value `0` or `1`.
    pub presence: ArrayView2<'a, F>,
    /// Frame spacing.
    pub dt: F,
    /// The 0/1 series that marks a frame as an origin, shaped like
    /// `presence` and `1` only where `presence` is; required by
    /// [`SurvivalOrigin::Inner`], refused otherwise.
    pub inner: Option<ArrayView2<'a, F>>,
    /// The number of reference particles the bonds were found around;
    /// required by [`SurvivalNormalization::PerReference`], refused
    /// otherwise.
    pub n_reference: Option<usize>,
}

impl Survival {
    /// The `kind` survival correlation over lags `0 … max_lag` frames,
    /// per bond, with origins wherever a bond is present.
    pub fn new(kind: SurvivalKind, max_lag: usize) -> Self {
        let kernel = match kind {
            SurvivalKind::Continuous => LagKernel::Persistence,
            SurvivalKind::Intermittent => LagKernel::Product,
        };
        Self {
            correlator: Correlator::new(kernel, max_lag),
            normalization: SurvivalNormalization::default(),
            origin: SurvivalOrigin::default(),
        }
    }

    /// Average per bond or per reference particle (default
    /// [`SurvivalNormalization::PerPair`]).
    pub fn with_normalization(mut self, normalization: SurvivalNormalization) -> Self {
        self.normalization = normalization;
        self
    }

    /// Start runs wherever a bond is present or only where `inner` is
    /// (default [`SurvivalOrigin::Present`]).
    pub fn with_origin(mut self, origin: SurvivalOrigin) -> Self {
        self.origin = origin;
        self
    }

    /// The correlator this survival correlation is.
    pub fn correlator(&self) -> &Correlator {
        &self.correlator
    }

    /// The normalisation.
    pub fn normalization(&self) -> SurvivalNormalization {
        self.normalization
    }

    /// The origin rule.
    pub fn origin(&self) -> SurvivalOrigin {
        self.origin
    }

    /// The `inner` series the origin rule needs, checked against `presence`.
    fn origin_series<'a>(
        &self,
        presence: ArrayView2<'_, F>,
        inner: Option<ArrayView2<'a, F>>,
    ) -> Result<Option<ArrayView2<'a, F>>, ComputeError> {
        match (self.origin, inner) {
            (SurvivalOrigin::Present, None) => Ok(None),
            (SurvivalOrigin::Present, Some(_)) => Err(ComputeError::OutOfRange {
                field: "inner",
                value: "given, but the present origin rule does not read it".into(),
            }),
            (SurvivalOrigin::Inner, None) => Err(ComputeError::OutOfRange {
                field: "inner",
                value: "missing: the inner origin rule starts runs where it is 1".into(),
            }),
            (SurvivalOrigin::Inner, Some(inner)) => {
                if inner.dim() != presence.dim() {
                    return Err(ComputeError::BadShape {
                        expected: format!("inner of shape {:?}, like presence", presence.shape()),
                        got: format!("{:?}", inner.shape()),
                    });
                }
                let stray = inner
                    .iter()
                    .zip(presence.iter())
                    .position(|(&origin, &present)| origin == 1.0 && present != 1.0);
                if let Some(index) = stray {
                    return Err(ComputeError::OutOfRange {
                        field: "inner",
                        value: format!("1 at index {index}, where the bond is not present"),
                    });
                }
                Ok(Some(inner))
            }
        }
    }

    /// The factor turning the per-bond curve into the configured
    /// normalisation: `1`, or `n_bonds / n_reference`.
    fn scale(&self, n_bonds: usize, n_reference: Option<usize>) -> Result<F, ComputeError> {
        match (self.normalization, n_reference) {
            (SurvivalNormalization::PerPair, None) => Ok(1.0),
            (SurvivalNormalization::PerPair, Some(n)) => Err(ComputeError::OutOfRange {
                field: "n_reference",
                value: format!("{n} given, but the per_pair normalization does not read it"),
            }),
            (SurvivalNormalization::PerReference, None | Some(0)) => {
                Err(ComputeError::OutOfRange {
                    field: "n_reference",
                    value: format!(
                        "{n_reference:?}: the per_reference normalization divides by a \
                         positive reference count"
                    ),
                })
            }
            (SurvivalNormalization::PerReference, Some(n)) => Ok(n_bonds as F / n as F),
        }
    }
}

impl Compute for Survival {
    type Input<'a> = SurvivalInput<'a>;
    type Output = TimeCorrelation;

    /// # Errors
    /// The [`Correlator`]'s, and [`ComputeError::OutOfRange`] /
    /// [`ComputeError::BadShape`] for an `inner` or `n_reference` the
    /// configuration does not match (missing, refused, misshapen, or `1`
    /// where the bond is absent; a zero reference count).
    fn compute(&self, input: SurvivalInput<'_>) -> Result<TimeCorrelation, ComputeError> {
        let inner = self.origin_series(input.presence, input.inner)?;
        let scale = self.scale(input.presence.ncols(), input.n_reference)?;
        let mut curve = self.correlator.compute(CorrelatorInput {
            a: inner.unwrap_or(input.presence).insert_axis(Axis(2)),
            b: inner.map(|_| input.presence.insert_axis(Axis(2))),
            dt: input.dt,
            boxes: None,
        })?;
        curve.values *= scale;
        Ok(curve)
    }
}

/// One frame of the presence a [`SurvivalAccumulator`] folds — a row of
/// [`SurvivalInput`].
#[derive(Debug, Clone, Copy)]
pub struct SurvivalFrame<'a> {
    /// This frame's presence, one value per bond.
    pub presence: ArrayView1<'a, F>,
    /// The sample interval; identical on every frame.
    pub dt: F,
    /// This frame's origin series ([`SurvivalInput::inner`]).
    pub inner: Option<ArrayView1<'a, F>>,
    /// The reference count ([`SurvivalInput::n_reference`]); identical on
    /// every frame.
    pub n_reference: Option<usize>,
}

/// The streaming form of a [`Survival`]: its [`Correlator`]'s accumulator,
/// fed `inner` / `presence` per the origin rule, and the normalisation
/// applied at [`finish`](Accumulate::finish).
#[derive(Debug, Clone)]
pub struct SurvivalAccumulator {
    survival: Survival,
    correlator: CorrelatorAccumulator,
    /// `(n_bonds, n_reference)`, latched by the first frame.
    latched: Option<(usize, Option<usize>)>,
}

impl Accumulate for SurvivalAccumulator {
    type Item<'a> = SurvivalFrame<'a>;
    type Output = TimeCorrelation;

    fn push(&mut self, frame: SurvivalFrame<'_>) -> Result<(), ComputeError> {
        let presence = frame.presence.insert_axis(Axis(0));
        let inner = self
            .survival
            .origin_series(presence, frame.inner.map(|i| i.insert_axis(Axis(0))))?;
        self.survival
            .scale(frame.presence.len(), frame.n_reference)?;
        match self.latched {
            None => self.latched = Some((frame.presence.len(), frame.n_reference)),
            Some((_, n_reference)) if n_reference != frame.n_reference => {
                return Err(ComputeError::OutOfRange {
                    field: "n_reference",
                    value: format!(
                        "{:?} differs from {n_reference:?} of the earlier frames",
                        frame.n_reference
                    ),
                });
            }
            Some(_) => {}
        }
        self.correlator.push(CorrelatorFrame {
            a: inner.unwrap_or(presence).reversed_axes(),
            b: inner.map(|_| presence.reversed_axes()),
            dt: frame.dt,
            simbox: None,
        })
    }

    fn finish(&self) -> Result<TimeCorrelation, ComputeError> {
        let mut curve = self.correlator.finish()?;
        let (n_bonds, n_reference) = self.latched.ok_or(ComputeError::EmptyInput)?;
        curve.values *= self.survival.scale(n_bonds, n_reference)?;
        Ok(curve)
    }
}

impl Stream for Survival {
    type Accumulator = SurvivalAccumulator;

    fn accumulator(&self) -> SurvivalAccumulator {
        SurvivalAccumulator {
            survival: *self,
            correlator: self.correlator.accumulator(),
            latched: None,
        }
    }
}

crate::compute::spec::compute_spec! {
    Survival as "Survival";
    params |s| {
        kind: Choice(SurvivalKind::NAMES);
        max_lag: Int;
        normalization: Choice(SurvivalNormalization::NAMES)
            = SurvivalNormalization::default() => s.normalization;
        origin: Choice(SurvivalOrigin::NAMES) = SurvivalOrigin::default() => s.origin;
    }
    inputs {
        presence: Series;
        dt: Scalar;
        inner: Series, optional;
        n_reference: Count, optional;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    /// One bond: present, present, absent, present, present, present.
    fn one_bond() -> Array2<F> {
        Array2::from_shape_vec((6, 1), vec![1.0, 1.0, 0.0, 1.0, 1.0, 1.0]).unwrap()
    }

    fn input(h: &Array2<F>) -> SurvivalInput<'_> {
        SurvivalInput {
            presence: h.view(),
            dt: 1.0,
            inner: None,
            n_reference: None,
        }
    }

    fn survival(kind: SurvivalKind, h: &Array2<F>, max_lag: usize) -> TimeCorrelation {
        Survival::new(kind, max_lag).compute(input(h)).unwrap()
    }

    #[test]
    fn surface_is_its_correlator() {
        let h = one_bond();
        for (kind, kernel) in [
            (SurvivalKind::Continuous, LagKernel::Persistence),
            (SurvivalKind::Intermittent, LagKernel::Product),
        ] {
            let explicit = Correlator::new(kernel, 3);
            assert_eq!(*Survival::new(kind, 3).correlator(), explicit);
            let via_correlator = explicit
                .compute(CorrelatorInput {
                    a: h.view().insert_axis(Axis(2)),
                    b: None,
                    dt: 1.0,
                    boxes: None,
                })
                .unwrap();
            assert_eq!(survival(kind, &h, 3), via_correlator);
        }
    }

    /// Counted by hand on `1 1 0 1 1 1`.
    #[test]
    fn continuous_and_intermittent_by_hand() {
        let h = one_bond();
        let s = survival(SurvivalKind::Continuous, &h, 3);
        let c = survival(SurvivalKind::Intermittent, &h, 3);
        // lag 0: 5 present of 6 origins, for both.
        // lag 1: unbroken pairs (0,1) (3,4) (4,5): 3 of 5 — also 3 of 5 gapped.
        // lag 2: unbroken run 3..=5 only: 1 of 4; gapped (1,3) (3,5): 2 of 4.
        // lag 3: no run of 4: 0 of 3; gapped (0,3) (1,4): 2 of 3.
        let continuous = [5.0 / 6.0, 3.0 / 5.0, 1.0 / 4.0, 0.0];
        let intermittent = [5.0 / 6.0, 3.0 / 5.0, 2.0 / 4.0, 2.0 / 3.0];
        for lag in 0..=3 {
            assert!(
                (s.values[[lag, 0]] - continuous[lag]).abs() < 1e-12,
                "S({lag})"
            );
            assert!(
                (c.values[[lag, 0]] - intermittent[lag]).abs() < 1e-12,
                "C({lag})"
            );
        }
    }

    /// Per reference particle the per-bond curve is scaled by
    /// `n_bonds / n_reference`: two always-present bonds around one
    /// reference give `C(t) = 2`, the coordination number.
    #[test]
    fn per_reference_is_the_coordination_number() {
        let h = Array2::from_elem((4, 2), 1.0);
        for kind in [SurvivalKind::Continuous, SurvivalKind::Intermittent] {
            let c = Survival::new(kind, 2)
                .with_normalization(SurvivalNormalization::PerReference)
                .compute(SurvivalInput {
                    n_reference: Some(1),
                    ..input(&h)
                })
                .unwrap();
            assert_eq!(c.values.column(0).to_vec(), vec![2.0; 3], "{kind}");
        }
    }

    /// Origins only where `inner` is 1, survival on the presence. Presence
    /// `1 1 0 1 1 1`, inner `1 0 0 1 0 0`. Continuous: lag 0 counts origins
    /// 0 and 3 of 6; lag 1 the runs (0,1) and (3,4) of 5; lag 2 only 3..=5
    /// of 4; lag 3 none of 3. Intermittent, `inner(τ) · h(τ + t)`: lags 0
    /// and 1 as continuous; lag 2 only (3,5) of 4 (`h(2) = 0`); lag 3 (0,3)
    /// of 3.
    #[test]
    fn inner_origins_by_hand() {
        let h = one_bond();
        let inner = Array2::from_shape_vec((6, 1), vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0]).unwrap();
        let run = |kind| {
            Survival::new(kind, 3)
                .with_origin(SurvivalOrigin::Inner)
                .compute(SurvivalInput {
                    inner: Some(inner.view()),
                    ..input(&h)
                })
                .unwrap()
        };
        let s = run(SurvivalKind::Continuous);
        let c = run(SurvivalKind::Intermittent);
        let continuous = [2.0 / 6.0, 2.0 / 5.0, 1.0 / 4.0, 0.0];
        let intermittent = [2.0 / 6.0, 2.0 / 5.0, 1.0 / 4.0, 1.0 / 3.0];
        for lag in 0..=3 {
            assert!(
                (s.values[[lag, 0]] - continuous[lag]).abs() < 1e-12,
                "S({lag})"
            );
            assert!(
                (c.values[[lag, 0]] - intermittent[lag]).abs() < 1e-12,
                "C({lag})"
            );
        }
    }

    /// An `inner` / `n_reference` the configuration does not read, or lacks,
    /// is an error — never ignored, never defaulted.
    #[test]
    fn mismatched_conventions_are_errors() {
        let h = one_bond();
        let inner = Array2::from_shape_vec((6, 1), vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0]).unwrap();
        let field = |s: Survival, i: SurvivalInput<'_>| match s.compute(i).unwrap_err() {
            ComputeError::OutOfRange { field, .. } => field,
            other => panic!("expected OutOfRange, got {other:?}"),
        };
        let plain = Survival::new(SurvivalKind::Continuous, 2);
        let gated = plain.with_origin(SurvivalOrigin::Inner);
        let per_reference = plain.with_normalization(SurvivalNormalization::PerReference);
        let with_inner = SurvivalInput {
            inner: Some(inner.view()),
            ..input(&h)
        };
        assert_eq!(field(plain, with_inner), "inner");
        assert_eq!(field(gated, input(&h)), "inner");
        assert_eq!(
            field(
                plain,
                SurvivalInput {
                    n_reference: Some(1),
                    ..input(&h)
                }
            ),
            "n_reference"
        );
        assert_eq!(field(per_reference, input(&h)), "n_reference");
        assert_eq!(
            field(
                per_reference,
                SurvivalInput {
                    n_reference: Some(0),
                    ..input(&h)
                }
            ),
            "n_reference"
        );
        // An origin where the bond is absent (frame 2) — e.g. swapped series.
        let stray = Array2::from_shape_vec((6, 1), vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0]).unwrap();
        assert_eq!(
            field(
                gated,
                SurvivalInput {
                    inner: Some(stray.view()),
                    ..input(&h)
                }
            ),
            "inner"
        );
        let short = Array2::from_elem((5, 1), 1.0);
        assert!(matches!(
            gated.compute(SurvivalInput {
                inner: Some(short.view()),
                ..input(&h)
            }),
            Err(ComputeError::BadShape { .. })
        ));
    }

    /// The stream folds the same curve as the batch, for every convention.
    #[test]
    fn streaming_equals_batch_for_every_convention() {
        let h = Array2::from_shape_vec(
            (8, 3),
            vec![
                1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, //
                1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            ],
        )
        .unwrap();
        let inner = &h * &Array2::from_shape_fn((8, 3), |(t, b)| ((t + b) % 2) as F);
        for kind in [SurvivalKind::Continuous, SurvivalKind::Intermittent] {
            for &origin in SurvivalOrigin::ALL {
                for &normalization in SurvivalNormalization::ALL {
                    let s = Survival::new(kind, 4)
                        .with_origin(origin)
                        .with_normalization(normalization);
                    let gate = (origin == SurvivalOrigin::Inner).then_some(&inner);
                    let n_reference =
                        (normalization == SurvivalNormalization::PerReference).then_some(2);
                    let batch = s
                        .compute(SurvivalInput {
                            presence: h.view(),
                            dt: 0.5,
                            inner: gate.map(|g| g.view()),
                            n_reference,
                        })
                        .unwrap();
                    let mut acc = s.accumulator();
                    for t in 0..h.nrows() {
                        acc.push(SurvivalFrame {
                            presence: h.row(t),
                            dt: 0.5,
                            inner: gate.map(|g| g.row(t)),
                            n_reference,
                        })
                        .unwrap();
                    }
                    let streamed = acc.finish().unwrap();
                    assert_eq!(streamed.lag_times, batch.lag_times);
                    for (a, b) in streamed.values.iter().zip(batch.values.iter()) {
                        assert!((a - b).abs() < 1e-12, "{kind} {origin} {normalization}");
                    }
                }
            }
        }
    }

    #[test]
    fn a_stream_keeps_one_reference_count() {
        let h = one_bond();
        let mut acc = Survival::new(SurvivalKind::Continuous, 2)
            .with_normalization(SurvivalNormalization::PerReference)
            .accumulator();
        let frame = |t: usize, n| SurvivalFrame {
            presence: h.row(t),
            dt: 1.0,
            inner: None,
            n_reference: Some(n),
        };
        acc.push(frame(0, 1)).unwrap();
        assert!(matches!(
            acc.push(frame(1, 2)),
            Err(ComputeError::OutOfRange {
                field: "n_reference",
                ..
            })
        ));
        assert!(matches!(
            Survival::new(SurvivalKind::Continuous, 2)
                .accumulator()
                .finish(),
            Err(ComputeError::EmptyInput)
        ));
    }

    #[test]
    fn presence_must_be_zero_or_one() {
        let mut h = one_bond();
        h[[2, 0]] = 0.5;
        let err = Survival::new(SurvivalKind::Continuous, 2)
            .compute(input(&h))
            .unwrap_err();
        assert!(matches!(err, ComputeError::OutOfRange { field: "a", .. }));
    }
}
