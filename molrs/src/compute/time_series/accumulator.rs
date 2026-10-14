//! The streaming form of a [`Correlator`]: one frame at a time, bounded
//! memory.

use std::collections::VecDeque;

use molrs::core::SimBox;
use molrs::op::F;
use ndarray::{Array2, ArrayView2, ArrayViewMut1};

use super::correlator::linear_term;
use super::guards::{first_wrapped_item, require_finite, require_presence};
use super::{Correlator, LagKernel, MeanPolicy, Origins, TimeCorrelation};
use crate::compute::{Accumulate, ComputeError, Stream};

/// One frame of the series a [`CorrelatorAccumulator`] folds.
#[derive(Debug, Clone, Copy)]
pub struct CorrelatorFrame<'a> {
    /// This frame of the series, `N × D`.
    pub a: ArrayView2<'a, F>,
    /// This frame of the second series of a cross-correlation; `None` for an
    /// autocorrelation. Must be given on every frame or on none.
    pub b: Option<ArrayView2<'a, F>>,
    /// The sample interval of the series; identical on every frame.
    pub dt: F,
    /// This frame's box, for the displacement kernel's wrapped-trajectory
    /// check against the previous frame; `None` skips the check.
    pub simbox: Option<&'a SimBox>,
}

/// What the first frame fixed.
#[derive(Debug, Clone, Copy)]
struct Latched {
    n_items: usize,
    n_components: usize,
    cross: bool,
    dt: F,
}

/// The accumulator of a [`Correlator`], from [`Stream::accumulator`].
///
/// Each pushed frame is correlated against the ring of the last `max_lag`
/// frames (all origins) or against the first frame (first origin), so memory
/// is `O(max_lag · N · D)` whatever the trajectory length. A mean-subtracted
/// product cannot subtract a mean it has not seen yet, so it accumulates raw
/// lagged products and applies the exact correction at
/// [`finish`](Accumulate::finish):
///
/// ```text
/// Σ_τ (x_τ − μ_x)(y_{τ+t} − μ_y)
///   = Σ_τ x_τ y_{τ+t} − μ_y (S_x − tail_t(x)) − μ_x (S_y − head_t(y)) + (T − t) μ_x μ_y
/// ```
///
/// with `S` the running totals and `head_t` / `tail_t` the sums of the first /
/// last `t` samples, which the retained head and ring hold. The result equals
/// [`Correlator::compute`](crate::compute::Compute::compute) on the same
/// frames to FFT round-off at every lag.
#[derive(Debug, Clone)]
pub struct CorrelatorAccumulator {
    correlator: Correlator,
    latched: Option<Latched>,
    n_frames: usize,
    /// Raw lagged sums, `(max_lag + 1) × channels`.
    sums: Array2<F>,
    /// The first `max_lag + 1` frames of `a` and `b`.
    head_a: Vec<Array2<F>>,
    head_b: Vec<Array2<F>>,
    /// The last `max(max_lag, 1)` frames of `a` and `b`, newest at the back.
    ring_a: VecDeque<Array2<F>>,
    ring_b: VecDeque<Array2<F>>,
    /// Running per-column totals of `a` and `b`.
    total_a: Array2<F>,
    total_b: Array2<F>,
    /// Persistence: consecutive present frames per item, ending now.
    runs: Vec<usize>,
}

impl CorrelatorAccumulator {
    fn new(correlator: Correlator) -> Self {
        Self {
            correlator,
            latched: None,
            n_frames: 0,
            sums: Array2::zeros((0, 0)),
            head_a: Vec::new(),
            head_b: Vec::new(),
            ring_a: VecDeque::new(),
            ring_b: VecDeque::new(),
            total_a: Array2::zeros((0, 0)),
            total_b: Array2::zeros((0, 0)),
            runs: Vec::new(),
        }
    }

    /// Number of frames folded so far.
    pub fn n_frames(&self) -> usize {
        self.n_frames
    }

    /// Validate `frame` against the configuration and the frames before it,
    /// latching the layout on the first frame.
    fn admit(&mut self, frame: &CorrelatorFrame<'_>) -> Result<Latched, ComputeError> {
        let (n_items, n_components) = frame.a.dim();
        if n_items == 0 || n_components == 0 {
            return Err(ComputeError::EmptyInput);
        }
        if let Some(b) = frame.b
            && b.dim() != frame.a.dim()
        {
            return Err(ComputeError::BadShape {
                expected: format!("b of shape {:?}, like a", frame.a.shape()),
                got: format!("{:?}", b.shape()),
            });
        }
        let cross = frame.b.is_some();
        let latched = match self.latched {
            None => {
                self.correlator.check_layout(n_components, cross)?;
                if !(frame.dt.is_finite() && frame.dt > 0.0) {
                    return Err(ComputeError::OutOfRange {
                        field: "dt",
                        value: frame.dt.to_string(),
                    });
                }
                let latched = Latched {
                    n_items,
                    n_components,
                    cross,
                    dt: frame.dt,
                };
                let n_channels = self.correlator.n_channels(n_items, n_components);
                self.sums = Array2::zeros((self.correlator.max_lag() + 1, n_channels));
                self.total_a = Array2::zeros((n_items, n_components));
                self.total_b = Array2::zeros((n_items, n_components));
                self.runs = vec![0; n_items];
                self.latched = Some(latched);
                latched
            }
            Some(latched) => {
                if n_items != latched.n_items {
                    return Err(ComputeError::DimensionMismatch {
                        expected: latched.n_items,
                        got: n_items,
                        what: "items per frame (latched by the first frame)",
                    });
                }
                if n_components != latched.n_components {
                    return Err(ComputeError::DimensionMismatch {
                        expected: latched.n_components,
                        got: n_components,
                        what: "components per item (latched by the first frame)",
                    });
                }
                if cross != latched.cross {
                    return Err(ComputeError::BadShape {
                        expected: format!(
                            "b given on every frame or none (first frame: {})",
                            latched.cross
                        ),
                        got: format!("b given: {cross}"),
                    });
                }
                if frame.dt != latched.dt {
                    return Err(ComputeError::OutOfRange {
                        field: "dt",
                        value: format!(
                            "{} differs from {} of the earlier frames",
                            frame.dt, latched.dt
                        ),
                    });
                }
                latched
            }
        };
        require_finite(frame.a.iter(), "a")?;
        if let Some(b) = frame.b {
            require_finite(b.iter(), "b")?;
        }
        if self.correlator.kernel() == LagKernel::Persistence {
            require_presence(frame.a.iter().copied())?;
        }
        if let Some(simbox) = frame.simbox {
            if self.correlator.kernel() != LagKernel::SquaredDisplacement {
                return Err(ComputeError::OutOfRange {
                    field: "simbox",
                    value: format!("the {} kernel takes no boxes", self.correlator.kernel()),
                });
            }
            let previous = [self.ring_a.back(), self.ring_b.back()];
            for (prev, now) in previous.into_iter().zip([Some(frame.a), frame.b]) {
                if let (Some(prev), Some(now)) = (prev, now)
                    && let Some(item) = first_wrapped_item(prev.view(), now, simbox)?
                {
                    return Err(ComputeError::Wrapped {
                        frame: self.n_frames,
                        item,
                    });
                }
            }
        }
        Ok(latched)
    }

    /// Fold the mean subtraction of a product correlation into raw sums.
    fn subtract_means(&self, sums: &mut Array2<F>, latched: Latched) {
        let n = self.n_frames;
        let inv_n = 1.0 / n as F;
        let max_lag = self.correlator.max_lag();
        let head_b = if latched.cross {
            &self.head_b
        } else {
            &self.head_a
        };
        let total_b = if latched.cross {
            &self.total_b
        } else {
            &self.total_a
        };
        for item in 0..latched.n_items {
            for component in 0..latched.n_components {
                let channel = self.correlator.channel_of(item, component);
                let (s_a, s_b) = (self.total_a[[item, component]], total_b[[item, component]]);
                let (mu_a, mu_b) = (s_a * inv_n, s_b * inv_n);
                match self.correlator.origins() {
                    Origins::All => {
                        let (mut tail_a, mut head_y) = (0.0, 0.0);
                        for lag in 0..=max_lag {
                            if lag > 0 {
                                tail_a += self.ring_a[self.ring_a.len() - lag][[item, component]];
                                head_y += head_b[lag - 1][[item, component]];
                            }
                            let origins = (n - lag) as F;
                            sums[[lag, channel]] += -mu_b * (s_a - tail_a) - mu_a * (s_b - head_y)
                                + origins * mu_a * mu_b;
                        }
                    }
                    Origins::First => {
                        let a0 = self.head_a[0][[item, component]];
                        for lag in 0..=max_lag {
                            sums[[lag, channel]] +=
                                -mu_b * a0 - mu_a * head_b[lag][[item, component]] + mu_a * mu_b;
                        }
                    }
                }
            }
        }
    }
}

/// Add the kernel between an earlier frame `early = (x, y)` and the current
/// one `now = (x, y)` into one lag's row of raw sums.
fn add_lag(
    correlator: Correlator,
    runs: &[usize],
    lag: usize,
    mut row: ArrayViewMut1<'_, F>,
    early: (&Array2<F>, &Array2<F>),
    now: (&Array2<F>, &Array2<F>),
) {
    let kernel = correlator.kernel();
    let ((x_early, y_early), (x_now, y_now)) = (early, now);
    for (item, (xe, xn)) in x_early.outer_iter().zip(x_now.outer_iter()).enumerate() {
        match kernel {
            LagKernel::Product | LagKernel::SquaredDisplacement => {
                for component in 0..xe.len() {
                    row[correlator.channel_of(item, component)] += linear_term(
                        kernel,
                        [xe[component], xn[component]],
                        [y_early[[item, component]], y_now[[item, component]]],
                    );
                }
            }
            LagKernel::Persistence => {
                // The presence that began `lag` frames ago is still unbroken.
                if runs[item] > lag {
                    row[correlator.channel_of(item, 0)] += 1.0;
                }
            }
            LagKernel::Legendre { .. } | LagKernel::WignerD { .. } => {
                row[correlator.channel_of(item, 0)] += kernel.item_value(
                    xe.as_slice().expect("stored frames are in standard layout"),
                    xn.as_slice().expect("stored frames are in standard layout"),
                );
            }
        }
    }
}

impl Accumulate for CorrelatorAccumulator {
    type Item<'a> = CorrelatorFrame<'a>;
    type Output = TimeCorrelation;

    fn push(&mut self, frame: CorrelatorFrame<'_>) -> Result<(), ComputeError> {
        let latched = self.admit(&frame)?;
        let mut a = frame.a.as_standard_layout().into_owned();
        self.correlator
            .prepare_sample(a.view_mut(), self.n_frames)?;
        let b = frame.b.map(|b| b.as_standard_layout().into_owned());
        if self.correlator.kernel() == LagKernel::Persistence {
            for (run, present) in self.runs.iter_mut().zip(a.iter()) {
                *run = if *present == 1.0 { *run + 1 } else { 0 };
            }
        }

        let t = self.n_frames;
        let max_lag = self.correlator.max_lag();
        let y_now = b.as_ref().unwrap_or(&a);
        let correlator = self.correlator;
        let (ring_y, head_y) = if latched.cross {
            (&self.ring_b, &self.head_b)
        } else {
            (&self.ring_a, &self.head_a)
        };
        match correlator.origins() {
            Origins::All => {
                for lag in 0..=max_lag.min(t) {
                    let early = if lag == 0 {
                        (&a, y_now)
                    } else {
                        let at = self.ring_a.len() - lag;
                        (&self.ring_a[at], &ring_y[at])
                    };
                    add_lag(
                        correlator,
                        &self.runs,
                        lag,
                        self.sums.row_mut(lag),
                        early,
                        (&a, y_now),
                    );
                }
            }
            Origins::First => {
                if t <= max_lag {
                    let early = if t == 0 {
                        (&a, y_now)
                    } else {
                        (&self.head_a[0], &head_y[0])
                    };
                    add_lag(
                        correlator,
                        &self.runs,
                        t,
                        self.sums.row_mut(t),
                        early,
                        (&a, y_now),
                    );
                }
            }
        }

        self.total_a += &a;
        if let Some(b) = &b {
            self.total_b += b;
        }
        if self.head_a.len() <= max_lag {
            self.head_a.push(a.clone());
            if let Some(b) = &b {
                self.head_b.push(b.clone());
            }
        }
        self.ring_a.push_back(a);
        if let Some(b) = b {
            self.ring_b.push_back(b);
        }
        // The ring holds the last `max_lag` frames, and at least the previous
        // frame for the wrapped-trajectory check.
        if self.ring_a.len() > max_lag && self.ring_a.len() > 1 {
            self.ring_a.pop_front();
            if latched.cross {
                self.ring_b.pop_front();
            }
        }
        self.n_frames += 1;
        Ok(())
    }

    fn finish(&self) -> Result<TimeCorrelation, ComputeError> {
        let Some(latched) = self.latched else {
            return Err(ComputeError::EmptyInput);
        };
        self.correlator.check_length(self.n_frames, latched.dt)?;
        let mut sums = self.sums.clone();
        if self.correlator.kernel() == LagKernel::Product
            && self.correlator.mean() == MeanPolicy::Subtract
        {
            self.subtract_means(&mut sums, latched);
        }
        Ok(self
            .correlator
            .normalize(sums, self.n_frames, latched.n_items, latched.dt))
    }
}

/// A correlator streams through its ring accumulator. Its
/// [`compute`](crate::compute::Compute::compute) is the faster batch
/// evaluation (FFT where the kernel allows), pinned equal to the fold of this
/// accumulator by the unit tests.
impl Stream for Correlator {
    type Accumulator = CorrelatorAccumulator;

    fn accumulator(&self) -> CorrelatorAccumulator {
        CorrelatorAccumulator::new(*self)
    }
}
