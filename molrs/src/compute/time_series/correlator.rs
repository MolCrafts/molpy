//! [`Correlator`] — the one time-correlation estimator of molrs.

use std::ops::Range;

use molrs::core::SimBox;
use molrs::op::F;
use ndarray::{Array1, Array2, Array3, ArrayView3, ArrayViewMut2, Axis};
use rustfft::FftPlanner;

use super::guards::{first_wrapped_item, require_finite, require_presence};
use super::{
    LagKernel, LagKernelKind, MeanPolicy, Normalization, Origins, Reduction, TimeCorrelation,
};
use crate::compute::{Compute, ComputeError};

mod column_fft;
use column_fft::ColumnFft;

/// Items per parallel work unit of the FFT path. Fixed, so the summation
/// order — and therefore every bit of the result — does not depend on the
/// number of threads.
const ITEMS_PER_CHUNK: usize = 32;

/// The time-correlation estimator: every lagged correlation in molrs is one
/// configuration of this type.
///
/// The data is a `T × N × D` series — `T` frames, `N` items (atoms,
/// molecules, degrees of freedom, bonds, …), `D` components per item — given
/// to [`compute`](Compute::compute) together with its sample interval `dt`.
/// For each lag `t = 0 … max_lag` the [`LagKernel`] is evaluated between the
/// samples at `τ` and `τ + t` of every item, summed over the [`Origins`] `τ`,
/// scaled by the [`Normalization`] and combined into channels by the
/// [`Reduction`]:
///
/// ```text
/// C(t) = ⊕_items  1/n(t) Σ_τ  K( x_i(τ), y_i(τ + t) )
/// ```
///
/// with `y = x` (auto) unless a second series `b` is given (cross), `n(t)` the
/// number of origins under `Normalization::Unbiased` (`T − t` for `All`,
/// `⌊(T − 1 − t)/s⌋ + 1` with an origin [stride](Self::with_stride) `s`, `1`
/// for `First`), `n(0)` under `Normalization::Biased` and `1` under
/// `Normalization::None`, and `⊕` the mean over items (`Reduction::Mean`), one
/// channel per item (`PerItem`) or one per component (`PerComponent`).
///
/// # Evaluation
///
/// The linear kernels over all origins run through the zero-padded FFT
/// (Wiener–Khinchin; Press et al., *Numerical Recipes* §13.2): the lagged
/// product is `IFFT(conj(X)·Y)` with padding to `(2T).next_power_of_two()`,
/// which makes the correlation linear rather than circular, in `O(T log T)`
/// per column; the lagged squared displacement adds the `S`-recurrence of
/// Kneller et al., *Comput. Phys. Commun.* **91**, 191 (1995) (nMoldyn).
/// A strided origin set stays on the FFT path as a cross-correlation of the
/// origin-masked series (the squared displacement expanded into masked
/// cross-correlations plus a prefix sum). The Legendre kernel of order
/// `l ≤ 2` factorises into products of the unit vectors and of their
/// quadratic components and runs on the same FFT. Everything else — a
/// single origin, or a kernel that is nonlinear in the item's vector — is
/// summed directly. The streaming form,
/// [`accumulator`](crate::compute::Stream::accumulator), folds one frame at a
/// time over a ring of the last `max_lag` frames and equals this batch result
/// to FFT round-off at every lag.
///
/// # Preconditions, all checked
///
/// At least two frames, at least one item and one component; `max_lag ≤ T − 1`
/// (a lag with no origin is [`ComputeError::OutOfRange`], never clamped);
/// `dt > 0`; every value finite. A displacement kernel given per-frame
/// `boxes` rejects a trajectory wrapped into the box: a frame-to-frame step
/// longer than half the box's shortest perpendicular height is
/// [`ComputeError::Wrapped`] — unwrap it first with
/// [`Unwrap`](super::Unwrap).
///
/// ```
/// use molrs::compute::{Compute, Correlator, CorrelatorInput, LagKernel, MeanPolicy};
/// use ndarray::Array3;
///
/// // One item, one component: x = [1, 2, 3].
/// let x = Array3::from_shape_vec((3, 1, 1), vec![1.0, 2.0, 3.0]).unwrap();
/// let acf = Correlator::new(LagKernel::Product, 1)
///     .compute(CorrelatorInput { a: x.view(), b: None, dt: 0.5, boxes: None })
///     .unwrap();
/// // C(0) = (1 + 4 + 9) / 3, C(1) = (1·2 + 2·3) / 2.
/// assert!((acf.values[[0, 0]] - 14.0 / 3.0).abs() < 1e-12);
/// assert!((acf.values[[1, 0]] - 4.0).abs() < 1e-12);
/// assert_eq!(acf.lag_times.to_vec(), vec![0.0, 0.5]);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Correlator {
    kernel: LagKernel,
    max_lag: usize,
    origins: Origins,
    mean: MeanPolicy,
    norm: Normalization,
    reduce: Reduction,
    stride: usize,
    dimension: Option<usize>,
}

/// The data a [`Correlator`] correlates.
#[derive(Debug, Clone, Copy)]
pub struct CorrelatorInput<'a> {
    /// The series, `T × N × D`.
    pub a: ArrayView3<'a, F>,
    /// A second series of the same shape for a cross-correlation (`a` at `τ`
    /// against `b` at `τ + t`); `None` for an autocorrelation.
    pub b: Option<ArrayView3<'a, F>>,
    /// Sample interval; sets [`TimeCorrelation::lag_times`].
    pub dt: F,
    /// One box per frame, for the displacement kernel only: enables the
    /// wrapped-trajectory check. `None` skips it.
    pub boxes: Option<&'a [SimBox]>,
}

impl Correlator {
    /// A correlator of `kernel` over lags `0 … max_lag` (in frames), with all
    /// origins, values kept as given, the unbiased normalisation and a mean
    /// over items.
    pub fn new(kernel: LagKernel, max_lag: usize) -> Self {
        Self {
            kernel,
            max_lag,
            origins: Origins::default(),
            mean: MeanPolicy::default(),
            norm: Normalization::default(),
            reduce: Reduction::default(),
            stride: Self::DEFAULT_STRIDE,
            dimension: None,
        }
    }

    /// The origin stride when not set: every frame is an origin.
    pub const DEFAULT_STRIDE: usize = 1;

    /// Take every `stride`-th frame, `τ = 0, s, 2s, …`, as a time origin
    /// (default [`DEFAULT_STRIDE`](Self::DEFAULT_STRIDE)): fewer, less
    /// correlated origins for a long trajectory. Applies to [`Origins::All`];
    /// [`Origins::First`] with a stride other than 1 is refused at `compute`.
    /// The linear kernels stay on the FFT path (a masked cross-correlation).
    ///
    /// # Errors
    /// [`ComputeError::OutOfRange`] for `stride == 0`.
    pub fn with_stride(mut self, stride: usize) -> Result<Self, ComputeError> {
        if stride == 0 {
            return Err(ComputeError::OutOfRange {
                field: "stride",
                value: stride.to_string(),
            });
        }
        self.stride = stride;
        Ok(self)
    }

    /// Average over these time origins.
    pub fn with_origins(mut self, origins: Origins) -> Self {
        self.origins = origins;
        self
    }

    /// Subtract (or keep) each column's time mean. Linear kernels only; for
    /// [`LagKernel::SquaredDisplacement`] both policies give the same curve.
    pub fn with_mean(mut self, mean: MeanPolicy) -> Self {
        self.mean = mean;
        self
    }

    /// Scale each lag by its origin count (or not).
    pub fn with_normalization(mut self, norm: Normalization) -> Self {
        self.norm = norm;
        self
    }

    /// Combine items and components into these channels.
    pub fn with_reduction(mut self, reduce: Reduction) -> Self {
        self.reduce = reduce;
        self
    }

    /// Require every item to have exactly `components` components (e.g. 3
    /// for a Cartesian vector); a series of another width is
    /// [`ComputeError::DimensionMismatch`].
    pub fn with_dimension(mut self, components: usize) -> Self {
        self.dimension = Some(components);
        self
    }

    /// The lag kernel.
    pub fn kernel(&self) -> LagKernel {
        self.kernel
    }

    /// The largest lag, in frames.
    pub fn max_lag(&self) -> usize {
        self.max_lag
    }

    /// The time origins.
    pub fn origins(&self) -> Origins {
        self.origins
    }

    /// The mean policy.
    pub fn mean(&self) -> MeanPolicy {
        self.mean
    }

    /// The normalisation.
    pub fn normalization(&self) -> Normalization {
        self.norm
    }

    /// The reduction.
    pub fn reduction(&self) -> Reduction {
        self.reduce
    }

    /// The origin stride.
    pub fn stride(&self) -> usize {
        self.stride
    }

    /// The number of time origins `n(t)` lag `lag` is averaged over in a
    /// trajectory of `n_frames`: `⌊(T − 1 − t)/s⌋ + 1` for [`Origins::All`]
    /// (`T − t` at stride 1), `1` for [`Origins::First`].
    pub(super) fn origin_count(&self, n_frames: usize, lag: usize) -> usize {
        match self.origins {
            Origins::All => (n_frames - 1 - lag) / self.stride + 1,
            Origins::First => 1,
        }
    }

    /// The required number of components per item, if fixed.
    pub fn dimension(&self) -> Option<usize> {
        self.dimension
    }

    /// Number of output channels for `n_items × n_components` data.
    pub(super) fn n_channels(&self, n_items: usize, n_components: usize) -> usize {
        match self.reduce {
            Reduction::Mean => 1,
            Reduction::PerItem => n_items,
            Reduction::PerComponent => n_components,
        }
    }

    /// The output channel an `(item, component)` column contributes to.
    pub(super) fn channel_of(&self, item: usize, component: usize) -> usize {
        match self.reduce {
            Reduction::Mean => 0,
            Reduction::PerItem => item,
            Reduction::PerComponent => component,
        }
    }

    /// Checks that depend on the configuration and the data's width only —
    /// shared by the batch and the streaming forms.
    pub(super) fn check_layout(
        &self,
        n_components: usize,
        cross: bool,
    ) -> Result<(), ComputeError> {
        for required in [self.dimension, self.kernel.required_components()]
            .into_iter()
            .flatten()
        {
            if n_components != required {
                return Err(ComputeError::DimensionMismatch {
                    expected: required,
                    got: n_components,
                    what: "components per item",
                });
            }
        }
        if self.origins == Origins::First && self.stride != 1 {
            return Err(ComputeError::OutOfRange {
                field: "stride (a single first origin has no stride)",
                value: self.stride.to_string(),
            });
        }
        if !self.kernel.is_linear() {
            let refuse = |field: &'static str, what: &str| {
                Err(ComputeError::OutOfRange {
                    field,
                    value: format!("{what} is undefined for the {} kernel", self.kernel),
                })
            };
            // Persistence crosses: `a` gates the origin, `b` must persist.
            if cross && self.kernel != LagKernel::Persistence {
                return refuse("b", "a cross-correlation");
            }
            if self.reduce == Reduction::PerComponent {
                return refuse("reduce", "per_component");
            }
            if self.mean == MeanPolicy::Subtract {
                return refuse("mean", "subtract");
            }
        }
        Ok(())
    }

    /// Checks on the trajectory length and `dt`, shared by both forms.
    pub(super) fn check_length(&self, n_frames: usize, dt: F) -> Result<(), ComputeError> {
        if n_frames < 2 {
            return Err(ComputeError::EmptyInput);
        }
        if self.max_lag > n_frames - 1 {
            return Err(ComputeError::OutOfRange {
                field: "max_lag",
                value: format!(
                    "{} exceeds n_frames - 1 = {}: that lag has no time origin",
                    self.max_lag,
                    n_frames - 1
                ),
            });
        }
        if !(dt.is_finite() && dt > 0.0) {
            return Err(ComputeError::OutOfRange {
                field: "dt",
                value: dt.to_string(),
            });
        }
        Ok(())
    }

    /// Scale raw lagged sums `(max_lag + 1) × channels` into the curve.
    pub(super) fn normalize(
        &self,
        mut sums: Array2<F>,
        n_frames: usize,
        n_items: usize,
        dt: F,
    ) -> TimeCorrelation {
        let per_item = match self.reduce {
            Reduction::Mean | Reduction::PerComponent => 1.0 / n_items as F,
            Reduction::PerItem => 1.0,
        };
        for (lag, mut row) in sums.axis_iter_mut(Axis(0)).enumerate() {
            let factor = match self.norm {
                Normalization::Unbiased => per_item / self.origin_count(n_frames, lag) as F,
                Normalization::Biased => per_item / self.origin_count(n_frames, 0) as F,
                Normalization::None => per_item,
            };
            row *= factor;
        }
        TimeCorrelation {
            lag_times: Array1::from_iter((0..=self.max_lag).map(|i| i as F * dt)),
            values: sums,
        }
    }

    /// Validate the batch input; returns `(T, N, D)`.
    fn validate(&self, input: &CorrelatorInput<'_>) -> Result<(usize, usize, usize), ComputeError> {
        let (n_frames, n_items, n_components) = input.a.dim();
        if n_frames == 0 || n_items == 0 || n_components == 0 {
            return Err(ComputeError::EmptyInput);
        }
        if let Some(b) = input.b
            && b.dim() != input.a.dim()
        {
            return Err(ComputeError::BadShape {
                expected: format!("b of shape {:?}, like a", input.a.shape()),
                got: format!("{:?}", b.shape()),
            });
        }
        self.check_length(n_frames, input.dt)?;
        require_finite(input.a.iter(), "a")?;
        if let Some(b) = input.b {
            require_finite(b.iter(), "b")?;
        }
        self.check_layout(n_components, input.b.is_some())?;
        if self.kernel == LagKernel::Persistence {
            require_presence(input.a.iter().copied(), "a")?;
            if let Some(b) = input.b {
                require_presence(b.iter().copied(), "b")?;
            }
        }
        if let Some(boxes) = input.boxes {
            if self.kernel != LagKernel::SquaredDisplacement {
                return Err(ComputeError::OutOfRange {
                    field: "boxes",
                    value: format!("the {} kernel takes no boxes", self.kernel),
                });
            }
            if boxes.len() != n_frames {
                return Err(ComputeError::DimensionMismatch {
                    expected: n_frames,
                    got: boxes.len(),
                    what: "boxes (one per frame)",
                });
            }
            for series in std::iter::once(input.a).chain(input.b) {
                let steps = series.outer_iter().zip(series.outer_iter().skip(1));
                for (frame, ((prev, now), simbox)) in steps.zip(&boxes[1..]).enumerate() {
                    if let Some(item) = first_wrapped_item(prev, now, simbox)? {
                        return Err(ComputeError::Wrapped {
                            frame: frame + 1,
                            item,
                        });
                    }
                }
            }
        }
        Ok((n_frames, n_items, n_components))
    }

    /// Lagged sums of a linear kernel over all origins, by FFT. Returns
    /// `(max_lag + 1) × channels`.
    fn fft_sums(&self, a: ArrayView3<'_, F>, b: Option<ArrayView3<'_, F>>) -> Array2<F> {
        let (_, n_items, n_components) = a.dim();
        let chunks: Vec<Range<usize>> = (0..n_items)
            .step_by(ITEMS_PER_CHUNK)
            .map(|start| start..(start + ITEMS_PER_CHUNK).min(n_items))
            .collect();
        let work =
            |planner: &mut FftPlanner<F>, items: Range<usize>| self.fft_chunk(planner, a, b, items);
        #[cfg(feature = "rayon")]
        let partials: Vec<Array2<F>> = {
            use rayon::prelude::*;
            chunks
                .into_par_iter()
                .map_init(FftPlanner::new, work)
                .collect()
        };
        #[cfg(not(feature = "rayon"))]
        let partials: Vec<Array2<F>> = {
            let mut planner = FftPlanner::new();
            chunks
                .into_iter()
                .map(|items| work(&mut planner, items))
                .collect()
        };

        let n_channels = self.n_channels(n_items, n_components);
        let mut channel_major = Array2::<F>::zeros((n_channels, self.max_lag + 1));
        let mut next_item = 0;
        for partial in partials {
            match self.reduce {
                Reduction::PerItem => {
                    let rows = next_item..next_item + partial.nrows();
                    channel_major
                        .slice_mut(ndarray::s![rows, ..])
                        .assign(&partial);
                    next_item += partial.nrows();
                }
                Reduction::Mean | Reduction::PerComponent => channel_major += &partial,
            }
        }
        if self.kernel == LagKernel::SquaredDisplacement {
            // A displacement over no time is exactly zero; the FFT leaves
            // round-off there.
            channel_major.column_mut(0).fill(0.0);
        }
        channel_major
            .reversed_axes()
            .as_standard_layout()
            .into_owned()
    }

    /// One chunk of items of the FFT path, channel-major: one row per channel
    /// the chunk feeds (its items for `PerItem`, else all channels).
    fn fft_chunk(
        &self,
        planner: &mut FftPlanner<F>,
        a: ArrayView3<'_, F>,
        b: Option<ArrayView3<'_, F>>,
        items: Range<usize>,
    ) -> Array2<F> {
        let (n_frames, _, n_components) = a.dim();
        let n_lags = self.max_lag + 1;
        let first_item = items.start;
        let rows = match self.reduce {
            Reduction::PerItem => items.len(),
            Reduction::Mean => 1,
            Reduction::PerComponent => n_components,
        };
        let mut out = Array2::<F>::zeros((rows, n_lags));
        let mut fft = ColumnFft::new(planner, n_frames);
        let mut col_a = vec![0.0; n_frames];
        let mut col_b = vec![0.0; n_frames];
        let mut scratch = vec![0.0; if self.stride > 1 { 3 * n_frames } else { 0 }];
        let center =
            self.mean == MeanPolicy::Subtract || self.kernel == LagKernel::SquaredDisplacement;
        for item in items {
            for component in 0..n_components {
                fill_column(a, item, component, center, &mut col_a);
                let row = match self.reduce {
                    Reduction::PerItem => item - first_item,
                    _ => self.channel_of(item, component),
                };
                let target = out
                    .row_mut(row)
                    .into_slice()
                    .expect("a fresh Array2 row is contiguous");
                let y = match b {
                    Some(b) => {
                        fill_column(b, item, component, center, &mut col_b);
                        &col_b
                    }
                    None => &col_a,
                };
                match (self.stride, b.is_some(), self.kernel) {
                    (1, false, LagKernel::Product) => fft.auto_product(&col_a, target),
                    (1, false, _) => fft.auto_displacement(&col_a, target),
                    (1, true, LagKernel::Product) => fft.cross_product(&col_a, y, target),
                    (1, true, _) => fft.cross_displacement(&col_a, y, target),
                    (s, _, LagKernel::Product) => {
                        fft.strided_product(&col_a, y, s, &mut scratch[..n_frames], target)
                    }
                    (s, _, _) => fft.strided_displacement(&col_a, y, s, &mut scratch, target),
                }
            }
        }
        out
    }

    /// Lagged sums by direct summation: a single origin, or a kernel that is
    /// nonlinear in the item's vector. Returns `(max_lag + 1) × channels`.
    fn direct_sums(
        &self,
        a: ArrayView3<'_, F>,
        b: Option<ArrayView3<'_, F>>,
    ) -> Result<Array2<F>, ComputeError> {
        let (n_frames, n_items, n_components) = a.dim();
        let n_channels = self.n_channels(n_items, n_components);
        let centered_a;
        let centered_b;
        let (a, b) = if self.mean == MeanPolicy::Subtract {
            centered_a = centered(a);
            centered_b = b.map(centered);
            (centered_a.view(), centered_b.as_ref().map(|b| b.view()))
        } else {
            (a, b)
        };
        let samples = self.kernel_samples(a)?;
        // Persistence runs of the series that must persist: `b` when crossed.
        let runs = (self.kernel == LagKernel::Persistence).then(|| presence_runs(b.unwrap_or(a)));

        let lag_row = |lag: usize| -> Vec<F> {
            let mut row = vec![0.0; n_channels];
            let origins = match self.origins {
                Origins::All => (0..n_frames - lag).step_by(self.stride),
                Origins::First => (0..1).step_by(1),
            };
            for origin in origins {
                let late = origin + lag;
                for item in 0..n_items {
                    match self.kernel {
                        LagKernel::Product | LagKernel::SquaredDisplacement => {
                            let y = b.unwrap_or(a);
                            for component in 0..n_components {
                                row[self.channel_of(item, component)] += linear_term(
                                    self.kernel,
                                    [a[[origin, item, component]], a[[late, item, component]]],
                                    [y[[origin, item, component]], y[[late, item, component]]],
                                );
                            }
                        }
                        LagKernel::Persistence => {
                            let runs = runs.as_ref().expect("runs exist for persistence");
                            if a[[origin, item, 0]] == 1.0 && runs[[late, item]] > lag {
                                row[self.channel_of(item, 0)] += 1.0;
                            }
                        }
                        LagKernel::Legendre { .. } | LagKernel::WignerD { .. } => {
                            let samples = samples.as_ref().expect("orientation samples");
                            row[self.channel_of(item, 0)] += self.kernel.item_value(
                                sample(samples, origin, item),
                                sample(samples, late, item),
                            );
                        }
                    }
                }
            }
            row
        };

        #[cfg(feature = "rayon")]
        let rows: Vec<Vec<F>> = {
            use rayon::prelude::*;
            (0..=self.max_lag).into_par_iter().map(lag_row).collect()
        };
        #[cfg(not(feature = "rayon"))]
        let rows: Vec<Vec<F>> = (0..=self.max_lag).map(lag_row).collect();

        Ok(Array2::from_shape_vec(
            (self.max_lag + 1, n_channels),
            rows.into_iter().flatten().collect(),
        )
        .expect("one row of n_channels per lag"))
    }

    /// The samples an orientation kernel reads — unit vectors for Legendre,
    /// non-zero quaternions for Wigner-D — in standard layout; `None` for the
    /// other kernels.
    /// Lagged sums of the Legendre kernel of order `l ≤ 2` over all origins,
    /// by FFT. `P_l` is a polynomial in the dot product of the unit vectors,
    /// so it factorises: `P_1(û·v̂) = Σ_d û_d v̂_d` is the product kernel of
    /// the unit vectors, and `P_2(û·v̂) = (3 Q(û)·Q(v̂) − 1)/2` with
    /// `Q = (x², y², z², √2 xy, √2 xz, √2 yz)` for 3-vectors (in general the
    /// `D(D+1)/2` independent components of `û ⊗ û`, off-diagonals weighted
    /// `√2`), whose dot product is `(û·v̂)²`. The lagged sums of `Q·Q` come
    /// from the product kernel; the constant `−½` contributes once per origin
    /// and item. Returns `(max_lag + 1) × channels`, raw.
    fn legendre_fft_sums(&self, a: ArrayView3<'_, F>, l: u32) -> Result<Array2<F>, ComputeError> {
        let samples = self
            .kernel_samples(a)?
            .expect("the Legendre kernel normalises its samples");
        let (n_frames, n_items, _) = samples.dim();
        let product = Self {
            kernel: LagKernel::Product,
            mean: MeanPolicy::Keep,
            dimension: None,
            ..*self
        };
        let mut sums = match l {
            0 => Array2::zeros((self.max_lag + 1, self.n_channels(n_items, 1))),
            1 => return Ok(product.fft_sums(samples.view(), None)),
            _ => product.fft_sums(quadratic_components(&samples).view(), None),
        };
        let items_per_channel = match self.reduce {
            Reduction::PerItem => 1.0,
            Reduction::Mean | Reduction::PerComponent => n_items as F,
        };
        for (lag, mut row) in sums.axis_iter_mut(Axis(0)).enumerate() {
            // P_0 = 1 and the -1/2 of P_2 count once per origin and item.
            let ones = items_per_channel * self.origin_count(n_frames, lag) as F;
            if l == 0 {
                row.fill(ones);
            } else {
                row.mapv_inplace(|qq| 1.5 * qq - 0.5 * ones);
            }
        }
        Ok(sums)
    }

    fn kernel_samples(&self, a: ArrayView3<'_, F>) -> Result<Option<Array3<F>>, ComputeError> {
        match self.kernel {
            LagKernel::Legendre { .. } | LagKernel::WignerD { .. } => {
                let mut samples = a.as_standard_layout().into_owned();
                for (frame, sample) in samples.outer_iter_mut().enumerate() {
                    self.prepare_sample(sample, frame)?;
                }
                Ok(Some(samples))
            }
            _ => Ok(None),
        }
    }

    /// Bring one frame `N × D` into the form the kernel reads: Legendre
    /// vectors are normalised; every orientation must have non-zero length.
    pub(super) fn prepare_sample(
        &self,
        mut sample: ArrayViewMut2<'_, F>,
        frame: usize,
    ) -> Result<(), ComputeError> {
        if !matches!(
            self.kernel,
            LagKernel::Legendre { .. } | LagKernel::WignerD { .. }
        ) {
            return Ok(());
        }
        for (item, mut row) in sample.outer_iter_mut().enumerate() {
            let norm = row.iter().map(|v| v * v).sum::<F>().sqrt();
            if !(norm.is_finite() && norm > 0.0) {
                return Err(ComputeError::OutOfRange {
                    field: "a",
                    value: format!(
                        "item {item} at frame {frame} has zero length; an orientation needs a direction"
                    ),
                });
            }
            if matches!(self.kernel, LagKernel::Legendre { .. }) {
                let inv = norm.recip();
                row.mapv_inplace(|v| v * inv);
            }
        }
        Ok(())
    }
}

impl Compute for Correlator {
    type Input<'a> = CorrelatorInput<'a>;
    type Output = TimeCorrelation;

    fn compute(&self, input: CorrelatorInput<'_>) -> Result<TimeCorrelation, ComputeError> {
        let (n_frames, n_items, _) = self.validate(&input)?;
        let sums = match self.kernel {
            _ if self.origins == Origins::First => self.direct_sums(input.a, input.b)?,
            LagKernel::Product | LagKernel::SquaredDisplacement => self.fft_sums(input.a, input.b),
            LagKernel::Legendre { l } if l <= 2 => self.legendre_fft_sums(input.a, l)?,
            _ => self.direct_sums(input.a, input.b)?,
        };
        Ok(self.normalize(sums, n_frames, n_items, input.dt))
    }
}

/// The value a linear kernel adds for one column, between `x = [x(τ), x(τ+t)]`
/// and `y = [y(τ), y(τ+t)]`.
#[inline]
pub(super) fn linear_term(kernel: LagKernel, x: [F; 2], y: [F; 2]) -> F {
    match kernel {
        LagKernel::Product => x[0] * y[1],
        _ => (x[1] - x[0]) * (y[1] - y[0]),
    }
}

/// The `D(D+1)/2` independent components of `û ⊗ û` per item, the
/// off-diagonal ones weighted `√2`, so that `Q(û) · Q(v̂) = (û · v̂)²`.
fn quadratic_components(samples: &Array3<F>) -> Array3<F> {
    let (n_frames, n_items, n_components) = samples.dim();
    let n_q = n_components * (n_components + 1) / 2;
    let mut q = Array3::<F>::zeros((n_frames, n_items, n_q));
    for frame in 0..n_frames {
        for item in 0..n_items {
            let u = sample(samples, frame, item);
            let mut k = 0;
            for i in 0..n_components {
                for j in i..n_components {
                    q[[frame, item, k]] = if i == j {
                        u[i] * u[i]
                    } else {
                        std::f64::consts::SQRT_2 * u[i] * u[j]
                    };
                    k += 1;
                }
            }
        }
    }
    q
}

/// One item's sample of a standard-layout `T × N × D` array.
#[inline]
fn sample(samples: &Array3<F>, frame: usize, item: usize) -> &[F] {
    samples
        .slice(ndarray::s![frame, item, ..])
        .to_slice()
        .expect("samples are in standard layout")
}

/// Copy the `(item, component)` column into `out`, minus its time mean when
/// `center` (mean as `Σ x · (1/T)`).
fn fill_column(
    series: ArrayView3<'_, F>,
    item: usize,
    component: usize,
    center: bool,
    out: &mut [F],
) {
    let lane = series.slice(ndarray::s![.., item, component]);
    let mean = if center {
        lane.iter().sum::<F>() * (1.0 / lane.len() as F)
    } else {
        0.0
    };
    for (slot, value) in out.iter_mut().zip(lane.iter()) {
        *slot = value - mean;
    }
}

/// The series with every `(item, component)` column's time mean removed.
fn centered(series: ArrayView3<'_, F>) -> Array3<F> {
    let mut out = series.to_owned();
    let (_, n_items, n_components) = series.dim();
    let mut column = vec![0.0; series.dim().0];
    for item in 0..n_items {
        for component in 0..n_components {
            fill_column(series, item, component, true, &mut column);
            out.slice_mut(ndarray::s![.., item, component])
                .iter_mut()
                .zip(&column)
                .for_each(|(slot, v)| *slot = *v);
        }
    }
    out
}

/// `runs[t][i]`: how many consecutive frames, ending at `t`, item `i` has
/// been present. The presence starting at `τ` survives to `τ + t` exactly
/// when `runs[τ + t][i] > t`.
fn presence_runs(presence: ArrayView3<'_, F>) -> Array2<usize> {
    let (n_frames, n_items, _) = presence.dim();
    let mut runs = Array2::<usize>::zeros((n_frames, n_items));
    for frame in 0..n_frames {
        for item in 0..n_items {
            runs[[frame, item]] = if presence[[frame, item, 0]] == 1.0 {
                if frame == 0 {
                    1
                } else {
                    runs[[frame - 1, item]] + 1
                }
            } else {
                0
            };
        }
    }
    runs
}

crate::compute::spec::compute_spec! {
    Correlator as "Correlator";
    params |s| {
        kernel: Choice(LagKernelKind::NAMES);
        max_lag: Int;
        l: Int = crate::compute::ParamDefault::UNSET => s.kernel.order();
        origins: Choice(Origins::NAMES) = Origins::default() => s.origins;
        mean: Choice(MeanPolicy::NAMES) = MeanPolicy::default() => s.mean;
        normalization: Choice(Normalization::NAMES) = Normalization::default() => s.norm;
        reduction: Choice(Reduction::NAMES) = Reduction::default() => s.reduce;
        stride: Int = Correlator::DEFAULT_STRIDE => s.stride;
        dimension: Int = crate::compute::ParamDefault::UNSET => s.dimension;
    }
    inputs {
        a: Trajectory;
        b: Trajectory, optional;
        dt: Scalar;
        boxes: Boxes, optional;
    }
}

#[cfg(test)]
mod product_tests;
#[cfg(test)]
mod tests;
