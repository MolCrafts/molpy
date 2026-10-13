use std::sync::Arc;

use ndarray::Array1;
use rustfft::num_traits::Zero;
use rustfft::{Fft, FftPlanner, num_complex::Complex64};

/// Linear (zero-padded) auto- and cross-correlation of length-`n` series by
/// the Wiener–Khinchin route, with the FFT plans made once.
///
/// This is the one FFT kernel behind every time-correlation function in
/// molrs: the free functions [`acf_fft`], [`acf_fft_accumulate`], [`xcorr_fft`]
/// and [`xcorr_fft_accumulate`] are thin wrappers over it, and the correlation
/// engine in `molrs::compute` drives it column by column.
///
/// Every method accumulates `weight · r[t]` into `out[0..=max_lag]`, where `r`
/// is the **un-normalized** linear correlation (`Σ_{τ=0}^{n-1-t} …`), so a
/// caller can fold several columns — and per-column weights — into one curve
/// with a single buffer. The padding is `(2n).next_power_of_two()`, which is
/// what makes the correlation linear rather than circular.
///
/// A `LinearCorrelator` is `Send + Sync` (the plans are `Arc`s); only the
/// complex scratch buffers are per-call, so threads can share one correlator
/// and own their scratch.
#[derive(Clone)]
pub struct LinearCorrelator {
    n: usize,
    n_pad: usize,
    fwd: Arc<dyn Fft<f64>>,
    inv: Arc<dyn Fft<f64>>,
}

impl std::fmt::Debug for LinearCorrelator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LinearCorrelator")
            .field("n", &self.n)
            .field("n_pad", &self.n_pad)
            .finish()
    }
}

impl LinearCorrelator {
    /// Plans for series of length `n` (padded to `(2n).next_power_of_two()`).
    pub fn new(planner: &mut FftPlanner<f64>, n: usize) -> Self {
        let n_pad = (2 * n.max(1)).next_power_of_two();
        Self {
            n,
            n_pad,
            fwd: planner.plan_fft_forward(n_pad),
            inv: planner.plan_fft_inverse(n_pad),
        }
    }

    /// Series length the plans were made for.
    pub fn len(&self) -> usize {
        self.n
    }

    /// `true` when planned for empty series (never a usable correlator).
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// Padded transform length.
    pub fn n_pad(&self) -> usize {
        self.n_pad
    }

    fn check(&self, len: usize, max_lag: usize, out_len: usize) -> Result<(), SignalError> {
        if len != self.n {
            return Err(SignalError::MaxLagTooLarge {
                max_lag,
                len: len.min(self.n),
            });
        }
        validate_series(self.n, max_lag)?;
        if out_len < max_lag + 1 {
            return Err(SignalError::MaxLagTooLarge {
                max_lag,
                len: out_len.saturating_sub(1).max(1),
            });
        }
        Ok(())
    }

    /// `out[t] += weight · Σ_{τ} data[τ]·data[τ+t]` for `t = 0..=max_lag`.
    ///
    /// In-place Wiener–Khinchin: pad, forward FFT, `|X|²`, inverse FFT; each
    /// lag is added as `weight · (re · 1/n_pad)`, one multiply per factor, so
    /// `weight = 1.0` is bit-for-bit the unweighted sum.
    pub fn acf_into(
        &self,
        data: &[f64],
        max_lag: usize,
        weight: f64,
        out: &mut [f64],
        scratch: &mut Vec<Complex64>,
    ) -> Result<(), SignalError> {
        self.check(data.len(), max_lag, out.len())?;
        fill_padded(scratch, data, self.n_pad);
        self.fwd.process(scratch);
        for z in scratch.iter_mut() {
            *z = Complex64::new(z.norm_sqr(), 0.0);
        }
        self.inv.process(scratch);
        let scale = 1.0 / self.n_pad as f64;
        for (k, slot) in out.iter_mut().enumerate().take(max_lag + 1) {
            *slot += weight * (scratch[k].re * scale);
        }
        Ok(())
    }

    /// `out[t] += weight · Σ_{τ} a[τ]·b[τ+t]` for `t = 0..=max_lag`
    /// (`IFFT(conj(A)·B)`, formed in place in `scratch_a`).
    ///
    /// `xcorr_into(a, a, …)` reproduces `acf_into(a, …)` bit for bit:
    /// `conj(A)·A` and `|A|²` agree to the last ULP.
    #[allow(clippy::too_many_arguments)]
    pub fn xcorr_into(
        &self,
        a: &[f64],
        b: &[f64],
        max_lag: usize,
        weight: f64,
        out: &mut [f64],
        scratch_a: &mut Vec<Complex64>,
        scratch_b: &mut Vec<Complex64>,
    ) -> Result<(), SignalError> {
        self.check(a.len(), max_lag, out.len())?;
        if b.len() != a.len() {
            return Err(SignalError::MaxLagTooLarge {
                max_lag,
                len: b.len(),
            });
        }
        fill_padded(scratch_a, a, self.n_pad);
        fill_padded(scratch_b, b, self.n_pad);
        self.fwd.process(scratch_a);
        self.fwd.process(scratch_b);
        for (xa, yb) in scratch_a.iter_mut().zip(scratch_b.iter()) {
            *xa = xa.conj() * yb;
        }
        self.inv.process(scratch_a);
        let scale = 1.0 / self.n_pad as f64;
        for (k, slot) in out.iter_mut().enumerate().take(max_lag + 1) {
            *slot += weight * (scratch_a[k].re * scale);
        }
        Ok(())
    }

    /// `out[t] += weight · Σ_{τ} (a[τ]·b[τ+t] + b[τ]·a[τ+t])`: the symmetric
    /// sum `r_ab + r_ba` from one inverse FFT of `2·Re(conj(A)·B)`.
    ///
    /// For `a == b` this is `2 · acf` exactly (the input to the inverse FFT is
    /// `2·|A|²`, and scaling by two commutes with every rounding).
    #[allow(clippy::too_many_arguments)]
    pub fn xcorr_sym_into(
        &self,
        a: &[f64],
        b: &[f64],
        max_lag: usize,
        weight: f64,
        out: &mut [f64],
        scratch_a: &mut Vec<Complex64>,
        scratch_b: &mut Vec<Complex64>,
    ) -> Result<(), SignalError> {
        self.check(a.len(), max_lag, out.len())?;
        if b.len() != a.len() {
            return Err(SignalError::MaxLagTooLarge {
                max_lag,
                len: b.len(),
            });
        }
        fill_padded(scratch_a, a, self.n_pad);
        fill_padded(scratch_b, b, self.n_pad);
        self.fwd.process(scratch_a);
        self.fwd.process(scratch_b);
        for (xa, yb) in scratch_a.iter_mut().zip(scratch_b.iter()) {
            let prod = xa.conj() * yb;
            *xa = Complex64::new(2.0 * prod.re, 0.0);
        }
        self.inv.process(scratch_a);
        let scale = 1.0 / self.n_pad as f64;
        for (k, slot) in out.iter_mut().enumerate().take(max_lag + 1) {
            *slot += weight * (scratch_a[k].re * scale);
        }
        Ok(())
    }
}

/// Linear autocorrelation via the Wiener-Khinchin theorem (FFT-based).
///
/// Returns the un-normalized linear autocorrelation `r[t] = Σ_{τ=0}^{n-1-t} x[τ]·x[τ+t]`
/// for `t = 0, 1, …, max_lag`. The caller is responsible for any further
/// normalization (e.g. division by `(n - t)` for the unbiased estimator,
/// or by `r[0]` for the normalized ACF).
///
/// Reference: Wiener (1930) "Generalized harmonic analysis";
/// Khinchin (1934). Implementation follows the standard linear-ACF recipe
/// (Press et al., *Numerical Recipes* §13.2): zero-pad to ≥ 2·n,
/// forward FFT, take the squared magnitude, inverse FFT.
///
/// Padding to `(2·n).next_power_of_two()` is required: with `n_pad = n`
/// the inverse FFT yields the *circular* autocorrelation, whose tail
/// wraps around and contaminates the lag-t entries with `r_linear[n − t]`.
pub fn acf_fft(data: &Array1<f64>, max_lag: usize) -> Result<Array1<f64>, SignalError> {
    let mut planner = FftPlanner::new();
    acf_fft_with_planner(&mut planner, data, max_lag)
}

/// Same as [`acf_fft`] but reuses the caller-provided `FftPlanner`.
///
/// `rustfft` caches plans inside the planner, so passing one planner across
/// many calls (e.g. component-wise ACFs in dielectric spectra) avoids paying
/// the plan-construction cost on every invocation.
pub fn acf_fft_with_planner(
    planner: &mut FftPlanner<f64>,
    data: &Array1<f64>,
    max_lag: usize,
) -> Result<Array1<f64>, SignalError> {
    let mut out = Array1::<f64>::zeros(max_lag + 1);
    let mut scratch = Vec::new();
    let owned;
    let slice = match data.as_slice() {
        Some(s) => s,
        None => {
            owned = data.to_vec();
            owned.as_slice()
        }
    };
    acf_fft_accumulate(
        planner,
        slice,
        max_lag,
        out.as_slice_mut().expect("zeros array is contiguous"),
        &mut scratch,
    )?;
    Ok(out)
}

/// Accumulate one series' un-normalized linear ACF into `out[0..=max_lag]`.
///
/// **Primitive for multi-component correlators**: call once per Cartesian
/// component, reusing `scratch` across calls so the complex FFT workspace is
/// allocated once. [`LinearCorrelator::acf_into`] with weight 1 and a plan
/// looked up in `planner` (rustfft caches plans, so repeated calls on one
/// length re-plan nothing).
pub fn acf_fft_accumulate(
    planner: &mut FftPlanner<f64>,
    data: &[f64],
    max_lag: usize,
    out: &mut [f64],
    scratch: &mut Vec<Complex64>,
) -> Result<(), SignalError> {
    validate_series(data.len(), max_lag)?;
    LinearCorrelator::new(planner, data.len()).acf_into(data, max_lag, 1.0, out, scratch)
}

/// Linear cross-correlation via the Wiener–Khinchin cross spectrum (FFT-based).
///
/// Returns the un-normalized linear cross-correlation
/// `r_ab[t] = Σ_{τ=0}^{n-1-t} a[τ]·b[τ+t]` for `t = 0, 1, …, max_lag`, using the
/// same zero-pad to `(2·n).next_power_of_two()`, `IFFT(conj(A)·B)` recipe, and
/// `1/n_pad` scaling as [`acf_fft_with_planner`]. Consequently
/// `xcorr_fft_with_planner(p, a, a, k)` reproduces `acf_fft_with_planner(p, a, k)`
/// **bit-for-bit** (`conj(A)·A` and `|A|²` agree to the last ULP).
///
/// Both inputs must share the length `n` of `a` (the caller precondition — like
/// [`acf_fft`] this does not itself validate `b.len()`). Errors when `a` is empty
/// or `max_lag ≥ n`.
/// Convenience wrapper around [`xcorr_fft_with_planner`] that owns its planner.
pub fn xcorr_fft(
    a: &Array1<f64>,
    b: &Array1<f64>,
    max_lag: usize,
) -> Result<Array1<f64>, SignalError> {
    let mut planner = FftPlanner::new();
    xcorr_fft_with_planner(&mut planner, a, b, max_lag)
}

pub fn xcorr_fft_with_planner(
    planner: &mut FftPlanner<f64>,
    a: &Array1<f64>,
    b: &Array1<f64>,
    max_lag: usize,
) -> Result<Array1<f64>, SignalError> {
    let mut out = Array1::<f64>::zeros(max_lag + 1);
    let mut sa = Vec::new();
    let mut sb = Vec::new();
    let owned_a;
    let owned_b;
    let aa = match a.as_slice() {
        Some(s) => s,
        None => {
            owned_a = a.to_vec();
            owned_a.as_slice()
        }
    };
    let bb = match b.as_slice() {
        Some(s) => s,
        None => {
            owned_b = b.to_vec();
            owned_b.as_slice()
        }
    };
    xcorr_fft_accumulate(
        planner,
        aa,
        bb,
        max_lag,
        out.as_slice_mut().expect("zeros array is contiguous"),
        &mut sa,
        &mut sb,
    )?;
    Ok(out)
}

/// Accumulate one un-normalized linear cross-correlation into `out[0..=max_lag]`.
///
/// Reuses `scratch_a` / `scratch_b` across multi-component calls.
/// [`LinearCorrelator::xcorr_into`] with weight 1.
pub fn xcorr_fft_accumulate(
    planner: &mut FftPlanner<f64>,
    a: &[f64],
    b: &[f64],
    max_lag: usize,
    out: &mut [f64],
    scratch_a: &mut Vec<Complex64>,
    scratch_b: &mut Vec<Complex64>,
) -> Result<(), SignalError> {
    validate_series(a.len(), max_lag)?;
    LinearCorrelator::new(planner, a.len())
        .xcorr_into(a, b, max_lag, 1.0, out, scratch_a, scratch_b)
}

#[inline]
fn validate_series(n: usize, max_lag: usize) -> Result<(), SignalError> {
    if n == 0 {
        return Err(SignalError::EmptyInput);
    }
    if max_lag >= n {
        return Err(SignalError::MaxLagTooLarge { max_lag, len: n });
    }
    Ok(())
}

#[inline]
fn fill_padded(buf: &mut Vec<Complex64>, data: &[f64], n_pad: usize) {
    buf.resize(n_pad, Complex64::zero());
    let n = data.len();
    for i in 0..n {
        buf[i] = Complex64::new(data[i], 0.0);
    }
    for z in buf[n..].iter_mut() {
        *z = Complex64::zero();
    }
}

/// Failure modes for `molrs::signal` primitives.
#[derive(Debug, PartialEq)]
pub enum SignalError {
    /// Input array has length 0.
    EmptyInput,
    /// `max_lag` does not satisfy `max_lag < data.len()`.
    MaxLagTooLarge { max_lag: usize, len: usize },
    /// Window/axis-aware operation was asked to operate along an axis
    /// that does not exist on the input.
    AxisOutOfBounds { axis: usize, ndim: usize },
    /// A window name [`WindowType::from_name`](super::WindowType::from_name)
    /// does not know.
    UnknownWindow(String),
}

impl std::fmt::Display for SignalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SignalError::EmptyInput => write!(f, "input array is empty"),
            SignalError::MaxLagTooLarge { max_lag, len } => {
                write!(
                    f,
                    "max_lag ({max_lag}) must be less than data length ({len})"
                )
            }
            SignalError::AxisOutOfBounds { axis, ndim } => {
                write!(f, "axis ({axis}) out of bounds for ndim ({ndim})")
            }
            SignalError::UnknownWindow(name) => write!(
                f,
                "unknown window {name:?}; expected hann, blackman, cosine_sq or none"
            ),
        }
    }
}

impl std::error::Error for SignalError {}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    #[test]
    fn test_acf_constant_signal_un_normalized() {
        let data = arr1(&[2.0, 2.0, 2.0, 2.0]);
        let result = acf_fft(&data, 3).unwrap();
        assert_eq!(result.len(), 4);
        let expected_lag0 = 4.0 * 2.0_f64.powi(2); // N * c^2 = 4 * 4 = 16
        assert!((result[0] - expected_lag0).abs() < 1e-10);
    }

    #[test]
    fn test_acf_constant_signal_linear_decay() {
        // Linear ACF of constant c over n samples: r[t] = (n - t) · c²
        let data = arr1(&[3.0, 3.0, 3.0]);
        let result = acf_fft(&data, 2).unwrap();
        assert_eq!(result.len(), 3);
        let c_sq = 3.0_f64.powi(2);
        let n = 3.0;
        for (t, v) in result.iter().enumerate() {
            let expected = (n - t as f64) * c_sq;
            assert!(
                (v - expected).abs() < 1e-10,
                "lag {t}: got {v}, expected {expected}",
            );
        }
    }

    #[test]
    fn test_acf_single_element() {
        let data = arr1(&[5.0]);
        let result = acf_fft(&data, 0).unwrap();
        assert_eq!(result.len(), 1);
        assert!((result[0] - 25.0).abs() < 1e-10); // 1 * 5^2
    }

    #[test]
    fn test_acf_max_lag_zero() {
        let data = arr1(&[1.0, 2.0, 3.0]);
        let result = acf_fft(&data, 0).unwrap();
        assert_eq!(result.len(), 1);
        let expected = 1.0_f64.powi(2) + 2.0_f64.powi(2) + 3.0_f64.powi(2); // sum of squares
        assert!((result[0] - expected).abs() < 1e-10);
    }

    #[test]
    fn test_acf_max_lag_too_large() {
        let data = arr1(&[1.0, 2.0]);
        let err = acf_fft(&data, 2).unwrap_err();
        assert_eq!(err, SignalError::MaxLagTooLarge { max_lag: 2, len: 2 });
    }

    #[test]
    fn test_acf_empty_input() {
        let data = Array1::<f64>::zeros(0);
        let err = acf_fft(&data, 0).unwrap_err();
        assert_eq!(err, SignalError::EmptyInput);
    }

    #[test]
    fn test_acf_white_noise_peak_at_zero() {
        use rand::RngExt;
        let mut rng = rand::rng();
        let data: Vec<f64> = (0..1000).map(|_| rng.random()).collect();
        let arr = Array1::from_vec(data);
        let result = acf_fft(&arr, 10).unwrap();
        assert_eq!(result.len(), 11);
        for k in 1..result.len() {
            assert!(result[k].abs() < result[0]);
        }
    }

    #[test]
    fn test_acf_sine_wave_oscillatory() {
        let n = 128;
        let data: Vec<f64> = (0..n)
            .map(|i| (2.0 * std::f64::consts::PI * i as f64 / 16.0).sin())
            .collect();
        let arr = Array1::from_vec(data);
        let result = acf_fft(&arr, 32).unwrap();
        assert_eq!(result.len(), 33);
        assert!(result[0] > 0.0);
        // ACF at half-period (lag 8) should be negative (anti-phase)
        assert!(result[8] < 0.0);
    }

    #[test]
    fn xcorr_of_self_equals_acf_bit_for_bit() {
        // The contract: xcorr(a, a) reproduces acf_fft(a) exactly (conj(A)·A == |A|²).
        let data = arr1(&[1.0, -2.0, 3.0, 0.5, -1.5, 4.0, 2.0, -3.0]);
        let mut planner = FftPlanner::new();
        let acf = acf_fft_with_planner(&mut planner, &data, 5).unwrap();
        let xc = xcorr_fft_with_planner(&mut planner, &data, &data, 5).unwrap();
        assert_eq!(acf, xc, "xcorr(a, a) must equal acf_fft(a) exactly");
    }

    #[test]
    fn xcorr_matches_direct_cross_sum() {
        // r_ab[t] = Σ_{τ} a[τ]·b[τ+t] — check against the direct definition.
        let a = arr1(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let b = arr1(&[2.0, 0.0, -1.0, 3.0, 1.0]);
        let n = a.len();
        let max_lag = 3;
        let mut planner = FftPlanner::new();
        let xc = xcorr_fft_with_planner(&mut planner, &a, &b, max_lag).unwrap();
        for t in 0..=max_lag {
            let expected: f64 = (0..n - t).map(|tau| a[tau] * b[tau + t]).sum();
            assert!(
                (xc[t] - expected).abs() < 1e-12,
                "lag {t}: {} vs {expected}",
                xc[t]
            );
        }
    }

    #[test]
    fn xcorr_rejects_bad_max_lag_and_empty() {
        let mut planner = FftPlanner::new();
        let a = arr1(&[1.0, 2.0]);
        assert_eq!(
            xcorr_fft_with_planner(&mut planner, &a, &a, 2).unwrap_err(),
            SignalError::MaxLagTooLarge { max_lag: 2, len: 2 }
        );
        let empty = Array1::<f64>::zeros(0);
        assert_eq!(
            xcorr_fft_with_planner(&mut planner, &empty, &empty, 0).unwrap_err(),
            SignalError::EmptyInput
        );
    }

    #[test]
    fn accumulate_sums_two_components() {
        let a = arr1(&[1.0, 2.0, 3.0, 4.0]);
        let b = arr1(&[0.5, -1.0, 1.5, 0.0]);
        let max_lag = 2;
        let mut planner = FftPlanner::new();
        let mut out = vec![0.0; max_lag + 1];
        let mut scratch = Vec::new();
        acf_fft_accumulate(
            &mut planner,
            a.as_slice().unwrap(),
            max_lag,
            &mut out,
            &mut scratch,
        )
        .unwrap();
        acf_fft_accumulate(
            &mut planner,
            b.as_slice().unwrap(),
            max_lag,
            &mut out,
            &mut scratch,
        )
        .unwrap();
        let sa = acf_fft_with_planner(&mut planner, &a, max_lag).unwrap();
        let sb = acf_fft_with_planner(&mut planner, &b, max_lag).unwrap();
        for k in 0..=max_lag {
            assert!((out[k] - (sa[k] + sb[k])).abs() < 1e-14, "k={k}");
        }
    }

    #[test]
    fn correlator_wrappers_are_bit_identical_to_direct_kernel_use() {
        let a = arr1(&[1.0, -2.0, 3.0, 0.5, -1.5, 4.0, 2.0, -3.0]);
        let b = arr1(&[0.5, 1.0, -1.0, 2.0, 0.25, -0.75, 1.5, 0.0]);
        let max_lag = 5;
        let mut planner = FftPlanner::new();
        let corr = LinearCorrelator::new(&mut planner, a.len());
        let (mut sa, mut sb) = (Vec::new(), Vec::new());
        let mut out = vec![0.0; max_lag + 1];
        corr.acf_into(a.as_slice().unwrap(), max_lag, 1.0, &mut out, &mut sa)
            .unwrap();
        assert_eq!(
            Array1::from_vec(out.clone()),
            acf_fft_with_planner(&mut planner, &a, max_lag).unwrap()
        );
        let mut xc = vec![0.0; max_lag + 1];
        corr.xcorr_into(
            a.as_slice().unwrap(),
            b.as_slice().unwrap(),
            max_lag,
            1.0,
            &mut xc,
            &mut sa,
            &mut sb,
        )
        .unwrap();
        assert_eq!(
            Array1::from_vec(xc),
            xcorr_fft_with_planner(&mut planner, &a, &b, max_lag).unwrap()
        );
    }

    #[test]
    fn weighted_accumulate_is_weight_times_the_unweighted_curve() {
        let a = arr1(&[1.0, -2.0, 3.0, 0.5, -1.5, 4.0]);
        let max_lag = 3;
        let mut planner = FftPlanner::new();
        let corr = LinearCorrelator::new(&mut planner, a.len());
        let mut s = Vec::new();
        let mut unweighted = vec![0.0; max_lag + 1];
        corr.acf_into(a.as_slice().unwrap(), max_lag, 1.0, &mut unweighted, &mut s)
            .unwrap();
        let mut weighted = vec![0.0; max_lag + 1];
        corr.acf_into(a.as_slice().unwrap(), max_lag, 3.0, &mut weighted, &mut s)
            .unwrap();
        for k in 0..=max_lag {
            assert_eq!(weighted[k], 3.0 * unweighted[k], "k={k}");
        }
    }

    #[test]
    fn symmetric_xcorr_is_the_sum_of_both_orders_and_twice_the_acf_for_self() {
        let a = arr1(&[1.0, -2.0, 3.0, 0.5, -1.5, 4.0, 2.0]);
        let b = arr1(&[0.5, 1.0, -1.0, 2.0, 0.25, -0.75, 1.5]);
        let max_lag = 4;
        let mut planner = FftPlanner::new();
        let corr = LinearCorrelator::new(&mut planner, a.len());
        let (mut sa, mut sb) = (Vec::new(), Vec::new());
        let mut sym = vec![0.0; max_lag + 1];
        corr.xcorr_sym_into(
            a.as_slice().unwrap(),
            b.as_slice().unwrap(),
            max_lag,
            1.0,
            &mut sym,
            &mut sa,
            &mut sb,
        )
        .unwrap();
        let ab = xcorr_fft_with_planner(&mut planner, &a, &b, max_lag).unwrap();
        let ba = xcorr_fft_with_planner(&mut planner, &b, &a, max_lag).unwrap();
        for k in 0..=max_lag {
            assert!((sym[k] - (ab[k] + ba[k])).abs() < 1e-12, "k={k}");
        }
        // Self: exactly twice the ACF.
        let mut self_sym = vec![0.0; max_lag + 1];
        corr.xcorr_sym_into(
            a.as_slice().unwrap(),
            a.as_slice().unwrap(),
            max_lag,
            1.0,
            &mut self_sym,
            &mut sa,
            &mut sb,
        )
        .unwrap();
        let acf = acf_fft_with_planner(&mut planner, &a, max_lag).unwrap();
        for k in 0..=max_lag {
            assert_eq!(self_sym[k], 2.0 * acf[k], "k={k}");
        }
    }

    #[test]
    fn correlator_rejects_wrong_length_and_long_lag() {
        let mut planner = FftPlanner::new();
        let corr = LinearCorrelator::new(&mut planner, 4);
        let mut out = vec![0.0; 4];
        let mut s = Vec::new();
        assert!(
            corr.acf_into(&[1.0, 2.0, 3.0], 2, 1.0, &mut out, &mut s)
                .is_err()
        );
        assert!(
            corr.acf_into(&[1.0, 2.0, 3.0, 4.0], 4, 1.0, &mut out, &mut s)
                .is_err()
        );
        assert!(
            corr.acf_into(&[1.0, 2.0, 3.0, 4.0], 3, 1.0, &mut out, &mut s)
                .is_ok()
        );
    }
}
