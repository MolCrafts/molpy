//! [`ColumnFft`] — the zero-padded Wiener–Khinchin kernel of the
//! [`Correlator`](super::Correlator)'s FFT path, one column at a time.

use std::sync::Arc;

use molrs::op::F;
use rustfft::num_complex::Complex64;
use rustfft::num_traits::Zero;
use rustfft::{Fft, FftPlanner};

/// Zero-padded FFT of one column at a time, with buffers reused across
/// columns. Padding to `(2T).next_power_of_two()` makes every correlation
/// linear; each method *adds* lags `0 … out.len() − 1` into `out`.
pub(super) struct ColumnFft {
    forward: Arc<dyn Fft<F>>,
    inverse: Arc<dyn Fft<F>>,
    scale: F,
    buf_a: Vec<Complex64>,
    buf_b: Vec<Complex64>,
}

impl ColumnFft {
    pub(super) fn new(planner: &mut FftPlanner<F>, n_frames: usize) -> Self {
        let n_pad = (2 * n_frames).next_power_of_two();
        Self {
            forward: planner.plan_fft_forward(n_pad),
            inverse: planner.plan_fft_inverse(n_pad),
            scale: 1.0 / n_pad as F,
            buf_a: vec![Complex64::zero(); n_pad],
            buf_b: vec![Complex64::zero(); n_pad],
        }
    }

    fn load(buf: &mut [Complex64], column: &[F]) {
        for (slot, &v) in buf.iter_mut().zip(column) {
            *slot = Complex64::new(v, 0.0);
        }
        for slot in buf[column.len()..].iter_mut() {
            *slot = Complex64::zero();
        }
    }

    /// `buf_a ← IFFT(|FFT x|²)`: the unscaled linear autocorrelation.
    fn autocorrelate(&mut self, x: &[F]) {
        Self::load(&mut self.buf_a, x);
        self.forward.process(&mut self.buf_a);
        for z in self.buf_a.iter_mut() {
            *z = Complex64::new(z.norm_sqr(), 0.0);
        }
        self.inverse.process(&mut self.buf_a);
    }

    /// `buf_a ← IFFT(conj(FFT x) · FFT y)`: index `m` holds the unscaled
    /// `Σ_τ x(τ) y(τ + m)`, index `n_pad − m` the reverse `Σ_τ y(τ) x(τ + m)`.
    fn cross_correlate(&mut self, x: &[F], y: &[F]) {
        Self::load(&mut self.buf_a, x);
        Self::load(&mut self.buf_b, y);
        self.forward.process(&mut self.buf_a);
        self.forward.process(&mut self.buf_b);
        for (za, zb) in self.buf_a.iter_mut().zip(self.buf_b.iter()) {
            *za = za.conj() * zb;
        }
        self.inverse.process(&mut self.buf_a);
    }

    /// `Σ_τ x(τ) x(τ + t)`.
    pub(super) fn auto_product(&mut self, x: &[F], out: &mut [F]) {
        self.autocorrelate(x);
        for (slot, z) in out.iter_mut().zip(&self.buf_a) {
            *slot += z.re * self.scale;
        }
    }

    /// `Σ_τ x(τ) y(τ + t)`.
    pub(super) fn cross_product(&mut self, x: &[F], y: &[F], out: &mut [F]) {
        self.cross_correlate(x, y);
        for (slot, z) in out.iter_mut().zip(&self.buf_a) {
            *slot += z.re * self.scale;
        }
    }

    /// `Σ_τ (x(τ + t) − x(τ))²` as `S(t) − 2 Σ_τ x(τ) x(τ + t)`, with
    /// `S(t) = Σ_{τ < T − t} x(τ)² + x(τ + t)²` peeled one lag at a time.
    pub(super) fn auto_displacement(&mut self, x: &[F], out: &mut [F]) {
        self.autocorrelate(x);
        let n = x.len();
        let mut s: F = 2.0 * x.iter().map(|v| v * v).sum::<F>();
        for (lag, slot) in out.iter_mut().enumerate() {
            *slot += s - 2.0 * self.buf_a[lag].re * self.scale;
            s -= x[lag] * x[lag] + x[n - 1 - lag] * x[n - 1 - lag];
        }
    }

    /// `Σ_τ (x(τ + t) − x(τ)) · (y(τ + t) − y(τ))` as
    /// `S(t) − Σ_τ x(τ) y(τ + t) − Σ_τ y(τ) x(τ + t)`, with `S` over
    /// `x · y`.
    pub(super) fn cross_displacement(&mut self, x: &[F], y: &[F], out: &mut [F]) {
        self.cross_correlate(x, y);
        let n = x.len();
        let n_pad = self.buf_a.len();
        let xy = |t: usize| x[t] * y[t];
        let mut s: F = 2.0 * (0..n).map(xy).sum::<F>();
        for (lag, slot) in out.iter_mut().enumerate() {
            let forward = self.buf_a[lag].re * self.scale;
            let backward = self.buf_a[(n_pad - lag) % n_pad].re * self.scale;
            *slot += s - forward - backward;
            s -= xy(lag) + xy(n - 1 - lag);
        }
    }

    /// `Σ_{τ ∈ O} x(τ) y(τ + t)` over the strided origins
    /// `O = {0, s, 2s, …}`: the cross-correlation of the origin-masked
    /// `m_s · x` with `y`. `masked` is scratch of the column's length.
    pub(super) fn strided_product(
        &mut self,
        x: &[F],
        y: &[F],
        stride: usize,
        masked: &mut [F],
        out: &mut [F],
    ) {
        mask_origins(x, stride, masked);
        self.cross_product(masked, y, out);
    }

    /// `Σ_{τ ∈ O} (x(τ + t) − x(τ)) · (y(τ + t) − y(τ))` over the strided
    /// origins `O = {0, s, 2s, …}`, expanded as
    ///
    /// ```text
    /// Σ_O p(τ + t) + Σ_{O, τ ≤ T−1−t} p(τ) − Σ_O x(τ) y(τ + t) − Σ_O y(τ) x(τ + t)
    /// ```
    ///
    /// with `p = x · y`: the first term is the cross-correlation of the
    /// origin mask with `p`, the second a prefix sum of the masked `p`, the
    /// last two masked cross-correlations — every lag in `O(T log T)`.
    /// `scratch` holds three columns' worth of space.
    pub(super) fn strided_displacement(
        &mut self,
        x: &[F],
        y: &[F],
        stride: usize,
        scratch: &mut [F],
        out: &mut [F],
    ) {
        let n = x.len();
        let (mask, rest) = scratch.split_at_mut(n);
        let (p, masked) = rest.split_at_mut(n);
        for (t, slot) in mask.iter_mut().enumerate() {
            *slot = if t.is_multiple_of(stride) { 1.0 } else { 0.0 };
        }
        for ((slot, a), b) in p.iter_mut().zip(x).zip(y) {
            *slot = a * b;
        }
        // Σ_O p(τ + t).
        self.cross_product(mask, p, out);
        // Σ_{O, τ ≤ T−1−t} p(τ): prefix sums of the masked p, read backwards.
        let mut prefix = vec![0.0; n + 1];
        for t in 0..n {
            prefix[t + 1] = prefix[t] + mask[t] * p[t];
        }
        for (lag, slot) in out.iter_mut().enumerate() {
            *slot += prefix[n - lag];
        }
        // − Σ_O x(τ) y(τ + t) − Σ_O y(τ) x(τ + t).
        let mut cross = vec![0.0; out.len()];
        mask_origins(x, stride, masked);
        self.cross_product(masked, y, &mut cross);
        mask_origins(y, stride, masked);
        self.cross_product(masked, x, &mut cross);
        for (slot, c) in out.iter_mut().zip(&cross) {
            *slot -= c;
        }
    }
}

/// `out[τ] = column[τ]` at the origins `τ ≡ 0 (mod stride)`, `0` elsewhere.
fn mask_origins(column: &[F], stride: usize, out: &mut [F]) {
    for (t, (slot, v)) in out.iter_mut().zip(column).enumerate() {
        *slot = if t.is_multiple_of(stride) { *v } else { 0.0 };
    }
}
