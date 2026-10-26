//! Window functions, the one-sided spectrum of a sampled signal, and its
//! frequency grid.

use std::f64::consts::PI;

use molrs::op::F;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use rustfft::FftPlanner;
use rustfft::num_complex::Complex64;
use rustfft::num_traits::Zero;

use crate::compute::named_enum::named_enum;
use crate::compute::{Compute, ComputeError};

named_enum! {
    /// A taper applied to a finite signal before its Fourier transform, to
    /// suppress spectral leakage.
    ///
    /// Hann and Blackman use the symmetric (`N − 1` denominator) form of
    /// `scipy.signal.windows.{hann, blackman}(M, sym=True)`.
    ///
    /// The default is [`CosineSq`](Self::CosineSq): a one-sided correlation
    /// keeps its `t = 0` value (the static signal, which Hann and Blackman
    /// zero) and still tapers to zero at the last lag.
    #[derive(Default)]
    pub enum Window {
        /// `w[n] = ½ (1 − cos(2πn / (N − 1)))` — zero at both ends.
        Hann => "hann",
        /// `w[n] = 0.42 − 0.5 cos(2πn / (N − 1)) + 0.08 cos(4πn / (N − 1))`
        /// (the textbook three-term coefficients).
        Blackman => "blackman",
        /// `w[n] = cos²(πn / (2 (N − 1)))` — one at `n = 0`, zero at the last
        /// sample: tapers only the tail of a one-sided correlation function.
        #[default]
        CosineSq => "cosine_sq",
        /// No taper.
        None => "none",
    }
}

impl Window {
    /// The `n` window coefficients. A one-sample window is `[1]`.
    pub fn coefficients(self, n: usize) -> Array1<F> {
        if n == 1 {
            return Array1::ones(1);
        }
        let last = n.saturating_sub(1) as F;
        Array1::from_iter((0..n).map(|i| {
            let i = i as F;
            match self {
                Self::Hann => 0.5 * (1.0 - (2.0 * PI * i / last).cos()),
                Self::Blackman => {
                    0.42 - 0.5 * (2.0 * PI * i / last).cos() + 0.08 * (4.0 * PI * i / last).cos()
                }
                Self::CosineSq => (PI * i / (2.0 * last)).cos().powi(2),
                Self::None => 1.0,
            }
        }))
    }

    /// The tapered signal `x[n] · w[n]`; the input is not modified.
    pub fn apply(self, signal: ArrayView1<'_, F>) -> Array1<F> {
        if self == Self::None {
            return signal.to_owned();
        }
        &signal * &self.coefficients(signal.len())
    }
}

/// The one-sided angular-frequency grid of an `n_fft`-point transform of a
/// signal sampled every `dt`: `ω_k = k · 2π / (n_fft · dt)`,
/// `k = 0 … n_fft / 2` — DC first, Nyquist `π / dt` last. The unit is
/// radians per unit of `dt`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FrequencyGrid {
    n_fft: usize,
    dt: F,
}

impl FrequencyGrid {
    /// The grid of an `n_fft`-point transform at sample interval `dt`.
    ///
    /// # Errors
    /// [`ComputeError::OutOfRange`] for `n_fft == 0` or a non-positive or
    /// non-finite `dt`.
    pub fn new(n_fft: usize, dt: F) -> Result<Self, ComputeError> {
        if n_fft == 0 {
            return Err(ComputeError::OutOfRange {
                field: "n_fft",
                value: n_fft.to_string(),
            });
        }
        if !(dt.is_finite() && dt > 0.0) {
            return Err(ComputeError::OutOfRange {
                field: "dt",
                value: dt.to_string(),
            });
        }
        Ok(Self { n_fft, dt })
    }

    /// Number of frequencies, `n_fft / 2 + 1`.
    pub fn n_points(&self) -> usize {
        self.n_fft / 2 + 1
    }

    /// The spacing `Δω = 2π / (n_fft · dt)`.
    pub fn spacing(&self) -> F {
        2.0 * PI / (self.n_fft as F * self.dt)
    }

    /// The angular frequencies `ω_k`.
    pub fn angular(&self) -> Array1<F> {
        let spacing = self.spacing();
        Array1::from_iter((0..self.n_points()).map(|k| k as F * spacing))
    }
}

/// The one-sided discrete spectrum of a real signal: taper with a [`Window`],
/// zero-pad to `(pad · n).next_power_of_two()` points, forward FFT, keep the
/// non-negative frequencies.
///
/// The bins are **unscaled**, `X_k = Σ_n x̃[n] e^{−2πi kn / n_fft}` for the
/// tapered signal `x̃`; a caller turns them into a physical spectrum with its
/// own prefactor (`1 / n_fft` for a power spectrum, `dt` for a continuous
/// transform, …).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Spectrum {
    window: Window,
    pad: usize,
}

/// The data a [`Spectrum`] transforms.
#[derive(Debug, Clone, Copy)]
pub struct SpectrumInput<'a> {
    /// The sampled signal, e.g. one channel of a
    /// [`TimeCorrelation`](super::TimeCorrelation).
    pub signal: ArrayView1<'a, F>,
    /// Sample interval; sets the frequency grid.
    pub dt: F,
}

/// What a [`Spectrum`] returns.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct OneSidedSpectrum {
    /// Angular frequencies, radians per unit of `dt` (the [`FrequencyGrid`]).
    pub frequencies: Array1<F>,
    /// Real parts of the unscaled bins.
    pub real: Array1<F>,
    /// Imaginary parts of the unscaled bins.
    pub imag: Array1<F>,
    /// Transform length after zero padding.
    pub n_fft: usize,
}

impl Spectrum {
    /// The zero-padding factor of the transforms built on a `Spectrum` when
    /// none is given ([`RamanSpectrum`](crate::compute::RamanSpectrum), the
    /// four ε(ω) routes): 4× the correlation length.
    pub const DEFAULT_PAD: usize = 4;

    /// A spectrum tapered by `window` and zero-padded to `pad` times the
    /// signal length (rounded up to a power of two).
    ///
    /// # Errors
    /// [`ComputeError::OutOfRange`] for `pad == 0`.
    pub fn new(window: Window, pad: usize) -> Result<Self, ComputeError> {
        if pad == 0 {
            return Err(ComputeError::OutOfRange {
                field: "pad",
                value: pad.to_string(),
            });
        }
        Ok(Self { window, pad })
    }

    /// The same transform with taper `window`; the padding is already valid.
    pub(crate) fn with_window(self, window: Window) -> Self {
        Self { window, ..self }
    }

    /// The taper.
    pub fn window(&self) -> Window {
        self.window
    }

    /// The zero-padding factor.
    pub fn pad(&self) -> usize {
        self.pad
    }

    /// The first `n_fft / 2 + 1` unscaled forward-FFT bins of `signal`
    /// zero-padded to `n_fft` points (`n_fft ≥ signal.len()`). The one FFT
    /// of a real signal in `compute`; the dielectric transforms that pad a
    /// derived signal to a length of their own call it directly.
    pub(crate) fn onesided(signal: &[F], n_fft: usize) -> Vec<Complex64> {
        let mut bins: Vec<Complex64> = signal.iter().map(|&x| Complex64::new(x, 0.0)).collect();
        bins.resize(n_fft, Complex64::zero());
        FftPlanner::new().plan_fft_forward(n_fft).process(&mut bins);
        bins.truncate(n_fft / 2 + 1);
        bins
    }

    /// The power `|F(k)|²` of the 2-D discrete Fourier transform of a real
    /// `n × n` image (rows then columns, unscaled), FFT-shifted so `k = 0`
    /// sits at `(n/2, n/2)` (the `numpy.fft.fftshift` convention).
    ///
    /// The second form of the one FFT owner in `compute`: a 2-D power
    /// spectrum is not a windowed one-sided 1-D spectrum, but its transforms
    /// are planned here so no other file plans an FFT.
    pub(crate) fn power_2d_centered(image: ArrayView2<'_, F>) -> Array2<F> {
        let (rows, cols) = image.dim();
        let mut buf: Vec<Complex64> = image.iter().map(|&x| Complex64::new(x, 0.0)).collect();
        let mut planner = FftPlanner::<F>::new();
        // Row pass: FFT every row of length `cols`.
        let row_fft = planner.plan_fft_forward(cols);
        for row in buf.chunks_exact_mut(cols) {
            row_fft.process(row);
        }
        // Column pass: gather column → FFT → scatter back.
        let col_fft = planner.plan_fft_forward(rows);
        let mut column = vec![Complex64::zero(); rows];
        for j in 0..cols {
            for (i, c) in column.iter_mut().enumerate() {
                *c = buf[i * cols + j];
            }
            col_fft.process(&mut column);
            for (i, c) in column.iter().enumerate() {
                buf[i * cols + j] = *c;
            }
        }
        // Magnitude squared + FFT-shift (swap halves along both axes).
        let mut shifted = Array2::<F>::zeros((rows, cols));
        for i in 0..rows {
            for j in 0..cols {
                shifted[[(i + rows / 2) % rows, (j + cols / 2) % cols]] =
                    buf[i * cols + j].norm_sqr();
            }
        }
        shifted
    }
}

impl Compute for Spectrum {
    type Input<'a> = SpectrumInput<'a>;
    type Output = OneSidedSpectrum;

    fn compute(&self, input: SpectrumInput<'_>) -> Result<OneSidedSpectrum, ComputeError> {
        let n = input.signal.len();
        if n == 0 {
            return Err(ComputeError::EmptyInput);
        }
        if let Some(index) = input.signal.iter().position(|v| !v.is_finite()) {
            return Err(ComputeError::NonFinite {
                where_: "signal",
                index,
            });
        }
        let n_fft = (self.pad * n).next_power_of_two();
        let grid = FrequencyGrid::new(n_fft, input.dt)?;
        let tapered = self.window.apply(input.signal);
        let bins = Self::onesided(
            tapered.as_slice().expect("a fresh Array1 is contiguous"),
            n_fft,
        );
        Ok(OneSidedSpectrum {
            frequencies: grid.angular(),
            real: bins.iter().map(|b| b.re).collect(),
            imag: bins.iter().map(|b| b.im).collect(),
            n_fft,
        })
    }
}

crate::compute::spec::compute_spec! {
    Spectrum as "Spectrum";
    params |s| {
        window: Choice(Window::NAMES);
        pad: Int;
    }
    inputs {
        signal: Scalars;
        dt: Scalar;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    fn hann_analytical(n: usize, idx: usize) -> F {
        if n == 1 {
            return 1.0;
        }
        0.5 * (1.0 - (2.0 * PI * idx as F / (n - 1) as F).cos())
    }

    fn blackman_analytical(n: usize, idx: usize) -> F {
        if n == 1 {
            return 1.0;
        }
        0.42 - 0.5 * (2.0 * PI * idx as F / (n - 1) as F).cos()
            + 0.08 * (4.0 * PI * idx as F / (n - 1) as F).cos()
    }

    fn cosine_sq_analytical(n: usize, idx: usize) -> F {
        if n == 1 {
            return 1.0;
        }
        (PI * idx as F / (2.0 * (n - 1) as F)).cos().powi(2)
    }

    fn ones(n: usize) -> Array1<F> {
        Array1::ones(n)
    }

    /// The 2-D power of a constant image is all at `k = 0`, shifted to the
    /// centre: `|Σ x|² = (n²)²`.
    #[test]
    fn power_2d_of_a_constant_image_is_centred_dc() {
        let n = 4;
        let image = ndarray::Array2::<F>::ones((n, n));
        let power = Spectrum::power_2d_centered(image.view());
        for ((i, j), &p) in power.indexed_iter() {
            let want = if (i, j) == (n / 2, n / 2) { 256.0 } else { 0.0 };
            assert!((p - want).abs() < 1e-9, "({i}, {j}): {p}");
        }
    }

    #[test]
    fn windows_match_their_formulas() {
        let n = 5;
        for (window, formula) in [
            (Window::Hann, hann_analytical as fn(usize, usize) -> F),
            (Window::Blackman, blackman_analytical),
            (Window::CosineSq, cosine_sq_analytical),
        ] {
            let result = window.apply(ones(n).view());
            for (i, v) in result.iter().enumerate() {
                let expected = formula(n, i);
                assert!(
                    (v - expected).abs() < 1e-12,
                    "{window} idx {i}: {v} != {expected}"
                );
            }
        }
        assert_eq!(Window::None.apply(ones(n).view()), ones(n));
    }

    #[test]
    fn window_boundary_values() {
        let n = 100;
        let hann = Window::Hann.apply(ones(n).view());
        assert!(hann[0].abs() < 1e-10 && hann[n - 1].abs() < 1e-10);
        let blackman = Window::Blackman.apply(ones(n).view());
        assert!((blackman[0] - (0.42 - 0.5 + 0.08)).abs() < 1e-10);
        let cos_sq = Window::CosineSq.apply(ones(n).view());
        assert!((cos_sq[0] - 1.0).abs() < 1e-10);
        assert!(cos_sq[n - 1].abs() < 1e-10);
    }

    #[test]
    fn cosine_sq_midpoint_and_single_sample() {
        // w[2] for N=5: cos²(π·2/(2·4)) = cos²(π/4) = 0.5
        assert!((Window::CosineSq.apply(ones(5).view())[2] - 0.5).abs() < 1e-12);
        for window in Window::ALL.iter().copied() {
            assert_eq!(window.coefficients(1).to_vec(), vec![1.0]);
            assert!(window.apply(Array1::<F>::zeros(0).view()).is_empty());
        }
    }

    #[test]
    fn window_names_are_canonical() {
        assert_eq!(Window::NAMES, &["hann", "blackman", "cosine_sq", "none"]);
        assert_eq!("cosine_sq".parse::<Window>().unwrap(), Window::CosineSq);
        assert!("Hann".parse::<Window>().is_err());
    }

    #[test]
    fn frequency_grid_basic() {
        let grid = FrequencyGrid::new(8, 0.5).unwrap().angular();
        assert_eq!(grid.len(), 5); // 8/2 + 1
        assert!((grid[0] - 0.0).abs() < 1e-10);
        assert!((grid[4] - PI / 0.5).abs() < 1e-10);
        assert!((grid[1] - grid[0] - 2.0 * PI / (8.0 * 0.5)).abs() < 1e-10);
    }

    #[test]
    fn frequency_grid_minimal_and_dt_variation() {
        let grid = FrequencyGrid::new(2, 1.0).unwrap().angular();
        assert_eq!(grid.len(), 2);
        assert!((grid[1] - PI).abs() < 1e-10);

        let grid = FrequencyGrid::new(4, 2.0).unwrap();
        assert_eq!(grid.n_points(), 3);
        assert!((grid.spacing() - PI / 4.0).abs() < 1e-10);
        assert!((grid.angular()[2] - PI / 2.0).abs() < 1e-10);
    }

    #[test]
    fn frequency_grid_uniform_spacing_and_large_n() {
        let grid = FrequencyGrid::new(1024, 0.001).unwrap().angular();
        assert_eq!(grid.len(), 513);
        let spacing = 2.0 * PI / (1024.0 * 0.001);
        for i in 1..grid.len() {
            assert!((grid[i] - grid[i - 1] - spacing).abs() < 1e-10, "index {i}");
        }
        let grid = FrequencyGrid::new(65536, 0.002).unwrap().angular();
        assert_eq!(grid.len(), 32769);
        assert!((grid[32768] - PI / 0.002).abs() < 1e-10);
    }

    #[test]
    fn frequency_grid_refuses_degenerate_input() {
        assert!(FrequencyGrid::new(0, 1.0).is_err());
        assert!(FrequencyGrid::new(8, 0.0).is_err());
        assert!(FrequencyGrid::new(8, F::NAN).is_err());
    }

    /// Unscaled bins equal the DFT written out on the zero-padded signal.
    #[test]
    fn spectrum_bins_are_the_padded_dft() {
        let signal = Array1::from_vec(vec![1.0, -0.5, 0.25, 2.0, -1.0]);
        let out = Spectrum::new(Window::None, 2)
            .unwrap()
            .compute(SpectrumInput {
                signal: signal.view(),
                dt: 0.1,
            })
            .unwrap();
        assert_eq!(out.n_fft, 16); // (2 · 5).next_power_of_two()
        assert_eq!(out.real.len(), 9);
        for k in 0..9 {
            let (mut re, mut im) = (0.0, 0.0);
            for (n, x) in signal.iter().enumerate() {
                let phase = -2.0 * PI * (k * n) as F / 16.0;
                re += x * phase.cos();
                im += x * phase.sin();
            }
            assert!((out.real[k] - re).abs() < 1e-12, "re[{k}]");
            assert!((out.imag[k] - im).abs() < 1e-12, "im[{k}]");
        }
        assert_eq!(
            out.frequencies,
            FrequencyGrid::new(16, 0.1).unwrap().angular()
        );
    }

    #[test]
    fn spectrum_peaks_at_the_signal_frequency() {
        // cos(ω₀ t) with ω₀ on the grid of a 4×-padded 256-sample transform.
        let (n, dt) = (256, 0.5);
        let spectrum = Spectrum::new(Window::Hann, 4).unwrap();
        let grid = FrequencyGrid::new(1024, dt).unwrap();
        let omega0 = 100.0 * grid.spacing();
        let signal = Array1::from_shape_fn(n, |t| (omega0 * t as F * dt).cos());
        let out = spectrum
            .compute(SpectrumInput {
                signal: signal.view(),
                dt,
            })
            .unwrap();
        let power = &out.real * &out.real + &out.imag * &out.imag;
        let peak = power
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0;
        assert_eq!(peak, 100);
        assert!((out.frequencies[peak] - omega0).abs() < 1e-12);
    }

    #[test]
    fn spectrum_refusals() {
        assert!(matches!(
            Spectrum::new(Window::Hann, 0),
            Err(ComputeError::OutOfRange { field: "pad", .. })
        ));
        let spectrum = Spectrum::new(Window::Hann, 1).unwrap();
        let empty = Array1::<F>::zeros(0);
        assert!(matches!(
            spectrum.compute(SpectrumInput {
                signal: empty.view(),
                dt: 1.0
            }),
            Err(ComputeError::EmptyInput)
        ));
        let nan = Array1::from_vec(vec![0.0, F::NAN]);
        assert!(matches!(
            spectrum.compute(SpectrumInput {
                signal: nan.view(),
                dt: 1.0
            }),
            Err(ComputeError::NonFinite { index: 1, .. })
        ));
        let ok = Array1::from_vec(vec![0.0, 1.0]);
        assert!(matches!(
            spectrum.compute(SpectrumInput {
                signal: ok.view(),
                dt: -1.0
            }),
            Err(ComputeError::OutOfRange { field: "dt", .. })
        ));
    }
}
