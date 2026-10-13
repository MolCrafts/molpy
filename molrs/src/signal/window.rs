use ndarray::ArrayD;

use super::acf::SignalError;

/// Window function used to suppress spectral leakage when Fourier-
/// transforming a finite-length signal.
///
/// Both variants use the **symmetric** (`N − 1` denominator) form
/// matching `scipy.signal.windows.{hann,blackman}(M, sym=True)`. For
/// pure spectral analysis the periodic (`sym=False`) form is sometimes
/// preferred; if needed, add a `periodic` variant rather than changing
/// the existing semantics.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WindowType {
    /// Hann window: `w[n] = 0.5 · (1 − cos(2π·n/(N−1)))`.
    Hann,
    /// Approximate Blackman window with the textbook three-term
    /// coefficients `(0.42, 0.5, 0.08)`. Use the exact Blackman
    /// coefficients `(0.42659, 0.49656, 0.07685)` only if added as a
    /// separate variant.
    Blackman,
    /// Squared-cosine window: `w[n] = cos²(π·n / (2·(N−1)))`.
    ///
    /// This is the default window in SchNetPack's vibrational-spectra
    /// module. It tapers to zero at the last sample (`w[N−1] = 0`),
    /// unlike Hann which goes to zero at both ends. Preferred for
    /// one-sided autocorrelation signals where only the right edge
    /// needs suppression.
    CosineSq,
    /// Identity window (no taper). Required for DRS pipelines that match
    /// literature/scripts with `window="none"`.
    None,
}

impl WindowType {
    /// Parse a window name, case-sensitive, as every molrs entry point
    /// spells it: `"hann"`, `"blackman"`, `"cosine_sq"`, and `"none"` /
    /// `"None"` / `"identity"` for no taper.
    pub fn from_name(name: &str) -> Result<Self, SignalError> {
        match name {
            "hann" => Ok(Self::Hann),
            "blackman" => Ok(Self::Blackman),
            "cosine_sq" => Ok(Self::CosineSq),
            "none" | "None" | "identity" => Ok(Self::None),
            other => Err(SignalError::UnknownWindow(other.to_string())),
        }
    }

    /// The name [`from_name`](Self::from_name) accepts for this window.
    pub fn name(self) -> &'static str {
        match self {
            Self::Hann => "hann",
            Self::Blackman => "blackman",
            Self::CosineSq => "cosine_sq",
            Self::None => "none",
        }
    }
}

/// The `n` coefficients of `window` (symmetric, `N − 1` denominator; a
/// single-sample window is `[1.0]`).
///
/// This is the one place the window formulas are written: [`apply_window`]
/// and every spectral taper in `molrs::compute` multiply by these numbers.
pub fn window_coefficients(window: WindowType, n: usize) -> Vec<f64> {
    if n == 0 {
        return Vec::new();
    }
    match window {
        WindowType::None => vec![1.0; n],
        WindowType::Hann => (0..n)
            .map(|k| {
                if n == 1 {
                    1.0
                } else {
                    0.5 * (1.0 - (2.0 * std::f64::consts::PI * k as f64 / (n - 1) as f64).cos())
                }
            })
            .collect(),
        WindowType::Blackman => (0..n)
            .map(|k| {
                if n == 1 {
                    1.0
                } else {
                    0.42 - 0.5 * (2.0 * std::f64::consts::PI * k as f64 / (n - 1) as f64).cos()
                        + 0.08 * (4.0 * std::f64::consts::PI * k as f64 / (n - 1) as f64).cos()
                }
            })
            .collect(),
        WindowType::CosineSq => (0..n)
            .map(|k| {
                if n == 1 {
                    1.0
                } else {
                    let angle = std::f64::consts::PI * k as f64 / (2.0 * (n - 1) as f64);
                    angle.cos().powi(2)
                }
            })
            .collect(),
    }
}

/// Apply a window function element-wise along `axis`.
///
/// Returns a new array of the same shape; the input is never mutated.
/// The window is applied multiplicatively, with no amplitude
/// correction (callers that need a normalized window energy must
/// rescale themselves).
///
/// # Errors
/// * `AxisOutOfBounds` if `axis >= data.ndim()`.
pub fn apply_window(
    data: &ArrayD<f64>,
    window: WindowType,
    axis: usize,
) -> Result<ArrayD<f64>, SignalError> {
    let ndim = data.ndim();
    if axis >= ndim {
        return Err(SignalError::AxisOutOfBounds { axis, ndim });
    }

    let window_len = data.shape()[axis];
    if window_len == 0 || window == WindowType::None {
        return Ok(data.clone());
    }

    let w = window_coefficients(window, window_len);
    let mut result = data.clone();
    for idx in ndarray::indices_of(&result) {
        let axis_idx = idx[axis];
        result[idx] *= w[axis_idx];
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::ArrayD;

    fn hann_analytical(n: usize, idx: usize) -> f64 {
        if n == 1 {
            return 1.0;
        }
        0.5 * (1.0 - (2.0 * std::f64::consts::PI * idx as f64 / (n - 1) as f64).cos())
    }

    fn blackman_analytical(n: usize, idx: usize) -> f64 {
        if n == 1 {
            return 1.0;
        }
        0.42 - 0.5 * (2.0 * std::f64::consts::PI * idx as f64 / (n - 1) as f64).cos()
            + 0.08 * (4.0 * std::f64::consts::PI * idx as f64 / (n - 1) as f64).cos()
    }

    #[test]
    fn test_hann_1d_matches_analytical() {
        let n = 5;
        let data = ArrayD::from_shape_vec(ndarray::IxDyn(&[n]), vec![1.0; n]).unwrap();
        let result = apply_window(&data, WindowType::Hann, 0).unwrap();
        assert_eq!(result.shape(), &[5]);
        for (i, v) in result.iter().enumerate() {
            let expected = hann_analytical(n, i);
            assert!((v - expected).abs() < 1e-12, "idx {i}: {v} != {expected}");
        }
    }

    #[test]
    fn test_blackman_1d_matches_analytical() {
        let n = 5;
        let data = ArrayD::from_shape_vec(ndarray::IxDyn(&[n]), vec![1.0; n]).unwrap();
        let result = apply_window(&data, WindowType::Blackman, 0).unwrap();
        for (i, v) in result.iter().enumerate() {
            let expected = blackman_analytical(n, i);
            assert!((v - expected).abs() < 1e-12, "idx {i}: {v} != {expected}");
        }
    }

    #[test]
    fn test_hann_2d_axis0_each_column_is_window() {
        let data = ArrayD::from_elem(ndarray::IxDyn(&[5, 3]), 1.0);
        let result = apply_window(&data, WindowType::Hann, 0).unwrap();
        assert_eq!(result.shape(), &[5, 3]);
        let expected_col = (0..5).map(|i| hann_analytical(5, i)).collect::<Vec<_>>();
        for col in 0..3 {
            for row in 0..5 {
                let v = result[[row, col]];
                assert!((v - expected_col[row]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn test_hann_2d_axis1_each_row_is_window() {
        let data = ArrayD::from_elem(ndarray::IxDyn(&[3, 5]), 1.0);
        let result = apply_window(&data, WindowType::Hann, 1).unwrap();
        assert_eq!(result.shape(), &[3, 5]);
        let expected_row = (0..5).map(|i| hann_analytical(5, i)).collect::<Vec<_>>();
        for row in 0..3 {
            for col in 0..5 {
                let v = result[[row, col]];
                assert!((v - expected_row[col]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn test_hann_boundary_values() {
        let n = 100;
        let data = ArrayD::from_elem(ndarray::IxDyn(&[n]), 1.0);
        let result = apply_window(&data, WindowType::Hann, 0).unwrap();
        assert!((result[0] - 0.0).abs() < 1e-10);
        assert!((result[n - 1] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_blackman_boundary_values() {
        let n = 100;
        let data = ArrayD::from_elem(ndarray::IxDyn(&[n]), 1.0);
        let result = apply_window(&data, WindowType::Blackman, 0).unwrap();
        let expected_first = 0.42 - 0.5 + 0.08; // = 0.0 at n=0
        assert!((result[0] - expected_first).abs() < 1e-10);
        let expected_last = 0.42
            - 0.5 * (2.0 * std::f64::consts::PI * (n - 1) as f64 / (n - 1) as f64).cos()
            + 0.08 * (4.0 * std::f64::consts::PI * (n - 1) as f64 / (n - 1) as f64).cos();
        assert!((result[n - 1] - expected_last).abs() < 1e-10);
    }

    #[test]
    fn test_axis_out_of_bounds() {
        let data = ArrayD::from_elem(ndarray::IxDyn(&[5, 3]), 1.0);
        let err = apply_window(&data, WindowType::Hann, 2).unwrap_err();
        assert!(matches!(
            err,
            SignalError::AxisOutOfBounds { axis: 2, ndim: 2 }
        ));
    }

    fn cosine_sq_analytical(n: usize, idx: usize) -> f64 {
        if n == 1 {
            return 1.0;
        }
        let angle = std::f64::consts::PI * idx as f64 / (2.0 * (n - 1) as f64);
        angle.cos().powi(2)
    }

    #[test]
    fn test_cosine_sq_1d_matches_analytical() {
        let n = 5;
        let data = ArrayD::from_shape_vec(ndarray::IxDyn(&[n]), vec![1.0; n]).unwrap();
        let result = apply_window(&data, WindowType::CosineSq, 0).unwrap();
        assert_eq!(result.shape(), &[5]);
        for (i, v) in result.iter().enumerate() {
            let expected = cosine_sq_analytical(n, i);
            assert!(
                (v - expected).abs() < 1e-12,
                "CosineSq idx {i}: {v} != {expected}"
            );
        }
    }

    #[test]
    fn test_cosine_sq_boundary_values() {
        let n = 100;
        let data = ArrayD::from_elem(ndarray::IxDyn(&[n]), 1.0);
        let result = apply_window(&data, WindowType::CosineSq, 0).unwrap();
        // w[0] = cos²(0) = 1.0
        assert!((result[0] - 1.0).abs() < 1e-10);
        // w[N-1] = cos²(π/2) = 0.0
        assert!((result[n - 1] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_sq_midpoint() {
        let n = 5;
        let data = ArrayD::from_elem(ndarray::IxDyn(&[n]), 1.0);
        let result = apply_window(&data, WindowType::CosineSq, 0).unwrap();
        // w[2] for N=5: cos²(π*2/(2*4)) = cos²(π/4) = 0.5
        assert!((result[2] - 0.5).abs() < 1e-12);
    }

    #[test]
    fn test_cosine_sq_single_element() {
        let data = ArrayD::from_elem(ndarray::IxDyn(&[1]), 1.0);
        let result = apply_window(&data, WindowType::CosineSq, 0).unwrap();
        assert!((result[0] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_cosine_sq_2d_axis1() {
        let data = ArrayD::from_elem(ndarray::IxDyn(&[3, 5]), 1.0);
        let result = apply_window(&data, WindowType::CosineSq, 1).unwrap();
        assert_eq!(result.shape(), &[3, 5]);
        let expected_row = (0..5)
            .map(|i| cosine_sq_analytical(5, i))
            .collect::<Vec<_>>();
        for row in 0..3 {
            for col in 0..5 {
                let v = result[[row, col]];
                assert!(
                    (v - expected_row[col]).abs() < 1e-12,
                    "CosineSq 2D [{row},{col}]: {v} != {}",
                    expected_row[col]
                );
            }
        }
    }

    #[test]
    fn from_name_round_trips_and_rejects_unknown() {
        for (name, want) in [
            ("hann", WindowType::Hann),
            ("blackman", WindowType::Blackman),
            ("cosine_sq", WindowType::CosineSq),
            ("none", WindowType::None),
            ("None", WindowType::None),
            ("identity", WindowType::None),
        ] {
            assert_eq!(WindowType::from_name(name).unwrap(), want);
        }
        assert_eq!(
            WindowType::from_name(WindowType::Hann.name()).unwrap(),
            WindowType::Hann
        );
        assert!(matches!(
            WindowType::from_name("Hann"),
            Err(SignalError::UnknownWindow(_))
        ));
        assert!(matches!(
            WindowType::from_name("boxcar"),
            Err(SignalError::UnknownWindow(_))
        ));
    }

    #[test]
    fn coefficients_match_apply_window_and_the_one_sided_taper() {
        let n = 9;
        let ones = ArrayD::from_elem(ndarray::IxDyn(&[n]), 1.0);
        for w in [
            WindowType::Hann,
            WindowType::Blackman,
            WindowType::CosineSq,
            WindowType::None,
        ] {
            let via_apply = apply_window(&ones, w, 0).unwrap();
            let coeffs = window_coefficients(w, n);
            for k in 0..n {
                assert_eq!(via_apply[k], coeffs[k], "{w:?} k={k}");
            }
        }
        // The cos² taper spelled as `cos²(π k / (2 max_lag))` is the same numbers.
        let max_lag = n - 1;
        let denom = 2.0 * max_lag.max(1) as f64;
        let coeffs = window_coefficients(WindowType::CosineSq, n);
        for (k, c) in coeffs.iter().enumerate() {
            let angle = std::f64::consts::PI * k as f64 / denom;
            assert_eq!(*c, angle.cos().powi(2), "k={k}");
        }
        assert_eq!(window_coefficients(WindowType::Hann, 1), vec![1.0]);
        assert!(window_coefficients(WindowType::Hann, 0).is_empty());
    }
}
