//! Signal-processing primitives for molrs analysis crates.
//!
//! Four families:
//!
//! * [`LinearCorrelator`] — the one zero-padded Wiener–Khinchin kernel:
//!   weighted auto-, cross- and symmetric cross-correlation of equal-length
//!   series with the FFT plans made once. [`acf_fft`] / [`xcorr_fft`] (and
//!   their accumulating and planner-taking forms) are wrappers over it.
//! * [`apply_window`] / [`window_coefficients`] / [`WindowType`] — Hann,
//!   Blackman and cosine² window functions for spectral estimation, with
//!   [`WindowType::from_name`] as the one name parser.
//! * [`forward_fft_onesided`] — the unscaled one-sided spectrum of a
//!   zero-padded real signal.
//! * [`frequency_grid`] — equally-spaced angular-frequency grids matched to
//!   the conventions of [`acf_fft`] + zero-padded FFTs.
//!
//! No unit assumptions; the caller chooses dt and interprets the
//! returned ω accordingly (`rad / [time]` where `[time]` matches dt).
//!
//! Higher-level dielectric / MSD / VACF analyses live in
//! `molrs::compute` and compose these primitives.

mod acf;
mod fft;
mod frequency_grid;
mod window;

pub use acf::{
    LinearCorrelator, SignalError, acf_fft, acf_fft_accumulate, acf_fft_with_planner, xcorr_fft,
    xcorr_fft_accumulate, xcorr_fft_with_planner,
};
pub use fft::forward_fft_onesided;
pub use frequency_grid::frequency_grid;
pub use window::{WindowType, apply_window, window_coefficients};
