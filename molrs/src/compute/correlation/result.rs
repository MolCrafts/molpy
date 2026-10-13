//! [`TimeCorrelation`] — the one result type of every time-correlation
//! function in molrs — and the vocabulary it carries.

use ndarray::{Array1, Array2};

use crate::compute::ComputeResult;

/// How the raw lag sums are scaled.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Normalization {
    /// `Σ / (N · counts[t])`: every lag averages exactly the time origins
    /// that exist for it (and the `N` entities). The multiple-time-origin
    /// estimator of Allen & Tildesley §8.4; what every molrs correlator used
    /// before the engine and still uses.
    Unbiased,
    /// `Σ / (N · counts[0])`: one divisor for every lag, so the tail tapers
    /// as `counts[t]/counts[0]`. Legitimate for spectral work; never the
    /// default.
    Biased,
    /// `Σ`: the plain sum over origins, entities and (weighted) components.
    /// The spectroscopy raw correlators feed this to their windowed
    /// transforms, where the overall scale is arbitrary.
    Raw,
}

impl Normalization {
    /// Parse `"unbiased"`, `"biased"` or `"raw"`.
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "unbiased" => Some(Self::Unbiased),
            "biased" => Some(Self::Biased),
            "raw" => Some(Self::Raw),
            _ => None,
        }
    }

    /// The name [`from_name`](Self::from_name) accepts.
    pub fn name(self) -> &'static str {
        match self {
            Self::Unbiased => "unbiased",
            Self::Biased => "biased",
            Self::Raw => "raw",
        }
    }
}

/// What was correlated with what.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CorrelationKind {
    /// `⟨a(τ) · a(τ+t)⟩`.
    Auto,
    /// `⟨a(τ) · b(τ+t)⟩`.
    Cross,
    /// `⟨|a(τ+t) − a(τ)|²⟩` — a mean squared displacement.
    Displacement,
    /// `⟨(a(τ+t) − a(τ)) · (b(τ+t) − b(τ))⟩` — a cross displacement
    /// correlation (Onsager).
    CrossDisplacement,
    /// `⟨P_ℓ(û(τ) · û(τ+t))⟩` of unit vectors — a reorientational
    /// correlation of Legendre order `ell`.
    Legendre { ell: u32 },
}

/// One time-correlation curve with the sample counts and estimator settings
/// behind it.
///
/// `values[t]` is `C(t)` for `lags[t] = t` frames, `t = 0..=max_lag`; the
/// time axis is [`lag_times`](Self::lag_times) `= lags · dt` in the caller's
/// `dt` unit (`dt = 1.0` means frames). `counts[t]` is the number of time
/// origins that contributed to lag `t`, `(T − 1 − t) / stride + 1`, which is
/// the divisor of the [`Unbiased`](Normalization::Unbiased) form and what a
/// caller needs to re-normalize a [`Raw`](Normalization::Raw) curve.
///
/// `per_entity`, when requested, holds one row per entity in the same
/// normalization; `values` is then the entity mean of the rows.
#[derive(Debug, Clone)]
pub struct TimeCorrelation {
    /// Lags `0, 1, …, max_lag`, in frames.
    pub lags: Array1<usize>,
    /// Frame spacing in the caller's unit (`1.0` = frames).
    pub dt: f64,
    /// `C(t)`, summed over components, reduced over entities as
    /// `normalization` says. Units: the product of the two inputs' units.
    pub values: Array1<f64>,
    /// Time origins per lag.
    pub counts: Array1<usize>,
    /// Entities the curve is averaged (or summed) over.
    pub n_entities: usize,
    /// How `values` were scaled.
    pub normalization: Normalization,
    /// What was correlated.
    pub kind: CorrelationKind,
    /// `(n_entities, max_lag + 1)` per-entity curves, when requested.
    pub per_entity: Option<Array2<f64>>,
}

impl TimeCorrelation {
    /// `lags · dt`.
    pub fn lag_times(&self) -> Array1<f64> {
        lag_times(self.max_lag(), self.dt)
    }

    /// The last lag, in frames.
    pub fn max_lag(&self) -> usize {
        self.values.len().saturating_sub(1)
    }

    /// Number of lags (`max_lag + 1`).
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// `true` for an empty curve (never produced by the engine).
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
}

impl ComputeResult for TimeCorrelation {}

/// Lag grid `τ = i·dt` for `i = 0..=max_lag`.
#[inline]
pub fn lag_times(max_lag: usize, dt: f64) -> Array1<f64> {
    Array1::from_iter((0..=max_lag).map(|i| i as f64 * dt))
}

/// Time origins available at lag `t` when origins are taken every `stride`
/// frames from `0`: `(n_frames − 1 − t) / stride + 1`.
#[inline]
pub(crate) fn origin_count(n_frames: usize, lag: usize, stride: usize) -> usize {
    (n_frames - 1 - lag) / stride + 1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn origin_counts_follow_the_stride() {
        assert_eq!(origin_count(10, 0, 1), 10);
        assert_eq!(origin_count(10, 3, 1), 7);
        assert_eq!(origin_count(10, 0, 3), 4); // origins 0, 3, 6, 9
        assert_eq!(origin_count(10, 2, 3), 3); // origins 0, 3, 6 (9 + 2 > 9)
        assert_eq!(origin_count(10, 9, 3), 1);
    }

    #[test]
    fn names_round_trip() {
        for n in [
            Normalization::Unbiased,
            Normalization::Biased,
            Normalization::Raw,
        ] {
            assert_eq!(Normalization::from_name(n.name()), Some(n));
        }
        assert_eq!(Normalization::from_name("Unbiased"), None);
    }
}
