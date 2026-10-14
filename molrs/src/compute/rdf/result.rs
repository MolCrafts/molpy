//! Result type for the radial distribution function [`Rdf`](super::Rdf):
//! binned pair counts plus the normalized `g(r)` curve (bin edges/centers in
//! Å).

use molrs::op::F;
use ndarray::Array1;

/// The radial distribution function over one or more frames.
///
/// `g(r)` is the ratio of the number of pairs found at separation `r` to the
/// number an ideal gas of the same number density would have there, so it is
/// dimensionless and tends to 1 at large `r`. It is normalized when the
/// result is produced; there is no later step.
///
/// The normalization depends on how the pairs were searched, which the pair
/// source's [`QueryMode`](molrs::core::QueryMode) records:
///
/// - a **self-query** is half-shell — each unordered pair `{i, j}` stored
///   once — while the ideal-gas reference counts ordered pairs, so the
///   histogram is doubled: `g = 2 n(r) / (ρ V_shell N n_frames)`;
/// - a **cross-query** is directed — every query point reports all of its
///   reference neighbours, both orderings present — so it is not:
///   `g = n(r) / (N_a N_b V_shell / V · n_frames)`.
#[derive(Debug, Clone, PartialEq)]
pub struct RdfResult {
    /// Bin edges in angstrom (n_bins + 1).
    pub bin_edges: Array1<F>,
    /// Bin centers in angstrom (n_bins).
    pub bin_centers: Array1<F>,
    /// Normalized g(r), dimensionless.
    pub rdf: Array1<F>,
    /// Raw pair count per bin (dimensionless), summed across frames.
    pub n_r: Array1<F>,
    /// Number of reference points, summed across frames.
    pub n_points: usize,
    /// Number of query points, summed across frames (equal to `n_points` for
    /// a self-query).
    pub n_query_points: usize,
    /// Whether the pairs were a half-shell self-query (`true`) or a directed
    /// cross-query (`false`).
    pub self_query: bool,
    /// Total normalization volume in Å³, summed across frames (divided by
    /// `n_frames` before use, giving the mean box volume).
    pub volume: F,
    /// Inner cutoff (lower edge of bin 0), Å.
    pub r_min: F,
    /// Number of frames fed into `n_r` / `volume` / `n_points`.
    pub n_frames: usize,
    /// Dimensionality used for the shell-volume normalization (2 or 3).
    pub dimensionality: u8,
}
