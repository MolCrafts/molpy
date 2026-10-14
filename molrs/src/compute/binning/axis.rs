//! [`Axis`] — one binned coordinate: a validated `(min, max, n_bins)`.

use std::f64::consts::PI;

use molrs::op::F;
use ndarray::Array1;

use crate::compute::ComputeError;

/// A linear (cloud-in-cell) split: two bins and their weights.
pub(crate) type LinearSplit = ([usize; 2], [F; 2]);

/// One binned coordinate: `n_bins` equal-width bins over `[min, max)`.
///
/// Bin `i` is the half-open interval `[edges[i], edges[i + 1])`, with
/// `edges[i] = min + i · width`; a sample equal to `max` lies outside the
/// axis. [`index`](Self::index) is the one place a sample becomes a bin
/// number in `compute`, so every histogram, radial distribution and grid
/// bins the same way.
///
/// An `Axis` is configuration: it is validated once, in [`new`](Self::new),
/// and handed to an analysis's constructor.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Axis {
    min: F,
    max: F,
    n_bins: usize,
    width: F,
}

impl Axis {
    /// `n_bins` equal bins over `[min, max)`.
    ///
    /// # Errors
    /// [`ComputeError::OutOfRange`] for `n_bins == 0`, a non-finite bound or
    /// `max ≤ min`.
    pub fn new(min: F, max: F, n_bins: usize) -> Result<Self, ComputeError> {
        if n_bins == 0 {
            return Err(ComputeError::OutOfRange {
                field: "Axis::n_bins",
                value: n_bins.to_string(),
            });
        }
        if !(min.is_finite() && max.is_finite()) || max <= min {
            return Err(ComputeError::OutOfRange {
                field: "Axis::range",
                value: format!("min={min}, max={max}"),
            });
        }
        Ok(Self {
            min,
            max,
            n_bins,
            width: (max - min) / n_bins as F,
        })
    }

    /// Lower edge of bin 0.
    pub fn min(&self) -> F {
        self.min
    }

    /// Upper edge of the last bin.
    pub fn max(&self) -> F {
        self.max
    }

    /// Number of bins.
    pub fn n_bins(&self) -> usize {
        self.n_bins
    }

    /// Bin width, `(max − min) / n_bins`.
    pub fn width(&self) -> F {
        self.width
    }

    /// The lower edge of bin `i` (`i = n_bins` gives the last upper edge).
    fn edge(&self, i: usize) -> F {
        self.min + i as F * self.width
    }

    /// The `n_bins + 1` bin edges, `min + i · width`.
    pub fn edges(&self) -> Array1<F> {
        Array1::from_iter((0..=self.n_bins).map(|i| self.edge(i)))
    }

    /// The `n_bins` bin centres, `min + (i + ½) · width`.
    pub fn centers(&self) -> Array1<F> {
        Array1::from_iter((0..self.n_bins).map(|i| self.min + (i as F + 0.5) * self.width))
    }

    /// The bin holding `x`: `Some(i)` with `edges[i] ≤ x < edges[i + 1]`, or
    /// `None` when `x` lies outside `[min, max)`.
    ///
    /// # Errors
    /// [`ComputeError::NonFinite`] for a NaN or infinite sample (`index` 0:
    /// the axis sees one sample at a time). A non-finite sample never lands
    /// in a bin.
    pub fn index(&self, x: F) -> Result<Option<usize>, ComputeError> {
        if !x.is_finite() {
            return Err(Self::non_finite());
        }
        if x < self.min || x >= self.max {
            return Ok(None);
        }
        Ok(Some(self.locate(x)))
    }

    /// Like [`index`](Self::index), with the upper edge closed: `x = max` is
    /// in the last bin. For a coordinate whose range is attained at both
    /// ends — a polar angle in `[0, π]`, an azimuth in `[−π, π]`, a
    /// reciprocal-lattice magnitude up to `k_max`.
    ///
    /// # Errors
    /// [`ComputeError::NonFinite`] for a NaN or infinite sample.
    pub fn index_closed(&self, x: F) -> Result<Option<usize>, ComputeError> {
        if !x.is_finite() {
            return Err(Self::non_finite());
        }
        if x < self.min || x > self.max {
            return Ok(None);
        }
        Ok(Some(self.locate(x)))
    }

    /// Bin of an in-range `x`. The proportional guess is corrected against
    /// the edges, so the bins are exactly the half-open intervals
    /// [`edges`](Self::edges) reports.
    fn locate(&self, x: F) -> usize {
        let last = self.n_bins - 1;
        let guess = ((x - self.min) / (self.max - self.min) * self.n_bins as F) as usize;
        let mut i = guess.min(last);
        while i > 0 && x < self.edge(i) {
            i -= 1;
        }
        while i < last && x >= self.edge(i + 1) {
            i += 1;
        }
        i
    }

    /// The cloud-in-cell (linear) split of a sample between the two bins
    /// whose centres straddle it: `([i, i + 1], [1 − f, f])`.
    ///
    /// With `p = (x − min) · n_bins / (max − min) − ½` (the centre of bin `i`
    /// sits at `p = i`), `i = ⌊p⌋` clamped to `[0, n_bins − 2]` and
    /// `f = p − i`; a sample in the first or last half-bin goes wholly to bin
    /// 0 or `n_bins − 1`. A single-bin axis deposits everything in bin 0. The
    /// support is the closed range `[min, max]` — the linear deposit of the
    /// reference implementation's `CDF::AddToBin` (`src/df.cpp`), whose last
    /// half-bin includes `max`. `None` outside it.
    ///
    /// # Errors
    /// [`ComputeError::NonFinite`] for a NaN or infinite sample.
    pub(crate) fn linear(&self, x: F) -> Result<Option<LinearSplit>, ComputeError> {
        if !x.is_finite() {
            return Err(Self::non_finite());
        }
        if x < self.min || x > self.max {
            return Ok(None);
        }
        if self.n_bins == 1 {
            return Ok(Some(([0, 0], [1.0, 0.0])));
        }
        let fac = self.n_bins as F / (self.max - self.min);
        let p = (x - self.min) * fac - 0.5;
        let floor = p.floor();
        let (i, frac) = if floor < 0.0 {
            (0usize, 0.0)
        } else if floor > (self.n_bins - 2) as F {
            (self.n_bins - 2, 1.0)
        } else {
            (floor as usize, p - floor)
        };
        Ok(Some(([i, i + 1], [1.0 - frac, frac])))
    }

    /// The volume of each radial shell `[edges[i], edges[i + 1])` in `dim`
    /// dimensions: `π (r_o² − r_i²)` for `dim = 2`, `(4/3) π (r_o³ − r_i³)`
    /// for `dim = 3`.
    ///
    /// # Errors
    /// [`ComputeError::OutOfRange`] for a dimensionality other than 2 or 3,
    /// or an axis starting below zero (not a radius).
    pub fn shell_volumes(&self, dim: u8) -> Result<Array1<F>, ComputeError> {
        if self.min < 0.0 {
            return Err(ComputeError::OutOfRange {
                field: "Axis::min (a radial axis starts at r ≥ 0)",
                value: self.min.to_string(),
            });
        }
        let shell: fn(F, F) -> F = match dim {
            2 => |r_inner, r_outer| PI * (r_outer.powi(2) - r_inner.powi(2)),
            3 => |r_inner, r_outer| (4.0 / 3.0) * PI * (r_outer.powi(3) - r_inner.powi(3)),
            _ => {
                return Err(ComputeError::OutOfRange {
                    field: "dimensionality (2 or 3)",
                    value: dim.to_string(),
                });
            }
        };
        Ok(Array1::from_iter(
            (0..self.n_bins).map(|i| shell(self.edge(i), self.edge(i + 1))),
        ))
    }

    fn non_finite() -> ComputeError {
        ComputeError::NonFinite {
            where_: "histogram sample",
            index: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_rejects_degenerate_axes() {
        for (min, max, n) in [
            (0.0, 1.0, 0),
            (1.0, 1.0, 4),
            (2.0, 1.0, 4),
            (F::NAN, 1.0, 4),
            (0.0, F::INFINITY, 4),
        ] {
            assert!(
                matches!(Axis::new(min, max, n), Err(ComputeError::OutOfRange { .. })),
                "({min}, {max}, {n}) must be rejected"
            );
        }
    }

    #[test]
    fn edges_and_centers() {
        let axis = Axis::new(1.0, 3.0, 4).unwrap();
        assert_eq!(axis.width(), 0.5);
        assert_eq!(axis.edges().to_vec(), vec![1.0, 1.5, 2.0, 2.5, 3.0]);
        assert_eq!(axis.centers().to_vec(), vec![1.25, 1.75, 2.25, 2.75]);
    }

    /// Bins are `[lo, hi)`: every edge belongs to the bin above it, `max` to
    /// no bin.
    #[test]
    fn bins_are_half_open() {
        let axis = Axis::new(0.0, 1.0, 10).unwrap();
        let edges = axis.edges();
        for i in 0..10 {
            assert_eq!(axis.index(edges[i]).unwrap(), Some(i), "edge {i}");
            let below = edges[i + 1] - 1e-12;
            assert_eq!(axis.index(below).unwrap(), Some(i), "below edge {}", i + 1);
        }
        assert_eq!(axis.index(1.0).unwrap(), None);
        assert_eq!(axis.index(-1e-15).unwrap(), None);
        assert_eq!(axis.index_closed(1.0).unwrap(), Some(9));
        assert_eq!(axis.index_closed(1.0 + 1e-12).unwrap(), None);
    }

    #[test]
    fn non_finite_samples_are_errors() {
        let axis = Axis::new(0.0, 1.0, 4).unwrap();
        for x in [F::NAN, F::INFINITY, F::NEG_INFINITY] {
            assert!(matches!(axis.index(x), Err(ComputeError::NonFinite { .. })));
            assert!(matches!(
                axis.index_closed(x),
                Err(ComputeError::NonFinite { .. })
            ));
            assert!(matches!(
                axis.linear(x),
                Err(ComputeError::NonFinite { .. })
            ));
        }
    }

    /// Cloud-in-cell: centre-aligned samples stay whole, a midpoint splits
    /// evenly, the outer half-bins clamp, and `max` is inside.
    #[test]
    fn linear_split_matches_reference_rule() {
        let axis = Axis::new(0.0, 4.0, 4).unwrap();
        assert_eq!(axis.linear(0.5).unwrap(), Some(([0, 1], [1.0, 0.0])));
        assert_eq!(axis.linear(1.0).unwrap(), Some(([0, 1], [0.5, 0.5])));
        assert_eq!(axis.linear(1.75).unwrap(), Some(([1, 2], [0.75, 0.25])));
        assert_eq!(axis.linear(3.9).unwrap(), Some(([2, 3], [0.0, 1.0])));
        assert_eq!(axis.linear(4.0).unwrap(), Some(([2, 3], [0.0, 1.0])));
        assert_eq!(axis.linear(4.5).unwrap(), None);
        let single = Axis::new(0.0, 1.0, 1).unwrap();
        assert_eq!(single.linear(0.7).unwrap(), Some(([0, 0], [1.0, 0.0])));
    }

    #[test]
    fn shell_volumes_match_closed_forms() {
        let axis = Axis::new(0.0, 2.0, 2).unwrap();
        let v3 = axis.shell_volumes(3).unwrap();
        assert!((v3[0] - 4.0 / 3.0 * PI).abs() < 1e-12);
        assert!((v3[1] - 4.0 / 3.0 * PI * 7.0).abs() < 1e-12);
        let v2 = axis.shell_volumes(2).unwrap();
        assert!((v2[0] - PI).abs() < 1e-12);
        assert!((v2[1] - 3.0 * PI).abs() < 1e-12);
        // The shells tile the ball.
        assert!((v3.sum() - 4.0 / 3.0 * PI * 8.0).abs() < 1e-12);

        assert!(matches!(
            axis.shell_volumes(4),
            Err(ComputeError::OutOfRange { .. })
        ));
        assert!(matches!(
            Axis::new(-1.0, 1.0, 2).unwrap().shell_volumes(3),
            Err(ComputeError::OutOfRange { .. })
        ));
    }
}
