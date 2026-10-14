//! Generic distance-binned correlation `⟨A_i · B_j⟩(r)`.

use crate::compute::ComputeResult;
use molrs::core::FrameAccess;
use molrs::core::Neighbors;
use molrs::op::F;
use ndarray::Array1;

use crate::compute::ComputeError;
use crate::compute::LegacyCompute;
use crate::compute::require_dist_sq;

/// Correlation-function calculator. Stateless container of bin parameters.
///
/// Mirrors `freud.density.CorrelationFunction`
/// ([source](https://github.com/glotzerlab/freud/blob/main/freud/density/CorrelationFunction.cc)).
///
/// Given per-particle real scalar fields `A_i` and `B_j` and a neighbor
/// list, builds the histogram `⟨A_i · B_j⟩(r)` averaged over all pairs
/// that fall into each shell `[r, r + dr)`. Empty bins return 0.
///
/// freud also supports complex-valued fields with the convention
/// `A_i · conj(B_j)`; the real version is implemented here. The complex
/// variant requires only a handful of extra lines and will follow when the
/// first downstream consumer (e.g. `LocalDescriptors` in Phase 6) needs it.
#[derive(Debug, Clone)]
pub struct CorrelationFunction {
    n_bins: usize,
    r_min: F,
    r_max: F,
    bin_width: F,
    bin_edges: Array1<F>,
    bin_centers: Array1<F>,
}

impl CorrelationFunction {
    /// `n_bins` radial bins over `[r_min, r_max]` (Å).
    pub fn new(n_bins: usize, r_max: F, r_min: F) -> Result<Self, ComputeError> {
        if n_bins == 0 {
            return Err(ComputeError::OutOfRange {
                field: "CorrelationFunction::n_bins",
                value: "0".into(),
            });
        }
        if r_min.is_nan() || r_min < 0.0 {
            return Err(ComputeError::OutOfRange {
                field: "CorrelationFunction::r_min",
                value: r_min.to_string(),
            });
        }
        if r_max.is_nan() || r_max <= r_min {
            return Err(ComputeError::OutOfRange {
                field: "CorrelationFunction::r_max",
                value: format!("r_max={r_max}, r_min={r_min}"),
            });
        }
        let bin_width = (r_max - r_min) / n_bins as F;
        let bin_edges = Array1::from_iter((0..=n_bins).map(|i| r_min + i as F * bin_width));
        let bin_centers =
            Array1::from_iter((0..n_bins).map(|i| r_min + (i as F + 0.5) * bin_width));
        Ok(Self {
            n_bins,
            r_min,
            r_max,
            bin_width,
            bin_edges,
            bin_centers,
        })
    }

    pub fn n_bins(&self) -> usize {
        self.n_bins
    }
    pub fn bin_edges(&self) -> &Array1<F> {
        &self.bin_edges
    }
    pub fn bin_centers(&self) -> &Array1<F> {
        &self.bin_centers
    }

    fn one_frame(
        &self,
        nlist: &Neighbors,
        values_a: &[F],
        values_b: &[F],
    ) -> Result<CorrelationFunctionResult, ComputeError> {
        let r_min_sq = self.r_min * self.r_min;
        let r_max_sq = self.r_max * self.r_max;
        let i_idx = nlist.query_point_indices();
        let j_idx = nlist.point_indices();
        let n_pairs = nlist.n_pairs();
        // Every pair is binned by its separation; a table without `dist_sq`
        // cannot say which bin.
        let dist_sq = require_dist_sq(nlist)?;
        let symmetric = matches!(nlist.mode(), molrs::core::QueryMode::SelfQuery { .. });

        let mut sum = Array1::<F>::zeros(self.n_bins);
        let mut counts = Array1::<u64>::zeros(self.n_bins);

        for k in 0..n_pairs {
            let d2 = dist_sq[k];
            if d2 < r_min_sq || d2 >= r_max_sq || d2 == 0.0 {
                continue;
            }
            let d = d2.sqrt();
            let b = ((d - self.r_min) / self.bin_width) as usize;
            if b >= self.n_bins {
                continue;
            }
            let i = i_idx[k] as usize;
            let j = j_idx[k] as usize;
            if i >= values_a.len() || j >= values_b.len() {
                return Err(ComputeError::DimensionMismatch {
                    expected: i.max(j) + 1,
                    got: values_a.len().min(values_b.len()),
                    what: "CorrelationFunction values length",
                });
            }
            sum[b] += values_a[i] * values_b[j];
            counts[b] += 1;
            if symmetric {
                // For a self-query list, the symmetric pair (j → i) is
                // implicit; honour it for ⟨A · B⟩.
                sum[b] += values_a[j] * values_b[i];
                counts[b] += 1;
            }
        }

        let mut correlation = Array1::<F>::zeros(self.n_bins);
        for b in 0..self.n_bins {
            if counts[b] > 0 {
                correlation[b] = sum[b] / counts[b] as F;
            }
        }
        Ok(CorrelationFunctionResult {
            bin_edges: self.bin_edges.clone(),
            bin_centers: self.bin_centers.clone(),
            bin_counts: counts,
            correlation,
        })
    }
}

/// `Args` for [`CorrelationFunction`]: parallel per-frame triplets of
/// `(neighbor list, A values, B values)`. All three slices must have the
/// same length as `frames`.
pub struct CorrelationFunctionArgs<'a> {
    pub nlists: &'a [Neighbors],
    pub values_a: &'a [Vec<F>],
    pub values_b: &'a [Vec<F>],
}

impl LegacyCompute for CorrelationFunction {
    type Args<'a> = CorrelationFunctionArgs<'a>;
    type Output = Vec<CorrelationFunctionResult>;

    fn compute<'a, FA: FrameAccess + Sync + 'a>(
        &self,
        frames: &[&'a FA],
        args: CorrelationFunctionArgs<'a>,
    ) -> Result<Vec<CorrelationFunctionResult>, ComputeError> {
        if frames.is_empty() {
            return Err(ComputeError::EmptyInput);
        }
        let nf = frames.len();
        if args.nlists.len() != nf || args.values_a.len() != nf || args.values_b.len() != nf {
            return Err(ComputeError::DimensionMismatch {
                expected: nf,
                got: args
                    .nlists
                    .len()
                    .min(args.values_a.len())
                    .min(args.values_b.len()),
                what: "CorrelationFunction frame-aligned inputs",
            });
        }
        #[cfg(feature = "rayon")]
        const PAR_THRESHOLD: usize = 2;

        #[cfg(feature = "rayon")]
        if nf >= PAR_THRESHOLD {
            use rayon::prelude::*;
            return (0..nf)
                .into_par_iter()
                .map(|k| self.one_frame(&args.nlists[k], &args.values_a[k], &args.values_b[k]))
                .collect();
        }

        let mut out = Vec::with_capacity(nf);
        for k in 0..nf {
            out.push(self.one_frame(&args.nlists[k], &args.values_a[k], &args.values_b[k])?);
        }
        Ok(out)
    }
}

/// Per-frame correlation result.
#[derive(Debug, Clone, Default)]
pub struct CorrelationFunctionResult {
    /// Bin edges (length `n_bins + 1`).
    pub bin_edges: Array1<F>,
    /// Bin centres (length `n_bins`).
    pub bin_centers: Array1<F>,
    /// Running pair count per bin (length `n_bins`).
    pub bin_counts: Array1<u64>,
    /// `⟨A · B⟩(r)` per bin (length `n_bins`); zero for empty bins.
    pub correlation: Array1<F>,
}

impl ComputeResult for CorrelationFunctionResult {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute::fixtures::nlist_from_frame;
    use molrs::core::Block;
    use molrs::core::Frame;
    use molrs::core::SimBox;
    use ndarray::{Array1 as A1, array};

    fn frame_with(positions: &[[F; 3]], box_len: F) -> Frame {
        let x = A1::from_iter(positions.iter().map(|p| p[0]));
        let y = A1::from_iter(positions.iter().map(|p| p[1]));
        let z = A1::from_iter(positions.iter().map(|p| p[2]));
        let mut block = Block::new();
        block.insert("x", x.into_dyn()).unwrap();
        block.insert("y", y.into_dyn()).unwrap();
        block.insert("z", z.into_dyn()).unwrap();
        let mut frame = Frame::new();
        frame.insert("atoms", block);
        frame.simbox =
            Some(SimBox::cube(box_len, array![0.0 as F, 0.0 as F, 0.0 as F], [false; 3]).unwrap());
        frame
    }

    #[test]
    fn constant_values_yield_constant_correlation() {
        // A = B = 1.0 for all particles → ⟨A·B⟩(r) = 1 in every populated bin.
        let positions = [
            [1.0_f64, 1.0, 1.0],
            [2.0, 1.0, 1.0],
            [3.0, 1.0, 1.0],
            [4.0, 1.0, 1.0],
        ];
        let frame = frame_with(&positions, 20.0);
        let nl = nlist_from_frame(&frame, 5.0);
        let vals = vec![1.0_f64; positions.len()];
        let cf = CorrelationFunction::new(10, 5.0, 0.0).unwrap();
        let r = &cf
            .compute(
                &[&frame],
                CorrelationFunctionArgs {
                    nlists: &[nl],
                    values_a: std::slice::from_ref(&vals),
                    values_b: std::slice::from_ref(&vals),
                },
            )
            .unwrap()[0];
        for b in 0..10 {
            if r.bin_counts[b] > 0 {
                assert!(
                    (r.correlation[b] - 1.0).abs() < 1e-12,
                    "bin {b}: got {}",
                    r.correlation[b]
                );
            }
        }
    }

    #[test]
    fn alternating_values_average_to_minus_one() {
        // A = B = [+1, -1, +1, -1]. Each NN pair is opposite-sign so
        // A_i · B_j = -1; bin 0 (the only populated one for unit spacing)
        // should average to -1.
        let positions = [
            [0.0_f64, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ];
        let frame = frame_with(&positions, 20.0);
        let nl = nlist_from_frame(&frame, 1.2);
        let vals = vec![1.0_f64, -1.0, 1.0, -1.0];
        let cf = CorrelationFunction::new(5, 1.5, 0.0).unwrap();
        let r = &cf
            .compute(
                &[&frame],
                CorrelationFunctionArgs {
                    nlists: &[nl],
                    values_a: std::slice::from_ref(&vals),
                    values_b: std::slice::from_ref(&vals),
                },
            )
            .unwrap()[0];
        // The first non-empty bin contains the unit-spacing NN pairs.
        let first = r.bin_counts.iter().position(|&c| c > 0).unwrap();
        assert!((r.correlation[first] + 1.0).abs() < 1e-12);
    }

    #[test]
    fn empty_bin_returns_zero() {
        let frame = frame_with(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], 20.0);
        let nl = nlist_from_frame(&frame, 1.5);
        let vals = vec![1.0_f64, 1.0];
        let cf = CorrelationFunction::new(10, 5.0, 0.0).unwrap();
        let r = &cf
            .compute(
                &[&frame],
                CorrelationFunctionArgs {
                    nlists: &[nl],
                    values_a: std::slice::from_ref(&vals),
                    values_b: std::slice::from_ref(&vals),
                },
            )
            .unwrap()[0];
        // Bins beyond 1.5 should all be empty.
        for b in 4..10 {
            assert_eq!(r.bin_counts[b], 0);
            assert_eq!(r.correlation[b], 0.0);
        }
    }

    #[test]
    fn invalid_params_error() {
        assert!(CorrelationFunction::new(0, 1.0, 0.0).is_err());
        assert!(CorrelationFunction::new(10, 1.0, 1.0).is_err());
        assert!(CorrelationFunction::new(10, 0.5, 1.0).is_err());
    }

    /// The `nf >= PAR_THRESHOLD` rayon branch must match the serial path
    /// exactly, in frame order (frame-aligned nlists / values indexing).
    #[test]
    fn parallel_matches_serial() {
        let positions = [
            [1.0_f64, 1.0, 1.0],
            [2.0, 1.0, 1.0],
            [3.0, 1.0, 1.0],
            [4.0, 1.0, 1.0],
        ];
        let frame = frame_with(&positions, 20.0);
        let nl = nlist_from_frame(&frame, 5.0);
        let vals = vec![1.0_f64; positions.len()];
        let cf = CorrelationFunction::new(10, 5.0, 0.0).unwrap();
        let solo = cf
            .compute(
                &[&frame],
                CorrelationFunctionArgs {
                    nlists: std::slice::from_ref(&nl),
                    values_a: std::slice::from_ref(&vals),
                    values_b: std::slice::from_ref(&vals),
                },
            )
            .unwrap();
        let nls = vec![nl.clone(), nl];
        let vs = vec![vals.clone(), vals];
        let par = cf
            .compute(
                &[&frame, &frame],
                CorrelationFunctionArgs {
                    nlists: &nls,
                    values_a: &vs,
                    values_b: &vs,
                },
            )
            .unwrap();
        assert_eq!(par.len(), 2);
        assert_eq!(par[0].correlation, solo[0].correlation);
        assert_eq!(par[0].bin_counts, solo[0].bin_counts);
        assert_eq!(par[1].correlation, solo[0].correlation);
    }
}
