//! Streaming (frame-by-frame) MSD accumulation with bounded memory.

use std::collections::VecDeque;

use molrs::op::F;

use crate::compute::ComputeError;

/// Streaming MSD accumulator (Direct curve + lag-capped windowed sums).
///
/// Positions are fed as one flat slice per frame, blocked `x[0..n) y[..) z[..)`
/// (`n_dof = 3 · n_atoms`) — the raw-buffer convention of the FFI computes.
///
/// [`MsdAccumulator`] is the streaming counterpart of the batch [`Msd`](super::Msd)
/// compute. Feed one frame of flat positions at a time; it maintains
///
/// - the **Direct-mode** curve `Msd(t) = ⟨|r(t) − r(0)|²⟩_i` (frame 0 =
///   reference, one scalar per frame — exactly [`MsdMode::Direct`](super::MsdMode)),
/// - **windowed** per-lag sums `Σ_{i,τ} |r_i(τ+k) − r_i(τ)|²` for lags
///   `k ≤ window`, over a ring buffer of the last `window` frames — the same
///   estimator as [`MsdMode::Window`](super::MsdMode) truncated at `window`.
///
/// Memory is O(`window · n_dof`) for the ring plus O(n_frames) scalars for the
/// direct curve — never O(trajectory · n_dof). The windowed curve equals the
/// batch `Msd` `Window`-mode means for every lag `k ≤ window` (the batch path
/// computes all lags up to `n_frames − 1`; a streaming estimator must cap the
/// lag to bound the ring).
#[derive(Debug, Clone)]
pub struct MsdAccumulator {
    window: usize,
    n_dof: usize, // latched on the first frame; 0 = not yet latched
    reference: Vec<F>,
    direct: Vec<F>,
    ring: VecDeque<Vec<F>>,
    win_sum: Vec<F>, // win_sum[k] = Σ_{i,τ} |r_i(τ+k) − r_i(τ)|², k in [1, window]
    n_frames: usize,
}

impl MsdAccumulator {
    /// New accumulator keeping windowed-MSD sums up to lag `window` (frames).
    ///
    /// `window = 0` disables the windowed estimator (Direct curve only,
    /// O(n_dof) state).
    pub fn new(window: usize) -> Self {
        Self {
            window,
            n_dof: 0,
            reference: Vec::new(),
            direct: Vec::new(),
            ring: VecDeque::new(),
            win_sum: vec![0.0; window + 1],
            n_frames: 0,
        }
    }

    /// Max windowed lag (frames) this accumulator resolves.
    pub fn window(&self) -> usize {
        self.window
    }

    /// Number of frames accumulated so far.
    pub fn n_frames(&self) -> usize {
        self.n_frames
    }

    /// Fold one frame of positions (blocked `x|y|z`, `len = 3 · n_atoms`).
    ///
    /// The first frame latches `n_dof` and becomes the Direct-mode reference;
    /// later frames must match it ([`ComputeError::DimensionMismatch`]
    /// otherwise). `len` must be a positive multiple of 3
    /// ([`ComputeError::BadShape`]).
    pub fn accumulate(&mut self, positions: &[F]) -> Result<(), ComputeError> {
        if positions.is_empty() || !positions.len().is_multiple_of(3) {
            return Err(ComputeError::BadShape {
                expected: "a positive multiple of 3 (blocked x|y|z)".to_string(),
                got: positions.len().to_string(),
            });
        }
        if self.n_dof == 0 {
            self.n_dof = positions.len();
            self.reference = positions.to_vec();
        } else if positions.len() != self.n_dof {
            return Err(ComputeError::DimensionMismatch {
                expected: self.n_dof,
                got: positions.len(),
                what: "Msd DOF count",
            });
        }

        let n_atoms = self.n_dof / 3;

        // Direct mode vs the frame-0 reference — per-particle summation order
        // matches `msd_vs_reference` (dx² + dy² + dz² per particle).
        self.direct
            .push(sum_sq_disp(positions, &self.reference, n_atoms) / n_atoms as F);

        // Windowed sums against the last ≤ window frames.
        for k in 1..=self.window.min(self.ring.len()) {
            let past = &self.ring[self.ring.len() - k];
            self.win_sum[k] += sum_sq_disp(positions, past, n_atoms);
        }
        if self.window > 0 {
            self.ring.push_back(positions.to_vec());
            if self.ring.len() > self.window {
                self.ring.pop_front();
            }
        }
        self.n_frames += 1;
        Ok(())
    }

    /// Direct-mode curve `Msd(t)` (one value per accumulated frame, index 0 = 0).
    pub fn direct_curve(&self) -> &[F] {
        &self.direct
    }

    /// Windowed MSD per lag, `k in [0, min(window, n_frames − 1)]`:
    /// `Msd(k) = Σ_{i,τ} |r_i(τ+k) − r_i(τ)|² / ((n_frames − k) · n_atoms)` —
    /// the `Msd` `Window`-mode estimator truncated at `window`. Empty before the
    /// second frame.
    pub fn windowed_msd(&self) -> Vec<F> {
        if self.n_frames < 2 || self.n_dof == 0 {
            return Vec::new();
        }
        let n_atoms = (self.n_dof / 3) as F;
        let max_lag = self.window.min(self.n_frames - 1);
        let mut out = Vec::with_capacity(max_lag + 1);
        out.push(0.0);
        for k in 1..=max_lag {
            out.push(self.win_sum[k] / ((self.n_frames - k) as F * n_atoms));
        }
        out
    }
}

/// Sum over particles of `|a_i − b_i|²` for two blocked `x|y|z` frames.
fn sum_sq_disp(a: &[F], b: &[F], n_atoms: usize) -> F {
    let mut total: F = 0.0;
    for i in 0..n_atoms {
        let dx = a[i] - b[i];
        let dy = a[n_atoms + i] - b[n_atoms + i];
        let dz = a[2 * n_atoms + i] - b[2 * n_atoms + i];
        total += dx * dx + dy * dy + dz * dz;
    }
    total
}

#[cfg(test)]
mod tests {
    use super::super::Msd;
    use super::*;
    use crate::compute::LegacyCompute;
    use crate::compute::MsdMode;
    use molrs::core::Block;
    use molrs::core::Frame;
    use ndarray::Array1 as A1;

    fn make_frame(x: &[f64], y: &[f64], z: &[f64]) -> Frame {
        let mut block = Block::new();
        block
            .insert("x", A1::from_vec(x.to_vec()).into_dyn())
            .unwrap();
        block
            .insert("y", A1::from_vec(y.to_vec()).into_dyn())
            .unwrap();
        block
            .insert("z", A1::from_vec(z.to_vec()).into_dyn())
            .unwrap();
        let mut frame = Frame::new();
        frame.insert("atoms", block);
        frame
    }

    /// Deterministic pseudo-random walk: T frames × n atoms.
    fn walk(t_frames: usize, n_atoms: usize, seed: u64) -> Vec<Vec<f64>> {
        use rand::SeedableRng;
        use rand::rngs::StdRng;
        let mut rng = StdRng::seed_from_u64(seed);
        use rand::RngExt;
        let mut cur = vec![0.0_f64; 3 * n_atoms];
        let mut frames = Vec::with_capacity(t_frames);
        for _ in 0..t_frames {
            for v in cur.iter_mut() {
                *v += rng.random_range(-0.5..0.5);
            }
            frames.push(cur.clone());
        }
        frames
    }

    fn to_molrs_frames(flat: &[Vec<f64>], n_atoms: usize) -> Vec<Frame> {
        flat.iter()
            .map(|f| {
                make_frame(
                    &f[0..n_atoms],
                    &f[n_atoms..2 * n_atoms],
                    &f[2 * n_atoms..3 * n_atoms],
                )
            })
            .collect()
    }

    #[test]
    fn direct_curve_matches_batch_direct_mode() {
        let (t, n) = (16, 5);
        let flat = walk(t, n, 42);
        let frames = to_molrs_frames(&flat, n);
        let refs: Vec<&Frame> = frames.iter().collect();
        let batch = Msd::new().compute(&refs, ()).unwrap();

        let mut acc = MsdAccumulator::new(0);
        for f in &flat {
            acc.accumulate(f).unwrap();
        }
        assert_eq!(acc.direct_curve().len(), t);
        for k in 0..t {
            assert!(
                (acc.direct_curve()[k] - batch.per_frame[k].mean).abs() < 1e-12,
                "direct lag {k}"
            );
        }
    }

    #[test]
    fn windowed_msd_matches_batch_window_mode_up_to_window() {
        let (t, n, w) = (24, 4, 10);
        let flat = walk(t, n, 7);
        let frames = to_molrs_frames(&flat, n);
        let refs: Vec<&Frame> = frames.iter().collect();
        let batch = Msd::with_mode(MsdMode::Window).compute(&refs, ()).unwrap();

        let mut acc = MsdAccumulator::new(w);
        for f in &flat {
            acc.accumulate(f).unwrap();
        }
        let streamed = acc.windowed_msd();
        assert_eq!(streamed.len(), w + 1);
        for (k, &s) in streamed.iter().enumerate() {
            assert!(
                (s - batch.per_frame[k].mean).abs() < 1e-9,
                "windowed lag {k}: streamed {s} vs batch {}",
                batch.per_frame[k].mean
            );
        }
    }

    #[test]
    fn window_shorter_than_trajectory_bounds_ring() {
        let (t, n, w) = (50, 3, 4);
        let flat = walk(t, n, 3);
        let mut acc = MsdAccumulator::new(w);
        for f in &flat {
            acc.accumulate(f).unwrap();
        }
        assert_eq!(acc.windowed_msd().len(), w + 1);
        assert_eq!(acc.n_frames(), t);
    }

    #[test]
    fn dof_mismatch_is_error() {
        let mut acc = MsdAccumulator::new(2);
        acc.accumulate(&[0.0; 6]).unwrap();
        assert!(matches!(
            acc.accumulate(&[0.0; 9]).unwrap_err(),
            ComputeError::DimensionMismatch { .. }
        ));
        assert!(matches!(
            acc.accumulate(&[0.0; 4]).unwrap_err(),
            ComputeError::BadShape { .. }
        ));
    }
}
