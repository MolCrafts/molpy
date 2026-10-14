//! Streaming (frame-by-frame) velocity-ACF accumulation with bounded memory.

use std::collections::VecDeque;

use molrs::op::F;

use crate::compute::ComputeError;

/// Streaming DOF-averaged velocity-ACF accumulator (lag ≤ `resolution`).
///
/// Velocities are fed as one flat slice per frame (any fixed `n_dof ≥ 1`;
/// Atomiverse feeds blocked `vx|vy|vz`). Finalize returns the same
/// unnormalized, mean-subtracted, DOF-averaged ACF curve as the batch
/// [`Vacf`](super::Vacf) compute truncated at `min(resolution, n_frames − 1)`.
///
/// [`VacfAccumulator`] is the streaming counterpart of the batch
/// [`Vacf`](super::Vacf) compute (`velocity_acf`: per-DOF mean-subtract →
/// FFT-ACF → DOF-average). A streaming pass cannot subtract the global
/// per-DOF mean up front, so it accumulates the **raw** lagged products and
/// applies the exact algebraic correction at finalize:
///
/// ```text
/// Σ_τ (x_τ − μ)(x_{τ+k} − μ)
///   = Σ_τ x_τ x_{τ+k} − μ·(A_k + B_k) + (T − k)·μ²
/// A_k = Σ_{τ=0}^{T-1-k} x_τ = S − (sum of the last k samples)
/// B_k = Σ_{τ=k}^{T-1}   x_τ = S − (sum of the first k samples)
/// ```
///
/// so only the first `resolution` and last `resolution` frames are retained
/// (two rings), plus the per-DOF totals and the `resolution + 1` lag sums.
/// Memory is O(`n_dof · resolution`) — never O(trajectory). The result equals
/// the batch FFT path to FFT round-off (~1e-12 relative); a unit test pins the
/// equivalence.
#[derive(Debug, Clone)]
pub struct VacfAccumulator {
    resolution: usize,
    n_dof: usize,           // latched on the first frame; 0 = not yet latched
    acc: Vec<F>,            // acc[k] = Σ_t Σ_d v_d(t)·v_d(t−k), raw (no mean subtraction)
    dof_sum: Vec<F>,        // per-DOF running total S_d
    head: Vec<Vec<F>>,      // first ≤ resolution frames
    ring: VecDeque<Vec<F>>, // last ≤ resolution frames (most recent at back)
    n_frames: usize,
}

impl VacfAccumulator {
    /// New accumulator resolving lags `0..=resolution` (frames).
    ///
    /// # Errors
    /// [`ComputeError::OutOfRange`] when `resolution == 0` — a zero-lag-only
    /// ACF is a variance, not a correlation curve.
    pub fn new(resolution: usize) -> Result<Self, ComputeError> {
        if resolution == 0 {
            return Err(ComputeError::OutOfRange {
                field: "VacfAccumulator::resolution",
                value: resolution.to_string(),
            });
        }
        Ok(Self {
            resolution,
            n_dof: 0,
            acc: vec![0.0; resolution + 1],
            dof_sum: Vec::new(),
            head: Vec::new(),
            ring: VecDeque::new(),
            n_frames: 0,
        })
    }

    /// Max lag (frames) this accumulator resolves.
    pub fn resolution(&self) -> usize {
        self.resolution
    }

    /// Number of frames accumulated so far.
    pub fn n_frames(&self) -> usize {
        self.n_frames
    }

    /// Fold one frame of velocities (flat, fixed length across frames).
    ///
    /// The first frame latches `n_dof`; later frames must match it
    /// ([`ComputeError::DimensionMismatch`] otherwise). An empty frame is
    /// [`ComputeError::BadShape`].
    pub fn accumulate(&mut self, velocities: &[F]) -> Result<(), ComputeError> {
        if velocities.is_empty() {
            return Err(ComputeError::BadShape {
                expected: "a non-empty velocity frame".to_string(),
                got: "0".to_string(),
            });
        }
        if self.n_dof == 0 {
            self.n_dof = velocities.len();
            self.dof_sum = vec![0.0; self.n_dof];
        } else if velocities.len() != self.n_dof {
            return Err(ComputeError::DimensionMismatch {
                expected: self.n_dof,
                got: velocities.len(),
                what: "Vacf DOF count",
            });
        }

        // Lag 0 (self product) + lags 1..=K against the ring history.
        let mut s0 = 0.0;
        for &v in velocities {
            s0 += v * v;
        }
        self.acc[0] += s0;
        let ring_len = self.ring.len();
        for k in 1..=self.resolution.min(ring_len) {
            let past = &self.ring[ring_len - k];
            let mut sk = 0.0;
            for (a, b) in velocities.iter().zip(past.iter()) {
                sk += a * b;
            }
            self.acc[k] += sk;
        }
        for (s, v) in self.dof_sum.iter_mut().zip(velocities.iter()) {
            *s += v;
        }
        if self.head.len() < self.resolution {
            self.head.push(velocities.to_vec());
        }
        // Reuse the slot dropped off the front once the ring is saturated —
        // steady-state allocates zero velocity buffers per frame.
        if self.ring.len() == self.resolution {
            let mut buf = self.ring.pop_front().expect("ring full");
            buf.copy_from_slice(velocities);
            self.ring.push_back(buf);
        } else {
            self.ring.push_back(velocities.to_vec());
        }
        self.n_frames += 1;
        Ok(())
    }

    /// The mean-subtracted, DOF-averaged ACF curve for lags
    /// `0..=min(resolution, n_frames − 1)` — the batch
    /// [`Vacf`](super::Vacf) output.
    ///
    /// # Errors
    /// [`ComputeError::EmptyInput`] with fewer than two accumulated frames.
    pub fn finalize(&self) -> Result<Vec<F>, ComputeError> {
        let t = self.n_frames;
        if t < 2 {
            return Err(ComputeError::EmptyInput);
        }
        let max_lag = self.resolution.min(t - 1);
        let tf = t as F;
        let inv_n_dof = 1.0 / self.n_dof as F;

        // μ_d = S_d / T;  Σ_d μ_d² and Σ_d μ_d·S_d are lag-independent.
        let mu: Vec<F> = self.dof_sum.iter().map(|s| s / tf).collect();
        let sum_mu_sq: F = mu.iter().map(|m| m * m).sum();
        let sum_mu_s: F = mu.iter().zip(self.dof_sum.iter()).map(|(m, s)| m * s).sum();

        // Running per-lag correction terms: Σ_d μ_d·(tail_k)_d and
        // Σ_d μ_d·(head_k)_d, built incrementally in O(max_lag · n_dof).
        let mut mu_dot_tail: F = 0.0; // Σ_d μ_d · (sum of last k samples of DOF d)
        let mut mu_dot_head: F = 0.0; // Σ_d μ_d · (sum of first k samples of DOF d)

        let mut out = Vec::with_capacity(max_lag + 1);
        // k = 0: A_0 = B_0 = S_d. Divide by n_origins = T for unbiased C(0).
        out.push((self.acc[0] - 2.0 * sum_mu_s + tf * sum_mu_sq) * inv_n_dof / tf);
        for k in 1..=max_lag {
            let newest = &self.ring[self.ring.len() - k]; // sample x_{T-k}
            let oldest = &self.head[k - 1]; // sample x_{k-1}
            mu_dot_tail += mu.iter().zip(newest.iter()).map(|(m, v)| m * v).sum::<F>();
            mu_dot_head += mu.iter().zip(oldest.iter()).map(|(m, v)| m * v).sum::<F>();
            // A_k + B_k = 2·S − tail_k − head_k  (per DOF, folded with μ_d).
            let corr = 2.0 * sum_mu_s - mu_dot_tail - mu_dot_head;
            let n_origins = tf - k as F;
            out.push((self.acc[k] - corr + n_origins * sum_mu_sq) * inv_n_dof / n_origins);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute::LegacyCompute;
    use crate::compute::Vacf;
    use molrs::core::Frame;
    use ndarray::Array2;
    use rand::{RngExt, SeedableRng};

    fn no_frames() -> Vec<&'static Frame> {
        Vec::new()
    }

    fn rng_series(n: usize, cols: usize, seed: u64) -> Array2<f64> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let mut s = Array2::zeros((n, cols));
        for t in 0..n {
            for c in 0..cols {
                // Nonzero mean so the mean-subtraction correction is exercised.
                s[[t, c]] = 0.3 + rng.random_range(-1.0..1.0);
            }
        }
        s
    }

    #[test]
    fn streaming_matches_batch_fft_vacf() {
        let (n, dof, res) = (128, 9, 20);
        let v = rng_series(n, dof, 11);

        let batch = Vacf.compute(&no_frames(), (&v, 1.0, res)).unwrap();

        let mut acc = VacfAccumulator::new(res).unwrap();
        for t in 0..n {
            let frame: Vec<f64> = (0..dof).map(|d| v[[t, d]]).collect();
            acc.accumulate(&frame).unwrap();
        }
        let streamed = acc.finalize().unwrap();

        assert_eq!(streamed.len(), batch.acf.len());
        let scale = batch.acf[0].abs().max(1.0);
        for (k, &s) in streamed.iter().enumerate() {
            assert!(
                (s - batch.acf[k]).abs() / scale < 1e-12,
                "lag {k}: streamed {} vs batch {}",
                s,
                batch.acf[k]
            );
        }
    }

    #[test]
    fn short_trajectory_caps_lag_at_t_minus_one() {
        let (n, dof, res) = (6, 3, 20); // T − 1 < resolution
        let v = rng_series(n, dof, 5);

        let batch = Vacf.compute(&no_frames(), (&v, 1.0, res)).unwrap();
        let mut acc = VacfAccumulator::new(res).unwrap();
        for t in 0..n {
            let frame: Vec<f64> = (0..dof).map(|d| v[[t, d]]).collect();
            acc.accumulate(&frame).unwrap();
        }
        let streamed = acc.finalize().unwrap();
        assert_eq!(streamed.len(), n); // max_lag = T − 1
        let scale = batch.acf[0].abs().max(1.0);
        for (k, &s) in streamed.iter().enumerate() {
            assert!((s - batch.acf[k]).abs() / scale < 1e-12, "lag {k}");
        }
    }

    #[test]
    fn zero_resolution_and_dof_mismatch_are_errors() {
        assert!(matches!(
            VacfAccumulator::new(0).unwrap_err(),
            ComputeError::OutOfRange { .. }
        ));
        let mut acc = VacfAccumulator::new(4).unwrap();
        acc.accumulate(&[1.0, 2.0]).unwrap();
        assert!(matches!(
            acc.accumulate(&[1.0]).unwrap_err(),
            ComputeError::DimensionMismatch { .. }
        ));
        assert!(matches!(
            acc.finalize().unwrap_err(),
            ComputeError::EmptyInput
        ));
    }
}
