//! Green–Kubo diffusion raw compute — the velocity-ACF route to D.

use molrs::core::FrameAccess;

use super::vacf::{VacfArgs, VacfResult, velocity_acf};
use crate::compute::ComputeError;
use crate::compute::LegacyCompute;

/// Named workflow over the unbiased [`Vacf`](super::Vacf) curve for Green–Kubo
/// self-diffusion. Same math as `Vacf` — not a second estimator.
///
/// Fit step (caller): \(D = \frac{1}{d}\int_0^{\tau_{\max}} C(\tau)\,\mathrm{d}\tau\)
/// via [`CumulativeTrapezoid`](crate::compute::CumulativeTrapezoid)
/// then scale by `1/d` (`d = 3` in 3D).
#[derive(Debug, Clone, Copy, Default)]
pub struct GreenKuboDiffusion;

impl LegacyCompute for GreenKuboDiffusion {
    type Args<'a> = VacfArgs<'a>;
    type Output = VacfResult;

    fn compute<'a, FA: FrameAccess + Sync + 'a>(
        &self,
        _frames: &[&'a FA],
        args: Self::Args<'a>,
    ) -> Result<Self::Output, ComputeError> {
        let (velocities, dt, resolution) = args;
        velocity_acf(velocities, dt, resolution)
    }
}

#[cfg(test)]
mod tests {
    use super::super::vacf::Vacf;
    use super::*;
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
                s[[t, c]] = rng.random_range(-1.0..1.0);
            }
        }
        s
    }

    #[test]
    fn green_kubo_diffusion_equals_vacf() {
        let n = 64;
        let dt = 1.0;
        let v = rng_series(n, 3, 11);
        let a = Vacf.compute(&no_frames(), (&v, dt, 20)).unwrap();
        let b = GreenKuboDiffusion
            .compute(&no_frames(), (&v, dt, 20))
            .unwrap();
        assert_eq!(a.acf, b.acf);
        assert_eq!(a.lag_times, b.lag_times);
    }
}
