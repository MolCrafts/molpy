//! Random variates for the stochastic paths (thermostats, velocity draws,
//! synthetic test data): one implementation of each draw, generic over the
//! caller's seeded RNG.

use rand::RngExt;

use crate::op::F;

/// One standard normal deviate, `N(0, 1)`, by the Box–Muller transform.
///
/// Consumes two uniforms per call and returns the cosine branch. The first
/// uniform is floored at the smallest positive `f64` so `ln` never sees zero.
pub fn standard_normal<R: RngExt + ?Sized>(rng: &mut R) -> F {
    let u1 = rng.random::<F>().max(F::MIN_POSITIVE);
    let u2 = rng.random::<F>();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn the_draws_have_zero_mean_and_unit_variance() {
        let mut rng = StdRng::seed_from_u64(7);
        let n = 200_000;
        let xs: Vec<F> = (0..n).map(|_| standard_normal(&mut rng)).collect();
        let mean = xs.iter().sum::<F>() / n as F;
        let var = xs.iter().map(|x| (x - mean) * (x - mean)).sum::<F>() / n as F;
        assert!(mean.abs() < 0.01, "mean {mean}");
        assert!((var - 1.0).abs() < 0.01, "variance {var}");
    }
}
