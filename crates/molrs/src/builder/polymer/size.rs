//! How large a stochastic object grows: a [`Distribution`] of target sizes
//! and the [`Measure`] a target is counted in.
//!
//! Growth adds fragments until the measured size reaches the drawn target (the
//! generative-BigSMILES rule), so a distribution is a distribution of
//! *targets*: the realised size overshoots it by at most one fragment.

use rand::{Rng, RngExt};
use rand_distr::{Distribution as _, Gamma, LogNormal, Normal, Poisson};

use super::PolymerError;
use crate::op::F;

/// The six chain-size distributions of generative BigSMILES.
///
/// Each names its parameters as the notation writes them
/// (`|schulz_zimm(Mw, Mn)|`, `|log_normal(Mn, D)|`, …). What a drawn value
/// counts — repeat units, heavy-atom mass or mass — is the [`Measure`] of the
/// [`ChainSize`] it sits in, not a property of the shape; masses are in g/mol.
///
/// References: Schneider, L.; de Pablo, J. J. *Generative BigSMILES: an
/// extension for polymer informatics, computer simulations & ML/AI*, Digital
/// Discovery **3**, 51–61 (2024), DOI: 10.1039/D3DD00147D.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Distribution {
    /// `flory_schulz(a)`: `P(k) = a² k (1 − a)^(k − 1)` over `k = 1, 2, …`,
    /// `0 < a < 1` (the most probable distribution of step growth, `a` the
    /// fraction of unreacted ends).
    FlorySchulz { a: F },
    /// `schulz_zimm(Mw, Mn)`: a gamma distribution of shape
    /// `z = Mn / (Mw − Mn)` and mean `Mn`, so its weight average is `Mw`;
    /// `Mw > Mn > 0`.
    SchulzZimm { mw: F, mn: F },
    /// `gauss(mu, sigma)`: a normal distribution, `sigma ≥ 0` (`0` is the
    /// constant `mu`).
    Gauss { mu: F, sigma: F },
    /// `uniform(low, high)`: uniform over `[low, high]`, `0 ≤ low ≤ high`.
    Uniform { low: F, high: F },
    /// `log_normal(Mn, D)`: a log-normal distribution of number average `Mn`
    /// and dispersity `D = Mw / Mn`, i.e. `ln M ~ N(ln Mn − σ²/2, σ²)` with
    /// `σ² = ln D`; `Mn > 0`, `D > 1`.
    LogNormal { mn: F, d: F },
    /// `poisson(N)`: a Poisson distribution of mean `N > 0`.
    Poisson { n: F },
}

impl Distribution {
    /// Check the parameters against the ranges each variant documents, and
    /// that the sampler they parameterise can be built — a mean or shape the
    /// sampler cannot represent (`poisson(1e30)`) is refused here, not at the
    /// first draw.
    ///
    /// # Errors
    ///
    /// [`PolymerError::InvalidParameter`] naming the distribution, for a
    /// non-finite or out-of-range parameter.
    pub fn validate(&self) -> Result<(), PolymerError> {
        self.sampler().map(|_| ())
    }

    /// The sampler these parameters state.
    fn sampler(&self) -> Result<Sampler, PolymerError> {
        let invalid = || {
            PolymerError::InvalidParameter(format!(
                "{self:?} is outside its documented parameter range"
            ))
        };
        let built = match *self {
            Distribution::FlorySchulz { a } => {
                (a.is_finite() && a > 0.0 && a < 1.0).then_some(Sampler::FlorySchulz(a))
            }
            Distribution::SchulzZimm { mw, mn } if mn > 0.0 && mw > mn => {
                let z = mn / (mw - mn);
                Gamma::new(z, mn / z).ok().map(Sampler::Gamma)
            }
            Distribution::Gauss { mu, sigma } if sigma >= 0.0 => {
                Normal::new(mu, sigma).ok().map(Sampler::Normal)
            }
            Distribution::Uniform { low, high } => {
                (low.is_finite() && high.is_finite() && low >= 0.0 && high >= low)
                    .then_some(Sampler::Uniform(low, high))
            }
            Distribution::LogNormal { mn, d } if mn > 0.0 && d > 1.0 => {
                let s2 = d.ln();
                LogNormal::new(mn.ln() - s2 / 2.0, s2.sqrt())
                    .ok()
                    .map(Sampler::LogNormal)
            }
            Distribution::Poisson { n } if n > 0.0 => Poisson::new(n).ok().map(Sampler::Poisson),
            _ => None,
        };
        // The constructors accept an infinite mean or scale; a target is finite.
        built
            .filter(|_| self.parameters().iter().all(|x| x.is_finite()))
            .ok_or_else(invalid)
    }

    fn parameters(&self) -> [F; 2] {
        match *self {
            Distribution::FlorySchulz { a } => [a, a],
            Distribution::SchulzZimm { mw, mn } => [mw, mn],
            Distribution::Gauss { mu, sigma } => [mu, sigma],
            Distribution::Uniform { low, high } => [low, high],
            Distribution::LogNormal { mn, d } => [mn, d],
            Distribution::Poisson { n } => [n, n],
        }
    }
}

/// A validated distribution, ready to draw from.
#[derive(Debug, Clone, Copy)]
enum Sampler {
    FlorySchulz(F),
    Gamma(Gamma<F>),
    Normal(Normal<F>),
    Uniform(F, F),
    LogNormal(LogNormal<F>),
    Poisson(Poisson<F>),
}

impl Sampler {
    /// Draw one value; negative draws (a wide `gauss`) are clipped to `0`.
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> F {
        let x = match self {
            // The number of trials to the second success, less one, is
            // distributed as a² k (1 − a)^(k − 1).
            Sampler::FlorySchulz(a) => (geometric(*a, rng) + geometric(*a, rng) - 1) as F,
            Sampler::Gamma(d) => d.sample(rng),
            Sampler::Normal(d) => d.sample(rng),
            Sampler::Uniform(low, high) if high > low => rng.random_range(*low..=*high),
            Sampler::Uniform(low, _) => *low,
            Sampler::LogNormal(d) => d.sample(rng),
            Sampler::Poisson(d) => d.sample(rng),
        };
        x.max(0.0)
    }
}

/// The number of Bernoulli(`p`) trials up to and including the first
/// success, by inversion: `⌈ln U / ln(1 − p)⌉` for `U ∈ (0, 1]`.
fn geometric<R: Rng + ?Sized>(p: F, rng: &mut R) -> u64 {
    let u: F = 1.0 - rng.random::<F>(); // (0, 1]
    let k = (u.ln() / (1.0 - p).ln()).ceil();
    // `u == 1` gives 0 trials; a success needs at least one.
    (k as u64).max(1)
}

/// What a drawn size target counts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Measure {
    /// The number of repeat units added (end groups do not count).
    RepeatUnits,
    /// The summed atomic mass of every non-hydrogen atom, g/mol — the
    /// measure generative BigSMILES writes its distributions and system
    /// sizes in.
    HeavyAtomMass,
    /// The summed atomic mass of every atom, hydrogens included, g/mol.
    Mass,
}

/// A distribution of size targets counted in one measure.
///
/// A target counted in [`Measure::RepeatUnits`] is a whole number: a
/// `uniform(low, high)` draws the integers of `[low, high]` with equal
/// probability, and every other continuous draw is rounded to the nearest
/// integer. Whatever the target, an object grows at least one repeat unit.
#[derive(Debug, Clone, Copy)]
pub struct ChainSize {
    distribution: Distribution,
    measure: Measure,
    sampler: Sampler,
}

impl PartialEq for ChainSize {
    fn eq(&self, other: &Self) -> bool {
        self.distribution == other.distribution && self.measure == other.measure
    }
}

impl ChainSize {
    /// Targets drawn from `distribution`, counted in `measure`.
    ///
    /// # Errors
    ///
    /// [`PolymerError::InvalidParameter`] for parameters outside the
    /// distribution's range ([`Distribution::validate`]), or a `uniform`
    /// counted in repeat units whose range holds no integer.
    pub fn new(distribution: Distribution, measure: Measure) -> Result<Self, PolymerError> {
        let mut sampler = distribution.sampler()?;
        if let (Measure::RepeatUnits, Sampler::Uniform(low, high)) = (measure, sampler) {
            let (lo, hi) = (low.ceil(), high.floor());
            if lo > hi {
                return Err(PolymerError::InvalidParameter(format!(
                    "{distribution:?} holds no whole number of repeat units"
                )));
            }
            sampler = Sampler::Uniform(lo, hi);
        }
        Ok(Self {
            distribution,
            measure,
            sampler,
        })
    }

    /// The distribution targets are drawn from.
    pub fn distribution(&self) -> Distribution {
        self.distribution
    }

    /// What a target counts.
    pub fn measure(&self) -> Measure {
        self.measure
    }

    /// Draw one target.
    pub fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> F {
        match (self.measure, self.sampler) {
            (Measure::RepeatUnits, Sampler::Uniform(lo, hi)) => {
                rng.random_range(lo as u64..=hi as u64) as F
            }
            (Measure::RepeatUnits, sampler) => sampler.sample(rng).round(),
            (_, sampler) => sampler.sample(rng),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    const N: usize = 40_000;

    fn moments(d: Distribution) -> (F, F) {
        let mut rng = StdRng::seed_from_u64(7);
        let sampler = d.sampler().unwrap();
        let xs: Vec<F> = (0..N).map(|_| sampler.sample(&mut rng)).collect();
        let mean = xs.iter().sum::<F>() / N as F;
        let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<F>() / N as F;
        (mean, var)
    }

    fn draw(d: Distribution, rng: &mut StdRng) -> F {
        d.sampler().unwrap().sample(rng)
    }

    fn close(actual: F, expected: F, rel: F) {
        assert!(
            (actual - expected).abs() <= rel * expected.abs(),
            "{actual} is not within {rel} of {expected}"
        );
    }

    #[test]
    fn flory_schulz_has_mean_two_minus_a_over_a() {
        // E[k] = (2 − a) / a for P(k) = a² k (1 − a)^(k − 1).
        let a = 0.1;
        let (mean, _) = moments(Distribution::FlorySchulz { a });
        close(mean, (2.0 - a) / a, 0.02);
    }

    #[test]
    fn flory_schulz_draws_are_positive_integers() {
        let mut rng = StdRng::seed_from_u64(1);
        for _ in 0..1000 {
            let k = draw(Distribution::FlorySchulz { a: 0.9 }, &mut rng);
            assert!(k >= 1.0 && k.fract() == 0.0);
        }
    }

    #[test]
    fn schulz_zimm_has_number_and_weight_averages_mn_and_mw() {
        // Mw = E[M²] / E[M] = Mn + Var / Mn.
        let (mw, mn) = (1500.0, 1000.0);
        let (mean, var) = moments(Distribution::SchulzZimm { mw, mn });
        close(mean, mn, 0.02);
        close(mean + var / mean, mw, 0.03);
    }

    #[test]
    fn log_normal_has_number_average_mn_and_dispersity_d() {
        let (mn, d) = (2000.0, 1.2);
        let (mean, var) = moments(Distribution::LogNormal { mn, d });
        close(mean, mn, 0.02);
        close((mean + var / mean) / mean, d, 0.02);
    }

    #[test]
    fn gauss_poisson_and_uniform_have_their_means() {
        close(
            moments(Distribution::Gauss {
                mu: 500.0,
                sigma: 50.0,
            })
            .0,
            500.0,
            0.01,
        );
        close(moments(Distribution::Poisson { n: 30.0 }).0, 30.0, 0.02);
        close(
            moments(Distribution::Uniform {
                low: 100.0,
                high: 300.0,
            })
            .0,
            200.0,
            0.01,
        );
    }

    #[test]
    fn zero_width_distributions_are_constant() {
        let mut rng = StdRng::seed_from_u64(3);
        assert_eq!(
            draw(
                Distribution::Gauss {
                    mu: 42.0,
                    sigma: 0.0
                },
                &mut rng
            ),
            42.0
        );
        assert_eq!(
            draw(
                Distribution::Uniform {
                    low: 5.0,
                    high: 5.0
                },
                &mut rng
            ),
            5.0
        );
    }

    #[test]
    fn negative_gauss_draws_clip_to_zero() {
        let mut rng = StdRng::seed_from_u64(3);
        for _ in 0..100 {
            assert!(
                draw(
                    Distribution::Gauss {
                        mu: -10.0,
                        sigma: 1.0
                    },
                    &mut rng
                ) >= 0.0
            );
        }
    }

    #[test]
    fn out_of_range_parameters_are_refused() {
        for bad in [
            Distribution::FlorySchulz { a: 1.0 },
            Distribution::FlorySchulz { a: 0.0 },
            Distribution::SchulzZimm {
                mw: 900.0,
                mn: 1000.0,
            },
            Distribution::Gauss {
                mu: 1.0,
                sigma: -1.0,
            },
            Distribution::Uniform {
                low: 3.0,
                high: 2.0,
            },
            Distribution::LogNormal { mn: 1000.0, d: 1.0 },
            Distribution::Poisson { n: 0.0 },
            Distribution::Poisson { n: F::NAN },
            // Valid by range, but no sampler represents them.
            Distribution::Poisson { n: 1e30 },
            Distribution::Gauss {
                mu: F::INFINITY,
                sigma: 1.0,
            },
            Distribution::SchulzZimm {
                mw: F::INFINITY,
                mn: 1.0,
            },
        ] {
            assert!(
                matches!(
                    ChainSize::new(bad, Measure::HeavyAtomMass),
                    Err(PolymerError::InvalidParameter(_))
                ),
                "{bad:?} must be refused"
            );
        }
    }

    #[test]
    fn equal_seeds_draw_equal_targets() {
        let size = ChainSize::new(Distribution::Poisson { n: 12.0 }, Measure::RepeatUnits).unwrap();
        let draw = |seed| {
            let mut rng = StdRng::seed_from_u64(seed);
            (0..10).map(|_| size.sample(&mut rng)).collect::<Vec<_>>()
        };
        assert_eq!(draw(5), draw(5));
    }

    #[test]
    fn uniform_repeat_units_draws_each_integer_of_its_range_equally() {
        let size = ChainSize::new(
            Distribution::Uniform {
                low: 2.0,
                high: 5.0,
            },
            Measure::RepeatUnits,
        )
        .unwrap();
        let mut rng = StdRng::seed_from_u64(11);
        let mut counts = [0usize; 6];
        for _ in 0..N {
            let k = size.sample(&mut rng);
            assert_eq!(k.fract(), 0.0);
            counts[k as usize] += 1;
        }
        assert_eq!(counts[0] + counts[1], 0);
        for count in &counts[2..=5] {
            close(*count as F, N as F / 4.0, 0.05);
        }
    }

    #[test]
    fn repeat_unit_targets_are_rounded_not_raised() {
        // Rounding keeps the mean; rounding up would add one half.
        let size = ChainSize::new(
            Distribution::Gauss {
                mu: 20.0,
                sigma: 3.0,
            },
            Measure::RepeatUnits,
        )
        .unwrap();
        let mut rng = StdRng::seed_from_u64(5);
        let xs: Vec<F> = (0..N).map(|_| size.sample(&mut rng)).collect();
        assert!(xs.iter().all(|x| x.fract() == 0.0));
        close(xs.iter().sum::<F>() / N as F, 20.0, 0.005);
    }

    #[test]
    fn uniform_repeat_units_without_an_integer_is_refused() {
        let d = Distribution::Uniform {
            low: 2.2,
            high: 2.8,
        };
        assert!(ChainSize::new(d, Measure::RepeatUnits).is_err());
        assert!(ChainSize::new(d, Measure::Mass).is_ok());
    }
}
