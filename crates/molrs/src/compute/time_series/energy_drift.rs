//! [`EnergyDrift`] — how fast a conserved energy wanders, per atom.

use molrs::op::F;

use crate::compute::{Accumulate, ComputeError, Stream};

/// The drift of a conserved quantity (NVE total energy) over a run.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct EnergyDriftResult {
    /// The ordinary-least-squares slope of `E(t)` divided by the atom count:
    /// energy per atom per time unit.
    pub drift: F,
    /// The rms of `E(t)` about that fitted line (population form, over `n`
    /// samples), divided by the atom count:
    /// the fluctuation per atom with the drift taken out. It is read to about
    /// `√ε` of the spread of `E`, far below any thermal fluctuation.
    pub rms: F,
    /// Samples pushed.
    pub n_samples: usize,
    /// `t_last − t_first`.
    pub duration: F,
}

/// One sample of the conserved energy.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EnergyDriftSample {
    /// The sample's time.
    pub time: F,
    /// The conserved energy at `time` (the total, not per atom).
    pub energy: F,
}

/// Energy drift per atom: the slope of a least-squares line through the
/// conserved energy `E(t)`, divided by `N`, with the rms about the line.
///
/// This is the per-atom drift GROMACS reports ("conserved energy drift"),
/// fitted rather than taken from the endpoints, and the one an NVE gate
/// should bound. Normalising by `|E₀|` instead says
/// nothing for a potential whose zero is arbitrary: a MatPES total near
/// −5 eV/atom lets a relative gate of 1e-3 pass 5 meV/atom.
///
/// The fit is accumulated in one pass with centred co-moments (Welford), so a
/// large energy offset over a long run does not cancel away the slope.
///
/// ```
/// use molrs::compute::{Accumulate, EnergyDrift, EnergyDriftSample, Stream};
///
/// // E = −100 + 0.002 t over four samples of a 4-atom system.
/// let mut drift = EnergyDrift::new(4)?.accumulator();
/// for t in 0..4 {
///     let time = t as f64;
///     drift.push(EnergyDriftSample { time, energy: -100.0 + 0.002 * time })?;
/// }
/// let out = drift.finish()?;
/// assert!((out.drift / 5e-4 - 1.0).abs() < 1e-10);
/// # Ok::<(), molrs::compute::ComputeError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EnergyDrift {
    n_atoms: usize,
}

impl EnergyDrift {
    /// A drift per atom of an `n_atoms` system.
    ///
    /// # Errors
    ///
    /// [`ComputeError::OutOfRange`] for `n_atoms == 0`.
    pub fn new(n_atoms: usize) -> Result<Self, ComputeError> {
        if n_atoms == 0 {
            return Err(ComputeError::OutOfRange {
                field: "n_atoms",
                value: "0".into(),
            });
        }
        Ok(Self { n_atoms })
    }

    /// The atom count the drift is divided by.
    pub fn n_atoms(&self) -> usize {
        self.n_atoms
    }
}

impl Stream for EnergyDrift {
    type Accumulator = EnergyDriftAccumulator;

    fn accumulator(&self) -> EnergyDriftAccumulator {
        EnergyDriftAccumulator {
            n_atoms: self.n_atoms,
            n: 0,
            first_time: 0.0,
            last_time: 0.0,
            mean_t: 0.0,
            mean_e: 0.0,
            m2_t: 0.0,
            m2_e: 0.0,
            c_te: 0.0,
        }
    }
}

/// The running fit of an [`EnergyDrift`].
#[derive(Debug, Clone)]
pub struct EnergyDriftAccumulator {
    n_atoms: usize,
    n: usize,
    first_time: F,
    last_time: F,
    mean_t: F,
    mean_e: F,
    /// `Σ (t − t̄)²`.
    m2_t: F,
    /// `Σ (E − Ē)²`.
    m2_e: F,
    /// `Σ (t − t̄)(E − Ē)`.
    c_te: F,
}

impl Accumulate for EnergyDriftAccumulator {
    type Item<'a> = EnergyDriftSample;
    type Output = EnergyDriftResult;

    /// # Errors
    ///
    /// [`ComputeError::NonFinite`] for a non-finite time or energy, and
    /// [`ComputeError::OutOfRange`] for a time not after the previous one.
    fn push(&mut self, sample: EnergyDriftSample) -> Result<(), ComputeError> {
        if !(sample.time.is_finite() && sample.energy.is_finite()) {
            return Err(ComputeError::NonFinite {
                where_: "energy drift sample",
                index: self.n,
            });
        }
        if self.n > 0 && sample.time <= self.last_time {
            return Err(ComputeError::OutOfRange {
                field: "time",
                value: format!("{} after {}", sample.time, self.last_time),
            });
        }
        if self.n == 0 {
            self.first_time = sample.time;
        }
        self.last_time = sample.time;
        self.n += 1;
        let n = self.n as F;
        let dt = sample.time - self.mean_t;
        let de = sample.energy - self.mean_e;
        self.mean_t += dt / n;
        self.mean_e += de / n;
        self.m2_t += dt * (sample.time - self.mean_t);
        self.m2_e += de * (sample.energy - self.mean_e);
        self.c_te += dt * (sample.energy - self.mean_e);
        Ok(())
    }

    /// # Errors
    ///
    /// [`ComputeError::EmptyInput`] before two samples: no line is defined.
    fn finish(&self) -> Result<EnergyDriftResult, ComputeError> {
        if self.n < 2 {
            return Err(ComputeError::EmptyInput);
        }
        let slope = self.c_te / self.m2_t;
        let residual = (self.m2_e - slope * self.c_te).max(0.0) / self.n as F;
        let n_atoms = self.n_atoms as F;
        Ok(EnergyDriftResult {
            drift: slope / n_atoms,
            rms: residual.sqrt() / n_atoms,
            n_samples: self.n,
            duration: self.last_time - self.first_time,
        })
    }
}

crate::compute::spec::compute_spec! {
    EnergyDrift as "EnergyDrift", Frames;
    params |s| {
        n_atoms: Int;
    }
    inputs {
        time: Scalar;
        energy: Scalar;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fit(n_atoms: usize, samples: &[(F, F)]) -> EnergyDriftResult {
        EnergyDrift::new(n_atoms)
            .unwrap()
            .accumulator()
            .fold(
                samples
                    .iter()
                    .map(|&(time, energy)| EnergyDriftSample { time, energy }),
            )
            .unwrap()
    }

    /// `(δ, −δ, −δ, δ)` at t = 0…3: slope 0, rms δ.
    #[test]
    fn a_symmetric_wobble_has_no_drift_and_rms_delta() {
        let d = 0.25;
        let out = fit(1, &[(0.0, d), (1.0, -d), (2.0, -d), (3.0, d)]);
        assert!(out.drift.abs() < 1e-15, "{}", out.drift);
        assert!((out.rms - d).abs() < 1e-15, "{}", out.rms);
        assert_eq!(out.n_samples, 4);
        assert_eq!(out.duration, 3.0);
    }

    /// `E = −100 + 0.002 t`, N = 4: 5e-4 per atom per unit time, no rms.
    #[test]
    fn a_line_gives_its_slope_per_atom() {
        let samples: Vec<(F, F)> = (0..4).map(|t| (t as F, -100.0 + 0.002 * t as F)).collect();
        let out = fit(4, &samples);
        assert!((out.drift / 5e-4 - 1.0).abs() < 1e-10, "{}", out.drift);
        // The residual is a difference of two co-moments: its floor is
        // √ε times the spread of E, here 1e-10.
        assert!(out.rms < 1e-8, "{}", out.rms);
    }

    /// A large offset over a long run: −2.2e4 + 1e-6·t over 10⁵ samples is
    /// still read as 1e-6, which the raw-sum form (Σt·E − Σt·ΣE/n) loses.
    #[test]
    fn an_offset_does_not_cancel_the_slope() {
        let samples: Vec<(F, F)> = (0..100_000)
            .map(|t| (t as F, -2.2e4 + 1e-6 * t as F))
            .collect();
        let out = fit(1, &samples);
        assert!((out.drift / 1e-6 - 1.0).abs() < 1e-6, "{}", out.drift);
    }

    #[test]
    fn one_sample_is_not_a_fit() {
        let mut acc = EnergyDrift::new(1).unwrap().accumulator();
        acc.push(EnergyDriftSample {
            time: 0.0,
            energy: 1.0,
        })
        .unwrap();
        assert!(matches!(acc.finish(), Err(ComputeError::EmptyInput)));
    }

    #[test]
    fn time_must_advance_and_values_be_finite() {
        let mut acc = EnergyDrift::new(1).unwrap().accumulator();
        acc.push(EnergyDriftSample {
            time: 1.0,
            energy: 0.0,
        })
        .unwrap();
        assert!(
            acc.push(EnergyDriftSample {
                time: 1.0,
                energy: 0.0
            })
            .is_err()
        );
        assert!(
            acc.push(EnergyDriftSample {
                time: 2.0,
                energy: F::NAN
            })
            .is_err()
        );
        assert!(EnergyDrift::new(0).is_err());
    }
}
