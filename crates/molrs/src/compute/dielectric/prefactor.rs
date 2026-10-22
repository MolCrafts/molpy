//! [`DielectricPrefactor`] — the one place the permittivity and conductivity
//! prefactors are formed.

use molrs::core::{FOUR_PI, FOUR_THIRDS_PI, UnitPreset};
use molrs::op::F;

use crate::compute::ComputeError;

/// The thermodynamic scale of a dielectric fluctuation formula: the volume
/// `V` and temperature `T` of the state, and the Boltzmann and Coulomb
/// constants `k_B`, `κ = 1/(4π ε₀)` of the configured [`UnitPreset`].
///
/// Every permittivity and conductivity compute — [`StaticDielectric`],
/// the four ε(ω) routes and the conductivity sum rule — forms its prefactor
/// here and nowhere else. No prefactor carries a time unit: the time in a
/// transform cancels between the lag spacing and the angular frequency, so
/// any consistent time unit works (frequencies come out in radians per unit
/// of the lag spacing).
///
/// [`StaticDielectric`]: crate::compute::StaticDielectric
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct DielectricPrefactor {
    volume: F,
    temperature: F,
    boltzmann: F,
    coulomb: F,
}

impl DielectricPrefactor {
    /// The scale of a state of `volume` (preset length³) at `temperature`.
    ///
    /// # Errors
    /// [`ComputeError::OutOfRange`] unless `volume` and `temperature` are
    /// finite and positive.
    pub(crate) fn new(units: &UnitPreset, volume: F, temperature: F) -> Result<Self, ComputeError> {
        for (field, value) in [("volume", volume), ("temperature", temperature)] {
            if !(value.is_finite() && value > 0.0) {
                return Err(ComputeError::OutOfRange {
                    field,
                    value: value.to_string(),
                });
            }
        }
        Ok(Self {
            volume,
            temperature,
            boltzmann: units.boltzmann(),
            coulomb: units.coulomb(),
        })
    }

    /// `A = 4π κ / (3 V k_B T)`: the Neumann fluctuation amplitude, so that
    /// `ε − ε_∞ = A ⟨|δM|²⟩` under conducting boundaries. Per (charge ·
    /// length)².
    pub(crate) fn fluctuation(&self) -> F {
        FOUR_THIRDS_PI * self.coulomb / (self.volume * self.boltzmann * self.temperature)
    }

    /// `V / (3 k_B T)`: the Green–Kubo conductivity prefactor of a
    /// current-density correlation `⟨J · J⟩`, `J = Ṁ / V`.
    pub(crate) fn current_density(&self) -> F {
        self.volume / (3.0 * self.boltzmann * self.temperature)
    }

    /// `1 / (3 V k_B T)`: the Green–Kubo prefactor of a total-current
    /// correlation `⟨Ṁ · Ṁ⟩`.
    pub(crate) fn total_current(&self) -> F {
        1.0 / (3.0 * self.volume * self.boltzmann * self.temperature)
    }

    /// `1 / ε₀ = 4π κ`.
    pub(crate) fn inverse_permittivity(&self) -> F {
        FOUR_PI * self.coulomb
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use molrs::core::constants::COULOMB_REAL;

    #[test]
    fn real_units_reproduce_the_md_formulas() {
        let p = DielectricPrefactor::new(&UnitPreset::real(), 1000.0, 300.0).unwrap();
        let kb = UnitPreset::real().boltzmann();
        assert_eq!(
            p.fluctuation(),
            FOUR_THIRDS_PI * COULOMB_REAL / (1000.0 * kb * 300.0)
        );
        assert_eq!(p.current_density(), 1000.0 / (3.0 * kb * 300.0));
        assert_eq!(p.inverse_permittivity(), FOUR_PI * COULOMB_REAL);
        let ratio = p.total_current() * 1000.0 * 1000.0 / p.current_density();
        assert!((ratio - 1.0).abs() < 1e-14);
    }

    #[test]
    fn a_non_physical_state_is_refused() {
        let units = UnitPreset::real();
        for (v, t) in [
            (0.0, 300.0),
            (-1.0, 300.0),
            (1000.0, 0.0),
            (F::NAN, 300.0),
            (1000.0, F::INFINITY),
        ] {
            assert!(matches!(
                DielectricPrefactor::new(&units, v, t),
                Err(ComputeError::OutOfRange { .. })
            ));
        }
    }
}
