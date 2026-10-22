//! Dipoles, the static dielectric constant and kinetic energy — the WASM face
//! of `molrs::compute::{dielectric, kinetic}`.
//!
//! ```js
//! const moments = new DipoleMoment().compute({ positions: traj, charges });
//! const dipoles = moments.toJSON().moments;   // { dim: [T, 1, 3], data }
//! const eps = new StaticDielectric("real").compute({
//!     dipoles: { dim: [dipoles.dim[0], 3], data: dipoles.data },
//!     volume, temperature: 300, boundary: "tin_foil",
//! });
//! ```

use molrs::compute::{self as mc, Compute};
use wasm_bindgen::prelude::*;

use super::config::{Given, unit_preset};
use super::handle::{described, output};
use super::wire::Fields;

output! {
    /// Kinetic energy `energy` and the Boltzmann constant `boltzmann` in the
    /// configured units.
    KineticEnergyResult => mc::KineticEnergyResult
}

#[wasm_bindgen]
impl KineticEnergyResult {
    /// The temperature these `nDof` degrees of freedom carry.
    pub fn temperature(&self, n_dof: usize) -> Result<f64, JsError> {
        Ok(self.inner.temperature(n_dof)?)
    }
}

output! {
    /// Dipole moment per frame and group: `groups`, `moments` (`T × G × 3`).
    DipoleMomentResult => mc::DipoleMomentResult
}

output! {
    /// The static dielectric constant and its fluctuation terms.
    StaticDielectricResult => mc::StaticDielectricResult
}

/// Kinetic energy of one snapshot, in a unit preset's units — one
/// constructor, as `KineticEnergy::new(units).with_mvv2e(..).with_boltzmann(..)`.
#[wasm_bindgen]
pub struct KineticEnergy {
    inner: mc::KineticEnergy,
}

#[wasm_bindgen]
impl KineticEnergy {
    /// `units` is a registered unit preset name. `mvv2e` replaces the
    /// preset's `mass · velocity²` → energy factor (1 for consistent units)
    /// and `boltzmann` its Boltzmann constant in that energy unit, each only
    /// when given (both positive).
    ///
    /// ```js
    /// const ke  = new KineticEnergy("real");
    /// const raw = new KineticEnergy("real", 1.0, 0.0019872);
    /// ```
    #[wasm_bindgen(constructor)]
    pub fn new(
        units: &str,
        mvv2e: Option<f64>,
        boltzmann: Option<f64>,
    ) -> Result<KineticEnergy, JsError> {
        let inner = mc::KineticEnergy::new(unit_preset(units)?)?
            .given(mvv2e, mc::KineticEnergy::with_mvv2e)?
            .given(boltzmann, mc::KineticEnergy::with_boltzmann)?;
        Ok(Self { inner })
    }

    /// `{ masses, velocities }`: `masses` is `N`, `velocities` `N × 3`.
    pub fn compute(&self, data: &JsValue) -> Result<KineticEnergyResult, JsError> {
        let data = Fields::new(data, "KineticEnergy")?;
        let masses = data.reals("masses")?;
        let velocities = data.matrix("velocities", Some(3))?;
        let input = mc::KineticEnergyInput {
            masses: masses.view(),
            velocities: velocities.view(),
        };
        Ok(self.inner.compute(input)?.into())
    }
}

/// Total (or per-group) dipole moment of every frame.
#[wasm_bindgen]
pub struct DipoleMoment {
    inner: mc::DipoleMoment,
}

#[wasm_bindgen]
impl DipoleMoment {
    #[wasm_bindgen(constructor)]
    #[allow(clippy::new_without_default)]
    pub fn new() -> DipoleMoment {
        Self {
            inner: mc::DipoleMoment::new(),
        }
    }

    /// `{ positions, charges, partition? }`: `positions` is `T × N × 3`.
    pub fn compute(&self, data: &JsValue) -> Result<DipoleMomentResult, JsError> {
        let data = Fields::new(data, "DipoleMoment")?;
        let positions = data.cube("positions")?;
        let charges = data.reals("charges")?;
        let partition = data.opt_partition("partition")?;
        let input = mc::DipoleMomentInput {
            positions: positions.view(),
            charges: charges.view(),
            partition: partition.as_ref().map(|p| p.as_dyn()),
        };
        Ok(self.inner.compute(input)?.into())
    }
}

/// The static dielectric constant from total-dipole fluctuations.
#[wasm_bindgen]
pub struct StaticDielectric {
    inner: mc::StaticDielectric,
}

#[wasm_bindgen]
impl StaticDielectric {
    #[wasm_bindgen(constructor)]
    pub fn new(units: &str, epsilon_inf: Option<f64>) -> Result<StaticDielectric, JsError> {
        let inner = mc::StaticDielectric::new(unit_preset(units)?)
            .given(epsilon_inf, |c, v| Ok(c.with_epsilon_inf(v)))?;
        Ok(Self { inner })
    }

    /// `{ dipoles, volume, temperature, boundary }`: `dipoles` is `T × 3`,
    /// `boundary` an [`EwaldBoundary`](mc::EwaldBoundary) name.
    pub fn compute(&self, data: &JsValue) -> Result<StaticDielectricResult, JsError> {
        let data = Fields::new(data, "StaticDielectric")?;
        let dipoles = data.matrix("dipoles", Some(3))?;
        let input = mc::StaticDielectricInput {
            dipoles: dipoles.view(),
            volume: data.scalar("volume")?,
            temperature: data.scalar("temperature")?,
            boundary: data.choice("boundary")?,
        };
        Ok(self.inner.compute(input)?.into())
    }
}

described!(
    KineticEnergy => mc::KineticEnergy,
    DipoleMoment => mc::DipoleMoment,
    StaticDielectric => mc::StaticDielectric,
);
