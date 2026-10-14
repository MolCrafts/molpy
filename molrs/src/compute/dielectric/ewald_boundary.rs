//! [`EwaldBoundary`] — the electrostatic boundary a dipole trajectory was
//! sampled under.

use crate::compute::ComputeError;

/// Ewald boundary condition under which the dipole fluctuations were sampled.
///
/// The Debye / Neumann–Kirkwood amplitude `ε₀ − ε∞` depends on this boundary
/// condition, so a dielectric spectrum route takes it with the volume and
/// temperature of the state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EwaldBoundary {
    /// Conducting / "tin-foil" boundary (ε' = ∞) — the common MD default.
    TinFoil,
    /// Vacuum boundary (ε' = 1).
    Vacuum,
}

impl EwaldBoundary {
    /// Parse a boundary-condition name (case-insensitive): `"tinfoil"` /
    /// `"tin-foil"` / `"tin_foil"` / `"conducting"` → [`TinFoil`](Self::TinFoil),
    /// `"vacuum"` → [`Vacuum`](Self::Vacuum).
    pub fn from_name(name: &str) -> Result<Self, ComputeError> {
        match name.to_ascii_lowercase().as_str() {
            "tinfoil" | "tin-foil" | "tin_foil" | "conducting" => Ok(Self::TinFoil),
            "vacuum" => Ok(Self::Vacuum),
            other => Err(ComputeError::OutOfRange {
                field: "EwaldBoundary",
                value: other.to_string(),
            }),
        }
    }
}
