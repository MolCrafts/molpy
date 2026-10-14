//! [`EwaldBoundary`] — the electrostatic boundary a dipole trajectory was
//! sampled under.

use crate::compute::ComputeError;
use crate::compute::named_enum::named_enum;

named_enum! {
    /// Ewald boundary condition under which the dipole fluctuations were
    /// sampled.
    ///
    /// The fluctuation formulas for the permittivity depend on the dielectric
    /// constant ε′ of the continuum surrounding the periodic system (Neumann,
    /// *Mol. Phys.* **50**, 841 (1983)), so every permittivity compute takes
    /// the boundary with the volume and temperature of the state. The formulas
    /// molrs implements are the conducting-boundary ones; a vacuum boundary is
    /// refused rather than analysed with the wrong formula.
    pub enum EwaldBoundary {
        /// Conducting / "tin-foil" boundary (ε′ = ∞) — the common MD default.
        TinFoil => "tin_foil",
        /// Vacuum boundary (ε′ = 1).
        Vacuum => "vacuum",
    }
}

impl EwaldBoundary {
    /// `Ok` for the boundaries the permittivity formulas hold under.
    ///
    /// # Errors
    /// [`ComputeError::OutOfRange`] for [`Vacuum`](Self::Vacuum): its
    /// `(ε − 1)(2ε′ + 1)/(2ε′ + ε)` correction is not implemented.
    pub(crate) fn require_supported(self) -> Result<(), ComputeError> {
        match self {
            Self::TinFoil => Ok(()),
            Self::Vacuum => Err(ComputeError::OutOfRange {
                field: "boundary",
                value: "vacuum: the vacuum-boundary fluctuation formula is not implemented; \
                        sample under tin-foil (conducting) Ewald boundaries"
                    .into(),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_are_canonical() {
        assert_eq!(EwaldBoundary::NAMES, &["tin_foil", "vacuum"]);
        assert_eq!(
            "tin_foil".parse::<EwaldBoundary>().unwrap(),
            EwaldBoundary::TinFoil
        );
        // The old `from_name` aliases are gone.
        for alias in ["tinfoil", "tin-foil", "conducting", "Vacuum"] {
            assert!(matches!(
                alias.parse::<EwaldBoundary>(),
                Err(ComputeError::UnknownName { .. })
            ));
        }
    }

    #[test]
    fn vacuum_is_refused_not_treated_as_tin_foil() {
        assert!(EwaldBoundary::TinFoil.require_supported().is_ok());
        assert!(matches!(
            EwaldBoundary::Vacuum.require_supported(),
            Err(ComputeError::OutOfRange {
                field: "boundary",
                ..
            })
        ));
    }
}
