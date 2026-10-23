//! [`PairRole`]: what part of the nonbonded treatment a pair style plays, a
//! declaration of its [`StyleSpec`](crate::ff::ir::StyleSpec).

/// What a pair style is to the system's nonbonded treatment — the fact an
/// engine deck reads to pick its electrostatics method and cutoffs.
///
/// Declared at registration rather than guessed from the style's name: a new
/// Coulomb style states what it is, and no engine's name list goes stale.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PairRole {
    /// Van-der-Waals: Lennard-Jones, Buckingham, …
    Dispersion,
    /// The system's electrostatics.
    Coulomb(CoulombRange),
    /// A short-range screening of other electrostatics (Thole, Tang–Toennies
    /// dampings of Drude dipoles): it sets no method and no cutoff.
    Damping,
}

/// How far a Coulomb style reaches.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CoulombRange {
    /// Truncated at its cutoff.
    Cut,
    /// Summed to infinity (Ewald, PME, PPPM).
    Long,
}
