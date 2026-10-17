//! LAMMPS Verlet phases. Every step walks [`Phase::ALL`], including phases
//! with no component bound.

/// One point in a molecular-dynamics step, in walk order.
///
/// Names are the LAMMPS Verlet names. The discriminant is the walk index.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Phase {
    /// First half-step: `v += ½ dt f/m`, `x += dt v`.
    InitialIntegrate = 0,
    /// Hook immediately after the first half-step.
    PostIntegrate = 1,
    /// Re-home atoms that crossed a rank boundary. Empty in a single process.
    PreExchange = 2,
    /// Build ghost atoms. Empty when the run has no halo.
    PreNeighbor = 3,
    /// Rebuild the neighbour list.
    Neighbor = 4,
    /// Zero force, energy, and stress before evaluation.
    PreForce = 5,
    /// Accumulate energy, forces, and virial.
    Force = 6,
    /// Constraints and thermostats that add to the force, then ghost-force fold.
    PostForce = 7,
    /// Second half-step: `v += ½ dt f/m`.
    FinalIntegrate = 8,
    /// Reductions, barostat, observers, output.
    EndOfStep = 9,
}

impl Phase {
    /// Walk order. `COUNT` is the length, not a phase.
    pub const ALL: [Phase; 10] = [
        Phase::InitialIntegrate,
        Phase::PostIntegrate,
        Phase::PreExchange,
        Phase::PreNeighbor,
        Phase::Neighbor,
        Phase::PreForce,
        Phase::Force,
        Phase::PostForce,
        Phase::FinalIntegrate,
        Phase::EndOfStep,
    ];

    /// Number of phases. Not itself a phase a component can bind.
    pub const COUNT: usize = Self::ALL.len();

    /// LAMMPS spelling: `initial_integrate`, `post_force`, `end_of_step`.
    pub const fn name(self) -> &'static str {
        match self {
            Phase::InitialIntegrate => "initial_integrate",
            Phase::PostIntegrate => "post_integrate",
            Phase::PreExchange => "pre_exchange",
            Phase::PreNeighbor => "pre_neighbor",
            Phase::Neighbor => "neighbor",
            Phase::PreForce => "pre_force",
            Phase::Force => "force",
            Phase::PostForce => "post_force",
            Phase::FinalIntegrate => "final_integrate",
            Phase::EndOfStep => "end_of_step",
        }
    }

    /// The phase named by the LAMMPS spelling, or `None`.
    pub fn parse(name: &str) -> Option<Phase> {
        Self::ALL.into_iter().find(|phase| phase.name() == name)
    }
}
