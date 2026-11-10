//! A system: molecules sampled until each reaches its share of a total size.

use std::collections::HashMap;

use rand::SeedableRng;
use rand::rngs::StdRng;

use super::PolymerError;
use super::molecule::{Molecule, add_sites};
use super::size::Measure;
use crate::core::{Atomistic, CoarseGrain};
use crate::op::F;

/// How much of the system one molecule kind makes up, counted in the
/// system's [`Measure`] — generative BigSMILES `.|5e5|` and `.|20%|`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Amount {
    /// An absolute amount, `> 0`.
    Total(F),
    /// A share of the whole system, in percent, `0 < p < 100`.
    Percent(F),
}

/// What one sampled molecule is: its kind and its size.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MoleculeRecord {
    /// Index of the component (molecule kind) it was sampled from.
    pub component: usize,
    /// The number of fragments placed — its sites, in site order.
    pub n_fragments: usize,
    /// How many of them are repeat units.
    pub n_repeat_units: usize,
    /// Summed non-hydrogen atomic mass, g/mol.
    pub heavy_atom_mass: F,
    /// Summed atomic mass with the hydrogens each fragment keeps, g/mol.
    pub mass: F,
}

/// One sample of a system: every molecule's sites in one graph (a
/// connected component per molecule, in record order) and a record per
/// molecule.
#[derive(Debug)]
pub struct SystemSample {
    /// The site graph; feed it to an
    /// [`Assembler`](crate::builder::Assembler) over [`System::library`].
    pub sites: CoarseGrain,
    /// One record per molecule, in site order.
    pub molecules: Vec<MoleculeRecord>,
}

/// Molecule kinds sampled until each reaches its [`Amount`] of the total.
#[derive(Debug, Clone)]
pub struct System {
    components: Vec<(Molecule, Amount)>,
    measure: Measure,
}

impl System {
    /// A system of `components`, amounts counted in `measure`.
    ///
    /// The total is fixed by the absolute amounts: with absolute amounts
    /// summing to `T` and percentages summing to `P`, the system totals
    /// `T / (1 − P/100)`, and a `p%` component makes up `p/100` of it.
    ///
    /// # Errors
    ///
    /// [`PolymerError::InvalidSystem`] for no component, no absolute amount,
    /// an amount out of range, percentages summing to 100 or more, a molecule
    /// that can measure zero in `measure` (no number of copies reaches a
    /// target), or two different fragments under one name across components.
    pub fn new(
        components: Vec<(Molecule, Amount)>,
        measure: Measure,
    ) -> Result<Self, PolymerError> {
        let invalid = |reason: String| Err(PolymerError::InvalidSystem(reason));
        if components.is_empty() {
            return invalid("a system needs at least one molecule".to_owned());
        }
        let mut percent = 0.0;
        let mut absolute = false;
        for (_, amount) in &components {
            match *amount {
                Amount::Total(t) if t.is_finite() && t > 0.0 => absolute = true,
                Amount::Percent(p) if p.is_finite() && p > 0.0 && p < 100.0 => percent += p,
                other => return invalid(format!("{other:?} is out of range")),
            }
        }
        if !absolute {
            return invalid("at least one amount must be absolute".to_owned());
        }
        if percent >= 100.0 {
            return invalid(format!("percentages sum to {percent} ≥ 100"));
        }
        let system = Self {
            components,
            measure,
        };
        if let Some(index) = system
            .components
            .iter()
            .position(|(molecule, _)| molecule.least(measure) <= 0.0)
        {
            return invalid(format!(
                "a molecule of component {index} can measure zero in {measure:?}"
            ));
        }
        super::check_names(
            system.components.iter().flat_map(|(m, _)| m.fragments()),
            PolymerError::InvalidSystem,
        )?;
        Ok(system)
    }

    /// The molecule kinds and their amounts.
    pub fn components(&self) -> &[(Molecule, Amount)] {
        &self.components
    }

    /// What the amounts count.
    pub fn measure(&self) -> Measure {
        self.measure
    }

    /// The target of each component, in the system's measure.
    pub fn targets(&self) -> Vec<F> {
        let (absolute, percent) = self
            .components
            .iter()
            .fold((0.0, 0.0), |(t, p), (_, amount)| match *amount {
                Amount::Total(x) => (t + x, p),
                Amount::Percent(x) => (t, p + x),
            });
        let total = absolute / (1.0 - percent / 100.0);
        self.components
            .iter()
            .map(|(_, amount)| match *amount {
                Amount::Total(x) => x,
                Amount::Percent(x) => total * x / 100.0,
            })
            .collect()
    }

    /// Every fragment any molecule may place, under its name.
    pub fn library(&self) -> HashMap<String, Atomistic> {
        self.components
            .iter()
            .flat_map(|(m, _)| m.library())
            .collect()
    }

    /// Sample molecules of each component, in component order, until the
    /// component's summed size reaches its target. Deterministic in `seed`.
    ///
    /// A component stops at the first molecule that reaches its target, so
    /// it overshoots by less than one molecule; nothing is trimmed and the
    /// number of molecules is not bounded — the amounts are sampled as
    /// written.
    ///
    /// # Errors
    ///
    /// What [`Molecule::sample`] raises.
    pub fn sample(&self, seed: u64) -> Result<SystemSample, PolymerError> {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut sites = CoarseGrain::new();
        let mut molecules = Vec::new();
        for (component, ((molecule, _), target)) in
            self.components.iter().zip(self.targets()).enumerate()
        {
            let mut reached = 0.0;
            while reached < target {
                let sample = molecule.grow(&mut rng)?;
                let (heavy_atom_mass, mass) = sample.masses();
                let size = match self.measure {
                    Measure::RepeatUnits => sample.n_repeat_units as F,
                    Measure::HeavyAtomMass => heavy_atom_mass,
                    Measure::Mass => mass,
                };
                add_sites(&mut sites, &sample)?;
                molecules.push(MoleculeRecord {
                    component,
                    n_fragments: sample.instances.len(),
                    n_repeat_units: sample.n_repeat_units,
                    heavy_atom_mass,
                    mass,
                });
                reached += size;
            }
        }
        Ok(SystemSample { sites, molecules })
    }
}
