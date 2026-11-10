//! A molecule: fixed fragments and stochastic objects joined end to end, and the
//! site graph one sample of it is.

use std::collections::HashMap;

use rand::Rng;

use super::PolymerError;
use super::fragment::Fragment;
use super::size::Measure;
use super::stochastic::{Growth, StochasticObject};
use crate::core::{Atomistic, CoarseGrain, MolRsError, NodeId, Props, keys};
use crate::op::F;

/// One piece of a [`Molecule`], in written order.
#[derive(Debug, Clone)]
pub enum Part {
    /// A fragment placed once: a prefix, a connector between objects, a suffix,
    /// or the whole of a small molecule. Its first port joins the part
    /// before it and its last port the part after it.
    Fixed(Fragment),
    /// A fragment chain of random size.
    Stochastic(StochasticObject),
}

/// Fixed fragments and stochastic objects joined end to end — the molecule of a
/// generative-BigSMILES string (`[H]{[] …; … []}|…|`, a prefix, objects,
/// connectors, a suffix).
#[derive(Debug, Clone)]
pub struct Molecule {
    parts: Vec<Part>,
}

/// One sample of a molecule: its fragments in placement order and how they join.
#[derive(Debug, Default)]
pub(super) struct Sample {
    pub instances: Vec<Fragment>,
    pub joins: Vec<((usize, usize), (usize, usize))>,
    pub n_repeat_units: usize,
}

impl Molecule {
    /// The molecule `parts` spell, joined in order.
    ///
    /// Neighbouring parts must agree on the port between them: a fixed fragment
    /// gives its last port to the part after it and takes the part before it
    /// on its first; a stochastic object's left terminal is the port of the
    /// part before it and its right terminal the port of the part after it.
    /// So a fixed fragment before an object carries the object's left terminal,
    /// one after it the right terminal, and between two objects the first
    /// one's right terminal accepts the second one's left.
    ///
    /// # Errors
    ///
    /// [`PolymerError::InvalidMolecule`] for no part, a fixed fragment whose port
    /// count is not its number of neighbours, a missing or disagreeing
    /// terminal, a terminal facing no part, a first object without an end
    /// group to start from, or two different fragments under one name.
    pub fn new(parts: Vec<Part>) -> Result<Self, PolymerError> {
        let invalid = |reason: String| Err(PolymerError::InvalidMolecule(reason));
        if parts.is_empty() {
            return invalid("a molecule needs at least one part".to_owned());
        }
        let n = parts.len();
        for (i, part) in parts.iter().enumerate() {
            if let Part::Fixed(fragment) = part {
                let neighbours = usize::from(i > 0) + usize::from(i + 1 < n);
                if fragment.ports().len() != neighbours {
                    return invalid(format!(
                        "fixed fragment '{}' has {} ports for {} neighbouring parts",
                        fragment.name(),
                        fragment.ports().len(),
                        neighbours
                    ));
                }
            }
        }
        // A port with nothing to join is an error, not an open end: an object
        // nothing precedes starts from an end group, and a terminal faces a
        // part.
        if let Some(Part::Stochastic(first)) = parts.first() {
            if first.left().is_some() {
                return invalid(
                    "the first object has a left terminal and nothing before it".to_owned(),
                );
            }
            if first.end_groups().is_empty() {
                return invalid(
                    "the first object has nothing before it and no end group to start from"
                        .to_owned(),
                );
            }
        }
        if let Some(Part::Stochastic(last)) = parts.last()
            && last.right().is_some()
        {
            return invalid("the last object has a right terminal and nothing after it".to_owned());
        }
        for (i, pair) in parts.windows(2).enumerate() {
            agree(&pair[0], &pair[1]).map_err(|reason| {
                PolymerError::InvalidMolecule(format!("parts {i} and {}: {reason}", i + 1))
            })?;
        }
        let molecule = Self { parts };
        super::check_names(molecule.fragments(), PolymerError::InvalidMolecule)?;
        Ok(molecule)
    }

    /// The parts, in written order.
    pub fn parts(&self) -> &[Part] {
        &self.parts
    }

    /// Every fragment the molecule may place, under its name: the library an
    /// [`Assembler`](crate::builder::Assembler) builds a sample's sites from.
    pub fn library(&self) -> HashMap<String, Atomistic> {
        self.fragments()
            .map(|u| (u.name().to_owned(), u.template().clone()))
            .collect()
    }

    /// One sample as a site graph: a site per placed fragment, its `bead_type`
    /// the fragment's name, and a bond per join with `src_port` / `dst_port`
    /// indexing the two fragments' ports and `bond_type` / `bond_number` the
    /// joined order. Deterministic in `seed`.
    ///
    /// # Errors
    ///
    /// [`PolymerError::Unsized`] when a stochastic object has no size
    /// distribution; [`PolymerError::Stuck`] when growth cannot place or cap
    /// a fragment the object's ports require — an open port no end group
    /// accepts, or a branched object that closed its way to the right
    /// terminal.
    pub fn sample(&self, seed: u64) -> Result<CoarseGrain, PolymerError> {
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let sample = self.grow(&mut rng)?;
        let mut sites = CoarseGrain::new();
        add_sites(&mut sites, &sample)?;
        Ok(sites)
    }

    pub(super) fn fragments(&self) -> impl Iterator<Item = &Fragment> {
        self.parts
            .iter()
            .flat_map(|part| -> Box<dyn Iterator<Item = &Fragment>> {
                match part {
                    Part::Fixed(fragment) => Box::new(std::iter::once(fragment)),
                    Part::Stochastic(object) => Box::new(object.fragments()),
                }
            })
    }

    /// Place every part and join neighbours.
    pub(super) fn grow<R: Rng + ?Sized>(&self, rng: &mut R) -> Result<Sample, PolymerError> {
        let mut sample = Sample::default();
        // The port the previous part offers the next one, as (instance, port).
        let mut offered: Option<(usize, usize)> = None;
        for (i, part) in self.parts.iter().enumerate() {
            let has_next = i + 1 < self.parts.len();
            match part {
                Part::Fixed(fragment) => {
                    let inst = sample.instances.len();
                    sample.instances.push(fragment.clone());
                    if let Some(prev) = offered.take() {
                        sample.joins.push((prev, (inst, 0)));
                    }
                    if has_next {
                        offered = Some((inst, fragment.ports().len() - 1));
                    }
                }
                Part::Stochastic(object) => {
                    let base = sample.instances.len();
                    let Growth {
                        instances,
                        joins,
                        entry,
                        exit,
                        n_repeat_units,
                    } = object.grow(offered.is_some(), rng)?;
                    let shift = |(inst, port): (usize, usize)| (inst + base, port);
                    sample.n_repeat_units += n_repeat_units;
                    sample.instances.extend(instances);
                    sample
                        .joins
                        .extend(joins.into_iter().map(|(a, b)| (shift(a), shift(b))));
                    if let Some(prev) = offered.take() {
                        let entry = entry.expect("an entered object reports its entry");
                        sample.joins.push((prev, shift(entry)));
                    }
                    offered = exit.map(shift);
                }
            }
        }
        Ok(sample)
    }
}

impl Molecule {
    /// The least a sample can measure in `measure`: its fixed fragments and
    /// each object's smallest repeat unit.
    pub(super) fn least(&self, measure: Measure) -> F {
        self.parts
            .iter()
            .map(|part| match part {
                Part::Fixed(fragment) => fragment.size_in(measure, false),
                Part::Stochastic(object) => object.least(measure),
            })
            .sum()
    }
}

impl Sample {
    /// The summed heavy-atom mass and mass of the placed fragments, g/mol.
    pub(super) fn masses(&self) -> (F, F) {
        self.instances.iter().fold((0.0, 0.0), |(h, m), u| {
            (h + u.heavy_atom_mass(), m + u.mass())
        })
    }
}

/// Whether neighbouring parts `a` then `b` agree on the port between them.
fn agree(a: &Part, b: &Part) -> Result<(), String> {
    use crate::core::Port;
    let last = |u: &Fragment| -> Port { u.ports().last().cloned().expect("checked port count") };
    let first = |u: &Fragment| -> Port { u.ports().first().cloned().expect("checked port count") };
    match (a, b) {
        (Part::Fixed(x), Part::Fixed(y)) => (last(x).accepts(&first(y)))
            .then_some(())
            .ok_or_else(|| format!("'{}' and '{}' have no matching ports", x.name(), y.name())),
        (Part::Fixed(x), Part::Stochastic(o)) => match o.left() {
            Some(t) if t.describes(&last(x)) => Ok(()),
            Some(_) => Err(format!(
                "the last port of '{}' is not the left terminal",
                x.name()
            )),
            None => Err("an object after a part needs a left terminal".to_owned()),
        },
        (Part::Stochastic(o), Part::Fixed(y)) => match o.right() {
            Some(t) if t.describes(&first(y)) => Ok(()),
            Some(_) => Err(format!(
                "the first port of '{}' is not the right terminal",
                y.name()
            )),
            None => Err("an object before a part needs a right terminal".to_owned()),
        },
        (Part::Stochastic(p), Part::Stochastic(q)) => match (p.right(), q.left()) {
            (Some(r), Some(l)) if r.port().accepts(&l.port()) => Ok(()),
            (Some(_), Some(_)) => {
                Err("the right terminal does not accept the next left terminal".to_owned())
            }
            _ => Err("adjacent objects need facing terminals".to_owned()),
        },
    }
}

/// Append `sample` to `sites`: one bead per fragment, one bond per join.
pub(super) fn add_sites(sites: &mut CoarseGrain, sample: &Sample) -> Result<(), PolymerError> {
    let build = |e: MolRsError| PolymerError::Sites(e.to_string());
    let beads = sample
        .instances
        .iter()
        .map(|u| sites.add_bead(Props::new().with(keys::BEAD_TYPE, u.name())))
        .collect::<Result<Vec<NodeId>, _>>()
        .map_err(build)?;
    for &((a, pa), (b, pb)) in &sample.joins {
        let order = sample.instances[a].ports()[pa].order;
        sites
            .add_site_bond(beads[a], beads[b], (pa, pb), order.implied_type(), order)
            .map_err(build)?;
    }
    Ok(())
}
