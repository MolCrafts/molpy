//! [`Retype::Perceive`](super::Retype::Perceive): atom types perceived again
//! by a typifier over the whole linked graph, written everywhere or, verified,
//! only near the junctions.

use std::collections::HashSet;
use std::fmt;
use std::sync::Arc;

use super::JunctionError;
use crate::core::keys;
use crate::core::{Atomistic, Junction, NodeId, PropValue};
use crate::ff::typifier::{Annotation, Typifier};

/// Atom types perceived again by `typifier` over the whole linked graph.
///
/// The typifier's [`assign`](Typifier::assign) runs on a copy of the graph;
/// of what it assigns only each atom's `type` name is read, and written back
/// on the atoms whose type it changes. Its bonded terms, parameters and
/// output force field are not used: pricing the graph is the parameter
/// assignment's step, which overwrites what the old types left.
///
/// **Within a radius** ([`within`](Self::within)): the perception is still of
/// the whole graph — a typifier's types can depend on far atoms (bond orders
/// solved over a residue, ring systems, the phase of a conjugated chain), so
/// typing a cut-out piece would not give the same answer. The call is refused
/// when the perception changes the type of any atom more than `radius` bonds
/// from every junction anchor: the
/// rest of the graph is verified to be already what the typifier would make
/// it, not assumed to be, so only atoms near the anchors change. The typifier
/// knows nothing of junctions; the scope is this stage's.
#[derive(Clone)]
pub struct Perceive {
    typifier: Arc<dyn Typifier + Send + Sync>,
    radius: Option<usize>,
}

impl fmt::Debug for Perceive {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Perceive")
            .field("forcefield", &self.typifier.source_forcefield().name)
            .field("radius", &self.radius)
            .finish()
    }
}

impl Perceive {
    /// Perceive types with `typifier` and write them on every atom.
    pub fn new(typifier: Arc<dyn Typifier + Send + Sync>) -> Self {
        Self {
            typifier,
            radius: None,
        }
    }

    /// Change only atoms at most `radius` bonds from a junction anchor:
    /// refuse when the perception changes any other atom's type.
    pub fn within(self, radius: usize) -> Self {
        Self {
            radius: Some(radius),
            ..self
        }
    }

    /// The `(atom, type)` writes for `junctions`: every atom's perceived type,
    /// or within a radius only the atoms near an anchor; `world` is not
    /// touched.
    ///
    /// # Errors
    ///
    /// [`JunctionError::Perception`] when the typifier refuses the graph;
    /// [`JunctionError::Untyped`] when it gives an atom no type, or does not
    /// keep the graph's atoms; [`JunctionError::ChangedBeyondRadius`] when,
    /// within a radius, an atom beyond it would change type.
    pub(super) fn plan(
        &self,
        world: &Atomistic,
        junctions: &[Junction],
    ) -> Result<Vec<(NodeId, String)>, JunctionError> {
        let perceived = self.perceive(world)?;
        let Some(radius) = self.radius else {
            return Ok(perceived);
        };
        let anchors: Vec<NodeId> = junctions.iter().flat_map(|j| j.anchors).collect();
        let near: HashSet<NodeId> = world
            .topo_distances(&anchors, Some(radius as i64))
            .into_iter()
            .map(|(atom, _)| atom)
            .collect();
        // Beyond the radius every type must already be the perceived one; the
        // writes there are then no-ops, so the plan is the whole perception.
        let changed: Vec<NodeId> = perceived
            .iter()
            .filter(|(atom, ty)| {
                !near.contains(atom)
                    && world
                        .get_atom(*atom)
                        .map(|props| props.get_str(keys::TYPE) != Some(ty.as_str()))
                        .unwrap_or(true)
            })
            .map(|(atom, _)| *atom)
            .collect();
        if !changed.is_empty() {
            return Err(JunctionError::ChangedBeyondRadius {
                radius,
                atoms: changed,
            });
        }
        Ok(perceived)
    }

    /// Every atom's `(atom, type)` as the typifier perceives the whole graph.
    fn perceive(&self, world: &Atomistic) -> Result<Vec<(NodeId, String)>, JunctionError> {
        let mut copy = world.clone();
        let assignment = self
            .typifier
            .assign(&mut copy)
            .map_err(JunctionError::Perception)?;
        let atoms: Vec<NodeId> = world.atom_ids().collect();
        if copy.atom_ids().ne(atoms.iter().copied()) {
            return Err(JunctionError::Untyped {
                atom: None,
                detail: "the typifier changed the graph's atoms",
            });
        }
        atoms
            .iter()
            .enumerate()
            .map(|(row, &atom)| {
                let name = assignment.nodes.get(row).and_then(|annotations| {
                    annotations.iter().find_map(|(key, annotation)| {
                        (key == keys::TYPE)
                            .then_some(annotation)
                            .and_then(|a| match a {
                                Annotation::Type { name, .. } => Some(name.clone()),
                                Annotation::Value(PropValue::Str(name)) => Some(name.clone()),
                                Annotation::Value(_) => None,
                            })
                    })
                });
                name.map(|name| (atom, name)).ok_or(JunctionError::Untyped {
                    atom: Some(atom),
                    detail: "the typifier gave it no type",
                })
            })
            .collect()
    }
}
