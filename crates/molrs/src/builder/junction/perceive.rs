//! [`Retype::Perceive`](super::Retype::Perceive): atom types perceived again
//! by a typifier, over the whole linked graph.

use std::fmt;
use std::sync::Arc;

use super::JunctionError;
use crate::core::keys;
use crate::core::{Atomistic, NodeId, PropValue};
use crate::ff::typifier::{Annotation, Typifier};

/// Atom types perceived again by `typifier` over the whole linked graph.
///
/// The typifier's [`assign`](Typifier::assign) runs on a copy of the graph;
/// of what it assigns only each atom's `type` name is read, and written back
/// on the atoms whose type it changes. Its bonded terms, parameters and
/// output force field are not used: pricing the graph is the parameter
/// assignment's step, which overwrites what the old types left. Perceiving
/// the whole graph is correct for any typifier; a junction-local variant would
/// need the typifier to declare how far its rules look.
#[derive(Clone)]
pub struct Perceive {
    typifier: Arc<dyn Typifier + Send + Sync>,
}

impl fmt::Debug for Perceive {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Perceive")
            .field("forcefield", &self.typifier.source_forcefield().name)
            .finish()
    }
}

impl Perceive {
    /// Perceive types with `typifier`.
    pub fn new(typifier: Arc<dyn Typifier + Send + Sync>) -> Self {
        Self { typifier }
    }

    /// The `(atom, type)` of every atom of `world`, as the typifier perceives
    /// them; `world` is not touched.
    ///
    /// # Errors
    ///
    /// [`JunctionError::Perception`] when the typifier refuses the graph;
    /// [`JunctionError::Untyped`] when it gives an atom no type, or does not
    /// keep the graph's atoms.
    pub(super) fn plan(&self, world: &Atomistic) -> Result<Vec<(NodeId, String)>, JunctionError> {
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
