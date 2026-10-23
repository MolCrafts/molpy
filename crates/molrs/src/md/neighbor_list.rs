//! The `neighbor` phase: the run's one [`NeighborList`], kept current by a
//! Verlet skin, publishing its pairs onto [`Scratch`] for a later `force`
//! phase to read.

use ndarray::ArrayView2;

use molrs::core::{Frame, NeighborList, NeighborPolicy, VerletSkin};
use molrs::op::F;

use super::columns::read_pos;
use super::component::Component;
use super::error::MdError;
use super::phase::Phase;
use super::scratch::Scratch;

/// Where the list lives between builds: handed in, or wrapped for the cell
/// of the last build.
enum Held {
    Bare(Box<NeighborList>),
    Wrapped(Box<VerletSkin>),
}

/// The neighbour component [`MD::set_neighbor`](crate::md::MD::set_neighbor)
/// mounts. The list searches at its own cutoff; pairs within `cutoff − skin`
/// reach the kernels, and the list is searched again once an atom has moved
/// half the skin. Periodicity is the frame box's, not the list's.
pub(crate) struct Neighbor {
    held: Option<Held>,
    policy: NeighborPolicy,
}

impl Neighbor {
    /// `skin` in the frame's length unit, inside `list.cutoff()`.
    pub(crate) fn new(list: NeighborList, skin: F) -> Result<Self, MdError> {
        if !skin.is_finite() || skin < 0.0 || skin >= list.cutoff() {
            return Err(MdError::Invalid(format!(
                "skin must be >= 0 and below the list's cutoff {}, got {skin}",
                list.cutoff()
            )));
        }
        Ok(Self {
            held: Some(Held::Bare(Box::new(list))),
            policy: NeighborPolicy {
                skin,
                every: 1,
                delay: 0,
                check: true,
            },
        })
    }
}

impl Component for Neighbor {
    fn name(&self) -> &str {
        "neighbor_list"
    }

    fn default_bindings(&self) -> Vec<(Phase, i32)> {
        vec![(Phase::Neighbor, -10)]
    }

    fn build(&mut self, frame: &mut Frame) -> Result<(), MdError> {
        let pos = read_pos(frame)?;
        let simbox = frame
            .simbox
            .clone()
            .ok_or_else(|| MdError::Invalid("the neighbor list needs a frame box".into()))?;
        let list = match self.held.take() {
            Some(Held::Bare(list)) => *list,
            Some(Held::Wrapped(skin)) => skin.into_search(),
            None => {
                return Err(MdError::Invalid(
                    "the neighbor list was lost by a failed build".into(),
                ));
            }
        };
        let cutoff = list.cutoff() - self.policy.skin;
        let skin = VerletSkin::new(list, cutoff, self.policy, ArrayView2::from(&pos), simbox)
            .map_err(MdError::from)?;
        self.held = Some(Held::Wrapped(Box::new(skin)));
        Ok(())
    }

    fn launch(
        &mut self,
        phase: Phase,
        frame: &mut Frame,
        scratch: &mut Scratch,
    ) -> Result<(), MdError> {
        if phase != Phase::Neighbor {
            return Ok(());
        }
        let Some(Held::Wrapped(skin)) = self.held.as_mut() else {
            return Err(MdError::Invalid(
                "neighbor list was launched before build".into(),
            ));
        };
        let pos = read_pos(frame)?;
        scratch.pairs = Some(skin.pairs_at(pos.view()).map_err(MdError::from)?.clone());
        Ok(())
    }
}
