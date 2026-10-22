//! Neighbour-list component. Publishes pairs onto [`Scratch`] during `neighbor`
//! for a later `force` phase to read.

use ndarray::ArrayView2;

use molrs::core::{Frame, NeighborList as CoreList, NeighborPolicy, VerletSkin};
use molrs::op::F;

use super::columns::read_pos;
use super::component::Component;
use super::error::MdError;
use super::phase::Phase;
use super::scratch::Scratch;

/// Minimum-image Verlet list bound to the `neighbor` phase.
///
/// `cutoff` is the distance the potential sees. The list is built at
/// `cutoff + skin`.
pub struct NeighborList {
    cutoff: F,
    policy: NeighborPolicy,
    skin: Option<VerletSkin>,
}

impl NeighborList {
    /// `cutoff` and `skin` in the frame's length unit.
    pub fn new(cutoff: F, skin: F) -> Result<Self, MdError> {
        if !cutoff.is_finite() || cutoff <= 0.0 {
            return Err(MdError::Invalid(format!(
                "cutoff must be > 0, got {cutoff}"
            )));
        }
        if !skin.is_finite() || skin < 0.0 {
            return Err(MdError::Invalid(format!("skin must be >= 0, got {skin}")));
        }
        Ok(Self {
            cutoff,
            policy: NeighborPolicy {
                skin,
                every: 1,
                delay: 0,
                check: true,
            },
            skin: None,
        })
    }
}

impl Component for NeighborList {
    fn name(&self) -> &str {
        "neighbor_list"
    }

    fn default_bindings(&self) -> Vec<(Phase, i32)> {
        vec![(Phase::Neighbor, -10)]
    }

    fn build(&mut self, frame: &mut Frame) -> Result<(), MdError> {
        let pos = read_pos(frame)?;
        let simbox = frame.simbox.clone().ok_or_else(|| {
            MdError::Invalid(
                "neighbor list requires a frame box; a free boundary has no minimum image".into(),
            )
        })?;
        let search = CoreList::new(self.cutoff + self.policy.skin);
        self.skin = Some(
            VerletSkin::new(
                search,
                self.cutoff,
                self.policy,
                ArrayView2::from(&pos),
                simbox,
            )
            .map_err(MdError::from)?,
        );
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
        let skin = self
            .skin
            .as_mut()
            .ok_or_else(|| MdError::Invalid("neighbor list was launched before build".into()))?;
        let pos = read_pos(frame)?;
        let pairs = skin.pairs_at(pos.view()).map_err(MdError::from)?;
        scratch.pairs = Some(pairs.clone());
        Ok(())
    }
}
