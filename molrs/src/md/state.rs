//! Typed array containers for the MD engine: [`ForceOutput`] and [`MdState`].
//!
//! There was a third, `MDObservables`, for a runner's hooks. It had no
//! consumer and could not have had a correct one: `kinetic`, `total` and
//! `temperature` need a mass array, a `k_B` and a removed-degrees-of-freedom
//! count that it did not carry, and the only object with all three is the
//! runner that does not exist yet. It will be designed with that runner, when
//! there is something to hand it to.
#![allow(dead_code)]

use ndarray::Array2;

use ndarray::Array1;

use molrs::core::Frame;
use molrs::core::SimBox;
use molrs::core::keys;
use molrs::core::schema::block_names::ATOMS;
use molrs::op::{F, Fnx3, I};

use super::error::MdError;
use molrs::core::Virial;

/// Energy + forces from one integrator force evaluation.
///
/// [`ForceOutput`] and [`MdState`] are the data contract crossing the
/// component boundary: a force evaluation hands back energy, forces and
/// virial, and the state carries them at `pos` alongside the coordinates and
/// their image flags. Frame topology stays at the composer; the hot step sees
/// these two structs.
///
/// `energy` is a scalar in amu·Å²/fs². `forces` is `(N, 3)` `= -∂E/∂pos`
/// in the same unit per Å.
#[derive(Clone, Debug)]
pub struct ForceOutput {
    /// Scalar total energy `()` in amu·Å²/fs².
    pub energy: F,
    /// Per-atom forces `(N, 3)`.
    pub forces: Fnx3,
    /// `Σ f ⊗ r` at `pos`, when the provider tallies one.
    ///
    /// `None` is not zero. A provider that does not tally says so, because a
    /// pressure computed from a fabricated zero is wrong and looks entirely
    /// plausible — the same distinction `Neighbors::disp` draws, for the same
    /// reason.
    pub virial: Option<Virial>,
}

/// Dynamical state advanced one step by an integrator.
///
/// Carries the force-cache (`forces` / `energy` at `pos`, both from the same
/// force-field evaluation) so the loop does one evaluation per step.
///
/// # Canonical coordinates: wrapped, plus image flags
///
/// `pos` is **wrapped** — it stays in the primary cell — and `images` records
/// how many cells each atom has crossed to get there. The continuous
/// trajectory is a derived view, `r^u = pos + H·images`
/// ([`SimBox::unwrap`](molrs::core::SimBox::unwrap)), never a second
/// float array integrated alongside the first: two independently-advanced
/// copies of the same quantity drift apart, and the one that drifts is
/// whichever the reader did not check.
///
/// Physics reads `pos`. History reads the reconstruction: mean-squared
/// displacement, diffusion, and any other quantity that must not see a
/// boundary crossing as a jump.
#[derive(Clone, Debug)]
pub struct MdState {
    /// Wrapped positions `(N, 3)` in Å — inside the primary cell.
    pub pos: Fnx3,
    /// Accumulated box crossings `(N, 3)`, one signed count per lattice
    /// vector, in the schema's image integer type ([`I`], i32) — the type of
    /// the `ix`/`iy`/`iz` columns it is written to. `pos + H·images` is the
    /// continuous position. An atom would have to cross the same face two
    /// billion times before the count overflowed, far beyond any physical run.
    pub images: Array2<I>,
    /// Velocities `(N, 3)` in Å/fs.
    pub vel: Fnx3,
    /// Cached forces `(N, 3)` at `pos`.
    pub forces: Fnx3,
    /// Cached scalar energy at `pos`.
    pub energy: F,
    /// Cached virial at `pos`, from the same evaluation as `forces`.
    ///
    /// It belongs to the force cache under the same invariant: the forces, the
    /// energy and the virial at `pos` all come from one evaluation, or the
    /// pressure and the trajectory describe different configurations.
    pub virial: Option<Virial>,
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::*;

    #[test]
    fn forces_field_is_plural() {
        let out = ForceOutput {
            energy: 1.0,
            forces: array![[0.0, 0.0, 0.0]],
            virial: None,
        };
        assert_eq!(out.forces.nrows(), 1);
        let state = MdState {
            pos: array![[0.0, 0.0, 0.0]],
            images: Array2::zeros((1, 3)),
            vel: array![[0.0, 0.0, 0.0]],
            forces: array![[1.0, 0.0, 0.0]],
            energy: 0.0,
            virial: None,
        };
        assert_eq!(state.forces[[0, 0]], 1.0);
    }
}

impl MdState {
    /// Write the canonical state into a frame's `atoms` block: wrapped
    /// positions **and** the image flags that go with them.
    ///
    /// The two travel together and this is why the method exists. A wrapped
    /// coordinate on its own has lost the atom's history — how many times it
    /// crossed, and therefore where it has actually been — and a trajectory
    /// written without the flags cannot answer a mean-squared displacement, a
    /// diffusion coefficient, or any other question that reads a path rather
    /// than a configuration. Nothing about such a file looks wrong; the
    /// analysis simply comes back with a number that is too small, because
    /// every crossing was read as the atom teleporting back.
    ///
    /// Writing them one at a time is the mistake this closes off: a caller with
    /// two calls will eventually make one of them.
    ///
    /// The columns are the canonical [`keys::COORDS`] and [`keys::IMAGES`], so
    /// a writer that emits whatever the block holds — the LAMMPS dump writer
    /// does — carries them without being told.
    pub fn apply_to(&self, frame: &mut Frame) -> Result<(), MdError> {
        let n = self.pos.nrows();
        if self.images.nrows() != n {
            return Err(MdError::Invalid(format!(
                "state has {n} positions but {} image flags",
                self.images.nrows()
            )));
        }
        let atoms = frame.get_mut(ATOMS).ok_or_else(|| {
            MdError::Invalid("frame has no \"atoms\" block to write into".to_string())
        })?;
        if let Some(rows) = atoms.n_rows()
            && rows != n
        {
            return Err(MdError::Invalid(format!(
                "state has {n} atoms but the frame's atoms block has {rows}"
            )));
        }

        atoms
            .set_coords(self.pos.view())
            .map_err(|e| MdError::Invalid(format!("atoms coordinates: {e}")))?;
        for (axis, key) in keys::IMAGES.iter().enumerate() {
            let col: Array1<I> = self.images.column(axis).to_owned();
            atoms
                .insert(*key, col.into_dyn(), None)
                .map_err(|e| MdError::Invalid(format!("atoms.{key}: {e}")))?;
        }
        Ok(())
    }

    /// The continuous positions, `pos + H·images`.
    ///
    /// This is the view history-dependent analysis wants — mean-squared
    /// displacement, diffusion, any measurement of a path. Force evaluation
    /// must not use it: the numbers grow without bound, and a potential handed
    /// them is being asked to subtract two large coordinates to recover a small
    /// separation.
    pub fn unwrapped(&self, bx: &SimBox) -> Fnx3 {
        bx.unwrap(self.pos.view(), self.images.view())
    }
}

#[cfg(test)]
mod persistence_tests {
    use super::*;
    use crate::core::Cell;
    use molrs::core::Block;
    use ndarray::{Array2, array};

    fn frame_with(n: usize) -> Frame {
        let mut atoms = Block::new();
        atoms
            .insert(
                "id",
                Array1::from((0..n as u64).collect::<Vec<_>>()).into_dyn(),
                None,
            )
            .unwrap();
        atoms
            .insert("x", Array1::from(vec![0.0_f64; n]).into_dyn(), None)
            .unwrap();
        atoms
            .insert("y", Array1::from(vec![0.0_f64; n]).into_dyn(), None)
            .unwrap();
        atoms
            .insert("z", Array1::from(vec![0.0_f64; n]).into_dyn(), None)
            .unwrap();
        let mut f = Frame::new();
        f.insert("atoms", atoms);
        f
    }

    /// Positions and image flags reach the frame together, under the canonical
    /// keys, so a writer that emits what the block holds carries both.
    #[test]
    fn the_state_writes_its_flags_beside_its_coordinates() {
        let state = MdState {
            pos: array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            images: array![[1 as I, 0, -2], [0, 3, 0]],
            vel: Array2::zeros((2, 3)),
            forces: Array2::zeros((2, 3)),
            energy: 0.0,
            virial: None,
        };
        let mut frame = frame_with(2);
        state.apply_to(&mut frame).unwrap();

        let atoms = frame.get("atoms").unwrap();
        for (axis, key) in keys::COORDS.iter().enumerate() {
            let col = atoms.get(key).and_then(|c| c.get::<f64>()).unwrap();
            for i in 0..2 {
                assert_eq!(col[[i]], state.pos[[i, axis]], "{key}[{i}]");
            }
        }
        for (axis, key) in keys::IMAGES.iter().enumerate() {
            let col = atoms
                .get(key)
                .and_then(|c| c.get::<i32>())
                .unwrap_or_else(|| panic!("atoms block is missing {key}"));
            for i in 0..2 {
                assert_eq!(col[[i]], state.images[[i, axis]], "{key}[{i}]");
            }
        }
    }

    /// The point of persisting the flags: the continuous trajectory survives a
    /// round trip through a frame, and a file written without them would not
    /// fail — it would quietly answer a displacement question with the wrong
    /// number.
    #[test]
    fn a_trajectory_written_with_flags_can_be_unwrapped_again() {
        let bx = SimBox::new(Cell::Lengths([10.0; 3]), [0.0_f64, 0.0, 0.0], [true; 3]).unwrap();
        // An atom that has crossed +x three times: stored at 2.0, really at 32.
        let state = MdState {
            pos: array![[2.0, 5.0, 5.0]],
            images: array![[3 as I, 0, 0]],
            vel: Array2::zeros((1, 3)),
            forces: Array2::zeros((1, 3)),
            energy: 0.0,
            virial: None,
        };
        assert!((state.unwrapped(&bx)[[0, 0]] - 32.0).abs() < 1e-12);

        let mut frame = frame_with(1);
        state.apply_to(&mut frame).unwrap();

        // Read it back the way an analysis would, from the frame alone.
        let atoms = frame.get("atoms").unwrap();
        let x = atoms.get("x").and_then(|c| c.get::<f64>()).unwrap()[[0]];
        let ix = atoms.get("ix").and_then(|c| c.get::<i32>()).unwrap()[[0]] as F;
        assert!(
            (x + ix * 10.0 - 32.0).abs() < 1e-12,
            "reconstructed {} from the frame, expected 32",
            x + ix * 10.0
        );

        // Without the flags the same frame says 2.0 — off by three cells, and
        // nothing about it looks wrong.
        assert!((x - 2.0).abs() < 1e-12);
    }

    /// A state and a frame that disagree about how many atoms they have is a
    /// caller error, and saying so beats writing a column of the wrong length.
    #[test]
    fn a_size_mismatch_is_refused() {
        let state = MdState {
            pos: array![[0.0, 0.0, 0.0]],
            images: Array2::zeros((1, 3)),
            vel: Array2::zeros((1, 3)),
            forces: Array2::zeros((1, 3)),
            energy: 0.0,
            virial: None,
        };
        let mut frame = frame_with(3);
        let err = state.apply_to(&mut frame).unwrap_err();
        assert!(format!("{err}").contains("1 atoms"), "{err}");
    }
}
