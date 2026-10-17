//! Read and write the atom columns a step touches.
//!
//! The hot step does not keep a parallel [`MdState`](crate::md::MdState).
//! Positions, velocities, forces, masses, and image flags are the `atoms`
//! columns. Scalar energy and kinetic energy are frame metadata.

use ndarray::{Array1, Array2};

use molrs::core::keys::{self, COORDS};
use molrs::core::schema::block_names::ATOMS;
use molrs::core::{Block, Frame};
use molrs::op::{F, Fnx3, I};

use super::error::MdError;

const VEL: [&str; 3] = [keys::VX, keys::VY, keys::VZ];
const FORCE: [&str; 3] = [keys::FX, keys::FY, keys::FZ];

pub(crate) fn atoms(frame: &Frame) -> Result<&Block, MdError> {
    frame
        .get(ATOMS)
        .ok_or_else(|| MdError::Invalid("frame has no \"atoms\" block".into()))
}

pub(crate) fn atoms_mut(frame: &mut Frame) -> Result<&mut Block, MdError> {
    frame
        .get_mut(ATOMS)
        .ok_or_else(|| MdError::Invalid("frame has no \"atoms\" block".into()))
}

pub(crate) fn n_atoms(block: &Block) -> Result<usize, MdError> {
    block.n_rows().ok_or_else(|| {
        MdError::Invalid("atoms block has no rows; set coordinates before run".into())
    })
}

fn read_f64(block: &Block, key: &str, n: usize) -> Result<Array1<F>, MdError> {
    let col = block
        .get(key)
        .ok_or_else(|| MdError::Invalid(format!("atoms.{key} is missing")))?;
    let values = col
        .get::<F>()
        .ok_or_else(|| MdError::Invalid(format!("atoms.{key} is not float64")))?;
    if values.len() != n {
        return Err(MdError::Invalid(format!(
            "atoms.{key} has {} rows, atoms has {n}",
            values.len()
        )));
    }
    Ok(Array1::from_iter(values.iter().copied()))
}

fn read_nx3(block: &Block, keys: [&str; 3], n: usize) -> Result<Fnx3, MdError> {
    let mut out = Fnx3::zeros((n, 3));
    for (axis, key) in keys.into_iter().enumerate() {
        let col = read_f64(block, key, n)?;
        out.column_mut(axis).assign(&col);
    }
    Ok(out)
}

fn write_nx3(block: &mut Block, keys: [&str; 3], data: &Array2<F>) -> Result<(), MdError> {
    if data.ncols() != 3 {
        return Err(MdError::Invalid(format!(
            "expected (N, 3), got {:?}",
            data.shape()
        )));
    }
    for (axis, key) in keys.into_iter().enumerate() {
        block
            .insert(key, data.column(axis).to_owned().into_dyn(), None)
            .map_err(|e| MdError::Invalid(format!("atoms.{key}: {e}")))?;
    }
    Ok(())
}

pub(crate) fn read_pos(frame: &Frame) -> Result<Fnx3, MdError> {
    let block = atoms(frame)?;
    read_nx3(block, COORDS, n_atoms(block)?)
}

pub(crate) fn read_vel(frame: &Frame) -> Result<Fnx3, MdError> {
    let block = atoms(frame)?;
    read_nx3(block, VEL, n_atoms(block)?)
}

pub(crate) fn read_force(frame: &Frame) -> Result<Fnx3, MdError> {
    let block = atoms(frame)?;
    let n = n_atoms(block)?;
    if FORCE.iter().all(|key| block.get(key).is_none()) {
        return Ok(Fnx3::zeros((n, 3)));
    }
    read_nx3(block, FORCE, n)
}

pub(crate) fn read_mass(frame: &Frame) -> Result<Array1<F>, MdError> {
    let block = atoms(frame)?;
    let mass = read_f64(block, keys::MASS, n_atoms(block)?)?;
    if mass.iter().any(|m| !m.is_finite() || *m <= 0.0) {
        return Err(MdError::Invalid(
            "atoms.mass must be strictly positive".into(),
        ));
    }
    Ok(mass)
}

pub(crate) fn read_images(frame: &Frame) -> Result<Array2<I>, MdError> {
    let block = atoms(frame)?;
    let n = n_atoms(block)?;
    let mut out = Array2::<I>::zeros((n, 3));
    for (axis, key) in keys::IMAGES.iter().enumerate() {
        let Some(col) = block.get(key) else {
            continue;
        };
        let values = col
            .get::<I>()
            .ok_or_else(|| MdError::Invalid(format!("atoms.{key} is not int32")))?;
        if values.len() != n {
            return Err(MdError::Invalid(format!(
                "atoms.{key} has {} rows, atoms has {n}",
                values.len()
            )));
        }
        for (dst, &v) in out.column_mut(axis).iter_mut().zip(values.iter()) {
            *dst = v;
        }
    }
    Ok(out)
}

/// Atom id when the column is present, otherwise the row index.
///
/// The Langevin stream is keyed by this so a reorder of storage does not
/// silently change the noise of a tagged atom. A missing `id` column falls
/// back to the row, which is stable for a run that does not migrate atoms.
pub(crate) fn read_tags(frame: &Frame) -> Result<Vec<u64>, MdError> {
    let block = atoms(frame)?;
    let n = n_atoms(block)?;
    let Some(col) = block.get(keys::ID) else {
        return Ok((0..n as u64).collect());
    };
    if let Some(values) = col.get::<u64>() {
        if values.len() != n {
            return Err(MdError::Invalid(format!(
                "atoms.id has {} rows, atoms has {n}",
                values.len()
            )));
        }
        return Ok(values.iter().copied().collect());
    }
    if let Some(values) = col.get::<u32>() {
        if values.len() != n {
            return Err(MdError::Invalid(format!(
                "atoms.id has {} rows, atoms has {n}",
                values.len()
            )));
        }
        return Ok(values.iter().map(|&v| u64::from(v)).collect());
    }
    Err(MdError::Invalid(
        "atoms.id is neither uint32 nor uint64".into(),
    ))
}

pub(crate) fn write_pos(frame: &mut Frame, pos: &Fnx3) -> Result<(), MdError> {
    write_nx3(atoms_mut(frame)?, COORDS, pos)
}

pub(crate) fn write_vel(frame: &mut Frame, vel: &Fnx3) -> Result<(), MdError> {
    write_nx3(atoms_mut(frame)?, VEL, vel)
}

pub(crate) fn write_force(frame: &mut Frame, force: &Fnx3) -> Result<(), MdError> {
    write_nx3(atoms_mut(frame)?, FORCE, force)
}

pub(crate) fn write_images(frame: &mut Frame, images: &Array2<I>) -> Result<(), MdError> {
    let block = atoms_mut(frame)?;
    for (axis, key) in keys::IMAGES.iter().enumerate() {
        let col: Array1<I> = images.column(axis).to_owned();
        block
            .insert(*key, col.into_dyn(), None)
            .map_err(|e| MdError::Invalid(format!("atoms.{key}: {e}")))?;
    }
    Ok(())
}

pub(crate) fn meta_f64(frame: &Frame, key: &str) -> F {
    frame.meta.get(key).and_then(|v| v.as_f64()).unwrap_or(0.0)
}

pub(crate) fn set_meta_f64(frame: &mut Frame, key: &str, value: F) {
    frame.meta.insert(key, molrs::core::MetaValue::F64(value));
}
