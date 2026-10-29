//! Read and write the atom columns a step touches.
//!
//! The hot step does not keep a parallel state struct.
//! Positions, velocities, forces, masses, and image flags are the `atoms`
//! columns. Scalar energy and kinetic energy are frame metadata.

use ndarray::{Array1, Array2};

use molrs::core::keys::{self, COORDS};
use molrs::core::schema::block_names::ATOMS;
use molrs::core::{Block, BlockDtype, Frame, MetaValue, Virial};
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

/// `atoms.{key}` as `n` contiguous `T`s to write in place; a column that is
/// absent, of another dtype, or not contiguous is replaced by zeros first.
fn column_mut<'a, T: BlockDtype + Default>(
    block: &'a mut Block,
    key: &str,
    n: usize,
) -> Result<&'a mut [T], MdError> {
    let fits = block
        .get(key)
        .and_then(|c| c.get::<T>())
        .is_some_and(|a| a.ndim() == 1 && a.len() == n && a.is_standard_layout());
    if !fits {
        block
            .insert(
                key,
                Array1::<T>::from_elem(n, T::default()).into_dyn(),
                None,
            )
            .map_err(|e| MdError::Invalid(format!("atoms.{key}: {e}")))?;
    }
    block
        .get_mut(key)
        .and_then(|c| c.get_mut::<T>())
        .and_then(|a| a.as_slice_mut())
        .ok_or_else(|| MdError::Invalid(format!("atoms.{key} is not a contiguous column")))
}

fn write_nx3(block: &mut Block, keys: [&str; 3], data: &Array2<F>) -> Result<(), MdError> {
    if data.ncols() != 3 {
        return Err(MdError::Invalid(format!(
            "expected (N, 3), got {:?}",
            data.shape()
        )));
    }
    for (axis, key) in keys.into_iter().enumerate() {
        let column = column_mut::<F>(block, key, data.nrows())?;
        for (dst, &v) in column.iter_mut().zip(data.column(axis)) {
            *dst = v;
        }
    }
    Ok(())
}

pub(crate) fn read_pos(frame: &Frame) -> Result<Fnx3, MdError> {
    let block = atoms(frame)?;
    read_nx3(block, COORDS, n_atoms(block)?)
}

/// The positions as flat `[x0, y0, z0, x1, …]` into `out`, resized to `3N`.
pub(crate) fn gather_pos(frame: &Frame, out: &mut Vec<F>) -> Result<(), MdError> {
    let block = atoms(frame)?;
    let n = n_atoms(block)?;
    out.resize(3 * n, 0.0);
    for (axis, key) in COORDS.into_iter().enumerate() {
        let col = block
            .get(key)
            .ok_or_else(|| MdError::Invalid(format!("atoms.{key} is missing")))?
            .get::<F>()
            .ok_or_else(|| MdError::Invalid(format!("atoms.{key} is not float64")))?;
        if col.len() != n {
            return Err(MdError::Invalid(format!(
                "atoms.{key} has {} rows, atoms has {n}",
                col.len()
            )));
        }
        for (i, &v) in col.iter().enumerate() {
            out[3 * i + axis] = v;
        }
    }
    Ok(())
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

/// Zero `fx`/`fy`/`fz` in place, inserting them when absent.
pub(crate) fn zero_force(frame: &mut Frame) -> Result<(), MdError> {
    let block = atoms_mut(frame)?;
    let n = n_atoms(block)?;
    for key in FORCE {
        column_mut::<F>(block, key, n)?.fill(0.0);
    }
    Ok(())
}

pub(crate) fn write_images(frame: &mut Frame, images: &Array2<I>) -> Result<(), MdError> {
    let block = atoms_mut(frame)?;
    for (axis, key) in keys::IMAGES.iter().enumerate() {
        let column = column_mut::<I>(block, key, images.nrows())?;
        for (dst, &v) in column.iter_mut().zip(images.column(axis)) {
            *dst = v;
        }
    }
    Ok(())
}

/// Add one force provider's flat `forces`, `energy` and `virial` onto what
/// the pre-force clear left.
///
/// The virial is `meta["virial"]` (`[xx, yy, zz, xy, xz, yz]`). A provider
/// without one removes it, since a virial missing a term is a different
/// quantity, not a smaller one.
pub(crate) fn add_force_terms(
    frame: &mut Frame,
    forces: &[F],
    energy: F,
    virial: Option<Virial>,
) -> Result<(), MdError> {
    let block = atoms_mut(frame)?;
    let n = n_atoms(block)?;
    if forces.len() != 3 * n {
        return Err(MdError::Invalid(format!(
            "a force provider produced {} atoms, the frame has {n}",
            forces.len() / 3
        )));
    }
    for (axis, key) in FORCE.into_iter().enumerate() {
        let column = column_mut::<F>(block, key, n)?;
        for (i, dst) in column.iter_mut().enumerate() {
            *dst += forces[3 * i + axis];
        }
    }
    let total = meta_f64(frame, "energy") + energy;
    set_meta_f64(frame, "energy", total);
    match (virial, frame.meta.get(VIRIAL)) {
        (Some(w), Some(MetaValue::F64x6(held))) => {
            let sum: [F; 6] = std::array::from_fn(|k| held[k] + w.components[k]);
            frame.meta.insert(VIRIAL, MetaValue::F64x6(sum));
        }
        _ => {
            frame.meta.remove(VIRIAL);
        }
    }
    Ok(())
}

/// `meta` key of the step's virial `Σ f ⊗ r`, `[xx, yy, zz, xy, xz, yz]`, in
/// the run's energy unit.
pub(crate) const VIRIAL: &str = "virial";

pub(crate) fn meta_f64(frame: &Frame, key: &str) -> F {
    frame.meta.get(key).and_then(|v| v.as_f64()).unwrap_or(0.0)
}

pub(crate) fn set_meta_f64(frame: &mut Frame, key: &str, value: F) {
    frame.meta.insert(key, MetaValue::F64(value));
}
