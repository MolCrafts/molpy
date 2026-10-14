//! The per-particle orientations a PMFT reads off a frame — one extractor,
//! shared by every binding.

use crate::compute::ComputeError;
use crate::core::Frame;
use crate::core::FrameAccess;
use crate::core::keys;
use crate::op::{F, Quat};

/// The `orientations` block an axis-derived orientation reads.
const ORIENTATIONS: &str = "orientations";

/// Per-atom unit quaternions `(w, i, j, k)` from the [`keys::QUAT`] columns of
/// the `atoms` block, each normalised (a zero quaternion reads as the
/// identity). `None` when any of the four columns is absent.
///
/// Two sources can state an orientation:
///
/// * the **quaternion columns** [`keys::QUAT`] (`quatw`, `quati`, `quatj`,
///   `quatk`) on the `atoms` block — the stored orientation of each particle;
/// * an **`orientations` topology block** of `(head, tail)` atom pairs, one per
///   query particle — an orientation derived from positions, the `head − tail`
///   axis of each particle in this frame.
///
/// A frame states at most one of them: carrying both is refused rather than
/// one silently winning over the other.
pub fn orientation_quaternions(frame: &Frame) -> Option<Vec<Quat>> {
    let atoms = frame.get("atoms")?;
    let cols = keys::QUAT.map(|key| atoms.get(key).and_then(|c| c.as_float()));
    let [Some(w), Some(i), Some(j), Some(k)] = cols else {
        return None;
    };
    Some(
        w.iter()
            .zip(i.iter())
            .zip(j.iter())
            .zip(k.iter())
            .map(|(((&w, &i), &j), &k)| {
                let norm = (w * w + i * i + j * j + k * k).sqrt();
                if norm > 0.0 {
                    [w / norm, i / norm, j / norm, k / norm]
                } else {
                    [1.0, 0.0, 0.0, 0.0]
                }
            })
            .collect(),
    )
}

/// Per-particle orientation angles in the xy plane (radians), for the 2-D
/// PMFTs.
///
/// From the quaternion columns, the z-rotation of each quaternion,
/// `θ = 2·atan2(q_k, q_w)`; from an `orientations` block, the angle of each
/// `head − tail` axis, `θ = atan2(Δy, Δx)`, at this frame's positions.
/// `Ok(None)` when the frame states no orientation (the lab frame).
///
/// # Errors
///
/// [`ComputeError::BadShape`] when the frame carries both sources;
/// [`ComputeError::OutOfRange`] when an `orientations` row names an atom the
/// frame does not have; whatever reading the block or the coordinates raises.
pub fn planar_orientation_angles(frame: &Frame) -> Result<Option<Vec<F>>, ComputeError> {
    let quaternions = orientation_quaternions(frame);
    let has_axes = frame.get(ORIENTATIONS).is_some();
    match (quaternions, has_axes) {
        (Some(_), true) => Err(ComputeError::BadShape {
            expected: format!(
                "one orientation source: the {} columns or an '{ORIENTATIONS}' block",
                keys::QUAT.join("/")
            ),
            got: "both".into(),
        }),
        (Some(quats), false) => Ok(Some(quats.iter().map(|q| 2.0 * q[3].atan2(q[0])).collect())),
        (None, true) => {
            // The block's first two endpoint columns: (head, tail) per row.
            let mut ends: Vec<Vec<crate::op::Idx>> = Vec::with_capacity(2);
            for &col in &keys::ENDPOINTS[..2] {
                let view = frame
                    .column(ORIENTATIONS, col)
                    .and_then(|c| c.as_uint())
                    .ok_or(ComputeError::MissingColumn {
                        block: ORIENTATIONS,
                        col,
                    })?;
                ends.push(view.iter().copied().collect());
            }
            if ends[0].len() != ends[1].len() {
                return Err(ComputeError::BadShape {
                    expected: "2 endpoint columns of equal length".into(),
                    got: "mismatched endpoint column lengths".into(),
                });
            }
            let xyz = frame.coords().map_err(|e| ComputeError::OutOfRange {
                field: "orientations coordinates",
                value: e.to_string(),
            })?;
            let n = xyz.nrows();
            (0..ends[0].len())
                .map(|g| {
                    let (head, tail) = (ends[0][g] as usize, ends[1][g] as usize);
                    if head >= n || tail >= n {
                        return Err(ComputeError::OutOfRange {
                            field: "orientations atom index",
                            value: format!("({head}, {tail}) with {n} atoms"),
                        });
                    }
                    Ok((xyz[[head, 1]] - xyz[[tail, 1]]).atan2(xyz[[head, 0]] - xyz[[tail, 0]]))
                })
                .collect::<Result<Vec<F>, _>>()
                .map(Some)
        }
        (None, false) => Ok(None),
    }
}

/// An angle wrapped into `[0, 2π)` (radians), the range the orientation-bin
/// axes of [`super::PmftXyt`] and [`super::PmftR12`] span.
#[inline]
pub(super) fn wrap_2pi(a: F) -> F {
    a.rem_euclid(std::f64::consts::TAU)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::Block;
    use crate::op::Idx;
    use ndarray::Array1;

    fn frame_with_atoms(x: Vec<F>, y: Vec<F>) -> Frame {
        let mut atoms = Block::new();
        let z = vec![0.0; x.len()];
        atoms.insert("x", Array1::from_vec(x).into_dyn()).unwrap();
        atoms.insert("y", Array1::from_vec(y).into_dyn()).unwrap();
        atoms.insert("z", Array1::from_vec(z).into_dyn()).unwrap();
        let mut frame = Frame::new();
        frame.insert("atoms", atoms);
        frame
    }

    fn with_quaternion(mut frame: Frame, q: [F; 4]) -> Frame {
        let atoms = frame.get_mut("atoms").unwrap();
        let n = atoms.n_rows().unwrap();
        for (key, value) in keys::QUAT.iter().zip(q) {
            atoms
                .insert(*key, Array1::from_elem(n, value).into_dyn())
                .unwrap();
        }
        frame
    }

    #[test]
    fn no_source_is_the_lab_frame() {
        let frame = frame_with_atoms(vec![0.0], vec![0.0]);
        assert_eq!(planar_orientation_angles(&frame).unwrap(), None);
        assert!(orientation_quaternions(&frame).is_none());
    }

    #[test]
    fn a_quaternion_gives_its_z_rotation() {
        let half = std::f64::consts::FRAC_PI_4; // a 90° turn about z
        let frame = with_quaternion(
            frame_with_atoms(vec![0.0], vec![0.0]),
            [half.cos(), 0.0, 0.0, half.sin()],
        );
        let angles = planar_orientation_angles(&frame).unwrap().unwrap();
        assert!((angles[0] - std::f64::consts::FRAC_PI_2).abs() < 1e-12);
    }

    #[test]
    fn a_head_tail_axis_gives_its_angle() {
        let mut frame = frame_with_atoms(vec![0.0, 0.0], vec![0.0, 1.0]);
        let mut block = Block::new();
        block
            .insert("atomi", Array1::<Idx>::from_vec(vec![1]).into_dyn())
            .unwrap();
        block
            .insert("atomj", Array1::<Idx>::from_vec(vec![0]).into_dyn())
            .unwrap();
        frame.insert(ORIENTATIONS, block);
        let angles = planar_orientation_angles(&frame).unwrap().unwrap();
        assert!((angles[0] - std::f64::consts::FRAC_PI_2).abs() < 1e-12);

        let both = with_quaternion(frame, [1.0, 0.0, 0.0, 0.0]);
        assert!(planar_orientation_angles(&both).is_err());
    }
}
