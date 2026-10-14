//! [`Unwrap`] — continuous coordinates from a trajectory wrapped into its box.

use molrs::core::SimBox;
use molrs::op::{F, I};
use ndarray::{Array3, ArrayView3};

use crate::compute::{Compute, ComputeError};

/// Undo the periodic wrapping of a trajectory, so a displacement analysis
/// ([`Msd`](super::Msd), [`OnsagerCorrelation`](super::OnsagerCorrelation))
/// sees how far each item really moved.
///
/// Two routes, chosen by the data given:
///
/// - **Image flags** (`images` given): `r(t) + H(t) · n(t)`, with `n` the
///   integer box crossings a simulation engine records and `H` the box
///   matrix (lattice vectors as columns). Exact at any step size.
/// - **Continuity** (`images` absent): the first frame is kept, and each
///   later frame adds the minimum-image step from the previous one,
///   `r_u(t) = r_u(t−1) + mic(r(t) − r(t−1))`. Exact as long as no item moves
///   more than half the box between frames — the same bound the displacement
///   kernels enforce with [`ComputeError::Wrapped`].
///
/// ```
/// use molrs::compute::{Compute, Unwrap, UnwrapInput};
/// use molrs::core::SimBox;
/// use ndarray::{Array3, array};
///
/// // One atom stepping +1 Å per frame across the face of a 10 Å cube.
/// let wrapped = Array3::from_shape_vec((3, 1, 3), vec![
///     9.0, 0.0, 0.0,
///     0.0, 0.0, 0.0, // 10 Å, wrapped to 0
///     1.0, 0.0, 0.0,
/// ]).unwrap();
/// let cube = SimBox::cube(10.0, array![0.0, 0.0, 0.0], [true; 3]).unwrap();
/// let boxes = vec![cube; 3];
/// let unwrapped = Unwrap::new()
///     .compute(UnwrapInput { positions: wrapped.view(), images: None, boxes: &boxes })
///     .unwrap();
/// assert!((unwrapped[[2, 0, 0]] - 11.0).abs() < 1e-12);
/// ```
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Unwrap;

/// The data [`Unwrap`] unwraps.
#[derive(Debug, Clone, Copy)]
pub struct UnwrapInput<'a> {
    /// Wrapped positions, `T × N × 3`.
    pub positions: ArrayView3<'a, F>,
    /// Box crossings per item and axis, `T × N × 3`; `None` to unwrap by
    /// continuity.
    pub images: Option<ArrayView3<'a, I>>,
    /// One box per frame.
    pub boxes: &'a [SimBox],
}

impl Unwrap {
    /// An unwrapper; the route is set by the data passed to `compute`.
    pub fn new() -> Self {
        Self
    }
}

impl Compute for Unwrap {
    type Input<'a> = UnwrapInput<'a>;
    type Output = Array3<F>;

    fn compute(&self, input: UnwrapInput<'_>) -> Result<Array3<F>, ComputeError> {
        let positions = input.positions;
        let (n_frames, n_items, n_components) = positions.dim();
        if n_frames == 0 || n_items == 0 {
            return Err(ComputeError::EmptyInput);
        }
        if n_components != 3 {
            return Err(ComputeError::DimensionMismatch {
                expected: 3,
                got: n_components,
                what: "components per position",
            });
        }
        if input.boxes.len() != n_frames {
            return Err(ComputeError::DimensionMismatch {
                expected: n_frames,
                got: input.boxes.len(),
                what: "boxes (one per frame)",
            });
        }
        if let Some(index) = positions.iter().position(|v| !v.is_finite()) {
            return Err(ComputeError::NonFinite {
                where_: "positions",
                index,
            });
        }
        let mut out = positions.to_owned();
        match input.images {
            Some(images) => {
                if images.dim() != positions.dim() {
                    return Err(ComputeError::BadShape {
                        expected: format!(
                            "images of shape {:?}, like positions",
                            positions.shape()
                        ),
                        got: format!("{:?}", images.shape()),
                    });
                }
                for (frame, (mut sample, crossings)) in
                    out.outer_iter_mut().zip(images.outer_iter()).enumerate()
                {
                    let h = input.boxes[frame].matrix();
                    for (mut r, n) in sample.outer_iter_mut().zip(crossings.outer_iter()) {
                        for (d, row) in h.iter().enumerate() {
                            r[d] += row[0] * n[0] as F + row[1] * n[1] as F + row[2] * n[2] as F;
                        }
                    }
                }
            }
            None => {
                for frame in 1..n_frames {
                    let mic = input.boxes[frame].mic();
                    for item in 0..n_items {
                        let step = mic.apply([
                            positions[[frame, item, 0]] - positions[[frame - 1, item, 0]],
                            positions[[frame, item, 1]] - positions[[frame - 1, item, 1]],
                            positions[[frame, item, 2]] - positions[[frame - 1, item, 2]],
                        ]);
                        for (d, delta) in step.into_iter().enumerate() {
                            out[[frame, item, d]] = out[[frame - 1, item, d]] + delta;
                        }
                    }
                }
            }
        }
        Ok(out)
    }
}

crate::compute::spec::compute_spec! {
    Unwrap as "Unwrap";
    params |s| {}
    inputs {
        positions: Trajectory;
        images: Images, optional;
        boxes: Boxes;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute::time_series::test_support::trajectory;
    use ndarray::array;

    fn ortho(lengths: [F; 3]) -> SimBox {
        SimBox::ortho(
            array![lengths[0], lengths[1], lengths[2]],
            array![0.0, 0.0, 0.0],
            [true; 3],
        )
        .unwrap()
    }

    /// Two atoms in a 10 × 8 × 6 box: atom 0 walks +3 Å/frame along x
    /// (crossing at frames 3 and 6), atom 1 walks −2 Å/frame along z.
    fn walkers() -> (Array3<F>, Array3<I>, Array3<F>) {
        let lengths = [10.0, 8.0, 6.0];
        let n = 7;
        let unwrapped = Array3::from_shape_fn((n, 2, 3), |(t, i, d)| match (i, d) {
            (0, 0) => 1.0 + 3.0 * t as F,
            (0, _) => 1.0,
            (1, 2) => 5.0 - 2.0 * t as F,
            (1, _) => 2.0,
            _ => unreachable!(),
        });
        let images = unwrapped.mapv(|_| 0)
            + Array3::from_shape_fn((n, 2, 3), |(t, i, d)| {
                (unwrapped[[t, i, d]] / lengths[d]).floor() as I
            });
        let wrapped = Array3::from_shape_fn((n, 2, 3), |(t, i, d)| {
            unwrapped[[t, i, d]] - images[[t, i, d]] as F * lengths[d]
        });
        (wrapped, images, unwrapped)
    }

    fn unwrap(
        wrapped: &Array3<F>,
        images: Option<&Array3<I>>,
        boxes: &[SimBox],
    ) -> Result<Array3<F>, ComputeError> {
        Unwrap::new().compute(UnwrapInput {
            positions: wrapped.view(),
            images: images.map(|i| i.view()),
            boxes,
        })
    }

    #[test]
    fn image_flags_restore_the_unwrapped_trajectory() {
        let (wrapped, images, unwrapped) = walkers();
        assert!(
            images.iter().any(|&n| n != 0),
            "the walk must cross the box"
        );
        let boxes = vec![ortho([10.0, 8.0, 6.0]); 7];
        let got = unwrap(&wrapped, Some(&images), &boxes).unwrap();
        for (g, u) in got.iter().zip(unwrapped.iter()) {
            assert!((g - u).abs() < 1e-12);
        }
    }

    #[test]
    fn continuity_restores_the_trajectory_up_to_the_first_frame() {
        let (wrapped, _, unwrapped) = walkers();
        let boxes = vec![ortho([10.0, 8.0, 6.0]); 7];
        let got = unwrap(&wrapped, None, &boxes).unwrap();
        // Frame 0 is kept as given; every later frame follows by continuity.
        for t in 0..7 {
            for i in 0..2 {
                for d in 0..3 {
                    let shift = unwrapped[[0, i, d]] - wrapped[[0, i, d]];
                    assert!((got[[t, i, d]] + shift - unwrapped[[t, i, d]]).abs() < 1e-12);
                }
            }
        }
    }

    #[test]
    fn image_flags_follow_a_tilted_cell() {
        // One crossing of the b lattice vector (0, 4, 0) + (3, 0, 0) tilt.
        let cell = SimBox::from_matrix(
            [[5.0, 3.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 6.0]],
            [0.0; 3],
            [true; 3],
        )
        .unwrap();
        let wrapped = trajectory(&[vec![[1.0, 1.0, 1.0]], vec![[1.0, 0.5, 1.0]]]);
        let images = Array3::from_shape_vec((2, 1, 3), vec![0, 0, 0, 0, 1, 0]).unwrap();
        let got = unwrap(&wrapped, Some(&images), &[cell.clone(), cell]).unwrap();
        assert_eq!(
            got.slice(ndarray::s![1, 0, ..]).to_vec(),
            vec![4.0, 4.5, 1.0]
        );
    }

    #[test]
    fn refusals() {
        let (wrapped, images, _) = walkers();
        let boxes = vec![ortho([10.0, 8.0, 6.0]); 7];
        assert!(matches!(
            unwrap(&wrapped, None, &boxes[..6]),
            Err(ComputeError::DimensionMismatch {
                expected: 7,
                got: 6,
                ..
            })
        ));
        let short_images = images.slice(ndarray::s![..6, .., ..]).to_owned();
        assert!(matches!(
            unwrap(&wrapped, Some(&short_images), &boxes),
            Err(ComputeError::BadShape { .. })
        ));
        let planar = Array3::<F>::zeros((7, 2, 2));
        assert!(matches!(
            unwrap(&planar, None, &boxes),
            Err(ComputeError::DimensionMismatch { expected: 3, .. })
        ));
        let mut nan = wrapped.clone();
        nan[[2, 1, 1]] = F::NAN;
        assert!(matches!(
            unwrap(&nan, None, &boxes),
            Err(ComputeError::NonFinite { index: 16, .. })
        ));
        let empty = Array3::<F>::zeros((0, 2, 3));
        assert!(matches!(
            unwrap(&empty, None, &[]),
            Err(ComputeError::EmptyInput)
        ));
    }
}
