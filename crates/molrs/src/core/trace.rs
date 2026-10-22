//! An ordered path of 3D points: [`Trace`].

use crate::op::Vec3;

/// An ordered path of 3D points.
///
/// A trace has no chemistry: it says *where* consecutive samples of a structure
/// lie, not *what* sits there. Points are in the caller's length unit, Å
/// throughout molrs.
///
/// `builder::SelfAvoidingWalk` generates one trace per chain; the generator is
/// not the trace.
#[derive(Debug, Clone, PartialEq)]
pub struct Trace {
    points: Vec<Vec3>,
}

impl Trace {
    /// The path through `points`, in order. An empty list is an empty trace.
    pub fn from_points(points: Vec<Vec3>) -> Self {
        Self { points }
    }

    /// Every point, in order.
    pub fn points(&self) -> &[Vec3] {
        &self.points
    }
}

#[cfg(test)]
mod tests {
    use super::Trace;

    #[test]
    fn from_points_keeps_the_points_in_order() {
        let pts = vec![[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [-1.0, 0.5, 0.0]];
        let t = Trace::from_points(pts.clone());
        assert_eq!(t.points(), pts.as_slice());
    }
}
