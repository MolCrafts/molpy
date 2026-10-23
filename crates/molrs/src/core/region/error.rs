//! Why a region primitive refuses its parameters: [`RegionError`].

use crate::core::BoxError;
use crate::op::F;

/// Why [`Cylinder`](super::Cylinder), [`Ellipsoid`](super::Ellipsoid),
/// [`HalfSpace`](super::HalfSpace) or
/// [`Parallelepiped`](super::Parallelepiped) cannot be built from its
/// parameters. Each variant names the shape and the parameter it refuses.
#[derive(Debug, Clone)]
pub enum RegionError {
    /// A vector that fixes a direction (an axis, a normal) is zero or not
    /// finite.
    NoDirection {
        /// The shape, e.g. `"Cylinder"`.
        shape: &'static str,
        /// The parameter, e.g. `"axis"`.
        param: &'static str,
    },
    /// A length (a radius, a semi-axis) is not positive and finite.
    NonPositive {
        /// The shape.
        shape: &'static str,
        /// The parameter.
        param: &'static str,
        /// The refused value.
        value: F,
    },
    /// A point (a centre, a base, a point on a plane) is not finite.
    NonFinite {
        /// The shape.
        shape: &'static str,
        /// The parameter.
        param: &'static str,
    },
    /// A [`Cell::Undefined`](crate::core::Cell::Undefined) has no edges to
    /// span a solid.
    UndefinedCell,
    /// The cell is one [`Cell`](crate::core::Cell) refuses, singular, or not
    /// finite.
    Cell(BoxError),
}

impl std::fmt::Display for RegionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoDirection { shape, param } => {
                write!(f, "{shape}: {param} must be a finite non-zero vector")
            }
            Self::NonPositive {
                shape,
                param,
                value,
            } => write!(f, "{shape}: {param} must be > 0, got {value}"),
            Self::NonFinite { shape, param } => write!(f, "{shape}: {param} must be finite"),
            Self::UndefinedCell => write!(
                f,
                "Parallelepiped: Cell::Undefined has no edges; give the cell matrix or its lengths"
            ),
            Self::Cell(err) => write!(f, "Parallelepiped: {err}"),
        }
    }
}

impl std::error::Error for RegionError {}
