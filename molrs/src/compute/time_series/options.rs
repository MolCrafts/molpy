//! The configuration of a [`Correlator`](super::Correlator): which lag kernel
//! it evaluates and how the lagged sums become a curve.

use molrs::core::legendre_plm;
use molrs::op::{F, Quat, quat_conj, quat_mul, quat_norm};

use crate::compute::ComputeError;
use crate::compute::named_enum::named_enum;

/// What a [`Correlator`](super::Correlator) evaluates between a sample at time
/// `τ` and one at `τ + t`.
///
/// The set is closed — it is the set of time correlations molrs ships — so
/// it is an enum, not a generic: one concrete `Correlator` type binds to every
/// language with a kernel argument.
///
/// | Kernel | Per item, between `x(τ)` and `y(τ+t)` | Evaluation |
/// |---|---|---|
/// | `Product` | `Σ_d x_d(τ) · y_d(τ+t)` | FFT (Wiener–Khinchin) |
/// | `SquaredDisplacement` | `Σ_d (x_d(τ+t) − x_d(τ)) · (y_d(τ+t) − y_d(τ))` | FFT + `S` recurrence |
/// | `Legendre { l }` | `P_l(x̂(τ) · x̂(τ+t))` of the normalised vectors | direct |
/// | `WignerD { l }` | `χ_l(θ) / (2l + 1)`, `θ` the angle of `q(τ+t) · q(τ)*` | direct |
/// | `Persistence` | `1` if the 0/1 presence stayed `1` over `[τ, τ+t]` | direct |
///
/// `Product` and `SquaredDisplacement` are linear in each component, so they
/// decompose over components and evaluate in `O(T log T)` per column; the
/// other three are nonlinear in the item's vector and are summed directly
/// (`O(T · L)` per item).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LagKernel {
    /// Lagged product, auto (`x = y`) or cross.
    Product,
    /// Lagged squared displacement, auto (the MSD) or cross (Onsager).
    SquaredDisplacement,
    /// Legendre polynomial `P_l` of the cosine between two orientation
    /// vectors (any length; each is normalised, zero length is an error).
    Legendre {
        /// Polynomial order.
        l: u32,
    },
    /// Normalised Wigner-D character of the relative rotation between two
    /// orientation quaternions `(w, x, y, z)` (zero norm is an error).
    WignerD {
        /// Angular order.
        l: u32,
    },
    /// Continuous persistence of a 0/1 presence series.
    Persistence,
}

named_enum! {
    /// The kind of a [`LagKernel`], by its canonical name. `legendre` and
    /// `wigner_d` also take an order `l` ([`LagKernel::new`]).
    pub enum LagKernelKind {
        /// [`LagKernel::Product`].
        Product => "product",
        /// [`LagKernel::SquaredDisplacement`].
        SquaredDisplacement => "squared_displacement",
        /// [`LagKernel::Legendre`].
        Legendre => "legendre",
        /// [`LagKernel::WignerD`].
        WignerD => "wigner_d",
        /// [`LagKernel::Persistence`].
        Persistence => "persistence",
    }
}

impl LagKernel {
    /// The kernel of `kind`, with its order `l`.
    ///
    /// # Errors
    /// [`ComputeError::OutOfRange`] when `l` is missing for `legendre` /
    /// `wigner_d`, which require it, or given for any other kind, which
    /// takes none.
    pub fn new(kind: LagKernelKind, l: Option<u32>) -> Result<Self, ComputeError> {
        use LagKernelKind as K;
        match (kind, l) {
            (K::Product, None) => Ok(Self::Product),
            (K::SquaredDisplacement, None) => Ok(Self::SquaredDisplacement),
            (K::Persistence, None) => Ok(Self::Persistence),
            (K::Legendre, Some(l)) => Ok(Self::Legendre { l }),
            (K::WignerD, Some(l)) => Ok(Self::WignerD { l }),
            (K::Legendre | K::WignerD, None) => Err(ComputeError::OutOfRange {
                field: "l (required by legendre / wigner_d)",
                value: format!("{kind} without l"),
            }),
            (K::Product | K::SquaredDisplacement | K::Persistence, Some(l)) => {
                Err(ComputeError::OutOfRange {
                    field: "l (only legendre / wigner_d take one)",
                    value: format!("{kind} with l = {l}"),
                })
            }
        }
    }

    /// The kind of this kernel.
    pub const fn kind(self) -> LagKernelKind {
        match self {
            Self::Product => LagKernelKind::Product,
            Self::SquaredDisplacement => LagKernelKind::SquaredDisplacement,
            Self::Legendre { .. } => LagKernelKind::Legendre,
            Self::WignerD { .. } => LagKernelKind::WignerD,
            Self::Persistence => LagKernelKind::Persistence,
        }
    }

    /// The order `l` of `legendre` / `wigner_d`; `None` for the other kinds.
    pub const fn order(self) -> Option<u32> {
        match self {
            Self::Legendre { l } | Self::WignerD { l } => Some(l),
            _ => None,
        }
    }

    /// Whether the kernel is a sum over components (and so FFT-evaluable and
    /// reducible per component).
    pub(super) const fn is_linear(self) -> bool {
        matches!(self, Self::Product | Self::SquaredDisplacement)
    }

    /// The number of components an item must have, if the kernel fixes it.
    pub(super) const fn required_components(self) -> Option<usize> {
        match self {
            Self::WignerD { .. } => Some(4),
            Self::Persistence => Some(1),
            _ => None,
        }
    }

    /// The per-item value of a nonlinear kernel between an earlier sample
    /// `early` and a later one `late`. Legendre vectors arrive normalised;
    /// quaternions arrive validated non-zero.
    pub(super) fn item_value(self, early: &[F], late: &[F]) -> F {
        match self {
            Self::Legendre { l } => {
                let cosine: F = early.iter().zip(late).map(|(a, b)| a * b).sum();
                legendre_plm(l, 0, cosine)
            }
            Self::WignerD { l } => {
                let q_ref: Quat = [early[0], early[1], early[2], early[3]];
                let q_now: Quat = [late[0], late[1], late[2], late[3]];
                normalised_character(l, rotation_angle(quat_mul(q_now, quat_conj(q_ref))))
            }
            Self::Product | Self::SquaredDisplacement | Self::Persistence => {
                unreachable!("item_value is only evaluated for the orientation kernels")
            }
        }
    }
}

impl std::fmt::Display for LagKernel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.order() {
            Some(l) => write!(f, "{}(l={l})", self.kind()),
            None => write!(f, "{}", self.kind()),
        }
    }
}

/// Rotation angle of a (non-zero) quaternion. `|w|` folds `q` and `−q`, which
/// are the same rotation.
fn rotation_angle(q: Quat) -> F {
    2.0 * (q[0].abs() / quat_norm(q)).clamp(0.0, 1.0).acos()
}

/// `χ_l(θ) / (2l + 1)` with `χ_l(θ) = sin((l + ½)θ) / sin(θ/2)`, continuous
/// (`= 1`) at `θ = 0`.
fn normalised_character(l: u32, theta: F) -> F {
    if theta.abs() < 1e-12 {
        return 1.0;
    }
    let num = ((l as F + 0.5) * theta).sin();
    let den = (0.5 * theta).sin();
    if den == 0.0 {
        1.0
    } else {
        num / den / (2.0 * l as F + 1.0)
    }
}

named_enum! {
    /// Which time origins `τ` each lag `t` is averaged over.
    #[derive(Default)]
    pub enum Origins {
        /// Every origin `τ = 0 … T − 1 − t` (the windowed estimator).
        #[default]
        All => "all",
        /// The first frame only, `τ = 0` (a single reference).
        First => "first",
    }
}

named_enum! {
    /// Whether each `(item, component)` column loses its own time mean first.
    #[derive(Default)]
    pub enum MeanPolicy {
        /// Correlate the values as given.
        #[default]
        Keep => "keep",
        /// Correlate the fluctuations `x − ⟨x⟩_T` (mean over all `T` frames).
        Subtract => "subtract",
    }
}

named_enum! {
    /// How the lagged sum at lag `t` is scaled.
    #[derive(Default)]
    pub enum Normalization {
        /// Divide by the number of origins that exist at that lag (`T − t` for
        /// [`Origins::All`], `1` for [`Origins::First`]) — the unbiased
        /// estimator.
        #[default]
        Unbiased => "unbiased",
        /// Leave the raw sum over origins.
        None => "none",
    }
}

named_enum! {
    /// How items and components combine into output channels.
    #[derive(Default)]
    pub enum Reduction {
        /// One channel: sum over components, mean over items.
        #[default]
        Mean => "mean",
        /// One channel per item: sum over its components.
        PerItem => "per_item",
        /// One channel per component (linear kernels only): mean over items.
        PerComponent => "per_component",
    }
}
