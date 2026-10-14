//! [`ComputeSpec`] — the machine-readable descriptor of a compute type.
//!
//! Every public analysis in this module describes itself through
//! [`Described`]: its **name**, its **configuration parameters** (name, kind,
//! default) and the **fields of its `Input`** (name, kind, optional). The
//! descriptor is the single source of truth bindings and catalogs read instead
//! of retyping a signature:
//!
//! - a binding's defaults-parity test builds each compute with no optional
//!   argument and compares [`Described::tunables`] with the spec's defaults;
//! - a binding's name-parity test compares its class names with
//!   [`ComputeSpec::all`];
//! - a UI catalog is generated from [`ComputeSpec::all`].
//!
//! A default is never typed into a descriptor. It is read from the type: an
//! instance built by the type's own no-argument constructor, the `Default` of
//! a configuration enum, or the named constant (`DEFAULT_*`) the constructor
//! itself uses. A unit test builds every registered type from its required
//! arguments alone and checks that what the instance holds is exactly what the
//! spec claims ([`Described::tunables`] against [`ParamSpec::default`]).
//!
//! The descriptor holds only what the core knows: names, kinds, defaults,
//! required-ness and input kinds. Labels, units, categories and slider ranges
//! are presentation, and belong to the UI that presents them.
//!
//! ```
//! use molrs::compute::{ComputeSpec, Described, Kmeans, ParamDefault, ParamValue};
//!
//! let spec = Kmeans::spec();
//! assert_eq!(spec.name, "Kmeans");
//! assert!(spec.param("k").unwrap().is_required());
//! assert_eq!(
//!     spec.param("max_iter").unwrap().default,
//!     ParamDefault::Value(ParamValue::Int(Kmeans::DEFAULT_MAX_ITER as u64)),
//! );
//! assert_eq!(ComputeSpec::find("Kmeans"), Some(&spec));
//! ```

mod registry;

use molrs::op::F;

use crate::compute::Compute;

/// The descriptor of one compute type.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct ComputeSpec {
    /// The public name — the Rust type name, or for a generic type the name
    /// of the instantiation (`PmftXyz` for `Pmft<Xyz>`).
    pub name: &'static str,
    /// The configuration parameters, in constructor-then-setter order. Each is
    /// either a constructor argument or the value of a `with_*` setter of the
    /// same name.
    pub params: Vec<ParamSpec>,
    /// The fields of the `Input` struct `compute` takes (of one frame record
    /// when [`form`](Self::form) is [`InputForm::Frames`]).
    pub inputs: &'static [InputSpec],
    /// How `compute` receives those fields.
    pub form: InputForm,
}

impl ComputeSpec {
    /// The parameter named `name`.
    pub fn param(&self, name: &str) -> Option<&ParamSpec> {
        self.params.iter().find(|p| p.name == name)
    }

    /// The input field named `name`.
    pub fn input(&self, name: &str) -> Option<&InputSpec> {
        self.inputs.iter().find(|i| i.name == name)
    }
}

/// How `compute` receives its input fields.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub enum InputForm {
    /// One `Input` struct per call.
    Single,
    /// A slice of per-frame records, each with the listed fields; the type is
    /// also a [`Stream`](crate::compute::Stream) folding them one at a time.
    Frames,
}

/// One configuration parameter.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct ParamSpec {
    /// The constructor argument or `with_*` setter name.
    pub name: &'static str,
    /// What values it takes.
    pub kind: ParamKind,
    /// What it is when the caller gives none.
    pub default: ParamDefault,
}

impl ParamSpec {
    /// Whether the caller must give this parameter.
    pub fn is_required(&self) -> bool {
        matches!(
            self.default,
            ParamDefault::Required | ParamDefault::RequiredIn(_)
        )
    }
}

/// The values a parameter takes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub enum ParamKind {
    /// A real number.
    Float,
    /// A non-negative integer.
    Int,
    /// A flag.
    Bool,
    /// One of the canonical names of a `named_enum!` choice.
    Choice(&'static [&'static str]),
    /// A list of choices, one per axis.
    ChoiceList(&'static [&'static str]),
    /// A list of reals.
    FloatList,
    /// A list of non-negative integers.
    IntList,
    /// A list of 3-vectors.
    Vec3List,
    /// A binning [`Axis`](crate::compute::Axis): `(min, max, n_bins)`.
    Axis,
    /// One [`Axis`](crate::compute::Axis) per dimension.
    AxisList,
    /// A [`RegularGrid`](crate::compute::RegularGrid): origin, basis, dims,
    /// periodicity.
    Grid,
    /// A `(start, end)` window of a curve, as fractions of its length.
    Range,
    /// A [`LagKernel`](crate::compute::LagKernel): one of its `NAMES`, plus the
    /// order `l` for `legendre` / `wigner_d`.
    LagKernel,
    /// A [`UnitPreset`](molrs::core::UnitPreset), by name.
    Units,
}

/// A parameter value — a default, or the value an instance holds.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub enum ParamValue {
    /// A real number.
    Float(F),
    /// A non-negative integer.
    Int(u64),
    /// A flag.
    Bool(bool),
    /// A choice's canonical name.
    Name(&'static str),
    /// A list of reals.
    FloatList(Vec<F>),
    /// A list of non-negative integers.
    IntList(Vec<u64>),
    /// A `(start, end)` fraction window.
    Range(F, F),
    /// An optional setting that is off: nothing is applied unless given.
    Unset,
}

/// What a parameter is when the caller gives none.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub enum ParamDefault {
    /// No default: the caller must give it.
    Required,
    /// Required by the named constructor, which is one of several mutually
    /// exclusive ways to build the type (`StaticStructureFactorDirect::new`
    /// takes explicit k-vectors, `::isotropic` a k-axis).
    RequiredIn(&'static str),
    /// This value ([`ParamValue::Unset`] for an option that is off).
    Value(ParamValue),
    /// The value of the named parameter (`PairPresence`'s `r1` is `r0`).
    SameAs(&'static str),
    /// `factor` times the value of the parameter `of` (a Gaussian's cutoff is
    /// `3σ`).
    Scaled { of: &'static str, factor: F },
}

impl ParamDefault {
    /// An option that is off unless given.
    pub const UNSET: Self = Self::Value(ParamValue::Unset);
}

/// One field of a compute's `Input`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct InputSpec {
    /// The field name — the keyword argument of `compute` in a binding.
    pub name: &'static str,
    /// What data it holds.
    pub kind: InputKind,
    /// Whether it is an `Option` the caller may leave out.
    pub optional: bool,
}

/// The data an input field holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub enum InputKind {
    /// One snapshot's positions, `N × 3`.
    Positions,
    /// One snapshot's per-item vectors (velocities, directors, projection
    /// axes), `N × 3`.
    Vectors,
    /// A `T × N × D` stack of frames.
    Trajectory,
    /// A `T × C` time series of channels.
    Series,
    /// A 1-D array of reals (per-item masses, charges, radii; a signal).
    Scalars,
    /// One real (`dt`, a volume, a temperature).
    Scalar,
    /// One index (a channel of an upstream curve).
    Index,
    /// A list of item indices.
    Indices,
    /// A list of `(i, j)` index pairs.
    IndexPairs,
    /// A `M × k` table of item tuples (pairs, triples, quadruples).
    Tuples,
    /// One tuple table per axis.
    TuplesList,
    /// A per-item flag mask.
    Mask,
    /// Integer image flags, `T × N × 3`.
    Images,
    /// Unit quaternions, `N × 4` (scalar first).
    Quaternions,
    /// Planar orientation angles, `N`.
    Angles,
    /// A [`SimBox`](molrs::core::SimBox).
    SimBox,
    /// One [`SimBox`](molrs::core::SimBox) per frame.
    Boxes,
    /// A [`Neighbors`](molrs::core::Neighbors) table.
    Neighbors,
    /// One [`Neighbors`](molrs::core::Neighbors) table per frame.
    NeighborFrames,
    /// A [`PairSource`](molrs::core::PairSource).
    PairSource,
    /// One [`PairSource`](molrs::core::PairSource) per frame.
    PairSourceFrames,
    /// A [`Partition`](crate::compute::Partition) of the items into groups.
    Partition,
    /// A [`FeatureMatrix`](crate::compute::FeatureMatrix).
    FeatureMatrix,
    /// A [`RegularGrid`](crate::compute::RegularGrid) the values live on.
    Grid,
    /// Values on a grid, `nx × ny × nz`.
    Field,
    /// One of a choice's canonical names.
    Choice(&'static [&'static str]),
    /// The output of another compute, by type name.
    Upstream(&'static str),
    /// A list of outputs of another compute, by type name.
    UpstreamList(&'static str),
}

/// A compute type that describes itself.
///
/// Implemented by every public compute type through `compute_spec!`, next to
/// the type, so the descriptor reads the type's own fields.
pub trait Described: Compute {
    /// The descriptor: name, parameters with their defaults, input fields.
    fn spec() -> ComputeSpec;

    /// The value this instance holds for every parameter that has a default
    /// (neither [`Required`](ParamDefault::Required) nor
    /// [`RequiredIn`](ParamDefault::RequiredIn)), in spec order.
    ///
    /// On an instance built from the required arguments alone this equals the
    /// spec's defaults — the property the defaults-parity tests check.
    fn tunables(&self) -> Vec<(&'static str, ParamValue)>;
}

macro_rules! scalar_values {
    ($($t:ty => |$v:ident| $value:expr),+ $(,)?) => {$(
        impl From<$t> for ParamValue {
            fn from($v: $t) -> Self {
                $value
            }
        }

        impl From<$t> for ParamDefault {
            fn from(value: $t) -> Self {
                Self::Value(ParamValue::from(value))
            }
        }
    )+};
}

scalar_values! {
    F => |v| Self::Float(v),
    bool => |v| Self::Bool(v),
    u8 => |v| Self::Int(u64::from(v)),
    u32 => |v| Self::Int(u64::from(v)),
    u64 => |v| Self::Int(v),
    usize => |v| Self::Int(v as u64),
}

impl From<ParamValue> for ParamDefault {
    fn from(value: ParamValue) -> Self {
        Self::Value(value)
    }
}

/// `None` is [`ParamValue::Unset`]; `Some(v)` is `v`.
impl<T: Into<ParamValue>> From<Option<T>> for ParamValue {
    fn from(value: Option<T>) -> Self {
        value.map_or(Self::Unset, Into::into)
    }
}

/// An option a default-constructed instance leaves off is
/// [`ParamDefault::UNSET`].
impl<T: Into<ParamValue>> From<Option<T>> for ParamDefault {
    fn from(value: Option<T>) -> Self {
        Self::Value(value.into())
    }
}

impl From<&[u32]> for ParamValue {
    fn from(value: &[u32]) -> Self {
        Self::IntList(value.iter().map(|&v| u64::from(v)).collect())
    }
}

impl From<(F, F)> for ParamValue {
    fn from((start, end): (F, F)) -> Self {
        Self::Range(start, end)
    }
}

/// Implement [`Described`] for a compute type.
///
/// ```ignore
/// compute_spec! {
///     Msd as "Msd";
///     params |s| {
///         max_lag: Int;                                    // required
///         origins: Choice(Origins::NAMES)
///             = Origins::default() => s.correlator.origins(); // default => readback
///     }
///     inputs {
///         positions: Trajectory;
///         boxes: Boxes, optional;
///     }
/// }
/// ```
///
/// A parameter without `= default` is [`ParamDefault::Required`]. A default is
/// any expression converting into [`ParamDefault`]; it must be read from the
/// type (a `DEFAULT_*` constant the constructor uses, an enum's `Default`, a
/// field of the default-constructed instance), never written as a literal.
/// `=> readback` reads the instance's current value through `s` and is given
/// for every parameter that has a default. Append `, Frames` after the name
/// for a compute whose input is a slice of per-frame records; prefix the type
/// with `@[const N: usize]` (any generic parameters) to describe every
/// instantiation at once.
macro_rules! compute_spec {
    (@default) => { $crate::compute::ParamDefault::Required };
    (@default $default:expr) => { $crate::compute::ParamDefault::from($default) };
    (@optional) => { false };
    (@optional optional) => { true };
    (@form) => { $crate::compute::InputForm::Single };
    (@form $form:ident) => { $crate::compute::InputForm::$form };
    (
        $(@[$($generics:tt)*])? $ty:ty as $name:expr $(, $form:ident)?;
        params |$s:ident| {
            $(
                $param:ident : $pkind:ident $(( $($parg:tt)* ))?
                $(= $default:expr $(=> $read:expr)?)?;
            )*
        }
        inputs {
            $(
                $input:ident : $ikind:ident $(( $($iarg:tt)* ))? $(, $opt:ident)?;
            )*
        }
    ) => {
        impl<$($($generics)*)?> $crate::compute::Described for $ty {
            fn spec() -> $crate::compute::ComputeSpec {
                const INPUTS: &[$crate::compute::InputSpec] = &[$(
                    $crate::compute::InputSpec {
                        name: stringify!($input),
                        kind: $crate::compute::InputKind::$ikind $(( $($iarg)* ))?,
                        optional: $crate::compute::spec::compute_spec!(@optional $($opt)?),
                    },
                )*];
                $crate::compute::ComputeSpec {
                    name: $name,
                    params: vec![$(
                        $crate::compute::ParamSpec {
                            name: stringify!($param),
                            kind: $crate::compute::ParamKind::$pkind $(( $($parg)* ))?,
                            default: $crate::compute::spec::compute_spec!(@default $($default)?),
                        },
                    )*],
                    inputs: INPUTS,
                    form: $crate::compute::spec::compute_spec!(@form $($form)?),
                }
            }

            #[allow(unused_variables)]
            fn tunables(&self) -> Vec<(&'static str, $crate::compute::ParamValue)> {
                let $s = self;
                vec![$($($(
                    (stringify!($param), $crate::compute::ParamValue::from($read)),
                )?)?)*]
            }
        }
    };
}

pub(crate) use compute_spec;
