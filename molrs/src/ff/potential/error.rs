//! [`CompileError`]: why a force field did not compile into kernels.

use std::fmt;

use crate::ff::ir::IrError;

/// Why [`PotentialCompiler`](crate::ff::compile::PotentialCompiler) (or a
/// kernel constructor it calls) built no kernels.
///
/// A refusal of the force-field IR stays typed ([`CompileError::Ir`]), so a
/// caller — a binding raising one exception class per [`IrError`] variant —
/// matches on it instead of reading a message.
///
/// ```
/// use molrs::core::Frame;
/// use molrs::ff::forcefield::ForceField;
/// use molrs::ff::ir::Params;
/// use molrs::ff::ir::IrError;
/// use molrs::ff::style_registry::Registry;
/// use molrs::ff::potential::CompileError;
/// use molrs::ff::compile::PotentialCompiler;
/// use std::sync::Arc;
///
/// let mut ff = ForceField::new("t");
/// ff.def_style("bond", "harmonic", Params::new()).unwrap();
/// // Compiled against a registry that declares no `bond` category.
/// let ff = ff.with_registry(Arc::new(Registry::new()));
/// let err = PotentialCompiler::new(&ff)
///     .compile(&Frame::new())
///     .unwrap_err();
/// let refused = IrError::UnknownCategory { category: "bond".into() };
/// assert_eq!(err, CompileError::Ir(refused));
/// ```
#[derive(Clone, Debug, PartialEq)]
pub enum CompileError {
    /// The force-field IR refused a style, a parameter or a term.
    Ir(IrError),
    /// A style that reads the simulation box — `pair coul/long/pme`'s
    /// Ewald sums, as LAMMPS's kspace reads its simulation box — compiled
    /// against a frame whose box it cannot use: none, one not periodic in
    /// every direction, or a cell outside LAMMPS's restricted triclinic form.
    /// The box is the frame's, never a force-field parameter.
    NoBox {
        category: String,
        style: String,
        reason: String,
    },
    /// Anything else that stops the compile: a missing block or column, an
    /// unknown type label, special-bonds weights a compiled pair list cannot
    /// carry.
    Invalid(String),
}

impl CompileError {
    /// The IR refusal, when this is one.
    pub fn ir(&self) -> Option<&IrError> {
        match self {
            CompileError::Ir(e) => Some(e),
            CompileError::NoBox { .. } | CompileError::Invalid(_) => None,
        }
    }
}

impl fmt::Display for CompileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CompileError::Ir(e) => e.fmt(f),
            CompileError::NoBox {
                category,
                style,
                reason,
            } => write!(
                f,
                "{category} style `{style}` reads the frame's periodic simulation box: {reason}"
            ),
            CompileError::Invalid(message) => f.write_str(message),
        }
    }
}

impl std::error::Error for CompileError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            CompileError::Ir(e) => Some(e),
            CompileError::NoBox { .. } | CompileError::Invalid(_) => None,
        }
    }
}

impl From<IrError> for CompileError {
    fn from(e: IrError) -> Self {
        CompileError::Ir(e)
    }
}

impl From<String> for CompileError {
    fn from(message: String) -> Self {
        CompileError::Invalid(message)
    }
}

impl From<&str> for CompileError {
    fn from(message: &str) -> Self {
        CompileError::Invalid(message.to_owned())
    }
}
